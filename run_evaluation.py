import time
from dataset import SENTENCES_LIST
from typedefinitions import *
from davidson_model import DavidsonBT
from latency_logger import log_latency
from itertools import combinations
import threading
from queue import Queue
from random import choice
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import numpy as np
import random
import re
import json
from scipy.stats import mannwhitneyu
from utils import md5hash, get_translation_with_cache_check, inference_slot


def all_non_duplicate_sets(input_list):
    result = []
    for r in range(1, len(input_list) + 1):
        result.extend([list(c) for c in combinations(input_list, r)])
    return result


def pair_every(pairwise_items, input_data, start, interval, limit):
    if len(input_data) <= start:
        return

    last = input_data[start]

    count = 0
    for i, x in enumerate(input_data[start:]):
        if count >= limit:
            break

        if i % interval == 0 and last != x:
            pairwise_items.append((last, x))
            last = x
            count += 1


def log_sentence_data(*args, **kwargs):
    print("S", *args, **kwargs)


def deterministic_sample(translations, key):
    rng = random.Random(key)
    return list(enumerate(rng.sample(list(translations), len(translations))))


def process_sentence(
    language, category, sentence, testing_models, cache, compare_models, output_queue
):
    wait_t = choice(range(0, 3))
    time.sleep(wait_t)

    log_sentence_data("Checking", language, category, sentence)
    print("Lang", language)

    # {text: [compare_models_list]}
    translations = {}
    refusals = []

    # sampling to help with rate limits
    for testing_model in testing_models:
        translation = get_translation_with_cache_check(
            testing_model,
            TranslatableLanguage.English,
            language,
            category,
            sentence,
            cache,
        )
        if translation == None:
            refusals.append(testing_model)

        else:
            if translations.get(translation):
                translations[translation].append(testing_model)
            else:
                translations[translation] = [testing_model]

    log_sentence_data("Translations", translations)

    judge_translations(
        language,
        category,
        sentence,
        translations,
        refusals,
        cache,
        compare_models,
        output_queue,
    )


def judge_translations(
    language,
    category,
    sentence,
    translations,
    refusals,
    cache,
    compare_models,
    output_queue,
    context_pairs=None,
):
    """Have each judge score every distinct translation of ``sentence``.

    ``translations`` maps each distinct translation text to the list of tested
    models that produced it; ``refusals`` is the list of models that refused.
    Shared by the batch (``process_sentence``) and live pipelines so the judge
    prompt, deterministic ID shuffling, and JSON parsing contract stay identical.

    ``context_pairs`` is an optional list of ``(source_segment, best_translation)``
    pairs for the preceding segments (oldest first) of a live talk. When given,
    the judge is shown that running source-and-translation context and asked to
    score each candidate for accuracy *and* coherence with it. The translation in
    each pair is the one the judge panel previously rated best, so the judge sees
    a single coherent thread rather than every model's history. When ``None`` the
    prompt is byte-identical to the batch prompt, so the existing judge cache
    stays valid.

    Returns the translation text rated best for ``sentence`` (highest mean score
    across the judge panel), or ``None`` if nothing scored — used by the live
    pipeline to extend ``context_pairs`` for the next segment.
    """
    context_block = ""
    if context_pairs:
        transcript = "\n".join(
            f"Source: {src}\nTranslation: {best}" for src, best in context_pairs
        )
        context_block = (
            "Preceding context (the immediately prior segments of the same live "
            "talk, each with its best accepted translation — for reference only, "
            "do NOT score these). The original text below continues directly from "
            "this, so also judge whether each translation is coherent with it: "
            "resolved references, and consistent terminology and register.\n"
            f"```\n{transcript}\n```\n\n"
        )

    scores_by_text = defaultdict(list)

    def run_judge(comparison_model, comparison_inference):
        """Score every distinct translation with one judge.

        Emits ``RankItem``s (including ``-483`` refusals) to ``output_queue`` and
        returns ``{translation_text: score}`` for this judge so the caller can
        aggregate the panel. Judges are independent, so this runs concurrently.
        """
        local_scores = {}
        randomised_id_lookup = {}

        prompt_translations_text = ""
        determ_key = (
            "|".join(sorted(list(translations.keys()))) + "-" + comparison_model
        )
        # log_sentence_data("Determ key", determ_key)

        for i, text in deterministic_sample(
            sorted(list(translations.keys())), determ_key
        ):
            models = translations[text]
            randomised_id_lookup[i] = (text, models)
            prompt_translations_text += f"""Translation ID {i}:\n```{text}```\n"""

            json_example = """```
[
{ "id": [TRANSLATION_ID], "score": [SCORE_0_TO_100] },
{ "id": [ANOTHER_TRANSLATION_ID], "score": [SCORE_0_TO_100] }
]
```"""

        comparison_prompt = f"""You're an unforgiving professional translator. Your job is to critique, compare, and score multiple translations of the same sentence.
IGNORE subjective or arguable style differences (e.g. loanwords; passive vs active). If a choice is defensible as style, don't criticise it. Favour natural, native-sounding translations over literal or clunky ones. The objective is to assess translation *quality* independent of *style*.

Think step-by-step in *minimal per point*, like "1 is better at 'Hallo'. 2 more idiomatic. 1 misuses 'Schlecht'." AVOID VERBOSITY! Just terse, for yourself. Save tokens.

Evaluation order:
- Accuracy (Same meaning?)
- Vocab (Check for any mistakes, however subtle)
- Grammar accuracy
- Tone matching
- *Consistency* in formality, both within the translation and compared to the original
- Idiomaticity & style (native phrasing? Remember, no stylistic judgements)

{context_block}Original text:
```{sentence}```

Translations into `{language.value}`:
{prompt_translations_text}

Critique tersely, compare, and score each 0-100. Use the full range, including <40 for poor translations.

Once you're ready to give your final answer, write out a triple-backtick code block with a JSON response in the following form:
{json_example}

This will be parsed, so keep your output exact, and remember the triple-backticks and correct format!
"""

        # log_sentence_data("Prompt", comparison_prompt)

        key = f"COMPARISON hash:{md5hash(comparison_prompt)} Model:{comparison_model}"
        # log_sentence_data("KEY", key)
        comparison_data = None

        if cache.get(key):
            comparison_data = cache.get(key)
            log_sentence_data("Cache hit")
        else:
            log_sentence_data("Inference for comparison")
            wait_t = choice(range(0, 2))
            time.sleep(wait_t)

            for i in range(3):
                try:
                    with inference_slot():
                        comparison_data = comparison_inference.infer(
                            comparison_prompt, 0
                        )
                    if comparison_data:
                        cache.set(key, comparison_data)
                        break
                except Exception as e:
                    log_sentence_data(
                        "Error during inference:",
                        e,
                        "with comp model",
                        comparison_model,
                    )
                    time.sleep(i * choice(range(5, 10)))

        if not comparison_data:
            print("Failed to get data for", comparison_model, sentence)
            return local_scores

        log_sentence_data("Resp", comparison_data)

        def extract_json_from_response(response):
            parts = response.split("```")
            for part in reversed(parts):
                try:
                    return json.loads(part.replace("json", "").strip())
                except json.JSONDecodeError:
                    continue

            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return None

        try:
            scores_list = extract_json_from_response(comparison_data)

            if not scores_list:
                print(
                    "Error handling JSON - could not parse!!!",
                    f"full data: [{comparison_data}]",
                )
                return local_scores

            if len(scores_list) != len(randomised_id_lookup.keys()):
                print("Incorrect response length on", comparison_data)

            for refusal in refusals:
                output_queue.put(
                    RankItem(
                        language=language,
                        tested_entry=refusal,
                        sentence=sentence,
                        sentence_category=category,
                        evaluating_model=comparison_model,
                        translation=text,
                        score=-483,
                    )
                )

            for entry in scores_list:
                entry_id = int(str(entry["id"]).replace("ID", ""))
                entry_score = float(entry["score"])
                if entry_id in randomised_id_lookup:
                    text, models = randomised_id_lookup[entry_id]
                    local_scores[text] = entry_score
                    for testing_model in models:
                        output_queue.put(
                            RankItem(
                                language=language,
                                tested_entry=testing_model,
                                sentence=sentence,
                                sentence_category=category,
                                evaluating_model=comparison_model,
                                translation=text,
                                score=entry_score,
                            )
                        )
                else:
                    print("Incorrect entry id", entry_id, "on msg", comparison_data)

        except Exception as e:
            print(
                "Error handling JSON",
                e,
                "model",
                comparison_model,
                f"full data: [{comparison_data}]",
            )

        return local_scores

    # Judges are independent; run the panel concurrently. Actual API concurrency
    # is still bounded by the global inference_slot() semaphore.
    with ThreadPoolExecutor(max_workers=max(1, len(compare_models))) as executor:
        for local_scores in executor.map(
            lambda cm: run_judge(cm[0], cm[1]), compare_models
        ):
            for text, score in local_scores.items():
                scores_by_text[text].append(score)

    if not scores_by_text:
        return None
    return max(
        scores_by_text,
        key=lambda t: sum(scores_by_text[t]) / len(scores_by_text[t]),
    )


def _extract_json_from_response(response):
    """Return the JSON parsed from the last triple-backtick block (the judge
    contract), falling back to the whole response. ``None`` if nothing parses."""
    parts = response.split("```")
    for part in reversed(parts):
        try:
            return json.loads(part.replace("json", "").strip())
        except json.JSONDecodeError:
            continue
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return None


def judge_translations_windowed(
    language,
    category,
    segments,
    context_pairs,
    cache,
    compare_models,
    output_queue,
):
    """Score a window of consecutive live segments — one judge call per judge.

    ``segments`` is a list (oldest first) of per-segment payloads:
    ``{"index": int, "segment": str, "translations": {text: [models]},
    "refusals": [models]}``. Each judge scores every distinct translation of
    every segment in the window in a single call, giving a one-clause
    justification per score (a lightweight stand-in for chain-of-thought — no
    thinking model). Translation IDs are deterministically shuffled per
    (segment, judge) to avoid positional bias; the JSON the judge returns keys
    each score by ``(segment, id)``.

    Batching amortises the shared prompt prefix and cuts the judge-call count
    (the live pipeline's bottleneck) by ~``len(segments)``. The tradeoff: within
    one window, a segment does not see the *confirmed best* translation of the
    segment immediately before it (only the source text and every candidate) —
    ``context_pairs`` carries the confirmed bests of segments *before* the
    window. Call with single-element windows for exact per-segment fidelity.

    Emits ``RankItem``s (incl. ``-483`` refusals) to ``output_queue`` and returns
    ``{segment_index: best_translation_text or None}`` for the live driver to
    extend ``context_pairs``.
    """
    seg_text = {s["index"]: s["segment"] for s in segments}

    context_block = ""
    if context_pairs:
        transcript = "\n".join(
            f"Source: {src}\nTranslation: {best}" for src, best in context_pairs
        )
        context_block = (
            "Preceding context (the immediately prior segments of the same live "
            "talk, each with its best accepted translation — for reference only, "
            "do NOT score these). The segments below continue directly from this, "
            "so also judge whether each translation is coherent with it and with "
            "the other segments in this batch: resolved references, and consistent "
            "terminology and register.\n"
            f"```\n{transcript}\n```\n\n"
        )

    # Only segments with at least one candidate are scored; refusals are emitted
    # for every segment regardless.
    scorable = [s for s in segments if s["translations"]]

    # {segment_index: {text: [scores across the judge panel]}}
    scores_by_segment = defaultdict(lambda: defaultdict(list))

    def run_judge(comparison_model, comparison_inference):
        # {segment_index: {text: score}} for this judge.
        local_scores = defaultdict(dict)

        # Refusals are independent of inference — emit them first.
        for s in segments:
            for refusal in s["refusals"]:
                output_queue.put(
                    RankItem(
                        language=language,
                        tested_entry=refusal,
                        sentence=s["segment"],
                        sentence_category=category,
                        evaluating_model=comparison_model,
                        translation="",
                        score=-483,
                    )
                )

        if not scorable:
            return local_scores

        # {segment_index: {local_id: (text, [models])}}
        lookup = {}
        segments_block = ""
        for s in scorable:
            seg_index = s["index"]
            translations = s["translations"]
            determ_key = (
                "|".join(sorted(translations))
                + "-"
                + str(seg_index)
                + "-"
                + comparison_model
            )
            lookup[seg_index] = {}
            segments_block += (
                f"=== Segment {seg_index} ===\n"
                f"Original:\n```{s['segment']}```\n"
                f"Translations into `{language.value}`:\n"
            )
            for local_id, text in deterministic_sample(
                sorted(translations), determ_key
            ):
                lookup[seg_index][local_id] = (text, translations[text])
                segments_block += f"(segment {seg_index}, ID {local_id}):\n```{text}```\n"
            segments_block += "\n"

        json_example = """```
[
{ "segment": [SEGMENT], "id": [TRANSLATION_ID], "score": [SCORE_0_TO_100], "reason": "[ONE TERSE CLAUSE]" },
{ "segment": [SEGMENT], "id": [ANOTHER_ID], "score": [SCORE_0_TO_100], "reason": "[ONE TERSE CLAUSE]" }
]
```"""

        comparison_prompt = f"""You're an unforgiving professional translator. Your job is to critique, compare, and score multiple candidate translations across a batch of consecutive segments from the same live talk.
IGNORE subjective or arguable style differences (e.g. loanwords; passive vs active). If a choice is defensible as style, don't criticise it. Favour natural, native-sounding translations over literal or clunky ones. The objective is to assess translation *quality* independent of *style*.

Evaluation order, per candidate:
- Accuracy (Same meaning?)
- Vocab (Check for any mistakes, however subtle)
- Grammar accuracy
- Tone matching
- *Consistency* in formality, both within the translation and compared to the original
- Coherence with the preceding context and the other segments in this batch
- Idiomaticity & style (native phrasing? Remember, no stylistic judgements)

{context_block}The segments below are consecutive and in order. Score every candidate translation of every segment 0-100. Use the full range, including <40 for poor translations. Give each score a single terse justification clause — this enforces rigour; keep it short to save tokens.

{segments_block}When you're ready, write out a single triple-backtick code block with a JSON array — one object per (segment, translation) — in the following form:
{json_example}

This will be parsed, so keep your output exact: emit exactly one JSON code block as the LAST thing in your response, with a "reason" for every entry. Remember the triple-backticks and correct format!
"""

        key = f"WINDOWCOMPARISON hash:{md5hash(comparison_prompt)} Model:{comparison_model}"
        comparison_data = None
        if cache.get(key):
            comparison_data = cache.get(key)
            log_sentence_data("Cache hit")
        else:
            log_sentence_data("Inference for windowed comparison")
            wait_t = choice(range(0, 2))
            time.sleep(wait_t)
            for i in range(3):
                try:
                    with inference_slot():
                        comparison_data = comparison_inference.infer(
                            comparison_prompt, 0
                        )
                    if comparison_data:
                        cache.set(key, comparison_data)
                        break
                except Exception as e:
                    log_sentence_data(
                        "Error during inference:",
                        e,
                        "with comp model",
                        comparison_model,
                    )
                    time.sleep(i * choice(range(5, 10)))

        if not comparison_data:
            print(
                "Failed to get data for",
                comparison_model,
                "window starting",
                segments[0]["index"],
            )
            return local_scores

        scores_list = _extract_json_from_response(comparison_data)
        if not scores_list:
            print(
                "Error handling JSON - could not parse!!!",
                f"full data: [{comparison_data}]",
            )
            return local_scores

        expected = sum(len(v) for v in lookup.values())
        if len(scores_list) != expected:
            print(
                "Incorrect response length on windowed judge",
                len(scores_list),
                "vs",
                expected,
            )

        for entry in scores_list:
            try:
                seg_index = int(str(entry["segment"]).replace("ID", ""))
                entry_id = int(str(entry["id"]).replace("ID", ""))
                entry_score = float(entry["score"])
            except (KeyError, ValueError, TypeError):
                print("Bad entry in windowed judge", entry)
                continue

            if seg_index in lookup and entry_id in lookup[seg_index]:
                text, models = lookup[seg_index][entry_id]
                local_scores[seg_index][text] = entry_score
                for testing_model in models:
                    output_queue.put(
                        RankItem(
                            language=language,
                            tested_entry=testing_model,
                            sentence=seg_text[seg_index],
                            sentence_category=category,
                            evaluating_model=comparison_model,
                            translation=text,
                            score=entry_score,
                        )
                    )
            else:
                print("Incorrect (segment,id) in windowed judge", entry)

        return local_scores

    # Judges are independent; run the panel concurrently. Actual API concurrency
    # is still bounded by the global inference_slot() semaphore.
    with ThreadPoolExecutor(max_workers=max(1, len(compare_models))) as executor:
        for local_scores in executor.map(
            lambda cm: run_judge(cm[0], cm[1]), compare_models
        ):
            for seg_index, text_scores in local_scores.items():
                for text, score in text_scores.items():
                    scores_by_segment[seg_index][text].append(score)

    best_by_segment = {}
    for s in segments:
        text_scores = scores_by_segment.get(s["index"])
        if text_scores:
            best_by_segment[s["index"]] = max(
                text_scores.keys(),
                key=lambda t, ts=text_scores: sum(ts[t]) / len(ts[t]),
            )
        else:
            best_by_segment[s["index"]] = None
    return best_by_segment


def compare_set(language, target_models, cache, compare_models):
    output = []
    threads = []
    output_queue = Queue()

    for category in SENTENCES_LIST.keys():
        for sentence in SENTENCES_LIST[category]:
            thread = threading.Thread(
                target=process_sentence,
                args=(
                    language,
                    category,
                    sentence,
                    target_models,
                    cache,
                    compare_models,
                    output_queue,
                ),
            )
            threads.append(thread)
            thread.start()

    for thread in threads:
        thread.join()

    while not output_queue.empty():
        output.append(output_queue.get())

    return output


# not doing lots here - the idea is that if we do want to filter by other cats, we do that at the higher level
def produce_summary(dataset, target_models, judge_names, sentence_cats, i):
    print("Summarising for", i, judge_names, sentence_cats)

    filtered_items = [
        x
        for x in dataset
        if x.evaluating_model in judge_names and x.sentence_category in sentence_cats
    ]

    model_data_list = []
    for i, model in enumerate(target_models):
        model_data_filtered = [x for x in filtered_items if x.tested_entry == model]
        scores = [x.score for x in model_data_filtered]
        mean = sum(scores) / len(scores)
        median = scores[len(scores) // 2]
        std_dev = np.std(scores)

        model_data_list.append(
            {
                "mean": mean,
                "median": median,
                "scores": scores,
                "std_dev": std_dev,
                "model": {
                    "name": model.model_name.value,
                    "company": model.model_company.value,
                    "temp": model.temp,
                    "thinking": model.thinking,
                },
                "id": i,
            }
        )

    for item in model_data_list:
        item["p_vals"] = {}

        for item_2 in model_data_list:
            id_2 = item_2["id"]

            if item == item_2:
                item["p_vals"][id_2] = "N/A"

            stat, p_value = mannwhitneyu(
                item["scores"], item_2["scores"], alternative="two-sided"
            )

            item["p_vals"][id_2] = p_value

    return model_data_list


def evaluate_datasets(target_languages, target_models, cache, compare_models):
    out = []

    for lang in target_languages:
        print("Lang", lang)
        data = compare_set(lang, target_models, cache, compare_models)

        # now make it into a minimal set of data that's OK to be handed to the client...

        model_data = []
        for i, model in enumerate(target_models):
            data_filtered = [x for x in data if x.tested_entry == model]

            scored_by_narrowed_pairs = []

            for item in data_filtered:
                found = False

                for existing in scored_by_narrowed_pairs:
                    if (
                        existing["sentence_category"] == item.sentence_category
                        and existing["evaluating_model"] == item.evaluating_model
                    ):
                        existing["scores"].append(item.score)
                        found = True
                        break

                if not found:
                    scored_by_narrowed_pairs.append(
                        {
                            "sentence_category": item.sentence_category,
                            "evaluating_model": item.evaluating_model,
                            "scores": [item.score],
                        }
                    )

            model_data.append(
                {
                    "id": i,
                    "model": {
                        "name": model.model_name.value,
                        "company": model.model_company.value,
                        "temp": model.temp,
                        "thinking": model.thinking,
                    },
                    "comparison_items": scored_by_narrowed_pairs,
                    "compare_models": [x for x, y in compare_models],
                    "sentence_types": list(SENTENCES_LIST.keys()),
                }
            )

        out.append({"language": lang.value, "models": model_data})

    return out
