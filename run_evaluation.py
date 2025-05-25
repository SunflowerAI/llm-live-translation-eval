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
from scipy.stats import mannwhitneyu
from utils import md5hash, get_translation_with_cache_check


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

    for comparison_model, comparison_inference in compare_models:
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

Original text:
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
                    comparison_data = comparison_inference.infer(comparison_prompt, 0)
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
            continue

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
                break

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
