import hashlib
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


def md5hash(text):
    return hashlib.md5(bytes(text, "utf-8")).hexdigest()


def deterministic_coin_flip(s: str) -> bool:
    h = hashlib.sha256(s.encode()).digest()
    return bool(h[0] & 1)


def log_sentence_data(*args, **kwargs):
    # return
    print("S", *args, **kwargs)


def get_translation_with_cache_check(model, language, category, sentence, cache):
    cache_key = f"TRANSLATION Language:{language.value}|Model: {model.unique_id()}|Sentence category:{category}|Sentence md5:{md5hash(sentence)}"
    check = cache.get(cache_key)
    if check:
        return check
    else:
        log_sentence_data("Translating", cache_key)

        # this is to avoid getting ratelimited...
        sleep_t = choice(range(0, 4))
        time.sleep(sleep_t)

        start_t = None
        end_t = None
        translation = None

        i = 0
        while True:
            log_sentence_data("Translation attempt", i)
            i += 1
            try:
                start_t = time.time()
                translation = model.inference_source.translate(
                    TranslatableLanguage.English, language, sentence, model.temp
                )
                end_t = time.time()
                break
            except Exception as e:
                log_sentence_data("Err on translate", e, "with m", model)
                time.sleep(i * choice(range(3, 12)))

        cache.set(cache_key, translation)
        log_latency(model.unique_id(), category, sentence, end_t - start_t)
        return translation


def deterministic_sample(translations, key):
    rng = random.Random(key)
    return list(enumerate(rng.sample(list(translations), len(translations))))


def process_sentence(
    language, category, sentence, testing_models, cache, compare_models, output_queue
):
    log_sentence_data("Checking", language, category, sentence)

    # {text: [compare_models_list]}
    translations = {}

    for testing_model in testing_models:
        translation = get_translation_with_cache_check(
            testing_model, language, category, sentence, cache
        )
        if translations.get(translation):
            translations[translation].append(testing_model)
        else:
            translations[translation] = [testing_model]

    log_sentence_data("Translations", translations)

    for comparison_model, comparison_inference in compare_models:
        randomised_id_lookup = {}

        prompt_translations_text = ""
        determ_key = "|".join(sorted(translations.keys())) + "-" + comparison_model
        log_sentence_data("Determ key", determ_key)

        for i, text in deterministic_sample(sorted(translations.keys()), determ_key):
            models = translations[text]
            randomised_id_lookup[i] = (text, models)
            prompt_translations_text += f"""Translation ID {i}:\n```{text}```\n"""

            json_example = """```
[
{ "id": [TRANSLATION_ID], "score": [SCORE_0_TO_100] },
{ "id": [ANOTHER_TRANSLATION_ID], "score": [SCORE_0_TO_100] }
]
```"""

        comparison_prompt = f"""You're an unforgiving professional translator. You will be tasked with critiquing, comparing, and ranking multiple translations of the same sentence.
IGNORE subjective or arguable style differences (e.g. using a common or technical English loan word; passive vs active voice). If a choice can be reasonably defended as a subjective but valid choice, do not criticise it. Prefer natural translations over literal ones that sound non-native. The objective is to assess translation *quality* independent of *style*.

Think step-by-step in *only a few words per point*, like "1 is better at 'Hallo'. 2 more idiomatic. 1 misuses 'Schlecht'." AVOID VERBOSITY! Just terse, for yourself. Save tokens.

Evaluate in order of priority (high-to-low):
- Accuracy (Same meaning?)
- Vocab (Check for any mistakes, however subtle)
- Grammar accuracy
- Tone (does it match the original?)
- *Consistency* in formality, both within the translation and compared to the original
- Idiomaticity & style (native phrasing? Remember, no stylistic judgements)
Remember priorities when tiebreaking.

This is the original text:
```{sentence}```

Here are the translations:
{prompt_translations_text}

You should think aloud for as long as you need,
- critiquing
- comparing
- then, finally, scoring each translation between 0 and 100.
You should try and use a broad range and avoid being overly positive or forgiving. It's OK to use the low end (<40) when describing truly poor translations!

Once you're ready to give your final answer, write out a triple-backtick code block with a JSON response in the following form:
{json_example}

This will be parsed, so keep your output exact, and remember the triple-backticks and correct format!
"""

        # log_sentence_data("Prompt", comparison_prompt)

        key = f"COMPARISON hash:{md5hash(comparison_prompt)} Model:{comparison_model}"
        log_sentence_data("KEY", key)
        comparison_data = None

        if cache.get(key):
            comparison_data = cache.get(key)
            log_sentence_data("Cache hit")
        else:
            log_sentence_data("Inference for comparison")
            wait_t = choice(range(0, 6))
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
                    time.sleep(i * choice(range(5, 25)))

        if not comparison_data:
            continue

        log_sentence_data("Resp", comparison_data)

        # now parse...
        def extract_json_from_response(response):
            matches = re.findall(r"```(.*?)```", response, re.DOTALL)
            if matches:
                return matches[-1].strip().replace("json", "")
            else:
                return response

        json_str = extract_json_from_response(comparison_data)
        if json_str:
            try:
                scores_list = json.loads(json_str)

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
            except Exception as e:
                print("Error handling JSON", e, f"data:[{json_str}]")
        else:
            log_sentence_data("No JSON found in response")
            continue


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
    permutations = []
    perms_dict = {}

    i = 0
    for lang in target_languages:
        data = compare_set(lang, target_models, cache, compare_models)

        # things to do now:
        # - have a function that gets all the data for a specific model - mean, median, s.d., and all values
        # - also have it calculate p values
        # - have that function accept an arg for exclusion of judges
        # - do it for all permutations... and add to output
        judge_names = [x for x, y in compare_models]
        judge_combs = all_non_duplicate_sets(judge_names)
        sentence_combs = all_non_duplicate_sets(SENTENCES_LIST.keys())

        for judge_comb in judge_combs:
            for sentence_comb in sentence_combs:
                i += 1

                summary = produce_summary(
                    data, target_models, judge_comb, sentence_comb, i
                )
                permutations.append(
                    {
                        "lang": lang.value,
                        "judges": judge_comb,
                        "sentences": sentence_comb,
                        "id": i,
                    }
                )
                perms_dict[i] = summary

    return permutations, perms_dict
