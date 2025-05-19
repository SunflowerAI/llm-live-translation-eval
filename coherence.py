from typedefinitions import *
from openrouter_inference_source import *
from dataset import SENTENCES_LIST
from utils import md5hash, get_translation_with_cache_check
from secrets_env import *
import time

# The purpose of this file is to sanity check a given language-model pair
# through the (now mostly deprecated) "coherence" approach, which translates English->Language->English
# over multiple iterations, and has LLMs score how close it is to the original text each time.
# it lets you graph a coherence curve over multiple back-and-forth translations. Interesting!


def inference_coherence_batch(
    language, to_test, evaluators, depth, first_x_in_cat, temp, cache
):
    sentences_list = []
    for cat, sentences in SENTENCES_LIST.items():
        for x in sentences[:first_x_in_cat]:
            sentences_list.append((cat, x))

    out_data = []

    for model in to_test:
        for sentence_cat, sentence_to_test in sentences_list:
            last_iteration_english = sentence_to_test

            for n in range(1, depth + 1):
                print("Depth", n)

                iteration_target = get_translation_with_cache_check(
                    model,
                    TranslatableLanguage.English,
                    language,
                    "Coherence-" + sentence_cat,
                    last_iteration_english,
                    cache,
                )
                print("Target translated", iteration_target)

                last_iteration_english = get_translation_with_cache_check(
                    model,
                    language,
                    TranslatableLanguage.English,
                    "Coherence-" + sentence_cat,
                    last_iteration_english,
                    cache,
                )
                print("English backtranslated", last_iteration_english)

                # (str, int)
                evaluations_out = []

                for evaluator_str, evaluator_inference in evaluators:
                    prompt = f"""Your role is to rate how close two English sentences in a game of chinese whispers are, from 0 to 100. Evaluate based on how close it is to the original *meaning*, tone, etc.
Original text:
```{sentence_to_test}```
New text:
```{last_iteration_english}```

You may think aloud in truly minimal, terse points for your own use, saving tokens.
Your final response should be a single integer between 0 and 100, on a new line, with NOTHING ELSE on that last line!"""

                    score = None
                    while True:
                        eval_temp = 0
                        cache_key = (
                            md5hash(prompt) + "|" + evaluator_str + "|" + str(eval_temp)
                        )
                        print(cache_key)

                        resp = None
                        if cache.get(cache_key):
                            resp = cache.get(cache_key)
                        else:
                            try:
                                resp = evaluator_inference.infer(prompt, eval_temp)
                                cache.set(cache_key, resp)
                            except Exception as e:
                                print("Err on inf for", evaluator_str, ":", e)
                                continue

                        try:
                            score = int(resp.strip().split("\n")[-1])
                            if score >= 0 and score <= 100:
                                print("Got score", score)
                                break
                            else:
                                print("Failed to parse correct score from", resp)
                        except Exception as e:
                            print("Err on parsing resp", resp)

                        if eval_temp < 1:
                            eval_temp += 0.1

                        time.sleep(4)

                    evaluations_out.append((evaluator_str, score))

                out_data.append(
                    CoherenceIteration(
                        sentence_cat=sentence_cat,
                        sentence=sentence_to_test,
                        depth=n,
                        staged_text=last_iteration_english,
                        evaluations=evaluations_out,
                        tested_entry=model,
                    )
                )

    return out_data
