from typedefinitions import *
from anthropic_inference_source import AnthropicExecutableTranslator
from openrouter_inference_source import (
    OpenrouterExecutableTranslator,
    OpenrouterGenericInference,
)
from groq_inference_source import GroqExecutableTranslator
from deepl_inference_source import DeeplExecutableTranslator
from lingvanex_inference_source import LingvanexExecutableTranslator
from mistral_inference_source import MistralExecutableTranslator
from nuenki_inference_source import NuenkiHybridExecutableTranslator
from google_inference_source import GoogleGenericInference

from secrets_env import *

from sqlitekv import SQLiteKVCache
from test_data import *
from run_evaluation import evaluate_datasets
from live_evaluation import evaluate_live_datasets

import os
import shutil

from coherence import (
    inference_coherence_batch,
    chart_coherence,
    chart_coherence_by_category,
)

cache = SQLiteKVCache("./cache.db")

# define comparing methods
compare_models = [
    # (
    #    "openai/gpt-4.1-comparison-system",
    #    OpenrouterGenericInference(OPENROUTER_API_KEY, "openai/gpt-4.1"),
    # ),
    # (
    #    "anthropic/claude-3.7-sonnet-comparison-system",
    #    OpenrouterGenericInference(OPENROUTER_API_KEY, "anthropic/claude-3.7-sonnet"),
    # ),
    # (
    #    "x-ai/grok-3-beta-comparison-system",
    #    OpenrouterGenericInference(OPENROUTER_API_KEY, "x-ai/grok-3-beta"),
    # ),
    # (
    #    "anthropic/haiku-3.5-comparison-system",
    #    OpenrouterGenericInference(OPENROUTER_API_KEY, "anthropic/claude-3.5-haiku"),
    # ),
    # Dropped (2026-06-21): qwen3-235b-a22b is a reasoning model — slow as a judge
    # (it dominated the judging wall-clock) and errored on probe. Removed for the
    # live run, leaving four fast, non-reasoning judges.
    # (
    #     "qwen/qwen3-235b-a22b-comparison-system",
    #     OpenrouterGenericInference(OPENROUTER_API_KEY, "qwen/qwen3-235b-a22b"),
    # ),
    (
        "deepseek/deepseek-v3-comparison-system",
        OpenrouterGenericInference(
            OPENROUTER_API_KEY, "deepseek/deepseek-chat-v3-0324"
        ),
    ),
    # (
    #    "google/gemini-2.5-pro-exp-comparison-system-thinking",
    #    GoogleGenericInference(GEMINI_API_KEY, "gemini-2.5-pro-exp-03-25", True),
    # ),
    (
        "google/gemini-2.5-flash-comparison-system",
        OpenrouterGenericInference(
            OPENROUTER_API_KEY, "google/gemini-2.5-flash"
        ),
    ),
    (
        "meta/llama-4-maverick-comparison-system",
        OpenrouterGenericInference(OPENROUTER_API_KEY, "meta-llama/llama-4-maverick"),
    ),
    # Dropped (2026-06-21): mistralai/mistral-medium-3 returns 404 "No endpoints
    # matching your data policy" — its providers are excluded by this account's
    # OpenRouter privacy settings (openrouter.ai/settings/privacy). It contributed
    # no scores, so judging was effectively a 3-judge panel anyway. Re-enable by
    # adjusting the data policy if you want it back.
    # (
    #     "mistralai/mistral-medium-3",
    #     OpenrouterGenericInference(OPENROUTER_API_KEY, "mistralai/mistral-medium-3"),
    # ),
]

compare_models_coherence = [
    (
        "deepseek/deepseek-v3-comparison-system",
        OpenrouterGenericInference(
            OPENROUTER_API_KEY, "deepseek/deepseek-chat-v3-0324"
        ),
    ),
    (
        "google/gemini-2.5-flash-comparison-system",
        OpenrouterGenericInference(
            OPENROUTER_API_KEY, "google/gemini-2.5-flash"
        ),
    ),
]

# test run on coherence
"""data = inference_coherence_batch(
    TranslatableLanguage.German,
    coherence_test_run,
    compare_models_coherence,
    4,
    24,
    0,
    cache,
)

chart_coherence(data, save_path="german_coherence_depth.png")"""

# first run on testing dataset

"""dataset = evaluate_datasets(
    target_languages_testing, evaluation_targets_testing, cache, compare_models
)

with open("out_testing.json", "w") as f:
    f.write(json.dumps(dataset, indent=4))"""

# Live translation: the Flash Lite price-class roster translating ordered sermon
# segments one at a time, each model seeing its own previous five translations as
# context (see live_evaluation.py). Kept small (a couple of sermons, capped
# segments) so a run is tractable; widen sermon_ids / max_segments to scale up.
data_live = evaluate_live_datasets(
    # Run one language at a time, starting with Simplified Chinese. Swap in the
    # next language (or add several) when ready:
    #   Cantonese, Japanese, Korean, Indonesian, Farsi, Spanish,
    #   BrazilianPortuguese, Burmese, Khmer, Bislama, Maori, Samoan, Tongan, Fijian
    [
        TranslatableLanguage.SimplifiedChinese,
    ],
    evaluation_targets_flash_lite_price_class,
    cache,
    compare_models,
    sermon_ids=[
        "1054361551",
        "1077837019",
    ],
    max_segments=None,
    window=5,
    # max_concurrency is the single rate-limit knob (total simultaneous API
    # calls); raise it if your OpenRouter limits allow, to finish proportionally
    # faster. language_workers/max_workers just need to keep it saturated.
    max_concurrency=16,
    language_workers=8,
)

# Name the output by the language(s) run, so one-language-at-a-time runs don't
# clobber each other (e.g. out_live_SimplifiedChinese.json).
langs_slug = "_".join(d["language"].replace(" ", "") for d in data_live)
with open(f"out_live_{langs_slug}.json", "w") as f:
    f.write(json.dumps(data_live, indent=4))

import sys

sys.exit()

# Top 10 OpenRouter models in the Gemini 2.5 Flash Lite price class, for
# translation. Roster defined in test_data.py as
# evaluation_targets_flash_lite_price_class.
data_flash_lite_class = evaluate_datasets(
    [
        TranslatableLanguage.German,
        TranslatableLanguage.Chinese,
        TranslatableLanguage.Hungarian,
        TranslatableLanguage.French,
        TranslatableLanguage.Japanese,
        TranslatableLanguage.Italian,
        # TranslatableLanguage.EuropeanSpanish,
        TranslatableLanguage.Ukrainian,
        TranslatableLanguage.Swedish,
        TranslatableLanguage.Korean,
        # TranslatableLanguage.Welsh,
        # TranslatableLanguage.Swahili,
    ],
    evaluation_targets_flash_lite_price_class,
    cache,
    compare_models,
)

with open("out_flash_lite_price_class.json", "w") as f:
    f.write(json.dumps(data_flash_lite_class, indent=4))

import sys

sys.exit()

data_updated_comparison = evaluate_datasets(
    [
        TranslatableLanguage.German,
        TranslatableLanguage.Chinese,
        TranslatableLanguage.Hungarian,
        TranslatableLanguage.French,
        TranslatableLanguage.Japanese,
        TranslatableLanguage.Italian,
        # TranslatableLanguage.EuropeanSpanish,
        TranslatableLanguage.Ukrainian,
        TranslatableLanguage.Swedish,
        TranslatableLanguage.Korean,
        # TranslatableLanguage.Welsh,
        # TranslatableLanguage.Swahili,
    ],
    evaluation_targets_new_multithink_additional_updated,
    cache,
    compare_models,
)

with open("out_major_comparison_updated.json", "w") as f:
    f.write(json.dumps(data_updated_comparison, indent=4))

# then run on sensible_large on German for a broad idea
data_initial_comparison = evaluate_datasets(
    [
        TranslatableLanguage.German,
        TranslatableLanguage.Chinese,
        TranslatableLanguage.Hungarian,
        TranslatableLanguage.French,
        TranslatableLanguage.Japanese,
        TranslatableLanguage.Italian,
        # TranslatableLanguage.EuropeanSpanish,
        TranslatableLanguage.Ukrainian,
        TranslatableLanguage.Swedish,
        TranslatableLanguage.Korean,
        # TranslatableLanguage.Welsh,
        # TranslatableLanguage.Swahili,
    ],
    evaluation_targets_new_multithink_additional,
    cache,
    compare_models,
)

with open("out_major_comparison.json", "w") as f:
    f.write(json.dumps(data_initial_comparison, indent=4))

data_initial_comparison = evaluate_datasets(
    [
        TranslatableLanguage.Welsh,
    ],
    evaluation_targets_new_multithink_additional_nodeepl_nolingvanex,
    cache,
    compare_models,
)

with open("out_major_comparison_nodeepl_nolingvanex.json", "w") as f:
    f.write(json.dumps(data_initial_comparison, indent=4))

data_initial_comparison = evaluate_datasets(
    [TranslatableLanguage.Swahili],
    evaluation_targets_new_multithink_additional_nodeepl_nolingvanex_nonuenki,
    cache,
    compare_models,
)

with open("out_major_comparison_nodeepl_nolingvanex_nonuenki.json", "w") as f:
    f.write(json.dumps(data_initial_comparison, indent=4))

data_temp_comparison = evaluate_datasets(
    [
        TranslatableLanguage.German,
        TranslatableLanguage.Chinese,
        TranslatableLanguage.Hungarian,
        TranslatableLanguage.French,
        TranslatableLanguage.Japanese,
        TranslatableLanguage.Italian,
        # TranslatableLanguage.EuropeanSpanish,
        TranslatableLanguage.Ukrainian,
        TranslatableLanguage.Swedish,
        TranslatableLanguage.Korean,
        # TranslatableLanguage.Welsh,
        # TranslatableLanguage.Swahili,
    ],
    evaluation_targets_top_models_multi_temp,
    cache,
    compare_models,
)

with open("out_temp_comparison.json", "w") as f:
    f.write(json.dumps(data_temp_comparison, indent=4))
