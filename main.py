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
#
# Judge panel — upgraded (2026-06-22) from the flash-class trio (DeepSeek V3,
# Gemini 2.5 Flash, Llama 4 Maverick) to stronger, non-reasoning frontier
# judges. Paired with *windowed* judging (see batch_size on the
# evaluate_live_datasets call below), which amortises the shared prompt prefix
# and cuts the judge-call count, so the higher per-token cost of these models
# doesn't fully undo the savings. NOT reasoning/thinking judges — those dominate
# the judging wall-clock (see the qwen3 note); instead each judge now justifies
# every score inline (a lightweight stand-in for chain-of-thought).
#
# DeepSeek V3 is kept in the mix for cost balance and provider diversity; swap
# in other frontier slugs your OpenRouter access supports for different/stronger
# judges (e.g. an Opus- or GPT-5-class model), or drop deepseek for max strength.
compare_models = [
    (
        "openai/gpt-4.1-comparison-system",
        OpenrouterGenericInference(OPENROUTER_API_KEY, "openai/gpt-4.1"),
    ),
    (
        "anthropic/claude-sonnet-4.6-comparison-system",
        OpenrouterGenericInference(OPENROUTER_API_KEY, "anthropic/claude-sonnet-4.6"),
    ),
    (
        "deepseek/deepseek-v3-comparison-system",
        OpenrouterGenericInference(
            OPENROUTER_API_KEY, "deepseek/deepseek-chat-v3-0324"
        ),
    ),
]
# Previous flash-class panel — revert to this block for cheaper, faster judging:
#   ("google/gemini-2.5-flash-comparison-system",
#       OpenrouterGenericInference(OPENROUTER_API_KEY, "google/gemini-2.5-flash")),
#   ("meta/llama-4-maverick-comparison-system",
#       OpenrouterGenericInference(OPENROUTER_API_KEY, "meta-llama/llama-4-maverick")),
#
# Dropped (2026-06-21): qwen3-235b-a22b is a reasoning model — slow as a judge
# (it dominated the judging wall-clock) and errored on probe.
# Dropped (2026-06-21): mistralai/mistral-medium-3 returns 404 "No endpoints
# matching your data policy" — providers excluded by this account's OpenRouter
# privacy settings (openrouter.ai/settings/privacy).

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
    # Run one language at a time. Done: Simplified Chinese. Now running Korean.
    # Swap in the next language (or add several) when ready:
    #   Cantonese, Japanese, Indonesian, Farsi, Spanish,
    #   BrazilianPortuguese, Burmese, Khmer, Bislama, Maori, Samoan, Tongan, Fijian
    [
        TranslatableLanguage.Korean,
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
    # Windowed judging: how many consecutive segments each judge scores per
    # call. >1 amortises the shared prompt prefix and cuts the judge-call count /
    # wall-clock, offsetting the stronger (frontier) judge panel. The tradeoff:
    # segments inside one batch don't see each other's *confirmed best* as locked
    # context (only the source text + every candidate). 1 = exact original
    # fidelity, one call per segment.
    batch_size=5,
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
