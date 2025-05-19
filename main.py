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

import os
import shutil

from coherence import inference_coherence_batch

cache = SQLiteKVCache("./cache.db")

# define comparing methods
compare_models = [
    (
        "openai/gpt-4.1-comparison-system",
        OpenrouterGenericInference(OPENROUTER_API_KEY, "openai/gpt-4.1"),
    ),
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
    (
        "qwen/qwen3-235b-a22b-comparison-system",  # has a claude lineage, because they trained off of outputs
        OpenrouterGenericInference(OPENROUTER_API_KEY, "qwen/qwen3-235b-a22b"),
    ),
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
        "google/gemini-2.5-flash-preview-comparison-system",
        OpenrouterGenericInference(
            OPENROUTER_API_KEY, "google/gemini-2.5-flash-preview"
        ),
    ),
    (
        "meta/llama-4-maverick-comparison-system",
        OpenrouterGenericInference(OPENROUTER_API_KEY, "meta-llama/llama-4-maverick"),
    ),
    (
        "mistralai/mistral-medium-3",
        OpenrouterGenericInference(OPENROUTER_API_KEY, "mistralai/mistral-medium-3"),
    ),
]

compare_models_coherence = [
    (
        "deepseek/deepseek-v3-comparison-system",
        OpenrouterGenericInference(
            OPENROUTER_API_KEY, "deepseek/deepseek-chat-v3-0324"
        ),
    ),
    (
        "google/gemini-2.5-flash-preview-comparison-system",
        OpenrouterGenericInference(
            OPENROUTER_API_KEY, "google/gemini-2.5-flash-preview"
        ),
    ),
]

# test run on coherence
print(
    inference_coherence_batch(
        TranslatableLanguage.German,
        coherence_test_run,
        compare_models_coherence,
        4,
        2,
        0,
        cache,
    )
)
import sys

sys.exit()

# first run on testing dataset

"""dataset = evaluate_datasets(
    target_languages_testing, evaluation_targets_testing, cache, compare_models
)

with open("out_testing.json", "w") as f:
    f.write(json.dumps(dataset, indent=4))"""

# then run on sensible_large on German for a broad idea
data_initial_comparison = evaluate_datasets(
    [TranslatableLanguage.German],
    evaluation_targets_new_multitemp_multithink,
    cache,
    compare_models,
)

with open("out_german_multitemp_multithink.json", "w") as f:
    f.write(json.dumps(data_initial_comparison, indent=4))

# then run on the temperature testing dataset

# then test the top models vs thinking models vs thinking on-off

# then run on the low_resource_inform on the best existing models on all niche Nuenki langs
