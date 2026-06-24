"""Sentence-level judging run for Korean + Simplified Chinese, smart & naive.

Live translation stays segment-by-segment (cached for both languages and both
segmentations from the earlier segmentation A/B), but here the *judging* unit is
the **sentence**: segments are regrouped into source-aligned sentences,
reconstructed per model, and judged independently with the ORIGINAL developer's
5-judge panel (``judge_translations``, the same procedure the batch benchmark
uses). See live_evaluation.group_sentences / reconstruct_sentence.

Because every segment translation is already cached, this only spends judge
calls. Sentence-mode judge prompts are new, so nothing judge-side is cached yet.

Writes one file per condition:
    out_live_Korean_smart_sentence.json
    out_live_Korean_naive_sentence.json
    out_live_SimplifiedChinese_smart_sentence.json
    out_live_SimplifiedChinese_naive_sentence.json

Usage:
    python -u run_sentence_compare.py [smart|naive ...]   (default: both)
"""

import json
import sys

from typedefinitions import TranslatableLanguage
from openrouter_inference_source import OpenrouterGenericInference
from secrets_env import *

from sqlitekv import SQLiteKVCache
from test_data import evaluation_targets_flash_lite_price_class
from live_evaluation import evaluate_live_datasets
from utils import set_inference_concurrency

cache = SQLiteKVCache("./cache.db")

GLOBAL_CONCURRENCY = 24
set_inference_concurrency(GLOBAL_CONCURRENCY)

# The original Nuenki 5-judge panel (the one behind out_major_comparison*.json),
# temp 0. mistral-medium-3 routes via the per-request data_collection="allow"
# override in openrouter_inference_source.py; qwen3-235b is the reasoning judge.
compare_models = [
    (
        "qwen/qwen3-235b-a22b-comparison-system",
        OpenrouterGenericInference(OPENROUTER_API_KEY, "qwen/qwen3-235b-a22b"),
    ),
    (
        "deepseek/deepseek-v3-comparison-system",
        OpenrouterGenericInference(OPENROUTER_API_KEY, "deepseek/deepseek-chat-v3-0324"),
    ),
    (
        "google/gemini-2.5-flash-comparison-system",
        OpenrouterGenericInference(OPENROUTER_API_KEY, "google/gemini-2.5-flash"),
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

SERMONS = ["1054361551", "1077837019"]

# Languages to judge this run. The original sentence-mode A/B was Korean +
# SimplifiedChinese (translations cached); 2026-06-23 also running Fijian, Tongan,
# Burmese (translations NOT yet cached, so those spend translation calls too).
# Override on the CLI with language names (e.g. `python -u run_sentence_compare.py
# smart Fijian Tongan`).
DEFAULT_LANGS = [
    TranslatableLanguage.Fijian,
    TranslatableLanguage.Tongan,
    TranslatableLanguage.Burmese,
]


def _slug(lang):
    # Slug matches what cached translations were stored under: the space-stripped
    # language value (SimplifiedChinese for "Simplified Chinese", Korean, Fijian…).
    return lang.value.replace(" ", "")


def run_segmentation(segmentation, langs):
    """Judge both languages under one segmentation; write a file per language."""
    print(f"==== SEGMENTATION START {segmentation} (sentence judging) ====", flush=True)
    data = evaluate_live_datasets(
        langs,
        evaluation_targets_flash_lite_price_class,
        cache,
        compare_models,
        sermon_ids=SERMONS,
        max_segments=None,
        window=5,
        judging_unit="sentence",
        segmentation=segmentation,
        max_concurrency=GLOBAL_CONCURRENCY,
        language_workers=3,
        max_workers=8,
    )
    for lang, entry in zip(langs, data):
        out_path = f"out_live_{_slug(lang)}_{segmentation}_sentence.json"
        # Wrap in a length-1 list to match the existing out_live_*.json shape that
        # paired_model_stats.py / segmentation_compare.py load via [0].
        with open(out_path, "w") as f:
            f.write(json.dumps([entry], indent=4))
        print(f"WROTE {out_path}", flush=True)


if __name__ == "__main__":
    wanted = [a for a in sys.argv[1:] if a in ("smart", "naive")]
    segmentations = wanted or ["smart", "naive"]
    lang_names = [a for a in sys.argv[1:] if a not in ("smart", "naive")]
    langs = [getattr(TranslatableLanguage, n) for n in lang_names] or DEFAULT_LANGS
    print(f"SEGMENTATIONS={segmentations}  LANGS={[l.value for l in langs]}", flush=True)
    # Each segmentation already saturates the shared cap internally (2 langs × 497
    # sentences × 5 judges fanned through 16 slots), so run them sequentially —
    # avoids two concurrent set_inference_concurrency re-binds briefly
    # over-subscribing the semaphore.
    for seg in segmentations:
        run_segmentation(seg, langs)
    print("\nALL REQUESTED CONDITIONS DONE", flush=True)
