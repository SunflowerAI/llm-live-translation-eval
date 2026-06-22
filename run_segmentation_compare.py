"""Segmentation A/B: smart (clause-level ``_segments.json``) vs naive (raw ASR
streaming finals) for Korean and Simplified Chinese.

Both strategies run under the *same* config — the new windowed judging + the
flash judge panel (the documented ranking default) — so the four conditions are
directly comparable. The underlying words are identical across the two
segmentations (verified: same per-sermon word/char counts); only the segment
boundaries differ, so a quality comparison isolates segmentation.

Cost note: the smart translations are already cached from the earlier live runs,
so the smart conditions only spend the (cheap) flash judge calls. The naive
segments are new text, so all models re-translate them — that is the dominant
cost. Judging is the flash panel either way.

Writes one file per condition:
    out_live_Korean_smart_flash.json            out_live_Korean_naive_flash.json
    out_live_SimplifiedChinese_smart_flash.json out_live_SimplifiedChinese_naive_flash.json

Usage:
    python -u run_segmentation_compare.py [condition ...]
where each condition is ``<lang>_<seg>`` e.g. ``Korean_naive`` (default: all four,
smart conditions first so the cheap cache-served runs land before the long ones).
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor

from typedefinitions import TranslatableLanguage
from openrouter_inference_source import OpenrouterGenericInference
from secrets_env import *

from sqlitekv import SQLiteKVCache
from test_data import evaluation_targets_flash_lite_price_class
from live_evaluation import evaluate_live_datasets
from utils import set_inference_concurrency

cache = SQLiteKVCache("./cache.db")

# One global cap on simultaneous inference calls, shared across ALL conditions.
# The conditions run concurrently (see __main__) so their judging phases overlap
# and saturate this cap — windowed judging is only ~6-wide within a single
# condition (2 sermons × 3 judges, windows sequential), so running them in
# parallel is what keeps the 16 slots busy.
GLOBAL_CONCURRENCY = 16
set_inference_concurrency(GLOBAL_CONCURRENCY)

# The flash judge panel — exactly the judge names in the existing
# out_live_SimplifiedChinese.json / out_live_Korean_oldmethod_flashpanel.json, so
# the cross-strategy comparison stays on one consistent panel.
compare_models_flash = [
    (
        "google/gemini-2.5-flash-comparison-system",
        OpenrouterGenericInference(OPENROUTER_API_KEY, "google/gemini-2.5-flash"),
    ),
    (
        "meta/llama-4-maverick-comparison-system",
        OpenrouterGenericInference(OPENROUTER_API_KEY, "meta-llama/llama-4-maverick"),
    ),
    (
        "deepseek/deepseek-v3-comparison-system",
        OpenrouterGenericInference(OPENROUTER_API_KEY, "deepseek/deepseek-chat-v3-0324"),
    ),
]

# The frontier (strong non-reasoning) panel — mirrors main.py. Re-judging the
# four cached-translation conditions with this panel confirms the flash-panel
# segmentation conclusions on stronger judges. KO/smart frontier judging is
# already cached (it is out_live_Korean.json), so it costs nothing to reuse.
compare_models_frontier = [
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
        OpenrouterGenericInference(OPENROUTER_API_KEY, "deepseek/deepseek-chat-v3-0324"),
    ),
]

PANELS = {"flash": compare_models_flash, "frontier": compare_models_frontier}

SERMONS = ["1054361551", "1077837019"]

# value used must match the enum the existing smart run cached under: the live
# Chinese run used SimplifiedChinese ("Simplified Chinese"), not Chinese.
LANGS = {
    "Korean": TranslatableLanguage.Korean,
    "SimplifiedChinese": TranslatableLanguage.SimplifiedChinese,
}


def run_condition(lang_slug, segmentation, panel):
    lang = LANGS[lang_slug]
    print(f"==== CONDITION START {lang_slug} / {segmentation} / {panel} ====", flush=True)
    data = evaluate_live_datasets(
        [lang],
        evaluation_targets_flash_lite_price_class,
        cache,
        PANELS[panel],
        sermon_ids=SERMONS,
        max_segments=None,
        window=5,
        batch_size=5,
        segmentation=segmentation,
        max_concurrency=GLOBAL_CONCURRENCY,
        language_workers=1,
    )
    out_path = f"out_live_{lang_slug}_{segmentation}_{panel}.json"
    with open(out_path, "w") as f:
        f.write(json.dumps(data, indent=4))
    print(f"WROTE {out_path}", flush=True)
    return out_path


ALL = [
    ("Korean", "smart"),
    ("SimplifiedChinese", "smart"),
    ("Korean", "naive"),
    ("SimplifiedChinese", "naive"),
]

if __name__ == "__main__":
    # First arg may select the panel (flash|frontier; default flash); any
    # remaining args filter conditions by "<lang>_<seg>".
    args = sys.argv[1:]
    panel = "flash"
    if args and args[0] in PANELS:
        panel = args.pop(0)
    if args:
        wanted = set(args)
        conditions = [(l, s) for (l, s) in ALL if f"{l}_{s}" in wanted]
    else:
        conditions = ALL

    # Run the conditions concurrently so their (sequential, ~6-wide) judging
    # phases overlap and keep the shared GLOBAL_CONCURRENCY cap saturated. Stagger
    # the starts a little so the per-condition concurrency re-set doesn't briefly
    # over-subscribe. batch_size is held at 5, so any judge windows already in the
    # cache from an earlier run are reused rather than recomputed.
    import time

    def staggered(arg):
        i, (lang_slug, segmentation) = arg
        time.sleep(i * 8)
        return run_condition(lang_slug, segmentation, panel)

    print(f"PANEL={panel}  CONDITIONS={conditions}", flush=True)
    with ThreadPoolExecutor(max_workers=len(conditions)) as ex:
        list(ex.map(staggered, enumerate(conditions)))
    print("\nALL REQUESTED CONDITIONS DONE", flush=True)
