"""Tiny smoke test for the windowed live-judging path + the new judge panel.

Runs one sermon, the first few segments, with batch_size=2, using the upgraded
non-reasoning judge panel. Confirms the new slugs resolve and the windowed JSON
contract parses before committing to a full run. Writes a separate output file
so it never touches out_live_Korean.json. Reuses cache.db, so the first few
segments (already translated by the full Korean run) are served from cache and
only the new windowed judge calls hit the network.
"""

from typedefinitions import TranslatableLanguage
from openrouter_inference_source import OpenrouterGenericInference
from secrets_env import *

from sqlitekv import SQLiteKVCache
from test_data import evaluation_targets_flash_lite_price_class
from live_evaluation import evaluate_live_datasets

import json

cache = SQLiteKVCache("./cache.db")

# The new non-reasoning frontier panel (mirrors main.py).
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
        OpenrouterGenericInference(OPENROUTER_API_KEY, "deepseek/deepseek-chat-v3-0324"),
    ),
]

data = evaluate_live_datasets(
    [TranslatableLanguage.Korean],
    evaluation_targets_flash_lite_price_class,
    cache,
    compare_models,
    sermon_ids=["1054361551"],
    max_segments=6,
    window=5,
    batch_size=2,
    max_concurrency=8,
    language_workers=1,
)

with open("out_live_Korean_smoke.json", "w") as f:
    f.write(json.dumps(data, indent=4))

print("SMOKE TEST DONE -> out_live_Korean_smoke.json")
