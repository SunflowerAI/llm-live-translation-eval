"""Reconstruct a single judged window (smart vs naive) from the cache, to quote
real translations + the judges' actual scores and reasons in the report.

No new API calls: every model's translation is already cached, and each judge's
raw scored response is cached under the windowed-judge prompt hash. We rebuild
the exact prompt the run used (verbatim template from
run_evaluation.judge_translations_windowed), recompute its md5 key, read the
cached judge response, and print it. A cache MISS here means the rebuilt prompt
didn't match — treated as an assertion failure, not silently skipped.

Usage:
    python report_examples.py <sermon_id> <Korean|SimplifiedChinese> <smart|naive> <window_start> <panel>
"""

import sys

from typedefinitions import TranslatableLanguage
from secrets_env import *
from sqlitekv import SQLiteKVCache
from test_data import evaluation_targets_flash_lite_price_class
from live_evaluation import load_live_segments, translate_chain
from run_evaluation import deterministic_sample, _extract_json_from_response
from utils import md5hash

cache = SQLiteKVCache("./cache.db")

LANGS = {
    "Korean": TranslatableLanguage.Korean,
    "SimplifiedChinese": TranslatableLanguage.SimplifiedChinese,
}
PANELS = {
    "flash": [
        "google/gemini-2.5-flash-comparison-system",
        "meta/llama-4-maverick-comparison-system",
        "deepseek/deepseek-v3-comparison-system",
    ],
    "frontier": [
        "openai/gpt-4.1-comparison-system",
        "anthropic/claude-sonnet-4.6-comparison-system",
        "deepseek/deepseek-v3-comparison-system",
    ],
}

JSON_EXAMPLE = """```
[
{ "segment": [SEGMENT], "id": [TRANSLATION_ID], "score": [SCORE_0_TO_100], "reason": "[ONE TERSE CLAUSE]" },
{ "segment": [SEGMENT], "id": [ANOTHER_ID], "score": [SCORE_0_TO_100], "reason": "[ONE TERSE CLAUSE]" }
]
```"""


def build_window(sermon_id, language, segmentation, start, size=5, window=5):
    """Replay the cached translations and assemble the window-`start` batch."""
    sermon = load_live_segments([sermon_id], None, segmentation)[0]
    seg_list = sermon["segments"]
    models = evaluation_targets_flash_lite_price_class
    # Each model's full translation chain (cache hits) — aligned to seg_list.
    chains = {
        m.unique_id(): translate_chain(
            m, language, sermon_id, seg_list, cache, window=window
        )
        for m in models
    }
    batch = []
    for idx in range(start, min(start + size, len(seg_list))):
        translations = {}
        refusals = []
        for m in models:
            t = chains[m.unique_id()][idx]
            if t is None:
                refusals.append(m)
            else:
                translations.setdefault(t, []).append(m)
        batch.append(
            {"index": idx, "segment": seg_list[idx],
             "translations": translations, "refusals": refusals}
        )
    return batch


def rebuild_prompt(scorable, language, comparison_model):
    """Verbatim copy of the prompt builder in judge_translations_windowed."""
    lookup = {}
    segments_block = ""
    for s in scorable:
        seg_index = s["index"]
        translations = s["translations"]
        determ_key = "|".join(sorted(translations)) + "-" + str(seg_index) + "-" + comparison_model
        lookup[seg_index] = {}
        segments_block += (
            f"=== Segment {seg_index} ===\n"
            f"Original:\n```{s['segment']}```\n"
            f"Translations into `{language.value}`:\n"
        )
        for local_id, text in deterministic_sample(sorted(translations), determ_key):
            lookup[seg_index][local_id] = (text, translations[text])
            segments_block += f"(segment {seg_index}, ID {local_id}):\n```{text}```\n"
        segments_block += "\n"

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

The segments below are consecutive and in order. Score every candidate translation of every segment 0-100. Use the full range, including <40 for poor translations. Give each score a single terse justification clause — this enforces rigour; keep it short to save tokens.

{segments_block}When you're ready, write out a single triple-backtick code block with a JSON array — one object per (segment, translation) — in the following form:
{JSON_EXAMPLE}

This will be parsed, so keep your output exact: emit exactly one JSON code block as the LAST thing in your response, with a "reason" for every entry. Remember the triple-backticks and correct format!
"""
    return comparison_prompt, lookup


def short(name):
    return name.split("/")[-1].replace("-comparison-system", "")


def main(sermon_id, lang_slug, segmentation, start, panel):
    language = LANGS[lang_slug]
    batch = build_window(sermon_id, language, segmentation, int(start))
    scorable = [s for s in batch if s["translations"]]

    # Gather each judge's per-(segment,id) score+reason from cache.
    judged = {}  # comparison_model -> {(seg,id): (score, reason)}
    lookups = {}
    for cm in PANELS[panel]:
        prompt, lookup = rebuild_prompt(scorable, language, cm)
        lookups[cm] = lookup
        key = f"WINDOWCOMPARISON hash:{md5hash(prompt)} Model:{cm}"
        raw = cache.get(key)
        assert raw, f"CACHE MISS for {short(cm)} — rebuilt prompt mismatch"
        data = _extract_json_from_response(raw)
        judged[cm] = {(int(d["segment"]), int(d["id"])): (d.get("score"), d.get("reason", ""))
                      for d in (data or [])}

    print(f"\n######## {lang_slug} / {segmentation} / sermon {sermon_id} / window@{start} / {panel} ########")
    for s in scorable:
        si = s["index"]
        print(f"\n— Segment {si}: «{s['segment']}»")
        # canonical id order from the first judge's lookup
        cm0 = PANELS[panel][0]
        for lid, (text, models) in sorted(lookups[cm0][si].items()):
            tag = ", ".join(sorted(m.model_name.value for m in models))
            print(f"   [{tag}] → «{text}»")
            for cm in PANELS[panel]:
                sc, rs = judged[cm].get((si, lid), ("?", ""))
                print(f"       {short(cm):<18} {sc!s:>4}  — {rs}")


if __name__ == "__main__":
    main(*sys.argv[1:])
