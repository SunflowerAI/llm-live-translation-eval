"""Live, segment-by-segment translation evaluation.

Where ``run_evaluation.py`` treats each sentence as independent and translates
the whole dataset in parallel, live translation is a *stream*: each sermon is an
ordered list of speech segments, and a model translates them one at a time,
seeing its own previous translations as context (see
``generate_translation_prompt`` in ``utils.py``).

That makes each ``(sermon, model)`` chain inherently sequential. So the pipeline
runs in two phases:

  1. **Translate** — for every ``(sermon, model)`` pair, walk the segments in
     order, threading the last ``window`` ``(source, translation)`` pairs forward
     as context. Pairs are parallelised across models and sermons; segments
     within a chain are not (they depend on each other).
  2. **Judge** — once every model has translated a sermon, score the models'
     translations a window of segments at a time (``judge_translations_windowed``,
     ``batch_size`` segments per judge call), each judge giving a brief
     justification per score.

The sermon id is used as the ``sentence_category`` on each ``RankItem``, so the
existing aggregation and ``produce_summary`` work unchanged, with per-sermon
breakdowns available.
"""

import json
import os
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

from typedefinitions import TranslatableLanguage
from utils import get_translation_with_cache_check, set_inference_concurrency
from run_evaluation import judge_translations_windowed


SEGMENTS_DIR = os.path.join(
    os.path.dirname(__file__), "live_test_data", "segments"
)
TRANSCRIPTIONS_DIR = os.path.join(
    os.path.dirname(__file__), "live_test_data", "transcriptions"
)
MANIFEST_PATH = os.path.join(
    os.path.dirname(__file__), "live_test_data", "manifest.json"
)


def _smart_segments(sermon_id):
    """Clause-level segments (the default): the re-segmented ``_segments.json``."""
    path = os.path.join(SEGMENTS_DIR, f"{sermon_id}_segments.json")
    with open(path) as f:
        raw = json.load(f)
    return [s["text"] for s in raw if s.get("text", "").strip()]


def _naive_segments(sermon_id):
    """Naive segments: the raw ASR streaming finals, translated as emitted.

    Each Deepgram streaming "final" result (``final_results[].transcript`` in the
    transcription JSON) is one segment, in order — the un-reprocessed chunks the
    recogniser flushed live, with no clause-aware re-segmentation. The underlying
    words are identical to the smart segments; only the boundaries differ, so a
    smart-vs-naive quality comparison isolates segmentation.
    """
    path = os.path.join(TRANSCRIPTIONS_DIR, f"{sermon_id}_transcription.json")
    with open(path) as f:
        raw = json.load(f)
    return [
        r["transcript"]
        for r in raw.get("final_results", [])
        if r.get("transcript", "").strip()
    ]


def load_live_segments(sermon_ids=None, max_segments=None, segmentation="smart"):
    """Load ordered sermon segments from ``live_test_data``.

    ``sermon_ids`` selects which sermons (by id string); ``None`` loads all.
    ``max_segments`` caps the number of leading segments per sermon (the sermons
    run to hundreds of segments, so a cap keeps a run tractable). ``segmentation``
    picks the boundary strategy: ``"smart"`` (clause-level ``_segments.json``, the
    default) or ``"naive"`` (raw ASR streaming finals). Returns a list of
    ``{"id", "title", "segments": [str, ...]}`` dicts, preserving the order of
    ``sermon_ids`` when given.
    """
    if segmentation not in ("smart", "naive"):
        raise ValueError(f"unknown segmentation {segmentation!r}")
    load_segments = _naive_segments if segmentation == "naive" else _smart_segments

    titles = {}
    try:
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
        titles = {s["id"]: s.get("title", s["id"]) for s in manifest["sermons"]}
    except (OSError, KeyError, json.JSONDecodeError):
        pass

    if sermon_ids is None:
        sermon_ids = sorted(
            fname[: -len("_segments.json")]
            for fname in os.listdir(SEGMENTS_DIR)
            if fname.endswith("_segments.json")
        )

    sermons = []
    for sermon_id in sermon_ids:
        segments = load_segments(sermon_id)
        if max_segments is not None:
            segments = segments[:max_segments]

        sermons.append(
            {
                "id": sermon_id,
                "title": titles.get(sermon_id, sermon_id),
                "segments": segments,
            }
        )

    return sermons


def translate_chain(model, language, sermon_id, segments, cache, window=5):
    """Translate one sermon's segments in order for a single model.

    Threads the last ``window`` successful ``(source, translation)`` pairs forward
    as context for each segment. Returns a list aligned with ``segments`` whose
    entries are the translation text, or ``None`` where the model refused or
    failed (those segments are omitted from the context handed to later ones).
    """
    context = []
    results = []

    for segment in segments:
        window_context = context[-window:]
        translation = get_translation_with_cache_check(
            model,
            TranslatableLanguage.English,
            language,
            sermon_id,
            segment,
            cache,
            context=window_context if window_context else None,
        )
        results.append(translation)
        if translation is not None:
            context.append((segment, translation))

    return results


def compare_live_set(
    language,
    target_models,
    cache,
    compare_models,
    sermons,
    window,
    max_workers,
    batch_size=1,
):
    """Run the translate-then-judge pipeline for one target language."""
    # Phase 1: build every (sermon, model) chain. Chains are sequential inside,
    # but independent across models and sermons, so fan them out.
    chains = {}  # (sermon_id, model.unique_id()) -> [translation | None, ...]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for sermon in sermons:
            for model in target_models:
                future = executor.submit(
                    translate_chain,
                    model,
                    language,
                    sermon["id"],
                    sermon["segments"],
                    cache,
                    window,
                )
                futures[future] = (sermon["id"], model.unique_id())

        for future in futures:
            sermon_id, model_uid = futures[future]
            chains[(sermon_id, model_uid)] = future.result()

    # Phase 2: judge each sermon's segments. The judge scores a segment in
    # context — the preceding source segments AND the single best translation of
    # each (the one the panel rated highest). That "best" comes from judging the
    # earlier segments, so each sermon is a sequential chain; sermons are
    # independent, so they run in parallel.
    def judge_sermon(sermon):
        local_queue = Queue()
        best_context = []  # rolling [(source_segment, best_translation), ...]
        seg_list = sermon["segments"]

        # Judge the segments in windows of ``batch_size``: each judge scores a
        # whole window in one call (see judge_translations_windowed). Windows are
        # sequential because each one's confirmed-best translations feed the next
        # window's context; the segments within a window are judged together.
        for start in range(0, len(seg_list), batch_size):
            batch = []
            for offset in range(start, min(start + batch_size, len(seg_list))):
                translations = {}  # text -> [models that produced it]
                refusals = []
                for model in target_models:
                    translation = chains[(sermon["id"], model.unique_id())][offset]
                    if translation is None:
                        refusals.append(model)
                    else:
                        translations.setdefault(translation, []).append(model)
                batch.append(
                    {
                        "index": offset,
                        "segment": seg_list[offset],
                        "translations": translations,
                        "refusals": refusals,
                    }
                )

            window_context = best_context[-window:]
            best_by_segment = judge_translations_windowed(
                language,
                sermon["id"],
                batch,
                window_context if window_context else None,
                cache,
                compare_models,
                local_queue,
            )
            for item in batch:
                best = best_by_segment.get(item["index"])
                if best is not None:
                    best_context.append((item["segment"], best))

        items = []
        while not local_queue.empty():
            items.append(local_queue.get())
        return items

    output = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for items in executor.map(judge_sermon, sermons):
            output.extend(items)

    return output


def evaluate_live_datasets(
    target_languages,
    target_models,
    cache,
    compare_models,
    sermon_ids=None,
    max_segments=50,
    window=5,
    batch_size=1,
    segmentation="smart",
    max_workers=8,
    max_concurrency=16,
    language_workers=4,
):
    """Live-translation counterpart to ``evaluate_datasets``.

    Translates each sermon's segments sequentially (with rolling context) for
    every model, judges the results per segment, and aggregates per
    ``(sermon, judge)`` exactly like the batch pipeline — sermon ids stand in for
    the sentence categories. Returns one entry per language, each carrying the
    per-model breakdown plus an ``id -> title`` sermon map.

    Concurrency: ``max_concurrency`` is the single cap on simultaneous inference
    (network) calls across the whole run — the rate-limit knob. ``language_workers``
    is how many languages are evaluated at once and ``max_workers`` how many
    translation chains / sermons run at once within a language; both only need to
    be large enough to keep ``max_concurrency`` saturated.
    """
    set_inference_concurrency(max_concurrency)

    sermons = load_live_segments(sermon_ids, max_segments, segmentation)
    sermon_labels = [sermon["id"] for sermon in sermons]

    # Languages are independent; evaluate them concurrently. Every API call still
    # passes through the global inference_slot() cap, so this widens overlap
    # without multiplying provider load.
    def run_language(lang):
        print("Live lang", lang)
        return compare_live_set(
            lang,
            target_models,
            cache,
            compare_models,
            sermons,
            window,
            max_workers,
            batch_size,
        )

    data_by_language = {}
    with ThreadPoolExecutor(max_workers=language_workers) as executor:
        for lang, data in zip(
            target_languages, executor.map(run_language, target_languages)
        ):
            data_by_language[lang] = data

    out = []
    for lang in target_languages:
        data = data_by_language[lang]

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
                    "sentence_types": sermon_labels,
                }
            )

        out.append(
            {
                "language": lang.value,
                "models": model_data,
                "sermons": {sermon["id"]: sermon["title"] for sermon in sermons},
            }
        )

    return out
