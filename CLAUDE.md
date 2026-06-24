# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A benchmark harness that compares **LLM/translation-API translation quality** using **LLMs as judges**. Batch, sentence-level today; intended to be adapted for live/streaming translation later (see README "Toward live translation"). There is no CLI and no test runner — `main.py` *is* the experiment config.

## Running

- Run an experiment: edit `main.py` to select languages + a model roster + a judge panel, then `python main.py`. It writes an `out_*.json`. Historical experiments are kept as commented blocks; an active block runs and a `sys.exit()` guards the rest.
- Self-test the stats model: `python davidson_model.py` (inline asserts; the only thing resembling a test suite).
- Dependencies: `pip install -r requirements.txt`.
- Secrets: create git-ignored `secrets_env.py` defining the `*_API_KEY` constants (`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `MISTRAL_API_KEY`, `COHERE_API_KEY`, `DEEPL_API_KEY`, `LINGVANEX_API_KEY`, `NUENKI_API_KEY`). Imported via `from secrets_env import *`.

## Architecture

Two pipelines share the inference abstraction and cache:

1. **Quality scoring** (`run_evaluation.py`, entry `evaluate_datasets`). For each (language, category, sentence): every tested model translates; identical translations are deduped; each judge LLM scores all distinct translations 0–100 in one prompt (IDs **deterministically shuffled** by content+judge to avoid positional bias); scores become `RankItem`s; aggregated per (category, judge) with mean/median/std and **Mann–Whitney U** p-values.
2. **Coherence / round-trip** (`coherence.py`, entry `inference_coherence_batch`). Repeated EN→target→EN "Chinese whispers" to a depth; judges rate drift from the original; `chart_coherence` plots score-vs-depth.
3. **Live / streaming** (`live_evaluation.py`, entry `evaluate_live_datasets`). Each sermon is an ordered list of speech segments; a model translates them one at a time, seeing its own previous translations as context (`generate_translation_prompt` builds a chat list), so each (sermon, model) chain is sequential. Sermon id stands in for the `sentence_category`, so the same aggregation + `produce_summary` apply. Two **judging units** (`judging_unit` knob): `"segment"` scores each clause-level segment in rolling best-context (`judge_translations_windowed`, sequential per sermon, `batch_size` segments/call); `"sentence"` regroups segments back into source-aligned sentences and judges those **independently and in parallel** via the batch `judge_translations` — see the reconstruction gotcha below. A `segmentation` knob picks `"smart"` (clause-level `_segments.json`) vs `"naive"` (raw ASR finals).

**Inference abstraction** (`typedefinitions.py`): providers implement either `AbstractExecutableTranslator.translate(...)` (model under test) or `AbstractGenericInference.infer(...)` (judge). One concrete class per `*_inference_source.py`. `OpenrouterGenericInference` is the workhorse judge.

**Data & rosters:** `dataset.py` → `SENTENCES_LIST` (category → English strings). `test_data.py` → lists of `TestedEntry` (model + service + temp + thinking; the same model recurs at multiple temps) and judge panels. `main.py` wires a roster + a `compare_models` judge list into `evaluate_datasets`.

**Caching** (`sqlitekv.py`, `cache.db`): thread-safe WAL SQLite KV. **Both translations and judge outputs are cached**, keyed by md5(prompt)+model identity (`utils.py`, `run_evaluation.py`). Re-runs are cheap; changing a prompt invalidates only affected keys. `cache.db` is **git-ignored** (it grew too large to track; it lives only on disk and re-runs rebuild it). `inference_cache.py` is a superseded JSON cache — do not use it.

## Conventions & gotchas

- **Refusal sentinel `483`**: `generate_translation_prompt` tells models to emit `483` to refuse; a `483` or <3-char translation is treated as failure and scored **`-483`** (a magic number that flows into outputs — filter it when analyzing).
- **`TestedEntry` identity** is `unique_id()` (name+company+service+temp+thinking). It is both the cache discriminator and the comparison key — two entries differing only by temp are distinct competitors.
- **`TranslatableLanguage`** (`typedefinitions.py`) is canonical; it carries provider code maps (`deepl_code()`, `nuenki_code()`). Not all providers support all languages — DeepL/Lingvanex rosters are pruned for Welsh/Swahili (see the `_nodeepl_nolingvanex*` roster variants and output files).
- **`DavidsonBT`** (`davidson_model.py`) and `run_evaluation_old.py` are a *legacy pairwise* approach. `DavidsonBT` is still imported in `run_evaluation.py` but unused — current scoring is absolute 0–100, not pairwise. Don't assume the imported `DavidsonBT` is live.
- **Concurrency:** one thread per sentence; retries are 3× with jittered `time.sleep` back-off, plus scattered random sleeps to dodge rate limits. Failures degrade gracefully (skip/continue, print) rather than raising.
- **Judge JSON parsing** reads the *last* triple-backtick block; prompts must keep that contract if edited.
- **Sentence-mode reconstruction (live, `judging_unit="sentence"`):** `group_sentences` (in `live_evaluation.py`) regroups source segments into sentences by terminal `.?!`; `reconstruct_sentence` concatenates a model's segment translations for each sentence (language-aware joiner — empty for CJK/Burmese/Khmer in `_NO_SPACE`, space otherwise incl. Korean). For **smart** segmentation this is *exact*: no segment straddles a sentence boundary (verified across the corpus), so each sentence is a contiguous run of whole segments. For **naive** segmentation ~10% of segments straddle; the straddling target is split at its first interior sentence-final mark (`_split_target`, script-aware marks incl. full-width `。！？`), with a coarse-assign + per-language printed count when the target lacks a usable mark. Sentence-mode `RankItem`s are keyed by sermon id like segment mode, but the pairing unit is the **sentence** — `paired_model_stats.py` only pairs sentence-mode files against other sentence-mode files (and segment- against segment-).
- Committed artifacts (`out_*.json`, `*_old.*`) are intentional for reproducibility — don't "clean them up" without asking. (`cache.db` and `latency.jsonl` are git-ignored — too large to track — and kept only on disk; both were purged from history.)
- **Model-vs-model significance (standing convention, decided 2026-06-22):** to compare two models on an `out_live_*.json` file, use **paired** tests via `paired_model_stats.py`, not the unpaired Mann–Whitney U in `produce_summary` (which understates effects — it leaves the big between-segment/between-judge variance in). Quality: **paired Wilcoxon signed-rank within each judge** (report per-judge results + matched-pairs rank-biserial; the pooled-across-judges row has correlated pairs — read the per-judge rows). Reliability / dropped segments (the `-483` sentinel): **McNemar** on the discordant pairs. Pairing is valid only *within a judge* (segment-aligned by order; no segment ids are stored — never pair across judges by index). For ranking the roster the **flash judge panel is the default** (reproduces the frontier panel's ordering at ~10× less cost, ρ≈0.98); reserve the frontier panel for confirming a finalist or a too-close-to-call pair.

## Project rules in effect

`~/.claude/rules/` apply here: use **Context7 MCP** for any library/API/SDK docs question; for Hobby splines use the pinned upstream `hobby.py`; follow the prose/emphasis writing-style rules for any user-facing copy.
