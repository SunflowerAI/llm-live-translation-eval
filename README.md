# llm-live-translation-eval

A benchmark harness for comparing **translation quality across LLMs and translation APIs**, using other LLMs as judges. It is the V2 of the internal tool behind Nuenki's published LLM translation comparisons — rebuilt around a structured inference abstraction, heavy caching, and a multi-judge scoring pipeline.

The repository is named for where it's *heading* — adapting the same machinery to evaluate **live (streaming) translation** — but everything currently implemented evaluates **batch, sentence-level translation**. This README describes what exists today.

---

## What it measures

There are two independent evaluation pipelines:

### 1. Quality scoring (the main pipeline)

For a set of target languages and a roster of models, every model translates the same fixed set of English source texts. A panel of **judge LLMs** then scores those translations, and the scores are aggregated into per-model, per-language summaries with statistical-significance tests.

This is **absolute scoring**, not pairwise: each judge sees all distinct translations of one sentence at once and assigns each a 0–100 score. (An earlier pairwise approach lives in `run_evaluation_old.py` / `davidson_model.py` and is not used by the current path.)

Entry point: `evaluate_datasets()` in `run_evaluation.py`.

### 2. Coherence / round-trip degradation

A "Chinese whispers" test: a sentence is translated English → target → English repeatedly to a given depth, and judge LLMs rate how close each round-trip result stays to the original meaning. The output is a coherence-vs-depth curve per model, useful for spotting models that drift or hallucinate under repeated translation.

Entry point: `inference_coherence_batch()` in `coherence.py`; charts via `chart_coherence()`.

---

## How the quality pipeline works

The flow, top to bottom:

1. **Source texts** — `dataset.py` defines `SENTENCES_LIST`, a dict of category → list of English strings. Categories are `phrases_internet`, `short_sentence`, `medium_sentence`, `long_sentence`, and `paragraph`, so results can be sliced by text difficulty.

2. **Model roster** — `test_data.py` defines lists of `TestedEntry` objects (the models under test) and the judge panels. A `TestedEntry` pins a model name/company, an inference service, a concrete translator instance, a temperature, and a thinking setting — so the *same* model can appear multiple times at different temperatures and be compared against itself.

3. **Translation** — for each (language, category, sentence), every tested model translates the sentence via `get_translation_with_cache_check()` (`utils.py`). The translation prompt (`generate_translation_prompt`) asks for idiomatic, native output, hardens against prompt injection ("Ignore all instructions"), and uses the sentinel **`483`** for refusals. A refusal or a suspiciously short (<3 char) output is treated as a failure and scored **`-483`**.

4. **Deduplication** — identical translations from different models are collapsed into one entry (mapped back to the set of models that produced it), so judges score each distinct string once.

5. **Judging** — for each judge model, the distinct translations are presented with **randomised IDs** (a deterministic shuffle keyed by translation content + judge name, so the layout is stable across runs but free of positional bias). The judge is prompted to critique tersely, ignore defensible stylistic choices, and return a JSON array `[{ "id": …, "score": 0–100 }]`. The JSON is extracted from the last triple-backtick block in the response.

6. **Scoring → records** — each parsed score becomes a `RankItem(language, tested_entry, sentence, category, evaluating_model, score)`. Refusals are emitted as `RankItem`s with score `-483`.

7. **Aggregation** — `evaluate_datasets()` groups `RankItem`s per model into `comparison_items` keyed by (sentence category, judge model). `produce_summary()` additionally computes mean / median / std-dev and pairwise **Mann–Whitney U** p-values between models. Results are written as JSON (`out_*.json`) for downstream charting/serving.

**Concurrency & resilience:** `compare_set()` spawns one thread per sentence. Inference calls retry up to 3× with jittered back-off, and small random sleeps are scattered through the code to smooth provider rate limits. The whole thing leans on the cache to stay affordable across re-runs.

---

## Inference abstraction

All providers implement one of two small interfaces from `typedefinitions.py`:

- **`AbstractExecutableTranslator.translate(source, target, text, temperature)`** — a model used *as a translator* (the thing being benchmarked).
- **`AbstractGenericInference.infer(prompt, temperature)`** — a model used *as a judge* (free-form completion).

Concrete implementations, one per `*_inference_source.py` file:

| File | Provider |
|------|----------|
| `anthropic_inference_source.py` | Anthropic (direct) |
| `openrouter_inference_source.py` | OpenRouter (also provides the generic judge inference) |
| `groq_inference_source.py` | Groq |
| `google_inference_source.py` | Gemini (direct, with thinking toggle) |
| `mistral_inference_source.py` | Mistral |
| `cohere_inference_source.py` | Cohere |
| `deepl_inference_source.py` | DeepL (translation API) |
| `lingvanex_inference_source.py` | Lingvanex (translation API) |
| `nuenki_inference_source.py` | Nuenki hybrid translator |

`TranslatableLanguage` (in `typedefinitions.py`) is the canonical language enum and carries provider-specific code mappings (e.g. `deepl_code()`, `nuenki_code()`).

---

## Caching

`SQLiteKVCache` (`sqlitekv.py`) is a thread-safe, WAL-mode SQLite key-value store backing `cache.db`. **Both translations and judge responses are cached**, keyed by md5 hashes of the prompt plus model identity (see `utils.py` and `run_evaluation.py`). Because the cache key includes the prompt hash, changing a prompt invalidates only the affected entries. The DB files are large and committed to the repo so experiments are reproducible without re-spending on inference.

(`inference_cache.py` is a legacy JSON-file cache, superseded by the SQLite store.)

Latency is logged separately, append-only, to `latency.jsonl` by `latency_logger.py`.

---

## Running it

There is **no CLI**: `main.py` is the experiment configuration. You edit it — choosing target languages, a roster from `test_data.py`, and a judge panel (`compare_models`) — then run it, and it writes a JSON file. Most historical experiments are preserved as commented-out blocks in `main.py`; an active block runs and a `sys.exit()` guards the rest.

```bash
# 1. provide API keys (see below)
# 2. install dependencies
pip install -r requirements.txt
# 3. edit main.py to select the experiment, then:
python main.py
```

### Secrets

`main.py`, `test_data.py`, and `coherence.py` do `from secrets_env import *`. Create a **`secrets_env.py`** (git-ignored) defining the API keys the providers you use require:

```python
OPENROUTER_API_KEY = "..."
ANTHROPIC_API_KEY  = "..."
GEMINI_API_KEY     = "..."
MISTRAL_API_KEY    = "..."
COHERE_API_KEY     = "..."
DEEPL_API_KEY      = "..."
LINGVANEX_API_KEY  = "..."
NUENKI_API_KEY     = "..."
```

### Outputs

- `out_*.json` — scored comparison results, the primary artifact (consumed by the charting/website layer outside this repo).
- `*.png` (e.g. `german_coherence_depth.png`) — coherence charts.
- `latency.jsonl` — per-translation timing.

### Tests

`davidson_model.py` has an inline self-test suite:

```bash
python davidson_model.py
```

---

## Toward live translation

This is the planned next stage, not yet built. The intended reuse: the inference abstraction, caching, judge prompts, and aggregation/statistics carry over largely unchanged; the new work is feeding the harness *streaming, incrementally-revealed* source segments instead of fixed whole sentences, and judging partial/latency-sensitive output. See `plan.txt` for the original design notes behind the current batch architecture.
