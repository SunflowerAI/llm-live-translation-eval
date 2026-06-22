# Live sermon translation: model choice and segmentation technique

*Benchmark of streaming English→target translation for live church sermons, judged by LLM panels. Korean and Simplified Chinese, June 2026.*

## TL;DR

- **Pick Gemma 4 31B.** It tops the roster of eight Flash‑Lite‑price‑class models in every condition tested — both languages, both judge panels, both segmentation strategies. The ordering of the rest is largely stable, but Gemma's lead is the one result that never moves.
- **Segment at clause boundaries, not raw ASR output.** Translating the recogniser's raw streaming "finals" ("naive" segmentation) costs **~2–3 quality points** across the whole roster versus a clause‑level re‑segmentation ("smart"), and for Chinese it **roughly doubles the dropped‑segment rate**. The penalty is small but utterly consistent — every model, every judge, both languages — and it is free to avoid.
- **The two findings are independent.** Segmentation shifts the whole roster down by a near‑constant amount; it does not change which model wins. So you can choose the model and choose the segmenter separately.

---

## 1. What was measured

Each sermon is an ordered stream of English speech segments. A model translates them one at a time, seeing its own previous five `(source, translation)` pairs as rolling context, exactly as the production interpreter does. Once every model has translated a sermon, a panel of LLM judges scores each candidate translation 0–100 in context.

**The roster under test** (eight models, the Flash‑Lite price class on OpenRouter): Gemma 4 31B, GPT‑4.1 Nano, Hermes 4 70B, Gemini 2.5 Flash Lite, Mistral Small 3.2 24B, Llama 4 Scout, Llama 3.3 70B, Qwen 3 32B.

**The two judge panels:**
- *Flash panel* (the cheap default for ranking): Gemini 2.5 Flash, Llama 4 Maverick, DeepSeek V3.
- *Frontier panel* (stronger non‑reasoning judges, for confirmation): GPT‑4.1, Claude Sonnet 4.6, DeepSeek V3.

**The two segmentation strategies:**
- *Smart* — a clause‑level re‑segmentation (`live_test_data/segments/*_segments.json`): short, syntactically clean units.
- *Naive* — the raw Deepgram streaming finals (`transcriptions/*_transcription.json → final_results`): the chunks the recogniser flushed live, cut wherever the stream happened to break.

**The clean‑comparison control.** The two segmentations cover **identical underlying words** — verified to the character: same 3,497 words for sermon 1054361551 and 2,369 for 1077837019, with normalised text matching exactly. Only the boundaries differ. So a smart‑vs‑naive quality difference isolates segmentation, with transcript content held constant.

Two sermons (1054361551, 1077837019) × 8 models × {smart, naive} × {flash, frontier} = eight scored conditions, ~2,300–2,660 judged scores per model per condition.

**Statistics.** Because the smart and naive segments do not align one‑to‑one, the alignable unit for the cross‑strategy comparison is the *model*: every model is translated under both strategies and scored by the same judges. So the strategy effect is a **paired Wilcoxon signed‑rank test over the eight models, within each judge** (matched‑pairs rank‑biserial as the effect size), supported by a per‑model unpaired Mann–Whitney with Cliff's δ on the segment scores. Dropped‑segment reliability (the `-483` failure sentinel) is compared with Fisher's exact test on the totals.

---

## 2. Model choice

### Rankings under the flash panel (mean rating, 0–100)

| Korean | smart | naive | | Simplified Chinese | smart | naive |
|---|---|---|---|---|---|---|
| **Gemma 4 31B** | **86.8** | **85.9** | | **Gemma 4 31B** | **87.8** | **86.2** |
| GPT 4.1 Nano | 85.0 | 82.9 | | GPT 4.1 Nano | 87.1 | 84.2 |
| Hermes 4 70B | 83.0 | 81.0 | | Mistral Small 3.2 24B | 86.3 | 84.6 |
| Gemini 2.5 Flash Lite | 83.0 | 79.9 | | Hermes 4 70B | 86.2 | 84.4 |
| Mistral Small 3.2 24B | 82.5 | 80.2 | | Llama 3.3 70B | 85.7 | 84.2 |
| Llama 4 Scout | 82.1 | 80.2 | | Gemini 2.5 Flash Lite | 85.4 | 81.5 |
| Llama 3.3 70B | 81.8 | 76.9 | | Qwen 3 32B | 85.3 | 81.9 |
| Qwen 3 32B | 79.3 | 78.0 | | Llama 4 Scout | 76.5 | 73.7 |

### Rankings under the frontier panel (mean rating, 0–100)

| Korean | smart | naive | | Simplified Chinese | smart | naive |
|---|---|---|---|---|---|---|
| **Gemma 4 31B** | **86.2** | **84.3** | | **Gemma 4 31B** | **86.8** | **85.5** |
| GPT 4.1 Nano | 83.0 | 80.4 | | GPT 4.1 Nano | 85.1 | 82.6 |
| Gemini 2.5 Flash Lite | 81.8 | 78.6 | | Mistral Small 3.2 24B | 85.0 | 83.9 |
| Hermes 4 70B | 80.9 | 78.8 | | Qwen 3 32B | 84.8 | 82.2 |
| Mistral Small 3.2 24B | 80.3 | 78.0 | | Gemini 2.5 Flash Lite | 84.8 | 80.5 |
| Llama 4 Scout | 80.3 | 78.5 | | Hermes 4 70B | 84.7 | 83.6 |
| Qwen 3 32B | 77.5 | 76.0 | | Llama 3.3 70B | 84.3 | 83.6 |
| Llama 3.3 70B | 77.2 | 75.2 | | Llama 4 Scout | 76.4 | 73.8 |

**Read these as orderings, not as cross‑panel levels.** The frontier panel is harsher in absolute terms (Claude Sonnet 4.6 pulls levels down), so do not compare a frontier number to a flash number; compare ranks within a panel.

### The recommendation

**Gemma 4 31B is the model to ship.** It is first in all eight columns above, and its margin over the field is the most robust signal in the whole benchmark. The flash panel reproduces the frontier panel's ordering closely on the smart baseline (Spearman ρ ≈ 0.98 in earlier work), so the cheap panel is a sound basis for this choice.

Below first place, read the rankings with appropriate caution:
- **Korean** is stable across panels — GPT 4.1 Nano is a clear second, and the bottom (Qwen, Llama 3.3 70B) is settled.
- **The Chinese mid‑table is a near‑tie** (six models inside 84.3–87.1 under the frontier panel) and reorders under different panels and segmentations. Don't over‑read distinctions inside that cluster.
- **Llama 4 Scout is the clear laggard for Chinese** (~76, ten points below the pack) while sitting mid‑field for Korean — a reminder that the choice is per‑language.

---

## 3. Segmentation technique

### The headline

Naive segmentation lowers translation quality across the whole roster. The roster‑mean rating, smart → naive:

| | Flash panel | Frontier panel |
|---|---|---|
| Korean | 82.9 → 80.6 (**−2.3**) | 80.9 → 78.7 (**−2.2**) |
| Simplified Chinese | 85.0 → 82.6 (**−2.4**) | 84.0 → 82.0 (**−2.0**) |

The two panels agree on the magnitude to within half a point.

### The effect is small but exceptionless

Under the paired‑by‑model test, **every model scores lower under naive segmentation**, in every judge, both languages — 7/8 to 8/8 of the matched pairs negative, driving the matched‑pairs rank‑biserial to its floor of **−1.000** (p ≈ 0.012 per judge; pooled p ≈ 2e‑5). At the segment level, the per‑model Mann–Whitney is significant for all 32 model×language×panel cells, with Cliff's δ between −0.05 and −0.14 (small). Across both panels there is **not a single reversal**: no model, under any judge, is better off with raw ASR segments.

The one place the *size* of the effect wobbles is judge‑dependent: under the frontier panel, **DeepSeek‑as‑judge barely registers the Korean penalty** (median model‑level Δ −0.3, n.s.), though even there the direction stays negative at the segment level. Whether smart segmentation helps is not in doubt; how much it helps depends a little on who is judging.

### Rankings survive the level shift — at the ends

Because naive segmentation moves the whole roster down by a near‑constant amount, the model ordering is largely preserved:

| Spearman ρ (smart‑rank vs naive‑rank) | Flash | Frontier |
|---|---|---|
| Korean | +0.905 | +0.952 |
| Simplified Chinese | +0.833 | **+0.548 (n.s.)** |

Korean is robust either way. The **Chinese mid‑table is sensitive** — under the frontier panel its tight cluster reshuffles enough that the rank correlation is no longer significant (Llama 3.3 70B rises 7→3, Qwen falls 4→6, Gemini Flash Lite 5→7). But **the winner (Gemma) and the laggard (Llama 4 Scout) never move**; only the indistinguishable middle does. The practical reading: don't read fine rank distinctions off naive‑segmented Chinese data.

### Reliability: naive also drops more segments

Dropped‑segment rate (the `-483` refusal/failure sentinel), smart → naive:

| | Flash | Frontier |
|---|---|---|
| Korean | 0.4% → 0.6% (Fisher p=0.038) | 0.4% → 0.6% (p=0.054, borderline) |
| Simplified Chinese | 0.8% → **1.8%** (p=9e‑16) | 0.8% → **1.6%** (p=3e‑16) |

For Korean the reliability cost is real but marginal. For **Chinese it is unambiguous — naive roughly doubles the failure rate** under both panels.

---

## 4. Worked examples — how the same words score differently

The mechanism is visible in the very first line of sermon 1054361551, a scripture announcement: *"So the first reading is two Sam chapter seven one to sixteen, and I'm gonna read that."* The chapter‑and‑verse reference is the load‑bearing content, and the two strategies cut it differently.

### Example A (Korean): a split reference makes a model invent a chapter

**Smart** keeps the whole reference in one segment, so models render it cleanly. Frontier‑panel scores for Mistral Small 3.2 24B:

> **Source (smart segment 0):** *"So the first reading is two Sam chapter seven one to sixteen,"*
> **Mistral Small 3.2 24B →** 첫 번째 읽기는 사무엘하 7장 1절부터 16절까지입니다. *("…2 Samuel chapter 7, verses 1 to 16.")*
> GPT‑4.1 **98** · Sonnet 4.6 **83** · DeepSeek **95**

**Naive** cuts mid‑reference: segment 0 ends at *"…chapter seven"* (the verses gone), and segment 1 becomes *"one to sixteen, and I'm gonna read that."* — a bare number range with no referent. GPT 4.1 Nano, translating that orphaned fragment, **mistook the dangling "one" for a chapter number** and invented one:

> **Source (naive segment 1):** *"one to sixteen, and I'm gonna read that."*
> **GPT 4.1 Nano →** 1장 1절부터 16절까지 읽겠습니다. *("I'll read **chapter 1**, verses 1 to 16.")*
> GPT‑4.1 **60** ("incorrect chapter; '1장' is not in the original") · Sonnet 4.6 **35** · DeepSeek **30**

That error is **structurally impossible under smart segmentation**, where "one to sixteen" never leaves the side of "chapter seven". (Under the flash panel, Gemini 2.5 Flash additionally **failed to return parseable scores for this entire naive window** — a concrete instance of the elevated naive drop rate.)

### Example B (Simplified Chinese): the judges themselves disagree

The same opening, segmented naively into *"So the first reading is two Sam chapter seven"*, shows how widely the *judges* can diverge on one candidate — here Llama 4 Scout, which picked the wrong book (撒母耳记**上** = *1* Samuel, where "two Sam" = 2 Samuel = 撒母耳记**下**):

> **Source (naive segment 0):** *"So the first reading is two Sam chapter seven"*
> **Llama 4 Scout →** 第一篇经文是撒母耳记**上**第七章 *("…**1** Samuel chapter 7")*
> GPT‑4.1 **40** ("incorrect book: 上 instead of 下") · Sonnet 4.6 **80** · DeepSeek **90**

The three judges span 40 to 90 on the *same* translation of the *same* segment. This is the noise that the benchmark averages out over thousands of scores — and the reason the *aggregate*, not any single rating, is what the conclusions rest on (see §5). It is also why DeepSeek‑heavy panels are lenient on exactly this class of proper‑noun error: DeepSeek repeatedly scored wrong‑book renderings 90–100.

### What the examples illustrate

Naive boundaries fall mid‑constituent — splitting a verse reference, orphaning a number range, truncating a clause — and a streaming model translating one fragment at a time has less to work with, so it guesses (a hallucinated chapter) or renders awkwardly, and the judges mark it down. Smart clause‑level units avoid the cut. The same mechanism explains the larger Chinese reliability hit: mid‑clause cuts interact badly with a script that has no word spacing, so more fragments come back unparseable or refused.

---

## 5. Judge behaviour and confidence

- **Per‑item, judges are noisy** (Example B: 40 vs 90 on one translation). They occasionally misread a candidate or attach a wrong reason to a right score. No single rating should be trusted.
- **In aggregate, they agree on what matters.** The flash and frontier panels reach the same model winner, the same segmentation verdict, the same magnitude, and the same reliability story. The flash panel is a faithful ~10×‑cheaper proxy; the frontier panel refined only the Chinese mid‑table rank detail.
- **DeepSeek is the lenient/erratic member** of both panels on proper‑noun accuracy; GPT‑4.1 and Claude Sonnet 4.6 are stricter. Because both panels share DeepSeek, the panels are not fully independent — but their agreement on the headline results, reached from different stricter judges, is reassuring.

---

## 6. Recommendations

1. **Ship Gemma 4 31B** for both Korean and Chinese live translation. It is the robust roster winner; nothing in the data argues for another model.
2. **Segment at clause boundaries** before translating, rather than feeding raw ASR streaming finals to the translator. It buys ~2–3 quality points across the board and halves the Chinese failure rate, at no inference cost.
3. **When sweeping the remaining languages, use the flash panel.** It reproduces the frontier panel's model ranking and its segmentation verdict; reserve the frontier panel for confirming a finalist or settling a too‑close‑to‑call pair.
4. **Treat Chinese mid‑table ranks as a tie.** Distinctions inside the 84–87 cluster are not stable across panels or segmentation and should not drive a decision.

---

## Appendix: reproduction

All eight scored outputs are committed: `out_live_{Korean,SimplifiedChinese}_{smart,naive}_{flash,frontier}.json`.

- **Run the conditions:** `python run_segmentation_compare.py [flash|frontier] [<Lang>_<seg> …]` — runs the four (language × segmentation) conditions concurrently on the chosen panel, reusing cached translations.
- **Cross‑strategy statistics:** `python segmentation_compare.py [flash|frontier] Korean SimplifiedChinese` — ranking + Spearman, paired‑by‑model Wilcoxon within judge, per‑model Mann–Whitney/Cliff's δ, and drop‑rate Fisher.
- **Reconstruct a worked example** (real translations + each judge's score and reason, read from cache): `python report_examples.py <sermon_id> <Korean|SimplifiedChinese> <smart|naive> <window_start> <flash|frontier>`.
- **Two‑model head‑to‑head** (the standing convention): `python paired_model_stats.py <file> "<Model A>" "<Model B>"`.

The segmentation source is selected by the `segmentation="smart"|"naive"` argument to `load_live_segments` / `evaluate_live_datasets` in `live_evaluation.py`. Naive segments are the raw `final_results[].transcript` from the transcription JSON; smart segments are the clause‑level `_segments.json`.
