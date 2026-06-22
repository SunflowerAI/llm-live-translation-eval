"""Standing convention for comparing two models on a live-eval output.

Decided 2026-06-22: when comparing two models head-to-head on an
``out_live_*.json`` file, use **paired** tests, not the unpaired Mann-Whitney U
that ``run_evaluation.produce_summary`` computes. Pairing each model's score for
the *same segment, scored by the same judge* cancels the large between-segment
and between-judge variance that otherwise swamps the signal — in practice this
upgraded the Gemma-vs-Gemini quality gap from "significant but negligible"
(unpaired Cliff's d ~0.08) to a small/medium matched-pairs effect (rank-biserial
0.18-0.42) that every judge agreed on.

  - Quality (0-100 scores): paired Wilcoxon signed-rank, *within each judge*.
    Report per-judge results (so cross-judge harshness can't pseudo-replicate)
    plus matched-pairs rank-biserial as the effect size. A pooled-across-judges
    row is shown for convenience but its pairs are correlated — read the
    per-judge rows for inference.
  - Reliability (dropped segments = the -483 refusal/failure sentinel): McNemar
    on the discordant pairs (segments where exactly one model dropped). Wilcoxon
    does not apply to a binary drop/no-drop outcome.

Alignment: within one judge, each model has exactly one score per segment in
segment order, and a judge's parse-failure drops that segment for *all* models
equally, so two models' per-judge lists stay segment-aligned. (The output stores
no segment ids, so this within-judge ordering is the *only* valid way to pair;
do not pair across judges by index.) A rare off-by-one between two models on a
single judge/sermon is tolerated by truncating to the shorter list.

Usage:
    python paired_model_stats.py out_live_Korean.json "Gemma 4 31B" "Gemini 2.5 Flash Lite"
"""

import json
import sys

import numpy as np
from scipy.stats import wilcoxon, binomtest

REFUSAL = -483  # dropped-segment sentinel (see CLAUDE.md)


def load(path):
    """Return (models, judges, cats): models maps name -> {(judge, cat): [scores]}."""
    d = json.load(open(path))[0]
    models = {}
    for m in d["models"]:
        mp = {}
        for ci in m["comparison_items"]:
            mp[(ci["evaluating_model"], ci["sentence_category"])] = list(ci["scores"])
        models[m["model"]["name"]] = mp
    judges = sorted({k[0] for mp in models.values() for k in mp})
    cats = sorted({k[1] for mp in models.values() for k in mp})
    return models, judges, cats


def _align(mA, mB, judges, cats):
    """Per-judge aligned score arrays for two models. Returns {judge: (a, b)}."""
    perj = {}
    for j in judges:
        a, b = [], []
        for c in cats:
            la, lb = mA.get((j, c), []), mB.get((j, c), [])
            n = min(len(la), len(lb))  # tolerate a rare off-by-one
            a += la[:n]
            b += lb[:n]
        perj[j] = (np.array(a, float), np.array(b, float))
    return perj


def _rank_biserial(diff):
    nz = diff[diff != 0]
    if not len(nz):
        return 0.0
    r = np.argsort(np.argsort(np.abs(nz))) + 1
    rp, rn = r[nz > 0].sum(), r[nz < 0].sum()
    return (rp - rn) / (rp + rn) if (rp + rn) else 0.0


def paired_quality(mA, mB, judges, cats, nameA="A", nameB="B"):
    """Paired Wilcoxon per judge (real scores only) + a pooled row."""
    perj = _align(mA, mB, judges, cats)
    pooled = []
    print(" QUALITY (paired Wilcoxon, within judge):")
    for j in judges:
        ga, gb = perj[j]
        mask = (ga != REFUSAL) & (gb != REFUSAL)
        x, y = ga[mask], gb[mask]
        diff = x - y
        npos, nneg, ntie = int((diff > 0).sum()), int((diff < 0).sum()), int((diff == 0).sum())
        try:
            _, p = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox", method="approx")
        except ValueError:
            p = float("nan")
        jn = j.split("/")[-1].replace("-comparison-system", "")
        print(
            f"   {jn:<22} pairs={mask.sum():>5}  {nameA}>{nameB}:{npos} <:{nneg} =:{ntie}"
            f"  rank-biserial={_rank_biserial(diff):+.3f}  p={p:.2e} {'SIG' if p < 0.05 else 'n.s.'}"
        )
        pooled.append((x, y))
    X = np.concatenate([a for a, _ in pooled])
    Y = np.concatenate([b for _, b in pooled])
    _, p = wilcoxon(X, Y, alternative="two-sided", zero_method="wilcox", method="approx")
    print(
        f"   {'POOLED (corr. pairs)':<22} pairs={len(X):>5}  median diff={np.median(X - Y):+.1f}"
        f"  rank-biserial={_rank_biserial(X - Y):+.3f}  p={p:.2e} {'SIG' if p < 0.05 else 'n.s.'}"
    )


def paired_reliability(mA, mB, judges, cats, nameA="A", nameB="B"):
    """McNemar on dropped segments, using the judge with the most aligned segments."""
    perj = _align(mA, mB, judges, cats)
    j = max(judges, key=lambda j: len(perj[j][0]))
    ga, gb = perj[j]
    b = int(((ga == REFUSAL) & (gb != REFUSAL)).sum())
    c = int(((gb == REFUSAL) & (ga != REFUSAL)).sum())
    both = int(((ga == REFUSAL) & (gb == REFUSAL)).sum())
    p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else 1.0
    jn = j.split("/")[-1].replace("-comparison-system", "")
    print(" RELIABILITY (McNemar on dropped segments):")
    print(
        f"   judge={jn} ({len(ga)} segs): {nameA}-only-drop={b}  {nameB}-only-drop={c}"
        f"  both={both}  p={p:.3f} {'SIG' if p < 0.05 else 'n.s.'}"
    )


if __name__ == "__main__":
    path, nameA, nameB = sys.argv[1], sys.argv[2], sys.argv[3]
    models, judges, cats = load(path)
    print(f"==== {path} : {nameA} vs {nameB} ====")
    paired_quality(models[nameA], models[nameB], judges, cats, nameA, nameB)
    paired_reliability(models[nameA], models[nameB], judges, cats, nameA, nameB)
