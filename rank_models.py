"""Rank the whole roster on a single live-eval output file.

Complements paired_model_stats.py (two named models, head-to-head) and
segmentation_compare.py (one model across two strategies): this ranks ALL models
within one ``out_live_*.json``.

Ranking key is the **mean of each judge's mean** (equal weight per judge, so a
lenient or refusal-light judge can't dominate the order); per-judge means and
per-judge ranks are shown alongside so you can see whether the ordering is robust
across the panel or an artefact of one judge. Scores exclude the -483 refusal
sentinel. The winner vs runner-up gap is then tested with the standing
convention — paired Wilcoxon within judge + McNemar reliability
(paired_model_stats) — since a ranking by means alone says nothing about
significance.

Usage:
    python rank_models.py out_live_Korean_smart_sentence.json
    python rank_models.py out_live_{Korean,SimplifiedChinese}_smart_sentence.json
"""

import sys

import numpy as np

from paired_model_stats import load, REFUSAL, paired_quality, paired_reliability


def _real(scores):
    a = np.array(scores, float)
    return a[a != REFUSAL]


def _jn(j):
    return j.split("/")[-1].replace("-comparison-system", "")


def _per_judge_mean(mp, judges, cats):
    out = {}
    for j in judges:
        vals = []
        for c in cats:
            vals += list(_real(mp.get((j, c), [])))
        out[j] = float(np.mean(vals)) if vals else float("nan")
    return out


def _pooled(mp, judges, cats):
    allv = []
    for j in judges:
        for c in cats:
            allv += list(_real(mp.get((j, c), [])))
    return (float(np.mean(allv)) if allv else float("nan")), len(allv)


def _refusals(mp, judges, cats):
    d = t = 0
    for j in judges:
        for c in cats:
            a = np.array(mp.get((j, c), []), float)
            d += int((a == REFUSAL).sum())
            t += len(a)
    return d, t


def rank(path):
    models, judges, cats = load(path)
    names = list(models)

    pjm = {m: _per_judge_mean(models[m], judges, cats) for m in names}
    pooled = {m: _pooled(models[m], judges, cats) for m in names}
    judge_avg = {m: float(np.nanmean([pjm[m][j] for j in judges])) for m in names}
    order = sorted(names, key=lambda m: judge_avg[m], reverse=True)

    # per-judge ranks, to expose whether the winner wins under every judge
    jrank = {
        j: {m: i + 1 for i, m in enumerate(sorted(names, key=lambda m: pjm[m][j], reverse=True))}
        for j in judges
    }

    print(f"\n################ {path} ################")
    print(f"  models={len(names)}  judges={len(judges)}  sermons={cats}")

    hdr = f"   {'#':>2} {'model':<26}" + "".join(f"{_jn(j)[:10]:>11}" for j in judges)
    hdr += f"{'judgeAvg':>9}{'pooled':>8}{'refus%':>8}"
    print("\n[RANKING] mean real score (−483 excluded); ranked by judgeAvg")
    print(hdr)
    for i, m in enumerate(order):
        row = f"   {i+1:>2} {m:<26}"
        row += "".join(f"{pjm[m][j]:>11.1f}" for j in judges)
        d, _ = pooled[m]
        rd, rt = _refusals(models[m], judges, cats)
        row += f"{judge_avg[m]:>9.1f}{d:>8.1f}{(rd/rt*100 if rt else 0):>7.2f}%"
        print(row)

    # judge harshness (roster mean per judge) — context for the spread above
    print("\n[JUDGE HARSHNESS] roster-mean score each judge gives")
    for j in judges:
        rm = float(np.nanmean([pjm[m][j] for m in names]))
        print(f"   {_jn(j):<22} {rm:6.1f}")

    # winner consistency across judges
    winner = order[0]
    wins = sum(1 for j in judges if jrank[j][winner] == 1)
    print(f"\n[WINNER] {winner} — ranked #1 by {wins}/{len(judges)} judges; "
          f"per-judge rank = {[jrank[j][winner] for j in judges]}")

    # significance of the top gap, per the standing convention
    runner = order[1]
    print(f"\n[TOP GAP] {winner} vs {runner} (paired, the standing convention)")
    paired_quality(models[winner], models[runner], judges, cats, winner, runner)
    paired_reliability(models[winner], models[runner], judges, cats, winner, runner)


if __name__ == "__main__":
    for path in sys.argv[1:] or ["out_live_Korean_smart_sentence.json"]:
        rank(path)
