"""Compare translation quality across segmentation strategies (smart vs naive).

The two strategies translate *different* segment boundaries over identical
underlying words, so segments do NOT align one-to-one and the within-segment
paired test (``paired_model_stats.py``) does not apply across strategies. The
alignable unit here is the **model**: every model is translated under both
strategies and scored by the same judges, so we pair on the model.

Three views, for one language (loads ``out_live_<slug>_{smart,naive}_flash.json``):

  1. RANKING — each model's mean real score (the -483 sentinel excluded) under
     each strategy, side by side, with the rank under each. Answers "does the
     model ordering change when you segment naively?" (Spearman rho between the
     two orderings).

  2. STRATEGY EFFECT (paired by model, within judge) — for each judge, pair the
     8 models' (smart mean, naive mean) and run paired Wilcoxon across the
     roster; report matched-pairs rank-biserial. This holds model and judge
     fixed and isolates the segmentation effect. A pooled-across-judges row is
     shown but its pairs are correlated (same models) — read the per-judge rows.

  3. PER-MODEL (unpaired) — for each model, Mann-Whitney U on its smart vs naive
     segment-score distributions (pooled across judges; harshness is balanced
     since both strategies face the same judges) with Cliff's delta. Shows
     whether the strategy effect is consistent across the roster.

  4. RELIABILITY — dropped-segment rate (fraction scored -483) per model and
     overall, smart vs naive, with a two-proportion (Fisher) test on the totals.

Usage:
    python segmentation_compare.py Korean
    python segmentation_compare.py SimplifiedChinese
"""

import sys

import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu, spearmanr, fisher_exact

from paired_model_stats import load, REFUSAL


def _real(scores):
    a = np.array(scores, float)
    return a[a != REFUSAL]


def _model_means(models, judges, cats):
    """name -> {judge: mean real score over both sermons} and pooled overall."""
    per_judge, overall = {}, {}
    for name, mp in models.items():
        pj = {}
        allreal = []
        for j in judges:
            vals = []
            for c in cats:
                vals += list(_real(mp.get((j, c), [])))
            pj[j] = float(np.mean(vals)) if vals else float("nan")
            allreal += vals
        per_judge[name] = pj
        overall[name] = float(np.mean(allreal)) if allreal else float("nan")
    return per_judge, overall


def _rank_biserial_pairs(s, n):
    diff = np.array(s, float) - np.array(n, float)
    nz = diff[diff != 0]
    if not len(nz):
        return 0.0
    r = np.argsort(np.argsort(np.abs(nz))) + 1
    rp, rn = r[nz > 0].sum(), r[nz < 0].sum()
    return (rp - rn) / (rp + rn) if (rp + rn) else 0.0


def _cliffs_delta(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not len(x) or not len(y):
        return float("nan")
    gt = sum((x[:, None] > y[None, :]).sum(axis=1))
    lt = sum((x[:, None] < y[None, :]).sum(axis=1))
    return (gt - lt) / (len(x) * len(y))


def _jn(j):
    return j.split("/")[-1].replace("-comparison-system", "")


def compare(slug, panel="flash"):
    smodels, sjudges, scats = load(f"out_live_{slug}_smart_{panel}.json")
    nmodels, njudges, ncats = load(f"out_live_{slug}_naive_{panel}.json")
    judges = sorted(set(sjudges) & set(njudges))
    names = [m for m in smodels if m in nmodels]

    print(f"\n################ {slug}: smart vs naive segmentation ################")
    print(f"  models={len(names)}  judges={len(judges)}  "
          f"smart sermons={scats}  naive sermons={ncats}")

    s_pj, s_ov = _model_means(smodels, judges, scats)
    n_pj, n_ov = _model_means(nmodels, judges, ncats)

    # ---- 1. RANKING ----
    order = sorted(names, key=lambda m: s_ov[m], reverse=True)
    print("\n[1] RANKING (mean real score, -483 excluded)")
    print(f"   {'model':<26} {'smart':>7} {'naive':>7} {'Δ(n-s)':>7}   smartRk naiveRk")
    s_rank = {m: i + 1 for i, m in enumerate(sorted(names, key=lambda m: s_ov[m], reverse=True))}
    n_rank = {m: i + 1 for i, m in enumerate(sorted(names, key=lambda m: n_ov[m], reverse=True))}
    for m in order:
        print(f"   {m:<26} {s_ov[m]:>7.1f} {n_ov[m]:>7.1f} {n_ov[m]-s_ov[m]:>+7.1f}"
              f"   {s_rank[m]:>5}   {n_rank[m]:>5}")
    rho, prho = spearmanr([s_rank[m] for m in names], [n_rank[m] for m in names])
    print(f"   Spearman rho(smart-rank, naive-rank) = {rho:+.3f}  (p={prho:.3f})")

    # ---- 2. STRATEGY EFFECT, paired by model within judge ----
    print("\n[2] STRATEGY EFFECT — paired Wilcoxon over models, within judge")
    print("    (pair = one model's mean under smart vs naive; +Δ ⇒ naive higher)")
    pooled_s, pooled_n = [], []
    for j in judges:
        s = [s_pj[m][j] for m in names]
        n = [n_pj[m][j] for m in names]
        pooled_s += s
        pooled_n += n
        diff = np.array(n) - np.array(s)
        try:
            _, p = wilcoxon(n, s, alternative="two-sided", zero_method="wilcox", method="approx")
        except ValueError:
            p = float("nan")
        npos, nneg = int((diff > 0).sum()), int((diff < 0).sum())
        print(f"   {_jn(j):<22} naive>smart:{npos} <:{nneg}  median Δ={np.median(diff):+.1f}"
              f"  rank-biserial={_rank_biserial_pairs(n, s):+.3f}  p={p:.3e} "
              f"{'SIG' if p < 0.05 else 'n.s.'}")
    diff = np.array(pooled_n) - np.array(pooled_s)
    _, p = wilcoxon(pooled_n, pooled_s, alternative="two-sided", zero_method="wilcox", method="approx")
    print(f"   {'POOLED (corr. pairs)':<22} naive>smart:{int((diff>0).sum())} <:{int((diff<0).sum())}"
          f"  median Δ={np.median(diff):+.1f}  rank-biserial={_rank_biserial_pairs(pooled_n, pooled_s):+.3f}"
          f"  p={p:.3e} {'SIG' if p < 0.05 else 'n.s.'}")

    # ---- 3. PER-MODEL unpaired (segment-level) ----
    print("\n[3] PER-MODEL — Mann-Whitney on segment scores (pooled judges), Cliff's δ")
    print("    (δ>0 ⇒ naive segments scored higher for that model)")
    for m in order:
        s_all = np.concatenate([_real(smodels[m].get((j, c), [])) for j in judges for c in scats]) \
            if any(smodels[m].get((j, c)) for j in judges for c in scats) else np.array([])
        n_all = np.concatenate([_real(nmodels[m].get((j, c), [])) for j in judges for c in ncats]) \
            if any(nmodels[m].get((j, c)) for j in judges for c in ncats) else np.array([])
        try:
            _, p = mannwhitneyu(n_all, s_all, alternative="two-sided")
        except ValueError:
            p = float("nan")
        d = _cliffs_delta(n_all, s_all)
        print(f"   {m:<26} smartN={len(s_all):>5} naiveN={len(n_all):>5}"
              f"  meanΔ={n_all.mean()-s_all.mean():+6.1f}  Cliff δ={d:+.3f}  p={p:.2e} "
              f"{'SIG' if p < 0.05 else 'n.s.'}")

    # ---- 4. RELIABILITY (dropped-segment rate) ----
    print("\n[4] RELIABILITY — dropped-segment rate (-483), smart vs naive")
    print(f"   {'model':<26} {'smart drop':>12} {'naive drop':>12}")
    S_drop = S_tot = N_drop = N_tot = 0

    def drops(models, judges, cats, m):
        d = t = 0
        for j in judges:
            for c in cats:
                a = np.array(models[m].get((j, c), []), float)
                d += int((a == REFUSAL).sum())
                t += len(a)
        return d, t
    for m in order:
        sd, st = drops(smodels, judges, scats, m)
        nd, nt = drops(nmodels, judges, ncats, m)
        S_drop += sd; S_tot += st; N_drop += nd; N_tot += nt
        sr = sd / st if st else 0.0
        nr = nd / nt if nt else 0.0
        print(f"   {m:<26} {sd:>5}/{st:<5} {sr:5.1%} {nd:>5}/{nt:<5} {nr:5.1%}")
    _, pf = fisher_exact([[S_drop, S_tot - S_drop], [N_drop, N_tot - N_drop]])
    print(f"   {'TOTAL':<26} {S_drop:>5}/{S_tot:<5} {S_drop/S_tot:5.1%}"
          f" {N_drop:>5}/{N_tot:<5} {N_drop/N_tot:5.1%}   Fisher p={pf:.3e} "
          f"{'SIG' if pf < 0.05 else 'n.s.'}")


if __name__ == "__main__":
    # Optional first arg selects the panel (flash|frontier); rest are languages.
    args = sys.argv[1:]
    panel = "flash"
    if args and args[0] in ("flash", "frontier", "sentence"):
        panel = args.pop(0)
    for slug in (args or ["Korean", "SimplifiedChinese"]):
        compare(slug, panel)
