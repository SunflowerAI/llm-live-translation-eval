"""Quantify three switches from the (Gemini 2.5 Flash Lite, naive) baseline:

  1. segmentation only  -> (Gemini, smart)
  2. model only         -> (Gemma 4 31B, naive)
  3. both               -> (Gemma 4 31B, smart)

Per language, for each switch: change in mean score (accuracy), change in
dropped-sentence rate (the -483 sentinel), and significance.

Test choice follows the alignment: switch 2 stays inside the naive file, where all
models share the same per-(judge,sermon) dropped set, so it is PAIRED (Wilcoxon
within judge for quality; McNemar for drops). Switches 1 and 3 cross the smart and
naive files, which dropped different sentences to judge parse-failures and store no
sentence ids — so they are UNPAIRED (Mann-Whitney + Cliff's delta for quality;
Fisher two-proportion for drops). All comparisons face the same 5 judges on both
sides, so judge harshness is balanced.

Usage: python switch_analysis.py [Korean SimplifiedChinese]
"""

import sys

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon, fisher_exact, binomtest

from paired_model_stats import load, REFUSAL, _align, _rank_biserial

GEMINI = "Gemini 2.5 Flash Lite"
GEMMA = "Gemma 4 31B"


def _real(scores):
    a = np.array(scores, float)
    return a[a != REFUSAL]


def _pool(models, who, judges, cats):
    out = []
    for j in judges:
        for c in cats:
            out += list(_real(models[who].get((j, c), [])))
    return np.array(out)


def _drops(models, who, judges, cats):
    d = t = 0
    for j in judges:
        for c in cats:
            a = np.array(models[who].get((j, c), []), float)
            d += int((a == REFUSAL).sum())
            t += len(a)
    return d, t


def _cliffs(x, y):
    """Cliff's delta of x vs y (>0 => x stochastically larger)."""
    if not len(x) or not len(y):
        return float("nan")
    gt = sum((x[:, None] > y[None, :]).sum(axis=1))
    lt = sum((x[:, None] < y[None, :]).sum(axis=1))
    return (gt - lt) / (len(x) * len(y))


def _unpaired_quality(base, targ):
    """Mann-Whitney + Cliff's delta, target vs baseline (pooled)."""
    try:
        _, p = mannwhitneyu(targ, base, alternative="two-sided")
    except ValueError:
        p = float("nan")
    return targ.mean() - base.mean(), _cliffs(targ, base), p


def _paired_quality(mBase, mTarg, judges, cats):
    """Paired Wilcoxon within judge (real-score pairs); returns per-judge + pooled."""
    perj = _align(mBase, mTarg, judges, cats)
    rows = []
    poolX, poolY = [], []
    for j in judges:
        gb, gt = perj[j]
        mask = (gb != REFUSAL) & (gt != REFUSAL)
        x, y = gt[mask], gb[mask]  # x=target, y=baseline
        try:
            _, p = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox", method="approx")
        except ValueError:
            p = float("nan")
        rows.append((j, x.mean() - y.mean(), _rank_biserial(x - y), p, int(mask.sum())))
        poolX.append(x); poolY.append(y)
    X, Y = np.concatenate(poolX), np.concatenate(poolY)
    return rows, (X.mean() - Y.mean(), _rank_biserial(X - Y), len(X))


def _mcnemar(mBase, mTarg, judges, cats):
    """McNemar on dropped sentences using the judge with most aligned pairs."""
    perj = _align(mBase, mTarg, judges, cats)
    j = max(judges, key=lambda j: len(perj[j][0]))
    gb, gt = perj[j]
    base_only = int(((gb == REFUSAL) & (gt != REFUSAL)).sum())  # baseline drops, target doesn't (good)
    targ_only = int(((gt == REFUSAL) & (gb != REFUSAL)).sum())
    p = binomtest(min(base_only, targ_only), base_only + targ_only, 0.5).pvalue if (base_only + targ_only) else 1.0
    return j, base_only, targ_only, p


def _drop_block(base_d, base_t, models, who, judges, cats, paired=None):
    td, tt = _drops(models, who, judges, cats)
    br, tr = base_d / base_t, td / tt
    rr = tr / br if br else float("nan")  # risk ratio target vs baseline
    line = (f"   drops: {br:.2%} ({base_d}/{base_t}) -> {tr:.2%} ({td}/{tt})"
            f"   reduction {br - tr:+.2%}  RR={rr:.2f}")
    if paired is None:  # unpaired Fisher (odds ratio is its first return value)
        orat, p = fisher_exact([[td, tt - td], [base_d, base_t - base_d]])
        print(line + f"  OR={orat:.2f}   Fisher p={p:.2e} {'SIG' if p < 0.05 else 'n.s.'}")
    else:  # paired McNemar; effect = discordant odds ratio targ-only/base-only
        jn, bo, to, p = paired
        dor = to / bo if bo else float("inf")
        print(line + f"   McNemar({jn.split('/')[-1][:8]}): base-only-drop={bo} targ-only={to}"
              f" discordantOR={dor:.2f} p={p:.3f} {'SIG' if p < 0.05 else 'n.s.'}")


def analyse(lang):
    sm, sj, sc = load(f"out_live_{lang}_smart_sentence.json")
    nm, nj, nc = load(f"out_live_{lang}_naive_sentence.json")
    judges = sorted(set(sj) & set(nj))

    base = _pool(nm, GEMINI, nj, nc)
    bd, bt = _drops(nm, GEMINI, nj, nc)
    print(f"\n################ {lang} ################")
    print(f"  BASELINE: {GEMINI} + naive   mean={base.mean():.1f}  drop={bd/bt:.2%} ({bd}/{bt})")

    # 1. segmentation only (cross-file, unpaired)
    print(f"\n[1] switch to SMART segmentation (still {GEMINI})")
    t = _pool(sm, GEMINI, sj, sc)
    dm, d, p = _unpaired_quality(base, t)
    print(f"   accuracy: mean {base.mean():.1f} -> {t.mean():.1f}  (Δ {dm:+.1f})  Cliff δ={d:+.3f}"
          f"  MannWhitney p={p:.2e} {'SIG' if p < 0.05 else 'n.s.'}")
    _drop_block(bd, bt, sm, GEMINI, sj, sc)

    # 2. model only (within naive file, paired)
    print(f"\n[2] switch to GEMMA 4 31B (still naive)")
    t = _pool(nm, GEMMA, nj, nc)
    rows, (pdm, prb, pn) = _paired_quality(nm[GEMINI], nm[GEMMA], judges, nc)
    sig = sum(1 for _, _, _, pv, _ in rows if pv < 0.05)
    print(f"   accuracy: mean {base.mean():.1f} -> {t.mean():.1f}  (Δ {t.mean()-base.mean():+.1f})")
    print(f"   paired Wilcoxon within judge: SIG {sig}/{len(rows)} judges; "
          f"pooled rank-biserial={prb:+.3f} (n={pn})")
    for j, dmean, rb, pv, npairs in rows:
        print(f"      {j.split('/')[-1].replace('-comparison-system',''):<20} Δ{dmean:+5.1f}"
              f"  rb={rb:+.3f}  p={pv:.2e} {'SIG' if pv < 0.05 else 'n.s.'}")
    _drop_block(bd, bt, nm, GEMMA, nj, nc, paired=_mcnemar(nm[GEMINI], nm[GEMMA], judges, nc))

    # 3. both (cross-file, unpaired)
    print(f"\n[3] switch to GEMMA 4 31B + SMART segmentation")
    t = _pool(sm, GEMMA, sj, sc)
    dm, d, p = _unpaired_quality(base, t)
    print(f"   accuracy: mean {base.mean():.1f} -> {t.mean():.1f}  (Δ {dm:+.1f})  Cliff δ={d:+.3f}"
          f"  MannWhitney p={p:.2e} {'SIG' if p < 0.05 else 'n.s.'}")
    _drop_block(bd, bt, sm, GEMMA, sj, sc)


if __name__ == "__main__":
    for lang in sys.argv[1:] or ["Korean", "SimplifiedChinese"]:
        analyse(lang)
