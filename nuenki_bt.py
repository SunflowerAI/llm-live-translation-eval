"""Rank a roster with Nuenki's original Bradley-Terry / Davidson methodology.

The original Nuenki pipeline (run_evaluation_old.py + davidson_model.py) judged
translations *pairwise* (win1 / win2 / tie) and fit the Davidson extension of
Bradley-Terry to get per-model strengths and pairwise z-test p-values.

This script applies that SAME model to the absolute-0-100 data we already have,
by DERIVING a pairwise outcome from each (sentence, judge) score pair: model A
beats B on that sentence+judge iff A's score is higher, a tie iff equal. Pairs
where either model dropped the segment (-483) are skipped. Comparisons are pooled
over all sentences and judges (as the original BT fit pooled them), then fit once
per language.

Caveat: this is the BT *statistics* applied to absolute scores, not a true
pairwise re-judge — judges may behave differently when comparing two translations
than when scoring all at once. It throws away score magnitude (a 90-vs-30 win and
a 71-vs-70 win count equally), which is exactly the pairwise philosophy. For a
faithful reproduction, re-judge with pairwise prompts (compare_set in
run_evaluation_old.py) — far more API calls.

Usage:
    python nuenki_bt.py out_live_Korean_smart_sentence.json
    python nuenki_bt.py out_live_{Korean,SimplifiedChinese}_smart_sentence.json
"""

import sys

import numpy as np

from paired_model_stats import load, REFUSAL
from davidson_model import DavidsonBT


def _comparisons(models, judges, cats):
    """Derive pairwise comparisons from absolute scores, tagged by sentence block.

    Within one (judge, cat) the models' score lists are segment-aligned (a judge
    parse-fail drops the segment for all models equally), so index k is the same
    sentence for every model. Pairs are truncated to the shorter list to tolerate
    a rare off-by-one, matching paired_model_stats._align.

    Returns (names, ai, bi, out, block, n_blocks): parallel int arrays where out is
    0=win1 (a>b), 1=win2 (b>a), 2=tie, and ``block`` is the sentence position
    (cat, k) shared across all judges — the cluster unit for the block-bootstrap,
    since the 28 pairwise outcomes from one sentence are highly correlated.
    """
    names = list(models)
    ai, bi, out, block = [], [], [], []
    block_id = {}
    for j in judges:
        for c in cats:
            for a in range(len(names)):
                for b in range(a + 1, len(names)):
                    la, lb = models[names[a]].get((j, c), []), models[names[b]].get((j, c), [])
                    for k in range(min(len(la), len(lb))):
                        sa, sb = la[k], lb[k]
                        if sa == REFUSAL or sb == REFUSAL:
                            continue
                        ai.append(a); bi.append(b)
                        out.append(0 if sa > sb else (1 if sb > sa else 2))
                        block.append(block_id.setdefault((c, k), len(block_id)))
    return (names, np.array(ai), np.array(bi), np.array(out),
            np.array(block), len(block_id))


def _build_wt(ai, bi, out, n):
    """Win and tie count matrices from comparison arrays (a<b always)."""
    win = np.zeros((n, n), int)
    tie = np.zeros((n, n), int)
    w1, w2, tt = out == 0, out == 1, out == 2
    np.add.at(win, (ai[w1], bi[w1]), 1)   # a beats b
    np.add.at(win, (bi[w2], ai[w2]), 1)   # b beats a
    np.add.at(tie, (ai[tt], bi[tt]), 1)
    return win, tie


def _fit_strengths(ai, bi, out, n):
    win, tie = _build_wt(ai, bi, out, n)
    return DavidsonBT().fit(win, tie).get_strengths()


def analyse(path, n_boot=300, seed=0):
    models, judges, cats = load(path)
    names, ai, bi, out, block, n_blocks = _comparisons(models, judges, cats)
    n = len(names)

    strengths = _fit_strengths(ai, bi, out, n)
    order = sorted(range(n), key=lambda i: strengths[i], reverse=True)

    # Block-bootstrap by sentence: resample sentence positions with replacement,
    # refit, and record strengths. This respects the within-sentence correlation
    # the analytic Davidson SE ignores (and which made its CIs uselessly wide).
    members = [np.where(block == b)[0] for b in range(n_blocks)]
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, n))
    for t in range(n_boot):
        pick = rng.integers(0, n_blocks, n_blocks)
        idx = np.concatenate([members[b] for b in pick])
        boot[t] = _fit_strengths(ai[idx], bi[idx], out[idx], n)

    print(f"\n################ {path} — Nuenki Bradley-Terry (Davidson) ################")
    print(f"  models={n}  judges={len(judges)}  sermons={cats}  "
          f"comparisons={len(ai)}  sentence-blocks={n_blocks}  bootstrap={n_boot}")

    print("\n[BT STRENGTHS] derived from absolute scores; 95% block-bootstrap CI")
    print(f"   {'#':>2} {'model':<26} {'strength':>9}  {'95% CI (boot)':>18}")
    for rk, i in enumerate(order):
        lo, hi = np.percentile(boot[:, i] * 100, [2.5, 97.5])
        print(f"   {rk+1:>2} {names[i]:<26} {strengths[i]*100:>8.2f}%  [{lo:>6.2f},{hi:>6.2f}]%")

    # Winner vs each rival: bootstrap significance on the strength gap (fraction of
    # resamples where the winner's strength does NOT exceed the rival's, two-sided).
    w = order[0]
    rank1 = float(np.mean(np.argmax(boot, axis=1) == w))
    print(f"\n[WINNER] {names[w]} — ranked #1 in {rank1:.1%} of bootstrap resamples")
    print(f"   {'rival':<26} {'Δstrength':>10} {'95% CI':>18} {'p_boot':>9}")
    for i in order[1:]:
        d = (boot[:, w] - boot[:, i]) * 100
        lo, hi = np.percentile(d, [2.5, 97.5])
        # two-sided bootstrap p: 2× the smaller tail mass past 0
        frac_le = float(np.mean(d <= 0))
        p = min(1.0, 2 * min(frac_le, 1 - frac_le))
        sig = "SIG" if p < 0.05 else "n.s."
        print(f"   {names[i]:<26} {strengths[w]*100-strengths[i]*100:>+9.2f}%"
              f" [{lo:>6.2f},{hi:>6.2f}] {p:>8.3f} {sig}")


if __name__ == "__main__":
    for path in sys.argv[1:] or ["out_live_Korean_smart_sentence.json"]:
        analyse(path)
