"""Selection vs no-selection: per-subject Nadeau-Bengio corrected resampled t-tests on fold-wise
differences, and a cross-subject equivalence test (TOST) on subject-level mean differences.
Inputs: the merged Haxby final results and the StarPlus ablation results. Writes selection_stats.json."""
import json, sys
import numpy as np
from scipy import stats

hx = json.load(open(sys.argv[1]))["subjects"]
sp = json.load(open(sys.argv[2]))["subjects"]
MARGIN = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05   # equivalence margin, accuracy units

def nb_test(sel, nos, test_over_train):
    d = np.array(sel) - np.array(nos); J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return float(d.mean()), float("nan"), 1.0
    t = d.mean() / np.sqrt((1 / J + test_over_train) * var)
    return float(d.mean()), float(t), float(2 * stats.t.sf(abs(t), J - 1))

rows, deltas = [], []
for s in sorted(hx, key=int):
    a, b = hx[s]["active_voxel"]["per_fold_scores"], hx[s]["all_vt_baseline"]["per_fold_scores"]
    m, t, p = nb_test(a, b, 1 / (len(a) - 1))
    rows.append({"dataset": "Haxby", "subject": s, "delta": m, "t": t, "p": p}); deltas.append(m)
for s in sorted(sp):
    a, b = sp[s]["selection"]["per_fold_scores"], sp[s]["no_selection"]["per_fold_scores"]
    m, t, p = nb_test(a, b, 1 / 9)
    rows.append({"dataset": "StarPlus", "subject": s, "delta": m, "t": t, "p": p}); deltas.append(m)

d = np.array(deltas); n = len(d); se = d.std(ddof=1) / np.sqrt(n)
t_lo = (d.mean() + MARGIN) / se; t_hi = (d.mean() - MARGIN) / se
p_tost = max(stats.t.sf(t_lo, n - 1), stats.t.cdf(t_hi, n - 1))
ci90 = stats.t.interval(0.90, n - 1, loc=d.mean(), scale=se)
out = {"per_subject": rows, "n_subjects": n, "mean_delta": float(d.mean()), "ci90": [float(c) for c in ci90],
       "equivalence_margin": MARGIN, "p_tost": float(p_tost),
       "n_significant_uncorrected": int(sum(r["p"] < 0.05 for r in rows))}
json.dump(out, open("selection_stats.json", "w"), indent=2)
for r in rows:
    print(f"  {r['dataset']:<8s} {r['subject']:>5s}: delta={100*r['delta']:+5.1f} pts  t={r['t']:+.2f}  p={r['p']:.3f}")
print(f"  mean delta over {n} subjects = {100*d.mean():+.2f} pts, 90% CI [{100*ci90[0]:+.2f}, {100*ci90[1]:+.2f}]")
print(f"  TOST equivalence within +/-{100*MARGIN:.0f} pts: p = {p_tost:.4f}")
