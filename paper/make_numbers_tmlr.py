"""Macros for the TMLR manuscript that are not in numbers_biorxiv.tex: the closed-form selection-bias
check (selection_bias_theory.json) and the selector / classifier robustness analysis
(sel_robust_{a,b,c}.json). Writes numbers_tmlr.tex (one macro per line).
Usage: python make_numbers_tmlr.py <extras_dir>"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats
from scipy.stats import norm

EXTRAS = sys.argv[1]
macros = {}


def pts(x):
    return f"{100 * x:.1f}"


def spts(x):
    """Signed difference in percentage points, math mode."""
    return f"${100 * x:+.1f}$"


def predicted(n, p, k, folds=10):
    q = k / (2.0 * p)
    c = norm.isf(q)
    m = 1.0 + c * norm.pdf(c) / q
    n_t, n_v = n * (folds - 1) / folds, n / folds
    return norm.cdf(np.sqrt(k) * (m / n - 1.0 / n_t) / np.sqrt(m / n + n_v / n_t ** 2))


th = json.load(open(os.path.join(EXTRAS, "selection_bias_theory.json")))
rows = th["rows"]
macros["thNconf"] = str(len(rows))
macros["thSeeds"] = str(th["seeds"])
macros["thMaxErrCent"] = pts(th["summary"]["max_abs_err_centroid"])
macros["thMeanErrCent"] = pts(th["summary"]["mean_abs_err_centroid"])
macros["thMaxErrLR"] = pts(th["summary"]["max_abs_err_logreg"])
lo, hi = th["summary"]["nested_range"]
macros["thNestedRange"] = f"{lo:.3f}--{hi:.3f}"
big = next(r for r in rows if (r["n"], r["p"], r["k"]) == (320, 5000, 100))
macros["thExampleLarge"] = f"{big['leak_centroid']:.3f}"
macros["thStarplus"] = f"{predicted(80, 5000, 100):.3f}"

# ---- selector / classifier robustness -------------------------------------------------------------
subj, versions = {}, set()
for path in sorted(glob.glob(os.path.join(EXTRAS, "sel_robust_*.json"))):
    d = json.load(open(path))
    subj.update(d["subjects"])
    versions.add((d["versions"]["scikit-learn"], d["versions"]["nilearn"]))
CONTROLS = [("none", "None (all voxels)"), ("sumf", "Summed one-vs-rest $F$ (Eq.~\\ref{eq:score})"),
            ("anova", "Multi-class ANOVA $F$"), ("random", "Random voxels"),
            ("variance", "Highest-variance voxels"), ("pca", "Principal components")]
CLFS = [("logreg", "Logistic regression"), ("linsvm", "Linear SVM"), ("rbfsvm", "Gaussian-kernel SVM"),
        ("rbftuned", "Gaussian-kernel SVM, tuned")]
ARMS = [(m, "logreg") for m, _ in CONTROLS] + [("none", "linsvm"), ("sumf", "linsvm"),
                                               ("none", "rbfsvm"), ("sumf", "rbfsvm"),
                                               ("none", "rbftuned"), ("sumf", "rbftuned")]
# Gaussian-kernel SVM with C and kernel width tuned in the inner loop (haxby_rbf_tuned.py)
tuned = {}
for path in sorted(glob.glob(os.path.join(EXTRAS, "rbf_tuned_*.json"))):
    d = json.load(open(path))
    tuned.update(d["subjects"])
    versions.add((d["versions"]["scikit-learn"], d["versions"]["nilearn"]))
    tuned_grid = (d["c_grid"], d["g_grid"])
LABEL = dict(CONTROLS)
CLF_LABEL = dict(CLFS)
complete = (len(subj) == 6 and all(len(s["arms"]) == 20 for s in subj.values())
            and len(tuned) == 6 and all(len(s["arms"]) == 8 for s in tuned.values()))
if complete:
    ids = sorted(subj, key=int)

    def arm(pool, method, clf):
        if clf == "rbftuned":
            return np.array([tuned[s]["arms"][f"{pool}/{method}/tuned"]["mean"] for s in ids])
        return np.array([subj[s]["arms"][f"{pool}/{method}/{clf}"]["mean"] for s in ids])

    def diff(pool, method, clf):
        return arm(pool, method, clf) - arm(pool, "none", clf)

    def n_pos(d):
        """Subjects with a positive difference at the reported precision (0.1 point)."""
        return int((np.round(100 * d, 1) > 0).sum())

    def tost(d, margin):
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n)
        return max(stats.t.sf((d.mean() + margin) / se, n - 1), stats.t.cdf((d.mean() - margin) / se, n - 1))

    def cells(pool, method, clf):
        a = arm(pool, method, clf)
        if method == "none":
            return f"{a.mean():.3f} & --"
        d = diff(pool, method, clf)
        return f"{a.mean():.3f} & {spts(d.mean())} ({n_pos(d)}/6)"

    lines, prev = [], None
    for method, clf in ARMS:
        head = CLF_LABEL[clf] if clf != prev else ""
        if prev is not None and clf != prev:
            lines[-1] += " \\addlinespace"
        prev = clf
        lines.append(f"{head} & {LABEL[method]} & {cells('wb', method, clf)} & {cells('vt', method, clf)} \\\\")
    macros["rbRows"] = " ".join(lines)

    # per-subject table (appendix)
    ps = []
    for pool, title in (("wb", "Whole brain"), ("vt", "Prior (VT)")):
        ps.append(f"\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\")
        for method, clf in ARMS:
            a = arm(pool, method, clf)
            lab = LABEL[method].replace(" (Eq.~\\ref{eq:score})", "").replace(" (all voxels)", "")
            ps.append(f"{CLF_LABEL[clf]} & {lab} & " + " & ".join(f"{x:.3f}" for x in a) + " \\\\")
        if pool == "wb":
            ps[-1] += " \\addlinespace"
    macros["rbSubjRows"] = " ".join(ps)

    name = {"sumf": "Sumf", "anova": "Anova", "random": "Random", "variance": "Variance", "pca": "Pca"}
    for pool in ("wb", "vt"):
        P = pool.upper()
        macros[f"rbNone{P}"] = f"{arm(pool, 'none', 'logreg').mean():.3f}"
        for m in name:
            d = diff(pool, m, "logreg")
            macros[f"rb{name[m]}{P}delta"] = spts(d.mean())
            macros[f"rb{name[m]}{P}abs"] = pts(abs(d.mean()))
            macros[f"rb{name[m]}{P}pos"] = str(n_pos(d))
        for c, cn in (("linsvm", "Linsvm"), ("rbfsvm", "Rbf"), ("rbftuned", "Tun")):
            d = diff(pool, "sumf", c)
            macros[f"rb{cn}{P}none"] = f"{arm(pool, 'none', c).mean():.3f}"
            macros[f"rb{cn}{P}sel"] = f"{arm(pool, 'sumf', c).mean():.3f}"
            macros[f"rb{cn}{P}delta"] = spts(d.mean())
            macros[f"rb{cn}{P}pos"] = str(n_pos(d))

    # within the prior: linear classifiers
    lin = {("sumf", "logreg"): diff("vt", "sumf", "logreg"), ("anova", "logreg"): diff("vt", "anova", "logreg"),
           ("random", "logreg"): diff("vt", "random", "logreg"), ("sumf", "linsvm"): diff("vt", "sumf", "linsvm"),
           ("variance", "logreg"): diff("vt", "variance", "logreg"), ("pca", "logreg"): diff("vt", "pca", "logreg")}
    means = [d.mean() for d in lin.values()]
    macros["rbVTlinRange"] = f"{100 * abs(max(means)):.1f} to {100 * abs(min(means)):.1f}"
    assert max(means) < 0, "expected every reduction to lower the VT mean"
    tight = [("sumf", "logreg"), ("anova", "logreg"), ("random", "logreg"), ("sumf", "linsvm")]
    loose = [("variance", "logreg"), ("pca", "logreg")]
    macros["rbTostThreeMax"] = f"{max(tost(100 * lin[a], 3) for a in tight):.3f}"
    assert max(tost(100 * lin[a], 3) for a in tight) < 0.05
    macros["rbTostThreeVariance"] = f"{tost(100 * lin[('variance', 'logreg')], 3):.2f}"
    macros["rbTostThreePca"] = f"{tost(100 * lin[('pca', 'logreg')], 3):.2f}"
    macros["rbTostFiveLoose"] = f"{max(tost(100 * lin[a], 5) for a in loose):.3f}"
    assert min(tost(100 * lin[a], 3) for a in loose) > 0.05 > max(tost(100 * lin[a], 5) for a in loose)
    d = 100 * diff("vt", "sumf", "logreg")
    ci = stats.t.interval(0.90, len(d) - 1, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(len(d)))
    macros["rbSumfVTci"] = f"${ci[0]:+.1f}$ to ${ci[1]:+.1f}$"

    # Gaussian-kernel SVM within the prior: who gains
    d = np.round(100 * diff("vt", "sumf", "rbfsvm"), 1)
    gain = d[d > 1.0]
    macros["rbRbfVTgainN"] = str(len(gain))
    macros["rbRbfVTgainRange"] = f"{gain.min():.1f}--{gain.max():.1f}"
    macros["rbRbfVTflatN"] = str(int((np.abs(d) <= 1.0).sum()))
    assert len(gain) + int((np.abs(d) <= 1.0).sum()) == 6
    macros["rbRbfVTtostThree"] = f"{tost(100 * diff('vt', 'sumf', 'rbfsvm'), 3):.2f}"
    macros["rbRbfVTdeficit"] = pts(arm("vt", "none", "logreg").mean() - arm("vt", "none", "rbfsvm").mean())

    # tuned Gaussian-kernel SVM: reproduction of the default arm, effect of tuning, selection within the prior
    repro = [abs(tuned[s]["arms"][f"{pool}/{m}/default"]["mean"] - subj[s]["arms"][f"{pool}/{m}/rbfsvm"]["mean"])
             for s in ids for pool in ("wb", "vt") for m in ("none", "sumf")]
    macros["rbTunReproMax"] = pts(max(repro))
    assert max(repro) < 0.01, max(repro)
    d = 100 * diff("vt", "sumf", "rbftuned")
    ci = stats.t.interval(0.90, len(d) - 1, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(len(d)))
    macros["rbTunVTci"] = f"${ci[0]:+.1f}$ to ${ci[1]:+.1f}$"
    macros["rbTunVTtostThree"] = f"{tost(d, 3):.3f}"
    macros["rbTunVTtostFive"] = f"{tost(d, 5):.3f}"
    macros["rbTunVTrange"] = f"${d.min():+.1f}$ to ${d.max():+.1f}$"
    macros["rbTunVTgain"] = pts(arm("vt", "none", "rbftuned").mean() - arm("vt", "none", "rbfsvm").mean())
    macros["rbTunVTvsLogreg"] = spts(arm("vt", "none", "rbftuned").mean() - arm("vt", "none", "logreg").mean())
    macros["rbTunWBgain"] = pts(arm("wb", "none", "rbftuned").mean() - arm("wb", "none", "rbfsvm").mean())
    chosen = [c for s_ in ids for a_ in ("none", "sumf") for c in tuned[s_]["arms"][f"vt/{a_}/tuned"]["chosen"]]
    macros["rbTunWideN"] = str(sum(c["g"] == min(tuned_grid[1]) for c in chosen))
    macros["rbTunFoldsN"] = str(len(chosen))
    macros["rbTunCgrid"] = "\\{" + ",".join(f"{c:g}" for c in tuned_grid[0]) + "\\}"
    macros["rbTunGgrid"] = "\\{" + ",".join(f"{g:g}" for g in tuned_grid[1]) + "\\}"
    print("tuned RBF, VT: none", arm("vt", "none", "rbftuned").round(3), "sel", arm("vt", "sumf", "rbftuned").round(3),
          "diff", d.round(1), "TOST3", tost(d, 3), "TOST5", tost(d, 5), file=sys.stderr)
    print("tuned RBF, WB: none", arm("wb", "none", "rbftuned").round(3), "sel", arm("wb", "sumf", "rbftuned").round(3),
          file=sys.stderr)

    # the two label-based rankings against each other
    for pool in ("wb", "vt"):
        d = 100 * (arm(pool, "anova", "logreg") - arm(pool, "sumf", "logreg"))
        macros[f"rbAnovaVsSumf{pool.upper()}"] = f"${d.mean():+.1f}$"
    assert len(versions) == 1
    sk, nl = versions.pop()
    macros["rbVersions"] = f"\\texttt{{scikit-learn}} {sk} / \\texttt{{nilearn}} {nl}"
    macros["rbKpca"] = "\\{20,50,100,200,400\\}"
else:
    print(f"robustness incomplete: {len(subj)} subjects, tuned {len(tuned)}", file=sys.stderr)

with open("numbers_tmlr.tex", "w", encoding="utf-8", newline="\n") as f:
    f.write("% AUTO-GENERATED by make_numbers_tmlr.py -- do not edit by hand\n")
    for k, v in macros.items():
        assert "\n" not in v
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(macros)} macros; robustness complete: {complete}")
