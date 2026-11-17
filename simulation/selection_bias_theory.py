"""
Closed-form approximation of the accuracy that selection before the split reports on pure noise, and
its validation by simulation.

Two balanced classes, n examples, p independent standard-normal features, labels independent of the
features. The leaky protocol ranks all p features by the squared two-sample statistic on ALL n examples,
keeps the top k, and then cross-validates a classifier (J folds) on those k features.

Write T_j = n^{-1/2} sum_i y_i x_ij ~ N(0, 1). Conditional on T_j, x_ij = y_i T_j / sqrt(n) + e_ij with
sum_i y_i e_ij = 0: the selected noise features are distributed exactly like features with a true class
difference of 2 T_j / sqrt(n). With m = m_k(p) the mean of the k largest of p chi-square(1) variables,
the nearest-centroid classifier trained on n_T = n (J - 1) / J examples and tested on n_V = n / J has

    accuracy ~= Phi( sqrt(k) (m / n - 1 / n_T) / sqrt(m / n + n_V / n_T^2) ).

The script compares this with simulated cross-validation for the nearest-centroid classifier and for
l2 logistic regression, and with the nested protocol (selection inside each training fold).
"""
import json
import sys

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

N_VALUES = [40, 80, 160, 320]
P_VALUES = [500, 5000, 40000]
K_VALUES = [10, 100, 500]
J = 10
SEEDS = 20
OUT = sys.argv[1] if len(sys.argv) > 1 else "selection_bias_theory.json"


def top_mean_chi2(k, p):
    """Mean of the k largest of p chi-square(1) variables (large-p approximation)."""
    q = k / (2.0 * p)
    c = norm.isf(q)
    return 1.0 + c * norm.pdf(c) / q


def predicted(n, p, k, folds=J):
    m = top_mean_chi2(k, p)
    n_t, n_v = n * (folds - 1) / folds, n / folds
    return float(norm.cdf(np.sqrt(k) * (m / n - 1.0 / n_t) / np.sqrt(m / n + n_v / n_t ** 2)))


def rank(X, y):
    return (y @ X) ** 2


def centroid_acc(Xtr, ytr, Xte, yte):
    mu_p, mu_n = Xtr[ytr > 0].mean(0), Xtr[ytr < 0].mean(0)
    s = (Xte - (mu_p + mu_n) / 2) @ (mu_p - mu_n)
    return float(np.mean(np.sign(s) == yte))


def logreg_acc(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xtr), ytr)
    return float(clf.score(sc.transform(Xte), yte))


def one_run(n, p, k, seed):
    rng = np.random.RandomState(seed)
    X = rng.standard_normal((n, p)).astype(np.float32)
    y = np.repeat([1.0, -1.0], n // 2)
    leak = np.argsort(-rank(X, y))[:k]
    out = {"leak_centroid": [], "leak_logreg": [], "nested_centroid": []}
    for tr, te in StratifiedKFold(J, shuffle=True, random_state=seed).split(X, y):
        out["leak_centroid"].append(centroid_acc(X[tr][:, leak], y[tr], X[te][:, leak], y[te]))
        out["leak_logreg"].append(logreg_acc(X[tr][:, leak], y[tr], X[te][:, leak], y[te]))
        inner = np.argsort(-rank(X[tr], y[tr]))[:k]
        out["nested_centroid"].append(centroid_acc(X[tr][:, inner], y[tr], X[te][:, inner], y[te]))
    return {a: float(np.mean(v)) for a, v in out.items()}


def main():
    rows = []
    for n in N_VALUES:
        for p in P_VALUES:
            for k in K_VALUES:
                if k >= p:
                    continue
                runs = [one_run(n, p, k, s) for s in range(SEEDS)]
                row = {"n": n, "p": p, "k": k, "predicted": predicted(n, p, k)}
                for a in runs[0]:
                    v = [r[a] for r in runs]
                    row[a], row[a + "_sd"] = float(np.mean(v)), float(np.std(v))
                rows.append(row)
                print(json.dumps(row), flush=True)
    err = [abs(r["predicted"] - r["leak_centroid"]) for r in rows]
    summary = {"max_abs_err_centroid": float(np.max(err)), "mean_abs_err_centroid": float(np.mean(err)),
               "max_abs_err_logreg": float(np.max([abs(r["predicted"] - r["leak_logreg"]) for r in rows])),
               "nested_range": [float(min(r["nested_centroid"] for r in rows)),
                                float(max(r["nested_centroid"] for r in rows))]}
    print(json.dumps(summary))
    with open(OUT, "w") as f:
        json.dump({"J": J, "seeds": SEEDS, "rows": rows, "summary": summary}, f, indent=1)


if __name__ == "__main__":
    main()
