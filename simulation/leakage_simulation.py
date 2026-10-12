"""
Ground-truth simulation of the two leakage mechanisms quantified in the paper. Data are pure noise
(true accuracy = 1/8 = 0.125) laid out like the Haxby design: 12 runs x 8 stimulus blocks x 9 volumes
(864 volumes), each run's blocks in random order. Voxel time series are independent AR(1) processes
within each run (phi controls the temporal autocorrelation that makes adjacent volumes of a block
similar, as with the BOLD response). A "beta-series" design (one pattern per block, 96 patterns) is
also simulated.

Protocols (all use summed one-vs-rest F ranking, top k = 500 voxels, StandardScaler, OvR logistic C=1):
  correct          selection inside the pipeline, leave-one-run-out (LORO)
  select_before    voxels ranked once on ALL data, then LORO          (feature-selection leakage)
  ungrouped_cv5    selection inside the pipeline, cross_val_score(cv=5) over volumes (no run grouping)
  both             ranked on all data AND ungrouped cv=5
"""
import json
import sys
import time

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

N_RUNS, N_CLASSES, BLOCK_LEN, K = 12, 8, 9, 500
P_VALUES = [300, 2000, 10000, 40000]
PHI_VALUES = [0.0, 0.5, 0.8]
SEEDS = range(5)


class TopK(BaseEstimator, TransformerMixin):
    def __init__(self, k=K):
        self.k = k

    def fit(self, X, y):
        s = np.zeros(X.shape[1])
        for c in np.unique(y):
            f, _ = f_classif(X, (y == c).astype(int))
            s += np.nan_to_num(f, nan=0.0)
        self.idx_ = np.argsort(-s)[:min(self.k, X.shape[1])]
        return self

    def transform(self, X):
        return X[:, self.idx_]


def design(rng, block_len):
    y, g = [], []
    for r in range(N_RUNS):
        for c in rng.permutation(N_CLASSES):
            y += [c] * block_len
            g += [r] * block_len
    return np.array(y), np.array(g)


def ar1_noise(rng, groups, p, phi):
    X = rng.standard_normal((len(groups), p)).astype(np.float32)
    if phi > 0:
        for r in np.unique(groups):
            idx = np.where(groups == r)[0]
            for t in range(1, len(idx)):
                X[idx[t]] = phi * X[idx[t - 1]] + np.sqrt(1 - phi ** 2) * X[idx[t]]
    return X


def clf():
    return Pipeline([("scale", StandardScaler()), ("clf", OneVsRestClassifier(LogisticRegression(max_iter=2000)))])


def run_protocols(X, y, g):
    inside = Pipeline([("sel", TopK())] + clf().steps)
    pre_idx = TopK().fit(X, y).idx_
    Xpre = X[:, pre_idx]
    logo = LeaveOneGroupOut()
    return {
        "correct": cross_val_score(inside, X, y, cv=logo, groups=g, n_jobs=-1).mean(),
        "select_before": cross_val_score(clf(), Xpre, y, cv=logo, groups=g, n_jobs=-1).mean(),
        "ungrouped_cv5": cross_val_score(inside, X, y, cv=5, n_jobs=-1).mean(),
        "both": cross_val_score(clf(), Xpre, y, cv=5, n_jobs=-1).mean(),
    }


def main():
    out = {"config": {"n_runs": N_RUNS, "n_classes": N_CLASSES, "block_len": BLOCK_LEN, "k": K,
                      "p_values": P_VALUES, "phi_values": PHI_VALUES, "n_seeds": len(SEEDS)},
           "frames": [], "betas": []}
    t0 = time.time()
    for p in P_VALUES:
        for phi in PHI_VALUES:
            for s in SEEDS:
                rng = np.random.RandomState(1000 * s + int(10 * phi) + p)
                y, g = design(rng, BLOCK_LEN)
                res = run_protocols(ar1_noise(rng, g, p, phi), y, g)
                out["frames"].append({"p": p, "phi": phi, "seed": s, **{k: float(v) for k, v in res.items()}})
            m = {k: np.mean([r[k] for r in out["frames"] if r["p"] == p and r["phi"] == phi])
                 for k in ("correct", "select_before", "ungrouped_cv5", "both")}
            print(f"frames p={p:>5d} phi={phi:.1f}: " + "  ".join(f"{k}={v:.3f}" for k, v in m.items())
                  + f"  ({time.time() - t0:.0f}s)", flush=True)
        for s in SEEDS:   # beta-series layout: one pattern per block, 96 patterns
            rng = np.random.RandomState(7000 + 1000 * s + p)
            y, g = design(rng, 1)
            res = run_protocols(rng.standard_normal((len(y), p)).astype(np.float32), y, g)
            out["betas"].append({"p": p, "seed": s, **{k: float(v) for k, v in res.items()}})
        m = {k: np.mean([r[k] for r in out["betas"] if r["p"] == p]) for k in ("correct", "select_before")}
        print(f"betas  p={p:>5d}        : " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()), flush=True)
    import os
    default = "/kaggle/working/leakage_simulation.json" if os.path.isdir("/kaggle/working") else "leakage_simulation.json"
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else default, "w"), indent=2)


if __name__ == "__main__":
    main()
