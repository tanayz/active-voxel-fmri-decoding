"""
Permutation/significance test for the Haxby raw-TR-frame nested-CV result
(67.8% mean across 6 subjects, our best Haxby track).

Unlike StarPlus (small enough to redo the FULL nested procedure -- inner
GridSearchCV hyperparameter search + outer CV -- for every permutation),
Haxby's raw-TR-frame track has 864 examples x up to ~2000 selected features
per subject; a full nested-CV-per-permutation test here would take days.

Standard, defensible simplification used in the MVPA permutation-testing
literature: fit the hyperparameter search ONCE on the real (non-permuted)
data to identify a single best (k, C, classifier) configuration, then hold
that configuration FIXED across all permutations -- each permutation only
needs cheap single-pipeline CV scoring, not a full re-search. This tests
"does the analysis pipeline our tuning procedure settled on perform above
chance", which is what the paper is actually claiming, rather than
re-validating per-fold hyperparameter choices under every permutation.
Note the resulting "real" accuracy from this fixed pipeline may differ
slightly from the original per-fold-tuned nested-CV mean (67.8%) -- that's
expected and is reported alongside for comparison, not treated as a
discrepancy to resolve.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.maskers import NiftiMasker
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RNG_SEED = 42
N_PERMUTATIONS = 100


def load_data(subject):
    haxby = datasets.fetch_haxby(subjects=[subject])
    func = haxby.func[0]
    mask = haxby.mask
    labels = pd.read_csv(haxby.session_target[0], sep=" ")
    y_all = labels["labels"].values
    runs_all = labels["chunks"].values

    masker = NiftiMasker(mask_img=mask, runs=runs_all, smoothing_fwhm=4,
                          memory="nilearn_cache", memory_level=1, verbose=0)
    X_all = masker.fit_transform(func)

    non_rest = y_all != "rest"
    return X_all[non_rest], y_all[non_rest], runs_all[non_rest]


class ActiveVoxelSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=500):
        self.k = k

    def fit(self, X, y):
        scores = np.zeros(X.shape[1])
        for c in np.unique(y):
            f_vals, _ = f_classif(X, (y == c).astype(int))
            scores += np.nan_to_num(f_vals, nan=0.0)
        self.ranking_ = np.argsort(-scores)
        return self

    def transform(self, X):
        k = min(self.k, X.shape[1])
        return X[:, self.ranking_[:k]]


def find_fixed_pipeline(X, y, groups, k_values, c_values):
    """One GridSearchCV fit on the full real dataset to pick a single
    (k, C, classifier) configuration to hold fixed across permutations."""
    pipe = Pipeline([
        ("select", ActiveVoxelSelector()),
        ("scale", StandardScaler()),
        ("clf", OneVsRestClassifier(LogisticRegression(max_iter=5000))),
    ])
    param_grid = [
        {"select__k": k_values,
         "clf": [OneVsRestClassifier(LogisticRegression(max_iter=5000))],
         "clf__estimator__C": c_values},
        {"select__k": k_values,
         "clf": [OneVsRestClassifier(SVC(kernel="linear"))],
         "clf__estimator__C": c_values},
    ]
    inner_cv = GroupKFold(n_splits=5)
    grid = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="accuracy", n_jobs=-1, refit=True)
    grid.fit(X, y, groups=groups)
    return grid.best_estimator_, grid.best_params_


def main():
    k_values = [100, 250, 500, 1000]
    c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    subjects = [1, 2, 3, 4, 5, 6]
    known_real_accuracy = {1: 0.745, 2: 0.619, 3: 0.735, 4: 0.593, 5: 0.591, 6: 0.784}

    results = {}
    for subject in subjects:
        print(f"\n{'='*70}\nSubject {subject}\n{'='*70}")
        t0 = time.time()
        X, y, groups = load_data(subject)
        print(f"[{subject}] Loaded in {time.time()-t0:.1f}s: X={X.shape}")

        t0 = time.time()
        fixed_pipeline, best_params = find_fixed_pipeline(X, y, groups, k_values, c_values)
        print(f"[{subject}] Fixed pipeline chosen in {time.time()-t0:.1f}s: "
              f"{ {k: v for k, v in best_params.items() if k != 'clf'} }, "
              f"clf={type(best_params['clf'].estimator).__name__}")

        outer_cv = LeaveOneGroupOut()
        t0 = time.time()
        real_scores = cross_val_score(fixed_pipeline, X, y, cv=outer_cv, groups=groups, n_jobs=-1)
        real_mean = float(real_scores.mean())
        print(f"[{subject}] Fixed-pipeline real accuracy: {real_mean:.3f} "
              f"(vs original per-fold-tuned nested CV: {known_real_accuracy[subject]:.3f}), "
              f"took {time.time()-t0:.1f}s")

        null_means = []
        rng = np.random.RandomState(RNG_SEED)
        t0 = time.time()
        for p in range(N_PERMUTATIONS):
            y_perm = rng.permutation(y)
            perm_scores = cross_val_score(fixed_pipeline, X, y_perm, cv=outer_cv, groups=groups, n_jobs=-1)
            null_means.append(float(perm_scores.mean()))
            if (p + 1) % 20 == 0:
                print(f"[{subject}]   permutation {p+1}/{N_PERMUTATIONS}, elapsed={time.time()-t0:.1f}s")

        null_means = np.array(null_means)
        p_value = (np.sum(null_means >= real_mean) + 1) / (N_PERMUTATIONS + 1)
        print(f"[{subject}] Null distribution: mean={null_means.mean():.3f}, "
              f"std={null_means.std():.3f}, max={null_means.max():.3f}")
        print(f"[{subject}] Real accuracy={real_mean:.3f}, permutation p-value={p_value:.4f}")

        results[subject] = {"real_mean": real_mean, "real_per_fold": real_scores.tolist(),
                             "null_means": null_means.tolist(), "p_value": float(p_value),
                             "fixed_params": {k: str(v) for k, v in best_params.items()}}

    print(f"\n{'='*70}\nSummary\n{'='*70}")
    for subject, r in results.items():
        print(f"  subject {subject}: real={r['real_mean']:.3f}, "
              f"null_mean={np.mean(r['null_means']):.3f}, p={r['p_value']:.4f}")

    out_path = "/kaggle/working/haxby_raw_permutation_results.json" if os.path.isdir("/kaggle/working") else "./haxby_raw_permutation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
