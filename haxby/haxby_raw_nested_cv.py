"""
Same leakage fix applied to the ORIGINAL raw-TR-frame Haxby track
(haxby_benchmark.py) -- this is the one behind the initial "beats nilearn's
54.5%/41.4% baseline with 72.0%" headline claim, so it needs the same
correction before that claim can be trusted. stage_a_reproduce_baseline's
nilearn-protocol numbers (41.4% grouped / 54.5% as-published) are NOT
affected by this bug -- nilearn's own pipeline already nests SelectKBest
inside the sklearn Pipeline, which cross_val_score correctly refits per
fold -- so those remain the valid comparison point.

Focused on subject 2 (nilearn's default, and the subject the original
54.5%/41.4% baseline numbers were computed on) for a direct, apples-to-apples
comparison, rather than re-running all 6 subjects again.
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
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RNG_SEED = 42


def load_data(subject=2):
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


def build_grid_search(k_values, c_values, inner_cv):
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
    return GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="accuracy", n_jobs=-1, refit=True)


def nested_cv_accuracy(X, y, groups, k_values, c_values):
    outer_cv = LeaveOneGroupOut()          # 12 folds, matches the original convention
    inner_cv = GroupKFold(n_splits=5)      # trimmed from leave-one-run-out (11 folds) for
                                            # compute -- 864 examples here vs 96 in the GLM
                                            # track, so each inner fit is much more expensive
    scores = []
    for train_idx, test_idx in outer_cv.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        grid = build_grid_search(k_values, c_values, inner_cv)
        grid.fit(X_train, y_train, groups=groups_train)
        scores.append(grid.score(X_test, y_test))
    return np.array(scores)


def main():
    # subject 2's own results (nilearn's default subject, and the one the
    # published 54.5%/41.4% baseline numbers were computed on) already
    # confirmed: honest nested CV = 0.619, beating both nilearn baselines.
    # Extending to all 6 subjects for a fair aggregate comparison against
    # the GLM (0.635 mean) and VT-restricted (0.668 mean) tracks.
    k_values = [100, 250, 500, 1000]
    c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    subjects = [1, 2, 3, 4, 5, 6]
    known_subject2 = 0.619  # sanity check against the already-completed run

    results = {}
    for subject in subjects:
        print(f"\n{'='*70}\nSubject {subject}\n{'='*70}")
        t0 = time.time()
        X, y, runs = load_data(subject=subject)
        print(f"[{subject}] Loaded in {time.time()-t0:.1f}s: X={X.shape}")

        t0 = time.time()
        scores = nested_cv_accuracy(X, y, runs, k_values, c_values)
        elapsed = time.time() - t0
        mean_acc = float(scores.mean())
        print(f"[{subject}] Nested CV done in {elapsed:.1f}s")
        print(f"[{subject}] Per-fold scores: {['%.3f' % s for s in scores]}")
        print(f"[{subject}] Mean nested CV accuracy: {mean_acc:.3f}")
        results[subject] = {"nested_cv_mean": mean_acc, "per_fold_scores": scores.tolist()}

    print(f"\n{'='*70}\nSummary: raw-TR-frame honest nested CV, all subjects\n{'='*70}")
    for subject, r in results.items():
        print(f"  subject {subject}: {r['nested_cv_mean']:.3f}")
    mean_all = float(np.mean([r["nested_cv_mean"] for r in results.values()]))
    print(f"  MEAN across subjects: {mean_all:.3f}")
    print(f"  (subject 2 sanity check: {results[2]['nested_cv_mean']:.3f} vs previously-run {known_subject2:.3f})")
    print(f"  Compare: GLM whole-brain mean=0.635, GLM VT-restricted mean=0.668, "
          f"nilearn published (subject 2 only)=0.545, nilearn leakage-safe (subject 2 only)=0.414")

    out_path = "/kaggle/working/haxby_raw_nested_cv_all_subjects.json" if os.path.isdir("/kaggle/working") else "./haxby_raw_nested_cv_all_subjects.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
