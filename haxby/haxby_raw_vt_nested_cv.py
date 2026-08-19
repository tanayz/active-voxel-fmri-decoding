"""
Combines the two levers that each helped separately but were never tried
together: raw-TR-frame classification (our strongest Haxby track, 67.8%
mean across 6 subjects) + ventral-temporal mask restriction (which helped
the GLM track's weakest subjects substantially, e.g. subject 2: +19.8 pts).

Motivation: raw-TR-frame's own weak points (subject 4: 59.3%, subject 5:
59.1%) are exactly the kind of case VT restriction helped elsewhere -- a
regularizing effect where whole-brain search was likely overfitting to
noise on a weaker signal. Untested combination, worth trying before
concluding 67.8% is the ceiling for this approach.
"""
import json
import os
import time

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.maskers import NiftiMasker
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RNG_SEED = 42


def load_data_and_vt_index(subject):
    haxby = datasets.fetch_haxby(subjects=[subject])
    func = haxby.func[0]
    mask = haxby.mask
    mask_vt = haxby.mask_vt[0] if isinstance(haxby.mask_vt, list) else haxby.mask_vt
    labels = pd.read_csv(haxby.session_target[0], sep=" ")
    y_all = labels["labels"].values
    runs_all = labels["chunks"].values

    masker = NiftiMasker(mask_img=mask, runs=runs_all, smoothing_fwhm=4,
                          memory="nilearn_cache", memory_level=1, verbose=0)
    X_all = masker.fit_transform(func)

    non_rest = y_all != "rest"
    X, y, runs = X_all[non_rest], y_all[non_rest], runs_all[non_rest]

    # same direct-numpy VT index lookup verified working in haxby_vt_nested_cv.py
    # (two earlier attempts via NiftiMasker.transform() both produced 0 voxels
    # despite the masks being correctly aligned -- bypassed entirely here)
    whole_brain_data = nib.load(mask).get_fdata() > 0
    vt_data = nib.load(mask_vt).get_fdata() > 0
    whole_brain_flat_positions = np.where(whole_brain_data.ravel())[0]
    is_vt_per_column = vt_data.ravel()[whole_brain_flat_positions]
    vt_indices = np.where(is_vt_per_column)[0]
    if len(vt_indices) == 0:
        raise ValueError(f"subject {subject}: VT mask projected to 0 voxels -- unexpected given "
                          f"this exact logic was already verified working for the GLM track.")

    return X, y, runs, vt_indices


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
    outer_cv = LeaveOneGroupOut()
    scores = []
    for train_idx, test_idx in outer_cv.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]

        grid = build_grid_search(k_values, c_values, outer_cv)
        grid.fit(X_train, y_train, groups=groups_train)
        scores.append(grid.score(X_test, y_test))
    return np.array(scores)


def main():
    c_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    subjects = [1, 2, 3, 4, 5, 6]
    raw_whole_brain = {1: 0.745, 2: 0.619, 3: 0.735, 4: 0.593, 5: 0.591, 6: 0.784}

    results = {}
    for subject in subjects:
        print(f"\n{'='*70}\nSubject {subject}\n{'='*70}")
        t0 = time.time()
        X, y, runs, vt_idx = load_data_and_vt_index(subject)
        X_vt = X[:, vt_idx]
        print(f"[{subject}] Loaded in {time.time()-t0:.1f}s: whole_brain={X.shape[1]} voxels, "
              f"VT={X_vt.shape[1]} voxels, examples={X.shape[0]}")

        k_values = sorted(set(min(k, X_vt.shape[1]) for k in [50, 100, 200, 300, 500, X_vt.shape[1]]))

        t0 = time.time()
        scores = nested_cv_accuracy(X_vt, y, runs, k_values, c_values)
        elapsed = time.time() - t0
        mean_acc = float(scores.mean())
        print(f"[{subject}] Nested CV done in {elapsed:.1f}s")
        print(f"[{subject}] Per-fold scores: {['%.3f' % s for s in scores]}")
        print(f"[{subject}] Mean: {mean_acc:.3f} (vs raw whole-brain: {raw_whole_brain[subject]:.3f})")
        results[subject] = {"n_vt_voxels": int(X_vt.shape[1]), "mean": mean_acc,
                             "per_fold_scores": scores.tolist(), "raw_whole_brain": raw_whole_brain[subject]}

    print(f"\n{'='*70}\nSummary: raw-TR-frame + VT-restriction vs raw-TR-frame whole-brain\n{'='*70}")
    for subject, r in results.items():
        delta = r["mean"] - r["raw_whole_brain"]
        print(f"  subject {subject}: VT+raw={r['mean']:.3f}  whole_brain={r['raw_whole_brain']:.3f}  delta={delta:+.3f}")
    mean_all = float(np.mean([r["mean"] for r in results.values()]))
    print(f"  MEAN across subjects: {mean_all:.3f}  (vs raw whole-brain mean 0.678, "
          f"vs GLM VT-restricted mean 0.668)")

    out_path = "/kaggle/working/haxby_raw_vt_results.json" if os.path.isdir("/kaggle/working") else "./haxby_raw_vt_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
