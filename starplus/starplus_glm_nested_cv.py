"""
Fixes the same feature-selection leakage bug confirmed in
starplus_glm_benchmark.py's sweep(): it computed active_voxel_ranking(X, y)
ONCE on the full 80-example dataset before cross-validation, meaning every
fold's voxel selection could "see" that fold's own test labels. This is very
likely the real explanation for the suspiciously perfect 100% accuracy on
all 3 subjects reported by that script.

Same fix pattern as haxby_glm_nested_cv.py: voxel selection becomes a proper
sklearn transformer INSIDE a Pipeline, tuned (along with regularization C
and classifier choice) via an inner GridSearchCV that only ever sees the
outer-training folds' data. Outer GroupKFold(10), grouped by trial index
(40 groups -> matches the original 10-fold convention used throughout this
project), gives the honest, unpeeked accuracy estimate.
"""
import json
import os
import time
import urllib.request

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from nilearn.maskers import NiftiMasker
from scipy.io import loadmat
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RNG_SEED = 42
TR = 0.5
STIM_DURATION = 4.0
SECOND_STIM_OFFSET = 8.0

DATA_DIR = "/kaggle/working/data" if os.path.isdir("/kaggle/working") else "./data"
STARPLUS_DIR = os.path.join(DATA_DIR, "starplus")
os.makedirs(STARPLUS_DIR, exist_ok=True)

STARPLUS_URLS = {
    "04799": "http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-81/www/data-starplus-04799-v7.mat",
    "04820": "http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-83/www/data-starplus-04820-v7.mat",
    "04847": "http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-81/www/data-starplus-04847-v7.mat",
}


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        return
    urllib.request.urlretrieve(url, dest)


def load_subject(path):
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    return np.atleast_1d(mat["info"]), np.atleast_1d(mat["data"]), mat["meta"]


def reconstruct_volume_and_mask(info, data, meta):
    dimx, dimy, dimz = int(meta.dimx), int(meta.dimy), int(meta.dimz)
    col_to_coord = np.asarray(meta.colToCoord, dtype=int) - 1

    n_timepoints = int(max(int(e.maxt) for e in info))
    volume = np.zeros((dimx, dimy, dimz, n_timepoints), dtype=np.float32)
    mask = np.zeros((dimx, dimy, dimz), dtype=np.uint8)
    mask[col_to_coord[:, 0], col_to_coord[:, 1], col_to_coord[:, 2]] = 1

    for i, entry in enumerate(info):
        d = np.asarray(data[i], dtype=np.float32)
        t0 = int(entry.mint) - 1
        t1 = t0 + d.shape[0]
        volume[col_to_coord[:, 0], col_to_coord[:, 1], col_to_coord[:, 2], t0:t1] = d.T

    affine = np.eye(4)
    return nib.Nifti1Image(volume, affine), nib.Nifti1Image(mask, affine)


def build_events(info):
    rows = []
    for i, entry in enumerate(info):
        cond = int(entry.cond)
        if cond <= 1:
            continue
        onset0 = (int(entry.mint) - 1) * TR
        fs = str(entry.firstStimulus)
        first_label, second_label = ("picture", "sentence") if fs == "P" else ("sentence", "picture")
        rows.append({"onset": onset0, "duration": STIM_DURATION,
                     "trial_type": f"t{i}_{first_label}", "base_label": first_label, "trial": i})
        rows.append({"onset": onset0 + SECOND_STIM_OFFSET, "duration": STIM_DURATION,
                     "trial_type": f"t{i}_{second_label}", "base_label": second_label, "trial": i})
    return pd.DataFrame(rows)


def fit_beta_series(path):
    info, data, meta = load_subject(path)
    func_img, mask_img = reconstruct_volume_and_mask(info, data, meta)
    events_full = build_events(info)

    glm = FirstLevelModel(t_r=TR, hrf_model="spm", drift_model="cosine", high_pass=0.01,
                           mask_img=mask_img, smoothing_fwhm=None, minimize_memory=False, verbose=0)
    glm.fit(func_img, events=events_full[["onset", "duration", "trial_type"]])

    masker = NiftiMasker(mask_img=mask_img, standardize=False)
    masker.fit()

    X_rows, y_rows, group_rows = [], [], []
    for _, ev in events_full.iterrows():
        beta_img = glm.compute_contrast(ev["trial_type"], output_type="effect_size")
        X_rows.append(np.asarray(masker.transform(beta_img)).reshape(-1))
        y_rows.append(ev["base_label"])
        group_rows.append(ev["trial"])

    shapes = {r.shape for r in X_rows}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent beta-vector lengths: {sorted(shapes)}")
    return np.vstack(X_rows), np.array(y_rows), np.array(group_rows)


class ActiveVoxelSelector(BaseEstimator, TransformerMixin):
    """Same Chowdhury-style ranking as before, but as a proper transformer
    so k is a nested-CV-safe hyperparameter and .fit() only ever sees
    whatever data the surrounding Pipeline/GridSearchCV gives it."""
    def __init__(self, k=100):
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
        ("clf", LogisticRegression(max_iter=5000)),
    ])
    param_grid = [
        {"select__k": k_values, "clf": [LogisticRegression(max_iter=5000)], "clf__C": c_values},
        {"select__k": k_values, "clf": [SVC(kernel="linear")], "clf__C": c_values},
    ]
    return GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="accuracy", n_jobs=-1, refit=True)


def nested_cv_accuracy(X, y, groups, k_values, c_values):
    outer_cv = GroupKFold(n_splits=10)
    inner_cv = GroupKFold(n_splits=6)
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
    k_values = [10, 25, 50, 100, 250, 500, 1000]
    c_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    prior_flat_sweep = {"04799": 1.000, "04820": 1.000, "04847": 1.000}

    results = {}
    for sid, url in STARPLUS_URLS.items():
        print(f"\n{'='*70}\nSubject {sid}\n{'='*70}")
        dest = os.path.join(STARPLUS_DIR, f"data-starplus-{sid}-v7.mat")
        download(url, dest)

        t0 = time.time()
        X, y, groups = fit_beta_series(dest)
        print(f"[{sid}] beta-series: {time.time()-t0:.1f}s, X={X.shape}")

        t0 = time.time()
        scores = nested_cv_accuracy(X, y, groups, k_values, c_values)
        elapsed = time.time() - t0
        mean_acc = float(scores.mean())
        print(f"[{sid}] nested CV: {elapsed:.1f}s, per-fold={['%.3f' % s for s in scores]}, "
              f"mean={mean_acc:.3f} (vs flat-sweep={prior_flat_sweep[sid]:.3f})")

        results[sid] = {"nested_cv_mean": mean_acc, "per_fold_scores": scores.tolist(),
                         "flat_sweep": prior_flat_sweep[sid]}

    print(f"\n{'='*70}\nSummary: nested CV (honest) vs flat-sweep (leaked) accuracy\n{'='*70}")
    for sid, r in results.items():
        delta = r["nested_cv_mean"] - r["flat_sweep"]
        print(f"  {sid}: nested={r['nested_cv_mean']:.3f}  flat_sweep={r['flat_sweep']:.3f}  delta={delta:+.3f}")

    out_path = "/kaggle/working/starplus_nested_cv_results.json" if os.path.isdir("/kaggle/working") else "./starplus_nested_cv_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
