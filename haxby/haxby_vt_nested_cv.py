"""
Anatomically-restricted feature selection for Haxby, as a legitimate
(non-leaky) attempt to improve on the honest 63.5% mean nested-CV accuracy
from the whole-brain GLM track (haxby_glm_nested_cv.py).

Rationale: the whole-brain track searches ~40,000 candidate voxels for only
96 examples -- an extreme p >> n ratio that nested CV correctly showed was
partly responsible for the earlier inflated numbers. Haxby ships its own
ventral-temporal mask (`mask_vt`), and the original Haxby et al. (2001)
paper is specifically about object/face representation in that region.
Restricting to VT uses domain knowledge to shrink the candidate feature
space BEFORE any statistical ranking, rather than relying purely on
data-driven search over the whole brain -- a different lever than more
hyperparameter search, not just re-running the same thing hoping for a
better number.

Reuses the whole-brain beta patterns already computed by fit_beta_patterns
(no need to refit any GLMs) -- the VT mask is projected into the SAME
flattened voxel-index space as X's columns via direct numpy boolean
indexing on the mask arrays (see fit_beta_patterns_and_vt_index -- two
earlier attempts routing this through NiftiMasker.transform() both
produced 0 voxels despite the underlying masks being confirmed correctly
aligned), then X is subset to those columns before the same nested-CV grid
search used for the whole-brain track (so results are directly comparable).
"""
import json
import os
import time

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, image
from nilearn.glm.first_level import FirstLevelModel
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


# --- inlined from haxby_glm_benchmark.py / haxby_glm_nested_cv.py ----------
def build_events(labels, chunks, t_r):
    events_per_run = {}
    for run in np.unique(chunks):
        run_labels = labels[chunks == run]
        rows, i, n = [], 0, len(run_labels)
        while i < n:
            lab = run_labels[i]
            j = i
            while j < n and run_labels[j] == lab:
                j += 1
            if lab != "rest":
                rows.append({"onset": i * t_r, "duration": (j - i) * t_r, "trial_type": lab})
            i = j
        events_per_run[run] = pd.DataFrame(rows)
    return events_per_run


def fit_beta_patterns_and_vt_index(subject):
    """Returns whole-brain X, y, runs, PLUS the column indices of X that
    fall within the ventral-temporal mask."""
    haxby = datasets.fetch_haxby(subjects=[subject])
    func_img = haxby.func[0]
    mask_img = haxby.mask
    mask_vt_img = haxby.mask_vt[0] if isinstance(haxby.mask_vt, list) else haxby.mask_vt
    t_r = float(haxby.t_r)
    labels_df = pd.read_csv(haxby.session_target[0], sep=" ")
    labels, chunks = labels_df["labels"].values, labels_df["chunks"].values

    events_per_run = build_events(labels, chunks, t_r)
    masker = NiftiMasker(mask_img=mask_img, smoothing_fwhm=4, standardize=False)
    masker.fit()

    # project the VT mask into the SAME flattened voxel-index space as X's
    # columns. Two prior attempts via NiftiMasker.transform() on the mask
    # image both produced 0 voxels despite the masks being confirmed
    # perfectly aligned (identical shape/affine, VT genuinely a 577-voxel
    # subset of the 39912-voxel whole-brain mask -- verified directly with
    # nibabel). Bypassing NiftiMasker.transform() for this specific lookup
    # and replicating its flattening with plain numpy boolean indexing
    # instead, which is directly verifiable: nilearn's own masking
    # internally does `img_data[mask_data.astype(bool)]`, i.e. standard
    # numpy C-order boolean fancy-indexing -- the same thing done here.
    whole_brain_data = nib.load(mask_img).get_fdata() > 0
    vt_data = nib.load(mask_vt_img).get_fdata() > 0
    whole_brain_flat_positions = np.where(whole_brain_data.ravel())[0]
    is_vt_per_column = vt_data.ravel()[whole_brain_flat_positions]
    vt_indices = np.where(is_vt_per_column)[0]
    print(f"  VT mask: {len(vt_indices)} voxels found within whole-brain mask "
          f"(raw VT nonzero count: {int(vt_data.sum())})")
    if len(vt_indices) == 0:
        raise ValueError(f"subject {subject}: VT mask projected to 0 voxels within the "
                          f"whole-brain mask even with direct numpy indexing -- something "
                          f"more fundamental is wrong, needs further investigation.")

    X_rows, y_rows, run_rows = [], [], []
    for run in np.unique(chunks):
        run_idx = np.where(chunks == run)[0]
        run_img = image.index_img(func_img, run_idx)
        events = events_per_run[run]
        if events.empty:
            print(f"  run {run}: no non-rest events, skipping")
            continue
        conditions = sorted(events["trial_type"].unique())

        glm = FirstLevelModel(t_r=t_r, hrf_model="spm", drift_model="cosine",
                               high_pass=0.01, mask_img=mask_img, smoothing_fwhm=4,
                               minimize_memory=False, verbose=0)
        glm.fit(run_img, events=events)

        for cond in conditions:
            beta_img = glm.compute_contrast(cond, output_type="effect_size")
            row = np.asarray(masker.transform(beta_img)).reshape(-1)
            X_rows.append(row)
            y_rows.append(cond)
            run_rows.append(run)
        print(f"  run {run}: fit GLM, {len(conditions)} conditions")

    shapes = {r.shape for r in X_rows}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent beta-vector lengths: {sorted(shapes)}")
    return np.vstack(X_rows), np.array(y_rows), np.array(run_rows), vt_indices
# -----------------------------------------------------------------------


class ActiveVoxelSelector(BaseEstimator, TransformerMixin):
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

        grid = build_grid_search(k_values, c_values, outer_cv)  # LeaveOneGroupOut again for inner
        grid.fit(X_train, y_train, groups=groups_train)
        scores.append(grid.score(X_test, y_test))
    return np.array(scores)


def main():
    c_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    subjects = [1, 2, 3, 4, 5, 6]
    whole_brain_honest = {1: 0.510, 2: 0.375, 3: 0.750, 4: 0.646, 5: 0.830, 6: 0.698}

    results = {}
    for subject in subjects:
        print(f"\n{'='*70}\nSubject {subject}\n{'='*70}")
        t0 = time.time()
        X, y, runs, vt_idx = fit_beta_patterns_and_vt_index(subject)
        X_vt = X[:, vt_idx]
        print(f"[{subject}] beta patterns: {time.time()-t0:.1f}s, "
              f"whole_brain={X.shape[1]} voxels, VT={X_vt.shape[1]} voxels")

        k_values = sorted(set(min(k, X_vt.shape[1]) for k in [25, 50, 100, 150, 200, 300, X_vt.shape[1]]))

        t0 = time.time()
        scores = nested_cv_accuracy(X_vt, y, runs, k_values, c_values)
        elapsed = time.time() - t0
        mean_acc = float(scores.mean())
        print(f"[{subject}] VT nested CV: {elapsed:.1f}s, per-fold={['%.3f' % s for s in scores]}, "
              f"mean={mean_acc:.3f} (vs whole-brain honest={whole_brain_honest[subject]:.3f})")

        results[subject] = {"n_vt_voxels": int(X_vt.shape[1]), "vt_nested_cv_mean": mean_acc,
                             "per_fold_scores": scores.tolist(),
                             "whole_brain_honest": whole_brain_honest[subject]}

    print(f"\n{'='*70}\nSummary: VT-restricted vs whole-brain (both honest, nested CV)\n{'='*70}")
    for subject, r in results.items():
        delta = r["vt_nested_cv_mean"] - r["whole_brain_honest"]
        print(f"  subject {subject}: VT={r['vt_nested_cv_mean']:.3f} (n_voxels={r['n_vt_voxels']})  "
              f"whole_brain={r['whole_brain_honest']:.3f}  delta={delta:+.3f}")
    print(f"  MEAN VT accuracy: {np.mean([r['vt_nested_cv_mean'] for r in results.values()]):.3f} "
          f"(vs whole-brain mean 0.635)")

    out_path = "/kaggle/working/haxby_vt_results.json" if os.path.isdir("/kaggle/working") else "./haxby_vt_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
