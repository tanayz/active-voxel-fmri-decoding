"""
Lever 3, first pass: cross-subject hyperalignment for subject 4 (levers 1+2
left it essentially flat, 57.6% -> 58.0%/58.0%, so this tests whether
borrowing other subjects' data actually helps where within-subject tuning
didn't).

REVISION: the first attempt at this operated in raw whole-brain space
(39,912 voxels) -- pooling 5 auxiliary subjects' full data into a fresh
~5,100 x 39,912 array in every one of 12 folds, plus repeated F-test ranking
over the same huge dimensionality, got OOM-killed (`Killed` in the log, no
Python traceback) after ~3.45 hours with zero results.

Fix, which is also more methodologically sound, not just a compute hack:
restrict the whole procedure to ventral temporal cortex, exactly what
Haxby's own hyperalignment work does (functional alignment is far more
reliable within a category-selective region than across the noisy whole
brain). Subjects have different-sized VT masks (307-675 voxels), and
orthogonal_procrustes requires matching dimensions, so this uses the UNION
of all involved subjects' VT indices -- still positions within the shared
39,912-voxel whole-brain grid (confirmed identical across all 6 subjects),
so still anatomically consistent, just a common feature space every subject
can be indexed into.

Still a cheap sanity check first: fixed hyperparameters (Logistic C=1.0, k
capped by the (much smaller) VT-union size), plain LeaveOneGroupOut CV, no
inner grid search -- confirms both whether the idea has any signal and real
per-fold timing before investing in a full nested-CV version.

Method: for each of subject 4's 12 outer folds, build subject 4's own
per-category template (in VT-union space) using ONLY its training runs (11
runs) -- the only part that must stay fold-restricted, since it's derived
from the target subject's own (soon-to-be-tested) data. Each auxiliary
subject's template is built from ALL of their data (safe -- they're never
evaluated, only used as auxiliary training signal). Align each auxiliary
subject to subject 4's training-derived template via orthogonal Procrustes,
transform their full VT-union-space trial-level data into subject 4's
space, and pool it into the training set. Test on subject 4's held-out run,
untransformed (already in subject 4's own space).
"""
import json
import os
import time

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.maskers import NiftiMasker
from scipy.linalg import orthogonal_procrustes
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RNG_SEED = 42
TARGET_SUBJECT = 4
AUX_SUBJECTS = [1, 2, 3, 5, 6]
FIXED_C = 1.0


def load_data_and_vt(subject):
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

    # same direct-numpy VT lookup verified working elsewhere in this project
    # (NiftiMasker.transform() on a mask image produced 0 voxels twice, for
    # reasons never fully diagnosed -- bypassed entirely)
    whole_brain_data = nib.load(mask).get_fdata() > 0
    vt_data = nib.load(mask_vt).get_fdata() > 0
    whole_brain_flat_positions = np.where(whole_brain_data.ravel())[0]
    is_vt_per_column = vt_data.ravel()[whole_brain_flat_positions]
    vt_indices = np.where(is_vt_per_column)[0]
    if len(vt_indices) == 0:
        raise ValueError(f"subject {subject}: VT mask projected to 0 voxels.")

    return X, y, runs, vt_indices


def category_template(X, y, categories):
    """Mean pattern per category, shape (n_categories, n_features)."""
    return np.vstack([X[y == c].mean(axis=0) for c in categories])


class ActiveVoxelSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=200):
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


def fixed_pipeline(k):
    return Pipeline([
        ("select", ActiveVoxelSelector(k=k)),
        ("scale", StandardScaler()),
        ("clf", OneVsRestClassifier(LogisticRegression(max_iter=5000, C=FIXED_C))),
    ])


def main():
    print(f"Loading target subject {TARGET_SUBJECT} and {len(AUX_SUBJECTS)} auxiliary subjects...")
    t0 = time.time()
    X_target_full, y_target, runs_target, vt_target = load_data_and_vt(TARGET_SUBJECT)
    print(f"  target {TARGET_SUBJECT}: loaded in {time.time()-t0:.1f}s, "
          f"whole_brain={X_target_full.shape[1]}, VT={len(vt_target)}")

    aux_full = {}
    all_vt_indices = [vt_target]
    for s in AUX_SUBJECTS:
        t0 = time.time()
        Xa, ya, _, vt_a = load_data_and_vt(s)
        assert Xa.shape[1] == X_target_full.shape[1], \
            f"Subject {s} has {Xa.shape[1]} voxels, target has {X_target_full.shape[1]} -- masks don't match!"
        aux_full[s] = (Xa, ya)
        all_vt_indices.append(vt_a)
        print(f"  aux {s}: loaded in {time.time()-t0:.1f}s, VT={len(vt_a)}")

    # union of all involved subjects' VT indices, in the shared whole-brain
    # index space -- fixes both the OOM/runtime problem (39912 -> ~1-2k
    # dims) and gives every subject the SAME feature dimensionality for
    # orthogonal_procrustes, which requires matching shapes.
    vt_union = np.unique(np.concatenate(all_vt_indices))
    print(f"\nVT union across all {1+len(AUX_SUBJECTS)} subjects: {len(vt_union)} voxels "
          f"(vs {X_target_full.shape[1]} whole-brain)")

    X_target = X_target_full[:, vt_union]
    aux_data = {s: (Xa[:, vt_union], ya) for s, (Xa, ya) in aux_full.items()}
    del X_target_full, aux_full  # free the whole-brain arrays, no longer needed

    categories = sorted(np.unique(y_target))
    k = min(200, len(vt_union))
    print(f"Categories: {categories}, using k={k} of {len(vt_union)} VT-union voxels")

    outer_cv = LeaveOneGroupOut()
    solo_scores, pooled_scores = [], []
    fold_times = []

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X_target, y_target, runs_target)):
        t0 = time.time()
        X_tr, y_tr = X_target[train_idx], y_target[train_idx]
        X_te, y_te = X_target[test_idx], y_target[test_idx]

        solo_pipe = fixed_pipeline(k)
        solo_pipe.fit(X_tr, y_tr)
        solo_scores.append(solo_pipe.score(X_te, y_te))

        target_template = category_template(X_tr, y_tr, categories)

        X_pooled_parts, y_pooled_parts = [X_tr], [y_tr]
        for s, (Xa, ya) in aux_data.items():
            aux_template = category_template(Xa, ya, categories)
            R, _ = orthogonal_procrustes(aux_template, target_template)
            Xa_aligned = Xa @ R
            X_pooled_parts.append(Xa_aligned)
            y_pooled_parts.append(ya)

        X_pooled = np.vstack(X_pooled_parts)
        y_pooled = np.concatenate(y_pooled_parts)

        pooled_pipe = fixed_pipeline(k)
        pooled_pipe.fit(X_pooled, y_pooled)
        pooled_scores.append(pooled_pipe.score(X_te, y_te))

        elapsed = time.time() - t0
        fold_times.append(elapsed)
        print(f"  fold {fold_i+1}/12: solo={solo_scores[-1]:.3f}, pooled={pooled_scores[-1]:.3f}, "
              f"pooled_train_n={X_pooled.shape[0]}, took {elapsed:.1f}s")

    solo_mean = float(np.mean(solo_scores))
    pooled_mean = float(np.mean(pooled_scores))
    print(f"\n{'='*70}\nResults for subject {TARGET_SUBJECT} (VT-union space, {len(vt_union)} voxels)\n{'='*70}")
    print(f"Solo (fixed pipeline, no pooling):    mean={solo_mean:.3f}, per-fold={['%.3f'%s for s in solo_scores]}")
    print(f"Pooled (hyperaligned aux subjects):   mean={pooled_mean:.3f}, per-fold={['%.3f'%s for s in pooled_scores]}")
    print(f"Delta: {pooled_mean - solo_mean:+.3f}")
    print(f"Mean time per fold: {np.mean(fold_times):.1f}s, total: {np.sum(fold_times):.1f}s")
    print(f"(Compare against prior results for subject 4: whole-brain raw nested CV=0.593, "
          f"raw+VT=0.576, combined-levers=0.580)")

    out = {"vt_union_size": int(len(vt_union)), "k": k,
           "solo_mean": solo_mean, "solo_scores": solo_scores,
           "pooled_mean": pooled_mean, "pooled_scores": pooled_scores,
           "fold_times": fold_times}
    out_path = "/kaggle/working/haxby_hyperalignment_sanity_results.json" if os.path.isdir("/kaggle/working") else "./haxby_hyperalignment_sanity_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
