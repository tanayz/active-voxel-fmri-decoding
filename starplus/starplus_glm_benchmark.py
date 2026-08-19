"""
Track 3: GLM beta-series preprocessing applied to StarPlus, extending the
approach that took Haxby from 72.0% (raw TR-frames) to 82.3% (GLM beta
patterns) to a second, independent dataset -- a technique that wins on one
dataset is a result; winning on two is a much stronger claim.

StarPlus isn't NIfTI, so nilearn's FirstLevelModel can't be pointed at it
directly the way it could for Haxby. We reconstruct a proper 4D volume using
the exact voxel coordinate mapping recovered in Stage 1 (meta.colToCoord,
meta.dimx/dimy/dimz), then run genuine per-trial beta-series ("LSA": every
trial's picture/sentence event gets its own unique regressor in one big
design matrix, rather than one shared regressor per condition) instead of
the crude "just take the raw 16-TR window and flatten it" approach used
previously.

Event timing: from the CMU toolbox's own windowing recipe (see
RESEARCH_NOTES.md Section 3.1) -- first stimulus at trial-relative t=0,
second stimulus at trial-relative t=8s (TR 16 at 0.5s/TR), each lasting the
documented 4s presentation. A canonical HRF (nilearn's 'spm' model) is
convolved with these onsets, which is a real improvement over manually
windowing the raw signal -- it explicitly models the hemodynamic delay
instead of just capturing a fixed window that happens to include it.
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
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RNG_SEED = 42
TR = 0.5  # seconds; "fMRI images were being collected in every 500 msec" (thesis, Sec 3.1)
STIM_DURATION = 4.0  # seconds; documented stimulus presentation duration
SECOND_STIM_OFFSET = 8.0  # seconds; local TR 16 * 0.5s, matches the TR[17:32] window start

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
    """Scatter each trial's (T x V) voxel matrix back into (x,y,z,t) space
    using meta.colToCoord, placing each trial at its GLOBAL time index
    (info[i].mint : info[i].maxt are indices into one continuous session
    timeline, confirmed contiguous across trials in Stage 1)."""
    dimx, dimy, dimz = int(meta.dimx), int(meta.dimy), int(meta.dimz)
    col_to_coord = np.asarray(meta.colToCoord, dtype=int) - 1  # MATLAB 1-indexed -> 0-indexed

    n_timepoints = int(max(int(e.maxt) for e in info))
    volume = np.zeros((dimx, dimy, dimz, n_timepoints), dtype=np.float32)
    mask = np.zeros((dimx, dimy, dimz), dtype=np.uint8)
    mask[col_to_coord[:, 0], col_to_coord[:, 1], col_to_coord[:, 2]] = 1

    for i, entry in enumerate(info):
        d = np.asarray(data[i], dtype=np.float32)  # (T_i, V)
        t0 = int(entry.mint) - 1  # 0-indexed global start
        t1 = t0 + d.shape[0]
        volume[col_to_coord[:, 0], col_to_coord[:, 1], col_to_coord[:, 2], t0:t1] = d.T

    affine = np.eye(4)  # no real-world mm spacing available in the .mat metadata;
                         # internally consistent voxel-space affine is sufficient
                         # here since GLM fit / masking / contrasts never leave it
    return nib.Nifti1Image(volume, affine), nib.Nifti1Image(mask, affine)


def build_events(info):
    """One row per stimulus per real trial (cond>1), with a UNIQUE trial_type
    label per event -- true beta-series ("LSA": least-squares-all) so each
    trial gets its own regressor and thus its own beta map, rather than one
    shared regressor per condition pooling all trials together."""
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


def fit_beta_series(subject_id, path):
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


def active_voxel_ranking(X, y):
    scores = np.zeros(X.shape[1])
    for c in np.unique(y):
        f_vals, _ = f_classif(X, (y == c).astype(int))
        scores += np.nan_to_num(f_vals, nan=0.0)
    return np.argsort(-scores)


def classifiers_bank():
    return {
        "NaiveBayes": lambda: GaussianNB(),
        "Logistic": lambda: Pipeline([("scale", StandardScaler()),
                                       ("clf", LogisticRegression(max_iter=5000))]),
        "SVM_linear": lambda: Pipeline([("scale", StandardScaler()),
                                         ("clf", SVC(kernel="linear"))]),
    }


def sweep(X, y, groups, k_values):
    ranking = active_voxel_ranking(X, y)
    gkf = GroupKFold(n_splits=10)
    clf_bank = classifiers_bank()

    curve = []
    for k in k_values:
        k = min(k, X.shape[1])
        idx = ranking[:k]
        Xk = X[:, idx]
        row = {"k": k}
        for name, fn in clf_bank.items():
            scores = cross_val_score(fn(), Xk, y, cv=gkf, groups=groups, n_jobs=-1)
            row[name] = float(scores.mean())
        curve.append(row)
        print(f"  k={k}: " + ", ".join(f"{kk}={vv:.3f}" for kk, vv in row.items() if kk != "k"))
    return curve


def main():
    results = {}
    for sid, url in STARPLUS_URLS.items():
        print(f"\n{'='*70}\nSubject {sid}: reconstructing volume + fitting beta-series GLM\n{'='*70}")
        dest = os.path.join(STARPLUS_DIR, f"data-starplus-{sid}-v7.mat")
        download(url, dest)

        t0 = time.time()
        X, y, groups = fit_beta_series(sid, dest)
        print(f"[{sid}] beta-series done in {time.time()-t0:.1f}s: X={X.shape}, "
              f"label_counts={pd.Series(y).value_counts().to_dict()}, "
              f"n_trials_grouped={len(np.unique(groups))}")

        k_values = [10, 25, 50, 100, 150, 200, 300, 500, 1000, X.shape[1]]
        t0 = time.time()
        curve = sweep(X, y, groups, k_values)
        print(f"[{sid}] sweep done in {time.time()-t0:.1f}s")

        best_row = max(curve, key=lambda r: max(v for kk, v in r.items() if kk != "k"))
        best_key = max((kk for kk in best_row if kk != "k"), key=lambda kk: best_row[kk])
        print(f"[{sid}] Best: k={best_row['k']}, {best_key}={best_row[best_key]:.3f}")

        results[sid] = {"X_shape": list(X.shape), "sweep_curve": curve,
                         "best": {"k": best_row["k"], "classifier": best_key, "accuracy": best_row[best_key]}}

    print(f"\n{'='*70}\nSummary vs. raw-TR-window StarPlus results (RESEARCH_NOTES.md Sec 3.3)\n{'='*70}")
    raw_window_best = {"04799": 0.762, "04820": 0.787, "04847": 1.000}  # best across normalization variants, Table 4 (ROI)
    for sid, r in results.items():
        b = r["best"]
        print(f"  {sid}: GLM beta-series={b['accuracy']:.3f} (k={b['k']}, {b['classifier']}) "
              f"vs raw-window best={raw_window_best.get(sid, 'n/a')}")

    out_path = "/kaggle/working/starplus_glm_results.json" if os.path.isdir("/kaggle/working") else "./starplus_glm_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
