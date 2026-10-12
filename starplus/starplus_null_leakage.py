"""
StarPlus counterpart of haxby_null_leakage.py: what does the pre-fix "flat sweep" protocol report when
the labels carry no information? Voxels are ranked ONCE on all 80 beta-series patterns (summed OvR F)
before splitting, then k in {10,25,50,100,150,250,500,1000} x {GaussianNB, LogisticRegression, linear SVC}
is swept on the SAME trial-grouped 10-fold CV used for evaluation and the best cell reported -- the
pre-fix pipeline of starplus_glm_benchmark.py. Scored on the real labels and on N_NULL label
permutations (permuted across examples; trial grouping of folds retained). Chance = 0.5.
Loader / beta-series code is spliced verbatim from starplus_ablation.py.
"""
import json
import os
import platform
import sys
import time
import urllib.request
from collections import Counter

import nibabel as nib
import numpy as np
import pandas as pd
import scipy
import sklearn
import nilearn
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
K_VALUES = [10, 25, 50, 100, 250, 500, 1000, "all"]
C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
N_PERMUTATIONS = 50   # permutation test for the NO-SELECTION arm (full nested procedure per permutation,
                      # labels permuted across examples with the trial grouping of the folds retained --
                      # the same scheme as starplus_permutation_test.py uses for the selection arm)

DATA_DIR = "/kaggle/working/data" if os.path.isdir("/kaggle/working") else "./data"
STARPLUS_DIR = os.path.join(DATA_DIR, "starplus")
os.makedirs(STARPLUS_DIR, exist_ok=True)

STARPLUS_URLS = {
    "04799": "http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-81/www/data-starplus-04799-v7.mat",
    "04820": "http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-83/www/data-starplus-04820-v7.mat",
    "04847": "http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-81/www/data-starplus-04847-v7.mat",
}


def versions():
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "nilearn": nilearn.__version__,
            "nibabel": nib.__version__}


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



from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

N_NULL = 50
SWEEP_K = [10, 25, 50, 100, 150, 250, 500, 1000]


def summed_f_ranking(X, y):
    scores = np.zeros(X.shape[1])
    for c in np.unique(y):
        f_vals, _ = f_classif(X, (y == c).astype(int))
        scores += np.nan_to_num(f_vals, nan=0.0)
    return np.argsort(-scores)


def leaky_flat_sweep(X, y, groups):
    ranking = summed_f_ranking(X, y)
    best = 0.0
    for k in SWEEP_K:
        Xk = X[:, ranking[:min(k, X.shape[1])]]
        for clf in (GaussianNB(),
                    Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))]),
                    Pipeline([("scale", StandardScaler()), ("clf", SVC(kernel="linear"))])):
            acc = cross_val_score(clf, Xk, y, cv=GroupKFold(n_splits=10), groups=groups, n_jobs=-1).mean()
            best = max(best, float(acc))
    return best


def out_path():
    return os.path.join("/kaggle/working", "starplus_null_leakage.json")         if os.path.isdir("/kaggle/working") else "starplus_null_leakage.json"


def main():
    print("Library versions:", json.dumps(versions()))
    results = {"versions": versions(), "n_null": N_NULL, "sweep_k": SWEEP_K, "subjects": {}}
    for sid, url in STARPLUS_URLS.items():
        dest = os.path.join(STARPLUS_DIR, f"data-starplus-{sid}-v7.mat")
        download(url, dest)
        X, y, groups = fit_beta_series(dest)
        real = leaky_flat_sweep(X, y, groups)
        rng = np.random.RandomState(RNG_SEED)
        null = [leaky_flat_sweep(X, rng.permutation(y), groups) for _ in range(N_NULL)]
        results["subjects"][sid] = {"real": real, "null": null, "null_mean": float(np.mean(null)),
                                    "null_sd": float(np.std(null, ddof=1)), "null_max": float(np.max(null))}
        print(f"[{sid}] leaky sweep: real={real:.3f}  null mean={np.mean(null):.3f} "
              f"sd={np.std(null, ddof=1):.3f} max={np.max(null):.3f}")
        sys.stdout.flush()
        with open(out_path(), "w") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
