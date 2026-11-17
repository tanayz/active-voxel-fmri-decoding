"""
Does a tuned Gaussian-kernel SVM still gain from feature selection inside the anatomical prior?

Companion to haxby_selector_robustness.py, which ran the Gaussian-kernel SVM with default hyperparameters
(C = 1, gamma = "scale"). Here C and the kernel width are tuned in the inner loop of the same nested
leave-one-run-out protocol, for both candidate pools (whole brain, 39,912 voxels; ventral temporal mask)
and for two arms:

    none   all voxels of the pool;           inner grid over (C, gamma)
    sumf   summed one-vs-rest F ranking;     inner grid over (k, C, gamma)

gamma = g / (n_features * variance), i.e. g times scikit-learn's "scale" value, so that the grid means
the same thing for every k. The inner loop is GroupKFold(4) over runs; ranking and standardization use
training runs only. Because the kernel for a given width depends on the data only through pairwise
squared distances, these are computed once per split and every (C, gamma) is fitted on a precomputed
kernel, which makes the grid cheap.

The default arm (C = 1, g = 1; k chosen in the inner loop) is recomputed from the same tables and must
reproduce the corresponding arm of haxby_selector_robustness.py.
Set SUBJECTS to split across kernels.
"""
import json
import os
import platform
import sys
import time

import nibabel as nib
import numpy as np
import pandas as pd
import sklearn
import nilearn
from nilearn import datasets
from nilearn.maskers import NiftiMasker
from sklearn.feature_selection import f_classif
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.svm import SVC

SUBJECTS = [1, 2, 3, 4, 5, 6]
K_WB = [100, 500, 1000, 2000, 5000]
K_VT = [50, 100, 200, 300, 500]
C_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]
G_GRID = [0.01, 0.1, 1.0, 10.0]
DEFAULT = (1.0, 1.0)
INNER_SPLITS = 4
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUT_NAME = "haxby_rbf_tuned.json"


def versions():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scikit-learn": sklearn.__version__, "nilearn": nilearn.__version__, "nibabel": nib.__version__}


def fetch_with_retry(subject, attempts=5, wait=120):
    for i in range(attempts):
        try:
            return datasets.fetch_haxby(subjects=[subject])
        except Exception as exc:
            print(f"fetch_haxby attempt {i + 1}/{attempts} failed: {exc!r}; retrying in {wait}s", flush=True)
            time.sleep(wait)
    return datasets.fetch_haxby(subjects=[subject])


def summed_f(X, y):
    s = np.zeros(X.shape[1])
    for c in np.unique(y):
        f, _ = f_classif(X, (y == c).astype(int))
        s += np.nan_to_num(f, nan=0.0)
    return s


def standardize(Xtr, Xte):
    """StandardScaler semantics: population SD, constant features left at zero."""
    Xtr = Xtr.astype(np.float64)
    Xte = Xte.astype(np.float64)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def accuracy_table(Xtr, ytr, Xte, yte):
    """Accuracy on (Xte, yte) of an RBF SVM trained on (Xtr, ytr) for every (C, g); features given raw."""
    Ztr, Zte = standardize(Xtr, Xte)
    base = 1.0 / (Ztr.shape[1] * Ztr.var())                      # scikit-learn's gamma="scale"
    d_tr = euclidean_distances(Ztr, squared=True)
    d_te = euclidean_distances(Zte, Ztr, squared=True)
    acc = np.zeros((len(C_GRID), len(G_GRID)))
    for gi, g in enumerate(G_GRID):
        k_tr, k_te = np.exp(-g * base * d_tr), np.exp(-g * base * d_te)
        for ci, c in enumerate(C_GRID):
            clf = SVC(C=c, kernel="precomputed").fit(k_tr, ytr)
            acc[ci, gi] = clf.score(k_te, yte)
    return acc


def run_pool(X, y, runs, k_grid):
    """Nested LORO for the none and sumf arms, tuned and default, from shared inner-loop tables."""
    k_grid = [k for k in k_grid if k < X.shape[1]] or [X.shape[1]]
    ci0, gi0 = C_GRID.index(DEFAULT[0]), G_GRID.index(DEFAULT[1])
    out = {a: {"per_fold": [], "chosen": []} for a in
           ("none/tuned", "none/default", "sumf/tuned", "sumf/default")}
    for tr, te in LeaveOneGroupOut().split(X, y, runs):
        Xo, yo, go = X[tr], y[tr], runs[tr]
        inner_none = np.zeros((len(C_GRID), len(G_GRID)))
        inner_sel = np.zeros((len(k_grid), len(C_GRID), len(G_GRID)))
        for itr, iva in GroupKFold(n_splits=INNER_SPLITS).split(Xo, yo, go):
            inner_none += accuracy_table(Xo[itr], yo[itr], Xo[iva], yo[iva]) / INNER_SPLITS
            order = np.argsort(-summed_f(Xo[itr], yo[itr]))
            for ki, k in enumerate(k_grid):
                idx = order[:k]
                inner_sel[ki] += accuracy_table(Xo[itr][:, idx], yo[itr], Xo[iva][:, idx], yo[iva]) / INNER_SPLITS
        outer_none = accuracy_table(Xo, yo, X[te], y[te])
        order = np.argsort(-summed_f(Xo, yo))
        outer_sel = [accuracy_table(Xo[:, order[:k]], yo, X[te][:, order[:k]], y[te]) for k in k_grid]

        ci, gi = np.unravel_index(np.argmax(inner_none), inner_none.shape)       # first maximum, grid order
        out["none/tuned"]["per_fold"].append(float(outer_none[ci, gi]))
        out["none/tuned"]["chosen"].append({"C": C_GRID[ci], "g": G_GRID[gi]})
        out["none/default"]["per_fold"].append(float(outer_none[ci0, gi0]))
        out["none/default"]["chosen"].append({"C": DEFAULT[0], "g": DEFAULT[1]})
        ki, ci, gi = np.unravel_index(np.argmax(inner_sel), inner_sel.shape)
        out["sumf/tuned"]["per_fold"].append(float(outer_sel[ki][ci, gi]))
        out["sumf/tuned"]["chosen"].append({"k": k_grid[ki], "C": C_GRID[ci], "g": G_GRID[gi]})
        ki = int(np.argmax(inner_sel[:, ci0, gi0]))
        out["sumf/default"]["per_fold"].append(float(outer_sel[ki][ci0, gi0]))
        out["sumf/default"]["chosen"].append({"k": k_grid[ki], "C": DEFAULT[0], "g": DEFAULT[1]})
    for a in out:
        out[a]["mean"] = float(np.mean(out[a]["per_fold"]))
        out[a]["sd"] = float(np.std(out[a]["per_fold"]))
    return out


def main():
    print("Library versions:", json.dumps(versions()), flush=True)
    out_path = os.path.join(OUT_DIR, OUT_NAME)
    results = {"versions": versions(), "k_wb": K_WB, "k_vt": K_VT, "c_grid": C_GRID, "g_grid": G_GRID,
               "inner_splits": INNER_SPLITS, "subjects": {}}
    for subject in SUBJECTS:
        h = fetch_with_retry(subject)
        labels = pd.read_csv(h.session_target[0], sep=" ")
        y_all, runs_all = labels["labels"].values, labels["chunks"].values
        whole = nib.load(h.mask).get_fdata() > 0
        masker = NiftiMasker(mask_img=h.mask, runs=runs_all, smoothing_fwhm=4,
                             memory="nilearn_cache", memory_level=1, verbose=0)
        X_all = masker.fit_transform(h.func[0]).astype(np.float32)
        keep = y_all != "rest"
        X, y, runs = X_all[keep], y_all[keep], runs_all[keep]
        mvt = h.mask_vt[0] if isinstance(h.mask_vt, list) else h.mask_vt
        vt_vol = nib.load(mvt).get_fdata() > 0
        vt = np.where(vt_vol.ravel()[np.where(whole.ravel())[0]])[0]
        print(f"[{subject}] loaded {X.shape}, VT={len(vt)}", flush=True)
        res = {"n_wb": int(X.shape[1]), "n_vt": int(len(vt)), "arms": {}}
        for pool, Xp, k_grid in (("vt", X[:, vt], K_VT), ("wb", X, K_WB)):
            t0 = time.time()
            arms = run_pool(Xp, y, runs, k_grid)
            for a, r in arms.items():
                res["arms"][f"{pool}/{a}"] = r
                print(f"[{subject}] {pool}/{a}: {r['mean']:.3f} +/- {r['sd']:.3f}", flush=True)
            print(f"[{subject}] {pool} done ({time.time() - t0:.0f}s)", flush=True)
            results["subjects"][str(subject)] = res
            with open(out_path, "w") as f:
                json.dump(results, f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
