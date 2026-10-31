"""
Is there object-category information OUTSIDE ventral temporal cortex? Permutation test for decoding from
the whole-brain mask minus the VT mask (haxby_selection_anatomy.py reports the nested accuracies).

Pipeline per leave-one-run-out fold, with fixed hyperparameters at the values the nested search chose
most often for this candidate pool (k = 1000, C = 1): summed one-vs-rest F ranking on the training runs,
top-k voxels, training-fold standardization, one-vs-rest logistic regression. The identical pipeline is
re-run under N_PERMUTATIONS block-wise within-run label permutations (block structure, block length and
run membership preserved); p = (1 + #{null >= observed}) / (N + 1). Set SUBJECTS to split across kernels.
"""
import json
import os
import platform
import time

import nibabel as nib
import numpy as np
import pandas as pd
import scipy
import sklearn
import nilearn
from nilearn import datasets
from nilearn.maskers import NiftiMasker
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

SUBJECTS = [1, 2, 3, 4, 5, 6]
K, C = 1000, 1.0
N_PERMUTATIONS = 50
SEED = 0
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUT_NAME = "haxby_outside_vt_perm.json"


def versions():
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "nilearn": nilearn.__version__}


def fetch_with_retry(subject, attempts=5, wait=120):
    for i in range(attempts):
        try:
            return datasets.fetch_haxby(subjects=[subject])
        except Exception as exc:
            print(f"fetch_haxby attempt {i + 1}/{attempts} failed: {exc!r}; retrying in {wait}s", flush=True)
            time.sleep(wait)
    return datasets.fetch_haxby(subjects=[subject])


def block_permute_labels(y, groups, rng):
    """Shuffle category labels among the contiguous stimulus blocks WITHIN each run."""
    y_perm = y.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        yg = y[idx]
        boundaries = np.r_[0, np.where(yg[1:] != yg[:-1])[0] + 1, len(yg)]
        blocks = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
        permuted = rng.permutation([yg[s] for s, _ in blocks])
        for (s, e), lab in zip(blocks, permuted):
            y_perm[idx[s:e]] = lab
    return y_perm


def summed_f(X, y):
    s = np.zeros(X.shape[1])
    for c in np.unique(y):
        f, _ = f_classif(X, (y == c).astype(int))
        s += np.nan_to_num(f, nan=0.0)
    return s


def loro_accuracy(X, y, runs):
    scores = []
    for tr, te in LeaveOneGroupOut().split(X, y, runs):
        idx = np.argsort(-summed_f(X[tr], y[tr]))[:K]
        sc = StandardScaler().fit(X[tr][:, idx])
        clf = OneVsRestClassifier(LogisticRegression(C=C, max_iter=5000), n_jobs=-1)
        clf.fit(sc.transform(X[tr][:, idx]), y[tr])
        scores.append(float(clf.score(sc.transform(X[te][:, idx]), y[te])))
    return float(np.mean(scores)), scores


def main():
    print("Library versions:", json.dumps(versions()), flush=True)
    rng = np.random.default_rng(SEED)
    results = {"versions": versions(), "config": {"k": K, "C": C, "n_permutations": N_PERMUTATIONS, "seed": SEED},
               "subjects": {}}
    for subject in SUBJECTS:
        t0 = time.time()
        h = fetch_with_retry(subject)
        labels = pd.read_csv(h.session_target[0], sep=" ")
        y_all, runs_all = labels["labels"].values, labels["chunks"].values
        whole = nib.load(h.mask).get_fdata() > 0
        mvt = h.mask_vt[0] if isinstance(h.mask_vt, list) else h.mask_vt
        vt = (nib.load(mvt).get_fdata() > 0).ravel()[np.where(whole.ravel())[0]]
        masker = NiftiMasker(mask_img=h.mask, runs=runs_all, smoothing_fwhm=4,
                             memory="nilearn_cache", memory_level=1, verbose=0)
        X_all = masker.fit_transform(h.func[0]).astype(np.float32)
        keep = y_all != "rest"
        X, y, runs = X_all[keep][:, ~vt], y_all[keep], runs_all[keep]
        obs, folds = loro_accuracy(X, y, runs)
        print(f"[{subject}] outside VT ({X.shape[1]} voxels), k={K}, C={C}: {obs:.3f} ({time.time() - t0:.0f}s)", flush=True)
        null = []
        for b in range(N_PERMUTATIONS):
            null.append(loro_accuracy(X, block_permute_labels(y, runs, rng), runs)[0])
            if (b + 1) % 10 == 0:
                print(f"[{subject}] {b + 1}/{N_PERMUTATIONS} perms, null mean {np.mean(null):.3f} max {max(null):.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        p = (1 + sum(v >= obs for v in null)) / (N_PERMUTATIONS + 1)
        results["subjects"][str(subject)] = {"n_candidates": int(X.shape[1]), "observed": obs, "per_fold": folds,
                                             "null": null, "null_mean": float(np.mean(null)),
                                             "null_max": float(max(null)), "p": p}
        print(f"[{subject}] p = {p:.4f}", flush=True)
        with open(os.path.join(OUT_DIR, OUT_NAME), "w") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
