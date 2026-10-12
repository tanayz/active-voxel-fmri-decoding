"""
What do leaky evaluation protocols report when the true accuracy is chance? On real Haxby data, labels
are permuted block-wise within each run (block structure and run membership preserved; any genuine
stimulus information destroyed), and three protocols are scored on both the real and the permuted
labels:

  A  nilearn example as published: SelectKBest(F, 500) -> linear SVC (OvR), cross_val_score(cv=5),
     i.e. stratified 5-fold over volumes, NOT grouped by run (whole-brain candidate pool);
  B  the same pipeline under leave-one-run-out (run-grouped; selection inside the pipeline);
  C  "pre-fix" flat sweep: voxels ranked ONCE on all data (summed OvR F) before the split, then
     k in {50,100,200,500,1000} x C in {0.1,1,10} swept on the SAME leave-one-run-out folds used for
     evaluation and the best cell reported -- exactly the leaky pattern quantified in the paper.
     Run on both the whole-brain and the VT candidate pool.

Under the null, a protocol that is not biased should average 1/8 = 0.125. Set SUBJECTS to split
across kernels. Results are saved after each permutation.
"""
import json
import os
import platform
import sys
import time

import nibabel as nib
import numpy as np
import pandas as pd
import scipy
import sklearn
import nilearn
from nilearn import datasets
from nilearn.maskers import NiftiMasker
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SUBJECTS = [1, 2, 3, 4, 5, 6]
N_PERMUTATIONS = 20
SWEEP_K = [50, 100, 200, 500, 1000]
SWEEP_C = [0.1, 1.0, 10.0]
RNG_SEED = 7
OUT_NAME = "haxby_null_leakage.json"


def versions():
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "nilearn": nilearn.__version__}


def load_data(subject):
    haxby = datasets.fetch_haxby(subjects=[subject])
    mask_vt = haxby.mask_vt[0] if isinstance(haxby.mask_vt, list) else haxby.mask_vt
    labels = pd.read_csv(haxby.session_target[0], sep=" ")
    y_all, runs_all = labels["labels"].values, labels["chunks"].values
    masker = NiftiMasker(mask_img=haxby.mask, runs=runs_all, smoothing_fwhm=4,
                         memory="nilearn_cache", memory_level=1, verbose=0)
    X_all = masker.fit_transform(haxby.func[0]).astype(np.float32)
    keep = y_all != "rest"
    whole = nib.load(haxby.mask).get_fdata() > 0
    vt = nib.load(mask_vt).get_fdata() > 0
    vt_idx = np.where(vt.ravel()[np.where(whole.ravel())[0]])[0]
    return X_all[keep], y_all[keep], runs_all[keep], vt_idx


def block_permute_labels(y, groups, rng):
    y_perm = y.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        yg = y[idx]
        b = np.r_[0, np.where(yg[1:] != yg[:-1])[0] + 1, len(yg)]
        blocks = [(b[i], b[i + 1]) for i in range(len(b) - 1)]
        for (s, e), lab in zip(blocks, rng.permutation([yg[s] for s, _ in blocks])):
            y_perm[idx[s:e]] = lab
    return y_perm


def tutorial_pipeline():
    return OneVsRestClassifier(Pipeline([("anova", SelectKBest(f_classif, k=500)), ("svc", SVC(kernel="linear"))]))


def summed_f_ranking(X, y):
    scores = np.zeros(X.shape[1])
    for c in np.unique(y):
        f_vals, _ = f_classif(X, (y == c).astype(int))
        scores += np.nan_to_num(f_vals, nan=0.0)
    return np.argsort(-scores)


def leaky_flat_sweep(X, y, runs):
    """Rank on ALL data, then sweep (k, C) on the evaluation folds and report the best cell."""
    ranking = summed_f_ranking(X, y)
    best = 0.0
    for k in SWEEP_K:
        Xk = X[:, ranking[:min(k, X.shape[1])]]
        for C in SWEEP_C:
            clf = Pipeline([("scale", StandardScaler()),
                            ("clf", OneVsRestClassifier(LogisticRegression(C=C, max_iter=5000)))])
            acc = cross_val_score(clf, Xk, y, cv=LeaveOneGroupOut(), groups=runs, n_jobs=-1).mean()
            best = max(best, float(acc))
    return best


def score_all(X, y, runs, vt_idx):
    return {
        "A_tutorial_cv5_ungrouped": float(cross_val_score(tutorial_pipeline(), X, y, cv=5, n_jobs=-1).mean()),
        "B_tutorial_loro": float(cross_val_score(tutorial_pipeline(), X, y, cv=LeaveOneGroupOut(),
                                                 groups=runs, n_jobs=-1).mean()),
        "C_leaky_sweep_wholebrain": leaky_flat_sweep(X, y, runs),
        "C_leaky_sweep_vt": leaky_flat_sweep(X[:, vt_idx], y, runs),
    }


def out_path():
    return os.path.join("/kaggle/working", OUT_NAME) if os.path.isdir("/kaggle/working") else OUT_NAME


def main():
    print("Library versions:", json.dumps(versions()))
    results = {"versions": versions(), "n_permutations": N_PERMUTATIONS,
               "permutation_scheme": "block-wise within run", "subjects": {}}
    for subject in SUBJECTS:
        t0 = time.time()
        X, y, runs, vt_idx = load_data(subject)
        print(f"\n[{subject}] loaded {X.shape}, VT={len(vt_idx)} ({time.time() - t0:.0f}s)")
        real = score_all(X, y, runs, vt_idx)
        print(f"[{subject}] real labels: " + ", ".join(f"{k}={v:.3f}" for k, v in real.items()))
        sys.stdout.flush()
        rec = {"real": real, "null": {k: [] for k in real}}
        rng = np.random.RandomState(RNG_SEED + subject)
        t0 = time.time()
        for p in range(N_PERMUTATIONS):
            s = score_all(X, block_permute_labels(y, runs, rng), runs, vt_idx)
            for k, v in s.items():
                rec["null"][k].append(v)
            print(f"[{subject}] perm {p + 1:>2d}/{N_PERMUTATIONS}: " +
                  ", ".join(f"{k}={v:.3f}" for k, v in s.items()) + f" ({time.time() - t0:.0f}s)")
            sys.stdout.flush()
            rec["null_mean"] = {k: float(np.mean(v)) for k, v in rec["null"].items()}
            results["subjects"][str(subject)] = rec
            with open(out_path(), "w") as fh:
                json.dump(results, fh, indent=2)
        print(f"[{subject}] null means: " + ", ".join(f"{k}={v:.3f}" for k, v in rec["null_mean"].items()))


if __name__ == "__main__":
    main()
