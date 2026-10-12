"""
Boundary condition for the "selection adds nothing" finding. Within the ventral temporal (VT) mask
(307-675 voxels) active-voxel selection added nothing on Haxby; StarPlus (~5,000 voxels, 80 patterns)
showed the same. Does selection matter once the candidate pool is the WHOLE brain (39,912 voxels,
864 volumes)? This script runs the protocol of haxby_vt_final.py unchanged except for the candidate
pool, in two arms per subject:

  (1) active-voxel selection, k in {100, 500, 1000, 2000, 5000, "all"} (per-fold choice recorded);
  (2) no selection: every whole-brain voxel.

Both: StandardScaler -> OvR LogisticRegression, C in {0.1, 1, 10}; outer LeaveOneGroupOut over runs,
inner GroupKFold(4). Set SUBJECTS to split across Kaggle kernels. Results are saved after each outer
fold so a killed kernel leaves partial output.
"""
import json
import os
import platform
import sys
import time
from collections import Counter

import nibabel as nib
import numpy as np
import pandas as pd
import scipy
import sklearn
import nilearn
from nilearn import datasets
from nilearn.maskers import NiftiMasker
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SUBJECTS = [1, 2, 3, 4, 5, 6]
INNER_SPLITS = 4
K_VALUES = [100, 500, 1000, 2000, 5000, "all"]
C_VALUES = [0.1, 1.0, 10.0]
OUT_NAME = "haxby_wholebrain_ablation.json"
ARMS = ["no_selection", "selection"]   # restrict to run one arm only


def versions():
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "nilearn": nilearn.__version__,
            "nibabel": nib.__version__}


def fetch_with_retry(subject, attempts=5, wait=120):
    for i in range(attempts):
        try:
            return datasets.fetch_haxby(subjects=[subject])
        except Exception as exc:   # transient mirror/TLS failures (e.g. nitrc.org) must not kill a long run
            print(f"fetch_haxby attempt {i + 1}/{attempts} failed: {exc!r}; retrying in {wait}s")
            sys.stdout.flush()
            time.sleep(wait)
    return datasets.fetch_haxby(subjects=[subject])


def load_data(subject):
    haxby = fetch_with_retry(subject)
    labels = pd.read_csv(haxby.session_target[0], sep=" ")
    y_all, runs_all = labels["labels"].values, labels["chunks"].values
    masker = NiftiMasker(mask_img=haxby.mask, runs=runs_all, smoothing_fwhm=4,
                         memory="nilearn_cache", memory_level=1, verbose=0)
    X_all = masker.fit_transform(haxby.func[0]).astype(np.float32)
    keep = y_all != "rest"
    return X_all[keep], y_all[keep], runs_all[keep]


class ActiveVoxelSelector(BaseEstimator, TransformerMixin):
    """Summed one-vs-rest F ranking; k='all' retains every voxel."""

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
        if self.k == "all":
            return X
        return X[:, self.ranking_[:min(int(self.k), X.shape[1])]]


def make_grid(with_selection):
    steps = [("select", ActiveVoxelSelector())] if with_selection else []
    steps += [("scale", StandardScaler()), ("clf", OneVsRestClassifier(LogisticRegression(max_iter=5000)))]
    g = {"clf__estimator__C": C_VALUES}
    if with_selection:
        g["select__k"] = K_VALUES
    return GridSearchCV(Pipeline(steps), g, cv=GroupKFold(n_splits=INNER_SPLITS),
                        scoring="accuracy", n_jobs=-1, refit=True)


def out_path():
    return os.path.join("/kaggle/working", OUT_NAME) if os.path.isdir("/kaggle/working") else OUT_NAME


def main():
    print("Library versions:", json.dumps(versions()))
    print(f"Config: subjects={SUBJECTS} k={K_VALUES} C={C_VALUES} inner=GroupKFold({INNER_SPLITS})")
    sys.stdout.flush()
    results = {"versions": versions(),
               "config": {"k_values": [str(k) for k in K_VALUES], "c_values": C_VALUES,
                          "outer_cv": "LeaveOneGroupOut(run)", "inner_cv": f"GroupKFold({INNER_SPLITS})",
                          "candidate_pool": "whole-brain mask", "arms": ARMS},
               "subjects": {}}
    for subject in SUBJECTS:
        t0 = time.time()
        X, y, runs = load_data(subject)
        print(f"\n[{subject}] loaded in {time.time() - t0:.0f}s: X={X.shape}")
        sys.stdout.flush()
        rec = {"n_voxels": int(X.shape[1]), "n_examples": int(X.shape[0])}
        for arm, with_sel in [(a, a == "selection") for a in ARMS]:
            scores, choices = [], []
            t0 = time.time()
            for f, (tr, te) in enumerate(LeaveOneGroupOut().split(X, y, runs)):
                grid = make_grid(with_sel)
                grid.fit(X[tr], y[tr], groups=runs[tr])
                scores.append(float(grid.score(X[te], y[te])))
                bp = grid.best_params_
                choices.append({"k": str(bp.get("select__k", "all")), "C": bp["clf__estimator__C"]})
                print(f"[{subject}] {arm:<12s} fold {f + 1:>2d}: acc={scores[-1]:.3f} "
                      f"k={choices[-1]['k']} C={choices[-1]['C']} ({time.time() - t0:.0f}s)")
                sys.stdout.flush()
                rec[arm] = {"per_fold_scores": scores, "per_fold_choices": choices,
                            "mean": float(np.mean(scores)),
                            "sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                            "complete": False}
                results["subjects"][str(subject)] = rec
                with open(out_path(), "w") as fh:
                    json.dump(results, fh, indent=2)
            rec[arm]["complete"] = True
            if with_sel:
                rec[arm]["k_histogram"] = dict(Counter(c["k"] for c in choices))
            print(f"[{subject}] {arm}: mean={np.mean(scores):.3f} sd={np.std(scores, ddof=1):.3f}")
        if "selection" in rec and "no_selection" in rec:
            d = rec["selection"]["mean"] - rec["no_selection"]["mean"]
            print(f"[{subject}] delta(selection - no_selection) = {d:+.3f}  k={rec['selection']['k_histogram']}")
        with open(out_path(), "w") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
