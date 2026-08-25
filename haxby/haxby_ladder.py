"""
Decomposition ladder for Haxby: which ingredients carry the run-grouped nilearn example (0.372 on
subject 2) up to our no-selection pipeline (0.708)? Every step below is evaluated with leave-one-run-out
over scanner runs, rest removed, on the SAME masked/smoothed data used by the paper's final analysis.

  A  tutorial pipeline, VT mask:  VT voxels -> OvR[ SelectKBest(F, k=500) -> linear SVC(C=1), unscaled ]
     (identical to nilearn's plot_haxby_multiclass.py except that the candidate pool is the subject's VT
     mask instead of the whole brain; with 307-675 VT voxels, k=500 keeps all or most of them)
  B  + feature standardisation:   VT voxels -> OvR[ StandardScaler -> linear SVC(C=1) ]
  C  + logistic regression:       VT voxels -> OvR[ StandardScaler -> LogisticRegression(C=1) ]
  D  + tuned C (nested):          = the "ours, no selection" column of haxby_vt_final.py (not re-run here)

Per-fold scores are saved so fold-level SDs can be reported. Library versions are recorded.
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
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SUBJECTS = [1, 2, 3, 4, 5, 6]
OUT_NAME = "haxby_ladder_results.json"


def versions():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scikit-learn": sklearn.__version__, "nilearn": nilearn.__version__, "nibabel": nib.__version__}


def load_data_and_vt_index(subject):
    haxby = datasets.fetch_haxby(subjects=[subject])
    func, mask = haxby.func[0], haxby.mask
    mask_vt = haxby.mask_vt[0] if isinstance(haxby.mask_vt, list) else haxby.mask_vt
    labels = pd.read_csv(haxby.session_target[0], sep=" ")
    y_all, runs_all = labels["labels"].values, labels["chunks"].values
    masker = NiftiMasker(mask_img=mask, runs=runs_all, smoothing_fwhm=4,
                         memory="nilearn_cache", memory_level=1, verbose=0)
    X_all = masker.fit_transform(func)
    non_rest = y_all != "rest"
    X, y, runs = X_all[non_rest], y_all[non_rest], runs_all[non_rest]
    whole = nib.load(mask).get_fdata() > 0
    vt = nib.load(mask_vt).get_fdata() > 0
    vt_idx = np.where(vt.ravel()[np.where(whole.ravel())[0]])[0]
    if len(vt_idx) == 0:
        raise ValueError(f"subject {subject}: VT mask projected to 0 voxels")
    return X[:, vt_idx], y, runs


def steps():
    return {
        "A_tutorial_pipeline_VT": OneVsRestClassifier(Pipeline([
            ("anova", SelectKBest(f_classif, k=500)), ("svc", SVC(kernel="linear"))])),
        "B_plus_standardisation": OneVsRestClassifier(Pipeline([
            ("scale", StandardScaler()), ("svc", SVC(kernel="linear"))])),
        "C_plus_logistic_regression": OneVsRestClassifier(Pipeline([
            ("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))])),
    }


def main():
    print("Library versions:", json.dumps(versions()))
    results = {"versions": versions(), "outer_cv": "LeaveOneGroupOut(run)", "subjects": {}}
    for subject in SUBJECTS:
        t0 = time.time()
        X, y, runs = load_data_and_vt_index(subject)
        print(f"\n[{subject}] loaded in {time.time() - t0:.0f}s: VT voxels={X.shape[1]}, examples={X.shape[0]}")
        sys.stdout.flush()
        r = {"n_vt_voxels": int(X.shape[1]), "n_examples": int(X.shape[0])}
        for name, clf in steps().items():
            t0 = time.time()
            scores = cross_val_score(clf, X, y, cv=LeaveOneGroupOut(), groups=runs, n_jobs=-1)
            r[name] = {"mean": float(scores.mean()), "sd": float(scores.std(ddof=1)), "per_fold": scores.tolist()}
            print(f"[{subject}] {name:<28s} {scores.mean():.3f} ± {scores.std(ddof=1):.3f}  ({time.time() - t0:.0f}s)")
            sys.stdout.flush()
        results["subjects"][str(subject)] = r
        out = os.path.join("/kaggle/working", OUT_NAME) if os.path.isdir("/kaggle/working") else OUT_NAME
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
    print("\nSummary (mean over subjects):")
    for name in steps():
        m = np.mean([results["subjects"][s][name]["mean"] for s in results["subjects"]])
        print(f"  {name:<28s} {m:.3f}")


if __name__ == "__main__":
    main()
