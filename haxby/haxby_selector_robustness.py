"""
Does the interchangeability of feature selection and the anatomical prior depend on the selector or the
classifier?

For each Haxby subject and each candidate pool (whole brain, 39,912 voxels; ventral temporal mask), the
nested leave-one-run-out protocol of the main analysis is repeated for a grid of dimensionality controls
and classifiers:

  dimensionality control (with one-vs-rest logistic regression, C = 1)
    none      all voxels of the pool
    sumf      summed one-vs-rest F ranking (the selection step of the main analysis)
    anova     multi-class one-way ANOVA F ranking (scikit-learn's f_classif, as in SelectKBest)
    random    k voxels drawn at random (control for dimensionality alone)
    variance  the k highest-variance voxels (label-free filter)
    pca       the first k principal components of the standardized pool (label-free projection)

  classifier (with no control and with the sumf selection)
    logreg    one-vs-rest l2 logistic regression, C = 1
    linsvm    linear support vector machine, C = 1
    rbfsvm    Gaussian-kernel support vector machine, C = 1, gamma = "scale"

k (or the number of components) is chosen in an inner GroupKFold(4) over runs; every fit, ranking and
projection sees training runs only. C is fixed at 1 because the main analysis found it immaterial.
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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

SUBJECTS = [1, 2, 3, 4, 5, 6]
K_WB = [100, 500, 1000, 2000, 5000]
K_VT = [50, 100, 200, 300, 500]
K_PCA = [20, 50, 100, 200, 400]
INNER_SPLITS = 4
SEED = 0
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUT_NAME = "haxby_selector_robustness.json"

CONTROL_ARMS = [("none", "logreg"), ("sumf", "logreg"), ("anova", "logreg"), ("random", "logreg"),
                ("variance", "logreg"), ("pca", "logreg")]
CLASSIFIER_ARMS = [("none", "linsvm"), ("sumf", "linsvm"), ("none", "rbfsvm"), ("sumf", "rbfsvm")]


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


class Reduce(BaseEstimator, TransformerMixin):
    """Dimensionality control fitted on the training data only."""

    def __init__(self, method="sumf", k=500, seed=SEED):
        self.method = method
        self.k = k
        self.seed = seed

    def fit(self, X, y):
        p = X.shape[1]
        k = min(int(self.k), p)
        self.pca_ = None
        if self.method == "sumf":
            self.idx_ = np.argsort(-summed_f(X, y))[:k]
        elif self.method == "anova":
            f, _ = f_classif(X, y)
            self.idx_ = np.argsort(-np.nan_to_num(f, nan=0.0))[:k]
        elif self.method == "random":
            self.idx_ = np.random.RandomState(self.seed).permutation(p)[:k]
        elif self.method == "variance":
            self.idx_ = np.argsort(-X.var(axis=0))[:k]
        elif self.method == "pca":
            self.scaler_ = StandardScaler().fit(X)
            k = min(k, X.shape[0] - 1, p)
            self.pca_ = PCA(n_components=k, svd_solver="randomized", random_state=self.seed)
            self.pca_.fit(self.scaler_.transform(X))
        else:
            raise ValueError(self.method)
        return self

    def transform(self, X):
        if self.pca_ is not None:
            return self.pca_.transform(self.scaler_.transform(X))
        return X[:, self.idx_]


def make_clf(name):
    if name == "logreg":
        return OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=5000))
    if name == "linsvm":
        return LinearSVC(C=1.0, max_iter=20000)
    if name == "rbfsvm":
        return SVC(C=1.0, kernel="rbf", gamma="scale")
    raise ValueError(name)


def run_arm(X, y, runs, method, clf_name, k_grid):
    outer = LeaveOneGroupOut()
    scores, ks = [], []
    if method == "none":
        pipe = Pipeline([("scale", StandardScaler()), ("clf", make_clf(clf_name))])
        for tr, te in outer.split(X, y, runs):
            pipe.fit(X[tr], y[tr])
            scores.append(float(pipe.score(X[te], y[te])))
            ks.append(int(X.shape[1]))
        return scores, ks
    pipe = Pipeline([("reduce", Reduce(method=method)), ("scale", StandardScaler()),
                     ("clf", make_clf(clf_name))])
    for tr, te in outer.split(X, y, runs):
        grid = [k for k in k_grid if k < X.shape[1]] or [X.shape[1]]
        g = GridSearchCV(pipe, {"reduce__k": grid}, cv=GroupKFold(n_splits=INNER_SPLITS),
                         scoring="accuracy", n_jobs=-1)
        g.fit(X[tr], y[tr], groups=runs[tr])
        scores.append(float(g.score(X[te], y[te])))
        ks.append(int(g.best_params_["reduce__k"]))
    return scores, ks


def main():
    print("Library versions:", json.dumps(versions()), flush=True)
    out_path = os.path.join(OUT_DIR, OUT_NAME)
    results = {"versions": versions(), "k_wb": K_WB, "k_vt": K_VT, "k_pca": K_PCA,
               "inner_splits": INNER_SPLITS, "seed": SEED, "subjects": {}}
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
        pools = {"wb": (X, K_WB), "vt": (X[:, vt], K_VT)}
        print(f"[{subject}] loaded {X.shape}, VT={len(vt)}", flush=True)
        res = {"n_wb": int(X.shape[1]), "n_vt": int(len(vt)), "arms": {}}
        for pool, (Xp, k_grid) in pools.items():
            for method, clf_name in CONTROL_ARMS + CLASSIFIER_ARMS:
                t0 = time.time()
                grid = K_PCA if method == "pca" else k_grid
                scores, ks = run_arm(Xp, y, runs, method, clf_name, grid)
                key = f"{pool}/{method}/{clf_name}"
                res["arms"][key] = {"mean": float(np.mean(scores)), "sd": float(np.std(scores)),
                                    "per_fold": scores, "k": ks}
                print(f"[{subject}] {key}: {np.mean(scores):.3f} +/- {np.std(scores):.3f} "
                      f"k={sorted(set(ks))} ({time.time() - t0:.0f}s)", flush=True)
                results["subjects"][str(subject)] = res
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
