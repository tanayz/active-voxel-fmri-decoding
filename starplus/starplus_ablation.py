"""
StarPlus attribution ablation (requested implicitly by the JBHI screening comments): the paper's
headline StarPlus result combines TWO changes relative to the active-ROI comparison column --
beta-series features AND active-voxel selection -- so the gain could not be assigned to the
selection strategy. This script isolates the selection step on the beta-series features:

  (1) beta-series + active-voxel selection, with the k grid EXTENDED by "all" (no selection) and the
      per-fold chosen (k, C, classifier) recorded -- exactly the design of haxby_vt_final.py, so the
      paper can report how often the tuner declines to select on StarPlus too;
  (2) beta-series with NO selection (all ~4,700-5,000 voxels), identical protocol otherwise
      (StandardScaler -> {LogisticRegression | linear SVC}, C tuned).

Everything else matches starplus_glm_nested_cv.py verbatim: the CMU download, the 4D volume
reconstruction from colToCoord metadata, the LSA beta-series (one regressor per stimulus event,
SPM HRF, cosine drift), outer GroupKFold(10) grouped by trial, inner GroupKFold(6),
C in {1e-3..1e2}. Library versions are printed for pinning.
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


class ActiveVoxelSelector(BaseEstimator, TransformerMixin):
    """Summed one-vs-rest F ranking; k='all' retains every voxel (no selection)."""

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
        if self.k == "all":
            return X
        k = min(int(self.k), X.shape[1])
        return X[:, self.ranking_[:k]]


def build_grid_search(with_selection):
    steps = []
    if with_selection:
        steps.append(("select", ActiveVoxelSelector()))
    steps += [("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))]
    pipe = Pipeline(steps)
    grids = []
    for clf, cparam in ((LogisticRegression(max_iter=5000), "clf__C"), (SVC(kernel="linear"), "clf__C")):
        g = {"clf": [clf], cparam: C_VALUES}
        if with_selection:
            g["select__k"] = K_VALUES
        grids.append(g)
    return GridSearchCV(pipe, grids, cv=GroupKFold(n_splits=6), scoring="accuracy", n_jobs=-1, refit=True)


def describe_params(best_params):
    return {"k": best_params.get("select__k", "all"), "C": best_params["clf__C"],
            "clf": type(best_params["clf"]).__name__}


def nested_cv(X, y, groups, with_selection):
    outer = GroupKFold(n_splits=10)
    scores, choices = [], []
    for train_idx, test_idx in outer.split(X, y, groups):
        grid = build_grid_search(with_selection)
        grid.fit(X[train_idx], y[train_idx], groups=groups[train_idx])
        scores.append(float(grid.score(X[test_idx], y[test_idx])))
        choices.append(describe_params(grid.best_params_))
    return np.array(scores), choices


def out_path():
    return os.path.join("/kaggle/working", "starplus_ablation_results.json") \
        if os.path.isdir("/kaggle/working") else "starplus_ablation_results.json"


def main():
    print("Library versions:", json.dumps(versions()))
    print(f"Config: k={K_VALUES} C={C_VALUES} outer=GroupKFold(10,trial) inner=GroupKFold(6)")
    sys.stdout.flush()

    results = {"versions": versions(),
               "config": {"k_values": [str(k) for k in K_VALUES], "c_values": C_VALUES,
                          "outer_cv": "GroupKFold(10, trial)", "inner_cv": "GroupKFold(6)",
                          "features": "LSA beta-series (SPM HRF, cosine drift)"},
               "subjects": {}}

    for sid, url in STARPLUS_URLS.items():
        print(f"\n{'=' * 70}\nSubject {sid}\n{'=' * 70}")
        dest = os.path.join(STARPLUS_DIR, f"data-starplus-{sid}-v7.mat")
        download(url, dest)

        t0 = time.time()
        X, y, groups = fit_beta_series(dest)
        print(f"[{sid}] beta-series: {time.time() - t0:.1f}s, X={X.shape}")
        sys.stdout.flush()

        arms = {}
        for name, with_sel in (("selection", True), ("no_selection", False)):
            t0 = time.time()
            scores, choices = nested_cv(X, y, groups, with_sel)
            k_hist = Counter(str(c["k"]) for c in choices) if with_sel else None
            arms[name] = {"mean": float(scores.mean()), "sd": float(scores.std(ddof=1)),
                          "per_fold_scores": scores.tolist(), "per_fold_choices": choices}
            if with_sel:
                arms[name]["k_histogram"] = dict(k_hist)
            print(f"[{sid}] {name:<12s} mean={scores.mean():.3f} sd={scores.std(ddof=1):.3f} "
                  f"({time.time() - t0:.0f}s)" + (f"  k choices: {dict(k_hist)}" if with_sel else ""))
            sys.stdout.flush()

        delta = arms["selection"]["mean"] - arms["no_selection"]["mean"]
        print(f"[{sid}] delta(selection - no_selection) = {delta:+.3f}")

        rng = np.random.RandomState(RNG_SEED)
        null_means = []
        t0 = time.time()
        for p in range(N_PERMUTATIONS):
            y_perm = rng.permutation(y)
            perm_scores, _ = nested_cv(X, y_perm, groups, with_selection=False)
            null_means.append(float(perm_scores.mean()))
            if (p + 1) % 10 == 0:
                el = time.time() - t0
                print(f"[{sid}]   no-selection permutation {p + 1}/{N_PERMUTATIONS}: "
                      f"null_mean so far={np.mean(null_means):.3f}, elapsed={el:.0f}s")
                sys.stdout.flush()
        null_means = np.array(null_means)
        p_value = float((np.sum(null_means >= arms["no_selection"]["mean"]) + 1) / (N_PERMUTATIONS + 1))
        print(f"[{sid}] no-selection permutation: null mean={null_means.mean():.3f} "
              f"max={null_means.max():.3f} p={p_value:.4f}")
        arms["no_selection"]["permutation"] = {"n": N_PERMUTATIONS, "null_means": null_means.tolist(),
                                               "null_mean": float(null_means.mean()),
                                               "null_max": float(null_means.max()), "p_value": p_value}

        results["subjects"][sid] = {"n_voxels": int(X.shape[1]), **arms}
        with open(out_path(), "w") as f:
            json.dump(results, f, indent=2)
        sys.stdout.flush()

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    for sid, r in results["subjects"].items():
        print(f"  {sid}: selection={r['selection']['mean']:.3f}±{r['selection']['sd']:.3f}  "
              f"no_selection={r['no_selection']['mean']:.3f}±{r['no_selection']['sd']:.3f}  "
              f"k={r['selection'].get('k_histogram')}")
    sel = np.mean([r["selection"]["mean"] for r in results["subjects"].values()])
    nos = np.mean([r["no_selection"]["mean"] for r in results["subjects"].values()])
    print(f"  MEAN: selection={sel:.3f}  no_selection={nos:.3f}  delta={sel - nos:+.3f}")


if __name__ == "__main__":
    main()
