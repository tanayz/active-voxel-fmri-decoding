"""
Final Haxby raw-frame + VT-restricted analysis, all from ONE procedure so that the
reported accuracy, the no-selection baseline, and the permutation p-value are
mutually consistent (they were not in the earlier version of the paper: Table III
paired raw+VT accuracies with p-values from a permutation test of the whole-brain
pipeline whose fixed hyperparameters had been chosen on the full dataset).

For every subject this script computes:

  (1) Nested-CV accuracy of the active-voxel pipeline
      (ActiveVoxelSelector(k) -> StandardScaler -> OvR LogisticRegression),
      outer = LeaveOneGroupOut over scanner runs, inner = GroupKFold(INNER_SPLITS)
      grid search over (k, C, classifier) on the outer-training runs only.
      The per-fold chosen (k, C, classifier) is recorded so the paper can report how
      often the tuner preferred selection over using all VT voxels.

  (2) The standard "all VT voxels, no selection" baseline under the IDENTICAL nested
      protocol (grid over C and classifier only). This is the canonical Haxby-style
      ROI analysis and the fair reference point for whether active-voxel selection
      adds anything on top of the anatomical restriction.

  (3) A permutation test in which the FULL nested procedure of (1) -- inner grid
      search included -- is re-run under each of N_PERMUTATIONS label permutations.
      Labels are permuted at the BLOCK level within each run (the category labels of
      the ~8 stimulus blocks inside a run are shuffled among those blocks), which
      preserves the temporal block structure and run membership of the data and is
      the exchangeability-respecting null for a block design evaluated with
      leave-one-run-out CV. p = (#null >= real + 1) / (N + 1).

Library versions are printed so the paper can pin them. Results are saved after every
subject so a killed kernel still leaves partial output. To split across Kaggle kernels,
edit SUBJECTS (one subject per kernel) -- the JSON files merge trivially.
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

# ----------------------------------------------------------------------------- config
SUBJECTS = [1, 2, 3, 4, 5, 6]
N_PERMUTATIONS = 50          # p-value floor = 1/(N+1)
INNER_SPLITS = 4             # GroupKFold splits for the inner grid search
K_VALUES = [50, 100, 200, 300, 500, "all"]   # "all" = every VT voxel (no selection)
C_VALUES = [0.1, 1.0, 10.0]     # features are standardised, so this covers the useful range
RNG_SEED = 42
N_JOBS = -1                  # parallelism lives at exactly ONE level: the inner GridSearchCV
OUT_NAME = "haxby_vt_final_results.json"
# -------------------------------------------------------------------------------------


def versions():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "nilearn": nilearn.__version__,
        "nibabel": nib.__version__,
    }


def load_data_and_vt_index(subject):
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

    # Direct numpy projection of the VT mask into the whole-brain masker's column space
    # (NiftiMasker.transform() on the mask image produced 0 voxels in earlier attempts).
    whole_brain_data = nib.load(mask).get_fdata() > 0
    vt_data = nib.load(mask_vt).get_fdata() > 0
    whole_brain_flat_positions = np.where(whole_brain_data.ravel())[0]
    is_vt_per_column = vt_data.ravel()[whole_brain_flat_positions]
    vt_indices = np.where(is_vt_per_column)[0]
    if len(vt_indices) == 0:
        raise ValueError(f"subject {subject}: VT mask projected to 0 voxels")
    return X[:, vt_indices], y, runs


class ActiveVoxelSelector(BaseEstimator, TransformerMixin):
    """Rank voxels by the summed one-vs-rest F statistic across classes; keep the top k.
    k='all' disables selection (all candidate voxels are retained)."""

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
        k = min(int(self.k), X.shape[1])
        return X[:, self.ranking_[:k]]


def make_grid(k_values, c_values, with_selection):
    steps = []
    if with_selection:
        steps.append(("select", ActiveVoxelSelector()))
    steps += [("scale", StandardScaler()),
              ("clf", OneVsRestClassifier(LogisticRegression(max_iter=5000)))]
    pipe = Pipeline(steps)
    # One classifier family (OvR logistic regression -- the majority choice in every earlier run and
    # the linear model recommended for this regime) keeps the grid at 18 configurations so that the
    # FULL nested procedure can be repeated under every label permutation.
    g = {"clf": [OneVsRestClassifier(LogisticRegression(max_iter=5000))], "clf__estimator__C": c_values}
    if with_selection:
        g["select__k"] = k_values
    return GridSearchCV(pipe, [g], cv=GroupKFold(n_splits=INNER_SPLITS),
                        scoring="accuracy", n_jobs=N_JOBS, refit=True)


def describe_params(best_params):
    clf = best_params["clf"].estimator
    return {"k": best_params.get("select__k", "all"),
            "C": best_params["clf__estimator__C"],
            "clf": type(clf).__name__}


def nested_cv(X, y, groups, with_selection, record_choices):
    outer = LeaveOneGroupOut()
    scores, choices = [], []
    for train_idx, test_idx in outer.split(X, y, groups):
        grid = make_grid(K_VALUES, C_VALUES, with_selection)
        grid.fit(X[train_idx], y[train_idx], groups=groups[train_idx])
        scores.append(float(grid.score(X[test_idx], y[test_idx])))
        if record_choices:
            choices.append(describe_params(grid.best_params_))
    return np.array(scores), choices


def block_permute_labels(y, groups, rng):
    """Shuffle category labels among the contiguous stimulus blocks WITHIN each run.
    Block structure, block lengths, and run membership are all preserved."""
    y_perm = y.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]           # indices of this run, in acquisition order
        yg = y[idx]
        # contiguous blocks of identical label within the run
        boundaries = np.r_[0, np.where(yg[1:] != yg[:-1])[0] + 1, len(yg)]
        blocks = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
        block_labels = [yg[s] for s, _ in blocks]
        permuted = rng.permutation(block_labels)
        for (s, e), lab in zip(blocks, permuted):
            y_perm[idx[s:e]] = lab
    return y_perm


def out_path():
    return os.path.join("/kaggle/working", OUT_NAME) if os.path.isdir("/kaggle/working") else OUT_NAME


def main():
    print("Library versions:", json.dumps(versions()))
    print(f"Config: subjects={SUBJECTS} n_perm={N_PERMUTATIONS} inner_splits={INNER_SPLITS} "
          f"k={K_VALUES} C={C_VALUES}")
    sys.stdout.flush()

    results = {"versions": versions(),
               "config": {"n_permutations": N_PERMUTATIONS, "inner_splits": INNER_SPLITS,
                          "k_values": [str(k) for k in K_VALUES], "c_values": C_VALUES,
                          "outer_cv": "LeaveOneGroupOut(run)", "inner_cv": f"GroupKFold({INNER_SPLITS})",
                          "permutation_scheme": "block-wise within run", "seed": RNG_SEED},
               "subjects": {}}

    for subject in SUBJECTS:
        print(f"\n{'=' * 70}\nSubject {subject}\n{'=' * 70}")
        t0 = time.time()
        X, y, runs = load_data_and_vt_index(subject)
        print(f"[{subject}] Loaded in {time.time() - t0:.1f}s: VT voxels={X.shape[1]}, "
              f"examples={X.shape[0]}, runs={len(np.unique(runs))}")
        sys.stdout.flush()

        # (1) active-voxel pipeline, real labels
        t0 = time.time()
        real_scores, choices = nested_cv(X, y, runs, with_selection=True, record_choices=True)
        real_mean = float(real_scores.mean())
        t_nested = time.time() - t0
        k_hist = Counter(str(c["k"]) for c in choices)
        clf_hist = Counter(c["clf"] for c in choices)
        print(f"[{subject}] Active-voxel nested CV: mean={real_mean:.3f} sd={real_scores.std(ddof=1):.3f} "
              f"({t_nested:.0f}s)")
        print(f"[{subject}]   per-fold: {['%.3f' % s for s in real_scores]}")
        print(f"[{subject}]   chosen k per fold: {dict(k_hist)}; classifiers: {dict(clf_hist)}")
        sys.stdout.flush()

        # (2) all-VT no-selection baseline, identical protocol
        t0 = time.time()
        base_scores, base_choices = nested_cv(X, y, runs, with_selection=False, record_choices=True)
        base_mean = float(base_scores.mean())
        print(f"[{subject}] All-VT baseline nested CV: mean={base_mean:.3f} sd={base_scores.std(ddof=1):.3f} "
              f"({time.time() - t0:.0f}s)  delta(ours - baseline)={real_mean - base_mean:+.3f}")
        print(f"[{subject}]   per-fold: {['%.3f' % s for s in base_scores]}")
        sys.stdout.flush()

        # (3) permutation test: full nested procedure under block-wise label permutation
        rng = np.random.RandomState(RNG_SEED + subject)
        null_means = []
        t0 = time.time()
        for p in range(N_PERMUTATIONS):
            y_perm = block_permute_labels(y, runs, rng)
            perm_scores, _ = nested_cv(X, y_perm, runs, with_selection=True, record_choices=False)
            null_means.append(float(perm_scores.mean()))
            if (p + 1) % 5 == 0 or p == 0:
                el = time.time() - t0
                print(f"[{subject}]   permutation {p + 1}/{N_PERMUTATIONS}: null_mean so far="
                      f"{np.mean(null_means):.3f}, elapsed={el:.0f}s, eta={el / (p + 1) * (N_PERMUTATIONS - p - 1):.0f}s")
                sys.stdout.flush()
        null_means = np.array(null_means)
        p_value = float((np.sum(null_means >= real_mean) + 1) / (N_PERMUTATIONS + 1))
        print(f"[{subject}] Null: mean={null_means.mean():.3f} sd={null_means.std(ddof=1):.3f} "
              f"max={null_means.max():.3f}; real={real_mean:.3f}; p={p_value:.4f}")
        sys.stdout.flush()

        results["subjects"][str(subject)] = {
            "n_vt_voxels": int(X.shape[1]), "n_examples": int(X.shape[0]),
            "n_runs": int(len(np.unique(runs))),
            "active_voxel": {"mean": real_mean, "sd": float(real_scores.std(ddof=1)),
                             "per_fold_scores": real_scores.tolist(), "per_fold_choices": choices,
                             "k_histogram": dict(k_hist), "clf_histogram": dict(clf_hist),
                             "seconds": t_nested},
            "all_vt_baseline": {"mean": base_mean, "sd": float(base_scores.std(ddof=1)),
                                "per_fold_scores": base_scores.tolist(),
                                "per_fold_choices": base_choices},
            "permutation": {"n": N_PERMUTATIONS, "null_means": null_means.tolist(),
                            "null_mean": float(null_means.mean()), "null_max": float(null_means.max()),
                            "p_value": p_value},
        }
        with open(out_path(), "w") as f:
            json.dump(results, f, indent=2)
        print(f"[{subject}] saved -> {out_path()}")
        sys.stdout.flush()

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    for s, r in results["subjects"].items():
        print(f"  subject {s}: ours={r['active_voxel']['mean']:.3f}±{r['active_voxel']['sd']:.3f}  "
              f"all-VT baseline={r['all_vt_baseline']['mean']:.3f}±{r['all_vt_baseline']['sd']:.3f}  "
              f"p={r['permutation']['p_value']:.4f}  k-choices={r['active_voxel']['k_histogram']}")
    if results["subjects"]:
        ours = np.mean([r["active_voxel"]["mean"] for r in results["subjects"].values()])
        base = np.mean([r["all_vt_baseline"]["mean"] for r in results["subjects"].values()])
        print(f"  MEAN: ours={ours:.3f}  all-VT baseline={base:.3f}  delta={ours - base:+.3f}")


if __name__ == "__main__":
    main()
