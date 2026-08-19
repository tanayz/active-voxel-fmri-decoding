"""
Stage 2 (CPU-only): faithfully reproduce Chowdhury (2014) Tables 1/2/4 on the
StarPlus picture-vs-sentence task, using the exact conventions found in the
original CMU fmri_core toolbox (README + transformIDM_*.m), not guesses:

  - Example construction (README tutorial section 1):
      trials = find(info.cond > 1)                      # drop calibration/fixation
      split by firstStimulus ('P' vs 'S')
      P1st trials: TR[0:16] = picture, TR[16:32] = sentence
      S1st trials: TR[0:16] = sentence, TR[16:32] = picture
    -> 40 trials x 2 windows = 80 examples, 40/class, one picture + one
       sentence example per raw trial (this pairing is the CV grouping unit
       for leakage avoidance, per thesis section 3.3).

  - Voxel activity ranking (mri_computeTvalues.m + transformIDM_selectActiveVoxact.m):
      one-tailed unpaired t-test per voxel, pooling every raw timepoint,
      cond==1 (fixation) vs cond==c (c in {2,3}) samples, alternative='greater'.
      Rank ascending p-value (smaller p = more active), select top-n per
      condition with an exact-total round-robin merge (selectActiveVoxact).

  - ROI averaging (transformIDM_avgROIVoxels.m): mean voxel signal within each
    named ROI at each timepoint -> one supervoxel per ROI.

Classifier notes: SMO -> linear-kernel SVC (Weka's SMO is a linear/poly SVM
solver); HoeffdingTree/SimpleCart have no direct scikit-learn equivalent
(they're Weka/MOA streaming & CART-style trees) -- approximated as an
unrestricted DecisionTreeClassifier ("H", strong learner) and a depth-2
DecisionTreeClassifier ("S", weak learner), matching the thesis's own
characterization of H as strong / S as weak for the Adaboost comparison.
This substitution is a stated limitation, not a silent one.
"""
import json
import os
import time
import urllib.request

import numpy as np
from scipy.io import loadmat
from scipy.stats import ttest_ind
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DATA_DIR = "/kaggle/working/data" if os.path.isdir("/kaggle/working") else "./data"
STARPLUS_DIR = os.path.join(DATA_DIR, "starplus")
os.makedirs(STARPLUS_DIR, exist_ok=True)

STARPLUS_URLS = {
    "04799": "http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-81/www/data-starplus-04799-v7.mat",
    "04820": "http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-83/www/data-starplus-04820-v7.mat",
    "04847": "http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-81/www/data-starplus-04847-v7.mat",
}

RNG_SEED = 42
N_FOLDS = 10
VOXEL_SWEEP = list(range(5, 505, 5))  # thesis: n=5, step+5, stop after 3 flat/declining steps


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        return
    urllib.request.urlretrieve(url, dest)


def load_subject(path):
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    return np.atleast_1d(mat["info"]), np.atleast_1d(mat["data"]), mat["meta"]


def normalize_trials(data):
    """transformIDM_normalizeTrials.m: per trial, per voxel, subtract that
    voxel's own mean activity over the trial (removes slow drift per the
    toolbox's own stated rationale, applied before any other step)."""
    return np.array([np.asarray(d, dtype=np.float64) - np.asarray(d, dtype=np.float64).mean(axis=0, keepdims=True)
                      for d in data], dtype=object)


def normalize_images(data):
    """transformIDM_normalizeImages.m: per trial, per timepoint ('image'),
    z-score across voxels so each image has mean 0 / std 1 over the voxel
    dimension -- explicitly documented as helping eliminate slow drift/noise
    for Naive Bayes and similar classifiers. Independent of normalize_trials
    (that one normalizes across time per voxel; this normalizes across
    voxels per timepoint)."""
    out = []
    for d in data:
        d = np.asarray(d, dtype=np.float64)
        mu = d.mean(axis=1, keepdims=True)
        sd = d.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        out.append((d - mu) / sd)
    return np.array(out, dtype=object)


PREPROCESSORS = {
    "raw": lambda data: data,
    "normTrials": normalize_trials,
    "normImages": normalize_images,
    "normTrials_normImages": lambda data: normalize_images(normalize_trials(data)),
}


def build_examples(info, data):
    """Returns X (n_examples, 16, n_voxels), y (0=sentence,1=picture), groups (raw trial index)."""
    X, y, groups = [], [], []
    for i, entry in enumerate(info):
        cond = int(entry.cond)
        if cond <= 1:
            continue
        d = np.asarray(data[i], dtype=np.float64)
        if d.shape[0] < 32:
            continue  # guard against any short/malformed trial
        first_half, second_half = d[0:16, :], d[16:32, :]
        fs = str(entry.firstStimulus)
        if fs == "P":
            X += [first_half, second_half]
            y += [1, 0]
        elif fs == "S":
            X += [first_half, second_half]
            y += [0, 1]
        else:
            continue
        groups += [i, i]
    return np.array(X), np.array(y), np.array(groups)


def voxel_activity_pvalues(info, data):
    """Per-voxel one-tailed p-value for cond==1 (fixation) vs cond==c, c in {2,3}."""
    nvoxels = np.asarray(data[0]).shape[1]
    bins = {1: [], 2: [], 3: []}
    for i, entry in enumerate(info):
        cond = int(entry.cond)
        if cond in bins:
            bins[cond].append(np.asarray(data[i], dtype=np.float64))
    pooled = {c: np.concatenate(v, axis=0) for c, v in bins.items() if v}

    pvals = {}
    fixation = pooled[1]
    for c in (2, 3):
        if c not in pooled:
            continue
        _, p = ttest_ind(pooled[c], fixation, axis=0, alternative="greater")
        pvals[c] = p  # shape (nvoxels,); smaller = more active
    return pvals, nvoxels


def select_active_voxact(pvals, nvoxels, n_to_keep):
    """Exact-total round-robin merge across conditions {2,3} (selectActiveVoxact.m)."""
    conditions = list(pvals.keys())
    ranked = {c: np.argsort(pvals[c]) for c in conditions}  # ascending p-value
    idx_ptr = {c: 0 for c in conditions}
    active = np.zeros(nvoxels, dtype=bool)
    n_left = n_to_keep
    n_per_cond = max(1, n_left // len(conditions))
    while n_left > 0:
        for c in conditions:
            take = ranked[c][idx_ptr[c]: idx_ptr[c] + n_per_cond]
            active[take] = True
            idx_ptr[c] += n_per_cond
            n_left = n_to_keep - active.sum()
            if n_left <= 0:
                break
        if n_left <= 0:
            break
        n_per_cond = max(1, n_left // len(conditions))
    return np.where(active)[0]


def roi_names_and_columns(meta):
    rois = np.atleast_1d(meta.rois)
    return {r.name: np.atleast_1d(r.columns).astype(int) - 1 for r in rois}  # MATLAB 1-indexed -> 0-indexed


def flatten(X_subset):
    return X_subset.reshape(X_subset.shape[0], -1)


def grouped_kfold_indices(groups, n_folds, seed):
    unique_groups = np.unique(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_groups)
    fold_groups = np.array_split(unique_groups, n_folds)
    folds = []
    for fg in fold_groups:
        test_mask = np.isin(groups, fg)
        folds.append((np.where(~test_mask)[0], np.where(test_mask)[0]))
    return folds


def classifiers():
    tree_h = DecisionTreeClassifier(random_state=RNG_SEED)  # "strong" stand-in for HoeffdingTree
    tree_s = DecisionTreeClassifier(max_depth=2, random_state=RNG_SEED)  # "weak" stand-in for SimpleCart
    return {
        "NaiveBayes": lambda: GaussianNB(),
        "Logistic": lambda: LogisticRegression(max_iter=2000),
        "SMO_linearSVC": lambda: SVC(kernel="linear"),
        "SVM_sigmoid": lambda: SVC(kernel="sigmoid"),
        "Tree_H": lambda: DecisionTreeClassifier(random_state=RNG_SEED),
        "Tree_H_Adaboost": lambda: AdaBoostClassifier(estimator=DecisionTreeClassifier(random_state=RNG_SEED),
                                                        n_estimators=50, random_state=RNG_SEED),
        "Tree_S": lambda: DecisionTreeClassifier(max_depth=2, random_state=RNG_SEED),
        "Tree_S_Adaboost": lambda: AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2, random_state=RNG_SEED),
                                                        n_estimators=50, random_state=RNG_SEED),
    }


def cv_accuracy(X2d, y, groups, clf_fn, n_folds=N_FOLDS, seed=RNG_SEED):
    folds = grouped_kfold_indices(groups, n_folds, seed)
    accs = []
    for train_idx, test_idx in folds:
        clf = clf_fn()
        clf.fit(X2d[train_idx], y[train_idx])
        accs.append(clf.score(X2d[test_idx], y[test_idx]))
    return float(np.mean(accs))


def run_voxel_sweep(X, y, groups, pvals, nvoxels):
    """Algorithm 3.5.1: step n by 5, stop after 3 consecutive non-improving steps."""
    clf_bank = classifiers()
    curve = []  # list of dicts: {n, per-classifier accuracy}
    best_overall = -1
    flat_count = 0
    for n in VOXEL_SWEEP:
        if n >= nvoxels:
            break
        active_idx = select_active_voxact(pvals, nvoxels, n)
        Xs = flatten(X[:, :, active_idx])
        row = {"n": n}
        for name, fn in clf_bank.items():
            row[name] = cv_accuracy(Xs, y, groups, fn)
        curve.append(row)

        best_this_n = max(v for k, v in row.items() if k != "n")
        if best_this_n <= best_overall:
            flat_count += 1
        else:
            flat_count = 0
            best_overall = best_this_n
        if flat_count >= 3:
            break
    return curve


def run_roi_method(info, data, meta, X, y, groups):
    """Algorithm 3.5.2: rank ROIs by standalone accuracy, then greedily add."""
    roi_cols = roi_names_and_columns(meta)
    clf_bank = classifiers()

    def roi_features(names):
        idx_groups = [roi_cols[n] for n in names if n in roi_cols and len(roi_cols[n]) > 0]
        avgs = [X[:, :, idx].mean(axis=2) for idx in idx_groups]  # (n_examples, 16) per ROI
        return np.concatenate(avgs, axis=1)  # (n_examples, 16*len(names))

    roi_scores = {}
    for name in roi_cols:
        if len(roi_cols[name]) == 0:
            continue
        Xs = roi_features([name])
        best = max(cv_accuracy(Xs, y, groups, fn) for fn in clf_bank.values())
        roi_scores[name] = best

    ranked_rois = sorted(roi_scores, key=roi_scores.get, reverse=True)

    curve = []
    best_overall = -1
    flat_count = 0
    for k in range(5, len(ranked_rois) + 1):
        names = ranked_rois[:k]
        Xs = roi_features(names)
        row = {"n_rois": k, "rois": names}
        for name, fn in clf_bank.items():
            row[name] = cv_accuracy(Xs, y, groups, fn)
        curve.append(row)
        best_this_k = max(v for kk, v in row.items() if kk not in ("n_rois", "rois"))
        if best_this_k <= best_overall:
            flat_count += 1
        else:
            flat_count = 0
            best_overall = best_this_k
        if flat_count >= 3:
            break
    return roi_scores, curve


# max value across the classifier row in each of the thesis's own tables
# (Tables 1/2/4), used only as a same-scale reference point in the summary.
THESIS_BEST = {
    "04799": {"table1": 0.90, "table2": 0.92, "table4": 0.92},
    "04820": {"table1": 0.96, "table2": 0.96, "table4": 0.96},
    "04847": {"table1": 0.92, "table2": 0.96, "table4": 1.00},
}


def analyze_subject(sid, url, preprocess_fn):
    dest = os.path.join(STARPLUS_DIR, f"data-starplus-{sid}-v7.mat")
    download(url, dest)
    info, data, meta = load_subject(dest)
    data = preprocess_fn(data)

    X, y, groups = build_examples(info, data)
    pvals, nvoxels = voxel_activity_pvalues(info, data)
    voxel_curve = run_voxel_sweep(X, y, groups, pvals, nvoxels)
    roi_scores, roi_curve = run_roi_method(info, data, meta, X, y, groups)

    best_voxel_row = max(voxel_curve, key=lambda r: max(v for k, v in r.items() if k != "n"))
    best_roi_row = max(roi_curve, key=lambda r: max(v for k, v in r.items() if k not in ("n_rois", "rois")))

    return {
        "n_examples": int(X.shape[0]),
        "nvoxels": int(X.shape[2]),
        "voxel_sweep_curve": voxel_curve,
        "roi_individual_scores": roi_scores,
        "roi_greedy_curve": roi_curve,
        "table1_equivalent": voxel_curve[0],
        "table2_equivalent": best_voxel_row,
        "table4_equivalent": best_roi_row,
    }


def best_of(row, exclude):
    return max(v for k, v in row.items() if k not in exclude)


def main():
    results = {}
    for variant_name, preprocess_fn in PREPROCESSORS.items():
        print(f"\n{'#'*70}\nPREPROCESSING VARIANT: {variant_name}\n{'#'*70}")
        results[variant_name] = {}
        for sid, url in STARPLUS_URLS.items():
            print(f"[{variant_name}/{sid}] running...")
            t0 = time.time()
            r = analyze_subject(sid, url, preprocess_fn)
            results[variant_name][sid] = r
            print(f"[{variant_name}/{sid}] done in {time.time()-t0:.1f}s | "
                  f"n_examples={r['n_examples']} nvoxels={r['nvoxels']}")

        print(f"\n{'-'*70}\nSummary for variant '{variant_name}' "
              f"(best classifier accuracy vs thesis's same-table max)\n{'-'*70}")
        print(f"{'subject':<10}{'table1(ours/thesis)':<24}{'table2(ours/thesis)':<24}{'table4(ours/thesis)':<24}")
        for sid in STARPLUS_URLS:
            r = results[variant_name][sid]
            t1 = best_of(r["table1_equivalent"], {"n"})
            t2 = best_of(r["table2_equivalent"], {"n"})
            t4 = best_of(r["table4_equivalent"], {"n_rois", "rois"})
            ref = THESIS_BEST[sid]
            print(f"{sid:<10}"
                  f"{t1:.3f}/{ref['table1']:.2f}{'':<12}"
                  f"{t2:.3f}/{ref['table2']:.2f}{'':<12}"
                  f"{t4:.3f}/{ref['table4']:.2f}")

    out_path = "/kaggle/working/results.json" if os.path.isdir("/kaggle/working") else "./results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
