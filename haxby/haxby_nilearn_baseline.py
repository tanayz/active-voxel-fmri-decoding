"""
Pinned-version reproduction of nilearn's public Haxby multi-class example
(examples/02_decoding/plot_haxby_multiclass.py): SelectKBest(f_classif, k=500) -> linear SVC,
one-vs-rest and one-vs-one, on the default subject (subject 2), rest condition removed.

Two evaluations of the identical pipeline:
  (a) as published: cross_val_score(..., cv=5)  -> StratifiedKFold, NOT grouped by run, so
      temporally adjacent frames of the same block land in both train and test folds;
  (b) run-grouped: LeaveOneGroupOut over the 12 scanner runs (the leakage-free protocol).

The feature selection is inside the Pipeline in both cases (as in the example), so the ONLY
difference between (a) and (b) is the fold assignment. Library versions are printed so the
paper can cite the exact nilearn/scikit-learn versions behind these numbers.
"""
import json
import os
import platform

import numpy as np
import pandas as pd
import sklearn
import nilearn
from nilearn import datasets
from nilearn.maskers import NiftiMasker
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

SUBJECT = 2


def main():
    versions = {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                "nilearn": nilearn.__version__, "numpy": np.__version__}
    print("Library versions:", json.dumps(versions))

    haxby = datasets.fetch_haxby(subjects=[SUBJECT])
    func = haxby.func[0]
    labels = pd.read_csv(haxby.session_target[0], sep=" ")
    y_all = labels["labels"].values
    runs_all = labels["chunks"].values

    masker = NiftiMasker(mask_img=haxby.mask, runs=runs_all, smoothing_fwhm=4,
                         memory="nilearn_cache", memory_level=1, verbose=0)
    X_all = masker.fit_transform(func)
    non_rest = y_all != "rest"
    X, y, runs = X_all[non_rest], y_all[non_rest], runs_all[non_rest]
    print(f"X={X.shape}, classes={sorted(set(y))}, runs={len(set(runs))}")

    pipe = Pipeline([("anova", SelectKBest(f_classif, k=500)), ("svc", SVC(kernel="linear"))])
    results = {"versions": versions, "subject": SUBJECT}
    for name, wrapper in (("OvR", OneVsRestClassifier), ("OvO", OneVsOneClassifier)):
        clf = wrapper(pipe)
        published = cross_val_score(clf, X, y, cv=5, n_jobs=-1)
        grouped = cross_val_score(clf, X, y, cv=LeaveOneGroupOut(), groups=runs, n_jobs=-1)
        results[name] = {"cv5_ungrouped_mean": float(published.mean()), "cv5_ungrouped_folds": published.tolist(),
                         "logo_grouped_mean": float(grouped.mean()), "logo_grouped_folds": grouped.tolist()}
        print(f"{name}: as published (cv=5, ungrouped) = {published.mean():.3f}   "
              f"run-grouped (leave-one-run-out) = {grouped.mean():.3f}   "
              f"delta = {published.mean() - grouped.mean():+.3f}")

    out = os.path.join("/kaggle/working", "haxby_nilearn_baseline_results.json") if os.path.isdir("/kaggle/working") \
        else "haxby_nilearn_baseline_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("saved ->", out)


if __name__ == "__main__":
    main()
