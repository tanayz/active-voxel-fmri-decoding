"""
Where does data-driven voxel selection put its voxels? (anatomical analysis for the whole-brain arm)

For each Haxby subject and each leave-one-run-out outer fold, voxels of the WHOLE-BRAIN mask
(39,912 candidates) are ranked by the summed one-vs-rest F statistic on the TRAINING runs only -- the
selection step exactly as in the nested pipeline, so nothing here uses held-out data. For the top k
(k = 500, 1000) we record:
  * fraction inside the ventral temporal (VT) mask, its chance level (|VT| / |brain|), the enrichment
    ratio and a hypergeometric p-value;
  * fraction inside the dataset's face- and house-selective masks;
  * for selected voxels outside VT: distance (mm) to the nearest VT voxel, and position along the
    posterior-anterior and inferior-superior axes (normalized to the brain mask's extent);
  * selection frequency of every voxel across folds (stability map, saved as NIfTI with the anatomy).
A second, nested decoding arm restricts the candidate pool to the whole brain MINUS VT (selection,
k in {100, 500, 1000, 2000, 5000}, C in {0.1, 1, 10}; outer LORO, inner GroupKFold(4)) to test whether
category information is confined to VT. Set SUBJECTS to split across kernels.
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
from nilearn.image import mean_img
from nilearn.maskers import NiftiMasker
from scipy.spatial import cKDTree
from scipy.stats import hypergeom
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SUBJECTS = [1, 2, 3, 4, 5, 6]
K_REPORT = [500, 1000]
K_VALUES = [100, 500, 1000, 2000, 5000]
C_VALUES = [0.1, 1.0, 10.0]
INNER_SPLITS = 4
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
OUT_NAME = "haxby_selection_anatomy.json"


def versions():
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "nilearn": nilearn.__version__, "nibabel": nib.__version__}


def fetch_with_retry(subject, attempts=5, wait=120):
    for i in range(attempts):
        try:
            return datasets.fetch_haxby(subjects=[subject])
        except Exception as exc:
            print(f"fetch_haxby attempt {i + 1}/{attempts} failed: {exc!r}; retrying in {wait}s", flush=True)
            time.sleep(wait)
    return datasets.fetch_haxby(subjects=[subject])


def mask_columns(whole_flat_pos, mask_path, shape):
    m = nib.load(mask_path).get_fdata() > 0
    assert m.shape == shape, (m.shape, shape)
    return np.where(m.ravel()[whole_flat_pos])[0]


def summed_f(X, y):
    s = np.zeros(X.shape[1])
    for c in np.unique(y):
        f, _ = f_classif(X, (y == c).astype(int))
        s += np.nan_to_num(f, nan=0.0)
    return s


class TopK(BaseEstimator, TransformerMixin):
    def __init__(self, k=500):
        self.k = k

    def fit(self, X, y):
        self.idx_ = np.argsort(-summed_f(X, y))[:min(int(self.k), X.shape[1])]
        return self

    def transform(self, X):
        return X[:, self.idx_]


def nested_outside_vt(X, y, runs):
    pipe = Pipeline([("select", TopK()), ("scale", StandardScaler()),
                     ("clf", OneVsRestClassifier(LogisticRegression(max_iter=5000)))])
    grid = {"select__k": K_VALUES, "clf__estimator__C": C_VALUES}
    scores, ks = [], []
    for tr, te in LeaveOneGroupOut().split(X, y, runs):
        g = GridSearchCV(pipe, grid, cv=GroupKFold(n_splits=INNER_SPLITS), scoring="accuracy", n_jobs=-1)
        g.fit(X[tr], y[tr], groups=runs[tr])
        scores.append(float(g.score(X[te], y[te])))
        ks.append(int(g.best_params_["select__k"]))
    return scores, ks


def main():
    print("Library versions:", json.dumps(versions()), flush=True)
    out_path = os.path.join(OUT_DIR, OUT_NAME)
    results = {"versions": versions(), "k_report": K_REPORT, "subjects": {}}
    for subject in SUBJECTS:
        t0 = time.time()
        h = fetch_with_retry(subject)
        labels = pd.read_csv(h.session_target[0], sep=" ")
        y_all, runs_all = labels["labels"].values, labels["chunks"].values
        whole_img = nib.load(h.mask)
        whole = whole_img.get_fdata() > 0
        shape = whole.shape
        masker = NiftiMasker(mask_img=h.mask, runs=runs_all, smoothing_fwhm=4,
                             memory="nilearn_cache", memory_level=1, verbose=0)
        X_all = masker.fit_transform(h.func[0]).astype(np.float32)
        keep = y_all != "rest"
        X, y, runs = X_all[keep], y_all[keep], runs_all[keep]
        flat_pos = np.where(whole.ravel())[0]
        p = len(flat_pos)

        mvt = h.mask_vt[0] if isinstance(h.mask_vt, list) else h.mask_vt
        vt = mask_columns(flat_pos, mvt, shape)
        extra = {}
        for name in ("mask_face", "mask_house"):
            path = getattr(h, name, None)
            if path is not None:
                path = path[0] if isinstance(path, list) else path
                extra[name.replace("mask_", "")] = mask_columns(flat_pos, path, shape)
        vt_set = set(vt.tolist())
        ijk = np.column_stack(np.unravel_index(flat_pos, shape))
        xyz = nib.affines.apply_affine(whole_img.affine, ijk)
        tree = cKDTree(xyz[vt])
        y_rng = (xyz[:, 1].min(), xyz[:, 1].max())
        z_rng = (xyz[:, 2].min(), xyz[:, 2].max())
        print(f"[{subject}] loaded {X.shape}, VT={len(vt)}, extra={ {k: len(v) for k, v in extra.items()} } "
              f"({time.time() - t0:.0f}s)", flush=True)

        rec = {"n_voxels": p, "n_vt": len(vt), "vt_chance": len(vt) / p,
               "extra_sizes": {k: len(v) for k, v in extra.items()}, "folds": []}
        freq = {k: np.zeros(p) for k in K_REPORT}
        n_folds = 0
        for o, (tr, te) in enumerate(LeaveOneGroupOut().split(X, y, runs)):
            order = np.argsort(-summed_f(X[tr], y[tr]))
            n_folds += 1
            fold = {"fold": o}
            for k in K_REPORT:
                sel = order[:k]
                freq[k][sel] += 1
                in_vt = np.isin(sel, vt)
                n_in = int(in_vt.sum())
                outside = sel[~in_vt]
                d = tree.query(xyz[outside])[0] if len(outside) else np.array([])
                fold[str(k)] = {
                    "frac_in_vt": n_in / k,
                    "enrichment": (n_in / k) / (len(vt) / p),
                    "hypergeom_p": float(hypergeom.sf(n_in - 1, p, len(vt), k)),
                    "frac_vt_covered": n_in / len(vt),
                    **{f"frac_in_{nm}": float(np.isin(sel, idx).mean()) for nm, idx in extra.items()},
                    "outside_dist_mm_median": float(np.median(d)) if len(d) else None,
                    "outside_frac_within_5mm": float((d <= 5).mean()) if len(d) else None,
                    "outside_frac_within_10mm": float((d <= 10).mean()) if len(d) else None,
                    "outside_y_norm_median": float(np.median((xyz[outside, 1] - y_rng[0]) / (y_rng[1] - y_rng[0]))) if len(outside) else None,
                    "outside_z_norm_median": float(np.median((xyz[outside, 2] - z_rng[0]) / (z_rng[1] - z_rng[0]))) if len(outside) else None,
                    "outside_frac_posterior_third": float(((xyz[outside, 1] - y_rng[0]) / (y_rng[1] - y_rng[0]) < 1 / 3).mean()) if len(outside) else None,
                }
            rec["folds"].append(fold)
        for k in K_REPORT:
            f = freq[k] / n_folds
            rec[f"stable_{k}"] = {"n_selected_every_fold": int((f == 1).sum()),
                                  "n_selected_any_fold": int((f > 0).sum()),
                                  "frac_of_always_selected_in_vt": float(np.isin(np.where(f == 1)[0], vt).mean()) if (f == 1).any() else None}
            nib.save(masker.inverse_transform(f.astype(np.float32)),
                     os.path.join(OUT_DIR, f"selfreq_s{subject}_k{k}.nii.gz"))
        anat = h.anat[0] if isinstance(h.anat, list) else h.anat
        if anat is not None:  # subject 6 ships without an anatomical image
            nib.save(nib.load(anat), os.path.join(OUT_DIR, f"anat_s{subject}.nii.gz"))
        nib.save(mean_img(h.func[0]), os.path.join(OUT_DIR, f"meanepi_s{subject}.nii.gz"))
        nib.save(nib.load(mvt), os.path.join(OUT_DIR, f"vt_s{subject}.nii.gz"))
        s5 = rec["folds"]
        print(f"[{subject}] k=500: in-VT {np.mean([f['500']['frac_in_vt'] for f in s5]):.3f} "
              f"(chance {len(vt) / p:.4f}); outside within 5mm {np.mean([f['500']['outside_frac_within_5mm'] for f in s5]):.2f}",
              flush=True)

        non_vt = np.setdiff1d(np.arange(p), vt)
        t0 = time.time()
        scores, ks = nested_outside_vt(X[:, non_vt], y, runs)
        rec["outside_vt_decoding"] = {"n_candidates": int(len(non_vt)), "per_fold_scores": scores,
                                      "mean": float(np.mean(scores)), "sd": float(np.std(scores, ddof=1)),
                                      "chosen_k": ks}
        print(f"[{subject}] decoding WITHOUT VT: {np.mean(scores):.3f} +/- {np.std(scores, ddof=1):.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        results["subjects"][str(subject)] = rec
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
