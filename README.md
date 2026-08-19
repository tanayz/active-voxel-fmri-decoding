# Active-Voxel/ROI Selection for Small-Sample fMRI Decoding

Code accompanying the paper *"Active-Voxel Selection for Small-Sample fMRI Decoding: Dataset-Dependent
Preprocessing and the Cost of Cross-Validation Leakage"* (Chowdhury, submitted to IEEE JBHI).

This repository contains the scripts needed to reproduce every number reported in the paper. It does not
include exploratory/debugging scripts or abandoned experimental variants that did not make it into the
final results.

## Datasets

Both datasets are fetched automatically by the scripts on first run — no manual download needed.

- **StarPlus** (Mitchell et al., 2004): downloaded directly from the CMU StarPlus archive via `urllib`.
- **Haxby** (Haxby et al., 2001): downloaded via `nilearn.datasets.fetch_haxby`.

## Setup

```bash
pip install -r requirements.txt
```

All scripts are CPU-only and self-contained (no cross-file imports) — each can be run directly, e.g.:

```bash
python starplus/starplus_glm_nested_cv.py
```

Each script writes its results to a `results.json` file in the working directory (or `/kaggle/working/`
if run inside a Kaggle kernel, which is how the original results were generated on Kaggle's free CPU
tier).

## Script-to-result mapping

### StarPlus (`starplus/`)

| Script | Produces |
|---|---|
| `replicate_baselines.py` | Raw-window baseline and active-ROI selection (Table I, "Raw-window baseline" column) — a faithful reproduction of Chowdhury (2014)'s original Algorithm 3.5.2 (rank ROIs by standalone accuracy, greedily average in the best-performing ones). |
| `starplus_glm_benchmark.py` | Pre-fix GLM beta-series pipeline with the feature-selection leakage bug described in the paper (voxel ranking computed once on the full dataset before cross-validation splitting). Included alongside the fixed version below so the leakage magnitude (13.7 points on one subject) is independently reproducible, not just asserted. |
| `starplus_glm_nested_cv.py` | Headline StarPlus result (Table I, GLM beta-series + active-voxel selection column; 92.9% mean, leak-free). |
| `starplus_permutation_test.py` | Permutation-test significance values reported in Table I. |

### Haxby (`haxby/`)

| Script | Produces |
|---|---|
| `haxby_benchmark.py` | `nilearn`'s published tutorial baseline and its run-grouped, leakage-corrected variant (Table II, both baseline rows). Also contains the pre-fix leaky voxel-ranking pipeline used to quantify the leakage bug's impact (up to 44.8 points on one subject) against the fixed version below. |
| `haxby_raw_nested_cv.py` | Fixed, leak-free raw-frame active-voxel selection, whole brain, all 6 subjects (Table II, "Ours" row for subject 2; also the fixed counterpart used to quantify the leakage delta above). |
| `haxby_raw_vt_nested_cv.py` | Headline Haxby result: raw-frame + VT-restricted active-voxel selection, all 6 subjects (Table III; 71.7% mean). |
| `haxby_vt_nested_cv.py` | GLM beta-series + VT-restricted active-voxel selection (Section IV-C, "Dataset-Dependent Preprocessing"; 66.8% mean, the best GLM configuration cited there). |
| `haxby_raw_permutation_test.py` | Permutation-test significance values reported in Table III. |
| `haxby_hyperalignment_sanity.py` | Cross-subject hyperalignment negative result (Section IV-D, "Unsuccessful Lever"; 52.8% solo vs. 25.1% pooled for subject 4). |

## Methodology notes

- All headline results use nested cross-validation: an outer `LeaveOneGroupOut`/`GroupKFold` loop for
  evaluation, with feature-selection and classifier hyperparameters tuned only on each outer fold's
  training data via an inner grid search. Feature selection is implemented as an
  `sklearn.base.BaseEstimator`/`TransformerMixin` living inside a `Pipeline`, so it is refit per fold
  rather than computed once on the full dataset — see `starplus_glm_nested_cv.py` or
  `haxby_raw_vt_nested_cv.py` for the canonical pattern (class `ActiveVoxelSelector`).
- The paired "pre-fix"/"fixed" scripts in each dataset's directory exist specifically to make the
  leakage-magnitude claims in the paper independently verifiable, not as a general changelog of the
  project's development history.

## License

MIT (see `LICENSE`).
