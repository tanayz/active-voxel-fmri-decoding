# Active-Voxel/ROI Selection for Small-Sample fMRI Decoding

Code accompanying the paper *"Active-Voxel Selection for Small-Sample fMRI Decoding: Dataset-Dependent
Preprocessing and the Cost of Cross-Validation Leakage"* (Chowdhury; preprint on bioRxiv, submitted to IEEE JBHI).

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
| `haxby_vt_final.py` | **Final Haxby analysis (the all-subjects table, the "ours" rows of the subject-2 table, columns D–E of the ladder table, and the last column of the preprocessing table).** For every subject, from one procedure: (i) nested-CV accuracy of VT-restricted active-voxel selection with the per-fold chosen `k` recorded; (ii) the whole-VT no-selection baseline under the identical nested protocol; (iii) a permutation test that repeats the *full* nested procedure under block-wise within-run label permutation. Prints library versions. Edit `SUBJECTS` to split across kernels (the per-subject JSONs merge trivially). |
| `haxby_ladder.py` | Decomposition ladder (Table II rows 3-5 and the all-subject ladder table): the nilearn example's pipeline restricted to the VT mask, then + feature standardisation, then + one-vs-rest logistic regression at C=1, all leave-one-run-out. Shows how the gap between the run-grouped tutorial and our no-selection pipeline splits between anatomical prior, scaling, classifier and tuning. |
| `haxby_nilearn_baseline.py` | Pinned-version reproduction of nilearn's `plot_haxby_multiclass.py` example (Table II, both nilearn rows): the identical SelectKBest+SVC pipeline scored with the example's ungrouped `cv=5` and with leave-one-run-out, so the fold assignment is the only difference. |
| `haxby_benchmark.py` | Original (unpinned) run of the nilearn baseline plus the pre-fix leaky voxel-ranking pipeline used to quantify the leakage bug's impact (leakage table) against the fixed version below. The Table II nilearn numbers now come from `haxby_nilearn_baseline.py`. |
| `haxby_raw_nested_cv.py` | Fixed, leak-free raw-frame active-voxel selection, whole brain, all 6 subjects (preprocessing table, "Raw, whole brain" column; also the fixed counterpart used to quantify the raw-frame leakage delta). |
| `haxby_raw_vt_nested_cv.py` | Earlier variant of the final analysis with an 11-fold leave-one-run-out inner loop and a larger grid (71.7% mean; cited in the text for continuity with the journal submission). Superseded by `haxby_vt_final.py`. |
| `haxby_vt_nested_cv.py` | GLM beta-series + VT-restricted active-voxel selection (preprocessing table, "Beta, VT" column; 66.8% mean). |
| `haxby_raw_permutation_test.py` | Earlier fixed-pipeline permutation test of the *whole-brain* raw-frame track. Kept for the record; its p-values are no longer reported because it tested a different pipeline than the one in Table III and chose its fixed hyperparameters on the full dataset. Superseded by the full-procedure test in `haxby_vt_final.py`. |
| `haxby_hyperalignment_sanity.py` | Cross-subject hyperalignment negative result (52.8% solo vs. 25.1% pooled for subject 4). |

### Paper numbers (`paper/`)

| Script | Produces |
|---|---|
| `merge_haxby_final.py` | Merges the per-subject `haxby_vt_final_s*.json` files from split runs into one results file, refusing to mix runs with different library versions or configs. |
| `make_numbers.py` | Generates the LaTeX macro file (`numbers_biorxiv.tex`) that feeds every results-derived number in the manuscript from the JSON outputs above, so tables and prose cannot drift from the result files. |
| `make_figures.py` | Draws Figure 1 (per-subject accuracy, whole-VT baseline vs. active-voxel selection, with the permutation null) from the same JSON files. |

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
- Permutation tests repeat the *entire* nested procedure (inner grid search included) under each label
  permutation. For Haxby, labels are permuted at the block level within each run so that block structure
  and run membership are preserved (`block_permute_labels` in `haxby_vt_final.py`).
- Library versions used for the reported numbers are printed at the top of each script's log and stored
  in its results JSON.

## License

MIT (see `LICENSE`).
