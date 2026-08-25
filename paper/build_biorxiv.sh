#!/usr/bin/env bash
# Final assembly for the bioRxiv manuscript: merge per-subject Haxby results -> regenerate number macros
# -> redraw Figure 1 -> compile (pdflatex, bibtex, pdflatex x2) -> sanity checks.
#
# Usage (from paper/):  bash build_biorxiv.sh <results_dir>
# where <results_dir> contains:
#   haxby_vt_final_s{1..6}.json            (one procedure, one library-version set)
#   haxby_nilearn_baseline_results.json    (same version set)
#   haxby_vt_final_local.json              (optional: second version set, robustness check)
#   haxby_nilearn_baseline_local.json      (optional: second version set, robustness check)
set -euo pipefail
R="${1:?results dir}"
RUN="../../_run_outputs"
PY="${PYTHON:-python}"
PYPLOT="${PYPLOT:-$PY}"       # interpreter with matplotlib, if different
export PATH="/e/MiKTeX/miktex/bin/x64:$PATH"

$PY merge_haxby_final.py haxby_vt_final_results.json "$R"/haxby_vt_final_s1.json "$R"/haxby_vt_final_s2.json \
    "$R"/haxby_vt_final_s3.json "$R"/haxby_vt_final_s4.json "$R"/haxby_vt_final_s5.json "$R"/haxby_vt_final_s6.json

ARGS=(--starplus "$RUN/starplus_nested_output/starplus_nested_cv_results.json"
      --starplus-perm "$RUN/starplus_perm_output/starplus_permutation_results.json"
      --combined "$RUN/combined_v2_output/haxby_combined_levers_results.json"
      --nilearn "$R/haxby_nilearn_baseline_results.json"
      --haxby-final haxby_vt_final_results.json)
[ -f "$R/haxby_vt_final_local.json" ] && ARGS+=(--haxby-local "$R/haxby_vt_final_local.json")
[ -f "$R/haxby_nilearn_baseline_local.json" ] && ARGS+=(--nilearn-local "$R/haxby_nilearn_baseline_local.json")
[ -f "$R/haxby_ladder_results.json" ] && ARGS+=(--ladder "$R/haxby_ladder_results.json")
$PY make_numbers.py "${ARGS[@]}"

$PYPLOT make_figures.py --haxby-final haxby_vt_final_results.json \
    --starplus "$RUN/starplus_nested_output/starplus_nested_cv_results.json" \
    --starplus-perm "$RUN/starplus_perm_output/starplus_permutation_results.json" 2>&1 | grep -v -i "warn" || true

pdflatex -interaction=nonstopmode -halt-on-error main_biorxiv.tex > build_biorxiv.log 2>&1
bibtex main_biorxiv > bibtex_biorxiv.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main_biorxiv.tex >> build_biorxiv.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error main_biorxiv.tex >> build_biorxiv.log 2>&1

echo "--- checks ---"
echo "pages: $(pdfinfo main_biorxiv.pdf | grep Pages)"
echo "unresolved macros (should be 0): $(grep -c '{?}' numbers_biorxiv.tex || true)"
echo "PENDING markers (should be 0): $(grep -c PENDING numbers_biorxiv.tex || true)"
echo "undefined refs/cites: $(grep -c 'undefined' main_biorxiv.log || true)"
echo "overfull boxes: $(grep -c Overfull main_biorxiv.log || true)"
pdftotext main_biorxiv.pdf - | grep -n "?\s*\\\\%\|??\| ? " | head -5 || true
