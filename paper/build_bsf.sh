#!/usr/bin/env bash
# Brain Structure and Function manuscript: regenerate numbers and Figures 1-2 (build_biorxiv.sh), draw
# Figure 3 (anatomy), then compile main_bsf.tex and the Online Resource main_bsf_supp.tex.
#
# Usage (from paper/):  MAPS=<dir with selfreq/vt/meanepi NIfTIs> bash build_bsf.sh <results_dir>
# EXTRAS must also hold sel_anatomy_{a,b,c}.json and outvt_perm_{a,b,c}.json.
set -euo pipefail
R="${1:?results dir}"
RUN="../../_run_outputs"
export EXTRAS="${EXTRAS:-$RUN/robustness_output}"
MAPS="${MAPS:?directory with the anatomy NIfTI maps}"
PYPLOT="${PYPLOT:-${PYTHON:-python}}"
export PATH="/e/MiKTeX/miktex/bin/x64:$PATH"

bash build_biorxiv.sh "$R" > build_bsf_master.log 2>&1 || { tail -20 build_bsf_master.log; exit 1; }
grep -A8 -- "--- checks ---" build_bsf_master.log
$PYPLOT make_fig3.py "$MAPS" "$EXTRAS" haxby_vt_final_results.json 2>&1 | grep -v -i "warn" || true

for doc in main_bsf main_bsf_supp; do
    pdflatex -interaction=nonstopmode -halt-on-error $doc.tex > build_$doc.log 2>&1 || { tail -30 build_$doc.log; exit 1; }
    bibtex $doc >> build_$doc.log 2>&1
    pdflatex -interaction=nonstopmode -halt-on-error $doc.tex >> build_$doc.log 2>&1
    pdflatex -interaction=nonstopmode -halt-on-error $doc.tex >> build_$doc.log 2>&1
    echo "--- $doc: $(pdfinfo $doc.pdf | grep Pages)"
    echo "undefined refs/cites: $(grep -c 'undefined' $doc.log || true); overfull: $(grep -c Overfull $doc.log || true)"
done
echo "unresolved macros (should be 0): $(grep -c '{?}' numbers_biorxiv.tex || true)"
echo "PENDING markers (should be 0): $(grep -c PENDING numbers_biorxiv.tex || true)"
