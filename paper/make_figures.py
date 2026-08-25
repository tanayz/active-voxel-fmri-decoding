"""
Figure 1 for the bioRxiv manuscript: per-subject nested-CV accuracy (mean +/- SD over outer folds) for
(A) Haxby -- whole-VT baseline vs VT-restricted active-voxel selection, with the permutation null, and
(B) StarPlus -- beta-series + active-voxel selection vs the original active-ROI method, with the null.

Usage: python make_figures.py --haxby-final haxby_vt_final_results.json \
           --starplus starplus_nested_cv_results.json --starplus-perm starplus_permutation_results.json
Writes fig1_accuracy.pdf (vector) and fig1_accuracy.png next to this script.

Palette: the validated default categorical palette (slot 1 blue, slot 2 orange) from the dataviz
reference instance; neutral gray for the null distribution; text in ink colours, never series colours.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 (light mode, validated)
NULL_GRAY = "#9a9891"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e4df"
SURFACE = "#ffffff"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def dot_series(ax, x, means, sds, color, label, offset):
    xs = np.array(x, float) + offset
    ax.errorbar(xs, means, yerr=sds, fmt="none", ecolor=color, elinewidth=1.2, capsize=2.5, capthick=1.2, zorder=3)
    ax.scatter(xs, means, s=42, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=4, label=label)
    return xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--haxby-final", required=True)
    ap.add_argument("--starplus", required=True)
    ap.add_argument("--starplus-perm", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig1_accuracy"))
    a = ap.parse_args()

    hx = json.load(open(a.haxby_final))["subjects"]
    sp = json.load(open(a.starplus))
    spp = json.load(open(a.starplus_perm))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "text.color": INK,
                         "axes.labelcolor": INK2, "axes.titlecolor": INK})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [2.0, 1.15]})
    fig.patch.set_facecolor(SURFACE)

    # ---------------------------------------------------------------- A: Haxby
    subs = sorted(hx, key=int)
    x = np.arange(1, len(subs) + 1)
    base_m = [hx[s]["all_vt_baseline"]["mean"] for s in subs]
    base_sd = [hx[s]["all_vt_baseline"]["sd"] for s in subs]
    ours_m = [hx[s]["active_voxel"]["mean"] for s in subs]
    ours_sd = [hx[s]["active_voxel"]["sd"] for s in subs]
    null_m = [hx[s]["permutation"]["null_mean"] for s in subs]
    null_max = [hx[s]["permutation"]["null_max"] for s in subs]

    style(axA)
    axA.axhline(0.125, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)), zorder=2)
    axA.text(x[-1] + 0.45, 0.125, "chance", color=INK2, fontsize=7, va="center", ha="left",
             bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
    # null distribution: mean marker with whisker to the maximum null accuracy
    axA.vlines(x, null_m, null_max, color=NULL_GRAY, linewidth=1.2, zorder=3)
    axA.scatter(x, null_m, s=18, color=NULL_GRAY, zorder=4, label="permutation null (mean, whisker to max)")
    xb = dot_series(axA, x, base_m, base_sd, BLUE, "ours, no selection (all VT voxels)", -0.13)
    xo = dot_series(axA, x, ours_m, ours_sd, ORANGE, "ours, + active-voxel selection", +0.13)
    # no per-point value labels: exact values are in the table; labels collided between subjects
    axA.set_xticks(x)
    axA.set_xticklabels([f"S{s}" for s in subs])
    axA.set_xlim(0.4, len(subs) + 0.9)
    axA.set_ylim(0, 1.0)
    axA.set_ylabel("accuracy (nested leave-one-run-out)")
    axA.set_title("A  Haxby, 8-way object category", loc="left", fontsize=9, fontweight="bold")
    axA.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=6.8,
               handletextpad=0.4, columnspacing=1.2)

    # ---------------------------------------------------------------- B: StarPlus
    sp_subs = ["04799", "04820", "04847"]
    roi = {"04799": 0.762, "04820": 0.787, "04847": 1.000}     # original active-ROI method (raw window)
    xs = np.arange(1, 4)
    ours_m = [float(np.mean(sp[s]["per_fold_scores"])) for s in sp_subs]
    ours_sd = [float(np.std(sp[s]["per_fold_scores"], ddof=1)) for s in sp_subs]
    null_m = [float(np.mean(spp[s]["null_means"])) for s in sp_subs]
    null_max = [float(np.max(spp[s]["null_means"])) for s in sp_subs]

    style(axB)
    axB.axhline(0.5, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)), zorder=2)
    axB.text(xs[-1] + 0.45, 0.5, "chance", color=INK2, fontsize=7, va="center", ha="left",
             bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
    axB.vlines(xs, null_m, null_max, color=NULL_GRAY, linewidth=1.2, zorder=3)
    axB.scatter(xs, null_m, s=18, color=NULL_GRAY, zorder=4)
    xr = axB.scatter(xs - 0.13, [roi[s] for s in sp_subs], s=42, color=BLUE, edgecolor=SURFACE, linewidth=1.2,
                     zorder=4, label="active-ROI, raw windows")
    xo = dot_series(axB, xs, ours_m, ours_sd, ORANGE, "beta-series + active-voxel (ours)", +0.13)
    axB.set_xticks(xs)
    axB.set_xticklabels(sp_subs)
    axB.set_xlim(0.4, 3.9)
    axB.set_ylim(0, 1.05)
    axB.set_ylabel("accuracy (nested 10-fold, trial-grouped)")
    axB.set_title("B  StarPlus, picture vs sentence", loc="left", fontsize=9, fontweight="bold")
    axB.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, frameon=False, fontsize=6.8,
               handletextpad=0.4)

    fig.tight_layout(w_pad=1.5)
    fig.savefig(a.out + ".pdf", bbox_inches="tight")
    fig.savefig(a.out + ".png", dpi=200, bbox_inches="tight")
    print("wrote", a.out + ".pdf", "and .png")


if __name__ == "__main__":
    main()
