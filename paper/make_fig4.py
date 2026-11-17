"""Figure: closed-form prediction of the accuracy that selection before the split reports on pure noise
(two classes, chance = 1/2) against simulation.
(A) accuracy vs number of examples for 5,000 candidate features and k = 10, 100, 500 selected: lines,
    prediction; markers, simulated cross-validation; (B) prediction vs simulation for every configuration.
Usage: python make_fig4.py selection_bias_theory.json"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1-3
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e4df", "#ffffff"
P_SHOWN = 5000


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=7.5, length=3, width=0.8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def predicted(n, p, k, folds):
    q = k / (2.0 * p)
    c = norm.isf(q)
    m = 1.0 + c * norm.pdf(c) / q
    n_t, n_v = n * (folds - 1) / folds, n / folds
    return norm.cdf(np.sqrt(k) * (m / n - 1.0 / n_t) / np.sqrt(m / n + n_v / n_t ** 2))


def main():
    sim = json.load(open(sys.argv[1]))
    rows, folds = sim["rows"], sim["J"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "text.color": INK, "axes.labelcolor": INK2})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.5, 1.0]})

    style(axA)
    axA.axhline(0.5, color=INK2, lw=0.8, ls=(0, (3, 3)))
    grid = np.geomspace(36, 360, 80)
    for k, col in ((10, BLUE), (100, ORANGE), (500, AQUA)):
        sel = sorted((r for r in rows if r["p"] == P_SHOWN and r["k"] == k), key=lambda r: r["n"])
        axA.plot(grid, [predicted(n, P_SHOWN, k, folds) for n in grid], color=col, lw=1.6)
        ns = [r["n"] for r in sel]
        axA.plot(ns, [r["leak_centroid"] for r in sel], ls="none", marker="o", ms=5, color=col, mec=SURFACE,
                 mew=0.8)
        axA.plot(ns, [r["leak_logreg"] for r in sel], ls="none", marker="s", ms=4.5, mfc=SURFACE, mec=col,
                 mew=1.1)
        axA.text(grid[-1] * 1.04, predicted(grid[-1], P_SHOWN, k, folds), f"k = {k}", color=INK2, fontsize=7,
                 va="center")
    nest = sorted((r for r in rows if r["p"] == P_SHOWN and r["k"] == 100), key=lambda r: r["n"])
    axA.plot([r["n"] for r in nest], [r["nested_centroid"] for r in nest], ls="none", marker="D", ms=4,
             color=INK2, mec=SURFACE, mew=0.8)
    axA.set_xscale("log")
    axA.set_xticks([40, 80, 160, 320]); axA.set_xticklabels(["40", "80", "160", "320"])
    axA.minorticks_off()
    axA.set_xlim(34, 470); axA.set_ylim(0.4, 1.03)
    axA.set_xlabel(f"examples n ({P_SHOWN:,} noise features, two classes)")
    axA.set_ylabel("reported accuracy")
    axA.set_title("A  Selection before the split, on pure noise", loc="left", fontsize=8.5, fontweight="bold")
    handles = [plt.Line2D([], [], color=INK2, lw=1.6, label="closed-form prediction"),
               plt.Line2D([], [], ls="none", marker="o", ms=5, color=INK2, mec=SURFACE, label="simulation, nearest centroid"),
               plt.Line2D([], [], ls="none", marker="s", ms=4.5, mfc=SURFACE, mec=INK2, mew=1.1, label="simulation, logistic regression"),
               plt.Line2D([], [], ls="none", marker="D", ms=4, color=INK2, mec=SURFACE, label="selection inside the split (k = 100)")]
    axA.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False,
               fontsize=6.5, columnspacing=1.0)

    style(axB)
    axB.plot([0.6, 1.0], [0.6, 1.0], color=INK2, lw=0.8, ls=(0, (3, 3)))
    axB.plot([r["predicted"] for r in rows], [r["leak_centroid"] for r in rows], ls="none", marker="o", ms=5,
             color=BLUE, mec=SURFACE, mew=0.8, label="nearest centroid")
    axB.plot([r["predicted"] for r in rows], [r["leak_logreg"] for r in rows], ls="none", marker="s", ms=4.5,
             mfc=SURFACE, mec=ORANGE, mew=1.1, label="logistic regression")
    axB.set_xlim(0.62, 1.02); axB.set_ylim(0.62, 1.02)
    axB.set_xlabel("predicted accuracy")
    axB.set_ylabel("simulated accuracy")
    axB.set_title(f"B  All {len(rows)} configurations", loc="left", fontsize=8.5, fontweight="bold")
    axB.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, frameon=False, fontsize=6.5)

    fig.subplots_adjust(left=0.08, right=0.985, top=0.9, bottom=0.34, wspace=0.28)
    out = os.path.dirname(os.path.abspath(__file__))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig4_theory.{ext}"), dpi=300, facecolor=SURFACE)


if __name__ == "__main__":
    main()
