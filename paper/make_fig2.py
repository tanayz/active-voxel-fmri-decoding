"""Figure 2: ground-truth leakage simulation (pure noise, chance = 1/8).
(A) reported accuracy vs number of candidate voxels, raw-frame layout at BOLD-like autocorrelation
    (phi = 0.5), plus the beta-series layout; (B) ungrouped cross-validation vs temporal autocorrelation.
Usage: python make_fig2.py leakage_simulation.json"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1-3
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e4df", "#ffffff"


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=7.5, length=3, width=0.8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def mean_sd(rows, key, **cond):
    v = [r[key] for r in rows if all(r[c] == x for c, x in cond.items())]
    return np.mean(v), np.std(v, ddof=1)


def main():
    sim = json.load(open(sys.argv[1]))
    fr, bt, ps, phis = sim["frames"], sim["betas"], sim["config"]["p_values"], sim["config"]["phi_values"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "text.color": INK, "axes.labelcolor": INK2})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.6, 1.0]})

    style(axA)
    axA.axhline(0.125, color=INK2, lw=0.8, ls=(0, (3, 3)))
    axA.text(ps[0] * 1.05, 0.075, "chance", color=INK2, fontsize=7, va="center")
    series = [("correct", fr, {"phi": 0.5}, BLUE, "-", "o", "correct protocol (raw frames)"),
              ("select_before", fr, {"phi": 0.5}, ORANGE, "-", "o", "selection before split (raw frames)"),
              ("ungrouped_cv5", fr, {"phi": 0.5}, AQUA, "-", "o", "ungrouped 5-fold CV (raw frames)"),
              ("correct", bt, {}, BLUE, (0, (4, 2)), "s", "correct protocol (beta-series)"),
              ("select_before", bt, {}, ORANGE, (0, (4, 2)), "s", "selection before split (beta-series)")]
    for key, rows, cond, col, ls, mk, lab in series:
        ms = [mean_sd(rows, key, p=p, **cond) for p in ps]
        m = np.array([x[0] for x in ms]); s = np.array([x[1] for x in ms])
        axA.plot(ps, m, color=col, ls=ls, lw=1.6, marker=mk, ms=4.5, mec=SURFACE, mew=0.8, label=lab)
        axA.fill_between(ps, m - s, m + s, color=col, alpha=0.12, lw=0)
    axA.set_xscale("log")
    axA.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axA.set_xticks(ps); axA.set_xticklabels([f"{p:,}" for p in ps])
    axA.set_ylim(0, 1.05); axA.set_xlim(ps[0] / 1.3, ps[-1] * 1.9)
    axA.set_xlabel("candidate voxels (pure noise)")
    axA.set_ylabel("reported accuracy")
    axA.set_title("A  Accuracy reported on data with no signal", loc="left", fontsize=8.5, fontweight="bold")
    axA.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, fontsize=6.5,
               handlelength=2.6, columnspacing=1.0)

    style(axB)
    axB.axhline(0.125, color=INK2, lw=0.8, ls=(0, (3, 3)))
    for key, col, lab in (("correct", BLUE, "correct protocol"), ("ungrouped_cv5", AQUA, "ungrouped 5-fold CV")):
        ms = [np.mean([mean_sd(fr, key, p=p, phi=phi)[0] for p in ps]) for phi in phis]
        axB.plot(phis, ms, color=col, lw=1.6, marker="o", ms=4.5, mec=SURFACE, mew=0.8, label=lab)
    axB.set_xticks(phis)
    axB.set_ylim(0, 0.35)
    axB.set_xlabel("temporal autocorrelation (AR(1) phi)")
    axB.set_title("B  Ungrouped CV vs autocorrelation", loc="left", fontsize=8.5, fontweight="bold")
    axB.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, frameon=False, fontsize=6.5)

    fig.tight_layout(w_pad=2.0)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig2_simulation")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
    print("wrote", out + ".pdf/.png")


if __name__ == "__main__":
    main()
