"""
Figure 3 -- where whole-brain active-voxel selection puts its voxels.

(A) Per subject, axial and sagittal slices of the mean EPI with the fraction of outer folds in which each
    voxel was among the top 500 (training data only), and the VT mask outline.
(B) Share of the selected voxels inside VT versus VT's share of the whole-brain mask (log scale).
(C) Nested LORO accuracy from all VT voxels, from whole-brain selection, and from the whole brain minus VT,
    with the permutation null of the outside-VT pipeline.
Style follows make_figures.py: validated categorical slots, gray for nulls, ink-colored text.

Usage: python make_fig3.py MAPS_DIR EXTRAS_DIR HAXBY_FINAL_JSON
  MAPS_DIR   selfreq_s{n}_k500.nii.gz, vt_s{n}.nii.gz, meanepi_s{n}.nii.gz
  EXTRAS_DIR sel_anatomy_{a,b,c}.json, outvt_perm_{a,b,c}.json, wb_ablation_{a,b}.json
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1-3
NULL_GRAY = "#9a9891"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e4df", "#ffffff"
SLAB = 2                                    # maps show the maximum over idx +/- SLAB slices
SEQ = LinearSegmentedColormap.from_list("seq_orange", ["#fbd3bf", "#eb6834", "#8a2c0b"])  # one hue, light -> dark

maps, extras, final = sys.argv[1:4]
subjects = [str(s) for s in range(1, 7)]


def jload(*names):
    out = {}
    for n in names:
        p = os.path.join(extras, n)
        if os.path.isfile(p):
            out.update(json.load(open(p))["subjects"])
    return out


anat = jload("sel_anatomy_a.json", "sel_anatomy_b.json", "sel_anatomy_c.json")
perm = jload("outvt_perm_a.json", "outvt_perm_b.json", "outvt_perm_c.json")
wb = jload("wb_ablation_a.json", "wb_ablation_b.json")
hx = json.load(open(final))["subjects"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=7, length=3, width=0.8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def show_slice(ax, bg, fq, vt, zooms, plane, idx):
    """plane 'axial': anterior up, subject's left on the left; 'sagittal': anterior right, superior up."""
    lo, hi = max(idx - SLAB, 0), idx + SLAB + 1
    if plane == "axial":
        sl = lambda a, red: red(a[:, :, lo:hi], axis=2).T[:, ::-1]   # rows y (A up), cols x flipped (L left)
        w, h = zooms[0], zooms[1]
    else:
        sl = lambda a, red: red(a[lo:hi, :, :], axis=0).T            # rows z (S up), cols y (A right)
        w, h = zooms[1], zooms[2]
    b, f, v = sl(bg, np.mean), sl(fq, np.max), sl(vt.astype(float), np.max) > 0
    ext = [0, b.shape[1] * w, 0, b.shape[0] * h]
    ax.imshow(b, cmap="gray", origin="lower", extent=ext, vmin=np.percentile(bg[bg > 0], 2),
              vmax=np.percentile(bg[bg > 0], 99.5), interpolation="nearest")
    ax.imshow(np.ma.masked_where(f <= 0, f), cmap=SEQ, vmin=0, vmax=1, origin="lower", extent=ext,
              interpolation="nearest")
    if v.any():
        ax.contour(v.astype(float), levels=[0.5], colors=BLUE, linewidths=0.9, origin="lower", extent=ext)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "text.color": INK,
                     "axes.labelcolor": INK2, "axes.titlecolor": INK})
fig = plt.figure(figsize=(7.0, 6.4))
fig.patch.set_facecolor(SURFACE)
gs = fig.add_gridspec(2, 6, top=0.905, bottom=0.535, hspace=0.06, wspace=0.04, left=0.06, right=0.99)
gsb = fig.add_gridspec(1, 2, top=0.405, bottom=0.17, wspace=0.30, left=0.08, right=0.97)
im = None
for i, s in enumerate(subjects):
    fq_img = nib.load(os.path.join(maps, f"selfreq_s{s}_k500.nii.gz"))
    fq, zooms = fq_img.get_fdata(), fq_img.header.get_zooms()[:3]
    vt = nib.load(os.path.join(maps, f"vt_s{s}.nii.gz")).get_fdata() > 0
    bg = nib.load(os.path.join(maps, f"meanepi_s{s}.nii.gz")).get_fdata()
    z = int(np.argmax(vt.sum(axis=(0, 1))))                    # axial slice with the most VT voxels
    x = int(np.argmax((fq * vt).sum(axis=(1, 2))))             # sagittal slice through the most-selected VT
    ax1, ax2 = fig.add_subplot(gs[0, i]), fig.add_subplot(gs[1, i])
    show_slice(ax1, bg, fq, vt, zooms, "axial", z)
    show_slice(ax2, bg, fq, vt, zooms, "sagittal", x)
    ax1.set_title(f"Subject {s}", fontsize=7.5, color=INK, pad=2)
    if i == 0:
        ax1.text(-0.06, 0.5, "axial", transform=ax1.transAxes, rotation=90, ha="right", va="center", fontsize=7, color=INK2)
        ax2.text(-0.06, 0.5, "sagittal", transform=ax2.transAxes, rotation=90, ha="right", va="center", fontsize=7, color=INK2)
        ax1.text(0.5, 0.98, "ant", transform=ax1.transAxes, ha="center", va="top", fontsize=6, color="w")
        ax1.text(0.02, 0.5, "L", transform=ax1.transAxes, ha="left", va="center", fontsize=6, color="w")
        ax2.text(0.98, 0.5, "ant", transform=ax2.transAxes, ha="right", va="center", fontsize=6, color="w")
fig.text(0.06, 0.95, "A  Where whole-brain selection puts its top 500 voxels (training runs only)",
         fontsize=9, fontweight="bold", color=INK, ha="left")
cax = fig.add_axes([0.40, 0.495, 0.22, 0.011])
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, 1)), cax=cax, orientation="horizontal")
cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["0", "0.5", "1"], fontfamily="DejaVu Sans"); cb.ax.tick_params(labelsize=6, colors=INK2, length=2); cb.outline.set_visible(False)
cax.set_xlabel("fraction of folds in which the voxel was selected", fontsize=6.5, color=INK2, labelpad=2)
fig.text(0.68, 0.497, "— VT mask outline", color=BLUE, fontsize=6.5, ha="left", va="center")

# (B) enrichment
axB = fig.add_subplot(gsb[0, 0])
style(axB)
xs = np.arange(len(subjects))
frac = [100 * np.mean([f["500"]["frac_in_vt"] for f in anat[s]["folds"]]) for s in subjects]
lo = [100 * min(f["500"]["frac_in_vt"] for f in anat[s]["folds"]) for s in subjects]
hi = [100 * max(f["500"]["frac_in_vt"] for f in anat[s]["folds"]) for s in subjects]
chance = [100 * anat[s]["vt_chance"] for s in subjects]
axB.scatter(xs, chance, s=34, marker="s", color=NULL_GRAY, edgecolor=SURFACE, linewidth=1.0, zorder=4,
            label="VT share of whole-brain mask (chance)")
axB.vlines(xs, lo, hi, color=ORANGE, linewidth=1.2, zorder=3)
axB.scatter(xs, frac, s=42, color=ORANGE, edgecolor=SURFACE, linewidth=1.2, zorder=4,
            label="share of selected voxels in VT (range over folds)")
for x, f, c in zip(xs, frac, chance):
    axB.text(x, f * 1.45, f"{f / c:.0f}×", fontsize=6.5, color=INK2, ha="center", va="bottom")
axB.set_yscale("log")
axB.set_ylim(0.5, 80)
axB.set_yticks([1, 3, 10, 30]); axB.set_yticklabels(["1%", "3%", "10%", "30%"])
axB.set_xticks(xs); axB.set_xticklabels(subjects, fontsize=7); axB.set_xlabel("Subject", fontsize=7.5)
axB.set_xlim(-0.5, len(xs) - 0.5)
axB.set_title("B  VT enrichment among selected voxels", loc="left", fontsize=9, fontweight="bold")
axB.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False, fontsize=6.6,
           labelcolor=INK2, handletextpad=0.3)

# (C) decoding with and without VT
axC = fig.add_subplot(gsb[0, 1])
style(axC)
off = 0.22
series = [("all VT voxels, no selection", BLUE, [hx[s]["all_vt_baseline"]["mean"] for s in subjects], -off),
          ("whole brain, selection", AQUA, [wb[s]["selection"]["mean"] for s in subjects], 0.0),
          ("whole brain minus VT, selection", ORANGE,
           [anat[s]["outside_vt_decoding"]["mean"] for s in subjects], off)]
if perm:
    nm = [perm[s]["null_mean"] for s in subjects]
    nx = [perm[s]["null_max"] for s in subjects]
    axC.vlines(xs + off, nm, nx, color=NULL_GRAY, linewidth=1.2, zorder=3)
    axC.scatter(xs + off, nm, s=14, color=NULL_GRAY, zorder=4, label="permutation null, minus VT (mean, to max)")
for label, color, vals, o in series:
    axC.scatter(xs + o, vals, s=30, color=color, edgecolor=SURFACE, linewidth=1.0, zorder=5, label=label)
axC.axhline(0.125, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)), zorder=2)
axC.text(xs[-1] + 0.45, 0.125, "chance", color=INK2, fontsize=6.5, va="center", ha="left",
         bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
axC.set_ylim(0, 0.9); axC.set_ylabel("Accuracy (nested LORO)", fontsize=7.5)
axC.set_xticks(xs); axC.set_xticklabels(subjects, fontsize=7); axC.set_xlabel("Subject", fontsize=7.5)
axC.set_xlim(-0.5, len(xs) - 0.1)
axC.set_title("C  Decoding with and without VT", loc="left", fontsize=9, fontweight="bold")
axC.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=6.6,
           labelcolor=INK2, handletextpad=0.3, columnspacing=0.8)

fig.savefig("fig3_anatomy.pdf", bbox_inches="tight")
fig.savefig("fig3_anatomy.png", dpi=200, bbox_inches="tight")
print("fig3_anatomy.{pdf,png} written")
