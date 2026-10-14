"""Fig A (Method 3.2) — multi-view depth-rendering demo (Lu R6 #8).

A real Pinus sylvestris point cloud (treeID 2573) and its six PCA-rotated depth
views, illustrating the render operation. Real data only. Vector PDF + PNG.
Run from the repo root with the project's virtualenv Python.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gen_fig1_pipeline import load_depth_view, load_tree_xyz
from pubfig_style import COL, savefig, set_pub_style

OUT = "runs/protocol_v2/figs/figrender_demo"


def build():
    set_pub_style()
    pts = load_tree_xyz()
    views = [load_depth_view(i) for i in range(6)]

    fig = plt.figure(figsize=(COL["double"], COL["double"] * 0.34))
    # left: the real point cloud
    axc = fig.add_axes([0.015, 0.06, 0.24, 0.88])
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    h = x if (x.max() - x.min()) >= (y.max() - y.min()) else y
    axc.scatter(h, z, c=z, cmap="viridis", s=0.9, marker=".", linewidths=0,
                rasterized=True)
    axc.set_axis_off(); axc.set_aspect("equal")
    hx0, hx1 = np.percentile(h, 0.5), np.percentile(h, 99.5)
    pad = (hx1 - hx0) * 0.06
    axc.set_xlim(hx0 - pad, hx1 + pad); axc.set_ylim(z.min() - pad, z.max() + pad)
    axc.set_title("Tree point cloud", fontsize=8)

    # arrow + label
    fig.text(0.285, 0.52, r"$\mathcal{R}$, six PCA azimuths", ha="center",
             va="bottom", fontsize=7, style="italic")
    fig.add_artist(FancyArrowPatch((0.255, 0.5), (0.325, 0.5),
                   transform=fig.transFigure, arrowstyle="-|>",
                   mutation_scale=11, lw=1.2, color="#444444"))

    # right: 6 depth views in a 2x3 grid, labelled view 1..6
    gx0, gw, gy0, gh = 0.345, 0.645, 0.06, 0.84
    cols, rows = 3, 2
    cellw = gw / cols * 0.93
    cellh = gh / rows * 0.86
    for k, v in enumerate(views):
        r, c = divmod(k, cols)
        ax = fig.add_axes([gx0 + c * (gw / cols), gy0 + (rows - 1 - r) * (gh / rows),
                           cellw, cellh])
        ax.imshow(v, aspect="equal", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#888888"); s.set_linewidth(0.5)
        ax.set_title(f"view {k + 1}", fontsize=6.5, pad=2)
    fig.text(gx0 + gw / 2, 0.965, r"Six $224\times224$ PCA-rotated depth views",
             ha="center", va="top", fontsize=8)

    savefig(fig, OUT)
    plt.close(fig)
    print(f"wrote {OUT}.pdf and {OUT}.png")


if __name__ == "__main__":
    build()
