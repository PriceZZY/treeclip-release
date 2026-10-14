"""Export the two REAL raster panels for the TikZ Fig 1 redraw:
  (a) clean Pinus sylvestris point cloud (no title, tight, transparent bg)
  (b) the 6 real PCA depth views composited into a clean 3x2 strip (white bg)
so the TikZ figure can \\includegraphics them as image nodes while the schematic
stages (c,d,e) are drawn natively in TikZ. No fabricated data. Run from repo root
with the project's virtualenv Python.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gen_fig1_pipeline import load_depth_view, load_tree_xyz  # reuse real loaders

OUT = "runs/protocol_v2/figs"
os.makedirs(OUT, exist_ok=True)


def export_cloud():
    pts = load_tree_xyz()
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    h = x if (x.max() - x.min()) >= (y.max() - y.min()) else y
    fig, ax = plt.subplots(figsize=(2.3, 3.4))
    ax.scatter(h, z, c=z, cmap="viridis", s=0.7, marker=".", linewidths=0,
               rasterized=True)
    ax.set_axis_off()
    ax.set_aspect("equal")
    hx0, hx1 = np.percentile(h, 0.5), np.percentile(h, 99.5)
    pad = (hx1 - hx0) * 0.06
    ax.set_xlim(hx0 - pad, hx1 + pad)
    ax.set_ylim(z.min() - pad, z.max() + pad)
    fig.savefig(f"{OUT}/fig1_asset_cloud.png", dpi=400, transparent=True,
                bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    print(f"wrote {OUT}/fig1_asset_cloud.png")


def export_views():
    tiles = [load_depth_view(i) for i in range(6)]
    h = min(t.shape[0] for t in tiles)
    w = min(t.shape[1] for t in tiles)
    tiles = [t[:h, :w] for t in tiles]
    gap = max(3, w // 18)
    cols, rows = 3, 2
    cw = cols * w + (cols - 1) * gap
    ch = rows * h + (rows - 1) * gap
    canvas = np.full((ch, cw, 3), 255, dtype=np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        canvas[r * (h + gap):r * (h + gap) + h, c * (w + gap):c * (w + gap) + w] = t
    Image.fromarray(canvas).save(f"{OUT}/fig1_asset_views.png")
    print(f"wrote {OUT}/fig1_asset_views.png  ({cw}x{ch})")


if __name__ == "__main__":
    export_cloud()
    export_views()
