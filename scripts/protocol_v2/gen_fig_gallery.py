"""Fig B (Results 5.5) — representative real tree point clouds (Lu R6 #8).

Four real FOR-species20K trees spanning conifer and broadleaf crown architecture,
to illustrate WHY the hard confusions concentrate among structurally similar
species (corrected finding: crown architecture, not genus). Real data only, from
poster/assets/gallery_*.npz. Vector PDF + PNG.
Run from the repo root with the project's virtualenv Python.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pubfig_style import COL, savefig, set_pub_style

ASSETS = r"poster/assets"  # poster assets are not shipped in this repository
OUT = "runs/protocol_v2/figs/figgallery_species"

# conifers first, then broadleaves, so the crown-type contrast reads left->right
SPECIES = [
    ("Picea_abies", "Picea abies", "conifer"),
    ("Pinus_sylvestris", "Pinus sylvestris", "conifer"),
    ("Fagus_sylvatica", "Fagus sylvatica", "broadleaf"),
    ("Betula_pendula", "Betula pendula", "broadleaf"),
]


def load_cloud(key: str, max_pts: int = 45000) -> np.ndarray:
    d = np.load(os.path.join(ASSETS, f"gallery_{key}.npz"))
    pts = np.asarray(d["points"], dtype=np.float32)
    if pts.shape[0] > max_pts:
        rng = np.random.default_rng(7)
        pts = pts[rng.choice(pts.shape[0], size=max_pts, replace=False)]
    return pts


def build():
    set_pub_style()
    fig, axes = plt.subplots(1, 4, figsize=(COL["double"], COL["double"] * 0.40))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.04, wspace=0.10)
    for ax, (key, name, crown) in zip(axes, SPECIES):
        pts = load_cloud(key)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        h = x if (x.max() - x.min()) >= (y.max() - y.min()) else y
        ax.scatter(h, z, c=z, cmap="viridis", s=0.8, marker=".", linewidths=0,
                   rasterized=True)
        ax.set_axis_off(); ax.set_aspect("equal")
        hx0, hx1 = np.percentile(h, 0.5), np.percentile(h, 99.5)
        pad = (hx1 - hx0) * 0.08
        ax.set_xlim(hx0 - pad, hx1 + pad); ax.set_ylim(z.min() - pad, z.max() + pad)
        ax.set_title(f"{name}\n({crown})", fontsize=7.5, fontstyle="italic", pad=3,
                     linespacing=1.2)
    savefig(fig, OUT)
    plt.close(fig)
    print(f"wrote {OUT}.pdf and {OUT}.png")


if __name__ == "__main__":
    build()
