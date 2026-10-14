"""DINOv2 twin of Fig 5 (qualitative cluster structure), to visually show the SAME
species-coherent-but-source-tinged structure emerges from a text-free backbone.

SPLIT into two double-column figures (mirrors gen_qualitative.py):
  fig5a_tsne_dino:      2 rows x 2 cols t-SNE [by species | by source] with legends.
  fig5b_confusion_dino: 2 Hungarian row-normalised confusions, shared viridis colorbar,
                        species CODES on both axes, x-labels rotated 45 deg.
DINOv2 aligned to the feature file by treeID. RNG seeds fixed.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pubfig_style import set_pub_style, savefig  # noqa: F401  (style applied below)
from gen_qualitative import analyzable_mask, compute, fig_tsne, fig_confusion, DATASETS

ROOT = Path(__file__).resolve().parent.parent.parent


def load_dino(which):
    if which == "for":
        d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
        dn = np.load(ROOT / "data/raw/for_species20k/for_dino.npz", allow_pickle=True)
        y = d["species"].astype(str); g = d["dataset"].astype(str); tref = d["treeID"]; fagus = None
    else:
        d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
        dn = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_dino.npz", allow_pickle=True)
        y = d["species"].astype(str); g = d["region"].astype(str); tref = d["treeID"]; fagus = 0.39
    idx = {str(t): i for i, t in enumerate(dn["treeID"])}
    rows = np.array([idx.get(str(t), -1) for t in tref]); assert (rows >= 0).all()
    Xd = dn["X_dino"][rows]
    m = analyzable_mask(y, g); Xd, y, g = Xd[m], y[m], g[m]
    if fagus is not None:
        c = Counter(y); top = max(c, key=c.get); n = len(y)
        if c[top] / n > fagus:
            tgt = int(fagus / (1 - fagus) * (n - c[top])); ii = np.where(y == top)[0]
            drop = set(np.random.RandomState(0).choice(ii, c[top] - tgt, replace=False))
            k = np.array([i not in drop for i in range(n)]); Xd, y, g = Xd[k], y[k], g[k]
    return Xd, y, g


def main():
    set_pub_style()
    results = {which: compute(which, loader=load_dino) for which, _ in DATASETS}
    fig_tsne(results, "fig5a_tsne_dino", backbone="DINOv2")
    fig_confusion(results, "fig5b_confusion_dino")


if __name__ == "__main__":
    main()
