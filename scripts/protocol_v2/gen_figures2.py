"""Figures 3-4 for the v1 revision (battery experiments).
  Fig 3: PCA dimension sweep — CLIP NMI vs dim for both datasets, with geo-11d reference lines.
  Fig 4: clustering-algorithm robustness — NMI for KMeans/GMM/Spectral/Agglomerative, CLIP-11d vs geo-11d, per dataset.
Numbers from dim_sweep_*.json + clustering_methods_*.json. CPU-only. Saves runs/protocol_v2/figs/.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
GEN = ROOT / "runs/protocol_v2/gen"
FIGS = ROOT / "runs/protocol_v2/figs"; FIGS.mkdir(parents=True, exist_ok=True)


def fig3_dim_sweep():
    f = json.load(open(GEN / "dim_sweep_for.json")); b = json.load(open(GEN / "dim_sweep_biodiv.json"))
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for d, lab, col in [(f, "FOR-species20K", "#1f77b4"), (b, "BioDiv-3DTrees (ULS)", "#2ca02c")]:
        sw = d["CLIP_pca_dim_sweep_NMI"]
        dims = sorted(int(k) for k in sw)
        ys = [sw[str(x)]["mean"] for x in dims]
        ax.plot(dims, ys, "o-", color=col, label=f"CLIP, {lab}")
        ax.axhline(d["geo_11d_NMI_reference"]["mean"], ls="--", color=col, alpha=0.6,
                   label=f"geometry-11d, {lab}")
    ax.set_xscale("log", base=2); ax.set_xticks([4, 8, 11, 16, 32, 64, 128, 256, 512])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("PCA dimensionality of CLIP feature"); ax.set_ylabel("NMI (species alignment)")
    ax.set_title("PCA dimension sweep: CLIP NMI vs geometry", fontsize=11)
    ax.set_ylim(0, 0.42); ax.legend(fontsize=8, loc="center right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "fig3_dim_sweep.png", dpi=300); plt.close(fig)
    print("wrote fig3_dim_sweep.png")


def fig4_methods():
    f = json.load(open(GEN / "clustering_methods_for.json")); b = json.load(open(GEN / "clustering_methods_biodiv.json"))
    methods = ["kmeans", "gmm", "spectral", "agglo"]; labels = ["KMeans", "GMM", "Spectral", "Agglom."]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, d, title in [(axes[0], f, "FOR-species20K"), (axes[1], b, "BioDiv-3DTrees (ULS)")]:
        clip = [d["fix_K_methods"][m]["CLIP_11d"]["NMI"]["mean"] for m in methods]
        geo = [d["fix_K_methods"][m]["geo_11d"]["NMI"]["mean"] for m in methods]
        x = np.arange(len(methods)); w = 0.38
        ax.bar(x - w/2, clip, w, label="CLIP-11d", color="#1f77b4")
        ax.bar(x + w/2, geo, w, label="geometry-11d", color="#d62728")
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_title(title); ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 0.42)
    axes[0].set_ylabel("NMI (species alignment)"); axes[0].legend(fontsize=9, loc="upper right")
    fig.suptitle("Clustering-algorithm robustness: CLIP > geometry on all four fixed-K algorithms", y=1.0)
    fig.tight_layout(); fig.savefig(FIGS / "fig4_clustering_methods.png", dpi=300); plt.close(fig)
    print("wrote fig4_clustering_methods.png")


if __name__ == "__main__":
    fig3_dim_sweep(); fig4_methods()
