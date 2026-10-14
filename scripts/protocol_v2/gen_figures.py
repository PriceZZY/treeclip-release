"""Generate the two manuscript figures (blind-review completeness):
  Fig 1: method pipeline schematic (point cloud -> 2 PCA depth views -> frozen CLIP -> L2-mean 512-d -> KMeans).
  Fig 2: per-dataset NMI bar chart, CLIP-512 / CLIP-11d / geo-11d, with the within-source (source-controlled)
         CLIP and geo NMI overlaid, so the source attenuation is visible.
Numbers from unsup_robust_*.json + source_control_robust.json. CPU-only. Saves runs/protocol_v2/figs/.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent.parent
GEN = ROOT / "runs/protocol_v2/gen"
FIGS = ROOT / "runs/protocol_v2/figs"; FIGS.mkdir(parents=True, exist_ok=True)
MANU_FIGS = ROOT / "manuscript/cas/figs"  # vector PDFs consumed by the manuscript build


def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(11, 2.6)); ax.axis("off")
    steps = ["Tree\npoint cloud", "2 orthogonal\nPCA depth\nrenders (224²)",
             "Frozen CLIP\nViT-B/16\nimage encoder", "L2-norm +\nmean →\n512-d", "KMeans\n(k=#species)\n+ NMI"]
    x = np.linspace(0.04, 0.84, len(steps))
    for i, (xi, s) in enumerate(zip(x, steps)):
        ax.add_patch(mpatches.FancyBboxPatch((xi, 0.32), 0.13, 0.42, boxstyle="round,pad=0.01",
                     linewidth=1.4, edgecolor="#2c6", facecolor="#eafbe7" if i < 4 else "#e7f0fb"))
        ax.text(xi + 0.065, 0.53, s, ha="center", va="center", fontsize=8.5)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x[i+1], 0.53), xytext=(xi + 0.13, 0.53),
                        arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.4))
    ax.text(0.5, 0.93, "TreeCLIP pipeline (frozen, unsupervised)", ha="center", fontsize=11, weight="bold")
    ax.text(0.5, 0.07, "No training / fine-tuning at any stage; species recovered by unsupervised clustering",
            ha="center", fontsize=8, style="italic", color="#555")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(FIGS / "fig1_pipeline.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print("wrote fig1_pipeline.png")


def fig2_bars():
    f = json.load(open(GEN / "unsup_robust_for.json")); b = json.load(open(GEN / "unsup_robust_biodiv.json"))
    sc = json.load(open(GEN / "source_control_robust.json"))
    lw = json.load(open(GEN / "local_weinmann.json"))  # F-17: fair local multi-scale Weinmann baseline
    datasets = ["FOR-species20K", "BioDiv-3DTrees (ULS)"]
    clip11 = [f["CLIP_matched_11d"]["NMI"]["mean"], b["CLIP_matched_11d"]["NMI"]["mean"]]
    geo_single = [f["geo_11d"]["NMI"]["mean"], b["geo_11d"]["NMI"]["mean"]]
    geo_weinmann = [lw["for"]["geo_local_multiscale_weinmann"]["NMI_mean"],
                    lw["biodiv"]["geo_local_multiscale_weinmann"]["NMI_mean"]]
    ws_clip = [sc["for"]["within_source_NMI_CLIP_11d_mean"], sc["biodiv"]["within_source_NMI_CLIP_11d_mean"]]
    ws_geo = [sc["for"]["within_source_NMI_geo_11d_mean"], sc["biodiv"]["within_source_NMI_geo_11d_mean"]]
    x = np.arange(2); w = 0.18
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.bar(x - 1.5*w, clip11, w, label="CLIP-11d (global, dim-matched)", color="#1f77b4")
    ax.bar(x - 0.5*w, geo_single, w, label="geometry: whole-tree single-scale (global)", color="#f0a3a3")
    ax.bar(x + 0.5*w, geo_weinmann, w, label="geometry: local multi-scale Weinmann (global)", color="#d62728")
    ax.bar(x + 1.5*w, ws_clip, w, label="CLIP-11d (within-source)", color="#9ecae1", hatch="//")
    # within-source geometry (single-scale) as black markers on the within-source column
    ax.scatter(x + 1.5*w, ws_geo, marker="_", s=380, color="black", zorder=5,
               label="geometry single-scale (within-source)")
    ax.axhline(0.003, ls=":", color="gray", lw=1); ax.text(1.42, 0.012, "random ≈0.003", fontsize=7, color="gray")
    ax.set_xticks(x); ax.set_xticklabels(datasets); ax.set_ylabel("NMI (species alignment)")
    ax.set_title("Unsupervised NMI: CLIP vs geometry (single-scale and fair local multi-scale),\nglobal and within-source (source-controlled)", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", ncol=2); ax.set_ylim(0, 0.42)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_nmi_bars.png", dpi=300)
    fig.savefig(MANU_FIGS / "fig2_nmi_bars.pdf")  # vector PDF for the manuscript build
    plt.close(fig)
    print("wrote fig2_nmi_bars.png + manuscript/cas/figs/fig2_nmi_bars.pdf "
          f"(geo single-scale={geo_single}, geo local-multiscale Weinmann={geo_weinmann})")


if __name__ == "__main__":
    fig1_pipeline(); fig2_bars()
