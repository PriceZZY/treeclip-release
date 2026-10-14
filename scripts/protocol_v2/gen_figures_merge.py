"""Backbone-agnostic manuscript figures for the DINOv2/ImageNet/random merge. Overwrites the same
PNG filenames the LaTeX build references, so the figures auto-update:
  Fig 1: pipeline schematic, generalized to "frozen vision-foundation encoder (CLIP / DINOv2 / ...)".
  Fig 2: two panels — (left) global matched-11d NMI and (right) within-source NMI, for
         CLIP / DINOv2 / ImageNet / random / geometry on both datasets. Random-below-geo is the
         visual punchline of the causal control.
  Fig 3: PCA dimension sweep, CLIP and DINOv2 curves vs geo reference.
  Fig 4: clustering-algorithm robustness (4 algos), CLIP / DINOv2 / geo.
Numbers from runs/protocol_v2/gen/{unsup_robust_multi,source_control_multi,dim_sweep*,clustering_methods*}.
CPU-only. Usage: python gen_figures_merge.py [fig1|fig2|fig3|fig4|all]
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pubfig_style import set_pub_style, COL, SEMANTIC, LABEL, savefig

ROOT = Path(__file__).resolve().parent.parent.parent
GEN = ROOT / "runs/protocol_v2/gen"
FIGS = ROOT / "runs/protocol_v2/figs"; FIGS.mkdir(parents=True, exist_ok=True)
DS = [("for", "FOR-species20K"), ("biodiv", "BioDiv-3DTrees (ULS)")]


def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(11, 2.6)); ax.axis("off")
    steps = ["Tree\npoint cloud", "6 PCA-rotated\ndepth\nrenders (224²)",
             "Frozen vision-\nfoundation encoder\n(CLIP / DINOv2)", "L2-norm +\nmean →\n512/768-d", "KMeans\n(k=#species)\n+ NMI"]
    x = np.linspace(0.04, 0.84, len(steps))
    for i, (xi, s) in enumerate(zip(x, steps)):
        ax.add_patch(mpatches.FancyBboxPatch((xi, 0.32), 0.13, 0.42, boxstyle="round,pad=0.01",
                     linewidth=1.4, edgecolor="#2c6", facecolor="#eafbe7" if i < 4 else "#e7f0fb"))
        ax.text(xi + 0.065, 0.53, s, ha="center", va="center", fontsize=8.0)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x[i+1], 0.53), xytext=(xi + 0.13, 0.53), arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.4))
    ax.text(0.5, 0.93, "Frozen vision-foundation pipeline (unsupervised)", ha="center", fontsize=11, weight="bold")
    ax.text(0.5, 0.07, "No training / fine-tuning at any stage; the same pipeline is run for each frozen backbone; species recovered by unsupervised clustering",
            ha="center", fontsize=7.5, style="italic", color="#555")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(FIGS / "fig1_pipeline.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print("wrote fig1_pipeline.png")


def fig2_multi():
    U = {k: json.load(open(GEN / f"unsup_robust_multi_{k}.json")) for k, _ in DS}
    S = {k: json.load(open(GEN / f"source_control_multi_{k}.json")) for k, _ in DS}
    order = ["CLIP", "DINOv2", "ImageNet", "geo", "random"]
    glob = {k: {"CLIP": U[k]["CLIP_matched_11d"]["NMI"]["mean"], "DINOv2": U[k]["DINOv2_matched_11d"]["NMI"]["mean"],
                "ImageNet": U[k]["ImageNet_matched_11d"]["NMI"]["mean"], "geo": U[k]["geo_11d"]["NMI"]["mean"],
                "random": U[k]["random_matched_11d"]["NMI"]["mean"]} for k, _ in DS}
    ws = {k: {"CLIP": S[k]["CLIP"]["within_source_NMI_mean"], "DINOv2": S[k]["DINOv2"]["within_source_NMI_mean"],
              "ImageNet": S[k]["ImageNet"]["within_source_NMI_mean"], "geo": S[k]["within_source_NMI_geo_11d_mean"],
              "random": S[k]["random"]["within_source_NMI_mean"]} for k, _ in DS}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(len(DS)); w = 0.16
    for ax, data, title in [(axes[0], glob, "Global (matched 11-d)"), (axes[1], ws, "Within-source (matched 11-d)")]:
        for i, name in enumerate(order):
            vals = [data[k][name] for k, _ in DS]
            ax.bar(x + (i - 2) * w, vals, w, label=name if name != "geo" else "geometry", color=SEMANTIC[name],
                   hatch="//" if name == "random" else None)
        ax.axhline(0.003, ls=":", color="gray", lw=1)
        ax.set_xticks(x); ax.set_xticklabels([n for _, n in DS]); ax.set_ylabel("NMI (species alignment)")
        ax.set_title(title); ax.set_ylim(0, 0.42)
    axes[0].legend(fontsize=8, loc="upper right", ncol=2)
    fig.suptitle("Unsupervised NMI by frozen backbone vs hand-crafted geometry (random-init = causal control)", fontsize=11)
    fig.tight_layout(); fig.savefig(FIGS / "fig2_nmi_bars.png", dpi=300); plt.close(fig)
    print("wrote fig2_nmi_bars.png")


def fig3_multi():
    # Distinguish backbone by COLOUR (CLIP blue / DINOv2 green) and dataset by
    # LINE STYLE (solid = FOR, dashed = BioDiv) — NOT a 4-colour mix.
    style = {"for": ("-", "o"), "biodiv": ("--", "s")}
    fig, ax = plt.subplots(figsize=(COL["single"], COL["single"] * 0.78))
    grefs = []
    for k, label in DS:
        ls, mk = style[k]
        clip = json.load(open(GEN / f"dim_sweep_{k}.json"))["CLIP_pca_dim_sweep_NMI"]
        dino = json.load(open(GEN / f"dim_sweep_dino_{k}.json"))["DINOv2_pca_dim_sweep_NMI"]
        grefs.append(json.load(open(GEN / f"dim_sweep_{k}.json"))["geo_11d_NMI_reference"]["mean"])
        cd = sorted(int(x) for x in clip)
        ax.plot(cd, [clip[str(x)]["mean"] for x in cd], ls=ls, marker=mk, color=SEMANTIC["CLIP"],
                label=f"CLIP, {label}")
        dd = sorted(int(x) for x in dino)
        ax.plot(dd, [dino[str(x)]["mean"] for x in dd], ls=ls, marker=mk, color=SEMANTIC["DINOv2"],
                label=f"DINOv2, {label}")
    # Geometry reference line, labelled on-axis.
    gref = float(np.mean(grefs))
    ax.axhline(gref, ls=":", lw=0.9, color=SEMANTIC["geo"], zorder=1)
    ax.text(0.985, gref + 0.006, "geometry (11-d)", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=6.5, color=SEMANTIC["geo"])
    ax.set_xscale("log", base=2)
    ticks = [4, 16, 64, 256, 512]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlabel("PCA dimensionality")
    ax.set_ylabel("NMI (species alignment)")
    ax.set_ylim(0.08, 0.40)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=6.5,
              columnspacing=1.0, handlelength=1.8, borderaxespad=0.0)
    savefig(fig, "runs/protocol_v2/figs/fig3_dim_sweep")
    plt.close(fig)
    print("wrote fig3_dim_sweep.{pdf,png}")


def fig4_multi():
    algos = ["kmeans", "gmm", "spectral", "agglo"]
    algo_labels = ["K-means", "GMM", "Spectral", "Agglom."]
    fig, axes = plt.subplots(1, 2, figsize=(COL["double"], COL["double"] * 0.40))
    x = np.arange(len(algos)); w = 0.25
    series = [
        ("CLIP_11d", SEMANTIC["CLIP"], "CLIP (11-d)"),
        ("DINOv2_11d", SEMANTIC["DINOv2"], "DINOv2 (11-d)"),
        ("geo_11d", SEMANTIC["geo"], "geometry (11-d)"),
    ]
    handles = None
    for ax, (k, label) in zip(axes, DS):
        clip = json.load(open(GEN / f"clustering_methods_{k}.json"))["fix_K_methods"]
        dino = json.load(open(GEN / f"clustering_methods_dino_{k}.json"))["fix_K_methods"]
        src = {"CLIP_11d": clip, "DINOv2_11d": dino, "geo_11d": clip}
        for i, (key, color, leg) in enumerate(series):
            d = src[key]
            means = [d[a][key]["NMI"]["mean"] for a in algos]
            stds = [d[a][key]["NMI"]["std"] for a in algos]
            bars = ax.bar(x + (i - 1) * w, means, w, yerr=stds, color=color, label=leg,
                          error_kw=dict(elinewidth=0.6, capthick=0.6), capsize=1.5)
            ax.bar_label(bars, labels=[f"{m:.3f}" for m in means], padding=1.5, fontsize=5.0)
        ax.set_xticks(x); ax.set_xticklabels(algo_labels)
        ax.set_title(label); ax.set_ylim(0, 0.42)
        ax.set_ylabel("NMI (species alignment)")
        if handles is None:
            handles, _ = ax.get_legend_handles_labels()
    fig.legend(handles, [s[2] for s in series], loc="lower center",
               bbox_to_anchor=(0.5, 0.99), ncol=3, fontsize=7, columnspacing=1.4)
    fig.tight_layout()
    savefig(fig, "runs/protocol_v2/figs/fig4_clustering_methods")
    plt.close(fig)
    print("wrote fig4_clustering_methods.{pdf,png}")


if __name__ == "__main__":
    set_pub_style()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("fig1", "all"): fig1_pipeline()
    if which in ("fig2", "all"): fig2_multi()
    if which in ("fig3", "all"): fig3_multi()
    if which in ("fig4", "all"): fig4_multi()
