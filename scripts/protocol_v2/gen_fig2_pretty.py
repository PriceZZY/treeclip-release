"""Publication-grade remake of Fig 2 (unsupervised NMI by frozen backbone vs geometry).

Reads the SAME JSONs as gen_figures_merge.py:fig2_multi (so every number is identical
and traceable) and restyles via the shared CAS style (pubfig_style): serif fonts,
Okabe-Ito colorblind-safe palette with FIXED semantic colours per method, value labels
only on the coloured learned bars, separated grey baselines (geo solid dark vs random
hatched light), and an on-axis annotation of the ~0.003 random-clustering reference line.
No baked title (the LaTeX caption does that). Writes the canonical stem (PDF + 300-dpi PNG).

Usage: python gen_fig2_pretty.py
"""
from __future__ import annotations
import os
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pubfig_style import set_pub_style, COL, SEMANTIC, LABEL, savefig

ROOT = Path(__file__).resolve().parent.parent.parent
GEN = ROOT / "runs/protocol_v2/gen"
FIGS = ROOT / "runs/protocol_v2/figs"
DS = [("for", "FOR-species20K"), ("biodiv", "BioDiv-3DTrees (ULS)")]
ORDER = ["CLIP", "DINOv2", "ImageNet", "geo", "random"]
LEARNED = {"CLIP", "DINOv2", "ImageNet"}  # only these get value labels

# value-label font (pt)
LABEL_FONT = 7
YMAX = 0.42


def load():
    U = {k: json.load(open(GEN / f"unsup_robust_multi_{k}.json")) for k, _ in DS}
    S = {k: json.load(open(GEN / f"source_control_multi_{k}.json")) for k, _ in DS}
    glob = {k: {"CLIP": U[k]["CLIP_matched_11d"]["NMI"]["mean"],
                "DINOv2": U[k]["DINOv2_matched_11d"]["NMI"]["mean"],
                "ImageNet": U[k]["ImageNet_matched_11d"]["NMI"]["mean"],
                "geo": U[k]["geo_11d"]["NMI"]["mean"],
                "random": U[k]["random_matched_11d"]["NMI"]["mean"]} for k, _ in DS}
    ws = {k: {"CLIP": S[k]["CLIP"]["within_source_NMI_mean"],
              "DINOv2": S[k]["DINOv2"]["within_source_NMI_mean"],
              "ImageNet": S[k]["ImageNet"]["within_source_NMI_mean"],
              "geo": S[k]["within_source_NMI_geo_11d_mean"],
              "random": S[k]["random"]["within_source_NMI_mean"]} for k, _ in DS}
    return glob, ws


def panel(ax, data, title, baseline, annotate_baseline=False):
    x = np.arange(len(DS))
    w = 0.16
    for i, name in enumerate(ORDER):
        vals = [data[k][name] for k, _ in DS]
        pos = x + (i - 2) * w
        if name == "geo":
            ax.bar(pos, vals, w, color=SEMANTIC["geo"], zorder=3, label=LABEL[name])
        elif name == "random":
            ax.bar(pos, vals, w, color=SEMANTIC["random"], hatch="////",
                   edgecolor="black", linewidth=0.4, zorder=3, label=LABEL[name])
        else:
            ax.bar(pos, vals, w, color=SEMANTIC[name], zorder=3, label=LABEL[name])
        # value labels ONLY on the coloured learned backbones (declutter baselines)
        if name in LEARNED:
            for p, v in zip(pos, vals):
                ax.text(p, v + 0.006, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=LABEL_FONT, color="#222222", zorder=4)
    # faint random-clustering reference line; annotated on-axis when requested
    ax.axhline(baseline, ls=(0, (1, 2.5)), color="#AAAAAA", lw=1.0, zorder=2)
    if annotate_baseline:
        ax.text(x[-1] + 3 * w, baseline + 0.005, "random clustering",
                ha="right", va="bottom", fontsize=6, color="#777777",
                family="serif", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in DS])
    ax.set_ylim(0, YMAX)
    ax.set_title(title, pad=6)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E6E6E6", lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=3)


def main():
    set_pub_style()
    glob, ws = load()
    baseline = 0.003  # paper value (NMI of a random clustering on FOR); ~0.0034 on BioDiv
    fig, axes = plt.subplots(1, 2, figsize=(COL["double"], COL["double"] * 0.42),
                             sharey=True)
    panel(axes[0], glob, "Global (matched 11-d)", baseline)
    panel(axes[1], ws, "Within-source (matched 11-d)", baseline)
    axes[0].set_ylabel("Unsupervised NMI (species alignment)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.05), handlelength=1.3, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    stem = "runs/protocol_v2/figs/fig2_nmi_bars"
    savefig(fig, str(ROOT / stem))
    plt.close(fig)
    print("wrote", ROOT / f"{stem}.pdf")
    print("wrote", ROOT / f"{stem}.png")
    # echo the numbers used, for an integrity glance
    for k, nm in DS:
        print(f"[{nm}] global:", {n: round(glob[k][n], 4) for n in ORDER},
              "| within:", {n: round(ws[k][n], 4) for n in ORDER})


if __name__ == "__main__":
    main()
