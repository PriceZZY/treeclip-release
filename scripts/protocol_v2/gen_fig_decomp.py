"""Decomposition figure (PFG round-1, R1 comment 6: 'a more intuitive figure or table that
separates the global advantage, the source-related component, and the remaining
within-source advantage').

For each dataset x baseline (whole-tree single-scale; local multi-scale Weinmann), three
bars: the global CLIP-geometry NMI gap, its source-related component (global minus
within-source), and the within-source residual. All values are read from the result JSONs
(matched 11-d); nothing is hardcoded. Global NMI values are 15-seed means; the within-source values
of the local multi-scale comparison (local_weinmann.json) come from one clustering run (seed 0,
n_init=10), so that comparison's source component mixes the two. Note the source component is NEGATIVE
for the local multi-scale baseline on FOR-species20K (about -0.044): there that descriptor gains
MORE from source structure than CLIP does, so controlling for source widens the margin. On
BioDiv-3DTrees-ULS the same component is positive (about +0.055).
Output: runs/protocol_v2/figs/fig_decomp.pdf/.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pubfig_style import set_pub_style, COL, savefig

ROOT = Path(__file__).resolve().parent.parent.parent
GEN = ROOT / "runs/protocol_v2/gen"
FIGS = ROOT / "runs/protocol_v2/figs"


def main():
    set_pub_style()
    multi = {w: json.load(open(GEN / f"unsup_robust_multi_{w}.json")) for w in ["for", "biodiv"]}
    sc = json.load(open(GEN / "source_control_robust.json"))
    lw = json.load(open(GEN / "local_weinmann.json"))
    groups = []
    for w, wl in [("for", "FOR-species20K"), ("biodiv", "BioDiv-3DTrees-ULS")]:
        clip_g = multi[w]["CLIP_matched_11d"]["NMI"]["mean"]
        geo_g = multi[w]["geo_11d"]["NMI"]["mean"]
        ws_delta_single = sc[w]["within_source_delta_mean"]
        g_single = clip_g - geo_g
        lw_g = lw[w]["geo_local_multiscale_weinmann"]["NMI_mean"]
        g_local = clip_g - lw_g
        ws_delta_local = lw[w]["clip"]["within_source_NMI"] - lw[w]["geo_local_multiscale_weinmann"]["within_source_NMI"]
        groups.append((f"{wl}\nvs single-scale", g_single, g_single - ws_delta_single, ws_delta_single))
        groups.append((f"{wl}\nvs local multi-scale", g_local, g_local - ws_delta_local, ws_delta_local))
    fig, ax = plt.subplots(figsize=(COL["double"], COL["double"] * 0.42))
    x = np.arange(len(groups)); wdt = 0.26
    cols = ["#4878a8", "#c8a464", "#5a9a68"]
    labs = ["global gap", "source-related component", "within-source residual"]
    for i, key in enumerate([1, 2, 3]):
        vals = [g[key] for g in groups]
        b = ax.bar(x + (i - 1) * wdt, vals, wdt, color=cols[i], label=labs[i])
        for r, v in zip(b, vals):
            ax.annotate(f"{v:+.3f}", (r.get_x() + r.get_width() / 2, v),
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=6.4,
                        xytext=(0, 1.5 if v >= 0 else -1.5), textcoords="offset points")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=7)
    ax.set_ylabel("CLIP $-$ geometry NMI gap")
    ax.legend(fontsize=6.8, loc="upper right", frameon=False)
    ax.set_ylim(min(-0.06, min(g[2] for g in groups) - 0.02), max(g[1] for g in groups) + 0.045)
    FIGS.mkdir(parents=True, exist_ok=True)
    savefig(fig, str(FIGS / "fig_decomp"))
    plt.close(fig)
    for g in groups:
        print(f"{g[0].replace(chr(10),' | ')}: global {g[1]:+.4f} = source {g[2]:+.4f} + within {g[3]:+.4f}")
    print("wrote fig_decomp.pdf/.png")


if __name__ == "__main__":
    main()
