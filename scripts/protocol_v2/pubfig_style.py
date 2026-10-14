"""Shared publication style for TreeCLIP CAS figures.

Elsevier CAS double-column. Serif fonts (Times/STIX, to match Lu's 3DCTN look)
+ Okabe-Ito colorblind-safe palette for data plots (Elsevier requires CB-safe).
Vector PDF primary output; fonts sized for the FINAL printed width, never
rescaled in LaTeX.

Usage from a generator (run from repo root):
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pubfig_style import set_pub_style, COL, SEMANTIC, LABEL, savefig
    set_pub_style()
    fig, ax = plt.subplots(figsize=(COL["double"], COL["double"] * 0.42))
    ...
    savefig(fig, "runs/protocol_v2/figs/fig2_nmi_bars")   # writes .pdf + .png
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

MM = 1 / 25.4
# Elsevier CAS column widths (inches): single=90mm, 1.5col=140mm, double=190mm
COL = {"single": 90 * MM, "onehalf": 140 * MM, "double": 190 * MM}

# Okabe-Ito (Wong 2011) colorblind-safe palette, with FIXED semantic roles so a
# colour means the same method in every figure.
SEMANTIC = {
    "CLIP": "#0072B2",      # blue
    "DINOv2": "#009E73",    # green
    "ImageNet": "#E69F00",  # orange
    "geo": "#555555",       # dark grey  (traditional hand-crafted baseline)
    "random": "#BBBBBB",    # light grey (random-init causal control)
}
LABEL = {
    "CLIP": "CLIP", "DINOv2": "DINOv2", "ImageNet": "sup-ImageNet",
    "geo": "geometry", "random": "random-init",
}


def set_pub_style() -> None:
    """Apply the shared rcParams. Safe to call once per generator."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42,            # embed as TrueType
        "axes.prop_cycle": mpl.cycler(color=list(SEMANTIC.values())),
        "image.cmap": "viridis",
        "axes.linewidth": 0.6, "lines.linewidth": 0.9, "lines.markersize": 3,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.grid": False, "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.dpi": 150,
    })


def savefig(fig, stem: str) -> None:
    """Save both vector PDF (primary) and a 300-dpi PNG preview for one figure.

    `stem` is a path without extension, e.g. 'runs/protocol_v2/figs/fig2_nmi_bars'.
    """
    fig.savefig(f"{stem}.pdf", dpi=600)
    fig.savefig(f"{stem}.png", dpi=400)


def species_code(latin: str) -> str:
    """Abbreviate a 'Genus_species' Latin name to a 'Gen.spe' binomial code.

    e.g. 'Fagus_sylvatica' -> 'Fag.syl', 'Picea_abies' -> 'Pic.abi'.
    Handles 'Genus_spec' (unresolved) -> 'Gen.spec'.
    """
    parts = str(latin).replace(" ", "_").split("_")
    genus = parts[0][:3].capitalize() if parts else str(latin)[:3]
    sp = parts[1][:3].lower() if len(parts) > 1 and parts[1] else "sp"
    return f"{genus}.{sp}"


# Perceptually-distinct, CB-aware discrete palette for K<=15 categorical classes.
# Built from Okabe-Ito + Paul Tol 'muted' so adjacent species stay distinguishable
# for the common colour-vision deficiencies. Ordered for max separation.
DISCRETE15 = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink/magenta
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#332288",  # indigo
    "#117733",  # dark green
    "#882255",  # wine
    "#88CCEE",  # pale cyan
    "#AA4499",  # purple
    "#999933",  # olive
    "#661100",  # maroon
    "#44AA99",  # teal
    "#DDCC77",  # sand (Tol muted) — 16th, for the FOR+BioDiv species union
]


def discrete_palette(k: int):
    """Return k perceptually-distinct CB-aware colours (k <= len(DISCRETE15))."""
    if k <= len(DISCRETE15):
        return DISCRETE15[:k]
    # graceful fallback for the unlikely K>15 case
    import matplotlib.cm as cm
    import numpy as _np
    return [cm.tab20(i % 20) for i in range(k)]
