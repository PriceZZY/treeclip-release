"""Manuscript Fig 1 — frozen vision-foundation pipeline, built from REAL assets.

Redraw per Lu's R6 #17 target (reports/_lu_pipeline_sketch_r6.png): a real
individual-tree LiDAR point cloud flows left-to-right through the label-free
pipeline, with tensor shapes on the arrows and a frozen-encoder snowflake. Adapts
the poster hero (poster/build/figscripts/gen_hero.py) into Elsevier CAS style
(serif, colourblind-safe, double-column) instead of the poster purple scheme.

Stages: (a) tree point cloud  ->  (b) 6 PCA depth views (6x224^2)  ->
(c) frozen image encoder (6x512)  ->  (d) L2-norm + mean fusion (1x512)  ->
(e) unsupervised KMeans (k = species count).

Real data: Pinus sylvestris treeID 2573 (poster/assets/hero_tree_2573.npz +
hero_depthview_2573_{0..5}.png). No fabricated geometry. Vector PDF + PNG.

Interpreter: the project's virtualenv Python; run from the repo root.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pubfig_style import COL, DISCRETE15, SEMANTIC, savefig, set_pub_style

ASSETS = r"poster/assets"  # poster assets are not shipped in this repository
HERO_NPZ = os.path.join(ASSETS, "hero_tree_2573.npz")
DEPTHVIEW = os.path.join(ASSETS, "hero_depthview_2573_{i}.png")
OUT = "runs/protocol_v2/figs/fig1_pipeline"

INK = "#1A1A1A"
BORDER = "#555555"
BOXFILL = "#EEF3F8"      # very light blue-grey card body
BARFILL = "#DCE6F0"
FROZEN = SEMANTIC["CLIP"]   # CLIP blue #0072B2 for the snowflake / encoder accent
ARROW = "#444444"
TAG = "#666666"          # tensor-shape labels


# --------------------------------------------------------------------------- #
# asset loaders
# --------------------------------------------------------------------------- #
def load_tree_xyz(max_pts: int = 22000) -> np.ndarray:
    d = np.load(HERO_NPZ)
    pts = np.asarray(d["points"], dtype=np.float32)
    if pts.shape[0] > max_pts:
        rng = np.random.default_rng(2573)
        pts = pts[rng.choice(pts.shape[0], size=max_pts, replace=False)]
    return pts


def load_depth_view(i: int) -> np.ndarray:
    """Load one real PCA depth view and trim its near-white margin (keep the
    natural viridis depth colouring)."""
    arr = np.asarray(Image.open(DEPTHVIEW.format(i=i)).convert("RGB"))
    gray = arr.mean(axis=2)
    mr = np.where((gray < 250).any(axis=1))[0]
    mc = np.where((gray < 250).any(axis=0))[0]
    if mr.size and mc.size:
        arr = arr[mr[0]:mr[-1] + 1, mc[0]:mc[-1] + 1]
    return arr


def snowflake(ax, cx: float, cy: float, r: float, color: str, lw: float = 1.1):
    """6-fold snowflake (frozen marker) in axes-fraction coords."""
    for k in range(6):
        ang = np.deg2rad(60 * k)
        dx, dy = r * np.cos(ang), r * np.sin(ang)
        ax.plot([cx, cx + dx], [cy, cy + dy], color=color, lw=lw,
                solid_capstyle="round", zorder=20, transform=ax.transAxes)
        for s in (0.55, 0.82):
            bx, by = cx + s * dx, cy + s * dy
            for bsign in (-1, 1):
                ba = ang + bsign * np.deg2rad(35)
                ax.plot([bx, bx + 0.30 * r * np.cos(ba)],
                        [by, by + 0.30 * r * np.sin(ba)],
                        color=color, lw=lw * 0.7, solid_capstyle="round",
                        zorder=20, transform=ax.transAxes)


# --------------------------------------------------------------------------- #
# stage renderers
# --------------------------------------------------------------------------- #
def stage1_cloud(ax, pts: np.ndarray):
    """(a) real tree point cloud, height-coloured side projection (viridis)."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    h_axis = x if (x.max() - x.min()) >= (y.max() - y.min()) else y
    ax.scatter(h_axis, z, c=z, cmap="viridis", s=0.5, marker=".",
               linewidths=0, rasterized=True)
    ax.set_axis_off()
    ax.set_aspect("equal")
    hx0, hx1 = np.percentile(h_axis, 0.5), np.percentile(h_axis, 99.5)
    pad = (hx1 - hx0) * 0.07
    ax.set_xlim(hx0 - pad, hx1 + pad)
    ax.set_ylim(z.min() - pad, z.max() + pad)


def stage2_depth(ax):
    """(b) real 6 PCA depth views composited into a clean 3x2 tile block."""
    ax.set_axis_off()
    tiles = [load_depth_view(i) for i in range(6)]
    h = min(t.shape[0] for t in tiles)
    w = min(t.shape[1] for t in tiles)
    tiles = [t[:h, :w] for t in tiles]
    gap = max(2, w // 20)
    cols, rows = 3, 2
    cw = cols * w + (cols - 1) * gap
    ch = rows * h + (rows - 1) * gap
    canvas = np.full((ch, cw, 3), 255, dtype=np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        canvas[r * (h + gap):r * (h + gap) + h, c * (w + gap):c * (w + gap) + w] = t
    ax.imshow(canvas, aspect="equal", interpolation="nearest")
    ax.add_patch(Rectangle((0, 0), cw - 1, ch - 1, fill=False,
                           edgecolor=BORDER, lw=0.6, zorder=5))


def stage3_frozen(ax):
    """(c) frozen image encoder schematic with a snowflake frozen badge."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    box = FancyBboxPatch(
        (0.06, 0.20), 0.88, 0.62,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0, edgecolor=BORDER, facecolor=BOXFILL, zorder=4,
        mutation_aspect=0.95)
    ax.add_patch(box)
    ax.text(0.50, 0.70, "Frozen image\nencoder", ha="center", va="center",
            fontsize=8, fontweight="bold", color=INK, zorder=8, linespacing=1.05)
    for yy in (0.34, 0.41, 0.48):
        ax.add_patch(FancyBboxPatch(
            (0.24, yy), 0.52, 0.040,
            boxstyle="round,pad=0.004,rounding_size=0.02",
            linewidth=0.5, edgecolor=BORDER, facecolor="white", zorder=5))
    ax.text(0.50, 0.255, "ViT-B/16", ha="center", va="center",
            fontsize=7, fontweight="bold", color=FROZEN, zorder=8)
    # snowflake "frozen" badge, top-right on the box border
    bcx, bcy = 0.86, 0.80
    ax.add_patch(Circle((bcx, bcy), 0.075, facecolor="white", edgecolor=FROZEN,
                        lw=0.9, zorder=18))
    snowflake(ax, bcx, bcy, 0.050, FROZEN, lw=1.0)


def stage4_fuse(ax):
    """(d) L2-norm + mean of 6 view embeddings into one tree embedding."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    vx, n, top, bot = 0.06, 6, 0.86, 0.18
    ys = np.linspace(top, bot, n)
    bar_h = (top - bot) / n * 0.60
    rng = np.random.default_rng(7)
    for yy in ys:
        cells = 5
        cw = 0.16 / cells
        shades = rng.uniform(0.4, 1.0, cells)
        for ci in range(cells):
            ax.add_patch(Rectangle((vx + ci * cw, yy - bar_h / 2), cw * 0.9, bar_h,
                                   facecolor=plt.cm.Blues(0.35 + 0.5 * shades[ci]),
                                   edgecolor="white", lw=0.3, zorder=6))
    ax.text(vx + 0.08, top + 0.10, "6 views", ha="center", va="bottom",
            fontsize=6.5, color=TAG, zorder=8)
    ox, cy = 0.52, (top + bot) / 2
    ax.add_patch(Circle((ox, cy), 0.075, facecolor="white", edgecolor=BORDER,
                        lw=0.9, zorder=7))
    ax.text(ox, cy, r"$\oplus$", ha="center", va="center", fontsize=12,
            color=INK, zorder=8)
    ax.text(ox, cy - 0.16, "L2-norm\n+ mean", ha="center", va="top",
            fontsize=7, color=INK, fontweight="bold", zorder=8, linespacing=1.0)
    for yy in ys:
        ax.add_patch(FancyArrowPatch((vx + 0.17, yy), (ox - 0.08, cy),
                                     arrowstyle="-", lw=0.5, color=BORDER,
                                     alpha=0.6, zorder=5))
    rx = 0.80
    cells = 5
    cw = 0.17 / cells
    out_h = bar_h * 1.8
    for ci in range(cells):
        ax.add_patch(Rectangle((rx + ci * cw, cy - out_h / 2), cw * 0.9, out_h,
                               facecolor=plt.cm.Blues(0.55 + 0.4 * (ci / cells)),
                               edgecolor="white", lw=0.4, zorder=6))
    ax.add_patch(FancyArrowPatch((ox + 0.08, cy), (rx - 0.015, cy),
                                 arrowstyle="-|>", mutation_scale=8,
                                 lw=1.0, color=ARROW, zorder=6))
    ax.text(rx + 0.085, cy + out_h / 2 + 0.10, "1 tree\nembedding", ha="center",
            va="bottom", fontsize=6.5, color=TAG, linespacing=1.0, zorder=8)


def stage5_cluster(ax):
    """(e) unsupervised KMeans clusters (schematic CB-safe blobs)."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rng = np.random.default_rng(11)
    centers = [(0.30, 0.66), (0.70, 0.72), (0.28, 0.30), (0.68, 0.30), (0.50, 0.50)]
    for ci, (cx, cy) in enumerate(centers):
        col = DISCRETE15[ci]
        p = rng.normal((cx, cy), 0.065, size=(26, 2))
        ax.scatter(p[:, 0], p[:, 1], s=4, c=col, marker="o", linewidths=0,
                   alpha=0.9, zorder=6)
    ax.text(0.5, 0.04, "k = species count", ha="center", va="bottom",
            fontsize=6.5, color=TAG, zorder=8)


# --------------------------------------------------------------------------- #
# compose
# --------------------------------------------------------------------------- #
def build():
    set_pub_style()
    pts = load_tree_xyz()

    fig = plt.figure(figsize=(COL["double"], COL["double"] * 0.40))
    fig.patch.set_facecolor("white")

    band_bottom, band_top = 0.20, 0.78
    band_h = band_top - band_bottom
    band_mid = (band_bottom + band_top) / 2

    left, right, gap = 0.012, 0.012, 0.052
    rel = np.array([0.92, 1.20, 0.98, 1.16, 1.00])
    avail = 1 - left - right - 4 * gap
    widths = rel / rel.sum() * avail
    xs, cur = [], left
    for w in widths:
        xs.append(cur)
        cur += w + gap
    centers = [xs[i] + widths[i] / 2 for i in range(5)]

    ax1 = fig.add_axes([xs[0], band_bottom, widths[0], band_h])
    stage1_cloud(ax1, pts)
    ax2 = fig.add_axes([xs[1], band_bottom + band_h * 0.12, widths[1], band_h * 0.76])
    stage2_depth(ax2)
    ax3 = fig.add_axes([xs[2], band_bottom, widths[2], band_h])
    stage3_frozen(ax3)
    ax4 = fig.add_axes([xs[3], band_bottom + band_h * 0.04, widths[3], band_h * 0.92])
    stage4_fuse(ax4)
    ax5 = fig.add_axes([xs[4], band_bottom, widths[4], band_h])
    stage5_cluster(ax5)

    # stage labels (a)-(e) + short text above each stage
    labels = [
        ("(a)", "Tree point cloud"),
        ("(b)", "6 depth views"),
        ("(c)", "Frozen encoder"),
        ("(d)", "L2-norm + mean"),
        ("(e)", "KMeans clusters"),
    ]
    for cx, (tag, txt) in zip(centers, labels):
        fig.text(cx, band_top + 0.135, tag, ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color=INK)
        fig.text(cx, band_top + 0.045, txt, ha="center", va="bottom",
                 fontsize=7.5, color=INK)

    # tensor shapes under each stage (Lu's sketch)
    shapes = [r"$N\times3$", r"$6\times224^2$", r"$6\times512$",
              r"$1\times512$", r"$k$ clusters"]
    for cx, sh in zip(centers, shapes):
        fig.text(cx, band_bottom - 0.085, sh, ha="center", va="top",
                 fontsize=7, color=TAG, family="serif")

    # flow arrows between stages
    for i in range(4):
        x0 = xs[i] + widths[i] + 0.006
        x1 = xs[i + 1] - 0.006
        fig.add_artist(FancyArrowPatch(
            (x0, band_mid), (x1, band_mid), transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=11, lw=1.3, color=ARROW,
            shrinkA=0, shrinkB=0, zorder=15, joinstyle="round"))

    savefig(fig, OUT)
    plt.close(fig)
    print(f"wrote {OUT}.pdf and {OUT}.png")


if __name__ == "__main__":
    build()
