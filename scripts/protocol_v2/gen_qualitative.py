"""Fig 5 — qualitative cluster structure (the gap vs the 3DCTN template), CLIP-11d.

SPLIT into two double-column figures for legibility:
  Fig 5a (fig5a_tsne):     2 rows (FOR, BioDiv) x 2 cols [t-SNE by species | by source],
                           with a species legend (colour -> abbreviated binomial code) under
                           the by-species column and a source legend under the by-source column.
  Fig 5b (fig5b_confusion): 2 cols (FOR, BioDiv) Hungarian row-normalised confusion matrices
                           sharing ONE viridis colorbar (vmin=0,vmax=1); axes labelled with
                           species CODES (not integers), x-labels rotated 45 deg.

sklearn t-SNE (no extra deps). FOR subsampled to 3000 for legibility. RNG seeds fixed.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pubfig_style import set_pub_style, COL, savefig, species_code, discrete_palette

ROOT = Path(__file__).resolve().parent.parent.parent
GEN = ROOT / "runs/protocol_v2/gen"
FIGS = ROOT / "runs/protocol_v2/figs"
VIZ_N = 3000
DATASETS = [("for", "FOR-species20K"), ("biodiv", "BioDiv-3DTrees (ULS)")]


def analyzable_mask(y, groups, min_n=30, min_blocks=2):
    keep = [s for s in set(y) if (y == s).sum() >= min_n and len(set(groups[y == s])) >= min_blocks]
    return np.isin(y, keep)


def load(which):
    if which == "for":
        d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
        Xc = d["X"]; y = d["species"].astype(str); g = d["dataset"].astype(str); fagus = None
    else:
        d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
        Xc = d["X_clip"]; y = d["species"].astype(str); g = d["region"].astype(str); fagus = 0.39
    m = analyzable_mask(y, g); Xc, y, g = Xc[m], y[m], g[m]
    if fagus is not None:
        c = Counter(y); top = max(c, key=c.get); n = len(y)
        if c[top] / n > fagus:
            tgt = int(fagus / (1 - fagus) * (n - c[top])); idx = np.where(y == top)[0]
            drop = set(np.random.RandomState(0).choice(idx, c[top] - tgt, replace=False))
            k = np.array([i not in drop for i in range(n)]); Xc, y, g = Xc[k], y[k], g[k]
    return Xc, y, g


def hungarian_pred(X11, y, K, seed=0):
    cl = KMeans(K, n_init=10, random_state=seed).fit_predict(X11)
    sp = sorted(set(y)); si = {s: i for i, s in enumerate(sp)}
    M = np.zeros((K, len(sp)))
    for c, s in zip(cl, y):
        M[c, si[s]] += 1
    row, col = linear_sum_assignment(-M)
    cmap = {int(r): sp[int(c)] for r, c in zip(row, col)}
    return np.array([cmap.get(int(c), sp[0]) for c in cl]), sp


def compute(which, loader=load):
    """Heavy lifting (PCA, t-SNE, KMeans+Hungarian) — shared by both figures."""
    Xc, y, g = loader(which)
    K = len(set(y))
    X11 = PCA(11, random_state=0).fit_transform(StandardScaler().fit_transform(Xc))
    if len(y) > VIZ_N:
        idx = np.random.RandomState(0).choice(len(y), VIZ_N, replace=False)
    else:
        idx = np.arange(len(y))
    emb = TSNE(n_components=2, init="pca", perplexity=30, random_state=0).fit_transform(X11[idx])
    yv, gv = y[idx], g[idx]
    pred, sp = hungarian_pred(X11, y, K)
    cm = confusion_matrix(y, pred, labels=sp).astype(float)
    cm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    print(f"{which}: n={len(y)} K={K} viz_n={len(idx)} diag_mean={np.diag(cm).mean():.3f}", flush=True)
    return dict(which=which, K=K, emb=emb, yv=yv, gv=gv, cm=cm, sp=sp)


def fig_tsne(results, stem, backbone="CLIP"):
    """Fig 5a: t-SNE by species | by source, 2 datasets, with per-column legends."""
    fig, axes = plt.subplots(2, 2, figsize=(COL["double"], COL["double"] * 0.55))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.18, wspace=0.06, hspace=0.22)
    # ONE global code -> colour map so the same species is the same colour in both
    # rows and the shared legend is truthful (union of codes, deterministic order).
    all_codes = sorted({species_code(s) for which, _ in DATASETS
                        for s in set(results[which]["yv"])})
    gpal = discrete_palette(len(all_codes))
    code_colour = {c: gpal[i] for i, c in enumerate(all_codes)}
    sp_handles = {c: plt.Line2D([], [], marker="o", linestyle="", markersize=4,
                                color=code_colour[c]) for c in all_codes}
    src_handles = {}  # source label -> handle
    for r, (which, title) in enumerate(DATASETS):
        R = results[which]
        emb, yv, gv = R["emb"], R["yv"], R["gv"]
        sp_sorted = sorted(set(yv))
        for s in sp_sorted:
            mm = yv == s
            axes[r, 0].scatter(emb[mm, 0], emb[mm, 1], s=8, alpha=0.65,
                               color=code_colour[species_code(s)], linewidths=0,
                               rasterized=True)
        axes[r, 0].set_title(f"{title}: {backbone} t-SNE by species (K={R['K']})")
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])

        src_sorted = sorted(set(gv))
        spal = discrete_palette(len(src_sorted))
        for i, s in enumerate(src_sorted):
            mm = gv == s
            axes[r, 1].scatter(emb[mm, 0], emb[mm, 1], s=8, alpha=0.65,
                               color=spal[i], linewidths=0, rasterized=True)
            if s not in src_handles:
                src_handles[s] = plt.Line2D([], [], marker="o", linestyle="",
                                            markersize=4, color=spal[i])
        axes[r, 1].set_title(f"{title}: same t-SNE by source ({len(src_sorted)} sources)")
        axes[r, 1].set_xticks([]); axes[r, 1].set_yticks([])

    # species legend under the by-species column. 2026-08-31: legend TITLES now spell out
    # that these are unions across the two panel rows (15 FOR + 6 BioDiv, 5 shared = 16;
    # 21 FOR sources + 3 BioDiv regions = 24) so the counts cannot be misread against the
    # per-panel K=15 / K=6 titles (PFG round-1: a reader miscounted exactly this way).
    n_for_sp = len(set(results["for"]["yv"])); n_bio_sp = len(set(results["biodiv"]["yv"]))
    n_for_src = len(set(results["for"]["gv"])); n_bio_src = len(set(results["biodiv"]["gv"]))
    sp_codes = sorted(sp_handles)
    leg1 = fig.legend([sp_handles[c] for c in sp_codes], sp_codes,
               loc="lower left", bbox_to_anchor=(0.03, 0.0), ncol=6,
               handletextpad=0.3, columnspacing=0.8,
               fontsize=6.2, borderaxespad=0.0,
               title=f"species codes, union of both rows ({n_for_sp} FOR $\cup$ {n_bio_sp} BioDiv = {len(sp_codes)})")
    leg1.get_title().set_fontsize(6.4)
    # source legend under the by-source column
    src_lab = sorted(src_handles)
    leg2 = fig.legend([src_handles[s] for s in src_lab], src_lab,
               loc="lower right", bbox_to_anchor=(0.97, 0.0), ncol=4,
               handletextpad=0.3, columnspacing=0.8,
               fontsize=6.2, borderaxespad=0.0,
               title=f"{n_for_src} FOR sources + {n_bio_src} BioDiv regions = {len(src_lab)} entries")
    leg2.get_title().set_fontsize(6.4)
    FIGS.mkdir(parents=True, exist_ok=True)
    savefig(fig, str((FIGS / stem)))
    plt.close(fig)
    print(f"wrote {stem}.pdf/.png", flush=True)


def fig_confusion(results, stem):
    """Fig 5b: 2 Hungarian row-normalised confusion matrices, shared viridis colorbar."""
    fig, axes = plt.subplots(1, 2, figsize=(COL["double"], COL["double"] * 0.50))
    fig.subplots_adjust(left=0.07, right=0.90, top=0.90, bottom=0.18, wspace=0.30)
    im = None
    cmax = max(float(results[w]["cm"].max()) for w, _ in DATASETS)
    vmax = min(1.0, float(np.ceil(cmax * 10) / 10))  # round data max up to 0.1: no clipping, better contrast
    for c, (which, title) in enumerate(DATASETS):
        R = results[which]
        cm, sp = R["cm"], R["sp"]
        codes = [species_code(s) for s in sp]
        im = axes[c].imshow(cm, cmap="viridis", vmin=0, vmax=vmax, rasterized=True)
        axes[c].set_title(f"{title}: Hungarian confusion (row-norm)")
        axes[c].set_xlabel("predicted cluster (mapped species)")
        if c == 0:
            axes[c].set_ylabel("true species")
        axes[c].set_xticks(range(len(sp))); axes[c].set_yticks(range(len(sp)))
        axes[c].set_xticklabels(codes, fontsize=6, rotation=45, ha="right")
        axes[c].set_yticklabels(codes, fontsize=6)
    cax = fig.add_axes([0.915, 0.18, 0.018, 0.72])
    fig.colorbar(im, cax=cax, label="row-normalized fraction")
    FIGS.mkdir(parents=True, exist_ok=True)
    savefig(fig, str((FIGS / stem)))
    plt.close(fig)
    print(f"wrote {stem}.pdf/.png", flush=True)


def main():
    set_pub_style()
    results = {which: compute(which) for which, _ in DATASETS}
    fig_tsne(results, "fig5a_tsne", backbone="CLIP")
    fig_confusion(results, "fig5b_confusion")


if __name__ == "__main__":
    main()
