"""Diagnostic figures for the two hypotheses:
  H1 (separability): can species be told apart by eye? -> structural montage (all 3),
     RGB montage (NEON), HSI mean-spectrum per species (NEON).
  H2 (data sufficiency): is the CLIP head data-limited? -> learning curve (CLIP+linear
     macro-F1 vs train fraction, under spatial CV).
CPU-only. Saves PNGs to runs/protocol_v2/figs/.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
from collections import Counter
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from treeclip.core.tree_sample import TreeSample

FIGS = ROOT / "runs/protocol_v2/figs"; FIGS.mkdir(parents=True, exist_ok=True)
SITES = {"Weiser": (ROOT / "data/cache_weiser", ["KA09", "KA10", "KA11", "BR01", "BR03", "BR05"]),
         "NEON-OSBS": (ROOT / "data/cache", ["OSBS"]),
         "NEON-MLBS": (ROOT / "data/cache", ["MLBS"])}


def collect(ds, need=("pc",)):
    root, sites = SITES[ds]
    by_sp = {}
    for site in sites:
        for p in sorted((root / site).glob("*.npz")):
            t = TreeSample.load_npz(p)
            if t.point_cloud.shape[0] < 3 or t.species is None:
                continue
            by_sp.setdefault(t.species, []).append(p)
    return by_sp


def top_species(by_sp, k=6):
    return [s for s, _ in sorted(by_sp.items(), key=lambda kv: -len(kv[1]))[:k]]


def structural_montage(ds, nsp=6, nex=5):
    by_sp = collect(ds); sps = top_species(by_sp, nsp)
    fig, ax = plt.subplots(nsp, nex, figsize=(nex * 1.8, nsp * 1.8))
    for r, sp in enumerate(sps):
        paths = by_sp[sp][:nex]
        for c in range(nex):
            a = ax[r, c]; a.axis("off")
            if c < len(paths):
                pc = TreeSample.load_npz(paths[c]).point_cloud
                x = pc[:, 0] - pc[:, 0].mean(); z = pc[:, 2] - pc[:, 2].min()
                a.scatter(x, z, s=1.5, c=z, cmap="viridis")
                a.set_aspect("equal")
            if c == 0:
                a.set_title(f"{sp} (n={len(by_sp[sp])})", fontsize=8, loc="left")
    fig.suptitle(f"{ds}: side-view structure (x vs height) — what CLIP sees [top {nsp} species]", fontsize=11)
    fig.tight_layout(); out = FIGS / f"structure_{ds}.png"; fig.savefig(out, dpi=110); plt.close(fig)
    print("wrote", out.name)


def rgb_montage(ds, nsp=6, nex=5):
    by_sp = collect(ds); sps = top_species(by_sp, nsp)
    fig, ax = plt.subplots(nsp, nex, figsize=(nex * 1.8, nsp * 1.8))
    for r, sp in enumerate(sps):
        paths = by_sp[sp][:nex]
        for c in range(nex):
            a = ax[r, c]; a.axis("off")
            if c < len(paths):
                rgb = TreeSample.load_npz(paths[c]).rgb_crop
                if rgb.size > 0:
                    a.imshow(rgb)
            if c == 0:
                a.set_title(f"{sp} (n={len(by_sp[sp])})", fontsize=8, loc="left")
    fig.suptitle(f"{ds}: RGB crowns — human-eye separability [top {nsp} species]", fontsize=11)
    fig.tight_layout(); out = FIGS / f"rgb_{ds}.png"; fig.savefig(out, dpi=110); plt.close(fig)
    print("wrote", out.name)


def hsi_signatures(ds, nsp=8):
    by_sp = collect(ds); sps = top_species(by_sp, nsp)
    fig, a = plt.subplots(figsize=(9, 5))
    for sp in sps:
        specs = []
        for p in by_sp[sp][:40]:
            t = TreeSample.load_npz(p)
            h, m = t.hsi_crown, t.hsi_mask
            if h.size == 0 or m.sum() == 0:
                continue
            specs.append(h[m].mean(0))  # mean over crown pixels -> (369,)
        if specs:
            mu = np.mean(specs, 0)
            a.plot(mu, label=f"{sp} (n={len(specs)})", lw=1.3)
    a.set_xlabel("HSI band index (369 bands)"); a.set_ylabel("mean reflectance")
    a.set_title(f"{ds}: mean per-species spectral signature — spectral separability")
    a.legend(fontsize=7, ncol=2)
    fig.tight_layout(); out = FIGS / f"hsi_{ds}.png"; fig.savefig(out, dpi=110); plt.close(fig)
    print("wrote", out.name)


def learning_curve():
    from data import load
    from spatial_cv import make_folds
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    fracs = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    fig, a = plt.subplots(figsize=(8, 5))
    for ds in ["Weiser", "NEON-OSBS", "NEON-MLBS"]:
        d = load(ds); folds = make_folds(d["group"], d["xy"], d["y"], ds)
        X, y = d["clip"], d["y"]
        means, stds = [], []
        for fr in fracs:
            f1s = []
            for seed in [0, 1, 2]:
                rng = np.random.RandomState(seed); yt_all, yp_all = [], []
                for f in folds:
                    tr, te = f["train_idx"], f["test_idx"]
                    if len(te) == 0:
                        continue
                    # stratified subsample of train to fraction fr
                    sub = []
                    for c in set(y[tr]):
                        idx = tr[y[tr] == c]; n = max(1, int(round(len(idx) * fr)))
                        sub.extend(rng.choice(idx, n, replace=False))
                    sub = np.array(sub)
                    sc = StandardScaler().fit(X[sub])
                    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(X[sub]), y[sub])
                    yp_all.extend(clf.predict(sc.transform(X[te])).tolist()); yt_all.extend(y[te].tolist())
                f1s.append(f1_score(yt_all, yp_all, average="macro", zero_division=0))
            means.append(np.mean(f1s)); stds.append(np.std(f1s))
        means, stds = np.array(means), np.array(stds)
        ntr = int(np.mean([len(f["train_idx"]) for f in folds]))
        a.errorbar([fr for fr in fracs], means, yerr=stds, marker="o", capsize=3,
                   label=f"{ds} (~{ntr} train/fold, {d['K']} sp)")
    a.set_xlabel("fraction of training data used"); a.set_ylabel("pooled macro-F1 (spatial CV)")
    a.set_title("CLIP+linear learning curve — is the head data-limited?")
    a.legend(); a.grid(alpha=0.3)
    fig.tight_layout(); out = FIGS / "learning_curve.png"; fig.savefig(out, dpi=120); plt.close(fig)
    print("wrote", out.name)


def main():
    for ds in ["Weiser", "NEON-OSBS", "NEON-MLBS"]:
        try: structural_montage(ds)
        except Exception as e: print(f"structure {ds} FAIL: {e}")
    for ds in ["NEON-OSBS", "NEON-MLBS"]:
        try: rgb_montage(ds)
        except Exception as e: print(f"rgb {ds} FAIL: {e}")
        try: hsi_signatures(ds)
        except Exception as e: print(f"hsi {ds} FAIL: {e}")
    try: learning_curve()
    except Exception as e: print(f"learning_curve FAIL: {e}")
    print("FIGS in", FIGS)


if __name__ == "__main__":
    main()
