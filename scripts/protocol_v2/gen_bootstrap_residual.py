"""B-1: DATA-LEVEL bootstrap CI on the within-source CLIP-minus-geo NMI residual.

The paper's existing "95% CI" on this residual is computed over 15 KMeans seeds
(clustering-init jitter), NOT data-level sampling. This produces a genuine
DATA-LEVEL bootstrap CI: resample TREES with replacement, blocked by source block,
then recompute the within-source CLIP(PCA-11d)-minus-geo(11d) NMI residual on the
resampled data. The KMeans seed is FIXED at 0 across replicates, so the variability
measured is DATA sampling, not init jitter. B=1000.

Reuses within_source_nmi_seed / analyzable_mask / Fagus cap EXACTLY from the harness
(no new modeling choices).

Usage: python gen_bootstrap_residual.py for | biodiv
Output: runs/protocol_v2/gen/bootstrap_residual_{which}.json
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from collections import Counter
from joblib import Parallel, delayed
import gen_score as G
from gen_source_control_robust import within_source_nmi_seed

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"
KMEANS_SEED = 0
B = 1000
N_JOBS = 12

CANON = {"for": (0.0436, [0.0427, 0.0446]), "biodiv": (0.2039, [0.2032, 0.2046])}


def load(which):
    if which == "for":
        d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
        g = np.load(ROOT / "data/raw/for_species20k/geo_features_matched.npz", allow_pickle=True)
        Xc = d["X"]; y = d["species"].astype(str); groups = d["dataset"].astype(str); Xg = g["X_sf"]; fag = None
    else:
        d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
        Xc = d["X_clip"]; Xg = d["X_sf"]; y = d["species"].astype(str); groups = d["region"].astype(str); fag = 0.39
    m, _ = G.analyzable_mask(y, groups); Xc, Xg, y, groups = Xc[m], Xg[m], y[m], groups[m]
    if fag is not None:
        c = Counter(y); top = max(c, key=c.get); n = len(y)
        if c[top] / n > fag:
            tgt = int(fag / (1 - fag) * (n - c[top])); idx = np.where(y == top)[0]
            drop = set(np.random.RandomState(0).choice(idx, c[top] - tgt, replace=False))
            k = np.array([i not in drop for i in range(n)])
            Xc, Xg, y, groups = Xc[k], Xg[k], y[k], groups[k]
    return Xc, Xg, y, groups


def residual(Xc, Xg, y, groups):
    clip = within_source_nmi_seed(Xc, y, groups, KMEANS_SEED, pca_dim=11)
    geo = within_source_nmi_seed(Xg, y, groups, KMEANS_SEED, pca_dim=None)
    return clip - geo


def one_rep(b, Xc, Xg, y, groups, block_idx):
    rng = np.random.default_rng(b + 1)
    sel = np.concatenate([idx[rng.integers(0, len(idx), len(idx))] for idx in block_idx])
    return residual(Xc[sel], Xg[sel], y[sel], groups[sel])


def main():
    which = sys.argv[1]
    Xc, Xg, y, groups = load(which)
    blocks = sorted(set(groups))
    block_idx = [np.where(groups == g)[0] for g in blocks]
    point = float(residual(Xc, Xg, y, groups))  # seed-0 residual on ORIGINAL (un-resampled) data
    t = time.time()
    reps = Parallel(n_jobs=N_JOBS)(delayed(one_rep)(b, Xc, Xg, y, groups, block_idx) for b in range(B))
    reps = np.asarray(reps, float)
    canon_mean, canon_ci = CANON[which]
    method = ("DATA-LEVEL bootstrap: resample TREES with replacement WITHIN each source block "
              "(preserves block structure), recompute within-source CLIP(PCA-11d)-minus-geo(11d) NMI "
              "residual at FIXED KMeans seed=0, so the measured spread is DATA sampling not init jitter.")
    if which == "for":
        method += " FOR: 21 source blocks; trees resampled within each block."
    else:
        method += (" BioDiv: only 3 region blocks -> a block-level bootstrap is uninformative, so "
                   "we resample TREES with replacement within each of the 3 regions (tree-level).")
    out = {
        "dataset": which, "B": B, "kmeans_seed": KMEANS_SEED,
        "n_blocks": len(blocks), "n_trees": int(len(y)),
        "point_estimate": round(point, 4),
        "bootstrap_mean": round(float(reps.mean()), 4),
        "bootstrap_median": round(float(np.median(reps)), 4),
        "ci95": [round(float(np.percentile(reps, 2.5)), 4), round(float(np.percentile(reps, 97.5)), 4)],
        "frac_positive": round(float((reps > 0).mean()), 4),
        "canonical_15seed_mean_residual": canon_mean,
        "canonical_15seed_ci95_seedjitter": canon_ci,
        "method": method,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"bootstrap_residual_{which}.json", "w"), indent=2)
    print(json.dumps(out, indent=2))
    print("elapsed %.1fs" % (time.time() - t))


if __name__ == "__main__":
    main()
