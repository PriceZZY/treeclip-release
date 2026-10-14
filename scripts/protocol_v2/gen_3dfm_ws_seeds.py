"""OpenShape within-source NMI over 15 seeds (PFG round-1, R1 comment 6: the frozen-3D
within-source leg was a single seed, "preliminary or indicative"; we upgrade it to the
full 15-seed protocol instead of merely labelling it).

Reuses the cached embeddings runs/protocol_v2/gen/emb_3dfm_{which}.npz (proj 1280-d,
pooled 256-d) and the exact within_source_nmi block loop of gen_3dfm_baseline (skip rule
<2 species or <4 trees, per-block PCA-11, KMeans at the block's own K).
Output: runs/protocol_v2/gen/tdfm_ws_seeds.json. CPU-only, minutes.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score as NMI
import gen_score as G

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"
SEEDS = list(range(15))


def ws_nmi(X, y, groups, seed, pca_dim=11):
    tot, wsum = 0.0, 0
    for g in sorted(set(groups)):
        mask = groups == g; yy = y[mask]
        if len(set(yy)) < 2 or mask.sum() < 4: continue
        Xs = StandardScaler().fit_transform(X[mask])
        k = min(pca_dim, Xs.shape[1], Xs.shape[0] - 1)
        Xs = PCA(n_components=k, random_state=seed).fit_transform(Xs)
        cl = KMeans(n_clusters=len(set(yy)), random_state=seed, n_init=10).fit_predict(Xs)
        tot += NMI(yy, cl) * mask.sum(); wsum += mask.sum()
    return tot / wsum


def main():
    out = {"n_seeds": len(SEEDS)}
    for which in ["for", "biodiv"]:
        if which == "for":
            d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
            y = d["species"].astype(str); groups = d["dataset"].astype(str); tref = d["treeID"]
            fag = None
        else:
            d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
            y = d["species"].astype(str); groups = d["region"].astype(str); tref = d["treeID"]
            fag = 0.39
        m, _ = G.analyzable_mask(y, groups)
        y, groups, tref = y[m], groups[m], tref[m]
        if fag:
            c = Counter(y); top = max(c, key=c.get); n = len(y)
            if c[top] / n > fag:
                tgt = int(fag/(1-fag)*(n-c[top])); idx = np.where(y == top)[0]
                drop = set(np.random.RandomState(0).choice(idx, c[top]-tgt, replace=False))
                k = np.array([i not in drop for i in range(n)]); y, groups, tref = y[k], groups[k], tref[k]
        cache = np.load(OUT / f"emb_3dfm_{which}.npz", allow_pickle=True)
        idx = {str(t): i for i, t in enumerate(cache["treeID"])}
        rows = np.array([idx.get(str(t), -1) for t in tref])
        assert (rows >= 0).all(), f"{which}: {int((rows<0).sum())} treeIDs missing from 3DFM cache"
        out[which] = {}
        for readout in ["proj", "pooled"]:
            X = cache[readout][rows]
            v = [ws_nmi(X, y, groups, s) for s in SEEDS]
            out[which][readout] = {"ws_NMI_mean": round(float(np.mean(v)), 4),
                                   "ws_NMI_std": round(float(np.std(v)), 4),
                                   "ws_NMI_seed0": round(float(v[0]), 4)}
            print(which, readout, out[which][readout], flush=True)
    json.dump(out, open(OUT / "tdfm_ws_seeds.json", "w"), indent=2)


if __name__ == "__main__":
    main()
