"""ARI + Rand Index for every Table-2 row (PFG round-1, reviewer 2: "RandIndex and
AdjustedRandIndex are easy to interpret. Why are they not included?").

Same protocol as gen_unsup_robust / _multi: StandardScaler -> optional PCA -> KMeans(K,
n_init=10), 15 seeds, analyzable subset, BioDiv Fagus capped 39%. Matching-free scores
(no Hungarian needed). Output: runs/protocol_v2/gen/ari_multi.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, rand_score
import gen_score as G

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"
SEEDS = list(range(15))


def align(tref, tsrc, X):
    idx = {str(t): i for i, t in enumerate(tsrc)}
    rows = np.array([idx.get(str(t), -1) for t in tref])
    assert (rows >= 0).all()
    return X[rows]


def one(X, y, K, seed, pca_dim):
    Xs = StandardScaler().fit_transform(X)
    if pca_dim is not None:
        k = min(pca_dim, Xs.shape[1], Xs.shape[0] - 1)
        Xs = PCA(n_components=k, random_state=seed).fit_transform(Xs)
    cl = KMeans(n_clusters=K, random_state=seed, n_init=10).fit_predict(Xs)
    return adjusted_rand_score(y, cl), rand_score(y, cl)


def main():
    out = {"n_seeds": len(SEEDS)}
    for which in ["for", "biodiv"]:
        if which == "for":
            d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
            g = np.load(ROOT / "data/raw/for_species20k/geo_features_matched.npz", allow_pickle=True)
            dn = np.load(ROOT / "data/raw/for_species20k/for_dino.npz", allow_pickle=True)
            ex = np.load(ROOT / "data/raw/for_species20k/for_extra.npz", allow_pickle=True)
            Xc = d["X"]; y = d["species"].astype(str); groups = d["dataset"].astype(str)
            Xg = g["X_sf"]; tref = d["treeID"]; fag = None
        else:
            d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
            dn = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_dino.npz", allow_pickle=True)
            ex = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_extra.npz", allow_pickle=True)
            Xc = d["X_clip"]; Xg = d["X_sf"]; y = d["species"].astype(str)
            groups = d["region"].astype(str); tref = d["treeID"]; fag = 0.39
        Xd = align(tref, dn["treeID"], dn["X_dino"])
        Xr = align(tref, ex["treeID"], ex["X_rand"])
        Xi = align(tref, ex["treeID"], ex["X_imnet"])
        m, _ = G.analyzable_mask(y, groups)
        Xc, Xg, Xd, Xr, Xi, y, groups = Xc[m], Xg[m], Xd[m], Xr[m], Xi[m], y[m], groups[m]
        if fag:
            c = Counter(y); top = max(c, key=c.get); n = len(y)
            if c[top] / n > fag:
                tgt = int(fag/(1-fag)*(n-c[top])); idx = np.where(y == top)[0]
                drop = set(np.random.RandomState(0).choice(idx, c[top]-tgt, replace=False))
                k = np.array([i not in drop for i in range(n)])
                Xc, Xg, Xd, Xr, Xi, y, groups = Xc[k], Xg[k], Xd[k], Xr[k], Xi[k], y[k], groups[k]
        K = len(set(y))
        res = {}
        for name, X, pd_ in [("CLIP-512", Xc, None), ("CLIP-11d", Xc, 11), ("DINOv2-11d", Xd, 11),
                             ("ImageNet-11d", Xi, 11), ("random-11d", Xr, 11), ("geo-11d", Xg, None)]:
            vals = [one(X, y, K, s, pd_) for s in SEEDS]
            ari = [v[0] for v in vals]; ri = [v[1] for v in vals]
            res[name] = {"ARI_mean": round(float(np.mean(ari)), 4), "ARI_std": round(float(np.std(ari)), 4),
                         "RI_mean": round(float(np.mean(ri)), 4), "RI_std": round(float(np.std(ri)), 4)}
            print(which, name, res[name], flush=True)
        out[which] = {"K": K, "n": int(len(y)), **res}
    json.dump(out, open(OUT / "ari_multi.json", "w"), indent=2)


if __name__ == "__main__":
    main()
