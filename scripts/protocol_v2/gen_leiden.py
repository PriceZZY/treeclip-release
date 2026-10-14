"""Leiden community detection on the matched 11-d features (PFG round-1, reviewer 2:
community detection / "more advanced clustering"). Leiden does NOT take K: it estimates
the community count itself, so it joins DP-GMM and HDBSCAN in the estimate-K family.

kNN graph (k=15, euclidean on StandardScaler->PCA-11), igraph + leidenalg modularity
partition, 15 seeds. Reports K-hat and species-NMI for CLIP-11d and geo-11d on both
datasets. Output: runs/protocol_v2/gen/leiden_estimateK.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import normalized_mutual_info_score as NMI
import igraph as ig
import leidenalg as la
import gen_score as G

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"
SEEDS = list(range(15))
KNN = 15


def leiden_once(X11, y, seed):
    nn = NearestNeighbors(n_neighbors=KNN + 1).fit(X11)
    _, ind = nn.kneighbors(X11)
    edges = {(min(i, j), max(i, j)) for i, row in enumerate(ind) for j in row[1:]}
    g = ig.Graph(n=len(X11), edges=list(edges))
    part = la.find_partition(g, la.ModularityVertexPartition, seed=seed)
    cl = np.array(part.membership)
    return len(set(cl)), NMI(y, cl)


def main():
    out = {"n_seeds": len(SEEDS), "knn": KNN}
    for which in ["for", "biodiv"]:
        if which == "for":
            d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
            g = np.load(ROOT / "data/raw/for_species20k/geo_features_matched.npz", allow_pickle=True)
            Xc = d["X"]; y = d["species"].astype(str); groups = d["dataset"].astype(str); Xg = g["X_sf"]; fag = None
        else:
            d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
            Xc = d["X_clip"]; Xg = d["X_sf"]; y = d["species"].astype(str); groups = d["region"].astype(str); fag = 0.39
        m, _ = G.analyzable_mask(y, groups); Xc, Xg, y, groups = Xc[m], Xg[m], y[m], groups[m]
        if fag:
            c = Counter(y); top = max(c, key=c.get); n = len(y)
            if c[top] / n > fag:
                tgt = int(fag/(1-fag)*(n-c[top])); idx = np.where(y == top)[0]
                drop = set(np.random.RandomState(0).choice(idx, c[top]-tgt, replace=False))
                k = np.array([i not in drop for i in range(n)]); Xc, Xg, y = Xc[k], Xg[k], y[k]
        res = {}
        for name, X, pd_ in [("CLIP-11d", Xc, 11), ("geo-11d", Xg, None)]:
            Xs = StandardScaler().fit_transform(X)
            if pd_ is not None:
                Xs = PCA(n_components=min(pd_, Xs.shape[1]), random_state=0).fit_transform(Xs)
            kh, nm = zip(*[leiden_once(Xs, y, s) for s in SEEDS])
            res[name] = {"K_hat_mean": round(float(np.mean(kh)), 1), "K_hat_range": [int(min(kh)), int(max(kh))],
                         "NMI_mean": round(float(np.mean(nm)), 4), "NMI_std": round(float(np.std(nm)), 4)}
            print(which, name, "true_K", len(set(y)), res[name], flush=True)
        out[which] = {"true_K": len(set(y)), **res}
    json.dump(out, open(OUT / "leiden_estimateK.json", "w"), indent=2)


if __name__ == "__main__":
    main()
