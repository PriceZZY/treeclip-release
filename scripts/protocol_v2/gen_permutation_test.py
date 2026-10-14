"""Data-level permutation test for the headline unsupervised NMI gap (blind-review: 'CIs are init-jitter only,
need a data-level significance test'). For each dataset we test H0: CLIP and geometry cluster species no
differently. Observed statistic = NMI(CLIP clustering, true labels) - NMI(geo clustering, true labels) at
matched 11-d. Null: permute the species labels and recompute the SAME statistic; the permutation breaks any
true feature-species relationship while preserving cluster structure and label marginals. p = fraction of
permutations with null-delta >= observed-delta. CPU-only. Output: runs/protocol_v2/gen/perm_test.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score as NMI
import gen_score as G
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"
N_PERM = 10000


def cluster_11d(X, K, seed=0):
    Xs = StandardScaler().fit_transform(X)
    k = min(11, Xs.shape[1], Xs.shape[0] - 1)
    Xs = PCA(n_components=k, random_state=seed).fit_transform(Xs)
    return KMeans(n_clusters=K, random_state=seed, n_init=10).fit_predict(Xs)


def main():
    out = {"n_perm": N_PERM, "note": "permute species labels; stat = NMI(CLIP)-NMI(geo) at matched 11-d global clustering"}
    for which in ["for", "biodiv"]:
        if which == "for":
            d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
            g = np.load(ROOT / "data/raw/for_species20k/geo_features_matched.npz", allow_pickle=True)
            Xc = d["X"]; y = d["species"].astype(str); groups = d["dataset"].astype(str); Xsf = g["X_sf"]
        else:
            d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
            Xc = d["X_clip"]; Xsf = d["X_sf"]; y = d["species"].astype(str); groups = d["region"].astype(str)
        m, _ = G.analyzable_mask(y, groups); Xc, Xsf, y, groups = Xc[m], Xsf[m], y[m], groups[m]
        if which == "biodiv":
            c = Counter(y); top = max(c, key=c.get); n = len(y); cap = 0.39
            if c[top] / n > cap:
                tgt = int(cap/(1-cap)*(n-c[top])); idx = np.where(y == top)[0]
                drop = set(np.random.RandomState(0).choice(idx, c[top]-tgt, replace=False))
                k = np.array([i not in drop for i in range(n)]); Xc, Xsf, y = Xc[k], Xsf[k], y[k]
        K = len(set(y))
        # fixed clusterings (seed 0) — permutation acts on labels, not on clustering
        cl_clip = cluster_11d(Xc, K); cl_geo = cluster_11d(Xsf, K)
        obs = NMI(y, cl_clip) - NMI(y, cl_geo)
        rng = np.random.default_rng(0); ge = 0
        null = []
        for _ in range(N_PERM):
            yp = rng.permutation(y)
            nd = NMI(yp, cl_clip) - NMI(yp, cl_geo)
            null.append(nd)
            if nd >= obs: ge += 1
        p = (ge + 1) / (N_PERM + 1)
        out[which] = {"observed_delta_NMI": round(float(obs), 4),
                      "null_delta_mean": round(float(np.mean(null)), 5),
                      "null_delta_p95": round(float(np.percentile(null, 95)), 5),
                      "p_value": round(float(p), 5), "K": K, "n": int(len(y))}
    json.dump(out, open(OUT / "perm_test.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
