"""INDEPENDENT recompute of unsup_robust_for.json (verification, not importing the original arms).
Reimplements analyzable_mask, hungarian, NMI/macroF1 from scratch for 3 seeds x 3 arms.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root

# --- independent analyzable mask ---
def analyzable_mask(y, groups, min_n=30, min_blocks=2):
    keep = []
    for s in set(y):
        sel = (y == s)
        if sel.sum() >= min_n and len(set(groups[sel])) >= min_blocks:
            keep.append(s)
    return np.isin(y, keep)

# --- independent hungarian mapping cluster->species ---
def hungarian_pred(yt, cl):
    sp = sorted(set(yt)); cs = sorted(set(cl))
    cost = np.zeros((len(cs), len(sp)))
    for i, c in enumerate(cs):
        for j, s in enumerate(sp):
            cost[i, j] = -np.sum((cl == c) & (yt == s))
    r, cc = linear_sum_assignment(cost)
    m = {cs[r[i]]: sp[cc[i]] for i in range(len(r))}
    return np.array([m.get(c, sp[0]) for c in cl])

d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
g = np.load(ROOT / "data/raw/for_species20k/geo_features_matched.npz", allow_pickle=True)
Xc = d["X"]; y = d["species"].astype(str); groups = d["dataset"].astype(str)
Xg = g["X_sf"]
print("raw shapes: Xc", Xc.shape, "Xg", Xg.shape, "len match:", len(Xc) == len(Xg))

m = analyzable_mask(y, groups)
Xc, Xg, y = Xc[m], Xg[m], y[m]
K = len(set(y))
print("after mask: n =", len(y), "K =", K, "geo_dim =", Xg.shape[1], "clip_dim =", Xc.shape[1])
print("species kept:", sorted(set(y)))

def run_arm(X, K, seed, pca_dim=None):
    Xs = StandardScaler().fit_transform(X)
    if pca_dim is not None:
        k = min(pca_dim, Xs.shape[1], Xs.shape[0]-1)
        Xs = PCA(n_components=k, random_state=seed).fit_transform(Xs)
    cl = KMeans(n_clusters=K, random_state=seed, n_init=10).fit_predict(Xs)
    yp = hungarian_pred(y, cl)
    return f1_score(y, yp, average="macro", zero_division=0), normalized_mutual_info_score(y, cl)

seeds = [0, 1, 2]
for label, X, pca in [("CLIP_512", Xc, None), ("CLIP_PCA11", Xc, 11), ("geo_sf_11", Xg, None)]:
    f1s, nmis = [], []
    for s in seeds:
        f1, nmi = run_arm(X, K, s, pca)
        f1s.append(f1); nmis.append(nmi)
    print(f"{label:12s} seeds{seeds}  macroF1={[round(v,4) for v in f1s]} mean={np.mean(f1s):.4f}  "
          f"NMI={[round(v,4) for v in nmis]} mean={np.mean(nmis):.4f}")
