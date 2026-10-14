"""Independent recompute to verify unsup_robust_{for,biodiv}.json.
Re-derives NMI + Hungarian macroF1/OA/mIoU from cached npz, a few seeds, both datasets.
Also inspects geo feature contents and Fagus downsample.
"""
import sys, json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score, jaccard_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent.parent


def analyzable_mask(y, groups, min_n=30, min_blocks=2):
    keep = []
    for s in set(y):
        if (y == s).sum() >= min_n and len(set(groups[y == s])) >= min_blocks:
            keep.append(s)
    return np.isin(y, keep), sorted(keep)


def hungarian_pred(yt, cl):
    sp = sorted(set(yt)); cs = sorted(set(cl)); cost = np.zeros((len(cs), len(sp)))
    for i, c in enumerate(cs):
        for j, s in enumerate(sp):
            cost[i, j] = -np.sum((cl == c) & (yt == s))
    r, cc = linear_sum_assignment(cost)
    m = {cs[r[i]]: sp[cc[i]] for i in range(len(r))}
    return np.array([m.get(c, sp[0]) for c in cl])


def metrics_for_seed(X, y, K, seed, pca_dim=None):
    Xs = StandardScaler().fit_transform(X)
    if pca_dim is not None:
        k = min(pca_dim, Xs.shape[1], Xs.shape[0] - 1)
        Xs = PCA(n_components=k, random_state=seed).fit_transform(Xs)
    cl = KMeans(n_clusters=K, random_state=seed, n_init=10).fit_predict(Xs)
    yp = hungarian_pred(y, cl)
    return {"macroF1": f1_score(y, yp, average="macro", zero_division=0),
            "OA": accuracy_score(y, yp),
            "mIoU": jaccard_score(y, yp, average="macro", zero_division=0),
            "NMI": normalized_mutual_info_score(y, cl)}


def random_nmi(y, K, n_draws=20, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_draws):
        cl = rng.integers(0, K, len(y))
        vals.append(normalized_mutual_info_score(y, cl))
    return float(np.mean(vals)), float(np.std(vals))


def run(which, seeds):
    if which == "for":
        d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
        g = np.load(ROOT / "data/raw/for_species20k/geo_features_matched.npz", allow_pickle=True)
        Xc = d["X"]; y = d["species"].astype(str); groups = d["dataset"].astype(str); Xg = g["X_sf"]
        print("FOR geo npz keys:", list(g.keys()), "X_sf shape:", Xg.shape)
        print("FOR clip n:", len(Xc), "geo n:", len(Xg), "match:", len(Xc) == len(Xg))
        fag = None
    else:
        d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
        Xc = d["X_clip"]; Xg = d["X_sf"]; y = d["species"].astype(str); groups = d["region"].astype(str)
        print("BioDiv npz keys:", list(d.keys()), "X_sf shape:", Xg.shape, "X_clip shape:", Xc.shape)
        fag = 0.39
    m, keep = analyzable_mask(y, groups)
    print(f"[{which}] analyzable species kept ({len(keep)}):", keep)
    print(f"[{which}] full n={len(y)}, after mask n={m.sum()}")
    Xc, Xg, y, groups = Xc[m], Xg[m], y[m], groups[m]
    if fag is not None:
        c = Counter(y); top = max(c, key=c.get); n = len(y)
        print(f"[{which}] before fagus downsample: top={top} frac={c[top]/n:.3f}, counts={dict(c)}")
        if c[top] / n > fag:
            tgt = int(fag / (1 - fag) * (n - c[top])); idx = np.where(y == top)[0]
            drop = set(np.random.RandomState(0).choice(idx, c[top] - tgt, replace=False))
            k = np.array([i not in drop for i in range(n)]); Xc, Xg, y = Xc[k], Xg[k], y[k]
        cc = Counter(y)
        print(f"[{which}] after fagus downsample: n={len(y)}, top frac={cc[max(cc,key=cc.get)]/len(y):.3f}, counts={dict(cc)}")
    K = len(set(y)); gdim = Xg.shape[1]
    print(f"[{which}] K={K}, n={len(y)}, geo_dim={gdim}, clip_dim={Xc.shape[1]}")
    rnmi_m, rnmi_s = random_nmi(y, K)
    print(f"[{which}] RANDOM-clustering NMI baseline (K={K}): {rnmi_m:.5f} +/- {rnmi_s:.5f}")

    for arm, pca in [("CLIP_512", None), ("CLIP_matched_%dd" % gdim, gdim), ("geo_%dd" % gdim, None)]:
        Xuse = Xg if arm.startswith("geo") else Xc
        nmis, f1s, oas, ious = [], [], [], []
        for s in seeds:
            r = metrics_for_seed(Xuse, y, K, s, pca)
            nmis.append(r["NMI"]); f1s.append(r["macroF1"]); oas.append(r["OA"]); ious.append(r["mIoU"])
        print(f"[{which}] {arm}: NMI={np.mean(nmis):.4f}(sd{np.std(nmis):.4f}) "
              f"F1={np.mean(f1s):.4f} OA={np.mean(oas):.4f} mIoU={np.mean(ious):.4f}  per-seed NMI={[round(v,4) for v in nmis]}")


if __name__ == "__main__":
    which = sys.argv[1]
    nseeds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    run(which, list(range(nseeds)))
