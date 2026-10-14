"""Fully independent recompute - does NOT import gen_unsup_robust arm/metrics."""
import sys, numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score, jaccard_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment


def my_mask(y, groups, min_n=30, min_blocks=2):
    keep = [s for s in set(y) if (y == s).sum() >= min_n and len(set(groups[y == s])) >= min_blocks]
    return np.isin(y, keep)


def my_hungarian(yt, cl):
    sp = sorted(set(yt)); cs = sorted(set(cl))
    cost = np.zeros((len(cs), len(sp)))
    for i, c in enumerate(cs):
        for j, s in enumerate(sp):
            cost[i, j] = -np.sum((cl == c) & (yt == s))
    r, cc = linear_sum_assignment(cost)
    m = {cs[r[i]]: sp[cc[i]] for i in range(len(r))}
    return np.array([m.get(c, sp[0]) for c in cl])


def run_arm(X, y, K, seeds, pca_dim=None):
    out = {m: [] for m in ["macroF1", "OA", "mIoU", "NMI"]}
    perseed = {}
    for s in seeds:
        Xs = StandardScaler().fit_transform(X)
        if pca_dim is not None:
            k = min(pca_dim, Xs.shape[1], Xs.shape[0] - 1)
            Xs = PCA(n_components=k, random_state=s).fit_transform(Xs)
        cl = KMeans(n_clusters=K, random_state=s, n_init=10).fit_predict(Xs)
        yp = my_hungarian(y, cl)
        out["macroF1"].append(f1_score(y, yp, average="macro", zero_division=0))
        out["OA"].append(accuracy_score(y, yp))
        out["mIoU"].append(jaccard_score(y, yp, average="macro", zero_division=0))
        nmi = normalized_mutual_info_score(y, cl)
        out["NMI"].append(nmi)
        perseed[s] = nmi
    return {m: (np.mean(v), np.std(v)) for m, v in out.items()}, out["NMI"]


def random_nmi(y, K, ntrial=10):
    rng = np.random.default_rng(123)
    vals = [normalized_mutual_info_score(y, rng.integers(0, K, len(y))) for _ in range(ntrial)]
    return np.mean(vals), np.std(vals)


which = sys.argv[1]
ROOT = "."
if which == "for":
    d = np.load("data/raw/for_species20k/clip_features.npz", allow_pickle=True)
    g = np.load("data/raw/for_species20k/geo_features_matched.npz", allow_pickle=True)
    Xc = d["X"]; y = d["species"].astype(str); groups = d["dataset"].astype(str); Xg = g["X_sf"]
    fag = None; seeds = [0, 1, 2]
else:
    d = np.load("data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
    Xc = d["X_clip"]; Xg = d["X_sf"]; y = d["species"].astype(str); groups = d["region"].astype(str)
    fag = 0.39; seeds = [0, 1, 2, 3, 4]

m = my_mask(y, groups)
Xc, Xg, y, groups = Xc[m], Xg[m], y[m], groups[m]
print("after mask n=%d K=%d" % (len(y), len(set(y))))
if fag is not None:
    c = Counter(y); top = max(c, key=c.get); n = len(y)
    if c[top] / n > fag:
        tgt = int(fag / (1 - fag) * (n - c[top])); idx = np.where(y == top)[0]
        drop = set(np.random.RandomState(0).choice(idx, c[top] - tgt, replace=False))
        k = np.array([i not in drop for i in range(n)]); Xc, Xg, y = Xc[k], Xg[k], y[k]
    cc = Counter(y)
    print("post-downsample n=%d, top=%s frac=%.4f, Fagus-like top species=%s frac=%.4f"
          % (len(y), max(cc, key=cc.get), cc[max(cc, key=cc.get)] / len(y),
             top, cc.get(top, 0) / len(y)))
K = len(set(y)); gdim = Xg.shape[1]
print("n=%d K=%d geo_dim=%d clip_dim=%d" % (len(y), K, gdim, Xc.shape[1]))

c512, nmi512 = run_arm(Xc, y, K, seeds)
cm, nmim = run_arm(Xc, y, K, seeds, pca_dim=gdim)
cg, nmig = run_arm(Xg, y, K, seeds)
print("\nARM        NMI(mean,std)   macroF1   OA      mIoU")
for name, a in [("CLIP_512", c512), ("CLIP_match%dd" % gdim, cm), ("geo_%dd" % gdim, cg)]:
    print("%-12s NMI=%.4f(%.4f) F1=%.4f OA=%.4f mIoU=%.4f"
          % (name, a["NMI"][0], a["NMI"][1], a["macroF1"][0], a["OA"][0], a["mIoU"][0]))
print("\nper-seed NMI CLIP512:", [round(x, 4) for x in nmi512])
print("per-seed NMI geo:", [round(x, 4) for x in nmig])
print("per-seed NMI CLIPmatched:", [round(x, 4) for x in nmim])
print("delta matched-geo NMI = %.4f" % (cm["NMI"][0] - cg["NMI"][0]))
print("delta 512-geo NMI = %.4f" % (c512["NMI"][0] - cg["NMI"][0]))
rn = random_nmi(y, K)
print("random-cluster NMI baseline = %.4f +/- %.4f" % rn)
