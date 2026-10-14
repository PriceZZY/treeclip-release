"""SOURCE-CONFOUND CONTROL (blind-review Major-1/2): does the unsupervised CLIP>geo NMI advantage survive
when acquisition source/region is partialled out?

Method: WITHIN-SOURCE clustering. For each source block g (FOR: dataset; BioDiv: region), cluster ONLY the trees
from g (k = #species in g), Hungarian-match + NMI within that block, then average NMI across blocks weighted by
block size. If CLIP still beats geo within-source, the advantage is not merely between-source/sensor clustering.
Also reports: (a) how well each feature predicts the SOURCE itself (the confound magnitude), at MATCHED 11-d
(geo_sf) for a like-for-like comparison; (b) the global NMI for reference.

CPU-only. Output: runs/protocol_v2/gen/source_control.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import normalized_mutual_info_score as NMI, balanced_accuracy_score
import gen_score as G
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"


def within_source_nmi(X, y, groups, pca_dim=None, seed=0):
    """cluster within each source block separately; size-weighted mean NMI."""
    tot, wsum = 0.0, 0
    for g in sorted(set(groups)):
        mask = groups == g
        yy = y[mask]
        if len(set(yy)) < 2 or mask.sum() < 4:
            continue
        Xs = StandardScaler().fit_transform(X[mask])
        if pca_dim is not None:
            k = min(pca_dim, Xs.shape[1], Xs.shape[0] - 1)
            Xs = PCA(n_components=k, random_state=seed).fit_transform(Xs)
        K = len(set(yy))
        cl = KMeans(n_clusters=K, random_state=seed, n_init=10).fit_predict(Xs)
        tot += NMI(yy, cl) * mask.sum(); wsum += mask.sum()
    return round(tot / wsum, 4) if wsum else 0.0


def source_predictability(X, groups, pca_dim=None, seed=0):
    """how well does the feature predict the SOURCE block (confound magnitude), matched dim, 3-fold balanced-acc."""
    Xs = StandardScaler().fit_transform(X)
    if pca_dim is not None:
        k = min(pca_dim, Xs.shape[1], Xs.shape[0] - 1)
        Xs = PCA(n_components=k, random_state=seed).fit_transform(Xs)
    # keep groups with >=3 members for CV
    cnt = Counter(groups); keep = np.array([cnt[g] >= 6 for g in groups])
    if keep.sum() < 20 or len(set(groups[keep])) < 2: return None
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=1)
    sc = cross_val_score(clf, Xs[keep], groups[keep], cv=cv, scoring="balanced_accuracy")
    chance = 1.0 / len(set(groups[keep]))
    return {"bal_acc": round(float(sc.mean()), 4), "chance": round(chance, 4)}


def main():
    out = {}
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
                k = np.array([i not in drop for i in range(n)]); Xc, Xsf, y, groups = Xc[k], Xsf[k], y[k], groups[k]
        out[which] = {
            "n_blocks": len(set(groups)),
            "global_NMI_CLIP11": round(NMI(y, KMeans(len(set(y)), random_state=0, n_init=10).fit_predict(
                PCA(min(11, Xc.shape[1]), random_state=0).fit_transform(StandardScaler().fit_transform(Xc)))), 4),
            "within_source_NMI_CLIP_11d": within_source_nmi(Xc, y, groups, pca_dim=11),
            "within_source_NMI_geo_11d": within_source_nmi(Xsf, y, groups, pca_dim=None),
            "source_predict_CLIP_11d": source_predictability(Xc, groups, pca_dim=11),
            "source_predict_geo_11d": source_predictability(Xsf, groups, pca_dim=None),
        }
        wc = out[which]["within_source_NMI_CLIP_11d"]; wg = out[which]["within_source_NMI_geo_11d"]
        out[which]["within_source_delta_CLIP_minus_geo"] = round(wc - wg, 4)
    json.dump(out, open(OUT / "source_control.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
