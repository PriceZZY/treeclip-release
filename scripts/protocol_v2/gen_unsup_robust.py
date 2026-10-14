"""Robustness controls for the unsupervised CLIP>geo claim (the 2 cheap fixes the adjudicator required):
  (1) MULTI-SEED: 15 KMeans seeds -> mean +/- 95% bootstrap-of-seeds CI on every metric and on the CLIP-geo delta,
      to show the +0.05~0.08 margin survives seed variance (not a single-seed fluke).
  (2) DIMENSION-MATCHED CONTROL: report CLIP at native 512-d AND CLIP PCA-reduced to geo's 11-d, to rule out the
      512-vs-11 capacity confound. Also a geo-as-is 11-d reference.
Metrics: macro-F1, OA, mIoU (all Hungarian open-book = cluster-quality upper bound) + NMI (matching-free,
scale-free -- the attack-immune metric the adjudicator said to lead with).
Leakage-aware analyzable subset (>=30 trees & >=2 blocks); BioDiv Fagus downsampled <40%. CPU-only.

Usage: python gen_unsup_robust.py for | biodiv
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score, jaccard_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from collections import Counter
import gen_score as G

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"
SEEDS = list(range(15))


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


def boot_ci(vals, n=2000, seed=0):
    rng = np.random.default_rng(seed); vals = np.asarray(vals)
    bs = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def arm(X, y, K, pca_dim=None):
    per = {m: [] for m in ["macroF1", "OA", "mIoU", "NMI"]}
    for s in SEEDS:
        r = metrics_for_seed(X, y, K, s, pca_dim)
        for m in per: per[m].append(r[m])
    return {m: {"mean": round(float(np.mean(per[m])), 4), "std": round(float(np.std(per[m])), 4),
                "ci": boot_ci(per[m])} for m in per}, per


def main():
    which = sys.argv[1]
    if which == "for":
        d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
        g = np.load(ROOT / "data/raw/for_species20k/geo_features_matched.npz", allow_pickle=True)
        Xc = d["X"]; y = d["species"].astype(str); groups = d["dataset"].astype(str); Xg = g["X_sf"]
        name = "FOR-species20K"; fag = None
    else:
        d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
        Xc = d["X_clip"]; Xg = d["X_sf"]; y = d["species"].astype(str); groups = d["region"].astype(str)
        name = "BioDiv-3DTrees-ULS"; fag = 0.39
    m, _ = G.analyzable_mask(y, groups); Xc, Xg, y, groups = Xc[m], Xg[m], y[m], groups[m]
    if fag is not None:
        c = Counter(y); top = max(c, key=c.get); n = len(y)
        if c[top] / n > fag:
            tgt = int(fag / (1 - fag) * (n - c[top])); idx = np.where(y == top)[0]
            drop = set(np.random.RandomState(0).choice(idx, c[top] - tgt, replace=False))
            k = np.array([i not in drop for i in range(n)]); Xc, Xg, y = Xc[k], Xg[k], y[k]
    K = len(set(y)); gdim = Xg.shape[1]
    res = {"dataset": name, "K": K, "n": int(len(y)), "n_seeds": len(SEEDS), "geo_dim": int(gdim), "clip_dim": int(Xc.shape[1])}
    clip512, clip512_per = arm(Xc, y, K)
    clipmatched, clipm_per = arm(Xc, y, K, pca_dim=gdim)
    geo, geo_per = arm(Xg, y, K)
    res["CLIP_512"] = clip512
    res["CLIP_matched_%dd" % gdim] = clipmatched
    res["geo_%dd" % gdim] = geo
    # delta distributions (paired across seeds) + CI, for each metric: CLIP512-geo AND CLIPmatched-geo
    res["delta_512_minus_geo"] = {}; res["delta_matched_minus_geo"] = {}
    for met in ["macroF1", "OA", "mIoU", "NMI"]:
        d512 = np.array(clip512_per[met]) - np.array(geo_per[met])
        dm = np.array(clipm_per[met]) - np.array(geo_per[met])
        res["delta_512_minus_geo"][met] = {"mean": round(float(d512.mean()), 4), "ci": boot_ci(d512),
                                           "frac_positive": round(float((d512 > 0).mean()), 3)}
        res["delta_matched_minus_geo"][met] = {"mean": round(float(dm.mean()), 4), "ci": boot_ci(dm),
                                               "frac_positive": round(float((dm > 0).mean()), 3)}
    json.dump(res, open(OUT / f"unsup_robust_{which}.json", "w"), indent=2)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
