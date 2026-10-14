"""Unsupervised (Lu's original TreeCLIP idea) re-run under clean protocol, on the GEN datasets.

Unsupervised = NO labels used for training. K-means (k = #analyzable species) on CLIP features
(and geo features for comparison), then Hungarian-match clusters->species, report macro-F1 + OA.
Labels are used ONLY for the post-hoc Hungarian matching + scoring (this is the standard, but
OPTIMISTIC, clustering-accuracy protocol — clusters are assigned the best-matching species using
the test labels themselves; report honestly as an upper bound, not a deployable classifier).

Two views (since unsupervised has no train/test):
  (1) GLOBAL: cluster all analyzable trees at once (matches Lu's original setup).
  (2) PER-BLOCK: cluster within each spatial block separately, pooled macro-F1 (checks whether the
      "natural clustering" is stable across sites or driven by site/source structure).
Same analyzable subset (>=30 trees & >=2 blocks) + BioDiv Fagus downsample as the supervised run.

Usage: python gen_unsup.py for | biodiv
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"; OUT.mkdir(parents=True, exist_ok=True)


def hungarian_f1(y_true, clusters):
    sp = sorted(set(y_true)); cl = sorted(set(clusters))
    cost = np.zeros((len(cl), len(sp)))
    for i, c in enumerate(cl):
        for j, s in enumerate(sp):
            cost[i, j] = -np.sum((clusters == c) & (y_true == s))
    r, cc = linear_sum_assignment(cost)
    m = {cl[r[i]]: sp[cc[i]] for i in range(len(r))}
    ym = np.array([m.get(c, sp[0]) for c in clusters])
    return (round(float(f1_score(y_true, ym, average="macro", zero_division=0)), 4),
            round(float(accuracy_score(y_true, ym)), 4),
            round(float(normalized_mutual_info_score(y_true, clusters)), 4))


def cluster_global(X, y, k, seed=0):
    Xs = StandardScaler().fit_transform(X)
    cl = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(Xs)
    return hungarian_f1(y, cl)


def cluster_per_block(X, y, groups, k_per, seed=0):
    yt_all, ym_all = [], []
    for g in sorted(set(groups)):
        m = groups == g
        yy = y[m]
        if len(set(yy)) < 2: continue
        kk = min(k_per, len(set(yy)), m.sum())
        Xs = StandardScaler().fit_transform(X[m])
        cl = KMeans(n_clusters=kk, random_state=seed, n_init=10).fit_predict(Xs)
        # hungarian within block
        sp = sorted(set(yy)); cls = sorted(set(cl))
        cost = np.zeros((len(cls), len(sp)))
        for i, c in enumerate(cls):
            for j, s in enumerate(sp):
                cost[i, j] = -np.sum((cl == c) & (yy == s))
        rr, ccc = linear_sum_assignment(cost)
        mp = {cls[rr[i]]: sp[ccc[i]] for i in range(len(rr))}
        ym = np.array([mp.get(c, sp[0]) for c in cl])
        yt_all.extend(yy.tolist()); ym_all.extend(ym.tolist())
    yt = np.array(yt_all); ym = np.array(ym_all)
    return (round(float(f1_score(yt, ym, average="macro", zero_division=0)), 4),
            round(float(accuracy_score(yt, ym)), 4))


def analyzable_mask(y, groups, min_n=30, min_blocks=2):
    keep = [s for s in set(y) if (y == s).sum() >= min_n and len(set(groups[y == s])) >= min_blocks]
    return np.isin(y, keep), sorted(keep)


def main():
    which = sys.argv[1]
    if which == "for":
        d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
        Xc = d["X"]; y = d["species"].astype(str); groups = d["dataset"].astype(str)
        gp = ROOT / "data/raw/for_species20k/geo_features_matched.npz"
        if not gp.exists(): gp = ROOT / "data/raw/for_species20k/geo_features.npz"
        g = np.load(gp, allow_pickle=True); Xsf = g["X_sf"]
        name = "FOR-species20K"; fagus_cap = None
    else:
        d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
        Xc = d["X_clip"]; Xsf = d["X_sf"]; y = d["species"].astype(str); groups = d["region"].astype(str)
        name = "BioDiv-3DTrees-ULS"; fagus_cap = 0.39

    mask, analyzable = analyzable_mask(y, groups)
    Xc, Xsf, y, groups = Xc[mask], Xsf[mask], y[mask], groups[mask]
    if fagus_cap is not None:
        from collections import Counter
        c = Counter(y); top = max(c, key=c.get); n = len(y)
        if c[top] / n > fagus_cap:
            tgt = int(fagus_cap / (1 - fagus_cap) * (n - c[top])); idx = np.where(y == top)[0]
            drop = set(np.random.RandomState(0).choice(idx, c[top] - tgt, replace=False))
            km = np.array([i not in drop for i in range(n)])
            Xc, Xsf, y, groups = Xc[km], Xsf[km], y[km], groups[km]

    k = len(set(y))
    res = {"dataset": name, "n": int(len(y)), "k_species": k, "n_blocks": len(set(groups)),
           "note": "unsupervised K-means + Hungarian match; labels used ONLY for post-hoc matching (optimistic upper bound)"}
    # majority baseline macro-F1 (predict the single most frequent species)
    vals, cnts = np.unique(y, return_counts=True); maj = vals[cnts.argmax()]
    res["majority_f1"] = round(float(f1_score(y, np.array([maj] * len(y)), average="macro", zero_division=0)), 4)

    cf, co, cn = cluster_global(Xc, y, k)
    gf, go, gn = cluster_global(Xsf, y, k)
    res["CLIP_global"] = {"macro_f1": cf, "oa": co, "nmi": cn}
    res["geo_sf_global"] = {"macro_f1": gf, "oa": go, "nmi": gn}
    pcf, pco = cluster_per_block(Xc, y, groups, k)
    pgf, pgo = cluster_per_block(Xsf, y, groups, k)
    res["CLIP_per_block"] = {"macro_f1": pcf, "oa": pco}
    res["geo_sf_per_block"] = {"macro_f1": pgf, "oa": pgo}

    json.dump(res, open(OUT / f"unsup_{which}.json", "w"), indent=2)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
