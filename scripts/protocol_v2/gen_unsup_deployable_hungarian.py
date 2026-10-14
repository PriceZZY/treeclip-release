"""FAIRER deployable score: same closed-book, spatially-isolated protocol as gen_unsup_deployable.py,
but the cluster->species map is built by an OPTIMAL ONE-TO-ONE (Hungarian) assignment on the TRAIN
confusion matrix instead of per-cluster majority vote.

Why: train-majority labeling is many-to-one, so several clusters can map to the same species and some
species end up with NO cluster -> 0 recall -> macro-F1 collapses (a known harshness, flagged by the
corresponding author). One-to-one assignment (k = #train species, square) forces every train species to
receive exactly one cluster, removing the 'species with no cluster' artifact. It STILL uses only TRAIN
labels (Hungarian is solved on the train confusion matrix), so it remains closed-book / deployable and
the TEST labels are never consulted.

This is the deployable analogue of the standard ACC-Hungarian metric, but with the matching learned on
TRAIN (honest) rather than on TEST (the upper bound). Output: runs/protocol_v2/gen/unsup_deployable_hungarian_{which}.json
Usage: python gen_unsup_deployable_hungarian.py for | biodiv
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from scipy.optimize import linear_sum_assignment
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"


def analyzable_mask(y, groups, min_n=30, min_blocks=2):
    keep = [s for s in set(y) if (y == s).sum() >= min_n and len(set(groups[y == s])) >= min_blocks]
    return np.isin(y, keep), sorted(keep)


def deployable_fold(X, y, groups, g, seed=0):
    te = groups == g; tr = ~te
    train_sp = sorted(set(y[tr])); keep = te & np.isin(y, train_sp)
    if keep.sum() == 0 or len(train_sp) < 2: return None
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[keep])
    k = len(train_sp)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Xtr)
    cl_tr = km.labels_; ytr = y[tr]
    sp_idx = {s: i for i, s in enumerate(train_sp)}
    # TRAIN confusion matrix: clusters x species (counts), then optimal one-to-one assignment
    conf = np.zeros((k, k), dtype=float)
    for c, s in zip(cl_tr, ytr):
        conf[c, sp_idx[s]] += 1
    row, col = linear_sum_assignment(-conf)        # maximize train agreement, one-to-one
    cmap = {int(r): train_sp[int(cc)] for r, cc in zip(row, col)}
    cl_te = km.predict(Xte)
    pred = np.array([cmap[int(c)] for c in cl_te])
    return f1_score(y[keep], pred, average="macro", zero_division=0), accuracy_score(y[keep], pred), y[keep], pred


def run(X, y, groups):
    yt_all, yp_all, per = [], [], []
    for g in sorted(set(groups)):
        r = deployable_fold(X, y, groups, g)
        if r is None: continue
        f1, oa, yt, yp = r; per.append(round(f1, 4)); yt_all.extend(yt.tolist()); yp_all.extend(yp.tolist())
    yt = np.array(yt_all); yp = np.array(yp_all)
    return {"pooled_f1": round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
            "pooled_oa": round(float(accuracy_score(yt, yp)), 4), "per_fold_f1": per}


def main():
    which = sys.argv[1]
    if which == "for":
        d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
        Xc = d["X"]; y = d["species"].astype(str); groups = d["dataset"].astype(str)
        gp = ROOT / "data/raw/for_species20k/geo_features_matched.npz"
        if not gp.exists(): gp = ROOT / "data/raw/for_species20k/geo_features.npz"
        Xsf = np.load(gp, allow_pickle=True)["X_sf"]; name = "FOR-species20K"; fagus = None
    else:
        d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
        Xc = d["X_clip"]; Xsf = d["X_sf"]; y = d["species"].astype(str); groups = d["region"].astype(str)
        name = "BioDiv-3DTrees-ULS"; fagus = 0.39

    mask, analyz = analyzable_mask(y, groups)
    Xc, Xsf, y, groups = Xc[mask], Xsf[mask], y[mask], groups[mask]
    if fagus is not None:
        c = Counter(y); top = max(c, key=c.get); n = len(y)
        if c[top] / n > fagus:
            tgt = int(fagus / (1 - fagus) * (n - c[top])); idx = np.where(y == top)[0]
            drop = set(np.random.RandomState(0).choice(idx, c[top] - tgt, replace=False))
            km_ = np.array([i not in drop for i in range(n)])
            Xc, Xsf, y, groups = Xc[km_], Xsf[km_], y[km_], groups[km_]

    res = {"dataset": name, "n": int(len(y)), "k_species": len(set(y)), "n_blocks": len(set(groups)),
           "protocol": "DEPLOYABLE unsup, TRAIN-Hungarian one-to-one cluster->species map (closed-book, spatial CV); "
                       "fairer than train-majority (forces species coverage), still uses TRAIN labels only"}
    res["CLIP_deployable_hungarian"] = run(Xc, y, groups)
    res["geo_sf_deployable_hungarian"] = run(Xsf, y, groups)
    vals, cnts = np.unique(y, return_counts=True); maj = vals[cnts.argmax()]
    res["majority_f1"] = round(float(f1_score(y, np.array([maj]*len(y)), average="macro", zero_division=0)), 4)
    json.dump(res, open(OUT / f"unsup_deployable_hungarian_{which}.json", "w"), indent=2)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
