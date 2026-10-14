"""DEPLOYABLE unsupervised score (Lu's abstract step 4: '每堆标几棵即可'), under the SAME
leakage-free spatial CV as the supervised run — so it IS comparable to supervised.

Per fold (leave-one-block-out):
  1. K-means on TRAIN features only (k = #species in train).
  2. Label each cluster by MAJORITY VOTE of its TRAIN members (uses only a few train labels — the
     'label a few per cluster' idea). Freeze the cluster->species map.
  3. Assign each TEST tree to nearest train centroid -> predict via the frozen map. (TEST labels NEVER used.)
  4. Shared-species scoring; pooled macro-F1 across folds.
This is closed-book + spatially isolated, so directly comparable to the supervised linear probe.

Usage: python gen_unsup_deployable.py for | biodiv
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"


def analyzable_mask(y, groups, min_n=30, min_blocks=2):
    keep = [s for s in set(y) if (y == s).sum() >= min_n and len(set(groups[y == s])) >= min_blocks]
    return np.isin(y, keep), sorted(keep)


def deployable_fold(X, y, groups, g, seed=0):
    te = groups == g; tr = ~te
    train_sp = set(y[tr]); keep = te & np.isin(y, list(train_sp))
    if keep.sum() == 0 or len(train_sp) < 2: return None
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[keep])
    k = len(train_sp)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Xtr)
    cl_tr = km.labels_
    ytr = y[tr]
    # label each cluster by majority vote of TRAIN members (a few labels per cluster)
    cmap = {}
    for c in range(k):
        members = ytr[cl_tr == c]
        cmap[c] = Counter(members).most_common(1)[0][0] if len(members) else list(train_sp)[0]
    # assign test to nearest centroid, predict via frozen map (TEST labels unused)
    cl_te = km.predict(Xte)
    pred = np.array([cmap[c] for c in cl_te])
    return f1_score(y[keep], pred, average="macro", zero_division=0), y[keep], pred


def run(X, y, groups):
    yt_all, yp_all, per = [], [], []
    for g in sorted(set(groups)):
        r = deployable_fold(X, y, groups, g)
        if r is None: continue
        f1, yt, yp = r; per.append(round(f1, 4)); yt_all.extend(yt.tolist()); yp_all.extend(yp.tolist())
    yt = np.array(yt_all); yp = np.array(yp_all)
    return {"pooled_f1": round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
            "pooled_oa": round(float(accuracy_score(yt, yp)), 4), "per_fold": per}


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
           "protocol": "DEPLOYABLE unsupervised: K-means on train, cluster labelled by train majority, "
                       "predict test via frozen map (closed-book, spatial CV) — comparable to supervised"}
    res["CLIP_deployable_unsup"] = run(Xc, y, groups)
    res["geo_sf_deployable_unsup"] = run(Xsf, y, groups)
    vals, cnts = np.unique(y, return_counts=True); maj = vals[cnts.argmax()]
    res["majority_f1"] = round(float(f1_score(y, np.array([maj]*len(y)), average="macro", zero_division=0)), 4)
    json.dump(res, open(OUT / f"unsup_deployable_{which}.json", "w"), indent=2)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
