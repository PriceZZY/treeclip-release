"""Permutation test vs geometry for the two NEW backbones (ImageNet-supervised, random-init);
CLIP and DINOv2 already have perm_test.json / perm_test_dino.json. stat = NMI(backbone)-NMI(geo)
at matched 11-d global clustering; null permutes species labels. Aligned by treeID.
Output: runs/protocol_v2/gen/perm_test_multi_{which}.json
Usage: python gen_permutation_multi.py for | biodiv
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as NMI
import gen_score as G
from collections import Counter
from gen_permutation_test import cluster_11d, N_PERM

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"


def align(tref, tsrc, X):
    idx = {str(t): i for i, t in enumerate(tsrc)}
    rows = np.array([idx.get(str(t), -1) for t in tref])
    return X[rows], int((rows < 0).sum())


def main():
    which = sys.argv[1]
    if which == "for":
        d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
        g = np.load(ROOT / "data/raw/for_species20k/geo_features_matched.npz", allow_pickle=True)
        ex = np.load(ROOT / "data/raw/for_species20k/for_extra.npz", allow_pickle=True)
        y = d["species"].astype(str); groups = d["dataset"].astype(str); Xg = g["X_sf"]; tref = d["treeID"]; fag = None
    else:
        d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
        ex = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_extra.npz", allow_pickle=True)
        Xg = d["X_sf"]; y = d["species"].astype(str); groups = d["region"].astype(str); tref = d["treeID"]; fag = 0.39
    Xi, m1 = align(tref, ex["treeID"], ex["X_imnet"])
    Xr, m2 = align(tref, ex["treeID"], ex["X_rand"])
    assert m1 == m2 == 0
    mask, _ = G.analyzable_mask(y, groups); Xi, Xr, Xg, y, groups = Xi[mask], Xr[mask], Xg[mask], y[mask], groups[mask]
    if fag is not None:
        c = Counter(y); top = max(c, key=c.get); n = len(y)
        if c[top] / n > fag:
            tgt = int(fag / (1 - fag) * (n - c[top])); idx = np.where(y == top)[0]
            drop = set(np.random.RandomState(0).choice(idx, c[top] - tgt, replace=False))
            k = np.array([i not in drop for i in range(n)]); Xi, Xr, Xg, y = Xi[k], Xr[k], Xg[k], y[k]
    K = len(set(y)); cl_geo = cluster_11d(Xg, K)
    out = {"dataset": which, "n_perm": N_PERM, "K": K, "n": int(len(y))}
    for name, X in [("ImageNet", Xi), ("random", Xr)]:
        cl = cluster_11d(X, K); obs = NMI(y, cl) - NMI(y, cl_geo)
        rng = np.random.default_rng(0); ge = 0; null = []
        for _ in range(N_PERM):
            yp = rng.permutation(y); nd = NMI(yp, cl) - NMI(yp, cl_geo); null.append(nd)
            if nd >= obs: ge += 1
        out[name] = {"observed_delta_NMI": round(float(obs), 4), "null_delta_mean": round(float(np.mean(null)), 5),
                     "null_delta_p95": round(float(np.percentile(null, 95)), 5),
                     "p_value": round(float((ge + 1) / (N_PERM + 1)), 5)}
    json.dump(out, open(OUT / f"perm_test_multi_{which}.json", "w"), indent=2)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
