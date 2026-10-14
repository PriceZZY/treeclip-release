"""Data-level permutation test for the DINOv2-vs-geometry NMI gap (mirrors gen_permutation_test.py
but for the DINOv2 backbone). Observed statistic = NMI(DINOv2 clustering, labels) - NMI(geo clustering,
labels) at matched 11-d global clustering; null permutes species labels. DINOv2 aligned by treeID.
Output: runs/protocol_v2/gen/perm_test_dino.json
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as NMI
import gen_score as G
from collections import Counter
from gen_permutation_test import cluster_11d, N_PERM

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"


def align(tref, tsrc, Xsrc):
    idx = {str(t): i for i, t in enumerate(tsrc)}
    rows = np.array([idx.get(str(t), -1) for t in tref])
    return Xsrc[rows], int((rows < 0).sum())


def main():
    out = {"n_perm": N_PERM, "note": "permute species labels; stat = NMI(DINOv2)-NMI(geo) at matched 11-d global clustering"}
    for which in ["for", "biodiv"]:
        if which == "for":
            d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
            g = np.load(ROOT / "data/raw/for_species20k/geo_features_matched.npz", allow_pickle=True)
            dn = np.load(ROOT / "data/raw/for_species20k/for_dino.npz", allow_pickle=True)
            y = d["species"].astype(str); groups = d["dataset"].astype(str); Xsf = g["X_sf"]; tref = d["treeID"]
        else:
            d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
            dn = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_dino.npz", allow_pickle=True)
            Xsf = d["X_sf"]; y = d["species"].astype(str); groups = d["region"].astype(str); tref = d["treeID"]
        Xd, miss = align(tref, dn["treeID"], dn["X_dino"])
        assert miss == 0, f"{which}: {miss} trees missing from DINOv2 file"
        m, _ = G.analyzable_mask(y, groups); Xd, Xsf, y, groups = Xd[m], Xsf[m], y[m], groups[m]
        if which == "biodiv":
            c = Counter(y); top = max(c, key=c.get); n = len(y); cap = 0.39
            if c[top] / n > cap:
                tgt = int(cap/(1-cap)*(n-c[top])); idx = np.where(y == top)[0]
                drop = set(np.random.RandomState(0).choice(idx, c[top]-tgt, replace=False))
                k = np.array([i not in drop for i in range(n)]); Xd, Xsf, y = Xd[k], Xsf[k], y[k]
        K = len(set(y))
        cl_dino = cluster_11d(Xd, K); cl_geo = cluster_11d(Xsf, K)
        obs = NMI(y, cl_dino) - NMI(y, cl_geo)
        rng = np.random.default_rng(0); ge = 0; null = []
        for _ in range(N_PERM):
            yp = rng.permutation(y)
            nd = NMI(yp, cl_dino) - NMI(yp, cl_geo)
            null.append(nd)
            if nd >= obs: ge += 1
        out[which] = {"observed_delta_NMI": round(float(obs), 4),
                      "null_delta_mean": round(float(np.mean(null)), 5),
                      "null_delta_p95": round(float(np.percentile(null, 95)), 5),
                      "p_value": round(float((ge + 1) / (N_PERM + 1)), 5), "K": K, "n": int(len(y))}
    json.dump(out, open(OUT / "perm_test_dino.json", "w"), indent=2)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
