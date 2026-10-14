"""Within-source control, multi-backbone (CLIP / DINOv2 / ImageNet / random / geo), reusing the
exact arm_ws primitive (per-block KMeans at matched 11-d, size-weighted NMI, 15 seeds, bootstrap CI).
All backbones aligned by treeID; same analyzable subset + BioDiv Fagus<39%.
Output: runs/protocol_v2/gen/source_control_multi_{which}.json
Usage: python gen_source_control_multi.py for | biodiv
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from collections import Counter
import gen_score as G
from gen_source_control_robust import arm_ws, ci, within_species_source

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
        dn = np.load(ROOT / "data/raw/for_species20k/for_dino.npz", allow_pickle=True)
        ex = np.load(ROOT / "data/raw/for_species20k/for_extra.npz", allow_pickle=True)
        Xc = d["X"]; y = d["species"].astype(str); groups = d["dataset"].astype(str); Xg = g["X_sf"]; tref = d["treeID"]; fag = None
    else:
        d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
        dn = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_dino.npz", allow_pickle=True)
        ex = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_extra.npz", allow_pickle=True)
        Xc = d["X_clip"]; Xg = d["X_sf"]; y = d["species"].astype(str); groups = d["region"].astype(str); tref = d["treeID"]; fag = 0.39

    Xd, m1 = align(tref, dn["treeID"], dn["X_dino"])
    Xr, m2 = align(tref, ex["treeID"], ex["X_rand"])
    Xi, m3 = align(tref, ex["treeID"], ex["X_imnet"])
    assert m1 == m2 == m3 == 0
    mask, _ = G.analyzable_mask(y, groups)
    Xc, Xg, Xd, Xr, Xi = Xc[mask], Xg[mask], Xd[mask], Xr[mask], Xi[mask]; y, groups = y[mask], groups[mask]
    if fag is not None:
        c = Counter(y); top = max(c, key=c.get); n = len(y)
        if c[top] / n > fag:
            tgt = int(fag / (1 - fag) * (n - c[top])); idx = np.where(y == top)[0]
            drop = set(np.random.RandomState(0).choice(idx, c[top] - tgt, replace=False))
            k = np.array([i not in drop for i in range(n)])
            Xc, Xg, Xd, Xr, Xi, y, groups = Xc[k], Xg[k], Xd[k], Xr[k], Xi[k], y[k], groups[k]

    geo_v = arm_ws(Xg, y, groups, pca_dim=None)
    out = {"dataset": which, "n_blocks": len(set(groups)), "n_seeds": 15,
           "within_source_NMI_geo_11d_mean": round(float(np.mean(geo_v)), 4)}
    for name, X in [("CLIP", Xc), ("DINOv2", Xd), ("ImageNet", Xi), ("random", Xr)]:
        v = arm_ws(X, y, groups, pca_dim=11)
        dlt = np.array(v) - np.array(geo_v)
        out[name] = {
            "within_source_NMI_mean": round(float(np.mean(v)), 4),
            "minus_geo_mean": round(float(dlt.mean()), 4),
            "minus_geo_ci95": ci(dlt),
            "minus_geo_frac_positive": round(float((dlt > 0).mean()), 3),
            "within_species_source_balacc": within_species_source(X, y, groups, pca_dim=11)[0],
        }
    out["within_species_source_balacc_geo"] = within_species_source(Xg, y, groups, pca_dim=None)[0]
    json.dump(out, open(OUT / f"source_control_multi_{which}.json", "w"), indent=2)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
