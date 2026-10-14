"""Within-source control, three-way (CLIP / DINOv2 / geometry), reusing the EXACT primitives
of gen_source_control_robust.py (per-block KMeans at matched 11-d, size-weighted NMI, 15 seeds,
bootstrap CI; plus within-species source-recoverability). DINOv2 rows aligned to the CLIP/geo
feature file by treeID. Same analyzable subset + BioDiv Fagus<39% down-sample.

Output: runs/protocol_v2/gen/source_control_robust_dino.json
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from collections import Counter
import gen_score as G
from gen_source_control_robust import arm_ws, ci, within_species_source

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"


def align(tref, tsrc, Xsrc):
    idx = {str(t): i for i, t in enumerate(tsrc)}
    rows = np.array([idx.get(str(t), -1) for t in tref])
    return Xsrc[rows], int((rows < 0).sum())


def main():
    out = {"n_seeds": 15}
    for which in ["for", "biodiv"]:
        if which == "for":
            d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
            g = np.load(ROOT / "data/raw/for_species20k/geo_features_matched.npz", allow_pickle=True)
            dn = np.load(ROOT / "data/raw/for_species20k/for_dino.npz", allow_pickle=True)
            Xc = d["X"]; y = d["species"].astype(str); groups = d["dataset"].astype(str); Xsf = g["X_sf"]
            tref = d["treeID"]
        else:
            d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
            dn = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_dino.npz", allow_pickle=True)
            Xc = d["X_clip"]; Xsf = d["X_sf"]; y = d["species"].astype(str); groups = d["region"].astype(str)
            tref = d["treeID"]
        Xd, miss = align(tref, dn["treeID"], dn["X_dino"])
        assert miss == 0, f"{which}: {miss} trees missing from DINOv2 file"
        m, _ = G.analyzable_mask(y, groups); Xc, Xsf, Xd, y, groups = Xc[m], Xsf[m], Xd[m], y[m], groups[m]
        if which == "biodiv":
            c = Counter(y); top = max(c, key=c.get); n = len(y); cap = 0.39
            if c[top] / n > cap:
                tgt = int(cap/(1-cap)*(n-c[top])); idx = np.where(y == top)[0]
                drop = set(np.random.RandomState(0).choice(idx, c[top]-tgt, replace=False))
                k = np.array([i not in drop for i in range(n)]); Xc, Xsf, Xd, y, groups = Xc[k], Xsf[k], Xd[k], y[k], groups[k]
        clip_v = arm_ws(Xc, y, groups, pca_dim=11)
        dino_v = arm_ws(Xd, y, groups, pca_dim=11)
        geo_v = arm_ws(Xsf, y, groups, pca_dim=None)
        d_dino = np.array(dino_v) - np.array(geo_v)
        d_clip = np.array(clip_v) - np.array(geo_v)
        out[which] = {
            "n_blocks": len(set(groups)),
            "within_source_NMI_CLIP_11d_mean": round(float(np.mean(clip_v)), 4),
            "within_source_NMI_DINOv2_11d_mean": round(float(np.mean(dino_v)), 4),
            "within_source_NMI_geo_11d_mean": round(float(np.mean(geo_v)), 4),
            "within_source_DINO_minus_geo_mean": round(float(d_dino.mean()), 4),
            "within_source_DINO_minus_geo_ci95": ci(d_dino),
            "within_source_DINO_minus_geo_frac_positive": round(float((d_dino > 0).mean()), 3),
            "within_source_CLIP_minus_geo_mean": round(float(d_clip.mean()), 4),
            "within_species_source_balacc_DINOv2_11d": within_species_source(Xd, y, groups, pca_dim=11)[0],
            "within_species_source_balacc_CLIP_11d": within_species_source(Xc, y, groups, pca_dim=11)[0],
            "within_species_source_balacc_geo_11d": within_species_source(Xsf, y, groups, pca_dim=None)[0],
        }
    json.dump(out, open(OUT / "source_control_robust_dino.json", "w"), indent=2)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
