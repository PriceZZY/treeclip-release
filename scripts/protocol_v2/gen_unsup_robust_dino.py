"""Three-way unsupervised comparison: CLIP / DINOv2 / geometry, reusing the EXACT arm()
protocol from gen_unsup_robust.py (15 seeds; StandardScaler -> optional PCA(pca_dim) ->
KMeans(K, n_init=10) -> Hungarian; NMI/macroF1/OA/mIoU), the SAME analyzable subset, and the
SAME BioDiv Fagus<39% down-sample. DINOv2 rows are aligned to the CLIP/geo feature file by
treeID, so all three feature sets describe identical trees.

Adds DINOv2 as a frozen, purely self-supervised (no-text, no-label) backbone: a positive
DINOv2-vs-geo result makes the unsupervised claim defensible and shows the advantage is not
CLIP-specific.

Usage: python gen_unsup_robust_dino.py for | biodiv
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from collections import Counter
import gen_score as G
from gen_unsup_robust import arm, boot_ci  # identical eval primitives

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"


def align_by_treeid(tref, tsrc, Xsrc):
    idx = {str(t): i for i, t in enumerate(tsrc)}
    rows = np.array([idx.get(str(t), -1) for t in tref])
    return rows, int((rows < 0).sum())


def deltas(a_per, b_per):
    o = {}
    for met in ["macroF1", "OA", "mIoU", "NMI"]:
        dd = np.array(a_per[met]) - np.array(b_per[met])
        o[met] = {"mean": round(float(dd.mean()), 4), "ci": boot_ci(dd),
                  "frac_positive": round(float((dd > 0).mean()), 3)}
    return o


def main():
    which = sys.argv[1]
    if which == "for":
        d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
        g = np.load(ROOT / "data/raw/for_species20k/geo_features_matched.npz", allow_pickle=True)
        dn = np.load(ROOT / "data/raw/for_species20k/for_dino.npz", allow_pickle=True)
        Xc = d["X"]; y = d["species"].astype(str); groups = d["dataset"].astype(str); Xg = g["X_sf"]
        tref = d["treeID"]; name = "FOR-species20K"; fag = None
    else:
        d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
        dn = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_dino.npz", allow_pickle=True)
        Xc = d["X_clip"]; Xg = d["X_sf"]; y = d["species"].astype(str); groups = d["region"].astype(str)
        tref = d["treeID"]; name = "BioDiv-3DTrees-ULS"; fag = 0.39

    rows, miss = align_by_treeid(tref, dn["treeID"], dn["X_dino"])
    assert (rows >= 0).all(), f"{miss}/{len(tref)} reference trees missing from DINOv2 file; cannot align"
    Xd = dn["X_dino"][rows]

    m, _ = G.analyzable_mask(y, groups)
    Xc, Xg, Xd, y, groups = Xc[m], Xg[m], Xd[m], y[m], groups[m]
    if fag is not None:
        c = Counter(y); top = max(c, key=c.get); n = len(y)
        if c[top] / n > fag:
            tgt = int(fag / (1 - fag) * (n - c[top])); idx = np.where(y == top)[0]
            drop = set(np.random.RandomState(0).choice(idx, c[top] - tgt, replace=False))
            k = np.array([i not in drop for i in range(n)]); Xc, Xg, Xd, y = Xc[k], Xg[k], Xd[k], y[k]

    K = len(set(y)); gdim = Xg.shape[1]
    res = {"dataset": name, "K": K, "n": int(len(y)), "n_seeds": 15, "geo_dim": int(gdim),
           "clip_dim": int(Xc.shape[1]), "dino_dim": int(Xd.shape[1]), "dino_missing": int(miss)}

    clip_n, clip_n_per = arm(Xc, y, K)
    clip_m, clip_m_per = arm(Xc, y, K, pca_dim=gdim)
    dino_n, dino_n_per = arm(Xd, y, K)
    dino_m, dino_m_per = arm(Xd, y, K, pca_dim=gdim)
    geo, geo_per = arm(Xg, y, K)

    res["CLIP_native"] = clip_n
    res["CLIP_matched_%dd" % gdim] = clip_m
    res["DINOv2_native"] = dino_n
    res["DINOv2_matched_%dd" % gdim] = dino_m
    res["geo_%dd" % gdim] = geo
    res["delta_DINO_matched_minus_geo"] = deltas(dino_m_per, geo_per)
    res["delta_DINO_native_minus_geo"] = deltas(dino_n_per, geo_per)
    res["delta_DINO_matched_minus_CLIP_matched"] = deltas(dino_m_per, clip_m_per)
    res["delta_CLIP_matched_minus_geo"] = deltas(clip_m_per, geo_per)

    json.dump(res, open(OUT / f"unsup_robust_dino_{which}.json", "w"), indent=2)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
