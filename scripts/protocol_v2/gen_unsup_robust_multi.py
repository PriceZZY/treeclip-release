"""Multi-backbone headline comparison: CLIP / DINOv2 / ImageNet-supervised / random-init / geometry,
reusing the EXACT arm() protocol (15 seeds; StandardScaler -> optional PCA -> KMeans(K,n_init=10) ->
Hungarian; NMI/macroF1/OA/mIoU). All backbones aligned to the CLIP/geo feature file by treeID; same
analyzable subset + BioDiv Fagus<39% down-sample.

random-init ViT is the causal control: if it sits near geometry (far below the pretrained backbones),
the advantage is due to PRE-TRAINING, not the render->ViT->cluster pipeline. ImageNet-supervised is a
third pre-training paradigm for the generality claim.

Usage: python gen_unsup_robust_multi.py for | biodiv
Output: runs/protocol_v2/gen/unsup_robust_multi_{which}.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from collections import Counter
import gen_score as G
from gen_unsup_robust import arm, boot_ci

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"


def align(tref, tsrc, X):
    idx = {str(t): i for i, t in enumerate(tsrc)}
    rows = np.array([idx.get(str(t), -1) for t in tref])
    return X[rows], int((rows < 0).sum())


def deltas(a_per, b_per):
    o = {}
    for m in ["macroF1", "OA", "mIoU", "NMI"]:
        dd = np.array(a_per[m]) - np.array(b_per[m])
        o[m] = {"mean": round(float(dd.mean()), 4), "ci": boot_ci(dd), "frac_positive": round(float((dd > 0).mean()), 3)}
    return o


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
    assert m1 == m2 == m3 == 0, f"alignment failed: dino {m1}, rand {m2}, imnet {m3}"

    mask, _ = G.analyzable_mask(y, groups)
    Xc, Xg, Xd, Xr, Xi = Xc[mask], Xg[mask], Xd[mask], Xr[mask], Xi[mask]; y, groups = y[mask], groups[mask]
    if fag is not None:
        c = Counter(y); top = max(c, key=c.get); n = len(y)
        if c[top] / n > fag:
            tgt = int(fag / (1 - fag) * (n - c[top])); idx = np.where(y == top)[0]
            drop = set(np.random.RandomState(0).choice(idx, c[top] - tgt, replace=False))
            k = np.array([i not in drop for i in range(n)])
            Xc, Xg, Xd, Xr, Xi, y = Xc[k], Xg[k], Xd[k], Xr[k], Xi[k], y[k]

    K = len(set(y)); gdim = Xg.shape[1]
    res = {"dataset": which, "K": K, "n": int(len(y)), "geo_dim": int(gdim), "n_seeds": 15,
           "backbones": {"CLIP": int(Xc.shape[1]), "DINOv2": int(Xd.shape[1]),
                         "ImageNet": int(Xi.shape[1]), "random": int(Xr.shape[1])}}
    geo, geo_per = arm(Xg, y, K)
    res["geo_%dd" % gdim] = geo
    per = {}
    for name, X in [("CLIP", Xc), ("DINOv2", Xd), ("ImageNet", Xi), ("random", Xr)]:
        nat, _ = arm(X, y, K)
        mat, mat_per = arm(X, y, K, pca_dim=gdim)
        res[name + "_native"] = nat
        res[name + "_matched_%dd" % gdim] = mat
        per[name] = mat_per
    for name in ["CLIP", "DINOv2", "ImageNet", "random"]:
        res["delta_%s_matched_minus_geo" % name] = deltas(per[name], geo_per)
    json.dump(res, open(OUT / f"unsup_robust_multi_{which}.json", "w"), indent=2)
    # concise print
    print(f"[{which}] K={K} n={len(y)}  NMI (matched {gdim}d):")
    for name in ["CLIP", "DINOv2", "ImageNet", "random"]:
        a = res[name + "_matched_%dd" % gdim]["NMI"]; dd = res["delta_%s_matched_minus_geo" % name]["NMI"]
        print(f"  {name:9s} {a['mean']:.3f}  (delta vs geo {dd['mean']:+.3f}, fracpos {dd['frac_positive']})")
    print(f"  geo       {geo['NMI']['mean']:.3f}")


if __name__ == "__main__":
    main()
