"""Regenerate supervised_matched_sf.json at the current feature set: supervised leave-source-out
macro-F1 of CLIP (PCA-reduced to 11-d, train-only) vs the 11-d scale-free geo_sf, BOTH through the
SAME logistic head (apples-to-apples), on the analyzable subset with BioDiv Fagus<39% cap. Reuses
gen_score's exact machinery (analyzable_mask, fold_score with train-only PCA, run_method, LIN) so
the protocol is identical to the headline supervised arm. View-count change propagates through the
CLIP features automatically.

Output: runs/protocol_v2/gen/supervised_matched_sf.json {for:{CLIP_pca11_macroF1, geo_sf11_macroF1,
delta_matched_sf}, biodiv:{...}}
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from collections import Counter
import gen_score as G

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"


def load(which):
    if which == "for":
        d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
        g = np.load(ROOT / "data/raw/for_species20k/geo_features_matched.npz", allow_pickle=True)
        Xc = d["X"]; y = d["species"].astype(str); groups = d["dataset"].astype(str); Xsf = g["X_sf"]
        fag = None
    else:
        d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
        Xc = d["X_clip"]; Xsf = d["X_sf"]; y = d["species"].astype(str); groups = d["region"].astype(str)
        fag = 0.39
    mask, _ = G.analyzable_mask(y, groups)
    Xc, Xsf, y, groups = Xc[mask], Xsf[mask], y[mask], groups[mask]
    if fag is not None:
        c = Counter(y); top = max(c, key=c.get); n = len(y)
        if c[top] / n > fag:
            tgt = int(fag / (1 - fag) * (n - c[top])); idx = np.where(y == top)[0]
            drop = set(np.random.RandomState(0).choice(idx, c[top] - tgt, replace=False))
            keep = np.array([i not in drop for i in range(n)])
            Xc, Xsf, y, groups = Xc[keep], Xsf[keep], y[keep], groups[keep]
    return Xc, Xsf, y, groups


def main():
    out = {}
    for which in ["for", "biodiv"]:
        Xc, Xsf, y, groups = load(which)
        clip, _ = G.run_method(Xc, y, groups, G.LIN, pca_dim=11)   # CLIP PCA->11d (train-only), LIN head
        geo, _ = G.run_method(Xsf, y, groups, G.RF)                # geo_sf 11-d via RF = geo_sf_rf headline baseline
                                                                    # (matches original supervised_matched_sf: CLIP-LIN vs geo_sf-RF)
        out[which] = {"CLIP_pca11_macroF1": clip["pooled_f1"],
                      "geo_sf11_macroF1": geo["pooled_f1"],
                      "delta_matched_sf": round(clip["pooled_f1"] - geo["pooled_f1"], 4)}
    json.dump(out, open(OUT / "supervised_matched_sf.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
