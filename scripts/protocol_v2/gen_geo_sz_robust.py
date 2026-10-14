"""B-2: size-AWARE geometry (geo_sz, 17-d X_sz) robustness row.

Rebuts "you crippled geometry by making it size-invariant (11-d)". Runs the repo's
already-cached size-aware Weinmann+size descriptor (X_sz, aka geo_sz_17d in
gen_score.py / rev_predict_source.py) through the SAME two harnesses as the headline:
  - GLOBAL 15-seed NMI  (arm() from gen_unsup_robust: StandardScaler->KMeans, native dim)
  - WITHIN-SOURCE NMI   (arm_ws() from gen_source_control_robust: per-block StandardScaler->KMeans, 15 seeds)
for BOTH datasets, beside CLIP and the size-invariant geo (X_sf, 11-d).

No new modeling choices: geometry variants use native dim (pca_dim=None), CLIP uses
PCA-11d for the matched/within-source arm exactly as the headline does.

Usage: python gen_geo_sz_robust.py
Output: runs/protocol_v2/gen/geo_sz_robust.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from collections import Counter
import gen_score as G
from gen_unsup_robust import arm
from gen_source_control_robust import arm_ws

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"


def load(which):
    if which == "for":
        d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
        g = np.load(ROOT / "data/raw/for_species20k/geo_features_matched.npz", allow_pickle=True)
        Xc = d["X"]; Xsf = g["X_sf"]; Xsz = g["X_sz"]
        y = d["species"].astype(str); groups = d["dataset"].astype(str); fag = None
    else:
        d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
        Xc = d["X_clip"]; Xsf = d["X_sf"]; Xsz = d["X_sz"]
        y = d["species"].astype(str); groups = d["region"].astype(str); fag = 0.39
    m, _ = G.analyzable_mask(y, groups); Xc, Xsf, Xsz, y, groups = Xc[m], Xsf[m], Xsz[m], y[m], groups[m]
    if fag is not None:
        c = Counter(y); top = max(c, key=c.get); n = len(y)
        if c[top] / n > fag:
            tgt = int(fag / (1 - fag) * (n - c[top])); idx = np.where(y == top)[0]
            drop = set(np.random.RandomState(0).choice(idx, c[top] - tgt, replace=False))
            k = np.array([i not in drop for i in range(n)])
            Xc, Xsf, Xsz, y, groups = Xc[k], Xsf[k], Xsz[k], y[k], groups[k]
    return Xc, Xsf, Xsz, y, groups


def run(which):
    Xc, Xsf, Xsz, y, groups = load(which)
    K = len(set(y))
    # GLOBAL 15-seed NMI (native dim for each feature; CLIP also PCA-matched to 11-d)
    clip_native = arm(Xc, y, K)[0]["NMI"]["mean"]
    clip_matched = arm(Xc, y, K, pca_dim=11)[0]["NMI"]["mean"]
    geo_sf_g = arm(Xsf, y, K)[0]["NMI"]["mean"]
    geo_sz_g = arm(Xsz, y, K)[0]["NMI"]["mean"]
    # WITHIN-SOURCE 15-seed NMI (geometry at native dim, CLIP at PCA-11d, matching the headline)
    clip_ws = float(np.mean(arm_ws(Xc, y, groups, pca_dim=11)))
    geo_sf_ws = float(np.mean(arm_ws(Xsf, y, groups, pca_dim=None)))
    geo_sz_ws = float(np.mean(arm_ws(Xsz, y, groups, pca_dim=None)))
    return {
        "K": K, "n": int(len(y)), "n_blocks": len(set(groups)),
        "geo_sz_dim": int(Xsz.shape[1]), "geo_sf_dim": int(Xsf.shape[1]), "n_seeds": 15,
        "global_NMI": {
            "CLIP_512": round(clip_native, 4),
            "CLIP_matched_11d": round(clip_matched, 4),
            "geo_sf_11d_size_invariant": round(geo_sf_g, 4),
            "geo_sz_17d_size_aware": round(geo_sz_g, 4),
        },
        "within_source_NMI": {
            "CLIP_11d": round(clip_ws, 4),
            "geo_sf_11d_size_invariant": round(geo_sf_ws, 4),
            "geo_sz_17d_size_aware": round(geo_sz_ws, 4),
        },
        "clip_leads_geo_sz_global": bool(clip_native > geo_sz_g),
        "clip_leads_geo_sz_within_source": bool(clip_ws > geo_sz_ws),
        "clip_minus_geo_sz_global": round(clip_native - geo_sz_g, 4),
        "clip_minus_geo_sz_within_source": round(clip_ws - geo_sz_ws, 4),
    }


def main():
    out = {"note": ("geo_sz = repo-cached size-AWARE descriptor (X_sz, 17-d; geo_sz_17d in "
                    "gen_score.py). Same 15-seed StandardScaler->KMeans harness as headline; "
                    "geometry at native dim, CLIP PCA-11d for matched/within-source arm.")}
    for which in ["for", "biodiv"]:
        out[which] = run(which)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "geo_sz_robust.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
