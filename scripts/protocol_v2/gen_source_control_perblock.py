"""Per-block within-source NMI + leave-one-source-out (LOO) sensitivity of the
within-source CLIP-minus-geo residual (PFG round-1 revision, plan item B2/#19;
GOAL_R1_REVISION red line 4).

Reuses the exact block loop, skip rule (<2 species or <4 trees), PCA-11 matching and
15-seed protocol of gen_source_control_robust.py, but DUMPS per-block values and the
LOO statistics of the size-weighted mean delta. Sanity: the reproduced weighted delta
must match source_control_robust.json (0.0436 FOR / 0.2039 BioDiv) to 1e-3.

Outputs runs/protocol_v2/gen/source_control_perblock.json. CPU-only.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score as NMI
import gen_score as G
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"
SEEDS = list(range(15))


def block_nmi(Xb, yb, seed, pca_dim):
    Xs = StandardScaler().fit_transform(Xb)
    if pca_dim is not None:
        k = min(pca_dim, Xs.shape[1], Xs.shape[0] - 1)
        Xs = PCA(n_components=k, random_state=seed).fit_transform(Xs)
    cl = KMeans(n_clusters=len(set(yb)), random_state=seed, n_init=10).fit_predict(Xs)
    return NMI(yb, cl)


def main():
    out = {"n_seeds": len(SEEDS)}
    for which in ["for", "biodiv"]:
        if which == "for":
            d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
            g = np.load(ROOT / "data/raw/for_species20k/geo_features_matched.npz", allow_pickle=True)
            Xc = d["X"]; y = d["species"].astype(str); groups = d["dataset"].astype(str); Xsf = g["X_sf"]
        else:
            d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
            Xc = d["X_clip"]; Xsf = d["X_sf"]; y = d["species"].astype(str); groups = d["region"].astype(str)
        m, _ = G.analyzable_mask(y, groups); Xc, Xsf, y, groups = Xc[m], Xsf[m], y[m], groups[m]
        if which == "biodiv":
            c = Counter(y); top = max(c, key=c.get); n = len(y); cap = 0.39
            if c[top] / n > cap:
                tgt = int(cap/(1-cap)*(n-c[top])); idx = np.where(y == top)[0]
                drop = set(np.random.RandomState(0).choice(idx, c[top]-tgt, replace=False))
                k = np.array([i not in drop for i in range(n)]); Xc, Xsf, y, groups = Xc[k], Xsf[k], y[k], groups[k]

        blocks = []
        for gname in sorted(set(groups)):
            mask = groups == gname; yy = y[mask]
            if len(set(yy)) < 2 or mask.sum() < 4:
                blocks.append(dict(block=gname, n=int(mask.sum()), K=len(set(yy)),
                                   skipped=True)); continue
            cv = float(np.mean([block_nmi(Xc[mask], yy, s, 11) for s in SEEDS]))
            gv = float(np.mean([block_nmi(Xsf[mask], yy, s, None) for s in SEEDS]))
            blocks.append(dict(block=gname, n=int(mask.sum()), K=len(set(yy)), skipped=False,
                               clip=round(cv, 4), geo=round(gv, 4), delta=round(cv - gv, 4)))
            print(f"{which}/{gname}: n={mask.sum()} K={len(set(yy))} clip={cv:.4f} geo={gv:.4f} d={cv-gv:+.4f}", flush=True)

        used = [b for b in blocks if not b["skipped"]]
        w = np.array([b["n"] for b in used], float)
        dl = np.array([b["delta"] for b in used], float)
        weighted = float((w * dl).sum() / w.sum())
        # LOO of the size-weighted mean
        loo = []
        for i, b in enumerate(used):
            k2 = np.ones(len(used), bool); k2[i] = False
            loo.append(dict(left_out=b["block"],
                            weighted_delta=round(float((w[k2]*dl[k2]).sum()/w[k2].sum()), 4)))
        loo_vals = [x["weighted_delta"] for x in loo]
        out[which] = dict(
            n_blocks_total=len(blocks), n_blocks_used=len(used),
            weighted_delta=round(weighted, 4),
            unweighted_delta=round(float(dl.mean()), 4),
            n_blocks_delta_positive=int((dl > 0).sum()),
            per_block=blocks,
            loo=loo,
            loo_min=min(loo_vals), loo_max=max(loo_vals),
            loo_min_leftout=loo[int(np.argmin(loo_vals))]["left_out"],
            largest_block=max(used, key=lambda b: b["n"])["block"],
        )
    json.dump(out, open(OUT / "source_control_perblock.json", "w"), indent=2)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("per_block", "loo")}
                      for k, v in out.items() if isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
