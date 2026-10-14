"""Regenerate rev_perclass_for.json at the current feature set: per-species supervised macro-F1
(leave-source-out) for CLIP-11d and geo_sf-11d through the SAME logistic head (apples-to-apples),
against each species' acquisition-source concentration share. Tests whether CLIP's per-species
advantage is negatively correlated with source-concentration (a partial-confound signature).
Reuses gen_score.fold_score (train-only PCA, same LIN). FOR-species20K only (block = source study).

Output rows: [species, source_share, CLIP_f1, geo_f1, delta(CLIP-geo)] sorted by share desc;
plus corr(share,delta), mean delta on top-5 (high-confound) and bottom-10 (low-confound) species.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.metrics import f1_score
import gen_score as G

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"


def pooled_pred(X, y, groups, pca_dim=None):
    """leave-source-out pooled (y_true, y_pred) via the same LIN head as gen_score."""
    yt_all, yp_all = [], []
    for g in sorted(set(groups)):
        r = G.fold_score(X, y, groups, g, G.LIN, pca_dim=pca_dim)
        if r is None:
            continue
        _, yt, yp = r
        yt_all.extend(yt.tolist()); yp_all.extend(yp.tolist())
    return np.array(yt_all), np.array(yp_all)


def main():
    d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
    g = np.load(ROOT / "data/raw/for_species20k/geo_features_matched.npz", allow_pickle=True)
    Xc = d["X"]; y = d["species"].astype(str); groups = d["dataset"].astype(str); Xsf = g["X_sf"]
    mask, _ = G.analyzable_mask(y, groups)
    Xc, Xsf, y, groups = Xc[mask], Xsf[mask], y[mask], groups[mask]

    species = sorted(set(y))
    # source-concentration share per species: max fraction of a species' trees from a single source
    share = {}
    for s in species:
        gs = groups[y == s]
        _, cnts = np.unique(gs, return_counts=True)
        share[s] = round(float(cnts.max() / cnts.sum()), 2)

    ytc, ypc = pooled_pred(Xc, y, groups, pca_dim=11)
    ytg, ypg = pooled_pred(Xsf, y, groups, pca_dim=None)
    f1c = dict(zip(species, f1_score(ytc, ypc, average=None, labels=species, zero_division=0)))
    f1g = dict(zip(species, f1_score(ytg, ypg, average=None, labels=species, zero_division=0)))

    rows = []
    for s in species:
        cf, gf = round(float(f1c[s]), 3), round(float(f1g[s]), 3)
        rows.append([s, share[s], cf, gf, round(cf - gf, 3)])
    rows.sort(key=lambda r: -r[1])

    sh = np.array([r[1] for r in rows]); de = np.array([r[4] for r in rows])
    corr = round(float(np.corrcoef(sh, de)[0, 1]), 3)
    n_high = 5
    res = {"note": "both CLIP and geo via SAME logistic-regression head (apples-to-apples), leave-source-out",
           "rows": rows, "corr_share_delta": corr,
           "mean_delta_high_confound": round(float(de[:n_high].mean()), 3), "n_high": n_high,
           "mean_delta_low_confound": round(float(de[n_high:].mean()), 3), "n_low": len(rows) - n_high}
    json.dump(res, open(OUT / "rev_perclass_for.json", "w"), indent=1)
    print(f"corr(share,delta)={corr}  high5 mean delta={res['mean_delta_high_confound']}  "
          f"low{res['n_low']} mean delta={res['mean_delta_low_confound']}")
    for r in rows:
        print("  ", r)


if __name__ == "__main__":
    main()
