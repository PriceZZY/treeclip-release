"""TRUSTWORTHY follow-up: is group-predictability actual CV-unit leakage?

The CV unit is: Weiser=plot (group itself), NEON=spatial BAND (a fold = several tiles).
Part2 showed features predict the fine group (tile). For NEON, tile-within-band is NOT
leakage because the whole band is held out. The decisive question:

(A) Can features predict the held-out FOLD identity (the actual CV unit)? For Weiser the
    fold==plot==group, so this equals part2. For NEON, fold=band; predict band-id.
(B) DECISIVE leakage test for Weiser: species and plot are confounded if species are
    plot-specific. Measure how much of CLIP's species-prediction is really plot-prediction:
    within a SINGLE plot (no cross-plot transfer), can CLIP still separate species? If the
    cross-plot species signal is a spatial artifact, within-plot species acc should be
    near chance while cross-plot 'works'. We instead expect the opposite if real.
    Report within-plot (pooled) species macro-F1 via stratified CV inside each plot.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from data import load, DATASETS
from spatial_cv import make_folds
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


def band_probe(d, folds, fk):
    """Predict fold/band id from features, group-aware: hold out whole tiles is hard here,
    so use stratified CV over band labels (same spirit as part2 but at CV-unit granularity)."""
    X = d[fk]; N = len(d["y"])
    band = -np.ones(N, int)
    for bi, f in enumerate(folds):
        # full band membership = group's members for that fold; reconstruct from train/test
        te = f["test_idx"]
        band[te] = bi
    # also assign species-dropped test rows: use group->band map
    # rebuild via group: any idx not assigned, find its band by majority — instead recompute mask
    grp = np.asarray(d["group"])
    # map each group to the band where it is test
    g2b = {}
    for bi, f in enumerate(folds):
        for g in set(grp[f["test_idx"]]):
            g2b[g] = bi
    band = np.array([g2b.get(g, -1) for g in grp])
    mask = band >= 0
    Xb, yb = X[mask], band[mask]
    uniq, counts = np.unique(yb, return_counts=True)
    if counts.min() < 2 or len(uniq) < 2:
        return None, None
    ns = min(5, counts.min())
    skf = StratifiedKFold(n_splits=ns, shuffle=True, random_state=0)
    yt_all, yp_all = [], []
    for tr, te in skf.split(Xb, yb):
        sc = StandardScaler().fit(Xb[tr])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xb[tr]), yb[tr])
        yp_all.extend(clf.predict(sc.transform(Xb[te])).tolist()); yt_all.extend(yb[te].tolist())
    acc = float(np.mean(np.array(yt_all) == np.array(yp_all)))
    return acc, 1.0/len(uniq)


def within_group_species(d, fk):
    """Pooled WITHIN-group species macro-F1: stratified CV inside each group separately,
    pool predictions. If CLIP's cross-plot species signal were pure spatial artifact,
    within-group species separation would be near-zero."""
    X = d[fk]; y = np.asarray(d["y"]); grp = np.asarray(d["group"])
    yt_all, yp_all = [], []
    for g in sorted(set(grp)):
        m = np.where(grp == g)[0]
        yy = y[m]
        uy, cy = np.unique(yy, return_counts=True)
        keep = uy[cy >= 2]
        if len(keep) < 2:
            continue
        sel = m[np.isin(yy, keep)]
        ys = y[sel]
        uy2, cy2 = np.unique(ys, return_counts=True)
        ns = min(3, cy2.min())
        if ns < 2:
            continue
        skf = StratifiedKFold(n_splits=ns, shuffle=True, random_state=0)
        for tr, te in skf.split(X[sel], ys):
            sc = StandardScaler().fit(X[sel][tr])
            clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(X[sel][tr]), ys[tr])
            yp_all.extend(clf.predict(sc.transform(X[sel][te])).tolist()); yt_all.extend(ys[te].tolist())
    if not yt_all:
        return None
    return float(f1_score(yt_all, yp_all, average="macro", zero_division=0))


def main():
    for ds, _, _ in DATASETS:
        d = load(ds)
        folds = make_folds(d["group"], d["xy"], d["y"], ds)
        print("="*70)
        print(f"{ds}: N={len(d['y'])} K={d['K']} CV-folds={len(folds)}")
        print("[A] predict FOLD/BAND id (the actual held-out CV unit) from features")
        for fk in ["clip", "geo_sf", "geo_sz"]:
            acc, ch = band_probe(d, folds, fk)
            if acc is None:
                print(f"   {fk:7s}: SKIP")
            else:
                print(f"   {fk:7s}: band_acc={acc:.3f} chance={ch:.3f} lift={acc/ch:.1f}x")
        print("[B] WITHIN-group species macro-F1 (spatial confound removed)")
        for fk in ["clip", "geo_sf"]:
            f1 = within_group_species(d, fk)
            print(f"   {fk:7s}: within_group_speciesF1={'NA' if f1 is None else round(f1,3)}")


if __name__ == "__main__":
    main()
