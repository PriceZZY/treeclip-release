"""Leakage-free spatial cross-validation folds.

NEON: tiles sorted along the principal spatial axis (PC1 of tile UTM centroids), then cut
      into 5 contiguous bands BALANCED by tree count; each band is held out as one test
      fold. No tile ever spans train/test (verified). KNOWN MINOR LIMITATION: a few tile
      pairs that are spatially adjacent but fall on opposite sides of a band boundary end
      up in different folds (~3-5 of the 20 closest pairs); since no crown/tree is shared,
      this is mild edge-adjacency, not train/test leakage.
Weiser: leave-one-plot-out (6 folds).

Shared-species scoring (audit C1 fix): each fold's TEST set is restricted to species that
also appear in that fold's TRAIN set. Per-fold bookkeeping records species scored/dropped
and trees scored/dropped, so nothing is silently hidden.
"""
from __future__ import annotations
import numpy as np


def _tile_centroids(group, xy):
    tiles = sorted(set(group))
    cent = np.array([xy[group == t].mean(0) for t in tiles])
    return tiles, cent


def make_folds(group, xy, y, ds_name, n_neon_folds=5, seed=0):
    """Return list of fold dicts: {name, train_idx, test_idx (shared-species only),
    info{n_train, n_test_full, n_test_scored, species_scored, species_dropped}}."""
    group = np.asarray(group); y = np.asarray(y); xy = np.asarray(xy)
    N = len(y)

    if ds_name.startswith("NEON"):
        tiles, cent = _tile_centroids(group, xy)
        k = min(n_neon_folds, len(tiles))
        # project tile centroids onto principal spatial axis -> sort -> balanced bands
        c0 = cent - cent.mean(0)
        pc1 = c0 @ np.linalg.svd(c0, full_matrices=False)[2][0]
        order = np.argsort(pc1)
        tile_n = {t: int((group == t).sum()) for t in tiles}
        target = len(y) / k
        band = {}; b = 0; cum = 0
        for ti in order:
            t = tiles[ti]
            if cum >= target and b < k - 1:
                b += 1; cum = 0
            band[t] = b; cum += tile_n[t]
        cluster_of = np.array([band[g] for g in group])
        fold_keys = list(range(k))
        def test_mask(fk): return cluster_of == fk
        def fold_name(fk):
            members = [t for t in tiles if band[t] == fk]
            return f"spatialband{fk}({len(members)}tiles)"
    else:  # Weiser leave-one-plot-out
        fold_keys = sorted(set(group))
        def test_mask(fk): return group == fk
        def fold_name(fk): return f"plot:{fk}"

    folds = []
    for fk in fold_keys:
        tem = test_mask(fk)
        tr_idx = np.where(~tem)[0]
        te_full = np.where(tem)[0]
        train_species = set(y[tr_idx])
        shared = train_species & set(y[te_full])
        te_idx = np.array([i for i in te_full if y[i] in shared], dtype=int)
        dropped = sorted(set(y[te_full]) - shared)
        folds.append({
            "name": fold_name(fk),
            "train_idx": tr_idx,
            "test_idx": te_idx,
            "info": {
                "n_train": int(len(tr_idx)),
                "n_test_full": int(len(te_full)),
                "n_test_scored": int(len(te_idx)),
                "species_scored": len(shared),
                "species_dropped": len(dropped),
                "dropped_list": dropped,
            },
        })
    return folds


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data import load, DATASETS
    for ds, _, _ in DATASETS:
        d = load(ds)
        folds = make_folds(d["group"], d["xy"], d["y"], ds)
        print(f"\n=== {ds}: {len(folds)} folds ===")
        for f in folds:
            i = f["info"]
            print(f"  {f['name']:28s} train={i['n_train']:4d} test={i['n_test_scored']:3d}/"
                  f"{i['n_test_full']:3d} species_scored={i['species_scored']:2d} "
                  f"dropped={i['species_dropped']}")
