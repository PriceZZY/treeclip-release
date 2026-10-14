"""Recover NEON spatial metadata (tile + UTM centroid) keyed by tree_id.

NEON tree_ids (indvdID) ARE globally unique within a site, so a tree_id->meta dict
is safe. Weiser is NOT keyed here: Weiser tree_ids are plot-local and collide across
plots (1233 npz, only 915 unique tree_ids). Weiser grouping is handled in data.py by
iteration order (group = plot dir, coords = point-cloud XY centroid).

Output: runs/spatial_meta.pkl = {"NEON-OSBS": {tree_id: {tile,x,y,species}}, "NEON-MLBS": {...}}
"""
from __future__ import annotations
import sys, pickle, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import geopandas as gpd
from treeclip.core.tree_sample import TreeSample

ROOT = Path(__file__).resolve().parent.parent.parent
RUNS = ROOT / "runs"


def neon_meta(site: str):
    shp = gpd.read_file(ROOT / f"data/raw/idtrees/train/ITC/train_{site}.shp")
    field = pd.read_csv(ROOT / "data/raw/idtrees/train/Field/train_data.csv")
    rsfile = pd.read_csv(ROOT / "data/raw/idtrees/train/Field/itc_rsFile.csv")
    tax = field.set_index("indvdID")["taxonID"].to_dict()
    tile = rsfile.set_index("indvdID")["rsFile"].to_dict()
    m = {}
    for _, row in shp.iterrows():
        tid = str(row["indvdID"]); geom = row.geometry
        if geom is None:
            continue
        c = geom.centroid
        m[tid] = {"tile": Path(str(tile.get(tid, "NA"))).stem,
                  "x": float(c.x), "y": float(c.y), "species": tax.get(tid)}
    return m


def labeled_ids(site_dir: Path):
    ids = []
    for p in sorted(site_dir.glob("*.npz")):
        t = TreeSample.load_npz(p)
        if t.point_cloud.shape[0] >= 3 and t.species is not None:
            ids.append(t.tree_id)
    return ids


def main():
    meta = {}
    for ds, site in [("NEON-OSBS", "OSBS"), ("NEON-MLBS", "MLBS")]:
        full = neon_meta(site)
        labeled = labeled_ids(ROOT / "data/cache" / site)
        sub = {tid: full[tid] for tid in labeled if tid in full}
        meta[ds] = sub
        miss = [t for t in labeled if t not in full]
        dup = len(labeled) - len(set(labeled))
        tiles = {v["tile"] for v in sub.values()}
        print(f"{ds}: labeled={len(labeled)} mapped={len(sub)} tiles={len(tiles)} "
              f"unmapped={len(miss)} dup_ids={dup}")
        assert len(miss) == 0, f"{ds}: {len(miss)} tree_ids missing coords: {miss[:5]}"
        assert dup == 0, f"{ds}: {dup} duplicate tree_ids (unexpected for NEON)"
    RUNS.mkdir(exist_ok=True)
    with open(RUNS / "spatial_meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    print(f"wrote {RUNS/'spatial_meta.pkl'}  (NEON only; Weiser grouped in data.py)")


if __name__ == "__main__":
    main()
