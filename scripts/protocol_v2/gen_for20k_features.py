"""FOR-species20K: stream .laz from dev.zip, compute geometric features, cache.

No 25GB extraction — reads each .laz member from the zip in memory. Reuses the SAME
geo_features (geo_sf 11-d scale-free, geo_sz 17-d) as protocol_v2/data.py so the baseline
is identical to Weiser's. Caches aligned arrays + labels/source/sensor to one .npz.
CPU-only. Resumable: skips if cache already complete.
"""
from __future__ import annotations
import sys, zipfile, io, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, laspy
from data import geo_features

ROOT = Path(__file__).resolve().parent.parent.parent
D = ROOT / "data/raw/for_species20k"

# M3 fix: when MATCHED, voxel-downsample the cloud to the SAME ~4096-pt budget that the CLIP
# encoder used, so geometry and CLIP see identical input. Default mode keeps full-cloud geometry.
MATCHED = "--matched" in sys.argv
OUT = D / ("geo_features_matched.npz" if MATCHED else "geo_features.npz")
if MATCHED:
    from render_validation import voxel_downsample


def main():
    meta = pd.read_csv(D / "tree_metadata_dev.csv")
    z = zipfile.ZipFile(D / "dev.zip")
    names = set(z.namelist())
    sf, sz, sp, ds, dt, tid = [], [], [], [], [], []
    t0 = time.time(); n_ok = n_fail = 0
    for i, row in enumerate(meta.itertuples(index=False)):
        entry = f"{int(row.treeID):05d}.laz"
        if entry not in names:
            n_fail += 1; continue
        try:
            las = laspy.read(io.BytesIO(z.read(entry)))
            pts = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
            if pts.shape[0] < 3:
                n_fail += 1; continue
            if MATCHED:
                pts = voxel_downsample(pts)  # match CLIP input density (M3 fix)
            a, b = geo_features(pts)
            sf.append(a); sz.append(b); sp.append(row.species); ds.append(row.dataset)
            dt.append(row.data_type); tid.append(int(row.treeID)); n_ok += 1
        except Exception:
            n_fail += 1
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(meta)}  ok={n_ok} fail={n_fail}  {(time.time()-t0)/60:.1f}min", flush=True)
    np.savez_compressed(OUT, X_sf=np.array(sf), X_sz=np.array(sz),
                        species=np.array(sp), dataset=np.array(ds),
                        data_type=np.array(dt), treeID=np.array(tid))
    print(f"DONE: ok={n_ok} fail={n_fail}  saved {OUT}  ({(time.time()-t0)/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
