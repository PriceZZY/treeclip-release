"""Phase-0 gate: verify render_n_views_pca(n=2) is the SAME algorithm as render_pca_depth_views.
view0 (theta=0) = major-axis projection -> must be byte-identical.
view1 (theta=pi/2) = major rotated 90deg = +/- minor axis -> identical OR exact horizontal mirror
(eigh sign convention is arbitrary). Reports max-abs-diff on 8 real FOR trees.
"""
from __future__ import annotations
import sys, io, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np, pandas as pd, laspy
from render_validation import voxel_downsample
from treeclip.encoders.render_views import render_pca_depth_views, render_n_views_pca

D = Path(__file__).resolve().parent.parent.parent / "data/raw/for_species20k"
meta = pd.read_csv(D / "tree_metadata_dev.csv").head(8)
z = zipfile.ZipFile(D / "dev.zip"); names = set(z.namelist())
print("renderer consistency (render_n_views_pca n=2  vs  render_pca_depth_views):")
all_ok = True
for row in meta.itertuples(index=False):
    e = f"{int(row.treeID):05d}.laz"
    if e not in names: continue
    las = laspy.read(io.BytesIO(z.read(e)))
    pts = voxel_downsample(np.vstack([las.x, las.y, las.z]).T.astype(np.float32))
    r2 = render_pca_depth_views(pts, 224)
    rn = render_n_views_pca(pts, 2, 224)
    d0 = int(np.abs(r2[0].astype(int) - rn[0].astype(int)).max())
    d1 = int(np.abs(r2[1].astype(int) - rn[1].astype(int)).max())
    d1m = int(np.abs(np.fliplr(r2[1]).astype(int) - rn[1].astype(int)).max())
    v1ok = (d1 == 0) or (d1m == 0)
    all_ok = all_ok and (d0 == 0) and v1ok
    print(f"  tree {int(row.treeID):5d}: view0 diff={d0}  view1 diff={d1} (mirror diff={d1m}) "
          f"{'OK' if (d0==0 and v1ok) else 'MISMATCH'}")
print("GATE0_RENDERER:", "PASS" if all_ok else "FAIL")
