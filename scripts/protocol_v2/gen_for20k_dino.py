"""FOR-species20K DINOv2 features (GPU). Mirrors gen_for20k_clip.py EXACTLY (same dev.zip
.laz streaming, same metadata order, same S1-LOCKED voxel-downsample) but swaps
CLIPStructEncoder -> DINOv2StructEncoder. Saves for_dino.npz {X_dino(N,768), species, dataset,
data_type, treeID} so the eval aligns DINOv2 rows to clip_features.npz by treeID.

Usage: python gen_for20k_dino.py [LIMIT]   (LIMIT optional smoke)
Progress -> runs/render_validation/_for_dino_log.txt
"""
from __future__ import annotations
import sys, zipfile, io, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np, pandas as pd, laspy
from treeclip.core.tree_sample import TreeSample
from treeclip.encoders.dinov2_struct import DINOv2StructEncoder
from render_validation import voxel_downsample

ROOT = Path(__file__).resolve().parent.parent.parent
D = ROOT / "data/raw/for_species20k"
LOG = ROOT / "runs/render_validation/_for_dino_log.txt"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None
OUTNAME = "for_dino_smoke.npz" if LIMIT else "for_dino.npz"


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(m + "\n")
    print(m, flush=True)


def main():
    open(LOG, "w").close()
    meta = pd.read_csv(D / "tree_metadata_dev.csv")
    if LIMIT: meta = meta.head(LIMIT)
    z = zipfile.ZipFile(D / "dev.zip"); names = set(z.namelist())
    enc = DINOv2StructEncoder(device="cuda")
    log(f"encoder loaded: kind={enc.kind} dim={enc.embedding_dim}")
    z0 = np.zeros((1, 1, 3), np.uint8); h0 = np.zeros((1, 1, 1), np.float32); m0 = np.zeros((1, 1), bool)
    X, sp, ds, dt, tid = [], [], [], [], []
    t0 = time.time(); ok = fail = 0
    for i, row in enumerate(meta.itertuples(index=False)):
        e = f"{int(row.treeID):05d}.laz"
        if e not in names: fail += 1; continue
        try:
            las = laspy.read(io.BytesIO(z.read(e)))
            pts = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
            if pts.shape[0] < 3: fail += 1; continue
            pts = voxel_downsample(pts)
            v = enc.encode(TreeSample("t", "FOR", str(row.species), pts, z0, h0, m0, 0.0)).numpy()
            X.append(v); sp.append(row.species); ds.append(row.dataset); dt.append(row.data_type); tid.append(int(row.treeID)); ok += 1
        except Exception as ex:
            fail += 1; log(f"  FAIL {e}: {str(ex)[:160]}")
        if (i + 1) % 500 == 0:
            log(f"  {i+1}/{len(meta)} ok={ok} fail={fail} {(time.time()-t0)/60:.1f}min")
    X = np.array(X)
    np.savez_compressed(D / OUTNAME, X_dino=X, species=np.array(sp), dataset=np.array(ds),
                        data_type=np.array(dt), treeID=np.array(tid))
    log(f"DONE ok={ok} fail={fail} X={X.shape} normmean={float(np.linalg.norm(X,axis=1).mean()):.3f} "
        f"nan={bool(np.isnan(X).any())} saved={OUTNAME} ({(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    main()
