"""BioDiv-3DTrees ULS: DINOv2 features (GPU). Mirrors gen_biodiv_clip.py EXACTLY
(same .laz streaming, same S1-LOCKED voxel-downsample, same per-tree order) but swaps
CLIPStructEncoder -> DINOv2StructEncoder. Saves biodiv_dino.npz {X_dino(N,768), treeID, species}
so the eval can align DINOv2 rows to biodiv_features.npz by treeID.

Usage: python gen_biodiv_dino.py [LIMIT]   (LIMIT optional smoke, e.g. 8)
Progress -> runs/render_validation/_biodiv_dino_log.txt
"""
from __future__ import annotations
import sys, tarfile, io, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np, pandas as pd, laspy, zstandard
from treeclip.core.tree_sample import TreeSample
from treeclip.encoders.dinov2_struct import DINOv2StructEncoder
from render_validation import voxel_downsample

ROOT = Path(__file__).resolve().parent.parent.parent
D = ROOT / "data/raw/biodiv_3dtrees"
LOG = ROOT / "runs/render_validation/_biodiv_dino_log.txt"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None
OUTNAME = "biodiv_dino_smoke.npz" if LIMIT else "biodiv_dino.npz"


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(m + "\n")
    print(m, flush=True)


def main():
    open(LOG, "w").close()
    lab = pd.read_csv(D / "biodiv_labels.csv")
    sp_of = dict(zip(lab.treeID.astype(str), lab.species.astype(str)))
    ep_of = dict(zip(lab.treeID.astype(str), lab.ep.astype(str)))
    enc = DINOv2StructEncoder(device="cuda")
    log(f"encoder loaded: kind={enc.kind} dim={enc.embedding_dim}")
    z0 = np.zeros((1, 1, 3), np.uint8); h0 = np.zeros((1, 1, 1), np.float32); m0 = np.zeros((1, 1), bool)

    f = open(D / "biodiv_ULS.tar.zst", "rb")
    tar = tarfile.open(fileobj=zstandard.ZstdDecompressor().stream_reader(f), mode="r|")
    Xd, sp, tid = [], [], []
    t0 = time.time(); ok = fail = 0; seen = 0
    for m in tar:
        if not m.isfile() or not m.name.lower().endswith(".laz"):
            continue
        seen += 1
        if LIMIT and seen > LIMIT:
            break
        stem = Path(m.name).name[:-4]
        key = stem[:-4] if stem.endswith("_ULS") else stem
        if key not in sp_of:
            fail += 1; continue
        try:
            data = tar.extractfile(m).read()
            las = laspy.read(io.BytesIO(data))
            pts = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
            if pts.shape[0] < 3:
                fail += 1; continue
            pd_ = voxel_downsample(pts)
            v = enc.encode(TreeSample("t", "BIO", sp_of[key], pd_, z0, h0, m0, 0.0)).numpy()
            Xd.append(v); sp.append(sp_of[key]); tid.append(key); ok += 1
        except Exception as ex:
            fail += 1; log(f"  FAIL {key}: {str(ex)[:160]}")
        if ok % 250 == 0 and ok > 0:
            log(f"  ok={ok} fail={fail} {(time.time()-t0)/60:.1f}min")
    Xd = np.array(Xd)
    np.savez_compressed(D / OUTNAME, X_dino=Xd, treeID=np.array(tid), species=np.array(sp))
    log(f"DONE ok={ok} fail={fail} Xdino={Xd.shape} "
        f"normmean={float(np.linalg.norm(Xd,axis=1).mean()):.3f} nan={bool(np.isnan(Xd).any())} "
        f"saved={OUTNAME} ({(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    main()
