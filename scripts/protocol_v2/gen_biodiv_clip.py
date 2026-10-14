"""BioDiv-3DTrees ULS: CLIP features (GPU). Streams .laz from ULS.tar.zst (no 20GB extract),
joins to labels.csv by treeID (= filename stem), applies S1-LOCKED voxel-downsample, frozen
CLIPStructEncoder. region = ep[:3] (AEW/HEW/SEW) for leave-region-out spatial CV. Also computes
geo_sf/geo_sz for the RF/SVM baseline in the SAME pass.

Saves data/raw/biodiv_3dtrees/biodiv_features.npz {X_clip(N,512), X_sf(N,11), X_sz(N,17),
species, region, ep, treeID, nULS}. Usage: python gen_biodiv_clip.py [LIMIT].
Progress -> runs/render_validation/_biodiv_clip_log.txt
"""
from __future__ import annotations
import sys, tarfile, io, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np, pandas as pd, laspy, zstandard
from treeclip.core.tree_sample import TreeSample
from treeclip.encoders.clip_struct import CLIPStructEncoder
from data import geo_features
from render_validation import voxel_downsample

ROOT = Path(__file__).resolve().parent.parent.parent
D = ROOT / "data/raw/biodiv_3dtrees"
LOG = ROOT / "runs/render_validation/_biodiv_clip_log.txt"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None
OUTNAME = "biodiv_features_smoke.npz" if LIMIT else "biodiv_features.npz"


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(m + "\n")


def main():
    open(LOG, "w").close()
    lab = pd.read_csv(D / "biodiv_labels.csv")
    sp_of = dict(zip(lab.treeID.astype(str), lab.species.astype(str)))
    ep_of = dict(zip(lab.treeID.astype(str), lab.ep.astype(str)))
    enc = CLIPStructEncoder(device="cuda")
    z0 = np.zeros((1, 1, 3), np.uint8); h0 = np.zeros((1, 1, 1), np.float32); m0 = np.zeros((1, 1), bool)

    f = open(D / "biodiv_ULS.tar.zst", "rb")
    tar = tarfile.open(fileobj=zstandard.ZstdDecompressor().stream_reader(f), mode="r|")
    Xc, Xsf, Xsz, sp, reg, ep, tid, npts = [], [], [], [], [], [], [], []
    t0 = time.time(); ok = fail = 0; seen = 0
    for m in tar:
        if not m.isfile() or not m.name.lower().endswith(".laz"):
            continue
        seen += 1
        if LIMIT and seen > LIMIT:
            break
        stem = Path(m.name).name[:-4]              # AEW33_GD_482_ULS
        key = stem[:-4] if stem.endswith("_ULS") else stem   # -> AEW33_GD_482
        if key not in sp_of:
            fail += 1; continue
        try:
            data = tar.extractfile(m).read()
            las = laspy.read(io.BytesIO(data))
            pts = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
            if pts.shape[0] < 3:
                fail += 1; continue
            n0 = pts.shape[0]
            pd_ = voxel_downsample(pts)             # S1-LOCKED
            a, b = geo_features(pd_)
            v = enc.encode(TreeSample("t", "BIO", sp_of[key], pd_, z0, h0, m0, 0.0)).numpy()
            Xc.append(v); Xsf.append(a); Xsz.append(b); sp.append(sp_of[key])
            reg.append(ep_of[key][:3]); ep.append(ep_of[key]); tid.append(key); npts.append(n0); ok += 1
        except Exception as ex:
            fail += 1; log(f"  FAIL {key}: {str(ex)[:120]}")
        if ok % 500 == 0 and ok > 0:
            log(f"  ok={ok} fail={fail} {(time.time()-t0)/60:.1f}min")
    Xc = np.array(Xc)
    np.savez_compressed(D / OUTNAME, X_clip=Xc, X_sf=np.array(Xsf), X_sz=np.array(Xsz),
                        species=np.array(sp), region=np.array(reg), ep=np.array(ep),
                        treeID=np.array(tid), nULS=np.array(npts))
    log(f"DONE ok={ok} fail={fail} Xclip={Xc.shape} "
        f"normmean={float(np.linalg.norm(Xc,axis=1).mean()):.3f} nan={bool(np.isnan(Xc).any())} "
        f"saved={OUTNAME} ({(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    main()
