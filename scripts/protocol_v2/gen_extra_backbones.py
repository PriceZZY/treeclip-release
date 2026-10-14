"""Extra frozen backbones for the causal + generality controls, render-once-encode-twice
(pay the expensive per-tree rendering only once), WITH incremental checkpoint/resume so an
interrupted run (host freeze / editor close) resumes instead of restarting:

  - RANDOM-init ViT-B/16 (pretrained=False, seeded): causal control. If random weights cluster
    near geometry (far below CLIP/DINOv2), the advantage is due to PRE-TRAINING, not the
    render->ViT->cluster pipeline.
  - Supervised ImageNet ViT-B/16 (augreg in21k->in1k): a THIRD pre-training paradigm.

Pipeline identical to CLIP/DINOv2 (render_pca_depth_views 224 -> 3ch -> ImageNet norm -> frozen
ViT -> L2-norm per view -> mean). Checkpoints to {which}_extra{suffix}_partial.npz every
CKPT_EVERY trees; final {which}_extra{suffix}.npz {X_rand, X_imnet, treeID, species}.

Usage: python gen_extra_backbones.py biodiv|for [LIMIT]
Progress -> runs/render_validation/_extra_{which}_log.txt
"""
from __future__ import annotations
import sys, io, time, tarfile, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np, pandas as pd, laspy
import torch, timm
from PIL import Image
from torchvision import transforms
from render_validation import voxel_downsample
from treeclip.encoders.render_views import render_pca_depth_views

ROOT = Path(__file__).resolve().parent.parent.parent
WHICH = sys.argv[1]
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
LOG = ROOT / f"runs/render_validation/_extra_{WHICH}_log.txt"
SUFFIX = "_smoke" if LIMIT else ""
CKPT_EVERY = 1000
_MEAN = [0.485, 0.456, 0.406]; _STD = [0.229, 0.224, 0.225]
D = ROOT / ("data/raw/biodiv_3dtrees" if WHICH == "biodiv" else "data/raw/for_species20k")
STEM = "biodiv" if WHICH == "biodiv" else "for"
OUT = D / f"{STEM}_extra{SUFFIX}.npz"
PARTIAL = D / f"{STEM}_extra{SUFFIX}_partial.npz"


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(m + "\n")
    print(m, flush=True)


def make_supervised():
    for name in ["vit_base_patch16_224.augreg2_in21k_ft_in1k",
                 "vit_base_patch16_224.augreg_in21k_ft_in1k",
                 "vit_base_patch16_224.orig_in21k_ft_in1k"]:
        try:
            return timm.create_model(name, pretrained=True, num_classes=0), name
        except Exception:
            continue
    return timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0), "vit_base_patch16_224(default)"


def freeze(m):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.cuda()


def save_partial(Xr, Xi, tid, sp):
    np.savez(PARTIAL, X_rand=np.array(Xr), X_imnet=np.array(Xi), treeID=np.array(tid), species=np.array(sp))


def main():
    # resume from checkpoint if present
    Xr, Xi, tid, sp = [], [], [], []
    done = set()
    if PARTIAL.exists():
        with np.load(PARTIAL, allow_pickle=True) as p:   # `with` closes the handle so unlink() works on Windows
            Xr = list(p["X_rand"]); Xi = list(p["X_imnet"]); tid = list(p["treeID"]); sp = list(p["species"])
        done = set(str(t) for t in tid)
        log(f"RESUME from {PARTIAL.name}: {len(done)} trees already encoded")
    else:
        open(LOG, "w").close()

    torch.manual_seed(0)
    rand = freeze(timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0))
    imnet, imname = make_supervised(); imnet = freeze(imnet)
    log(f"random ViT-B/16 (pretrained=False, seed=0) | supervised={imname} | resume_done={len(done)}")
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)])

    @torch.no_grad()
    def encode_both(pts):
        views = render_pca_depth_views(pts, image_size=224)
        zr, zi = [], []
        for v in views:
            x = tf(Image.fromarray(np.stack([v, v, v], axis=-1))).unsqueeze(0).cuda()
            a = rand(x).squeeze(0); a = a / (torch.linalg.norm(a) + 1e-9)
            b = imnet(x).squeeze(0); b = b / (torch.linalg.norm(b) + 1e-9)
            zr.append(a); zi.append(b)
        return (torch.stack(zr).mean(0).cpu().numpy(), torch.stack(zi).mean(0).cpu().numpy())

    t0 = time.time(); ok = fail = 0

    if WHICH == "biodiv":
        lab = pd.read_csv(D / "biodiv_labels.csv")
        sp_of = dict(zip(lab.treeID.astype(str), lab.species.astype(str)))
        import zstandard
        f = open(D / "biodiv_ULS.tar.zst", "rb")
        tar = tarfile.open(fileobj=zstandard.ZstdDecompressor().stream_reader(f), mode="r|")
        seen = 0
        for m in tar:
            if not m.isfile() or not m.name.lower().endswith(".laz"):
                continue
            seen += 1
            if LIMIT and seen > LIMIT:
                break
            stem = Path(m.name).name[:-4]; key = stem[:-4] if stem.endswith("_ULS") else stem
            if key in done or key not in sp_of:
                if key not in sp_of: fail += 1
                continue
            try:
                las = laspy.read(io.BytesIO(tar.extractfile(m).read()))
                pts = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
                if pts.shape[0] < 3:
                    fail += 1; continue
                a, b = encode_both(voxel_downsample(pts))
                Xr.append(a); Xi.append(b); tid.append(key); sp.append(sp_of[key]); done.add(key); ok += 1
            except Exception as ex:
                fail += 1; log(f"  FAIL {key}: {str(ex)[:160]}")
            if len(tid) % CKPT_EVERY == 0 and ok > 0:
                save_partial(Xr, Xi, tid, sp); log(f"  ckpt ok={ok} total={len(tid)} {(time.time()-t0)/60:.1f}min")
    else:
        meta = pd.read_csv(D / "tree_metadata_dev.csv")
        if LIMIT: meta = meta.head(LIMIT)
        z = zipfile.ZipFile(D / "dev.zip"); names = set(z.namelist())
        for i, row in enumerate(meta.itertuples(index=False)):
            key = str(int(row.treeID)); e = f"{int(row.treeID):05d}.laz"
            if key in done:
                continue
            if e not in names:
                fail += 1; continue
            try:
                las = laspy.read(io.BytesIO(z.read(e)))
                pts = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
                if pts.shape[0] < 3:
                    fail += 1; continue
                a, b = encode_both(voxel_downsample(pts))
                Xr.append(a); Xi.append(b); tid.append(int(row.treeID)); sp.append(row.species); done.add(key); ok += 1
            except Exception as ex:
                fail += 1; log(f"  FAIL {e}: {str(ex)[:160]}")
            if len(tid) % CKPT_EVERY == 0 and ok > 0:
                save_partial(Xr, Xi, tid, sp); log(f"  {i+1}/{len(meta)} ok={ok} total={len(tid)} {(time.time()-t0)/60:.1f}min")

    Xr = np.array(Xr); Xi = np.array(Xi)
    np.savez_compressed(OUT, X_rand=Xr, X_imnet=Xi, treeID=np.array(tid), species=np.array(sp))
    try:
        if PARTIAL.exists():
            PARTIAL.unlink()
    except Exception as ex:
        log(f"  (note: could not remove partial: {str(ex)[:80]})")
    log(f"DONE ok_this_run={ok} fail={fail} total={len(tid)} Xrand={Xr.shape} Ximnet={Xi.shape} "
        f"rand_norm={float(np.linalg.norm(Xr,axis=1).mean()):.3f} imnet_norm={float(np.linalg.norm(Xi,axis=1).mean()):.3f} "
        f"nan={bool(np.isnan(Xr).any() or np.isnan(Xi).any())} saved={OUT.name} ({(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    main()
