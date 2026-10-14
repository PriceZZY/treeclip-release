"""Multi-view multi-backbone feature extraction (render-once-encode-all) with a TUNABLE view
count, for the view-count re-run. Renders n_views PCA-rotated depth images once per tree
(render_n_views_pca: angles linspace(0,pi,n_views, endpoint=False); n_views=2 reproduces the
major/minor pair of render_pca_depth_views), then encodes each requested frozen backbone on the
SAME views (per-view L2-norm then mean), reproducing each encoder's recipe exactly:
  - clip : open_clip ViT-B-16-quickgelu (openai),  open_clip preprocess,      512-d
  - dino : timm vit_base_patch14_dinov2.lvd142m (img_size=224), ImageNet norm, 768-d
  - rand : timm vit_base_patch16_224 (pretrained=False, torch.manual_seed(0)), ImageNet norm
  - imnet: timm augreg2_in21k_ft_in1k (supervised in21k->in1k),    ImageNet norm, 768-d

Writes canonical-schema feature files (suffix-namespaced for the ablation) so existing eval
scripts run unchanged:
  FOR    -> clip_features{sfx}.npz {X,species,dataset,data_type,treeID}
            for_dino{sfx}.npz      {X_dino,species,dataset,data_type,treeID}
            for_extra{sfx}.npz     {X_rand,X_imnet,treeID,species}
  BioDiv -> biodiv_features{sfx}.npz {X_clip,X_sf,X_sz,species,region,ep,treeID,nULS}
            biodiv_dino{sfx}.npz     {X_dino,treeID,species}
            biodiv_extra{sfx}.npz    {X_rand,X_imnet,treeID,species}
geo (X_sf/X_sz) is view-independent but lives inside biodiv_features.npz, so it is recomputed
here to keep that file complete and aligned. FOR geo lives in geo_features_matched.npz (untouched).

ONLY the view count changes vs the 2-view pipeline; voxel_downsample, PCA, L2-norm-then-mean,
and the per-backbone weights/transforms are identical. Checkpoint/resume: one
_mv_{stem}_v{N}{sfx}_partial.npz every CKPT_EVERY trees; resume skips done treeIDs (host-freeze
safe; closes the NpzFile before unlink for Windows).

Usage:
  python gen_multiview_backbones.py for|biodiv --nviews N [--backbones clip,dino,rand,imnet]
         [--suffix _v6] [--limit K]
Progress -> runs/render_validation/_mv_{which}_v{N}{sfx}_log.txt
"""
from __future__ import annotations
import sys, io, time, zipfile, tarfile, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np, pandas as pd, laspy
import torch
from PIL import Image
from torchvision import transforms
from render_validation import voxel_downsample
from treeclip.encoders.render_views import render_n_views_pca

ROOT = Path(__file__).resolve().parent.parent.parent
_MEAN = [0.485, 0.456, 0.406]; _STD = [0.229, 0.224, 0.225]
IMNET_TF = transforms.Compose([transforms.ToTensor(), transforms.Normalize(_MEAN, _STD)])
CKPT_EVERY = 1000


def freeze(m):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m.cuda()


def make_supervised():
    import timm
    for name in ["vit_base_patch16_224.augreg2_in21k_ft_in1k",
                 "vit_base_patch16_224.augreg_in21k_ft_in1k",
                 "vit_base_patch16_224.orig_in21k_ft_in1k"]:
        try:
            return timm.create_model(name, pretrained=True, num_classes=0), name
        except Exception:
            continue
    return timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0), "vit_base_patch16_224(default)"


def build_backbones(which):
    """Return dict name->model (+ clip preprocess). rand created FIRST under seed 0 so its weights
    are deterministic regardless of which other backbones load."""
    bb = {}; clip_pre = None; notes = []
    import timm
    if "rand" in which or "imnet" in which:
        torch.manual_seed(0)
        if "rand" in which:
            bb["rand"] = freeze(timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0))
            notes.append("rand=vit_base_patch16_224(pretrained=False,seed0)")
        if "imnet" in which:
            im, imname = make_supervised(); bb["imnet"] = freeze(im); notes.append(f"imnet={imname}")
    if "dino" in which:
        bb["dino"] = freeze(timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True,
                                              num_classes=0, img_size=224))
        notes.append("dino=vit_base_patch14_dinov2.lvd142m")
    if "clip" in which:
        import open_clip
        m, _, clip_pre = open_clip.create_model_and_transforms("ViT-B-16-quickgelu", pretrained="openai")
        bb["clip"] = freeze(m); notes.append("clip=ViT-B-16-quickgelu(openai)")
    return bb, clip_pre, notes


@torch.no_grad()
def encode_tree(pts, n_views, bb, clip_pre):
    """render_n_views once, encode each requested backbone (per-view L2-norm then mean)."""
    views = render_n_views_pca(pts, n_views, 224)               # (n,224,224) uint8
    pil = [Image.fromarray(np.stack([v, v, v], axis=-1)) for v in views]
    out = {}
    if "clip" in bb:
        x = torch.stack([clip_pre(p) for p in pil]).cuda()
        z = bb["clip"].encode_image(x); z = z / (z.norm(dim=1, keepdim=True) + 1e-9)
        out["clip"] = z.mean(0).cpu().numpy()
    imnet_keys = [k for k in ("dino", "rand", "imnet") if k in bb]
    if imnet_keys:
        x = torch.stack([IMNET_TF(p) for p in pil]).cuda()
        for k in imnet_keys:
            z = bb[k](x); z = z / (z.norm(dim=1, keepdim=True) + 1e-9)
            out[k] = z.mean(0).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["for", "biodiv"])
    ap.add_argument("--nviews", type=int, required=True)
    ap.add_argument("--backbones", default="clip,dino,rand,imnet")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    which = a.which; n_views = a.nviews; sfx = a.suffix
    req = [b.strip() for b in a.backbones.split(",") if b.strip()]
    D = ROOT / ("data/raw/biodiv_3dtrees" if which == "biodiv" else "data/raw/for_species20k")
    stem = "biodiv" if which == "biodiv" else "for"
    LOG = ROOT / f"runs/render_validation/_mv_{which}_v{n_views}{sfx}_log.txt"
    PARTIAL = D / f"_mv_{stem}_v{n_views}{sfx}_partial.npz"

    def log(m):
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(m + "\n")
        print(m, flush=True)

    # ---- resume ----
    cols = {"tid": [], "sp": []}
    for k in ("clip", "dino", "rand", "imnet"):
        if k in req: cols[k] = []
    if which == "for":
        cols["ds"] = []; cols["dt"] = []
    else:
        cols["sf"] = []; cols["sz"] = []; cols["reg"] = []; cols["ep"] = []; cols["npts"] = []
    done = set()
    if PARTIAL.exists():
        with np.load(PARTIAL, allow_pickle=True) as p:
            for k in list(cols):
                if k in p.files: cols[k] = list(p[k])
        done = set(str(t) for t in cols["tid"])
        log(f"RESUME {PARTIAL.name}: {len(done)} trees already done")
    else:
        open(LOG, "w").close()

    bb, clip_pre, notes = build_backbones(req)
    log(f"[{which}] n_views={n_views} backbones={req} suffix='{sfx}' | " + " | ".join(notes))

    def save_partial():
        np.savez(PARTIAL, **{k: np.array(v) for k, v in cols.items()})

    t0 = time.time(); ok = fail = 0

    if which == "for":
        meta = pd.read_csv(D / "tree_metadata_dev.csv")
        if a.limit: meta = meta.head(a.limit)
        z = zipfile.ZipFile(D / "dev.zip"); names = set(z.namelist())
        for i, row in enumerate(meta.itertuples(index=False)):
            key = str(int(row.treeID)); e = f"{int(row.treeID):05d}.laz"
            if key in done: continue
            if e not in names: fail += 1; continue
            try:
                las = laspy.read(io.BytesIO(z.read(e)))
                pts = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
                if pts.shape[0] < 3: fail += 1; continue
                emb = encode_tree(voxel_downsample(pts), n_views, bb, clip_pre)
                for k in ("clip", "dino", "rand", "imnet"):
                    if k in req: cols[k].append(emb[k])
                cols["tid"].append(int(row.treeID)); cols["sp"].append(row.species)
                cols["ds"].append(row.dataset); cols["dt"].append(row.data_type)
                done.add(key); ok += 1
            except Exception as ex:
                fail += 1; log(f"  FAIL {e}: {str(ex)[:160]}")
            if len(cols["tid"]) % CKPT_EVERY == 0 and ok > 0:
                save_partial(); log(f"  {i+1}/{len(meta)} ok={ok} total={len(cols['tid'])} {(time.time()-t0)/60:.1f}min")
    else:
        from data import geo_features
        import zstandard
        lab = pd.read_csv(D / "biodiv_labels.csv")
        sp_of = dict(zip(lab.treeID.astype(str), lab.species.astype(str)))
        ep_of = dict(zip(lab.treeID.astype(str), lab.ep.astype(str)))
        f = open(D / "biodiv_ULS.tar.zst", "rb")
        tar = tarfile.open(fileobj=zstandard.ZstdDecompressor().stream_reader(f), mode="r|")
        seen = 0
        for m in tar:
            if not m.isfile() or not m.name.lower().endswith(".laz"): continue
            seen += 1
            if a.limit and seen > a.limit: break
            st = Path(m.name).name[:-4]; key = st[:-4] if st.endswith("_ULS") else st
            if key in done: continue
            if key not in sp_of: fail += 1; continue
            try:
                las = laspy.read(io.BytesIO(tar.extractfile(m).read()))
                pts = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
                if pts.shape[0] < 3: fail += 1; continue
                n0 = pts.shape[0]; pdn = voxel_downsample(pts)
                sf, sz = geo_features(pdn)
                emb = encode_tree(pdn, n_views, bb, clip_pre)
                for k in ("clip", "dino", "rand", "imnet"):
                    if k in req: cols[k].append(emb[k])
                cols["tid"].append(key); cols["sp"].append(sp_of[key])
                cols["sf"].append(sf); cols["sz"].append(sz)
                cols["reg"].append(ep_of[key][:3]); cols["ep"].append(ep_of[key]); cols["npts"].append(n0)
                done.add(key); ok += 1
            except Exception as ex:
                fail += 1; log(f"  FAIL {key}: {str(ex)[:160]}")
            if len(cols["tid"]) % CKPT_EVERY == 0 and ok > 0:
                save_partial(); log(f"  ok={ok} total={len(cols['tid'])} {(time.time()-t0)/60:.1f}min")

    # ---- write canonical-schema files ----
    tid = np.array(cols["tid"]); sp = np.array(cols["sp"])
    written = []
    if which == "for":
        ds = np.array(cols["ds"]); dt = np.array(cols["dt"])
        if "clip" in req:
            np.savez_compressed(D / f"clip_features{sfx}.npz", X=np.array(cols["clip"]), species=sp,
                                dataset=ds, data_type=dt, treeID=tid); written.append(f"clip_features{sfx}.npz")
        if "dino" in req:
            np.savez_compressed(D / f"for_dino{sfx}.npz", X_dino=np.array(cols["dino"]), species=sp,
                                dataset=ds, data_type=dt, treeID=tid); written.append(f"for_dino{sfx}.npz")
        if "rand" in req or "imnet" in req:
            np.savez_compressed(D / f"for_extra{sfx}.npz",
                                X_rand=np.array(cols["rand"]) if "rand" in req else np.zeros((len(tid), 0)),
                                X_imnet=np.array(cols["imnet"]) if "imnet" in req else np.zeros((len(tid), 0)),
                                treeID=tid, species=sp); written.append(f"for_extra{sfx}.npz")
    else:
        if "clip" in req:
            np.savez_compressed(D / f"biodiv_features{sfx}.npz", X_clip=np.array(cols["clip"]),
                                X_sf=np.array(cols["sf"]), X_sz=np.array(cols["sz"]), species=sp,
                                region=np.array(cols["reg"]), ep=np.array(cols["ep"]), treeID=tid,
                                nULS=np.array(cols["npts"])); written.append(f"biodiv_features{sfx}.npz")
        if "dino" in req:
            np.savez_compressed(D / f"biodiv_dino{sfx}.npz", X_dino=np.array(cols["dino"]),
                                treeID=tid, species=sp); written.append(f"biodiv_dino{sfx}.npz")
        if "rand" in req or "imnet" in req:
            np.savez_compressed(D / f"biodiv_extra{sfx}.npz",
                                X_rand=np.array(cols["rand"]) if "rand" in req else np.zeros((len(tid), 0)),
                                X_imnet=np.array(cols["imnet"]) if "imnet" in req else np.zeros((len(tid), 0)),
                                treeID=tid, species=sp); written.append(f"biodiv_extra{sfx}.npz")

    nan = any(bool(np.isnan(np.array(cols[k])).any()) for k in req if k in cols and len(cols[k]) > 0)
    try:
        if PARTIAL.exists():
            PARTIAL.unlink()
    except Exception as ex:
        log(f"  (note: could not remove partial: {str(ex)[:80]})")
    norms = {k: round(float(np.linalg.norm(np.array(cols[k]), axis=1).mean()), 3) for k in req if k in cols and len(cols[k]) > 0}
    log(f"DONE ok={ok} fail={fail} total={len(tid)} nan={nan} norms={norms} wrote={written} ({(time.time()-t0)/60:.1f}min)")


if __name__ == "__main__":
    main()
