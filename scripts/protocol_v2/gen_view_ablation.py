"""Phase-1 view-count ablation (CLIP, FOR + BioDiv). DECODE-ONCE-RENDER-MANY: each tree's .laz is
decoded + voxel_downsampled ONCE, then rendered at every n_views in NLIST and CLIP-encoded, so the
expensive decode/voxel cost is amortized across all view counts. CLIP recipe identical to
CLIPStructEncoder (open_clip ViT-B-16-quickgelu openai, per-view L2-norm then mean).

Eval per n uses the EXACT headline protocol (gen_unsup_robust.arm + gen_score.analyzable_mask +
BioDiv Fagus<39% down-sample, 15 seeds, matched-11d PCA NMI), so the chosen n* is directly comparable
to the full eval. The archived headline values in unsup_robust_multi (CLIP_matched_11d) correspond to
six views (n=6); n=2 is an ablation condition and does not reproduce them.

Checkpoint/resume: _viewabl_{which}_partial.npz every 1000 trees (skips done treeIDs).
Usage: python gen_view_ablation.py for|biodiv [--nlist 2,4,6,8,12] [--limit K]
Output: runs/protocol_v2/gen/view_ablation_{which}.json
"""
from __future__ import annotations
import sys, io, time, json, zipfile, tarfile, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import numpy as np, pandas as pd, laspy
import torch
from PIL import Image
from collections import Counter
from render_validation import voxel_downsample
from treeclip.encoders.render_views import render_n_views_pca
import gen_unsup_robust as R
import gen_score as G

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["for", "biodiv"])
    ap.add_argument("--nlist", default="2,4,6,8,12")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    which = a.which
    NLIST = [int(x) for x in a.nlist.split(",")]
    D = ROOT / ("data/raw/biodiv_3dtrees" if which == "biodiv" else "data/raw/for_species20k")
    LOG = ROOT / f"runs/render_validation/_viewabl_{which}_log.txt"
    PARTIAL = D / f"_viewabl_{which}_partial.npz"

    def log(m):
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(m + "\n")
        print(m, flush=True)

    # ---- CLIP encoder (frozen) ----
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16-quickgelu", pretrained="openai")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.cuda()

    @torch.no_grad()
    def clip_fuse(pts, n):
        views = render_n_views_pca(pts, n, 224)
        x = torch.stack([preprocess(Image.fromarray(np.stack([v, v, v], -1))) for v in views]).cuda()
        z = model.encode_image(x); z = z / (z.norm(dim=1, keepdim=True) + 1e-9)
        return z.mean(0).cpu().numpy()

    # ---- resume ----
    store = {f"v{n}": [] for n in NLIST}
    tid, sp, grp, sf = [], [], [], []
    done = set()
    if PARTIAL.exists():
        with np.load(PARTIAL, allow_pickle=True) as p:
            for n in NLIST:
                if f"v{n}" in p.files: store[f"v{n}"] = list(p[f"v{n}"])
            tid = list(p["tid"]); sp = list(p["sp"]); grp = list(p["grp"])
            if "sf" in p.files: sf = list(p["sf"])
        done = set(str(t) for t in tid)
        log(f"RESUME {PARTIAL.name}: {len(done)} trees done")
    else:
        open(LOG, "w").close()
    log(f"[{which}] view ablation NLIST={NLIST} | CLIP ViT-B-16-quickgelu(openai) | resume={len(done)}")

    def save_partial():
        d = {f"v{n}": np.array(store[f"v{n}"]) for n in NLIST}
        d.update(tid=np.array(tid), sp=np.array(sp), grp=np.array(grp))
        if which == "biodiv": d["sf"] = np.array(sf)
        np.savez(PARTIAL, **d)

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
                pdn = voxel_downsample(pts)
                for n in NLIST: store[f"v{n}"].append(clip_fuse(pdn, n))
                tid.append(int(row.treeID)); sp.append(str(row.species)); grp.append(str(row.dataset)); ok += 1
            except Exception as ex:
                fail += 1; log(f"  FAIL {e}: {str(ex)[:160]}")
            if len(tid) % 1000 == 0 and ok > 0:
                save_partial(); log(f"  {i+1}/{len(meta)} ok={ok} {(time.time()-t0)/60:.1f}min")
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
                pdn = voxel_downsample(pts)
                gsf, _ = geo_features(pdn)
                for n in NLIST: store[f"v{n}"].append(clip_fuse(pdn, n))
                tid.append(key); sp.append(sp_of[key]); grp.append(ep_of[key][:3]); sf.append(gsf); ok += 1
            except Exception as ex:
                fail += 1; log(f"  FAIL {key}: {str(ex)[:160]}")
            if len(tid) % 1000 == 0 and ok > 0:
                save_partial(); log(f"  ok={ok} {(time.time()-t0)/60:.1f}min")

    log(f"EXTRACT DONE ok={ok} fail={fail} total={len(tid)} ({(time.time()-t0)/60:.1f}min)")

    # ---- eval per n (exact headline protocol) ----
    y = np.array(sp); groups = np.array(grp)
    Xn = {n: np.array(store[f"v{n}"]) for n in NLIST}
    if which == "for":
        g = np.load(ROOT / "data/raw/for_species20k/geo_features_matched.npz", allow_pickle=True)
        gmap = {str(t): i for i, t in enumerate(g["treeID"])}
        rows = np.array([gmap[str(t)] for t in tid])
        Xg = g["X_sf"][rows]
        fag = None
    else:
        Xg = np.array(sf); fag = 0.39

    mask, _ = G.analyzable_mask(y, groups)
    y, groups, Xg = y[mask], groups[mask], Xg[mask]
    for n in NLIST: Xn[n] = Xn[n][mask]
    if fag is not None:
        c = Counter(y); top = max(c, key=c.get); nn = len(y)
        if c[top] / nn > fag:
            tgt = int(fag / (1 - fag) * (nn - c[top])); idx = np.where(y == top)[0]
            drop = set(np.random.RandomState(0).choice(idx, c[top] - tgt, replace=False))
            keep = np.array([i not in drop for i in range(nn)])
            y, Xg = y[keep], Xg[keep]
            for n in NLIST: Xn[n] = Xn[n][keep]

    K = len(set(y)); gdim = Xg.shape[1]
    geo, _ = R.arm(Xg, y, K)
    res = {"dataset": which, "K": int(K), "n": int(len(y)), "geo_dim": int(gdim), "n_seeds": 15,
           "nlist": NLIST, "subset": ("FULL" if a.limit is None else f"LIMIT={a.limit}"),
           "geo_matched": geo, "per_n": {}}
    for n in NLIST:
        m_, _ = R.arm(Xn[n], y, K, pca_dim=gdim)
        res["per_n"][str(n)] = m_
    # pick n*: max NMI; report knee too
    nmis = {n: res["per_n"][str(n)]["NMI"]["mean"] for n in NLIST}
    nstar = max(nmis, key=nmis.get)
    res["nstar_by_max_NMI"] = int(nstar)
    res["nmi_by_n"] = {str(n): nmis[n] for n in NLIST}
    json.dump(res, open(OUT / f"view_ablation_{which}.json", "w"), indent=2)
    try:
        if PARTIAL.exists(): PARTIAL.unlink()
    except Exception:
        pass
    print(f"\n[{which}] K={K} n={len(y)} geo NMI={geo['NMI']['mean']:.4f}  (matched-{gdim}d NMI by views:)")
    for n in NLIST:
        print(f"  n_views={n:2d}: NMI={nmis[n]:.4f}  macroF1={res['per_n'][str(n)]['macroF1']['mean']:.4f}")
    print(f"  -> n* (max NMI) = {nstar}")


if __name__ == "__main__":
    main()
