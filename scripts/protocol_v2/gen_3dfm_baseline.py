"""Frozen 3D foundation-model (OpenShape PointBERT/PPAT, ShapeNet-pretrained) baseline for
the TreeCLIP unsupervised comparison. See reports/PREREG_3d_baseline.md.

Pipeline (per prereg):
  - Align to the SAME analyzable subset as CLIP/geo (gen_score.analyzable_mask + BioDiv Fagus
    cap 0.39, RandomState(0)); load raw LAZ for exactly those treeIDs.
  - Preprocess (§5): resample to N=1024 (random down/upsample, RandomState(0)); center to
    centroid + scale to unit max-norm (== OpenShape normalize_pc).
  - Frozen forward (torch.no_grad, eval, no fine-tune/probe) -> per-tree embedding:
      * PRIMARY  = 1280-d CLIP-aligned projected embedding (OpenShape's published shape embedding).
      * SECONDARY= 256-d pooled cls-token (pre-projection), reported alongside (anti-cherry-pick).
    Input features = concat([normalized_xyz(3), rgb=0.4 constant(3)]) = 6-d (OpenShape no-color std).
  - Evaluate with the EXACT existing harness: StandardScaler -> PCA(11) -> KMeans(K, n_init=10),
    matching-free global NMI + AMI over 15 seeds (mean +/- sd; within_source_NMI is one run, seed 0). Re-score CLIP + geo on the SAME subset
    (self-check: CLIP FOR ~0.2746 / BioDiv ~0.3501). Within-source residual NMI mirrors
    gen_source_control.within_source_nmi.

Embeddings are cached to runs/protocol_v2/gen/emb_3dfm_{for,biodiv}.npz (re-runnable).
Output: runs/protocol_v2/gen/unsup_3dfm.json

Usage:  python gen_3dfm_baseline.py [biodiv|for|both]   (default: both)
"""
from __future__ import annotations
import sys, io, json, time, zipfile, tarfile
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import laspy
import zstandard
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_mutual_info_score as AMI

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_score as G          # analyzable_mask (exact reuse)
import oshape_model as M

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"; OUT.mkdir(parents=True, exist_ok=True)

N_POINTS = 1024
DS_SEED = 0
SEEDS = list(range(15))
BATCH = 128
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_REPO = "OpenShape/openshape-pointbert-shapenet"
SELFCHECK_TARGET = {"for": 0.2746, "biodiv": 0.3501}
SELFCHECK_TOL = 0.01
RGB_CONST = 0.4  # OpenShape no-color convention


# ----------------------------- alignment (exact reuse) -----------------------------
def load_aligned(which):
    """Reproduce the harness subset EXACTLY (analyzable_mask [+ BioDiv Fagus cap])."""
    if which == "for":
        d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
        g = np.load(ROOT / "data/raw/for_species20k/geo_features_matched.npz", allow_pickle=True)
        Xc = d["X"]; y = d["species"].astype(str); groups = d["dataset"].astype(str)
        Xsf = g["X_sf"]; tid = d["treeID"]
    else:
        d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
        Xc = d["X_clip"]; Xsf = d["X_sf"]; y = d["species"].astype(str)
        groups = d["region"].astype(str); tid = d["treeID"].astype(str)
    m, _ = G.analyzable_mask(y, groups)
    Xc, Xsf, y, groups, tid = Xc[m], Xsf[m], y[m], groups[m], tid[m]
    if which == "biodiv":
        c = Counter(y); top = max(c, key=c.get); n = len(y); cap = 0.39
        if c[top] / n > cap:
            tgt = int(cap / (1 - cap) * (n - c[top])); idx = np.where(y == top)[0]
            drop = set(np.random.RandomState(0).choice(idx, c[top] - tgt, replace=False))
            k = np.array([i not in drop for i in range(n)])
            Xc, Xsf, y, groups, tid = Xc[k], Xsf[k], y[k], groups[k], tid[k]
    return {"treeID": tid, "y": y, "groups": groups, "Xc": Xc, "Xsf": Xsf}


# ----------------------------- preprocessing (§5) -----------------------------
def preprocess(xyz):
    """Resample to N=1024 (RandomState(0)); center to centroid + unit max-norm."""
    n = xyz.shape[0]
    rng = np.random.RandomState(DS_SEED)
    if n >= N_POINTS:
        idx = rng.choice(n, N_POINTS, replace=False)
    else:
        idx = rng.choice(n, N_POINTS, replace=True)
    p = xyz[idx].astype(np.float32)
    p = p - p.mean(0)
    m = float(np.max(np.linalg.norm(p, axis=1)))
    p = p / m if m > 1e-6 else p * 0.0
    rgb = np.full((N_POINTS, 3), RGB_CONST, np.float32)
    feat = np.concatenate([p, rgb], axis=1)
    return p.astype(np.float32), feat.astype(np.float32)


# ----------------------------- model + forward -----------------------------
def get_model():
    ckpt = __import__("huggingface_hub").hf_hub_download(CKPT_REPO, "model.pt")
    model = M.build_openshape_pointbert().to(DEV).eval()
    miss, unexp = M.load_checkpoint(model, ckpt, map_location=DEV)
    assert len(miss) == 0 and len(unexp) == 0, f"state_dict mismatch miss={miss} unexp={unexp}"
    return model, ckpt


@torch.no_grad()
def embed_batch(model, xyz_arr, feat_arr):
    xyz = torch.from_numpy(np.stack(xyz_arr)).float().to(DEV)
    feat = torch.from_numpy(np.stack(feat_arr)).float().to(DEV)
    pooled, proj = model(xyz, feat)
    return proj.cpu().numpy(), pooled.cpu().numpy()


def read_clouds_for(treeIDs):
    """Yield (i, xyz) for FOR treeIDs from dev.zip (random access by name)."""
    z = zipfile.ZipFile(ROOT / "data/raw/for_species20k/dev.zip")
    for i, t in enumerate(treeIDs):
        las = laspy.read(io.BytesIO(z.read(f"{int(t):05d}.laz")))
        yield i, np.stack([las.x, las.y, las.z], 1)


def read_clouds_biodiv(treeIDs):
    """Stream tar.zst once, collect needed entries, then yield in treeID order."""
    want = {f"ULS/{t}_ULS.laz": i for i, t in enumerate(treeIDs)}
    store = {}
    dctx = zstandard.ZstdDecompressor()
    with open(ROOT / "data/raw/biodiv_3dtrees/biodiv_ULS.tar.zst", "rb") as f:
        tar = tarfile.open(fileobj=dctx.stream_reader(f), mode="r|")
        for mem in tar:
            if mem.isfile() and mem.name in want:
                store[want[mem.name]] = tar.extractfile(mem).read()
                if len(store) == len(want):
                    break
    missing = [t for t, i in zip(treeIDs, range(len(treeIDs))) if i not in store]
    assert not missing, f"BioDiv LAZ missing for {len(missing)} treeIDs e.g. {missing[:5]}"
    for i in range(len(treeIDs)):
        las = laspy.read(io.BytesIO(store[i]))
        yield i, np.stack([las.x, las.y, las.z], 1)


def extract_embeddings(which, treeIDs, model):
    """Compute (or load cached) 3D-FM embeddings aligned to treeIDs order."""
    cache = OUT / f"emb_3dfm_{which}.npz"
    tid_str = np.array([str(t) for t in treeIDs])
    if cache.exists():
        c = np.load(cache, allow_pickle=True)
        ctid = c["treeID"].astype(str)
        cmap = {t: i for i, t in enumerate(ctid)}
        if all(t in cmap for t in tid_str):
            order = [cmap[t] for t in tid_str]
            print(f"[{which}] loaded cached embeddings {cache.name} n={len(order)}")
            return c["proj"][order], c["pooled"][order]
        print(f"[{which}] cache incomplete -> recomputing")
    reader = read_clouds_for if which == "for" else read_clouds_biodiv
    n = len(treeIDs)
    proj = np.zeros((n, 1280), np.float32); pooled = np.zeros((n, 256), np.float32)
    bx, bf, bi = [], [], []
    t0 = time.time(); done = 0
    for i, xyz in reader(treeIDs):
        p, fe = preprocess(xyz)
        bx.append(p); bf.append(fe); bi.append(i)
        if len(bx) == BATCH:
            pj, pl = embed_batch(model, bx, bf)
            proj[bi] = pj; pooled[bi] = pl
            done += len(bi); bx, bf, bi = [], [], []
            if done % (BATCH * 8) == 0:
                print(f"[{which}] {done}/{n}  {time.time()-t0:.0f}s", flush=True)
    if bx:
        pj, pl = embed_batch(model, bx, bf)
        proj[bi] = pj; pooled[bi] = pl; done += len(bi)
    np.savez_compressed(cache, treeID=tid_str, proj=proj, pooled=pooled)
    print(f"[{which}] extracted {done}/{n} in {time.time()-t0:.0f}s -> {cache.name}")
    return proj, pooled


# ----------------------------- eval harness (exact reuse) -----------------------------
def to11(X, seed):
    Xs = StandardScaler().fit_transform(X)
    k = min(11, Xs.shape[1], Xs.shape[0] - 1)
    return PCA(n_components=k, random_state=seed).fit_transform(Xs)


def cluster_score(X, y, K, metric, seeds=SEEDS):
    vals = []
    for s in seeds:
        Z = to11(X, s)
        cl = KMeans(n_clusters=K, random_state=s, n_init=10).fit_predict(Z)
        vals.append(metric(y, cl))
    return float(np.mean(vals)), float(np.std(vals))


def within_source_nmi(X, y, groups, pca_dim=None, seed=0):
    """Mirror gen_source_control.within_source_nmi: per-block cluster, size-weighted mean NMI."""
    tot, wsum = 0.0, 0
    for g in sorted(set(groups)):
        mask = groups == g; yy = y[mask]
        if len(set(yy)) < 2 or mask.sum() < 4:
            continue
        Xs = StandardScaler().fit_transform(X[mask])
        if pca_dim is not None:
            k = min(pca_dim, Xs.shape[1], Xs.shape[0] - 1)
            Xs = PCA(n_components=k, random_state=seed).fit_transform(Xs)
        K = len(set(yy))
        cl = KMeans(n_clusters=K, random_state=seed, n_init=10).fit_predict(Xs)
        tot += NMI(yy, cl) * mask.sum(); wsum += mask.sum()
    return round(tot / wsum, 4) if wsum else 0.0


def score_features(X, y, K, groups, pca_dim_ws):
    nmi_m, nmi_s = cluster_score(X, y, K, NMI)
    ami_m, ami_s = cluster_score(X, y, K, AMI)
    return {
        "NMI_mean": round(nmi_m, 4), "NMI_sd": round(nmi_s, 4),
        "AMI_mean": round(ami_m, 4), "AMI_sd": round(ami_s, 4),
        "within_source_NMI": within_source_nmi(X, y, groups, pca_dim=pca_dim_ws),
    }


def run(which, model):
    print(f"\n===== {which} =====")
    a = load_aligned(which)
    y, groups, tid = a["y"], a["groups"], a["treeID"]
    K = len(set(y)); n = len(y)
    print(f"[{which}] n={n} K={K} sources={len(set(groups))}")
    proj, pooled = extract_embeddings(which, tid, model)
    # sanity on embeddings
    assert not np.isnan(proj).any(), "NaN in proj embeddings"
    # self-check on cached CLIP features of THIS subset
    clip_nmi_m, clip_nmi_s = cluster_score(a["Xc"], y, K, NMI)
    tgt = SELFCHECK_TARGET[which]
    ok = abs(clip_nmi_m - tgt) <= SELFCHECK_TOL
    print(f"[{which}] SELF-CHECK CLIP NMI={clip_nmi_m:.4f} (target {tgt}) -> {'PASS' if ok else 'FAIL'}")
    res = {
        "n": n, "K": K, "n_sources": len(set(groups)),
        "self_check": {"clip_NMI_mean": round(clip_nmi_m, 4), "clip_NMI_sd": round(clip_nmi_s, 4),
                       "target": tgt, "tol": SELFCHECK_TOL, "pass": bool(ok)},
        "clip": score_features(a["Xc"], y, K, groups, pca_dim_ws=11),
        "geo": score_features(a["Xsf"], y, K, groups, pca_dim_ws=None),
        "tdfm_proj_1280": score_features(proj, y, K, groups, pca_dim_ws=11),
        "tdfm_pooled_256": score_features(pooled, y, K, groups, pca_dim_ws=11),
    }
    for kk in ["clip", "geo", "tdfm_proj_1280", "tdfm_pooled_256"]:
        v = res[kk]
        print(f"[{which}] {kk:16s} NMI {v['NMI_mean']:.4f}±{v['NMI_sd']:.4f}  "
              f"AMI {v['AMI_mean']:.4f}  within-src {v['within_source_NMI']:.4f}")
    return res


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    targets = ["biodiv", "for"] if which == "both" else [which]
    model, ckpt = get_model()
    print(f"model=OpenShape PointBERT/PPAT (scaling=1) ckpt={ckpt} dev={DEV}")
    out = {
        "note": "Frozen OpenShape PointBERT (PPAT) 3D-FM baseline vs CLIP vs geo; matching-free "
                "global NMI/AMI over 15 seeds; within_source_NMI from one run (seed 0, n_init=10); primary 3D-FM embedding = 1280-d CLIP-aligned projected.",
        "model": "OpenShape/openshape-pointbert-shapenet (PPAT scaling=1, ShapeNet-pretrained, FROZEN)",
        "embedding_primary": "tdfm_proj_1280", "N_points": N_POINTS, "seeds": len(SEEDS),
        "grouping": "pure-torch FPS (start=0) + ball-query (r=0.4,nsamp=256); no custom CUDA op",
    }
    # merge with existing file if present (so a single-dataset run doesn't clobber the other)
    fp = OUT / "unsup_3dfm.json"
    if fp.exists():
        try:
            prev = json.load(open(fp)); out = {**prev, **out}
        except Exception:
            pass
    for w in targets:
        out[w] = run(w, model)
    json.dump(out, open(fp, "w"), indent=2)
    print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
