"""F-15 robustness check: re-run the frozen OpenShape PointBERT 3D-FM baseline at a LARGER
point budget than the pre-registered N=1024 (which was inherited from the Point-MAE convention),
to test whether the 2D>3D>geometry ordering survives when the 3D model is given its native-scale
input. Everything else is IDENTICAL to gen_3dfm_baseline.py (same subset, same frozen model,
same StandardScaler->PCA11->KMeans 15-seed harness for the global scores; within-source NMI is one run, seed 0). Separate cache + output so the N=1024
results are untouched.

Usage:  python gen_3dfm_np.py <for|biodiv|both> <N_points>   e.g.  gen_3dfm_np.py both 10000
Output: runs/protocol_v2/gen/unsup_3dfm_n{N}.json ; cache emb_3dfm_{which}_n{N}.npz
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, json
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_3dfm_baseline as B   # reuse EVERYTHING; only override N + cache

WHICH = sys.argv[1] if len(sys.argv) > 1 else "both"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
B.N_POINTS = N                  # patch the global used by B.preprocess()
OUT = B.OUT


def extract(which, treeIDs, model):
    cache = OUT / f"emb_3dfm_{which}_n{N}.npz"
    tid_str = np.array([str(t) for t in treeIDs])
    if cache.exists():
        c = np.load(cache, allow_pickle=True)
        cmap = {t: i for i, t in enumerate(c["treeID"].astype(str))}
        if all(t in cmap for t in tid_str):
            order = [cmap[t] for t in tid_str]
            print(f"[{which}] loaded cached {cache.name}")
            return c["proj"][order], c["pooled"][order]
    reader = B.read_clouds_for if which == "for" else B.read_clouds_biodiv
    n = len(treeIDs)
    proj = np.zeros((n, 1280), np.float32); pooled = np.zeros((n, 256), np.float32)
    bx, bf, bi = [], [], []; t0 = time.time(); done = 0
    for i, xyz in reader(treeIDs):
        p, fe = B.preprocess(xyz)       # uses patched B.N_POINTS
        bx.append(p); bf.append(fe); bi.append(i)
        if len(bx) == B.BATCH:
            pj, pl = B.embed_batch(model, bx, bf)
            proj[bi] = pj; pooled[bi] = pl; done += len(bi); bx, bf, bi = [], [], []
            if done % (B.BATCH * 8) == 0:
                print(f"[{which}] {done}/{n}  {time.time()-t0:.0f}s N={N}", flush=True)
    if bx:
        pj, pl = B.embed_batch(model, bx, bf); proj[bi] = pj; pooled[bi] = pl; done += len(bi)
    np.savez_compressed(cache, treeID=tid_str, proj=proj, pooled=pooled)
    print(f"[{which}] extracted {done}/{n} at N={N} in {time.time()-t0:.0f}s")
    return proj, pooled


def run(which, model):
    print(f"\n===== {which} (N={N}) =====")
    a = B.load_aligned(which); y, groups, tid = a["y"], a["groups"], a["treeID"]
    K = len(set(y)); n = len(y)
    print(f"[{which}] n={n} K={K} sources={len(set(groups))}")
    proj, pooled = extract(which, tid, model)
    assert not np.isnan(proj).any()
    clip_m, clip_s = B.cluster_score(a["Xc"], y, K, B.NMI)
    res = {
        "n": n, "K": K, "N_points": N,
        "clip": B.score_features(a["Xc"], y, K, groups, pca_dim_ws=11),
        "geo": B.score_features(a["Xsf"], y, K, groups, pca_dim_ws=None),
        "tdfm_proj_1280": B.score_features(proj, y, K, groups, pca_dim_ws=11),
        "tdfm_pooled_256": B.score_features(pooled, y, K, groups, pca_dim_ws=11),
    }
    for kk in ["clip", "geo", "tdfm_proj_1280", "tdfm_pooled_256"]:
        v = res[kk]
        print(f"[{which}] {kk:16s} NMI {v['NMI_mean']:.4f}  within-src {v['within_source_NMI']:.4f}")
    # explicit ordering verdict
    c, t, g = res["clip"]["NMI_mean"], res["tdfm_proj_1280"]["NMI_mean"], res["geo"]["NMI_mean"]
    res["ordering_2d_gt_3d_gt_geo"] = bool(c > t > g)
    res["clip_minus_3dfm_NMI"] = round(c - t, 4)
    print(f"[{which}] ORDERING 2D>3D>geo = {res['ordering_2d_gt_3d_gt_geo']}  (CLIP-3DFM={c-t:+.4f})")
    return res


def main():
    targets = ["biodiv", "for"] if WHICH == "both" else [WHICH]
    model, ckpt = B.get_model()
    print(f"model=OpenShape PointBERT ckpt={ckpt} dev={B.DEV} N_points={N}")
    out = {"note": f"F-15 robustness: OpenShape 3D-FM at N={N} (vs pre-reg N=1024). "
                   "Same subset/model/harness as gen_3dfm_baseline.py.", "N_points": N}
    for w in targets:
        out[w] = run(w, model)
    fp = OUT / f"unsup_3dfm_n{N}.json"
    json.dump(out, open(fp, "w"), indent=2)
    print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
