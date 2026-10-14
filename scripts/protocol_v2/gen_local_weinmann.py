"""F-17 robustness check: a TRUE local multi-scale Weinmann geometric baseline (per-point
eigenfeatures at multiple neighbourhood scales, aggregated per tree), vs the paper's
whole-tree single-scale covariance descriptor and vs CLIP. Tests whether the CLIP>geometry
coherence gap survives against a *stronger, properly-Weinmann* geometry baseline (the paper's
descriptor is a global single-scale covariance, which R4/R3 flagged as a mislabel + weakening).

Same analyzable subset / same StandardScaler->PCA11->KMeans 15-seed global NMI harness (within-source NMI: one run, seed 0) as
gen_3dfm_baseline.py. CPU only. Output: runs/protocol_v2/gen/local_weinmann.json
Usage: python gen_local_weinmann.py <for|biodiv|both>
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_3dfm_baseline as B

WHICH = sys.argv[1] if len(sys.argv) > 1 else "both"
SCALES = [10, 20, 40]
MAXP = 4096
OUT = B.OUT


def tree_descriptor(xyz):
    n = xyz.shape[0]
    if n > MAXP:
        xyz = xyz[np.random.RandomState(0).choice(n, MAXP, replace=False)]
    xyz = xyz.astype(np.float64)
    P = xyz.shape[0]
    if P < max(SCALES) + 2:
        return np.zeros(len(SCALES) * 7, np.float32)
    kd = cKDTree(xyz)
    feats = []
    for k in SCALES:
        kk = min(k, P - 1)
        _, nn = kd.query(xyz, k=kk + 1)               # (P, kk+1) incl self
        neigh = xyz[nn]
        c = neigh - neigh.mean(1, keepdims=True)
        cov = np.einsum('pki,pkj->pij', c, c) / (kk + 1)
        ev = np.linalg.eigvalsh(cov)                  # ascending (P,3)
        e3, e2, e1 = ev[:, 0], ev[:, 1], ev[:, 2]     # e1>=e2>=e3
        e1c = np.clip(e1, 1e-12, None); s = e1 + e2 + e3 + 1e-12
        lin = (e1 - e2) / e1c; pla = (e2 - e3) / e1c; sph = e3 / e1c; ani = (e1 - e3) / e1c
        p1, p2, p3 = e1 / s, e2 / s, e3 / s
        ent = -(p1 * np.log(p1 + 1e-12) + p2 * np.log(p2 + 1e-12) + p3 * np.log(p3 + 1e-12))
        cur = e3 / s
        feats += [lin.mean(), pla.mean(), sph.mean(), ani.mean(), ent.mean(), cur.mean(), p1.mean()]
    return np.array(feats, np.float32)


def build_descriptors(which, treeIDs):
    cache = OUT / f"local_weinmann_{which}.npz"
    tid_str = np.array([str(t) for t in treeIDs])
    if cache.exists():
        c = np.load(cache, allow_pickle=True)
        cmap = {t: i for i, t in enumerate(c["treeID"].astype(str))}
        if all(t in cmap for t in tid_str):
            order = [cmap[t] for t in tid_str]
            print(f"[{which}] loaded cached {cache.name}"); return c["X"][order]
    reader = B.read_clouds_for if which == "for" else B.read_clouds_biodiv
    n = len(treeIDs); X = np.zeros((n, len(SCALES) * 7), np.float32)
    t0 = time.time()
    for i, xyz in reader(treeIDs):
        X[i] = tree_descriptor(xyz)
        if (i + 1) % 1000 == 0:
            print(f"[{which}] {i+1}/{n}  {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(cache, treeID=tid_str, X=X)
    print(f"[{which}] built {n} local-Weinmann descriptors in {time.time()-t0:.0f}s")
    return X


def run(which):
    print(f"\n===== {which} =====")
    a = B.load_aligned(which); y, groups, tid = a["y"], a["groups"], a["treeID"]
    K = len(set(y))
    Xlw = build_descriptors(which, tid)
    res = {
        "n": len(y), "K": K, "scales": SCALES, "dim": Xlw.shape[1],
        "clip": B.score_features(a["Xc"], y, K, groups, pca_dim_ws=11),
        "geo_global_single_scale": B.score_features(a["Xsf"], y, K, groups, pca_dim_ws=None),
        "geo_local_multiscale_weinmann": B.score_features(Xlw, y, K, groups, pca_dim_ws=11),
    }
    for kk in ["clip", "geo_global_single_scale", "geo_local_multiscale_weinmann"]:
        v = res[kk]
        print(f"[{which}] {kk:32s} NMI {v['NMI_mean']:.4f}  within-src {v['within_source_NMI']:.4f}")
    c = res["clip"]["NMI_mean"]; lw = res["geo_local_multiscale_weinmann"]["NMI_mean"]
    res["clip_beats_local_weinmann"] = bool(c > lw); res["clip_minus_local_weinmann_NMI"] = round(c - lw, 4)
    print(f"[{which}] CLIP>local-Weinmann = {res['clip_beats_local_weinmann']}  (delta={c-lw:+.4f})")
    return res


def main():
    targets = ["biodiv", "for"] if WHICH == "both" else [WHICH]
    out = {"note": "F-17 robustness: true local multi-scale Weinmann (k=10/20/40 per-point "
                   "eigenfeatures, per-tree mean) vs paper's whole-tree single-scale covariance vs CLIP. "
                   "Same subset/harness.", "scales": SCALES}
    for w in targets:
        out[w] = run(w)
    fp = OUT / "local_weinmann.json"; json.dump(out, open(fp, "w"), indent=2)
    print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
