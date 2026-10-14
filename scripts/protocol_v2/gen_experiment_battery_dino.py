"""DINOv2 half of the experiment battery (for the three-way Fig 3 / Fig 4): clustering-method
comparison (kmeans/gmm/spectral/agglo fix-K + hdbscan/dpgmm estimate-K) and PCA dimension sweep
for DINOv2, reusing gen_experiment_battery's EXACT primitives (rep/fixk_method/estimatek_method).
DINOv2 aligned to the CLIP/geo feature file by treeID, identical analyzable subset + Fagus<39%.

Outputs runs/protocol_v2/gen/{clustering_methods_dino,dim_sweep_dino}_{which}.json
Usage: python gen_experiment_battery_dino.py for | biodiv
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI
from gen_experiment_battery import rep, fixk_method, estimatek_method, mean_std, analyzable_mask, SEEDS

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"


def load_dino(which):
    if which == "for":
        d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
        g = np.load(ROOT / "data/raw/for_species20k/geo_features_matched.npz", allow_pickle=True)
        dn = np.load(ROOT / "data/raw/for_species20k/for_dino.npz", allow_pickle=True)
        y = d["species"].astype(str); groups = d["dataset"].astype(str); Xsf = g["X_sf"]; tref = d["treeID"]; fagus = None
    else:
        d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
        dn = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_dino.npz", allow_pickle=True)
        Xsf = d["X_sf"]; y = d["species"].astype(str); groups = d["region"].astype(str); tref = d["treeID"]; fagus = 0.39
    idx = {str(t): i for i, t in enumerate(dn["treeID"])}
    rows = np.array([idx.get(str(t), -1) for t in tref]); assert (rows >= 0).all(), "DINOv2 alignment failed"
    Xd = dn["X_dino"][rows]
    mask, _ = analyzable_mask(y, groups); Xd, Xsf, y, groups = Xd[mask], Xsf[mask], y[mask], groups[mask]
    if fagus is not None:
        c = Counter(y); top = max(c, key=c.get); n = len(y)
        if c[top] / n > fagus:
            tgt = int(fagus / (1 - fagus) * (n - c[top])); ii = np.where(y == top)[0]
            drop = set(np.random.RandomState(0).choice(ii, c[top] - tgt, replace=False))
            k = np.array([i not in drop for i in range(n)]); Xd, Xsf, y = Xd[k], Xsf[k], y[k]
    return Xd, Xsf, y


def main():
    which = sys.argv[1]
    Xd, Xsf, y = load_dino(which)
    K = len(set(y)); dino11 = rep(Xd, 11); geo11 = rep(Xsf, 11)
    print(f"[{which}] n={len(y)} K={K} (DINOv2 battery)", flush=True)

    methods = {}
    for m in ["kmeans", "gmm", "spectral", "agglo"]:
        methods[m] = {"DINOv2_11d": fixk_method(m, dino11, K, y), "geo_11d": fixk_method(m, geo11, K, y)}
        print(f"  {m}: DINOv2 NMI {methods[m]['DINOv2_11d']['NMI']['mean']} | geo {methods[m]['geo_11d']['NMI']['mean']}", flush=True)
    estk = {m: {"DINOv2_11d": estimatek_method(m, dino11, y, K)} for m in ["hdbscan", "dpgmm"]}
    json.dump({"dataset": which, "n": int(len(y)), "true_K": K, "fix_K_methods": methods, "estimate_K_methods": estk},
              open(OUT / f"clustering_methods_dino_{which}.json", "w"), indent=2)

    sweep = {}
    for dd in [4, 8, 11, 16, 32, 64, 128, 256, 512, 768]:
        if dd > Xd.shape[1]:
            continue
        Xz = rep(Xd, dd)
        nmis = [NMI(y, KMeans(K, n_init=5, random_state=s).fit_predict(Xz)) for s in SEEDS]
        sweep[str(dd)] = mean_std(nmis); print(f"  dim {dd}: DINOv2 NMI {sweep[str(dd)]['mean']}", flush=True)
    geo_ref = mean_std([NMI(y, KMeans(K, n_init=5, random_state=s).fit_predict(geo11)) for s in SEEDS])
    json.dump({"dataset": which, "true_K": K, "DINOv2_pca_dim_sweep_NMI": sweep, "geo_11d_NMI_reference": geo_ref},
              open(OUT / f"dim_sweep_dino_{which}.json", "w"), indent=2)
    print(f"[{which}] DINOv2 battery done.", flush=True)


if __name__ == "__main__":
    main()
