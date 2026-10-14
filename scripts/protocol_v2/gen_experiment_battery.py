"""Experiment battery to thicken the paper to field-standard (Lu's asks + DEC/SCAN/TEMI/PointDC template):
  (1) clustering-method comparison at matched 11-d: KMeans/GMM/Spectral/Agglomerative (fix-K) vs
      HDBSCAN/DP-GMM (estimate-K), for CLIP-11d and geo-11d, primary metric NMI (+ Hungarian ACC).
  (2) dimensionality sweep: CLIP PCA dim in {4,8,11,16,32,64,128,256,512}, KMeans@K, NMI; geo-11d reference.
  (3) K-sensitivity: KMeans at K = {0.5,0.8,1.0,1.2,1.5,2,5}x #species, NMI + AMI (AMI for K>>#species);
      plus estimated K-hat from DP-GMM and HDBSCAN vs the true #species.
CPU-only, n_jobs=1, modest n_init; O(n^2) methods (spectral/agglomerative) subsample large datasets.
Outputs runs/protocol_v2/gen/{clustering_methods,dim_sweep,k_sensitivity}_{which}.json
Usage: python gen_experiment_battery.py for | biodiv
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, HDBSCAN
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_mutual_info_score as AMI
from scipy.optimize import linear_sum_assignment
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"
SEEDS = list(range(5))
SUBSAMPLE = 4000  # cap for O(n^2) spectral/agglomerative


def analyzable_mask(y, groups, min_n=30, min_blocks=2):
    keep = [s for s in set(y) if (y == s).sum() >= min_n and len(set(groups[y == s])) >= min_blocks]
    return np.isin(y, keep), sorted(keep)


def rep(X, d, seed=0):
    Xs = StandardScaler().fit_transform(X)
    if d is None or d >= Xs.shape[1]:
        return Xs
    return PCA(n_components=d, random_state=seed).fit_transform(Xs)


def hung_acc(y, cl):
    clusters = sorted(set(cl)); classes = sorted(set(y))
    ci = {c: i for i, c in enumerate(clusters)}; yi = {s: i for i, s in enumerate(classes)}
    M = np.zeros((len(clusters), len(classes)))
    for a, b in zip(cl, y):
        M[ci[a], yi[b]] += 1
    r, c = linear_sum_assignment(-M)
    return float(M[r, c].sum() / len(y))


def mean_std(vals):
    v = np.asarray(vals, float)
    return {"mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4)}


def fixk_method(method, X, K, y):
    nmis, accs = [], []
    Xc = X
    if method in ("spectral", "agglo") and X.shape[0] > SUBSAMPLE:
        idx = np.random.RandomState(0).choice(X.shape[0], SUBSAMPLE, replace=False)
        Xc, yc = X[idx], y[idx]
    else:
        yc = y
    seeds = SEEDS if method in ("kmeans", "gmm", "spectral") else [0]
    for s in seeds:
        if method == "kmeans":
            cl = KMeans(K, n_init=5, random_state=s).fit_predict(Xc)
        elif method == "gmm":
            cl = GaussianMixture(K, covariance_type="diag", reg_covar=1e-3,
                                 max_iter=200, random_state=s).fit_predict(Xc)
        elif method == "spectral":
            cl = SpectralClustering(K, affinity="nearest_neighbors", n_neighbors=15,
                                    assign_labels="kmeans", random_state=s).fit_predict(Xc)
        elif method == "agglo":
            cl = AgglomerativeClustering(K, linkage="ward").fit_predict(Xc)
        nmis.append(NMI(yc, cl)); accs.append(hung_acc(yc, cl))
    out = {"NMI": mean_std(nmis), "hungarian_acc": mean_std(accs), "n_seeds": len(seeds)}
    if Xc.shape[0] != X.shape[0]:
        out["subsampled_n"] = int(Xc.shape[0])
    return out


def estimatek_method(method, X, y, true_k):
    if method == "hdbscan":
        cl = HDBSCAN(min_cluster_size=30).fit_predict(X)
        khat = len(set(cl) - {-1}); noise = float((cl == -1).mean())
        return {"k_hat": int(khat), "true_k": int(true_k), "noise_frac": round(noise, 3),
                "NMI": round(float(NMI(y, cl)), 4), "AMI": round(float(AMI(y, cl)), 4)}
    if method == "dpgmm":
        bgm = BayesianGaussianMixture(n_components=min(3 * true_k, 40),
                                      covariance_type="diag", reg_covar=1e-3,
                                      weight_concentration_prior_type="dirichlet_process",
                                      max_iter=300, random_state=0).fit(X)
        cl = bgm.predict(X); khat = int((bgm.weights_ > 0.01).sum())
        return {"k_hat": khat, "true_k": int(true_k),
                "NMI": round(float(NMI(y, cl)), 4), "AMI": round(float(AMI(y, cl)), 4)}


def load(which):
    if which == "for":
        d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
        Xc = d["X"]; y = d["species"].astype(str); groups = d["dataset"].astype(str)
        gp = ROOT / "data/raw/for_species20k/geo_features_matched.npz"
        Xsf = np.load(gp, allow_pickle=True)["X_sf"]; fagus = None
    else:
        d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
        Xc = d["X_clip"]; Xsf = d["X_sf"]; y = d["species"].astype(str); groups = d["region"].astype(str)
        fagus = 0.39
    mask, _ = analyzable_mask(y, groups)
    Xc, Xsf, y, groups = Xc[mask], Xsf[mask], y[mask], groups[mask]
    if fagus is not None:
        c = Counter(y); top = max(c, key=c.get); n = len(y)
        if c[top] / n > fagus:
            tgt = int(fagus / (1 - fagus) * (n - c[top])); idx = np.where(y == top)[0]
            drop = set(np.random.RandomState(0).choice(idx, c[top] - tgt, replace=False))
            k = np.array([i not in drop for i in range(n)])
            Xc, Xsf, y, groups = Xc[k], Xsf[k], y[k], groups[k]
    return Xc, Xsf, y


def main():
    which = sys.argv[1]
    Xc, Xsf, y = load(which)
    K = len(set(y))
    clip11 = rep(Xc, 11); geo11 = rep(Xsf, 11)
    print(f"[{which}] n={len(y)} K={K}")

    # (1) clustering methods
    methods = {}
    for m in ["kmeans", "gmm", "spectral", "agglo"]:
        methods[m] = {"CLIP_11d": fixk_method(m, clip11, K, y), "geo_11d": fixk_method(m, geo11, K, y)}
        print(f"  method {m}: CLIP NMI {methods[m]['CLIP_11d']['NMI']['mean']} | geo {methods[m]['geo_11d']['NMI']['mean']}")
    estk = {}
    for m in ["hdbscan", "dpgmm"]:
        estk[m] = {"CLIP_11d": estimatek_method(m, clip11, y, K), "geo_11d": estimatek_method(m, geo11, y, K)}
        print(f"  estimate-K {m}: CLIP k_hat {estk[m]['CLIP_11d']['k_hat']} (true {K}) NMI {estk[m]['CLIP_11d']['NMI']}")
    json.dump({"dataset": which, "n": int(len(y)), "true_K": K, "fix_K_methods": methods,
               "estimate_K_methods": estk,
               "note": "fix-K methods at K=#species, matched 11-d; spectral/agglo subsampled if n>%d" % SUBSAMPLE},
              open(OUT / f"clustering_methods_{which}.json", "w"), indent=2)

    # (2) dimension sweep (CLIP, KMeans), geo-11d reference
    dims = [4, 8, 11, 16, 32, 64, 128, 256, 512]
    sweep = {}
    for d in dims:
        if d > Xc.shape[1]:
            continue
        Xd = rep(Xc, d)
        nmis = [NMI(y, KMeans(K, n_init=5, random_state=s).fit_predict(Xd)) for s in SEEDS]
        sweep[str(d)] = mean_std(nmis)
        print(f"  dim {d}: CLIP NMI {sweep[str(d)]['mean']}")
    geo_ref = mean_std([NMI(y, KMeans(K, n_init=5, random_state=s).fit_predict(geo11)) for s in SEEDS])
    json.dump({"dataset": which, "true_K": K, "CLIP_pca_dim_sweep_NMI": sweep, "geo_11d_NMI_reference": geo_ref},
              open(OUT / f"dim_sweep_{which}.json", "w"), indent=2)

    # (3) K-sensitivity (CLIP-11d, KMeans), NMI + AMI
    mults = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 5.0]
    ksens = {}
    for mu in mults:
        k = max(2, int(round(mu * K)))
        nmis = [NMI(y, KMeans(k, n_init=5, random_state=s).fit_predict(clip11)) for s in SEEDS]
        amis = [AMI(y, KMeans(k, n_init=5, random_state=s).fit_predict(clip11)) for s in SEEDS]
        ksens[f"{mu}x_K{k}"] = {"NMI": mean_std(nmis), "AMI": mean_std(amis)}
        print(f"  K={k} ({mu}x): NMI {ksens[f'{mu}x_K{k}']['NMI']['mean']} AMI {ksens[f'{mu}x_K{k}']['AMI']['mean']}")
    json.dump({"dataset": which, "true_K": K, "CLIP_11d_K_sensitivity": ksens,
               "estimated_K": {m: {"CLIP_11d_k_hat": estk[m]["CLIP_11d"]["k_hat"]} for m in estk}},
              open(OUT / f"k_sensitivity_{which}.json", "w"), indent=2)
    print(f"[{which}] battery done.")


if __name__ == "__main__":
    main()
