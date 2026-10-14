"""Track-1 cached stats hardening (publishability strengthening, zero honesty risk).
Produces runs/protocol_v2/gen/stats_hardening.json with:
  (1) AMI  — chance-corrected AMI on the headline CLIP-vs-geo comparison, both datasets, 15-seed mean
             (so a referee cannot dismiss NMI 0.278 at K=15 as cluster-count-inflated).
  (2) SOURCE-SUBSPACE-REMOVAL recluster (PRE-SPECIFIED single variant, per reports/PREREG_3d_baseline.md
      spirit): remove the between-source linear subspace (span of centered per-source means,
      rank = n_source-1) from the standardized CLIP features, then recluster and score NMI. Shows the
      residual species signal is not merely a leftover source axis. Also reports source-recovery bal-acc
      BEFORE vs AFTER removal to confirm the source axis is actually gone. Single variant, no cherry-pick.
  (3) HOLM multiple-comparison correction across the pre-trained-backbone permutation family
      (CLIP/DINOv2/ImageNet x FOR/BioDiv), symmetric with the supervised arm's Bonferroni.

Reuses the EXACT clustering + analyzable-mask + BioDiv-cap logic of gen_permutation_test.py so raw NMI
reproduces the paper (self-check). CPU-only.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import balanced_accuracy_score
from collections import Counter
import gen_score as G

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"
SEEDS = list(range(15))


def load(which):
    if which == "for":
        d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
        g = np.load(ROOT / "data/raw/for_species20k/geo_features_matched.npz", allow_pickle=True)
        Xc = d["X"]; y = d["species"].astype(str); grp = d["dataset"].astype(str); Xsf = g["X_sf"]
    else:
        d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
        Xc = d["X_clip"]; Xsf = d["X_sf"]; y = d["species"].astype(str); grp = d["region"].astype(str)
    m, _ = G.analyzable_mask(y, grp); Xc, Xsf, y, grp = Xc[m], Xsf[m], y[m], grp[m]
    if which == "biodiv":
        c = Counter(y); top = max(c, key=c.get); n = len(y); cap = 0.39
        if c[top] / n > cap:
            tgt = int(cap / (1 - cap) * (n - c[top])); idx = np.where(y == top)[0]
            drop = set(np.random.RandomState(0).choice(idx, c[top] - tgt, replace=False))
            k = np.array([i not in drop for i in range(n)]); Xc, Xsf, y, grp = Xc[k], Xsf[k], y[k], grp[k]
    return Xc, Xsf, y, grp


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


def remove_source_subspace(X, grp):
    """Project out the between-source linear subspace (span of centered per-source means)."""
    Xs = StandardScaler().fit_transform(X)
    srcs = sorted(set(grp)); gm = Xs.mean(0)
    M = np.array([Xs[grp == s].mean(0) - gm for s in srcs])          # (n_src, d)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    rank = min(len(srcs) - 1, Vt.shape[0])
    B = Vt[:rank]                                                     # orthonormal rows
    return Xs - (Xs @ B.T) @ B, rank


def source_balacc(X, grp):
    """5-fold balanced-accuracy of recovering source from features (logistic)."""
    le = {s: i for i, s in enumerate(sorted(set(grp)))}
    g = np.array([le[s] for s in grp])
    Z = to11(X, 0)
    clf = LogisticRegression(max_iter=1000)
    sc = cross_val_score(clf, Z, g, cv=5, scoring="balanced_accuracy")
    return float(np.mean(sc)), 1.0 / len(le)


def holm(pvals: dict):
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); adj = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        a = min(1.0, p * (m - i)); running = max(running, a); adj[k] = round(running, 6)
    return adj


def main():
    out = {"note": "Track-1 stats hardening: AMI + pre-specified source-subspace-removal recluster + Holm MCC"}
    for which in ["for", "biodiv"]:
        Xc, Xsf, y, grp = load(which); K = len(set(y))
        ami_clip, _ = cluster_score(Xc, y, K, AMI)
        ami_geo, _ = cluster_score(Xsf, y, K, AMI)
        nmi_clip_raw, _ = cluster_score(Xc, y, K, NMI)          # self-check vs paper
        Xc_sr, rank = remove_source_subspace(Xc, grp)
        nmi_clip_sr, sd_sr = cluster_score(Xc_sr, y, K, NMI, seeds=SEEDS)
        ba_before, chance = source_balacc(Xc, grp)
        ba_after, _ = source_balacc(Xc_sr, grp)
        out[which] = {
            "K": K, "n": int(len(y)), "n_sources": len(set(grp)),
            "AMI_clip": round(ami_clip, 4), "AMI_geo": round(ami_geo, 4),
            "AMI_advantage": round(ami_clip - ami_geo, 4),
            "NMI_clip_raw_selfcheck": round(nmi_clip_raw, 4),
            "source_subspace_rank_removed": int(rank),
            "NMI_clip_after_source_removal": round(nmi_clip_sr, 4),
            "NMI_clip_after_source_removal_sd": round(sd_sr, 4),
            "source_balacc_before": round(ba_before, 4),
            "source_balacc_after_removal": round(ba_after, 4),
            "source_balacc_chance": round(chance, 4),
        }
        print(which, out[which])
    # Holm across pre-trained-backbone permutation family
    fam = {}
    for f, keys in [("perm_test.json", ["for", "biodiv"]),
                    ("perm_test_dino.json", ["for", "biodiv"])]:
        p = OUT / f
        if p.exists():
            j = json.load(open(p))
            for kk in keys:
                fam[f"{f.split('.')[0]}:{kk}"] = j[kk]["p_value"]
    for which in ["for", "biodiv"]:
        p = OUT / f"perm_test_multi_{which}.json"
        if p.exists():
            j = json.load(open(p))
            for bk, bv in j.items():
                if isinstance(bv, dict) and "p_value" in bv and "imagenet" in bk.lower():
                    fam[f"imagenet:{which}"] = bv["p_value"]
    out["holm_family_pvalues"] = fam
    out["holm_adjusted"] = holm(fam) if fam else {}
    out["holm_all_survive_0.05"] = all(v <= 0.05 for v in out["holm_adjusted"].values()) if fam else None
    json.dump(out, open(OUT / "stats_hardening.json", "w"), indent=2)
    print("\nHOLM:", out["holm_family_pvalues"], "->", out["holm_adjusted"])
    print("wrote", OUT / "stats_hardening.json")


if __name__ == "__main__":
    main()
