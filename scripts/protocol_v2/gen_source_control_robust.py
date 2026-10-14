"""Within-source control WITH uncertainty (blind-review round-3 Major #2) + within-species source recoverability
(#3). Adds multi-seed spread to the within-source NMI deltas and reports the within-species source-prediction
balanced accuracy at MATCHED 11-d (like-for-like with the headline), so the source-confound discussion carries
proper uncertainty and the competing 'residual may still be source' evidence is included.

Outputs runs/protocol_v2/gen/source_control_robust.json. CPU-only.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import normalized_mutual_info_score as NMI
import gen_score as G
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"
SEEDS = list(range(15))


def within_source_nmi_seed(X, y, groups, seed, pca_dim=None):
    tot, wsum = 0.0, 0
    for g in sorted(set(groups)):
        mask = groups == g; yy = y[mask]
        if len(set(yy)) < 2 or mask.sum() < 4: continue
        Xs = StandardScaler().fit_transform(X[mask])
        if pca_dim is not None:
            k = min(pca_dim, Xs.shape[1], Xs.shape[0] - 1)
            Xs = PCA(n_components=k, random_state=seed).fit_transform(Xs)
        cl = KMeans(n_clusters=len(set(yy)), random_state=seed, n_init=10).fit_predict(Xs)
        tot += NMI(yy, cl) * mask.sum(); wsum += mask.sum()
    return tot / wsum if wsum else 0.0


def arm_ws(X, y, groups, pca_dim=None):
    v = [within_source_nmi_seed(X, y, groups, s, pca_dim) for s in SEEDS]
    return v


def ci(v):
    v = np.asarray(v); rng = np.random.default_rng(0)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def within_species_source(X, y, groups, pca_dim=None, seed=0):
    """avg balanced-acc of predicting source WITHIN each species (matched dim), weighted by species size."""
    Xs = StandardScaler().fit_transform(X)
    if pca_dim is not None:
        k = min(pca_dim, Xs.shape[1], Xs.shape[0] - 1)
        Xs = PCA(n_components=k, random_state=seed).fit_transform(Xs)
    tot, wsum, chances = 0.0, 0, []
    for sp in set(y):
        m = y == sp; gg = groups[m]
        cnt = Counter(gg); keep_g = [g for g in cnt if cnt[g] >= 4]
        if len(keep_g) < 2: continue
        km = np.isin(gg, keep_g)
        if km.sum() < 12: continue
        Xsp = Xs[m][km]; gsp = gg[km]
        try:
            cv = StratifiedKFold(n_splits=min(3, min(Counter(gsp).values())), shuffle=True, random_state=seed)
            sc = cross_val_score(LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=1),
                                 Xsp, gsp, cv=cv, scoring="balanced_accuracy").mean()
            tot += sc * km.sum(); wsum += km.sum(); chances.append(1.0 / len(set(gsp)))
        except Exception:
            continue
    return (round(tot / wsum, 4) if wsum else None, round(float(np.mean(chances)), 4) if chances else None)


def main():
    out = {"n_seeds": len(SEEDS)}
    for which in ["for", "biodiv"]:
        if which == "for":
            d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
            g = np.load(ROOT / "data/raw/for_species20k/geo_features_matched.npz", allow_pickle=True)
            Xc = d["X"]; y = d["species"].astype(str); groups = d["dataset"].astype(str); Xsf = g["X_sf"]
        else:
            d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
            Xc = d["X_clip"]; Xsf = d["X_sf"]; y = d["species"].astype(str); groups = d["region"].astype(str)
        m, _ = G.analyzable_mask(y, groups); Xc, Xsf, y, groups = Xc[m], Xsf[m], y[m], groups[m]
        if which == "biodiv":
            c = Counter(y); top = max(c, key=c.get); n = len(y); cap = 0.39
            if c[top] / n > cap:
                tgt = int(cap/(1-cap)*(n-c[top])); idx = np.where(y == top)[0]
                drop = set(np.random.RandomState(0).choice(idx, c[top]-tgt, replace=False))
                k = np.array([i not in drop for i in range(n)]); Xc, Xsf, y, groups = Xc[k], Xsf[k], y[k], groups[k]
        clip_v = arm_ws(Xc, y, groups, pca_dim=11)
        geo_v = arm_ws(Xsf, y, groups, pca_dim=None)
        delta = np.array(clip_v) - np.array(geo_v)
        wss_clip = within_species_source(Xc, y, groups, pca_dim=11)
        wss_geo = within_species_source(Xsf, y, groups, pca_dim=None)
        out[which] = {
            "n_blocks": len(set(groups)),
            "within_source_NMI_CLIP_11d_mean": round(float(np.mean(clip_v)), 4),
            "within_source_NMI_CLIP_11d_std": round(float(np.std(clip_v)), 4),
            "within_source_NMI_geo_11d_mean": round(float(np.mean(geo_v)), 4),
            "within_source_delta_mean": round(float(delta.mean()), 4),
            "within_source_delta_ci95": ci(delta),
            "within_source_delta_frac_positive": round(float((delta > 0).mean()), 3),
            "within_species_source_balacc_CLIP_11d": wss_clip[0],
            "within_species_source_balacc_geo_11d": wss_geo[0],
            "within_species_source_chance": wss_clip[1],
        }
    json.dump(out, open(OUT / "source_control_robust.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
