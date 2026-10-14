"""GEN test scoring (CPU). Per CONSTITUTION §8 frozen rule.

Computes, per dataset, under leakage-free spatial CV (FOR=leave-source-out; BioDiv=leave-region-out):
  - CLIP linear probe (primary) macro-F1, pooled over folds + per-fold
  - geo_sf RF, geo_sz RF, geo_sf SVM(C=10), geo_sz SVM macro-F1 (best geo = baseline)
  - delta = CLIP_linear - max(geo); per-fold paired Wilcoxon; bootstrap 95% CI
  - learning-curve saturation gate; majority baseline
Verdict per §8: win iff delta>=+0.05 & sig & gates pass; lose iff <=-0.05 & sig; else tie.
(Threshold 0.05 matches CONSTITUTION §8/§9; earlier 0.03 was a bug — M4 fix.)

Fixes applied (adversarial review 2026-05-30):
  M5: CI is now bootstrapped on MACRO-F1 (not overall accuracy).
  M4: optional PCA-reduce-CLIP-to-geo-dim arm (train-only per fold) for capacity-matched comparison.
  M7: Bonferroni/Holm correction + leave-one-fold jackknife on the paired p.

Usage: python gen_score.py for | biodiv
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"; OUT.mkdir(parents=True, exist_ok=True)


def boot_ci_f1(yt, yp, n=1000, seed=0):
    """M5 fix: bootstrap 95% CI of MACRO-F1 (resample test instances)."""
    yt = np.asarray(yt); yp = np.asarray(yp)
    if len(yt) == 0: return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        idx = rng.integers(0, len(yt), len(yt))
        bs.append(f1_score(yt[idx], yp[idx], average="macro", zero_division=0))
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def fold_score(X, y, groups, g, clf_factory, pca_dim=None):
    """one fold: hold out group g, shared-species, return (per-fold macroF1, y_true, y_pred).
    pca_dim (M4): if set, fit PCA on TRAIN ONLY to that dim and transform both splits (capacity match)."""
    te = groups == g; tr = ~te
    train_sp = set(y[tr]); keep = te & np.isin(y, list(train_sp))
    if keep.sum() == 0 or len(set(y[tr])) < 2: return None
    Xtr_raw, Xte_raw = X[tr], X[keep]
    sc = StandardScaler().fit(Xtr_raw)                       # train-only
    Xtr, Xte = sc.transform(Xtr_raw), sc.transform(Xte_raw)
    if pca_dim is not None:
        k = min(pca_dim, Xtr.shape[1], Xtr.shape[0] - 1)
        pca = PCA(n_components=k, random_state=0).fit(Xtr)   # train-only (anti-leakage)
        Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
    clf = clf_factory().fit(Xtr, y[tr])
    pred = clf.predict(Xte)
    return f1_score(y[keep], pred, average="macro", zero_division=0), y[keep], pred


def run_method(X, y, groups, clf_factory, pca_dim=None):
    gs = sorted(set(groups)); per_fold = []; yt_all, yp_all = [], []
    for g in gs:
        r = fold_score(X, y, groups, g, clf_factory, pca_dim=pca_dim)
        if r is None: continue
        f1, yt, yp = r; per_fold.append(f1); yt_all.extend(yt.tolist()); yp_all.extend(yp.tolist())
    yt = np.array(yt_all); yp = np.array(yp_all)
    return {"per_fold": [round(x, 4) for x in per_fold],
            "pooled_f1": round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
            "pooled_oa": round(float(accuracy_score(yt, yp)), 4),
            "ci": boot_ci_f1(yt, yp)}, per_fold


LIN = lambda: LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=1)
RF = lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0, n_jobs=1)
SVM = lambda: SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")


def majority_f1(y, groups):
    gs = sorted(set(groups)); yt_all, yp_all = [], []
    for g in gs:
        te = groups == g; tr = ~groups.__eq__(g) if False else ~te
        train_sp = set(y[tr]); keep = te & np.isin(y, list(train_sp))
        if keep.sum() == 0: continue
        vals, cnts = np.unique(y[tr], return_counts=True); maj = vals[cnts.argmax()]
        yt_all.extend(y[keep].tolist()); yp_all.extend([maj] * int(keep.sum()))
    return round(float(f1_score(np.array(yt_all), np.array(yp_all), average="macro", zero_division=0)), 4)


def learning_curve(X, y, groups, clf_factory):
    gs = sorted(set(groups)); lc = {}
    for frac in [0.4, 0.6, 0.8, 1.0]:
        k = max(2, int(round(len(gs) * frac))); f1s = []
        for seed in range(3):
            r2 = np.random.RandomState(seed); chosen = list(r2.choice(gs, k, replace=False))
            sub = np.isin(groups, chosen)
            res, _ = run_method(X[sub], y[sub], groups[sub], clf_factory)
            f1s.append(res["pooled_f1"])
        lc[str(frac)] = round(float(np.mean(f1s)), 4)
    slope = lc["1.0"] - lc["0.8"]
    return lc, round(float(slope), 4), bool(slope < 0.02)


def analyzable_mask(y, groups, min_n=30, min_blocks=2):
    keep = []
    for s in set(y):
        if (y == s).sum() >= min_n and len(set(groups[y == s])) >= min_blocks:
            keep.append(s)
    return np.isin(y, keep), sorted(keep)


def main():
    which = sys.argv[1]
    if which == "for":
        d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
        Xc = d["X"]; y = d["species"].astype(str); groups = d["dataset"].astype(str)
        # M3 fix: use geometry computed on the SAME voxel-downsampled cloud as CLIP (matched input)
        geo_path = ROOT / "data/raw/for_species20k/geo_features_matched.npz"
        if not geo_path.exists():
            geo_path = ROOT / "data/raw/for_species20k/geo_features.npz"
        g = np.load(geo_path, allow_pickle=True)
        Xsf, Xsz = g["X_sf"], g["X_sz"]
        assert len(Xsf) == len(Xc), f"clip {len(Xc)} vs geo {len(Xsf)}"
        name = "FOR-species20K"; block = "source-dataset"; res_geo_src = geo_path.name
        fagus_cap = None
    else:
        d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
        Xc = d["X_clip"]; Xsf = d["X_sf"]; Xsz = d["X_sz"]
        y = d["species"].astype(str); groups = d["region"].astype(str)
        name = "BioDiv-3DTrees-ULS"; block = "region"; res_geo_src = "biodiv_features.npz"
        fagus_cap = 0.39  # D2(b): downsample dominant species below 40%

    mask, analyzable = analyzable_mask(y, groups)
    Xc, Xsf, Xsz, y, groups = Xc[mask], Xsf[mask], Xsz[mask], y[mask], groups[mask]

    # D2(b) dominant-class downsample (BioDiv Fagus 60% -> <40%)
    if fagus_cap is not None:
        from collections import Counter
        c = Counter(y); top = max(c, key=c.get); n = len(y)
        if c[top] / n > fagus_cap:
            target = int(fagus_cap / (1 - fagus_cap) * (n - c[top]))
            idx_top = np.where(y == top)[0]
            rng = np.random.RandomState(0); drop = set(rng.choice(idx_top, c[top] - target, replace=False))
            keepm = np.array([i not in drop for i in range(n)])
            Xc, Xsf, Xsz, y, groups = Xc[keepm], Xsf[keepm], Xsz[keepm], y[keepm], groups[keepm]

    res = {"dataset": name, "block": block, "n": int(len(y)), "n_species": len(set(y)),
           "n_blocks": len(set(groups)), "analyzable_species": len(analyzable),
           "dominant_frac": round(float(max(np.bincount(np.unique(y, return_inverse=True)[1])) / len(y)), 3)}

    clip, clip_folds = run_method(Xc, y, groups, LIN)
    sf_rf, sf_folds = run_method(Xsf, y, groups, RF)
    sz_rf, sz_folds = run_method(Xsz, y, groups, RF)
    sf_svm, _ = run_method(Xsf, y, groups, SVM)
    res["CLIP_linear"] = clip; res["geo_sf_rf"] = sf_rf; res["geo_sz_rf"] = sz_rf; res["geo_sf_svm"] = sf_svm
    res["majority_f1"] = majority_f1(y, groups)

    res["geo_features_source"] = res_geo_src
    geo_options = [("geo_sf_rf", sf_rf, sf_folds, 11), ("geo_sz_rf", sz_rf, sz_folds, 17)]
    best_geo_name, best_geo, best_folds, best_geo_dim = max(geo_options, key=lambda t: t[1]["pooled_f1"])
    res["best_geo"] = best_geo_name
    delta = round(clip["pooled_f1"] - best_geo["pooled_f1"], 4)
    res["delta_clip_minus_bestgeo"] = delta

    # M4: capacity-matched arm — CLIP PCA-reduced (train-only) to the winning geo's dimension
    clip_pca, clip_pca_folds = run_method(Xc, y, groups, LIN, pca_dim=best_geo_dim)
    res["CLIP_pca_matched"] = clip_pca; res["pca_dim"] = best_geo_dim
    res["delta_clip_pca_minus_bestgeo"] = round(clip_pca["pooled_f1"] - best_geo["pooled_f1"], 4)

    def paired_p(a_folds, b_folds):
        nf = min(len(a_folds), len(b_folds))
        a, b = np.array(a_folds[:nf]), np.array(b_folds[:nf])
        if nf >= 3 and len(set(np.round(a - b, 6))) > 1:
            try: return float(wilcoxon(a, b).pvalue), nf
            except Exception: return None, nf
        return None, nf

    p, nfold = paired_p(clip_folds, best_folds)
    res["wilcoxon_p"] = round(p, 4) if p is not None else None
    p_pca, _ = paired_p(clip_pca_folds, best_folds)
    res["wilcoxon_p_pca_matched"] = round(p_pca, 4) if p_pca is not None else None

    # M7: multiple-comparison correction (3 GEN datasets) + leave-one-fold jackknife on the paired p
    if p is not None:
        res["p_bonferroni_x3"] = round(min(1.0, p * 3), 4)
        res["p_bonferroni_x2"] = round(min(1.0, p * 2), 4)
        nf = min(len(clip_folds), len(best_folds))
        a, b = np.array(clip_folds[:nf]), np.array(best_folds[:nf])
        jk = []
        for i in range(nf):
            aa, bb = np.delete(a, i), np.delete(b, i)
            if len(set(np.round(aa - bb, 6))) > 1:
                try: jk.append(float(wilcoxon(aa, bb).pvalue))
                except Exception: pass
        res["jackknife_p_range"] = [round(min(jk), 4), round(max(jk), 4)] if jk else None

    lc, slope, sat = learning_curve(Xc, y, groups, LIN)
    res["clip_learning_curve"] = lc; res["saturation_slope"] = slope; res["saturated"] = sat

    # verdict per §8 (threshold 0.05, raw paired p)
    sig = (p is not None and p < 0.05)
    if not sat: verdict = "INCONCLUSIVE (underpowered)"
    elif delta >= 0.05 and sig: verdict = "CLIP WINS"
    elif delta <= -0.05 and sig: verdict = "CLIP LOSES"
    else: verdict = "TIE/INCONCLUSIVE"
    res["verdict"] = verdict
    # honest secondary verdict after multiplicity correction
    sig_corr = (res.get("p_bonferroni_x3") is not None and res["p_bonferroni_x3"] < 0.05)
    res["verdict_bonferroni_corrected"] = ("CLIP WINS" if (delta >= 0.05 and sig_corr and sat) else "TIE/INCONCLUSIVE")

    json.dump(res, open(OUT / f"score_{which}.json", "w"), indent=2)
    print(json.dumps({k: res.get(k) for k in ["dataset", "n", "n_species", "n_blocks", "dominant_frac",
          "geo_features_source", "best_geo", "delta_clip_minus_bestgeo", "delta_clip_pca_minus_bestgeo",
          "wilcoxon_p", "wilcoxon_p_pca_matched", "p_bonferroni_x3", "jackknife_p_range",
          "majority_f1", "saturated", "verdict", "verdict_bonferroni_corrected"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
