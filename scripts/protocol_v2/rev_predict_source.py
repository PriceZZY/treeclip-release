"""HOSTILE REVIEWER CHECK: does CLIP encode non-structural SOURCE/SITE artifacts?

Test: on the SAME analyzable subset used by gen_score.py, predict the
source-dataset (FOR) / region & plot (BioDiv) FROM the features.
If CLIP predicts source far better than geometry, the depth-render encodes
acquisition/sensor/site cues, and the cross-dataset 'generality' is confounded.

Two evaluation modes:
  (1) WITHIN-CLASS controlled: stratify by species so the source-predictor
      cannot cheat via species->source correlation. We hold species fixed and
      ask: within a species, can CLIP still tell which source/plot a tree came from?
  (2) plain stratified 5-fold predicting the source label.

Also: nearest-neighbor source purity and a permutation-style chance baseline.

CPU-only, recomputes from cached features. No training of nets.
Usage: python rev_predict_source.py for | biodiv
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"; OUT.mkdir(parents=True, exist_ok=True)


def analyzable_mask(y, groups, min_n=30, min_blocks=2):
    keep = []
    for s in set(y):
        if (y == s).sum() >= min_n and len(set(groups[y == s])) >= min_blocks:
            keep.append(s)
    return np.isin(y, keep), sorted(keep)


def predict_label(X, target, seed=0, max_classes=None):
    """Stratified 5-fold: predict `target` from X. Return macro-F1, balanced acc, OA, chance(1/K)."""
    lab, inv = np.unique(target, return_inverse=True)
    K = len(lab)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    yt_all, yp_all = [], []
    for tr, te in skf.split(X, inv):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=1)
        clf.fit(sc.transform(X[tr]), inv[tr])
        yp_all.extend(clf.predict(sc.transform(X[te])).tolist())
        yt_all.extend(inv[te].tolist())
    yt = np.array(yt_all); yp = np.array(yp_all)
    return {
        "K": int(K),
        "macro_f1": round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
        "balanced_acc": round(float(balanced_accuracy_score(yt, yp)), 4),
        "oa": round(float(accuracy_score(yt, yp)), 4),
        "chance_oa": round(1.0 / K, 4),
        "chance_balanced": round(1.0 / K, 4),
    }


def within_species_source(X, y, src, seed=0, min_per_cell=10):
    """For each species present in >=2 sources with enough samples, predict the
    SOURCE label within that species only (species held constant => purely
    non-structural-vs-structural source signal). Pool predictions, report
    balanced accuracy vs within-species chance.
    Returns per-species and pooled."""
    rng = np.random.RandomState(seed)
    per = {}
    pooled_yt, pooled_yp, pooled_chance = [], [], []
    for s in sorted(set(y)):
        m = y == s
        srcs = src[m]
        labs, cnts = np.unique(srcs, return_counts=True)
        ok = labs[cnts >= min_per_cell]
        if len(ok) < 2:
            continue
        cellmask = m & np.isin(src, ok)
        Xs = X[cellmask]; ss = src[cellmask]
        lab, inv = np.unique(ss, return_inverse=True)
        K = len(lab)
        if min(np.bincount(inv)) < 5:
            continue
        skf = StratifiedKFold(n_splits=min(5, int(min(np.bincount(inv)))), shuffle=True, random_state=seed)
        yt_all, yp_all = [], []
        for tr, te in skf.split(Xs, inv):
            sc = StandardScaler().fit(Xs[tr])
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=1)
            clf.fit(sc.transform(Xs[tr]), inv[tr])
            yp_all.extend(clf.predict(sc.transform(Xs[te])).tolist())
            yt_all.extend(inv[te].tolist())
        ba = balanced_accuracy_score(yt_all, yp_all)
        per[s] = {"n": int(cellmask.sum()), "n_sources": int(K),
                  "balanced_acc": round(float(ba), 4), "chance": round(1.0 / K, 4)}
        pooled_yt.extend([f"{s}|{t}" for t in yt_all])
        pooled_yp.extend([f"{s}|{p}" for p in yp_all])
        pooled_chance.append(1.0 / K)
    pooled_ba = None
    if per:
        # mean of per-species balanced acc (each vs its own chance)
        bas = [v["balanced_acc"] for v in per.values()]
        chs = [v["chance"] for v in per.values()]
        pooled_ba = {"mean_balanced_acc": round(float(np.mean(bas)), 4),
                     "mean_chance": round(float(np.mean(chs)), 4),
                     "n_species_tested": len(per),
                     "lift_over_chance": round(float(np.mean(bas) - np.mean(chs)), 4)}
    return per, pooled_ba


def main():
    which = sys.argv[1]
    if which == "for":
        d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
        Xc = d["X"]; y = d["species"].astype(str); groups = d["dataset"].astype(str)
        dtype = d["data_type"].astype(str)
        g = np.load(ROOT / "data/raw/for_species20k/geo_features.npz", allow_pickle=True)
        Xsf, Xsz = g["X_sf"], g["X_sz"]
        name = "FOR-species20K"
        mask, analyzable = analyzable_mask(y, groups)
        Xc, Xsf, Xsz, y, groups, dtype = Xc[mask], Xsf[mask], Xsz[mask], y[mask], groups[mask], dtype[mask]
        src = groups  # source-dataset is the spatial CV block here
        extra_targets = {"source_dataset": groups, "sensor_data_type": dtype}
    else:
        d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
        Xc = d["X_clip"]; Xsf = d["X_sf"]; Xsz = d["X_sz"]
        y = d["species"].astype(str); groups = d["region"].astype(str)
        ep = d["ep"].astype(str)
        name = "BioDiv-3DTrees-ULS"
        mask, analyzable = analyzable_mask(y, groups)
        Xc, Xsf, Xsz, y, groups, ep = Xc[mask], Xsf[mask], Xsz[mask], y[mask], groups[mask], ep[mask]
        # match scoring: Fagus downsample to <39% on the analyzable subset
        from collections import Counter
        c = Counter(y); top = max(c, key=c.get); n = len(y); cap = 0.39
        if c[top] / n > cap:
            target = int(cap / (1 - cap) * (n - c[top]))
            idx_top = np.where(y == top)[0]
            rng = np.random.RandomState(0); drop = set(rng.choice(idx_top, c[top] - target, replace=False))
            keepm = np.array([i not in drop for i in range(n)])
            Xc, Xsf, Xsz, y, groups, ep = Xc[keepm], Xsf[keepm], Xsz[keepm], y[keepm], groups[keepm], ep[keepm]
        src = ep  # finest site label = plot
        extra_targets = {"region": groups, "plot_ep": ep}

    res = {"dataset": name, "n": int(len(y)), "n_species": len(set(y)),
           "analyzable_species": analyzable}

    # ---- (A) predict each source/site target from CLIP vs geo ----
    res["predict_source"] = {}
    for tname, tgt in extra_targets.items():
        res["predict_source"][tname] = {
            "n_classes": int(len(set(tgt))),
            "CLIP_512d": predict_label(Xc, tgt),
            "geo_sf_11d": predict_label(Xsf, tgt),
            "geo_sz_17d": predict_label(Xsz, tgt),
        }

    # ---- (B) WITHIN-SPECIES source prediction (the clean confound test) ----
    per_clip, pooled_clip = within_species_source(Xc, y, src)
    per_geosz, pooled_geosz = within_species_source(Xsz, y, src)
    res["within_species_source"] = {
        "CLIP_512d": {"pooled": pooled_clip, "per_species": per_clip},
        "geo_sz_17d": {"pooled": pooled_geosz, "per_species": per_geosz},
        "interpretation": ("If CLIP within-species balanced-acc >> chance AND >> geo, "
                           "the render encodes non-structural source/site cues."),
    }

    json.dump(res, open(OUT / f"rev_predict_source_{which}.json", "w"), indent=2, default=str)

    # compact print
    print(f"=== {name}  N={len(y)}  species={len(set(y))} ===")
    for tname, r in res["predict_source"].items():
        c = r["CLIP_512d"]; gz = r["geo_sz_17d"]
        print(f"[predict {tname}] K={r['n_classes']} chance_oa={c['chance_oa']}")
        print(f"   CLIP   bal_acc={c['balanced_acc']} oa={c['oa']} macroF1={c['macro_f1']}")
        print(f"   geo_sz bal_acc={gz['balanced_acc']} oa={gz['oa']} macroF1={gz['macro_f1']}")
    print("[within-species source prediction]")
    print("   CLIP  ", pooled_clip)
    print("   geo_sz", pooled_geosz)


if __name__ == "__main__":
    main()
