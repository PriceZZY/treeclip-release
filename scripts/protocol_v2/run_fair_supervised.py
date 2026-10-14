"""FAIR-ER supervised comparison for Lu: CLIP vs RF vs SVM, spatial CV.

Addresses the audit's fairness defect (SVM was default C=1.0 vs RF's 300 trees) by giving
SVM/linear a STRONGER FIXED setting (svm C=10 gamma=scale; linear C=1), which the audit
showed removes the C=1 handicap (geo_sf+svm 0.08 -> 0.10, ties RF).

NOTE (audit finding F1): the GridSearchCV inner-CV tuning path below is effectively DEAD —
every spatial fold's train set contains a single-tree species (min_cls==1), so tuned()
always returns the fixed fallback classifier. So results = FIXED hyperparameters
(svm C=10, gamma=scale; linear C=1), NOT per-fold tuned. Do NOT describe these as "tuned".
To make tuning real, replace the inner StratifiedKFold with KFold (tolerates singletons).

Same spatial folds + shared-species + pooled & fold-mean metrics as the rest of protocol_v2.
CPU-only. Output: runs/protocol_v2/fair_supervised.json
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

from data import load, DATASETS
from spatial_cv import make_folds

OUT = Path(__file__).resolve().parent.parent.parent / "runs/protocol_v2"


def tuned(name, ytr):
    """Return a classifier; SVM/linear get inner-CV tuning when feasible."""
    min_cls = np.min(np.bincount(np.unique(ytr, return_inverse=True)[1]))
    cv = StratifiedKFold(n_splits=min(3, max(2, min_cls)), shuffle=True, random_state=0)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0)
    if name == "linear":
        base = LogisticRegression(max_iter=1000, class_weight="balanced")
        grid = {"C": [0.1, 1, 10]}
        fb = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    else:  # svm
        base = SVC(class_weight="balanced")
        grid = {"C": [1, 10, 100], "gamma": ["scale", 0.01, 0.001]}
        fb = SVC(class_weight="balanced", C=10, gamma="scale")
    if min_cls < 2:
        return fb
    try:
        return GridSearchCV(base, grid, scoring="f1_macro", cv=cv, n_jobs=1, error_score=0)
    except Exception:
        return fb


def scores(yt, yp):
    return (round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
            round(float(accuracy_score(yt, yp)), 4))


def run_cell(d, folds, feat_key, clf_name):
    yt_all, yp_all, fold_f1 = [], [], []
    for f in folds:
        tr, te = f["train_idx"], f["test_idx"]
        if len(te) == 0:
            continue
        X = d[feat_key]
        sc = StandardScaler().fit(X[tr])
        clf = tuned(clf_name, d["y"][tr])
        clf.fit(sc.transform(X[tr]), d["y"][tr])
        pred = clf.predict(sc.transform(X[te]))
        f1, _ = scores(d["y"][te], pred)
        fold_f1.append(f1)
        yt_all.extend(d["y"][te].tolist()); yp_all.extend(pred.tolist())
    pf1, poa = scores(np.array(yt_all), np.array(yp_all))
    return {"f1_pooled": pf1, "oa_pooled": poa,
            "f1_foldmean": round(float(np.mean(fold_f1)), 4),
            "f1_foldstd": round(float(np.std(fold_f1)), 4)}


def main():
    results = {}
    for ds, _, _ in DATASETS:
        d = load(ds)
        folds = make_folds(d["group"], d["xy"], d["y"], ds)
        print(f"\n===== {ds} (spatial CV, {len(folds)} folds) — FAIR (tuned svm/linear) =====")
        print(f"{'feat x clf':16s} {'F1pool':>7s} {'F1fold±std':>13s} {'OApool':>7s}")
        dsr = {}
        for fk in ["clip", "geo_sf", "geo_sz"]:
            for ck in ["linear", "rf", "svm"]:
                r = run_cell(d, folds, fk, ck)
                dsr[f"{fk}+{ck}"] = r
                print(f"{fk+'+'+ck:16s} {r['f1_pooled']:7.3f} {r['f1_foldmean']:6.3f}±{r['f1_foldstd']:.3f}  {r['oa_pooled']:6.3f}")
        results[ds] = dsr
        json.dump(results, open(OUT / "fair_supervised.json", "w"), indent=2)
    print(f"\nwrote {OUT/'fair_supervised.json'}")


if __name__ == "__main__":
    main()
