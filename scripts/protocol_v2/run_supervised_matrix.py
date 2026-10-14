"""Supervised feature x classifier matrix under the SAME leakage-free spatial CV.

Features: CLIP (frozen topview), geo_sf (scale-free), geo_sz (with size).
Classifiers: linear (LogReg), RF, SVM (rbf). All class_weight=balanced.
Same spatial folds + shared-species + pooled metrics as run_protocol.py.
CPU-only; safe to run alongside the GPU PointNet++ job. Output: runs/protocol_v2/supervised_matrix.json
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from data import load, DATASETS
from spatial_cv import make_folds
from run_protocol import run_sklearn   # reuse exact same spatial-CV + pooled-metric logic

OUT = Path(__file__).resolve().parent.parent.parent / "runs/protocol_v2"

CLFS = {
    "linear": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    "rf":     lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0),
    "svm":    lambda: SVC(kernel="rbf", class_weight="balanced"),
}
FEATS = ["clip", "geo_sf", "geo_sz"]


def main():
    results = {}
    for ds, _, _ in DATASETS:
        d = load(ds)
        folds = make_folds(d["group"], d["xy"], d["y"], ds)
        print(f"\n===== {ds} (spatial CV, {len(folds)} folds) =====")
        print(f"{'feat x clf':18s} {'macroF1':>8s} {'OA':>7s}  CI95")
        dsr = {}
        for fk in FEATS:
            for ck, cf in CLFS.items():
                r = run_sklearn(d, folds, fk, cf)
                dsr[f"{fk}+{ck}"] = r
                print(f"{fk+'+'+ck:18s} {r['f1_pooled']:8.3f} {r['oa_pooled']:7.3f}  {r['oa_ci95']}")
        results[ds] = dsr
        json.dump(results, open(OUT / "supervised_matrix.json", "w"), indent=2)
    print(f"\nwrote {OUT/'supervised_matrix.json'}")


if __name__ == "__main__":
    main()
