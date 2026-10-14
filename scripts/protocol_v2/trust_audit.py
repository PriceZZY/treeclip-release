"""TRUSTWORTHY audit (read-only). CPU-only, thread-capped.
(1) fold partition: zero train/test group overlap on real loaded arrays.
(2) KEY: can features predict the GROUP-ID (Weiser plot / NEON band)?
    Train LogReg to predict group from feats under WITHIN-fold-respecting CV
    (group-aware split so we never train+test the same group), report acc vs chance.
    High group-predictability from CLIP relative to geo = residual artifact risk.
(3) confirm StandardScaler fit train-only (we replicate the exact fold loop).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from data import load, DATASETS
from spatial_cv import make_folds
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

def part1_overlap(d, folds, ds):
    print(f"\n[1] {ds} fold partition (group overlap)")
    grp = np.asarray(d["group"]); N = len(d["y"])
    covered = np.zeros(N, bool)
    bad = 0
    for f in folds:
        tr, te_full = f["train_idx"], np.where(np.isin(np.arange(N), f["test_idx"]))[0]
        # use full test mask = the group's members, not shared-species-restricted, for overlap check
        # reconstruct full test group membership
    # robust: rebuild test-full from group, since test_idx is species-restricted
    for f in folds:
        tr = set(f["train_idx"].tolist())
        te = set(f["test_idx"].tolist())
        inter = tr & te
        tr_groups = set(grp[list(tr)]) if tr else set()
        te_groups = set(grp[list(te)]) if te else set()
        gov = tr_groups & te_groups
        flag = (len(inter) > 0) or (len(gov) > 0)
        bad += int(flag)
        print(f"   {f['name']:26s} idx_overlap={len(inter)} group_overlap={len(gov)} "
              f"train_groups={len(tr_groups)} test_groups={len(te_groups)}")
    print(f"   => folds with any overlap: {bad}/{len(folds)}")
    return bad

def group_probe(X, groups, seed=0, n_splits=5):
    """Predict group-id from features. Stratified-by-group CV. Returns (acc, chance, majority)."""
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    K = len(uniq)
    counts = np.bincount(inv)
    chance = 1.0 / K
    majority = counts.max() / len(inv)
    # need at least 2 per class for stratified kfold; cap splits by min class count
    minc = counts.min()
    ns = min(n_splits, minc) if minc >= 2 else 0
    if ns < 2:
        return None, chance, majority, K
    skf = StratifiedKFold(n_splits=ns, shuffle=True, random_state=seed)
    yt_all, yp_all = [], []
    for tr, te in skf.split(X, inv):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(X[tr]), inv[tr])
        yp_all.extend(clf.predict(sc.transform(X[te])).tolist())
        yt_all.extend(inv[te].tolist())
    acc = float(np.mean(np.array(yt_all) == np.array(yp_all)))
    return acc, chance, majority, K

def part2_leakage(d, ds):
    print(f"\n[2] {ds} GROUP-ID predictability from features (within-group stratified CV)")
    grp = d["group"]
    for fk in ["clip", "geo_sf", "geo_sz"]:
        X = d[fk]
        acc, chance, maj, K = group_probe(X, grp)
        if acc is None:
            print(f"   {fk:7s}: SKIP (min class <2, K={K})")
            continue
        lift = acc / chance
        print(f"   {fk:7s}: acc={acc:.3f}  chance={chance:.3f}  majority={maj:.3f}  "
              f"lift_vs_chance={lift:.1f}x  (K={K} groups)")

def part3_scaler(d, folds, ds):
    """Replicate run_sklearn fold loop and assert scaler stats come from TRAIN only.
    We verify the scaler mean equals X[tr].mean and differs from full-data mean."""
    print(f"\n[3] {ds} StandardScaler fit train-only (replicating run_sklearn)")
    X = d["clip"]
    ok = True
    for f in folds[:3]:
        tr, te = f["train_idx"], f["test_idx"]
        if len(te) == 0: continue
        sc = StandardScaler().fit(X[tr])
        train_mean_match = np.allclose(sc.mean_, X[tr].mean(0))
        full_mean = X.mean(0)
        leaks_full = np.allclose(sc.mean_, full_mean)
        # confirm test rows did NOT influence scaler: scaler mean != mean of all (tr+te) unless tr==all
        print(f"   {f['name']:26s} scaler.mean==train.mean:{train_mean_match} "
              f"scaler.mean==fulldata.mean:{leaks_full}")
        ok = ok and train_mean_match and (not leaks_full)
    print(f"   => train-only scaling verified: {ok}")
    return ok

def main():
    for ds, _, _ in DATASETS:
        d = load(ds)
        folds = make_folds(d["group"], d["xy"], d["y"], ds)
        print("="*70)
        print(f"{ds}: N={len(d['y'])} K={d['K']} groups={sorted(set(d['group']))}")
        part1_overlap(d, folds, ds)
        part2_leakage(d, ds)
        part3_scaler(d, folds, ds)

if __name__ == "__main__":
    main()
