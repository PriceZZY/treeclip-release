import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load
from spatial_cv import make_folds

d = load("NEON-MLBS")
y = d["y"]
folds = make_folds(d["group"], d["xy"], y, "NEON-MLBS")

# ---- Baselines per fold, then pooled, computed on SHARED-SPECIES test sets ----
# (mirrors run_protocol shared-species scoring)
from sklearn.metrics import f1_score, accuracy_score

pooled_pred_majority = []
pooled_true = []
pooled_pred_random = []
rng = np.random.RandomState(0)

print("MLBS folds: K_overall =", d["K"])
for f in folds:
    tr = f["train_idx"]; te = f["test_idx"]
    ytr = y[tr]; yte = y[te]
    # majority class from TRAIN
    vals, cnts = np.unique(ytr, return_counts=True)
    maj = vals[np.argmax(cnts)]
    pred_maj = np.full(len(yte), maj)
    # stratified-random: sample from train class distribution
    classes = vals
    probs = cnts / cnts.sum()
    pred_rand = rng.choice(classes, size=len(yte), p=probs)
    pooled_pred_majority.append(pred_maj)
    pooled_pred_random.append(pred_rand)
    pooled_true.append(yte)
    print(f"  {f['name']:28s} n_test={len(yte):3d} K_scored={len(set(yte))} maj_share_in_test={np.mean(yte==maj):.3f}")

yt = np.concatenate(pooled_true)
ymaj = np.concatenate(pooled_pred_majority)
yrnd = np.concatenate(pooled_pred_random)

f1_maj = f1_score(yt, ymaj, average="macro", zero_division=0)
oa_maj = accuracy_score(yt, ymaj)
f1_rnd = f1_score(yt, yrnd, average="macro", zero_division=0)
oa_rnd = accuracy_score(yt, yrnd)
n_classes_pooled = len(set(yt))

print(f"\nPOOLED MLBS baselines (shared-species test, n={len(yt)}, K_pooled={n_classes_pooled}):")
print(f"  majority-class:    macroF1={f1_maj:.4f}  OA={oa_maj:.4f}")
print(f"  stratified-random: macroF1={f1_rnd:.4f}  OA={oa_rnd:.4f}")
print(f"  1/K chance macroF1 ref (uniform) ~ {1.0/n_classes_pooled:.4f}")

# ---- Per-fold CLIP(linear) vs geo_sf(rf) from the JSON, recompute Wilcoxon + wins ----
J = json.load(open(Path(__file__).resolve().parent.parent.parent / "runs/protocol_v2/supervised_matrix.json"))
mlbs = J["NEON-MLBS"]
clip = [pf["f1"] for pf in mlbs["clip+linear"]["per_fold"]]
geo  = [pf["f1"] for pf in mlbs["geo_sf+rf"]["per_fold"]]
clip = np.array(clip); geo = np.array(geo)
diff = clip - geo
wins = int((clip > geo).sum())
print(f"\nMLBS per-fold clip+linear vs geo_sf+rf:")
for i,(c,g) in enumerate(zip(clip,geo)):
    print(f"  fold{i}: clip={c:.4f} geo={g:.4f} diff={c-g:+.4f}")
print(f"  clip wins {wins}/5; mean diff={diff.mean():+.4f}")
from scipy.stats import wilcoxon
try:
    stat, p = wilcoxon(clip, geo)
    print(f"  Wilcoxon signed-rank p={p:.4f}")
except Exception as e:
    print("  wilcoxon err", e)

# range check
allf1 = []
for feat in mlbs:
    allf1.append(mlbs[feat]["f1_pooled"])
print(f"\nMLBS all f1_pooled range: {min(allf1):.4f} - {max(allf1):.4f}")
print("values:", {k: mlbs[k]["f1_pooled"] for k in mlbs})
