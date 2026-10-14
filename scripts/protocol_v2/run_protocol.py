"""Protocol v2 unified experiment — leakage-free spatial CV, apples-to-apples.

Methods: CLIP+linear, geo+RF (scale-free AND size variants = size ablation), PointNet++.
Splits: spatial CV (NEON spatial bands / Weiser LOPO) with shared-species scoring.
Preprocessing: all StandardScaler/PCA fit on TRAIN ONLY, inside each fold.
Metrics: macro-F1 + OA per fold; mean +/- std across folds (x seeds for PN++);
         pooled-OA bootstrap 95% CI. Per-fold detail saved.

Scale policy: CLIP is scale-normalized by construction (report under primary).
geo reports BOTH scale-free (geo_sf) and size (geo_sz). PN++ runs unit-sphere
(normalized regime) only this pass (size regime deferred — too costly).
Point budget: canonical N=1024 (yanx27 standard); PN++ SSG sa1 needs >=512 and OSBS
median is 72, so sparse clouds are upsampled-with-replacement (fraction reported per spec C2).
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "third_party/pointnet2_pytorch/models"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import importlib

from data import load, DATASETS
from spatial_cv import make_folds

pn2 = importlib.import_module("pointnet2_cls_ssg")
OUT = ROOT / "runs/protocol_v2"; OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "protocol_v2.log"
SEEDS = [42, 43, 44]


def log(m):
    line = f"[{__import__('datetime').datetime.now().strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def scores(yt, yp):
    return (round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
            round(float(accuracy_score(yt, yp)), 4))


def boot_oa_ci(correct, n=2000, seed=0):
    """bootstrap 95% CI of overall accuracy from a 0/1 correctness array."""
    if len(correct) == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed); correct = np.asarray(correct)
    bs = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


# ---------------- pooled metrics (M2 fix) ----------------
def pooled_metrics(yt_all, yp_all, per_fold):
    """Primary = POOLED across folds (concatenate y_true/y_pred -> one macro-F1/OA over
    the union of scored species). Per-fold mean kept for transparency only."""
    yt = np.asarray(yt_all); yp = np.asarray(yp_all)
    f1, oa = scores(yt, yp)
    f1s = [p["f1"] for p in per_fold]
    return {
        "f1_pooled": f1, "oa_pooled": oa,
        "oa_ci95": boot_oa_ci((yt == yp).astype(int)),
        "f1_fold_mean": round(float(np.mean(f1s)), 4),
        "f1_fold_std": round(float(np.std(f1s)), 4),
        "n_test_total": int(len(yt)), "n_folds": len(per_fold), "per_fold": per_fold,
    }


# ---------------- sklearn methods (per fold) ----------------
def run_sklearn(d, folds, feat_key, clf_factory):
    per_fold = []; yt_all = []; yp_all = []
    for f in folds:
        tr, te = f["train_idx"], f["test_idx"]
        if len(te) == 0:
            continue
        X = d[feat_key]
        sc = StandardScaler().fit(X[tr])
        clf = clf_factory().fit(sc.transform(X[tr]), d["y"][tr])
        pred = clf.predict(sc.transform(X[te]))
        f1, oa = scores(d["y"][te], pred)
        per_fold.append({"fold": f["name"], "f1": f1, "oa": oa, "n_test": len(te)})
        yt_all.extend(d["y"][te].tolist()); yp_all.extend(pred.tolist())
    return pooled_metrics(yt_all, yp_all, per_fold)


# ---------------- PointNet++ ----------------
def normalize(p, unit_sphere=True):
    p = p - p.mean(0)
    if unit_sphere:
        p = p / (np.max(np.linalg.norm(p, axis=1)) + 1e-9)
    return p.astype(np.float32)


def sample_to(p, n, rng):
    if p.shape[0] >= n:
        idx = rng.choice(p.shape[0], n, replace=False)
    else:
        idx = rng.choice(p.shape[0], n, replace=True)  # minimal upsample (small clouds)
    return p[idx]


def augment(p, rng):
    th = rng.uniform(0, 2 * np.pi); c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
    return (p @ R.T) * rng.uniform(0.8, 1.25) + rng.normal(0, 0.01, p.shape).astype(np.float32)


def prep(points, idxs, n, rng, train, unit_sphere):
    out = []
    for i in idxs:
        pp = sample_to(points[i][:, :3].astype(np.float32), n, rng)
        if train:
            pp = augment(pp, rng)
        out.append(normalize(pp, unit_sphere))
    return np.transpose(np.stack(out), (0, 2, 1))


def stratified_val(y_tr_labels, frac=0.15, seed=0):
    """Carve a stratified val set from TRAIN only. Each class with >=2 samples contributes
    >=1 to val AND keeps >=1 in train (m1 fix); singletons stay in train."""
    rng = np.random.RandomState(seed); tr, va = [], []
    for c in set(y_tr_labels):
        idx = np.where(y_tr_labels == c)[0]; rng.shuffle(idx); n = len(idx)
        nv = 0 if n < 2 else min(n - 1, max(1, int(round(n * frac))))
        va.extend(idx[:nv]); tr.extend(idx[nv:])
    return np.array(tr), np.array(va)


def train_pn2_fold(d, f, npts, seed, unit_sphere=True, max_epochs=150, patience=30):
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    tr_all, te = f["train_idx"], f["test_idx"]
    if len(te) == 0:
        return None
    # label space = species present in TRAIN (classifier outputs over these)
    cls = sorted(set(d["y"][tr_all])); c2i = {c: i for i, c in enumerate(cls)}; K = len(cls)
    ylab = d["y"]
    # internal val carve (never from test)
    sub_tr, sub_va = stratified_val(ylab[tr_all], seed=seed)
    tr = tr_all[sub_tr]; va = tr_all[sub_va] if len(sub_va) else tr_all[:0]

    model = pn2.get_model(K, normal_channel=False).to("cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    yi = np.array([c2i[c] for c in ylab[tr]])
    cnt = np.bincount(yi, minlength=K).astype(np.float32)
    w = torch.tensor(cnt.sum() / np.maximum(cnt, 1), device="cuda", dtype=torch.float32)
    nll = nn.NLLLoss(weight=w)

    Xte = torch.tensor(prep(d["points"], te, npts, rng, False, unit_sphere), device="cuda")
    yte = d["y"][te]
    has_va = len(va) > 0
    if has_va:
        Xva = torch.tensor(prep(d["points"], va, npts, rng, False, unit_sphere), device="cuda")
        yva = np.array([c2i.get(c, -1) for c in ylab[va]])

    def predict(X):
        out = []
        for i in range(0, len(X), 32):
            pr, _ = model(X[i:i+32]); out.append(pr.argmax(1).cpu().numpy())
        return np.concatenate(out)

    best_va, best_pred = -1.0, None; wait = 0; tr_list = list(tr)
    for ep in range(max_epochs):
        model.train(); rng.shuffle(tr_list)
        Xtr = torch.tensor(prep(d["points"], tr_list, npts, rng, True, unit_sphere), device="cuda")
        ytr = torch.tensor([c2i[c] for c in ylab[tr_list]], device="cuda", dtype=torch.long)
        for i in range(0, len(Xtr), 16):
            xb, yb = Xtr[i:i+16], ytr[i:i+16]
            if len(xb) < 2: continue
            opt.zero_grad(); pr, _ = model(xb); loss = nll(pr, yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step(); model.eval()
        with torch.no_grad():
            score_va = f1_score(yva, predict(Xva), average="macro", zero_division=0) if has_va else ep
            if score_va > best_va:
                best_va = score_va; wait = 0
                idx2c = np.array(cls)
                best_pred = idx2c[predict(Xte)]
            else:
                wait += 1
                if wait >= patience: break
    f1, oa = scores(yte, best_pred)
    return {"fold": f["name"], "f1": f1, "oa": oa, "n_test": len(te)}, list(yte), list(best_pred)


def run_pn2(d, folds, npts):
    per_fold = []
    seed_yt = {sd: [] for sd in SEEDS}; seed_yp = {sd: [] for sd in SEEDS}
    for f in folds:
        f1s, oas = [], []
        for sd in SEEDS:
            r = train_pn2_fold(d, f, npts, sd)
            if r is None: continue
            rec, yte, yp = r
            f1s.append(rec["f1"]); oas.append(rec["oa"])
            seed_yt[sd].extend(yte); seed_yp[sd].extend(yp)
        if not f1s: continue
        per_fold.append({"fold": f["name"], "f1": round(float(np.mean(f1s)), 4),
                         "f1_seedstd": round(float(np.std(f1s)), 4),
                         "oa": round(float(np.mean(oas)), 4), "n_test": f["info"]["n_test_scored"]})
        log(f"    PN++ {f['name']}: F1={np.mean(f1s):.3f}+-{np.std(f1s):.3f} OA={np.mean(oas):.3f}")
    # pooled per seed -> mean +- seed-std; OA CI bootstraps over ALL seeds+folds (M3 fix)
    pooled_f1, pooled_oa, all_correct = [], [], []
    for sd in SEEDS:
        if not seed_yt[sd]: continue
        yt = np.asarray(seed_yt[sd]); yp = np.asarray(seed_yp[sd])
        f1, oa = scores(yt, yp); pooled_f1.append(f1); pooled_oa.append(oa)
        all_correct.extend((yt == yp).astype(int).tolist())
    return {
        "f1_pooled": round(float(np.mean(pooled_f1)), 4),
        "f1_pooled_seedstd": round(float(np.std(pooled_f1)), 4),
        "oa_pooled": round(float(np.mean(pooled_oa)), 4),
        "oa_ci95": boot_oa_ci(all_correct),
        "n_folds": len(per_fold), "per_fold": per_fold,
    }


def main():
    log("########## PROTOCOL v2 ##########")
    results = {}
    for ds, _, _ in DATASETS:
        log(f"===== {ds} =====")
        d = load(ds)
        folds = make_folds(d["group"], d["xy"], d["y"], ds)
        counts = np.array([p.shape[0] for p in d["points"]])
        npts = 1024  # canonical PointNet++ input (yanx27 standard); see spec C2 note
        ups = float(np.mean(counts < npts))
        log(f"  N={len(d['y'])} K={d['K']} folds={len(folds)} npts(PN++)={npts} "
            f"frac_upsampled={ups:.0%} median_pts={int(np.median(counts))}")
        dsr = {"meta": {"n": len(d["y"]), "K": d["K"], "n_folds": len(folds),
                        "npts_pn2": npts, "frac_upsampled": round(ups, 3),
                        "folds": [f["info"] | {"name": f["name"]} for f in folds]}}

        log("  -- CLIP+linear --")
        dsr["clip_linear"] = run_sklearn(d, folds, "clip",
                                         lambda: LogisticRegression(max_iter=1000, class_weight="balanced"))
        log(f"     F1(pooled)={dsr['clip_linear']['f1_pooled']} "
            f"OA={dsr['clip_linear']['oa_pooled']} CI={dsr['clip_linear']['oa_ci95']}")

        for fk, lbl in [("geo_sf", "geo_scalefree+RF"), ("geo_sz", "geo_size+RF")]:
            dsr[fk] = run_sklearn(d, folds, fk,
                                  lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0))
            log(f"  -- {lbl} -- F1(pooled)={dsr[fk]['f1_pooled']} OA={dsr[fk]['oa_pooled']}")

        log("  -- PointNet++ (normalized) --")
        dsr["pointnet2"] = run_pn2(d, folds, npts)
        log(f"     F1(pooled)={dsr['pointnet2']['f1_pooled']}+-{dsr['pointnet2']['f1_pooled_seedstd']} "
            f"OA={dsr['pointnet2']['oa_pooled']} CI={dsr['pointnet2']['oa_ci95']}")

        results[ds] = dsr
        json.dump(results, open(OUT / "results.json", "w"), indent=2)
    log("########## DONE ##########")


if __name__ == "__main__":
    main()
