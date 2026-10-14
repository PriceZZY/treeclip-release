"""FOR-species20K CPU段: geometric baseline + underfitting gate, under leave-one-source-out spatial CV.

Per CONSTITUTION §8: analyzable species = >=30 trees AND present in >=2 source datasets (threshold locked 30).
Leave-one-SOURCE-out spatial CV (the 25 contributing datasets are the spatial blocks). Shared-species scoring.
Geo baseline = RF on geo_sf (scale-free, primary) + geo_sz. Learning curve over fraction of SOURCE blocks ->
saturation gate (last-step macro-F1 gain <0.02 => saturated => adequately powered; else INCONCLUSIVE).
CPU-only. Output: runs/protocol_v2/gen/for20k_cpu.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

ROOT = Path(__file__).resolve().parent.parent.parent
D = ROOT / "data/raw/for_species20k"
OUT = ROOT / "runs/protocol_v2/gen"; OUT.mkdir(parents=True, exist_ok=True)


def boot_ci(corr, n=1000, seed=0):
    if len(corr) == 0: return [0.0, 0.0]
    rng = np.random.default_rng(seed); corr = np.asarray(corr)
    bs = [corr[rng.integers(0, len(corr), len(corr))].mean() for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def leave_source_out(X, y, src, sources):
    """pooled preds over leave-one-source-out, shared-species per fold."""
    yt_all, yp_all, maj_all = [], [], []
    for s in sources:
        te = src == s; tr = ~te
        train_sp = set(y[tr])
        keep = te & np.isin(y, list(train_sp))
        if keep.sum() == 0: continue
        sc = StandardScaler().fit(X[tr])
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=0, n_jobs=1).fit(sc.transform(X[tr]), y[tr])
        pred = clf.predict(sc.transform(X[keep]))
        # majority baseline = most frequent train class
        vals, cnts = np.unique(y[tr], return_counts=True); maj = vals[cnts.argmax()]
        yt_all.extend(y[keep].tolist()); yp_all.extend(pred.tolist())
        maj_all.extend([maj] * int(keep.sum()))
    yt = np.array(yt_all); yp = np.array(yp_all); mj = np.array(maj_all)
    f1 = round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4)
    oa = round(float(accuracy_score(yt, yp)), 4)
    mf1 = round(float(f1_score(yt, mj, average="macro", zero_division=0)), 4)
    return {"macro_f1": f1, "oa": oa, "majority_f1": mf1, "n_test": int(len(yt)),
            "f1_ci": boot_ci((yt == yp).astype(int)), "maj_ci": boot_ci((yt == mj).astype(int))}


def main():
    d = np.load(D / "geo_features.npz", allow_pickle=True)
    Xsf, Xsz = d["X_sf"], d["X_sz"]; y = d["species"].astype(str); src = d["dataset"].astype(str)
    # analyzable species: >=30 trees AND >=2 source datasets
    sp_n = {s: int((y == s).sum()) for s in set(y)}
    sp_src = {s: len(set(src[y == s])) for s in set(y)}
    analyzable = [s for s in sp_n if sp_n[s] >= 30 and sp_src[s] >= 2]
    mask = np.isin(y, analyzable)
    Xsf, Xsz, y, src = Xsf[mask], Xsz[mask], y[mask], src[mask]
    sources = sorted(set(src))
    res = {"meta": {"n_total": int(mask.sum()), "n_analyzable_species": len(analyzable),
                    "n_sources": len(sources), "threshold": "n>=30 & >=2 sources"}}
    print(f"analyzable: {len(analyzable)} sp, {mask.sum()} trees, {len(sources)} sources")

    res["geo_sf_rf"] = leave_source_out(Xsf, y, src, sources)
    res["geo_sz_rf"] = leave_source_out(Xsz, y, src, sources)
    print("geo_sf+rf:", res["geo_sf_rf"]["macro_f1"], "vs majority", res["geo_sf_rf"]["majority_f1"])

    # learning curve over fraction of SOURCE blocks (spatial), geo_sf + rf
    rng = np.random.RandomState(0); lc = {}
    for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ksrc = max(2, int(round(len(sources) * frac)))
        f1s = []
        for seed in range(3):
            r2 = np.random.RandomState(seed)
            chosen = list(r2.choice(sources, ksrc, replace=False))
            sub = np.isin(src, chosen)
            r = leave_source_out(Xsf[sub], y[sub], src[sub], chosen)
            f1s.append(r["macro_f1"])
        lc[str(frac)] = {"mean": round(float(np.mean(f1s)), 4), "std": round(float(np.std(f1s)), 4)}
        print(f"  LC frac={frac}: {lc[str(frac)]['mean']}")
    res["learning_curve"] = lc
    slope = lc["1.0"]["mean"] - lc["0.8"]["mean"]
    res["saturation"] = {"last_step_gain": round(float(slope), 4),
                         "saturated": bool(slope < 0.02),
                         "verdict": "SATURATED (adequately powered)" if slope < 0.02 else "STILL RISING (underpowered -> INCONCLUSIVE)"}
    print("saturation:", res["saturation"]["verdict"])
    json.dump(res, open(OUT / "for20k_cpu.json", "w"), indent=2)
    print("wrote", OUT / "for20k_cpu.json")


if __name__ == "__main__":
    main()
