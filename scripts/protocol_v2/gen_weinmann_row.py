"""Full Table-2 row for the local multi-scale Weinmann baseline (PFG round-1, R1 comment 4:
stronger baseline into the main results). Loads the saved 21-d local-Weinmann features
(local_weinmann_{for,biodiv}.npz), aligns by treeID to the analyzable subset, and runs the
EXACT Table-2 harness (StandardScaler -> PCA-11 -> KMeans(K), 15 seeds): NMI/ARI/RI/
macro-F1/OA/mIoU with mean±sd. Sanity: NMI must reproduce local_weinmann.json
(0.2265 FOR / 0.192 BioDiv) within 2e-3. Output: runs/protocol_v2/gen/weinmann_row.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (normalized_mutual_info_score, adjusted_rand_score, rand_score,
                             f1_score, accuracy_score, jaccard_score)
from scipy.optimize import linear_sum_assignment
import gen_score as G

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "runs/protocol_v2/gen"
SEEDS = list(range(15))


def hungarian_pred(y, cl):
    sp = sorted(set(y)); si = {s: i for i, s in enumerate(sp)}
    K = len(set(cl)); M = np.zeros((K, len(sp)))
    for c, s in zip(cl, y): M[c, si[s]] += 1
    row, col = linear_sum_assignment(-M)
    cmap = {int(r): sp[int(c)] for r, c in zip(row, col)}
    return np.array([cmap.get(int(c), sp[0]) for c in cl])


def main():
    out = {"n_seeds": len(SEEDS)}
    for which in ["for", "biodiv"]:
        if which == "for":
            d = np.load(ROOT / "data/raw/for_species20k/clip_features.npz", allow_pickle=True)
            y = d["species"].astype(str); groups = d["dataset"].astype(str); tref = d["treeID"]; fag = None
        else:
            d = np.load(ROOT / "data/raw/biodiv_3dtrees/biodiv_features.npz", allow_pickle=True)
            y = d["species"].astype(str); groups = d["region"].astype(str); tref = d["treeID"]; fag = 0.39
        m, _ = G.analyzable_mask(y, groups); y, groups, tref = y[m], groups[m], tref[m]
        if fag:
            c = Counter(y); top = max(c, key=c.get); n = len(y)
            if c[top] / n > fag:
                tgt = int(fag/(1-fag)*(n-c[top])); idx = np.where(y == top)[0]
                drop = set(np.random.RandomState(0).choice(idx, c[top]-tgt, replace=False))
                k = np.array([i not in drop for i in range(n)]); y, groups, tref = y[k], groups[k], tref[k]
        z = np.load(OUT / f"local_weinmann_{which}.npz", allow_pickle=True)
        idx = {str(t): i for i, t in enumerate(z["treeID"])}
        rows = np.array([idx.get(str(t), -1) for t in tref])
        assert (rows >= 0).all(), f"{which}: {int((rows<0).sum())} missing treeIDs"
        X = z["X"][rows]; K = len(set(y))
        per = {k: [] for k in ["NMI","ARI","RI","macroF1","OA","mIoU"]}
        for s in SEEDS:
            Xs = StandardScaler().fit_transform(X)
            Xs = PCA(n_components=11, random_state=s).fit_transform(Xs)
            cl = KMeans(n_clusters=K, random_state=s, n_init=10).fit_predict(Xs)
            yp = hungarian_pred(y, cl)
            per["NMI"].append(normalized_mutual_info_score(y, cl))
            per["ARI"].append(adjusted_rand_score(y, cl))
            per["RI"].append(rand_score(y, cl))
            per["macroF1"].append(f1_score(y, yp, average="macro", zero_division=0))
            per["OA"].append(accuracy_score(y, yp))
            per["mIoU"].append(jaccard_score(y, yp, average="macro", zero_division=0))
        out[which] = {k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
                      for k, v in per.items()}
        print(which, json.dumps(out[which]), flush=True)
    ref = json.load(open(OUT / "local_weinmann.json"))
    for which in ["for", "biodiv"]:
        want = ref[which]["geo_local_multiscale_weinmann"]["NMI_mean"]
        got = out[which]["NMI"]["mean"]
        assert abs(got - want) < 2e-3, f"{which}: sanity fail {got} vs {want}"
    print("SANITY OK: reproduces local_weinmann.json NMI")
    json.dump(out, open(OUT / "weinmann_row.json", "w"), indent=2)


if __name__ == "__main__":
    main()
