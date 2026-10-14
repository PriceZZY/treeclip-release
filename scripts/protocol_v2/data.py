"""Aligned data loader for protocol v2.

Single pass over the canonical sorted-glob order (identical to _common.load_dataset),
assembling everything index-aligned: tree_id, species(y), point cloud, geometric
features (scale-free + size variants), spatial group, UTM/relative xy, and the cached
CLIP topview feature (mean of h0-h5).

A hard assertion guarantees the CLIP-cache species order == freshly-loaded species
order, so any future drift fails loudly instead of silently misaligning.
"""
from __future__ import annotations
import sys, pickle, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from treeclip.core.tree_sample import TreeSample

RUNS = ROOT / "runs"

# canonical order — MUST match _common.DATASETS site ordering
DATASETS = [
    ("Weiser",    ROOT / "data/cache_weiser", ["KA09", "KA10", "KA11", "BR01", "BR03", "BR05"]),
    ("NEON-OSBS", ROOT / "data/cache",        ["OSBS"]),
    ("NEON-MLBS", ROOT / "data/cache",        ["MLBS"]),
]


def _eigs(pts):
    cov = np.cov(pts.T)
    ev = np.clip(np.sort(np.linalg.eigvalsh(cov))[::-1], 1e-9, None)
    return ev  # l1>=l2>=l3


def geo_features(pts):
    """Return (scale_free_11d, size_17d).

    scale_free (11): 6 DIMENSIONLESS covariance ratios (linearity, planarity, sphericity,
      anisotropy, eigenentropy, change-of-curvature) + 5 height-FRACTION percentiles (z/h).
      Contains NO absolute size — verified invariant under isotropic scaling.
    size (17): the 6 ratios + omnivariance + sum(eigs) (both length^2, carry scale)
      + absolute height + crown extents + absolute z-percentiles.
    """
    if pts.shape[0] < 3:
        return np.zeros(11, np.float32), np.zeros(17, np.float32)
    l1, l2, l3 = _eigs(pts); s = l1 + l2 + l3
    ev = np.array([l1, l2, l3])
    ratios6 = np.array([(l1-l2)/l1, (l2-l3)/l1, l3/l1, (l1-l3)/l1,
                        -sum((e/s)*np.log(e/s) for e in ev), l3/s], dtype=np.float32)
    z = pts[:, 2]; z0 = z - z.min(); h = float(z0.max()) + 1e-9
    zpct = np.percentile(z0, [10, 25, 50, 75, 90]).astype(np.float32)
    cx = float(np.ptp(pts[:, 0])); cy = float(np.ptp(pts[:, 1]))
    omni = float((l1 * l2 * l3) ** (1/3))
    scale_free = np.concatenate([ratios6, (zpct / h).astype(np.float32)])               # 11
    size = np.concatenate([ratios6, np.array([omni, s, h, max(cx, cy), cx, cy], np.float32), zpct])  # 17
    return scale_free, size


def load(ds_name):
    root = dict((d[0], d[1]) for d in DATASETS)[ds_name]
    sites = dict((d[0], d[2]) for d in DATASETS)[ds_name]

    # CLIP cache (tree_views ordered same as canonical glob; y for the alignment check)
    with open(RUNS / f"topview_cache_{ds_name}.pkl", "rb") as f:
        cd = pickle.load(f)
    clip = np.array([np.mean([v[f"h{i}"] for i in range(6)], axis=0) for v in cd["tree_views"]])
    cache_y = list(cd["y"])

    neon_meta = None
    if ds_name.startswith("NEON"):
        neon_meta = pickle.load(open(RUNS / "spatial_meta.pkl", "rb"))[ds_name]

    tree_id, y, points, gsf, gsz, group, xs, ys = [], [], [], [], [], [], [], []
    for site in sites:
        for p in sorted((root / site).glob("*.npz")):
            t = TreeSample.load_npz(p)
            if t.point_cloud.shape[0] < 3 or t.species is None:
                continue
            pc = t.point_cloud.astype(np.float32)
            a, b = geo_features(pc)
            tree_id.append(t.tree_id); y.append(t.species); points.append(pc)
            gsf.append(a); gsz.append(b)
            if neon_meta is not None:
                m = neon_meta[t.tree_id]
                group.append(m["tile"]); xs.append(m["x"]); ys.append(m["y"])
            else:  # Weiser: plot = dir, coords = pc centroid
                group.append(site); xs.append(float(pc[:, 0].mean())); ys.append(float(pc[:, 1].mean()))

    y = np.array(y)
    assert len(y) == len(cache_y), f"{ds_name}: loaded {len(y)} vs cache {len(cache_y)}"
    mism = sum(1 for i in range(len(y)) if str(y[i]) != str(cache_y[i]))
    assert mism == 0, f"{ds_name}: {mism} species/order mismatches vs CLIP cache — ALIGNMENT BROKEN"

    return {
        "ds": ds_name, "tree_id": tree_id, "y": y,
        "clip": clip.astype(np.float32),
        "geo_sf": np.array(gsf), "geo_sz": np.array(gsz),
        "points": points,
        "group": np.array(group), "xy": np.array([xs, ys]).T,
        "K": len(set(y)),
    }


if __name__ == "__main__":
    for ds, _, _ in DATASETS:
        d = load(ds)
        npts = np.array([p.shape[0] for p in d["points"]])
        print(f"{ds}: N={len(d['y'])} K={d['K']} groups={len(set(d['group']))} "
              f"clip{d['clip'].shape} geo_sf{d['geo_sf'].shape} geo_sz{d['geo_sz'].shape} "
              f"pts(median={int(np.median(npts))},max={npts.max()}) ALIGN-OK")
