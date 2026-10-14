"""Weiser 2022 SYSSIFOSS preprocessing.

For each per-tree LAS/LAZ file:
  * parse species + plot + tree_id from filename
  * read point cloud, center XY at centroid (no need to clip — already per-tree)
  * estimate point density (pts / m^2 in XY bbox)
  * build a TreeSample with point_cloud only (no RGB, HSI, or crown mask)
  * save to cache/{plot}/{tree_id}.npz

The TreeSample's rgb_crop / hsi_crown / hsi_mask fields are filled with empty arrays
to satisfy the dataclass contract; downstream encoders (CLIPStructEncoder) only read
point_cloud.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import laspy
from tqdm import tqdm

from treeclip.core.tree_sample import TreeSample
from treeclip.preprocess.weiser_layout import (
    WeiserLayout, LABELED_PLOTS, parse_single_tree_filename,
)


def _read_laz_xyz(path: Path) -> np.ndarray:
    las = laspy.read(str(path))
    return np.column_stack([las.x, las.y, las.z]).astype(np.float32)


def _center_xy_and_density(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Center the XY of points at their centroid; estimate pts / m^2 from XY bbox."""
    if pts.shape[0] == 0:
        return pts, 0.0
    centroid_xy = pts[:, :2].mean(axis=0)
    centered = pts.copy()
    centered[:, :2] -= centroid_xy.astype(np.float32)
    minx, miny = centered[:, :2].min(axis=0)
    maxx, maxy = centered[:, :2].max(axis=0)
    area = max((maxx - minx) * (maxy - miny), 1e-6)
    density = float(pts.shape[0] / area)
    return centered, density


def preprocess_one_tree_file(
    laz_path: Path,
    species: str,
    plot: str,
    tree_id: str,
) -> TreeSample:
    pts = _read_laz_xyz(laz_path)
    centered, density = _center_xy_and_density(pts)
    return TreeSample(
        tree_id=tree_id,
        site=plot,
        species=species,
        point_cloud=centered,
        # Weiser has no RGB / HSI — fill with empty arrays
        rgb_crop=np.zeros((0, 0, 0), dtype=np.uint8),
        hsi_crown=np.zeros((0, 0, 0), dtype=np.float32),
        hsi_mask=np.zeros((0, 0), dtype=bool),
        lidar_density=density,
    )


def discover_tree_files(
    plot_dir: Path, platform_filter: str | None = "ALS"
) -> list[tuple[Path, dict]]:
    """Walk a plot directory, find single-tree LAS/LAZ files matching the naming
    convention. Returns list of (path, parsed_fields)."""
    out = []
    for p in plot_dir.rglob("*.la[sz]"):
        # Filter on platform tag in path (ALS / ULS / TLS)
        if platform_filter is not None:
            parts_lower = [seg.lower() for seg in p.parts]
            if not any(platform_filter.lower() in s for s in parts_lower):
                continue
        fields = parse_single_tree_filename(p.name)
        if fields is None:
            continue
        out.append((p, fields))
    return out


def preprocess_plot(
    layout: WeiserLayout,
    cache_root: Path,
    plot: str,
    platform_filter: str | None = "ALS",
) -> tuple[int, int]:
    # Be robust to two zip layouts: root/PLOT/single_trees/... vs root/single_trees/...
    candidates = [layout.plot_dir(plot), layout.root]
    files: list = []
    for cand in candidates:
        if cand.exists():
            files = discover_tree_files(cand, platform_filter=platform_filter)
            # Filter by plot from filename (rglob from root would catch other plots)
            files = [(p, f) for p, f in files if f["plot"] == plot]
            if files:
                break
    out_dir = cache_root / plot
    out_dir.mkdir(parents=True, exist_ok=True)
    cached = 0
    skipped = 0
    for laz_path, fields in tqdm(files, desc=f"{plot}/{platform_filter or 'ALL'}"):
        try:
            sample = preprocess_one_tree_file(
                laz_path=laz_path,
                species=fields["species"],
                plot=fields["plot"],
                tree_id=fields["treeid"],
            )
        except Exception as e:
            print(f"  skip {laz_path.name}: {e}")
            skipped += 1
            continue
        sample.save_npz(out_dir / f"{fields['treeid']}.npz")
        cached += 1
    return cached, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weiser-root", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--plots", nargs="+", default=list(LABELED_PLOTS))
    ap.add_argument("--platform", default="ALS",
                    help="restrict to one platform (ALS / ULS / TLS); 'none' = all")
    args = ap.parse_args()

    platform = None if args.platform.lower() == "none" else args.platform
    layout = WeiserLayout(root=args.weiser_root)
    for plot in args.plots:
        n, sk = preprocess_plot(layout, args.cache_root, plot, platform_filter=platform)
        print(f"{plot}: {n} cached, {sk} skipped")


if __name__ == "__main__":
    main()
