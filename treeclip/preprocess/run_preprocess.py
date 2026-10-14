"""End-to-end IDTReeS preprocessing.

For each crown:
  * locate the matching tile in train/RemoteSensing/{modality}/ (or task2/...) by `rs_file_stem`
  * crop point cloud, HSI, RGB to crown polygon
  * build a TreeSample and save to data/cache/{site}/{tree_id}.npz
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Callable
import laspy
import numpy as np
from tqdm import tqdm

from treeclip.core.tree_sample import TreeSample
from treeclip.preprocess.crowns import (
    CrownRecord,
    load_train_crowns,
    load_test_crowns,
)
from treeclip.preprocess.crop_pointcloud import crop_and_center
from treeclip.preprocess.crop_hsi import extract_crown_hsi
from treeclip.preprocess.crop_rgb import crop_rgb_to_224
from treeclip.preprocess.idtrees_layout import (
    IDTReeSLayout,
    SITES,
    LABELED_SITES,
)


def _read_lidar_xyz(path: Path) -> np.ndarray:
    las = laspy.read(str(path))
    return np.column_stack([las.x, las.y, las.z]).astype(np.float32)


def preprocess_one_tree(
    crown: CrownRecord,
    rgb_path: Path,
    hsi_path: Path,
    lidar_path: Path,
) -> TreeSample:
    rgb = crop_rgb_to_224(rgb_path, crown.polygon, padding_m=0.5)
    hsi, mask = extract_crown_hsi(hsi_path, crown.polygon)
    pts = _read_lidar_xyz(lidar_path)
    pts_cropped, density = crop_and_center(pts, crown.polygon)
    return TreeSample(
        tree_id=crown.tree_id,
        site=crown.site,
        species=crown.species,
        point_cloud=pts_cropped,
        rgb_crop=rgb,
        hsi_crown=hsi,
        hsi_mask=mask,
        lidar_density=density,
    )


def _resolve_tiles(
    crown: CrownRecord,
    rgb_dir: Path,
    hsi_dir: Path,
    lidar_dir: Path,
) -> tuple[Path | None, Path | None, Path | None]:
    """Direct lookup via crown.rs_file_stem. RGB and HSI are .tif; LiDAR is .laz or .las."""
    stem = crown.rs_file_stem
    rgb = rgb_dir / f"{stem}.tif"
    hsi = hsi_dir / f"{stem}.tif"
    laz = lidar_dir / f"{stem}.laz"
    las_ = lidar_dir / f"{stem}.las"
    lidar = laz if laz.exists() else (las_ if las_.exists() else None)
    return (
        rgb if rgb.exists() else None,
        hsi if hsi.exists() else None,
        lidar,
    )


def _preprocess_records(
    records: list[CrownRecord],
    rgb_dir: Path,
    hsi_dir: Path,
    lidar_dir: Path,
    cache_root: Path,
    desc: str,
) -> tuple[int, int]:
    """Process a list of crowns, return (n_cached, n_skipped)."""
    cached = 0
    skipped = 0
    for crown in tqdm(records, desc=desc):
        rgb_path, hsi_path, lidar_path = _resolve_tiles(crown, rgb_dir, hsi_dir, lidar_dir)
        if rgb_path is None or hsi_path is None or lidar_path is None:
            skipped += 1
            continue
        sample = preprocess_one_tree(crown, rgb_path, hsi_path, lidar_path)
        out_dir = cache_root / crown.site
        out_dir.mkdir(parents=True, exist_ok=True)
        sample.save_npz(out_dir / f"{crown.tree_id}.npz")
        cached += 1
    return cached, skipped


def preprocess_train_site(layout: IDTReeSLayout, cache_root: Path, site: str) -> tuple[int, int]:
    records = load_train_crowns(layout, site)
    return _preprocess_records(
        records=records,
        rgb_dir=layout.train_modality_dir("RGB"),
        hsi_dir=layout.train_modality_dir("HSI"),
        lidar_dir=layout.train_modality_dir("LAS"),
        cache_root=cache_root,
        desc=f"train/{site}",
    )


def preprocess_test_site(layout: IDTReeSLayout, cache_root: Path, site: str) -> tuple[int, int]:
    records = load_test_crowns(layout, site)
    return _preprocess_records(
        records=records,
        rgb_dir=layout.test_modality_dir("RGB"),
        hsi_dir=layout.test_modality_dir("HSI"),
        lidar_dir=layout.test_modality_dir("LAS"),
        cache_root=cache_root,
        desc=f"test/{site}",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idtrees-root", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument(
        "--split", choices=["train", "test", "both"], default="train",
        help="train = labeled OSBS+MLBS only; test = task2 (incl. TALL, no labels); both = run both phases."
    )
    ap.add_argument("--sites", nargs="+", default=None,
                    help="restrict to specific sites; default depends on --split")
    args = ap.parse_args()

    layout = IDTReeSLayout(root=args.idtrees_root)
    if args.split in ("train", "both"):
        sites = args.sites or list(LABELED_SITES)
        for site in sites:
            if site not in LABELED_SITES:
                print(f"  WARN: skipping {site} for train (no public labels)")
                continue
            n, sk = preprocess_train_site(layout, args.cache_root, site)
            print(f"train/{site}: {n} cached, {sk} skipped")
    if args.split in ("test", "both"):
        sites = args.sites or list(SITES)
        # cache test crowns in a separate subtree so they don't get mixed with labeled
        test_cache = args.cache_root.parent / (args.cache_root.name + "_test")
        for site in sites:
            n, sk = preprocess_test_site(layout, test_cache, site)
            print(f"test/{site}: {n} cached, {sk} skipped  (-> {test_cache})")


if __name__ == "__main__":
    main()
