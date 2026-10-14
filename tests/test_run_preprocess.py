"""Test the orchestrator with synthetic fixtures matching the real IDTReeS layout."""
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon
from pathlib import Path
import laspy
import geopandas as gpd
import pandas as pd

from treeclip.preprocess.run_preprocess import (
    preprocess_one_tree,
    preprocess_train_site,
)
from treeclip.preprocess.crowns import CrownRecord
from treeclip.preprocess.idtrees_layout import IDTReeSLayout
from treeclip.core.tree_sample import TreeSample


def _write_rgb(path, h=50, w=50):
    data = np.ones((3, h, w), dtype=np.uint8) * 128
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w",
        driver="GTiff", height=h, width=w, count=3,
        dtype="uint8", crs="EPSG:32617",
        transform=from_origin(0, h * 0.1, 0.1, 0.1),
    ) as dst:
        dst.write(data)


def _write_hsi(path, bands=5, h=20, w=20):
    data = np.ones((bands, h, w), dtype=np.float32) * 0.3
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w",
        driver="GTiff", height=h, width=w, count=bands,
        dtype="float32", crs="EPSG:32617",
        transform=from_origin(0, h, 1.0, 1.0),
    ) as dst:
        dst.write(data)


def _write_lidar(path, n=500):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x = rng.uniform(0, 5, n)
    las.y = rng.uniform(0, 5, n)
    las.z = rng.uniform(0, 30, n)
    las.write(str(path))


def test_preprocess_one_tree_with_explicit_paths(tmp_path):
    rgb_path = tmp_path / "rgb.tif"; _write_rgb(rgb_path)
    hsi_path = tmp_path / "hsi.tif"; _write_hsi(hsi_path)
    las_path = tmp_path / "pts.las"; _write_lidar(las_path)
    crown = CrownRecord(
        tree_id="t001",
        site="OSBS",
        species="Pinus_palustris",
        polygon=Polygon([(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)]),
        crs="EPSG:32617",
        rs_file_stem="ignored_here",
    )

    sample = preprocess_one_tree(
        crown=crown, rgb_path=rgb_path, hsi_path=hsi_path, lidar_path=las_path,
    )
    assert isinstance(sample, TreeSample)
    assert sample.tree_id == "t001"
    assert sample.site == "OSBS"
    assert sample.species == "Pinus_palustris"
    assert sample.rgb_crop.shape == (224, 224, 3)
    assert sample.hsi_crown.ndim == 3
    assert sample.point_cloud.shape[1] == 3
    assert sample.lidar_density >= 0


def test_preprocess_train_site_end_to_end(tmp_path):
    """Mini IDTReeS-shaped fixture: one site, one crown, one tile per modality."""
    root = tmp_path / "idtrees"
    layout = IDTReeSLayout(root=root)
    # ITC shapefile
    (root / "train" / "ITC").mkdir(parents=True)
    gpd.GeoDataFrame(
        {"id": [1], "indvdID": ["OSBS00001"]},
        geometry=[Polygon([(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)])],
        crs="EPSG:32617",
    ).to_file(root / "train" / "ITC" / "train_OSBS.shp")
    # Field CSVs
    (root / "train" / "Field").mkdir(parents=True)
    pd.DataFrame({"indvdID": ["OSBS00001"], "siteID": ["OSBS"], "taxonID": ["PIPA2"]}).to_csv(
        root / "train" / "Field" / "train_data.csv", index=False
    )
    pd.DataFrame({"id": [1], "indvdID": ["OSBS00001"], "rsFile": ["OSBS_5.tif"]}).to_csv(
        root / "train" / "Field" / "itc_rsFile.csv", index=False
    )
    # Per-modality tile named OSBS_5
    _write_rgb(layout.train_modality_dir("RGB") / "OSBS_5.tif")
    _write_hsi(layout.train_modality_dir("HSI") / "OSBS_5.tif")
    _write_lidar(layout.train_modality_dir("LAS") / "OSBS_5.las")

    cache_root = tmp_path / "cache"
    n_cached, n_skipped = preprocess_train_site(layout, cache_root, "OSBS")
    assert n_cached == 1
    assert n_skipped == 0

    cached_file = cache_root / "OSBS" / "OSBS00001.npz"
    assert cached_file.exists()
    sample = TreeSample.load_npz(cached_file)
    assert sample.species == "PIPA2"
    assert sample.site == "OSBS"
    assert sample.rgb_crop.shape == (224, 224, 3)
