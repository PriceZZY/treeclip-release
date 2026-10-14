"""End-to-end synthetic test: preprocess → cache → dataset → split → load.

Wires preprocess + cache + dataset + within-site/LOSO splits together using a
minimal IDTReeS-shaped fixture. Real data is exercised by Task 14 (run script
against actual IDTReeS download).
"""
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon
import laspy
import geopandas as gpd
import pandas as pd

from treeclip.preprocess.idtrees_layout import IDTReeSLayout
from treeclip.preprocess.run_preprocess import preprocess_train_site
from treeclip.core.tree_sample import TreeSample
from treeclip.data.tree_dataset import TreeDataset
from treeclip.splits.within_site import make_within_site_split
from treeclip.splits.cross_site_loso import make_loso_splits


def _make_site_fixture(layout: IDTReeSLayout, site: str, n_trees: int):
    """Build a minimal labeled site under layout.root with n_trees crowns + 1 shared tile."""
    root = layout.root
    # ITC shapefile
    (root / "train" / "ITC").mkdir(parents=True, exist_ok=True)
    polys = []
    ids = []
    for i in range(n_trees):
        x = i * 3
        polys.append(Polygon([(x, 0), (x + 2, 0), (x + 2, 2), (x, 2)]))
        ids.append(f"{site}_{i:04d}")
    gpd.GeoDataFrame(
        {"id": list(range(n_trees)), "indvdID": ids},
        geometry=polys,
        crs="EPSG:32617",
    ).to_file(layout.train_crowns_shp(site))
    # Field CSVs — APPEND so calling with multiple sites accumulates rather than overwrites
    (root / "train" / "Field").mkdir(parents=True, exist_ok=True)
    new_field = pd.DataFrame(
        {"indvdID": ids, "siteID": [site] * n_trees, "taxonID": [f"sp_{i % 3}" for i in range(n_trees)]}
    )
    if layout.train_field_csv().exists():
        new_field = pd.concat([pd.read_csv(layout.train_field_csv()), new_field], ignore_index=True)
    new_field.to_csv(layout.train_field_csv(), index=False)
    new_rs = pd.DataFrame(
        {"id": list(range(n_trees)), "indvdID": ids, "rsFile": [f"{site}_1.tif"] * n_trees}
    )
    if layout.train_itc_rsfile_csv().exists():
        new_rs = pd.concat([pd.read_csv(layout.train_itc_rsfile_csv()), new_rs], ignore_index=True)
    new_rs.to_csv(layout.train_itc_rsfile_csv(), index=False)
    # One shared tile per modality (named SITE_1.tif / .las) that covers all crowns
    rgb_path = layout.train_modality_dir("RGB") / f"{site}_1.tif"
    hsi_path = layout.train_modality_dir("HSI") / f"{site}_1.tif"
    las_path = layout.train_modality_dir("LAS") / f"{site}_1.las"
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    hsi_path.parent.mkdir(parents=True, exist_ok=True)
    las_path.parent.mkdir(parents=True, exist_ok=True)
    # tile spans 3*n_trees x 5 with 10cm GSD (so it covers all polygons)
    h = 50
    w = max(30 * n_trees, 50)
    with rasterio.open(
        rgb_path, "w",
        driver="GTiff", height=h, width=w, count=3,
        dtype="uint8", crs="EPSG:32617",
        transform=from_origin(0, h * 0.1, 0.1, 0.1),
    ) as dst:
        dst.write(np.ones((3, h, w), dtype=np.uint8) * 100)
    with rasterio.open(
        hsi_path, "w",
        driver="GTiff", height=10, width=max(3 * n_trees, 10), count=5,
        dtype="float32", crs="EPSG:32617",
        transform=from_origin(0, 10, 1.0, 1.0),
    ) as dst:
        dst.write(np.ones((5, 10, max(3 * n_trees, 10)), dtype=np.float32) * 0.5)
    rng = np.random.default_rng(0)
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    ld = laspy.LasData(header)
    ld.x = rng.uniform(0, 3 * n_trees, 1000)
    ld.y = rng.uniform(0, 3, 1000)
    ld.z = rng.uniform(0, 30, 1000)
    ld.write(str(las_path))


def test_full_pipeline_two_sites(tmp_path):
    """Full pipeline: preprocess OSBS+MLBS, build cache, iterate via Dataset, make splits."""
    layout = IDTReeSLayout(root=tmp_path / "idtrees")
    _make_site_fixture(layout, "OSBS", n_trees=5)
    _make_site_fixture(layout, "MLBS", n_trees=4)

    cache_root = tmp_path / "cache"
    n_o, _ = preprocess_train_site(layout, cache_root, "OSBS")
    n_m, _ = preprocess_train_site(layout, cache_root, "MLBS")
    assert n_o == 5
    assert n_m == 4

    # Iterate the cache
    ds = TreeDataset(cache_root=cache_root, allowed_ids=None)
    assert len(ds) == 9
    by_site: dict[str, list[str]] = {}
    for i in range(len(ds)):
        sample = ds[i]
        by_site.setdefault(sample.site, []).append(sample.tree_id)
        assert sample.rgb_crop.shape == (224, 224, 3)
        assert sample.hsi_crown.ndim == 3
        assert sample.point_cloud.shape[1] == 3
        assert sample.species in {"sp_0", "sp_1", "sp_2"}
    assert set(by_site.keys()) == {"OSBS", "MLBS"}

    # Cross-site LOSO over the two labeled sites
    folds = make_loso_splits(by_site)
    assert len(folds) == 2
    for fold in folds:
        if fold.test_site == "OSBS":
            assert len(fold.test) == 5
            assert len(fold.train) == 4
        else:
            assert len(fold.test) == 4
            assert len(fold.train) == 5

    # Within-site split
    split = make_within_site_split(
        by_site["OSBS"], val_frac=0.2, test_frac=0.2, seed=42, output_path=None
    )
    assert len(split.train) + len(split.val) + len(split.test) == 5
