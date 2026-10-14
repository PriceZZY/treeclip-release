"""Test crown loader against synthetic IDTReeS-shaped fixtures."""
import pytest
import geopandas as gpd
import pandas as pd
from pathlib import Path
from shapely.geometry import Polygon
from treeclip.preprocess.idtrees_layout import IDTReeSLayout
from treeclip.preprocess.crowns import (
    CrownRecord,
    load_train_crowns,
    load_test_crowns,
)


def _make_train_fixtures(root: Path):
    """Build a minimal train/ tree mirroring the real IDTReeS layout."""
    (root / "train" / "ITC").mkdir(parents=True)
    (root / "train" / "Field").mkdir(parents=True)
    # train_OSBS.shp
    gpd.GeoDataFrame(
        {"id": [1, 2], "indvdID": ["OSBS00001", "OSBS00002"]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
        ],
        crs="EPSG:32617",
    ).to_file(root / "train" / "ITC" / "train_OSBS.shp")
    # Field/train_data.csv
    pd.DataFrame(
        {
            "indvdID": ["OSBS00001", "OSBS00002"],
            "siteID": ["OSBS", "OSBS"],
            "taxonID": ["PIPA2", "QULA2"],
        }
    ).to_csv(root / "train" / "Field" / "train_data.csv", index=False)
    # Field/itc_rsFile.csv
    pd.DataFrame(
        {
            "id": [1, 2],
            "indvdID": ["OSBS00001", "OSBS00002"],
            "rsFile": ["OSBS_5.tif", "OSBS_8.tif"],
        }
    ).to_csv(root / "train" / "Field" / "itc_rsFile.csv", index=False)


def _make_test_fixtures(root: Path):
    (root / "task2" / "ITC").mkdir(parents=True)
    gpd.GeoDataFrame(
        {"indvdID": ["TALLE00001", "TALLE00002"], "plotID": ["TALL_1.tif", "TALL_3.tif"]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
        ],
        crs="EPSG:32616",
    ).to_file(root / "task2" / "ITC" / "test_TALL.shp")


def test_load_train_crowns_joins_species_and_rsfile(tmp_path):
    _make_train_fixtures(tmp_path)
    layout = IDTReeSLayout(root=tmp_path)
    records = load_train_crowns(layout, "OSBS")
    assert len(records) == 2
    by_id = {r.tree_id: r for r in records}
    assert by_id["OSBS00001"].species == "PIPA2"
    assert by_id["OSBS00001"].rs_file_stem == "OSBS_5"
    assert by_id["OSBS00001"].site == "OSBS"
    assert by_id["OSBS00002"].species == "QULA2"
    assert by_id["OSBS00002"].rs_file_stem == "OSBS_8"


def test_load_test_crowns_has_no_species(tmp_path):
    _make_test_fixtures(tmp_path)
    layout = IDTReeSLayout(root=tmp_path)
    records = load_test_crowns(layout, "TALL")
    assert len(records) == 2
    by_id = {r.tree_id: r for r in records}
    assert by_id["TALLE00001"].species is None
    assert by_id["TALLE00001"].rs_file_stem == "TALL_1"
    assert by_id["TALLE00001"].site == "TALL"


def test_train_crown_skips_when_rsfile_missing(tmp_path):
    """If indvdID has no itc_rsFile entry, that crown is skipped (no tile to load)."""
    _make_train_fixtures(tmp_path)
    # Drop one row from itc_rsFile.csv
    pd.DataFrame(
        {"id": [1], "indvdID": ["OSBS00001"], "rsFile": ["OSBS_5.tif"]}
    ).to_csv(tmp_path / "train" / "Field" / "itc_rsFile.csv", index=False)
    layout = IDTReeSLayout(root=tmp_path)
    records = load_train_crowns(layout, "OSBS")
    assert len(records) == 1
    assert records[0].tree_id == "OSBS00001"
