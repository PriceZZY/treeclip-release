"""Crown polygons + species labels for IDTReeS.

Train crowns: shapefile (indvdID -> geometry) joined with `train_data.csv` (indvdID -> taxonID)
              and `itc_rsFile.csv` (indvdID -> rsFile). Yields fully-labeled CrownRecords.
Test crowns:  shapefile (indvdID, plotID -> geometry). plotID already gives the tile filename;
              species is None because the public release withholds test labels.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import math
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon

from treeclip.preprocess.idtrees_layout import IDTReeSLayout


@dataclass(frozen=True)
class CrownRecord:
    """One labeled or unlabeled crown.

    rs_file_stem: filename stem without extension, e.g. "OSBS_103".
                  Use with layout.{train,test}_modality_dir(modality) / f"{stem}.tif" (or .las).
    """
    tree_id: str
    site: str
    species: Optional[str]
    polygon: Polygon | MultiPolygon
    crs: str
    rs_file_stem: str


def _stem(file_name: str) -> str:
    """Strip extension from a tile filename like 'OSBS_103.tif' -> 'OSBS_103'."""
    return Path(file_name).stem


def _species_or_none(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    s = str(value)
    return None if s.strip() == "" or s.lower() in {"nan", "none"} else s


def load_train_crowns(layout: IDTReeSLayout, site: str) -> list[CrownRecord]:
    """Load train crowns for a site, joined with species labels + rs file lookup."""
    shp = gpd.read_file(layout.train_crowns_shp(site))
    field = pd.read_csv(layout.train_field_csv())
    rsfile = pd.read_csv(layout.train_itc_rsfile_csv())

    field_idx = field.set_index("indvdID")["taxonID"].to_dict()
    rsfile_idx = rsfile.set_index("indvdID")["rsFile"].to_dict()

    records: list[CrownRecord] = []
    for _, row in shp.iterrows():
        ind_id = str(row["indvdID"])
        species = _species_or_none(field_idx.get(ind_id))
        rs_file = rsfile_idx.get(ind_id)
        if rs_file is None:
            continue
        records.append(
            CrownRecord(
                tree_id=ind_id,
                site=site,
                species=species,
                polygon=row.geometry,
                crs=str(shp.crs),
                rs_file_stem=_stem(rs_file),
            )
        )
    return records


def load_test_crowns(layout: IDTReeSLayout, site: str) -> list[CrownRecord]:
    """Load test/task2 crowns. species is always None (public release withholds labels)."""
    shp = gpd.read_file(layout.test_crowns_shp(site))
    records: list[CrownRecord] = []
    for _, row in shp.iterrows():
        records.append(
            CrownRecord(
                tree_id=str(row["indvdID"]),
                site=site,
                species=None,
                polygon=row.geometry,
                crs=str(shp.crs),
                rs_file_stem=_stem(str(row["plotID"])),
            )
        )
    return records
