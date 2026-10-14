"""IDTReeS 2020 actual on-disk layout (Zenodo record 3934932 v2).

Real structure after unzipping `IDTREES_competition_{train,test}_v2.zip` into a single root::

    {root}/
      train/
        RemoteSensing/{RGB,HSI,LAS,CHM}/{SITE}_{N}.tif   # tiles, site in filename
        ITC/train_{SITE}.shp                              # crown polygons with `indvdID`
        Field/train_data.csv                              # species labels keyed by `indvdID`
        Field/itc_rsFile.csv                              # maps `indvdID` -> rsFile tile name
        Field/taxonID_ScientificName.csv                  # taxonID -> scientific name
      task2/
        RemoteSensing/{RGB,HSI,LAS,CHM}/{SITE}_{N}.tif
        ITC/test_{SITE}.shp                               # crown polygons, NO species labels in public release

Sites in TRAIN (labels available):  OSBS, MLBS
Sites in TEST/task2 (no public labels): OSBS, MLBS, TALL
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

SITES = ("OSBS", "MLBS", "TALL")
LABELED_SITES = ("OSBS", "MLBS")   # only these have public species labels
MODALITIES = ("RGB", "HSI", "LAS", "CHM")


@dataclass(frozen=True)
class IDTReeSLayout:
    root: Path

    # ----- TRAIN -----
    def train_modality_dir(self, modality: str) -> Path:
        self._check_modality(modality)
        return self.root / "train" / "RemoteSensing" / modality

    def train_crowns_shp(self, site: str) -> Path:
        self._check_labeled_site(site)
        return self.root / "train" / "ITC" / f"train_{site}.shp"

    def train_field_csv(self) -> Path:
        return self.root / "train" / "Field" / "train_data.csv"

    def train_itc_rsfile_csv(self) -> Path:
        return self.root / "train" / "Field" / "itc_rsFile.csv"

    def train_taxon_csv(self) -> Path:
        return self.root / "train" / "Field" / "taxonID_ScientificName.csv"

    # ----- TEST (task2) -----
    def test_modality_dir(self, modality: str) -> Path:
        self._check_modality(modality)
        return self.root / "task2" / "RemoteSensing" / modality

    def test_crowns_shp(self, site: str) -> Path:
        self._check_site(site)
        return self.root / "task2" / "ITC" / f"test_{site}.shp"

    # ----- validators -----
    def _check_site(self, site: str) -> None:
        if site not in SITES:
            raise ValueError(f"Unknown site {site!r}; expected one of {SITES}")

    def _check_labeled_site(self, site: str) -> None:
        if site not in LABELED_SITES:
            raise ValueError(
                f"Site {site!r} has no public training labels. Labeled sites: {LABELED_SITES}"
            )

    def _check_modality(self, modality: str) -> None:
        if modality not in MODALITIES:
            raise ValueError(
                f"Unknown modality {modality!r}; expected one of {MODALITIES}"
            )
