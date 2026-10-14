import pytest
from pathlib import Path
from treeclip.preprocess.idtrees_layout import (
    IDTReeSLayout,
    SITES,
    LABELED_SITES,
    MODALITIES,
)


def test_constants():
    assert SITES == ("OSBS", "MLBS", "TALL")
    assert LABELED_SITES == ("OSBS", "MLBS")
    assert set(MODALITIES) == {"RGB", "HSI", "LAS", "CHM"}


def test_train_paths(tmp_path):
    layout = IDTReeSLayout(root=tmp_path)
    assert layout.train_modality_dir("RGB") == tmp_path / "train" / "RemoteSensing" / "RGB"
    assert layout.train_modality_dir("HSI") == tmp_path / "train" / "RemoteSensing" / "HSI"
    assert layout.train_modality_dir("LAS") == tmp_path / "train" / "RemoteSensing" / "LAS"
    assert layout.train_crowns_shp("OSBS") == tmp_path / "train" / "ITC" / "train_OSBS.shp"
    assert layout.train_field_csv() == tmp_path / "train" / "Field" / "train_data.csv"
    assert layout.train_itc_rsfile_csv() == tmp_path / "train" / "Field" / "itc_rsFile.csv"


def test_test_paths(tmp_path):
    layout = IDTReeSLayout(root=tmp_path)
    assert layout.test_modality_dir("RGB") == tmp_path / "task2" / "RemoteSensing" / "RGB"
    assert layout.test_crowns_shp("TALL") == tmp_path / "task2" / "ITC" / "test_TALL.shp"


def test_train_crowns_rejects_tall(tmp_path):
    layout = IDTReeSLayout(root=tmp_path)
    with pytest.raises(ValueError, match="no public training labels"):
        layout.train_crowns_shp("TALL")


def test_unknown_modality_raises(tmp_path):
    layout = IDTReeSLayout(root=tmp_path)
    with pytest.raises(ValueError, match="Unknown modality"):
        layout.train_modality_dir("xxx")


def test_unknown_site_raises(tmp_path):
    layout = IDTReeSLayout(root=tmp_path)
    with pytest.raises(ValueError, match="Unknown site"):
        layout.test_crowns_shp("xxx")
