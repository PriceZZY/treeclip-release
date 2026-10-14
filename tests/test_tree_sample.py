import numpy as np
import pytest
from tests.fixtures.make_synthetic_tree import synthetic_tree
from treeclip.core.tree_sample import TreeSample


def test_treesample_holds_all_modalities():
    s = synthetic_tree()
    assert s.point_cloud.shape == (1024, 3)
    assert s.rgb_crop.shape == (224, 224, 3)
    assert s.hsi_crown.shape == (16, 16, 426)
    assert s.hsi_mask.shape == (16, 16)
    assert s.species == "Pinus_palustris"


def test_npz_roundtrip(tmp_path):
    s = synthetic_tree()
    path = tmp_path / "t001.npz"
    s.save_npz(path)
    s2 = TreeSample.load_npz(path)
    assert s.tree_id == s2.tree_id
    assert s.site == s2.site
    assert s.species == s2.species
    np.testing.assert_array_equal(s.point_cloud, s2.point_cloud)
    np.testing.assert_array_equal(s.rgb_crop, s2.rgb_crop)
    np.testing.assert_array_equal(s.hsi_crown, s2.hsi_crown)
    np.testing.assert_array_equal(s.hsi_mask, s2.hsi_mask)
    assert s.lidar_density == s2.lidar_density


def test_npz_handles_missing_species(tmp_path):
    s = synthetic_tree(species=None)
    path = tmp_path / "u001.npz"
    s.save_npz(path)
    s2 = TreeSample.load_npz(path)
    assert s2.species is None
