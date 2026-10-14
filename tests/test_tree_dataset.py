from pathlib import Path
import pytest
from tests.fixtures.make_synthetic_tree import synthetic_tree
from treeclip.data.tree_dataset import TreeDataset


@pytest.fixture
def populated_cache(tmp_path):
    for i, sp in enumerate(["OSBS", "MLBS", "TALL"]):
        site_dir = tmp_path / sp
        site_dir.mkdir(parents=True)
        for j in range(3):
            tree = synthetic_tree(tree_id=f"{sp}_t{j}", site=sp, species=f"sp_{j}")
            tree.save_npz(site_dir / f"{sp}_t{j}.npz")
    return tmp_path


def test_dataset_discovers_all_trees(populated_cache):
    ds = TreeDataset(cache_root=populated_cache, allowed_ids=None)
    assert len(ds) == 9


def test_dataset_filter_by_allowed_ids(populated_cache):
    allowed = {"OSBS_t0", "OSBS_t1", "MLBS_t0"}
    ds = TreeDataset(cache_root=populated_cache, allowed_ids=allowed)
    assert len(ds) == 3
    seen = {ds[i].tree_id for i in range(len(ds))}
    assert seen == allowed


def test_dataset_getitem_returns_treesample(populated_cache):
    ds = TreeDataset(cache_root=populated_cache, allowed_ids=None)
    sample = ds[0]
    assert sample.point_cloud.ndim == 2
    assert sample.rgb_crop.shape == (224, 224, 3)
