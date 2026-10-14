import pytest
from pathlib import Path
from treeclip.splits.within_site import make_within_site_split


def test_within_site_split_deterministic(tmp_path):
    tree_ids = [f"t{i:03d}" for i in range(100)]
    out = tmp_path / "split.yaml"
    split = make_within_site_split(
        tree_ids=tree_ids, val_frac=0.2, test_frac=0.2, seed=42, output_path=out
    )
    split2 = make_within_site_split(
        tree_ids=tree_ids, val_frac=0.2, test_frac=0.2, seed=42, output_path=None
    )
    assert split.train == split2.train
    assert split.val == split2.val
    assert split.test == split2.test


def test_within_site_split_sizes_and_disjoint():
    tree_ids = [f"t{i:03d}" for i in range(100)]
    split = make_within_site_split(
        tree_ids=tree_ids, val_frac=0.2, test_frac=0.2, seed=42, output_path=None
    )
    assert len(split.train) == 60
    assert len(split.val) == 20
    assert len(split.test) == 20
    assert set(split.train).isdisjoint(set(split.val))
    assert set(split.train).isdisjoint(set(split.test))
    assert set(split.val).isdisjoint(set(split.test))
    assert set(split.train) | set(split.val) | set(split.test) == set(tree_ids)


def test_within_site_split_writes_yaml(tmp_path):
    tree_ids = [f"t{i:03d}" for i in range(20)]
    out = tmp_path / "split.yaml"
    make_within_site_split(
        tree_ids=tree_ids, val_frac=0.2, test_frac=0.2, seed=0, output_path=out
    )
    assert out.exists()
    import yaml
    d = yaml.safe_load(out.read_text())
    assert set(d.keys()) >= {"train", "val", "test"}
