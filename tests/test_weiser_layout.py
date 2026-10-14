import pytest
from treeclip.preprocess.weiser_layout import (
    WeiserLayout, LABELED_PLOTS, ALL_PLOTS, parse_single_tree_filename,
)


def test_constants():
    assert LABELED_PLOTS == ("BR01", "BR03", "BR05", "KA09", "KA10", "KA11")
    assert len(ALL_PLOTS) == 12
    assert set(LABELED_PLOTS).issubset(set(ALL_PLOTS))


def test_layout_paths(tmp_path):
    layout = WeiserLayout(root=tmp_path)
    assert layout.plot_dir("KA10") == tmp_path / "KA10"
    assert layout.single_trees_dir("KA10") == tmp_path / "KA10" / "single_trees"


def test_unknown_plot_raises(tmp_path):
    layout = WeiserLayout(root=tmp_path)
    with pytest.raises(ValueError, match="Unknown plot"):
        layout.plot_dir("XX99")


def test_parse_single_tree_filename_basic():
    fields = parse_single_tree_filename("FagSyl_BR01_t003_2019-07-05.laz")
    assert fields is not None
    assert fields["species"] == "FagSyl"
    assert fields["plot"] == "BR01"
    assert fields["treeid"] == "t003"
    assert fields["date"] == "2019-07-05"


def test_parse_single_tree_filename_with_suffix():
    fields = parse_single_tree_filename("QueRob_KA10_005_2019-07-05_normalized.las")
    assert fields is not None
    assert fields["species"] == "QueRob"
    assert fields["plot"] == "KA10"
    assert fields["treeid"] == "005"
    assert fields["suffix"] == "normalized"


def test_parse_invalid_filename_returns_none():
    assert parse_single_tree_filename("not_a_valid_tree.txt") is None
    assert parse_single_tree_filename("ALS-on_BR01_2019-07-05.laz") is None  # plot-level, not tree
