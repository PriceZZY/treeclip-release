"""Test the Weiser preprocessing pipeline with synthetic LAS files."""
import numpy as np
import laspy
from pathlib import Path

from treeclip.preprocess.weiser_preprocess import (
    preprocess_one_tree_file,
    discover_tree_files,
    preprocess_plot,
)
from treeclip.preprocess.weiser_layout import WeiserLayout
from treeclip.core.tree_sample import TreeSample


def _write_synth_las(path: Path, n: int = 500, seed: int = 0):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x = rng.uniform(10, 15, n)
    las.y = rng.uniform(20, 25, n)
    las.z = rng.uniform(0, 30, n)
    las.write(str(path))


def test_preprocess_one_tree_file_returns_treesample(tmp_path):
    p = tmp_path / "tree.las"
    _write_synth_las(p, n=500)
    sample = preprocess_one_tree_file(
        laz_path=p, species="FagSyl", plot="BR01", tree_id="t001"
    )
    assert isinstance(sample, TreeSample)
    assert sample.species == "FagSyl"
    assert sample.site == "BR01"
    assert sample.tree_id == "t001"
    assert sample.point_cloud.shape == (500, 3)
    # XY should be centered at 0
    assert np.allclose(sample.point_cloud[:, :2].mean(axis=0), [0, 0], atol=1e-3)
    assert sample.lidar_density > 0


def test_discover_tree_files_filters_by_platform_and_pattern(tmp_path):
    layout = WeiserLayout(root=tmp_path)
    plot_dir = layout.plot_dir("KA10")
    # Make 3 ALS tree files + 1 ULS + 1 non-matching plot-level
    (plot_dir / "single_trees" / "ALS-on").mkdir(parents=True)
    (plot_dir / "single_trees" / "ULS-on").mkdir(parents=True)
    for sp, tid in [("FagSyl", "t1"), ("QueRob", "t2"), ("PicAbi", "t3")]:
        _write_synth_las(plot_dir / "single_trees" / "ALS-on" / f"{sp}_KA10_{tid}_2019-07-05.laz")
    _write_synth_las(plot_dir / "single_trees" / "ULS-on" / "FagSyl_KA10_t4_2019-09-12.laz")
    _write_synth_las(plot_dir / "ALS-on_KA10_2019-07-05.laz")  # plot-level, wrong pattern

    als = discover_tree_files(plot_dir, platform_filter="ALS")
    assert len(als) == 3
    assert all("ALS" in str(p).upper() for p, _ in als)

    uls = discover_tree_files(plot_dir, platform_filter="ULS")
    assert len(uls) == 1

    all_p = discover_tree_files(plot_dir, platform_filter=None)
    assert len(all_p) == 4  # 3 ALS + 1 ULS; plot-level file excluded by name pattern


def test_preprocess_plot_caches_npz(tmp_path):
    layout = WeiserLayout(root=tmp_path)
    plot_dir = layout.plot_dir("KA10")
    (plot_dir / "single_trees" / "ALS-on").mkdir(parents=True)
    for sp, tid in [("FagSyl", "t1"), ("QueRob", "t2"), ("FagSyl", "t3")]:
        _write_synth_las(plot_dir / "single_trees" / "ALS-on" / f"{sp}_KA10_{tid}_2019-07-05.laz")

    cache = tmp_path / "cache"
    n, sk = preprocess_plot(layout, cache, "KA10", platform_filter="ALS")
    assert n == 3
    assert sk == 0
    files = sorted((cache / "KA10").glob("*.npz"))
    assert len(files) == 3
    species = {TreeSample.load_npz(f).species for f in files}
    assert species == {"FagSyl", "QueRob"}
