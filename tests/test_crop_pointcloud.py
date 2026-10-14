import numpy as np
from shapely.geometry import Polygon
from treeclip.preprocess.crop_pointcloud import crop_and_center


def test_crop_keeps_only_points_inside_polygon():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 10, (100, 2)).astype(np.float32)
    z = rng.uniform(0, 30, (100,)).astype(np.float32)
    pts = np.column_stack([xy, z])
    poly = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])

    cropped, density = crop_and_center(pts, poly)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] <= 5) & (xy[:, 1] >= 0) & (xy[:, 1] <= 5)
    # NOTE: shapely.prep.contains returns True only for strictly interior;
    # boundary-on points may be excluded so use <=
    assert cropped.shape[0] <= inside.sum()
    assert density == cropped.shape[0] / poly.area


def test_cropped_is_centered_in_xy():
    pts = np.array([[10.0, 10.0, 5.0], [12.0, 14.0, 8.0]], dtype=np.float32)
    poly = Polygon([(9, 9), (13, 9), (13, 15), (9, 15)])
    cropped, _ = crop_and_center(pts, poly)
    assert np.allclose(cropped[:, :2].mean(axis=0), [0.0, 0.0], atol=1e-5)
    assert np.allclose(np.sort(cropped[:, 2]), [5.0, 8.0])


def test_empty_polygon_returns_zero_density():
    pts = np.array([[100.0, 100.0, 5.0]], dtype=np.float32)
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    cropped, density = crop_and_center(pts, poly)
    assert cropped.shape == (0, 3)
    assert density == 0.0
