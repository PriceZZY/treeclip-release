import numpy as np
from treeclip.encoders.render_views import render_pca_depth_views


def test_renders_two_views_with_correct_shape():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((500, 3)).astype(np.float32)
    views = render_pca_depth_views(pts, image_size=224)
    assert views.shape == (2, 224, 224)
    assert views.dtype == np.uint8


def test_views_are_nonempty_on_dense_input():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((1024, 3)).astype(np.float32) * 2.0
    views = render_pca_depth_views(pts, image_size=224)
    for i in range(2):
        assert (views[i] > 0).mean() > 0.01


def test_empty_pointcloud_returns_zero_images():
    pts = np.zeros((0, 3), dtype=np.float32)
    views = render_pca_depth_views(pts, image_size=64)
    assert views.shape == (2, 64, 64)
    assert views.sum() == 0
