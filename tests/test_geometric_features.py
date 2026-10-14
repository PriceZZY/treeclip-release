import numpy as np
from tests.fixtures.make_synthetic_tree import synthetic_tree
from treeclip.encoders.geometric_features import (
    extract_geometric_features,
    GeometricFeaturesEncoder,
)


def test_geometric_features_shape():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((500, 3)).astype(np.float32)
    feat = extract_geometric_features(pts, density=3.15)
    assert feat.shape == (15,)
    assert np.isfinite(feat).all()


def test_geometric_features_distinguish_tall_vs_short_tree():
    rng = np.random.default_rng(0)
    short = rng.standard_normal((500, 3)).astype(np.float32) * np.array([1, 1, 0.5])
    tall = rng.standard_normal((500, 3)).astype(np.float32) * np.array([1, 1, 10.0])
    f_short = extract_geometric_features(short, density=1.0)
    f_tall = extract_geometric_features(tall, density=1.0)
    # First feature is height
    assert f_tall[0] > f_short[0] * 5


def test_geometric_encoder_integrates_with_treesample():
    enc = GeometricFeaturesEncoder()
    sample = synthetic_tree()
    z = enc.encode(sample)
    assert z.shape == (15,)


def test_empty_pointcloud_returns_zeros():
    feat = extract_geometric_features(np.zeros((0, 3), dtype=np.float32), density=0)
    assert feat.shape == (15,)
    assert np.allclose(feat, 0)
