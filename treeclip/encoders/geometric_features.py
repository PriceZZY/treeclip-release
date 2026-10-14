"""Geometric feature extractor for point clouds — the classical baseline that
Lu's abstract claims to beat. Output is a compact hand-crafted vector per tree.

Features include height, point-density proxies, and the standard "linearity /
planarity / sphericity" eigenvalue ratios at the whole-tree scale. Together with
percentile heights, these are the typical features fed to RF / SVM in forest
inventory work.
"""
from __future__ import annotations
import numpy as np

from treeclip.encoders.base import Encoder
from treeclip.core.tree_sample import TreeSample


def _eigenvalue_features(pts: np.ndarray) -> np.ndarray:
    """Return (linearity, planarity, sphericity, anisotropy) from PCA of point coords."""
    if pts.shape[0] < 3:
        return np.zeros(4, dtype=np.float32)
    cov = np.cov(pts.T)
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1]  # descending
    e1, e2, e3 = eig[0], eig[1], eig[2]
    if e1 <= 0:
        return np.zeros(4, dtype=np.float32)
    linearity = (e1 - e2) / e1
    planarity = (e2 - e3) / e1
    sphericity = e3 / e1
    anisotropy = (e1 - e3) / e1
    return np.array([linearity, planarity, sphericity, anisotropy], dtype=np.float32)


def extract_geometric_features(pts: np.ndarray, density: float) -> np.ndarray:
    """Return a 15-dim hand-crafted feature vector per tree.

    Order: height, crown_diameter, n_pts, density,           (4)
           z_pct(10,25,50,75,90),                              (5)
           linearity, planarity, sphericity, anisotropy,       (4)
           crown_x_extent, crown_y_extent                      (2)
    """
    if pts.shape[0] == 0:
        return np.zeros(15, dtype=np.float32)

    # Heights (z is preserved in our pipeline, XY are centered)
    z = pts[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    height = z_max - z_min
    z_pct = np.percentile(z, [10, 25, 50, 75, 90]).astype(np.float32)  # 5 dims

    # Crown geometry (XY spread)
    xy = pts[:, :2]
    xy_minmax = np.array([xy[:, 0].max() - xy[:, 0].min(), xy[:, 1].max() - xy[:, 1].min()], dtype=np.float32)
    crown_diameter = float(xy_minmax.max())

    # Point count + density
    n_pts = float(pts.shape[0])

    # Eigenvalue features (4 dims)
    eig = _eigenvalue_features(pts)

    return np.concatenate([
        np.array([height, crown_diameter, n_pts, density], dtype=np.float32),
        z_pct,                                # 5
        eig,                                  # 4
        np.array([xy_minmax[0], xy_minmax[1]], dtype=np.float32),  # 2
    ])


class GeometricFeaturesEncoder(Encoder):
    """Drop-in Encoder that emits hand-crafted geometric features (for RF/SVM baselines)."""
    name = "geometric"
    embedding_dim = 15

    def __init__(self, device: str = "cpu"):
        self.device = device  # ignored — pure numpy

    def encode(self, sample: TreeSample):
        import torch
        feat = extract_geometric_features(sample.point_cloud, sample.lidar_density)
        return torch.from_numpy(feat)
