"""Crop a point cloud to a crown polygon, then center it in XY. Z is preserved."""
from __future__ import annotations
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.prepared import prep


def crop_and_center(
    points_xyz: np.ndarray, polygon: Polygon | MultiPolygon
) -> tuple[np.ndarray, float]:
    """
    Args:
        points_xyz: (N, 3) float array in the same CRS as polygon.
        polygon: shapely polygon defining the crown.

    Returns:
        (cropped_xyz, density)
        cropped_xyz: (M, 3) float32, XY centered at cropped centroid; Z untouched.
        density: M / polygon.area (pts per square unit; meaningful in metric CRS).
    """
    if points_xyz.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), 0.0

    prepared = prep(polygon)
    mask = np.fromiter(
        (prepared.contains(Point(x, y)) for x, y in points_xyz[:, :2]),
        dtype=bool,
        count=points_xyz.shape[0],
    )
    cropped = points_xyz[mask]
    area = float(polygon.area)
    if cropped.shape[0] == 0 or area == 0.0:
        return cropped.astype(np.float32), 0.0

    centroid_xy = cropped[:, :2].mean(axis=0)
    centered = cropped.copy().astype(np.float32)
    centered[:, :2] -= centroid_xy.astype(np.float32)
    density = cropped.shape[0] / area
    return centered, float(density)
