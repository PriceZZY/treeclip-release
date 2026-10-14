"""Extract a hyperspectral patch around a crown polygon, with crown mask."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.features import geometry_mask
from shapely.geometry import Polygon, MultiPolygon, mapping


def extract_crown_hsi(
    raster_path: Path | str, polygon: Polygon | MultiPolygon
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        patch: (H, W, B) float32 — clipped to polygon bbox, bands moved to last axis.
        mask:  (H, W) bool      — True for pixels inside the crown polygon.
    If the polygon does not intersect the raster, returns ((0,0,0), (0,0)).
    """
    with rasterio.open(raster_path) as src:
        try:
            clipped, transform = rio_mask(
                src, [mapping(polygon)], crop=True, filled=False, all_touched=False
            )
        except ValueError:
            return np.zeros((0, 0, 0), dtype=np.float32), np.zeros((0, 0), dtype=bool)

        bands, h, w = clipped.shape
        if h == 0 or w == 0:
            return np.zeros((0, 0, 0), dtype=np.float32), np.zeros((0, 0), dtype=bool)

        crown_mask = geometry_mask(
            [mapping(polygon)],
            out_shape=(h, w),
            transform=transform,
            invert=True,
        )
        patch = np.asarray(clipped.filled(0), dtype=np.float32)
        patch = np.moveaxis(patch, 0, -1)
        return patch, crown_mask.astype(bool)
