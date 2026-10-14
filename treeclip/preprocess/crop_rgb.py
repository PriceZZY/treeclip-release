"""Crop RGB orthophoto around a crown polygon, padding then resizing to 224x224 for CLIP."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import Polygon, MultiPolygon
from PIL import Image


def crop_rgb_to_224(
    raster_path: Path | str,
    polygon: Polygon | MultiPolygon,
    padding_m: float = 0.5,
) -> np.ndarray:
    """
    Returns:
        rgb_224: (224, 224, 3) uint8. Out-of-raster regions filled with 0.
    """
    minx, miny, maxx, maxy = polygon.bounds
    minx -= padding_m
    miny -= padding_m
    maxx += padding_m
    maxy += padding_m

    with rasterio.open(raster_path) as src:
        try:
            window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
        except Exception:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        data = src.read(
            window=window,
            boundless=True,
            fill_value=0,
            out_dtype="uint8",
        )
    if data.size == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    if data.shape[0] < 3:
        pad = np.zeros((3 - data.shape[0], data.shape[1], data.shape[2]), dtype=np.uint8)
        data = np.concatenate([data, pad], axis=0)
    elif data.shape[0] > 3:
        data = data[:3]
    img = np.moveaxis(data, 0, -1)
    pil = Image.fromarray(img).resize((224, 224), Image.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)
