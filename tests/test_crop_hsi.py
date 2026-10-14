import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon
from treeclip.preprocess.crop_hsi import extract_crown_hsi


def _write_synthetic_hsi(path, bands=10, h=20, w=20):
    """Write a tiny HSI raster: pixel (i,j) band b == i*1000 + j*100 + b."""
    data = np.zeros((bands, h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            for b in range(bands):
                data[b, i, j] = i * 1000 + j * 100 + b
    transform = from_origin(0, h, 1.0, 1.0)
    with rasterio.open(
        path, "w",
        driver="GTiff", height=h, width=w, count=bands,
        dtype="float32", crs="EPSG:32616", transform=transform,
    ) as dst:
        dst.write(data)


def test_extract_crown_hsi_returns_patch_and_mask(tmp_path):
    raster_path = tmp_path / "synth.tif"
    _write_synthetic_hsi(raster_path, bands=5, h=20, w=20)
    poly = Polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
    patch, mask = extract_crown_hsi(raster_path, poly)
    assert patch.ndim == 3
    assert patch.shape[-1] == 5
    assert mask.shape == patch.shape[:2]
    assert mask.dtype == bool
    assert mask.any()
    assert np.isfinite(patch[mask]).all()


def test_extract_crown_hsi_polygon_outside_raster(tmp_path):
    raster_path = tmp_path / "synth.tif"
    _write_synthetic_hsi(raster_path, bands=3, h=10, w=10)
    poly = Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])
    patch, mask = extract_crown_hsi(raster_path, poly)
    assert patch.size == 0 or mask.sum() == 0
