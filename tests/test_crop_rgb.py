import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon
from treeclip.preprocess.crop_rgb import crop_rgb_to_224


def _write_synthetic_rgb(path, h=200, w=200):
    data = np.zeros((3, h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            data[0, i, j] = i * 255 // max(h - 1, 1)
            data[1, i, j] = j * 255 // max(w - 1, 1)
            data[2, i, j] = ((i + j) * 255 // max(h + w - 2, 1)) & 0xFF
    transform = from_origin(0, h * 0.1, 0.1, 0.1)
    with rasterio.open(
        path, "w",
        driver="GTiff", height=h, width=w, count=3,
        dtype="uint8", crs="EPSG:32616", transform=transform,
    ) as dst:
        dst.write(data)


def test_crop_rgb_returns_224x224x3_uint8(tmp_path):
    raster_path = tmp_path / "rgb.tif"
    _write_synthetic_rgb(raster_path, h=200, w=200)
    poly = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    rgb = crop_rgb_to_224(raster_path, poly, padding_m=0.5)
    assert rgb.shape == (224, 224, 3)
    assert rgb.dtype == np.uint8


def test_crop_rgb_polygon_outside_raster_returns_black(tmp_path):
    raster_path = tmp_path / "rgb.tif"
    _write_synthetic_rgb(raster_path, h=50, w=50)
    poly = Polygon([(1000, 1000), (1001, 1000), (1001, 1001), (1000, 1001)])
    rgb = crop_rgb_to_224(raster_path, poly, padding_m=0.0)
    assert rgb.shape == (224, 224, 3)
    assert rgb.dtype == np.uint8
    assert (rgb == 0).mean() > 0.9
