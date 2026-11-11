#!/usr/bin/env python3
"""Generate the demonstration GeoTIFFs in samples/.

Run from the repository root:

    python scripts/generate_samples.py

The two mask scenes have different, non-zero tile origins.  Their grids are
offset by integral pixels from the parcel map and overlap each other, which is
exactly the condition the service must handle without resampling.

Two Float32 NDVI scenes with the same staggered, overlapping footprints are
also produced for the feathered mosaic endpoint.  They deliberately contain
every kind of invalid pixel the API must reject: outside-footprint positions,
the dataset NoData sentinel, NaN, an infinite value, and an internal
valid-data mask flag.  In the overlap the two scenes carry different NDVI
gradients so the weighted blend is visible.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

OUT = Path(__file__).resolve().parent.parent / "samples"

PIXEL_SIZE = 10.0
PARCEL_ORIGIN_X = 500_000.0
PARCEL_ORIGIN_Y = 3_000_000.0
WIDTH = 120
HEIGHT = 100

# (scene origin in parcels' pixel space, width, height)
SCENE_A = ((-7, -5), 100, 90)
SCENE_B = ((15, 10), 95, 85)

# Sentinel NoData for the Float32 NDVI demonstration scenes.
NDVI_NODATA = np.float32(-9999.0)


def write_raster(path: Path, data: np.ndarray, origin, nodata=0, crs="EPSG:32650"):
    height, width = data.shape
    transform = from_origin(
        origin[0], origin[1], xsize=PIXEL_SIZE, ysize=PIXEL_SIZE
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        tiled=True,
        blockxsize=64,
        blockysize=64,
        compress="deflate",
    ) as dst:
        dst.write(data, 1)


def global_grid(offset, width, height):
    """Coordinates in the parcel map's pixel index space."""
    rows = np.arange(offset[1], offset[1] + height)[:, None]
    cols = np.arange(offset[0], offset[0] + width)[None, :]
    return cols, rows


def write_ndvi_scene(path: Path, data, origin, invalid_mask):
    """Write a Float32 NDVI scene with an internal valid-data mask."""
    height, width = data.shape
    transform = from_origin(
        origin[0], origin[1], xsize=PIXEL_SIZE, ysize=PIXEL_SIZE
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs="EPSG:32650",
        transform=transform,
        nodata=NDVI_NODATA,
        tiled=True,
        blockxsize=64,
        blockysize=64,
        compress="deflate",
    ) as dst:
        dst.write(data.astype(np.float32, copy=False), 1)
        # GDAL mask band: 255 = valid, 0 = explicitly invalid.  NaN and the
        # nodata sentinel are additionally rejected by the service itself.
        dst.write_mask(np.where(invalid_mask, 0, 255).astype(np.uint8))


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    parcels = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
    parcels[0:40, 0:50] = 101
    parcels[0:40, 55:110] = 102
    parcels[45:100, 0:50] = 103
    parcels[45:100, 55:110] = 104
    write_raster(
        OUT / "parcel_map.tif",
        parcels,
        (PARCEL_ORIGIN_X, PARCEL_ORIGIN_Y),
        nodata=0,
    )

    offset, width, height = SCENE_A
    cols, rows = global_grid(offset, width, height)
    mask_a = ((cols < 70) & (rows < 55)).astype(np.uint8)
    write_raster(
        OUT / "scene_01_cloud_free.tif",
        mask_a,
        (
            PARCEL_ORIGIN_X + offset[0] * PIXEL_SIZE,
            PARCEL_ORIGIN_Y - offset[1] * PIXEL_SIZE,
        ),
        nodata=0,
    )

    offset, width, height = SCENE_B
    cols, rows = global_grid(offset, width, height)
    mask_b = ((cols >= 50) & (rows >= 40)).astype(np.uint8)
    write_raster(
        OUT / "scene_02_cloud_free.tif",
        mask_b,
        (
            PARCEL_ORIGIN_X + offset[0] * PIXEL_SIZE,
            PARCEL_ORIGIN_Y - offset[1] * PIXEL_SIZE,
        ),
        nodata=0,
    )

    # --- Float32 NDVI scenes for POST /mosaic -------------------------------
    offset_a, width_a, height_a = SCENE_A
    cols_a, rows_a = global_grid(offset_a, width_a, height_a)
    # Smooth, distinct NDVI gradients per scene so feathered blending shows.
    ndvi_a = (
        0.30
        + 0.0040 * (cols_a - offset_a[0])
        - 0.0020 * (rows_a - offset_a[1])
    ).astype(np.float32)
    invalid_a = np.zeros((height_a, width_a), dtype=bool)
    # Dataset nodata sentinel in the top-left corner.
    ndvi_a[2:6, 2:6] = NDVI_NODATA
    invalid_a[2:6, 2:6] = True
    # Explicit NaN block.
    ndvi_a[40:44, 30:36] = np.nan
    invalid_a[40:44, 30:36] = True
    # A non-finite (negative infinity) pixel.
    ndvi_a[60, 70] = -np.inf
    invalid_a[60, 70] = True
    # Internal valid-data mask flags a block even though its values are finite.
    invalid_a[15:19, 80:86] = True
    write_ndvi_scene(
        OUT / "scene_01_ndvi.tif",
        ndvi_a,
        (
            PARCEL_ORIGIN_X + offset_a[0] * PIXEL_SIZE,
            PARCEL_ORIGIN_Y - offset_a[1] * PIXEL_SIZE,
        ),
        invalid_a,
    )

    offset_b, width_b, height_b = SCENE_B
    cols_b, rows_b = global_grid(offset_b, width_b, height_b)
    ndvi_b = (
        0.60
        - 0.0030 * (cols_b - offset_b[0])
        + 0.0025 * (rows_b - offset_b[1])
    ).astype(np.float32)
    invalid_b = np.zeros((height_b, width_b), dtype=bool)
    # Nodata sentinel near the bottom-right corner.
    ndvi_b[height_b - 6 : height_b - 2, width_b - 6 : width_b - 2] = NDVI_NODATA
    invalid_b[height_b - 6 : height_b - 2, width_b - 6 : width_b - 2] = True
    # NaN column strip.
    ndvi_b[20:30, 10] = np.nan
    invalid_b[20:30, 10] = True
    # Positive infinity pixel.
    ndvi_b[55, 40] = np.inf
    invalid_b[55, 40] = True
    # Mask-only invalid block.
    invalid_b[60:64, 5:12] = True
    write_ndvi_scene(
        OUT / "scene_02_ndvi.tif",
        ndvi_b,
        (
            PARCEL_ORIGIN_X + offset_b[0] * PIXEL_SIZE,
            PARCEL_ORIGIN_Y - offset_b[1] * PIXEL_SIZE,
        ),
        invalid_b,
    )

    print("Generated:")
    for path in sorted(OUT.glob("*.tif")):
        print(f"  {path.relative_to(OUT.parent)} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
