"""Windowed feather-weight mosaic of single-band Float32 NDVI GeoTIFFs.

The parcel map only fixes the output extent and pixel grid.  Every scene is
validated against the same grid-compatibility rules used by the coverage
analysis (same CRS and resolution, origins offset by an integral number of
pixels).  A pixel of a scene is *invalid* when it lies outside that scene's
footprint, carries its NoData value, is non-finite, or is flagged invalid by
the dataset's valid-data mask.

Per scene, the weight of a valid pixel is the Euclidean distance (in pixels)
from the pixel centre to the nearest invalid pixel centre, capped at the
feather width.  Where several scenes carry valid observations of the same
output pixel they are merged by a weighted mean; outputs without any valid
observation are written as NoData.

All GeoTIFF I/O is windowed.  The distance transform is evaluated on tiles
padded by a halo of ``feather_width`` pixels: because the weight is capped at
the feather width, the halo result is bit-for-bit identical to the transform
of the whole scene, so neither the tiling layout nor the output boundaries can
change a weight.
"""

from __future__ import annotations

import math
from contextlib import ExitStack
from typing import Sequence

import numpy as np
import rasterio
from rasterio.windows import Window

try:  # scipy ships an exact, compiled Euclidean distance transform.
    from scipy.ndimage import distance_transform_edt as _scipy_edt
except ImportError:  # pragma: no cover - exercised only without scipy
    _scipy_edt = None

from app.analysis import (
    IncompatibleFile,
    RasterMeta,
    UploadedRaster,
    _display_crs,
    _fmt_num,
    _inspect,
    _pixel_offset,
    _same_resolution,
    _split_window,
)

# Output tiles are at most this many pixels per side.  Memory use is bounded
# by tile_size**2 times the number of open scenes; weights never depend on it.
_DEFAULT_TILE_SIZE = 512

# Minimum/maximum feather width accepted by the API.
MIN_FEATHER = 1
MAX_FEATHER = 64

# NoData written into the output GeoTIFF.  Float32 NaN round-trips through
# GDAL and is the conventional NDVI fill value.
OUTPUT_NODATA = np.float32(np.nan)


def validate_mosaic_uploads(
    parcel: UploadedRaster, scenes: Sequence[UploadedRaster]
) -> tuple[RasterMeta | None, list[RasterMeta], list[IncompatibleFile]]:
    """Validate mosaic inputs against the parcel-map reference grid.

    Mirrors :func:`app.analysis.validate_uploads` and additionally requires
    every scene to be a single-band Float32 raster.
    """

    errors: list[IncompatibleFile] = []
    parcel_meta, parcel_error = _inspect(parcel)
    if parcel_error is not None:
        errors.append(parcel_error)

    reference = parcel_meta if parcel_error is None else None

    scene_metas: list[RasterMeta] = []
    for scene in scenes:
        meta, own_error = _inspect(scene)
        if own_error is not None:
            errors.append(own_error)
        if meta is None:
            continue

        local_reasons: list[str] = []
        if meta.dtype != "float32":
            local_reasons.append(
                f"NDVI 景必须为 Float32 单波段，实际数据类型为 {meta.dtype}"
            )

        if reference is not None:
            if meta.crs is None or reference.crs is None:
                local_reasons.append("坐标系缺失")
            elif meta.crs != reference.crs:
                local_reasons.append(
                    "坐标系与地块图不一致"
                    f"（{_display_crs(meta.crs)} vs 地块图 "
                    f"{_display_crs(reference.crs)}）"
                )

            if not _same_resolution(
                meta.transform.a, reference.transform.a
            ) or not _same_resolution(meta.transform.e, reference.transform.e):
                local_reasons.append(
                    "分辨率与地块图不一致"
                    f"（{_fmt_num(abs(meta.transform.a))}"
                    f"×{_fmt_num(abs(meta.transform.e))} vs "
                    f"{_fmt_num(abs(reference.transform.a))}"
                    f"×{_fmt_num(abs(reference.transform.e))}）"
                )
            else:
                # Origins may differ, but only by an integral number of
                # pixels so the pixel grids stay coincident.
                ox = _pixel_offset(
                    reference.transform.c - meta.transform.c, reference.transform.a
                )
                oy = _pixel_offset(
                    reference.transform.f - meta.transform.f, reference.transform.e
                )
                if ox is None or oy is None:
                    qx = (reference.transform.c - meta.transform.c) / reference.transform.a
                    qy = (reference.transform.f - meta.transform.f) / reference.transform.e
                    local_reasons.append(
                        "原点与地块图未对齐到整数像元"
                        f"（偏移 {qx:.6f}, {qy:.6f} 像元）"
                    )

        if local_reasons and own_error is None:
            errors.append(
                IncompatibleFile(
                    filename=scene.filename,
                    role=scene.role,
                    reason="；".join(local_reasons),
                )
            )

        scene_metas.append(meta)

    if errors:
        return None, [], errors
    return parcel_meta, scene_metas, []


def _edt_1d_f(f: np.ndarray) -> np.ndarray:
    """1-D squared-distance transform for arbitrary site costs ``f``.

    Implements the lower-envelope algorithm of Felzenszwalb & Huttenlocher
    (2012), §3: given per-site costs ``f`` (0 at background sites, infinity
    elsewhere), returns ``d[i] = min_v ((i-v)**2 + f[v])``.
    """

    n = f.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    # A line without any finite site has no background point: every answer
    # stays infinite.  Guarding up front also avoids inf - inf = NaN in the
    # parabola intersections below.
    if not np.isfinite(f).any():
        return f.astype(np.float64, copy=True)

    f = f.astype(np.float64, copy=False)
    # Parabola vertices v, envelope boundaries z; q is the current vertex.
    vertices = np.empty(n, dtype=np.int64)
    boundaries = np.empty(n + 1, dtype=np.float64)
    # Seed with the first finite site; infinity-only lines are handled above.
    first = int(np.argmax(np.isfinite(f)))
    vertices[0] = first
    boundaries[0] = -np.inf
    boundaries[1] = np.inf
    count = 0  # index of the most recent envelope vertex

    for i in range(first + 1, n):
        if not math.isfinite(f[i]):
            # Foreground sites carry no parabola; they are evaluated later.
            continue
        # Intersection abscissa of parabolas centred at i and vertices[count].
        s = ((f[i] + i * i) - (f[vertices[count]] + vertices[count] ** 2)) / (
            2.0 * (i - vertices[count])
        )
        while count > 0 and s <= boundaries[count]:
            count -= 1
            s = ((f[i] + i * i) - (f[vertices[count]] + vertices[count] ** 2)) / (
                2.0 * (i - vertices[count])
            )
        count += 1
        vertices[count] = i
        boundaries[count] = s
        boundaries[count + 1] = np.inf

    distances = np.empty(n, dtype=np.float64)
    envelope_index = 0
    for i in range(n):
        while boundaries[envelope_index + 1] < i:
            envelope_index += 1
        v = vertices[envelope_index]
        distances[i] = (i - v) ** 2 + f[v]
    return distances


def _edt_1d(binary: np.ndarray) -> np.ndarray:
    """Squared distance to the nearest zero along a 1-D boolean line."""

    # Background (invalid) sites cost 0; foreground starts at infinity.
    f = np.where(binary.astype(bool, copy=False), np.inf, 0.0).astype(np.float64)
    return _edt_1d_f(f)


def _edt_numpy(valid: np.ndarray) -> np.ndarray:
    """Reference exact EDT built only from NumPy (used when scipy is absent).

    Separable application of the lower-envelope 1-D transform; kept as a
    dependency-free fallback and as an executable specification of the
    distance definition.
    """

    # First pass along rows: binary sites -> squared distances within rows.
    row_pass = np.apply_along_axis(_edt_1d, axis=1, arr=valid)
    # Second pass along columns, starting from the row-wise squared costs.
    squared = np.apply_along_axis(_edt_1d_f, axis=0, arr=row_pass)
    # Squared distances are exact integers; sqrt restores pixel units.
    return np.sqrt(squared, out=squared)


def euclidean_distance_to_invalid(valid: np.ndarray) -> np.ndarray:
    """Exact Euclidean distance from every pixel to the nearest invalid pixel.

    ``valid`` is a boolean array (True = valid observation).  Invalid pixels
    get distance zero; valid pixels get the Euclidean distance in pixel units
    to the nearest invalid pixel centre; an all-valid array returns infinity.
    Uses scipy's compiled transform when available and the pure-NumPy
    separable transform otherwise; both return identical values.
    """

    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("validity mask must be 2-dimensional")
    if valid.size == 0:
        return np.empty(valid.shape, dtype=np.float64)
    # With no invalid site there is no nearest invalid pixel: the distance is
    # infinite.  Normalise both backends to this semantics (scipy otherwise
    # reports the distance to the array's outer boundary).
    if valid.all():
        return np.full(valid.shape, np.inf, dtype=np.float64)
    if _scipy_edt is not None:
        # Invalid pixels are the background (False) and get distance zero.
        return _scipy_edt(valid)
    return _edt_numpy(valid)


def _scene_offsets(
    parcel_ds: rasterio.DatasetReader, scene_ds: rasterio.DatasetReader
) -> tuple[int, int]:
    """Pixel offset (col, row) to add to parcel indices to reach scene indices."""

    offset_x = int(
        round((parcel_ds.transform.c - scene_ds.transform.c) / parcel_ds.transform.a)
    )
    offset_y = int(
        round((parcel_ds.transform.f - scene_ds.transform.f) / parcel_ds.transform.e)
    )
    return offset_x, offset_y


def _read_scene_tile(
    scene_ds: rasterio.DatasetReader,
    scene_nodata: float | int | None,
    global_window: Window,
    offset_x: int,
    offset_y: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one parcel tile mapped into a scene, padded with invalid pixels.

    Returns ``(values, valid)`` at the parcel tile's shape.  Pixels outside
    the scene footprint are present as zero fill with ``valid == False``.
    """

    tile_width = int(global_window.width)
    tile_height = int(global_window.height)
    scene_col = int(global_window.col_off) + offset_x
    scene_row = int(global_window.row_off) + offset_y

    src_col = max(0, scene_col)
    src_row = max(0, scene_row)
    end_col = min(scene_ds.width, scene_col + tile_width)
    end_row = min(scene_ds.height, scene_row + tile_height)

    values = np.zeros((tile_height, tile_width), dtype=np.float32)
    valid = np.zeros((tile_height, tile_width), dtype=bool)

    if end_col <= src_col or end_row <= src_row:
        return values, valid

    read_window = Window(
        src_col, src_row, end_col - src_col, end_row - src_row
    )
    data = scene_ds.read(1, window=read_window).astype(np.float32, copy=False)
    # The GDAL valid-data mask flags nodata and any mask-band invalid pixels.
    # Read it through the same clipped, in-bounds window as the data so the
    # two arrays always share a shape regardless of driver edge handling.
    mask = scene_ds.read_masks(
        1, window=read_window
    ).astype(bool, copy=False)

    dest_col = src_col - scene_col
    dest_row = src_row - scene_row
    dest = (
        slice(dest_row, dest_row + data.shape[0]),
        slice(dest_col, dest_col + data.shape[1]),
    )
    values[dest] = data
    tile_valid = np.isfinite(data) & mask
    # Explicit nodata comparison as a belt-and-braces check: some drivers
    # expose nodata without an internal mask band.
    if scene_nodata is not None and np.isfinite(float(scene_nodata)):
        tile_valid &= data != np.float32(scene_nodata)
    valid[dest] = tile_valid
    return values, valid


def mosaic_scenes(
    parcel_meta: RasterMeta,
    scene_metas: Sequence[RasterMeta],
    output_path: str,
    feather_width: int,
    tile_size: int = _DEFAULT_TILE_SIZE,
) -> None:
    """Build the feathered mosaic and write a Float32 GeoTIFF.

    The output grid (extent, dimensions, CRS, transform) is taken from the
    parcel map.  Raises :class:`rasterio.RasterioIOError` (or a related
    rasterio error) if a source file cannot be read; callers surface the
    responsible file name.
    """

    if not (MIN_FEATHER <= int(feather_width) <= MAX_FEATHER):
        raise ValueError(
            f"feather_width must be within [{MIN_FEATHER}, {MAX_FEATHER}]"
        )
    feather = int(feather_width)
    tile_size = max(1, int(tile_size))

    # GDAL requires tiled raster block sizes to be multiples of 16; tiny
    # rasters are written as a single stripped band instead.
    if parcel_meta.width >= 16 and parcel_meta.height >= 16:
        block = max(16, min(tile_size, parcel_meta.width, parcel_meta.height))
        block = (block // 16) * 16
        tiling = {
            "tiled": True,
            "blockxsize": block,
            "blockysize": block,
        }
    else:
        tiling = {"tiled": False}

    profile = {
        "driver": "GTiff",
        "width": parcel_meta.width,
        "height": parcel_meta.height,
        "count": 1,
        "dtype": "float32",
        "crs": parcel_meta.crs,
        "transform": parcel_meta.transform,
        "nodata": OUTPUT_NODATA,
        "compress": "deflate",
        **tiling,
    }

    with ExitStack() as stack:
        parcel_ds = stack.enter_context(
            rasterio.open(parcel_meta.upload.path)
        )
        scene_datasets = [
            stack.enter_context(rasterio.open(meta.upload.path))
            for meta in scene_metas
        ]
        offsets = [
            _scene_offsets(parcel_ds, ds) for ds in scene_datasets
        ]
        destination = stack.enter_context(
            rasterio.open(output_path, "w", **profile)
        )

        full_window = Window(0, 0, parcel_meta.width, parcel_meta.height)
        for tile in _split_window(full_window, tile_size):
            acc_values = np.zeros(
                (tile.height, tile.width), dtype=np.float64
            )
            acc_weights = np.zeros(
                (tile.height, tile.width), dtype=np.float64
            )

            for scene_ds, scene_meta, (offset_x, offset_y) in zip(
                scene_datasets, scene_metas, offsets
            ):
                # Pad the read by exactly one feather width on every side.
                # A pixel's capped weight only depends on invalid pixels no
                # farther than ``feather`` away, so this halo always contains
                # every invalid centre that can influence the tile.
                halo_window = Window(
                    int(tile.col_off) - feather,
                    int(tile.row_off) - feather,
                    int(tile.width) + 2 * feather,
                    int(tile.height) + 2 * feather,
                )
                _, halo_valid = _read_scene_tile(
                    scene_ds,
                    scene_meta.nodata,
                    halo_window,
                    offset_x,
                    offset_y,
                )
                distances = euclidean_distance_to_invalid(halo_valid)
                # Capping turns an (unknown) distance beyond the halo into
                # the feather width; inside the halo the transform is exact.
                halo_weights = np.minimum(distances, float(feather))
                # Drop the halo; tile-local weights are independent of tiling.
                weights = halo_weights[
                    feather : feather + int(tile.height),
                    feather : feather + int(tile.width),
                ]

                values, valid = _read_scene_tile(
                    scene_ds,
                    scene_meta.nodata,
                    tile,
                    offset_x,
                    offset_y,
                )
                contribute = valid & (weights > 0.0)
                if not contribute.any():
                    continue
                acc_values[contribute] += (
                    values[contribute].astype(np.float64, copy=False)
                    * weights[contribute]
                )
                acc_weights[contribute] += weights[contribute]

            output = np.full(
                (tile.height, tile.width), OUTPUT_NODATA, dtype=np.float32
            )
            covered = acc_weights > 0.0
            if covered.any():
                blended = (acc_values[covered] / acc_weights[covered]).astype(
                    np.float32, copy=False
                )
                output[covered] = blended
            destination.write(output, 1, window=tile)
