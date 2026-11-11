"""End-to-end tests for the feathered mosaic endpoint.  Run with pytest."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin
from scipy import ndimage

from app.main import app

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _write_float(
    path: Path,
    values: np.ndarray,
    transform,
    nodata: float | None = -9999.0,
    mask: np.ndarray | None = None,
) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:32650",
        transform=transform,
        nodata=nodata,
        tiled=True,
        blockxsize=16,
        blockysize=16,
    ) as dst:
        dst.write(values.astype(np.float32, copy=False), 1)
        if mask is not None:
            dst.write_mask(mask.astype(np.uint8))


def _write_int(path: Path, values: np.ndarray, transform) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype=values.dtype,
        crs="EPSG:32650",
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(values, 1)


def _post_mosaic(
    client: TestClient,
    parcel: Path,
    scenes: list[Path],
    feather: int | str,
):
    files = [("parcel", (parcel.name, parcel.open("rb"), "image/tiff"))]
    files += [("scenes", (s.name, s.open("rb"), "image/tiff")) for s in scenes]
    try:
        return client.post(
            "/mosaic",
            files=files,
            data={"feather_width": str(feather)},
        )
    finally:
        for _, (_, handle, _) in files:
            handle.close()


def test_sample_mosaic_downloads_valid_geotiff(client: TestClient):
    response = _post_mosaic(
        client,
        SAMPLES / "parcel_map.tif",
        [SAMPLES / "scene_01_ndvi.tif", SAMPLES / "scene_02_ndvi.tif"],
        8,
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/tiff"
    assert "ndvi_mosaic.tif" in response.headers["content-disposition"]

    with rasterio.open(io.BytesIO(response.content)) as ds:
        assert ds.count == 1
        assert ds.dtypes[0] == "float32"
        assert ds.width == 120 and ds.height == 100
        with rasterio.open(SAMPLES / "parcel_map.tif") as ref:
            assert ds.transform.almost_equals(ref.transform)
            assert ds.crs == ref.crs
        assert np.isnan(ds.nodata)
        values = ds.read(1)
    # Some pixels are covered, some are outside every scene -> NoData.
    assert np.isfinite(values).any()
    assert np.isnan(values).any()
    # NDVI must stay within the physical range used by the sample scenes.
    finite = values[np.isfinite(values)]
    assert finite.min() >= -1.0 and finite.max() <= 1.0


def test_mosaic_matches_independent_weighted_mean(client: TestClient, tmp_path: Path):
    transform = from_origin(500_000.0, 3_000_000.0, 10, 10)
    parcel = np.ones((20, 20), dtype=np.uint16)
    parcel_path = tmp_path / "parcel.tif"
    _write_int(parcel_path, parcel, transform)

    # Scene A covers parcel cols 0:14, rows 0:14, with one invalid corner.
    a = np.full((14, 14), 0.40, dtype=np.float32)
    a[0, 0] = -9999.0
    path_a = tmp_path / "a.tif"
    _write_float(path_a, a, transform, nodata=-9999.0)

    # Scene B is shifted +6 cols/+6 rows, fully valid constant 0.80.
    b_transform = from_origin(500_000.0 + 6 * 10, 3_000_000.0 - 6 * 10, 10, 10)
    b = np.full((14, 14), 0.80, dtype=np.float32)
    path_b = tmp_path / "b.tif"
    _write_float(path_b, b, b_transform, nodata=-9999.0)

    feather = 3
    response = _post_mosaic(client, parcel_path, [path_a, path_b], feather)
    assert response.status_code == 200
    with rasterio.open(io.BytesIO(response.content)) as ds:
        got = ds.read(1)

    # Independent whole-raster reference implementation.
    valid_a = np.zeros((20, 20), dtype=bool)
    va = np.zeros((20, 20), dtype=np.float32)
    valid_a[0:14, 0:14] = True
    va[0:14, 0:14] = a
    valid_a[0, 0] = False
    valid_b = np.zeros((20, 20), dtype=bool)
    vb = np.zeros((20, 20), dtype=np.float32)
    valid_b[6:20, 6:20] = True
    vb[6:20, 6:20] = b

    wa = np.minimum(ndimage.distance_transform_edt(valid_a), feather)
    wb = np.minimum(ndimage.distance_transform_edt(valid_b), feather)
    weight_sum = wa + wb
    expected = np.full((20, 20), np.nan, dtype=np.float32)
    covered = weight_sum > 0
    expected[covered] = (
        (va * wa + vb * wb)[covered] / weight_sum[covered]
    ).astype(np.float32)

    assert np.array_equal(np.isfinite(got), covered)
    np.testing.assert_array_equal(got[covered], expected[covered])
    assert np.isnan(got[~covered]).all()

    # Pixel (10, 10) sits at least 3 px from every invalid centre of either
    # scene, so both weights hit the feather cap and the blend is the mean.
    assert got[10, 10] == pytest.approx(0.60, abs=1e-7)


def test_invalid_pixel_kinds_are_rejected(client: TestClient, tmp_path: Path):
    transform = from_origin(500_000.0, 3_000_000.0, 10, 10)
    parcel = np.ones((10, 10), dtype=np.uint16)
    parcel_path = tmp_path / "parcel.tif"
    _write_int(parcel_path, parcel, transform)

    scene = np.full((10, 10), 0.5, dtype=np.float32)
    scene[0, 0] = -9999.0  # dataset nodata
    scene[1, 1] = np.nan  # non-finite
    scene[2, 2] = np.inf  # non-finite
    mask = np.full((10, 10), 255, dtype=np.uint8)
    mask[3, 3] = 0  # valid-data mask flags invalid
    scene_path = tmp_path / "scene.tif"
    _write_float(scene_path, scene, transform, nodata=-9999.0, mask=mask)

    response = _post_mosaic(client, parcel_path, [scene_path], 2)
    assert response.status_code == 200
    with rasterio.open(io.BytesIO(response.content)) as ds:
        values = ds.read(1)
    for r, c in [(0, 0), (1, 1), (2, 2), (3, 3)]:
        assert np.isnan(values[r, c])
    # An untouched pixel remains a valid observation.
    assert values[8, 8] == pytest.approx(0.5)


def test_weight_is_distance_to_nearest_invalid(client: TestClient, tmp_path: Path):
    transform = from_origin(500_000.0, 3_000_000.0, 10, 10)
    parcel = np.ones((7, 7), dtype=np.uint16)
    parcel_path = tmp_path / "parcel.tif"
    _write_int(parcel_path, parcel, transform)

    # Only the centre pixel is invalid; every other pixel is valid.
    scene = np.full((7, 7), 0.7, dtype=np.float32)
    scene[3, 3] = -9999.0
    scene_path = tmp_path / "scene.tif"
    _write_float(scene_path, scene, transform, nodata=-9999.0)

    feather = 64  # larger than any distance, so the cap never binds
    response = _post_mosaic(client, parcel_path, [scene_path], feather)
    assert response.status_code == 200
    with rasterio.open(io.BytesIO(response.content)) as ds:
        values = ds.read(1)
    assert np.isnan(values[3, 3])
    # With one scene and constant values, weighting cannot change the value;
    # invalid stays NoData and every valid centre keeps its observation.
    np.testing.assert_array_equal(
        values[np.isfinite(values)],
        np.full(int(np.isfinite(values).sum()), np.float32(0.7)),
    )


def test_feather_width_changes_overlap_blend(client: TestClient, tmp_path: Path):
    transform = from_origin(500_000.0, 3_000_000.0, 10, 10)
    parcel = np.ones((10, 40), dtype=np.uint16)
    parcel_path = tmp_path / "parcel.tif"
    _write_int(parcel_path, parcel, transform)

    # Two overlapping scenes, each covering the full output but with a
    # different invalid column on opposite sides.
    a = np.full((10, 40), 0.20, dtype=np.float32)
    a[:, 0] = -9999.0  # invalid left column -> weights grow rightward
    path_a = tmp_path / "a.tif"
    _write_float(path_a, a, transform, nodata=-9999.0)

    b = np.full((10, 40), 0.80, dtype=np.float32)
    b[:, 39] = -9999.0  # invalid right column -> weights grow leftward
    path_b = tmp_path / "b.tif"
    _write_float(path_b, b, transform, nodata=-9999.0)

    def run(feather: int) -> np.ndarray:
        response = _post_mosaic(client, parcel_path, [path_a, path_b], feather)
        assert response.status_code == 200
        with rasterio.open(io.BytesIO(response.content)) as ds:
            return ds.read(1)

    narrow = run(1)
    wide = run(64)
    # The two feather widths shape the weights differently, so the blends
    # cannot be identical across the band.
    assert not np.allclose(narrow, wide)
    # At column 0, A is invalid there, so only B contributes in both runs.
    assert np.allclose(narrow[:, 0], 0.80)
    assert np.allclose(wide[:, 0], 0.80)
    # In the central columns the invalid columns are more than 64 px away,
    # so both weights hit the cap and the blend is the unweighted mean.
    assert np.allclose(wide[:, 18:22], 0.50, atol=1e-6)


def test_nan_nodata_scene(client: TestClient, tmp_path: Path):
    transform = from_origin(500_000.0, 3_000_000.0, 10, 10)
    parcel = np.ones((6, 6), dtype=np.uint16)
    parcel_path = tmp_path / "parcel.tif"
    _write_int(parcel_path, parcel, transform)

    scene = np.full((6, 6), 0.42, dtype=np.float32)
    scene[2, 2] = np.nan  # NaN used as the dataset nodata value
    scene_path = tmp_path / "nan_nodata.tif"
    with rasterio.open(
        scene_path,
        "w",
        driver="GTiff",
        width=6,
        height=6,
        count=1,
        dtype="float32",
        crs="EPSG:32650",
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(scene, 1)

    response = _post_mosaic(client, parcel_path, [scene_path], 4)
    assert response.status_code == 200
    with rasterio.open(io.BytesIO(response.content)) as ds:
        values = ds.read(1)
    assert np.isnan(values[2, 2])
    assert values[0, 0] == pytest.approx(0.42)


def test_feather_width_bounds(client: TestClient, tmp_path: Path):
    transform = from_origin(500_000.0, 3_000_000.0, 10, 10)
    parcel_path = tmp_path / "parcel.tif"
    _write_int(parcel_path, np.ones((4, 4), np.uint16), transform)
    scene_path = tmp_path / "scene.tif"
    _write_float(scene_path, np.ones((4, 4), np.float32), transform)

    for bad in (0, 65, -1, "abc"):
        response = _post_mosaic(client, parcel_path, [scene_path], bad)
        assert response.status_code == 400
        assert "feather_width" in response.json()["detail"]

    for good in (1, 64):
        response = _post_mosaic(client, parcel_path, [scene_path], good)
        assert response.status_code == 200


def test_non_float32_scene_reports_file(client: TestClient, tmp_path: Path):
    transform = from_origin(500_000.0, 3_000_000.0, 10, 10)
    parcel_path = tmp_path / "parcel.tif"
    _write_int(parcel_path, np.ones((6, 6), np.uint16), transform)
    int_scene = tmp_path / "int_scene.tif"
    _write_int(int_scene, np.ones((6, 6), np.uint16), transform)

    response = _post_mosaic(client, parcel_path, [int_scene], 4)
    assert response.status_code == 400
    errors = response.json()["detail"]["incompatible_files"]
    assert errors[0]["filename"] == "int_scene.tif"
    assert "Float32" in errors[0]["reason"]


def test_misaligned_origin_reports_file(client: TestClient, tmp_path: Path):
    transform = from_origin(500_000.0, 3_000_000.0, 10, 10)
    parcel_path = tmp_path / "parcel.tif"
    _write_int(parcel_path, np.ones((6, 6), np.uint16), transform)
    shifted = from_origin(500_005.0, 3_000_000.0, 10, 10)
    scene_path = tmp_path / "shifted.tif"
    _write_float(scene_path, np.ones((6, 6), np.float32), shifted)

    response = _post_mosaic(client, parcel_path, [scene_path], 4)
    assert response.status_code == 400
    errors = response.json()["detail"]["incompatible_files"]
    assert errors[0]["filename"] == "shifted.tif"
    assert "整数像元" in errors[0]["reason"]


def test_unreadable_file_is_named(client: TestClient, tmp_path: Path):
    transform = from_origin(500_000.0, 3_000_000.0, 10, 10)
    parcel_path = tmp_path / "parcel.tif"
    _write_int(parcel_path, np.ones((6, 6), np.uint16), transform)
    broken = tmp_path / "broken.tif"
    broken.write_bytes(b"this is not a geotiff")

    response = _post_mosaic(client, parcel_path, [broken], 4)
    assert response.status_code == 400
    errors = response.json()["detail"]["incompatible_files"]
    assert errors[0]["filename"] == "broken.tif"


def test_no_scenes_rejected(client: TestClient, tmp_path: Path):
    transform = from_origin(500_000.0, 3_000_000.0, 10, 10)
    parcel_path = tmp_path / "parcel.tif"
    _write_int(parcel_path, np.ones((4, 4), np.uint16), transform)
    with parcel_path.open("rb") as handle:
        # Omitting the required repeated ``scenes`` field is a malformed
        # multipart request and rejected at validation time.
        response = client.post(
            "/mosaic",
            files=[("parcel", (parcel_path.name, handle, "image/tiff"))],
            data={"feather_width": "4"},
        )
    assert response.status_code == 422
