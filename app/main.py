"""FastAPI application for parcel coverage analysis and NDVI mosaicking."""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

import rasterio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from app.analysis import (
    IncompatibleFile,
    UploadedRaster,
    analyze_coverage,
    validate_uploads,
)
from app.mosaic import (
    MAX_FEATHER,
    MIN_FEATHER,
    mosaic_scenes,
    validate_mosaic_uploads,
)

app = FastAPI(
    title="农情遥感分析 API",
    version="1.0.0",
    description=(
        "提供两类服务：上传地块编号图和各景无云掩膜，统计地块无云覆盖率；"
        "或上传多景单波段 Float32 NDVI 景和羽化宽度，按到最近无效像元的"
        "欧氏距离加权拼接，输出分块写入的 Float32 GeoTIFF。"
    ),
)


class ParcelCoverage(BaseModel):
    parcel_id: int = Field(description="地块编号（地块图中的正整数）")
    total_pixels: int = Field(description="该地块在地块图中的总有效像元数")
    covered_pixels: int = Field(
        description="至少一景无云掩膜值为 1 的像元数；多景重叠只计一次"
    )
    coverage_ratio: float = Field(description="covered_pixels / total_pixels")
    below_target: bool = Field(description="覆盖率是否低于请求中的目标覆盖率")


class AnalysisResponse(BaseModel):
    target_coverage: float
    total_parcels_analyzed: int
    below_target_count: int
    parcels: list[ParcelCoverage] = Field(
        description="低于目标覆盖率的地块，按地块编号升序排列"
    )


def _safe_upload_name(upload: UploadFile, index: int) -> str:
    raw = upload.filename or f"上传文件_{index}"
    # Keep a readable label for error messages but never use it as a path.
    name = raw.replace("\\", "/").rsplit("/", 1)[-1].strip()
    return name or f"上传文件_{index}"


def _save_upload(upload: UploadFile, destination: Path) -> None:
    upload.file.seek(0)
    with destination.open("wb") as out:
        shutil.copyfileobj(upload.file, out, length=1024 * 1024)


def _incompatible_response(errors: list[IncompatibleFile]) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={
            "message": "输入影像不兼容，未执行分析。",
            "incompatible_files": [e.as_dict() for e in errors],
        },
    )


@app.get("/health", tags=["系统"], summary="健康检查")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "name": "农情遥感分析 API",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    tags=["分析"],
    summary="找出无云观测覆盖不足的地块",
    responses={
        400: {
            "description": "存在不兼容的 GeoTIFF，响应中列出具体文件与原因",
        }
    },
)
def analyze(
    parcel: Annotated[
        UploadFile,
        File(
            description=(
                "地块编号图：单波段 GeoTIFF，北向上；有效地块编号为正整数"
            )
        ),
    ],
    masks: Annotated[
        list[UploadFile],
        File(
            description=(
                "各景无云掩膜：单波段 GeoTIFF，北向上，坐标系和分辨率与地块图"
                "相同；掩膜值 1 表示无云。可一次上传多景，景幅可错开重叠。"
            )
        ),
    ],
    target_coverage: Annotated[
        float,
        Form(
            description="目标覆盖率，范围 0 到 1；覆盖率严格低于该值的地块被返回"
        ),
    ],
) -> AnalysisResponse:
    if not math.isfinite(target_coverage) or not (0.0 <= target_coverage <= 1.0):
        raise HTTPException(
            status_code=400,
            detail="target_coverage 必须是 0 到 1 之间的有限数值。",
        )
    if not masks:
        raise HTTPException(status_code=400, detail="至少上传一景无云掩膜。")

    with tempfile.TemporaryDirectory(prefix="agri-coverage-") as tmp:
        tmp_dir = Path(tmp)
        parcels_dir = tmp_dir / "parcels"
        masks_dir = tmp_dir / "masks"
        parcels_dir.mkdir()
        masks_dir.mkdir()

        parcel_name = _safe_upload_name(parcel, 0)
        parcel_dest = parcels_dir / "parcel_0000.tif"
        _save_upload(parcel, parcel_dest)
        parcel_upload = UploadedRaster(
            path=str(parcel_dest), filename=parcel_name, role="parcel"
        )

        mask_uploads: list[UploadedRaster] = []
        for index, mask in enumerate(masks, start=1):
            mask_name = _safe_upload_name(mask, index)
            mask_dest = masks_dir / f"mask_{index:04d}.tif"
            _save_upload(mask, mask_dest)
            mask_uploads.append(
                UploadedRaster(
                    path=str(mask_dest), filename=mask_name, role="mask"
                )
            )

        parcel_meta, mask_metas, errors = validate_uploads(
            parcel_upload, mask_uploads
        )
        if errors:
            raise _incompatible_response(errors)

        assert parcel_meta is not None
        assert len(mask_metas) == len(mask_uploads)
        results = analyze_coverage(parcel_meta, mask_metas)

    below = [r for r in results if r.coverage_ratio < target_coverage]
    return AnalysisResponse(
        target_coverage=target_coverage,
        total_parcels_analyzed=len(results),
        below_target_count=len(below),
        parcels=[
            ParcelCoverage(
                parcel_id=r.parcel_id,
                total_pixels=r.total_pixels,
                covered_pixels=r.covered_pixels,
                coverage_ratio=r.coverage_ratio,
                below_target=True,
            )
            for r in below
        ],
    )


def _parse_feather_width(raw: str) -> int:
    try:
        feather = int(raw)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail=(
                f"feather_width 必须是 {MIN_FEATHER} 到 {MAX_FEATHER} 之间的整数。"
            ),
        )
    if not (MIN_FEATHER <= feather <= MAX_FEATHER):
        raise HTTPException(
            status_code=400,
            detail=(
                f"feather_width 必须在 {MIN_FEATHER} 到 {MAX_FEATHER} 像元之间，"
                f"收到 {feather}。"
            ),
        )
    return feather


@app.post(
    "/mosaic",
    tags=["拼接"],
    summary="多景 Float32 NDVI 羽化加权拼接，下载 GeoTIFF",
    response_class=FileResponse,
    responses={
        200: {
            "description": "拼接完成后的单波段 Float32 GeoTIFF，NoData 为 NaN",
            "content": {"image/tiff": {}},
        },
        400: {
            "description": (
                "参数非法或存在不兼容/不可读的 GeoTIFF，响应中列出具体文件与原因"
            ),
        },
    },
)
def mosaic(
    parcel: Annotated[
        UploadFile,
        File(
            description=(
                "地块编号图：单波段 GeoTIFF，北向上；只决定输出范围、"
                "尺寸和像元格网，不参与数值计算"
            )
        ),
    ],
    scenes: Annotated[
        list[UploadFile],
        File(
            description=(
                "各景单波段 Float32 NDVI GeoTIFF，北向上，坐标系和分辨率与"
                "地块图相同，原点可按整数像元错开；可一次上传多景"
            )
        ),
    ],
    feather_width: Annotated[
        str,
        Form(
            description=(
                f"羽化宽度（像元），{MIN_FEATHER}～{MAX_FEATHER} 的整数；"
                "每景权重为到最近无效像元的欧氏距离并按该值截断"
            ),
        ),
    ],
) -> FileResponse:
    feather = _parse_feather_width(feather_width)
    if not scenes:
        raise HTTPException(status_code=400, detail="至少上传一景 NDVI 影像。")

    tmp_dir = Path(tempfile.mkdtemp(prefix="agri-mosaic-"))
    parcels_dir = tmp_dir / "parcels"
    scenes_dir = tmp_dir / "scenes"
    parcels_dir.mkdir()
    scenes_dir.mkdir()

    def _cleanup() -> None:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    try:
        parcel_name = _safe_upload_name(parcel, 0)
        parcel_dest = parcels_dir / "parcel_0000.tif"
        _save_upload(parcel, parcel_dest)
        parcel_upload = UploadedRaster(
            path=str(parcel_dest), filename=parcel_name, role="parcel"
        )

        scene_uploads: list[UploadedRaster] = []
        for index, scene in enumerate(scenes, start=1):
            scene_name = _safe_upload_name(scene, index)
            scene_dest = scenes_dir / f"scene_{index:04d}.tif"
            _save_upload(scene, scene_dest)
            scene_uploads.append(
                UploadedRaster(
                    path=str(scene_dest), filename=scene_name, role="scene"
                )
            )

        parcel_meta, scene_metas, errors = validate_mosaic_uploads(
            parcel_upload, scene_uploads
        )
        if errors:
            raise _incompatible_response(errors)
        assert parcel_meta is not None
        assert len(scene_metas) == len(scene_uploads)

        output_path = tmp_dir / "ndvi_mosaic.tif"
        try:
            mosaic_scenes(parcel_meta, scene_metas, str(output_path), feather)
        except (rasterio.RasterioIOError, OSError) as exc:
            # Map the private temp path back to the user-provided file name so
            # the message points at the upload that could not be read.
            message = str(exc)
            responsible = "未知文件"
            for upload in [parcel_upload, *scene_uploads]:
                if upload.path in message:
                    responsible = upload.filename
                    message = message.replace(upload.path, upload.filename)
                    break
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "读取 GeoTIFF 失败，未生成拼接结果。",
                    "incompatible_files": [
                        {
                            "filename": responsible,
                            "role": (
                                "parcel"
                                if responsible == parcel_name
                                else "scene"
                            ),
                            "reason": f"无法读取栅格数据：{message}",
                        }
                    ],
                },
            )

        # The file is fully written and closed before FileResponse starts
        # streaming, so a failed request never exposes a partial download.
        # Cleanup runs exactly once, after the response has been sent.
        return FileResponse(
            output_path,
            media_type="image/tiff",
            filename="ndvi_mosaic.tif",
            background=BackgroundTask(_cleanup),
        )
    except BaseException:
        _cleanup()
        raise
