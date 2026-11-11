# 农情遥感分析 API

提供两类制图前处理服务，均通过 FastAPI + Swagger UI 直接上传 GeoTIFF 使用：

1. **覆盖率分析（`/analyze`）**：从地块编号图和各景无云掩膜中找出覆盖率不足的地块。
2. **NDVI 羽化拼接（`/mosaic`）**：把多景错幅重叠的单波段 Float32 NDVI 景按
   到最近无效像元的欧氏距离加权拼成无缝的一张 GeoTIFF。

## 启动

```bash
docker compose up --build -d
```

打开：

- Swagger：<http://localhost:8000/docs>
- 健康检查：<http://localhost:8000/health>

可用 `APP_PORT` 修改宿主机端口，例如：

```bash
APP_PORT=8080 docker compose up --build -d
```

## 接口

### `POST /analyze`

以 `multipart/form-data` 上传：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `parcel` | 文件，单个 | 地块编号图，单波段 GeoTIFF |
| `masks` | 文件，可重复 | 各景无云掩膜，单波段 GeoTIFF，至少一景 |
| `target_coverage` | 数值 | 目标覆盖率，范围 0 到 1 |

输入约束：

- 全部文件必须是单波段 GeoTIFF，且北向上（无旋转、无剪切）。
- 全部影像使用相同坐标系、相同分辨率。
- 掩膜原点可与地块图原点不同，但偏移必须是整数个像元，保证像元格网重合。
- 地块图中只有有效的正整数是地块编号；0、负数、小数和 nodata 会被忽略。
- 掩膜中有效值 `1` 表示该像元无云；同一像元被多景覆盖时只计一次。

不兼容时返回 HTTP 400，并在 `detail.incompatible_files` 中逐个列出文件名、
角色（`parcel` / `mask`）和原因，例如：

```json
{
  "detail": {
    "message": "输入影像不兼容，未执行分析。",
    "incompatible_files": [
      {
        "filename": "bad_scene.tif",
        "role": "mask",
        "reason": "坐标系与地块图不一致（EPSG:4326 vs 地块图 EPSG:32650）"
      }
    ]
  }
}
```

成功时返回覆盖率**严格低于** `target_coverage` 的地块，按编号升序排列：

```json
{
  "target_coverage": 0.9,
  "total_parcels_analyzed": 4,
  "below_target_count": 2,
  "parcels": [
    {
      "parcel_id": 102,
      "total_pixels": 2200,
      "covered_pixels": 600,
      "coverage_ratio": 0.2727272727272727,
      "below_target": true
    },
    {
      "parcel_id": 103,
      "total_pixels": 2750,
      "covered_pixels": 500,
      "coverage_ratio": 0.18181818181818182,
      "below_target": true
    }
  ]
}
```

### `POST /mosaic`

以 `multipart/form-data` 上传，成功后直接**下载**单波段 Float32 GeoTIFF
（`Content-Type: image/tiff`，文件名 `ndvi_mosaic.tif`）：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `parcel` | 文件，单个 | 地块编号图，单波段 GeoTIFF，北向上；只决定输出范围、尺寸和像元格网 |
| `scenes` | 文件，可重复 | 各景单波段 **Float32** NDVI GeoTIFF，至少一景，原点可按整数像元错开 |
| `feather_width` | 整数 | 羽化宽度，1～64 像元 |

输入与格网规则：

- 地块图与各景沿用与 `/analyze` 相同的格网兼容规则：相同坐标系、相同分辨率，
  原点之间只允许整数个像元的偏移。地块图本身的取值不参与计算，仅提供输出
  范围和格网。
- 各景必须是单波段 Float32 GeoTIFF、北向上。

无效像元（每景独立判定，包含四类）：

1. 位于该景景幅之外的位置；
2. 等于该景数据集 NoData 值的像元；
3. 非有限值（`NaN`、`+Inf`、`-Inf`）；
4. 被该景有效数据掩膜（GDAL mask band）标记为无效的像元。

权重与合成规则：

- **每景单独计算权重**：从有效像元中心到最近无效像元中心的**欧氏距离**
  （按像元计），并在 `feather_width` 处截断，即
  `weight = min(distance_to_nearest_invalid, feather_width)`。
- 同一输出位置若有多景有效观测，按各景权重做**加权平均**：
  `out = Σ(value × weight) / Σ(weight)`。
- 没有任何有效观测的位置写入输出 NoData（Float32 `NaN`）。

分块与边界：

- 所有 GeoTIFF 读写均按窗口分块进行，内存占用与影像总幅面解耦。
- 距离变换在每个输出瓦片外扩一圈恰好 `feather_width` 的光晕（halo）上计算。
  由于权重在羽化宽度处截断，光晕内的结果与整景变换逐位一致，因此**切块方式
  和输出边界都不会改变任何权重**；不同分块尺寸产出位级相同的结果。

参数非法或文件不兼容/不可读时返回 HTTP 400，并指出具体文件名，例如：

```json
{
  "detail": {
    "message": "输入影像不兼容，未执行分析。",
    "incompatible_files": [
      {
        "filename": "scene_02_ndvi.tif",
        "role": "scene",
        "reason": "原点与地块图未对齐到整数像元（偏移 0.500000, 0.000000 像元）"
      }
    ]
  }
}
```

输出文件在服务端完整写好并关闭后才开始流式返回，请求失败不会产生半成品下载；
上传的临时文件在响应发送后自动删除。

curl 示例：

```bash
curl -X POST http://localhost:8000/mosaic \
  -F "parcel=@samples/parcel_map.tif" \
  -F "scenes=@samples/scene_01_ndvi.tif" \
  -F "scenes=@samples/scene_02_ndvi.tif" \
  -F "feather_width=8" \
  --output ndvi_mosaic.tif
```

## 样例

仓库自带一组可直接在 Swagger 中上传的样例：

- `samples/parcel_map.tif`：120×100，EPSG:32650，像元 10 m，地块编号 101–104
- `samples/scene_01_cloud_free.tif`：相对地块图原点偏移 `(-7, -5)` 像元
- `samples/scene_02_cloud_free.tif`：相对地块图原点偏移 `(+15, +10)` 像元
- `samples/scene_01_ndvi.tif` / `samples/scene_02_ndvi.tif`：与上述错幅布局一致的
  单波段 Float32 NDVI 景，两景在中部重叠且各自带有 NoData、`NaN`、`Inf` 和
  有效数据掩膜标记的无效像元，用于在 Swagger 中直接试拼 `/mosaic`

两景的幅面尺寸不同、原点错开且空间范围互相重叠，像元格网与地块图对齐，
用于验证窗口映射和多景覆盖/加权合并。

同样可以用 curl 试算覆盖率：

```bash
curl -X POST http://localhost:8000/analyze \
  -F "parcel=@samples/parcel_map.tif" \
  -F "masks=@samples/scene_01_cloud_free.tif" \
  -F "masks=@samples/scene_02_cloud_free.tif" \
  -F "target_coverage=0.9"
```

样例中地块 101 完全覆盖，地块 104 覆盖率为 2750/3025 ≈ 90.91%，
不低于 0.9；因此只返回地块 102 和 103。

重新生成样例：

```bash
python scripts/generate_samples.py
```

本地运行与测试（可选）：

```bash
python -m pip install -r requirements-dev.txt
uvicorn app.main:app --reload
pytest -q
```

## 计算与内存

- 以地块图为基准，按 ≤512×512 的窗口遍历；异常大的条带/分块会再切分
  （`app.analysis` / `app.mosaic`）。不会整幅载入地块图或任何一景。
- 覆盖率分析中每个窗口只读取与地块窗口相交的掩膜窗口，越界部分按未覆盖处理；
  多景掩膜用布尔 OR 合并，重叠不重复计数。
- 拼接时每个输出瓦片对每景读取一个外扩 `feather_width` 的光晕窗口计算精确
  欧氏距离变换（优先使用 scipy 的编译实现，缺失时回退到纯 NumPy 的可分离
  变换，两者数值一致），再读取瓦片本身取值做加权平均。
- 上传文件先保存到私有临时目录，请求结束后自动删除；不使用上传文件名作为
  本地路径。
