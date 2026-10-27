# OCR Pipeline

这个目录负责把远程数据文件整理成适合截图的文本版，再通过截图和 OCR 还原成后续 `analysis/` 模块可用的车辆轨迹文本。

原始数据在远程桌面上先通过 `wrap_routes_for_screenshot.py` 整理成方便展示和翻页截图的 txt；随后本地或远程自动翻页截图，再对截图做裁剪和 Tesseract OCR，之后校验和修正 OCR 文本，最后按车辆 VIN 聚合、合并、按时间间隔切分，生成 `vin time edge_id time edge_id ...` 格式的轨迹文件。

所有脚本都不接收命令行参数，输入输出路径和处理参数全部以常量形式写死在各脚本开头，需要调整时直接改代码中的常量。

## 数据流

推荐从代码仓库的 `ocr/` 目录运行这些脚本。脚本读写真数据一律指向数据盘 `/data5/tzhang174/TrafficFlow/ocr_data/`，不把截图或 OCR 文本写进 git 仓库。

```text
远端原始路线文本
  -> wrap_routes_for_screenshot.py
  -> route_by_edge_wrapped.txt
  -> 远程桌面 txt 窗口
  -> get_snapshoot.py
  -> snapshoot_all/*.png
  -> extract_routes_from_screenshots.py
  -> ocr_output_all/all_ocr_no_blank_lines.txt
  -> refine_ocr_summary.py
  -> ocr_output_all/all_ocr_no_blank_lines_refined.txt
     ocr_output_all/log/all_ocr_no_blank_lines_refine_log.txt
  -> validate_ocr_summary.py（校验 refined 文本，剩余问题人工修正）
  -> ocr_output_all/log/invalid_line_details.txt
  -> build_routes.py
  -> ocr_output_all/route_by_edge.txt
     ocr_output_all/route_by_edge_no_merge.txt
     ocr_output_all/route_by_edge_split.txt
  -> analysis/matching/
```

`build_routes.py` 生成三份轨迹文件：

- `route_by_edge_no_merge.txt`：不合并 VIN，每个原始轨迹组单独一条。
- `route_by_edge.txt`：相同 VIN 的多个轨迹组合并到同一行。
- `route_by_edge_split.txt`：在 `route_by_edge.txt` 的基础上按时间间隔切分轨迹，**是后续 `analysis/` 模块使用的数据源**。

`wrap_routes_for_screenshot.py` 是截图前的远端准备脚本，不是 OCR 后处理；它不改变数据含义，只把过长的路线按固定数量的 `(time, edge_id)` 对折行，方便把文本展示出来再截图。

## 主要输入

- 远端原始路线文本：`wrap_routes_for_screenshot.py` 的输入，通常是单行很长的 `vin time edge_id time edge_id ...` 文本。
- 远程桌面中的折行后 txt 窗口：`get_snapshoot.py` 通过全屏截图和 `PageDown` 翻页采集它。
- `snapshoot_all/*.png`：自动翻页采集到的截图，`extract_routes_from_screenshots.py` 直接读取该目录。
- `analysis/road_network/basemap.geojson`：用于校验 OCR 出来的 OSM edge id 是否存在于路网中。
- `ocr_output_all/all_ocr_no_blank_lines.txt`：OCR 后合并去空行的文本，是修正阶段的输入。
- `ocr_output_all/all_ocr_no_blank_lines_refined.txt`：修正后的文本，是校验和轨迹构建阶段的输入。

## 主要输出

截图前文本整理（`wrap_routes_for_screenshot.py`）：

- `ocr_output/route_by_edge_wrapped.txt`：在远端生成的折行展示版，供打开窗口后翻页截图。

截图采集（`get_snapshoot.py`）：

- `snapshoot_all/*.png`：自动翻页截到的全屏截图。

截图 OCR（`extract_routes_from_screenshots.py`）：

- `ocr_output_all/<image_stem>.txt`：每张截图的 OCR 原始文本。
- `ocr_output_all/<image_stem>_crop_preview.png`：每张截图被送入 OCR 的裁剪区域。
- `ocr_output_all/all_ocr_no_blank_lines.txt`：所有 OCR 文本合并后去掉空行的结果。

OCR 修正与校验（`refine_ocr_summary.py`、`validate_ocr_summary.py`）：

- `ocr_output_all/all_ocr_no_blank_lines_refined.txt`：自动修正和去重后的 OCR 文本。
- `ocr_output_all/log/all_ocr_no_blank_lines_refine_log.txt`：修正动作日志。
- `ocr_output_all/log/invalid_line_details.txt`：非法行、相邻重复行、VIN 顺序异常的详细原因。

轨迹构建（`build_routes.py`）：

- `ocr_output_all/route_by_edge.txt`：按 VIN 合并后的车辆轨迹。
- `ocr_output_all/route_by_edge_no_merge.txt`：不合并 VIN、按原始轨迹组输出的车辆轨迹。
- `ocr_output_all/route_by_edge_split.txt`：在合并轨迹基础上按时间间隔切分后的轨迹，后续模块的数据源。
- `ocr_output_all/log/route_by_edge_build_log.txt`：构建轨迹时的合并、切分、跳过和异常日志。

## 文本格式

中间 OCR 文本允许两种行格式：

```text
vin time edge_id time edge_id ...
time edge_id time edge_id ...
```

第一种行以 VIN 开头，表示一个新车辆或新轨迹组开始；第二种没有 VIN，是上一条轨迹的续行。

最终输出给 `matching/` 的格式是每行一条轨迹：

```text
vin time edge_id time edge_id ...
```

这里的 `time` 是 `HH:MM:SS`，`edge_id` 是 `basemap.geojson` 里的 OSM edge id。切分之后同一个 VIN 可能对应多行，每行是一段独立轨迹。

## 脚本说明

### `wrap_routes_for_screenshot.py`

用途：在截图前运行，通常是在远程桌面环境里把过长的 `route_by_edge.txt` 折成多行，方便把路线文本展示在窗口里截图。这个脚本不生成 JSON，输出仍然是 txt。

输入：

- `ocr_output/route_by_edge.txt`（`INPUT_TXT`）

输出：

- `ocr_output/route_by_edge_wrapped.txt`（`OUTPUT_TXT`）

关键常量：

- `INPUT_TXT` / `OUTPUT_TXT`：输入输出路径。如果脚本在远程桌面机器上执行，应改成远端实际存放路线 txt 的路径。
- `MAX_PAIRS = 13`：每行最多输出的 `(time, edge_id)` 对数。

输出规则：

- 每行最多输出 `MAX_PAIRS` 个 `(time, edge_id)` 对。
- 第一行保留 VIN。
- 后续续行不重复 VIN。

### `get_snapshoot.py`

用途：自动采集远程桌面里 txt 窗口的翻页截图。

输入：

- 当前屏幕内容。运行脚本后，需要在倒计时内把远程桌面切到全屏、把目标 txt 窗口放到最前、滚动到起始位置。

输出：

- `snapshoot_all/000001.png` 这类按序号命名的全屏截图。

关键常量：

- `OUTPUT_DIR = Path.cwd() / "snapshoot_all"`：输出目录依赖运行时工作目录。
- `WAIT_BEFORE_START`：开始前预留的手动准备时间。
- `WAIT_AFTER_PAGEDOWN`：每次 `PageDown` 后等待页面稳定的时间。
- `MAX_SHOTS`：最多截图数量。

停止方式：

- `Ctrl+C`
- 或把鼠标移动到屏幕左上角触发 `pyautogui` 的 failsafe。

### `extract_routes_from_screenshots.py`

用途：对截图裁剪、二值化、放大，然后调用 Tesseract OCR，把图片里的轨迹文本识别成 txt。多线程并发处理。

输入：

- `snapshoot_all/*.png`

输出：

- `ocr_output_all/<image_stem>.txt`
- `ocr_output_all/<image_stem>_crop_preview.png`
- `ocr_output_all/all_ocr_no_blank_lines.txt`

关键常量：

- `IMG_DIR` / `OUT_DIR`：输入截图目录和输出目录。
- `CROP_BOX = (0, 75, 3800, 2025)`：裁剪区域。
- `TESS_CONFIG = "--oem 3 --psm 6"`：Tesseract OCR 参数。
- `WORKERS`：并发线程数，默认为 CPU 核心数的两倍。

脚本中保留了 `normalize_text()`，用于把 `O/o` 替换成 `0`、把 `I/l/|` 替换成 `1` 等常见 OCR 纠错；但当前主流程没有调用它，实际写出的 `<image_stem>.txt` 是 Tesseract 的原始识别结果。

### `refine_ocr_summary.py`

用途：在校验规则基础上自动修正一些 OCR 常见错误，生成更干净的 OCR 文本。

输入（常量 `INPUT_TXT`、`GEOJSON`）：

- `ocr_output_all/all_ocr_no_blank_lines.txt`
- `../analysis/road_network/basemap.geojson`

输出（常量 `OUTPUT_TXT`、`LOG_TXT`）：

- `ocr_output_all/all_ocr_no_blank_lines_refined.txt`
- `ocr_output_all/log/all_ocr_no_blank_lines_refine_log.txt`

修正逻辑：

- 删除以 `(` 开头的整行噪声。
- 删除相邻完全重复行。
- 尝试把被 OCR 错拆成两个数字 token 的 edge id 合并回来。
- 如果当前行非法，并且和下一行或上一条已保留行去空格后相同或只差一个字符，则删除当前行。
- 如果某个 VIN 行破坏了 VIN 非递减顺序，并且违例两端之一与其相邻行去空格后相同或近似，则删除该行（OCR 把同一行重复识别且 VIN 读错的情况）。

每一次合并或删除都会写入 refine log，便于人工复核。refine 之后可根据校验结果和日志文件再做人工修正。

### `validate_ocr_summary.py`

用途：校验修正后的 OCR 文本是否符合轨迹格式，并报告相邻重复行和 VIN 顺序异常，供人工修正参考。

输入（常量 `INPUT_TXT`、`GEOJSON`）：

- `ocr_output_all/all_ocr_no_blank_lines_refined.txt`
- `../analysis/road_network/basemap.geojson`

输出（常量 `DETAIL_OUTPUT_TXT`）：

- `ocr_output_all/log/invalid_line_details.txt`

校验内容：

- 行是否为空或 token 数不足。
- 是否能解析为 `vin time edge_id ...` 或 `time edge_id ...`。
- 时间是否为 `HH:MM:SS`。
- edge id 是否为整数（edge id 是否存在于路网 geojson 中的检查目前已关闭）。
- 是否存在相邻重复行。
- VIN 是否保持非递减顺序。

### `build_routes.py`

用途：把修正后的 OCR 文本按 VIN 分组，提取 `(time, edge_id)` 序列，生成最终给 `matching/` 使用的轨迹文件。

输入（常量 `INPUT_TXT`）：

- `ocr_output_all/all_ocr_no_blank_lines_refined.txt`

输出（常量 `OUTPUT_TXT`、`OUTPUT_NO_MERGE_TXT`、`OUTPUT_SPLIT_TXT`、`LOG_TXT`）：

- `ocr_output_all/route_by_edge.txt`
- `ocr_output_all/route_by_edge_no_merge.txt`
- `ocr_output_all/route_by_edge_split.txt`
- `ocr_output_all/log/route_by_edge_build_log.txt`

关键常量：

- `SPLIT_GAP_SECONDS = 3600`：切分阈值，相邻两条边的进入时间戳相差超过该值就切分轨迹。

处理逻辑：

- 遇到行首 VIN 时开始一个新轨迹组。
- 没有 VIN 的行会作为上一轨迹组的续行。
- 从每个轨迹组里提取合法的 `(time, edge_id)` 对。
- 检查同一轨迹内时间是否非递减。
- 记录 VIN 顺序异常、空轨迹、时间倒序等问题。
- 生成三份结果：
  - `route_by_edge_no_merge.txt`：不合并，保留每个原始轨迹组。
  - `route_by_edge.txt`：相同 VIN 的轨迹组合并。
  - `route_by_edge_split.txt`：在 `route_by_edge.txt` 基础上，相邻两条边进入时间戳相差超过 `SPLIT_GAP_SECONDS`（默认 1 小时）就切分成新轨迹，切出来的每段保持原 VIN。每次切分会在 build log 的 `=== split mode ===` 部分记录切分点。

## 运行顺序

常见运行顺序如下（所有脚本无命令行参数，直接运行即可）：

```shell
cd ocr
python wrap_routes_for_screenshot.py
python get_snapshoot.py
python extract_routes_from_screenshots.py
python refine_ocr_summary.py
python validate_ocr_summary.py
python build_routes.py
```

如果校验发现 refined 文本仍有问题，先人工修正 `ocr_output_all/all_ocr_no_blank_lines_refined.txt`，再重新运行 `validate_ocr_summary.py` 和 `build_routes.py`。

完成后，后续 `analysis/` 模块以 `ocr/ocr_output_all/route_by_edge_split.txt` 作为数据源，继续把 OSM edge 序列匹配到 SUMO edge 序列。
