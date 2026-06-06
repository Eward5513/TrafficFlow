# analysis 目录说明

本目录把整个 TrafficFlow 流水线按"阶段 + 数据形态"拆成 6 个模块：

1. `road_network/` —— 路网（OSM / SUMO net）
2. `ocr/` —— 截图 OCR → 原始 OSM 轨迹
3. `matching/` —— OSM 轨迹 → SUMO edge 路径
4. `simulation/` —— SUMO 仿真输入 / 配置 / 输出
5. `flow/` —— 基于 FCD 与 TLS 的过线流量统计
6. `metrics/` —— 估计误差指标输出

`run_sumo_pipeline.txt` 是手写的 SUMO 命令备忘录（netconvert / randomTrips / sumo / xml2csv 等典型调用）。

数据从左到右流动：

```
road_network ──┐
               ├──► matching ──► simulation ──► flow ──► metrics
ocr ───────────┘
```


## 1. road_network/

存放整个项目的"路网真值"。所有下游模块（matching、simulation、flow）使用的都是这里的同一份网。

| 文件 | 说明 |
|---|---|
| `download_osm.py` | 通过 Overpass API 拉取一块矩形区域的 OSM way / restriction，同时导出 `basemap.osm.xml`（SUMO 用）与 `basemap.geojson`（前端 / 校验用） |
| `basemap.osm.xml` | 原始 OSM XML，`netconvert` 的输入 |
| `basemap.geojson` | OSM way 的 LineString GeoJSON，`feature.properties.osm_id` 用作 OCR 阶段的合法 edge id 集合 |
| `net_tls.net.xml` | 由 `netconvert` 生成、带红绿灯（TLS）的 SUMO 路网，是后续所有 SUMO 仿真与 edge 匹配的真源 |

生成 `net_tls.net.xml` 的典型命令见 `run_sumo_pipeline.txt` 的 Step 1（`netconvert --osm ... --tls.guess-signals ...`）。


## 2. ocr/

把"远程桌面里翻 txt 截图"这种半结构化数据，OCR 还原成 `vin time edge_id time edge_id ...` 形式的逐车轨迹文本。

### 脚本

| 脚本 | 作用 |
|---|---|
| `get_snapshoot.py` | 用 `pyautogui` 自动截图 + PageDown，把 txt 滚动翻屏保存到 `snapshoot_all/` |
| `extract_routes_from_screenshots.py` | 对 `snapshoot/` 下的 PNG 裁剪 + 二值化 + Tesseract OCR，输出到 `ocr_output/`：`*.txt`、`*_crop_preview.png`、`*_crop_marked.png`，以及合并去空行的 `all_ocr_no_blank_lines.txt` |
| `validate_ocr_summary.py` | 用 `road_network/basemap.geojson` 里的 `osm_id` 集合校验 `all_ocr_no_blank_lines.txt`，给出非法行号与具体原因（时间戳格式 / edge id 非整数 / edge id 不在路网内 / 相邻重复等），输出 `ocr_output/invalid_line_numbers.txt`、`invalid_line_details.txt` |
| `refine_ocr_summary.py` | 用合法 edge id 集合修正 OCR 把一个 edge 切成两个 token 的错误、删除以 `(` 开头的无关行、删除连续重复行、合并 OCR 抖动产生的近似重复行，输出 `all_ocr_no_blank_lines_refined.txt` 与对应日志 |
| `build_routes.py` | 把"以 vin 起头的连续行"聚合成轨迹组，做时间戳单调性检查，输出两份：`route_by_edge.txt`（按 vin 合并）和 `route_by_edge_no_merge.txt`（每个原始组独立一条），以及合并/跳过日志 |
| `export_routes_json.py` | 把单行过长的 `route_by_edge.txt` 按每行最多 N 对 `(timestamp, edge_id)` 折行，便于后续阅读和分块处理 |
| `merge_basic_routes.py` | 把 `analysis/matching/matched_routes.rou.xml`、`matched_routes_shuffled_depart.rou.xml` 和 `analysis/simulation/random_routes_1w.rou.xml` 合并并按 depart 排序，输出到 `analysis/simulation/basic_data.rou.xml`（即仿真的"基础需求"） |

### 子目录

- `snapshoot/`、`snapshoot_all/` —— 屏幕截图原图。
- `ocr_output/` —— 所有 OCR 中间产物和最终的 `route_by_edge*.txt`。

### 典型顺序

```
get_snapshoot.py
  → extract_routes_from_screenshots.py
  → validate_ocr_summary.py          (查问题)
  → refine_ocr_summary.py            (修问题)
  → build_routes.py                  (聚合成轨迹)
  → 交给 analysis/matching/
```

`merge_basic_routes.py` 是出口阶段，把 OSM 匹配出的 SUMO 路由和随机生成的背景流量合并成仿真的统一输入。


## 3. matching/

把 OSM edge 序列匹配成 SUMO edge 序列。详细算法描述见 `matching/README.md`（分层 DP + 邻居缓存 + BFS 最短路）。

| 脚本 | 作用 |
|---|---|
| `build_sumo_edge_graph.py` | 流式读 `road_network/net_tls.net.xml`，过滤掉 internal edge 与 passenger 不可走的 edge，按 `<connection>` 建"以 SUMO edge 为节点"的有向图，并预算 1/2 跳邻居缓存 |
| `match_sumo_edges.py` | OSM→SUMO 候选索引、BFS 最短路、分层动态规划、模糊匹配兜底等核心匹配逻辑 |
| `parse_trajectories.py` | 入口脚本：流式读 `ocr/ocr_output/route_by_edge_no_merge.txt`，对每条轨迹调用匹配；写 `matched_routes.txt`、`failed_routes.txt`、`fuzzy_match_logs.txt` |
| `export_sumo_routes.py` | 把 `matched_routes.txt` 渲染成 SUMO 标准 `<routes>` XML：`matched_routes.rou.xml` 与按 depart 打散后的 `matched_routes_shuffled_depart.rou.xml` |
| `build_sumo_trips_from_matchable_edges.py` | 另一条路径：不要求顺序连通，只取每个 OSM edge 一个最优 SUMO 候选，按可达性裁剪后输出 `<trip>` 形式的 `matched_trips.rou.xml`（让 SUMO 自己 route） |

主要产物：

- `matched_routes.txt`、`matched_routes.rou.xml`、`matched_routes_shuffled_depart.rou.xml`
- `failed_routes.txt`、`fuzzy_match_logs.txt`、`reachability_prune_logs.txt`
- `matched_trips.rou.xml`


## 4. simulation/

整个 SUMO 仿真的配置、输入、输出都集中在这里。

### 仿真配置

| 文件 | 用途 |
|---|---|
| `simulation_output.sumocfg` | SUMO 输出模板配置，可由 `run_penetration_simulations.py` 按渗透率生成对应配置 |
| `simulation_output_5.sumocfg` | 渗透率 5% 的对照版本 |
| `simulation_replay.sumocfg` | 只含 input/processing/time 的复盘模板，由 `replay.py` 按渗透率生成对应 cfg |
| `run_penetration_simulations.py` | 按脚本顶部写死的渗透率批量生成 `sumocfg/simulation_output_<rate>_<seed>.sumocfg`，并并行执行 `sumo -c`，日志写到 `sumocfg/*.log` |
| `replay.py` | 复盘用 helper：`python replay.py 105 -p 50` ⇒ 生成 `sumocfg/simulation_replay_50_42.sumocfg` 后执行 `sumo-gui --load-state state/50/state_105.00.xml.gz --begin 105` |

三个 sumocfg 内的 `net-file` 都指向 `../road_network/net_tls.net.xml`，所有相对路径以 `analysis/simulation/` 为 cwd。

### 仿真输入

- `random_passenger` —— duarouter 输出的纯 vType 定义（`<vType id="random_passenger" vClass="passenger"/>`）。
- `random_trips_1w.trips.xml` / `random_routes_1w.rou.xml` —— `randomTrips.py --validate` 生成的 1 万条背景随机流，以及 duarouter 排好的对应路由。
- `basic_data.rou.xml` —— 由 `analysis/ocr/merge_basic_routes.py` 合成的"完整需求"（matched + matched_shuffled + random），是 `penetration/` 抽样的源头。

### penetration/

按渗透率对 `basic_data.rou.xml` 做随机抽样，给被抽中车染色、记录被抽中的 vin 列表，供 flow 模块作为"观测到的样本"使用。

- `generate_penetration_routes.py`
  - 入参：`-p` 渗透率 (%)、`--seed` 随机种子。
  - 出参：`<rate>_<seed>.rou.xml`（被抽中的染红、其余染灰的整份路由）和 `<rate>_<seed>_sample.txt`（被抽中车辆的 vin 列表）。
- 当前产物覆盖 5 / 10 / 20 / 30 / 40 / 50 几个渗透率。

### output/&lt;rate&gt;/

SUMO 跑完每个渗透率版本产生的输出：`fcd.csv`、`tripinfo.xml`、`vehroute.xml`、`summary.xml`、`statistics.xml`。flow 模块下游就吃 `fcd.csv`。

### state/&lt;rate&gt;/

`save-state.period=1` 下每秒一份的 `state_<t>.xml.gz` 快照，供 `replay.py` 在任意时刻打开 GUI 复盘。


## 5. flow/

利用 FCD 轨迹 + TLS 相位信息，按"周期 × 相位 × 流向"维度统计真实/采样过线流量，并算误差。

| 脚本 | 作用 |
|---|---|
| `print_junction_tls.py` | 流式读 `road_network/net_tls.net.xml`，把每个 TLS 路口的所有 phase 与允许通行的 connection（仅 `G`/`g`）按 `(from, to, dir)` 分组导出为 `junction_tls.json` |
| `count_edge_departures_by_cycle.py` | 给定渗透率，读 `analysis/simulation/output/<pen>/fcd.csv` + `junction_tls.json`，按周期切分相位窗口，识别每辆车跨过停车线的事件，按 `(cycleIndex, phaseIndex, direction, fromLane→toEdge)` 累加，结果写到 `analysis/flow/<pen>/<junctionId>_edge_departures_by_cycle.json` |
| `calculate_sample_error_metrics.py` | 对单个路口（默认 `cluster_1247897642_2350807770`），用 `simulation/penetration/<pen>_<seed>_sample.txt` 做"观测样本"，对比 `flow/<pen>/<junctionId>_edge_departures_by_cycle.json` 的真实计数，分两种估计 `obsonly` / `scale`（按渗透率倒数放大），输出 MAE / RMSE / NMAE 等指标，写到 `analysis/metrics/<pen>/...sample_error_metrics.json` |
| `calculate_queue_augmented_estimates.py` | 在 `scale` 基础上，对直行/左转独占车道，用周期开始前 1 秒采样车排队队尾位置反推一个队长估计（`floor(distance / spacing) + 1`），替换该 movement 的直接计数；同样输出到 `analysis/metrics/<pen>/...queue_estimates.json` |
| `check_route.py` | 调试用：从 `simulation/output/50/fcd.csv` 中按 vin 抽出该车实际经过的 edge 序列 |

辅助文件：

- `junction_tls.json` —— 全路口的 TLS 相位 + 连接结构。
- `edge_departures_by_cycle.json` —— 历史单文件输出，已被 `<pen>/` 分目录替代。
- `<pen>/*.json` —— 各渗透率下每个 TLS 路口的周期过线统计。


## 6. metrics/

只放最终指标输出，不放原始 FCD。

```
metrics/
├── 10/                # 10% 渗透率结果（占位，尚未跑）
├── 30/
│   ├── cluster_1247897642_2350807770_queue_estimates.json
│   └── cluster_1247897642_2350807770_sample_error_metrics.json
└── 50/
    ├── cluster_1247897642_2350807770_42_queue_estimates.json
    └── cluster_1247897642_2350807770_sample_error_metrics.json
```

文件名含 `_<seed>_` 的来自 `calculate_queue_augmented_estimates.py`，不含的是早期 `calculate_sample_error_metrics.py` 的输出。


## 端到端跑一遍的顺序

1. `road_network/download_osm.py` → `netconvert` 得到 `net_tls.net.xml`。
2. `ocr/get_snapshoot.py` → `extract_routes_from_screenshots.py` → `validate_ocr_summary.py` → `refine_ocr_summary.py` → `build_routes.py`。
3. `matching/parse_trajectories.py` → `matching/export_sumo_routes.py`。
4. `randomTrips.py` 生成背景流量（命令见 `run_sumo_pipeline.txt`）。
5. `ocr/merge_basic_routes.py` 合并出 `simulation/basic_data.rou.xml`。
6. `simulation/penetration/generate_penetration_routes.py -p <rate>` 抽样。
7. `cd analysis/simulation && python run_penetration_simulations.py` 跑仿真，得到 `output/<rate>/fcd.csv` 与 `state/<rate>/`。
8. `flow/print_junction_tls.py` → `flow/count_edge_departures_by_cycle.py -p <rate>`。
9. `flow/calculate_sample_error_metrics.py -p <rate>` 与 `flow/calculate_queue_augmented_estimates.py -p <rate>`，得到 `metrics/<rate>/...json`。


## 路径约定

所有脚本里的默认路径都基于 `__file__` 解析，并且只走"模块内部相对路径"或"`analysis/` 内的相对路径"。重新挪动目录时请同时更新：

- 各脚本顶部 `DEFAULT_*` 常量；
- `simulation/*.sumocfg` 中的 `net-file` / `route-files`；
- `matching/README.md` 与本 README 中的命令示例。
