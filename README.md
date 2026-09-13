# TrafficFlow

TrafficFlow 的主要代码和数据位于 `analysis/`，STGCN 数据整理在 `reimplementation/stgcn/`。流水线按"阶段 + 数据形态"拆成 8 个模块：

1. `analysis/road_network/` —— 路网（OSM / SUMO net）
2. `ocr/` —— 远端文本折行 → 截图 OCR → 原始 OSM 轨迹
3. `analysis/matching/` —— OSM 轨迹 → SUMO edge 路径
4. `analysis/simulation/` —— SUMO 仿真输入 / 配置 / 输出
5. `analysis/flow/` —— 基于 FCD 与 TLS 的过线流量统计
6. `analysis/metrics/` —— 估计误差指标输出
7. `analysis/graph/` —— 静态路网图（当前仅 R-only 道路级有向图）
8. `reimplementation/stgcn/` —— R-only STGCN 数据整理，以及原版 STGCN 的 PyTorch 迁移（尚未正式训练）

除非特别说明，下文中的模块与文件路径都以 `analysis/` 为根目录，例如 `road_network/` 实际对应 `analysis/road_network/`。OCR 辅助脚本单独放在仓库根目录的 `ocr/`；STGCN 脚本与产物在 `reimplementation/stgcn/`。

SUMO 相关的 `netconvert` / `randomTrips.py` / `sumo` / `xml2csv.py` 典型命令已整理在下文的「SUMO 命令备忘」。

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

生成 `net_tls.net.xml` 的典型命令见下文「SUMO 命令备忘」里的 `netconvert` 示例。


## 2. ocr/

先把远端原始路线文本折行成适合展示和截图的 txt，再对"远程桌面里翻 txt 截图"这种半结构化数据做 OCR，还原成 `vin time edge_id time edge_id ...` 形式的逐车轨迹文本。

详细脚本职责、输入输出和运行顺序见 `ocr/README.md`。

## 3. matching/

把 OSM edge 序列匹配成 SUMO edge 序列。详细算法描述见 `analysis/matching/README.md`（分层 DP + 邻居缓存 + BFS 最短路）。

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

仿真需求的生成脚本、以及体量很大的仿真输入输出都集中在这里。

### 需求生成

当前的需求模型是**连续 20 天、每天一份 route 文件**，由三个脚本前后衔接：

| 脚本 | 作用 |
|---|---|
| `generate_random_routes.py` | 调 `$SUMO_HOME/tools/randomTrips.py`，为每天各抽一个随机种子、生成 3 万条背景随机流，产物写到 `data/random_trip/<编号>_<种子>.{trips,rou}.xml`，日志写到 `data/random_trip/logs/`。需要 SUMO 环境（`--validate` 会拉起 `duarouter`），带 `--jobs` 控制并发、已生成的天自动跳过 |
| `merge_basic_routes.py` | 把 `analysis/matching/matched_routes.rou.xml` 的匹配轨迹翻倍、每辆车的 depart 各自随机抖动 ±5 分钟，再并入当天的背景随机流，按 depart 升序写出 `data/route_demand/<编号>_<种子>.rou.xml` |
| `run_simulations.py` | **整条链路的入口**：缺哪天的背景流或日需求就先调上面两个脚本补出来，再拿日需求加路网跑一整天 SUMO，自己生成 sumocfg，产物全部写进 `data/simulation/<编号>_<种子>/`。带 `--jobs` 控制并发、已跑完的天自动跳过 |

跑一次完整仿真只需要一条命令，不用手工一步步调前两个脚本：

```bash
export SUMO_HOME=/usr/share/sumo
python3 analysis/simulation/run_simulations.py
```

编号是 `1..20`（不补零），种子由 `generate_random_routes.py` 随机抽取后写进文件名永久留档。后两个脚本都直接从上游文件名解析编号与种子，彼此不靠公式约定；同一个种子既是当天背景流的生成种子、当天 depart 抖动的种子，也是当天 SUMO 的 `--seed`，所以每一步都可以随时重跑而结果不变。

每天数据的构成与清洗规则：

- 匹配轨迹翻倍：每条真实轨迹产出两辆车，原车 id 不变、复制车 id 加 `.dup1`，两辆车**各自独立**抽 `[-300, +300)` 秒加到原 depart 上，负值截到 0。
- 背景随机流既不翻倍也不抖动，depart 原样保留。
- 反向边去环：相邻两条边互为反向边（由 `road_network/find_reverse_edge_pairs.py` 给出）即整对删除，起点边不参与配对，删除后必须仍连通，不处理嵌套环。
- 去短轨迹：边序列不足 3 条边的车辆整辆丢弃。

去环和去短只看边序列、与 depart 无关，所以匹配轨迹只清洗一次即被 20 天共用。实测 161,374 条匹配轨迹去环影响 31,001 辆、删掉 81,038 条边，再滤掉 5,360 条短轨迹后保留 156,014 条，于是每天约 156,014 × 2 + 30,000 ≈ 34.2 万辆、单文件约 205 MB。

需要注意「需求翻倍」和「观测渗透率」是两件不同的事：翻倍改变路上真实存在的车辆数（拥堵水平会变），渗透率只改变哪些车被观测到。

### 仿真配置与输入

- `data/random_trip/<编号>_<种子>.rou.xml` —— 当天的 3 万条背景流，vType 固定为 `<vType id="random_passenger" vClass="passenger"/>`，车辆 id 前缀固定为 `random`（用来和匹配轨迹的 VIN 区分）。sumocfg 用 `device.rerouting.probability=1` 给全部车辆开动态重路由。
- `data/route_demand/<编号>_<种子>.rou.xml` —— 当天的完整需求，即仿真真正加载的 route 文件。
- `road_network/tls_schedule.add.xml` —— 作为 `additional-files` 挂进去的分时信号方案：`offpeak` / `evening_peak` 两套配时，加一张 WAUT 切换表（60000 s 切晚高峰、75000 s 切回）。不挂它的话全天都是路网里的 `programID="0"`，`run_simulations.py --no-tls-schedule` 可以关掉。

### 仿真运行与输出

`run_simulations.py` 不用外部模板，自己给每天生成一份 sumocfg，且配置里所有路径都相对该文件所在的当天文件夹，配置、日志、输出放在一起：

```text
data/simulation/<编号>_<种子>/
├── simulation.sumocfg
├── fcd.csv.gz        # 逐秒轨迹，按 .gz 后缀由 SUMO 直接压缩写出
├── tripinfo.xml
├── vehroute.xml      # 含进出边时刻
├── summary.xml
├── statistics.xml
└── simulation.log
```

不设 `end`，一天要跑到最后一辆车到达为止；`--end` 只用来做冒烟测试。五个输出文件齐全且非空的天默认跳过，`--force` 才重跑覆盖，脚本任何时候都不会删目录。

脚本按这个顺序解决依赖：

1. 先扫 `data/simulation/` 把已经跑完的天摘掉。这一步排在补数据之前，免得为一个早就跑完的天白造一份 200 MB 的日需求；
2. `data/random_trip/` 缺天就调 `generate_random_routes.py --days <最大编号>`。它只能按 `1..N` 连续生成，如果指定跑第 5 天会把第 1..5 天的背景流补齐（已有的跳过），但只为第 5 天造日需求、只跑第 5 天；
3. `data/route_demand/` 缺天就调**一次** `merge_basic_routes.py --only <所有缺的天>`。它每次都要重新清洗 16 万条匹配轨迹，按天分开调会把这份开销重复很多遍；
4. 最后才写 sumocfg、拉起 SUMO。

route 文件结尾没有 `</routes>` 的算作没生成，会被重做。`--no-upstream` 可以让缺数据直接报错而不是去生成。

### 子图轨迹与观测流量

当前 R-only 实验不再按渗透率重跑 SUMO。20 天全量 FCD 提取到 `data/processed/subgraph_trajectories/day_01` … `day_20` 后：

1. `count_edge_flow.py` 按道路进入事件写出每天的全量 `edge_flow_5min.csv`（56 条 R 道路 × 288 个 5 分钟窗）。
2. `sample_vehicle_ids.py` 在当天车辆集合上做嵌套前缀抽样（5/10/20/30/40/50/70%），写出 `sampled_vehicle_ids/vehicles_pXX.txt`。
3. `count_observed_edge_flow.py` 复用同一套进入事件规则，只替换参与统计的车辆集合，在每天的 `observed_edge_flow_5min/` 下写出 7 个观测流量 CSV。观测值是原始进入次数，不做 `1/p` 缩放。

```bash
python3 analysis/simulation/run_simulations.py                 # 完整 20 天，4 并发
python3 analysis/simulation/run_simulations.py --days 1        # 只跑第 1 天看效果
python3 analysis/simulation/run_simulations.py --days 3        # 跑前 3 天
python3 analysis/simulation/run_simulations.py --days 1 2 3    # 指定跑第 1、2、3 天
python3 analysis/simulation/run_simulations.py --generate-only # 备齐数据和 sumocfg，不跑 SUMO
```

### penetration/

按渗透率对某一天的 `data/route_demand/<编号>_<种子>.rou.xml` 做随机抽样，给被抽中车染色、记录被抽中的 vin 列表，供 flow 模块作为"观测到的样本"使用。

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


## 7. graph/

当前阶段只构建 **R-only** 静态有向道路图：节点是 `simulation/data/subgraph.txt` 里的唯一 SUMO edge ID，有向边只来自 `road_network/net_tls.net.xml` 中两端都在子图内的 `<connection>`。不读取轨迹、渗透率采样或流量，也不在这一步构建 M-only / MR 图。

```bash
python3 analysis/graph/build_r_graph.py
```

产物写到 `analysis/graph/r_graph/`。邻接矩阵约定为 `A[i, j] = 1` 当且仅当 source/from 节点 i 指向 target/to 节点 j；保存的是原始二值有向拓扑，不加自环、不对称化、不归一化。详细规则见 `analysis/graph/AGENTS.md`。


## 8. reimplementation/stgcn/

把已经聚合好的全量 / 观测 5 分钟 R 道路进入流量整理成窗口数据，并把原版 TensorFlow STGCN（`reference/STGCN_IJCAI-18`）迁到 PyTorch。默认按仿真日序号 `day_01`…`day_20` 的时间先后做完整日期划分（14/3/3），每天内部切 12 步窗口，标签是窗口最后一个时间片的全量流量（当前时刻重建，不是预测下一个 5 分钟）。观测输入按渗透率各自用训练集做全局 Z-score，全量目标共用一组训练集 scaler。

公共训练接口在 `reimplementation/common/`，STGCN 层与图算子在 `reimplementation/stgcn/`。公开张量格式为 `[B,T,V,C]`。加权邻接矩阵直接读取已经完成的 `stgcn_weighted_adjacency.npy`，训练时再构造拉普拉斯 / Chebyshev，不再做第二次高斯核。

```bash
python3 reimplementation/stgcn/prepared_data/prepare_r_only_stgcn_data.py --dry-run
python3 reimplementation/stgcn/prepared_data/prepare_r_only_stgcn_data.py --overwrite
python3 -m unittest discover -s reimplementation/stgcn/tests -v
```

产物写到 `reimplementation/stgcn/prepared_data/r-only/`。输入仍来自 `analysis/simulation/data/processed/subgraph_trajectories/` 与 `analysis/graph/r_graph/`。训练入口、评估、监控和未来输出结构见 `reimplementation/stgcn/README.md`。当前没有正式 checkpoint 或实验指标。

对称、按直接 SUMO connection 中心点距离加权的 STGCN 邻接矩阵由
`reimplementation/stgcn/prepared_data/prepare_stgcn_weighted_adjacency.py`
生成，产物目录是 `reimplementation/stgcn/prepared_data/r-only/adjacency_matrix/`。
只对 `A_sym` 邻居计算距离，不绕完整路网补反向。不覆盖
`analysis/graph/r_graph/r_adjacency.npy`，不加自环，也不在这一步做拉普拉斯。


## 端到端跑一遍的顺序

1. `road_network/download_osm.py` → `netconvert` 得到 `net_tls.net.xml`。
2. `ocr/wrap_routes_for_screenshot.py` → `ocr/get_snapshoot.py` → `extract_routes_from_screenshots.py` → `validate_ocr_summary.py` → `refine_ocr_summary.py` → `build_routes.py`。
3. `matching/parse_trajectories.py` → `matching/export_sumo_routes.py`。
4. `simulation/run_simulations.py` 一条命令跑完仿真：它会自己先生成 20 天的背景随机流到 `simulation/data/random_trip/`、合成 20 份日需求到 `simulation/data/route_demand/`，再输出 `simulation/data/simulation/<编号>_<种子>/fcd.csv.gz` 等结果（需要 SUMO 环境，命令细节见下方「SUMO 命令备忘」）。
5. `simulation/penetration/generate_penetration_routes.py -p <rate>` 抽样。
6. `flow/print_junction_tls.py` → `flow/count_edge_departures_by_cycle.py -p <rate>`。
7. `flow/calculate_sample_error_metrics.py -p <rate>` 与 `flow/calculate_queue_augmented_estimates.py -p <rate>`，得到 `metrics/<rate>/...json`。


## SUMO 命令备忘

前提：

- 已安装 SUMO，命令行能找到 `netconvert`、`sumo`、`sumo-gui`。
- 已设置 `SUMO_HOME`；若未设置，请把 `$SUMO_HOME/...` 换成本机 SUMO tools 的实际路径。
- 除特别注明外，下面命令从项目根目录执行。

### 生成 SUMO 路网

当前项目使用 `analysis/road_network/basemap.osm.xml` 作为 OSM 输入，输出带红绿灯的 `analysis/road_network/net_tls.net.xml`：

```bash
netconvert --osm analysis/road_network/basemap.osm.xml \
  -o analysis/road_network/net_tls.net.xml \
  --tls.guess-signals true \
  --tls.join true \
  --tls.join-dist 30 \
  --tls.layout opposites \
  --tls.cycle.time 90 \
  --tls.yellow.time 4 \
  --junctions.join --junctions.join-dist 6 \
  --roundabouts.guess false
```

可选验证路网：

```bash
sumo-gui -n analysis/road_network/net_tls.net.xml
```

常用参数含义：

- `netconvert`：SUMO 路网转换工具，把 OSM 转成 SUMO 可仿真的 `.net.xml`。
- `--geometry.remove`：清理多余几何点，减少异常折线。
- `--roundabouts.guess` / `--ramps.guess`：尝试识别环岛和匝道。
- `--junctions.join`：合并非常接近的路口节点。
- `--tls.guess-signals`：根据 OSM 信号灯信息推断 TLS。
- `--tls.join`：合并相邻信号灯控制单元。

### 生成背景车辆需求

20 天的背景随机流由 `analysis/simulation/generate_random_routes.py` 批量生成，它内部就是对每个种子调一次 `randomTrips.py`：

```bash
export SUMO_HOME=/usr/share/sumo          # 指向自己的 SUMO 安装目录
python3 analysis/simulation/generate_random_routes.py            # 20 天，4 并发
python3 analysis/simulation/generate_random_routes.py --days 1    # 先试一天
```

单次调用等价于下面这条命令（`<编号>`、`<种子>` 由脚本填入，`--period 2.88` 对应全天 86400 秒内恰好 3 万条）：

```bash
python3 $SUMO_HOME/tools/randomTrips.py \
  -n analysis/road_network/net_tls.net.xml \
  -o analysis/simulation/data/random_trip/<编号>_<种子>.trips.xml \
  -r analysis/simulation/data/random_trip/<编号>_<种子>.rou.xml \
  --begin 0 \
  --end 86400 \
  --period 2.88 \
  --validate \
  --min-distance 2000 \
  --vehicle-class passenger \
  --prefix random \
  --vtype analysis/simulation/data/random_trip/<编号>_<种子>.vtype.xml \
  --seed <种子>
```

对应的 `vType` 定义：

```xml
<vType id="random_passenger" vClass="passenger" />
```

`randomTrips.py` 常用参数含义：

- `-n`：基于指定路网生成需求。
- `--begin` / `--end`：生成车辆的起止仿真时间。
- `-p` / `--period`：平均发车间隔，数值越小车流越密。
- `--seed`：随机种子，保证可复现。
- `--route-file` / `-r`：输出可直接仿真的 route 文件。

### 跑 SUMO 仿真

正常走 `analysis/simulation/run_simulations.py`，它会先补齐缺的背景流和日需求，再为每天生成 sumocfg 并在当天的文件夹里执行 `sumo -c simulation.sumocfg`：

```bash
python3 analysis/simulation/run_simulations.py            # 完整 20 天
python3 analysis/simulation/run_simulations.py --days 1   # 只准备并跑第 1 天看效果
```

想手工跑一次等价的命令（输出路径相对 cfg 所在目录，这里改成从项目根目录写）：

```bash
sumo -n analysis/road_network/net_tls.net.xml \
  -r analysis/simulation/data/route_demand/<编号>_<种子>.rou.xml \
  -a analysis/road_network/tls_schedule.add.xml \
  --seed <种子> \
  --threads 1 \
  --begin 0 \
  --time-to-teleport 300 \
  --device.rerouting.probability 1 \
  --device.rerouting.period 60 \
  --device.rerouting.adaptation-interval 10 \
  --device.rerouting.adaptation-steps 30 \
  --fcd-output analysis/simulation/data/simulation/<编号>_<种子>/fcd.csv.gz \
  --fcd-output.geo \
  --device.fcd.period 1 \
  --fcd-output.attributes id,x,y,angle,type,speed,pos,lane \
  --output.column-header plain \
  --output.column-separator , \
  --tripinfo-output analysis/simulation/data/simulation/<编号>_<种子>/tripinfo.xml \
  --vehroute-output analysis/simulation/data/simulation/<编号>_<种子>/vehroute.xml \
  --vehroute-output.exit-times \
  --vehroute-output.sorted \
  --vehroute-output.write-unfinished \
  --summary-output analysis/simulation/data/simulation/<编号>_<种子>/summary.xml \
  --statistic-output analysis/simulation/data/simulation/<编号>_<种子>/statistics.xml \
  --duration-log.statistics \
  --no-step-log
```

可选 GUI 检查：

```bash
sumo-gui -n analysis/road_network/net_tls.net.xml \
  -r analysis/simulation/data/route_demand/<编号>_<种子>.rou.xml
```

### 输出转换与复盘

把 FCD XML 转成 CSV：

```bash
python $SUMO_HOME/tools/xml/xml2csv.py fcd_geo.xml > fcd_geo.csv
```

加载保存的仿真状态进行 GUI 复盘：

```bash
cd analysis/simulation

sumo-gui -c basic2_res/run_basic2.sumocfg --load-state=basic2_res/state/state_66600.00.xml.gz --begin=66600
```

或从项目根目录使用项目里的 helper：

```bash
python analysis/simulation/replay.py 105 -p 50
```


## 路径约定

项目级 README 位于仓库根目录；各脚本里的默认路径仍基于 `__file__` 解析，并且只走"模块内部相对路径"或"`analysis/` 内的相对路径"。重新挪动目录时请同时更新：

- 各脚本顶部 `DEFAULT_*` 常量；
- `simulation/run_simulations.py` 里的 `NET_FILE` / `TLS_SCHEDULE_FILE` / `DEMAND_DIR` / `SIMULATION_DIR`（生成的 sumocfg 里的相对路径由它们算出来，不用手改）；
- `matching/README.md` 与本 README 中的命令示例。