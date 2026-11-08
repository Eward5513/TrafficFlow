# Xuancheng26

这一目录只放宣城实验的代码。原始下载和仿真结果在数据盘上，不在这里。

- 只读原始数据：`/data5/tzhang174/Xuancheng26`
- 可写结果：`/data5/tzhang174/TrafficFlow/xuancheng26`

脚本分成两段：

| 目录 | 作用 |
| --- | --- |
| `simulation/` | 改原始出行、生成 SUMO 路线，并跑全天仿真 |
| `model_prep/` | 仿真结束后，把 FCD 做成各模型复现要用的流量、图和窗口 |

路网、信号附加文件、SUMO 配置和批次日志仍留在本目录。当前 3 秒相位仿真和这些脚本都按这个绝对路径读取它们，所以没有挪进子目录。

```text
dataset/xuancheng26/
├── simulation/          原始数据 -> SUMO
├── model_prep/          FCD -> 模型输入
├── xuancheng_tls_rebuilt.net.xml
├── night_peak.add.xml
├── xuancheng_tls_3s.net.xml
├── night_peak_3s.add.xml
├── fcd_configs/         已完成的 29 天批次
└── fcd_configs_3s/      正在跑的 3 秒相位批次
```

## 总流程

```text
只读原始出行 JSON
        |
        v
simulation/  修复断掉的行程，去掉重复日 2023-04-09
        |
        v
prepared/ 路线
        |
        v
refine_routes.py  按人工路网删边、压缩路口，写出 refined 路线
        |
        v
SUMO 全天仿真（人工路网 + 晚高峰 WAUT）
        |
        +---- fcd_data/       已完成的 29 天，相位时长未改
        |
        +---- fcd_data_3s/    只把相位时长改成 3 秒整数倍后重跑
        |
        v
model_prep/  早期 R-level 脚本默认 fcd_data/；多种子 v2 和 M/MR 脚本默认 3s 批次
        |
        v
training_data/  全量流量、抽样流量、路网图、12 步窗口
        |
        v
reimplementation/  各模型训练
```

29 个仿真日是 2023-04-01 到 2023-04-30，去掉与 04-08 重复的 04-09。

## 1. simulation：从原始数据到仿真

`simulation/config/repair_trips.yaml` 指向只读原始树和可写数据树。修复代码不会改 `/data5/tzhang174/Xuancheng26`。

1. `build_sumo_edge_graph.py` 从基础路网抽出边图。
2. `repair_trips.py` 修复走不通的行程。`run_repair_pipeline.py` 把建图、测试、修复和校验串起来。
3. `prepare_sumo_routes.py` 把修复后的 JSON 写成显式 SUMO 路线，输出到 `simulation_data/trip_data/prepared/routes_YYYY_MM_DD.rou.xml`。
4. `refine_routes.py` 按人工改过的 `xuancheng_tls_rebuilt.net.xml` 改这些路线。它删除已经不在路网里的 edge，压缩四个路口内部的连续 edge，再把受影响的车辆改成 `<trip>`。没被改到、而且相邻 edge 仍有连接的车辆保持原来的 `<route>`。脚本不调用 SUMO。输出写在输入文件旁边，名字是 `refined_<输入文件名>`。全天仿真读的是放到 `simulation_data/trip_data/refined/refined_routes_YYYY_MM_DD.rou.xml` 的这批文件。
5. `run_fcd_simulations.py` 用人工调整过的 `xuancheng_tls_rebuilt.net.xml` 和 `night_peak.add.xml` 跑 29 天。结果在 `simulation_data/fcd_data/YYYY_MM_DD/`。
6. `build_tls_3s.py` 只改相位时长：含 `y`/`Y` 的相位改为 3 秒，其余相位向上取整到 3 的倍数。路网拓扑、相位状态和晚高峰切换时间不动。审计在 `tls_3s_audit.md`。
7. `run_fcd_simulations_3s.py` 用新路网重跑 29 天，结果在 `simulation_data/fcd_data_3s/`。

已完成批次和 3 秒批次的 SUMO 参数相同：`--begin 0`，`--end 86400`，`--time-to-teleport 600`，`device.fcd.period=1`，种子 42，每个进程 1 个线程。配置里没有 `device.rerouting`。FCD 写到 `.xml.gz`，SUMO 1.19 由此压缩输出。

```bash
cd /home/tzhang174/TrafficFlow/dataset/xuancheng26/simulation

python3 run_repair_pipeline.py --resume --workers 10
python3 prepare_sumo_routes.py --workers 29
python3 refine_routes.py \
  --input /data5/tzhang174/TrafficFlow/xuancheng26/simulation_data/trip_data/prepared/routes_2023_04_01.rou.xml \
  --net /home/tzhang174/TrafficFlow/dataset/xuancheng26/xuancheng_tls_rebuilt.net.xml

python3 run_fcd_simulations.py --smoke-test
python3 build_tls_3s.py
python3 run_fcd_simulations_3s.py --smoke-test
```

`night_peak.add.xml` 在 55000 秒切到 program 1，75000 秒切回 program 0。`build_tls_3s.py` 生成的 `night_peak_3s.add.xml` 把切入晚高峰改到 54900 秒，75000 秒切回不变。54900 能被 3 整除。

`run_fcd_simulations_tmp.py` 用 `xuancheng_tls_tmp.net.xml` 再跑同一套 29 天。除路网和输出目录外，参数与 3 秒批次相同，包括 `night_peak_3s.add.xml` 和 refined 路线。配置写到 `fcd_configs_tmp/`，结果写到 `simulation_data/fcd_tmp/YYYY_MM_DD/`。它不改 `fcd_data_3s/` 里正在跑的那一批。

## 2. model_prep：从仿真结果到模型输入

这些脚本不改路网，也不改路线。早期 R-level 脚本默认读 `fcd_data/`，写出 `training_data/`；下文的多种子 v2 和 M/MR 脚本默认使用 3 秒批次。换一批仿真时用参数改路径，子目录结构保持不变：

- `--fcd-root`：FCD 日目录
- `--net-file`：生成路网图时用的 SUMO 路网
- `--training-root`：流量、图、窗口和配置的根目录

3 秒相位这批写到 `training_data_3s/`，输入是 `fcd_data_3s/` 和 `xuancheng_tls_3s.net.xml`。

建议顺序：

1. `generate_r_graph.py`：用 `xuancheng_tls_rebuilt.net.xml` 的真实连接做成有向路段图 `graph/r_graph.npz`。
2. `generate_r_level_full_flow.py`：按 5 分钟写出 `inflow.csv`、`outflow.csv`、`accumulation_start.csv`、`accumulation_end.csv`。部分观测由 `generate_r_level_partial_flow.py` 用同样四个文件名写到各渗透率目录。
3. `generate_multi_seed_vehicle_sampling.py`：五个采样种子、七个渗透率的车辆名单。seed 42 复用已经核对过的历史名单。
4. `generate_multi_seed_flow_data.py`：按这些名单写四状态 CSV 和 12→1 窗口。这是当前唯一的流量数据入口。
5. `generate_r_level_partial_flow.py` 和 `generate_r_level_windows.py` 仍可供单种子核对，默认写到 `sampling_seeds/seed_42/flow/`。正式五种子数据不要靠它们重跑。
6. `generate_r_level_sequence_targets.py`（可选，12→12 序列重建）：从全量 CSV 为每天写一份 `shared/sequence_targets_12step/{split}/{date}.npz`（`full_raw [288,1406,4]` float32 原始计数，节点按 `r_graph.npz` 的 `edge_ids`）和 `sequence_manifest.json`。标签 `Y = full[s:s+12]`，与输入 12 步逐步对齐，不是预测；`X` 继续读已有窗口文件，不重写，也不为每个渗透率重复存标签。沿用 `state_normalization.json` 的 target scaler，不重新拟合。脚本逐日核对 7 个渗透率 × 5 个状态文件的 12→1 标签（`y_raw_seq[-1] == y_raw[0]`），只读旧文件并在运行前后比对其大小、mtime 和抽样哈希。支持 `--dates`、`--skip-complete`、`--verify-only`；已存在且与 CSV 不一致的文件需显式 `--overwrite` 才会替换。
7. 模型各自的图，都跟 `r_graph` 的节点顺序：
   - `generate_stgcn_weighted_adjacency.py`
   - `generate_dcrnn_directed_adjacency.py`（DCRNN 和 Graph WaveNet 共用）
   - `generate_stsgcn_undirected_topology.py`
   - `generate_undirected_center_distance.py`
   - `generate_stfgnn_temporal_adjacency.py`（只用训练日的全量流量）
   - `generate_pdformer_relations.py`（只用训练日的全量流量）
8. `generate_training_support_files.py`：节点表、校验报告和各模型配置。
9. `prepare_rf_features.py`：随机森林的辅助特征。它不训练模型。
10. `run_all_models_training.py`：调用 `reimplementation/` 里的训练脚本。

```bash
cd /home/tzhang174/TrafficFlow/dataset/xuancheng26/model_prep

python3 generate_r_graph.py
python3 generate_r_level_full_flow.py --workers 29
python3 generate_multi_seed_vehicle_sampling.py --workers 29
python3 generate_multi_seed_flow_data.py --workers 24
python3 generate_training_support_files.py
python3 generate_r_level_sequence_targets.py --workers 6   # 默认 training_data_3s

python3 generate_r_graph.py \
  --net-file /home/tzhang174/TrafficFlow/dataset/xuancheng26/xuancheng_tls_3s.net.xml \
  --training-root /data5/tzhang174/TrafficFlow/xuancheng26/training_data_3s
python3 generate_r_level_full_flow.py \
  --fcd-root /data5/tzhang174/TrafficFlow/xuancheng26/simulation_data/fcd_data_3s \
  --net-file /home/tzhang174/TrafficFlow/dataset/xuancheng26/xuancheng_tls_3s.net.xml \
  --training-root /data5/tzhang174/TrafficFlow/xuancheng26/training_data_3s \
  --workers 29
```

## 五种子全渗透率数据（3 秒批次）

当前唯一的车辆采样和流量入口是 `training_data_3s/sampling_seeds/`。
训练只用 seed 42。验证和测试分别读取 42–46，再对每份指标取平均。
不要读已经删除的 `training_data_3s/window_data`、`sampled_vehicle` 或 R-level `partial_flow`。



新入口为两个独立阶段，默认通过 `data_root.py` 解析
`xuancheng26/training_data_3s` 和 `xuancheng26/simulation_data/fcd_data_3s`，
支持 `TRAFFICFLOW_DATA_ROOT`。不运行 SUMO、不训练模型，也不生成 Speed。
日期以 `training_data_3s/shared/split_manifest.json` 为准：train 20 天、validation 4 天、test 5 天。
五个采样种子 42–46、七档 p05/p10/p20/p30/p40/p50/p70，共 1015 组。

```text
training_data_3s/sampling_seeds/
├── vehicle_sampling_manifest_v2.json
├── vehicle_sampling_validation_v2.json
├── flow_generation_manifest_v2.json
├── flow_generation_validation_v2.json
├── audit_v2/<date>.json
└── seed_<sampling_seed>/
    ├── sampled_vehicle/<rate>/<date>.csv
    ├── flow/
    │   ├── partial_flow/<rate>/<date>/<state>.csv
    │   └── window_data/
    │       ├── split_manifest.json          # 链接到 training_data_3s/shared/
    │       ├── state_normalization.json
    │       ├── state_window_manifest.json
    │       ├── validation_summary.json
    │       ├── node_mapping.csv
    │       └── <rate>/<split>/<date>/{inflow,outflow,accumulation_start,accumulation_end,all}.npz
    └── graph -> ../../graph
training_data_3s/shared/                     # 五份种子共用的 scaler、manifest、节点表、12→12 标签
```

`generate_multi_seed_vehicle_sampling.py` 只处理车辆身份。每日按字符串排序有效车辆，
使用 `int.from_bytes(SHA256(f"{seed}_{date}")[:8], "big")` 派生种子，
`random.Random(...).shuffle` 后取 `int(N * rate)` 前缀，保存排序后的 ID。
逐日检查七档嵌套、唯一性、有效车辆归属、重复运行确定性和种子间差异。
seed 42 全部历史名单以及 42–46 的 p05 测试名单必须与理论集合逐一相同；
不一致就失败，不覆盖。其他特征以后可以直接复用这些清单，例如另建 `seed_43/speed/`。

`generate_multi_seed_flow_data.py` 必须读到第一阶段全量 PASS 报告，核对 manifest 和
全部车辆文件 SHA256 后才处理 Flow；它不进行随机采样。一天一个进程，缺数据时只流式扫描
一次 FCD，复用 `generate_r_level_full_flow.stream_fcd` 的真实道路、internal lane、
Entry/Exit、到达补齐规则，读取缓冲为 16 MiB，不缓存完整轨迹。
事件先计入每个种子的最低待生成渗透率层级，再沿渗透率轴累加。
四状态保持 288×1406、整数守恒、零日初存量、连续存量，并逐格检查非负、partial≤full、
渗透率单调和 R 图列顺序。

窗口为同日 12→1，共 277 个，目标是最后输入时间片的全量状态。
四通道顺序固定为 `[inflow,outflow,accumulation_start,accumulation_end]`。
保留 `x,y,x_raw,y_raw,day_index,window_start_slot,target_slot` 的原 dtype、shape 和语义。
所有种子的同一渗透率都使用 seed 42 原训练集 input scaler；所有目标共用原 full target scaler，
不拟合任何新 scaler。每一个 NPZ 的所有数组都会与 CSV 推导值精确比较，因此全量标签在
所有种子和渗透率间相同。`day_index` 仍是 Monday=0 的星期索引。

全量 CSV 和图只引用权威 `training_data_3s/full_flow/r-level`、`training_data_3s/graph`，
不在各 seed 下复制；NPZ 内保留读取接口要求的 `y/y_raw` 数组。
五个种子的窗口元数据都链接到 `training_data_3s/shared/`，不各存一份 scaler。
新实验只读 `seed_<id>/flow/window_data`、`seed_<id>/flow/partial_flow` 和
`seed_<id>/sampled_vehicle`。旧的顶层 `window_data`、`sampled_vehicle`、
R-level `partial_flow`，以及 `seed_*/window_data` 和 `seed_*/partial_flow`，
在核对进新目录之后已经删除。`full_flow/`、`graph/`、`config/`、M/MR 流量和
更早的 `xuancheng26/training_data/` 保留。

从仓库根目录运行：

```bash
python dataset/xuancheng26/model_prep/generate_multi_seed_vehicle_sampling.py --workers 29
python dataset/xuancheng26/model_prep/generate_multi_seed_flow_data.py --dates 2023_04_01 --workers 1
python dataset/xuancheng26/model_prep/generate_multi_seed_flow_data.py --workers 24
```

默认校验并跳过已存在数据，不需要 `--skip-complete`。中断后重复同一命令可继续，
缺失 CSV 的日期才扫描 FCD；只有缺窗口时直接由 CSV 补齐。也可用 `--dates` 仅重试失败日期，
然后运行全日期命令汇总验收。`--verify-only` 不生成 CSV/NPZ，但会更新 v2 审计报告。
子集报告写 `flow_generation_subset_v2.json`，不会将部分完成冒充完整成功。
运行时持有共享阶段锁，按日期隔离输出。新数据先写临时文件、回读校验后以原子且不覆盖
已有目标的方式发布；任何已有数据不一致都报错并保留原文件。
仅 v2 manifest/审计元数据允许原子更新。
磁盘按最多 300 GiB 新 Flow/窗口预估并保留额外空间；并发根据 CPU affinity 和可用内存限制，
不给 GPU 分配任务，给训练预留 CPU/内存。日志可通过 `tee` 保存到 `sampling_seeds/logs_v2/`。

## 3. M-level（Movement）数据

M-level 只做 3 秒批次，三个脚本默认就读 `fcd_data_3s/`、`xuancheng_tls_3s.net.xml`、`night_peak_3s.add.xml`，写到 `training_data_3s/`。它们不改 R-level 文件，只读 `graph/r_graph.npz` 做映射核对，读 R-level 全量总数写进 summary 作对照。

`fcd_data_3s/` 的 FCD 头写的是 `xuancheng_tls_tmp.net.xml` 和 `fcd_tmp/`。这两个路网文件逐字节相同，lane id 可以直接对上。

一个 Movement 是 `(junction, incoming real edge, outgoing real edge)`，ID 为 `M::<junction>::<in>::<out>`。同一三元组的所有 lane-level `<connection>` 合成一个节点，车道号、via 车道（含内部路口的第二段 via）、`tl`、`linkIndex`、`state` 按 lane connection 全部保留。节点按入边、再按出边在 R 图里的序号排序。`m_i -> m_j` 当且仅当 `out(m_i) == in(m_j)`，中间道路就是这条共享 edge，`free_flow_travel_time = 车道平均长度 / 车道平均限速`。

```text
training_data_3s/
├── graph/
│   ├── m_graph.npz              movement_ids、adjacency(uint8)、edge_index[2,E]、via 车道映射、graph_signature
│   ├── m_nodes.csv  m_transitions.csv
│   ├── m_signal_connections.csv m_signal_programs.json   TLS/linkIndex 多对多、各 program 的逐相位灯色、WAUT
│   └── m_graph_audit.json  m_graph_audit.md              度、连通分量、via 覆盖、信号覆盖、与 r_graph 的逐对核对
├── full_flow/m-level/YYYY_MM_DD/{inflow,outflow,accumulation_start,accumulation_end}.csv
│   ├── generation_summary.csv   day_stats/YYYY_MM_DD.json   anomalies/YYYY_MM_DD.csv
├── partial_flow/m-level/p05..p70/YYYY_MM_DD/  同样四个文件
│   └── generation_summary.csv   day_stats/YYYY_MM_DD.json
├── full_flow/mr-level/YYYY_MM_DD/  partial_flow/mr-level/p05..p70/YYYY_MM_DD/   MR 拼接数据，见第 5 节
```

CSV 与 R-level 同格式：`time_bin,start_time,<movement_id...>`，288 行，每格 300 秒，列顺序就是 `m_graph.npz` 的节点顺序。四个量都描述 Movement 连接区（via 车道），不是入边或出边上的存量：

- `inflow`：车辆第一次出现在该 Movement 的任一 via 车道。
- `outflow`：车辆随后第一次出现在该 Movement 的出边。
- `accumulation_*`：窗口起点/终点仍在连接区内的车辆数，`N_end = N_start + inflow - outflow`，`N_start[0] = 0`。

FCD 是 1 秒一帧，短连接常被整段跳过。车辆从入边直接出现在出边时，在出现时刻同一格记 1 次 inflow 和 1 次 outflow，不产生存量（`direct_transitions`）。一天结束仍在连接区的车，只有 tripinfo 的 `arrivalLane` 正是该 Movement 出边时，才在 arrival 时刻记 outflow（`final_exits`），否则留作 `unresolved`，不补 outflow。驶出到非预期道路（多为 teleport）记 outflow 并计入 `mismatched_exits`，从非预期道路驶入记 `mismatched_entries`，两条不相连的真实道路之间的跳变不分配 Movement，记 `unmapped_transitions`。逐条记录在 `anomalies/`。

部分观测只扫一遍 FCD，只跟踪 p70 车辆，每个事件写入车辆所属最低档及以上各档。逐 Movement、逐时间片检查 `partial <= full`（流量和存量）以及 p05 到 p70 的嵌套。

```bash
cd /home/tzhang174/TrafficFlow/dataset/xuancheng26/model_prep

python3 generate_m_graph.py
python3 generate_m_level_full_flow.py --workers 29 --skip-complete
python3 generate_m_level_partial_flow.py --workers 29 --skip-complete
```

`--skip-complete` 只在 `day_stats` 为 PASS、图签名一致、四个文件重读后守恒都通过时跳过该日。全量和部分观测不要同时跑，部分观测依赖全量结果。

## 4. MR（R-M 异构）图

`generate_mr_graph.py` 只读 `graph/r_graph.npz`、`graph/m_graph.npz`（及 `m_nodes.csv`、`m_transitions.csv`、`r_nodes.csv` 做一致性核对），不读路网和 FCD，不改任何已有文件。默认 `--training-root` 是 `training_data_3s/`；已有 `mr_*` 输出时需加 `--force`。所有文件先写临时文件，重读校验通过后才原子替换；校验失败时只写审计文件，退出码 2。

节点：先全部 R 节点（`R::<edge_id>`，global 0..R-1，顺序同 `r_graph` 的 `edge_ids`），再全部 M 节点（沿用 `M::<junction>::<in>::<out>`，global R..R+M-1，顺序同 `m_graph` 的 `movement_ids`）。`node_types`：0=R，1=M。`node_local_indices[g]` 就是该节点在 R-level 或 M-level 流量 CSV 中 `time_bin,start_time` 之后的列序号，列名等于 `node_original_ids[g]`；脚本逐个核对了 `full_flow/` 和 `partial_flow/` 下两级、全部渗透率、全部日期的四个 CSV 表头。R 与 M 的流量/存量物理含义不同，这里不合并。

四类有向关系分别编码，按 0..3 连续存放（`relation_edge_offsets`），不做无向化、不加自环：

| 编码 | 关系 | 来源 | 3 秒批次边数 |
| --- | --- | --- | --- |
| 0 | R->R | `r_graph` 邻接，行优先 | 3877 |
| 1 | M->M | `m_graph` 的 `edge_index`，顺序不变，可直接对齐 `transition_*` 数组 | 10551 |
| 2 | R->M | Movement 的入边 -> Movement，每个 Movement 恰一条 | 3877 |
| 3 | M->R | Movement -> 出边，每个 Movement 恰一条 | 3877 |

```text
training_data_3s/graph/
├── mr_graph.npz         node_ids/node_types/node_local_indices/node_original_ids、edge_index[2,E]、edge_type[E]、
│                        edge_type_names、r_r/m_m/r_m/m_r_edge_index（均为 global index）、
│                        edge_road_global_index/edge_movement_global_index（每条边关联的道路和 Movement，无则 -1）、
│                        m_incoming/outgoing_r_global_index、r_downstream_m_ptr/_global_index、r_upstream_m_ptr/_global_index（CSR）、
│                        输入 sha256、m_graph_signature、graph_signature；无需 allow_pickle
├── mr_nodes.csv         每个节点的 global/local index、流量层级与列、M 的路口/入边/出边、R 的上下游 Movement、按关系拆分的入/出度
├── mr_edges.csv         每条边的 source/target（global、ID、类型、local）、relation_type/name、关联道路与 Movement
└── mr_graph_audit.json  mr_graph_audit.md   逐边核对、两跳 R->M->R 与 R->R 一致性、度、孤立点、弱/强连通分量、流量列映射
```

`mr_edges.csv` 的关联字段：R->R 记对应的唯一 Movement（A->B），M->M 记共享的中间道路，R->M / M->R 同时记道路和 Movement。原 M 图的孤立 Movement 在 MR 图中经 R->M、M->R 接通，属正常现象。

## 5. MR-level 四状态数据

MR-level 不产生新的物理量：每天、每个状态把 R-level 矩阵（1406 列道路）和 M-level 矩阵（3877 列 Movement）按 `mr_graph.npz` 的节点顺序直接按列拼接，`X_MR[t] = [X_R[t], X_M[t]]`。不读 FCD，不重新抽样，不归一化。CSV 仍是 `time_bin,start_time,<node_ids...>`，288 行，5283 个数据列，列名就是 MR 的 `node_ids`：前 1406 列 `R::<edge_id>`，后 3877 列 Movement ID。

R 列是道路存量，M 列是连接区（via 车道）存量，两者不能相加，也不要求 R、M 总量相等。守恒只逐节点检查：`N_end = N_start + inflow - outflow`、`N_start[t+1] = N_end[t]`、`N_start[0] = 0`、无负值。

| 脚本 | 输入 | 输出 |
| --- | --- | --- |
| `generate_mr_level_full_flow.py` | `full_flow/r-level/<date>/`、`full_flow/m-level/<date>/` | `full_flow/mr-level/<date>/` |
| `generate_mr_level_partial_flow.py` | `partial_flow/<rate>/<date>/`（R-level，没有 `r-level` 中间层）、`partial_flow/m-level/<rate>/<date>/`、`full_flow/mr-level/` | `partial_flow/mr-level/<rate>/<date>/` |

两个脚本都默认读 `training_data_3s/`，一天一个 worker（默认 `--workers 29`）；部分观测的 worker 依次处理当天七档。每个文件先写 `.tmp` 再原子替换，然后严格重读：表头、0..287 时间片、非负整数，R 块和 M 块与输入逐格相等，逐节点守恒。部分观测还逐格检查四个状态的 `partial <= full MR` 和 `p05 <= p10 <= ... <= p70`。`--skip-complete` 只跳过 `day_stats` 为 PASS、图签名一致、且重读后与输入逐格相等的天（或档）；`--dates` 可单独重跑失败的日期。summary 由主进程汇总到各自的 `generation_summary.csv`（全量 29 行，部分观测 29×7 行）。`mr_level_metadata.json` 记录节点数、索引范围、两类节点四个状态的含义、列顺序（附表头 sha256）、数据来源、时间粒度和渗透率定义。全量要先跑完，部分观测依赖它。

```bash
cd /home/tzhang174/TrafficFlow/dataset/xuancheng26/model_prep
python3 generate_mr_graph.py --training-root /data5/tzhang174/TrafficFlow/xuancheng26/training_data_3s
python3 generate_mr_level_full_flow.py --training-root /data5/tzhang174/TrafficFlow/xuancheng26/training_data_3s --workers 29 --skip-complete
python3 generate_mr_level_partial_flow.py --training-root /data5/tzhang174/TrafficFlow/xuancheng26/training_data_3s --workers 29 --skip-complete
```

原来那一批的训练结果在 `/data5/tzhang174/TrafficFlow/xuancheng26/training_results/`。3 秒这批的配置把 `output_root` 指到 `/data5/tzhang174/TrafficFlow/xuancheng26/training_results_3s/`，`run_all_models_training.py` 也读 `training_data_3s` 并写到这个新目录。
