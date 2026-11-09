# 数据盘目录清单

审计日期：2026-11-09。本文只记录数据盘上当时实际存放的路径和文件，不按代码里的目标布局改写。代码在 `/home/tzhang174/TrafficFlow`，数据在 `/data5/tzhang174/TrafficFlow`。`data_root.py` 里的 `ocr_data/`、`simulation_data/`、`training_data/`、`training_results/` 是旧的平铺前缀；磁盘根下没有这四个目录，实际先按城市分开。

体积用 `du` 统计，并四舍五入。`training_data_3s` 里 `archive/`、`datasets/`、`full_flow/` 之间有硬链接，子目录体积不能相加。整棵 `training_data_3s` 的独立占用是 800G。

```text
/data5/tzhang174/TrafficFlow          2.5T
├── shanghai_sim                      122G
└── xuancheng26                       2.4T
    ├── simulation_data               1.1T
    ├── training_data                  18G
    ├── training_data_3s              800G
    ├── training_results               19G
    └── training_results_3s           503G
```

宣城原始出行不在这棵树里，在同级目录 `/data5/tzhang174/Xuancheng26`（`trips/`、`network/`、`metadata/` 等）。本清单不展开那棵只读原始树。

两套实验都是 5 分钟一个时间片、一天 288 片、用连续 12 步重建最后一步的全量状态。渗透率目录统一是 `p05`、`p10`、`p20`、`p30`、`p40`、`p50`、`p70`。

---

## 上海：`shanghai_sim/`（122G）

20 个仿真日，子图 56 条 R 道路。日目录名是 `day_01` … `day_20`，也对应仿真目录 `<编号>_<种子>`。划分在 `training_data/r-only/split_manifest.json`：训练 `day_01`–`day_14`（3878 个窗口），验证 `day_15`–`day_17`（831），测试 `day_18`–`day_20`（831）。每天 277 个窗口。

```text
shanghai_sim/
├── ocr_data/                 13G
├── simulation_data/         109G
├── training_data/           218M
├── training_results/        778M
└── analysis_results/        2.3M
```

### `ocr_data/`

远程桌面截图和 OCR 文本，平铺编号文件，没有再按日期分层。

| 目录 | 内容 | 数量 |
| --- | --- | --- |
| `snapshoot/` | `000001.png` 这类原始截图 | 1049 张 png |
| `ocr_output/` | `NNNNNN.txt`，以及 `NNNNNN_crop_marked.png`、`NNNNNN_crop_preview.png` | 1062 个 txt，2098 张 png |
| `snapshoot_all/` | 全量截图，同样是编号 png | 8444 张 |
| `ocr_output_all/` | 编号 txt 和 `NNNNNN_crop_preview.png`；另有 `log/` | 8451 个 txt，8444 张 png |

`ocr_output_all/log/` 里是三份文本日志：`all_ocr_no_blank_lines_refine_log.txt`、`invalid_line_details.txt`、`route_by_edge_build_log.txt`。

### `simulation_data/`

| 路径 | 体积 | 存放什么 |
| --- | --- | --- |
| `random_trip/` | 596M | 20 天背景随机流。每天三份：`<编号>_<种子>.{trips,rou,vtype}.xml`。`logs/` 下每天一份 `.log` |
| `route_demand/` | 4.0G | 20 份日需求 `<编号>_<种子>.rou.xml`，每份约 204MB，是仿真真正加载的路线 |
| `simulation/` | 95G | 20 个已跑完的日目录 |
| `processed/` | 9.8G | 从全量 FCD 抽出的子图轨迹和观测流量 |
| `subgraph.txt` | 693B | 56 行 SUMO edge id，子图道路名单 |

20 个仿真目录是 `1_428228`、`2_465872`、`3_217895`、`4_895787`、`5_859336`、`6_203516`、`7_173543`、`8_379780`、`9_415694`、`10_175`、`11_446447`、`12_583682`、`13_174157`、`14_110191`、`15_609497`、`16_650161`、`17_620690`、`18_86957`、`19_200008`、`20_250495`。每个目录七个文件：

```text
simulation/<编号>_<种子>/
├── simulation.sumocfg
├── fcd.csv.gz          逐秒 FCD，单日约 4GB
├── tripinfo.xml
├── vehroute.xml
├── summary.xml
├── statistics.xml
└── simulation.log
```

`1_428228/simulation.sumocfg` 把路网写成相对路径 `../../../../road_network/net_tls.net.xml`，附加文件写成 `../../../../road_network/tls_schedule.add.xml`。按当前数据盘根目录，这个相对路径不会落到 `shanghai_sim` 里面；数据盘根下也没有 `road_network/`。

`processed/subgraph_trajectories/` 有 `day_01` … `day_20`，以及跨日汇总：`extract_1_to_20.log`、`extract_day1.log`、`observed_edge_flow_run.log`、`sampling_summary_all_days.csv`、`observed_edge_flow_5min_summary_all_days.csv`、`validation_summary_all_days.csv`、`edge_flow_rank_5min_summary.csv`、两张汇总 png、`observed_edge_flow_5min_status.json`。

每一天的内部是：

```text
day_01/
├── trajectories.csv                         约 476MB
├── vehicle_ids.txt
├── edge_flow_5min.csv                       列：window_index, window_start, time_label, edge_id, vehicle_count
├── edge_flow_rank_5min.csv
├── edge_plot_index_5min.csv
├── edge_flow_compare_5min.png
├── edge_flow_mean_5min.png
├── validation_summary.json
├── edge_flow_plots_5min/                    56 张按 edge id 命名的 png
├── sampled_vehicle_ids/
│   ├── vehicles_p05.txt … vehicles_p70.txt
│   └── sampling_summary.json
└── observed_edge_flow_5min/
    ├── edge_flow_5min_p05.csv … edge_flow_5min_p70.csv
    └── validation_summary.json
```

### `training_data/`

```text
training_data/
├── r-only/          共享 56 节点窗口、scaler、STGCN/DCRNN 图
├── pdformer/        PDFormer 关系矩阵
└── stfgnn/          STFGNN 时间图
```

`r-only/` 根上是 `dataset_metadata.json`、`normalization.json`、`split_manifest.json`、`validation_summary.json`、`node_mapping.csv`、`sample_index_{train,validation,test}.csv`、`preparation_log.jsonl`、`preparation_status.json`、`prepare_run.log`。`node_mapping.csv` 的列是 `node_index,node_id,node_type,edge_id,from_junction,to_junction,num_lanes,length,speed,priority,edge_type`，`node_id` 形如 `R::850154335`，共 56 行。

`r-only/p05` … `p70` 每个只有三份合并窗口：`train.npz`、`validation.npz`、`test.npz`。`p05/train.npz` 的数组是：

| 键 | 形状 | 类型 |
| --- | --- | --- |
| `x`, `x_raw` | `(3878, 12, 56, 1)` | float32 |
| `y`, `y_raw` | `(3878, 1, 56, 1)` | float32 |
| `day_index`, `window_start_slot`, `target_slot` | `(3878,)` | int32 |

`r-only/adjacency_matrix/` 是 STGCN 用的无向加权图：`stgcn_weighted_adjacency.npy`、`stgcn_weighted_adjacency_sparse.npz`、`stgcn_center_distance.npy`、`stgcn_undirected_topology.npy`、`stgcn_weighted_edges.csv`，以及 metadata / validation json。

`r-only/dcrnn/` 是有向图：`dcrnn_adj_mx.pkl`、`dcrnn_weighted_adjacency.npy`、`dcrnn_weighted_adjacency_sparse.npz`、`dcrnn_directed_topology.npy`、`dcrnn_directed_center_distance.npy`、`dcrnn_directed_edges.csv`，以及 metadata / validation json。

`pdformer/` 八个文件：`pdformer_{dtw_distance,geo_mask,semantic_mask,laplacian_pe,pattern_keys,shortest_path}.npy`，加 `pdformer_relations_metadata.json`、`pdformer_relations_validation.json`。

`stfgnn/` 四个文件：`dtw_distance.npy`、`temporal_adjacency.npy`、`temporal_adjacency.csv`、`temporal_graph_metadata.json`。

### `training_results/`

八个模型各一套七档渗透率，外加启动日志：

```text
training_results/
├── _launch/          dcrnn.log、graph_wavenet.log、pdformer.log、staeformer.log、stfgnn.log、stgcn.log、stid.log、stsgcn.log
├── dcrnn/
├── graph_wavenet/
├── pdformer/
├── staeformer/
├── stfgnn/
├── stgcn/
├── stid/
└── stsgcn/
```

每个模型都是 `p05` … `p70/seed_42/`。一次运行的常见文件是 `best_checkpoint.pt`、`last_checkpoint.pt`、`resolved_config.json`、`run_metadata.json`、`training_history.json`、`test_metrics.json`、`test_predictions.npz`、`validation_metrics.json`、`validation_predictions.npz`。STGCN、DCRNN、STSGCN、STFGNN 还写了 `day_metrics.csv`、`node_metrics.csv`、`training_history.csv`、`run_validation.json`。Graph WaveNet 多一个 `learned_adaptive_adjacency.npy`。

### `analysis_results/`

八个模型在上海 56 条路上的汇总表，不是训练中间产物。根上是 csv / json：`final_results_56.csv`、`model_rankings.csv`、`model_level_summary.csv`、`model_efficiency_summary.csv`、`completion_audit.csv`、`convergence_summary.csv`、`mae_penetration_trends.csv`、`edge_wape.csv`、`edge_bias.csv`、`hourly_wape.csv`、`hourly_bias.csv`、`flow_bin_metrics.csv`、`zero_flow_metrics.csv`、`nonzero_flow_metrics.csv`、`signed_bias_summary.csv`、`validation_test_gap.csv` 等。

`figures/` 是八张 png：渗透率对 MAE / WAPE、p05 与 p70 的小时 WAPE、p05 道路 WAPE 箱线图、精度-耗时散点、平均训练时间和每 epoch 秒数。

`historical_mean/` 是历史均值基线：`historical_mean_template.npy`、`historical_mean_predictions.npz`、`historical_mean_metrics.json`、若干 csv 和 `report.md`。

---

## 宣城：`xuancheng26/`（2.4T）

29 个仿真日，日期是 2023-04-01 到 2023-04-30，没有 2023-04-09。R 级节点 1406。划分在 `training_data_3s/shared/split_manifest.json`，旧的 `training_data/window_data/split_manifest.json` 用的是同一组日期：

| 划分 | 日期 | 天数 | 窗口数 |
| --- | --- | --- | --- |
| train | 04-01 … 04-08、04-10 … 04-21 | 20 | 5540 |
| validation | 04-22 … 04-25 | 4 | 1108 |
| test | 04-26 … 04-30 | 5 | 1385 |

每天仍是 277 个同日窗口。`day_index` 按周一为 0 的星期编码，缺的 04-09 不把后面的日期顺延。

### `simulation_data/`（1.1T）

```text
simulation_data/
├── road_network/      12M
├── trip_data/         11G
├── fcd_data/         131G    只有压缩 FCD
└── fcd_data_3s/      916G    压缩包和未压缩 XML 都在
```

#### `road_network/`

| 目录 | 文件 |
| --- | --- |
| `base/` | `xuancheng.net.xml`（约 5.0MB） |
| `cache/` | `bridge_cache.pkl`、`bridge_cache_meta.json` |
| `topology/` | `sumo_graph.pkl`、`adjacency.json`、`edge_connections.json`、`lane_connections.jsonl`、`normal_edges.json`、`graph_stats.json` |

全天仿真实际读的路网在代码树 `dataset/xuancheng26/`，不在这个 `base/` 里。见下面 FCD 批次记录。

#### `trip_data/`

29 天修复后的行程。`tmp/` 是空目录。

| 目录 | 文件 |
| --- | --- |
| `repaired/` | `data_YYYY_MM_DD_repaired.json`，单日约 127–174MB |
| `repair_metadata/` | `data_YYYY_MM_DD_metadata.jsonl.gz` |
| `rejected/` | `data_YYYY_MM_DD_rejected.jsonl.gz` |
| `prepared/` | `routes_YYYY_MM_DD.rou.xml`；另有空的 `.tmp/` |
| `refined/` | `refined_routes_YYYY_MM_DD.rou.xml`，全天仿真加载的是这批 |
| `manifests/` | `manifest_YYYY_MM_DD.json`，以及 `input_hashes_before.json`、`input_hashes_after.json` |
| `reports/` | `trip_repair_report.md`、`daily_repair_statistics.csv`、`repair_action_distribution.csv`、`rejected_reason_counts.csv`、`top_deleted_edges.csv`、`top_inserted_paths.csv`、`source_and_output_manifest.json`、`validation_summary.json`、`repair_trips.log`、`unit_test_results.txt` |

#### `fcd_data/`

29 天，目录名 `2023_04_01` 这种下划线日期。`batch_status.json` 写于 2026-10-13，种子 42，29 个进程。根上还有 `cleanup_actions.log`。每一天：

```text
2023_04_01/
├── fcd_2023_04_01.xml.gz      约 4.6–5.6GB，没有未压缩 xml
├── tripinfo_2023_04_01.xml.gz
├── summary_2023_04_01.xml.gz
├── statistics_2023_04_01.xml.gz
├── sumo.log
├── console.log
└── error.log
```

这份 `batch_status.json` 没有写下 `net` 和 `fcd_root` 字段。

#### `fcd_data_3s/`

同样 29 天。根上有 `batch_status.json`、`batch_summary.json`、`batch_summary.md`。每一天比上一批多了未压缩 FCD 和 `day_result.json`：

```text
2023_04_01/
├── fcd_2023_04_01.xml         约 28GB
├── fcd_2023_04_01.xml.gz      约 4.6GB
├── tripinfo_2023_04_01.xml.gz
├── summary_2023_04_01.xml.gz
├── statistics_2023_04_01.xml.gz
├── day_result.json
├── sumo.log
├── console.log
└── error.log
```

29 天都同时有 `.xml` 和 `.xml.gz`。未压缩 XML 是这 916G 的主体。

这批文件现在的目录名是 `fcd_data_3s`，但 `day_result.json` 和两份 batch 摘要记录的产出路径是 `simulation_data/fcd_tmp/`。磁盘上已经没有 `fcd_tmp/`。`2023_04_01/day_result.json` 记录：2026-10-27 跑完，路网 `dataset/xuancheng26/xuancheng_tls_tmp.net.xml`，附加文件 `dataset/xuancheng26/night_peak_3s.add.xml`，路线 `trip_data/refined/refined_routes_2023_04_01.rou.xml`，种子与批次摘要一致为 42，`loaded_rerouting` 为 false。`event_breakdown.json` 后来读的是现在这个 `fcd_data_3s/.../fcd_*.xml.gz`。

### `training_data/`（18G）

这是较早的 R 级、单通道 inflow 包，对应 `dataset_metadata.json` 里的 `xuancheng26_r_level_reconstruction`。输入是渗透后的 R 级 inflow，目标是全量 inflow，节点 1406，历史 12，预测 1。

```text
training_data/
├── dataset_metadata.json
├── validation_summary.json
├── full_flow/            54M
├── partial_flow/        342M
├── sampled_vehicle/     481M
├── graph/                40M
├── window_data/          16G
└── config/              1.0G
```

`full_flow/r-level/YYYY_MM_DD/` 只有 `entry_flow.csv` 和 `exit_flow.csv`。列是 `time_bin,start_time,<edge_id>…`，288 行、1406 条边。旁边有 `generation_summary.csv`。

`partial_flow/pXX/YYYY_MM_DD/` 是 `inflow.csv` 和 `out_flow.csv`（文件名是 `out_flow`，不是 `outflow`）。根上有 `generation_summary.csv`。

`sampled_vehicle/pXX/YYYY_MM_DD.csv` 一列 `vehicle_id`，值如 `veh_20230401_000000023`。根上有 `sampling_summary.csv`。

`graph/` 九个文件，节点顺序跟 `r_graph.npz` 的 `edge_ids`：

| 文件 | 作用 |
| --- | --- |
| `r_graph.npz` | 有向 0/1 邻接，`edge_ids (1406,)`、`adjacency (1406, 1406) uint8` |
| `r_nodes.csv`、`node_mapping.csv` | 同一份 1406 行节点表 |
| `stgcn_weighted_adjacency.npy` | STGCN 加权邻接 |
| `dcrnn_weighted_adjacency.npy`、`dcrnn_adj_mx.pkl` | DCRNN / Graph WaveNet 有向加权图 |
| `stsgcn_undirected_topology.npy` | STSGCN 无向拓扑 |
| `undirected_center_distance.npy`、`undirected_center_distance_metadata.json` | 无向中心距离 |

`window_data/` 根上有 `dataset_metadata.json`、`merge_summary.json`、`node_mapping.csv`、`normalization.json`、`split_manifest.json`、`validation_summary.json`。其下 `p05` … `p70` 各有：

```text
p05/
├── normalization.json
├── train.npz / validation.npz / test.npz     按划分合并
└── train/ validation/ test/
    └── YYYY_MM_DD.npz                        按天拆开
```

单日 `2023_04_01.npz` 与上海包同一组键，形状是 `x/x_raw (277, 12, 1406, 1)`、`y/y_raw (277, 1, 1406, 1)`，索引三列长度 277。

`config/` 十个模型目录：

| 目录 | 实际文件 |
| --- | --- |
| `stgcn`、`dcrnn`、`gwn`、`stid`、`staeformer`、`s2s_gcn_lstm`、`stsgcn` | 各一份 `r_only_<model>.json` |
| `stfgnn` | `r_only_stfgnn.json`、`temporal_adjacency.npy`、`temporal_adjacency.before_activity_filter.npy`、`temporal_activity_per_road.csv`、两份 activity json |
| `pdformer` | `r_only_pdformer.json` 和 `pdformer_{dtw_distance,laplacian_pe,pattern_keys,shortest_path}.npy`、`pdformer_relations_metadata.json` |
| `RF` | `calendar.csv`、`road_attributes.csv`、`feature_schema.json`、`preprocessing_summary.json`，以及 `p05` … `p70` |

### `training_data_3s/`（800G）

3 秒相位这批仿真之后的训练包。现行四状态定义写在 `shared/physical_definition.json`，版本名 `independent_fcd_v1`：inflow / outflow 是直接进出事件，accumulation 是边界时刻的道路占有快照。5 分钟箱、288 箱。通道顺序是 `inflow, outflow, accumulation_start, accumulation_end`。

```text
training_data_3s/
├── sampling_seeds -> datasets/fixed_uniform/sampling_seeds
├── datasets/                 现行固定渗透率和可变渗透率
├── shared/                   划分、节点表、目标 scaler、12 步序列标签
├── full_flow/                R / M / MR 全量四状态
├── partial_flow/             只有 M 和 MR 的分渗透率 CSV
├── graph/                    R 图，以及 M、MR、ASTTN、STJGCN
├── config/                   模型配置和部分模型自己的图
├── archive/cumsum_stock_v1/ 目录名是旧口径；和现行树有硬链接
├── .staging/independent_fcd_v1/
├── architecture_audit/
├── dataset_metadata.json
├── validation_summary.json
├── cleanup_manifest.json
├── cleanup_report.json
└── .dataset_layout_migration.lock
```

`sampling_seeds` 是符号链接，不是第二份数据。

#### `datasets/fixed_uniform/`（固定渗透率）

`dataset_manifest.json`：`dataset_id=fixed_uniform`，算法 `xuancheng26_3s_multi_seed_v2`，物理定义 `independent_fcd_v1`，采样种子 42–46，观测键就是七档 `p05` … `p70`。`shared/` 里 `state_normalization.json`、`state_window_manifest.json`、`validation_summary.json` 是本目录文件；`node_mapping.csv`、`physical_definition.json`、`split_manifest.json`、`target_normalization.json`、`sequence_targets_12step` 链接到 `training_data_3s/shared/`。

```text
sampling_seeds/
├── seed_42 … seed_46
├── audit_v2/YYYY_MM_DD.json
├── logs_v2/
├── vehicle_sampling_manifest_v2.json
├── vehicle_sampling_validation_v2.json
├── flow_generation_manifest_v2.json
├── flow_generation_validation_v2.json
├── flow_generation_subset_v2.json
├── completion_audit_v2.json
├── completion_report_v2.md
├── sampling_manifest.json
├── legacy_inventory_before_v2.json
├── reader_validation_v2.json
└── baselines / staeformer 的 p05 sampling mae json
```

每个种子：

```text
seed_42/
├── graph -> ../../../../graph
├── sampled_vehicle/pXX/YYYY_MM_DD.csv          一列 vehicle_id
├── model_prepared/RF/pXX/YYYY_MM_DD.npz        29 天都有
└── flow/
    ├── partial_flow/pXX/YYYY_MM_DD/
    │   ├── inflow.csv
    │   ├── outflow.csv
    │   ├── accumulation_start.csv
    │   └── accumulation_end.csv
    └── window_data/
        ├── node_mapping.csv 等五个共享文件的符号链接
        └── pXX/{train,validation,test}/YYYY_MM_DD/
            ├── inflow.npz
            ├── outflow.npz
            ├── accumulation_start.npz
            ├── accumulation_end.npz
            └── all.npz
```

单通道 `inflow.npz`：`x/x_raw (277, 12, 1406, 1)`，`y/y_raw (277, 1, 1406, 1)`。`all.npz` 最后一维是 4，通道顺序与 `physical_definition.json` 相同。键还包括 `day_index`、`window_start_slot`、`target_slot`。

R 级分渗透率 CSV 和窗口只在这条 `datasets/.../flow/` 下面。`training_data_3s/partial_flow/` 没有 R 级。

#### `datasets/variable_*`

四套可变渗透率，结构相同，观测键都是一个目录 `p10_p20`，种子仍是 42–46。`dataset_manifest.json` 写明 `distribution=uniform_continuous`，`schedule_seed=42`，物理定义仍是 `independent_fcd_v1`。

| 目录 | `mode` | 每段秒数 | 渗透率范围 |
| --- | --- | --- | --- |
| `variable_day_p10_p20` | `day` | 86400 | 0.10–0.20 |
| `variable_2h_p10_p20` | `2h` | 7200 | 0.10–0.20 |
| `variable_1h_p10_p20` | `1h` | 3600 | 0.10–0.20 |
| `variable_30min_p10_p20` | `30min` | 1800 | 0.10–0.20 |

每套内部：

```text
<dataset_id>/
├── dataset_manifest.json
├── penetration_schedule.json
├── audit/                 days/ 和 stage_*.json；30 分钟这套 audit/days 是空的
├── logs/
├── shared/                与 fixed_uniform 一样，混合本地 scaler 和指向 ../../shared 的链接
└── sampling_seeds/seed_<id>/
    ├── sampled_vehicle/p10_p20/
    ├── flow/partial_flow/p10_p20/
    ├── flow/window_data/p10_p20/{train,validation,test}/YYYY_MM_DD/{inflow,outflow,accumulation_start,accumulation_end,all}.npz
    └── model_prepared/
```

`variable_day` 的 `seed_42/flow/window_data/p10_p20/train/2023_04_01/` 已确认是这五个 npz。

#### `shared/`

| 文件或目录 | 内容 |
| --- | --- |
| `split_manifest.json` | 上面的 20/4/5 天划分、277 窗口、1406 节点 |
| `physical_definition.json` | `independent_fcd_v1` 的四状态定义 |
| `node_mapping.csv` | 1406 行，列与上海节点表相同，`edge_id` 如 `34180200019` |
| `state_normalization.json` | 四状态输入 scaler |
| `target_normalization.json` | 目标 scaler |
| `state_window_manifest.json` | 窗口清单 |
| `validation_summary.json` | 短校验摘要 |
| `sequence_targets_12step/{train,validation,test}/YYYY_MM_DD.npz` | 全量日序列，不按渗透率复制 |
| `sequence_targets_12step/sequence_manifest.json` | 序列清单 |
| `provenance/` | 迁移前留下的 metadata；`retired_layout/` 里还有旧的 `partial_flow`、`sampled_vehicle`、`sampling_seeds`、`window_data` 目录壳 |

`sequence_targets_12step/train/2023_04_01.npz` 的 `full_raw` 是 `(288, 1406, 4)` float32，`channels` 为四状态名，另有 `edge_ids`、`time_bin`、`start_time`、`day_index`、`date`、`split`。train 约 126M，validation 约 26M，test 约 32M。

#### `full_flow/`

| 层级 | 日目录内容 | 规模 |
| --- | --- | --- |
| `r-level/YYYY_MM_DD/` | `inflow.csv`、`outflow.csv`、`accumulation_start.csv`、`accumulation_end.csv`、`event_breakdown.json` | 1406 条边；另有 `generation_summary.csv`、`state_migration_summary.csv` |
| `m-level/YYYY_MM_DD/` | 四个状态 csv，列名 `M::<junction>::<in>::<out>` | 3877 个 movement；另有 `day_stats/`、`anomalies/` |
| `mr-level/YYYY_MM_DD/` | 四个状态 csv | R 与 M 合在一张表；另有 `mr_level_metadata.json`、`day_stats/` |

CSV 前两列仍是 `time_bin,start_time`。`r-level` 的 `event_breakdown.json` 记录物理定义版本、读过的 `fcd_data_3s` 压缩包、86400 个样本和各类进出事件计数。

#### `partial_flow/`

只有 `m-level/`（约 1.9G）和 `mr-level/`（约 2.5G）。两边都是 `p05` … `p70/YYYY_MM_DD/` 四个状态 csv，加 `day_stats/` 和 `generation_summary.csv`。MR 还有 `mr_level_metadata.json`。这里没有 R 级渗透率流量。

#### `graph/`

R 级文件与旧 `training_data/graph/` 同名：`r_graph.npz`、`r_nodes.csv`、`node_mapping.csv`、`stgcn_weighted_adjacency.npy`、`dcrnn_weighted_adjacency.npy`、`dcrnn_adj_mx.pkl`、`stsgcn_undirected_topology.npy`、`undirected_center_distance.npy` 及其 metadata。

另外有 M / MR：

| 文件 | 内容 |
| --- | --- |
| `m_graph.npz`、`m_nodes.csv`、`m_signal_connections.csv`、`m_signal_programs.json`、`m_transitions.csv` | 3877 个 movement。审计写明由 `xuancheng_tls_3s.net.xml` 和 `night_peak_3s.add.xml` 生成 |
| `m_graph_audit.json`、`m_graph_audit.md` | M 图审计，状态 PASS |
| `mr_graph.npz`、`mr_nodes.csv`、`mr_edges.csv` | 5283 个节点：R 的 global index 0–1405，M 为 1406–5282 |
| `mr_graph_audit.json`、`mr_graph_audit.md` | 四类关系 R→R、M→M、R→M、M→R |
| `asttn/node2vec_r128.npy`、`asttn/metadata.json` | 128 维 node2vec |
| `stjgcn/predefined_stjg.npz`、`stjgcn/metadata.json` | STJGCN 预定义图，npz 约 189MB |

#### `config/`

模型集合与旧 `training_data/config/` 相同，没有 `asttn`、`stjgcn`、`MSTT`、`tlast` 子目录。差异是：`RF/` 不再有 `p05` … `p70`，多了 `legacy_speed_archive.json`；`stfgnn/` 的时间邻接按四个状态分开，有 `temporal_adjacency_{inflow,outflow,accumulation_start,accumulation_end}.npy`，没有旧包里的 `temporal_adjacency.before_activity_filter.npy`。PDFormer 的 npy 仍放在 `config/pdformer/`，不在 `graph/`。

#### `archive/cumsum_stock_v1/`

`metadata_archived.json` 只有两行：`physical_definition_version=cumsum_stock_v1`，`moves=20`。目录本身是：

```text
cumsum_stock_v1/
├── datasets/{fixed_uniform,variable_day_p10_p20,variable_2h_p10_p20,variable_1h_p10_p20,variable_30min_p10_p20}/sampling_seeds/
├── full_flow/r-level/YYYY_MM_DD/
├── shared/          自己的 scaler、split、sequence_targets_12step
└── model_prepared/RF_shared_across_seeds/
```

它不是一份内容独立的旧副本。抽查这些路径，archive 与现行树是同一个 inode（`nlink=2`）：

- `full_flow/r-level/2023_04_01/inflow.csv`
- `datasets/fixed_uniform/.../seed_42/flow/partial_flow/p10/2023_04_02/accumulation_start.csv`
- `datasets/fixed_uniform/.../seed_42/flow/window_data/p05/train/2023_04_01/inflow.npz`

archive 日目录比现行 `full_flow/r-level` 多两个文件：`entry_flow.csv`、`exit_flow.csv`。抽查的 `entry_flow.csv` 是 `nlink=1`，只在 archive 里。`shared/state_normalization.json` 也不是硬链接：archive 4004 字节，现行 `shared/state_normalization.json` 4437 字节。

#### `.staging/` 和 `architecture_audit/`

`.staging/independent_fcd_v1/` 有 29 个日期目录。`2023_04_01/` 含 `complete.json`、空的 `full/`，以及 `fixed/seed_42` … `seed_46/p05` … `p70` 和 `variable/`。抽查 `fixed/seed_42/p05` 是空目录。整棵 `.staging` 只有 7.9M。

`architecture_audit/` 是迁移和清理记录：`migration.json`、`dataset_layout_migration_journal.json`、`independent_fcd_v1_migration_journal.json`、`independent_fcd_v1_cleanup_report.json`、`independent_fcd_v1_counts.json`、`reader_audit.json`、`retired_input_verification.json`，以及 `model_data_compatibility/` 下的 `README.md`、`model_preparation_dependencies.json`、`smoke_inputs.json`。

### `training_results/`（19G）

旧单通道 inflow 包的训练输出。模型目录：`dcrnn` 838M、`graph_wavenet` 883M、`pdformer` 917M、`staeformer` 948M、`stgcn` 998M、`stid` 588M、`stfgnn` 7.3G、`stsgcn` 4.5G、`s2s_gcn_lstm` 1.3G、`RF` 84M。根上还有一份 `training_launcher.log`。

DCRNN、Graph WaveNet、PDFormer、STAEformer、STGCN、STID、STSGCN、STFGNN 的主体是 `p05` … `p70/seed_42/`，渗透率目录旁常有 `SUCCESS` 和 `train.log`。模型根上还有 `environment.json`、`resolved_config.json`、`overall_summary.json`、`source_code_manifest.json`、`training_log.jsonl`。图模型另有 `experiment_manifest.json` 和 runtime validation。STFGNN 根上还有 `full_stfgnn_after_p20_*.log` 和一个 `__pycache__/`。

`s2s_gcn_lstm/` 不是七档都在顶层。实际有 `p70/seed_42/`、`gaussian_stgcn/p70/seed_42/`、`searched_trial018/`，以及 `searched_trial018.pid`、`searched_trial018_train.log`、`paper.json`。

`RF/2026_10_23/` 有 `clusters_100/`、`clusters_200/`、`rf_full/`。后两者按 `p05` … `p70` 放结果，并带 `experiment_config.json`、`summary.csv`、`summary.json`、`worker_pXX.json`。`RF/` 根上还有 `deleted_model_files.json`。

一次 `seed_42` 目录里的文件与上海结果同类：checkpoint、`resolved_config.json`、`run_metadata.json`、history、validation/test metrics 和 predictions npz。

### `training_results_3s/`（503G）

3 秒包上的训练输出。顶层模型是 `MSTT` 18G、`RF` 309G、`asttn` 19M、`dcrnn` 9.9G、`graph_wavenet` 11G、`pdformer` 917M、`s2s_gcn_lstm` 8.3G、`staeformer` 26G、`stfgnn` 56G、`stgcn` 12G、`stid` 7.4G、`stjgcn` 21M、`stsgcn` 40G、`tlast` 8.6G，外加 `_scheduler/`（四份 MSTT p05 日志，共 28K）。

同一模型下面经常并排放着两代目录，都还在：

```text
<model>/
├── p05 … p70/seed_42/                         早期按渗透率直接铺开
├── in_<input>__out_<target>/pXX/seed_42/      九个输入输出对
└── 模型自己的日志和 summary
```

九个 pair 目录名是：

```text
in_inflow__out_inflow
in_outflow__out_outflow
in_accumulation_start__out_accumulation_start
in_accumulation_end__out_accumulation_end
in_all__out_all
in_all__out_inflow
in_all__out_outflow
in_all__out_accumulation_start
in_all__out_accumulation_end
```

DCRNN、Graph WaveNet、STGCN、STSGCN、STFGNN、STAEformer、STID、S2S GCN-LSTM 这八个同时有七档 `pXX` 和这九个 pair。TLAST 有九个 pair，没有顶层 `pXX`。PDFormer 只有顶层 `p05` … `p70`。磁盘上没有 `physical_independent_fcd_v1/` 这一层结果目录。

`staeformer/` 另外有 `sampling_protocol/in_<input>__out_<target>/pXX/seed_42/`。`in_inflow__out_inflow` 下面目前只看到 `p05`、`p10`、`p20`、`p30`。其 `p05/seed_42/` 比旧运行多了 `test_sampling_metrics.json`、`validation_sampling_metrics.json` 和 `predictions/seed_42` … `seed_46/`。

`RF/` 的 pair 只有八个，没有 `in_inflow__out_inflow`。另外保留 `2026_10_27/231446/`，里面是 `p05` … `p70`、`summary.csv`、`experiment_config.json`、`worker_pXX.json`。根上有 `rf_train.log` 和 `rf_train.pid`。

`MSTT/` 除了 `in_inflow__out_inflow/p05/seed_42/` 和顶层 `p05/{I2I,O2O}/`，还有这些实验目录：`C_inflow_aux`、`ablation_100ep`、`conservation_best_backbone`、`conservation_raw_weight_sweep`、`experiment_A_separate_value_mlp_projected`、`huber_delta1`、`huber_delta2`、`joint_st3_projected`、`v0.2_concat_embedding_bs16`、`v0.3_concat_adaptive_bs16`、`v0.4_global`，以及 `_benchmarks/`、`_logs/`、`overall_summary.csv`、`overall_summary.json`。

`asttn/` 只有 `_smoke/in_inflow__out_inflow/`。`stjgcn/` 只有 `_smoke/`。`tlast/` 有 `_smoke/` 下的 `in_inflow__out_inflow`、`in_all__out_all`、`in_all__out_outflow`，以及正式的九个 pair。
