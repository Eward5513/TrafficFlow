# 数据盘目录清单

更新日期：2026-11-09，按当时磁盘上的实际路径记录。代码在 `/home/tzhang174/TrafficFlow`，数据在 `/data5/tzhang174/TrafficFlow`。

```text
/data5/tzhang174/TrafficFlow          1.9T
├── shanghai_sim                      122G
└── xuancheng26                       1.8T
    ├── simulation_data               1.1T
    ├── training_data_3s              503G
    └── training_results_3s           190G
```

宣城非 3 秒的 `training_data/`、`training_results/` 已删除。`training_data_3s/sampling_seeds` 软链接、`archive/`、`.staging/`、`architecture_audit/` 也已删除。原始出行仍在数据盘外面的 `/data5/tzhang174/Xuancheng26/`，本清单不展开那棵树。

两套实验都是 5 分钟一个时间片、一天 288 片，用连续 12 步重建最后一步的全量状态。固定渗透率目录是 `p05`、`p10`、`p20`、`p30`、`p40`、`p50`、`p70`。

---

## 上海：`shanghai_sim/`（122G）

20 个仿真日，子图 56 条 R 道路。划分在 `training_data/r-only/split_manifest.json`：训练 `day_01`–`day_14`，验证 `day_15`–`day_17`，测试 `day_18`–`day_20`。这次清理没有改这个目录。

```text
shanghai_sim/
├── ocr_data/            截图和 OCR 文本
├── simulation_data/     20 天 SUMO 输入输出和子图流量
├── training_data/       56 节点窗口、STGCN/DCRNN 图、PDFormer/STFGNN 关系
├── training_results/    八个模型的 p05–p70/seed_42
└── analysis_results/    汇总表和历史均值基线
```

`ocr_data/` 是平铺编号文件：`snapshoot/`、`ocr_output/`，以及全量的 `snapshoot_all/`、`ocr_output_all/`。`ocr_output_all/log/` 里是三份文本日志。

`simulation_data/` 有 `random_trip/`、`route_demand/`、`simulation/<编号>_<种子>/` 和 `processed/subgraph_trajectories/day_01` … `day_20`。每个仿真日目录是 `simulation.sumocfg`、`fcd.csv.gz`、`tripinfo.xml`、`vehroute.xml`、`summary.xml`、`statistics.xml`、`simulation.log`。`subgraph.txt` 是 56 行 SUMO edge id。

`training_data/r-only/p05` … `p70` 各有 `train.npz`、`validation.npz`、`test.npz`。`p05/train.npz` 的 `x` 是 `(3878, 12, 56, 1)`。`adjacency_matrix/` 是 STGCN 图，`dcrnn/` 是有向图。`pdformer/` 和 `stfgnn/` 是上海这 56 个节点自己的关系矩阵。

`training_results/` 下是 `dcrnn`、`graph_wavenet`、`pdformer`、`staeformer`、`stfgnn`、`stgcn`、`stid`、`stsgcn`，外加 `_launch/` 日志。每个模型是 `pXX/seed_42/`，里面是 checkpoint、`resolved_config.json`、history、validation/test metrics 和 predictions。

---

## 宣城仿真：`xuancheng26/simulation_data/`

29 天，2023-04-01 到 2023-04-30，没有 2023-04-09。当前要加载的路网是代码树里的 `dataset/xuancheng26/xuancheng_tls_3s.net.xml`，附加文件是 `night_peak_3s.add.xml`。`xuancheng_tls_tmp.net.xml` 与 3 秒路网逐字节相同，文件已删除。`fcd_data_3s/` 里的 `day_result.json` 仍写着当时的 `fcd_tmp/` 和 tmp 路网路径，那些记录没有改。

```text
simulation_data/
├── road_network/     base/xuancheng.net.xml、cache/、topology/
├── trip_data/        repaired、prepared、refined、rejected、repair_metadata、manifests、reports
├── fcd_data/         29 天，只有 fcd_YYYY_MM_DD.xml.gz
└── fcd_data_3s/      29 天，未压缩 fcd_*.xml 和 .xml.gz 都在
```

`trip_data/refined/refined_routes_YYYY_MM_DD.rou.xml` 是全天仿真加载的路线。`fcd_data_3s/2023_04_01/` 里有 `fcd_2023_04_01.xml`、`fcd_2023_04_01.xml.gz`、压缩的 tripinfo/summary/statistics、`day_result.json` 和日志。未压缩 XML 是这棵仿真树的大头，还没有和 gzip 做完全流式比对，所以仍保留。

历史非 3 秒批次的 runner `run_fcd_simulations.py` 还在代码里，它用的是 `xuancheng_tls_rebuilt.net.xml`，对应 `fcd_data/`。当前批次的入口是 `run_fcd_simulations_3s.py`。

---

## 宣城训练数据：`xuancheng26/training_data_3s/`（503G）

四状态定义在 `shared/physical_definition.json`，版本名 `independent_fcd_v1`。通道顺序是 `inflow, outflow, accumulation_start, accumulation_end`。划分在 `shared/split_manifest.json`：训练 20 天（04-01…04-08、04-10…04-21），验证 04-22…04-25，测试 04-26…04-30。每天 277 个窗口，R 节点 1406。

```text
training_data_3s/
├── datasets/
│   ├── fixed_uniform/sampling_seeds/seed_42 … seed_46/
│   ├── variable_30min_p10_p20/ … variable_day_p10_p20/
│   ├── variable_30min_p5_p15/  … variable_day_p5_p15/
│   └── variable_30min_p5_p35/  … variable_day_p5_p35/
├── shared/
├── full_flow/{r-level,m-level,mr-level}/
├── partial_flow/{m-level,mr-level}/
├── graph/
├── config/
├── dataset_metadata.json
├── validation_summary.json
├── cleanup_manifest.json
└── cleanup_report.json
```

没有顶层 `sampling_seeds`。固定渗透率就是 `datasets/fixed_uniform/sampling_seeds/`。

每个固定种子：

```text
seed_42/
├── graph -> ../../../../graph
├── sampled_vehicle/pXX/YYYY_MM_DD.csv
├── model_prepared/RF/pXX/YYYY_MM_DD.npz
└── flow/
    ├── partial_flow/pXX/YYYY_MM_DD/{inflow,outflow,accumulation_start,accumulation_end}.csv
    └── window_data/pXX/{train,validation,test}/YYYY_MM_DD/{inflow,outflow,accumulation_start,accumulation_end,all}.npz
```

单通道 `inflow.npz` 的 `x` 是 `(277, 12, 1406, 1)`。`all.npz` 最后一维是 4。

可变渗透率按配置里的 `p_min`、`p_max` 生成，观测键由 `data_root.observation_key` 决定。0.10–0.20 是 `p10_p20`，0.05–0.15 是 `p5_p15`，0.05–0.35 是 `p5_p35`（5% 写成 `p5`，不是 `p05`）。固定渗透率文件仍用原来的 `p05`。每种频率一套 dataset，段长分别是 1800、3600、7200、86400 秒，种子都是 42–46。`p5_p15` 和 `p5_p35` 各四套的 `dataset_status` 和 13 个基线辅助资源都是 ready。静态图、训练全日统计和 RF 道路属性继续共用，不按采样种子复制。RF 速度在各自的 `model_prepared/RF/<observation_key>/`。

换一个区间时新增一份 JSON，不改生成脚本：

```bash
cd /home/tzhang174/TrafficFlow
python3 dataset/xuancheng26/model_prep/generate_variable_penetration_datasets.py \
  --config dataset/xuancheng26/model_prep/configs/variable_p5_p35.json \
  --workers 29 --stage all --prepare-models all-baselines --execute
```

`variable_p5_p15.json` 用同一条命令。

断点续跑把 `--stage` 换成 `vehicles`、`flow`、`scaler`、`windows`、`ready` 或 `models`。`--dates` 只处理列出的日期；scaler 仍用完整训练 split 和 sampling seed 42。窗口身份包含当天四个 partial CSV 的 sha256，旧身份缺这个字段时，只在本次日期范围内核对原始窗口，一致就补记身份、不重写 NPZ。只有 29 天都在参数里时才发布全量 ready。只补辅助资源用 `--stage models`。训练入口只改 `--dataset-id` 和输出目录；未传 `--rates` 时，单一观测键从该 dataset 的 manifest 读取。波动训练默认仍是 `p10_p20`；`run_variable_baselines.py plan --observation-key p5_p15` 使用另一套 `operations/variable_p5_p15_train42_eval5/` 状态。

`shared/` 有 `split_manifest.json`、`physical_definition.json`、`node_mapping.csv`、`state_normalization.json`、`target_normalization.json`、`state_window_manifest.json`、`validation_summary.json`、`sequence_targets_12step/{train,validation,test}/YYYY_MM_DD.npz` 和 `provenance/`。序列标签的 `full_raw` 是 `(288, 1406, 4)`。

`full_flow/r-level/YYYY_MM_DD/` 是四个状态 CSV 加 `event_breakdown.json`。`m-level` 是 3877 个 movement，`mr-level` 把 R 和 M 放在一张表里。`partial_flow/` 只有 M 和 MR，没有 R 级；R 级分渗透率 CSV 在 `datasets/.../flow/partial_flow/`。

`graph/` 有 `r_graph.npz`（1406 节点）、STGCN/DCRNN/STSGCN 的邻接、`m_graph.npz`、`mr_graph.npz`、`asttn/node2vec_r128.npy`、`stjgcn/predefined_stjg.npz`。

`config/` 仍是各模型的 JSON，以及训练会读的生成矩阵：

| 路径 | 谁在读 |
| --- | --- |
| `config/pdformer/pdformer_{shortest_path,dtw_distance,laplacian_pe,pattern_keys}.npy` 和 `pdformer_relations_metadata.json` | `r_only_pdformer.json` 的 `relations_dir`，PDFormer 训练从这里加载 |
| `config/stfgnn/temporal_adjacency.npy` | `r_only_stfgnn.json` 的 `temporal_adjacency` |
| `config/stfgnn/temporal_activity_*.json`、`temporal_activity_per_road.csv` | 生成时间图时留下的报告，训练不读 |
| `config/<model>/r_only_<model>.json` | 各模型默认配置 |

按状态拆开的 `temporal_adjacency_{inflow,outflow,accumulation_start,accumulation_end}.npy` 没有读取方，已删除。

---

## 宣城训练结果：`xuancheng26/training_results_3s/`（190G）

```text
training_results_3s/
├── datasets/legacy_unknown/physical_legacy_unknown/<model>/...
├── registry/runs.jsonl
├── reports/
└── operations/cleanup_20261109/
```

现存 631 个 run，全部在 `legacy_unknown` / `physical_legacy_unknown` 下面。这些历史运行的配置里没有写下 `dataset_id` 和 `physical_definition_version`，所以没有标成 `fixed_uniform` 或 `independent_fcd_v1`。模型有 `MSTT`、`RF`、`dcrnn`、`graph_wavenet`、`pdformer`、`s2s_gcn_lstm`、`staeformer`、`stfgnn`、`stgcn`、`stid`、`stsgcn`、`tlast`。

一个 run 的路径是：

```text
datasets/<dataset_id>/physical_<version>/<model>/in_<input>__out_<target>/<observation_key>/<experiment_id>/seed_<model_seed>/<run_id>/
```

历史 run 的 `run_id` 是 `legacy_<hash>`。目录里保留原来的 checkpoint、配置、history、metrics、predictions 文件名，并用 `run_manifest.json` 登记。新训练由 `data_root.sampling_run_results_dir` 写到同一层结构，并使用新的 `run_id`。

另有 103 个中断 run 已删除：79 个随机森林、19 个 STAEformer、3 个 STFGNN、2 个 MSTT。它们有模型文件，没有测试指标，也没有完成状态。清单在 `operations/cleanup_20261109/removed_interrupted_runs.jsonl`。

`operations/cleanup_20261109/` 只保存这次清理的迁移记录和从 `xuancheng26/` 根目录收进来的旧准备日志，不参与训练。`registry/runs.jsonl` 是留下的 run 索引。`reports/` 里是迁移时从旧模型根目录收来的 summary、日志和 loose 文件。
