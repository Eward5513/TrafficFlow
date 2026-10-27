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
model_prep/  目前读取的是 fcd_data/，不是 fcd_data_3s/
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

晚高峰切换在 `night_peak.add.xml`：55000 秒切到 program 1，75000 秒切回 program 0。55000 不是 3 的整数倍，3 秒版本没有改这个切换时刻。

## 2. model_prep：从仿真结果到模型输入

这些脚本不改路网，也不改路线。它们读 `fcd_data/` 里已经完成的那一轮，写出 `training_data/`。3 秒相位那一轮还在 `fcd_data_3s/`，当前脚本不会自动改读它。

建议顺序：

1. `generate_r_graph.py`：用 `xuancheng_tls_rebuilt.net.xml` 的真实连接做成有向路段图 `graph/r_graph.npz`。
2. `generate_r_level_full_flow.py`：按 5 分钟统计每条路段的进入和离开。
3. `generate_sampled_vehicle_ids.py`：每个渗透率抽取实际出发的车辆。种子 42，低比例是高比例的前缀。
4. `generate_r_level_partial_flow.py`：只用抽中的车辆再数一遍 5 分钟流量。
5. `generate_r_level_windows.py`：切同一天内的 12 步窗口。输入是抽样流量，目标是窗口最后一步的全量进入流量。
6. `merge_r_level_window_data.py`：按固定日期顺序拼成 train、validation、test。
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
python3 generate_sampled_vehicle_ids.py --workers 29
python3 generate_r_level_partial_flow.py --workers 29
python3 generate_r_level_windows.py --workers 29
python3 merge_r_level_window_data.py
python3 generate_training_support_files.py
```

训练结果写到 `/data5/tzhang174/TrafficFlow/xuancheng26/training_results/`。
