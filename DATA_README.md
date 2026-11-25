# 数据盘目录清单

更新日期：2026-11-25 15:25 +0800。体积是当时的 `du -h`，父子目录各自取整，相加不必等于上一级。代码在 `/home/tzhang174/TrafficFlow`。这一清单只统计 `/data5/tzhang174`，合计 3.8T。同一时刻 `/data5` 总容量 14T，剩余约 25G。

```text
/data5/tzhang174                                     3.8T
├── TrafficFlow                                      3.8T   上海仿真和宣城 3 秒相位的全部可写数据
├── Xuancheng26                                      5.5G   只读原始出行。trips/ 5.5G，其余审计、路网、文档不到 80M
├── EVData_RouteSys                                  967M   电动车插入路径的 C++ 工程和 Manhattan_Data，属主是 root
├── EVData_RouteSys1                                 778M   同一工程的另一份
└── vip                                              21M    route_by_edge_8-9.txt、route_by_edge_real.txt
```

```text
/data5/tzhang174/TrafficFlow                         3.8T
├── shanghai_sim/                                    122G   上海 20 天、56 条路
│   ├── simulation_data/                             109G   20 天 SUMO 输入输出和子图流量
│   ├── ocr_data/                                    13G    截图和 OCR 文本
│   ├── training_results/                            778M   八个模型的 p05–p70/seed_42
│   ├── training_data/                               218M   56 节点窗口和图
│   └── analysis_results/                            2.3M   汇总表和历史均值基线
└── xuancheng26/                                     3.6T   宣城 29 天、3 秒相位
    ├── training_results_3s/                         1.4T   checkpoint、指标、预测
    ├── training_data_3s/                            1.3T   四状态流量、图、固定档和波动渗透率窗口
    └── simulation_data/                             1.1T   路网、全天路线、FCD
```

运行目录的层级见下面各节。物理版本是 `independent_fcd_v1`。结果里最大的一项目录是 `training_results_3s/datasets/variable_30min_p5_p15/`，358G，其中随机森林 272G。

宣城非 3 秒的 `training_data/`、`training_results/` 已删除。`training_data_3s/` 顶层没有 `sampling_seeds` 软链接，也没有 `archive/`、`architecture_audit/`。`training_data_3s/.staging/` 还在，4.4M。原始出行在 `/data5/tzhang174/Xuancheng26/`，上面已经列到第一层。

两套实验都是 5 分钟一个时间片、一天 288 片，用连续 12 步重建最后一步的全量状态。固定渗透率目录是 `p05`、`p10`、`p20`、`p30`、`p40`、`p50`、`p70`。

---

## 上海：`shanghai_sim/`（122G）

20 个仿真日，子图 56 条 R 道路。划分在 `training_data/r-only/split_manifest.json`：训练 `day_01`–`day_14`，验证 `day_15`–`day_17`，测试 `day_18`–`day_20`。这次清理没有改这个目录。

```text
shanghai_sim/              122G
├── simulation_data/       109G   20 天 SUMO 输入输出和子图流量
├── ocr_data/              13G    截图和 OCR 文本
├── training_results/      778M   八个模型的 p05–p70/seed_42
├── training_data/         218M   56 节点窗口、STGCN/DCRNN 图、PDFormer/STFGNN 关系
└── analysis_results/      2.3M   汇总表和历史均值基线
```

`ocr_data/` 是平铺编号文件：`snapshoot/`、`ocr_output/`，以及全量的 `snapshoot_all/`、`ocr_output_all/`。`ocr_output_all/log/` 里是三份文本日志。

`simulation_data/` 有 `random_trip/`、`route_demand/`、`simulation/<编号>_<种子>/` 和 `processed/subgraph_trajectories/day_01` … `day_20`。每个仿真日目录是 `simulation.sumocfg`、`fcd.csv.gz`、`tripinfo.xml`、`vehroute.xml`、`summary.xml`、`statistics.xml`、`simulation.log`。`subgraph.txt` 是 56 行 SUMO edge id。

`training_data/r-only/p05` … `p70` 各有 `train.npz`、`validation.npz`、`test.npz`。`p05/train.npz` 的 `x` 是 `(3878, 12, 56, 1)`。`adjacency_matrix/` 是 STGCN 图，`dcrnn/` 是有向图。`pdformer/` 和 `stfgnn/` 是上海这 56 个节点自己的关系矩阵。

`training_results/` 下是 `dcrnn`、`graph_wavenet`、`pdformer`、`staeformer`、`stfgnn`、`stgcn`、`stid`、`stsgcn`，外加 `_launch/` 日志。每个模型是 `pXX/seed_42/`，里面是 checkpoint、`resolved_config.json`、history、validation/test metrics 和 predictions。

---

## 宣城仿真：`xuancheng26/simulation_data/`

29 天，2023-04-01 到 2023-04-30，没有 2023-04-09。当前要加载的路网是代码树里的 `dataset/xuancheng26/xuancheng_tls_3s.net.xml`，附加文件是 `night_peak_3s.add.xml`。`xuancheng_tls_tmp.net.xml` 与 3 秒路网逐字节相同，文件已删除。`fcd_data_3s/` 里的 `day_result.json` 仍写着当时的 `fcd_tmp/` 和 tmp 路网路径，那些记录没有改。

```text
simulation_data/      1.1T
├── fcd_data_3s/      916G   29 天，未压缩 fcd_*.xml 和 .xml.gz 都在
├── fcd_data/         131G   29 天，只有 fcd_YYYY_MM_DD.xml.gz
├── trip_data/        11G    repaired、prepared、refined、rejected、repair_metadata、manifests、reports
└── road_network/     12M    base/xuancheng.net.xml、cache/、topology/
```

`trip_data/refined/refined_routes_YYYY_MM_DD.rou.xml` 是全天仿真加载的路线。`fcd_data_3s/2023_04_01/` 里有 `fcd_2023_04_01.xml`、`fcd_2023_04_01.xml.gz`、压缩的 tripinfo/summary/statistics、`day_result.json` 和日志。未压缩 XML 是这棵仿真树的大头，还没有和 gzip 做完全流式比对，所以仍保留。

历史非 3 秒批次的 runner `run_fcd_simulations.py` 还在代码里，它用的是 `xuancheng_tls_rebuilt.net.xml`，对应 `fcd_data/`。当前批次的入口是 `run_fcd_simulations_3s.py`。

---

## 宣城训练数据：`xuancheng26/training_data_3s/`（1.3T）

四状态定义在 `shared/physical_definition.json`，版本名 `independent_fcd_v1`。通道顺序是 `inflow, outflow, accumulation_start, accumulation_end`。划分在 `shared/split_manifest.json`：训练 20 天（04-01…04-08、04-10…04-21），验证 04-22…04-25，测试 04-26…04-30。每天 277 个窗口，R 节点 1406。

```text
training_data_3s/                         1.3T
├── datasets/                             1.3T
│   ├── fixed_uniform/                    317G   五个采样种子各 64G，七档 p05–p70
│   └── variable_*/                       各 78G 十二套都是 78G。四个频率 × p10_p20、p5_p15、p5_p35
├── partial_flow/                         4.4G   m-level 1.9G，mr-level 2.5G。没有 R 级
├── full_flow/                            753M   r-level 103M，m-level 275M，mr-level 375M
├── shared/                               354M   划分、scaler、12 步标签。sequence_targets_12step 182M
├── graph/                                238M   R/M/MR 图和各模型邻接
├── config/                               24M    各模型 JSON 和生成矩阵
├── .staging/                             4.4M
├── logs/                                 164K
├── operations/                           60K
├── dataset_metadata.json
├── validation_summary.json
├── cleanup_manifest.json
└── cleanup_report.json
```

没有顶层 `sampling_seeds`。固定渗透率就是 `datasets/fixed_uniform/sampling_seeds/`。五个种子各 64G，seed_42 里 `flow/window_data` 62G，`model_prepared/RF` 979M，`flow/partial_flow` 665M，七档 `sampled_vehicle` 合计 481M。十二套波动 dataset 各 78G，一个种子 16G，同样主要是 `window_data`；seed_42 的 `model_prepared` 是 235M，其中 RF 140M、`traffic_observations` 95M。

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

已发布 `all_sp_c` 的可变数据集，在自己的观测键下另有 `model_prepared/traffic_observations/<observation_key>/YYYY_MM_DD.npz`，以及同级窗口 `window_data/<observation_key>/{train,validation,test}/YYYY_MM_DD/all_sp_c.npz`。`p10_p20`、`p5_p15` 和 `p5_p35` 四个频率都已经有这两份文件。键名分别是 `p10_p20`、`p5_p15`、`p5_p35`，5% 写成 `p5`，不写成 `p05`。每个区间只用自己的车辆名单和 scaler。`all_sp_c.npz` 的 `x` 和 `x_raw` 是 `(277, 12, 1406, 6)`。固定档和尚未跑过这个生成器的观测区间没有这两份文件。

`all_sp_c` 只作为 MSTT 输入，不是预测目标。通道顺序是 `inflow, outflow, accumulation_start, accumulation_end, average_speed_raw, speed_observation_count`。前四路直接沿用原来的 `all.npz`，不重新计算。后两路来自当天 `traffic_observations/<observation_key>/YYYY_MM_DD.npz`：`average_speed_raw` 是该道路、该 300 秒窗里所有有效 1 秒 FCD 快照速度的均值，单位 km/h，不按限速截断，空窗为 0；`speed_observation_count` 是这些快照的条数，同一辆车出现 30 次计 30，停车速度 0 也计入，不除以 300，也不去重成车辆数。`C=0` 时 `V=0`。日级 `count` 存整数，拼进窗口时转成 float32。

每个 dataset 自己的 `all_sp_c_normalization.json` 放在 `datasets/<dataset_id>/`。它只拟合这两个新通道，用该 dataset 的 sampling seed 42、20 个训练日、每天原始 288 个格子，空窗的 0 参加，不按滑动窗口重复加权，也不把不同频率或不同渗透率区间混在一起拟合。五个采样种子和三个 split 共用这份冻结结果。前四通道仍读原来的 `state_normalization.json`。`all_sp_c_ready.json` 只记录这批新文件，不改四状态 dataset 的 ready 状态。标签仍由 `target_type` 读取 `all.npz` 或 `inflow.npz`。

生成入口默认仍是四个 `p10_p20` dataset。`--dataset-ids` 按 manifest 里唯一的观测键处理一个或多个 dataset，可以混合区间；重复 id 会去掉并保持第一次出现的顺序。

```bash
cd /home/tzhang174/TrafficFlow
python3 dataset/xuancheng26/model_prep/prepare_all_sp_c.py --workers 29
python3 dataset/xuancheng26/model_prep/prepare_all_sp_c.py --workers 29 \
  --dataset-ids variable_30min_p5_p15 variable_1h_p5_p15 \
  variable_2h_p5_p15 variable_day_p5_p15
```

可变渗透率按配置里的 `p_min`、`p_max` 生成，观测键由 `data_root.observation_key` 决定。0.10–0.20 是 `p10_p20`，0.05–0.15 是 `p5_p15`，0.05–0.35 是 `p5_p35`（5% 写成 `p5`，不是 `p05`）。固定渗透率文件仍用原来的 `p05`。每种频率一套 dataset，段长分别是 1800、3600、7200、86400 秒，种子都是 42–46。`p5_p15` 和 `p5_p35` 各四套的 `dataset_status` 和 13 个基线辅助资源都是 ready。静态图、训练全日统计和 RF 道路属性继续共用，不按采样种子复制。RF 速度在各自的 `model_prepared/RF/<observation_key>/`。

换一个区间时新增一份 JSON，不改生成脚本：

```bash
cd /home/tzhang174/TrafficFlow
python3 dataset/xuancheng26/model_prep/generate_variable_penetration_datasets.py \
  --config dataset/xuancheng26/model_prep/configs/variable_p5_p35.json \
  --workers 29 --stage all --prepare-models all-baselines --execute
```

`variable_p5_p15.json` 用同一条命令。

断点续跑把 `--stage` 换成 `vehicles`、`flow`、`scaler`、`windows`、`ready` 或 `models`。`--dates` 只处理列出的日期；scaler 仍用完整训练 split 和 sampling seed 42。窗口身份包含当天四个 partial CSV 的 sha256，旧身份缺这个字段时，只在本次日期范围内核对原始窗口，一致就补记身份、不重写 NPZ。只有 29 天都在参数里时才发布全量 ready。只补辅助资源用 `--stage models`。训练入口只改 `--dataset-id` 和输出目录；未传 `--rates` 时，单一观测键从该 dataset 的 manifest 读取。波动训练默认是 `p5_p15`，状态目录是 `operations/variable_p5_p15_train42_eval5/`。10%–20% 用 `run_variable_baselines.py plan --observation-key p10_p20`，状态目录仍是 `operations/variable_p10_p20_train42_eval5/`。

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

## 宣城训练结果：`xuancheng26/training_results_3s/`（1.4T）

正式运行都在 `datasets/` 下面，一层一个身份字段。2026-11-25 的体积：

```text
training_results_3s/                                          1.4T
├── datasets/                                                 1.4T
│   ├── variable_30min_p5_p15/                                358G
│   ├── legacy_unknown/                                       190G   历史固定档 p05–p70
│   ├── variable_day_p5_p15/                                  100G
│   ├── variable_1h_p5_p15/                                   99G
│   ├── variable_2h_p5_p15/                                   99G
│   ├── variable_30min_p10_p20/                               71G
│   ├── variable_2h_p10_p20/                                  70G
│   ├── variable_1h_p10_p20/                                  69G
│   ├── variable_day_p10_p20/                                 69G
│   ├── variable_1h_p5_p35/                                   64G
│   ├── variable_2h_p5_p35/                                   64G
│   ├── variable_day_p5_p35/                                  64G
│   └── variable_30min_p5_p35/                                63G
├── MSTT/                                                     2.4G   结果根上的旧实验。v0.2_concat_A2A_projected 占 2.3G
├── operations/                                               53M    队列和调度日志，没有模型权重
├── reports/                                                  44M    historical 42M，其余是启动汇总
└── registry/ 以及根上的 asttn/、dcrnn/ 等                     各 8K–208K，只有 summary 或索引
```

`variable_30min_p5_p15/` 的 358G 里，随机森林 272G，MSTT 48G，其余十一个神经网络模型各 2.9G–5.8G。这 272G 是 14 个输入输出、训练采样种子 42–46 都落了盘。最大的两对是 `in_all_sp_c__out_all` 66G 和 `in_all__out_all` 48G；四个单状态对各 8.2G–15G，`all` 或 `all_sp_c` 预测单个状态的各 12G–17G。其他十一个 dataset 的随机森林只有训练种子 42 的 I2I、A2A、A2I，每个 dataset 的 RF 是 15G–28G。

`operations/` 里三个已完成的 144 条队列各 13M–14M：`variable_p10_p20_train42_eval5`、`variable_p5_p15_train42_eval5`、`variable_p5_p35_train42_eval5`。上一轮分组队列 `grouped_p5_p15_p5_p35_train5` 是 9.6M，留在原地。当前分组入口另写 `grouped_p5_p15_p5_p35_io5_s42_43_44_45_46_m42_101_102/`，不读、不迁这个旧目录。`cleanup_20261109` 是 560K。

四个 `variable_*_p10_p20` 结构相同，下面把 30 分钟展开；1 小时、2 小时、全天只是段长不同。`p5_p15` 和 `p5_p35` 的结果写在同样的层级下。观测键分别是 `p5_p15`、`p5_p35`，实验号是 `variable_p5_p15_train42_eval5`、`variable_p5_p35_train42_eval5`。分组补跑另用 `variable_<band>_train<s>_eval5`，这里的 s 是训练采样种子 42–46。同一次拟合的模型种子在下一层 `seed_<m>`，这一轮 m 是 42、101、102。

```text
training_results_3s/
├── datasets/                                      正式运行。checkpoint 和指标只在最底层 <run_id>/
│   ├── variable_30min_p10_p20/                    一套数据：渗透率每 30 分钟重抽一次，U(0.10, 0.20)
│   │   └── physical_independent_fcd_v1/           状态定义。通道顺序 inflow, outflow, accumulation_start, accumulation_end
│   │       ├── RF/  stgcn/  dcrnn/  …  stjgcn/    十二个基线，各一个模型
│   │       │   └── in_inflow__out_inflow/         输入输出。这一轮已落盘的是 I2I：inflow → inflow
│   │       │       └── p10_p20/                   观测键，对应上面的 0.10–0.20
│   │       │           └── variable_p10_p20_train42_eval5/   实验号。训练种子 42，评估种子 42–46
│   │       │               └── seed_42/           模型初始化种子，不是采样种子
│   │       │                   └── <run_id>/      一次运行。见下面的叶子
│   │       └── MSTT/                              不在基线队列里
│   │           └── in_all__out_all/               A2A：四个状态进，四个状态出
│   │               └── p10_p20/conservation_projected/seed_42/<run_id>/
│   ├── variable_1h_p10_p20/                       同上，一段 1 小时
│   ├── variable_2h_p10_p20/                       同上，一段 2 小时
│   ├── variable_day_p10_p20/                      同上，一段 1 天
│   └── legacy_unknown/physical_legacy_unknown/    历史固定档。观测键 p05…p70，run_id 为 legacy_<hash>
├── operations/                                    调度和清理记录，没有模型权重
│   ├── variable_p10_p20_train42_eval5/            144 条任务的 manifest、tasks、完成汇总、scheduler 日志，均已完成
│   ├── variable_p5_p15_train42_eval5/             同上，144 条都已完成
│   ├── variable_p5_p35_train42_eval5/             同上，144 条都已完成
│   └── cleanup_20261109/                          11 月 9 日迁移清单，以及当时删除的中断 run
├── reports/                                       日志和启动汇总，不代替 datasets/ 里的运行
│   ├── RF/<run_id>/                               随机森林四次运行的 config 和 summary
│   ├── historical/                                从旧模型根目录收进来的 loose 文件
│   └── mstt_variable_projected_a2a*.log           MSTT 这一轮的标准输出
├── registry/runs.jsonl                           631 行，只索引 legacy 运行
└── asttn/ dcrnn/ … tlast/                         结果根上的模型名。只有 overall_summary 或校验 json
```

叶子 `<run_id>/` 是一次运行的全部输出。写完并做过五种子评估之后，目录里是：

```text
<run_id>/
├── best_checkpoint.pt          按五种子验证 MAE 的平均选出的权重
├── last_checkpoint.pt          最后一个 epoch 的权重
├── resolved_config.json        这次实际用的配置
├── run_metadata.json           数据集、实验号、种子、是否早停
├── training_history.json       每个 epoch 的损失和验证指标
├── training_log.jsonl          训练过程日志
├── validation_metrics.json     选中 checkpoint 后的一次验证
├── validation_sampling_metrics.json    种子 42–46 的验证平均和标准差
├── test_metrics.json           一次测试
├── test_sampling_metrics.json  种子 42–46 的测试平均和标准差
├── test_predictions.npz        测试预测和真值
└── predictions/                按采样种子、按天拆开的预测
```

还在训练的叶子只有 checkpoint、配置和 history，没有 `test_sampling_metrics.json`。随机森林没有 checkpoint；它的运行本体仍在 `datasets/.../RF/.../<run_id>/`，`reports/RF/<run_id>/` 是另一份启动汇总。结果根上的 `dcrnn/`、`stgcn/` 等目录没有这些叶子。

### 2026-11-10 快照：`variable_*_p10_p20` / `physical_independent_fcd_v1`

这一节的 14G、11G、9.3G、8.1G 是 2026-11-10 18:18 的目录体积，不是现在的占用。现在的体积在本节开头的清单里。

四个 `dataset_id` 对应四套日程，渗透率在 0.10 到 0.20 之间均匀抽取，观测键都是 `p10_p20`。段长沿用训练数据：30 分钟 1800 秒，1 小时 3600 秒，2 小时 7200 秒，全天 86400 秒。训练 sampling seed 是 42，模型种子是 42，验证和测试对种子 42、43、44、45、46 取平均。

基线队列的实验号是 `variable_p10_p20_train42_eval5`。模型是 `RF`、`stgcn`、`dcrnn`、`graph_wavenet`、`stsgcn`、`stfgnn`、`stid`、`staeformer`、`s2s_gcn_lstm`、`tlast`、`asttn`、`stjgcn`。这一轮的清单把 `MSTT` 和 `pdformer` 留在队列外。计划中的输入输出是 I2I（`in_inflow__out_inflow`）、A2A（`in_all__out_all`）、A2I（`in_all__out_inflow`），共 4 × 12 × 3 = 144 条。队列状态在 `operations/variable_p10_p20_train42_eval5/`：`campaign_manifest.json`、`tasks.json`、`campaign_summary.json`、`campaign_summary.csv`、`scheduler.log`、`stdout.log`、`control.json`、`logs/`。

18:18 的 `campaign_summary.json` 里，144 条是 41 条 completed、7 条 running、96 条 pending。已经落盘的都是 I2I：

| dataset_id | 已完成 I2I | 18:18 仍在训练的 I2I | 运行目录 |
| --- | --- | --- | --- |
| `variable_30min_p10_p20` | 12 | — | 14G，十二个基线都有 `test_sampling_metrics.json` |
| `variable_1h_p10_p20` | 12 | — | 11G，十二个基线都有 `test_sampling_metrics.json` |
| `variable_2h_p10_p20` | 10 | `asttn` `20261109T225358_ef6d85d5`，`stjgcn` `20261109T225358_81ede9b1` | 9.3G |
| `variable_day_p10_p20` | 7：`RF`、`stgcn`、`dcrnn`、`graph_wavenet`、`stid`、`s2s_gcn_lstm`、`tlast` | `stsgcn` `20261109T225358_827a67cd`，`stfgnn` `20261109T225358_e71705b1`，`staeformer` `20261109T225358_cb529e29`，`asttn` `20261109T225358_7b8bb41a`，`stjgcn` `20261109T225358_e534e5fe` | 8.1G |

96 条 pending 是四个 dataset 的 A2A 和 A2I。这些任务还没有运行目录。

一条已经写完测试指标的基线，例如：

```text
datasets/variable_1h_p10_p20/physical_independent_fcd_v1/dcrnn/
    in_inflow__out_inflow/p10_p20/variable_p10_p20_train42_eval5/
    seed_42/20261109T225358_dbeadc83/
```

叶子里的文件就是上一棵树列出的那些。五种子平均在 `validation_sampling_metrics.json` 和 `test_sampling_metrics.json`。随机森林的四份启动汇总在 `reports/RF/<run_id>/`，run id 是 `20261109T225358_ffe1a013`（30 分钟）、`f03af6f8`（1 小时）、`0cc4c3ec`（2 小时）、`80a226ab`（全天）。

MSTT 单独跑 A2A，实验号 `conservation_projected`。它不进上面的 144 条队列。四个周期都已写出 `test_sampling_metrics.json`，数字在 [MSTT/TRAINING_RESULTS.md](MSTT/TRAINING_RESULTS.md)：

```text
datasets/variable_30min_p10_p20/.../seed_42/20261110T132316_29bc19aa/
datasets/variable_1h_p10_p20/.../seed_42/20261110T161911_f3edf8f9/
datasets/variable_2h_p10_p20/.../seed_42/20261110T183956_a9bd76cc/
datasets/variable_day_p10_p20/.../seed_42/20261110T210053_dceca753/
```

共同前缀是 `physical_independent_fcd_v1/MSTT/in_all__out_all/p10_p20/conservation_projected/`。日志是 `reports/mstt_variable_projected_a2a.log` 和 `reports/mstt_variable_projected_a2a_1h_2h_day.log`。

指标摘录在代码树的 [dataset/xuancheng26/TRAINING_RESULTS.md](dataset/xuancheng26/TRAINING_RESULTS.md)，MSTT 在 [MSTT/TRAINING_RESULTS.md](MSTT/TRAINING_RESULTS.md)。基线摘录整理到 2026-11-24 14:05。10%–20%、5%–15% 和 5%–35% 各 144 条都已写入。上面的 18:18 快照只说明当时的目录，完成数字以摘录和 `campaign_summary.json` 为准。

### 历史固定渗透率：`legacy_unknown` / `physical_legacy_unknown`（190G）

631 个历史 run 都在这一层。配置里没有写下 `dataset_id` 和 `physical_definition_version`，所以没有标成 `fixed_uniform` 或 `independent_fcd_v1`。模型有 `MSTT`、`dcrnn`、`graph_wavenet`、`pdformer`、`s2s_gcn_lstm`、`staeformer`、`stfgnn`、`stgcn`、`stid`、`stsgcn`、`tlast`。观测键是固定档 `p05`、`p10`、`p20`、`p30`、`p40`、`p50`、`p70`。实验号主要是 `baseline`，另外有 STAEformer 的 `sampling_protocol` 和若干 MSTT 实验号。模型种子都是 `seed_42`，`run_id` 是 `legacy_<hash>`。目录里保留原来的 checkpoint、配置、history、metrics、predictions 文件名，并用 `run_manifest.json` 登记。

另有 103 个中断 run 已删除：79 个随机森林、19 个 STAEformer、3 个 STFGNN、2 个 MSTT。它们有模型文件，没有测试指标，也没有完成状态。清单在 `operations/cleanup_20261109/removed_interrupted_runs.jsonl`。

`operations/cleanup_20261109/` 只保存这次清理的迁移记录和从 `xuancheng26/` 根目录收进来的旧准备日志。`registry/runs.jsonl` 仍是这 631 条 legacy 运行的索引，波动渗透率的新 run 不在这个文件里。`reports/historical/` 是迁移时从旧模型根目录收来的 summary、日志和 loose 文件。
