# 数据盘目录清单

更新日期：2026-11-10 18:18 +0800，按当时磁盘上的实际路径记录。代码在 `/home/tzhang174/TrafficFlow`，数据在 `/data5/tzhang174/TrafficFlow`。

```text
/data5/tzhang174/TrafficFlow                         2.3T
├── shanghai_sim/                                    122G   上海 20 天、56 条路：截图、SUMO、窗口、八个模型结果
└── xuancheng26/                                     2.1T   宣城 29 天、3 秒相位
    ├── simulation_data/                             1.1T   路网、全天路线、FCD
    ├── training_data_3s/                            864G   四状态流量、图、固定档和波动渗透率窗口
    └── training_results_3s/                         230G   checkpoint、指标、预测
        ├── datasets/variable_30min_p10_p20/         14G    最新一轮，日程 30 分钟
        ├── datasets/variable_1h_p10_p20/            11G    最新一轮，日程 1 小时
        ├── datasets/variable_2h_p10_p20/            9.3G   最新一轮，日程 2 小时
        ├── datasets/variable_day_p10_p20/           8.1G   最新一轮，日程 1 天
        ├── datasets/legacy_unknown/                 190G   历史固定渗透率 p05–p70
        ├── operations/                              队列、调度日志、清理记录
        ├── reports/                                 启动汇总和训练日志
        └── registry/                                631 条历史运行索引
```

最新一轮的层级和每一层里的文件，见下面「宣城训练结果」开头的树。物理版本是 `independent_fcd_v1`，观测键是 `p10_p20`。

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

## 宣城训练数据：`xuancheng26/training_data_3s/`（864G）

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

## 宣城训练结果：`xuancheng26/training_results_3s/`（230G）

正式运行都在 `datasets/` 下面，一层一个身份字段。四个 `variable_*_p10_p20` 结构相同，下面把 30 分钟展开；1 小时、2 小时、全天只是段长不同。`p5_p15` 和 `p5_p35` 还只有训练数据，结果树里没有这两个观测键。

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
│   ├── variable_p10_p20_train42_eval5/            144 条任务的 manifest、tasks、完成汇总、scheduler 日志
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

### 最新一轮：`variable_*_p10_p20` / `physical_independent_fcd_v1`

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

MSTT 单独跑 A2A，实验号 `conservation_projected`，配置是 `MSTT/configs/conservation_best_backbone/projected.json`。它不进上面的 144 条队列。四个周期都已写出 `test_sampling_metrics.json`：

```text
datasets/variable_30min_p10_p20/.../seed_42/20261110T132316_29bc19aa/
datasets/variable_1h_p10_p20/.../seed_42/20261110T161911_f3edf8f9/
datasets/variable_2h_p10_p20/.../seed_42/20261110T183956_a9bd76cc/
datasets/variable_day_p10_p20/.../seed_42/20261110T210053_dceca753/
```

共同前缀是 `physical_independent_fcd_v1/MSTT/in_all__out_all/p10_p20/conservation_projected/`。日志是 `reports/mstt_variable_projected_a2a.log` 和 `reports/mstt_variable_projected_a2a_1h_2h_day.log`。

指标摘录在代码树的 [dataset/xuancheng26/TRAINING_RESULTS.md](dataset/xuancheng26/TRAINING_RESULTS.md)，MSTT 在 [MSTT/TRAINING_RESULTS.md](MSTT/TRAINING_RESULTS.md)。基线摘录整理到 2026-11-11 14:11，当时 71 条已经写出五种子测试指标：四个周期的 I2I 都完成了，A2A 完成 23 条，A2I 还没有。上面的 18:18 快照只说明当时的目录，完成数字以摘录和 `campaign_summary.json` 为准。

### 历史固定渗透率：`legacy_unknown` / `physical_legacy_unknown`（190G）

631 个历史 run 都在这一层。配置里没有写下 `dataset_id` 和 `physical_definition_version`，所以没有标成 `fixed_uniform` 或 `independent_fcd_v1`。模型有 `MSTT`、`dcrnn`、`graph_wavenet`、`pdformer`、`s2s_gcn_lstm`、`staeformer`、`stfgnn`、`stgcn`、`stid`、`stsgcn`、`tlast`。观测键是固定档 `p05`、`p10`、`p20`、`p30`、`p40`、`p50`、`p70`。实验号主要是 `baseline`，另外有 STAEformer 的 `sampling_protocol` 和若干 MSTT 实验号。模型种子都是 `seed_42`，`run_id` 是 `legacy_<hash>`。目录里保留原来的 checkpoint、配置、history、metrics、predictions 文件名，并用 `run_manifest.json` 登记。

另有 103 个中断 run 已删除：79 个随机森林、19 个 STAEformer、3 个 STFGNN、2 个 MSTT。它们有模型文件，没有测试指标，也没有完成状态。清单在 `operations/cleanup_20261109/removed_interrupted_runs.jsonl`。

`operations/cleanup_20261109/` 只保存这次清理的迁移记录和从 `xuancheng26/` 根目录收进来的旧准备日志。`registry/runs.jsonl` 仍是这 631 条 legacy 运行的索引，波动渗透率的新 run 不在这个文件里。`reports/historical/` 是迁移时从旧模型根目录收来的 summary、日志和 loose 文件。
