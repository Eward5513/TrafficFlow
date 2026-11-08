# 实验配置一致性审计

比较只使用 `run_metadata.json` 中 `status=completed`、带有测试指标和测试预测、且路径符合约定的 seed 42 运行。原始单通道在 `/data5/tzhang174/TrafficFlow/xuancheng26/training_results/`。四状态和后来的单通道目录在 `/data5/tzhang174/TrafficFlow/xuancheng26/training_results_3s/`。请求里写的结果根目录只覆盖前一套。

三套结果分开比较：

- 原始信号配时上的单通道 inflow。标签来自 `training_data/window_data`。
- 3 秒相位仿真上的单通道 inflow。标签来自 `training_data_3s/sampling_seeds/seed_42/flow/window_data`。名称里的 3 秒指信号相位时长被对齐到 3 秒网格，时间聚合仍是 5 分钟、每天 288 个时间片。历史审计 JSON 里仍记录迁移前的 `training_data_3s/window_data` 路径。
- 同一套 3 秒相位仿真上的四状态实验。通道顺序是 inflow、outflow、accumulation_start、accumulation_end。

请求中的日期是 2026-04-01 至 2026-04-29。两份 `split_manifest.json` 实际保存的是 2023-04-01 至 2023-04-30，缺 2023-04-09，共 29 天。训练 20 天、验证 4 天（04-22 至 04-25）、测试 5 天（04-26 至 04-30）。下文使用文件里的日期。

## 完成数量

每个模型的四状态预期是 4 个 single-to-single、4 个 all-to-single、1 个 all-to-all，再乘 7 个渗透率，共 63 组。

| 模型 | 原始单通道 | 3 秒相位单通道 | 四状态 | 未进入比较的目录 |
| --- | --- | --- | --- | --- |
| STID | 7/7 | 7/7 | 63/63 | 0 |
| STGCN | 7/7 | 7/7 | 63/63 | 0 |
| DCRNN | 7/7 | 7/7 | 63/63 | 0 |
| Graph WaveNet | 7/7 | 7/7 | 63/63 | 0 |
| S2S-GCN-LSTM | 1/7 | 7/7 | 63/63 | 8 |
| TLAST | 0/7 | 0/7 | 63/63 | 3 |

TLAST 没有原始单通道结果，也没有单独的 3 秒相位单通道目录。S2S-GCN-LSTM 的原始单通道只有 p70。其余已进入比较的单元都覆盖了 7 个渗透率。四状态六种模型都是 63/63。

3 秒相位单通道的 35 组测试预测与四状态 single-to-single inflow 逐字节相同，见 `stats/error_summary.json`。它们不提供新的测试结果，下文不再单列排行榜。

未进入比较、但目录里确实有完成记录的运行：

- S2S-GCN-LSTM `/data5/tzhang174/TrafficFlow/xuancheng26/training_results/s2s_gcn_lstm/searched_trial018/p05/seed_42`，状态 completed，MAE 3.726。
- S2S-GCN-LSTM `/data5/tzhang174/TrafficFlow/xuancheng26/training_results/s2s_gcn_lstm/searched_trial018/p10/seed_42`，状态 completed，MAE 3.370。
- S2S-GCN-LSTM `/data5/tzhang174/TrafficFlow/xuancheng26/training_results/s2s_gcn_lstm/searched_trial018/p20/seed_42`，状态 completed，MAE 2.926。
- S2S-GCN-LSTM `/data5/tzhang174/TrafficFlow/xuancheng26/training_results/s2s_gcn_lstm/searched_trial018/p30/seed_42`，状态 completed，MAE 2.796。
- S2S-GCN-LSTM `/data5/tzhang174/TrafficFlow/xuancheng26/training_results/s2s_gcn_lstm/searched_trial018/p40/seed_42`，状态 completed，MAE 2.636。
- S2S-GCN-LSTM `/data5/tzhang174/TrafficFlow/xuancheng26/training_results/s2s_gcn_lstm/searched_trial018/p50/seed_42`，状态 completed，MAE 2.438。
- S2S-GCN-LSTM `/data5/tzhang174/TrafficFlow/xuancheng26/training_results/s2s_gcn_lstm/gaussian_stgcn/p70/seed_42`，状态 completed，MAE 2.004。
- S2S-GCN-LSTM `/data5/tzhang174/TrafficFlow/xuancheng26/training_results/s2s_gcn_lstm/searched_trial018/p70/seed_42`，状态 completed，MAE 2.186。
- TLAST `/data5/tzhang174/TrafficFlow/xuancheng26/training_results_3s/tlast/_smoke/in_all__out_all/p70/seed_42`，状态 completed，MAE 4.966。
- TLAST `/data5/tzhang174/TrafficFlow/xuancheng26/training_results_3s/tlast/_smoke/in_all__out_outflow/p70/seed_42`，状态 completed，MAE 6.133。
- TLAST `/data5/tzhang174/TrafficFlow/xuancheng26/training_results_3s/tlast/_smoke/in_inflow__out_inflow/p70/seed_42`，状态 completed，MAE 7.259。

`searched_trial018` 是超参搜索，`gaussian_stgcn` 是另一种边权，`_smoke` 是冒烟测试。它们的 MAE 不能补进上面的完成数。原始数据上，搜索试验 p70 的 MAE 是 2.186，Gaussian 边权 p70 是 2.004，都低于协议运行的 4.392，也仍高于同一渗透率下 STID 的 1.124。协议运行本身使用 100 个 epoch；3 秒相位上的 S2S 预算是 200 个 epoch。因此原始单通道那一个 4.392 不能代表“把四状态里的 S2S 曲线换到旧数据上”。

清单：`experiment_inventory.csv`。

## 对齐、划分和指标定义

六个模型的测试指标都经过 `reimplementation/common/evaluation.py`，最终调用 `traffic_metrics`。计算发生在反归一化后的车辆数上，负预测不截断。MAE 和 RMSE 对全部道路和时间片取平均，包括真实值为 0 的位置。MAPE_nonzero 只在真实值大于 0 的位置计算 `|误差| / 真实值`，分母不加 epsilon，结果乘 100。WAPE 是 `sum(|误差|) / sum(|真实值|) × 100`。All-to-all 另外按最后一维的四个通道分别计算；通道顺序与这份函数里的名称一致。重算结果与 JSON 的差异见误差报告。

输入窗口都是 12 个 5 分钟时间片。`target_mode` 都是 `last-observed-step`，也就是 `target_slot == window_start_slot + 11`，不是下一个时间片。窗口不跨天：每个测试日 277 个窗口，起点是 0..276。时间片 0 对应仿真日 00:00，因为流量按 `event_time // 300` 分箱，仿真从 0 秒开始。

检查点由验证集选择，测试集不参与。STID、DCRNN、Graph WaveNet、S2S-GCN-LSTM、STGCN 的选择量是包含零值的验证集原始 MAE。TLAST 的选择量是原始尺度 Huber（delta=2）。TLAST 有 46 组选出的 epoch 不是验证 MAE 最小的 epoch。这些组的验证 MAE 比 MAE 最优 epoch 高，中位数 0.80%，最大 5.14%；其中 5 组达到或超过 2%。最大的一组是 accumulation end、p70，验证 MAE 0.329 对 0.313。测试指标仍按原始 MAE 报告，但对应的是 Huber 检查点。因此 TLAST 与其他模型相差约 2% 的测试名次不能当作稳定排序。后文里超过 10% 的存量差距大于这个选模差异，方向不受这一检查点替换的影响；具体会变多少则没有把 MAE 检查点再评一遍测试集，不能改写已有数字。

归一化都是训练集上的全局 z-score。单通道 inflow 使用各渗透率自己的输入 scaler 和所有渗透率共享的全量 inflow scaler。四状态训练在运行时用 `state_normalization.json` 替换目标 scaler，每个物理量一个标量，拟合划分是 train。配置文件里仍写着单通道 `normalization.json` 的路径，这是记录方式，不是又用了 inflow 的均值去缩放 accumulation。TLAST 的 `norm_flag=none` 是原论文的结构开关，损失仍然先反归一化再计算。

## 仍然存在的协议差异

这些差异不阻止报告测试集原始 MAE，但阻止把名次直接解释成结构优劣。

- 只有 seed 42。没有标准差。
- STGCN 的训练损失是归一化空间的 L2，最多 50 个 epoch，没有早停。其余模型的训练损失是原始尺度 MAE，TLAST 是原始尺度 Huber。
- DCRNN 有早停，耐心 50，最多 100 个 epoch。四状态里有多组在 74–93 epoch 停止。S2S 在四状态上的预算是 200，并且会早停。STID、Graph WaveNet、TLAST 固定 100 个 epoch，没有早停。
- 参数量大约是 S2S 7.1 万、STID 15.5 万、Graph WaveNet 32.8 万、DCRNN 37.1 万、TLAST 100 万、STGCN 109 万。输入通道数不同时，同一模型的参数量有几千的差别。
- STID 的 `input_dim` 包含时间通道。单通道任务是 3，四状态 all 输入是 6。交通状态通道仍分别是 1 和 4，清单里分开记录。
- 原始仿真和 3 秒相位仿真的全量标签不同。STID 在两套 inflow 上的零值比例并不相同。不能把两套 MAE 排进一张表。

问题条目也写在 `stats/audit_issues.json`，共 16 条。其中“训练损失”一条的自动说明把 STGCN 也写成了原始 MAE，以上面的列表为准。

## 这一阶段不能做的事

没有第二颗随机种子，不能报告均值和标准差。没有把测试集用于选模的证据，也没有发现跨天窗口。道路没有预先定义的低、中、高流量分组；误差报告里的分组是本次按测试集均值现划的。
