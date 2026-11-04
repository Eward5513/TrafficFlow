# 四状态实验

数据是 3 秒相位仿真。六种模型都完成了 7 个渗透率上的 single-to-single、all-to-single 和 all-to-all。All-to-all 的四个通道分开比较。四个通道直接平均的 MAE 只留在 `metrics_long.csv` 的 `pooled_not_for_state_ranking` 行里，不作为名次依据。

Single-to-single 用同一状态的部分观测重建该状态的全量值。All-to-single 把四种状态一起作为输入，只预测一种状态，用来看输入端多变量。All-to-all 输入和输出都是四种状态，和 all-to-single 对比时输入相同，用来看输出端联合学习。两者相减才把两种机制分开；all-to-all 直接减 single-to-single 会把它们混在一起。

## 四种状态的尺度

| 状态 | 渗透率 | 六模型 MAE 中位数 | 最小 MAE | 最大 MAE | WAPE 中位数 | MAPE_nonzero 中位数 | 零值比例中位数 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Inflow | p05 | 2.653 | 2.565 | 3.618 | 22.0 | 37.6 | 37.3 |
| Inflow | p70 | 1.143 | 1.124 | 1.973 | 9.5 | 17.4 | 37.3 |
| Outflow | p05 | 2.720 | 2.550 | 3.681 | 22.6 | 37.7 | 37.3 |
| Outflow | p70 | 1.145 | 1.126 | 2.005 | 9.5 | 17.6 | 37.3 |
| Accumulation start | p05 | 0.873 | 0.674 | 1.030 | 56.3 | 67.5 | 67.6 |
| Accumulation start | p70 | 0.318 | 0.290 | 0.531 | 20.5 | 29.3 | 67.6 |
| Accumulation end | p05 | 0.872 | 0.677 | 1.029 | 56.2 | 67.5 | 67.6 |
| Accumulation end | p70 | 0.322 | 0.291 | 0.525 | 20.8 | 29.6 | 67.6 |

Inflow 和 outflow 的 MAE 大约是 accumulation 的三倍，因为流量本身更大。WAPE 和 MAPE 相反：accumulation 的 WAPE 中位数明显高于流量，零值比例也大约是流量的两倍。按相对误差，存量比流量更难重建。Accumulation start 和 end 的中位数几乎一样，不能从这些汇总里把期初和期末分成两种难度。

## Single-to-single

### Inflow

| 渗透率 | STID | STGCN | DCRNN | Graph WaveNet | S2S-GCN-LSTM | TLAST |
| --- | --- | --- | --- | --- | --- | --- |
| p05 | 2.652 (#3) | 2.653 (#4) | 2.731 (#5) | 2.578 (#2) | 3.618 (#6) | 2.565 (#1) |
| p10 | 2.459 (#4) | 2.439 (#2) | 2.475 (#5) | 2.457 (#3) | 3.156 (#6) | 2.385 (#1) |
| p20 | 2.160 (#1) | 2.167 (#2) | 2.182 (#4) | 2.218 (#5) | 2.768 (#6) | 2.177 (#3) |
| p30 | 1.945 (#2) | 1.950 (#3) | 1.954 (#4) | 1.967 (#5) | 2.554 (#6) | 1.944 (#1) |
| p40 | 1.730 (#2) | 1.748 (#5) | 1.746 (#4) | 1.745 (#3) | 2.397 (#6) | 1.725 (#1) |
| p50 | 1.531 (#1) | 1.557 (#5) | 1.548 (#3) | 1.544 (#2) | 2.245 (#6) | 1.551 (#4) |
| p70 | 1.124 (#1) | 1.159 (#5) | 1.135 (#3) | 1.128 (#2) | 1.973 (#6) | 1.150 (#4) |

p05 为 TLAST（MAE 2.565，与 Graph WaveNet 相差 0.5%）；p10 为 TLAST（MAE 2.385，与 STGCN 相差 2.3%）；p20 为 STID（MAE 2.160，与 STGCN 相差 0.3%）；p30 为 TLAST（MAE 1.944，与 STID 相差 0.1%）；p40 为 TLAST（MAE 1.725，与 STID 相差 0.3%）；p50 为 STID（MAE 1.531，与 Graph WaveNet 相差 0.9%）；p70 为 STID（MAE 1.124，与 Graph WaveNet 相差 0.4%）。

前两名差距：p05 前两名差距 0.5%；p10 前两名差距 2.3%，超过 2% 的报告阈值；p20 前两名差距 0.3%；p30 前两名差距 0.1%；p40 前两名差距 0.3%；p50 前两名差距 0.9%；p70 前两名差距 0.4%。

去掉 S2S 后，inflow 上没有模型以超过约 3% 的差距稳居第一。TLAST 在 p05 和 p10 的 MAE 最低，但 p05 只比 Graph WaveNet 低 0.5%。p20 以后 STID 与 TLAST 交替领先，差距多在 2% 以内。S2S 在七个渗透率上都是最后一名，p05 比第一名高 41%，p70 高 76%。

### Outflow

| 渗透率 | STID | STGCN | DCRNN | Graph WaveNet | S2S-GCN-LSTM | TLAST |
| --- | --- | --- | --- | --- | --- | --- |
| p05 | 2.729 (#4) | 2.711 (#3) | 2.795 (#5) | 2.653 (#2) | 3.681 (#6) | 2.550 (#1) |
| p10 | 2.509 (#4) | 2.486 (#2) | 2.524 (#5) | 2.501 (#3) | 3.202 (#6) | 2.409 (#1) |
| p20 | 2.187 (#2) | 2.197 (#3) | 2.217 (#4) | 2.259 (#5) | 2.864 (#6) | 2.162 (#1) |
| p30 | 1.957 (#2) | 1.969 (#3) | 1.980 (#4) | 1.999 (#5) | 2.603 (#6) | 1.944 (#1) |
| p40 | 1.739 (#1) | 1.762 (#3) | 1.763 (#4) | 1.764 (#5) | 2.433 (#6) | 1.756 (#2) |
| p50 | 1.539 (#1) | 1.568 (#5) | 1.561 (#3) | 1.554 (#2) | 2.231 (#6) | 1.564 (#4) |
| p70 | 1.126 (#1) | 1.163 (#5) | 1.140 (#3) | 1.131 (#2) | 2.005 (#6) | 1.150 (#4) |

p05 为 TLAST（MAE 2.550，与 Graph WaveNet 相差 4.0%）；p10 为 TLAST（MAE 2.409，与 STGCN 相差 3.2%）；p20 为 TLAST（MAE 2.162，与 STID 相差 1.1%）；p30 为 TLAST（MAE 1.944，与 STID 相差 0.7%）；p40 为 STID（MAE 1.739，与 TLAST 相差 1.0%）；p50 为 STID（MAE 1.539，与 Graph WaveNet 相差 1.0%）；p70 为 STID（MAE 1.126，与 Graph WaveNet 相差 0.5%）。

Outflow 的名次和 inflow 相同方向：低渗透率 TLAST，高渗透率 STID，S2S 始终最后。p05 上 TLAST 与第二名 Graph WaveNet 的差距是 4.0%，比 inflow 上的 0.5% 大。p70 上除 S2S 外，STGCN 落后 STID 3.3%，其余低于 2.2%。

Inflow 与 outflow 的配对：

| 模型 | 渗透率 | Inflow MAE | Outflow MAE | 右减左 | Inflow WAPE | Outflow WAPE |
| --- | --- | --- | --- | --- | --- | --- |
| STID | p05 | 2.652 | 2.729 | 0.076 | 22.0 | 22.7 |
| STID | p70 | 1.124 | 1.126 | 0.002 | 9.3 | 9.4 |
| STGCN | p05 | 2.653 | 2.711 | 0.058 | 22.0 | 22.5 |
| STGCN | p70 | 1.159 | 1.163 | 0.004 | 9.6 | 9.7 |
| DCRNN | p05 | 2.731 | 2.795 | 0.064 | 22.7 | 23.2 |
| DCRNN | p70 | 1.135 | 1.140 | 0.005 | 9.4 | 9.5 |
| Graph WaveNet | p05 | 2.578 | 2.653 | 0.075 | 21.4 | 22.0 |
| Graph WaveNet | p70 | 1.128 | 1.131 | 0.003 | 9.4 | 9.4 |
| S2S-GCN-LSTM | p05 | 3.618 | 3.681 | 0.062 | 30.1 | 30.6 |
| S2S-GCN-LSTM | p70 | 1.973 | 2.005 | 0.032 | 16.4 | 16.7 |
| TLAST | p05 | 2.565 | 2.550 | -0.015 | 21.3 | 21.2 |
| TLAST | p70 | 1.150 | 1.150 | -0.000 | 9.6 | 9.6 |

同一模型、同一渗透率下，outflow MAE 通常略高于 inflow，p05 的差别大约是 0 到 0.08 辆车，p70 大约是 0 到 0.03。WAPE 也很接近。没有出现一种模型只把其中一个方向做好、另一个方向明显失败的情况。

### Accumulation

Start：

| 渗透率 | STID | STGCN | DCRNN | Graph WaveNet | S2S-GCN-LSTM | TLAST |
| --- | --- | --- | --- | --- | --- | --- |
| p05 | 0.725 (#2) | 0.931 (#5) | 0.877 (#4) | 0.869 (#3) | 1.030 (#6) | 0.674 (#1) |
| p10 | 0.684 (#2) | 0.837 (#5) | 0.795 (#4) | 0.776 (#3) | 0.932 (#6) | 0.648 (#1) |
| p20 | 0.615 (#2) | 0.715 (#5) | 0.687 (#4) | 0.669 (#3) | 0.819 (#6) | 0.598 (#1) |
| p30 | 0.550 (#1) | 0.627 (#5) | 0.603 (#4) | 0.585 (#3) | 0.736 (#6) | 0.556 (#2) |
| p40 | 0.489 (#1) | 0.553 (#5) | 0.525 (#4) | 0.513 (#3) | 0.671 (#6) | 0.506 (#2) |
| p50 | 0.424 (#1) | 0.481 (#5) | 0.451 (#4) | 0.442 (#2) | 0.628 (#6) | 0.451 (#3) |
| p70 | 0.290 (#1) | 0.340 (#5) | 0.306 (#3) | 0.300 (#2) | 0.531 (#6) | 0.330 (#4) |

End：

| 渗透率 | STID | STGCN | DCRNN | Graph WaveNet | S2S-GCN-LSTM | TLAST |
| --- | --- | --- | --- | --- | --- | --- |
| p05 | 0.720 (#2) | 0.939 (#5) | 0.877 (#4) | 0.867 (#3) | 1.029 (#6) | 0.677 (#1) |
| p10 | 0.684 (#2) | 0.840 (#5) | 0.796 (#4) | 0.780 (#3) | 0.930 (#6) | 0.647 (#1) |
| p20 | 0.613 (#2) | 0.715 (#5) | 0.686 (#4) | 0.667 (#3) | 0.817 (#6) | 0.599 (#1) |
| p30 | 0.552 (#1) | 0.627 (#5) | 0.603 (#4) | 0.588 (#3) | 0.741 (#6) | 0.556 (#2) |
| p40 | 0.488 (#1) | 0.553 (#5) | 0.524 (#4) | 0.514 (#3) | 0.673 (#6) | 0.507 (#2) |
| p50 | 0.425 (#1) | 0.480 (#5) | 0.451 (#3) | 0.441 (#2) | 0.616 (#6) | 0.451 (#4) |
| p70 | 0.291 (#1) | 0.340 (#5) | 0.306 (#3) | 0.299 (#2) | 0.525 (#6) | 0.338 (#4) |

Start 的领先者：p05 为 TLAST（MAE 0.674，与 STID 相差 7.5%）；p10 为 TLAST（MAE 0.648，与 STID 相差 5.6%）；p20 为 TLAST（MAE 0.598，与 STID 相差 2.9%）；p30 为 STID（MAE 0.550，与 TLAST 相差 1.1%）；p40 为 STID（MAE 0.489，与 TLAST 相差 3.5%）；p50 为 STID（MAE 0.424，与 Graph WaveNet 相差 4.3%）；p70 为 STID（MAE 0.290，与 Graph WaveNet 相差 3.4%）。

这里的差距和流量不是一个量级。p05 上 TLAST 的 start MAE 是 0.674，STID 高 7.5%，Graph WaveNet 高 28.9%，STGCN 高 38.1%，S2S 高 52.7%。p70 反过来，STID 为 0.290，TLAST 高 14.0%，STGCN 高 17.5%，S2S 高 83.5%。End 的名次和差距与 start 平行。

Start 与 end：

| 模型 | 渗透率 | Accumulation start MAE | Accumulation end MAE | 右减左 | Accumulation start WAPE | Accumulation end WAPE |
| --- | --- | --- | --- | --- | --- | --- |
| STID | p05 | 0.725 | 0.720 | -0.005 | 46.7 | 46.4 |
| STID | p70 | 0.290 | 0.291 | 0.001 | 18.7 | 18.8 |
| STGCN | p05 | 0.931 | 0.939 | 0.008 | 60.0 | 60.6 |
| STGCN | p70 | 0.340 | 0.340 | -0.000 | 21.9 | 21.9 |
| DCRNN | p05 | 0.877 | 0.877 | 0.001 | 56.5 | 56.6 |
| DCRNN | p70 | 0.306 | 0.306 | 0.000 | 19.7 | 19.8 |
| Graph WaveNet | p05 | 0.869 | 0.867 | -0.003 | 56.0 | 55.9 |
| Graph WaveNet | p70 | 0.300 | 0.299 | -0.000 | 19.3 | 19.3 |
| S2S-GCN-LSTM | p05 | 1.030 | 1.029 | -0.001 | 66.4 | 66.3 |
| S2S-GCN-LSTM | p70 | 0.531 | 0.525 | -0.007 | 34.3 | 33.8 |
| TLAST | p05 | 0.674 | 0.677 | 0.003 | 43.5 | 43.7 |
| TLAST | p70 | 0.330 | 0.338 | 0.008 | 21.3 | 21.8 |

同一模型在两个存量上的 MAE 差不超过约 0.01。重建难度没有因为“期初”或“期末”分开。

RMSE、MAPE 和 WAPE 的完整名次在 `tables/rankings.csv`。Single-to-single 的曲线在 `figures/four_state/s2s_mae_<state>.png`、`s2s_rmse_<state>.png` 和 `s2s_wape_<state>.png`。

## 输入融合和联合输出

负的百分数表示后一种设置的 MAE 更低。融合差距是 all-to-single 减 single-to-single，再除以后者。多任务差距是 all-to-all 的对应通道减 all-to-single，输入都是四种状态。

- Inflow p05。联合输入相对单独输入：TLAST -2.5%, S2S-GCN-LSTM -2.1%, STGCN -0.4%, DCRNN -0.2%, STID 0.7%, Graph WaveNet 0.9%。联合输出相对单输出：Graph WaveNet -1.9%, TLAST -0.7%, DCRNN -0.5%, STID -0.5%, STGCN 1.2%, S2S-GCN-LSTM 1.4%。
- Inflow p30。联合输入相对单独输入：S2S-GCN-LSTM -1.7%, TLAST -1.0%, STID -0.6%, STGCN -0.2%, Graph WaveNet 0.0%, DCRNN 0.0%。联合输出相对单输出：Graph WaveNet -0.7%, STID -0.4%, DCRNN 0.1%, S2S-GCN-LSTM 0.4%, TLAST 0.8%, STGCN 0.9%。
- Inflow p70。联合输入相对单独输入：S2S-GCN-LSTM -3.3%, STID -0.3%, Graph WaveNet -0.2%, DCRNN -0.1%, STGCN 0.4%, TLAST 1.7%。联合输出相对单输出：S2S-GCN-LSTM -1.4%, TLAST -0.4%, Graph WaveNet -0.2%, STID -0.1%, DCRNN 0.1%, STGCN 0.9%。
- Outflow p05。联合输入相对单独输入：S2S-GCN-LSTM -2.8%, STID -1.0%, DCRNN -0.7%, STGCN -0.6%, TLAST -0.3%, Graph WaveNet 0.5%。联合输出相对单输出：TLAST -2.2%, Graph WaveNet -1.9%, DCRNN -0.6%, STID 0.0%, STGCN 0.8%, S2S-GCN-LSTM 1.5%。
- Outflow p30。联合输入相对单独输入：S2S-GCN-LSTM -2.0%, Graph WaveNet -1.2%, STGCN -0.7%, STID -0.5%, DCRNN -0.4%, TLAST 1.1%。联合输出相对单输出：TLAST -1.0%, STID -0.8%, DCRNN 0.1%, S2S-GCN-LSTM 0.3%, Graph WaveNet 0.3%, STGCN 0.9%。
- Outflow p70。联合输入相对单独输入：S2S-GCN-LSTM -5.0%, STID -0.4%, DCRNN -0.3%, STGCN 0.2%, Graph WaveNet 0.4%, TLAST 1.0%。联合输出相对单输出：Graph WaveNet -0.5%, TLAST -0.0%, STID -0.0%, S2S-GCN-LSTM 0.1%, DCRNN 0.1%, STGCN 0.7%。
- Accumulation start p05。联合输入相对单独输入：S2S-GCN-LSTM -4.0%, DCRNN -2.3%, Graph WaveNet -2.3%, STGCN -1.9%, STID -1.8%, TLAST 0.0%。联合输出相对单输出：STGCN -0.5%, Graph WaveNet 4.5%, S2S-GCN-LSTM 5.1%, DCRNN 6.5%, TLAST 13.0%, STID 21.9%。
- Accumulation start p30。联合输入相对单独输入：S2S-GCN-LSTM -4.3%, DCRNN -3.2%, Graph WaveNet -2.3%, STID -1.8%, STGCN -1.6%, TLAST 0.9%。联合输出相对单输出：STGCN -0.0%, Graph WaveNet 3.6%, DCRNN 4.4%, STID 5.4%, TLAST 5.9%, S2S-GCN-LSTM 11.8%。
- Accumulation start p70。联合输入相对单独输入：S2S-GCN-LSTM -9.7%, Graph WaveNet -1.8%, DCRNN -0.9%, TLAST -0.7%, STID -0.6%, STGCN -0.5%。联合输出相对单输出：STGCN 0.5%, STID 2.5%, Graph WaveNet 3.1%, DCRNN 3.3%, TLAST 3.8%, S2S-GCN-LSTM 25.5%。
- Accumulation end p05。联合输入相对单独输入：S2S-GCN-LSTM -3.1%, STGCN -2.3%, Graph WaveNet -1.8%, DCRNN -1.2%, STID -1.1%, TLAST 0.1%。联合输出相对单输出：STGCN -0.2%, Graph WaveNet 4.8%, S2S-GCN-LSTM 4.8%, DCRNN 5.9%, TLAST 12.6%, STID 22.4%。
- Accumulation end p30。联合输入相对单独输入：S2S-GCN-LSTM -4.7%, DCRNN -2.1%, Graph WaveNet -1.8%, STID -1.4%, STGCN -0.7%, TLAST 1.1%。联合输出相对单输出：STGCN -0.0%, Graph WaveNet 3.4%, DCRNN 4.2%, STID 5.1%, TLAST 6.1%, S2S-GCN-LSTM 12.2%。
- Accumulation end p70。联合输入相对单独输入：S2S-GCN-LSTM -7.3%, TLAST -2.2%, Graph WaveNet -1.6%, DCRNN -0.6%, STID -0.5%, STGCN 0.2%。联合输出相对单输出：STGCN 0.4%, STID 2.0%, TLAST 2.8%, DCRNN 3.2%, Graph WaveNet 4.5%, S2S-GCN-LSTM 24.8%。

流量上，两种改动大多落在 ±3% 以内，符号也不统一。S2S 从额外输入里得到的相对降幅稍大，大约 2%–10%，但它的绝对 MAE 仍然最高。对 STID、STGCN、DCRNN、Graph WaveNet 和 TLAST，现有结果不支持“把四种状态一起输入就能明显改善流量重建”。

存量上，额外输入的收益仍然小，多数模型在 0 到 −4% 之间。联合输出则系统地变差：p05 的 start 通道，STID 的 MAE 上升 21.9%，TLAST 上升 13.0%，DCRNN 上升 6.5%，Graph WaveNet 上升 4.5%，STGCN 基本不变。p70 上 S2S 的联合输出比单输出高 25.5%。因此，all-to-all 相对 single-to-single 的变化不能解释成“模型利用了额外状态”；存量上的主要变化来自输出端同时预测四个量，而且这个变化是损失。

每个状态的三套曲线画在 `figures/four_state/mechanism_<state>.png`。逐格数字在 `tables/four_state_mechanism_gaps.csv`。

## 这一节不能推出的结构结论

TLAST 在低渗透率存量上的 MAE 更低，STID 在高渗透率存量上更低。这是结果差异，不是已经证明代理节点或身份嵌入是原因。STGCN 只用 50 个 epoch 和归一化 L2，S2S 的参数量和训练预算都不同。要比较结构，需要先把损失、epoch 和选择指标对齐。
