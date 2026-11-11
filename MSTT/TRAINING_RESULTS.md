# MSTT 训练结果

整理时间：2026-11-11 18:41 +0800。下面是数据盘上已经写出测试指标的 MSTT 运行。冒烟运行不收录。`projection_raw_loss_weight` 为 0.25 和 1 的两次守恒投影扫描在写出指标前因显存不足失败，没有数字。10%–20% 波动渗透率里，原版 C 的四个周期，以及四状态独立输入编码的 30 分钟、1 小时、2 小时，都已写出五种子测试指标。全天没有跑独立输入编码。

数字都在 `/data5/tzhang174/TrafficFlow/xuancheng26/training_results_3s/`。固定渗透率的历史运行迁在：

```text
datasets/legacy_unknown/physical_legacy_unknown/MSTT/
    in_<input>__out_<target>/<rate>/<experiment_id>/seed_<model_seed>/<run_id>/
```

新的运行写在：

```text
datasets/<dataset_id>/physical_<version>/MSTT/
    in_<input>__out_<target>/<observation_key>/<experiment_id>/
    seed_<model_seed>/<run_id>/
```

模型种子都是 42。道路数 1406，输入 12 步，预测最后一个已观测时间片。通道顺序是 inflow、outflow、accumulation_start、accumulation_end。优化器是 AdamW，学习率 0.001，权重衰减 0.0001，余弦调度，预热 5 个 epoch。`ablation_100ep` 和其后的五种子运行开着 RoPE 与 QK-Norm。

## 数字怎么读

有两种测试。

一次测试来自 `test_metrics.json`。checkpoint 按那次运行自己的验证 MAE 选取，验证集没有对采样种子 42–46 取平均。早期目录的早停耐心是 20。`ablation_100ep` 没有早停，跑满 100 个 epoch。

五种子测试来自 `test_sampling_metrics.json`，或者只迁走了 `sampling_seed_means.json` 的那几条。训练用 sampling seed 42。评估种子是 42、43、44、45、46。表里的数字是不加权平均 ± 样本标准差（`ddof=1`）。测试没有参与选 checkpoint。

A2A 里有两种选法，结果分表列出。一种按四个状态的验证 MAE 选。一种按五个验证种子的 inflow MAE 平均选，目录里的 `effective_selection_metric` 是 `val_inflow_mae`。

`p05` 是固定 5% 渗透率。`p10_p20` 是 10%–20% 的波动渗透率。

## 固定 p05，I2I

I2I 的测试 MAE 就是 inflow MAE。FFN 隐层宽度都是 176，空间注意力是 global，嵌入是拼接。除非单独写出，自适应嵌入是关掉的，batch size 是 16，损失是原始尺度 MAE。

### 100 个 epoch 的结构比较

checkpoint 按一次验证 MAE 选取。这一组跑满 100 个 epoch。

| 实验 | 块 | d_model | 注意力头 | 自适应嵌入 | 参数量 | best epoch | 测试 MAE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `single_ts_ffn_100ep` | 1 层 `[T, S, FFN]` | 64 | 4 | 无 | 92,569 | 82 | 2.474 |
| `single_ts_ffn_temporal_head1_100ep` | 1 层 `[T, S, FFN]` | 64 | 时间 1，空间 4 | 无 | 92,665 | 69 | 2.492 |
| `single_ts_ffn_spatial_head1_100ep` | 1 层 `[T, S, FFN]` | 64 | 时间 4，空间 1 | 无 | 92,665 | 51 | 2.548 |
| `single_ts_ffn_both_head1_100ep` | 1 层 `[T, S, FFN]` | 64 | 时间 1，空间 1 | 无 | 92,761 | 79 | 2.476 |
| `single_ts_ffn_wide_head128_100ep` | 1 层 `[T, S, FFN]` | 64 | 4，每头 128 | 无 | 322,393 | 68 | 2.524 |
| `single_layer_tfsf_100ep` | 1 层 `[T, FFN, S, FFN]` | 64 | 4 | 无 | 126,425 | 82 | 2.476 |
| `single_sftf_100ep` | 1 层 `[S, FFN, T, FFN]` | 64 | 4 | 无 | 126,425 | 56 | 2.509 |
| `control_global_100ep` | 2 层 `[T, S, FFN]` | 64 | 4 | 无 | 159,385 | 85 | 2.494 |
| `ffn_after_each_attention_100ep` | 2 层 `[T, FFN, S, FFN]` | 64 | 4 | 无 | 227,097 | 77 | 2.493 |
| `two_layer_sftf_100ep` | 2 层 `[S, FFN, T, FFN]` | 64 | 4 | 无 | 227,097 | 85 | 2.474 |
| `base_concat_128_100ep` | 2 层 `[T, S, FFN]` | 128 | 4 | 无 | 449,841 | 50 | 2.476 |
| `adaptive_concat_128_100ep` | 2 层 `[T, S, FFN]` | 128 | 4 | 有，64 维拼接 | 1,504,729 | 37 | 2.461 |
| `ttfssf_adaptive128_100ep` | `[T, T, FFN, S, S, FFN]` | 128 | 4 | 有，64 维拼接 | 1,504,729 | 34 | 2.448 |
| `single_tsf_adaptive128_100ep` | 1 层 `[T, S, FFN]` | 128 | 4 | 有，64 维拼接 | 1,305,561 | 60 | 2.448 |

自适应嵌入是 12×1406×64，拼在基础嵌入后面。基础嵌入是 value 32、道路 16、TOD 8、DOW 8。`single_tsf_adaptive128_100ep` 是后来冻结进 `configs/default.json` 的骨干，FFN 宽度 176 也来自这次运行。它和 `ttfssf_adaptive128_100ep` 的一次测试 MAE 在三位小数上都是 2.448。

同一套单层骨干后来又按五种子协议训了一次，实验号在目录里是 `baseline`，run id `legacy_5a8a9edfeb7ee5cc`。checkpoint 按种子 42–46 的验证 MAE 平均选取，best epoch 69，验证 MAE 2.278。测试 MAE 是 **2.442 ± 0.008**。这是另一次训练，不是把上面 epoch 60 的 checkpoint 重新评分。

### 同一骨干上的 Huber 损失

结构与冻结骨干相同，1,305,561 个参数。checkpoint 按五种子验证 MAE 的平均选取，损失换成原始尺度 Huber。

| 实验 | delta | best epoch | 验证 MAE | 测试 MAE |
| --- | --- | --- | --- | --- |
| `huber_delta1` | 1 | 65 | 2.285 | 2.466 ± 0.009 |
| `huber_delta2` | 2 | 38 | 2.289 | 2.467 ± 0.010 |

### 更早的两层栈

早停耐心是 20。checkpoint 按一次验证损失选取。测试是一次测试。

| 实验 | 配置 | 参数量 | best epoch | 实际 epoch | 测试 MAE |
| --- | --- | --- | --- | --- | --- |
| `baseline` / `legacy_df70edddfabda9f7` | 2 层，d_model 64，稀疏空间注意力 | 243,489 | 61 | 82 | 2.651 |
| `v0.2_concat_embedding_bs16` | 2 层，d_model 64，稀疏空间注意力，嵌入改为拼接 | 159,417 | 67 | 88 | 2.613 |
| `v0.3_concat_adaptive_bs16` | 在 v0.2 上打开自适应嵌入 | 1,239,225 | 55 | 76 | 2.608 |
| `v0.4_global` | 2 层，d_model 64，拼接嵌入，空间注意力改为 global | 159,385 | 17 | 38 | 2.534 |

## 固定 p05，A2I

输入是四个状态，输出只有 inflow。

| 实验 | 配置 | 选 checkpoint | best epoch | 测试 MAE |
| --- | --- | --- | --- | --- |
| `v0.2_concat_embedding_bs16` | 2 层 d_model 64，稀疏空间注意力，拼接嵌入，159,513 参数，早停耐心 20 | 一次验证损失 | 53 | 2.579 |
| `single_tsf_adaptive128_all2i_100ep` | 冻结骨干，1,305,657 参数，无守恒项 | 一次验证 MAE，验证值 2.323 | 59 | 2.505 ± 0.007 |

后一条的五种子数字在 `sampling_seed_means.json`。测试种子没有参与选 checkpoint。

## 固定 p05，A2A

输入和输出都是四个状态。总 MAE 是四个状态的算术平均。下面先给总表，再给分状态。

无守恒的单层骨干是 1,310,268 个参数。加上守恒投影 MLP 后是 1,314,656。soft 只改损失，参数量仍是 1,310,268。联合注意力是 1,248,928。

### 按五个验证种子的 inflow MAE 选取

这些运行的 `effective_selection_metric` 是 `val_inflow_mae`。投影实验的主损失是投影后四个状态的 MAE，另外加上 `projection_raw_loss_weight` 乘投影前四个状态的 MAE。

| 实验 | 相对冻结骨干的改动 | best epoch | 验证 inflow MAE | 测试总 MAE | 测试 inflow MAE |
| --- | --- | --- | --- | --- | --- |
| `conservation_best_backbone__projected` | 守恒投影，原始项权重 0.5 | 64 | 2.256 | 1.585 ± 0.004 | 2.438 ± 0.008 |
| `conservation_raw_weight_sweep__lambda_2` | 守恒投影，原始项权重 2 | 50 | 2.255 | 1.598 ± 0.003 | 2.440 ± 0.006 |
| `conservation_best_backbone__soft` | soft 守恒惩罚，原始项权重 0.5 | 53 | 2.275 | 1.600 ± 0.004 | 2.452 ± 0.008 |
| `single_tsf_adaptive128_all2a_100ep` | 无守恒项 | 55 | 2.258 | 1.599 ± 0.002 | 2.454 ± 0.004 |
| `joint_st3_projected` | 用居中 3 步联合注意力替换 T 和 S，守恒投影，原始项权重 0.5 | 21 | 2.318 | 1.711 ± 0.005 | 2.491 ± 0.009 |
| `C_inflow_aux` | 守恒投影仍在；只监督投影后的 inflow，再加上 0.5 倍投影前守恒误差 | 37 | 2.326 | 3.313 ± 0.004 | 2.504 ± 0.007 |

这一组里测试 inflow MAE 最低的是原始项权重 0.5 的守恒投影。权重 2 的验证 inflow MAE 是 2.255，测试 inflow MAE 是 2.440。`C_inflow_aux` 的另外三个状态没有任务监督，所以总 MAE 到了 3.313，其中 outflow 5.079、accumulation_start 4.058、accumulation_end 1.609。

分状态的测试 MAE：

| 实验 | inflow | outflow | accumulation_start | accumulation_end |
| --- | --- | --- | --- | --- |
| 投影，权重 0.5 | 2.438 ± 0.008 | 2.443 ± 0.007 | 0.728 ± 0.001 | 0.730 ± 0.001 |
| 投影，权重 2 | 2.440 ± 0.006 | 2.451 ± 0.005 | 0.745 ± 0.001 | 0.755 ± 0.001 |
| soft | 2.452 ± 0.008 | 2.458 ± 0.007 | 0.744 ± 0.001 | 0.748 ± 0.001 |
| 无守恒 | 2.454 ± 0.004 | 2.458 ± 0.004 | 0.740 ± 0.001 | 0.745 ± 0.001 |
| 联合 3 步注意力 | 2.491 ± 0.009 | 2.543 ± 0.008 | 0.887 ± 0.001 | 0.924 ± 0.001 |
| `C_inflow_aux` | 2.504 ± 0.007 | 5.079 ± 0.013 | 4.058 ± 0.004 | 1.609 ± 0.001 |

权重 0.25 和 1 的投影扫描没有完成。两次都在 GPU 上分配内存时失败，目录里没有测试指标。

四路独立 value MLP 的实验 A 没有进入上面的完成目录。它的配置在 `configs/experiment_A_separate_value_mlp_projected.json`，日志停在 p05 的训练过程中，没有五种子测试结果。

### 按一次四个状态验证 MAE 选取

结构和上一节对应实验相同。checkpoint 用的是一次验证的四状态 MAE，五种子测试是事后评的，写在 `sampling_seed_means.json`。

| 实验 | best epoch | 验证总 MAE | 测试总 MAE | 测试 inflow MAE |
| --- | --- | --- | --- | --- |
| 投影，权重 0.5 / `legacy_db96a4616d8860c9` | 78 | 1.478 | 1.587 ± 0.004 | 2.442 ± 0.009 |
| 无守恒 / `legacy_98fdc2fb17e91749` | 78 | 1.476 | 1.585 ± 0.002 | 2.444 ± 0.004 |
| soft / `legacy_f795d4325edec350` | 71 | 1.494 | 1.594 ± 0.004 | 2.460 ± 0.007 |

| 实验 | inflow | outflow | accumulation_start | accumulation_end |
| --- | --- | --- | --- | --- |
| 投影，权重 0.5 | 2.442 ± 0.009 | 2.446 ± 0.007 | 0.729 ± 0.001 | 0.732 ± 0.001 |
| 无守恒 | 2.444 ± 0.004 | 2.448 ± 0.004 | 0.723 ± 0.001 | 0.724 ± 0.001 |
| soft | 2.460 ± 0.007 | 2.468 ± 0.007 | 0.724 ± 0.001 | 0.726 ± 0.001 |

更早的 `v0.2_concat_embedding_bs16` A2A 是 2 层 d_model 64、稀疏空间注意力、拼接嵌入，161,820 个参数。它跑满 100 个 epoch，best epoch 83，按一次验证损失选取。一次测试的总 MAE 是 1.717，inflow MAE 是 2.568。

## 波动渗透率

实验沿用固定 p05 上测试 inflow MAE 最低的那套结构：单层 `[T, S, FFN]`、d_model 128、自适应嵌入、守恒投影、原始项权重 0.5，1,314,656 个参数。任务是 A2A。checkpoint 按种子 42–46 的验证 inflow MAE 平均选取，再在对应测试种子上评分。渗透率在 0.10 到 0.20 之间，观测标识 `p10_p20`，物理版本 `independent_fcd_v1`。四个周期都跑满 100 个 epoch，没有早停。测试没有参与选 checkpoint。

全天的测试 inflow MAE 最低，是 **2.338 ± 0.008**，测试总 MAE 是 **1.522 ± 0.005**。1 小时的测试 inflow MAE 最高，是 **2.558 ± 0.009**。

| 波动周期 | best epoch | 验证 inflow MAE | 测试总 MAE | 测试 inflow MAE | run id |
| --- | --- | --- | --- | --- | --- |
| 30 分钟 | 72 | 2.162 | 1.589 ± 0.003 | 2.459 ± 0.006 | `20261110T132316_29bc19aa` |
| 1 小时 | 74 | 2.174 | 1.644 ± 0.005 | 2.558 ± 0.009 | `20261110T161911_f3edf8f9` |
| 2 小时 | 78 | 2.209 | 1.579 ± 0.005 | 2.447 ± 0.009 | `20261110T183956_a9bd76cc` |
| 全天 | 51 | 2.152 | 1.522 ± 0.005 | 2.338 ± 0.008 | `20261110T210053_dceca753` |

| 波动周期 | inflow | outflow | accumulation_start | accumulation_end |
| --- | --- | --- | --- | --- |
| 30 分钟 | 2.459 ± 0.006 | 2.469 ± 0.007 | 0.710 ± 0.000 | 0.717 ± 0.001 |
| 1 小时 | 2.558 ± 0.009 | 2.575 ± 0.009 | 0.718 ± 0.001 | 0.723 ± 0.001 |
| 2 小时 | 2.447 ± 0.009 | 2.462 ± 0.008 | 0.702 ± 0.001 | 0.706 ± 0.001 |
| 全天 | 2.338 ± 0.008 | 2.353 ± 0.008 | 0.697 ± 0.001 | 0.701 ± 0.001 |

目录是 `datasets/variable_<period>_p10_p20/physical_independent_fcd_v1/MSTT/in_all__out_all/p10_p20/conservation_projected/seed_42/<run_id>/`。`<period>` 分别是 `30min`、`1h`、`2h`、`day`。

### 四状态独立输入编码

这一组和上一节对照。唯一的结构差别是 Value Embedding：上一节是 `joint_linear`，四个通道一起映射到 32 维；这里是 `separate_mlp`，每个通道一个独立的 `Linear(1, 8) → SiLU → Linear(8, 8)`，再拼成 32 维，参数不共享。参数量是 1,314,848。任务、守恒投影、原始项权重 0.5、100 个 epoch、无早停、五种子协议都和上一节相同。实验号是 `separate_value_mlp_projected`。全天没有跑。

比较时看测试 inflow MAE。30 分钟是 **2.374 ± 0.006**，1 小时是 **2.454 ± 0.018**，2 小时是 **2.391 ± 0.013**。同一周期的 `joint_linear` 分别是 2.459、2.558、2.447。

| 波动周期 | best epoch | 验证 inflow MAE | 测试 inflow MAE | 测试总 MAE | run id |
| --- | --- | --- | --- | --- | --- |
| 30 分钟 | 72 | 2.159 | 2.374 ± 0.006 | 1.560 ± 0.003 | `20261111T145211_d7a3b081` |
| 1 小时 | 65 | 2.146 | 2.454 ± 0.018 | 1.606 ± 0.009 | `20261111T145211_f653a402` |
| 2 小时 | 85 | 2.180 | 2.391 ± 0.013 | 1.544 ± 0.007 | `20261111T145312_7031d8cc` |

投影后的分状态测试 MAE：

| 波动周期 | inflow | outflow | accumulation_start | accumulation_end |
| --- | --- | --- | --- | --- |
| 30 分钟 | 2.374 ± 0.006 | 2.420 ± 0.005 | 0.724 ± 0.001 | 0.724 ± 0.001 |
| 1 小时 | 2.454 ± 0.018 | 2.511 ± 0.017 | 0.732 ± 0.002 | 0.728 ± 0.002 |
| 2 小时 | 2.391 ± 0.013 | 2.404 ± 0.013 | 0.689 ± 0.001 | 0.691 ± 0.001 |

投影前的 inflow MAE 分别是 2.374 ± 0.006、2.454 ± 0.018、2.393 ± 0.013。投影前守恒 MAE 分别是 0.103、0.090、0.097。投影后守恒 MAE 约 6×10⁻⁷。

目录是 `datasets/variable_<period>_p10_p20/physical_independent_fcd_v1/MSTT/in_all__out_all/p10_p20/separate_value_mlp_projected/seed_42/<run_id>/`。`<period>` 分别是 `30min`、`1h`、`2h`。