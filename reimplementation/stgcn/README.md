# R-only STGCN（PyTorch 迁移）

本目录包含两件事：已经完成的 R-only 训练数据 / 加权邻接矩阵准备，以及原版 STGCN 的 PyTorch 迁移代码。

原版代码：`reference/STGCN_IJCAI-18/`（VeritasYin / IJCAI-18，TensorFlow 1.x）。  
公共张量格式：`x = [B, 12, 56, 1]`，`y = [B, 1, 56, 1]`。  
标签语义：`target_mode = last-observed-step`，即 `Y = full_flow[s+11]`，不是原版 PeMS 的未来 `n_pred` 步。

**当前状态：代码已写完。本文中的训练 / 评估 / 冒烟命令是未来入口示例，尚未作为正式实验执行。** 仓库里没有正式 checkpoint、预测文件或实验指标。

## 路径

相对仓库根目录：

```text
PROJECT_ROOT            = .
ORIGINAL_STGCN_ROOT     = reference/STGCN_IJCAI-18
PREPARED_DATA_ROOT      = reimplementation/stgcn/prepared_data/r-only
WEIGHTED_ADJACENCY      = reimplementation/stgcn/prepared_data/r-only/adjacency_matrix/stgcn_weighted_adjacency.npy
GRAPH_METADATA          = reimplementation/stgcn/prepared_data/r-only/adjacency_matrix/stgcn_weighted_adjacency_metadata.json
GRAPH_VALIDATION        = reimplementation/stgcn/prepared_data/r-only/adjacency_matrix/stgcn_weighted_adjacency_validation.json
R_NODES_CSV             = analysis/graph/r_graph/r_nodes.csv
NODE_MAPPING_CSV        = reimplementation/stgcn/prepared_data/r-only/node_mapping.csv
NORMALIZATION_JSON      = reimplementation/stgcn/prepared_data/r-only/normalization.json
SPLIT_MANIFEST          = reimplementation/stgcn/prepared_data/r-only/split_manifest.json
DATASET_METADATA        = reimplementation/stgcn/prepared_data/r-only/dataset_metadata.json
DATA_VALIDATION         = reimplementation/stgcn/prepared_data/r-only/validation_summary.json
REIMPLEMENTATION_ROOT   = reimplementation
FUTURE_OUTPUT_ROOT      = reimplementation/experiments/r_only/stgcn
```

不要移动、复制或重命名上述数据文件。训练代码读取已经完成高斯加权的 `stgcn_weighted_adjacency.npy`，不再执行 `W/10000` 或 `weight_matrix()`。

## 目录结构

```text
reimplementation/
├── common/                         # 后续模型可复用
│   ├── data/r_only_npz_dataset.py
│   ├── graph/adjacency.py          # 只加载/校验 W，不再高斯加权
│   ├── metrics/traffic_metrics.py
│   └── utils/                      # seed、checkpoint、JSONL、SHA256、原子写
├── stgcn/                          # STGCN 专用数学结构与训练入口
│   ├── layers.py / graph.py / model.py / losses.py
│   ├── engine.py / validation.py
│   ├── train_r_only_stgcn.py
│   ├── evaluate_r_only_stgcn.py
│   ├── monitor_training.py
│   ├── configs/r_only_stgcn.json
│   ├── tests/
│   └── prepared_data/              # 已有数据准备脚本与产物，本阶段不修改
└── experiments/                    # 未来训练输出；本次不得创建正式结果
```

## 原版 → PyTorch 映射

| 原版 | PyTorch |
|---|---|
| `inputs[:, 0:n_his]` NHWC | 公共输入 `[B,T,V,C]`；`Conv2d` 前 `permute(0,3,1,2)` |
| `temporal_conv_layer` VALID + GLU/ReLU | `TemporalConvLayer`，残差裁剪 `[:, Kt-1:T]` |
| `gconv` + `spatio_conv_layer` | `SpatialConvLayer.gconv`，Chebyshev kernel 为 buffer |
| `st_conv_block` × 2 | `STConvBlock` × 2，通道 `[[1,32,64],[64,32,128]]` |
| `layer_norm` 对节点+通道 | `SpatialTemporalNorm`，`unbiased=False` |
| `output_layer` | `OutputLayer`：剩余时间维 GLU → LN → sigmoid k=1 → 1×1 |
| `tf.nn.l2_loss(y - label)` | `0.5 * sum((pred-y)^2)`，不把 collected L2 加入反向传播 |
| `scaled_laplacian` / `cheb_poly_approx` | `reimplementation/stgcn/graph.py` |
| `weight_matrix()` | **跳过**；直接读最终 W |
| `n_pred` 未来切片 | **不使用**；标签固定为窗口最后一步全量流量 |

默认超参数来自原版 `main.py` / `trainer.py`：`n_his=12`，`Ks=Kt=3`，`batch_size=50`，`epochs=50`，`lr=1e-3`，RMSProp `alpha=0.9` `eps=1e-10`，每 5 个 epoch 对应的 batch 数把学习率乘 `0.7`，训练时 `keep_prob=1.0`。因任务适配而改动的项写在 `configs/r_only_stgcn.json` 的 `adaptations`。

不可避免的框架差异见下文「框架差异」。

## 单元测试

测试只用合成数据，不依赖正式 NPZ / 邻接矩阵理解基本逻辑。

```bash
cd /mnt/f/workspace/TrafficFlow
python3 -m unittest discover -s reimplementation/stgcn/tests -v
```

覆盖：拉普拉斯与 Chebyshev、Dataset 字段与最后一步标签、DataLoader 末 batch、模型形状与 graph buffer、MAE/RMSE/MAPE_nonzero/WAPE、checkpoint 渗透率 / 图哈希校验。

## 未来命令（不要在未准备好输出目录时误跑正式训练）

在仓库根目录、WSL 中执行。所有路径相对仓库根目录。

### 预检查 + 冒烟（不写正式 checkpoint）

```bash
python3 reimplementation/stgcn/train_r_only_stgcn.py \
  --config reimplementation/stgcn/configs/r_only_stgcn.json \
  --data-root reimplementation/stgcn/prepared_data/r-only \
  --weighted-adjacency reimplementation/stgcn/prepared_data/r-only/adjacency_matrix/stgcn_weighted_adjacency.npy \
  --r-nodes analysis/graph/r_graph/r_nodes.csv \
  --normalization reimplementation/stgcn/prepared_data/r-only/normalization.json \
  --split-manifest reimplementation/stgcn/prepared_data/r-only/split_manifest.json \
  --output-root reimplementation/experiments/r_only/stgcn \
  --rates 5 10 20 30 40 50 70 \
  --n-his 12 \
  --target-mode last-observed-step \
  --seed 42 \
  --device auto \
  --smoke-test-only \
  --overwrite
```

### 单渗透率训练

```bash
python3 reimplementation/stgcn/train_r_only_stgcn.py \
  --config reimplementation/stgcn/configs/r_only_stgcn.json \
  --data-root reimplementation/stgcn/prepared_data/r-only \
  --weighted-adjacency reimplementation/stgcn/prepared_data/r-only/adjacency_matrix/stgcn_weighted_adjacency.npy \
  --r-nodes analysis/graph/r_graph/r_nodes.csv \
  --normalization reimplementation/stgcn/prepared_data/r-only/normalization.json \
  --split-manifest reimplementation/stgcn/prepared_data/r-only/split_manifest.json \
  --output-root reimplementation/experiments/r_only/stgcn \
  --rate 5 \
  --n-his 12 \
  --target-mode last-observed-step \
  --seed 42 \
  --device auto \
  --overwrite
```

### 全部渗透率训练

每个渗透率会重新 seed、重新初始化模型，使用同一张图和同一套目标 scaler，不继承其他渗透率 checkpoint。

```bash
python3 reimplementation/stgcn/train_r_only_stgcn.py \
  --config reimplementation/stgcn/configs/r_only_stgcn.json \
  --data-root reimplementation/stgcn/prepared_data/r-only \
  --weighted-adjacency reimplementation/stgcn/prepared_data/r-only/adjacency_matrix/stgcn_weighted_adjacency.npy \
  --r-nodes analysis/graph/r_graph/r_nodes.csv \
  --normalization reimplementation/stgcn/prepared_data/r-only/normalization.json \
  --split-manifest reimplementation/stgcn/prepared_data/r-only/split_manifest.json \
  --output-root reimplementation/experiments/r_only/stgcn \
  --rates 5 10 20 30 40 50 70 \
  --n-his 12 \
  --target-mode last-observed-step \
  --seed 42 \
  --device auto \
  --overwrite
```

`--resume` 必须与 `--rate` 一起使用，禁止跨渗透率恢复。

### 评估与监控

```bash
python3 reimplementation/stgcn/evaluate_r_only_stgcn.py \
  --config reimplementation/stgcn/configs/r_only_stgcn.json \
  --checkpoint reimplementation/experiments/r_only/stgcn/p05/seed_42/best_checkpoint.pt \
  --output-dir reimplementation/experiments/r_only/stgcn/p05/seed_42 \
  --rate 5 \
  --overwrite

python3 reimplementation/stgcn/monitor_training.py \
  --output-root reimplementation/experiments/r_only/stgcn \
  --once
```

监控脚本只读状态、日志和 checkpoint 时间，不改训练文件、不启动第二份训练。

## 未来输出

正式训练时应写入 `reimplementation/experiments/r_only/stgcn/`：

```text
experiment_manifest.json
resolved_config.json
environment.json
source_code_manifest.json
data_runtime_validation.json
graph_runtime_validation.json
smoke_test.json          # 仅 --smoke-test-only
overall_summary.csv
overall_summary.json
training_log.jsonl
pXX/seed_42/
  best_checkpoint.pt
  last_checkpoint.pt
  training_history.csv
  validation_metrics.json
  test_metrics.json          # 由 evaluate 脚本写；训练循环不用 test 选模
  node_metrics.csv
  day_metrics.csv
  validation_predictions.npz
  test_predictions.npz
  run_metadata.json
  run_validation.json
```

## 框架差异

这些差异无法在 PyTorch 中做成与 TensorFlow 1.x 逐 bit 相同，已在代码注释中标明：

1. 卷积核布局：TF NHWC `[Kt,1,C_in,C_out]` vs PyTorch `[C_out,C_in,Kt,1]`。数学上在入口/出口做 `permute`，不改变 VALID 时间卷积。
2. 权重初始化：TF `get_variable` 默认 Glorot uniform；PyTorch 使用 `xavier_uniform_` / 等价 fan-in fan-out，随机实现不同，不能复现 TF 数值。
3. RMSProp：`torch.optim.RMSprop(alpha=0.9, eps=1e-10)` 对应 TF `RMSPropOptimizer` 的 `decay`/`epsilon`，内部累加器实现仍可能不同。
4. 学习率：按原版 `exponential_decay(..., decay_steps=5*epoch_step, staircase=True)`，在每个 **training batch** 后 `scheduler.step()`。与「每个 epoch 调一次」的常见 PyTorch 写法不同。
5. `tf.nn.dropout(keep_prob)` vs `nn.Dropout(p=1-keep_prob)`。原版训练始终 `keep_prob=1.0`，因此实际无 dropout。
6. 不实现原版 `copy_loss`（它比较历史最后一帧与 **未来** 第一帧）以及 recursive multi-step inference。
7. 评估指标使用本项目定义的原始车辆计数 MAE/RMSE/MAPE_nonzero/WAPE，不用原版 tester 的未来多步协议。

## 禁止事项

- 不要对 `stgcn_weighted_adjacency.npy` 再做高斯核、`/10000`、自环或二值化。
- 不要重新拟合 `normalization.json`，不要用输入 scaler 反归一化输出，不要乘 `1/p`。
- 不要把标签改成 `full_flow[s+12]`。
- 不要用 test 做 early stopping 或选 best checkpoint。
- 不要默认裁剪负预测。
- 不要把其他模型的 attention / 自适应邻接 / embedding 加进 STGCN。
