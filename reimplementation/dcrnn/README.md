# R-only DCRNN（PyTorch 迁移）

本目录包含两件事：已经完成的 R-only **有向**加权邻接矩阵准备，以及原版 DCRNN 的 PyTorch 迁移。

原版代码：`reference/dcrnn/`（Yaguang Li / ICLR 2018，TensorFlow 1.x）。  
公共张量格式：`x = [B, 12, 56, 1]`，`y = [B, 1, 56, 1]`。  
标签语义：`target_mode = last-observed-step`，即 `Y = full_flow[s+11]`。  
图：有向 `dcrnn_weighted_adjacency.npy`，运行时按官方配置构造 `dual_random_walk` 正反向 support。

## 路径

```text
ORIGINAL_DCRNN_ROOT   = reference/dcrnn
PREPARED_DATA_ROOT    = reimplementation/stgcn/prepared_data/r-only
DCRNN_ADJACENCY_ROOT  = reimplementation/dcrnn/prepared_data/r-only/adjacency_matrix
TRAIN_SCRIPT          = reimplementation/dcrnn/train_r_only_dcrnn.py
OUTPUT_ROOT           = reimplementation/dcrnn/experiments/r_only
```

## 训练

```bash
python3 reimplementation/dcrnn/train_r_only_dcrnn.py --help
python3 reimplementation/dcrnn/train_r_only_dcrnn.py
```

七种渗透率独立训练，共享同一张有向图。超参数来自原版 METR-LA yaml（Adam / `dual_random_walk` / 100 epoch），仅将 `horizon` 改为 1、`input_dim` 改为 1、节点数改为 56。
