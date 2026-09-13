# R-only STSGCN

本目录是原版 STSGCN（`reference/STSGCN/`，Davidham3 / Song et al., AAAI 2020，MXNet）的 PyTorch 迁移。

官方训练命令：

```bash
python3 main.py --config config/PEMS03/individual_GLU_mask_emb.json
```

公共张量格式：`x = [B, 12, 56, 1]`，`y = [B, 1, 56, 1]`。  
标签语义：`target_mode = last-observed-step`，即 `Y = full_flow[s+11]`。  
空间图：只读 `stgcn_undirected_topology.npy`（原版 `connectivity` 0/1 无向图）。  
局部图：`construct_adj(A, steps=3)` 得到 `3N×3N`，相邻时间步同一节点双向边，全部自环，不归一化。

GCN/GLU（以原版代码为准，先 `A@X` 再一个 FC 到 `2C'` 再 split）：

```text
Y = A_ST X
P, Q = split(Linear(Y, 2C'))
out = P ⊙ sigmoid(Q)
```

每个 STSGCM：三层 GCN，每层取中心时间片 `[N:2N]`，再 element-wise max。  
每个 STSGCL：长度 3、步长 1 滑窗，时间维 `T → T-2`。官方 `module_type=individual`。  
四层之后 `12 → 10 → 8 → 6 → 4`，输出头 `horizon=1`。

## 路径

```text
ORIGINAL_STSGCN_ROOT  = reference/STSGCN
PREPARED_DATA_ROOT    = reimplementation/stgcn/prepared_data/r-only
SPATIAL_TOPOLOGY      = reimplementation/stgcn/prepared_data/r-only/adjacency_matrix/stgcn_undirected_topology.npy
TRAIN_SCRIPT          = reimplementation/stsgcn/train_r_only_stsgcn.py
OUTPUT_ROOT           = reimplementation/stsgcn/experiments/r_only
```

不得写入 DCRNN / GWN / STGCN 实验目录，也不得修改 prepared data、图文件或原版 STSGCN。

## 测试

当前环境未安装 pytest 时使用 unittest：

```bash
PYTHONPATH=. python3 -m unittest discover -s reimplementation/stsgcn/tests -v
PYTHONPATH=. python3 reimplementation/stsgcn/tests/run_and_report.py
python3 reimplementation/stsgcn/train_r_only_stsgcn.py --help
python3 reimplementation/stsgcn/train_r_only_stsgcn.py --smoke-test-only
```

`--smoke-test-only` 只跑一个 training batch，不写 `experiments/r_only`。
