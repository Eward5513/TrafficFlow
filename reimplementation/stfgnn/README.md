# R-only STFGNN

本目录是原版 STFGNN（`reference/STFGNN/`，[MengzhangLI/STFGNN](https://github.com/MengzhangLI/STFGNN)，Li & Zhu, AAAI 2021，MXNet）的 PyTorch 迁移。

官方训练命令：

```bash
python main_4n0_3layer_12T_res.py --config config/XXXX/individual_3layer_12T.json
```

公共张量格式：`x = [B, 12, 56, 1]`，`y = [B, 1, 56, 1]`。
标签语义：`target_mode = last-observed-step`，即 `Y = full_flow[s+11]`。
空间图：只读 `stgcn_undirected_topology.npy`（原版 `connectivity` 0/1 无向图）。
时间图：原版 `Temporal_Graph_gen.py` 的带窗 DTW，只允许用 training 日的全量道路流量；本会话不生成正式 56 节点图。
融合图：`construct_adj_fusion(A_SG, A_TG, steps=4)` 得到 `4N×4N`。对角块为 `[T, S, S, T]`，相邻时间步同一节点双向边，`t0–t3` 放置整张 `A_TG`，再把 `(t0,t1)` 的同节点连接拷到跨两步位置，最后全部自环，不归一化。

GCN/GLU（以原版代码为准，先 `A@X` 再一个 FC 到 `2C'` 再 split）：

```text
Y = A_STFG X
P, Q = split(Linear(Y, 2C'))
out = P ⊙ sigmoid(Q)
```

每个 STFGCM：多层 GCN，每层取第二时间片节点 `[N:2N]`，再 element-wise max。
每个 STFGCL：长度 4、步长 1 滑窗，与 `kernel=(1,2), dilation=(1,3), pad=0` 的 gated CNN 残差相加，时间维 `T → T-3`。
官方 `module_type=individual`。三层之后 `12 → 9 → 6 → 3`，输出头 `horizon=1`。

gated CNN（原版左右支路）：

```text
sigmoid(Conv) ⊙ tanh(Conv)
```

## 路径

```text
ORIGINAL_STFGNN_ROOT  = reference/STFGNN
PREPARED_DATA_ROOT    = reimplementation/stgcn/prepared_data/r-only
SPATIAL_TOPOLOGY      = reimplementation/stgcn/prepared_data/r-only/adjacency_matrix/stgcn_undirected_topology.npy
TEMPORAL_GRAPH_SCRIPT = reimplementation/stfgnn/prepare_temporal_graph.py
TRAIN_SCRIPT          = reimplementation/stfgnn/train_r_only_stfgnn.py
OUTPUT_ROOT           = reimplementation/stfgnn/experiments/r_only
```

不得写入 DCRNN / GWN / STSGCN / STGCN 实验目录，也不得修改 prepared data、空间图或原版 STFGNN。

## 测试

当前环境未安装 pytest 时使用 unittest：

```bash
PYTHONPATH=. python3 -m unittest discover -s reimplementation/stfgnn/tests -v
PYTHONPATH=. python3 reimplementation/stfgnn/tests/run_and_report.py
python3 reimplementation/stfgnn/train_r_only_stfgnn.py --help
python3 reimplementation/stfgnn/train_r_only_stfgnn.py --smoke-test-only
```

`--smoke-test-only` 只跑一个 training batch，使用内存中的 `test_only_temporal_adjacency`，不写 `experiments/r_only`，不生成正式 DTW 图。
