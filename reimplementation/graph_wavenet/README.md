# R-only Graph WaveNet

本目录是原版 Graph WaveNet（`reference/Graph-WaveNet/`，Wu et al., IJCAI 2019，PyTorch）的迁移。

官方训练命令：

```bash
python train.py --gcn_bool --adjtype doubletransition --addaptadj --randomadj
```

公共张量格式：`x = [B, 12, 56, 1]`，`y = [B, 1, 56, 1]`。  
标签语义：`target_mode = last-observed-step`，即 `Y = full_flow[s+11]`。  
静态图：有向 `dcrnn_weighted_adjacency.npy`，按原版 `doubletransition` 生成正反向 transition。  
自适应图：`softmax(ReLU(nodevec1 @ nodevec2), dim=1)`，`--randomadj` 随机初始化。

输入 12 步、receptive field 13：按原版 `engine.py` 在时间维**左侧**补 1 个 0。

## 路径

```text
ORIGINAL_GWN_ROOT     = reference/Graph-WaveNet
PREPARED_DATA_ROOT    = reimplementation/stgcn/prepared_data/r-only
DIRECTED_ADJACENCY    = reimplementation/dcrnn/prepared_data/r-only/adjacency_matrix/dcrnn_weighted_adjacency.npy
TRAIN_SCRIPT          = reimplementation/graph_wavenet/train_r_only_gwn.py
OUTPUT_ROOT           = reimplementation/graph_wavenet/experiments/r_only
```

不得写入 DCRNN / STGCN 实验目录，也不得修改 prepared data 或图文件。

## 训练

```bash
python3 reimplementation/graph_wavenet/train_r_only_gwn.py --help
python3 reimplementation/graph_wavenet/train_r_only_gwn.py --smoke-test-only
python3 reimplementation/graph_wavenet/train_r_only_gwn.py
```
