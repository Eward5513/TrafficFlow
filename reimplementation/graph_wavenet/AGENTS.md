# R-only Graph WaveNet instructions

## Scope

This directory contains a PyTorch port of the original Graph WaveNet in
`reference/Graph-WaveNet` (Wu et al., IJCAI 2019). Shared helpers live in
`reimplementation/common/`.

Do not train from this `AGENTS.md` unless the user explicitly asks. The
training script is the entry point for a requested run.

## Original code

The authoritative original is `reference/Graph-WaveNet`. Do not substitute the
Shleifer et al. improvement linked from that README. Do not modify files under
`reference/Graph-WaveNet/`.

Official command:

```bash
python train.py --gcn_bool --adjtype doubletransition --addaptadj --randomadj
```

## Isolation from DCRNN

- Output is `reimplementation/graph_wavenet/experiments/r_only/`.
- Never write into `reimplementation/dcrnn/experiments/`.
- Never modify DCRNN source while a DCRNN run is in progress.
- Prepared NPZ and the directed adjacency are read-only.

If DCRNN is already using the only CPU/GPU, finish unit tests and
`--smoke-test-only` here and wait to start the 7-rate run.

## Inputs

- Prepared windows: `reimplementation/stgcn/prepared_data/r-only/pXX/{train,validation,test}.npz`
- Directed weighted adjacency: `reimplementation/dcrnn/prepared_data/r-only/adjacency_matrix/dcrnn_weighted_adjacency.npy`
- R nodes: `analysis/graph/r_graph/r_nodes.csv`

Do not use the STGCN symmetric W. Do not scale observed flow by `1/p`.

## Task definition

```text
X = observed_flow[s:s+12]
Y = full_flow[s+11]
target_slot == window_start_slot + 11
horizon = 1
```

This is last-observed-step reconstruction, not future forecasting.

## Training

```bash
python3 reimplementation/graph_wavenet/train_r_only_gwn.py --help
python3 reimplementation/graph_wavenet/train_r_only_gwn.py --smoke-test-only
python3 reimplementation/graph_wavenet/train_r_only_gwn.py
```

Checkpoint monitor: raw validation MAE including zeros. Test is evaluated
once after loading the best checkpoint.
