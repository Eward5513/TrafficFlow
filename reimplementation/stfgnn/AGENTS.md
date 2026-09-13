# R-only STFGNN instructions

## Scope

This directory is a PyTorch port of the original STFGNN in
`reference/STFGNN` (MengzhangLI / Li & Zhu, AAAI 2021, MXNet). Shared
helpers live in `reimplementation/common/`.

Do not train from this `AGENTS.md` unless the user explicitly asks. The
training script is the entry point for a requested run. This migration
session implements code, unit tests, and a single training-batch smoke
test only.

## Original code

The authoritative original is `reference/STFGNN` =
https://github.com/MengzhangLI/STFGNN. Do not substitute a third-party
PyTorch STFGNN. Do not modify files under `reference/STFGNN/`. Do not
install MXNet, fastdtw, dtaidistance, PyG, DGL, or graphviz.

Official command:

```bash
python main_4n0_3layer_12T_res.py --config config/XXXX/individual_3layer_12T.json
```

Official module settings: `individual`, `GLU`, `use_mask=true`, spatial
and temporal embeddings on, `first_layer_embedding_size=64`, three STFGCL
blocks with filters `[64,64,64]`, Adam `lr=1e-3`, `batch_size=32`,
`epochs=200`, Huber `rho=1`. Spatial graph is 0/1 connectivity. Temporal
graph is custom Sakoe-Chiba DTW with `sparsity=0.01`.

## Isolation

- Output is `reimplementation/stfgnn/experiments/r_only/`.
- Never write into DCRNN, Graph WaveNet, STSGCN, or STGCN experiment directories.
- Never modify DCRNN, Graph WaveNet, or STSGCN source while those runs are in progress.
- Prepared NPZ and spatial adjacency files are read-only.
- Unit tests and `--smoke-test-only` must not create official checkpoints
  or the official 56-node DTW temporal graph.

## Inputs

- Prepared windows: `reimplementation/stgcn/prepared_data/r-only/pXX/{train,validation,test}.npz`
- Spatial graph: `.../adjacency_matrix/stgcn_undirected_topology.npy` (0/1 undirected connectivity)
- Temporal graph: produced later by `prepare_temporal_graph.py` from training-day full flow only
- R nodes: `analysis/graph/r_graph/r_nodes.csv`

Do not use `stgcn_weighted_adjacency.npy` or DCRNN directed W.

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
python3 reimplementation/stfgnn/train_r_only_stfgnn.py --help
python3 reimplementation/stfgnn/train_r_only_stfgnn.py --smoke-test-only
python3 reimplementation/stfgnn/train_r_only_stfgnn.py --temporal-adjacency <dtw.npy>
```

`--smoke-test-only` uses an in-memory `test_only_temporal_adjacency`.
Official training cannot default to that matrix.

Checkpoint monitor: raw validation MAE including zeros. Test is evaluated
once after loading the best checkpoint.
