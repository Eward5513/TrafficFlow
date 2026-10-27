# R-only DCRNN instructions

## Scope

This directory contains:

1. Directed R-only graph preparation for original DCRNN.
2. A PyTorch port of the original TensorFlow DCRNN in `reference/dcrnn`.

Follow the repository-root instructions as well. Shared helpers live in
`reimplementation/common/`. DCRNN-specific math stays in this directory.

Do not train from this `AGENTS.md` unless the user explicitly asks. The
training script is the entry point for a requested run.

## Original code

The authoritative original is `reference/dcrnn` (Yaguang Li et al., ICLR 2018,
TensorFlow 1.x). Do not substitute `reference/STID`, `reference/Graph-WaveNet`,
or the third-party PyTorch DCRNN mentioned in that README. Do not modify
files under `reference/dcrnn/`.

Graph load path: `lib/utils.py` `load_graph_data()`. Official METR-LA /
PEMS-BAY yaml files use `filter_type: dual_random_walk`. Supports are built
at runtime from the finished directed `adj_mx`.

## Inputs

Paths below are from the repository root.

- Prepared windows: `training_data/r-only/pXX/{train,validation,test}.npz`
- Directed weighted adjacency: `training_data/r-only/dcrnn/dcrnn_weighted_adjacency.npy`
- Original pickle: `training_data/r-only/dcrnn/dcrnn_adj_mx.pkl`
- R nodes: `training_data/r-only/node_mapping.csv`

Do not open trajectories, vehicle IDs, or raw FCD except to SHA256-protect
them. Do not scale observed flow by `1/p`. Do not refit `normalization.json`.
Do not symmetrize or re-kernel the DCRNN adjacency. Do not use the STGCN
symmetric W.

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
python3 reimplementation/dcrnn/train_r_only_dcrnn.py --help
python3 reimplementation/dcrnn/train_r_only_dcrnn.py --overwrite
```

- Output is `/data5/tzhang174/TrafficFlow/training_results/dcrnn/`.
Checkpoint monitor: raw validation MAE including zeros. Test is evaluated
once after loading the best checkpoint. Each official penetration trains
an independent model from seed 42. A run may pass one rate or several.

## Safety

- `--overwrite` replaces only the DCRNN experiment directory.
- Do not overwrite prepared NPZ files, DCRNN/STGCN graphs, or `reference/dcrnn`.
- Resume must not cross penetration rates.
