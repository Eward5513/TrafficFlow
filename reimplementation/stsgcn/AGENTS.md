# R-only STSGCN instructions

## Scope

This directory is a PyTorch port of the original STSGCN in
`reference/STSGCN` (Davidham3 / Song et al., AAAI 2020, MXNet). Shared
helpers live in `reimplementation/common/`.

Do not train from this `AGENTS.md` unless the user explicitly asks. The
training script is the entry point for a requested run. This migration
session implements code, unit tests, and a single training-batch smoke
test only.

## Original code

The authoritative original is `reference/STSGCN`. Do not substitute a
third-party PyTorch STSGCN. Do not modify files under `reference/STSGCN/`.
Do not install MXNet.

Official command (config directory is missing from this copy; the JSON
was taken from the upstream GitHub repo):

```bash
python3 main.py --config config/PEMS03/individual_GLU_mask_emb.json
```

Official module settings: `individual`, `GLU`, `use_mask=true`, spatial
and temporal embeddings on, `first_layer_embedding_size=64`, four STSGCL
blocks with filters `[64,64,64]`, Adam `lr=1e-3`, `batch_size=32`,
`epochs=200`, Huber `rho=1`.

## Isolation

- Output is `reimplementation/stsgcn/experiments/r_only/`.
- Never write into DCRNN, Graph WaveNet, or STGCN experiment directories.
- Never modify DCRNN or Graph WaveNet source while those runs are in progress.
- Prepared NPZ and adjacency files are read-only.
- Unit tests and `--smoke-test-only` must not create official checkpoints.

## Inputs

- Prepared windows: `reimplementation/stgcn/prepared_data/r-only/pXX/{train,validation,test}.npz`
- Spatial graph: `.../adjacency_matrix/stgcn_undirected_topology.npy` (0/1 undirected connectivity)
- R nodes: `analysis/graph/r_graph/r_nodes.csv`

Do not use `stgcn_weighted_adjacency.npy` or DCRNN directed W. Original
`get_adjacency_matrix(..., type_='connectivity')` stores binary edges.

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
python3 reimplementation/stsgcn/train_r_only_stsgcn.py --help
python3 reimplementation/stsgcn/train_r_only_stsgcn.py --smoke-test-only
python3 reimplementation/stsgcn/train_r_only_stsgcn.py
```

Checkpoint monitor: raw validation MAE including zeros. Test is evaluated
once after loading the best checkpoint.
