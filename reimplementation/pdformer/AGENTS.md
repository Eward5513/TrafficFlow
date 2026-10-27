# R-only PDFormer instructions

## Scope

This directory is a PyTorch port of the original PDFormer in
`reference/PDFormer` (Jiang, Han, Zhao, Wang, AAAI 2023, LibCity). Shared
helpers live in `reimplementation/common/`.

Do not train from this `AGENTS.md` unless the user explicitly asks. The
training script is the entry point for a requested run. This migration
session implements code, unit tests, and a single training-batch smoke
test only.

## Original code

The authoritative original is `reference/PDFormer` =
https://github.com/aptx1231/PDFormer (also NickHan-cs). MIT License,
Copyright 2022 aptx1231. Do not substitute a third-party simplified
Transformer. Do not copy the full LibCity training framework. Do not
modify files under `reference/PDFormer/`.

Official command:

```bash
python run_model.py --task traffic_state_pred --model PDFormer --dataset PeMS04 --config_file PeMS04
```

Target overlay is PeMS04/08, not NYC grid: `type_ln=pre`, `far_mask_delta=7`,
`set_loss=huber`, `huber_delta=2`, `geo/sem/t heads = 4/2/2`, `embed_dim=64`,
`enc_depth=6`, `drop_path=0.3`, `s_attn_size=3`, `type_short_path=hop`,
`bidir=true`.

## Isolation

- Output is `/data5/tzhang174/TrafficFlow/training_results/pdformer/`.
- Relation preprocessing writes `/data5/tzhang174/TrafficFlow/training_data/pdformer/`.
- Never write into DCRNN, Graph WaveNet, STSGCN, STFGNN, or STGCN experiment directories.
- Prepared NPZ, R graphs, and STGCN adjacency files are read-only.
- Unit tests and `--smoke-test-only` must not create official checkpoints
  or official 56-node hop/DTW/Laplacian files.
- Default device is `cuda`. `--smoke-test-only` still forces CPU.

## Inputs

- Prepared windows: `training_data/r-only/pXX/{train,validation,test}.npz`
- Hop / Laplacian topology: `.../adjacency_matrix/stgcn_undirected_topology.npy`
- R nodes: `analysis/graph/r_graph/r_nodes.csv`

Do not use `stgcn_weighted_adjacency.npy` as a hop matrix.

## Task definition

```text
X = observed_flow[s:s+12]
Y = full_flow[s+11]
target_slot == window_start_slot + 11
output_window = 1
```

This is last-observed-step reconstruction, not future forecasting.

Time-of-day comes from 5-minute slots. Day-of-week is off by default because
simulation days have no calendar dates. Do not use `day_index % 7`.

## Training

```bash
python3 reimplementation/pdformer/train_r_only_pdformer.py --help
python3 reimplementation/pdformer/train_r_only_pdformer.py --smoke-test-only
PDFormer relation files (hop, DTW, Laplacian, pattern keys) are dataset inputs, not produced in this package.
python3 reimplementation/pdformer/train_r_only_pdformer.py --relations-dir training_data/pdformer
```

`--smoke-test-only` uses in-memory `test_only` relations. Official training
cannot default to those matrices.

Official `run_single_rate` trains the configured 200 epochs (the PeMS
`patience: 50` field is present in config but is not implemented; do not
add early stopping without an explicit protocol change). It selects
`best_checkpoint.pt` by validation raw MAE, then reloads that checkpoint
and writes `test_metrics.json` plus `test_predictions.npz` through
`reimplementation.common.evaluation`. Test is never used for checkpoint
selection.
