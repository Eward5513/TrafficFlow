# R-only Seq2seq GCN-LSTM instructions

## Scope

This directory is a PyTorch reconstruction of Abdelraouf, Abdel-Aty, and
Mahmoud, IEEE T-ITS 2023 (DOI 10.1109/TITS.2022.3168865), adapted to the
Xuancheng26 12-to-1 full-flow task. It is not an official implementation.
Shared windows, scalers, and the road graph stay read-only under
`reimplementation/common/` and `xuancheng26/training_data/`.

Do not train from this file unless the user explicitly asks. The training
script is the entry point.

## Task

```text
X = penetrated R-edge inflow[s:s+12]
Y = full R-edge inflow[s+11]
target_slot == window_start_slot + 11
output_steps = 1
input_channels = 1
```

This is last-observed-step reconstruction. Do not add probe speed. Do not
treat the target as the next slot.

## Graph

Read `undirected_center_distance.npy` as undirected metre weights. Its support
must match `stsgcn_undirected_topology.npy`, and its node order must match
`r_graph.npz` and the node tables. Do not load the Gaussian STGCN or DCRNN
kernels as `A`. Self-loops are added only inside `Â = D̃^{-1/2}(A+I)D̃^{-1/2}`.

## Architecture assumptions

Fixed-context decoder, zero decoder LSTM state, node-wise shared LSTM, and
one shared decoder GCN/LSTM/FC. Record these as assumptions. The label is
`Seq2seq GCN-LSTM (Reimplemented, 12-to-1 Adaptation)`.

## Isolation

- Output is `/data5/tzhang174/TrafficFlow/xuancheng26/training_results/s2s_gcn_lstm/`.
- Never write into another model's experiment directory or into prepared NPZ,
  adjacency, or `normalization.json`.
- `--smoke-test-only` uses CPU, reads p70 train plus the graph, and writes
  no checkpoint.

## Training

```bash
python3 reimplementation/s2s_gcn_lstm/tests/test_s2s_gcn_lstm.py
python3 reimplementation/s2s_gcn_lstm/train_r_only_s2s_gcn_lstm.py --smoke-test-only
python3 reimplementation/s2s_gcn_lstm/train_r_only_s2s_gcn_lstm.py
```

Loss is unmasked raw MAE, including zeros. The best checkpoint is the lowest
validation raw MAE. Test runs once after that reload. The FC output is linear
because targets are z-scores.
