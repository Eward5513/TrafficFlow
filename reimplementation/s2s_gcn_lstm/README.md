# Seq2seq GCN-LSTM (Reimplemented, 12-to-1 Adaptation)

PyTorch reimplementation of Abdelraouf, Abdel-Aty, and Mahmoud, *Sequence-to-Sequence Recurrent Graph Convolutional Networks for Traffic Estimation and Prediction Using Connected Probe Vehicle Data*, IEEE Transactions on Intelligent Transportation Systems, 2023. DOI: [10.1109/TITS.2022.3168865](https://doi.org/10.1109/TITS.2022.3168865).

This is **not** an official implementation. No confirmed author code was available. Widths and training lengths that the paper left to KerasTuner are starting points, not recovered optima.

The experiment is the existing Xuancheng26 R-only task: 12 steps of penetrated inflow reconstruct the full inflow at the last of those steps.

```text
X = penetrated_inflow[s:s+12]     # [B, 12, 1406, 1]
Y = full_inflow[s+11]             # [B, 1, 1406, 1]
target_slot == window_start_slot + 11
```

## Input feature

Channel 0 of the prepared window NPZ `x` is the penetrated R-edge inflow count. `dataset_metadata.json` records `feature_count = 1` and `input_variable = penetrated R-edge inflow counts`. Probe speed is not in that tensor, so it is not added. The target is full R-edge inflow (`y` / `y_raw`). Scalers stay the training-only global z-scores in `normalization.json`.

## Graph

`A` is the undirected STGCN Gaussian kernel `stgcn_weighted_adjacency.npy`, built from the same centre distances. Node order matches `node_mapping.csv`, `r_nodes.csv`, and `r_graph.npz`. Eighteen very long edges are zero in that float32 kernel; every positive weight still lies on `stsgcn_undirected_topology.npy`. DCRNN's directed kernel is not used.

Propagation is the paper's symmetric normalization, with self-loops added only here:

```text
Ã = A + I
Â = D̃^{-1/2} Ã D̃^{-1/2}
H = ReLU(Â H W)
```

## Data flow

```text
X_t [B, N, 1]
  -> Encoder GCN, once per t = 1..12     Z_t [B, N, 64]
  -> node-wise Encoder LSTM              E = h_12 [B, N, 64]
  -> Decoder GCN, once                   Z_dec [B, N, 64]
  -> Decoder LSTM from zero state, once  h_1 [B, N, 64]
  -> Dropout -> linear FC                Ŷ [B, 1, N, 1]
```

Nodes share LSTM weights. Encoder and decoder parameters are distinct. The decoder does not feed predictions back and does not inherit the encoder LSTM state.

## Paper-faithful components

- Static undirected graph with metre edge lengths, and Kipf-Welling GCN: `ReLU(Â H W)` with `Â` from `A + I`.
- Separate GCN-LSTM encoder and GCN-LSTM decoder.
- Dropout immediately before the fully connected output layer.
- Training objective is MAE. Optimizer is Adam. `ReduceLROnPlateau` multiplies the learning rate by 0.5. Early stopping watches validation loss.
- No attention, adaptive adjacency, or directional diffusion.

## Task adaptations

- The paper uses 13 input steps and 13 outputs (current estimate plus 60 minutes). This task is 12 to 1, and the label is the last observed slot, not a future slot.
- The paper's inputs are probe volume and probe speed, and its outputs are total volume and average speed. This task uses penetrated inflow only and predicts full inflow.
- Windows, splits, node order, z-score scalers, and raw MAE / RMSE / nonzero MAPE / WAPE follow the other Xuancheng26 baselines. Zeros in `y_raw` are real empty counts and stay in the loss.
- Because `y` is a z-score, the FC layer is linear. A sigmoid would clip the normalized target. Metrics are computed after inverting with `mean_y_full` / `std_y_full`.

## Implementation assumptions

These are not statements the paper explicitly confirms:

- Fixed-context decoding: the decoder GCN always reads the encoder context `E`. There is no GO token, teacher forcing, or scheduled sampling.
- The decoder LSTM hidden and cell states start at zero. With one output step, the previous state is the zero vector, so the decoder's recurrent matrix `weight_hh` does not affect that step. The input-to-hidden weights, gates, and biases do. A longer decoder unroll would train `weight_hh`; the official task does not.
- GCN, LSTM, and FC weights are shared across decoder steps. The official task uses one step, so the loop runs once.
- Each LSTM is node-wise: one shared cell runs along time at every node, and only the GCN mixes nodes.
- One GCN layer and one LSTM layer on each side. Hidden size 64, dropout 0.2, batch size 32, learning rate 0.001, 200 epochs, and early-stopping patience 20 are unpublished starting points.
- `ReduceLROnPlateau` patience 5 and `min_lr` 1e-6 are also unpublished. The factor 0.5 is the published coefficient.
- GCN layers have no bias, matching `Â H W`. The FC layer keeps a bias. GCN and FC weights use Xavier initialization; LSTM weights keep the PyTorch default.
- The undirected distance on a two-way neighbour is the STGCN selection: the reverse metres only when they are strictly shorter, otherwise the forward metres.

## Hyperparameter search

Table II ranges are sampled uniformly at random. Results for one penetration go under that model's existing result directory, for example `xuancheng26/training_results/s2s_gcn_lstm/p70/hyperparameter_search/`. Test metrics are not used.

```bash
python3 reimplementation/s2s_gcn_lstm/search_hyperparameters.py --dry-run --device cpu --penetration p70
CUDA_VISIBLE_DEVICES=2 python3 reimplementation/s2s_gcn_lstm/search_hyperparameters.py --device cuda --penetration p70 --num-trials 30
```

## Run

Smoke test (one p70 training batch, CPU, no checkpoint):

```bash
python3 reimplementation/s2s_gcn_lstm/train_r_only_s2s_gcn_lstm.py --smoke-test-only
```

Unit checks:

```bash
python3 reimplementation/s2s_gcn_lstm/tests/test_s2s_gcn_lstm.py
```

Official training is separate and is not started by the smoke test. It writes to `/data5/tzhang174/TrafficFlow/xuancheng26/training_results/s2s_gcn_lstm/<rate>/seed_42/`. Set `CUDA_VISIBLE_DEVICES` so the run metadata can record the physical GPU.

```bash
python3 reimplementation/s2s_gcn_lstm/train_r_only_s2s_gcn_lstm.py --rates 5 10
python3 reimplementation/s2s_gcn_lstm/train_r_only_s2s_gcn_lstm.py
```

The entry config is `xuancheng26/training_data/config/s2s_gcn_lstm/r_only_s2s_gcn_lstm.json`. The same JSON is kept at `reimplementation/s2s_gcn_lstm/configs/r_only_s2s_gcn_lstm.json`.
