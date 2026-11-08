# STJGCN on Xuancheng26

PyTorch port of the PeMSD7 implementation in
[zhengchuanpan/STJGCN](https://github.com/zhengchuanpan/STJGCN)
(`reference/STJGCN/PeMSD7`). The paper is Spatio-Temporal Joint Graph
Convolutional Networks for Traffic Forecasting, IEEE TKDE 2024. That local
snapshot does not include a LICENSE file. File hashes are in `provenance.py`.

The TensorFlow training runtime is not reused. D3, D4, and D8 variants are
not mixed in. Training does not import `reference/` or its PeMS arrays.

## Task

Same 12-to-1 reconstruction as the other Xuancheng26 baselines. `P=12` and
`Q=1`. One prediction head emits `C_out`. Public tensors stay
`[B, 12, 1406, C_in]` and `[B, 1, 1406, C_out]`.

Train on sampling seed 42. Select by the mean raw MAE across seeds 42–46,
including zeros. All-to-all uses the four-state elementwise mean, not an
inflow-only rule. Test does not select the checkpoint.

The training loss is still the PeMSD7 objective on the raw scale:
`MAE + 0.5 * 100 * abs(error) / (label + 1)`, with zero labels masked and the
mask divided by the valid fraction. An all-zero batch returns a finite zero.
Reported MAE, RMSE, and `MAPE_nonzero` stay on the shared metric functions.

## Reference settings that stay

`K=2`, `d=64`, `T=288`, `delta_pdf=0.9`, `delta_adt=0.7`, `beta=0.5`, four
STJGC layers, shared interaction `B`, Adam `lr=0.001`, `200` epochs, no
weight decay and no added schedule. The PeMSD7 batch is 64. On a 24GB GPU a
batch of 64 for 1406 nodes did not finish one step (the run was already
holding about 16GB). The default batch is therefore 32, which completed a
step at about 8.9GB allocated. This is not equivalent to batch 64 for the
batch-norm statistics. `--batch-size 64` is still accepted when the GPU has
room.

Layer time positions are `[1,3,5,7,9,11]`, `[3,7,11]`, `[7,11]`, `[11]`, with
predefined graph indexes `[1,0]`, `[2,0]`, `[4,0]`, `[4,0]`. Residuals follow
those assignments. Logits below `delta_adt` are set to 0 before softmax, not
to negative infinity. Batch norm runs only when the TensorFlow conv has an
activation, with population variance, epsilon `1e-3`, and EMA decay `0.9`.

## Necessary differences

- `Q=1`, and the last prediction layer emits `C_out` instead of 1. The output
  is permuted to `[B, 1, N, C_out]` without `squeeze`.
- Time features are the 12 observed slots only. Day-of-week is the stored
  Monday=0 weekday, not a series that starts on Monday at interval 0.
- The predefined graph uses directed Xuancheng centre distances
  (`0.5 * source lane + via + 0.5 * target lane` on legal connections). Sigma
  is the population standard deviation of those finite distances. Non-edges
  are infinite. `A` and `A.T` are each normalized by `D^{-1/2} A D^{-1/2}`.
  Generate it with
  `python3 dataset/xuancheng26/model_prep/generate_stjgcn_predefined_graph.py`.
  The asset is `training_data_3s/graph/stjgcn/predefined_stjg.npz`.
- Adaptive attention is evaluated in node chunks with activation checkpointing.
  The chunk width does not change the products. Default chunk is 64 nodes.
- Inputs are already normalized. The loss applies the full-state target scaler
  once.

## Run

```bash
CUDA_VISIBLE_DEVICES=<gpu> python3 reimplementation/stjgcn/train_r_only_stjgcn.py \
  --input-type inflow --target-type inflow --rates 5

CUDA_VISIBLE_DEVICES=<gpu> python3 reimplementation/stjgcn/train_r_only_stjgcn.py \
  --input-type all --target-type all --rates 5

CUDA_VISIBLE_DEVICES=<gpu> python3 reimplementation/stjgcn/train_r_only_stjgcn.py \
  --input-type all --target-type inflow --rates 5

CUDA_VISIBLE_DEVICES=<gpu> python3 reimplementation/stjgcn/train_r_only_stjgcn.py \
  --input-type inflow --target-type inflow \
  --rates 5 10 20 30 40 50 70
```

`--rates 5` is `p05`. Results go to
`training_results_3s/stjgcn/in_<input>__out_<target>/<rate>/seed_<model_seed>/`.
Resume with `--resume <run>/last_checkpoint.pt`.
