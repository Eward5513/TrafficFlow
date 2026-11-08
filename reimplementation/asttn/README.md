# ASTTN on Xuancheng26

PyTorch port of the default `Model_Adp` branch in
[yokifly/ASTTN_pytorch](https://github.com/yokifly/ASTTN_pytorch)
(`reference/ASTTN_pytorch`). The paper is Adaptive Graph Spatial-Temporal
Transformer Network for Traffic Forecasting, CIKM 2022. That local snapshot
does not include a LICENSE file. File hashes are in `provenance.py`.

Training does not import `reference/` and does not read its PeMS files.

## Task

Input `x` is `[B, 12, 1406, C_in]`. The label is the full state at the last
input slot, `y` `[B, 1, 1406, C_out]`. `C_in` and `C_out` are 1 or 4 for the
nine `--input-type` / `--target-type` tasks. Channel order is inflow, outflow,
accumulation_start, accumulation_end.

Train on sampling seed 42. Each epoch scores validation seeds 42–46 separately
and keeps the checkpoint with the lowest unweighted mean raw MAE (latest epoch
on a tie). All-to-all uses the four-state elementwise MAE, the same field the
other baselines use. Test is scored the same way after that checkpoint is
restored.

## Reference settings that stay

`L=1`, `K=8`, `d=8`, encoder window `6`, adaptive node vectors `20`, at most
`40` neighbors, Adam `lr=0.001`, StepLR step `20` gamma `0.9`, batch `32`,
`100` epochs, no weight decay and no early stop. The training loss is
`masked_mae` with `null_val=NaN`, so zero counts are kept. Each time offset
still has its own incoming-edge softmax, and those outputs are averaged.
Adaptive-edge weights stay unused, so the node vectors do not receive gradient.

## Necessary differences

- `num_pred` is 1. The decoder is built with `T=1` and `window=1`. The encoder
  keeps `T=12`, `window=6`, and the original shift wrap. That wrap uses
  `arange(-(window-1)//2, (window-1)//2+1)`, which is the six offsets
  `-3..2` because Python floor-divides the unary minus.
- The input convolution accepts `C_in` and the output convolution emits
  `C_out`. There is no parameter-free squeeze.
- Time input is `[B, 13, 2]`: twelve history slots plus a copy of the last
  slot. Day-of-week is the stored Monday=0 weekday. Time-of-day is the real
  slot in `0..287`.
- DGL is not installed. Message passing uses the same source-to-destination
  edges, destination softmax, and zero vector for a node with no incoming edge.
  `num_nodes=1406` is explicit, so an isolated node is not dropped. Edge
  messages are computed in batch chunks of 4 and recomputed on the backward
  pass with the saved RNG state. The training batch stays 32, and batch norm
  still uses that full batch.
- Node2vec is a frozen 128-d table. The reference does not give a Xuancheng
  header, so 128 is an explicit adaptation. Walks use the static directed R
  graph and seed 42. Skip-gram uses two epochs, a summed pair loss, and
  gradient-norm clip 5 so a large batch does not explode. Generate it with:
  `python3 dataset/xuancheng26/model_prep/generate_asttn_node2vec.py`.
  The asset is `training_data_3s/graph/asttn/node2vec_r128.npy`.
- Inputs are already normalized. The loss applies the full-state target scaler
  once. Evaluation uses that same scaler through the shared metric path.

## Run

```bash
CUDA_VISIBLE_DEVICES=<gpu> python3 reimplementation/asttn/train_r_only_asttn.py \
  --input-type inflow --target-type inflow --rates 5

CUDA_VISIBLE_DEVICES=<gpu> python3 reimplementation/asttn/train_r_only_asttn.py \
  --input-type all --target-type all --rates 5

CUDA_VISIBLE_DEVICES=<gpu> python3 reimplementation/asttn/train_r_only_asttn.py \
  --input-type all --target-type inflow --rates 5

CUDA_VISIBLE_DEVICES=<gpu> python3 reimplementation/asttn/train_r_only_asttn.py \
  --input-type inflow --target-type inflow \
  --rates 5 10 20 30 40 50 70
```

`--rates 5` is `p05`. Results go to
`training_results_3s/asttn/in_<input>__out_<target>/<rate>/seed_<model_seed>/`.
Resume with `--resume <run>/last_checkpoint.pt`. The checkpoint stores the
model `state_dict`, Adam, StepLR, epoch, and RNG state.
