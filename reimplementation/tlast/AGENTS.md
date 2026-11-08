# Xuancheng26 TLAST instructions

## Scope

This directory ports `reference/TLAST` (Zheng, Shao, and Zhang, IEEE T-ITS 2025,
https://github.com/zhuoshu/TLAST) onto the Xuancheng26 four-state
reconstruction task. Do not edit `reference/TLAST/`.

The authoritative weights are the released PEMS07 12-to-12 configuration:
`hid_dim=64`, `d_out=1024`, `d_ff=256`, `n_heads=1`, `M=8`, `num_layers=1`,
`tau=3`, dropout `0.1`, `norm_flag=none`, proxy attention, AdamW `lr=0.001`,
`weight_decay=0.0001`, batch `32`, 100 epochs, no early stopping, Huber
delta `2`, cosine warmup of 5 epochs. PEMS07 is the closest published road
network (883 nodes) below these 1406 edges. CA's `M=2` is for 8600 nodes
and is not used.

## Task

Data is the 3-second-phase Xuancheng26 package. Training reads sampling
seed 42; validation and test read seeds 42–46:

`.../training_data_3s/sampling_seeds/seed_<id>/flow/window_data`

```text
X = partial state[s:s+12]
Y = full state[s+11]
target_slot == window_start_slot + 11
```

Train / validation / test days are the manifest's 20 / 4 / 5 days
(5540 / 1108 / 1385 windows). Windows do not cross days. Channel order is
`inflow`, `outflow`, `accumulation_start`, `accumulation_end`.

One model class serves three modes via `--input-type` and `--target-type`:

| Mode | input | target | tasks per rate |
| --- | --- | --- | --- |
| S2S | one state | the same state | 4 |
| A2A | all | all | 1 |
| A2S | all | one state | 4 |

Public tensors stay `[B, T, N, C]`. The head returns `[B, 1, N, C_out]`.

`use_input_residual` defaults to false. Baselines predict the full state
directly, and the partial observation does not share the target scaler.
The encoder skip is unchanged. Setting the flag adds the latest partial
observation of the predicted channels after converting it with the target
scaler. All-to-single uses that target channel. The full label is never
added.

## Training

```bash
python3 reimplementation/tlast/train_r_only_tlast.py --input-type inflow --target-type inflow --rates 70
```

Results go to
`.../training_results_3s/tlast/in_<input>__out_<target>/pXX/seed_<model_seed>/`.
The training loss stays Huber. The checkpoint is the lowest mean validation
MAE across the five sampling seeds. Test is not used for selection.
