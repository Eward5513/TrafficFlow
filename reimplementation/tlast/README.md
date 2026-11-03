# TLAST on Xuancheng26

PyTorch port of [TLAST](https://github.com/zhuoshu/TLAST) for the 3-second-phase
Xuancheng26 four-state reconstruction task. The reference tree under
`reference/TLAST/` is read-only.

The model keeps cross-time embedding, TOD/DOW embeddings, spatial ids,
time-lag embedding of those TOD/DOW vectors, a kernel-3 temporal convolution,
last-step proxy readout, and two separate proxy attention maps. It predicts
the last observed slot (`[B, 1, N, C_out]`), not the next 12 slots.

Hyperparameters follow `configurations/PEMS07_12_12.json` because that
published road network is the closest to 1406 nodes. See `AGENTS.md` for the
list of values that stay at the author's settings and the task adaptations.

`use_input_residual` defaults to false so the comparison with STID,
STAEformer, Graph WaveNet, and STFGNN uses the same direct full-state target.
Pass `--use-input-residual 1` to add the latest partial observation with the
target scaler, matching the original ``head * std + latest_raw`` update.

## Commands

One task:

```bash
python3 reimplementation/tlast/train_r_only_tlast.py \
  --input-type inflow --target-type inflow --rates 70 --device cuda:0
```

Single-to-single uses the same state for `--input-type` and `--target-type`.
All-to-all is `--input-type all --target-type all`. All-to-single is
`--input-type all` and one target state. Omit `--rates` to train every
penetration in the config. A finished rate is skipped unless `--overwrite`
is set.

Outputs:

`/data5/tzhang174/TrafficFlow/xuancheng26/training_results_3s/tlast/`
