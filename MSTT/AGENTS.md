# MSTT instructions

MSTT is this project's backbone, not a port of a published model. It reuses
`reimplementation/common/` for windows, metrics, and the five-seed evaluation
protocol. Model math stays in `MSTT/model/`.

## Frozen backbone

The default is the completed single-layer TSF run
`ablation_100ep/single_tsf_adaptive128_100ep`. `configs/default.json` copies
that run's resolved settings, including `ffn_hidden_dim=176`. Do not recompute
the FFN width from `d_model`.

```text
Input [B, 12, N, C]
  value 32 + road 16 + TOD 8 + DOW 8
  concat base embedding 64
  concat adaptive embedding 64
  temporal attention, residual + RMSNorm
  global spatial attention, residual + RMSNorm
  SwiGLU FFN, residual + RMSNorm
  flatten + linear head
  output [B, N, C_out]
```

`num_layers` is 1. The default `attention_layout` is `factorized_ts`:
`attention_order` is `TS` and `ffn_placement` is `after_spatial`.
`validate_config` rejects any other factorized stack. QK-Norm, temporal
RoPE, and pre-RMSNorm stay on. Sparse spatial attention remains registered in
`model/attention.py::SPATIAL_ATTENTION` and is not the default.

`attention_layout=joint_st` is the authorized single-run alternative. It
removes the separate temporal and spatial sublayers. One centered 3-step
joint attention reads every road at `t-1`, `t`, and `t+1` (the two existing
steps at either end of the window; the missing step is not copied), applies
one softmax over that time-major key set, then the same SwiGLU, flatten
head, and conservation projection. QK-Norm and temporal RoPE stay on the real
time index `0..11`. The default config remains `factorized_ts`. The joint
experiment config is `configs/joint_st3_projected.json`.

## Adding a module

Keep Embedding, Adaptive Embedding, Attention, FFN, Prediction Head, and
Conservation in their own files. A new spatial attention registers in
`SPATIAL_ATTENTION` and is selected with `spatial_attention`. Do not fold a
new traffic input into `mstt.py`; add it beside `InputEmbedding`.

## Data and outputs

- Train windows: `training_data_3s/datasets/fixed_uniform/sampling_seeds/seed_42/flow/window_data/{rate}/{split}/{date}/{state}.npz`.
  Pass `--dataset-id` to read another dataset under `training_data_3s/datasets/<dataset_id>/`.
  `p10_p20` is that dataset's observation key, not a numeric penetration.
- Validation and test: the same layout for seeds 42, 43, 44, 45, and 46.
  The fixed scaler, split manifest, and node order live in `training_data_3s/shared/`
  and are linked from each fixed seed. A variable dataset keeps its own input scaler.
- Graph: `training_data_3s/graph/r_graph.npz`. `adjacency[u, v] = 1` means
  traffic moves from road `u` to road `v`.
- 12→12 labels, when needed, are `training_data_3s/shared/sequence_targets_12step/`.
  They are not the default 12→1 trainer.
- Results:
  `training_results_3s/datasets/<dataset_id>/physical_<version>/MSTT/in_<input>__out_<target>/<observation_key>/<experiment_id>/seed_<model_seed>/<run_id>/`.
  `seed_<model_seed>` is the initialization seed. The config records
  `train_sampling_seed=42`, `evaluation_sampling_seeds=[42,43,44,45,46]`, and
  `physical_definition_version`. A new run gets its own `run_id` and does not
  overwrite an existing one. Historical 3s runs that did not record a physical
  version are under `legacy_unknown`.

## Protocol

Train on sampling seed 42 only (5540 windows at every rate). Each epoch
scores validation seeds 42–46 separately on raw-scale MAE and selects
`best_checkpoint.pt` by the unweighted mean. Test is scored the same way
after that checkpoint is restored, with per-seed metrics, the mean, and the
sample standard deviation (`ddof=1`). Test never selects the checkpoint.

`conservation_mode` is `none`, `soft`, or `projected`. Soft and projected
are All2All task losses, not extra backbone branches. Launch them with
`conservation_best_backbone.py` and an empty output root.

`projection_supervision` is `all` or `inflow_only`. Missing or `all` keeps
original C: four-state final MAE + `projection_raw_loss_weight` * raw MAE +
`projection_consistency_weight` * pre-projection conservation MAE.
`inflow_only` (C_inflow_aux) changes only that loss. It supervises the
projected inflow channel and adds `projection_consistency_weight` times the
pre-projection conservation MAE, with `projection_raw_loss_weight` at 0.
The MLP and projection math stay in `model/conservation.py`. All2All still
selects `best_checkpoint.pt` by the mean validation inflow MAE over seeds
42–46. Config: `configs/conservation_best_backbone/C_inflow_aux.json`.
Results: `training_results_3s/MSTT/C_inflow_aux/`.

## Rules

- `MSTT/model/` must not read files or know task names. Pass the graph in.
- Default loss for the frozen backbone is raw-scale MAE. Do not change another
  model's loss to match MSTT.
- Smoke runs (`--max-train-batches` / `--max-eval-batches`) go under `MSTT/_smoke/`.
- Nothing under `CODE_ROOT`.
