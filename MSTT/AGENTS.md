# MSTT instructions

MSTT is this project's backbone, not a port of a published model. It reuses
`reimplementation/common/` for windows, metrics, and the five-seed evaluation
protocol. Model math stays in `MSTT/model/`.

Finished experiments that are no longer a training path are recorded in
`MSTT/TRAINING_RESULTS.md`. Do not put their modules, configs, or launchers
back in order to reload an old checkpoint. A saved config that explicitly
selects a removed structure must fail. Inert leftover fields on a live
checkpoint (`history_fusion_mode=none`, `temporal_aux_enabled=false`, joint
attention defaults while `attention_layout` stays `factorized_ts`,
`backbone_mode=shared`, `state_context_sharing=independent`,
`state_node_sharing=independent`, and `fusion_mlp_hidden_dim` on a flatten
head) are dropped at read time. Live weights still load with `strict=True`.

## Current paths

Three configs share one TSF. Do not change the optimizer, loss weights,
default model, or data definition while editing them.

| Path | Config | Structure |
| --- | --- | --- |
| Default | `configs/default.json` | joint value linear, one shared TSF, flatten head |
| Four-state control | `configs/experiment_A_separate_value_mlp_projected.json` | four value MLPs, one shared TSF, flatten head, conservation projection |
| Speed and count gate | `configs/all_sp_c_state_aux_gate_d128_projected_all.json` | four value MLPs, 32-d gate from speed and observation count, same TSF, flatten head, projection |

The same gate config is reused across frequencies and penetrations by passing
that dataset id and its observation key. `p5_p15` does not reuse the
`p10_p20` scaler, vehicle lists, or checkpoints.
`run_all_sp_c_state_aux_gate.sh` launches the four `p10_p20` datasets.

```text
Input [B, 12, N, C]
  default and four-state control: value 32 + road 16 + TOD 8 + DOW 8
  gate: the same 32-d value after the product with g
  concat base embedding 64
  concat adaptive embedding 64
  temporal attention, residual + RMSNorm
  global spatial attention, residual + RMSNorm
  SwiGLU FFN, residual + RMSNorm
  flatten head
  output [B, N, C_out]
```

`num_layers` is 1. `attention_layout` is `factorized_ts`: `attention_order`
is `TS` and `ffn_placement` is `after_spatial`. QK-Norm, temporal RoPE, and
pre-RMSNorm stay on. Sparse spatial attention remains registered in
`model/attention.py::SPATIAL_ATTENTION` and is not the default.
`ffn_hidden_dim=176` is copied from the frozen run. Do not recompute it
from `d_model`.

The default value embedding is one `Linear(C_in, 32)`. The four-state
control and the gate use one `Linear(1, 8) → SiLU → Linear(8, 8)` per state,
concatenated to 32. The four-state control's experiment id is
`separate_value_mlp_projected`. The gate reads `all_sp_c`: the four states plus uncapped
`average_speed_raw` (km/h) and `speed_observation_count`. Those two channels
are not value features. They produce

```text
g = 2 * sigmoid(Linear(32, 32)(SiLU(Linear(2, 32)(aux))))
```

The last linear layer is zero-initialized, so `g = 1` at the start and the
product is applied before the road, TOD, and DOW concat. Conservation still
corrects the four outputs and does not treat `g` as a physical quantity.
Experiment id `state_aux_gate_d128_projected_all`. Do not resume it from a
four-channel checkpoint.

`conservation_mode` is `none`, `soft`, or `projected`. Soft and projected
are All2All or AllSpC2All losses, not extra backbone branches. Projected
loss is four-state final MAE plus `projection_raw_loss_weight` times raw MAE
plus `projection_consistency_weight` times the pre-projection conservation
MAE. The live projected configs use raw weight 0.5 and consistency weight 0.
`projection_supervision` is `all`.

## Adding a module

Keep Embedding, Adaptive Embedding, Attention, FFN, Prediction Head, and
Conservation in their own files. A new spatial attention registers in
`SPATIAL_ATTENTION` and is selected with `spatial_attention`. Do not fold a
new traffic input into `mstt.py`; add it beside `InputEmbedding`.

## Retired experiments

Four independent TSFs, the shared-TSF MLP head, history fusion, temporal
auxiliary losses, and the other finished branches are not training paths.
Their datasets, frequencies, metrics, and run ids are in `TRAINING_RESULTS.md`.
The road-width-4 full-context independent TSF stopped on 2026-11-16 at epochs
63, 40, 38, and 28 with no test metrics and no queued continuation. That stop
is its final record.

## Data and outputs

- Train windows: `training_data_3s/datasets/fixed_uniform/sampling_seeds/seed_42/flow/window_data/{rate}/{split}/{date}/{state}.npz`.
  Pass `--dataset-id` to read another dataset under `training_data_3s/datasets/<dataset_id>/`.
  `p10_p20` is that dataset's observation key, not a numeric penetration.
- Validation and test: the same layout for seeds 42, 43, 44, 45, and 46.
  The fixed scaler, split manifest, and node order live in `training_data_3s/shared/`
  and are linked from each fixed seed. A variable dataset keeps its own input scaler.
- Graph: `training_data_3s/graph/r_graph.npz`. `adjacency[u, v] = 1` means
  traffic moves from road `u` to road `v`.
- Results:
  `training_results_3s/datasets/<dataset_id>/physical_<version>/MSTT/in_<input>__out_<target>/<observation_key>/<experiment_id>/seed_<model_seed>/<run_id>/`.
  `seed_<model_seed>` is the initialization seed. `--experiment-id` defaults to
  `baseline`; a projected run with that default is stored as
  `conservation_projected`. Pass another id to keep a separate directory.
  The config records
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
All2All and AllSpC2All select by the five-seed mean validation inflow MAE.

## Rules

- `MSTT/model/` must not read files or know task names. Pass the graph in.
- Default loss for the frozen backbone is raw-scale MAE. Do not change another
  model's loss to match MSTT.
- Smoke runs (`--max-train-batches` / `--max-eval-batches`) keep the `_smoke` segment that `train.py` appends. For a projected run that segment sits under `MSTT/v0.2_concat_A2A_projected/_smoke/datasets/...`. The canonical builder would otherwise drop it and write the smoke into the official experiment directory.
- Nothing under `CODE_ROOT`.
