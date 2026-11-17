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

`conservation_mode` is `none`, `soft`, or `projected`. Soft and projected
are All2All task losses, not extra backbone branches. Launch them with
`conservation_best_backbone.py` and an empty output root.

`all_sp_c` is an input-only task pair: `AllSpC2All` and `AllSpC2I`. Its six
channels are the four states plus uncapped `average_speed_raw` (km/h) and
`speed_observation_count`. Windows live beside `all.npz` as `all_sp_c.npz`.
The two new scales are `datasets/<dataset_id>/all_sp_c_normalization.json`,
fit on sampling seed 42's raw 288 training bins. The four-state scaler is
unchanged. Separate MLPs follow the input width: 32-d for four channels and
48-d for six, with `d_model=144` in
`configs/all_sp_c_separate_mlp_d144_*.json`. Conservation stays on the four
outputs and is allowed for `AllSpC2All`. Selection for these inputs is the
five-seed mean validation inflow MAE. Do not resume a four-channel checkpoint
into these runs.

`backbone_mode=state_independent` is an All2All experiment on the four-state
`all.npz` windows. Four `StateTSFBranch` modules each encode one state
channel with their own value MLP, road table, and one T→S→FFN. Config
widths are totals and are split by 4. `ffn_hidden_dim` and
`attention_head_dim` are not split.

`state_context_sharing` controls the TOD, DOW, and adaptive tables.
`state_node_sharing` controls the road table. A missing `state_node_sharing`
stays `independent`, so older checkpoints keep one private road table per
branch. `shared` is legal only for `backbone_mode=state_independent` with
`state_context_sharing=shared`.

- `independent` (default), including
  `configs/experiment_state_independent_tsf_mlp_projected.json`: each branch
  owns its tables. The 128-d run is value 8, road 4, TOD 2, DOW 2, adaptive
  16, branch width 32, concatenated to 128, head `1536→128→4`.
- Historical full context, road width 4. These ids and parameter counts stay
  the record for that structure:
  `configs/experiment_state_independent_full_context_independent_tsf_mlp_projected.json`
  gives each branch TOD 8, DOW 8, adaptive 64, and a private road table of
  width 4. The branch is 92-d (`attention_head_dim=24`, inner width 96, FFN
  176). Four branches concatenate to 368. The head is `4416→128→4`. The
  projection MLP reads 372. At N=1406 this model has 5,408,264 parameters.
  `configs/experiment_state_independent_full_context_shared_tsf_mlp_projected.json`
  uses the same 4×92 branches, but one parent TOD `[288, 8]`, DOW `[7, 8]`,
  and adaptive `[12, N, 64]` is read in full by every branch. Each branch
  still owns a road table `[N, 4]`. At N=1406 the model has 2,161,760
  parameters, exactly 3,246,504 fewer than the four-copy context model.
- Node-16 full context. Value MLPs and the four T→S→FFN blocks stay private.
  Config totals are still divided by 4, so `node_embedding_dim=64` is 16 per
  branch. Each branch is value 8 + road 16 + TOD 8 + DOW 8 + adaptive 64 =
  104. `attention_head_dim` stays 24 and `ffn_hidden_dim` stays 176. Four
  branches concatenate to 416. The head is `4992→128→4`. The projection MLP
  reads 420.
  `configs/experiment_state_independent_full_context_shared_node16_tsf_mlp_projected.json`
  has one parent road table `[N, 16]`, TOD `[288, 8]`, DOW `[7, 8]`, and
  adaptive `[12, N, 64]`. Every branch reads each table in full, and dropout
  stays inside the branch.
  `configs/experiment_state_independent_full_context_independent_node16_tsf_mlp_projected.json`
  gives each branch its own copy of those four tables. At N=1406 the
  shared model has 2,299,424 parameters and the independent model has
  5,613,416, a gap of `3 × (1406×16 + 288×8 + 7×8 + 12×1406×64) = 3,313,992`.

`head_type=state_token_transformer` keeps that shared Node16 D backbone and
replaces only the prediction head. The four finished `p5_p15` runs are in
`TRAINING_RESULTS.md`. Four private value MLPs and four private TSF blocks stay in
place. TSF attention remains `attention_head_dim=24` and `ffn_hidden_dim=176`.
The concatenated backbone hidden state stays `[B, 12, N, 416]`. Each branch is
projected by its own `Linear(104, 128)`. A learned state id `[4, 128]` and a
learned time id `[12, 128]` are added, both initialized with truncated normal
standard deviation 0.02. The 48 tokens are state-major,
`token_index = state_index * 12 + time_index`, so the last step of each state
is token 11, 23, 35, or 47. One pre-norm Transformer layer attends inside
those 48 tokens: 4 heads, head dim 32, QK-Norm on, RoPE off, SwiGLU hidden
176, dropout 0.1. The TSF blocks keep their own RoPE. Four independent
`Linear(128, 1)` read the last token of each state. Conservation still calls
`projection(backbone_hidden, raw_pred)`; the condition is the 416-d backbone
step, and the MLP remains `420→32→4`. `fusion_chunk_size=1024` and
`fusion_checkpoint=true` only split that per-road attention for memory. At
N=1406 the model has 1,849,280 parameters. Config:
`configs/experiment_state_independent_full_context_shared_node16_tsf_token_transformer_projected.json`.
Experiment id
`state_independent_full_context_shared_node16_tsf_token_transformer_projected`.
Do not load it from a `flatten_mlp` checkpoint, and do not set backbone
`d_model` to 128. The finished runs read `all.npz` for `variable_30min_p5_p15`,
`variable_1h_p5_p15`, `variable_2h_p5_p15`, and `variable_day_p5_p15`.

`configs/experiment_shared_tsf_mlp_projected.json` is the one-TSF control.
`backbone_mode=shared` accepts only `state_context_sharing=independent` and
`state_node_sharing=independent` as unused placeholders. The frozen default
stays `backbone_mode=shared` and `head_type=flatten`. Do not resume one of
these structures into another.
`state_aux_gate` cannot be combined with `state_independent`.
`python MSTT/tests/test_state_context_sharing.py` checks the historical
full-context models and the node-16 pair on CPU.
`python MSTT/tests/test_state_token_transformer.py` checks the 48-token head
and the auxiliary-gate initialization on CPU.

`value_embedding_mode=state_aux_gate` is a separate `AllSpC2All` experiment
on the same `all_sp_c.npz` windows and merged scaler. Four independent state
MLPs stay 8-d and concatenate to 32-d. The standardized speed and
observation-count channels are not value features. They only produce
`g = 2 * sigmoid(Linear(32, 32)(SiLU(Linear(2, 32)(aux))))`. The last linear
layer is zero-initialized, so `g = 1` at the start and the gated embedding
equals the four-state embedding. The product is applied before the road,
TOD, and DOW concat. The backbone is again 128-d: value 32 + road 16 + TOD 8
+ DOW 8, then adaptive 64. Conservation still corrects the four outputs and
does not treat `g` as a physical quantity. Config:
`configs/all_sp_c_state_aux_gate_d128_projected_all.json`. Experiment id
`state_aux_gate_d128_projected_all`. Do not write it into a `separate_mlp`
directory or resume an older checkpoint. The same config is used for another
published range by passing that dataset id and its own observation key.
`p5_p15` reads `datasets/variable_<period>_p5_p15/.../window_data/p5_p15/.../all_sp_c.npz`
and that dataset's `all_sp_c_normalization.json`. It does not reuse the
`p10_p20` scaler, vehicle lists, or checkpoints. `run_all_sp_c_state_aux_gate.sh`
still launches only the four `p10_p20` datasets.

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
- Smoke runs (`--max-train-batches` / `--max-eval-batches`) keep the `_smoke` segment that `train.py` appends. For a projected run that segment sits under `MSTT/v0.2_concat_A2A_projected/_smoke/datasets/...`. The canonical builder would otherwise drop it and write the smoke into the official experiment directory.
- Nothing under `CODE_ROOT`.
