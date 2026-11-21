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

`value_embedding_mode=state_aux_gate_split_mlp` keeps that six-channel input,
the four state MLPs, the product with `g`, and the 128-d TSF. Speed and the
observation count each have their own `1→8→8` MLP. Those 16 features go through
one `Linear(16, 32) → SiLU → Linear(32, 32)`, and only the last linear layer
starts at zero, so `g = 1` at initialization. There is no temporal auxiliary
loss and no history fusion. At N=1406 the gate has 1,776 parameters and the
model has 1,316,624, 624 more than `state_aux_gate`. Config:
`configs/all_sp_c_state_aux_gate_split_mlp_d128_projected_all.json`. Experiment
id `state_aux_gate_split_mlp_d128_projected_all`. Do not resume it from
`state_aux_gate_d128_projected_all`.
`MSTT/run_mstt_state_aux_gate_split_mlp_p5_p15.py` queues the four `p5_p15`
frequencies and takes a free GPU only when the live baseline scheduler will
not launch. At its current cap, a card freed while two or more baseline CUDA
tasks are still alive stays available.

`temporal_aux_enabled` defaults to false and adds no parameters. When true, one
shared `Linear(d_model, 4)` reads the encoder hidden state after the existing
modules are constructed. Training adds `temporal_aux_weight` times the
raw-scale MAE of steps 0..10 (`temporal_aux_scope=history_except_last`). The
12th auxiliary step is not in that loss. `history_fusion_mode` defaults to
`none`, and in that mode the auxiliary head is not fused into the last-step
head. The training split also returns `y_seq_raw [12, N, 4]` from
`shared/sequence_targets_12step`. Validation, test, and checkpoint selection
stay on the last-step inflow MAE. A checkpoint that omits the switch loads as
disabled. Changing the weight or scope does not resume an older auxiliary run.
The p5_p15 config is
`configs/experiment_temporal_aux_history11_projected.json`, experiment id
`separate_value_mlp_temporal_aux11_l05_projected`, weight 0.5.

`configs/experiment_state_aux_gate_temporal_aux_history11_projected.json`
combines that auxiliary loss with `value_embedding_mode=state_aux_gate`.
The task is `AllSpC2All`. `history_fusion_mode` stays `none`, so the
auxiliary head does not enter the flatten head or the projection. At N=1406
the model has 1,316,516 parameters, 516 more than the speed-gate parent.
Experiment id `state_aux_gate_d128_temporal_aux11_l05_projected_all`.
Do not resume it from `state_aux_gate_d128_projected_all` or from
`separate_value_mlp_temporal_aux11_l05_projected`.
`MSTT/run_mstt_state_aux_gate_temporal_aux_p5_p15.py` launches the four
`p5_p15` frequencies and sets each process to 15 threads.

`history_fusion_mode=gated_cross_attention` is the follow-up on that auxiliary
run. It requires `temporal_aux_enabled`, 12 steps, four output states,
`backbone_mode=shared`, `head_type=flatten`, `conservation_mode=projected`,
and `d_model=128`. The module is `HistoryPredictionFusion` in
`model/history_fusion.py`, constructed after `temporal_aux_head`, so the same
seed keeps the shared initialization. One forward still calls `_encode` and
the auxiliary head once. `base` is the flatten head. `P [B, 12, N, 4]` is the
auxiliary head in the standardized target scale. The fusion reads
`h = H[:, -1]` and `P` steps 0..10, earliest first. Each history step is
concatenated with a learned position row from `[11, 8]`, initialized normal
with std 0.02. Q is `Linear(128, 64)(h)`. K and V are independent
`Linear(12, 64)` maps of that token. Four heads of width 16 attend over the
11 history steps of that road only (`softmax(QK^T / sqrt(16))`, dropout 0),
and `Linear(64, 128)` produces `C`. The gate is
`sigmoid(Linear(32, 128)(SiLU(Linear(256, 32)(cat(h, C)))))`, then
`h_fused = h + g * C`. The residual is
`Linear(32, 4)(SiLU(Linear(256, 32)(cat(h_fused, h))))`. Only that last
linear starts at zero, so the initial raw prediction equals `base`. The
existing projection then reads the corrected raw and the full hidden state.
The loss is unchanged: projected four-state MAE + 0.5 times corrected raw MAE
+ 0.5 times auxiliary MAE on steps 0..10. Step 11 of `P` stays out of the
auxiliary loss and out of the history keys. `predict_with_aux` still packs
`raw_pred` and `final_pred` as `[B, 1, N, 4]` and `temporal_aux_pred` as
`[B, 12, N, 4]`. Attention weights, gates, and deltas are not written into
that dict or the five-seed NPZ. A checkpoint that omits the mode loads as
`none` and still rebuilds with `strict=True`. At N=1406 the fusion adds
39,132 parameters and the model has 1,354,496. Config:
`configs/experiment_temporal_aux_history11_gated_cross_attention_projected.json`.
Experiment id
`separate_value_mlp_temporal_aux11_l05_gated_cross_attention_projected`.
The four `p5_p15` frequencies use this same structure and the parent
hyperparameters. Do not resume it from an auxiliary-only checkpoint.

`history_fusion_mode=dual_path_history_tsf` is a separate readout, not a
residual on the flatten head. It keeps the same shared encoder, auxiliary
head, and projected conservation. Path A is the existing flatten head.
Path B embeds auxiliary steps 0..10 with `Linear(4, 64)`, a learned
`[11, 64]` time table, and a learned `[N, 64]` road table, then one new
T→S→FFN block (`d_model=64`, 4 heads, FFN 176). Its own flatten head maps
`704 → 4`. Cross-attention reads `cat(h, A)` against that history block
output, 4 heads of width 16, dropout 0. A four-state gate mixes A and B.
Its last linear starts at zero, so the initial gate is 0.5 and the raw
prediction is the average of A and B. Conservation projects only that
mixture. The loss is final MAE + 0.25 path-A MAE + 0.25 path-B MAE + 0.5
auxiliary MAE on steps 0..10. `projection_raw_loss_weight` and
`projection_consistency_weight` are 0. The history block's SwiGLU is not
counted as a backbone FFN. At N=1406 the new module has 194,824 parameters
and the model has 1,510,188. Config:
`configs/experiment_dual_path_history_tsf_projected.json`. Experiment id
`dual_path_history_tsf_gate4_projected_v1`. Do not resume it from an
auxiliary-only or gated-fusion checkpoint.

`history_fusion_mode=two_stage_projected_tsf` is a second full-width TSF, not
a residual on the flatten head. The shared embedding (four value MLPs, road,
TOD, DOW, and adaptive) runs twice. The first TSF reads the partial
observation. `temporal_aux_head` reconstructs all 12 steps. A new projection
MLP, separate from the existing last-step projection, puts every step on the
conservation surface in the target z-score. The second TSF has its own
T→S→FFN parameters and reads only those 12 projected values. `head_type=last`
is the residual head: its linear layer starts at zero, so the initial raw
last step equals the projected first-stage last step. The existing projection
then reads that sum and the second TSF's last hidden state. The loss is the
projected last-step four-state MAE plus 0.5 times the 12-step projected MAE
(`temporal_aux_scope=all_steps`). `projection_raw_loss_weight`,
`projection_consistency_weight`, and both branch weights are 0. The second
block's SwiGLU is not counted as a backbone FFN. At N=1406 the model has
1,513,288 parameters. Config:
`configs/experiment_two_stage_projected_tsf.json`. Experiment id
`two_stage_projected_tsf_shared_embedding_v1`. Do not resume it from an
auxiliary-only, gated-fusion, or dual-path checkpoint.

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
