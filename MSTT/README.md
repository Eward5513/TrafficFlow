# MSTT

Single-layer temporal-spatial-FFN backbone with a 64-d base embedding
concatenated to a 64-d adaptive spatio-temporal embedding. The completed
I2I experiment `single_tsf_adaptive128_100ep` is the source of
`configs/default.json`. Its FFN hidden width is 176, taken from that run's
`resolved_config.json`. With 1,406 roads the I2I network has 1,305,561
parameters and All2All has 1,310,268.

## Layout

```text
MSTT/
├── config.py
├── train.py
├── evaluate.py
├── trainer.py
├── conservation_best_backbone.py
├── run_experiments.py
├── configs/default.json
├── configs/experiment_A_separate_value_mlp_projected.json
├── configs/experiment_shared_tsf_mlp_projected.json
├── configs/experiment_state_independent_tsf_mlp_projected.json
├── configs/experiment_state_independent_full_context_shared_tsf_mlp_projected.json
├── configs/experiment_state_independent_full_context_independent_tsf_mlp_projected.json
├── configs/experiment_state_independent_full_context_shared_node16_tsf_mlp_projected.json
├── configs/experiment_state_independent_full_context_independent_node16_tsf_mlp_projected.json
├── configs/conservation_best_backbone/
├── model/
│   ├── mstt.py
│   ├── state_branch.py
│   ├── state_context.py
│   ├── embedding.py
│   ├── auxiliary_gate.py
│   ├── adaptive_embedding.py
│   ├── attention.py
│   ├── joint_attention.py
│   ├── transformer_block.py
│   ├── ffn.py
│   ├── rope.py
│   ├── graph.py
│   ├── head.py
│   └── conservation.py
├── data/
└── utils/
```

`model/attention.py` keeps `SPATIAL_ATTENTION`. The default name is `global`.
`sparse` stays available for a later spatial module. New embeddings or heads
belong in their own modules and are wired from `MSTT`, not pasted into the block.

## Data

Training reads only

```text
DATA_ROOT/xuancheng26/training_data_3s/sampling_seeds/seed_42/flow/window_data/
```

Validation and test each load seeds 42, 43, 44, 45, and 46 from
`seed_<id>/flow/window_data/`, one seed at a time. Scaler, split manifest,
window manifest, validation summary, and node order are the shared files under
`training_data_3s/shared/`. The graph is `training_data_3s/graph/r_graph.npz`.

Each rate has 5540 train windows, 1108 validation windows, and 1385 test
windows. Channel order is inflow, outflow, accumulation_start,
accumulation_end. The label is the full state at the last of the 12 steps.

## Selection

```text
Train:       sampling_seed = 42
Validation:  seeds 42–46
Test:        seeds 42–46, report per seed, mean, and std (ddof=1)
```

The training loss stays `mae_raw`, the mean absolute error over every
predicted value. For a single output channel the checkpoint is that
validation MAE, averaged over seeds 42–46. When the input is all four
states and inflow is one of the outputs, the loss is unchanged but
`best_checkpoint.pt` is the epoch with the smallest validation inflow MAE
(again the mean over the five seeds). Test batches are not used to choose
the checkpoint.

## Commands

From the repository root, with the PyTorch environment:

```bash
python MSTT/train.py --task I2I --penetration p05
python MSTT/evaluate.py --run-dir <run directory> --splits validation test
python MSTT/conservation_best_backbone.py --mode soft --output-root <empty directory>
python MSTT/conservation_best_backbone.py --mode projected --output-root <empty directory>
```

A new run is written to

```text
training_results_3s/MSTT/in_<input>__out_<target>/<rate>/seed_<model_seed>/
```

`seed_<model_seed>` is the parameter seed, default 42. The resolved config
records `train_sampling_seed` and `evaluation_sampling_seeds`. If that
directory already holds a run from the earlier protocol, training stops
instead of replacing it. Pass an empty `--output-root` for another copy.

Soft and projected conservation are All2All only. They add a loss term or a
differentiable projection. They do not change the TSF block.

`configs/conservation_best_backbone/C_inflow_aux.json` keeps projected C's
MLP and conservation projection. Its loss is projected-inflow MAE plus
`0.5` times the pre-projection conservation MAE. The other three states
and the raw four-state MAE are not supervised. Checkpoints are still the
epoch with the smallest mean validation inflow MAE over seeds 42–46.
Results go to `training_results_3s/MSTT/C_inflow_aux/`.

```bash
python MSTT/train.py --config MSTT/configs/conservation_best_backbone/C_inflow_aux.json \
  --task All2All --penetration p05 --gpu 0
```

## Joint 3-step attention

`configs/joint_st3_projected.json` keeps the projected All2All backbone and
replaces T then S with one joint attention. For each of the 12 steps, the
N road queries attend once to every road in the centered 3-step window.
The first step uses times 0 and 1, the last uses times 10 and 11, and no
boundary token is repeated. RoPE uses the original time index. The FFN
width stays 176, the loss stays `final MAE + 0.5 * raw MAE`, and the
checkpoint is the mean validation inflow MAE over seeds 42–46.

```bash
python MSTT/train.py --config MSTT/configs/joint_st3_projected.json \
  --task All2All --penetration p05 --gpu 1
```

Results go to `training_results_3s/MSTT/joint_st3_projected/in_all__out_all/p05/seed_42/`.

## Experiment A: separate value MLPs

`configs/experiment_A_separate_value_mlp_projected.json` keeps the projected
All2All backbone: one T→S→FFN block, `d_model=128`, 4 heads,
`ffn_hidden_dim=176`, `attention_layout=factorized_ts`, and
`projection_raw_loss_weight=0.5` on all four states. Only the value
embedding changes. In `STATE_CHANNELS` order — inflow, outflow,
accumulation_start, accumulation_end — each channel has its own
`Linear(1, 8) → SiLU → Linear(8, 8)`. The four 8-d outputs are
concatenated to the original 32-d value vector. Road, TOD, DOW, embedding
dropout, the 64-d adaptive table, the TSF block, the flatten head, and the
conservation projection are unchanged.

Results go to
`training_results_3s/MSTT/experiment_A_separate_value_mlp_projected/in_all__out_all/<rate>/seed_<model_seed>/`.
The historical unconstrained All2All run stays at
`ablation_100ep/single_tsf_adaptive128_all2a_100ep`.

```bash
python MSTT/train.py --config MSTT/configs/experiment_A_separate_value_mlp_projected.json \
  --task All2All --penetration p05 --gpu <GPU>
```

## Six-channel input: `all_sp_c`

`all_sp_c` is an MSTT input, not a new target. The channel order is inflow,
outflow, accumulation_start, accumulation_end, average_speed_raw,
speed_observation_count. The first four arrays are copied from `all.npz`.
The last two are the uncapped snapshot-mean speed in km/h and the raw
snapshot count. Targets still come from `all.npz` or `inflow.npz`.

`value_embedding_mode=separate_mlp` builds one `Linear(1, 8) → SiLU →
Linear(8, 8)` per input channel and concatenates them. Four channels stay
32-d. Six channels are 48-d. With road 16, TOD 8, and DOW 8 the base
embedding is 80, the adaptive table stays 64, and `d_model` is 144. Head
dimension is 36. The FFN hidden width stays 176. There is no 48→32
projection.

Conservation still corrects the four output states. `AllSpC2All` may use
`projected` with supervision `all` or `inflow_only`. `AllSpC2I` stays a
single-output task without that projection. Checkpoint selection for both
new tasks is the mean validation inflow MAE over sampling seeds 42–46.
Speed and count are not loss targets.

The four `p10_p20` frequencies are launched by
`MSTT/run_all_sp_c_d144_variable.sh`. It uses the projected all-state config
above. GPU 5 runs 30 minutes, then 1 hour. GPU 7 runs 2 hours, then one day.
The two cards start together. A card does not start its second run until the
first run has exited.

```bash
bash MSTT/run_all_sp_c_d144_variable.sh
```

A run is written to

```text
training_results_3s/datasets/<dataset_id>/physical_independent_fcd_v1/MSTT/
  in_all_sp_c__out_all/p10_p20/separate_mlp_d144_projected_all/seed_42/<run_id>/
```

`in_all_sp_c__out_inflow` and the experiment ids
`separate_mlp_d144_projected_inflow_only` and `separate_mlp_d144_none` follow
the same layout. These runs do not resume a four-channel checkpoint.

## Auxiliary gate on the four state embeddings

`value_embedding_mode=state_aux_gate` reads the same six-channel `all_sp_c`
windows and the existing merged scaler. It does not add a label or a mask.
The four traffic states each keep an independent
`Linear(1, 8) → SiLU → Linear(8, 8)` and concatenate to 32. Standardized
`average_speed_raw` and `speed_observation_count` are not concatenated into
the backbone and do not enter the head or the conservation network. They
only produce

```text
g = 2 * sigmoid(Linear(32, 32)(SiLU(Linear(2, 32)(aux))))
```

`g` has shape `[B, 12, N, 32]`, one coefficient per state-embedding channel,
in `(0, 2)`. The two auxiliary channels jointly set all four groups of
coefficients. Parameters are shared across the batch, time, and roads; each
`(b, t, n)` is computed on its own. The last linear layer, weight and bias,
is zero-initialized, so the logits start at 0, `g` starts at 1, and the
gated embedding equals the four-state embedding. There is no second
normalization, no auxiliary residual, and no direct auxiliary path to the
output.

The gated 32-d value is concatenated with road 16, TOD 8, and DOW 8 (base
64), then the 64-d adaptive table, so `d_model` is 128. The single T→S→FFN
block, flatten head, and four-state conservation projection stay as they
are. The gate is not a physical count and is not a projection weight.

The config is `configs/all_sp_c_state_aux_gate_d128_projected_all.json`.
The experiment id is `state_aux_gate_d128_projected_all`. Training loss is
still four-state final MAE plus `0.5` times raw MAE. The checkpoint is still
chosen by the mean validation inflow MAE over seeds 42–46. This run starts
from scratch. `joint_linear`, the four-channel `separate_mlp`, and the
six-channel `separate_mlp` configs and result directories are unchanged.

```text
training_results_3s/datasets/<dataset_id>/physical_independent_fcd_v1/MSTT/
  in_all_sp_c__out_all/p10_p20/state_aux_gate_d128_projected_all/seed_42/<run_id>/
```

One frequency:

```bash
python MSTT/train.py --config MSTT/configs/all_sp_c_state_aux_gate_d128_projected_all.json \
  --task AllSpC2All --penetration p10_p20 \
  --dataset-id variable_30min_p10_p20 --gpu <GPU>
```

The other dataset ids are `variable_1h_p10_p20`, `variable_2h_p10_p20`, and
`variable_day_p10_p20`. `run_all_sp_c_state_aux_gate.sh` takes a GPU list
and keeps one dataset on each GPU at a time. Its PID file only stops a
second copy of that same chain. It does not see other jobs on the GPU, and
each launch overwrites `gpu<N>.log`. That shell script stays on `p10_p20`.

The same JSON config and experiment id also apply to `p5_p15`. Pass
`--dataset-id variable_<period>_p5_p15` and `--penetration p5_p15`. The
result directory then uses that dataset id and the `p5_p15` observation key,
so it does not overwrite a `p10_p20` run. Each frequency keeps its own
`all_sp_c_normalization.json`. Do not load a `p10_p20` checkpoint into it.

The four `p5_p15` frequencies finished on 2026-11-17. Each ran 100 epochs
with no early stop. The checkpoint is the epoch with the smallest mean
validation inflow MAE over seeds 42–46. The test metric is projected inflow
MAE on those same seeds. The standard deviation is the sample standard
deviation (`ddof=1`) across the five observation samples. The baseline is
`separate_value_mlp_projected` on the same frequencies.

| frequency | baseline | aux gate | delta |
| --- | ---: | ---: | ---: |
| 30 min | 2.527 ± 0.004 | 2.472 ± 0.010 | -0.055 |
| 1 h | 2.540 ± 0.014 | 2.508 ± 0.011 | -0.032 |
| 2 h | 2.494 ± 0.005 | 2.456 ± 0.002 | -0.038 |
| day | 2.505 ± 0.008 | 2.545 ± 0.010 | +0.040 |

The equal-weight mean of the four frequencies is 2.495 for the gate and
2.517 for the baseline, a difference of -0.021. Best epochs are 75, 74, 74,
and 53. Run ids are `20261117T144759_1e0955e2`,
`20261117T144759_4361df93`, `20261117T144759_12c97223`, and
`20261117T144759_c9a19f09`. Per-state metrics are in
[TRAINING_RESULTS.md](TRAINING_RESULTS.md).

```text
training_results_3s/datasets/<dataset_id>/physical_independent_fcd_v1/MSTT/
  in_all_sp_c__out_all/p5_p15/state_aux_gate_d128_projected_all/seed_42/<run_id>/
```

## No conservation projection

`separate_value_mlp_no_conservation` keeps the four-channel
`separate_mlp` backbone of `separate_value_mlp_projected` and sets
`conservation_mode=none`. The head output is the prediction. The run starts
from scratch on `all.npz` for the four `p5_p15` frequencies, uses 100 epochs
with no early stop, and selects the checkpoint by the mean validation inflow
MAE over seeds 42–46. At N=1406 the network has 1,310,460 parameters.
Experiment id `separate_value_mlp_no_conservation`.

The comparison below is test inflow MAE. The projected column is
`separate_value_mlp_projected` on the same frequencies. The conservation
column is the conservation MAE of the unprojected prediction.

| frequency | projected | no projection | delta | conservation MAE |
| --- | ---: | ---: | ---: | ---: |
| 30 min | 2.527 ± 0.004 | 2.564 ± 0.004 | +0.037 | 0.048 |
| 1 h | 2.540 ± 0.014 | 2.601 ± 0.011 | +0.061 | 0.050 |
| 2 h | 2.494 ± 0.005 | 2.534 ± 0.006 | +0.039 | 0.073 |
| day | 2.505 ± 0.008 | 2.499 ± 0.011 | -0.006 | 0.077 |

The equal-weight mean is 2.549 without projection and 2.517 after projection,
a difference of +0.033. Best epochs are 79, 78, 66, and 41. Run ids are
`20261117T162427_a874f7d7`, `20261117T162427_48a72854`,
`20261117T162427_bda49c12`, and `20261117T162427_fe1929a9`.

```text
training_results_3s/datasets/<dataset_id>/physical_independent_fcd_v1/MSTT/
  in_all__out_all/p5_p15/separate_value_mlp_no_conservation/seed_42/<run_id>/
```

## Independent state TSFs

`backbone_mode=state_independent` keeps the four-state `all` windows and
`separate_mlp`. Inflow, outflow, accumulation_start, and accumulation_end
each have their own value MLP, road/TOD/DOW tables, adaptive table, and one
T→S→FFN block. Nothing is mixed across states until those four 32-d
encodings are concatenated to 128. `head_type=flatten_mlp` then applies
RMSNorm, flattens all 12 steps, and maps `1536 → 128 → 4` with SiLU. The
existing conservation projection reads that 128-d hidden state.

Widths on each branch are value 8, road 4, TOD 2, DOW 2, adaptive 16, and a
32-d block with 4 heads (`head_dim=8`). `ffn_hidden_dim=176` is the width
inside each FFN. `d_model=128` is the total after concatenation. The same
fields can describe another total width when every total is divisible by 4
and the branch head size stays valid. The default config stays the shared
backbone with `head_type=flatten`.

Two configs separate the backbone change from the head change:

| Config | Backbone | Head | Parameters at N=1406 |
|---|---|---|---:|
| `experiment_A_separate_value_mlp_projected.json` | shared 128-d TSF | flatten linear | existing run |
| `experiment_shared_tsf_mlp_projected.json` | shared 128-d TSF | flatten MLP | 1,505,952 |
| `experiment_state_independent_tsf_mlp_projected.json` | four 32-d TSFs | flatten MLP | 1,407,648 |
| `experiment_state_independent_full_context_shared_tsf_mlp_projected.json` | four 92-d TSFs, shared TOD/DOW/adaptive, private road 4 | flatten MLP | 2,161,760 |
| `experiment_state_independent_full_context_independent_tsf_mlp_projected.json` | four 92-d TSFs, four context tables, private road 4 | flatten MLP | 5,408,264 |
| `experiment_state_independent_full_context_shared_node16_tsf_mlp_projected.json` | four 104-d TSFs, one shared road-16 context | flatten MLP | 2,299,424 |
| `experiment_state_independent_full_context_independent_node16_tsf_mlp_projected.json` | four 104-d TSFs, four road-16 context copies | flatten MLP | 5,613,416 |

A and B use the same fusion MLP and the same projected loss, final MAE plus
`0.5` times raw MAE. A has four width-32 blocks and 1,407,648 parameters.
B has one width-128 block and 1,505,952 parameters.
Checkpoint selection remains the mean validation inflow MAE over sampling
seeds 42–46. Those seeds resample the observed vehicles. The model seed
stays 42.

Results use the canonical layout, with `p10_p20` and `all.npz`:

```text
training_results_3s/datasets/<dataset_id>/physical_independent_fcd_v1/MSTT/
  in_all__out_all/p10_p20/<experiment_id>/seed_42/<run_id>/
```

`<dataset_id>` is `variable_30min_p10_p20`, `variable_1h_p10_p20`,
`variable_2h_p10_p20`, or `variable_day_p10_p20`. `<experiment_id>` is
`state_independent_tsf_mlp_projected` or `shared_tsf_mlp_projected`.
Run one task at a time, and only on a GPU whose compute-process list is empty.

```bash
python MSTT/train.py \
  --config MSTT/configs/experiment_state_independent_tsf_mlp_projected.json \
  --task All2All --penetration p10_p20 \
  --dataset-id variable_30min_p10_p20 --device cuda:0
```

The shared-head control uses
`MSTT/configs/experiment_shared_tsf_mlp_projected.json` and the same task,
penetration, and dataset id. In that file, `shared` means one TSF. The
full-context runs below keep four TSFs and share only the context tables.

The 128-d four-branch config stays value 8, road 4, TOD 2, DOW 2, and
adaptive 16. A full-context run has its own experiment id and checkpoint.

## Full context on four independent TSFs

`state_context_sharing` applies only when `backbone_mode=state_independent`.
It switches the TOD, DOW, and adaptive tables together. Value MLPs, road
tables, attention, FFN, and branch RMSNorm stay private either way. The
default is `independent`, which is what the 128-d run above already does.
`backbone_mode=shared` accepts only that unused placeholder.

Both new configs use the same widths. Each branch reads value 8, road 4,
TOD 8, DOW 8, then dropout, then adaptive 64, so the branch is 92-d.
`num_heads=4` and `attention_head_dim=24` give an attention inner width of
96, with QKV `92→288` and the output projection `96→92`. `ffn_hidden_dim=176`
is the width inside each FFN and is not divided by 4. The four 92-d states
concatenate to 368. The head is RMSNorm, then `4416→128→4`. The conservation
MLP reads the last hidden step plus the four raw outputs, `372→32→4`.

`shared` keeps one parent table, TOD `[288, 8]`, DOW `[7, 8]`, and adaptive
`[12, N, 64]`, and every branch reads that full vector. The four branches
add their gradients into those parameters. `independent` gives each branch
its own copy of the same shapes. At N=1406 one copy has 1,082,168
parameters, so the independent full-context model has 3,246,504 more
parameters than the shared one (5,408,264 versus 2,161,760). Between these
two models the only change is context sharing, and the parameter counts
still differ. A comparison with the 128-d four-branch model also changes
branch width, attention inner width, and head input width.

Loss, selection, and the training schedule are the same projected All2All
protocol: final four-state MAE plus 0.5 times raw MAE, checkpoint by the
mean validation inflow MAE over sampling seeds 42–46. The model seed stays
42.

```text
training_results_3s/datasets/<dataset_id>/physical_<version>/MSTT/
  in_all__out_all/<observation_key>/<experiment_id>/seed_42/<run_id>/
```

For `variable_30min_p10_p20` the observation key is `p10_p20` and the
physical version recorded on that dataset is `independent_fcd_v1`. The
experiment ids are `state_independent_full_context_shared_tsf_mlp_projected`
and `state_independent_full_context_independent_tsf_mlp_projected`. Run one
task at a time.

```bash
python MSTT/train.py \
  --config MSTT/configs/experiment_state_independent_full_context_shared_tsf_mlp_projected.json \
  --task All2All \
  --dataset-id variable_30min_p10_p20 \
  --penetration p10_p20 \
  --device cuda:0
```

```bash
python MSTT/train.py \
  --config MSTT/configs/experiment_state_independent_full_context_independent_tsf_mlp_projected.json \
  --task All2All \
  --dataset-id variable_30min_p10_p20 \
  --penetration p10_p20 \
  --device cuda:0
```

The same command with `--dataset-id variable_1h_p10_p20`,
`variable_2h_p10_p20`, or `variable_day_p10_p20` reads that frequency. The
5–15% manifests are `variable_30min_p5_p15`, `variable_1h_p5_p15`,
`variable_2h_p5_p15`, and `variable_day_p5_p15`, with observation key
`p5_p15`.

The two configs above are the historical road-width-4 full-context models.
Their parameter counts, 2,161,760 and 5,408,264, belong to that structure.
`state_node_sharing` is absent there and therefore stays `independent`: each
branch keeps a private `[N, 4]` road table.

## Full 16-d road table

`state_node_sharing=shared` is allowed only beside
`state_context_sharing=shared`. The new pair keeps four private value MLPs
and four private T→S→FFN blocks. Totals are still split by 4:

| Part | Per branch | Shared copies | Independent copies |
|---|---:|---:|---:|
| Value MLP | 8 | 4 | 4 |
| Road | 16 | 1 `[N, 16]` | 4 `[N, 16]` |
| TOD | 8 | 1 `[288, 8]` | 4 `[288, 8]` |
| DOW | 8 | 1 `[7, 8]` | 4 `[7, 8]` |
| Adaptive | 64 | 1 `[12, N, 64]` | 4 `[12, N, 64]` |
| TSF | 104 | 4 | 4 |

`node_embedding_dim=64` is the total road width, not one `[N, 64]` table.
Base width is 40 and the branch is 104. Attention stays at
`attention_head_dim=24` (inner width 96, QKV `104→288`, output `96→104`)
and `ffn_hidden_dim=176`. The concatenated hidden state is 416. The head is
RMSNorm then `4992→128→4`. The projection MLP reads `420→32→4`. At N=1406 the shared model has 2,299,424 parameters and the independent
model has 5,613,416. The difference is 3,313,992.

```bash
python MSTT/train.py \
  --config MSTT/configs/experiment_state_independent_full_context_shared_node16_tsf_mlp_projected.json \
  --task All2All \
  --dataset-id variable_30min_p5_p15 \
  --penetration p5_p15 \
  --device cuda:0
```

The independent node-16 config is
`MSTT/configs/experiment_state_independent_full_context_independent_node16_tsf_mlp_projected.json`
with experiment id
`state_independent_full_context_independent_node16_tsf_mlp_projected`.
Results use the same canonical directory with those experiment ids and
`p5_p15`. The four `p5_p15` frequencies of both node-16 configs have finished.
Do not resume a road-width-4 checkpoint into either run.

## 48-token state fusion

`state_token_transformer` starts from the shared Node16 D backbone above and
changes the prediction head. It is an implementation choice for one group of
runs. The four TSF blocks still emit 104 dimensions each and still use
`attention_head_dim=24`. The fusion layer is separate: each branch is mapped
by its own `Linear(104, 128)`, then one 48-token Transformer mixes the four
states and twelve history steps of a single road. Token order is state-major.
The last token of each state, indexes 11, 23, 35, and 47, has its own
`Linear(128, 1)`. Fusion attention uses 4 heads of 32 with QK-Norm and without
RoPE. The TSF RoPE is unchanged. Conservation still conditions on the 416-d
backbone hidden state, so the projection MLP stays `420→32→4`. At N=1406 this
configuration has 1,849,280 parameters.

```bash
python MSTT/train.py \
  --config MSTT/configs/experiment_state_independent_full_context_shared_node16_tsf_token_transformer_projected.json \
  --task All2All \
  --dataset-id variable_30min_p5_p15 \
  --penetration p5_p15 \
  --device cuda:0 \
  --experiment-id state_independent_full_context_shared_node16_tsf_token_transformer_projected
```

The same command with `variable_1h_p5_p15`, `variable_2h_p5_p15`, or
`variable_day_p5_p15` reads that frequency. All four use observation key
`p5_p15` and input `all.npz`, and all four have finished. Do not resume the
Node16 `flatten_mlp` checkpoint into this head.

The auxiliary-gate configuration is unchanged. Its finished runs use
`all_sp_c.npz` and `all_sp_c_state_aux_gate_d128_projected_all.json`.
The `p10_p20` runs cover `variable_30min_p10_p20`, `variable_1h_p10_p20`,
`variable_2h_p10_p20`, and `variable_day_p10_p20`. The `p5_p15` runs cover
`variable_30min_p5_p15`, `variable_1h_p5_p15`, `variable_2h_p5_p15`, and
`variable_day_p5_p15`. Task is `AllSpC2All`.

```bash
python MSTT/tests/test_state_context_sharing.py
python MSTT/tests/test_state_token_transformer.py
```

## Historical results

Completed ablations, All2I, All2All, and conservation runs now live under
`training_results_3s/datasets/legacy_unknown/physical_legacy_unknown/MSTT/`.
Finished `p10_p20` runs, including four-channel `separate_value_mlp_projected`,
six-channel `separate_mlp_d144_projected_all`, and
`state_aux_gate_d128_projected_all`, live under
`training_results_3s/datasets/variable_<period>_p10_p20/`. On `p5_p15`, the
finished runs under `training_results_3s/datasets/variable_<period>_p5_p15/`
are `separate_value_mlp_projected`, `separate_value_mlp_no_conservation`,
`shared_tsf_mlp_projected`,
`state_independent_tsf_mlp_projected`,
`state_independent_full_context_shared_tsf_mlp_projected`,
`state_independent_full_context_shared_node16_tsf_mlp_projected`,
`state_independent_full_context_independent_node16_tsf_mlp_projected`,
`state_independent_full_context_shared_node16_tsf_token_transformer_projected`,
and `state_aux_gate_d128_projected_all`.
The config and metrics for each completed run are in
[TRAINING_RESULTS.md](TRAINING_RESULTS.md). This tree no longer launches the
deleted ablation scripts.
