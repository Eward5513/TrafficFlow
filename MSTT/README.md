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
├── configs/conservation_best_backbone/
├── model/
│   ├── mstt.py
│   ├── embedding.py
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

## Historical results

Completed ablations, All2I, All2All, and conservation runs now live under
`training_results_3s/datasets/legacy_unknown/physical_legacy_unknown/MSTT/`.
Finished `p10_p20` runs, including four-channel `separate_value_mlp_projected`
and six-channel `separate_mlp_d144_projected_all`, live under
`training_results_3s/datasets/variable_<period>_p10_p20/`. The config and
metrics for each completed run are in
[TRAINING_RESULTS.md](TRAINING_RESULTS.md). This tree no longer launches the
deleted ablation scripts.
