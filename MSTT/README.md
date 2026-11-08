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
├── configs/conservation_best_backbone/
├── model/
│   ├── mstt.py
│   ├── embedding.py
│   ├── adaptive_embedding.py
│   ├── attention.py
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

## Historical results

Completed ablations, All2I, All2All, and conservation runs remain under
`training_results_3s/MSTT/`, including `ablation_100ep/` and
`conservation_best_backbone/`. Those directories are evidence. This tree no
longer launches the deleted ablation scripts.
