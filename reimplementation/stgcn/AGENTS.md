# R-only STGCN instructions

## Scope

This directory contains:

1. Data and weighted-adjacency preparation for the R-only STGCN experiment.
2. A PyTorch port of the original TensorFlow STGCN in `reference/STGCN_IJCAI-18`.

Follow the repository-root instructions as well. Shared PyTorch helpers live in
`reimplementation/common/`. STGCN-specific math stays in this directory.

Do not train from this `AGENTS.md` unless the user explicitly asks. The training
scripts are entry points for a later run.

## Inputs

Paths below are from the data disk unless noted (`data_root.py`).

- Daily full flow: `simulation_data/processed/subgraph_trajectories/day_*/edge_flow_5min.csv`
- Daily observed flow: `simulation_data/processed/subgraph_trajectories/day_*/observed_edge_flow_5min/edge_flow_5min_pXX.csv`
- R nodes: `training_data/r-only/node_mapping.csv` (`node_index` 0..55, `edge_id` as strings)
- R adjacency: `analysis/graph/r_graph/r_adjacency.npy` (rebuild with `build_r_graph.py` if missing)
- Prepared tensors: `training_data/r-only/pXX/{train,validation,test}.npz`
- Finished weighted adjacency: `training_data/r-only/adjacency_matrix/stgcn_weighted_adjacency.npy`

Do not open `trajectories.csv`, vehicle-ID files, or raw FCD except to
SHA256-protect them. Do not scale observed flow by `1/p`. Do not refit
`normalization.json`. Do not run `weight_matrix()` or `W/10000` on the finished W.

## Task definition

Each sample is a same-day 12-step window of observed R flow. The target is
the **full** flow at the last of those 12 slots (current-slot reconstruction),
not the next 5-minute slot.

```text
X = observed_flow[s:s+12]
Y = full_flow[s+11]
target_slot == window_start_slot + 11
```

Public tensors stay `[B, T, V, C]`. STGCN permutes to NCHW only inside `Conv2d`.

Default split is chronological by simulation day index parsed from `day_<NN>`:

- train: first 14 days
- validation: next 3 days
- test: last 3 days

Splits are whole days. Do not shuffle windows across days. Z-score mean/std
come only from training tensors: one input scaler per penetration, one shared
target scaler for full flow. Inverse transform predictions with
`mean_y_full` / `std_y_full` only.

## Outputs

Default prepared-data root: `training_data/r-only/` on the data disk.
NPZ files are generated artifacts. Python, JSON configs, tests, README, and
this `AGENTS.md` are source.

Default future training root: `training_results/stgcn/` on the data disk.
Do not create official experiment outputs unless the user asks to train.

Training / eval / monitor commands are in `reimplementation/stgcn/README.md`.
Window NPZ and the weighted adjacency are dataset inputs. This package reads
`stgcn_weighted_adjacency.npy` as finished W and builds the scaled Laplacian
and Chebyshev kernel at runtime. It must not apply a second Gaussian kernel.

## PyTorch port rules

- Faithful layer mapping from `reference/STGCN_IJCAI-18/models/layers.py`.
- No extra attention, embeddings, adaptive graphs, or modernized STGCN variants.
- Graph kernel is a buffer, not a trainable parameter. Do not add self-loops to W.
- Training loss is original `tf.nn.l2_loss` (`sum(sq)/2`) on prediction error.
  Collected weight L2 is logged only unless `weight_decay_in_optimizer` is true.
- Best checkpoint uses validation MAE on the raw count scale. Test is not used
  for model selection.
- Resume must not cross penetration rates.
- Negative predictions are not clipped by default.

## Safety

- `--dry-run` only scans and reports; it must not write the dataset.
- `--overwrite` replaces prepared artifacts in the output root, not source
  flow CSVs or the R graph.
- Training `--overwrite` applies to the experiment directory, not prepared data.
- Do not move, rename, or rewrite the daily flow files to match a template.
