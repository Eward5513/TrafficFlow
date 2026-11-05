# MSTT instructions

MSTT is this project's own backbone, not a port of a published model. It
reuses the baseline data and evaluation code in `reimplementation/common/`
but never edits it, the baselines, or their results.

## Data and outputs

- Windows: `DATA_ROOT/xuancheng26/training_data_3s/window_data/{rate}/{split}/{date}/{state}.npz`,
  read through `AlignedStateWindowDataset`. Do not regenerate or refit.
- Graph: `training_data_3s/graph/r_graph.npz`, `adjacency[u, v] = 1` means
  traffic moves from road `u` to road `v`.
- Results: `DATA_ROOT/xuancheng26/training_results_3s/MSTT/<rate>/<task>/seed_<seed>/`.
  Smoke runs (`--max-train-batches` / `--max-eval-batches`) go to `MSTT/_smoke/`.
  Nothing under `CODE_ROOT`. The top-level `MSTT/<rate>/` tree is v0.1
  (`embedding_fusion=add`); every later experiment uses
  `--output-root .../MSTT/<experiment_id>` and must not write into v0.1 runs.

## Task contract

- 9 tasks (`config.TASKS`) x 7 rates. Channel order is
  `inflow, outflow, accumulation_start, accumulation_end`.
- `x` = partial state of one rate, `y` = full state at `window_start_slot + 11`.
- 12→12 reconstruction data (`y = full[s:s+12]`) is read only through
  `data/sequence_dataset.py` from `window_data/sequence_targets_12step/`.
  It reuses the 12→1 `x` and target scaler; never write per-rate copies of it.
- Scalers come from `state_normalization.json` (training split only).
- Default loss/selection is unmasked raw-scale MAE on validation (same as STID,
  Graph WaveNet, DCRNN, Seq2seq GCN-LSTM). Test is never used for selection.

## Rules

- `MSTT/model/` must not read files or know about tasks; pass the graph in.
- New spatial modules register in `model/attention.py::SPATIAL_ATTENTION`.
- Every new mechanism gets a config switch.
- Changing defaults changes the experiment; record it in `README.md`.

## All2All conservation experiments

- `conservation_mode=none|soft|projected`; default `none` preserves legacy
  model parameters/forward. B/C are restricted to All2All, 12→1, v0.2 concat,
  sparse R-Graph, no adaptive embedding, raw MAE without channel balancing.
- Use `configs/v0.2_concat_all2all.json` and separate result roots
  `v0.2_concat_A2A_{none,soft,projected}`. Never overwrite earlier experiments.
- The projection receives training target mean/std from Trainer as persistent
  buffers; model code never reads scaler files. All conservation arithmetic is
  differentiable torch in physical counts. No post-projection clamping.
- Select on final raw-scale validation MAE. `val_loss` in B/C remains only the
  final supervised MAE, not the composite training objective. The supplied
  three-mode config explicitly uses `val_mae` for all three modes.
- Checkpoint mode must match on resume and standalone evaluation. Missing mode
  in historical checkpoints means `none`. C prediction archives use `pred_raw`
  for final physical predictions and `raw_pred_physical` for pre-projection
  values; do not confuse the historical meaning of `raw` (physical scale) with
  the new meaning (before projection).
