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
  Nothing under `CODE_ROOT`.

## Task contract

- 9 tasks (`config.TASKS`) x 7 rates. Channel order is
  `inflow, outflow, accumulation_start, accumulation_end`.
- `x` = partial state of one rate, `y` = full state at `window_start_slot + 11`.
- Scalers come from `state_normalization.json` (training split only).
- Default loss/selection is unmasked raw-scale MAE on validation (same as STID,
  Graph WaveNet, DCRNN, Seq2seq GCN-LSTM). Test is never used for selection.

## Rules

- `MSTT/model/` must not read files or know about tasks; pass the graph in.
- New spatial modules register in `model/attention.py::SPATIAL_ATTENTION`.
- Every new mechanism gets a config switch.
- Changing defaults changes the experiment; record it in `README.md`.
