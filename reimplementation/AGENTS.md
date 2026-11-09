# PyTorch reimplementation tree

Shared training infrastructure for spatiotemporal traffic models lives here.
Model-specific mathematics stay inside each model directory (`stgcn/`,
`dcrnn/`, `graph_wavenet/`, `stsgcn/`, `stfgnn/`, `pdformer/`, `stid/`,
`staeformer/`, `s2s_gcn_lstm/`, `tlast/`, `asttn/`, `stjgcn/`).

Rules:

- Public tensors are `[B, T, V, C]`. Convert layouts inside the model, not in Dataset.
- Do not overwrite prepared NPZ files, adjacency matrices, or `analysis/` inputs.
- This tree does not build datasets. The 3-second windows, scalers, and node
  order come from `data_root.sampling_window_root`. Training uses sampling
  seed 42. Validation and test call `sampling_evaluation` once per seed
  42–46 and average the raw metrics. Each model keeps its own loss,
  optimizer, and training loop.
- Xuancheng results go to
  `xuancheng26/training_results_3s/datasets/<dataset_id>/physical_<version>/<model>/in_<input>__out_<target>/<observation_key>/<experiment_id>/seed_<model_seed>/<run_id>/`.
  `model_seed` initializes parameters. `dataset_id`, `train_sampling_seed`, and
  `evaluation_sampling_seeds` are written into the resolved config.
  A new `run_id` does not overwrite an existing run. The non-3s
  `xuancheng26/training_data/` and `training_results/` trees have been removed.
  Never write artifacts under `CODE_ROOT`. See `data_root.py`.
- Default `model_seed` is `42`. IDs stay strings. Paths are relative to the
  repository root or the data-disk root resolved by `data_root.py`.
- Common code may be reused; do not flatten model differences into one Trainer class.
- STGCN is a faithful port of `reference/STGCN_IJCAI-18`, not a simplified third-party STGCN.

See `reimplementation/stgcn/AGENTS.md`, `reimplementation/stgcn/README.md`,
`reimplementation/dcrnn/AGENTS.md`, `reimplementation/graph_wavenet/AGENTS.md`,
`reimplementation/stsgcn/AGENTS.md`, `reimplementation/stfgnn/AGENTS.md`,
`reimplementation/pdformer/AGENTS.md`, `reimplementation/stid/AGENTS.md`,
`reimplementation/staeformer/AGENTS.md`,
`reimplementation/s2s_gcn_lstm/AGENTS.md`,
`reimplementation/tlast/AGENTS.md`,
`reimplementation/asttn/AGENTS.md`, and
`reimplementation/stjgcn/AGENTS.md`.
DCRNN, Graph WaveNet, STSGCN, STFGNN, PDFormer, STID, STAEformer,
Seq2seq GCN-LSTM, TLAST, ASTTN, and STJGCN have separate experiment directories; do not
write one model's outputs into another.
STID, STAEformer, and TLAST do not use a road adjacency matrix.
