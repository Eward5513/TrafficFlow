# PyTorch reimplementation tree

Shared training infrastructure for spatiotemporal traffic models lives here.
Model-specific mathematics stay inside each model directory (`stgcn/`,
`dcrnn/`, `graph_wavenet/`, `stsgcn/`, `stfgnn/`, `pdformer/`, `stid/`,
`staeformer/`, `s2s_gcn_lstm/`).

Rules:

- Public tensors are `[B, T, V, C]`. Convert layouts inside the model, not in Dataset.
- Do not overwrite prepared NPZ files, adjacency matrices, or `analysis/` inputs.
- This tree does not build datasets. Xuancheng26 windows, scalers, node
  tables, and model configs live under
  `DATA_ROOT/xuancheng26/training_data/`. Xuancheng26 training
  writes `xuancheng26/training_results/<model>/` on the data disk, with per-rate
  `pXX/seed_42/` subdirectories. Never write those artifacts under
  `CODE_ROOT`. See `data_root.py`.
- Default seed is `42`. IDs stay strings. Paths are relative to the repository root
  or the data-disk root resolved by `data_root.py`.
- Common code may be reused; do not flatten model differences into one Trainer class.
- STGCN is a faithful port of `reference/STGCN_IJCAI-18`, not a simplified third-party STGCN.

See `reimplementation/stgcn/AGENTS.md`, `reimplementation/stgcn/README.md`,
`reimplementation/dcrnn/AGENTS.md`, `reimplementation/graph_wavenet/AGENTS.md`,
`reimplementation/stsgcn/AGENTS.md`, `reimplementation/stfgnn/AGENTS.md`,
`reimplementation/pdformer/AGENTS.md`, `reimplementation/stid/AGENTS.md`,
`reimplementation/staeformer/AGENTS.md`, and
`reimplementation/s2s_gcn_lstm/AGENTS.md`.
DCRNN, Graph WaveNet, STSGCN, STFGNN, PDFormer, STID, STAEformer, and
Seq2seq GCN-LSTM have separate experiment directories; do not write one
model's outputs into another.
STID and STAEformer do not use a road adjacency matrix.
