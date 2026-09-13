# PyTorch reimplementation tree

Shared training infrastructure for spatiotemporal traffic models lives here.
Model-specific mathematics stay inside each model directory (`stgcn/`,
`dcrnn/`, `graph_wavenet/`, `stsgcn/`, `stfgnn/`, `pdformer/`, `stid/`,
`staeformer/`; later others).

Rules:

- Public tensors are `[B, T, V, C]`. Convert layouts inside the model, not in Dataset.
- Do not overwrite prepared NPZ files, adjacency matrices, or `analysis/` inputs.
- Default seed is `42`. IDs stay strings. Paths are relative to the repository root.
- Common code may be reused; do not flatten model differences into one Trainer class.
- STGCN is a faithful port of `reference/STGCN_IJCAI-18`, not a simplified third-party STGCN.

See `reimplementation/stgcn/AGENTS.md`, `reimplementation/stgcn/README.md`,
`reimplementation/dcrnn/AGENTS.md`, `reimplementation/graph_wavenet/AGENTS.md`,
`reimplementation/stsgcn/AGENTS.md`, `reimplementation/stfgnn/AGENTS.md`,
`reimplementation/pdformer/AGENTS.md`, `reimplementation/stid/AGENTS.md`,
and `reimplementation/staeformer/AGENTS.md`.
DCRNN, Graph WaveNet, STSGCN, STFGNN, PDFormer, STID, and STAEformer have
separate experiment directories; do not write one model's outputs into another.
STID and STAEformer do not use a road adjacency matrix.
