# R-only STAEformer instructions

## Scope

This directory is a PyTorch port of the original STAEformer in
`reference/STAEformer` (Liu et al., CIKM 2023,
https://github.com/XDZhelheim/STAEformer). Shared helpers live in
`reimplementation/common/`.

Do not train from this `AGENTS.md` unless the user explicitly asks. The
training script is the entry point for a requested run. This migration
session implements code, unit tests, and a single training-batch smoke
test only.

## Original code

The authoritative original is `reference/STAEformer` =
https://github.com/XDZhelheim/STAEformer. The official GitHub has no
LICENSE file. Do not substitute a third-party rewrite. Do not add graph
convolution. `torchinfo` and `matplotlib` are original extras and must
not be required to run the model.

Official command:

```bash
cd model && python train.py -d PEMS04 -g 0
```

Traffic target config is `STAEformer.yaml::PEMS04`: `input_dim=3`,
`spatial_embedding_dim=0`, `adaptive_embedding_dim=80`, `num_heads=4`,
`num_layers=3`, `use_mixed_proj` constructor default `True` (YAML does
not override it), attention `mask=False`, Huber loss, Adam `lr=0.001`,
`weight_decay=0.0005`, MultiStepLR `[15,30,50] γ=0.1`, batch `16`,
`max_epochs=300`, `early_stop=20`. Gradient clip is commented out.

## Isolation

- Output is `reimplementation/staeformer/experiments/r_only/`.
- Never write into DCRNN, Graph WaveNet, STSGCN, STFGNN, PDFormer, STID,
  or STGCN experiment / prepared-data directories.
- Prepared NPZ and R graphs are read-only. STAEformer does not load
  adjacency.
- Unit tests and `--smoke-test-only` must not create official checkpoints.
- Prefer CPU while other models may be using the machine.

## Task definition

```text
X = observed_flow[s:s+12]
Y = full_flow[s+11]
target_slot == window_start_slot + 11
out_steps = 1
```

This is last-observed-step reconstruction, not future forecasting.

Time-of-day is constructed for every history step. Day-of-week follows
the original PeMS npz script `(day_index % 7)` as an integer 0..6,
recorded as `original_pems_npz_sequential_index_mod_7`. That is not
calendar Monday. Calendar mapping is optional and must fail if missing.

## Training

```bash
python3 reimplementation/staeformer/tests/run_and_report.py
python3 reimplementation/staeformer/train_r_only_staeformer.py --smoke-test-only
python3 reimplementation/staeformer/train_r_only_staeformer.py
```

`--smoke-test-only` runs one p70 training batch on CPU and discards the
model. Do not pass graph flags; the CLI does not accept them.
