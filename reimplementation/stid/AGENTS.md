# R-only STID instructions

## Scope

This directory is a PyTorch port of the original STID in
`reference/STID` (Shao et al., CIKM 2022, https://github.com/zezhishao/STID).
Shared helpers live in `reimplementation/common/`.

Do not train from this `AGENTS.md` unless the user explicitly asks. The
training script is the entry point for a requested run. This migration
session implements code, unit tests, and a single training-batch smoke
test only.

## Original code

The authoritative original is `reference/STID` =
https://github.com/zezhishao/STID. Apache License 2.0. Built on BasicTS /
EasyTorch. Do not install or copy those frameworks. Do not substitute a
third-party "simple STID". Do not add graph convolution to make STID look
like the other baselines.

Official command:

```bash
python experiments/train.py --cfg stid/PEMS04.py --gpus '0'
```

Traffic target config is PeMS04/08 / METR-LA style: `embed_dim=32`,
`num_layer=3`, `if_node/if_T_i_D/if_D_i_W=True`, `node_dim=temp_dim_tid=
temp_dim_diw=32`, `time_of_day_size=288`, `input_dim=3`, Adam `lr=0.002`,
`weight_decay=0.0001`, MultiStepLR `[1,50,80] γ=0.5`, clip `5.0`,
batch `64`, epochs `100`. MLP dropout is hardcoded `0.15` in `mlp.py`.

## Isolation

- Output is `reimplementation/stid/experiments/r_only/`.
- Never write into DCRNN, Graph WaveNet, STSGCN, STFGNN, PDFormer, or STGCN
  experiment directories.
- Prepared NPZ and R graphs are read-only. STID does not load adjacency.
- Unit tests and `--smoke-test-only` must not create official checkpoints.
- Prefer CPU while DCRNN may be using the machine.

## Task definition

```text
X = observed_flow[s:s+12]
Y = full_flow[s+11]
target_slot == window_start_slot + 11
output_len = 1
```

This is last-observed-step reconstruction, not future forecasting.

Time-of-day is the last history slot (`target_slot`). Day-of-week follows
the original PeMS script `(day_index % 7)`, recorded as
`original_sequential_index_mod_7`. That is not calendar Monday. Calendar
mapping is optional and must fail if missing.

## Training

```bash
python3 reimplementation/stid/train_r_only_stid.py --help
python3 reimplementation/stid/train_r_only_stid.py --smoke-test-only
python3 reimplementation/stid/train_r_only_stid.py
```

`--smoke-test-only` runs one p70 training batch on CPU and discards the
model. Do not pass graph flags; the CLI does not accept them.
