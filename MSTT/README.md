# MSTT v0.1 — Modern Spatio-Temporal Transformer

Encoder-only Transformer backbone for Xuancheng26 R-level (road) traffic-state
reconstruction. It plugs into the same data, split, scaler, and metric code
as the 9-task baselines in `reimplementation/`, so results are directly
comparable with `training_results_3s/<baseline>/in_X__out_Y/`.

This is a baseline backbone for later research, not a claim of better accuracy.

## Task

```text
x = partial state of rate pXX, slots s .. s+11      [B, 12, 1406, C_in]
y = full state at slot s+11 (last input slot)        [B, 1406, C_out]
```

| Task | Input | Output | | Task | Input | Output |
|---|---|---|---|---|---|---|
| I2I | inflow | inflow | | All2I | all 4 | inflow |
| O2O | outflow | outflow | | All2O | all 4 | outflow |
| AS2AS | accumulation_start | accumulation_start | | All2AS | all 4 | accumulation_start |
| AE2AE | accumulation_end | accumulation_end | | All2AE | all 4 | accumulation_end |
| All2All | all 4 | all 4 | | | | |

Rates: `p05 p10 p20 p30 p40 p50 p70` (63 runs). Channel order is always
`inflow, outflow, accumulation_start, accumulation_end`.

Data: `DATA_ROOT/xuancheng26/training_data_3s/window_data/{rate}/{split}/{date}/{state}.npz`
(the four-state package; `training_data/` only has inflow). Split: 20 / 4 / 5
days = 5540 / 1108 / 1385 windows, windows never cross midnight. The loader
is the shared `AlignedStateWindowDataset`; `x` is already z-scored with the
rate's training input scaler, labels are inverted per channel with the
training target scaler from `state_normalization.json`. Nothing is refit.
Zeros are real counts: loss and metrics are unmasked; MAPE skips zero targets.

## Architecture

```text
x [B,T,N,C_in] ─ Linear ─┐
road id  ─ Embedding ────┤ sum -> Dropout -> H [B,T,N,D]
TOD slot ─ Embedding ────┤   (TOD = window_start_slot + t, 288 slots)
DOW      ─ Embedding ────┘   (DOW = day_index % 7, Monday = 0)

L x block:
  H += Drop(TemporalAttn(RMSNorm(H)))   per node over T steps; QK-Norm, RoPE, SDPA, no causal mask
  H += Drop(SpatialAttn(RMSNorm(H)))    per step over R-graph neighbours (+ self); sparse edge softmax
  H += Drop(SwiGLU(RMSNorm(H)))         W_out(SiLU(W_gate H) * W_value H)

RMSNorm -> flatten T*D -> Linear -> [B, N, C_out]
```

| Module | File | Switches |
|---|---|---|
| Input embedding | `model/embedding.py` | `use_node_embedding`, `use_tod`, `use_dow` |
| Temporal attention | `model/attention.py::TemporalSelfAttention` | `use_qk_norm`, `use_rope`, `use_gated_attention` |
| Spatial attention | `model/attention.py::SparseSpatialAttention` | `spatial_direction`, `use_edge_type_bias`, `spatial_chunk_size`, `spatial_checkpoint` |
| FFN | `model/ffn.py::SwiGLU` | `ffn_hidden_dim` (default 8/3·d rounded to 8 = 176) |
| Block | `model/transformer_block.py` | `dropout`, `norm_eps` |
| Head | `model/head.py` | `head_type` = `flatten` / `last` |
| Graph edges | `model/graph.py` | — |

**Spatial graph.** `r_graph.npz` has `adjacency[u, v] = 1` when vehicles leave
road `u` and enter road `v` (checked: `to_junction(u) == from_junction(v)` for
all 3877 links). An attention edge `(src, dst)` sends `k, v` from `src` to the
query at `dst`; softmax runs over the edges of each `dst`.
`spatial_direction`: `forward` (attend to upstream), `backward` (downstream),
`bidirectional` (default, union). Every road has a self-loop. Each edge keeps
a relation type (self / upstream / downstream / both) and, with
`use_edge_type_bias`, gets one learnable logit per head and type so the
union graph still distinguishes direction. Default graph: 8336 edges
(in-degree 2–10) versus 1406² = 1.98 M dense entries.

**Sparse kernel.** `edge_softmax_attention` gathers `q[dst]`, `k[src]`, computes
`E` scores per head, subtracts a per-destination max (`scatter_reduce amax`),
normalizes with `index_add`, and aggregates `alpha * v[src]`. Scores and sums
run in float32 even under AMP. The `B*T` axis is chunked and each chunk is
activation-checkpointed so per-edge tensors are recomputed in backward
instead of stored.

**Gated attention (off by default).** `sigmoid(W_g x + b_g)`, `W_g` [D, D],
computed from the normalized sublayer input at the query position,
multiplies the concatenated head outputs elementwise (per head, per channel)
before the output projection.

## Training defaults (`config.py`)

`d_model 64, heads 4, layers 2, dropout 0.1, QK-Norm on, RoPE on, gate off,
sparse bidirectional spatial attention, TOD/DOW/node embeddings on`.
AdamW lr 1e-3, weight decay 1e-4 (not on biases / norms / edge logits),
5-epoch linear warmup then cosine (shared `LibCityCosineLRScheduler`),
100 epochs, early stopping patience 20 on the selection metric, gradient
clipping 5.0, batch 16 (eval 32), fp32 (`--amp bf16|fp16` available), seed 42.

Loss: unmasked raw-scale MAE (`mae_raw`), the objective STID, Graph WaveNet,
DCRNN and Seq2seq GCN-LSTM train and select on. For All2All this pools the
four channels, so inflow/outflow (target std ≈ 17) weigh more than
accumulation (std ≈ 4). `--channel-balance target_std` divides each channel
by its std (z-space MAE) as an explicit ablation; `--selection-metric
val_scaled_mae` selects on `mean_c(MAE_c / std_c)`. Both are logged every epoch.

## Commands

Run from the repository root (or from `MSTT/` with `python train.py ...`).

```bash
python MSTT/train.py --task I2I --penetration p05
python MSTT/train.py --task All2All --penetration p30 --gpu 4
python MSTT/train.py --task All2AE --penetration p70 --amp bf16 --use-gated-attention 1

python MSTT/evaluate.py --task All2All --penetration p30          # from best_checkpoint.pt
python MSTT/run_experiments.py --all_tasks --all_penetrations --dry_run
python MSTT/run_experiments.py --all_tasks --all_penetrations --gpus 0 4 2 7
python MSTT/run_experiments.py --tasks All2All All2I --penetrations p05 p70 --gpus 5 -- --amp bf16
python MSTT/run_experiments.py --summarize
```

Every config key is a flag (`--d-model 96`, `--spatial-direction forward`,
`--early-stop-patience none`, ...); `--config file.json` overrides defaults.
`--device auto` (default) picks the GPU with the most free memory. A
completed run is skipped unless `--overwrite`; an interrupted run resumes
from `last_checkpoint.pt` (model, optimizer, scheduler, RNG and shuffle state).

## Outputs

`DATA_ROOT/xuancheng26/training_results_3s/MSTT/<rate>/<task>/seed_42/`

| File | Content |
|---|---|
| `best_checkpoint.pt`, `last_checkpoint.pt` | weights, optimizer, scheduler, RNG, history |
| `resolved_config.json` | full config including the resolved model config |
| `training_history.json` / `.csv` | per-epoch loss, LR, validation metrics, per-channel MAE, time, memory |
| `validation_metrics.json`, `test_metrics.json` | raw-scale metrics, per channel for All2All |
| `test_predictions.npz` | same keys as the baselines (`--save-test-predictions 0` to skip) |
| `run_metadata.json` | parameters, timing, GPU, peak memory, hashes, status |
| `training_log.jsonl` | structured epoch log |

`MSTT/overall_summary.csv` holds one row per (task, rate, seed) with
validation/test MAE, RMSE, MAPE_nonzero, WAPE, `scaled_mae`, per-channel
test metrics, time and memory.
