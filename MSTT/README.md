# MSTT v0.4 — Modern Spatio-Temporal Transformer

## v0.2 All2All conservation modes (12→1)

`configs/v0.2_concat_all2all.json` pins the common experiment: concat
32/16/8/8, D=64, 4 heads, 2 layers, SwiGLU hidden=176, flatten prediction
head, sparse bidirectional R-Graph with edge-type bias, no adaptive embedding,
batch 16 / evaluation 32, fp32, seed 42, AdamW and the existing cosine schedule.
It uses existing `training_data_3s/window_data` and its training-only
`state_normalization.json`; no data, scaler, graph or target changes.
Input `[B,12,1406,4]` reconstructs `[B,1406,4]` at the last observed slot;
the dataset/trainer retains the existing singleton output-step dimension.
Channel order is always `inflow, outflow, accumulation_start, accumulation_end`.

| Mode | Training objective (physical vehicle counts) | Parameters |
| --- | --- | ---: |
| A / `none` | existing `RawScaleLoss` MAE | 161,820 |
| B / `soft` | final MAE + λ mean(abs(r)) | 161,820 |
| C / `projected` | final MAE + α raw MAE + β mean(abs(raw r)) | 164,160 |

Here `r = N_end - N_start - I + O`. Defaults are λ=0.1, α=0.5, β=0.0,
`projection_mlp_hidden_dim=32`. B changes only the objective and preserves the
original prediction. C retains the original head and concatenates the last
encoder node state with its normalized four-state prediction, then applies
`Linear(68,32) → SiLU → Linear(32,4) → Softmax`. The last linear layer starts
at zero, giving weights `(0.25,0.25,0.25,0.25)`. These are **correction allocation
weights**: larger weight means a larger correction, not greater reliability.

The projection is `y_final = y_raw - r * (a * w)`, `a=(-1,+1,-1,+1)`, performed
after per-channel denormalization in FP32 torch (also under AMP), followed by
renormalization for the Trainer interface. The target scaler is injected once
and saved as model buffers. No labels enter the projection. No clamping is
applied, so the final physical outputs conserve up to floating-point roundoff
but may be negative. Raw and final negative fractions and magnitudes are logged.

`model(x,tod,dow)` returns final normalized predictions for all modes.
`model.forward_with_aux(...)` reuses the same encoder/head and, in C, returns
`raw_pred` and `final_pred` (normalized), `correction_weights`, and
`raw_conservation_residual` (physical counts). At initialization the zero last
MLP layer makes the first MLP layer gradient zero on the first step; it receives
nonzero gradients after the last layer updates. This is expected initialization
behavior, not a detached graph.

The supplied config selects all three modes on final physical validation MAE.
B/C also interpret `selection_metric=val_loss` as final prediction MAE;
the composite training loss never selects checkpoints. Their config rejects
single-output tasks, channel balancing, other loss types, adaptive embedding,
global attention and sequence reconstruction. Historical `none` configurations
retain their loss and selection options, state dict and forward behavior.

Independent commands from the repository root (do not run together on the same
GPU unless memory/capacity has been checked):

```bash
python MSTT/train.py --config MSTT/configs/v0.2_concat_all2all.json \
  --task All2All --penetration p05 --gpu 3 --conservation-mode none \
  --output-root /data5/tzhang174/TrafficFlow/xuancheng26/training_results_3s/MSTT/v0.2_concat_A2A_none

python MSTT/train.py --config MSTT/configs/v0.2_concat_all2all.json \
  --task All2All --penetration p05 --gpu 3 --conservation-mode soft \
  --conservation-lambda 0.1 \
  --output-root /data5/tzhang174/TrafficFlow/xuancheng26/training_results_3s/MSTT/v0.2_concat_A2A_soft

python MSTT/train.py --config MSTT/configs/v0.2_concat_all2all.json \
  --task All2All --penetration p05 --gpu 3 --conservation-mode projected \
  --projection-raw-loss-weight 0.5 --projection-consistency-weight 0.0 \
  --projection-mlp-hidden-dim 32 \
  --output-root /data5/tzhang174/TrafficFlow/xuancheng26/training_results_3s/MSTT/v0.2_concat_A2A_projected
```

Each root contains `p05/All2All/seed_42/` with resolved config, mode-tagged best
and last checkpoints, epoch histories, metadata (parameter count, best epoch,
time and GPU peak memory), validation/test metrics and test predictions.
`train_supervised_mae`, `train_conservation_loss` (raw residual in C),
`train_raw_mae` (C), and `train_total_loss` distinguish loss components.
Metrics include each state's MAE/RMSE/WAPE, `conservation_mae/max_abs`, negative
fraction/mean magnitude/max magnitude, and per-channel negative diagnostics.
C additionally records `raw_metrics`, `raw_mae`, `raw_conservation_mae/max_abs`,
projection correction magnitude and per-channel weight mean/std/min/max/p05/p50/p95.
`pred_raw` / `y_pred_raw` in the NPZ retain the shared meaning **final physical
prediction**. C also stores `raw_pred_physical`, `raw_pred_normalized`, and
`correction_weights`, aligned by `sample_index`.

When using the default result root, explicit `--conservation-mode` on All2All
routes to the corresponding new root; omitting the switch preserves legacy
Mode A locations. B/C reject legacy version directories and cross-mode roots,
even with `--overwrite`. Resume and standalone evaluation reject a checkpoint
from another mode; B/C additionally check coefficients and normalization hash.
Old checkpoints lacking mode metadata are treated as `none`.

Standalone evaluation reads the saved config and uses the same final output:

```bash
python MSTT/evaluate.py \
  --run-dir /data5/tzhang174/TrafficFlow/xuancheng26/training_results_3s/MSTT/v0.2_concat_A2A_projected/p05/All2All/seed_42 \
  --device cuda:3 --splits validation test --save-predictions
```

Replace the mode directory to evaluate A or B. Results go under `reeval/`.

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

### 12→12 sequence reconstruction data (not wired into the trainer)

`data/sequence_dataset.py::SequenceStateWindowDataset` serves the same `x`
with `y = full[s:s+12]` (`[12, 1406, C_out]`, step `t` aligned with `x[t]`;
reconstruction, not forecasting). Targets come from one rate-independent
`window_data/sequence_targets_12step/{split}/{date}.npz` (`full_raw
[288, 1406, 4]`, raw counts) per date and use the same 12→1 target scaler, so
`y[-1]` equals the 12→1 `y`. Both `x` and `full_raw` are memory-mapped, so
DataLoader workers share the page cache.

```python
from MSTT.data.sequence_dataset import open_sequence_datasets, build_sequence_loader
data = open_sequence_datasets(config["data_root"], "p20", "All2All")
loader = build_sequence_loader(data["train"], batch_size=16, shuffle=True, seed=42, num_workers=4)
```

## Architecture

```text
x [B,T,N,C_in] ─ Linear ─────┐  (32)
road id  ─ Embedding ────────┤  (16)  concat on the last axis -> Dropout -> H [B,T,N,D=64]
TOD slot ─ Embedding ────────┤  (8)   (TOD = window_start_slot + t, 288 slots)
DOW      ─ Embedding ────────┘  (8)   (DOW = day_index % 7, Monday = 0)

H += A[None]   optional adaptive embedding A [T, N, D] (use_adaptive_embedding, off by default)

L x block:
  H += Drop(TemporalAttn(RMSNorm(H)))   per node over T steps; QK-Norm, RoPE, SDPA, no causal mask
  H += Drop(SpatialAttn(RMSNorm(H)))    per step over R-graph neighbours (+ self); sparse edge softmax
  H += Drop(SwiGLU(RMSNorm(H)))         W_out(SiLU(W_gate H) * W_value H)

RMSNorm -> flatten T*D -> Linear -> [B, N, C_out]
```

**Embedding fusion.** `embedding_fusion=concat` (v0.2 default) broadcasts
each part to `[B, T, N, d_i]` and concatenates value, node, TOD, DOW in that
order, with widths `value_embedding_dim / node_embedding_dim /
tod_embedding_dim / dow_embedding_dim` = 32 / 16 / 8 / 8. Every width must be
positive and the sum must equal `d_model`, otherwise config validation fails.
A part switched off with `use_node_embedding / use_tod / use_dow` contributes
zeros of its width, so the backbone still sees `d_model`. No extra projection
follows the concat. `embedding_fusion=add` is v0.1: every part is `d_model`
wide and summed; the four widths are ignored. Parameter names are the same
in both modes, so v0.1 checkpoints load strictly with `add`. A v0.1
`resolved_config.json` has no `embedding_fusion` key; `evaluate.py` and the
trainer read it as `add` (`config.LEGACY_DEFAULTS`). Embedding tables
keep the v0.1 init (`trunc_normal(std=0.02)`).

**Adaptive embedding (off by default).** `use_adaptive_embedding=1` adds the
STAEformer `adaptive_embedding`: one learnable `[input_window, num_nodes,
d_model]` = `[12, 1406, 64]` table (1,079,808 parameters), `xavier_uniform_`
init, shared by every sample. `T` is the relative history position 0–11,
`N` the road in R-graph order; it reads no input values, no calendar, and
nothing beyond the 12 history slots. It is a separate module
(`model/adaptive_embedding.py`, state key `adaptive.adaptive_embedding`),
not the node or TOD/DOW embedding. STAEformer concatenates it; MSTT adds it,
broadcast over the batch, to the fused embedding *after* the embedding
dropout (`H = Dropout(concat(...)) + A`), so the backbone width stays 64, the
32/16/8/8 split is unchanged, there is no extra projection, and the table
itself is not dropped. It is built after every other module, so the other
weights initialize identically with the switch on or off; off reproduces the
previous model bit for bit. It sits in the AdamW weight-decay group like the
other embedding tables. Old run directories have no
`use_adaptive_embedding` key and are read as off; checkpoints load strictly,
so a table missing from or unexpected in a checkpoint is an error.

| Module | File | Switches |
|---|---|---|
| Input embedding | `model/embedding.py` | `embedding_fusion` = `concat` / `add`, `{value,node,tod,dow}_embedding_dim`, `use_node_embedding`, `use_tod`, `use_dow` |
| Adaptive embedding | `model/adaptive_embedding.py` | `use_adaptive_embedding` |
| Temporal attention | `model/attention.py::TemporalSelfAttention` | `use_qk_norm`, `use_rope`, `use_gated_attention` |
| Spatial attention | `model/attention.py::SparseSpatialAttention` (`spatial_attention=sparse`) / `GlobalSpatialAttention` (`global`) | `spatial_direction`, `use_edge_type_bias` (sparse only), `spatial_chunk_size`, `spatial_checkpoint` |
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

**Global spatial attention (`spatial_attention=global`, v0.4).** At every
step each road attends to all 1406 roads of the same step (itself included):
`q/k/v` are `[B*T, H, N, d]` and go through
`F.scaled_dot_product_attention` with no mask, no top-k, no positional bias
and no RoPE; steps never mix. QK-Norm, the optional output gate, the QKV /
output projections and dropout are the same as the sparse module. The R
graph is still loaded and checked for the node count, but it restricts
nothing; `use_edge_type_bias` and `spatial_direction` do not apply
(`validate_config` rejects `use_edge_type_bias=1`). `spatial_chunk_size`
splits only the independent `B*T` axis and `spatial_checkpoint` recomputes
each chunk in backward; keys are never chunked, so both are memory knobs
only. On CUDA, SDPA runs a fused kernel and never materializes `N x N`
scores (batch 16, fp32: 5.37 GB peak, same as sparse). The module builds its
layers in the sparse order, so with the same seed it starts from the same
weights as the sparse model minus the edge logits.

**Gated attention (off by default).** `sigmoid(W_g x + b_g)`, `W_g` [D, D],
computed from the normalized sublayer input at the query position,
multiplies the concatenated head outputs elementwise (per head, per channel)
before the output projection.

## Training defaults (`config.py`)

`d_model 64, heads 4, layers 2, dropout 0.1, QK-Norm on, RoPE on, gate off,
sparse bidirectional spatial attention, TOD/DOW/node embeddings on,
concat embedding fusion 32/16/8/8, adaptive embedding off` (v0.1 used `add`;
pass `--embedding-fusion add` to reproduce it).
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
`--device auto` (default) ignores free memory and randomly picks a visible GPU with no compute process. If every visible GPU is busy, training does not start. A
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

`resolved_config.json` and `run_metadata.json` store the fusion mode and
the widths actually built under `embedding`.

**Experiment isolation.** The top level `MSTT/<rate>/<task>/seed_42/` holds the
v0.1 (`add`, batch 16) runs. Later experiments pass
`--output-root .../MSTT/<experiment_id>` and get their own
`<experiment_id>/<rate>/<task>/seed_42/`, `overall_summary.*` and `_logs/`.
`train.py` refuses to skip into or resume a run directory whose stored model
config or batch size differs from the request (`--overwrite` replaces it).

## v0.2 experiment: `v0.2_concat_embedding_bs16`

Change from v0.1: `embedding_fusion` `add` -> `concat` (32/16/8/8). Everything
else, including batch 16, is the v0.1 default. Parameters for I2I fell from
243,489 to 159,417 because the node and TOD tables are narrower.

Timed training steps (concat, p05 I2I, RTX 4090, fp32, spatial checkpointing on,
10 warm-up + 40 timed steps; `_benchmarks/v0.2_concat_batch_size_p05_I2I.json`):

| Batch | Peak allocated | Peak reserved | Step | Samples/s | Est. epoch (train + val) |
|---:|---:|---:|---:|---:|---:|
| 16 | 5.37 GB | 7.17 GB | 140 ms | 114.3 | 51.5 s |
| 24 | 8.02 GB | 10.88 GB | 215 ms | 111.5 | 52.6 s |
| 32 | 10.70 GB | 14.31 GB | 289 ms | 110.9 | 53.2 s |
| 48 | — | — | — | — | CUDA grid limit (B*N > 65535) |

Throughput is already saturated at batch 16; larger batches cost 1.5–2x the
memory and run slightly slower, so the run keeps batch 16. This also keeps
the optimization identical to v0.1, so the comparison isolates the fusion
change (the learning rate is not rescaled).

```bash
python MSTT/train.py --task I2I --penetration p05 --gpu 4 --batch-size 16 \
    --embedding-fusion concat \
    --output-root $DATA_ROOT/xuancheng26/training_results_3s/MSTT/v0.2_concat_embedding_bs16
```

## v0.3 experiment: `v0.3_concat_adaptive_bs16`

Change from v0.2: `use_adaptive_embedding` 0 -> 1. Everything else is the v0.2
configuration (concat 32/16/8/8, batch 16, seed 42, same optimizer, loss,
schedule, early stopping, split, and metrics). The reference is v0.2, not
v0.1. Parameters for I2I grow from 159,417 to 1,239,225 (+1,079,808, the
`[12, 1406, 64]` table), so a gain mixes the mechanism with a 7.8x larger
parameter count. Peak memory is unchanged (5.38 GB allocated at batch 16).

```bash
python MSTT/train.py --task I2I --penetration p05 --batch-size 16 \
    --embedding-fusion concat --use-adaptive-embedding 1 \
    --output-root $DATA_ROOT/xuancheng26/training_results_3s/MSTT/v0.3_concat_adaptive_bs16
```

## v0.4 experiment: `v0.4_global`

Change from v0.2: `spatial_attention` `sparse` -> `global` (and therefore
`use_edge_type_bias` 1 -> 0, not applicable). Adaptive embedding stays off;
everything else is the v0.2 configuration (concat 32/16/8/8, batch 16,
fp32, `spatial_chunk_size 64`, `spatial_checkpoint 1`, seed 42, same
optimizer, loss, schedule, early stopping, split, and metrics). The
reference is v0.2. Parameters for I2I: 159,385 (v0.2 159,417; the 2 x 16
edge logits are gone). Step time at batch 16 on an RTX 4090 is about 260 ms
(v0.2: 140 ms); peak memory 5.37 GB allocated.

```bash
python MSTT/train.py --task I2I --penetration p05 --gpu 3 --batch-size 16 \
    --embedding-fusion concat --use-adaptive-embedding 0 \
    --spatial-attention global --use-edge-type-bias 0 \
    --output-root $DATA_ROOT/xuancheng26/training_results_3s/MSTT/v0.4_global
```
