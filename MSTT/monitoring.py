"""Graded training monitor for one MSTT run.

Level 0 does not create files, hooks, parameter copies, or probes. Higher
levels accumulate. Records stay under ``<run_dir>/monitoring/`` and are not
part of the model ``state_dict`` or the checkpoint-selection rule.

``pred_raw`` in the official prediction NPZ is the final prediction in
physical units. The head output before projection, also in physical units,
is ``raw_pred_physical``. Inside ``forward_with_aux``, ``raw_pred`` and
``final_pred`` are still normalized.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from MSTT import CODE_VERSION
from MSTT.config import STATE_CHANNELS
from MSTT.data.dataset import time_features
from MSTT.model.auxiliary_gate import AuxiliaryStateGate
from MSTT.model.conservation import conservation_residual
from reimplementation.common.utils.atomic_io import atomic_write_json, atomic_write_npz, atomic_write_text

QUANTILE_CAP = 8192
NEAR_ZERO = 1e-3
UPDATE_EPS = 1e-12
GATE_LOW = 0.05
GATE_HIGH = 1.95
TOPK = 3
QUERY_ROADS = 16
QUERY_TIMES = (0, 3, 7, 11)
VAL_PROBE_SEED = 42
SPEED_EDGES = (20.0, 40.0, 60.0)
SILU_DERIVATIVE_NOTE = (
    "SiLU derivative statistics describe this probe only. "
    "A small derivative is not a verdict that the unit never trained."
)
SIGMOID_NOTE = (
    "Gate values are g=2*sigmoid(logit). A mean near 1 can still hide mass near 0 and 2. "
    "count=0 speed is a fill, not a congestion measurement."
)


def _py(value: Any) -> Any:
    if torch.is_tensor(value):
        if int(value.numel()) > 64:
            return {"tensor_shape": [int(item) for item in value.shape], "omitted": True}
        return _py(value.detach().float().cpu().tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_py(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _py(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_py(item) for item in value]
    return value


def _sync(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _rss_mb() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[1]),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_py(record), ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _truncate_jsonl(path: Path, committed_epoch: int) -> int:
    """Drop an interrupted epoch and duplicate keys. ``committed_epoch`` is kept."""
    if not path.is_file():
        return 0
    kept: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    removed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            removed += 1
            continue
        epoch = row.get("epoch_index")
        if epoch is not None and int(epoch) > int(committed_epoch):
            removed += 1
            continue
        key = (
            row.get("kind"),
            row.get("epoch_index"),
            row.get("phase"),
            row.get("module"),
            row.get("source"),
            row.get("probe_id"),
        )
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        kept.append(row)
    payload = "".join(json.dumps(_py(row), ensure_ascii=False, sort_keys=True) + "\n" for row in kept)
    atomic_write_text(path, payload)
    return removed


def _tensor_summary(values: torch.Tensor, *, cap: int = QUANTILE_CAP) -> dict[str, Any]:
    flat = values.detach().float().reshape(-1)
    finite = flat[torch.isfinite(flat)]
    count = int(finite.numel())
    if count == 0:
        return {"count": 0, "nonfinite_count": int(flat.numel()), "missing": True}
    if count > cap:
        stride = (count + cap - 1) // cap
        sample = finite[::stride][:cap]
        method = "deterministic_stride"
    else:
        sample = finite
        method = "full"
    quantiles = torch.quantile(
        sample.float().cpu(),
        torch.tensor([0.05, 0.5, 0.95], dtype=torch.float32),
    )
    return {
        "count": count,
        "nonfinite_count": int(flat.numel()) - count,
        "missing": False,
        "mean": float(finite.mean()),
        "variance": float(finite.var(unbiased=False)),
        "rms": float(finite.square().mean().sqrt()),
        "near_zero_fraction": float((finite.abs() < NEAR_ZERO).float().mean()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "p05": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "quantile_sample_count": int(sample.numel()),
        "quantile_sample_cap": int(cap),
        "quantile_method": method,
        "near_zero_threshold": NEAR_ZERO,
    }


def _channel_variance(values: torch.Tensor) -> np.ndarray | None:
    if values.ndim < 1 or int(values.shape[-1]) > 256 or int(values.shape[-1]) < 1:
        return None
    flat = values.detach().float().reshape(-1, int(values.shape[-1]))
    if int(flat.shape[0]) < 1:
        return None
    return flat.var(dim=0, unbiased=False).cpu().numpy().astype(np.float32)


def _read_node_ids(path: Path) -> list[str]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames and "edge_id" in reader.fieldnames:
            return [str(row["edge_id"]) for row in reader]
        handle.seek(0)
        rows = list(csv.reader(handle))
    if not rows:
        return []
    start = 1 if rows[0] and not rows[0][0].isdigit() else 0
    return [str(row[0]) for row in rows[start:] if row]


def _day_spans(window_root: Path, rate: str, split: str, kind: str) -> list[dict[str, Any]]:
    manifest_path = Path(window_root) / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spans = []
    offset = 0
    for day in manifest[split]:
        path = Path(window_root) / rate / split / str(day) / f"{kind}.npz"
        with np.load(path, mmap_mode="r") as payload:
            count = int(payload["y"].shape[0]) if "y" in payload.files else int(payload["x"].shape[0])
            slots = np.asarray(payload["target_slot"], dtype=np.int64)
            starts = np.asarray(payload["window_start_slot"], dtype=np.int64)
            weekdays = np.asarray(payload["day_index"], dtype=np.int64)
        if count != int(slots.shape[0]):
            raise RuntimeError(f"{path} target_slot length {slots.shape[0]} != windows {count}")
        spans.append(
            {
                "date": str(day),
                "start": offset,
                "count": count,
                "target_slot": slots,
                "window_start_slot": starts,
                "day_index": weekdays,
            }
        )
        offset += count
    return spans


def _choose_probes(spans: list[dict[str, Any]], size: int, *, split: str, sampling_seed: int) -> list[dict[str, Any]]:
    """Pick windows near evenly spaced times of day. Ties keep the earlier sample."""
    if size < 1:
        return []
    candidates = []
    for band in range(size):
        center = int((band + 0.5) * 288 / size)
        for span in spans:
            slots = span["target_slot"]
            if slots.size == 0:
                continue
            distance = np.abs(slots - center)
            local = int(np.argmin(distance))
            sample_index = int(span["start"] + local)
            candidates.append(
                (
                    int(distance[local]),
                    sample_index,
                    {
                        "split": split,
                        "sampling_seed": int(sampling_seed),
                        "date": span["date"],
                        "local_index": local,
                        "sample_index": sample_index,
                        "target_slot": int(slots[local]),
                        "window_start_slot": int(span["window_start_slot"][local]),
                        "day_index": int(span["day_index"][local]),
                        "band": int(band),
                        "band_center_slot": center,
                    },
                )
            )
    chosen: list[dict[str, Any]] = []
    used: set[int] = set()
    for band in range(size):
        pool = [item for item in candidates if item[2]["band"] == band and item[1] not in used]
        if not pool:
            continue
        pool.sort()
        meta = dict(pool[0][2])
        meta["probe_id"] = f"{split}-seed{sampling_seed}-{len(chosen)}"
        used.add(int(meta["sample_index"]))
        chosen.append(meta)
    return chosen


def _logical_module(name: str) -> str:
    prefixes = (
        "embedding.value_mlps.0",
        "embedding.value_mlps.1",
        "embedding.value_mlps.2",
        "embedding.value_mlps.3",
        "embedding.auxiliary_gate",
        "embedding.value_projection",
        "embedding.node_embedding",
        "embedding.tod_embedding",
        "embedding.dow_embedding",
        "adaptive",
        "blocks.0.temporal",
        "blocks.0.spatial",
        "blocks.0.ffn",
        "head",
        "projection",
    )
    for prefix in prefixes:
        if name == prefix or name.startswith(prefix + "."):
            return prefix
    return name.split(".", 1)[0]


def _parameter_plan(model: nn.Module, road_index: list[int]) -> list[dict[str, Any]]:
    plan = []
    generator = np.random.Generator(np.random.PCG64(42))
    for name, param in model.named_parameters():
        spec: dict[str, Any] = {
            "name": name,
            "module": _logical_module(name),
            "shape": [int(item) for item in param.shape],
            "numel": int(param.numel()),
        }
        rows = [int(item) for item in road_index]
        if name.endswith("node_embedding.weight") and param.ndim == 2:
            spec["index_kind"] = "rows"
            spec["rows"] = rows
            spec["row_axis"] = 0
        elif name.endswith("adaptive_embedding") and param.ndim == 3:
            spec["index_kind"] = "rows"
            spec["rows"] = rows
            spec["row_axis"] = 1
        elif int(param.numel()) > 65536:
            count = min(4096, int(param.numel()))
            chosen = np.sort(generator.choice(int(param.numel()), size=count, replace=False))
            spec["index_kind"] = "flat"
            spec["flat_index"] = [int(item) for item in chosen.tolist()]
        else:
            spec["index_kind"] = "full"
        plan.append(spec)
    return plan


def _gather(param: torch.Tensor, spec: Mapping[str, Any]) -> torch.Tensor:
    data = param.detach()
    kind = spec["index_kind"]
    if kind == "rows":
        rows = torch.as_tensor(spec["rows"], device=data.device, dtype=torch.long)
        data = data.index_select(int(spec["row_axis"]), rows)
    elif kind == "flat":
        index = torch.as_tensor(spec["flat_index"], device=data.device, dtype=torch.long)
        data = data.reshape(-1).index_select(0, index)
    return data.float().reshape(-1).cpu().clone()


def _rms(values: torch.Tensor) -> float:
    if int(values.numel()) == 0:
        return 0.0
    return float(values.float().square().mean().sqrt())


def _l2(values: torch.Tensor) -> float:
    if int(values.numel()) == 0:
        return 0.0
    return float(values.float().norm())


def _attention_distribution(weights: torch.Tensor) -> dict[str, Any]:
    """``weights`` [query, head, key], already softmaxed on the key axis."""
    probs = weights.float()
    entropy = -(probs.clamp_min(1e-12).log() * probs).sum(dim=-1)
    key_count = int(probs.shape[-1])
    normalizer = math.log(key_count) if key_count > 1 else 1.0
    top = torch.topk(probs, k=min(TOPK, key_count), dim=-1).values
    return {
        "query_count": int(probs.shape[0]),
        "head_count": int(probs.shape[1]),
        "key_count": key_count,
        "entropy_mean": float(entropy.mean()),
        "normalized_entropy_mean": float((entropy / normalizer).mean()),
        "top1_mass_mean": float(top[..., 0].mean()),
        "topk_mass_mean": float(top.sum(dim=-1).mean()),
        "topk": int(min(TOPK, key_count)),
    }


def _temporal_attention(module, hidden: torch.Tensor, roads: list[int]) -> dict[str, Any] | None:
    if not hasattr(module, "qkv"):
        return None
    batch, steps, nodes, _width = hidden.shape
    roads_t = [int(item) for item in roads if 0 <= int(item) < nodes]
    if batch != 1 or not roads_t:
        return {"applicable": False, "reason": "temporal attention probe needs one window and a road index"}
    seq = hidden.permute(0, 2, 1, 3).reshape(batch * nodes, steps, hidden.shape[-1])
    projected = module.qkv(seq).view(batch * nodes, steps, 3, module.num_heads, module.head_dim)
    q, k, _v = projected.permute(2, 0, 3, 1, 4)[:3]
    if module.q_norm is not None:
        q = module.q_norm(q)
        k = module.k_norm(k)
    if module.rope is not None:
        q, k = module.rope(q, k)
    index = torch.as_tensor(roads_t, device=hidden.device, dtype=torch.long)
    q = q.index_select(0, index)
    k = k.index_select(0, index)
    scale = 1.0 / math.sqrt(int(module.head_dim))
    scores = torch.einsum("rhqd,rhkd->rhqk", q, k) * scale
    weights = torch.softmax(scores.float(), dim=-1)
    stats = _attention_distribution(weights.permute(0, 2, 1, 3).reshape(-1, module.num_heads, steps))
    diagonal = weights[:, :, torch.arange(steps, device=weights.device), torch.arange(steps, device=weights.device)]
    stats.update(
        applicable=True,
        kind="temporal",
        self_weight_mean=float(diagonal.mean()),
        history_mass_mean=[float(item) for item in weights.mean(dim=(0, 1, 2)).detach().cpu().tolist()],
        scale="1/sqrt(head_dim)",
        head_dim=int(module.head_dim),
        query_roads=roads_t,
        note="softmax over the 12 history keys; weights are not stored",
    )
    return stats


def _spatial_attention(module, hidden: torch.Tensor, roads: list[int], times: tuple[int, ...]) -> dict[str, Any] | None:
    if not hasattr(module, "qkv"):
        return None
    if module.__class__.__name__ == "SparseSpatialAttention":
        return {
            "applicable": True,
            "kind": "spatial_sparse",
            "note": "training softmax is over graph edges, so a dense N-key entropy is not the training path",
            "key_count": int(getattr(module, "num_nodes", hidden.shape[2])),
        }
    _batch, steps, nodes, _width = hidden.shape
    use_times = [int(item) for item in times if 0 <= int(item) < steps]
    use_roads = [int(item) for item in roads if 0 <= int(item) < nodes]
    if not use_times or not use_roads:
        return {"applicable": False, "reason": "spatial probe selection is empty"}
    projected = module.qkv(hidden)
    projected = projected.view(steps, nodes, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4)
    q, k, _v = projected[:3]
    if getattr(module, "q_norm", None) is not None:
        q = module.q_norm(q)
        k = module.k_norm(k)
    time_index = torch.as_tensor(use_times, device=hidden.device, dtype=torch.long)
    road_index = torch.as_tensor(use_roads, device=hidden.device, dtype=torch.long)
    q = q.index_select(0, time_index).index_select(2, road_index)
    k = k.index_select(0, time_index)
    scale = 1.0 / math.sqrt(int(module.head_dim))
    entropies = []
    top1 = []
    topk = []
    self_mass = []
    for start in range(0, len(use_roads), 4):
        q_chunk = q[:, :, start : start + 4, :]
        scores = torch.einsum("thrd,thnd->thrn", q_chunk, k) * scale
        weights = torch.softmax(scores.float(), dim=-1)
        flat = weights.permute(0, 2, 1, 3).reshape(-1, module.num_heads, nodes)
        piece = _attention_distribution(flat)
        entropies.append(piece["entropy_mean"])
        top1.append(piece["top1_mass_mean"])
        topk.append(piece["topk_mass_mean"])
        local_roads = road_index[start : start + q_chunk.shape[2]]
        gathered = weights[:, :, torch.arange(q_chunk.shape[2], device=weights.device), local_roads]
        self_mass.append(float(gathered.mean()))
        del scores, weights, flat
    return {
        "applicable": True,
        "kind": "spatial",
        "entropy_mean": float(np.mean(entropies)),
        "normalized_entropy_mean": float(np.mean(entropies) / math.log(nodes)),
        "top1_mass_mean": float(np.mean(top1)),
        "topk_mass_mean": float(np.mean(topk)),
        "topk": TOPK,
        "self_weight_mean": float(np.mean(self_mass)),
        "key_count": int(nodes),
        "query_times": use_times,
        "query_roads": use_roads,
        "scale": "1/sqrt(head_dim)",
        "head_dim": int(module.head_dim),
        "note": "queries are chunked; the N×N matrix is not materialized or saved",
    }


def _count_key(count: np.ndarray) -> np.ndarray:
    labels = np.full(count.shape, "5+", dtype=object)
    labels[count <= 0] = "0"
    labels[np.round(count) == 1] = "1"
    labels[(count >= 2) & (count <= 4)] = "2-4"
    return labels


def _speed_key(speed: np.ndarray, count: np.ndarray) -> np.ndarray:
    labels = np.full(speed.shape, "count_0_excluded", dtype=object)
    valid = count > 0
    labels[valid & (speed < SPEED_EDGES[0])] = "0-20"
    labels[valid & (speed >= SPEED_EDGES[0]) & (speed < SPEED_EDGES[1])] = "20-40"
    labels[valid & (speed >= SPEED_EDGES[1]) & (speed < SPEED_EDGES[2])] = "40-60"
    labels[valid & (speed >= SPEED_EDGES[2])] = "60+"
    return labels


class _Bucket:
    def __init__(self) -> None:
        self.road_slots = 0
        self.windows: set[int] = set()
        self.abs = np.zeros(4, dtype=np.float64)
        self.signed_in = 0.0
        self.gate = 0.0
        self.gate_dev = 0.0
        self.gate_count = 0
        self.correction = np.zeros(4, dtype=np.float64)
        self.correction_count = 0

    def add(self, pred: np.ndarray, true: np.ndarray, window_ids: np.ndarray, gate=None, correction=None) -> None:
        error = pred.astype(np.float64) - true.astype(np.float64)
        self.road_slots += int(error.shape[0])
        self.windows.update(int(item) for item in window_ids.tolist())
        self.abs += np.abs(error).sum(axis=0)
        self.signed_in += float(error[:, 0].sum())
        if gate is not None:
            self.gate += float(gate.reshape(-1).sum())
            self.gate_dev += float(np.abs(gate.reshape(-1) - 1.0).sum())
            self.gate_count += int(gate.reshape(-1).size)
        if correction is not None:
            self.correction += np.abs(correction.astype(np.float64)).sum(axis=0)
            self.correction_count += int(correction.shape[0])

    def export(self) -> dict[str, Any]:
        if self.road_slots == 0:
            return {"missing": True, "road_slot_count": 0, "window_count": len(self.windows)}
        mae = self.abs / self.road_slots
        payload = {
            "missing": False,
            "road_slot_count": int(self.road_slots),
            "window_count": len(self.windows),
            "inflow_mae": float(mae[0]),
            "inflow_bias": float(self.signed_in / self.road_slots),
            "state_mae": {name: float(mae[index]) for index, name in enumerate(STATE_CHANNELS)},
            "count_unit": "road-slot (one road in one window)",
        }
        if self.gate_count:
            payload["gate_mean"] = float(self.gate / self.gate_count)
            payload["gate_abs_deviation_mean"] = float(self.gate_dev / self.gate_count)
        if self.correction_count:
            payload["projection_correction_mae"] = {
                name: float(self.correction[index] / self.correction_count) for index, name in enumerate(STATE_CHANNELS)
            }
        return payload


def _accumulate_into(
    tables: dict[str, dict[str, _Bucket]],
    pred,
    true,
    groups: Mapping[str, np.ndarray],
    window_ids,
    gate=None,
    correction=None,
) -> None:
    flat_pred = np.asarray(pred).reshape(-1, np.asarray(pred).shape[-1])
    flat_true = np.asarray(true).reshape(-1, np.asarray(true).shape[-1])
    flat_windows = np.asarray(window_ids).reshape(-1)
    flat_gate = None if gate is None else np.asarray(gate).reshape(-1)
    flat_correction = None if correction is None else np.asarray(correction).reshape(-1, np.asarray(correction).shape[-1])
    for name, labels in groups.items():
        flat_labels = np.asarray(labels).reshape(-1)
        dest = tables.setdefault(name, {})
        for key in sorted(set(flat_labels.tolist()), key=str):
            mask = flat_labels == key
            if not bool(np.any(mask)):
                continue
            bucket = dest.setdefault(str(key), _Bucket())
            bucket.add(
                flat_pred[mask],
                flat_true[mask],
                flat_windows[mask],
                None if flat_gate is None else flat_gate[mask],
                None if flat_correction is None else flat_correction[mask],
            )


def _export_tables(tables: Mapping[str, Mapping[str, _Bucket]]) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        name: {key: bucket.export() for key, bucket in table.items()}
        for name, table in tables.items()
    }


def _accumulate_groups(pred, true, groups: Mapping[str, np.ndarray], window_ids, gate=None, correction=None) -> dict[str, dict[str, dict[str, Any]]]:
    tables: dict[str, dict[str, _Bucket]] = {}
    _accumulate_into(tables, pred, true, groups, window_ids, gate, correction)
    return _export_tables(tables)


def _average_seed_tables(per_seed: Mapping[int, Mapping[str, Mapping[str, Mapping[str, Any]]]]) -> dict[str, Any]:
    seeds = [int(seed) for seed in per_seed]
    table_names = sorted({name for tables in per_seed.values() for name in tables})
    averaged: dict[str, list[dict[str, Any]]] = {}
    for table_name in table_names:
        keys = sorted({key for tables in per_seed.values() for key in tables.get(table_name, {})})
        rows = []
        for key in keys:
            present = []
            missing = []
            for seed in seeds:
                row = per_seed[seed].get(table_name, {}).get(key)
                if row is None or row.get("missing") or int(row.get("road_slot_count", 0)) == 0:
                    missing.append(seed)
                else:
                    present.append((seed, row))
            if not present:
                rows.append({"key": key, "missing": True, "n_seeds": 0, "missing_seeds": missing or seeds})
                continue
            def mean_of(getter):
                return float(np.mean([getter(row) for _seed, row in present]))

            state = {
                name: mean_of(lambda row, channel=name: row["state_mae"][channel])
                for name in STATE_CHANNELS
            }
            payload = {
                "key": key,
                "missing": False,
                "n_seeds": len(present),
                "missing_seeds": missing,
                "inflow_mae": mean_of(lambda row: row["inflow_mae"]),
                "inflow_bias": mean_of(lambda row: row["inflow_bias"]),
                "state_mae": state,
                "road_slot_count_sum": int(sum(row["road_slot_count"] for _seed, row in present)),
                "aggregation": "unweighted_mean_of_per_seed_metrics",
            }
            if any("gate_mean" in row for _seed, row in present):
                gated = [row for _seed, row in present if "gate_mean" in row]
                payload["gate_mean"] = float(np.mean([row["gate_mean"] for row in gated]))
                payload["gate_abs_deviation_mean"] = float(np.mean([row["gate_abs_deviation_mean"] for row in gated]))
                payload["gate_n_seeds"] = len(gated)
            if any("projection_correction_mae" in row for _seed, row in present):
                corrected = [row for _seed, row in present if "projection_correction_mae" in row]
                payload["projection_correction_mae"] = {
                    name: float(np.mean([row["projection_correction_mae"][name] for row in corrected]))
                    for name in STATE_CHANNELS
                }
            rows.append(payload)
        averaged[table_name] = rows
    return {
        "seeds": seeds,
        "n_seeds_expected": len(seeds),
        "aggregation": "unweighted_mean_of_per_seed_metrics",
        "tables": averaged,
    }


def _load_prediction(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"prediction archive is missing: {path}")
    with np.load(path) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _squeeze_prediction(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 4 and values.shape[1] == 1:
        return values[:, 0]
    if values.ndim == 3:
        return values
    raise ValueError(f"expected prediction [S, N, C] or [S, 1, N, C], got {values.shape}")


class RunMonitor:
    """Sampling and writers for one training process. Level 0 is inert."""

    def __init__(self, run, model: nn.Module, *, train_data_sha256: str | None = None) -> None:
        self.run = run
        self.model = model
        self.train_data_sha256 = train_data_sha256
        self.level = int(run.config.get("monitor_level", 0) or 0)
        self.interval = int(run.config.get("monitor_module_interval", 5))
        self.mechanism_epochs = [int(item) for item in run.config.get("monitor_mechanism_epochs", [1, 5, 20, 50, 100])]
        self.probe_size = int(run.config.get("monitor_probe_size", 4))
        self.directory = Path(run.run_dir) / "monitoring"
        self.probes: dict[str, list[dict[str, Any]]] = {}
        self.plan: list[dict[str, Any]] = []
        self.road_index: list[int] = []
        self.node_ids: list[str] = []
        self._before: dict[str, torch.Tensor] = {}
        self._grads: dict[str, torch.Tensor | None] = {}
        self._pending: list[dict[str, Any]] | None = None
        self._channel: dict[str, np.ndarray] = {}
        self._norms: list[float] = []
        self._over_clip = 0
        self._skipped = 0
        self._hooks: list[Any] = []
        self._accept_hooks = False
        self._hook_stats: dict[str, list[dict[str, Any]]] = {}
        self._hook_channels: dict[str, list[np.ndarray]] = {}
        self.monitor_seconds = 0.0
        self.health_epochs: list[int] = []
        self.module_epochs: list[int] = []
        self.mechanism_epochs_done: list[int] = []
        self.anomalies: list[str] = []
        self.associations_written = False
        self._closed = False

    def active(self) -> bool:
        return self.level > 0

    def prepare(self, *, snapshot_init: bool) -> None:
        if not self.active():
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        if not self.node_ids:
            self._select_context()
        elif not self.road_index:
            count = min(QUERY_ROADS, len(self.node_ids))
            generator = np.random.Generator(np.random.PCG64(42))
            self.road_index = sorted(int(item) for item in generator.choice(len(self.node_ids), size=count, replace=False))
        if self.level >= 2 and not self.plan:
            self.plan = _parameter_plan(self.model, self.road_index)
        if self.level >= 2 and not self.probes:
            self._select_train_val_probes()
        self._write_manifest()
        if snapshot_init and self.level >= 3 and any(self.probes.values()):
            self._mechanisms(epoch_index=None, phase="init")

    def truncate(self, committed_epoch_index: int) -> None:
        if not self.active():
            return
        removed = 0
        for name in ("health.jsonl", "module_stats.jsonl", "mechanisms.jsonl"):
            removed += _truncate_jsonl(self.directory / name, committed_epoch_index)
        self.anomalies.append(
            f"resume truncated monitoring rows above epoch_index {committed_epoch_index}; removed {removed}"
        )

    def wants_capture(self, epoch_index: int, step_index: int) -> bool:
        return self.level >= 2 and int(step_index) == 0 and self._module_epoch(epoch_index)

    def begin_capture(self, model: nn.Module) -> None:
        if self.level < 2:
            return
        started = self._clock()
        self._before = {}
        self._grads = {}
        self._channel = {}
        for spec in self.plan:
            param = dict(model.named_parameters())[spec["name"]]
            self._before[spec["name"]] = _gather(param, spec)
            if param.grad is None:
                self._grads[spec["name"]] = None
            else:
                self._grads[spec["name"]] = _gather(param.grad, spec)
                if param.ndim == 2 and int(param.shape[0]) <= 256 and spec["index_kind"] == "full":
                    grad = param.grad.detach().float()
                    self._channel[spec["name"]] = grad.square().mean(dim=1).sqrt().cpu().numpy().astype(np.float32)
        self.monitor_seconds += self._clock() - started

    def finish_capture(self, model: nn.Module, *, skipped: bool) -> None:
        if self.level < 2 or not self._before:
            return
        started = self._clock()
        records = []
        for spec in self.plan:
            param = dict(model.named_parameters())[spec["name"]]
            before = self._before[spec["name"]]
            after = _gather(param, spec)
            delta = after - before
            grad = self._grads.get(spec["name"])
            before_norm = _l2(before)
            records.append(
                {
                    "module": spec["module"],
                    "parameter": spec["name"],
                    "source": "train_step",
                    "index_kind": spec["index_kind"],
                    "element_count": int(before.numel()),
                    "observed_rows": spec.get("rows"),
                    "weight_rms": _rms(before),
                    "grad_rms": None if grad is None else _rms(grad),
                    "grad_norm": None if grad is None else _l2(grad),
                    "grad_missing": grad is None,
                    "grad_all_zero_on_observed": bool(grad is not None and float(grad.abs().max()) == 0.0),
                    "update_rms": _rms(delta),
                    "relative_update": _l2(delta) / (before_norm + UPDATE_EPS),
                    "zero_denominator": before_norm == 0.0,
                    "optimizer_step_skipped": bool(skipped),
                    "embedding_observation": (
                        "row sample only; a zero gradient can mean this row was absent from the step"
                        if spec["index_kind"] == "rows"
                        else None
                    ),
                }
            )
        self._pending = records
        self.monitor_seconds += self._clock() - started

    def note_train_step(self, grad_norm: float, clip: float, skipped: bool) -> None:
        if self.level < 1:
            return
        self._norms.append(float(grad_norm))
        if clip > 0.0 and float(grad_norm) > float(clip):
            self._over_clip += 1
        if skipped:
            self._skipped += 1

    def on_epoch_end(
        self,
        epoch_index: int,
        row: Mapping[str, Any],
        *,
        train_seconds: float,
        val_seconds: float,
        train_samples: int,
        val_metrics: Mapping[str, Any],
    ) -> None:
        if not self.active():
            return
        self._health(epoch_index, row, train_seconds, val_seconds, train_samples, val_metrics)
        probe_needed = self.level >= 2 and (self._module_epoch(epoch_index) or self._mechanism_epoch(epoch_index))
        if probe_needed:
            activation = self._probe_pass(epoch_index, phase="epoch")
            if self._module_epoch(epoch_index):
                self._write_module_stats(epoch_index, activation)
            if self._mechanism_epoch(epoch_index):
                self._mechanisms(epoch_index, phase="epoch", cached=activation)
        self._norms = []
        self._over_clip = 0
        self._skipped = 0
        self._pending = None
        self._channel = {}

    def on_best(self, model: nn.Module) -> None:
        if self.level < 3:
            return
        self.model = model
        self._mechanisms(epoch_index=None, phase="best")

    def on_predictions_ready(self, model: nn.Module) -> None:
        if self.level < 4:
            return
        self.model = model
        self._write_associations(model)
        self.associations_written = True

    def finish(self) -> None:
        self._summary("completed")

    def fail(self, exc: BaseException) -> None:
        if not self.active():
            return
        self.anomalies.append(repr(exc))
        self._summary("failed")

    def close(self) -> None:
        self._remove_hooks()
        self._closed = True

    def _clock(self) -> float:
        _sync(self._device())
        return time.perf_counter()

    def _device(self) -> torch.device:
        return self.run.device

    def _run_id(self) -> str:
        return str(self.run.config.get("run_id") or "")

    def _module_epoch(self, epoch_index: int) -> bool:
        epoch_no = int(epoch_index) + 1
        return epoch_no == 1 or epoch_no % self.interval == 0

    def _mechanism_epoch(self, epoch_index: int) -> bool:
        return int(epoch_index) + 1 in self.mechanism_epochs

    def _select_context(self) -> None:
        mapping = Path(self.run.package["node_mapping"])
        self.node_ids = _read_node_ids(mapping)
        count = min(QUERY_ROADS, int(self.run.num_nodes))
        generator = np.random.Generator(np.random.PCG64(42))
        self.road_index = sorted(int(item) for item in generator.choice(int(self.run.num_nodes), size=count, replace=False))

    def _select_train_val_probes(self) -> None:
        train_root = Path(self.run.data_root)
        from data_root import sampling_root_from_window, sampling_window_root

        sampling_root = sampling_root_from_window(train_root)
        expected = f"seed_{int(self.run.config['train_sampling_seed'])}"
        if train_root.parent.parent.name != expected:
            raise RuntimeError(f"monitor training probes expected {expected}, found {train_root.parent.parent.name}")
        train_spans = _day_spans(train_root, self.run.rate, "train", self.run.task.target_type)
        if "train" in self.run.datasets and sum(span["count"] for span in train_spans) != len(self.run.datasets["train"]):
            raise RuntimeError("training probe day spans do not match the open training dataset")
        val_root = sampling_window_root(sampling_root, VAL_PROBE_SEED)
        val_spans = _day_spans(val_root, self.run.rate, "validation", self.run.task.target_type)
        self.probes = {
            "train": _choose_probes(
                train_spans,
                self.probe_size,
                split="train",
                sampling_seed=int(self.run.config["train_sampling_seed"]),
            ),
            "validation": _choose_probes(
                val_spans,
                self.probe_size,
                split="validation",
                sampling_seed=VAL_PROBE_SEED,
            ),
        }
        self._val_root = val_root

    def _input_scaler_note(self) -> dict[str, Any]:
        try:
            from reimplementation.common.data.state_windows import all_sp_c_normalization_file

            path = all_sp_c_normalization_file(Path(self.run.data_root))
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            return {"applicable": False, "reason": str(exc)}
        except Exception as exc:
            from reimplementation.common.errors import ReimplementationError

            if not isinstance(exc, ReimplementationError):
                raise
            return {"applicable": False, "reason": str(exc)}
        return {
            "applicable": True,
            "path": str(path),
            "fit_sampling_seed": payload.get("fit_sampling_seed"),
            "fit_split": payload.get("fit_split"),
            "kept_fit_on_sampling_seed_42": int(payload.get("fit_sampling_seed", -1)) == 42,
        }

    def _write_manifest(self) -> None:
        config = self.run.config
        modules = sorted({spec["module"] for spec in self.plan}) if self.plan else []
        present = {name for name, _module in self.model.named_modules()}
        catalog = {
            "embedding.value_mlps.0": "embedding.value_mlps.0" in present or any(name.startswith("embedding.value_mlps.0.") for name in present),
            "embedding.value_mlps.1": any(name.startswith("embedding.value_mlps.1") for name in present),
            "embedding.value_mlps.2": any(name.startswith("embedding.value_mlps.2") for name in present),
            "embedding.value_mlps.3": any(name.startswith("embedding.value_mlps.3") for name in present),
            "embedding.auxiliary_gate": any(name.startswith("embedding.auxiliary_gate") for name in present),
            "embedding.node_embedding": "embedding.node_embedding.weight" in dict(self.model.named_parameters()),
            "embedding.tod_embedding": "embedding.tod_embedding.weight" in dict(self.model.named_parameters()),
            "embedding.dow_embedding": "embedding.dow_embedding.weight" in dict(self.model.named_parameters()),
            "adaptive": any(name.startswith("adaptive.") or name == "adaptive" for name in present),
            "blocks.0.temporal": any(name.startswith("blocks.0.temporal") for name in present),
            "blocks.0.spatial": any(name.startswith("blocks.0.spatial") for name in present),
            "blocks.0.ffn": any(name.startswith("blocks.0.ffn") for name in present),
            "head": any(name.startswith("head") for name in present),
            "projection": any(name.startswith("projection") for name in present),
        }
        payload = {
            "run_id": self._run_id(),
            "code_version": CODE_VERSION,
            "git_revision": _git_revision(),
            "monitor_level": self.level,
            "monitor_module_interval": self.interval,
            "monitor_mechanism_epochs": self.mechanism_epochs,
            "monitor_probe_size": self.probe_size,
            "dataset_id": config.get("dataset_id"),
            "experiment_id": config.get("experiment_id"),
            "train_sampling_seed": int(config.get("train_sampling_seed", 42)),
            "model_seed": int(config.get("model_seed", config.get("seed", 42))),
            "evaluation_sampling_seeds": [int(item) for item in config.get("evaluation_sampling_seeds", [])],
            "train_data_sha256": self.train_data_sha256,
            "input_scaler": self._input_scaler_note(),
            "normalization_sha256": self.run.package.get("state_normalization_sha256"),
            "split_manifest_sha256": self.run.package.get("split_manifest_sha256"),
            "thresholds": {
                "near_zero_abs": NEAR_ZERO,
                "gate_low": GATE_LOW,
                "gate_high": GATE_HIGH,
                "update_eps": UPDATE_EPS,
                "quantile_cap": QUANTILE_CAP,
                "silu_derivative_lt": 0.01,
                "speed_bins_kmh": ["0-20", "20-40", "40-60", "60+"],
                "count_bins": ["0", "1", "2-4", "5+"],
                "clip_grad_norm": config.get("clip_grad_norm"),
            },
            "units": {
                "speed": "km/h raw average_speed_raw",
                "count": "speed_observation_count vehicles",
                "states": "vehicle counts",
                "slot": "5-minute index within the day",
                "day_index": "weekday, Monday=0; the calendar date comes from the split manifest",
            },
            "field_notes": {
                "pred_raw_in_prediction_npz": "physical units of the final prediction after projection",
                "raw_pred_physical": "physical units of the head output before projection",
                "forward_with_aux.raw_pred": "normalized head output",
                "forward_with_aux.final_pred": "normalized output of forward",
                "correction_weights": "softmax correction weights, not a traffic state",
                "diagnostic_records": "analysis only; they do not change the optimizer or the selected checkpoint",
            },
            "modules_tracked": modules,
            "module_applicable": catalog,
            "query_roads": [
                {"index": index, "edge_id": self.node_ids[index] if index < len(self.node_ids) else None}
                for index in self.road_index
            ],
            "probes": self.probes,
            "validation_probe_sampling_seed": VAL_PROBE_SEED,
            "task_gradient_parameter": "blocks.0.ffn.w_out.weight",
            "levels": {
                "1": "every epoch health record",
                "2": "epoch 1 and every monitor_module_interval: one real train step plus fixed eval probes",
                "3": "mechanism epochs, initialization, and the restored best checkpoint",
                "4": "after the best checkpoint is selected and official predictions exist",
            },
        }
        atomic_write_json(self.directory / "manifest.json", _py(payload))

    def _health(self, epoch_index, row, train_seconds, val_seconds, train_samples, val_metrics) -> None:
        missing = []
        per_seed = {}
        report = val_metrics.get("sampling_metrics") if isinstance(val_metrics, Mapping) else None
        if isinstance(report, Mapping) and isinstance(report.get("per_seed"), Mapping):
            for seed, metrics in report["per_seed"].items():
                value = metrics.get("inflow_mae") if isinstance(metrics, Mapping) else None
                per_seed[str(seed)] = None if value is None else float(value)
                if value is None:
                    missing.append(f"validation_seed_{seed}_inflow_mae")
        else:
            missing.append("sampling_metrics.per_seed")
        state_mae = {}
        for name in self.run.channel_names:
            key = f"val_{name}_mae"
            if key in row:
                state_mae[name] = float(row[key])
            else:
                state_mae[name] = None
                missing.append(key)
        norms = np.asarray(self._norms, dtype=np.float64)
        record = {
            "kind": "health",
            "run_id": self._run_id(),
            "epoch_index": int(epoch_index),
            "epoch_no": int(epoch_index) + 1,
            "phase": "epoch",
            "learning_rate": row.get("learning_rate"),
            "train_loss": row.get("train_loss"),
            "loss_components": {
                key: row.get(key)
                for key in row
                if key.startswith("train_") and key not in {"train_loss", "train_batches"}
            },
            "grad_norm_mean": None if norms.size == 0 else float(norms.mean()),
            "grad_norm_max": None if norms.size == 0 else float(norms.max()),
            "grad_norm_over_clip_fraction": None if norms.size == 0 else self._over_clip / max(len(self._norms), 1),
            "optimizer_steps_skipped": self._skipped,
            "clip_grad_norm": self.run.config.get("clip_grad_norm"),
            "train_seconds": float(train_seconds),
            "val_seconds": float(val_seconds),
            "train_samples": int(train_samples),
            "train_windows_per_sec": None if train_seconds <= 0 else float(train_samples) / float(train_seconds),
            "val_sample_count_per_seed": val_metrics.get("sample_count"),
            "val_windows_per_sec": None
            if val_seconds <= 0 or val_metrics.get("sample_count") is None
            else float(val_metrics["sample_count"]) * max(len(per_seed), 1) / float(val_seconds),
            "selection_value": row.get("selection_value"),
            "val_state_mae": state_mae,
            "val_inflow_mae_per_seed": per_seed,
            "rss_mb": _rss_mb(),
            "monitor_seconds_cumulative": self.monitor_seconds,
            "missing": missing,
        }
        from reimplementation.common.experiment_protocol import peak_memory_mb

        record.update(peak_memory_mb(self._device()))
        record["gpu_peak_scope"] = "since train start; not reset per epoch, so official history peaks stay comparable"
        if self._device().type == "cuda":
            record["gpu_allocated_mb"] = float(torch.cuda.memory_allocated(self._device())) / (1024.0 * 1024.0)
            record["gpu_reserved_mb"] = float(torch.cuda.memory_reserved(self._device())) / (1024.0 * 1024.0)
        else:
            record["gpu_allocated_mb"] = None
            record["gpu_reserved_mb"] = None
            missing.append("gpu_memory")
        if norms.size == 0:
            missing.append("grad_norm")
        record["missing"] = missing
        _append_jsonl(self.directory / "health.jsonl", record)
        self.health_epochs.append(int(epoch_index))

    def _write_module_stats(self, epoch_index: int, activation: Mapping[str, Any]) -> None:
        base = {
            "kind": "module",
            "run_id": self._run_id(),
            "epoch_index": int(epoch_index),
            "epoch_no": int(epoch_index) + 1,
            "phase": "epoch",
        }
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in self._pending or []:
            groups.setdefault(record["module"], []).append(record)
            _append_jsonl(self.directory / "module_stats.jsonl", {**base, **record})
        for module, records in groups.items():
            relatives = [float(item["relative_update"]) for item in records]
            _append_jsonl(
                self.directory / "module_stats.jsonl",
                {
                    **base,
                    "module": module,
                    "parameter": None,
                    "source": "train_step_rollup",
                    "element_count": int(sum(item["element_count"] for item in records)),
                    "parameter_count": len(records),
                    "relative_update_mean": float(np.mean(relatives)),
                    "grad_missing_count": int(sum(bool(item["grad_missing"]) for item in records)),
                    "zero_denominator_count": int(sum(bool(item["zero_denominator"]) for item in records)),
                },
            )
        for module, stats in activation.get("activations", {}).items():
            _append_jsonl(
                self.directory / "module_stats.jsonl",
                {
                    **base,
                    "module": module,
                    "source": "probe_eval",
                    "sample_count": stats.get("probe_count"),
                    "activation": stats,
                },
            )
        arrays = {key: value for key, value in self._channel.items()}
        for name, values in activation.get("channels", {}).items():
            arrays[f"actvar__{name.replace('.', '_')}"] = values
        for name, values in self._channel.items():
            arrays[f"gradrms__{name.replace('.', '_')}"] = values
        if arrays:
            epoch_no = int(epoch_index) + 1
            atomic_write_npz(self.directory / "channels" / f"epoch_{epoch_no:03d}.npz", **arrays)
        self.module_epochs.append(int(epoch_index))

    def _probe_batch(self, probe: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        inline = probe.get("batch")
        if inline is not None:
            return {key: value.clone() if torch.is_tensor(value) else value for key, value in inline.items()}
        split = str(probe["split"])
        if split == "train":
            item = self.run.datasets["train"][int(probe["sample_index"])]
            batch = {}
            for key, value in item.items():
                tensor = value if torch.is_tensor(value) else torch.tensor(value)
                if key in {"day_index", "window_start_slot", "target_slot", "sample_index"}:
                    batch[key] = tensor.reshape(1).long()
                else:
                    batch[key] = tensor.unsqueeze(0)
            batch["sample_index"] = torch.tensor([int(probe["sample_index"])], dtype=torch.long)
            return batch
        root = self._val_root
        date = str(probe["date"])
        local = int(probe["local_index"])
        incoming = root / self.run.rate / split / date / f"{self.run.task.input_type}.npz"
        outgoing = root / self.run.rate / split / date / f"{self.run.task.target_type}.npz"
        with np.load(incoming, mmap_mode="r") as src, np.load(outgoing, mmap_mode="r") as tgt:
            return {
                "x": torch.from_numpy(np.array(src["x"][local], dtype=np.float32, copy=True)).unsqueeze(0),
                "x_raw": torch.from_numpy(np.array(src["x_raw"][local], dtype=np.float32, copy=True)).unsqueeze(0),
                "y": torch.from_numpy(np.array(tgt["y"][local], dtype=np.float32, copy=True)).unsqueeze(0),
                "y_raw": torch.from_numpy(np.array(tgt["y_raw"][local], dtype=np.float32, copy=True)).unsqueeze(0),
                "day_index": torch.tensor([int(tgt["day_index"][local])], dtype=torch.long),
                "window_start_slot": torch.tensor([int(tgt["window_start_slot"][local])], dtype=torch.long),
                "target_slot": torch.tensor([int(tgt["target_slot"][local])], dtype=torch.long),
                "sample_index": torch.tensor([int(probe["sample_index"])], dtype=torch.long),
            }

    def _install_hooks(self) -> None:
        self._remove_hooks()
        self._hook_stats = {}
        self._hook_channels = {}

        def make(name):
            def hook(_module, _inputs, output):
                if not self._accept_hooks or not torch.is_tensor(output):
                    return
                self._hook_stats.setdefault(name, []).append(_tensor_summary(output))
                variance = _channel_variance(output)
                if variance is not None:
                    self._hook_channels.setdefault(name, []).append(variance)

            return hook

        for name, module in self.model.named_modules():
            tracked = _logical_module(name)
            if tracked == name or name in {
                "embedding.auxiliary_gate",
                "blocks.0.temporal",
                "blocks.0.spatial",
                "blocks.0.ffn",
                "head",
                "projection",
            }:
                if isinstance(module, (nn.Linear, nn.Sequential, AuxiliaryStateGate)) or name in {
                    "blocks.0.temporal",
                    "blocks.0.spatial",
                    "blocks.0.ffn",
                    "head",
                    "projection",
                    "embedding.auxiliary_gate",
                }:
                    self._hooks.append(module.register_forward_hook(make(name)))

    def _remove_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks = []

    def _probe_pass(self, epoch_index: int | None, phase: str) -> dict[str, Any]:
        started = self._clock()
        was_training = self.model.training
        self._install_hooks()
        details = []
        try:
            self.model.eval()
            probes = [(probe, "validation") for probe in self.probes.get("validation", [])]
            probes.extend((probe, "train") for probe in self.probes.get("train", []))
            with torch.no_grad():
                for probe, _split in probes:
                    details.append(self._one_probe(probe))
        finally:
            self._remove_hooks()
            self.model.train(was_training)
        activations = {}
        channels = {}
        for name, rows in self._hook_stats.items():
            module = _logical_module(name)
            usable = [row for row in rows if not row.get("missing") and row.get("mean") is not None]
            activations[module] = {
                "probe_count": len(rows),
                "missing": not bool(usable),
                "mean": None if not usable else float(np.mean([row["mean"] for row in usable])),
                "rms": None if not usable else float(np.mean([row["rms"] for row in usable])),
                "variance": None if not usable else float(np.mean([row["variance"] for row in usable])),
                "near_zero_fraction": None if not usable else float(np.mean([row["near_zero_fraction"] for row in usable])),
                "p05": None if not usable else float(np.mean([row["p05"] for row in usable])),
                "p50": None if not usable else float(np.mean([row["p50"] for row in usable])),
                "p95": None if not usable else float(np.mean([row["p95"] for row in usable])),
            }
        for name, arrays in self._hook_channels.items():
            channels[name] = np.mean(np.stack(arrays, axis=0), axis=0).astype(np.float32)
        self.monitor_seconds += self._clock() - started
        return {"phase": phase, "epoch_index": epoch_index, "probes": details, "activations": activations, "channels": channels}

    def _one_probe(self, probe: Mapping[str, Any], *, return_arrays: bool = False) -> dict[str, Any]:
        batch = self._probe_batch(probe)
        device = self._device()
        moved = {key: value.to(device) for key, value in batch.items()}
        tod, dow = time_features(moved, self.model.config.input_window, steps_per_day=self.model.config.steps_per_day)
        self._accept_hooks = True
        try:
            hidden = self.model._embed_shared(moved["x"], tod, dow)
            block = self.model.blocks[0]
            temporal_in = block.temporal_norm(hidden)
            temporal_out = block.temporal(temporal_in)
            hidden = hidden + block.dropout(temporal_out)
            spatial_in = block.spatial_norm(hidden)
            spatial_out = block.spatial(spatial_in)
            hidden = hidden + block.dropout(spatial_out)
            ffn_in = block.ffn_norm(hidden)
            ffn_out = block.ffn(ffn_in)
            hidden = hidden + block.dropout(ffn_out)
            raw = self.model.head(hidden)
            outputs = {"raw_pred": raw, "final_pred": raw}
            if self.model.projection is not None:
                outputs = self.model.projection(hidden, raw)
        finally:
            self._accept_hooks = False
        detail = {
            "probe_id": probe["probe_id"],
            "split": probe["split"],
            "sampling_seed": probe["sampling_seed"],
            "date": probe["date"],
            "sample_index": probe["sample_index"],
            "target_slot": probe["target_slot"],
            "temporal": _temporal_attention(block.temporal, temporal_in, self.road_index),
            "spatial": _spatial_attention(block.spatial, spatial_in, self.road_index, QUERY_TIMES),
            "gate": self._gate_report(moved, outputs),
            "projection": self._projection_report(moved, outputs),
            "response": self._response_report(ffn_in, moved),
        }
        if return_arrays:
            final = self.run.criterion.denormalize(outputs["final_pred"]).detach()
            raw_physical = self.run.criterion.denormalize(outputs["raw_pred"]).detach()
            observed = moved["x_raw"][:, -1].detach()
            gate_mean = None
            gate = getattr(self.model.embedding, "auxiliary_gate", None)
            if gate is not None and int(moved["x"].shape[-1]) >= 6:
                gate_mean = gate(moved["x"][:, -1, :, 4:6]).mean(dim=-1)[0].detach().float().cpu().numpy()
            detail["_arrays"] = {
                "pred": final[0].float().cpu().numpy(),
                "true": moved["y_raw"][:, -1].detach().float().cpu().numpy(),
                "correction": (final - raw_physical)[0].float().cpu().numpy(),
                "count": None if observed.shape[-1] < 6 else observed[0, :, 5].float().cpu().numpy(),
                "speed": None if observed.shape[-1] < 6 else observed[0, :, 4].float().cpu().numpy(),
                "gate_mean": gate_mean,
            }
        return detail

    def _gate_report(self, batch, outputs) -> dict[str, Any]:
        gate = getattr(self.model.embedding, "auxiliary_gate", None)
        if gate is None or batch["x"].shape[-1] < 6:
            return {"applicable": False, "reason": "no auxiliary gate"}
        aux = batch["x"][..., 4:6]
        values = gate(aux)
        last = values[:, -1]
        summary = _tensor_summary(last)
        groups = {}
        for index, name in enumerate(STATE_CHANNELS):
            groups[name] = _tensor_summary(last[..., index * 8 : (index + 1) * 8])
        raw = batch["x_raw"][:, -1].detach().float()
        count = raw[..., 5]
        speed = raw[..., 4]
        observed = count > 0
        final = self.run.criterion.denormalize(outputs["final_pred"])
        truth = batch["y_raw"][:, -1].float()
        error = (final - truth).abs()
        return {
            "applicable": True,
            "formula": "g=2*sigmoid(logit)",
            "summary": summary,
            "groups_of_8": groups,
            "abs_deviation_mean": float((last - 1).abs().mean()),
            "fraction_below_0.05": float((last < GATE_LOW).float().mean()),
            "fraction_above_1.95": float((last > GATE_HIGH).float().mean()),
            "count_0": {
                "road_slots": int((~observed).sum()),
                "gate": _tensor_summary(last[~observed]) if int((~observed).sum()) else {"missing": True, "count": 0},
                "speed_note": "speed at count 0 is a fill value and is not summarized as congestion",
                "error_mae": None if int((~observed).sum()) == 0 else float(error[~observed].mean()),
            },
            "count_positive": {
                "road_slots": int(observed.sum()),
                "gate": _tensor_summary(last[observed]) if int(observed.sum()) else {"missing": True, "count": 0},
                "speed_kmh": _tensor_summary(speed[observed]) if int(observed.sum()) else {"missing": True, "count": 0},
                "error_mae": None if int(observed.sum()) == 0 else float(error[observed].mean()),
            },
            "note": SIGMOID_NOTE,
        }

    def _projection_report(self, batch, outputs) -> dict[str, Any]:
        if self.model.projection is None or "correction_weights" not in outputs:
            return {"applicable": False, "reason": "conservation projection is not in this model"}
        raw = self.run.criterion.denormalize(outputs["raw_pred"])
        final = self.run.criterion.denormalize(outputs["final_pred"])
        truth = batch["y_raw"][:, -1].float()
        raw_error = (raw - truth).abs()
        final_error = (final - truth).abs()
        raw_residual = conservation_residual(raw).abs()
        final_residual = conservation_residual(final).abs()
        weights = outputs["correction_weights"].detach().float()
        per_state = {}
        for index, name in enumerate(STATE_CHANNELS):
            per_state[name] = {
                "final_minus_raw_mae": float((final[..., index] - raw[..., index]).abs().mean()),
                "error_change_mae": float((final_error[..., index] - raw_error[..., index]).mean()),
                "negative_fraction_raw": float((raw[..., index] < 0).float().mean()),
                "negative_fraction_final": float((final[..., index] < 0).float().mean()),
                "weight": _tensor_summary(weights[..., index]),
            }
        return {
            "applicable": True,
            "units": "physical vehicle counts",
            "raw_residual_mean_abs": float(raw_residual.mean()),
            "raw_residual_max_abs": float(raw_residual.max()),
            "final_residual_mean_abs": float(final_residual.mean()),
            "final_residual_max_abs": float(final_residual.max()),
            "per_state": per_state,
            "field_note": "saved pred_raw is final physical; raw_pred_physical is the pre-projection physical head output",
        }

    def _response_report(self, ffn_in: torch.Tensor, batch) -> dict[str, Any]:
        ffn = self.model.blocks[0].ffn
        pre = ffn.w_gate(ffn_in)
        sigmoid = torch.sigmoid(pre)
        silu_derivative = sigmoid * (1.0 + pre * (1.0 - sigmoid))
        report = {
            "silu_derivative_p50": float(torch.quantile(silu_derivative.detach().float().cpu().reshape(-1), 0.5)),
            "fraction_silu_derivative_lt_0.01": float((silu_derivative < 0.01).float().mean()),
            "note": SILU_DERIVATIVE_NOTE,
        }
        gate = getattr(self.model.embedding, "auxiliary_gate", None)
        if gate is not None and batch["x"].shape[-1] >= 6:
            logits = gate.last(gate.activation(gate.first(batch["x"][..., 4:6])))
            gate_sigmoid = torch.sigmoid(logits)
            derivative = gate_sigmoid * (1.0 - gate_sigmoid)
            report["gate_sigmoid_derivative_mean"] = float(derivative.mean())
            report["gate_note"] = SIGMOID_NOTE
        return report

    def _mechanisms(self, epoch_index: int | None, phase: str, cached: Mapping[str, Any] | None = None) -> None:
        started = self._clock()
        details = list(cached.get("probes", [])) if cached is not None else []
        if not details:
            was_training = self.model.training
            try:
                self.model.eval()
                with torch.no_grad():
                    for probe in self.probes.get("validation", []) + self.probes.get("train", []):
                        details.append(self._one_probe(probe))
            finally:
                self.model.train(was_training)
        train_probe = self.probes.get("train", [None])[0]
        gradient = {"applicable": False, "reason": "no training probe"}
        if train_probe is not None and phase != "init":
            gradient = self._task_gradient(train_probe)
        record = {
            "kind": "mechanism",
            "run_id": self._run_id(),
            "epoch_index": None if epoch_index is None else int(epoch_index),
            "epoch_no": None if epoch_index is None else int(epoch_index) + 1,
            "phase": phase,
            "source": "fixed_probe_eval",
            "probes": details,
            "task_gradient": gradient,
        }
        _append_jsonl(self.directory / "mechanisms.jsonl", record)
        if epoch_index is not None:
            self.mechanism_epochs_done.append(int(epoch_index))
        self.monitor_seconds += self._clock() - started

    def _task_gradient(self, probe: Mapping[str, Any]) -> dict[str, Any]:
        name = "blocks.0.ffn.w_out.weight"
        params = dict(self.model.named_parameters())
        if name not in params:
            return {"applicable": False, "reason": f"{name} is not in this model"}
        param = params[name]
        witness = None if param.grad is None else param.grad.detach().flatten()[:8].cpu().clone()
        was_training = self.model.training
        batch = self._probe_batch(probe)
        device = self._device()
        moved = {key: value.to(device) for key, value in batch.items()}
        coefficient = float(self.run.config.get("projection_raw_loss_weight", 0.5))
        try:
            self.model.eval()
            with torch.enable_grad():
                tod, dow = time_features(moved, self.model.config.input_window, steps_per_day=self.model.config.steps_per_day)
                outputs = self.model.forward_with_aux(moved["x"], tod, dow)
                final = self.run.criterion.denormalize(outputs["final_pred"])
                raw = self.run.criterion.denormalize(outputs["raw_pred"])
                truth = moved["y_raw"][:, -1].float()
                grads = []
                losses = []
                for index in range(4):
                    task_loss = (final[..., index] - truth[..., index]).abs().mean() + coefficient * (
                        raw[..., index] - truth[..., index]
                    ).abs().mean()
                    losses.append(float(task_loss.detach()))
                    grad = torch.autograd.grad(task_loss, param, retain_graph=index < 3, allow_unused=True)[0]
                    grads.append(None if grad is None else grad.detach().float().reshape(-1).cpu())
                    del grad, task_loss
                del outputs, final, raw
        finally:
            self.model.train(was_training)
        after = None if param.grad is None else param.grad.detach().flatten()[:8].cpu()
        if witness is None:
            unchanged = after is None
        else:
            unchanged = after is not None and torch.equal(witness, after)
        if not unchanged:
            raise RuntimeError("diagnostic task gradient wrote into parameter.grad")
        norms = [None if grad is None else float(grad.norm()) for grad in grads]
        pairs = []
        nulls = 0
        for left in range(4):
            for right in range(left + 1, 4):
                a, b = grads[left], grads[right]
                if a is None or b is None or float(a.norm()) == 0.0 or float(b.norm()) == 0.0:
                    pairs.append({"states": [STATE_CHANNELS[left], STATE_CHANNELS[right]], "cosine": None})
                    nulls += 1
                else:
                    pairs.append(
                        {
                            "states": [STATE_CHANNELS[left], STATE_CHANNELS[right]],
                            "cosine": float(torch.dot(a, b) / (a.norm() * b.norm())),
                        }
                    )
        return {
            "applicable": True,
            "parameter": name,
            "probe_id": probe["probe_id"],
            "coefficient": coefficient,
            "units": "physical MAE",
            "task_losses": {STATE_CHANNELS[index]: losses[index] for index in range(4)},
            "task_loss_mean": float(np.mean(losses)),
            "grad_norms": {STATE_CHANNELS[index]: norms[index] for index in range(4)},
            "cosines": pairs,
            "null_cosine_count": nulls,
            "note": "one fixed training probe and one shared parameter; gradients were not written back",
        }

    def _write_associations(self, model: nn.Module) -> None:
        started = self._clock()
        directory = self.directory / "associations"
        directory.mkdir(parents=True, exist_ok=True)
        full = {}
        for split in ("validation", "test"):
            per_seed = {}
            for seed in self.run.config.get("evaluation_sampling_seeds", [42, 43, 44, 45, 46]):
                path = Path(self.run.run_dir) / "predictions" / f"seed_{int(seed)}" / f"{split}.npz"
                per_seed[int(seed)] = self._associate_seed(int(seed), split, path)
            averaged = _average_seed_tables(per_seed)
            averaged.update(
                {
                    "scope": "full_predictions",
                    "split": split,
                    "dataset_id": self.run.config.get("dataset_id"),
                    "note": "gate and projection corrections are not in the official prediction archive",
                }
            )
            atomic_write_json(directory / f"{split}_full.json", _py(averaged))
            full[split] = {"n_seeds": averaged["n_seeds_expected"], "tables": list(averaged["tables"])}
        probe_report = self._replay_probe_associations(model)
        atomic_write_json(directory / "probe_replay.json", _py(probe_report))
        atomic_write_json(
            directory / "summary.json",
            _py(
                {
                    "full_predictions": full,
                    "probe_replay": {
                        "scope": "fixed subsample replay of the best checkpoint",
                        "validation_sampling_seed": VAL_PROBE_SEED,
                        "test_sampling_seed": VAL_PROBE_SEED,
                        "probe_count": probe_report.get("probe_count"),
                    },
                    "comparison_note": (
                        "Full-prediction tables pool every official window. "
                        "Probe tables use the fixed subsample only and add gate and projection statistics."
                    ),
                }
            ),
        )
        self.monitor_seconds += self._clock() - started

    def _associate_seed(self, seed: int, split: str, path: Path) -> dict[str, dict[str, dict[str, Any]]]:
        from data_root import sampling_root_from_window, sampling_window_root

        packed = _load_prediction(path)
        pred = _squeeze_prediction(packed["pred_raw"])
        true = _squeeze_prediction(packed["y_raw"])
        sample_index = np.asarray(packed["sample_index"]).reshape(-1).astype(np.int64)
        target_slot = np.asarray(packed["target_slot"]).reshape(-1).astype(np.int64)
        position = {int(sample): pos for pos, sample in enumerate(sample_index.tolist())}
        if len(position) != int(sample_index.shape[0]):
            raise RuntimeError(f"{split} seed {seed} prediction sample_index values are not unique")
        window = sampling_window_root(sampling_root_from_window(self.run.data_root), seed)
        spans = _day_spans(window, self.run.rate, split, self.run.task.target_type)
        tables: dict[str, dict[str, _Bucket]] = {}
        seen = 0
        for span in spans:
            incoming = window / self.run.rate / split / span["date"] / f"{self.run.task.input_type}.npz"
            with np.load(incoming, mmap_mode="r") as payload:
                observed = (
                    np.asarray(payload["x_raw"][:, -1, :, 4:6], dtype=np.float32)
                    if int(payload["x_raw"].shape[-1]) >= 6
                    else None
                )
            global_index = span["start"] + np.arange(span["count"], dtype=np.int64)
            try:
                rows = np.asarray([position[int(item)] for item in global_index.tolist()], dtype=np.int64)
            except KeyError as exc:
                raise RuntimeError(f"{split} seed {seed} {span['date']} sample is missing from predictions") from exc
            if not np.array_equal(target_slot[rows], span["target_slot"]):
                raise RuntimeError(f"{split} seed {seed} {span['date']} target_slot does not match predictions")
            pred_day = pred[rows]
            true_day = true[rows]
            nodes = int(pred_day.shape[1])
            window_ids = np.repeat(global_index[:, None], nodes, axis=1)
            road_names = np.asarray(
                [self.node_ids[index] if index < len(self.node_ids) else str(index) for index in range(nodes)],
                dtype=object,
            )
            groups = {
                "by_date": np.full((span["count"], nodes), span["date"], dtype=object),
                "by_hour": np.broadcast_to((span["target_slot"] // 12).astype(str)[:, None], (span["count"], nodes)),
                "by_road": np.broadcast_to(road_names, (span["count"], nodes)),
            }
            if observed is not None:
                groups["by_count"] = _count_key(observed[..., 1])
                groups["by_speed"] = _speed_key(observed[..., 0], observed[..., 1])
            _accumulate_into(tables, pred_day, true_day, groups, window_ids)
            seen += span["count"]
            del observed, pred_day, true_day, groups
        if seen != int(pred.shape[0]):
            raise RuntimeError(f"{split} seed {seed} day windows {seen} != predictions {pred.shape[0]}")
        return _export_tables(tables)

    def _replay_probe_associations(self, model: nn.Module) -> dict[str, Any]:
        from data_root import sampling_root_from_window, sampling_window_root

        test_root = sampling_window_root(sampling_root_from_window(self.run.data_root), VAL_PROBE_SEED)
        test_spans = _day_spans(test_root, self.run.rate, "test", self.run.task.target_type)
        test_probes = _choose_probes(test_spans, self.probe_size, split="test", sampling_seed=VAL_PROBE_SEED)
        selected = list(self.probes.get("validation", [])) + test_probes
        was_training = model.training
        rows = []
        tables: dict[str, dict[str, _Bucket]] = {}
        try:
            model.eval()
            with torch.no_grad():
                for probe in selected:
                    saved_root = getattr(self, "_val_root", None)
                    if probe["split"] == "test":
                        self._val_root = test_root
                    try:
                        detail = self._one_probe(probe, return_arrays=True)
                    finally:
                        if probe["split"] == "test":
                            self._val_root = saved_root
                    arrays = detail.pop("_arrays")
                    rows.append(
                        {
                            "probe": {key: probe[key] for key in ("probe_id", "split", "sampling_seed", "date", "sample_index", "target_slot")},
                            "gate": detail["gate"],
                            "projection": detail["projection"],
                        }
                    )
                    pred = arrays["pred"]
                    nodes = int(pred.shape[0])
                    window_ids = np.full(nodes, int(probe["sample_index"]), dtype=np.int64)
                    road_names = np.asarray(
                        [self.node_ids[index] if index < len(self.node_ids) else str(index) for index in range(nodes)],
                        dtype=object,
                    )
                    groups = {
                        "by_date": np.full(nodes, probe["date"], dtype=object),
                        "by_hour": np.full(nodes, str(int(probe["target_slot"]) // 12), dtype=object),
                        "by_road": road_names,
                    }
                    if arrays["count"] is not None:
                        groups["by_count"] = _count_key(arrays["count"])
                        groups["by_speed"] = _speed_key(arrays["speed"], arrays["count"])
                    _accumulate_into(
                        tables,
                        pred,
                        arrays["true"],
                        groups,
                        window_ids,
                        arrays["gate_mean"],
                        arrays["correction"],
                    )
        finally:
            model.train(was_training)
        return {
            "scope": "probe_subsample",
            "probe_count": len(rows),
            "sampling_seed": VAL_PROBE_SEED,
            "tables": _export_tables(tables),
            "records": rows,
            "note": "fixed validation and test subsample of sampling seed 42; not the full-prediction aggregate",
        }

    def _summary(self, status: str) -> None:
        if not self.active():
            return
        payload = {
            "status": status,
            "run_id": self._run_id(),
            "monitor_level": self.level,
            "health_epochs": self.health_epochs,
            "module_epochs": self.module_epochs,
            "mechanism_epoch_indexes": self.mechanism_epochs_done,
            "associations_written": self.associations_written,
            "monitor_seconds": self.monitor_seconds,
            "rss_mb": _rss_mb(),
            "anomalies": self.anomalies,
            "outputs": {
                "manifest": (self.directory / "manifest.json").is_file(),
                "health": (self.directory / "health.jsonl").is_file(),
                "module_stats": (self.directory / "module_stats.jsonl").is_file(),
                "mechanisms": (self.directory / "mechanisms.jsonl").is_file(),
                "associations": (self.directory / "associations" / "summary.json").is_file(),
            },
        }
        from reimplementation.common.experiment_protocol import peak_memory_mb

        payload.update(peak_memory_mb(self._device()))
        atomic_write_json(self.directory / "summary.json", _py(payload))


def replay_level4(run_dir: Path, *, device: str | None = None) -> dict[str, Any]:
    """Redo level-4 associations from an existing best checkpoint and predictions."""
    run_dir = Path(run_dir)
    predictions = list((run_dir / "predictions").glob("seed_*/validation.npz"))
    if not predictions or not (run_dir / "best_checkpoint.pt").is_file():
        raise FileNotFoundError(f"{run_dir} is missing best_checkpoint.pt or validation predictions")
    from MSTT.config import load_config, restore_saved_model_fields, with_legacy_defaults
    from MSTT.trainer import TaskRun, assert_model_checkpoint_matches
    from reimplementation.common.utils.checkpoint import load_checkpoint
    from data_root import parse_canonical_run

    raw = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    saved = with_legacy_defaults(raw)
    config = load_config(None)
    config.update({key: saved[key] for key in config if key in saved})
    restore_saved_model_fields(config, raw)
    if device is not None:
        config["device"] = device
    identity = parse_canonical_run(run_dir)
    config.update({key: value for key, value in identity.items() if value})
    output_root = Path(identity.get("results_root") or run_dir.parents[2])
    run = TaskRun(config, task=saved["task"], rate=saved["penetration"], output_root=output_root, splits=("train", "validation"))
    run.run_dir = run_dir
    checkpoint = load_checkpoint(run_dir / "best_checkpoint.pt", map_location=run.device)
    assert_model_checkpoint_matches(checkpoint, run.model_config.to_dict())
    model = run.build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    monitor = RunMonitor(run, model, train_data_sha256=checkpoint.get("data_file_sha256"))
    monitor.level = max(monitor.level, 4)
    monitor.prepare(snapshot_init=False)
    monitor.on_predictions_ready(model)
    monitor.finish()
    return json.loads((monitor.directory / "summary.json").read_text(encoding="utf-8"))
