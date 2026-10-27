#!/usr/bin/env python3
"""Concatenate Xuancheng26 daily window NPZ files into split-level datasets.

This step only stacks samples that were already cut. It does not recut
windows, refit normalization, shuffle, or write model graphs.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import date, datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

WINDOW_ROOT = Path(
    "/data5/tzhang174/TrafficFlow/xuancheng26/training_data/window_data"
)
GRAPH_PATH = Path(
    "/data5/tzhang174/TrafficFlow/xuancheng26/training_data/graph/r_graph.npz"
)
RATES = ("p05", "p10", "p20", "p30", "p40", "p50", "p70")
SPLITS = ("train", "validation", "test")
NPZ_KEYS = (
    "x",
    "y",
    "x_raw",
    "y_raw",
    "day_index",
    "window_start_slot",
    "target_slot",
)
VALUE_KEYS = ("x", "y", "x_raw", "y_raw")
SAMPLES_PER_DAY = 277
N_HIS = 12
N_NODES = 1406
EXPECTED_COUNTS = {"train": 5540, "validation": 1108, "test": 1385}
EXPECTED_DAYS = {
    "train": (
        "2023_04_01", "2023_04_02", "2023_04_03", "2023_04_04", "2023_04_05",
        "2023_04_06", "2023_04_07", "2023_04_08", "2023_04_10", "2023_04_11",
        "2023_04_12", "2023_04_13", "2023_04_14", "2023_04_15", "2023_04_16",
        "2023_04_17", "2023_04_18", "2023_04_19", "2023_04_20", "2023_04_21",
    ),
    "validation": ("2023_04_22", "2023_04_23", "2023_04_24", "2023_04_25"),
    "test": ("2023_04_26", "2023_04_27", "2023_04_28", "2023_04_29", "2023_04_30"),
}
WEEKDAY_SPOTS = {
    "2023_04_22": 5,
    "2023_04_23": 6,
    "2023_04_29": 5,
    "2023_04_30": 6,
}
DENORM_ATOL = 1e-3


def log(message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"{stamp} {message}", flush=True)


def weekday_index(day_name: str) -> int:
    year, month, day = (int(part) for part in day_name.split("_"))
    return int(date(year, month, day).weekday())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Uncompressed np.savez, matching the old prepared-data writer."""
    temporary = path.with_name(f".{path.name}.tmp.npz")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def check_numeric(path: Path, name: str, array: np.ndarray) -> None:
    if array.dtype.hasobject or array.dtype.kind in {"O", "U", "S"}:
        raise RuntimeError(f"{path} {name} dtype {array.dtype} is not numeric")
    if name in VALUE_KEYS and not np.isfinite(array).all():
        raise RuntimeError(f"{path} {name} contains NaN or inf")


def load_manifest() -> dict[str, tuple[str, ...]]:
    manifest = load_json(WINDOW_ROOT / "split_manifest.json")
    loaded: dict[str, tuple[str, ...]] = {}
    for split in SPLITS:
        days = tuple(str(item) for item in manifest[split])
        if days != EXPECTED_DAYS[split]:
            raise RuntimeError(f"split_manifest {split} dates differ from the frozen split")
        if len(days) * SAMPLES_PER_DAY != EXPECTED_COUNTS[split]:
            raise RuntimeError(f"{split} day count does not produce {EXPECTED_COUNTS[split]} samples")
        loaded[split] = days
    return loaded


def check_graph_node_axis() -> int:
    with np.load(GRAPH_PATH, allow_pickle=False) as payload:
        edge_ids = payload["edge_ids"]
        num_nodes = int(np.asarray(payload["num_nodes"]).reshape(-1)[0])
    if num_nodes != N_NODES or len(edge_ids) != N_NODES:
        raise RuntimeError(f"r_graph.npz has {num_nodes} nodes / {len(edge_ids)} edge ids")
    probe = WINDOW_ROOT / "p05" / "train" / f"{EXPECTED_DAYS['train'][0]}.npz"
    with np.load(probe, allow_pickle=False) as payload:
        node_axis = int(payload["x"].shape[2])
    if node_axis != N_NODES:
        raise RuntimeError(f"{probe} node axis is {node_axis}, graph has {N_NODES}")
    return num_nodes


def merge_split(rate: str, split: str, days: tuple[str, ...]) -> dict[str, object]:
    chunks: dict[str, list[np.ndarray]] = {key: [] for key in NPZ_KEYS}
    dtypes: dict[str, np.dtype] = {}
    for day_name in days:
        path = WINDOW_ROOT / rate / split / f"{day_name}.npz"
        if not path.is_file():
            raise RuntimeError(f"missing daily NPZ {path}")
        with np.load(path, allow_pickle=False) as payload:
            missing = [key for key in NPZ_KEYS if key not in payload.files]
            if missing:
                raise RuntimeError(f"{path} missing {missing}")
            arrays = {key: np.array(payload[key], copy=True) for key in NPZ_KEYS}
        for key, array in arrays.items():
            check_numeric(path, key, array)
            if key not in dtypes:
                dtypes[key] = array.dtype
            elif array.dtype != dtypes[key]:
                raise RuntimeError(f"{path} {key} dtype {array.dtype} != {dtypes[key]}")
            chunks[key].append(array)
        count = int(arrays["x"].shape[0])
        if count != SAMPLES_PER_DAY:
            raise RuntimeError(f"{path} has {count} samples, expected {SAMPLES_PER_DAY}")
        if arrays["x"].shape != (SAMPLES_PER_DAY, N_HIS, N_NODES, 1):
            raise RuntimeError(f"{path} x shape {arrays['x'].shape}")
        if arrays["y"].shape != (SAMPLES_PER_DAY, 1, N_NODES, 1):
            raise RuntimeError(f"{path} y shape {arrays['y'].shape}")
        weekday = weekday_index(day_name)
        if not np.array_equal(arrays["day_index"], np.full(SAMPLES_PER_DAY, weekday, dtype=arrays["day_index"].dtype)):
            raise RuntimeError(f"{path} day_index is not weekday {weekday}")
        starts = np.arange(SAMPLES_PER_DAY, dtype=arrays["window_start_slot"].dtype)
        if not np.array_equal(arrays["window_start_slot"], starts):
            raise RuntimeError(f"{path} window_start_slot is not 0..276")
        if not np.array_equal(arrays["target_slot"], starts + (N_HIS - 1)):
            raise RuntimeError(f"{path} target_slot is not 11..287")

    merged = {key: np.concatenate(chunks[key], axis=0) for key in NPZ_KEYS}
    expected = EXPECTED_COUNTS[split]
    if int(merged["x"].shape[0]) != expected:
        raise RuntimeError(f"{rate}/{split} has {merged['x'].shape[0]} samples, expected {expected}")
    if merged["x"].shape != (expected, N_HIS, N_NODES, 1) or merged["x_raw"].shape != merged["x"].shape:
        raise RuntimeError(f"{rate}/{split} x shape {merged['x'].shape}")
    if merged["y"].shape != (expected, 1, N_NODES, 1) or merged["y_raw"].shape != merged["y"].shape:
        raise RuntimeError(f"{rate}/{split} y shape {merged['y'].shape}")
    for key in ("day_index", "window_start_slot", "target_slot"):
        if merged[key].shape != (expected,):
            raise RuntimeError(f"{rate}/{split} {key} shape {merged[key].shape}")
    for day_offset, day_name in enumerate(days):
        block = slice(day_offset * SAMPLES_PER_DAY, (day_offset + 1) * SAMPLES_PER_DAY)
        weekday = weekday_index(day_name)
        if not np.array_equal(
            merged["day_index"][block],
            np.full(SAMPLES_PER_DAY, weekday, dtype=merged["day_index"].dtype),
        ):
            raise RuntimeError(f"{rate}/{split} lost weekday encoding at {day_name}")
        starts = np.arange(SAMPLES_PER_DAY, dtype=merged["window_start_slot"].dtype)
        if not np.array_equal(merged["window_start_slot"][block], starts):
            raise RuntimeError(f"{rate}/{split} slots were renumbered at {day_name}")
        if not np.array_equal(merged["target_slot"][block], starts + (N_HIS - 1)):
            raise RuntimeError(f"{rate}/{split} target slots were renumbered at {day_name}")
        if day_name in WEEKDAY_SPOTS and int(merged["day_index"][block][0]) != WEEKDAY_SPOTS[day_name]:
            raise RuntimeError(f"{rate}/{split} {day_name} weekday is not {WEEKDAY_SPOTS[day_name]}")

    destination = WINDOW_ROOT / rate / f"{split}.npz"
    save_npz(destination, merged)
    with np.load(destination, allow_pickle=False) as payload:
        reloaded = {key: np.array(payload[key], copy=False) for key in NPZ_KEYS}
    for key in NPZ_KEYS:
        if reloaded[key].shape != merged[key].shape or reloaded[key].dtype != merged[key].dtype:
            raise RuntimeError(f"{destination} reload mismatch for {key}")
        if not np.array_equal(reloaded[key], merged[key]):
            raise RuntimeError(f"{destination} reload values differ for {key}")
        check_numeric(destination, key, reloaded[key])
    log(
        f"[{rate}/{split}] samples={expected} x={tuple(reloaded['x'].shape)} "
        f"dtype={reloaded['x'].dtype}"
    )
    return {
        "samples": expected,
        "x_shape": list(reloaded["x"].shape),
        "y_shape": list(reloaded["y"].shape),
        "dtypes": {key: str(reloaded[key].dtype) for key in NPZ_KEYS},
    }


def read_rate_normalization() -> dict[str, dict]:
    payloads: dict[str, dict] = {}
    reference: tuple[float, float] | None = None
    for rate in RATES:
        payload = load_json(WINDOW_ROOT / rate / "normalization.json")
        for key in ("mean_x", "std_x", "mean_y_full", "std_y_full"):
            if key not in payload:
                raise RuntimeError(f"{rate} normalization.json missing {key}")
            value = float(payload[key])
            if not np.isfinite(value) or (key.startswith("std") and value <= 0.0):
                raise RuntimeError(f"{rate} normalization {key}={value}")
        pair = (float(payload["mean_y_full"]), float(payload["std_y_full"]))
        if reference is None:
            reference = pair
        elif pair != reference:
            raise RuntimeError(
                f"{rate} mean_y_full/std_y_full {pair} != shared {reference}"
            )
        payloads[rate] = payload
    return payloads


def patch_normalization(payloads: dict[str, dict]) -> dict[str, dict]:
    """Add the nested fields read_target_scaler and the old writer use.

    Existing flat statistics stay unchanged. Nothing is refit.
    """
    shared_mean = float(payloads["p05"]["mean_y_full"])
    shared_std = float(payloads["p05"]["std_y_full"])
    training_days = list(EXPECTED_DAYS["train"])
    shared_inputs: dict[str, dict[str, float]] = {}
    patched: dict[str, dict] = {}
    for rate, payload in payloads.items():
        updated = dict(payload)
        updated["target_scaler"] = {
            "mean_y_full": float(payload["mean_y_full"]),
            "std_y_full": float(payload["std_y_full"]),
        }
        updated["input_scalers"] = {
            rate: {
                "mean_x": float(payload["mean_x"]),
                "std_x": float(payload["std_x"]),
            }
        }
        updated["used_validation_to_fit"] = False
        updated["used_test_to_fit"] = False
        if float(updated["mean_x"]) != float(payload["mean_x"]) or float(updated["std_x"]) != float(payload["std_x"]):
            raise RuntimeError(f"{rate} input scaler changed while patching schema")
        if float(updated["mean_y_full"]) != shared_mean or float(updated["std_y_full"]) != shared_std:
            raise RuntimeError(f"{rate} target scaler changed while patching schema")
        write_json(WINDOW_ROOT / rate / "normalization.json", updated)
        check_loader_schema(updated, rate)
        patched[rate] = updated
        shared_inputs[rate] = {
            "mean_x": float(payload["mean_x"]),
            "std_x": float(payload["std_x"]),
        }
    shared = {
        "formula": "z = (x - mean) / std",
        "inverse_formula": "x = z * std + mean",
        "ddof": 0,
        "fit_split": "training_only",
        "global_or_per_node": "global_scalar",
        "per_day_normalization": False,
        "per_node_normalization": False,
        "validation_and_test_reuse_training_scaler": True,
        "used_validation_to_fit": False,
        "used_test_to_fit": False,
        "training_days": training_days,
        "target_scaler": {
            "mean_y_full": shared_mean,
            "std_y_full": shared_std,
        },
        "input_scalers": shared_inputs,
        "source": "copied from per-rate training-only statistics; not refit",
    }
    write_json(WINDOW_ROOT / "normalization.json", shared)
    check_loader_schema(shared, None)
    return patched


def check_loader_schema(payload: dict, rate: str | None) -> None:
    """Mirror reimplementation.common.data.r_only_npz_dataset.read_target_scaler."""
    scaler = payload.get("target_scaler") or {}
    if "mean_y_full" not in scaler or "std_y_full" not in scaler:
        raise RuntimeError("normalization.json is missing target_scaler.mean_y_full/std_y_full")
    mean = float(scaler["mean_y_full"])
    std = float(scaler["std_y_full"])
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0.0:
        raise RuntimeError(f"invalid target scaler mean={mean} std={std}")
    if bool(payload.get("used_test_to_fit")) or bool(payload.get("used_validation_to_fit")):
        raise RuntimeError("normalization.json was not fit on training only")
    inputs = payload.get("input_scalers") or {}
    if rate is None:
        expected_rates = RATES
    else:
        expected_rates = (rate,)
    for tag in expected_rates:
        entry = inputs.get(tag) or {}
        if "mean_x" not in entry or "std_x" not in entry:
            raise RuntimeError(f"input_scalers[{tag}] is missing mean_x/std_x")
        if float(entry["std_x"]) <= 0.0:
            raise RuntimeError(f"input_scalers[{tag}] std_x is not positive")


def max_abs_denorm_error(rate: str, split: str, normalization: dict) -> dict[str, float]:
    path = WINDOW_ROOT / rate / f"{split}.npz"
    with np.load(path, allow_pickle=False) as payload:
        x = np.asarray(payload["x"], dtype=np.float64)
        y = np.asarray(payload["y"], dtype=np.float64)
        x_raw = np.asarray(payload["x_raw"], dtype=np.float64)
        y_raw = np.asarray(payload["y_raw"], dtype=np.float64)
    mean_x = float(normalization["mean_x"])
    std_x = float(normalization["std_x"])
    mean_y = float(normalization["mean_y_full"])
    std_y = float(normalization["std_y_full"])
    x_error = float(np.max(np.abs(x - (x_raw - mean_x) / std_x)))
    y_error = float(np.max(np.abs(y - (y_raw - mean_y) / std_y)))
    if x_error > DENORM_ATOL or y_error > DENORM_ATOL:
        raise RuntimeError(f"{path} z-score mismatch x={x_error} y={y_error}")
    return {"x_max_abs_error": x_error, "y_max_abs_error": y_error}


def assert_shared_targets(split: str) -> None:
    reference_path = WINDOW_ROOT / "p05" / f"{split}.npz"
    with np.load(reference_path, allow_pickle=False) as payload:
        reference_y = np.array(payload["y_raw"], copy=True)
        reference_index = (
            np.array(payload["day_index"], copy=True),
            np.array(payload["window_start_slot"], copy=True),
            np.array(payload["target_slot"], copy=True),
        )
    for rate in RATES[1:]:
        path = WINDOW_ROOT / rate / f"{split}.npz"
        with np.load(path, allow_pickle=False) as payload:
            y_raw = payload["y_raw"]
            if y_raw.shape != reference_y.shape or not np.array_equal(y_raw, reference_y):
                raise RuntimeError(f"{path} y_raw differs from p05/{split}.npz")
            if not (
                np.array_equal(payload["day_index"], reference_index[0])
                and np.array_equal(payload["window_start_slot"], reference_index[1])
                and np.array_equal(payload["target_slot"], reference_index[2])
            ):
                raise RuntimeError(f"{path} sample index differs from p05/{split}.npz")
    log(f"[{split}] y_raw identical across {len(RATES)} rates")


def assert_raw_input_nesting(split: str) -> dict[str, int]:
    previous: np.ndarray | None = None
    previous_rate = ""
    violations: dict[str, int] = {}
    for rate in RATES:
        path = WINDOW_ROOT / rate / f"{split}.npz"
        with np.load(path, allow_pickle=False) as payload:
            current = np.array(payload["x_raw"], copy=True)
        if previous is not None:
            mismatch = int(np.count_nonzero(previous > current))
            violations[f"{previous_rate}<={rate}"] = mismatch
            if mismatch:
                raise RuntimeError(f"{split} x_raw nesting failed: {previous_rate} > {rate} at {mismatch} elements")
        previous = current
        previous_rate = rate
    log(f"[{split}] x_raw nesting passed")
    return violations


def add_manifest_loader_aliases() -> None:
    """Expose the frozen split under the key names the old loader reads.

    Date membership is not changed.
    """
    path = WINDOW_ROOT / "split_manifest.json"
    manifest = load_json(path)
    for split in SPLITS:
        if tuple(str(item) for item in manifest[split]) != EXPECTED_DAYS[split]:
            raise RuntimeError("refusing to annotate a split_manifest whose dates changed")
    manifest["training_days"] = list(EXPECTED_DAYS["train"])
    manifest["validation_days"] = list(EXPECTED_DAYS["validation"])
    manifest["test_days"] = list(EXPECTED_DAYS["test"])
    manifest["actual_sample_counts"] = dict(EXPECTED_COUNTS)
    manifest["theoretical_sample_counts"] = dict(EXPECTED_COUNTS)
    for split in SPLITS:
        if tuple(str(item) for item in manifest[split]) != EXPECTED_DAYS[split]:
            raise RuntimeError("split date list changed while adding loader aliases")
    write_json(path, manifest)


def count_daily_npz() -> int:
    return sum(1 for rate in RATES for split in SPLITS for _day in EXPECTED_DAYS[split]
               if (WINDOW_ROOT / rate / split / f"{_day}.npz").is_file())


def main() -> int:
    log("merge Xuancheng26 daily windows into split NPZ")
    node_count = check_graph_node_axis()
    days = load_manifest()
    daily_before = count_daily_npz()
    if daily_before != 29 * len(RATES):
        raise RuntimeError(f"expected 203 daily NPZ files, found {daily_before}")
    normalizations = read_rate_normalization()
    summary: dict[str, object] = {
        "node_count": node_count,
        "node_order": "unchanged; concatenated on the existing daily node axis",
        "cross_day_windows": False,
        "shuffled": False,
        "normalization_refit": False,
        "double_normalized": False,
        "daily_npz_retained": True,
        "rates": {},
        "failures": [],
    }
    for rate in RATES:
        rate_report: dict[str, object] = {"splits": {}, "status": "ok"}
        try:
            for split in SPLITS:
                rate_report["splits"][split] = merge_split(rate, split, days[split])
        except Exception:
            detail = traceback.format_exc(limit=8).strip()
            log(f"[{rate}] FAIL\n{detail}")
            rate_report["status"] = "fail"
            rate_report["error"] = detail
            summary["failures"].append({"rate": rate, "error": detail})
        summary["rates"][rate] = rate_report

    if summary["failures"]:
        summary["status"] = "fail"
        write_json(WINDOW_ROOT / "merge_summary.json", summary)
        log(f"FAILED rates: {[item['rate'] for item in summary['failures']]}")
        return 2

    cross: dict[str, object] = {}
    denorm: dict[str, dict[str, float]] = {}
    try:
        for split in SPLITS:
            assert_shared_targets(split)
            nesting = assert_raw_input_nesting(split)
            cross[split] = {"y_raw_identical": True, "x_raw_nesting_violations": nesting}
        for rate in RATES:
            denorm[rate] = {}
            for split in SPLITS:
                denorm[rate][split] = max_abs_denorm_error(rate, split, normalizations[rate])
            log(f"[{rate}] z-score check passed")
        patched = patch_normalization(normalizations)
        add_manifest_loader_aliases()
    except Exception:
        detail = traceback.format_exc(limit=8).strip()
        log(f"SANITY FAIL\n{detail}")
        summary["status"] = "fail"
        summary["failures"].append({"rate": "cross-check", "error": detail})
        write_json(WINDOW_ROOT / "merge_summary.json", summary)
        return 2

    daily_after = count_daily_npz()
    split_files = [
        str(WINDOW_ROOT / rate / f"{split}.npz")
        for rate in RATES
        for split in SPLITS
        if (WINDOW_ROOT / rate / f"{split}.npz").is_file()
    ]
    if daily_after != daily_before or len(split_files) != 21:
        raise RuntimeError(f"file count drift daily={daily_after} split={len(split_files)}")

    for rate in RATES:
        payload = patched[rate]
        summary["rates"][rate]["normalization"] = {
            "mean_x": payload["mean_x"],
            "std_x": payload["std_x"],
            "mean_y_full": payload["mean_y_full"],
            "std_y_full": payload["std_y_full"],
            "target_scaler": payload["target_scaler"],
            "input_scalers": payload["input_scalers"],
            "schema_check": "pass",
            "refit": False,
        }
        summary["rates"][rate]["denorm_max_abs_error"] = denorm[rate]
        summary["rates"][rate]["y_consistency"] = "pass"
        summary["rates"][rate]["x_nesting_check"] = "pass"
        summary["rates"][rate]["status"] = "ok"
    summary["cross_split_checks"] = cross
    summary["shared_target_scaler"] = {
        "mean_y_full": normalizations["p05"]["mean_y_full"],
        "std_y_full": normalizations["p05"]["std_y_full"],
        "identical_across_rates": True,
    }
    summary["weekday_spots"] = {
        day_name: {"weekday": weekday, "encoding": "monday_0"}
        for day_name, weekday in WEEKDAY_SPOTS.items()
    }
    summary["split_npz_count"] = len(split_files)
    summary["daily_npz_count"] = daily_after
    summary["status"] = "ok"
    write_json(WINDOW_ROOT / "merge_summary.json", summary)
    log("Xuancheng26 merged R-level window datasets are ready.")
    log("STID / STAEformer dataset requirements are now satisfied.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
