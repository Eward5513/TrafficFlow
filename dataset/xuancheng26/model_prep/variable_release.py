"""Release checks for one variable-penetration range.

The shared full states, target scaler, split, and node order are read only.
A failure marks the selected dataset failed and leaves every other dataset alone.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import generate_r_level_windows as windows
from independent_states import CHANNELS, PHYSICAL_VERSION, definition_sha256, residual, summarize_residual

STATES = (*CHANNELS, "all")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _dataset_root(training_root: Path, dataset_id: str) -> Path:
    return training_root / "datasets" / dataset_id


def audit_flow_day(task: dict) -> dict:
    """Check one dataset-day: finite non-negative states, adjacent snapshots, partial <= full."""
    try:
        date = task["date"]
        edges = task["edges"]
        full_root = Path(task["full_root"]) / date
        full_states = {name: windows.load_aligned_matrix(full_root / f"{name}.csv", edges) for name in CHANNELS}
        windows._check_conservation(full_states, f"full {date}")
        residuals = []
        for seed in task["seeds"]:
            directory = Path(task["partial_root"]) / f"seed_{seed}" / "flow" / "partial_flow" / task["key"] / date
            partial = {name: windows.load_aligned_matrix(directory / f"{name}.csv", edges) for name in CHANNELS}
            windows._check_conservation(partial, f"{task['dataset_id']} seed {seed} {date}")
            for name in CHANNELS:
                if np.any(partial[name] > full_states[name]):
                    raise RuntimeError(f"{date} seed {seed} {name} partial exceeds full")
            summary = summarize_residual(residual(
                partial["accumulation_start"], partial["accumulation_end"], partial["inflow"], partial["outflow"],
            ))
            residuals.append({"sampling_seed": int(seed), "nonzero_cells": summary["nonzero_cells"], "max_abs": summary["max_abs"]})
        return {"date": date, "dataset_id": task["dataset_id"], "ok": True, "residuals": residuals}
    except Exception as exc:
        return {"date": task.get("date"), "dataset_id": task.get("dataset_id"), "ok": False, "error": str(exc)}


def _coverage(root: Path, spec: dict, splits: dict[str, list[str]]) -> tuple[dict, list[str]]:
    key = spec["observation_key"]
    seeds = [int(seed) for seed in spec["sampling_seeds"]]
    dates = [day for name in ("train", "validation", "test") for day in splits[name]]
    counts = {"vehicle_csv": 0, "state_csv": 0, "window_npz": 0}
    missing = []
    for seed in seeds:
        seed_root = root / "sampling_seeds" / f"seed_{seed}"
        for date in dates:
            vehicle = seed_root / "sampled_vehicle" / key / f"{date}.csv"
            if vehicle.is_file() and vehicle.stat().st_size > 0:
                counts["vehicle_csv"] += 1
            else:
                missing.append(str(vehicle))
            for name in CHANNELS:
                path = seed_root / "flow" / "partial_flow" / key / date / f"{name}.csv"
                if path.is_file() and path.stat().st_size > 0:
                    counts["state_csv"] += 1
                else:
                    missing.append(str(path))
            split = next(name for name, days in splits.items() if date in days)
            for state in STATES:
                path = seed_root / "flow" / "window_data" / key / split / date / f"{state}.npz"
                if path.is_file() and path.stat().st_size > 0:
                    counts["window_npz"] += 1
                else:
                    missing.append(str(path))
    expected = {
        "vehicle_csv": len(seeds) * len(dates),
        "state_csv": len(seeds) * len(dates) * len(CHANNELS),
        "window_npz": len(seeds) * len(dates) * len(STATES),
    }
    for name, count in expected.items():
        if counts[name] != count:
            missing.append(f"{spec['dataset_id']} {name}={counts[name]} expected {count}")
    return counts, missing


def _spot_windows(training_root: Path, spec: dict, splits: dict[str, list[str]], edges: list[str]) -> list[str]:
    failures = []
    root = _dataset_root(training_root, spec["dataset_id"])
    key = spec["observation_key"]
    normalization = json.loads((root / "shared" / "state_normalization.json").read_text(encoding="utf-8"))
    if normalization.get("fit_sampling_seed") != 42 or normalization.get("fit_split") != "train":
        failures.append(f"{spec['dataset_id']} scaler is not the frozen seed-42 training fit")
    if normalization.get("channels") != list(CHANNELS):
        failures.append(f"{spec['dataset_id']} channel order is {normalization.get('channels')}")
    if normalization.get("target") != json.loads((training_root / "shared" / "target_normalization.json").read_text(encoding="utf-8"))["target"]:
        failures.append(f"{spec['dataset_id']} target scaler differs from the shared full-flow scaler")
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    for seed in spec["sampling_seeds"]:
        for split, days in splits.items():
            date = days[0]
            full_states = {
                name: windows.load_aligned_matrix(training_root / "full_flow" / "r-level" / date / f"{name}.csv", edges)
                for name in CHANNELS
            }
            day_dir = root / "sampling_seeds" / f"seed_{seed}" / "flow" / "window_data" / key / split / date
            identity_path = day_dir / "day_identity.json"
            if not identity_path.is_file():
                failures.append(f"missing window identity {identity_path}")
                continue
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            if (
                identity.get("dataset_id") != spec["dataset_id"]
                or int(identity.get("sampling_seed")) != int(seed)
                or identity.get("observation_key") != key
                or identity.get("schedule_sha256") != manifest.get("schedule_sha256")
                or identity.get("physical_definition_version") != PHYSICAL_VERSION
            ):
                failures.append(f"{day_dir} identity does not match the dataset manifest")
            for state in STATES:
                path = day_dir / f"{state}.npz"
                names = CHANNELS if state == "all" else (state,)
                try:
                    with np.load(path, allow_pickle=False) as payload:
                        if tuple(payload["x"].shape) != (windows.N_WINDOWS, windows.N_HIS, windows.N_NODES, len(names)):
                            raise RuntimeError(f"shape {payload['x'].shape}")
                        if tuple(payload["y"].shape) != (windows.N_WINDOWS, 1, windows.N_NODES, len(names)):
                            raise RuntimeError(f"y shape {payload['y'].shape}")
                        if not np.isfinite(payload["x"]).all() or not np.isfinite(payload["y"]).all():
                            raise RuntimeError("non-finite window")
                        if np.any(payload["x_raw"] < 0) or np.any(payload["y_raw"] < 0):
                            raise RuntimeError("negative raw window")
                        starts = payload["window_start_slot"]
                        targets = payload["target_slot"]
                        if not np.array_equal(starts, np.arange(windows.N_WINDOWS)) or not np.array_equal(targets, starts + 11):
                            raise RuntimeError("target slot is not window_start_slot + 11")
                        for channel, name in enumerate(names):
                            observed = payload["y_raw"][:, 0, :, channel]
                            expected = np.stack([full_states[name][int(slot)] for slot in targets]).astype(np.float32)
                            if not np.array_equal(observed, expected):
                                raise RuntimeError(f"{name} label is not the full state at the target slot")
                            mean = np.float32(normalization["input"][key][name]["mean"])
                            std = np.float32(normalization["input"][key][name]["std"])
                            mean_y = np.float32(normalization["target"][name]["mean"])
                            std_y = np.float32(normalization["target"][name]["std"])
                            if not np.allclose(payload["x"][:, :, :, channel], (payload["x_raw"][:, :, :, channel] - mean) / std, atol=1e-5):
                                raise RuntimeError(f"{name} input scaler is not the frozen seed-42 scaler")
                            if not np.allclose(payload["y"][:, :, :, channel], (payload["y_raw"][:, :, :, channel] - mean_y) / std_y, atol=1e-5):
                                raise RuntimeError(f"{name} target scaler is not the shared scaler")
                except Exception as exc:
                    failures.append(f"{path}: {exc}")
    return failures


def _mark(root: Path, dataset_id: str, failures: list[str]) -> None:
    passed = not failures
    summary = {
        "overall_validation_passed": passed,
        "dataset_status": "ready" if passed else "failed",
        "failures": failures[:50],
        "dataset_id": dataset_id,
        "physical_definition_version": PHYSICAL_VERSION,
        "definition_sha256": definition_sha256(),
    }
    shared = root / "shared"
    normalization_path = shared / "state_normalization.json"
    manifest_path = root / "dataset_manifest.json"
    if normalization_path.is_symlink() or manifest_path.is_symlink():
        raise RuntimeError(f"refusing to publish status through a symlink in {root}")
    _write_json(shared / "validation_summary.json", summary)
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    normalization["dataset_status"] = summary["dataset_status"]
    _write_json(normalization_path, normalization)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_status"] = summary["dataset_status"]
    manifest["physical_definition_version"] = PHYSICAL_VERSION
    manifest["definition_sha256"] = definition_sha256()
    _write_json(manifest_path, manifest)


def audit_and_release(training_root: Path, specs: list[dict], splits: dict[str, list[str]], edges: list[str], workers: int) -> dict:
    training_root = Path(training_root)
    dates = [day for name in ("train", "validation", "test") for day in splits[name]]
    failures: list[str] = []
    counts: dict[str, dict] = {}
    tasks = []
    clear_ids = []
    for spec in specs:
        root = _dataset_root(training_root, spec["dataset_id"])
        manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
        own: list[str] = []
        if manifest.get("observation_keys") != [spec["observation_key"]]:
            own.append(f"{spec['dataset_id']} observation key drifted")
        if manifest.get("physical_definition_version") != PHYSICAL_VERSION or manifest.get("definition_sha256") != definition_sha256():
            own.append(f"{spec['dataset_id']} physical definition does not match the generator")
        shared_target = (root / "shared" / "target_normalization.json").resolve()
        if shared_target != (training_root / "shared" / "target_normalization.json").resolve():
            own.append(f"{spec['dataset_id']} target scaler link leaves the shared file")
        sequence = (root / "shared" / "sequence_targets_12step").resolve()
        if sequence != (training_root / "shared" / "sequence_targets_12step").resolve():
            own.append(f"{spec['dataset_id']} sequence target link leaves the shared directory")
        covered, missing = _coverage(root, {**spec, "sampling_seeds": manifest["sampling_seeds"]}, splits)
        counts[spec["dataset_id"]] = covered
        own.extend(missing[:20])
        failures.extend(own)
        if own:
            continue
        clear_ids.append(spec["dataset_id"])
        for date in dates:
            tasks.append({
                "date": date,
                "dataset_id": spec["dataset_id"],
                "key": spec["observation_key"],
                "seeds": [int(seed) for seed in manifest["sampling_seeds"]],
                "edges": edges,
                "full_root": str(training_root / "full_flow" / "r-level"),
                "partial_root": str(root / "sampling_seeds"),
            })
    if tasks:
        worker_count = max(1, min(int(workers), len(tasks)))
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(audit_flow_day, task) for task in tasks]
            for future in as_completed(futures):
                row = future.result()
                if not row.get("ok"):
                    failures.append(f"{row.get('dataset_id')} {row.get('date')}: {row.get('error')}")
    for spec in specs:
        if spec["dataset_id"] not in clear_ids or any(spec["dataset_id"] in item for item in failures):
            continue
        manifest = json.loads((_dataset_root(training_root, spec["dataset_id"]) / "dataset_manifest.json").read_text(encoding="utf-8"))
        checked = dict(spec)
        checked["sampling_seeds"] = manifest["sampling_seeds"]
        failures.extend(_spot_windows(training_root, checked, splits, edges))
    for spec in specs:
        root = _dataset_root(training_root, spec["dataset_id"])
        relevant = [item for item in failures if spec["dataset_id"] in item]
        _mark(root, spec["dataset_id"], relevant)
        _write_json(root / "audit" / "release.json", {
            "dataset_id": spec["dataset_id"],
            "counts": counts.get(spec["dataset_id"]),
            "failures": relevant[:50],
        })
    return {"failures": failures, "counts": counts}
