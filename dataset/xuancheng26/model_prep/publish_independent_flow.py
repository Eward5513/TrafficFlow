#!/usr/bin/env python3
"""Archive the cumsum stock and publish independent_fcd_v1 states, scalers, and windows."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import shutil
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import generate_r_level_windows as windows  # noqa: E402
from independent_states import (  # noqa: E402
    CHANNELS,
    PHYSICAL_VERSION,
    SEEDS,
    definition_document,
    definition_sha256,
)

ARCHIVE_NAME = "cumsum_stock_v1"
RATES = ("p05", "p10", "p20", "p30", "p40", "p50", "p70")
DATASETS = (
    ("fixed_uniform", RATES),
    ("variable_30min_p10_p20", ("p10_p20",)),
    ("variable_1h_p10_p20", ("p10_p20",)),
    ("variable_2h_p10_p20", ("p10_p20",)),
    ("variable_day_p10_p20", ("p10_p20",)),
)


def log(message: str) -> None:
    from datetime import datetime

    print(f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}", flush=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def link_relative(path: Path, target: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = os.path.relpath(target, path.parent)
    if path.is_symlink() and os.readlink(path) == text:
        return
    if path.is_symlink() or path.exists():
        path.unlink()
    path.symlink_to(text)


def move_tree(src: Path, dst: Path, journal: list) -> str:
    if not src.exists() and not src.is_symlink():
        return "absent"
    if dst.exists() or dst.is_symlink():
        if not src.exists():
            return "already"
        raise RuntimeError(f"both exist, refusing to overwrite {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.rename(src, dst)
    journal.append({"action": "rename", "src": str(src), "dst": str(dst)})
    return "moved"


def copy_bytes(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    target = src.resolve() if src.is_symlink() else src
    shutil.copy2(target, dst)


def snapshot_shared(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if path.is_dir() and not path.is_symlink():
            continue
        copy_bytes(path, dst / path.name)


def training_paths(root: Path) -> dict:
    return {
        "root": root,
        "staging": root / ".staging" / PHYSICAL_VERSION,
        "archive": root / "archive" / ARCHIVE_NAME,
    }


def assert_scan_idle(staging: Path) -> list[str]:
    dates = sorted(path.parent.name for path in staging.glob("*/complete.json"))
    if len(dates) != 29:
        raise RuntimeError(f"staging has {len(dates)} complete days, expected 29")
    busy = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\x00", b" ")
        except OSError:
            continue
        if b"independent_states.py" in command and b"scan" in command:
            busy.append(entry.name)
    if busy:
        raise RuntimeError(f"scan still running: {busy}")
    return dates


def archive_metadata(root: Path, journal: list) -> None:
    archive = root / "archive" / ARCHIVE_NAME
    if (archive / "metadata_archived.json").is_file():
        log("metadata archive already recorded")
        return
    shared = archive / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    live = root / "shared"
    for name in ("state_normalization.json", "state_window_manifest.json", "validation_summary.json"):
        move_tree(live / name, shared / name, journal)
    move_tree(live / "sequence_targets_12step", shared / "sequence_targets_12step", journal)
    for name in ("split_manifest.json", "node_mapping.csv"):
        copy_bytes(live / name, shared / name)
    fixed_shared = root / "datasets" / "fixed_uniform" / "shared"
    if fixed_shared.is_symlink():
        fixed_shared.unlink()
        journal.append({"action": "unlink", "src": str(fixed_shared)})
    fixed_shared.mkdir(parents=True, exist_ok=True)
    for dataset_id, _rates in DATASETS:
        if dataset_id == "fixed_uniform":
            continue
        snapshot_shared(root / "datasets" / dataset_id / "shared", archive / "datasets" / dataset_id / "shared")
    for dataset_id, _rates in DATASETS:
        for seed in SEEDS:
            src = root / "datasets" / dataset_id / "sampling_seeds" / f"seed_{seed}" / "flow" / "window_data"
            dst = archive / "datasets" / dataset_id / "sampling_seeds" / f"seed_{seed}" / "flow" / "window_data"
            if src.exists() and any(src.iterdir()):
                move_tree(src, dst, journal)
            src.mkdir(parents=True, exist_ok=True)
    write_json(archive / "metadata_archived.json", {"physical_definition_version": ARCHIVE_NAME, "moves": len(journal)})


def _identical(group: dict, name: str) -> bool:
    compared = (group.get("comparison") or {}).get(name) or {}
    return compared.get("status") == "compared" and compared.get("changed_cells") == 0


def _place(staged: Path, live: Path, archived: Path, identical: bool, journal: list) -> None:
    if not staged.exists():
        if live.exists():
            return
        raise RuntimeError(f"missing staged and live state file: {staged}")
    if live.exists() or live.is_symlink():
        move_tree(live, archived, journal)
    live.parent.mkdir(parents=True, exist_ok=True)
    if identical and archived.exists():
        os.link(archived, live)
        staged.unlink()
    else:
        os.rename(staged, live)


def publish_day(task: dict) -> dict:
    try:
        date = task["date"]
        root = Path(task["root"])
        staging = Path(task["staging"]) / date
        archive = Path(task["archive"])
        staged_breakdown = staging / "full" / "event_breakdown.json"
        live_breakdown = root / "full_flow" / "r-level" / date / "event_breakdown.json"
        source = staged_breakdown if staged_breakdown.exists() else live_breakdown
        breakdown = json.loads(source.read_text(encoding="utf-8"))
        journal = []
        full_live = root / "full_flow" / "r-level" / date
        full_archive = archive / "full_flow" / "r-level" / date
        for name in CHANNELS:
            live = full_live / f"{name}.csv"
            archived = full_archive / f"{name}.csv"
            staged = staging / "full" / f"{name}.csv"
            _place(staged, live, archived, _identical({"comparison": breakdown.get("comparison")}, name), journal)
        for legacy in ("entry_flow.csv", "exit_flow.csv"):
            move_tree(full_live / legacy, full_archive / legacy, journal)
        staged_breakdown = staging / "full" / "event_breakdown.json"
        if staged_breakdown.exists():
            os.replace(staged_breakdown, full_live / "event_breakdown.json")
        by_relative = {item["relative"]: item for item in breakdown["groups"]}
        for dataset_id, rates in DATASETS:
            for seed in SEEDS:
                for rate in rates:
                    if dataset_id == "fixed_uniform":
                        relative = f"fixed/seed_{seed}/{rate}"
                        live_dir = root / "datasets" / dataset_id / "sampling_seeds" / f"seed_{seed}" / "flow" / "partial_flow" / rate / date
                    else:
                        relative = f"variable/{dataset_id}/seed_{seed}"
                        live_dir = root / "datasets" / dataset_id / "sampling_seeds" / f"seed_{seed}" / "flow" / "partial_flow" / rate / date
                    group = by_relative[relative]
                    archive_dir = archive / "datasets" / dataset_id / "sampling_seeds" / f"seed_{seed}" / "flow" / "partial_flow" / rate / date
                    for name in CHANNELS:
                        _place(
                            staging / relative / f"{name}.csv",
                            live_dir / f"{name}.csv",
                            archive_dir / f"{name}.csv",
                            _identical(group, name),
                            journal,
                        )
        return {"date": date, "ok": True, "journal": journal}
    except Exception:
        return {"date": task["date"], "ok": False, "error": traceback.format_exc(limit=8)}


def _add_moment(bucket: dict, moment: tuple[float, float, int]) -> None:
    bucket["sum"] += moment[0]
    bucket["square"] += moment[1]
    bucket["count"] += moment[2]


def moment_day(task: dict) -> dict:
    try:
        date = task["date"]
        root = Path(task["root"])
        edges = windows.load_edge_ids()
        full_states = {
            name: windows.load_aligned_matrix(root / "full_flow" / "r-level" / date / f"{name}.csv", edges)
            for name in CHANNELS
        }
        if not np.array_equal(full_states["accumulation_start"][1:], full_states["accumulation_end"][:-1]):
            raise RuntimeError(f"{date} full boundary occupancy is not continuous")
        payload = {"date": date, "target": {}, "input": {}}
        for name in CHANNELS:
            _x, y_raw, _starts, _targets = windows.make_windows(full_states[name], full_states[name])
            total, square, count = windows._moment(y_raw)
            payload["target"][name] = {"sum": total, "square": square, "count": count}
            del _x, y_raw
        for dataset_id, rates in DATASETS:
            for rate in rates:
                directory = root / "datasets" / dataset_id / "sampling_seeds" / "seed_42" / "flow" / "partial_flow" / rate / date
                for name in CHANNELS:
                    partial = windows.load_aligned_matrix(directory / f"{name}.csv", edges)
                    if np.any(partial > full_states[name]):
                        raise RuntimeError(f"{date} {dataset_id} {rate} {name} exceeds full")
                    raw, _y, _starts, _targets = windows.make_windows(partial, full_states[name])
                    total, square, count = windows._moment(raw)
                    payload["input"][f"{dataset_id}|{rate}|{name}"] = {"sum": total, "square": square, "count": count}
                    del raw, partial
        return {"ok": True, **payload}
    except Exception:
        return {"date": task["date"], "ok": False, "error": traceback.format_exc(limit=8)}


def freeze_scalers(root: Path, train_days: list[str], rows: list[dict]) -> dict:
    target = {name: {"sum": 0.0, "square": 0.0, "count": 0} for name in CHANNELS}
    inputs: dict[str, dict] = {}
    for row in rows:
        for name in CHANNELS:
            _add_moment(target[name], (row["target"][name]["sum"], row["target"][name]["square"], row["target"][name]["count"]))
        for key, moment in row["input"].items():
            bucket = inputs.setdefault(key, {"sum": 0.0, "square": 0.0, "count": 0})
            _add_moment(bucket, (moment["sum"], moment["square"], moment["count"]))
    target_stats = {name: dict(zip(("mean", "std"), windows._mean_std(item["sum"], item["square"], item["count"]))) for name, item in target.items()}
    target_doc = {
        "physical_definition_version": PHYSICAL_VERSION,
        "definition_sha256": definition_sha256(),
        "fit_split": "train",
        "ddof": 0,
        "formula": "(value - mean) / std",
        "channels": list(CHANNELS),
        "training_days": list(train_days),
        "source": "full_flow/r-level training-window y_raw, one target per window, ddof 0",
        "target": target_stats,
    }
    write_json(root / "shared" / "target_normalization.json", target_doc)
    write_json(root / "shared" / "physical_definition.json", definition_document())
    split = json.loads((root / "shared" / "split_manifest.json").read_text(encoding="utf-8"))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, text=True).strip()
    scalers = {}
    for dataset_id, rates in DATASETS:
        shared = root / "datasets" / dataset_id / "shared"
        if shared.is_symlink():
            shared.unlink()
        shared.mkdir(parents=True, exist_ok=True)
        link_relative(shared / "split_manifest.json", root / "shared" / "split_manifest.json")
        link_relative(shared / "node_mapping.csv", root / "shared" / "node_mapping.csv")
        link_relative(shared / "physical_definition.json", root / "shared" / "physical_definition.json")
        link_relative(shared / "target_normalization.json", root / "shared" / "target_normalization.json")
        input_stats = {}
        for rate in rates:
            input_stats[rate] = {}
            for name in CHANNELS:
                moment = inputs[f"{dataset_id}|{rate}|{name}"]
                mean, std = windows._mean_std(moment["sum"], moment["square"], moment["count"])
                input_stats[rate][name] = {"mean": mean, "std": std}
        payload = {
            "physical_definition_version": PHYSICAL_VERSION,
            "definition_sha256": definition_sha256(),
            "dataset_status": "building",
            "dataset_id": dataset_id,
            "fit_split": "train",
            "ddof": 0,
            "formula": "(value - mean) / std",
            "scope": "global_scalar_per_variable",
            "channels": list(CHANNELS),
            "training_days": list(train_days),
            "fit_sampling_seed": 42,
            "input_source": "sampling seed 42 training-window x_raw, overlap retained, ddof 0",
            "target_source": "shared/target_normalization.json",
            "target": target_stats,
            "input": input_stats,
            "git_commit": commit,
        }
        write_json(shared / "state_normalization.json", payload)
        manifest = {
            "train": split["train"],
            "validation": split["validation"],
            "test": split["test"],
            "samples_per_day": windows.N_WINDOWS,
            "num_nodes": windows.N_NODES,
            "n_his": windows.N_HIS,
            "output_steps": 1,
            "target_mode": "last-observed-step",
            "label_definition": "Y is the full state at window_start_slot + 11",
            "channels": list(CHANNELS),
            "state_files": list(windows.STATE_FILES),
            "rates": list(rates),
            "dataset_id": dataset_id,
            "sampling_seeds": list(SEEDS),
            "physical_definition_version": PHYSICAL_VERSION,
            "fit_sampling_seed": 42,
        }
        write_json(shared / "state_window_manifest.json", manifest)
        write_json(shared / "validation_summary.json", {
            "overall_validation_passed": False,
            "dataset_status": "building",
            "failures": ["windows not audited"],
            "dataset_id": dataset_id,
            "physical_definition_version": PHYSICAL_VERSION,
        })
        for seed in SEEDS:
            window = root / "datasets" / dataset_id / "sampling_seeds" / f"seed_{seed}" / "flow" / "window_data"
            window.mkdir(parents=True, exist_ok=True)
            for name in ("split_manifest.json", "node_mapping.csv", "state_normalization.json", "state_window_manifest.json", "validation_summary.json"):
                link_relative(window / name, shared / name)
        scalers[dataset_id] = payload
        dataset_manifest = root / "datasets" / dataset_id / "dataset_manifest.json"
        current = json.loads(dataset_manifest.read_text(encoding="utf-8")) if dataset_manifest.is_file() else {"dataset_id": dataset_id}
        current.update({
            "physical_definition_version": PHYSICAL_VERSION,
            "definition_sha256": definition_sha256(),
            "dataset_status": "building",
            "sampling_algorithm_version": current.get("algorithm_version") or current.get("sampling_algorithm_version"),
        })
        write_json(dataset_manifest, current)
    fixed_norm = root / "datasets" / "fixed_uniform" / "shared" / "state_normalization.json"
    shutil.copy2(fixed_norm, root / "shared" / "state_normalization.json")
    shutil.copy2(root / "datasets" / "fixed_uniform" / "shared" / "state_window_manifest.json", root / "shared" / "state_window_manifest.json")
    shutil.copy2(root / "datasets" / "fixed_uniform" / "shared" / "validation_summary.json", root / "shared" / "validation_summary.json")
    return scalers


def window_day(task: dict) -> dict:
    try:
        date = task["date"]
        root = Path(task["root"])
        edges = windows.load_edge_ids()
        full_states = {
            name: windows.load_aligned_matrix(root / "full_flow" / "r-level" / date / f"{name}.csv", edges)
            for name in CHANNELS
        }
        weekday = windows.weekday_index(date)
        written = skipped = 0
        for dataset_id, rates in DATASETS:
            scaler = task["scalers"][dataset_id]
            for seed in SEEDS:
                for rate in rates:
                    partial_dir = root / "datasets" / dataset_id / "sampling_seeds" / f"seed_{seed}" / "flow" / "partial_flow" / rate / date
                    partial = {name: windows.load_aligned_matrix(partial_dir / f"{name}.csv", edges) for name in CHANNELS}
                    single = {}
                    starts = targets = None
                    for name in CHANNELS:
                        if np.any(partial[name] > full_states[name]):
                            raise RuntimeError(f"{date} {dataset_id} {seed} {rate} {name} exceeds full")
                        raw, label, starts, targets = windows.make_windows(partial[name], full_states[name])
                        single[name] = (raw, label)
                    bundles = dict(single)
                    bundles["all"] = (
                        np.concatenate([single[name][0] for name in CHANNELS], axis=-1),
                        np.concatenate([single[name][1] for name in CHANNELS], axis=-1),
                    )
                    day_dir = root / "datasets" / dataset_id / "sampling_seeds" / f"seed_{seed}" / "flow" / "window_data" / rate / task["split"] / date
                    day_index = np.full(windows.N_WINDOWS, weekday, dtype=np.int32)
                    for state_name, (x_raw, y_raw) in bundles.items():
                        names = CHANNELS if state_name == "all" else (state_name,)
                        mean_x = np.asarray([scaler["input"][rate][name]["mean"] for name in names], dtype=np.float32)
                        std_x = np.asarray([scaler["input"][rate][name]["std"] for name in names], dtype=np.float32)
                        mean_y = np.asarray([scaler["target"][name]["mean"] for name in names], dtype=np.float32)
                        std_y = np.asarray([scaler["target"][name]["std"] for name in names], dtype=np.float32)
                        path = day_dir / f"{state_name}.npz"
                        channels = 4 if state_name == "all" else 1
                        if windows._file_matches(path, channels=channels, weekday=weekday, mean_x=mean_x, std_x=std_x, mean_y=mean_y, std_y=std_y):
                            skipped += 1
                            continue
                        archived = (
                            root / "archive" / "cumsum_stock_v1" / "datasets" / dataset_id / "sampling_seeds"
                            / f"seed_{seed}" / "flow" / "window_data" / rate / task["split"] / date / f"{state_name}.npz"
                        )
                        if state_name in {"inflow", "outflow"} and [dataset_id, rate, state_name] in task.get("reusable", []) and archived.is_file():
                            path.parent.mkdir(parents=True, exist_ok=True)
                            os.link(archived, path)
                            skipped += 1
                            continue
                        windows.write_npz(path, {
                            "x": windows._normalize(x_raw, mean_x.reshape(1, 1, 1, -1), std_x.reshape(1, 1, 1, -1)),
                            "y": windows._normalize(y_raw, mean_y.reshape(1, 1, 1, -1), std_y.reshape(1, 1, 1, -1)),
                            "x_raw": np.ascontiguousarray(x_raw),
                            "y_raw": np.ascontiguousarray(y_raw),
                            "day_index": day_index,
                            "window_start_slot": starts,
                            "target_slot": targets,
                        })
                        if not windows._file_matches(path, channels=channels, weekday=weekday, mean_x=mean_x, std_x=std_x, mean_y=mean_y, std_y=std_y):
                            raise RuntimeError(f"window verification failed: {path}")
                        written += 1
                    del partial, single, bundles
        log(f"[{date}] windows written={written} skipped={skipped}")
        return {"date": date, "ok": True, "written": written, "skipped": skipped}
    except Exception:
        return {"date": task["date"], "ok": False, "error": traceback.format_exc(limit=8)}


def run_pool(function, tasks: list[dict], workers: int) -> list[dict]:
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(function, task): task["date"] for task in tasks}
        for future in as_completed(futures):
            date = futures[future]
            try:
                results[date] = future.result()
            except Exception:
                results[date] = {"date": date, "ok": False, "error": traceback.format_exc(limit=6)}
            row = results[date]
            if not row.get("ok"):
                log(f"[{date}] FAIL {(row.get('error') or '')[-400:]}")
    return [results[task["date"]] for task in tasks]


def mark_ready(root: Path, passed: bool, failures: list[str]) -> None:
    for dataset_id, _rates in DATASETS:
        shared = root / "datasets" / dataset_id / "shared"
        summary = {
            "overall_validation_passed": passed,
            "dataset_status": "ready" if passed else "failed",
            "failures": failures,
            "dataset_id": dataset_id,
            "physical_definition_version": PHYSICAL_VERSION,
            "definition_sha256": definition_sha256(),
        }
        write_json(shared / "validation_summary.json", summary)
        normalization = json.loads((shared / "state_normalization.json").read_text(encoding="utf-8"))
        normalization["dataset_status"] = summary["dataset_status"]
        write_json(shared / "state_normalization.json", normalization)
        manifest_path = root / "datasets" / dataset_id / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["dataset_status"] = summary["dataset_status"]
        write_json(manifest_path, manifest)
    shutil.copy2(root / "datasets" / "fixed_uniform" / "shared" / "state_normalization.json", root / "shared" / "state_normalization.json")
    shutil.copy2(root / "datasets" / "fixed_uniform" / "shared" / "validation_summary.json", root / "shared" / "validation_summary.json")
    marker = {
        "physical_definition_version": "legacy_cumsum_v1",
        "status": "not_rebuilt",
        "note": "M and MR states still use cumsum stock. They were not regenerated for independent_fcd_v1.",
    }
    write_json(root / "full_flow" / "m-level" / "physical_definition_status.json", marker)
    if (root / "full_flow" / "mr-level").is_dir():
        write_json(root / "full_flow" / "mr-level" / "physical_definition_status.json", marker)


def main() -> int:
    raise SystemExit(
        "publish_independent_flow is retired. It required archive/cumsum_stock_v1, "
        "which has been removed. Current windows are under training_data_3s/datasets."
    )
    import argparse
    from data_root import data_root

    parser = argparse.ArgumentParser(description="Publish independent_fcd_v1 training data")
    parser.add_argument("--training-root", type=Path, default=data_root() / "xuancheng26/training_data_3s")
    parser.add_argument("--workers", type=int, default=29)
    parser.add_argument("--stage", choices=("all", "csv", "scaler", "windows", "ready"), default="all")
    args = parser.parse_args()
    root = args.training_root.expanduser().resolve()
    paths = training_paths(root)
    dates = assert_scan_idle(paths["staging"])
    split = json.loads((root / "shared" / "split_manifest.json").read_text(encoding="utf-8"))
    split_of = {day: name for name, days in split.items() if name in {"train", "validation", "test"} for day in days}
    workers = max(1, min(args.workers, len(dates)))
    journal: list = []
    if args.stage in {"all", "csv"}:
        archive_metadata(root, journal)
        tasks = [{"date": date, "root": str(root), "staging": str(paths["staging"]), "archive": str(paths["archive"])} for date in dates]
        rows = run_pool(publish_day, tasks, workers)
        failed = [row for row in rows if not row.get("ok")]
        journal.extend(item for row in rows if row.get("ok") for item in row.get("journal", []))
        write_json(root / "architecture_audit" / "independent_fcd_v1_migration_journal.json", {"moves": journal, "failed": [row["date"] for row in failed]})
        if failed:
            return 1
        log("csv publish complete")
    if args.stage == "csv":
        return 0
    if args.stage in {"all", "scaler", "windows"}:
        tasks = [{"date": day, "root": str(root)} for day in split["train"]]
        rows = run_pool(moment_day, tasks, workers)
        if any(not row.get("ok") for row in rows):
            return 1
        scalers = freeze_scalers(root, split["train"], rows)
        write_json(paths["staging"] / "scalers.json", scalers)
        log("scalers frozen")
    if args.stage == "scaler":
        return 0
    scalers = json.loads((paths["staging"] / "scalers.json").read_text(encoding="utf-8"))
    reusable = []
    archive_root = root / "archive" / ARCHIVE_NAME
    old_fixed = json.loads((archive_root / "shared" / "state_normalization.json").read_text(encoding="utf-8"))
    for rate, channels in scalers["fixed_uniform"]["input"].items():
        for name in ("inflow", "outflow"):
            if channels[name] == old_fixed["input"][rate][name] and scalers["fixed_uniform"]["target"][name] == old_fixed["target"][name]:
                reusable.append([ "fixed_uniform", rate, name])
    for dataset_id in ("variable_day_p10_p20", "variable_2h_p10_p20"):
        old = json.loads((archive_root / "datasets" / dataset_id / "shared" / "state_normalization.json").read_text(encoding="utf-8"))
        for name in ("inflow", "outflow"):
            if scalers[dataset_id]["input"]["p10_p20"][name] == old["input"]["p10_p20"][name] and scalers[dataset_id]["target"][name] == old["target"][name]:
                reusable.append([dataset_id, "p10_p20", name])
    if args.stage in {"all", "windows"}:
        tasks = [{"date": date, "split": split_of[date], "root": str(root), "scalers": scalers, "reusable": reusable} for date in dates]
        rows = run_pool(window_day, tasks, workers)
        if any(not row.get("ok") for row in rows):
            return 1
        log("windows complete")
    if args.stage != "ready" and args.stage != "all":
        return 0
    counts = {
        "full_csv": sum(1 for _ in root.glob("full_flow/r-level/*/*.csv")),
        "vehicle_csv": sum(1 for _ in root.glob("datasets/*/sampling_seeds/seed_*/sampled_vehicle/*/*.csv")),
        "partial_csv": sum(1 for _ in root.glob("datasets/*/sampling_seeds/seed_*/flow/partial_flow/*/*/*.csv")),
        "window_npz": sum(1 for _ in root.glob("datasets/*/sampling_seeds/seed_*/flow/window_data/*/*/*/*.npz")),
    }
    expected = {"full_csv": 116, "vehicle_csv": 1595, "partial_csv": 6380, "window_npz": 7975}
    count_failures = [f"{key}={counts[key]} expected {expected[key]}" for key in expected if counts[key] != expected[key]]
    write_json(root / "architecture_audit" / "independent_fcd_v1_counts.json", {"counts": counts, "expected": expected, "failures": count_failures})
    if count_failures:
        mark_ready(root, False, count_failures)
        log(f"count mismatch {count_failures}")
        return 1
    mark_ready(root, False, ["sequence targets not rebuilt"])
    sequence = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("generate_r_level_sequence_targets.py")), "--workers", str(workers), "--overwrite"],
        cwd=Path(__file__).resolve().parent,
        check=False,
    )
    if sequence.returncode != 0:
        mark_ready(root, False, ["sequence target rebuild failed"])
        return sequence.returncode
    for dataset_id, _rates in DATASETS:
        link_relative(
            root / "datasets" / dataset_id / "shared" / "sequence_targets_12step",
            root / "shared" / "sequence_targets_12step",
        )
    mark_ready(root, True, [])
    log("published ready")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
