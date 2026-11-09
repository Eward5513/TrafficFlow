#!/usr/bin/env python3
"""Queue the 144 Xuancheng variable-penetration baseline tasks.

One task is one model, one dataset, and one input/target pair. It keeps one
physical GPU until the process exits. A GPU is free only when nvidia-smi lists
no compute process on it. Memory is not part of that decision. The scheduler
asks again every 30 seconds.

``control.json`` is ``{"max_gpus": N}``. A missing file starts at 8. The next
poll reads edits. ``SIGHUP`` wakes the scheduler and reads it immediately. A
line in ``messages.jsonl``, ``{"max_gpus": N}``, is the same change. The
first poll at or after 09:00 local time each day writes ``max_gpus`` back to
4. A later edit still applies until the next morning.

Lowering the cap does not stop a job that is already running. ``0`` pauses
new launches. RF stays on CPU and still reserves its GPU slot; an empty
nvidia-smi row does not release that slot.

Campaign files live under
``<training_results_3s>/operations/<campaign_id>/``.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data_root.py").is_file() and (parent / "reimplementation").is_dir():
            return parent
    raise RuntimeError(f"cannot find the TrafficFlow checkout from {here}")


CODE = repo_root()
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from data_root import (  # noqa: E402
    new_run_id,
    sampling_run_results_dir,
    xuancheng26_model_config,
    xuancheng26_training_3s_root,
    xuancheng26_training_results_3s_root,
)

CAMPAIGN_ID = "variable_p10_p20_train42_eval5"
EXPERIMENT_ID = "variable_p10_p20_train42_eval5"
OBSERVATION_KEY = "p10_p20"
PHYSICAL_VERSION = "independent_fcd_v1"
MODEL_SEED = 42
EVAL_SEEDS = (42, 43, 44, 45, 46)
POLL_SECONDS = 30
DEFAULT_MAX_GPUS = 8
MORNING_HOUR = 9
MORNING_CAP = 4
MIN_FREE_RAM_BYTES = 32 * 1024**3
SUMMARY_COLUMNS = (
    "model",
    "dataset_id",
    "period_seconds",
    "io",
    "train_sampling_seed",
    "model_seed",
    "eval_sampling_seeds",
    "status",
    "run_dir",
    "best_epoch",
    "fit",
    "val_mae_mean",
    "val_mae_std",
    "test_mae_mean",
    "test_mae_std",
    "test_inflow_mae_mean",
    "test_inflow_mae_std",
    "test_outflow_mae_mean",
    "test_outflow_mae_std",
    "test_accumulation_start_mae_mean",
    "test_accumulation_start_mae_std",
    "test_accumulation_end_mae_mean",
    "test_accumulation_end_mae_std",
    "duration",
    "error",
)
MODELS = (
    {"name": "RF", "script": "reimplementation/RF/run_rf.py", "cpu": True},
    {"name": "stgcn", "script": "reimplementation/stgcn/train_r_only_stgcn.py"},
    {"name": "dcrnn", "script": "reimplementation/dcrnn/train_r_only_dcrnn.py"},
    {"name": "graph_wavenet", "script": "reimplementation/graph_wavenet/train_r_only_gwn.py"},
    {"name": "stsgcn", "script": "reimplementation/stsgcn/train_r_only_stsgcn.py"},
    {"name": "stfgnn", "script": "reimplementation/stfgnn/train_r_only_stfgnn.py"},
    {"name": "stid", "script": "reimplementation/stid/train_r_only_stid.py"},
    {"name": "staeformer", "script": "reimplementation/staeformer/train_r_only_staeformer.py"},
    {"name": "s2s_gcn_lstm", "script": "reimplementation/s2s_gcn_lstm/train_r_only_s2s_gcn_lstm.py"},
    {"name": "tlast", "script": "reimplementation/tlast/train_r_only_tlast.py"},
    {"name": "asttn", "script": "reimplementation/asttn/train_r_only_asttn.py"},
    {"name": "stjgcn", "script": "reimplementation/stjgcn/train_r_only_stjgcn.py"},
)
DATASETS = (
    {"dataset_id": "variable_30min_p10_p20", "period_seconds": 1800},
    {"dataset_id": "variable_1h_p10_p20", "period_seconds": 3600},
    {"dataset_id": "variable_2h_p10_p20", "period_seconds": 7200},
    {"dataset_id": "variable_day_p10_p20", "period_seconds": 86400},
)
IOS = (
    {"io": "I2I", "input_type": "inflow", "target_type": "inflow"},
    {"io": "A2A", "input_type": "all", "target_type": "all"},
    {"io": "A2I", "input_type": "all", "target_type": "inflow"},
)
REPO_CONFIGS = {
    "asttn": CODE / "reimplementation" / "asttn" / "configs" / "r_only_asttn.json",
    "tlast": CODE / "reimplementation" / "tlast" / "configs" / "r_only_tlast.json",
    "stjgcn": CODE / "reimplementation" / "stjgcn" / "configs" / "r_only_stjgcn.json",
}
RELOAD_REQUESTED = False
STOP_REQUESTED = False


def local_now() -> datetime:
    return datetime.now().astimezone()


def stamp(when: datetime | None = None) -> str:
    return (when or local_now()).isoformat(timespec="seconds")


def log_line(ops: Path, message: str) -> None:
    line = f"{stamp()} {message}\n"
    ops.mkdir(parents=True, exist_ok=True)
    with (ops / "scheduler.log").open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
    print(line, end="", flush=True)


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def training_python() -> Path:
    override = os.environ.get("TRAFFICFLOW_PYTHON")
    if override:
        return Path(override)
    conda = Path("/home/tzhang174/miniforge3/envs/trafficflow/bin/python3")
    if conda.is_file():
        return conda
    return Path(sys.executable)


def config_path(model_name: str) -> Path:
    if model_name in REPO_CONFIGS:
        return REPO_CONFIGS[model_name]
    key = "gwn" if model_name == "graph_wavenet" else model_name
    return xuancheng26_model_config(key)


def morning_reset_due(now: datetime, applied_on: str | None) -> bool:
    return now.hour >= MORNING_HOUR and applied_on != now.date().isoformat()


def parse_cap(value: object, gpu_count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"max_gpus must be an integer, got {value!r}")
    upper = gpu_count if gpu_count > 0 else 64
    if not 0 <= value <= upper:
        raise ValueError(f"max_gpus must be in 0..{upper}, got {value}")
    return value


def thread_budget(cap: int) -> int:
    cpus = os.cpu_count() or 1
    slots = max(int(cap), 1)
    return max(1, min(16, (cpus - 4) // slots))


def mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    stat = Path(f"/proc/{pid}/stat")
    if not stat.is_file():
        return False
    fields = stat.read_text(encoding="utf-8").split()
    return len(fields) < 3 or fields[2] != "Z"


def request_reload(signum, frame) -> None:
    del signum, frame
    global RELOAD_REQUESTED
    RELOAD_REQUESTED = True


def request_stop(signum, frame) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def wait_poll(seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while not RELOAD_REQUESTED and not STOP_REQUESTED and time.monotonic() < deadline:
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def query_gpus() -> tuple[list[dict[str, object]], set[int]]:
    """Return GPU index/uuid rows and indexes that have any compute process.

    nvidia-smi must succeed. A failed query is not an empty card.
    """
    listed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, object]] = []
    uuid_to_index: dict[str, int] = {}
    for line in listed.stdout.splitlines():
        if not line.strip():
            continue
        index_text, uuid = [part.strip() for part in line.split(",", 1)]
        index = int(index_text)
        rows.append({"index": index, "uuid": uuid})
        uuid_to_index[uuid] = index
    occupied: set[int] = set()
    for line in apps.stdout.splitlines():
        if not line.strip():
            continue
        uuid = line.split(",", 1)[0].strip()
        if uuid not in uuid_to_index:
            raise RuntimeError(f"compute process on unknown GPU {uuid}")
        occupied.add(uuid_to_index[uuid])
    return rows, occupied


def dataset_problems(training_root: Path, dataset_id: str, period_seconds: int) -> tuple[str | None, str | None]:
    root = training_root / "datasets" / dataset_id
    try:
        manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("dataset_id") != dataset_id:
            message = f"{dataset_id} manifest id is {manifest.get('dataset_id')!r}"
            return message, message
        if list(manifest.get("observation_keys") or []) != [OBSERVATION_KEY]:
            message = f"{dataset_id} observation_keys={manifest.get('observation_keys')!r}"
            return message, message
        if manifest.get("physical_definition_version") != PHYSICAL_VERSION:
            message = f"{dataset_id} physical={manifest.get('physical_definition_version')!r}"
            return message, message
        segment = manifest.get("segment_seconds", manifest.get("period_seconds"))
        if int(segment) != int(period_seconds):
            message = f"{dataset_id} segment_seconds={segment!r}"
            return message, message
        if manifest.get("dataset_status") not in (None, "ready"):
            message = f"{dataset_id} dataset_status={manifest.get('dataset_status')!r}"
            return message, message
        window = root / "sampling_seeds" / "seed_42" / "flow" / "window_data"
        split = json.loads((window / "split_manifest.json").read_text(encoding="utf-8"))
        expected_days = {"train": 20, "validation": 4, "test": 5}
        expected_samples = {"train": 5540, "validation": 1108, "test": 1385}
        for name, days in expected_days.items():
            if len(split[name]) != days:
                message = f"{dataset_id} {name} days={len(split[name])}"
                return message, message
        for name, count in expected_samples.items():
            if int(split[f"{name}_samples_per_rate"]) != count:
                message = f"{dataset_id} {name} samples={split[f'{name}_samples_per_rate']}"
                return message, message
        if int(split["num_nodes"]) != 1406 or int(split["n_his"]) != 12:
            message = f"{dataset_id} shape num_nodes={split.get('num_nodes')} n_his={split.get('n_his')}"
            return message, message
        if split.get("target_mode") != "last-observed-step":
            message = f"{dataset_id} target_mode={split.get('target_mode')!r}"
            return message, message
        rf_error = None
        for seed in EVAL_SEEDS:
            flow = root / "sampling_seeds" / f"seed_{seed}" / "flow" / "window_data" / OBSERVATION_KEY
            for split_name in ("train", "validation", "test"):
                if not (flow / split_name).is_dir():
                    message = f"missing {flow / split_name}"
                    return message, message
            prepared = root / "sampling_seeds" / f"seed_{seed}" / "model_prepared" / "RF" / OBSERVATION_KEY
            if rf_error is None and (not prepared.is_dir() or not any(prepared.glob("*.npz"))):
                rf_error = f"missing RF features {prepared}"
        return None, rf_error
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        message = f"{dataset_id} unreadable: {exc}"
        return message, message


def build_tasks(training_root: Path, results_root: Path) -> list[dict]:
    problems = {
        row["dataset_id"]: dataset_problems(training_root, row["dataset_id"], row["period_seconds"])
        for row in DATASETS
    }
    tasks: list[dict] = []
    order = 0
    for io in IOS:
        for dataset in DATASETS:
            dataset_error, rf_error = problems[dataset["dataset_id"]]
            for model in MODELS:
                run_id = new_run_id(model["name"], dataset["dataset_id"])
                run_dir = sampling_run_results_dir(
                    results_root,
                    io["input_type"],
                    io["target_type"],
                    OBSERVATION_KEY,
                    MODEL_SEED,
                    physical_version=PHYSICAL_VERSION,
                    dataset_id=dataset["dataset_id"],
                    model_name=model["name"],
                    experiment_id=EXPERIMENT_ID,
                    run_id=run_id,
                )
                error = dataset_error or (rf_error if model.get("cpu") else None)
                task = {
                    "order": order,
                    "task_id": f"{model['name']}__{dataset['dataset_id']}__{io['io']}",
                    "model": model["name"],
                    "script": model["script"],
                    "cpu": bool(model.get("cpu")),
                    "dataset_id": dataset["dataset_id"],
                    "period_seconds": dataset["period_seconds"],
                    "io": io["io"],
                    "input_type": io["input_type"],
                    "target_type": io["target_type"],
                    "train_sampling_seed": MODEL_SEED,
                    "model_seed": MODEL_SEED,
                    "eval_sampling_seeds": list(EVAL_SEEDS),
                    "experiment_id": EXPERIMENT_ID,
                    "observation_key": OBSERVATION_KEY,
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "config": "" if model.get("cpu") else str(config_path(model["name"])),
                    "status": "blocked" if error else "pending",
                    "attempt": 0,
                    "pid": None,
                    "gpu_index": None,
                    "gpu_uuid": None,
                    "compute_backend": "cpu" if model.get("cpu") else "cuda",
                    "gpu_reserved": True,
                    "command": [],
                    "log": "",
                    "started_at": None,
                    "finished_at": None,
                    "exit_code": None,
                    "error": error,
                }
                tasks.append(task)
                order += 1
    if len(tasks) != 144:
        raise RuntimeError(f"expected 144 tasks, built {len(tasks)}")
    identities = {(row["model"], row["dataset_id"], row["io"]) for row in tasks}
    run_dirs = {row["run_dir"] for row in tasks}
    if len(identities) != 144 or len(run_dirs) != 144:
        raise RuntimeError("task identity or run_dir collision")
    if any(row["model"] in {"MSTT", "pdformer"} for row in tasks):
        raise RuntimeError("MSTT and PDFormer are outside this campaign")
    return tasks


def command_for(task: dict, python: Path, training_root: Path, results_root: Path, threads: int) -> list[str]:
    if task["cpu"]:
        return [
            str(python),
            task["script"],
            "--dataset-id",
            task["dataset_id"],
            "--training-root",
            str(training_root),
            "--results-root",
            str(results_root / "RF"),
            "--penetrations",
            OBSERVATION_KEY,
            "--input-type",
            task["input_type"],
            "--target-type",
            task["target_type"],
            "--parallel-models",
            "1",
            "--n-jobs-per-model",
            str(threads),
            "--seed",
            str(MODEL_SEED),
            "--run-id",
            task["run_id"],
            "--experiment-id",
            EXPERIMENT_ID,
            "--n-clusters",
            "50",
        ]
    if not Path(task["config"]).is_file():
        raise FileNotFoundError(task["config"])
    return [
        str(python),
        task["script"],
        "--config",
        task["config"],
        "--dataset-id",
        task["dataset_id"],
        "--experiment-id",
        EXPERIMENT_ID,
        "--run-id",
        task["run_id"],
        "--model-seed",
        str(MODEL_SEED),
        "--seed",
        str(MODEL_SEED),
        "--rates",
        OBSERVATION_KEY,
        "--input-type",
        task["input_type"],
        "--target-type",
        task["target_type"],
        "--device",
        "cuda:0",
    ]


def child_env(task: dict, gpu_index: int, threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CODE)
    env["PYTHONUNBUFFERED"] = "1"
    text = str(threads)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = text
    if task["cpu"]:
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    return env


def five_seed_metrics(path: Path) -> dict | None:
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    per_seed = payload.get("per_seed")
    if not isinstance(per_seed, dict):
        return None
    try:
        seeds = tuple(int(seed) for seed in per_seed)
    except (TypeError, ValueError):
        return None
    if seeds != EVAL_SEEDS:
        return None
    mean = payload.get("mean")
    std = payload.get("std")
    if not isinstance(mean, dict) or not isinstance(std, dict):
        return None
    try:
        mae = float(mean["mae"])
    except (KeyError, TypeError, ValueError):
        return None
    if mae != mae or mae in (float("inf"), float("-inf")):
        return None
    return payload


def predictions_ready(task: dict) -> bool:
    run_dir = Path(task["run_dir"])
    for seed in EVAL_SEEDS:
        if task["cpu"]:
            directory = run_dir / "predictions" / f"seed_{seed}" / "test"
            if not directory.is_dir() or not any(directory.glob("*.npz")):
                return False
        else:
            path = run_dir / "predictions" / f"seed_{seed}" / "test.npz"
            if not path.is_file() or path.stat().st_size <= 0:
                return False
    return True


def identity_matches(task: dict) -> tuple[bool, str]:
    path = Path(task["run_dir"]) / "resolved_config.json"
    if not path.is_file():
        return False, "missing resolved_config.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"resolved_config.json unreadable: {exc}"
    checks = {
        "dataset_id": task["dataset_id"],
        "experiment_id": EXPERIMENT_ID,
        "run_id": task["run_id"],
        "model_name": task["model"],
    }
    for key, expected in checks.items():
        if str(payload.get(key)) != str(expected):
            return False, f"{key}={payload.get(key)!r}"
    rates = [str(item) for item in (payload.get("rates") or [])]
    if rates != [OBSERVATION_KEY]:
        return False, f"rates={rates}"
    if str(payload.get("input_type")) != task["input_type"] or str(payload.get("target_type")) != task["target_type"]:
        return False, "input/target mismatch"
    return True, ""


def artifacts_complete(task: dict) -> tuple[bool, str]:
    matched, reason = identity_matches(task)
    if not matched:
        return False, reason
    for split in ("validation", "test"):
        if five_seed_metrics(Path(task["run_dir"]) / f"{split}_sampling_metrics.json") is None:
            return False, f"missing five-seed {split} metrics"
    if not predictions_ready(task):
        return False, "missing five-seed predictions"
    if not task["cpu"]:
        checkpoint = Path(task["run_dir"]) / "best_checkpoint.pt"
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            return False, "missing best_checkpoint.pt"
    return True, ""


def blank_number(value: object) -> object:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def summary_row(task: dict) -> dict[str, object]:
    row: dict[str, object] = {column: None for column in SUMMARY_COLUMNS}
    row.update(
        {
            "model": task["model"],
            "dataset_id": task["dataset_id"],
            "period_seconds": task["period_seconds"],
            "io": task["io"],
            "train_sampling_seed": task["train_sampling_seed"],
            "model_seed": task["model_seed"],
            "eval_sampling_seeds": ",".join(str(seed) for seed in task["eval_sampling_seeds"]),
            "status": task["status"],
            "run_dir": task["run_dir"],
            "error": task.get("error") or None,
            "fit": "fit_once" if task["cpu"] else None,
        }
    )
    if task.get("started_at") and task.get("finished_at"):
        start = datetime.fromisoformat(task["started_at"])
        finish = datetime.fromisoformat(task["finished_at"])
        row["duration"] = round((finish - start).total_seconds(), 1)
    matched, _reason = identity_matches(task)
    if matched:
        validation = five_seed_metrics(Path(task["run_dir"]) / "validation_sampling_metrics.json")
        test = five_seed_metrics(Path(task["run_dir"]) / "test_sampling_metrics.json")
        if validation is not None:
            row["val_mae_mean"] = blank_number(validation["mean"].get("mae"))
            row["val_mae_std"] = blank_number(validation["std"].get("mae"))
        if test is not None:
            row["test_mae_mean"] = blank_number(test["mean"].get("mae"))
            row["test_mae_std"] = blank_number(test["std"].get("mae"))
            inflow_key = "inflow_mae" if "inflow_mae" in test["mean"] else "mae"
            if task["io"] == "A2A" and "inflow_mae" not in test["mean"]:
                inflow_key = ""
            if inflow_key:
                row["test_inflow_mae_mean"] = blank_number(test["mean"].get(inflow_key))
                row["test_inflow_mae_std"] = blank_number(test["std"].get(inflow_key))
            if task["io"] == "A2A":
                for state in ("outflow", "accumulation_start", "accumulation_end"):
                    row[f"test_{state}_mae_mean"] = blank_number(test["mean"].get(f"{state}_mae"))
                    row[f"test_{state}_mae_std"] = blank_number(test["std"].get(f"{state}_mae"))
        metadata = read_json(Path(task["run_dir"]) / "run_metadata.json")
        if isinstance(metadata, dict) and not task["cpu"] and metadata.get("best_epoch") is not None:
            row["best_epoch"] = metadata.get("best_epoch")
    return row


def write_summary(ops: Path, tasks: list[dict]) -> None:
    rows = [summary_row(task) for task in tasks]
    atomic_write_json(ops / "campaign_summary.json", rows)
    temporary = ops / "campaign_summary.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SUMMARY_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row[key] is None else row[key] for key in SUMMARY_COLUMNS})
    os.replace(temporary, ops / "campaign_summary.csv")


def save_tasks(ops: Path, tasks: list[dict], limit: int) -> None:
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task["status"]] = counts.get(task["status"], 0) + 1
    atomic_write_json(
        ops / "tasks.json",
        {
            "campaign_id": CAMPAIGN_ID,
            "experiment_id": EXPERIMENT_ID,
            "updated_at": stamp(),
            "max_gpus": limit,
            "counts": counts,
            "tasks": tasks,
        },
    )
    write_summary(ops, tasks)


def load_tasks(ops: Path) -> list[dict] | None:
    payload = read_json(ops / "tasks.json")
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        return None
    if payload.get("campaign_id") != CAMPAIGN_ID:
        raise RuntimeError(f"tasks.json campaign is {payload.get('campaign_id')!r}")
    tasks = payload["tasks"]
    if len(tasks) != 144:
        raise RuntimeError(f"tasks.json has {len(tasks)} rows")
    return tasks


def operations_dir(results_root: Path, campaign_id: str) -> Path:
    return results_root / "operations" / campaign_id


def drain_messages(ops: Path) -> object:
    inbox = ops / "messages.jsonl"
    if not inbox.is_file() or inbox.stat().st_size == 0:
        return None
    processing = ops / "messages.processing"
    os.replace(inbox, processing)
    last: object = None
    archive = ops / "messages.archive.jsonl"
    with archive.open("a", encoding="utf-8") as handle:
        for line in processing.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            handle.write(line + "\n")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                log_line(ops, f"message rejected: {exc}")
                continue
            if isinstance(payload, dict) and "max_gpus" in payload:
                last = payload["max_gpus"]
    processing.unlink()
    return last


def refresh_cap(ops: Path, gpu_count: int, previous: int) -> int:
    control_path = ops / "control.json"
    state_path = ops / "scheduler_state.json"
    state = read_json(state_path) or {}
    now = local_now()
    if not control_path.is_file():
        initial = DEFAULT_MAX_GPUS if gpu_count <= 0 else min(DEFAULT_MAX_GPUS, gpu_count)
        atomic_write_json(control_path, {"max_gpus": initial})
        log_line(ops, f"CONTROL created {control_path} max_gpus={initial}")
    if morning_reset_due(now, state.get("morning_reset_date")):
        atomic_write_json(control_path, {"max_gpus": MORNING_CAP})
        state["morning_reset_date"] = now.date().isoformat()
        atomic_write_json(state_path, state)
        log_line(ops, f"morning reset max_gpus={MORNING_CAP} local_time={stamp(now)}")
    message = drain_messages(ops)
    if message is not None:
        try:
            parsed = parse_cap(message, gpu_count)
        except ValueError as exc:
            log_line(ops, f"message rejected ({exc})")
        else:
            atomic_write_json(control_path, {"max_gpus": parsed})
            log_line(ops, f"message set max_gpus={parsed}")
    try:
        payload = read_json(control_path) or {}
        return parse_cap(payload.get("max_gpus"), gpu_count)
    except ValueError as exc:
        log_line(ops, f"CONTROL rejected ({exc}); keeping max_gpus={previous}")
        return previous


def git_revision() -> str:
    try:
        done = subprocess.run(
            ["git", "-C", str(CODE), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return done.stdout.strip()


def write_manifest(ops: Path, training_root: Path, results_root: Path) -> None:
    path = ops / "campaign_manifest.json"
    if path.is_file():
        return
    atomic_write_json(
        path,
        {
            "campaign_id": CAMPAIGN_ID,
            "experiment_id": EXPERIMENT_ID,
            "observation_key": OBSERVATION_KEY,
            "physical_definition_version": PHYSICAL_VERSION,
            "train_sampling_seed": MODEL_SEED,
            "model_seed": MODEL_SEED,
            "evaluation_sampling_seeds": list(EVAL_SEEDS),
            "models": [row["name"] for row in MODELS],
            "excluded_models": ["MSTT", "pdformer"],
            "datasets": list(DATASETS),
            "ios": list(IOS),
            "code_revision": git_revision(),
            "training_root": str(training_root),
            "results_root": str(results_root),
            "poll_seconds": POLL_SECONDS,
            "morning_hour": MORNING_HOUR,
            "morning_cap": MORNING_CAP,
            "empty_gpu": "no compute process; memory is ignored",
        },
    )


def adopt_running(tasks: list[dict]) -> dict[int, str]:
    running: dict[int, str] = {}
    for task in tasks:
        if task["status"] != "running":
            continue
        pid = int(task["pid"] or 0)
        gpu = task.get("gpu_index")
        if pid_alive(pid) and isinstance(gpu, int):
            running[gpu] = task["task_id"]
            continue
        complete, reason = artifacts_complete(task)
        task["finished_at"] = stamp()
        if complete:
            task["status"] = "completed"
            task["error"] = None
        elif int(task.get("attempt") or 0) >= 3:
            task["status"] = "failed"
            task["error"] = reason or "process disappeared"
        else:
            task["status"] = "pending"
            task["error"] = reason or "process disappeared before completion"
            task["pid"] = None
            task["gpu_index"] = None
            task["gpu_uuid"] = None
    return running


def finish_task(task: dict, exit_code: int | None) -> None:
    complete, reason = artifacts_complete(task)
    task["finished_at"] = stamp()
    task["exit_code"] = exit_code
    # A restarted scheduler tracks adopted jobs by pid and does not have the
    # child exit status. A finished run directory is enough in that case.
    if complete and exit_code in (0, None):
        task["status"] = "completed"
        task["error"] = None
    else:
        task["status"] = "failed"
        task["error"] = reason or f"exit {exit_code}"


def launch_task(
    ops: Path,
    task: dict,
    gpu: dict[str, object],
    python: Path,
    training_root: Path,
    results_root: Path,
    threads: int,
) -> subprocess.Popen:
    command = command_for(task, python, training_root, results_root, threads)
    log_path = ops / "logs" / f"{task['task_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    handle.write(
        f"start_time={stamp()}\nphysical_gpu={gpu['index']}\ngpu_uuid={gpu['uuid']}\n"
        f"threads={threads}\ncommand={' '.join(command)}\n"
    )
    handle.flush()
    proc = subprocess.Popen(
        command,
        cwd=CODE,
        env=child_env(task, int(gpu["index"]), threads),
        stdout=handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    handle.close()
    task["status"] = "running"
    task["attempt"] = int(task.get("attempt") or 0) + 1
    task["pid"] = proc.pid
    task["gpu_index"] = int(gpu["index"])
    task["gpu_uuid"] = str(gpu["uuid"])
    task["command"] = command
    task["log"] = str(log_path)
    task["threads"] = threads
    task["started_at"] = task.get("started_at") or stamp()
    task["finished_at"] = None
    task["exit_code"] = None
    task["error"] = None
    return proc


def task_map(tasks: list[dict]) -> dict[str, dict]:
    return {task["task_id"]: task for task in tasks}


def counts_of(tasks: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task["status"]] = counts.get(task["status"], 0) + 1
    return counts


def format_running(tasks: list[dict]) -> str:
    rows = []
    for task in tasks:
        if task["status"] == "running":
            rows.append(
                f"{task['gpu_index']}:{task['task_id']}:pid={task['pid']}"
            )
    return ",".join(rows) if rows else "[]"


def acquire_lock(ops: Path):
    ops.mkdir(parents=True, exist_ok=True)
    handle = (ops / "scheduler.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        owner = handle.read().strip() or "unknown"
        handle.close()
        raise SystemExit(f"scheduler already running, lock pid {owner}")
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def run_campaign(ops: Path, training_root: Path, results_root: Path) -> None:
    global RELOAD_REQUESTED
    lock = acquire_lock(ops)
    signal.signal(signal.SIGHUP, request_reload)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    python = training_python()
    if not python.is_file():
        raise SystemExit(f"missing training python: {python}")
    write_manifest(ops, training_root, results_root)
    existing = load_tasks(ops)
    created = existing is None
    tasks = existing if existing is not None else build_tasks(training_root, results_root)
    running = adopt_running(tasks)
    procs: dict[str, subprocess.Popen] = {}
    limit = 0
    log_line(
        ops,
        f"scheduler start pid={os.getpid()} campaign={CAMPAIGN_ID} "
        f"created={created} counts={counts_of(tasks)} python={python}"
    )
    save_tasks(ops, tasks, limit)
    by_id = task_map(tasks)
    while not STOP_REQUESTED:
        if RELOAD_REQUESTED:
            RELOAD_REQUESTED = False
            log_line(ops, "SIGHUP reload")
        try:
            gpus, occupied = query_gpus()
        except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
            log_line(ops, f"GPU query failed: {exc}")
            gpus, occupied = [], set()
        limit = refresh_cap(ops, len(gpus), limit)
        for gpu_index, task_id in list(running.items()):
            task = by_id[task_id]
            proc = procs.get(task_id)
            if proc is not None:
                code = proc.poll()
                if code is None:
                    continue
                finish_task(task, code)
                log_line(ops, f"GPU {gpu_index} {task['status'].upper()} {task_id} exit={code} {task.get('error') or ''}")
                running.pop(gpu_index, None)
                procs.pop(task_id, None)
            elif not pid_alive(int(task.get("pid") or 0)):
                finish_task(task, task.get("exit_code"))
                log_line(ops, f"GPU {gpu_index} {task['status'].upper()} {task_id} {task.get('error') or ''}")
                running.pop(gpu_index, None)
        reserved = set(running)
        free = [row for row in gpus if int(row["index"]) not in occupied and int(row["index"]) not in reserved]
        ram = mem_available_bytes()
        threads = thread_budget(limit)
        pending = [task for task in tasks if task["status"] == "pending"]
        if ram < MIN_FREE_RAM_BYTES:
            log_line(ops, f"skip launch MemAvailable={ram} below {MIN_FREE_RAM_BYTES}")
            free = []
        while pending and free and len(running) < limit:
            task = pending.pop(0)
            gpu = free.pop(0)
            try:
                listed, occupied_now = query_gpus()
            except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
                log_line(ops, f"GPU query failed before launch: {exc}")
                pending.insert(0, task)
                free.insert(0, gpu)
                break
            current = {int(row["index"]): row for row in listed}
            index = int(gpu["index"])
            if index not in current or index in occupied_now or index in running:
                log_line(ops, f"GPU {index} was taken before launch")
                pending.insert(0, task)
                continue
            try:
                proc = launch_task(ops, task, current[index], python, training_root, results_root, threads)
            except Exception as exc:
                task["status"] = "failed"
                task["error"] = f"launch failed: {exc}"
                task["finished_at"] = stamp()
                log_line(ops, f"GPU {index} FAILED {task['task_id']} {exc}")
                continue
            running[index] = task["task_id"]
            procs[task["task_id"]] = proc
            log_line(
                ops,
                f"GPU {index} START {task['task_id']} pid={proc.pid} "
                f"backend={task['compute_backend']} threads={threads}"
            )
        save_tasks(ops, tasks, limit)
        pending_count = sum(task["status"] == "pending" for task in tasks)
        log_line(
            ops,
            f"poll local={stamp()} max={limit} running={format_running(tasks)} queued={pending_count}",
        )
        if pending_count == 0 and not running:
            break
        wait_poll(POLL_SECONDS)
    try:
        save_tasks(ops, tasks, limit)
        log_line(ops, f"scheduler stop counts={counts_of(tasks)}")
    finally:
        lock.seek(0)
        lock.truncate()
        lock.close()


def print_status(ops: Path) -> None:
    payload = read_json(ops / "tasks.json")
    control = read_json(ops / "control.json") or {}
    state = read_json(ops / "scheduler_state.json") or {}
    if not isinstance(payload, dict):
        print(f"no queue at {ops / 'tasks.json'}")
        return
    counts = payload.get("counts") or {}
    print(f"campaign={payload.get('campaign_id')} updated={payload.get('updated_at')}")
    print(f"max_gpus={control.get('max_gpus')} morning_reset_date={state.get('morning_reset_date')}")
    print(f"counts={counts}")
    print(f"control={ops / 'control.json'}")
    print(f"messages={ops / 'messages.jsonl'}")
    print(f"log={ops / 'scheduler.log'}")
    lock = read_text(ops / "scheduler.lock")
    print(f"scheduler_pid={lock}")
    for task in payload.get("tasks") or []:
        if task.get("status") == "running":
            print(
                f"RUNNING {task['task_id']} gpu={task.get('gpu_index')} "
                f"uuid={task.get('gpu_uuid')} pid={task.get('pid')} log={task.get('log')}"
            )


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def requeue_failed(ops: Path) -> None:
    tasks = load_tasks(ops)
    if tasks is None:
        raise SystemExit(f"no queue at {ops / 'tasks.json'}")
    changed = 0
    for task in tasks:
        if task["status"] != "failed":
            continue
        complete, _reason = artifacts_complete(task)
        if complete:
            task["status"] = "completed"
            task["error"] = None
            continue
        task["status"] = "pending"
        task["pid"] = None
        task["gpu_index"] = None
        task["gpu_uuid"] = None
        task["exit_code"] = None
        task["finished_at"] = None
        changed += 1
    control = read_json(ops / "control.json") or {}
    limit = control.get("max_gpus") if isinstance(control.get("max_gpus"), int) else 0
    save_tasks(ops, tasks, int(limit))
    print(f"requeued {changed}")


def print_plan(training_root: Path, results_root: Path) -> None:
    tasks = build_tasks(training_root, results_root)
    print(f"tasks={len(tasks)} counts={counts_of(tasks)} preview_run_ids_are_not_saved")
    for task in tasks:
        if task["status"] == "blocked":
            print(f"BLOCKED {task['task_id']} {task['error']}")
    print(f"example_run_dir={tasks[0]['run_dir']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Schedule the 144 variable-penetration baseline tasks.")
    parser.add_argument("command", choices=("run", "status", "plan", "requeue-failed"))
    parser.add_argument("--campaign-id", default=CAMPAIGN_ID)
    args = parser.parse_args(argv)
    if args.campaign_id != CAMPAIGN_ID:
        raise SystemExit(f"this entry only resumes {CAMPAIGN_ID}")
    training_root = xuancheng26_training_3s_root()
    results_root = xuancheng26_training_results_3s_root()
    ops = operations_dir(results_root, CAMPAIGN_ID)
    if args.command == "plan":
        print_plan(training_root, results_root)
        return 0
    if args.command == "status":
        print_status(ops)
        return 0
    if args.command == "requeue-failed":
        requeue_failed(ops)
        return 0
    run_campaign(ops, training_root, results_root)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
