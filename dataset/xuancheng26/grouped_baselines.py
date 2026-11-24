"""Grouped Xuancheng26 baseline campaign for two penetration bands and five training samples.

One group is one model, one dataset, and one training sampling seed. The group
runs its missing input/output items one at a time. Seed 42 of the model is the
initialization seed. It is not evidence that the training vehicles were sample 42.

This module plans and can run the campaign. Importing it does not start training.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from data_root import (  # noqa: E402
    model_report_dir,
    sampling_run_results_dir,
    xuancheng26_training_3s_root,
    xuancheng26_training_results_3s_root,
)


def _load_legacy_scheduler():
    path = Path(__file__).resolve().with_name("run_variable_baselines.py")
    spec = importlib.util.spec_from_file_location("run_variable_baselines", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY = _load_legacy_scheduler()

CAMPAIGN_ID = "grouped_p5_p15_p5_p35_train5"
PHYSICAL_VERSION = "independent_fcd_v1"
MODEL_SEED = 42
TRAIN_SEEDS = (42, 43, 44, 45, 46)
EVAL_SEEDS = (42, 43, 44, 45, 46)
POLL_SECONDS = 30
DEFAULT_MAX_GPUS = 4
MAX_ATTEMPTS = 2
MIN_FREE_RAM_BYTES = 32 * 1024**3
BANDS = ("p5_p15", "p5_p35")
PERIODS = (("30min", 1800), ("1h", 3600), ("2h", 7200), ("day", 86400))
STATES = ("inflow", "outflow", "accumulation_start", "accumulation_end")
EXPECTED_DATASETS = tuple(
    {"dataset_id": f"variable_{name}_{band}", "period_seconds": seconds, "observation_key": band}
    for band in BANDS
    for name, seconds in PERIODS
)
RUNNABLE = {"pending"}
TERMINAL_SUCCESS = {"reusable", "completed"}
HELD = {"active", "blocked", "conflict", "unreadable", "unverified"}


class CampaignSignals:
    def __init__(self) -> None:
        self.reload = False
        self.stop = False


SIGNALS = CampaignSignals()


def io_pairs() -> list[dict[str, str]]:
    """Fourteen training pairs in the comparison order. Identity is the type pair."""
    rows: list[dict[str, str]] = []
    for state in STATES:
        rows.append({"input_type": state, "target_type": state, "family": "single"})
    rows.append({"input_type": "all", "target_type": "all", "family": "A2A"})
    for state in STATES:
        rows.append({"input_type": "all", "target_type": state, "family": "A2S"})
    rows.append({"input_type": "all_sp_c", "target_type": "all", "family": "ASC2A"})
    for state in STATES:
        rows.append({"input_type": "all_sp_c", "target_type": state, "family": "ASC2S"})
    if len(rows) != 14 or len({(row["input_type"], row["target_type"]) for row in rows}) != 14:
        raise RuntimeError("the campaign matrix is not 14 unique input/target pairs")
    return rows


def experiment_id_for(observation_key: str, train_sampling_seed: int) -> str:
    return f"variable_{observation_key}_train{int(train_sampling_seed)}_eval5"


def new_attempt_id() -> str:
    """Run id unique inside one second. The logical experiment id stays separate."""
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex}"


def local_now() -> datetime:
    return datetime.now().astimezone()


def stamp(when: datetime | None = None) -> str:
    return (when or local_now()).isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict | list | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def log_line(ops: Path, message: str) -> None:
    line = f"{stamp()} {message}\n"
    ops.mkdir(parents=True, exist_ok=True)
    with (ops / "scheduler.log").open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
    print(line, end="", flush=True)


def operations_dir(results_root: Path, campaign_id: str = CAMPAIGN_ID) -> Path:
    return Path(results_root) / "operations" / campaign_id


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def build_groups(models: tuple[dict, ...] | None = None) -> list[dict]:
    catalog = tuple(models) if models is not None else tuple(LEGACY.MODELS)
    pairs = io_pairs()
    groups: list[dict] = []
    for model in catalog:
        for dataset in EXPECTED_DATASETS:
            for train_seed in TRAIN_SEEDS:
                group_id = f"{model['name']}__{dataset['dataset_id']}__train{train_seed}"
                items = []
                for pair in pairs:
                    items.append(
                        {
                            "item_id": f"{group_id}__{pair['input_type']}__{pair['target_type']}",
                            "model": model["name"],
                            "script": model["script"],
                            "cpu": bool(model.get("cpu")),
                            "dataset_id": dataset["dataset_id"],
                            "period_seconds": dataset["period_seconds"],
                            "observation_key": dataset["observation_key"],
                            "train_sampling_seed": int(train_seed),
                            "model_seed": MODEL_SEED,
                            "evaluation_sampling_seeds": list(EVAL_SEEDS),
                            "input_type": pair["input_type"],
                            "target_type": pair["target_type"],
                            "family": pair["family"],
                            "experiment_id": experiment_id_for(dataset["observation_key"], train_seed),
                            "status": "pending",
                            "attempt": 0,
                            "run_id": None,
                            "run_dir": None,
                            "reused_run_dir": None,
                            "source_experiment_id": None,
                            "evidence": None,
                            "reason": None,
                            "pid": None,
                            "command": [],
                            "log": None,
                            "started_at": None,
                            "finished_at": None,
                            "exit_code": None,
                            "gpu_index": None,
                            "gpu_uuid": None,
                        }
                    )
                groups.append(
                    {
                        "group_id": group_id,
                        "model": model["name"],
                        "script": model["script"],
                        "cpu": bool(model.get("cpu")),
                        "dataset_id": dataset["dataset_id"],
                        "period_seconds": dataset["period_seconds"],
                        "observation_key": dataset["observation_key"],
                        "train_sampling_seed": int(train_seed),
                        "model_seed": MODEL_SEED,
                        "status": "pending",
                        "gpu_index": None,
                        "gpu_uuid": None,
                        "items": items,
                    }
                )
    if len(groups) != 480 or sum(len(group["items"]) for group in groups) != 6720:
        raise RuntimeError("campaign matrix is not 480 groups and 6720 items")
    identities = {
        (
            item["model"],
            item["dataset_id"],
            item["input_type"],
            item["target_type"],
            item["train_sampling_seed"],
            item["model_seed"],
        )
        for group in groups
        for item in group["items"]
    }
    if len(identities) != 6720:
        raise RuntimeError("logical experiment identities are not unique")
    if any(group["model"] in {"MSTT", "pdformer"} for group in groups):
        raise RuntimeError("MSTT and PDFormer are outside this campaign")
    return groups


def dataset_report(training_root: Path) -> dict[str, dict]:
    """Read the eight published manifests. Missing mounts stay unverified."""
    root = Path(training_root)
    if not root.is_dir():
        return {
            row["dataset_id"]: {
                "status": "unverified",
                "detail": f"training root is not a directory: {root}",
            }
            for row in EXPECTED_DATASETS
        }
    found: dict[str, dict] = {}
    for row in EXPECTED_DATASETS:
        dataset_id = row["dataset_id"]
        manifest_path = root / "datasets" / dataset_id / "dataset_manifest.json"
        try:
            if not manifest_path.is_file():
                found[dataset_id] = {"status": "missing", "detail": f"missing {manifest_path}"}
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            problems = []
            if manifest.get("dataset_id") != dataset_id:
                problems.append(f"dataset_id={manifest.get('dataset_id')!r}")
            keys = [str(item) for item in manifest.get("observation_keys") or []]
            if keys != [row["observation_key"]]:
                problems.append(f"observation_keys={keys}")
            if manifest.get("physical_definition_version") != PHYSICAL_VERSION:
                problems.append(f"physical={manifest.get('physical_definition_version')!r}")
            segment = manifest.get("segment_seconds", manifest.get("period_seconds"))
            if int(segment) != int(row["period_seconds"]):
                problems.append(f"segment_seconds={segment!r}")
            window = root / "datasets" / dataset_id / "sampling_seeds" / "seed_42" / "flow" / "window_data"
            split_path = window / "split_manifest.json"
            if not split_path.is_file():
                problems.append("missing seed_42 split_manifest")
            else:
                split = json.loads(split_path.read_text(encoding="utf-8"))
                expected_days = {"train": 20, "validation": 4, "test": 5}
                expected_samples = {"train": 5540, "validation": 1108, "test": 1385}
                for name, days in expected_days.items():
                    if len(split[name]) != days:
                        problems.append(f"{name} days={len(split[name])}")
                for name, count in expected_samples.items():
                    if int(split[f"{name}_samples_per_rate"]) != count:
                        problems.append(f"{name} samples={split[f'{name}_samples_per_rate']}")
                if int(split.get("num_nodes", -1)) != 1406 or int(split.get("n_his", -1)) != 12:
                    problems.append(f"shape num_nodes={split.get('num_nodes')} n_his={split.get('n_his')}")
                if split.get("target_mode") != "last-observed-step":
                    problems.append(f"target_mode={split.get('target_mode')!r}")
            found[dataset_id] = {
                "status": "ready" if not problems else "mismatch",
                "detail": "; ".join(problems),
                "observation_key": row["observation_key"],
                "period_seconds": row["period_seconds"],
            }
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            found[dataset_id] = {"status": "unreadable", "detail": f"{dataset_id} unreadable: {exc}"}
    return found


def _seed_assets(training_root: Path, dataset_id: str, seed: int, observation_key: str) -> dict[str, bool]:
    root = Path(training_root) / "datasets" / dataset_id / "sampling_seeds" / f"seed_{int(seed)}"
    window = root / "flow" / "window_data" / observation_key
    partial = root / "flow" / "partial_flow" / observation_key
    speed = root / "model_prepared" / "RF" / observation_key
    traffic = root / "model_prepared" / "traffic_observations" / observation_key
    train = window / "train"
    return {
        "window": window.is_dir() and train.is_dir(),
        "asc": train.is_dir() and any(train.glob("*/all_sp_c.npz")),
        "partial": partial.is_dir() and any(partial.glob("*/*.csv")),
        "rf_speed": speed.is_dir() and any(speed.glob("*.npz")),
        "traffic": traffic.is_dir() and any(traffic.glob("*.npz")),
        "asc_scaler": (Path(training_root) / "datasets" / dataset_id / "all_sp_c_normalization.json").is_file(),
    }


def data_block(item: dict, training_root: Path, dataset_status: str, assets: dict[str, bool]) -> str | None:
    if dataset_status == "unverified":
        return None
    if dataset_status != "ready":
        return f"dataset {item['dataset_id']} {dataset_status}"
    if item["input_type"] == "all_sp_c" and not (assets["asc"] and assets["asc_scaler"]):
        return f"missing all_sp_c windows or scaler for seed {item['train_sampling_seed']}"
    if not assets["window"]:
        return f"missing windows for seed {item['train_sampling_seed']}"
    if item["cpu"]:
        if not assets["partial"]:
            return f"missing RF partial flow for seed {item['train_sampling_seed']}"
        if not assets["rf_speed"]:
            return f"missing RF speed features for seed {item['train_sampling_seed']}"
        if item["input_type"] == "all_sp_c" and not assets["traffic"]:
            return f"missing ASC raw observations for seed {item['train_sampling_seed']}"
    return None


def train_seed_evidence(resolved: dict, experiment_id: str) -> tuple[str, int | None, str]:
    """Require two agreeing signals. A result directory named seed_42 is not one of them."""
    signals: list[tuple[str, int]] = []
    if resolved.get("train_sampling_seed") is not None:
        try:
            signals.append(("resolved_train_sampling_seed", int(resolved["train_sampling_seed"])))
        except (TypeError, ValueError):
            return "conflict", None, "train_sampling_seed is not an integer"
    for label, value in (
        ("data_root", resolved.get("data_root")),
        ("normalization", resolved.get("normalization")),
        ("split_manifest", resolved.get("split_manifest")),
    ):
        match = re.search(r"/seed_(\d+)/flow/window_data(?:/|$)", str(value or ""))
        if match:
            signals.append((label, int(match.group(1))))
    experiment_match = re.search(r"train(\d+)_eval5$", str(experiment_id))
    if experiment_match:
        signals.append(("experiment_id", int(experiment_match.group(1))))
    if not signals:
        return "unproven", None, "no training-sample evidence"
    seeds = {seed for _name, seed in signals}
    detail = ",".join(f"{name}={seed}" for name, seed in signals)
    if len(seeds) != 1:
        return "conflict", None, f"training-sample evidence disagrees: {detail}"
    if len({name for name, _seed in signals}) < 2:
        return "unproven", next(iter(seeds)), f"only one training-sample signal: {detail}"
    seed = next(iter(seeds))
    if seed not in TRAIN_SEEDS:
        return "conflict", seed, f"training sample {seed} is outside 42-46"
    return "proven", seed, detail


def _finite_mae(payload: dict | None) -> bool:
    if not isinstance(payload, dict):
        return False
    per_seed = payload.get("per_seed")
    if not isinstance(per_seed, dict):
        return False
    try:
        seeds = tuple(int(seed) for seed in per_seed)
    except (TypeError, ValueError):
        return False
    if seeds != EVAL_SEEDS:
        return False
    mean = payload.get("mean")
    std = payload.get("std")
    if not isinstance(mean, dict) or not isinstance(std, dict):
        return False
    try:
        mae = float(mean["mae"])
        spread = float(std["mae"])
    except (KeyError, TypeError, ValueError):
        return False
    return mae == mae and spread == spread and mae not in (float("inf"), float("-inf")) and spread not in (
        float("inf"),
        float("-inf"),
    )


def _metrics_ready(run_dir: Path) -> bool:
    for split in ("validation", "test"):
        path = run_dir / f"{split}_sampling_metrics.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        except (OSError, json.JSONDecodeError):
            return False
        if not _finite_mae(payload):
            return False
    return True


def _predictions_ready(run_dir: Path, cpu: bool) -> bool:
    for seed in EVAL_SEEDS:
        if cpu:
            directory = run_dir / "predictions" / f"seed_{seed}" / "test"
            if not directory.is_dir() or not any(directory.glob("*.npz")):
                return False
        else:
            path = run_dir / "predictions" / f"seed_{seed}" / "test.npz"
            if not path.is_file() or path.stat().st_size <= 0:
                return False
    return True


def _rf_retained(run_dir: Path) -> bool:
    discarded = run_dir / "model_discarded.json"
    if discarded.is_file() and discarded.stat().st_size > 0:
        try:
            payload = json.loads(discarded.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return bool(payload.get("discarded"))
    model = run_dir / "model"
    return model.is_dir() and any(model.rglob("*"))


def _smoke_or_debug(run_dir: Path, resolved: dict) -> str | None:
    text = "/".join(run_dir.parts).lower()
    experiment = str(resolved.get("experiment_id") or run_dir.parent.parent.name)
    run_id = str(resolved.get("run_id") or run_dir.name)
    blob = f"{text} {experiment} {run_id}".lower()
    if "smoke" in blob or "/debug/" in blob or "debug_short" in blob:
        return "smoke or debug path"
    epochs = resolved.get("epochs")
    if epochs is not None:
        try:
            if int(epochs) <= 1:
                return "epochs<=1"
        except (TypeError, ValueError):
            return "epochs is not an integer"
    return None


def classify_run(run_dir: Path) -> dict:
    """Read one finished or partial run. Does not treat model seed_42 as the training sample."""
    run_dir = Path(run_dir)
    resolved_path = run_dir / "resolved_config.json"
    try:
        if not resolved_path.is_file():
            return {"status": "incomplete", "reason": "missing resolved_config.json", "run_dir": str(run_dir)}
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "unreadable", "reason": str(exc), "run_dir": str(run_dir)}
    if not isinstance(resolved, dict):
        return {"status": "unreadable", "reason": "resolved_config is not an object", "run_dir": str(run_dir)}
    debug = _smoke_or_debug(run_dir, resolved)
    if debug:
        return {"status": "ignored", "reason": debug, "run_dir": str(run_dir), "resolved": resolved}
    proof, train_seed, evidence = train_seed_evidence(resolved, str(resolved.get("experiment_id") or ""))
    try:
        model_seed = int(resolved.get("model_seed", resolved.get("seed", -1)))
    except (TypeError, ValueError):
        model_seed = -1
    directory_model_seed = run_dir.parent.name
    record = {
        "status": proof,
        "reason": evidence,
        "run_dir": str(run_dir),
        "resolved": resolved,
        "train_sampling_seed": train_seed,
        "model_seed": model_seed,
        "model": str(resolved.get("model_name") or ""),
        "dataset_id": str(resolved.get("dataset_id") or ""),
        "input_type": str(resolved.get("input_type") or ""),
        "target_type": str(resolved.get("target_type") or ""),
        "experiment_id": str(resolved.get("experiment_id") or ""),
        "run_id": str(resolved.get("run_id") or run_dir.name),
        "directory_model_seed": directory_model_seed,
        "physical_definition_version": str(resolved.get("physical_definition_version") or ""),
    }
    if proof != "proven":
        return record
    if directory_model_seed != f"seed_{model_seed}":
        record["status"] = "conflict"
        record["reason"] = f"directory {directory_model_seed} != model_seed {model_seed}"
        return record
    if model_seed != MODEL_SEED:
        record["status"] = "ignored"
        record["reason"] = f"model_seed={model_seed}"
        return record
    selection = resolved.get("checkpoint_selection")
    if selection not in (None, "mean_validation_mae", "single_fit_no_epoch_selection"):
        record["status"] = "ignored"
        record["reason"] = f"checkpoint_selection={selection}"
        return record
    rates = [str(item) for item in (resolved.get("rates") or [])]
    if rates and rates != [str(run_dir.parents[2].name)]:
        record["status"] = "conflict"
        record["reason"] = f"rates={rates} directory={run_dir.parents[2].name}"
        return record
    cpu = record["model"] == "RF"
    if not _metrics_ready(run_dir) or not _predictions_ready(run_dir, cpu):
        record["status"] = "incomplete"
        record["reason"] = "missing five-seed metrics or predictions"
        return record
    if cpu and not _rf_retained(run_dir):
        record["status"] = "incomplete"
        record["reason"] = "RF predictions exist without a discard record or fitted model"
        return record
    if not cpu:
        checkpoint = run_dir / "best_checkpoint.pt"
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            record["status"] = "incomplete"
            record["reason"] = "missing best_checkpoint.pt"
            return record
    record["status"] = "reusable"
    return record


def _result_identity(record: dict) -> tuple | None:
    if record.get("train_sampling_seed") is None:
        return None
    return (
        record.get("model"),
        record.get("dataset_id"),
        record.get("input_type"),
        record.get("target_type"),
        int(record["train_sampling_seed"]),
        int(record.get("model_seed") or -1),
    )


def scan_results(results_root: Path, groups: list[dict]) -> tuple[str, list[dict], list[dict]]:
    """Index existing runs. A missing results root is unverified, not zero completed."""
    root = Path(results_root)
    if not root.exists():
        return "unverified", [], [{"path": str(root), "reason": "results root is not mounted"}]
    anomalies: list[dict] = []
    found: list[dict] = []
    datasets = {item["dataset_id"] for group in groups for item in group["items"]}
    models = {group["model"] for group in groups}
    for dataset_id in sorted(datasets):
        base = root / "datasets" / dataset_id / f"physical_{PHYSICAL_VERSION}"
        if not base.is_dir():
            continue
        for model in sorted(models):
            model_dir = base / model
            if not model_dir.is_dir():
                continue
            try:
                task_dirs = [path for path in model_dir.iterdir() if path.is_dir() and path.name.startswith("in_")]
            except OSError as exc:
                anomalies.append({"path": str(model_dir), "reason": str(exc)})
                continue
            for task_dir in task_dirs:
                try:
                    run_dirs = [
                        run_dir
                        for key_dir in task_dir.iterdir()
                        if key_dir.is_dir()
                        for experiment_dir in key_dir.iterdir()
                        if experiment_dir.is_dir()
                        for seed_dir in experiment_dir.glob("seed_*")
                        if seed_dir.is_dir()
                        for run_dir in seed_dir.iterdir()
                        if run_dir.is_dir()
                    ]
                except OSError as exc:
                    anomalies.append({"path": str(task_dir), "reason": str(exc)})
                    continue
                for run_dir in run_dirs:
                    record = classify_run(run_dir)
                    record["model"] = record.get("model") or model
                    record["dataset_id"] = record.get("dataset_id") or dataset_id
                    if record["status"] == "unreadable":
                        anomalies.append({"path": str(run_dir), "reason": record["reason"]})
                    found.append(record)
    return "ok", found, anomalies


def live_commands(reader=None) -> list[dict]:
    if reader is not None:
        return list(reader())
    rows = []
    proc = Path("/proc")
    if not proc.is_dir():
        return rows
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if "train_r_only" in command or "run_rf.py" in command:
            rows.append({"pid": int(entry.name), "command": command})
    return rows


def command_matches_item(command: str, item: dict) -> bool:
    text = str(command)
    needed = (
        item["dataset_id"],
        item["script"],
        f"--input-type {item['input_type']}",
        f"--target-type {item['target_type']}",
        f"--train-sampling-seed {item['train_sampling_seed']}",
    )
    if item.get("run_id") and str(item["run_id"]) in text:
        return True
    return all(part in text for part in needed)


def apply_inventory(
    groups: list[dict],
    *,
    training_root: Path,
    results_root: Path,
    processes: list[dict] | None = None,
) -> dict:
    inventory_status, records, anomalies = scan_results(results_root, groups)
    datasets = dataset_report(training_root)
    reusable: dict[tuple, list[dict]] = {}
    conflicts: dict[tuple, list[dict]] = {}
    for record in records:
        if record["status"] == "conflict":
            identity = _result_identity(record)
            if identity is not None:
                conflicts.setdefault(identity, []).append(record)
            anomalies.append({"path": record["run_dir"], "reason": record["reason"]})
            continue
        if record["status"] != "reusable":
            if record["status"] in {"incomplete", "unproven"}:
                anomalies.append(
                    {
                        "path": record.get("run_dir"),
                        "reason": f"{record['status']}: {record.get('reason')}",
                        "safe_to_rerun": record["status"] != "unreadable",
                    }
                )
            continue
        identity = _result_identity(record)
        if identity is None:
            continue
        reusable.setdefault(identity, []).append(record)
    asset_cache: dict[tuple, dict[str, bool]] = {}
    training_available = Path(training_root).is_dir()
    for group in groups:
        for item in group["items"]:
            if inventory_status == "unverified":
                item["status"] = "unverified"
                item["reason"] = "results or training mount was not available"
                continue
            identity = (
                item["model"],
                item["dataset_id"],
                item["input_type"],
                item["target_type"],
                int(item["train_sampling_seed"]),
                int(item["model_seed"]),
            )
            if identity in conflicts:
                item["status"] = "conflict"
                item["reason"] = conflicts[identity][0]["reason"]
                continue
            candidates = reusable.get(identity) or []
            if candidates:
                chosen = sorted(
                    candidates,
                    key=lambda record: (
                        0 if record["experiment_id"] == item["experiment_id"] else 1,
                        record["run_dir"],
                    ),
                )[0]
                item["status"] = "reusable"
                item["reused_run_dir"] = chosen["run_dir"]
                item["run_dir"] = chosen["run_dir"]
                item["source_experiment_id"] = chosen["experiment_id"]
                item["evidence"] = chosen["reason"]
                item["reason"] = None
                continue
            dataset = datasets.get(item["dataset_id"], {"status": "missing", "detail": "not checked"})
            if dataset["status"] == "unverified":
                item["status"] = "unverified"
                item["reason"] = dataset["detail"]
                continue
            cache_key = (item["dataset_id"], int(item["train_sampling_seed"]), item["observation_key"])
            if training_available and cache_key not in asset_cache:
                try:
                    merged = _seed_assets(training_root, item["dataset_id"], int(item["train_sampling_seed"]), item["observation_key"])
                    for seed in EVAL_SEEDS:
                        if seed == int(item["train_sampling_seed"]):
                            continue
                        other = _seed_assets(training_root, item["dataset_id"], seed, item["observation_key"])
                        for key in ("window", "asc", "partial", "rf_speed", "traffic"):
                            merged[key] = bool(merged[key] and other[key])
                    asset_cache[cache_key] = merged
                except OSError as exc:
                    asset_cache[cache_key] = {"error": str(exc)}
            assets = asset_cache.get(cache_key) or {}
            if assets.get("error"):
                item["status"] = "unreadable"
                item["reason"] = assets["error"]
                anomalies.append({"path": item["dataset_id"], "reason": assets["error"]})
                continue
            blocked = data_block(item, training_root, dataset["status"], assets or {
                "window": False, "asc": False, "partial": False, "rf_speed": False, "traffic": False, "asc_scaler": False,
            })
            if blocked:
                item["status"] = "blocked"
                item["reason"] = blocked
                continue
            matched = [proc for proc in (processes or []) if command_matches_item(proc.get("command", ""), item)]
            if matched:
                item["status"] = "active"
                item["pid"] = matched[0].get("pid")
                item["reason"] = "live training process"
                anomalies.append({"item_id": item["item_id"], "reason": "active process", "pid": item["pid"]})
                continue
            item["status"] = "pending"
            item["reason"] = None
        _refresh_group_status(group)
    counts = count_items(groups)
    if sum(counts.values()) != 6720:
        raise RuntimeError(f"status counts {counts} do not sum to 6720")
    return {
        "inventory_status": inventory_status,
        "counts": counts,
        "anomalies": anomalies,
        "datasets": datasets,
        "reusable_examples": _reusable_examples(groups),
        "distribution": distribution(groups),
        "to_run": None if inventory_status == "unverified" or counts.get("unverified") else counts.get("pending", 0),
        "reusable": None if inventory_status == "unverified" else counts.get("reusable", 0),
    }


def _reusable_examples(groups: list[dict], limit: int = 5) -> list[str]:
    examples = []
    for group in groups:
        if int(group["train_sampling_seed"]) != 42:
            continue
        for item in group["items"]:
            if item["status"] == "reusable" and item.get("reused_run_dir"):
                examples.append(item["reused_run_dir"])
                if len(examples) >= limit:
                    return examples
    return examples


def count_items(groups: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for group in groups:
        for item in group["items"]:
            counts[item["status"]] = counts.get(item["status"], 0) + 1
    return counts


def distribution(groups: list[dict]) -> dict[str, dict[str, dict[str, int]]]:
    table: dict[str, dict[str, dict[str, int]]] = {}
    for group in groups:
        band = table.setdefault(group["observation_key"], {})
        seed = band.setdefault(str(group["train_sampling_seed"]), {})
        for item in group["items"]:
            family = seed.setdefault(item["family"], {})
            family[item["status"]] = family.get(item["status"], 0) + 1
    return table


def _holds_gpu(group: dict) -> bool:
    if group.get("cpu") or not group.get("gpu_uuid"):
        return False
    return any(item["status"] in {"pending", "running", "starting"} for item in group["items"])


def _refresh_group_status(group: dict) -> None:
    statuses = {item["status"] for item in group["items"]}
    if any(item["status"] in {"running", "starting"} for item in group["items"]):
        group["status"] = "running"
        return
    if _holds_gpu(group):
        group["status"] = "running"
        return
    group["gpu_uuid"] = None
    group["gpu_index"] = None
    if "pending" in statuses:
        group["status"] = "pending"
    elif "failed" in statuses:
        group["status"] = "partial_failed"
    elif "active" in statuses:
        group["status"] = "active"
    elif statuses <= TERMINAL_SUCCESS:
        group["status"] = "completed"
    elif "blocked" in statuses:
        group["status"] = "blocked"
    elif "unverified" in statuses:
        group["status"] = "unverified"
    else:
        group["status"] = "pending"


def command_for(item: dict, python: Path, training_root: Path, results_root: Path, threads: int) -> list[str]:
    if item["cpu"]:
        LEGACY.reject_failed_rf_preparation(item, Path(training_root))
        return [
            str(python),
            item["script"],
            "--dataset-id",
            item["dataset_id"],
            "--training-root",
            str(training_root),
            "--results-root",
            str(Path(results_root) / "RF"),
            "--penetrations",
            item["observation_key"],
            "--input-type",
            item["input_type"],
            "--target-type",
            item["target_type"],
            "--parallel-models",
            "1",
            "--n-jobs-per-model",
            str(threads),
            "--seed",
            str(MODEL_SEED),
            "--train-sampling-seed",
            str(item["train_sampling_seed"]),
            "--run-id",
            str(item["run_id"]),
            "--experiment-id",
            item["experiment_id"],
            "--n-clusters",
            "50",
        ]
    config = LEGACY.config_path(item["model"])
    if not Path(config).is_file():
        raise FileNotFoundError(config)
    sampling_root = Path(training_root) / "datasets" / item["dataset_id"] / "sampling_seeds"
    return [
        str(python),
        item["script"],
        "--config",
        str(config),
        "--dataset-id",
        item["dataset_id"],
        "--sampling-root",
        str(sampling_root),
        "--experiment-id",
        item["experiment_id"],
        "--run-id",
        str(item["run_id"]),
        "--model-seed",
        str(MODEL_SEED),
        "--seed",
        str(MODEL_SEED),
        "--train-sampling-seed",
        str(item["train_sampling_seed"]),
        "--evaluation-sampling-seeds",
        *[str(seed) for seed in EVAL_SEEDS],
        "--rates",
        item["observation_key"],
        "--input-type",
        item["input_type"],
        "--target-type",
        item["target_type"],
        "--output-root",
        str(Path(results_root) / item["model"]),
        "--device",
        "cuda:0",
    ]


def output_dir_for(item: dict, results_root: Path) -> Path:
    return sampling_run_results_dir(
        Path(results_root) / item["model"],
        item["input_type"],
        item["target_type"],
        item["observation_key"],
        MODEL_SEED,
        physical_version=PHYSICAL_VERSION,
        dataset_id=item["dataset_id"],
        model_name=item["model"],
        experiment_id=item["experiment_id"],
        run_id=str(item["run_id"]),
    )


def rf_worker_path(run_id: str, observation_key: str) -> Path:
    return model_report_dir("RF") / str(run_id) / f"worker_{observation_key}.json"


def child_env(item: dict, gpu_index: int | None, threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CODE)
    env["PYTHONUNBUFFERED"] = "1"
    text = str(threads)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = text
    if item["cpu"]:
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    return env


def parse_gpu_xml(xml_text: str) -> list[dict]:
    """Cards and every process the driver reported. A graphics process occupies the card."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise RuntimeError(f"nvidia-smi XML is not parseable: {exc}") from exc
    cards = []
    for gpu in root.findall("gpu"):
        uuid = (gpu.findtext("uuid") or "").strip()
        index_text = (gpu.findtext("minor_number") or "").strip()
        query_ok = bool(uuid) and index_text.isdigit()
        processes = []
        if query_ok:
            holder = gpu.find("processes")
            if holder is not None:
                for info in holder.findall("process_info"):
                    pid_text = (info.findtext("pid") or "").strip()
                    if not pid_text.isdigit():
                        continue
                    processes.append(
                        {
                            "pid": int(pid_text),
                            "type": (info.findtext("type") or "").strip(),
                            "name": (info.findtext("process_name") or "").strip(),
                        }
                    )
        cards.append(
            {
                "index": int(index_text) if index_text.isdigit() else None,
                "uuid": uuid,
                "processes": processes,
                "query_ok": query_ok,
            }
        )
    return cards


def query_gpus(runner=None) -> list[dict]:
    if runner is None:
        done = subprocess.run(["nvidia-smi", "-q", "-x"], check=False, capture_output=True, text=True)
        if done.returncode != 0:
            detail = (done.stderr or done.stdout or "").strip()
            raise RuntimeError(f"nvidia-smi failed: {detail}")
        return parse_gpu_xml(done.stdout)
    done = runner(["nvidia-smi", "-q", "-x"])
    if getattr(done, "returncode", 1) != 0:
        raise RuntimeError(getattr(done, "stderr", "") or "nvidia-smi failed")
    return parse_gpu_xml(done.stdout)


def card_is_free(card: dict) -> bool:
    return bool(card.get("query_ok")) and not card.get("processes")


def allowed_cards(cards: list[dict], gpu_indexes: set[int] | None, gpu_uuids: set[str] | None) -> list[dict]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_indexes: set[int] | None = None
    visible_uuids: set[str] | None = None
    if visible is not None and visible.strip() not in {"", "all"}:
        visible_indexes = set()
        visible_uuids = set()
        for part in visible.split(","):
            token = part.strip()
            if token.isdigit():
                visible_indexes.add(int(token))
            elif token:
                visible_uuids.add(token)
    chosen = []
    for card in cards:
        if not card.get("query_ok"):
            continue
        if gpu_indexes is not None and int(card["index"]) not in gpu_indexes:
            continue
        if gpu_uuids is not None and card["uuid"] not in gpu_uuids:
            continue
        if visible_indexes is not None and int(card["index"]) not in visible_indexes and card["uuid"] not in (visible_uuids or set()):
            continue
        chosen.append(card)
    return chosen


def interpret_cap(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def drain_messages(ops: Path) -> int | None:
    """Apply every legal max_gpus line. An illegal line does not undo an earlier legal one."""
    inbox = ops / "messages.jsonl"
    if not inbox.is_file() or inbox.stat().st_size == 0:
        return None
    processing = ops / "messages.processing"
    os.replace(inbox, processing)
    last: int | None = None
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
            if not isinstance(payload, dict) or "max_gpus" not in payload:
                log_line(ops, f"message rejected: {line}")
                continue
            parsed = interpret_cap(payload["max_gpus"])
            if parsed is None:
                log_line(ops, f"message rejected max_gpus={payload['max_gpus']!r}")
                continue
            last = parsed
    processing.unlink(missing_ok=True)
    return last


def refresh_cap(ops: Path, previous: int) -> int:
    """Read control.json. Invalid contents keep the previous legal value. No 09:00 reset."""
    message = drain_messages(ops)
    if message is not None:
        atomic_write_json(ops / "control.json", {"max_gpus": message})
        log_line(ops, f"message set max_gpus={message}")
        previous = message
    try:
        payload = read_json(ops / "control.json") or {}
        parsed = interpret_cap(payload.get("max_gpus") if isinstance(payload, dict) else None)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        log_line(ops, f"CONTROL rejected ({exc}); keeping max_gpus={previous}")
        return previous
    if parsed is None:
        log_line(ops, f"CONTROL rejected max_gpus={payload!r}; keeping {previous}")
        return previous
    return parsed


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


def _pid_command(pid: int, processes: list[dict] | None) -> str:
    if processes is not None:
        for row in processes:
            if int(row.get("pid") or -1) == int(pid):
                return str(row.get("command") or "")
        return ""
    path = Path(f"/proc/{pid}/cmdline")
    try:
        return path.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return ""


def _pid_alive(pid: int, processes: list[dict] | None) -> bool:
    if processes is not None:
        return any(int(row.get("pid") or -1) == int(pid) for row in processes)
    return bool(LEGACY.pid_alive(int(pid)))


def adopt_running(groups: list[dict], processes: list[dict] | None) -> None:
    """Take back a child only when its command carries this attempt's run id.

    A live pid with a different command is not adopted. A pending item whose
    run id is already in a live command is the gap between process start and
    the following state write.
    """
    rows = list(processes or [])
    for group in groups:
        for item in group["items"]:
            if item["status"] not in {"running", "starting", "pending"}:
                continue
            run_id = str(item.get("run_id") or "")
            pid = int(item.get("pid") or 0)
            matched = None
            if run_id:
                for row in rows:
                    if run_id in str(row.get("command") or ""):
                        matched = row
                        break
            if matched is None and item["status"] == "pending" and not run_id:
                for row in rows:
                    if command_matches_item(str(row.get("command") or ""), item):
                        item["status"] = "active"
                        item["pid"] = row.get("pid")
                        item["reason"] = "live training process"
                        matched = "external"
                        break
                if matched == "external":
                    continue
            if matched is None:
                if item["status"] == "pending":
                    continue
                complete = item["status"] == "running" and _item_artifacts_complete(item)
                item["finished_at"] = stamp()
                if complete:
                    item["status"] = "completed"
                    item["reason"] = None
                elif int(item.get("attempt") or 0) >= MAX_ATTEMPTS:
                    item["status"] = "failed"
                    item["reason"] = item.get("reason") or "process disappeared"
                else:
                    item["status"] = "pending"
                    item["reason"] = "process disappeared before completion"
                    item["pid"] = None
                continue
            item["status"] = "running"
            item["pid"] = int(matched["pid"])
            item["reason"] = "adopted live attempt"
            if item.get("gpu_uuid"):
                group["gpu_uuid"] = item["gpu_uuid"]
                group["gpu_index"] = item.get("gpu_index")
        _refresh_group_status(group)


def _item_artifacts_complete(item: dict) -> bool:
    if not item.get("run_dir"):
        return False
    record = classify_run(Path(item["run_dir"]))
    if record["status"] != "reusable":
        return False
    return (
        record.get("model") in {item["model"], ""}
        and int(record.get("train_sampling_seed") or -1) == int(item["train_sampling_seed"])
        and record.get("input_type") in {item["input_type"], ""}
        and record.get("target_type") in {item["target_type"], ""}
    )


def next_item(group: dict) -> dict | None:
    for item in group["items"]:
        if item["status"] in {"running", "starting"}:
            return None
        if item["status"] == "pending":
            return item
    return None


def group_reserved_uuid(group: dict) -> str | None:
    if _holds_gpu(group) or (
        group.get("gpu_uuid") and any(item["status"] in {"running", "starting"} for item in group["items"])
    ):
        return str(group["gpu_uuid"])
    return None


def save_state(ops: Path, groups: list[dict], cap: int, report: dict | None = None) -> None:
    payload = {
        "campaign_id": CAMPAIGN_ID,
        "updated_at": stamp(),
        "max_gpus": cap,
        "counts": count_items(groups),
        "groups": groups,
    }
    atomic_write_json(ops / "groups.json", payload)
    if report is not None:
        atomic_write_json(ops / "inventory.json", {key: value for key, value in report.items() if key != "datasets"})
        atomic_write_json(ops / "dataset_report.json", report.get("datasets") or {})


def load_state(ops: Path) -> list[dict] | None:
    payload = read_json(ops / "groups.json")
    if not isinstance(payload, dict) or payload.get("campaign_id") != CAMPAIGN_ID:
        return None
    groups = payload.get("groups")
    if not isinstance(groups, list) or len(groups) != 480:
        raise RuntimeError("groups.json does not contain the 480 campaign groups")
    return groups


def _launch_item(ops, groups, group, item, gpu, python, training_root, results_root, threads, launch, cap: int) -> dict:
    if not item.get("run_id"):
        item["run_id"] = new_attempt_id()
    item["run_dir"] = str(output_dir_for(item, results_root))
    item["command"] = command_for(item, python, training_root, results_root, threads)
    item["status"] = "starting"
    item["started_at"] = stamp()
    item["gpu_index"] = None if gpu is None else int(gpu["index"])
    item["gpu_uuid"] = None if gpu is None else str(gpu["uuid"])
    if gpu is not None:
        group["gpu_index"] = int(gpu["index"])
        group["gpu_uuid"] = str(gpu["uuid"])
        group["status"] = "running"
    log_path = ops / "logs" / f"{item['item_id']}__{item['run_id']}.log"
    item["log"] = str(log_path)
    append_jsonl(
        ops / "attempts.jsonl",
        {
            "item_id": item["item_id"],
            "run_id": item["run_id"],
            "run_dir": item["run_dir"],
            "command": item["command"],
            "gpu_uuid": item["gpu_uuid"],
            "started_at": item["started_at"],
        },
    )
    save_state(ops, groups, cap)
    proc = launch(item, gpu, log_path)
    item["pid"] = int(proc.pid)
    item["status"] = "running"
    item["attempt"] = int(item.get("attempt") or 0) + 1
    item["exit_code"] = None
    item["finished_at"] = None
    save_state(ops, groups, cap)
    return {"proc": proc, "item_id": item["item_id"]}


def default_launch(item: dict, gpu: dict | None, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    handle.write(f"start_time={stamp()}\ncommand={' '.join(item['command'])}\nrun_dir={item['run_dir']}\n")
    handle.flush()
    threads = int(os.environ.get("TRAFFICFLOW_THREADS", "1"))
    proc = subprocess.Popen(
        item["command"],
        cwd=CODE,
        env=child_env(item, None if gpu is None else int(gpu["index"]), threads),
        stdout=handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc


def run_campaign(
    ops: Path,
    training_root: Path,
    results_root: Path,
    *,
    query=None,
    launch=None,
    sleep=None,
    processes=None,
    gpu_indexes: set[int] | None = None,
    gpu_uuids: set[str] | None = None,
    install_signals: bool = True,
    max_polls: int | None = None,
    python: Path | None = None,
    memory_bytes: int | None = None,
) -> dict:
    """Run until the matrix is finished or a test asks the loop to stop.

    Polling rechecks the driver. It is not an atomic claim against programs
    that ignore this campaign lock.
    """
    lock = acquire_lock(ops)
    if install_signals:
        signal.signal(signal.SIGHUP, _request_reload)
        signal.signal(signal.SIGTERM, _request_stop)
        signal.signal(signal.SIGINT, _request_stop)
    query = query or query_gpus
    launch = launch or default_launch
    sleep = sleep or (lambda seconds: LEGACY.wait_poll(seconds))
    python = python or LEGACY.training_python()
    control_path = ops / "control.json"
    if not control_path.is_file():
        atomic_write_json(control_path, {"max_gpus": DEFAULT_MAX_GPUS})
    groups = load_state(ops)
    created = groups is None
    if groups is None:
        groups = build_groups()
        report = apply_inventory(groups, training_root=training_root, results_root=results_root, processes=processes if processes is not None else live_commands())
    else:
        report = {"inventory_status": "resumed", "counts": count_items(groups), "anomalies": [], "datasets": {}}
    adopt_running(groups, processes if processes is not None else live_commands())
    procs: dict[str, object] = {}
    cap = DEFAULT_MAX_GPUS
    log_line(ops, f"scheduler start campaign={CAMPAIGN_ID} created={created} counts={count_items(groups)}")
    save_state(ops, groups, cap, report if created else None)
    polls = 0
    try:
        while not SIGNALS.stop:
            if SIGNALS.reload:
                SIGNALS.reload = False
                log_line(ops, "SIGHUP reload")
            cap = refresh_cap(ops, cap)
            try:
                cards = query()
            except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
                log_line(ops, f"GPU query failed: {exc}")
                cards = []
            _reap(ops, groups, procs)
            reserved = {uuid for group in groups if (uuid := group_reserved_uuid(group))}
            pool = [card for card in allowed_cards(cards, gpu_indexes, gpu_uuids) if card_is_free(card) and card["uuid"] not in reserved]
            available_ram = LEGACY.mem_available_bytes() if memory_bytes is None else int(memory_bytes)
            ram_ok = available_ram >= MIN_FREE_RAM_BYTES
            threads = LEGACY.thread_budget(cap)
            os.environ["TRAFFICFLOW_THREADS"] = str(threads)
            holders = [group for group in groups if _holds_gpu(group)]
            cpu_running = [
                group for group in groups
                if group["cpu"] and any(item["status"] in {"running", "starting"} for item in group["items"])
            ]
            for group in holders:
                item = next_item(group)
                if item is None:
                    continue
                card = _card_by_uuid(cards, group.get("gpu_uuid"))
                if card is None or not card_is_free(card) or not _recheck_free(query, str(group.get("gpu_uuid")), gpu_indexes, gpu_uuids):
                    group["status"] = "waiting_external"
                    log_line(ops, f"WAIT external process on reserved {group.get('gpu_uuid')} {group['group_id']}")
                    continue
                group["status"] = "running"
                _start_group_item(ops, groups, group, item, card, python, training_root, results_root, threads, launch, procs, cap)
            if ram_ok and not cpu_running:
                rf_group = next((group for group in groups if group["cpu"] and next_item(group) is not None), None)
                if rf_group is not None:
                    item = next_item(rf_group)
                    if item is not None:
                        _start_group_item(ops, groups, rf_group, item, None, python, training_root, results_root, threads, launch, procs, cap)
            elif not ram_ok:
                log_line(ops, f"skip CPU launch MemAvailable below {MIN_FREE_RAM_BYTES}")
            if cap > 0 and len(holders) < cap:
                for group in groups:
                    if sum(_holds_gpu(row) for row in groups) >= cap:
                        break
                    if group["cpu"] or _holds_gpu(group) or group["status"] in {"completed", "waiting_external"}:
                        continue
                    item = next_item(group)
                    if item is None:
                        _refresh_group_status(group)
                        continue
                    if not pool:
                        break
                    card = pool.pop(0)
                    if card["uuid"] in reserved or not _recheck_free(query, card["uuid"], gpu_indexes, gpu_uuids):
                        log_line(ops, f"GPU {card['index']} was taken before launch")
                        continue
                    group["gpu_uuid"] = card["uuid"]
                    group["gpu_index"] = card["index"]
                    reserved.add(card["uuid"])
                    _start_group_item(ops, groups, group, item, card, python, training_root, results_root, threads, launch, procs, cap)
            elif cap == 0:
                log_line(ops, "max_gpus=0; not dispatching a new GPU group")
            for group in groups:
                _refresh_group_status(group)
            save_state(ops, groups, cap)
            pending = sum(item["status"] == "pending" for group in groups for item in group["items"])
            running = sum(item["status"] in {"running", "starting"} for group in groups for item in group["items"])
            log_line(ops, f"poll max={cap} running_items={running} queued_items={pending}")
            if pending == 0 and running == 0:
                break
            polls += 1
            if max_polls is not None and polls >= max_polls:
                break
            sleep(POLL_SECONDS)
        save_state(ops, groups, cap)
        log_line(ops, f"scheduler stop counts={count_items(groups)}")
        return {"counts": count_items(groups), "max_gpus": cap}
    finally:
        lock.seek(0)
        lock.truncate()
        lock.close()


def _card_by_uuid(cards: list[dict], uuid_value: str | None) -> dict | None:
    if not uuid_value:
        return None
    for card in cards:
        if card.get("uuid") == uuid_value and card.get("query_ok"):
            return card
    return None


def _recheck_free(query, uuid_value: str, gpu_indexes, gpu_uuids) -> bool:
    try:
        cards = query()
    except (OSError, subprocess.CalledProcessError, RuntimeError):
        return False
    card = _card_by_uuid(allowed_cards(cards, gpu_indexes, gpu_uuids), uuid_value)
    return card is not None and card_is_free(card)


def _start_group_item(ops, groups, group, item, gpu, python, training_root, results_root, threads, launch, procs, cap: int) -> None:
    try:
        handle = _launch_item(ops, groups, group, item, gpu, python, training_root, results_root, threads, launch, cap)
    except Exception as exc:
        item["status"] = "failed"
        item["attempt"] = int(item.get("attempt") or 0) + 1
        item["error"] = f"launch failed: {exc}"
        item["reason"] = item["error"]
        item["finished_at"] = stamp()
        log_line(ops, f"FAILED {item['item_id']} {exc}")
        return
    procs[item["item_id"]] = handle["proc"]
    where = "CPU" if gpu is None else f"GPU {gpu['index']}"
    log_line(ops, f"{where} START {item['item_id']} pid={item['pid']} run_id={item['run_id']}")


def _reap(ops: Path, groups: list[dict], procs: dict[str, object]) -> None:
    for group in groups:
        for item in group["items"]:
            if item["status"] not in {"running", "starting"}:
                continue
            proc = procs.get(item["item_id"])
            if proc is None:
                continue
            code = proc.poll()
            if code is None:
                continue
            item["exit_code"] = code
            item["finished_at"] = stamp()
            if code == 0 and _item_artifacts_complete(item):
                item["status"] = "completed"
                item["reason"] = None
            else:
                item["status"] = "failed"
                item["reason"] = f"exit {code}"
            log_line(ops, f"{item['status'].upper()} {item['item_id']} exit={code}")
            procs.pop(item["item_id"], None)
        _refresh_group_status(group)


def _request_reload(signum, frame) -> None:
    del signum, frame
    SIGNALS.reload = True


def _request_stop(signum, frame) -> None:
    del signum, frame
    SIGNALS.stop = True


def requeue_failed(ops: Path) -> int:
    groups = load_state(ops)
    if groups is None:
        raise SystemExit(f"no queue at {ops / 'groups.json'}")
    changed = 0
    for group in groups:
        for item in group["items"]:
            if item["status"] != "failed":
                continue
            if _item_artifacts_complete(item):
                item["status"] = "completed"
                item["reason"] = None
                continue
            if int(item.get("attempt") or 0) >= MAX_ATTEMPTS:
                continue
            item["status"] = "pending"
            item["pid"] = None
            item["exit_code"] = None
            item["finished_at"] = None
            changed += 1
        _refresh_group_status(group)
    control = read_json(ops / "control.json") or {}
    cap = control.get("max_gpus") if isinstance(control, dict) and isinstance(control.get("max_gpus"), int) else DEFAULT_MAX_GPUS
    save_state(ops, groups, int(cap))
    return changed


def format_plan(groups: list[dict], report: dict) -> str:
    lines = [
        f"campaign={CAMPAIGN_ID}",
        f"inventory={report['inventory_status']}",
        f"groups={len(groups)} items={sum(len(group['items']) for group in groups)}",
        f"counts={report['counts']}",
        f"reusable={report['reusable']}",
        f"to_run={report['to_run']}",
        f"anomalies={len(report['anomalies'])}",
    ]
    for example in report.get("reusable_examples") or []:
        lines.append(f"reused_seed42={example}")
    for dataset_id, row in sorted((report.get("datasets") or {}).items()):
        if row.get("status") not in {None, "ready"}:
            lines.append(f"dataset {dataset_id} {row.get('status')} {row.get('detail')}")
    for anomaly in report["anomalies"][:50]:
        lines.append(f"anomaly {anomaly}")
    lines.append("distribution=" + json.dumps(report.get("distribution") or {}, sort_keys=True))
    return "\n".join(lines)


def write_summary_csv(ops: Path, groups: list[dict]) -> None:
    rows = []
    for group in groups:
        for item in group["items"]:
            rows.append(
                {
                    "model": item["model"],
                    "dataset_id": item["dataset_id"],
                    "train_sampling_seed": item["train_sampling_seed"],
                    "input_type": item["input_type"],
                    "target_type": item["target_type"],
                    "status": item["status"],
                    "run_dir": item.get("run_dir") or "",
                    "reason": item.get("reason") or "",
                }
            )
    temporary = ops / "campaign_summary.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, ops / "campaign_summary.csv")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan or run the two-band five-sample baseline campaign.")
    parser.add_argument("command", choices=("plan", "status", "run", "requeue-failed"))
    parser.add_argument("--gpus", default=None, help="Physical indexes, comma-separated. The default pool is the visible GPUs.")
    parser.add_argument("--gpu-uuids", default=None, help="Physical GPU UUIDs, comma-separated.")
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--training-root", type=Path, default=None)
    parser.add_argument("--max-polls", type=int, default=None, help="Stop after this many polls. Used by tests.")
    args = parser.parse_args(argv)
    training_root = args.training_root or xuancheng26_training_3s_root()
    results_root = args.results_root or xuancheng26_training_results_3s_root()
    ops = operations_dir(results_root)
    if args.command == "plan":
        groups = build_groups()
        report = apply_inventory(groups, training_root=training_root, results_root=results_root, processes=live_commands())
        print(format_plan(groups, report))
        return 0
    if args.command == "status":
        payload = read_json(ops / "groups.json")
        control = read_json(ops / "control.json") if (ops / "control.json").is_file() else None
        print(f"campaign={CAMPAIGN_ID}")
        print(f"control={ops / 'control.json'}")
        print(f"lock={ops / 'scheduler.lock'}")
        print(f"messages={ops / 'messages.jsonl'}")
        print(f"log={ops / 'scheduler.log'}")
        if isinstance(control, dict):
            print(f"max_gpus={control.get('max_gpus')}")
        if not isinstance(payload, dict):
            print("no saved queue")
            return 0
        print(f"counts={payload.get('counts')}")
        for group in payload.get("groups") or []:
            if group.get("status") in {"running", "waiting_external"}:
                print(f"{group['status']} {group['group_id']} gpu={group.get('gpu_uuid')}")
        return 0
    if args.command == "requeue-failed":
        print(f"requeued {requeue_failed(ops)}")
        return 0
    indexes = None if not args.gpus else {int(part) for part in args.gpus.split(",") if part.strip()}
    uuids = None if not args.gpu_uuids else {part.strip() for part in args.gpu_uuids.split(",") if part.strip()}
    run_campaign(
        ops,
        training_root,
        results_root,
        gpu_indexes=indexes,
        gpu_uuids=uuids,
        max_polls=args.max_polls,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
