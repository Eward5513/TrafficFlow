"""Grouped Xuancheng26 baseline campaign for two penetration bands.

One group is one model, one dataset, one training sampling seed, and one model
seed. The group runs its five input/output items one at a time. The sampling
seed selects the published vehicle sample. The model seed is the initialization
and shuffle seed. A result directory named seed_<m> is not evidence that the
training vehicles were sample m.

This module plans and can run the campaign. Importing it does not start training.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.util
import json
import math
import os
import re
import shlex
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

PHYSICAL_VERSION = "independent_fcd_v1"
TRAIN_SAMPLING_SEEDS = (42, 43, 44, 45, 46)
MODEL_SEEDS = (42, 101, 102)
EVAL_SEEDS = (42, 43, 44, 45, 46)
IO_PAIRS = (
    {"input_type": "inflow", "target_type": "inflow", "family": "I2I"},
    {"input_type": "all", "target_type": "inflow", "family": "A2I"},
    {"input_type": "all", "target_type": "all", "family": "A2A"},
    {"input_type": "all_sp_c", "target_type": "inflow", "family": "ASC2I"},
    {"input_type": "all_sp_c", "target_type": "all", "family": "ASC2A"},
)
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


def _joined_seeds(seeds: tuple[int, ...]) -> str:
    return "_".join(str(int(seed)) for seed in seeds)


def campaign_id(
    sampling_seeds: tuple[int, ...] = TRAIN_SAMPLING_SEEDS,
    model_seeds: tuple[int, ...] = MODEL_SEEDS,
    task_count: int = len(IO_PAIRS),
) -> str:
    """Operations namespace. Changing either seed sequence starts a new campaign."""
    return (
        "grouped_p5_p15_p5_p35_"
        f"io{int(task_count)}_s{_joined_seeds(sampling_seeds)}_m{_joined_seeds(model_seeds)}"
    )


CAMPAIGN_ID = campaign_id()


def io_pairs() -> list[dict[str, str]]:
    """Five training pairs in comparison order. Identity is the type pair."""
    rows = [dict(row) for row in IO_PAIRS]
    keys = [(row["input_type"], row["target_type"]) for row in rows]
    if keys != [
        ("inflow", "inflow"),
        ("all", "inflow"),
        ("all", "all"),
        ("all_sp_c", "inflow"),
        ("all_sp_c", "all"),
    ] or len({(row["input_type"], row["target_type"]) for row in rows}) != len(rows):
        raise RuntimeError(
            "the campaign matrix is not the five input/target pairs I2I, A2I, A2A, ASC2I, ASC2A"
        )
    return rows


def matrix_shape(models: tuple[dict, ...] | None = None) -> tuple[int, int]:
    """Group count and item count for this catalog, both seed sequences, and the five tasks."""
    catalog = tuple(LEGACY.MODELS if models is None else models)
    group_count = len(catalog) * len(EXPECTED_DATASETS) * len(TRAIN_SAMPLING_SEEDS) * len(MODEL_SEEDS)
    return group_count, group_count * len(io_pairs())


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


def _assert_queue_matrix(groups: list[dict], models: tuple[dict, ...] | None = None) -> None:
    """Reject a queue whose sampling seeds, model seeds, or five tasks differ."""
    catalog = tuple(LEGACY.MODELS if models is None else models)
    expected_groups, expected_items = matrix_shape(catalog)
    if len(groups) != expected_groups:
        raise RuntimeError(f"campaign matrix is not {expected_groups} groups and {expected_items} items")
    pairs = io_pairs()
    expected_models = {row["name"] for row in catalog}
    expected_datasets = {row["dataset_id"] for row in EXPECTED_DATASETS}
    if {group.get("model") for group in groups} != expected_models:
        raise RuntimeError("campaign groups do not match the model catalog")
    if {group.get("dataset_id") for group in groups} != expected_datasets:
        raise RuntimeError("campaign groups do not match the eight datasets")
    identities: set[tuple] = set()
    for group in groups:
        try:
            train_seed = int(group["train_sampling_seed"])
            model_seed = int(group["model_seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"group {group.get('group_id')} is missing sampling or model seed") from exc
        if train_seed not in TRAIN_SAMPLING_SEEDS or model_seed not in MODEL_SEEDS:
            raise RuntimeError(
                f"group {group.get('group_id')} has s={train_seed} m={model_seed} "
                f"outside s={list(TRAIN_SAMPLING_SEEDS)} m={list(MODEL_SEEDS)}"
            )
        expected_id = f"{group.get('model')}__{group.get('dataset_id')}__train{train_seed}__model{model_seed}"
        if group.get("group_id") != expected_id:
            raise RuntimeError(f"group id {group.get('group_id')} does not match {expected_id}")
        items = group.get("items")
        if not isinstance(items, list) or len(items) != len(pairs):
            raise RuntimeError(f"group {expected_id} does not have the {len(pairs)} input/output items")
        for item, pair in zip(items, pairs):
            try:
                item_train = int(item["train_sampling_seed"])
                item_model = int(item["model_seed"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"item {item.get('item_id')} is missing sampling or model seed") from exc
            if (
                item.get("model") != group.get("model")
                or item.get("dataset_id") != group.get("dataset_id")
                or item.get("input_type") != pair["input_type"]
                or item.get("target_type") != pair["target_type"]
                or item.get("family") != pair["family"]
                or item_train != train_seed
                or item_model != model_seed
            ):
                raise RuntimeError(f"item {item.get('item_id')} does not match the five-task seed matrix")
            identities.add(
                (
                    item.get("model"),
                    item.get("dataset_id"),
                    item.get("input_type"),
                    item.get("target_type"),
                    item_train,
                    item_model,
                )
            )
    if len(identities) != expected_items or sum(len(group["items"]) for group in groups) != expected_items:
        raise RuntimeError("logical experiment identities are not unique")


def build_groups(models: tuple[dict, ...] | None = None) -> list[dict]:
    catalog = tuple(models) if models is not None else tuple(LEGACY.MODELS)
    pairs = io_pairs()
    groups: list[dict] = []
    for model in catalog:
        for dataset in EXPECTED_DATASETS:
            for train_seed in TRAIN_SAMPLING_SEEDS:
                for model_seed in MODEL_SEEDS:
                    sample = int(train_seed)
                    fitted = int(model_seed)
                    group_id = f"{model['name']}__{dataset['dataset_id']}__train{sample}__model{fitted}"
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
                                "train_sampling_seed": sample,
                                "model_seed": fitted,
                                "evaluation_sampling_seeds": list(EVAL_SEEDS),
                                "input_type": pair["input_type"],
                                "target_type": pair["target_type"],
                                "family": pair["family"],
                                "experiment_id": experiment_id_for(dataset["observation_key"], sample),
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
                            "train_sampling_seed": sample,
                            "model_seed": fitted,
                            "status": "pending",
                            "gpu_index": None,
                            "gpu_uuid": None,
                            "items": items,
                        }
                    )
    if any(group["model"] in {"MSTT", "pdformer"} for group in groups):
        raise RuntimeError("MSTT and PDFormer are outside this campaign")
    _assert_queue_matrix(groups, catalog)
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


def _manifest_dates(training_root: Path, dataset_id: str) -> dict[str, list[str]] | None:
    path = (
        Path(training_root) / "datasets" / dataset_id / "sampling_seeds" / "seed_42"
        / "flow" / "window_data" / "split_manifest.json"
    )
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {split: [str(day) for day in payload[split]] for split in ("train", "validation", "test")}


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


class SeedFiles:
    """Cached existence checks for one dataset, sampling seed, and observation key."""

    def __init__(self, training_root: Path, dataset_id: str, seed: int, observation_key: str, dates: dict[str, list[str]] | None) -> None:
        self.root = Path(training_root) / "datasets" / dataset_id / "sampling_seeds" / f"seed_{int(seed)}"
        self.key = str(observation_key)
        self.dates = dates
        self._cache: dict[tuple, bool] = {}

    def windows(self, split: str, kind: str) -> bool:
        token = ("window", split, kind)
        if token not in self._cache:
            days = [] if self.dates is None else self.dates[split]
            base = self.root / "flow" / "window_data" / self.key / split
            self._cache[token] = bool(days) and all(_nonempty_file(base / day / f"{kind}.npz") for day in days)
        return self._cache[token]

    def raw_features(self, split: str, states: tuple[str, ...], *, traffic: bool) -> bool:
        token = ("raw", split, states, traffic)
        if token not in self._cache:
            days = [] if self.dates is None else self.dates[split]
            partial = self.root / "flow" / "partial_flow" / self.key
            speed = self.root / "model_prepared" / "RF" / self.key
            raw = self.root / "model_prepared" / "traffic_observations" / self.key
            ready = bool(days)
            for day in days:
                if not all(_nonempty_file(partial / day / f"{state}.csv") for state in states):
                    ready = False
                    break
                if not _nonempty_file(speed / f"{day}.npz"):
                    ready = False
                    break
                if traffic and not _nonempty_file(raw / f"{day}.npz"):
                    ready = False
                    break
            self._cache[token] = ready
        return self._cache[token]


def load_observation_assets(training_root: Path, dataset_id: str, observation_key: str) -> dict:
    dates = _manifest_dates(training_root, dataset_id)
    scaler = _nonempty_file(Path(training_root) / "datasets" / dataset_id / "all_sp_c_normalization.json")
    return {
        "dates": dates,
        "asc_scaler": scaler,
        "seeds": {
            seed: SeedFiles(training_root, dataset_id, seed, observation_key, dates)
            for seed in EVAL_SEEDS
        },
    }


def data_block(item: dict, dataset_status: str, assets: dict | None) -> str | None:
    """Block only the files this item will read. One missing ASC file does not block inflow."""
    if dataset_status == "unverified":
        return None
    if dataset_status != "ready":
        return f"dataset {item['dataset_id']} {dataset_status}"
    if not assets or not assets.get("dates"):
        return f"{item['dataset_id']} split manifest is missing"
    train_seed = int(item["train_sampling_seed"])
    seeds = assets["seeds"]
    train = seeds[train_seed]
    if not train.windows("train", item["input_type"]) or not train.windows("train", item["target_type"]):
        return f"missing train windows for seed {train_seed}"
    if item["input_type"] == "all_sp_c" and not assets["asc_scaler"]:
        return f"missing all_sp_c scaler for {item['dataset_id']}"
    for seed in EVAL_SEEDS:
        view = seeds[seed]
        for split in ("validation", "test"):
            if not view.windows(split, item["input_type"]) or not view.windows(split, item["target_type"]):
                return f"missing {split} {item['input_type']}->{item['target_type']} windows for seed {seed}"
    if item["cpu"]:
        states = STATES if item["input_type"] in {"all", "all_sp_c"} else (item["input_type"],)
        traffic = item["input_type"] == "all_sp_c"
        if not train.raw_features("train", states, traffic=traffic):
            return f"missing RF train features for seed {train_seed}"
        for seed in EVAL_SEEDS:
            view = seeds[seed]
            for split in ("validation", "test"):
                if not view.raw_features(split, states, traffic=traffic):
                    return f"missing RF {split} features for seed {seed}"
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
    if seed not in TRAIN_SAMPLING_SEEDS:
        return "conflict", seed, f"training sample {seed} is outside {list(TRAIN_SAMPLING_SEEDS)}"
    return "proven", seed, detail


def _finite_mae(payload: dict | None) -> bool:
    """Five finite per-seed MAEs whose mean and sample std match the stored aggregate."""
    if not isinstance(payload, dict):
        return False
    per_seed = payload.get("per_seed")
    if not isinstance(per_seed, dict):
        return False
    values: list[float] = []
    for seed in EVAL_SEEDS:
        row = per_seed.get(str(seed))
        if not isinstance(row, dict) or "mae" not in row:
            return False
        try:
            value = float(row["mae"])
        except (TypeError, ValueError):
            return False
        if not math.isfinite(value):
            return False
        values.append(value)
    mean = payload.get("mean")
    std = payload.get("std")
    if not isinstance(mean, dict) or not isinstance(std, dict):
        return False
    try:
        recorded_mean = float(mean["mae"])
        recorded_std = float(std["mae"])
    except (KeyError, TypeError, ValueError):
        return False
    if not math.isfinite(recorded_mean) or not math.isfinite(recorded_std):
        return False
    computed_mean = sum(values) / float(len(values))
    computed_std = math.sqrt(sum((value - computed_mean) ** 2 for value in values) / float(len(values) - 1))
    tolerance = max(1e-5, 1e-4 * max(1.0, abs(computed_mean)))
    return abs(recorded_mean - computed_mean) <= tolerance and abs(recorded_std - computed_std) <= tolerance


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


def _npz_header(path: Path) -> bool:
    """Read the ZIP local header only. Inventory does not load prediction arrays."""
    try:
        if not path.is_file() or path.stat().st_size < 4:
            return False
        with path.open("rb") as handle:
            return handle.read(4) == b"PK\x03\x04"
    except OSError:
        return False


def _manifest_test_days(resolved: dict) -> list[str] | None:
    candidates: list[Path] = []
    if resolved.get("split_manifest"):
        candidates.append(Path(str(resolved["split_manifest"])))
    if resolved.get("data_root"):
        candidates.append(Path(str(resolved["data_root"])) / "split_manifest.json")
    dataset_id = resolved.get("dataset_id")
    training_root = resolved.get("training_root")
    if dataset_id and training_root:
        candidates.append(
            Path(str(training_root)) / "datasets" / str(dataset_id) / "sampling_seeds" / "seed_42"
            / "flow" / "window_data" / "split_manifest.json"
        )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        days = payload.get("test")
        if isinstance(days, list) and days:
            return [str(day) for day in days]
    return None


def _predictions_ready(run_dir: Path, cpu: bool, resolved: dict) -> bool:
    if cpu:
        days = _manifest_test_days(resolved)
        if not days:
            return False
        for seed in EVAL_SEEDS:
            directory = run_dir / "predictions" / f"seed_{seed}" / "test"
            if not all(_npz_header(directory / f"{day}.npz") for day in days):
                return False
        return True
    return all(_npz_header(run_dir / "predictions" / f"seed_{seed}" / "test.npz") for seed in EVAL_SEEDS)


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
    experiment = str(resolved.get("experiment_id") or "")
    run_id = str(resolved.get("run_id") or run_dir.name)
    blob = f"{text} {experiment} {run_id}".lower()
    if "smoke" in blob or "/debug/" in blob or "debug_short" in blob:
        return "smoke or debug path"
    if resolved.get("protocol_limit") or resolved.get("max_train_batches") or resolved.get("max_eval_batches"):
        return "protocol_limit"
    if "protocol_limit" in experiment.lower():
        return "protocol_limit"
    epochs = resolved.get("epochs")
    if epochs is not None:
        try:
            if int(epochs) <= 1:
                return "epochs<=1"
        except (TypeError, ValueError):
            return "epochs is not an integer"
    return None


def _explicit_failure(run_dir: Path) -> str | None:
    for name in ("run_metadata.json", "status.json"):
        path = run_dir / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return f"{name} unreadable"
        if not isinstance(payload, dict):
            continue
        status = str(payload.get("status", "")).lower()
        if status in {"failed", "fail"}:
            return f"{name} status={payload.get('status')}"
    return None


def _path_identity(run_dir: Path) -> dict[str, str]:
    parts = Path(run_dir).parts
    if "datasets" not in parts:
        return {}
    tail = parts[parts.index("datasets") + 1 :]
    if len(tail) < 8:
        return {}
    version = tail[1][len("physical_") :] if tail[1].startswith("physical_") else tail[1]
    task = tail[3]
    input_type = ""
    target_type = ""
    if task.startswith("in_") and "__out_" in task:
        input_type, target_type = task[len("in_") :].split("__out_", 1)
    model_seed = tail[6][len("seed_") :] if tail[6].startswith("seed_") else ""
    return {
        "dataset_id": tail[0],
        "physical_definition_version": version,
        "model": tail[2],
        "input_type": input_type,
        "target_type": target_type,
        "observation_key": tail[4],
        "experiment_id": tail[5],
        "model_seed": model_seed,
        "run_id": tail[7],
    }


def _fill_missing(record: dict, parsed: dict, key: str) -> None:
    if not record.get(key) and parsed.get(key):
        record[key] = parsed[key]


def classify_run(run_dir: Path) -> dict:
    """Read one finished or partial run. The model-seed directory is not the training sample."""
    run_dir = Path(run_dir)
    parsed = _path_identity(run_dir)
    resolved_path = run_dir / "resolved_config.json"
    try:
        if not resolved_path.is_file():
            return {"status": "incomplete", "reason": "missing resolved_config.json", "run_dir": str(run_dir)}
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "unreadable", "reason": str(exc), "run_dir": str(run_dir)}
    if not isinstance(resolved, dict):
        return {"status": "unreadable", "reason": "resolved_config is not an object", "run_dir": str(run_dir)}
    failure = _explicit_failure(run_dir)
    if failure:
        return {"status": "ignored", "reason": failure, "run_dir": str(run_dir), "resolved": resolved}
    debug = _smoke_or_debug(run_dir, resolved)
    if debug:
        return {"status": "ignored", "reason": debug, "run_dir": str(run_dir), "resolved": resolved}
    experiment_id = str(resolved.get("experiment_id") or parsed.get("experiment_id") or "")
    proof, train_seed, evidence = train_seed_evidence(resolved, experiment_id)
    try:
        model_seed = int(resolved.get("model_seed", resolved.get("seed", parsed.get("model_seed", -1))))
    except (TypeError, ValueError):
        model_seed = -1
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
        "experiment_id": experiment_id,
        "run_id": str(resolved.get("run_id") or parsed.get("run_id") or run_dir.name),
        "observation_key": str(parsed.get("observation_key") or ""),
        "physical_definition_version": str(resolved.get("physical_definition_version") or parsed.get("physical_definition_version") or ""),
    }
    for key in ("model", "dataset_id", "input_type", "target_type", "observation_key"):
        _fill_missing(record, parsed, key)
    if proof != "proven":
        return record
    mismatches = []
    for key in ("model", "dataset_id", "input_type", "target_type", "observation_key"):
        if parsed.get(key) and record.get(key) and str(parsed[key]) != str(record[key]):
            mismatches.append(f"{key} path={parsed[key]} record={record[key]}")
    path_version = parsed.get("physical_definition_version")
    meta_version = resolved.get("physical_definition_version")
    if path_version and path_version != PHYSICAL_VERSION:
        mismatches.append(f"physical path={path_version}")
    if meta_version and str(meta_version) != PHYSICAL_VERSION:
        mismatches.append(f"physical metadata={meta_version}")
    if parsed.get("model_seed") and str(parsed["model_seed"]) != str(model_seed):
        mismatches.append(f"model_seed path={parsed['model_seed']} record={model_seed}")
    if parsed.get("run_id") and resolved.get("run_id") and str(parsed["run_id"]) != str(resolved["run_id"]):
        mismatches.append(f"run_id path={parsed['run_id']} record={resolved['run_id']}")
    rates = [str(item) for item in (resolved.get("rates") or [])]
    if rates and parsed.get("observation_key") and rates != [parsed["observation_key"]]:
        mismatches.append(f"rates={rates}")
    eval_seeds = resolved.get("evaluation_sampling_seeds")
    if eval_seeds is not None and [int(seed) for seed in eval_seeds] != list(EVAL_SEEDS):
        mismatches.append(f"evaluation_sampling_seeds={eval_seeds}")
    if mismatches:
        record["status"] = "conflict"
        record["reason"] = "; ".join(mismatches)
        return record
    if model_seed not in MODEL_SEEDS:
        record["status"] = "ignored"
        record["reason"] = f"model_seed={model_seed}"
        return record
    selection = resolved.get("checkpoint_selection")
    if selection not in (None, "mean_validation_mae", "single_fit_no_epoch_selection"):
        record["status"] = "ignored"
        record["reason"] = f"checkpoint_selection={selection}"
        return record
    cpu = record["model"] == "RF"
    if cpu:
        clusters = resolved.get("n_clusters")
        config_path = run_dir / "config.json"
        if clusters is None and config_path.is_file():
            try:
                clusters = json.loads(config_path.read_text(encoding="utf-8")).get("n_clusters")
            except (OSError, json.JSONDecodeError):
                clusters = None
        if clusters is not None and int(clusters) != 50:
            record["status"] = "ignored"
            record["reason"] = f"n_clusters={clusters}"
            return record
    if not _metrics_ready(run_dir) or not _predictions_ready(run_dir, cpu, resolved):
        record["status"] = "incomplete"
        record["reason"] = "five-seed metrics or predictions are incomplete"
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


def _argv(command: object) -> list[str]:
    if isinstance(command, dict):
        if command.get("argv"):
            return [str(part) for part in command["argv"]]
        command = command.get("command", "")
    if isinstance(command, (list, tuple)):
        return [str(part) for part in command]
    return shlex.split(str(command))


def _flag_values(argv: list[str], name: str) -> list[str]:
    values: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == name and index + 1 < len(argv) and not str(argv[index + 1]).startswith("--"):
            values.append(str(argv[index + 1]))
            index += 2
            continue
        prefix = name + "="
        if token.startswith(prefix):
            values.append(token[len(prefix) :])
        index += 1
    return values


def process_start_id(pid: int) -> str:
    """Kernel start token. A reused PID has a different token."""
    path = Path(f"/proc/{int(pid)}/stat")
    try:
        text = path.read_text(encoding="utf-8")
        rest = text.rsplit(")", 1)[1].split()
        return rest[19]
    except (OSError, IndexError):
        return ""


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
            raw = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        argv = [part.decode("utf-8", "replace") for part in raw if part]
        if not any("train_r_only" in token or token.endswith("run_rf.py") or "train_worker" in token for token in argv):
            continue
        rows.append({"pid": int(entry.name), "argv": argv, "command": argv, "start_id": process_start_id(int(entry.name))})
    return rows


def _command_model_seed(argv: list[str], *, cpu: bool) -> int | None:
    """Fit seed claimed by one process, or None when the flags cannot be claimed.

    Neural commands read ``--model-seed``, or the older ``--seed`` when that
    flag is absent. Random forest reads ``--seed`` and does not take
    ``--model-seed``. When both flags are present they must agree. Omitting
    both is the historical default 42, so that process matches only model seed 42.
    """
    model_flags = _flag_values(argv, "--model-seed")
    seed_flags = _flag_values(argv, "--seed")
    try:
        model_value = int(model_flags[-1]) if model_flags else None
        seed_value = int(seed_flags[-1]) if seed_flags else None
    except ValueError:
        return None
    if model_value is not None and seed_value is not None and model_value != seed_value:
        return None
    if cpu:
        if seed_flags:
            return seed_value
        if model_flags:
            return None
        return 42
    if model_value is not None:
        return model_value
    if seed_value is not None:
        return seed_value
    return 42


def command_matches_item(command: object, item: dict) -> bool:
    """Match one argv vector. ``all`` does not match ``all_sp_c``.

    An omitted ``--train-sampling-seed`` means sampling seed 42. The model
    seed is read separately and does not select the vehicle sample.
    """
    argv = _argv(command)
    if not argv:
        return False
    script = str(item["script"])
    if not any(token == script or token.endswith("/" + script) for token in argv):
        return False
    datasets = _flag_values(argv, "--dataset-id")
    if datasets:
        if datasets[-1] != str(item["dataset_id"]):
            return False
    elif str(item["dataset_id"]) != "fixed_uniform":
        return False
    inputs = _flag_values(argv, "--input-type")
    targets = _flag_values(argv, "--target-type")
    input_type = inputs[-1] if inputs else "inflow"
    target_type = targets[-1] if targets else "inflow"
    if input_type != str(item["input_type"]) or target_type != str(item["target_type"]):
        return False
    seeds = _flag_values(argv, "--train-sampling-seed")
    try:
        train_seed = int(seeds[-1]) if seeds else 42
    except ValueError:
        return False
    if train_seed != int(item["train_sampling_seed"]):
        return False
    fit_seed = _command_model_seed(argv, cpu=bool(item.get("cpu")))
    if fit_seed is None or fit_seed != int(item["model_seed"]):
        return False
    run_ids = _flag_values(argv, "--run-id")
    if item.get("run_id") and run_ids and str(item["run_id"]) not in run_ids:
        return False
    return True


def _process_matches_attempt(row: dict, item: dict) -> bool:
    argv = _argv(row)
    run_id = str(item.get("run_id") or "")
    worker = str(item.get("worker_config") or "")
    if run_id and (run_id in argv or run_id in _flag_values(argv, "--run-id")):
        return True
    if worker and worker in argv:
        return True
    return False


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
            cache_key = (item["dataset_id"], item["observation_key"])
            if training_available and cache_key not in asset_cache:
                try:
                    asset_cache[cache_key] = load_observation_assets(training_root, item["dataset_id"], item["observation_key"])
                except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    asset_cache[cache_key] = {"error": str(exc)}
            assets = asset_cache.get(cache_key) or {}
            if isinstance(assets, dict) and assets.get("error"):
                item["status"] = "unreadable"
                item["reason"] = assets["error"]
                anomalies.append({"path": item["dataset_id"], "reason": assets["error"]})
                continue
            blocked = data_block(item, dataset["status"], assets if isinstance(assets, dict) else None)
            if blocked:
                item["status"] = "blocked"
                item["reason"] = blocked
                continue
            matched = [proc for proc in (processes or []) if command_matches_item(proc, item)]
            if matched:
                item["status"] = "active"
                item["pid"] = matched[0].get("pid")
                item["process_start_id"] = str(matched[0].get("start_id") or "")
                item["tracked"] = "external"
                item["reason"] = "live training process"
                anomalies.append({"item_id": item["item_id"], "reason": "active process", "pid": item["pid"]})
                continue
            item["status"] = "pending"
            item["reason"] = None
        _refresh_group_status(group)
    counts = count_items(groups)
    _expected_groups, expected_items = matrix_shape()
    if sum(counts.values()) != expected_items:
        raise RuntimeError(f"status counts {counts} do not sum to {expected_items}")
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
    """One reused directory per sampling/model seed pair, labeled with both."""
    examples = []
    seen: set[tuple[int, int]] = set()
    for group in groups:
        sample = int(group["train_sampling_seed"])
        model_seed = int(group["model_seed"])
        if (sample, model_seed) in seen:
            continue
        for item in group["items"]:
            if item["status"] == "reusable" and item.get("reused_run_dir"):
                examples.append(f"s={sample} m={model_seed} {item['reused_run_dir']}")
                seen.add((sample, model_seed))
                break
        if len(examples) >= limit:
            return examples
    return examples


def count_items(groups: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for group in groups:
        for item in group["items"]:
            counts[item["status"]] = counts.get(item["status"], 0) + 1
    return counts


def distribution(groups: list[dict]) -> dict[str, dict[str, dict[str, dict[str, dict[str, int]]]]]:
    """Counts by observation band, sampling seed s, model seed m, and task."""
    table: dict[str, dict[str, dict[str, dict[str, dict[str, int]]]]] = {}
    for group in groups:
        band = table.setdefault(str(group["observation_key"]), {})
        sample = band.setdefault(f"s{int(group['train_sampling_seed'])}", {})
        fitted = sample.setdefault(f"m{int(group['model_seed'])}", {})
        for item in group["items"]:
            family = fitted.setdefault(str(item["family"]), {})
            family[item["status"]] = family.get(item["status"], 0) + 1
    return table


def _holds_gpu(group: dict) -> bool:
    """Keep a card between this group's own items. An external active job does not reserve it."""
    if group.get("cpu") or not group.get("gpu_uuid"):
        return False
    if any(item["status"] in {"running", "starting"} for item in group["items"]):
        return True
    if any(item["status"] == "active" for item in group["items"]):
        return False
    return any(item["status"] == "pending" for item in group["items"])


def _refresh_group_status(group: dict) -> None:
    statuses = {item["status"] for item in group["items"]}
    if any(item["status"] in {"running", "starting"} for item in group["items"]):
        group["status"] = "running"
        return
    if "active" in statuses:
        group["gpu_uuid"] = None
        group["gpu_index"] = None
        group["status"] = "active"
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
    elif statuses <= TERMINAL_SUCCESS:
        group["status"] = "completed"
    elif "blocked" in statuses:
        group["status"] = "blocked"
    elif "unverified" in statuses:
        group["status"] = "unverified"
    else:
        group["status"] = "pending"


def command_for(item: dict, python: Path, training_root: Path, results_root: Path, threads: int) -> list[str]:
    model_seed = int(item["model_seed"])
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
            str(model_seed),
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
        str(model_seed),
        "--seed",
        str(model_seed),
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
        int(item["model_seed"]),
        physical_version=PHYSICAL_VERSION,
        dataset_id=item["dataset_id"],
        model_name=item["model"],
        experiment_id=item["experiment_id"],
        run_id=str(item["run_id"]),
    )


def rf_worker_path(run_id: str, observation_key: str) -> Path:
    return model_report_dir("RF") / str(run_id) / f"worker_{observation_key}.json"


def child_env(item: dict, gpu: dict | int | None, threads: int) -> dict[str, str]:
    """Bind a CUDA child to the rechecked UUID and record its physical index."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CODE)
    env["PYTHONUNBUFFERED"] = "1"
    text = str(threads)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = text
    env.pop("TRAFFICFLOW_PHYSICAL_GPU_INDEX", None)
    env.pop("TRAFFICFLOW_GPU_UUID", None)
    if item.get("cpu") or gpu is None:
        env["CUDA_VISIBLE_DEVICES"] = ""
        return env
    if isinstance(gpu, dict):
        env["CUDA_VISIBLE_DEVICES"] = str(gpu["uuid"])
        env["TRAFFICFLOW_GPU_UUID"] = str(gpu["uuid"])
        env["TRAFFICFLOW_PHYSICAL_GPU_INDEX"] = str(gpu.get("minor_number", gpu.get("index")))
        return env
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TRAFFICFLOW_PHYSICAL_GPU_INDEX"] = str(gpu)
    return env


def _process_list_state(gpu: ET.Element) -> tuple[str, str, list[dict]]:
    """Return occupancy, reason, and process rows.

    Only an explicit empty ``<processes>`` list is free. A missing list, an
    ``N/A`` list, or any process record is not free.
    """
    holder = gpu.find("processes")
    if holder is None:
        return "unknown", "processes element missing", []
    infos = list(holder.findall("process_info"))
    if infos:
        processes = []
        for info in infos:
            pid_text = (info.findtext("pid") or "").strip()
            processes.append(
                {
                    "pid": int(pid_text) if pid_text.isdigit() else None,
                    "type": (info.findtext("type") or "").strip() or "unknown",
                    "name": (info.findtext("process_name") or "").strip(),
                    "pid_readable": pid_text.isdigit(),
                }
            )
        return "busy", "process record present", processes
    text = "".join(holder.itertext()).strip()
    if text == "":
        return "free", "", []
    return "unknown", f"process list unreadable: {text[:80]}", []


def parse_gpu_xml(xml_text: str) -> list[dict]:
    """Cards, driver index, minor number, and UUID. Unknown process lists are not free."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise RuntimeError(f"nvidia-smi XML is not parseable: {exc}") from exc
    cards = []
    for driver_index, gpu in enumerate(root.findall("gpu")):
        uuid = (gpu.findtext("uuid") or "").strip()
        minor_text = (gpu.findtext("minor_number") or "").strip()
        occupancy, reason, processes = _process_list_state(gpu)
        identity_ok = bool(uuid) and minor_text.isdigit()
        if not identity_ok:
            occupancy = "unknown"
            reason = reason or "gpu uuid or minor_number is unreadable"
        cards.append(
            {
                "index": driver_index,
                "minor_number": int(minor_text) if minor_text.isdigit() else None,
                "uuid": uuid,
                "processes": processes,
                "occupancy": occupancy,
                "reason": reason,
                "query_ok": identity_ok and occupancy != "unknown",
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
    return card.get("occupancy") == "free" and bool(card.get("query_ok"))


def parent_gpu_allowance() -> tuple[str, set[int] | None, set[str] | None]:
    """Distinguish an unset mask, an explicit empty mask, and a device list.

    The scheduler can only narrow this set. An empty parent mask selects no card.
    """
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        return "unset", None, None
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if raw.strip() == "" or raw.strip().lower() in {"-1", "none", "void", "nodevfiles"}:
        return "empty", set(), set()
    if raw.strip().lower() == "all":
        return "unset", None, None
    indexes: set[int] = set()
    uuids: set[str] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if token.isdigit():
            indexes.add(int(token))
        else:
            uuids.add(token)
    return "restricted", indexes, uuids


def allowed_cards(cards: list[dict], gpu_indexes: set[int] | None, gpu_uuids: set[str] | None) -> list[dict]:
    mode, visible_indexes, visible_uuids = parent_gpu_allowance()
    if mode == "empty":
        return []
    chosen = []
    for card in cards:
        if not card.get("query_ok"):
            continue
        if gpu_indexes is not None and int(card["index"]) not in gpu_indexes:
            continue
        if gpu_uuids is not None and card["uuid"] not in gpu_uuids:
            continue
        if visible_indexes is not None or visible_uuids is not None:
            index_ok = visible_indexes is not None and int(card["index"]) in visible_indexes
            minor_ok = visible_indexes is not None and card.get("minor_number") in visible_indexes
            uuid_ok = visible_uuids is not None and card["uuid"] in visible_uuids
            if not (index_ok or minor_ok or uuid_ok):
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


def _matching_process(item: dict, rows: list[dict]) -> dict | None:
    pid = int(item.get("pid") or 0)
    for row in rows:
        row_pid = int(row.get("pid") or 0)
        if pid and row_pid and row_pid != pid:
            continue
        if not (_process_matches_attempt(row, item) or (not item.get("run_id") and command_matches_item(row, item))):
            continue
        return row
    return None


def adopt_running(groups: list[dict], processes: list[dict] | None) -> None:
    """Take back a live attempt only when PID, start token, and run id agree."""
    rows = list(processes or [])
    for group in groups:
        for item in group["items"]:
            if item["status"] not in {"running", "starting", "pending"}:
                continue
            matched = _matching_process(item, rows) if item.get("run_id") or item.get("worker_config") else None
            if matched is not None:
                start = str(matched.get("start_id") or "")
                recorded = str(item.get("process_start_id") or "")
                if recorded and start and recorded != start:
                    matched = None
                    item["reason"] = "pid reused"
            if matched is None and item["status"] == "pending" and not item.get("run_id"):
                for row in rows:
                    if command_matches_item(row, item):
                        item["status"] = "active"
                        item["pid"] = row.get("pid")
                        item["process_start_id"] = str(row.get("start_id") or "")
                        item["reason"] = "live training process"
                        item["tracked"] = "external"
                        matched = "external"
                        break
                if matched == "external":
                    continue
            if matched is None:
                if item["status"] == "pending":
                    continue
                item["finished_at"] = stamp()
                item["exit_code"] = None
                if _item_artifacts_complete(item):
                    item["status"] = "completed"
                    item["reason"] = "process ended; exit code unknown"
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
            item["tracked"] = "adopted"
            item["process_start_id"] = str(matched.get("start_id") or item.get("process_start_id") or "")
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
        str(record.get("run_id") or "") == str(item.get("run_id") or "")
        and record.get("model") in {item["model"], ""}
        and int(record.get("train_sampling_seed") or -1) == int(item["train_sampling_seed"])
        and int(record.get("model_seed") if record.get("model_seed") is not None else -1) == int(item["model_seed"])
        and record.get("input_type") in {item["input_type"], ""}
        and record.get("target_type") in {item["target_type"], ""}
    )


def next_item(group: dict) -> dict | None:
    """Do not start another item while any item in the group is still in flight."""
    if any(item["status"] in {"running", "starting", "active"} for item in group["items"]):
        return None
    for item in group["items"]:
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
    if not isinstance(groups, list):
        raise RuntimeError("groups.json does not contain the campaign groups")
    _assert_queue_matrix(groups)
    return groups


def _archive_attempt(item: dict) -> None:
    if not item.get("run_id") and not item.get("run_dir") and not item.get("pid"):
        return
    history = list(item.get("attempt_history") or [])
    history.append(
        {
            "run_id": item.get("run_id"),
            "run_dir": item.get("run_dir"),
            "pid": item.get("pid"),
            "command": list(item.get("command") or []),
            "log": item.get("log"),
            "reason": item.get("reason"),
            "exit_code": item.get("exit_code"),
            "started_at": item.get("started_at"),
            "finished_at": item.get("finished_at"),
            "gpu_uuid": item.get("gpu_uuid"),
        }
    )
    item["attempt_history"] = history


def _launch_item(ops, groups, group, item, gpu, python, training_root, results_root, threads, launch, cap: int) -> dict | None:
    """Start a new attempt. An adopted attempt is not launched again."""
    if item.get("run_dir") and item.get("run_id") and _item_artifacts_complete(item):
        item["status"] = "completed"
        item["reason"] = None
        return None
    _archive_attempt(item)
    item["run_id"] = new_attempt_id()
    item["run_dir"] = str(output_dir_for(item, results_root))
    item["command"] = command_for(item, python, training_root, results_root, threads)
    item["worker_config"] = str(rf_worker_path(item["run_id"], item["observation_key"])) if item.get("cpu") else None
    item["status"] = "starting"
    item["started_at"] = stamp()
    item["finished_at"] = None
    item["exit_code"] = None
    item["pid"] = None
    item["process_start_id"] = ""
    item["tracked"] = "child"
    item["attempt"] = int(item.get("attempt") or 0) + 1
    item["gpu_index"] = None if gpu is None else int(gpu.get("index"))
    item["gpu_uuid"] = None if gpu is None else str(gpu.get("uuid"))
    if gpu is not None:
        group["gpu_index"] = int(gpu["index"])
        group["gpu_uuid"] = str(gpu["uuid"])
        group["status"] = "running"
    env = child_env(item, gpu, threads)
    item["launch_env"] = {
        "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", ""),
        "TRAFFICFLOW_PHYSICAL_GPU_INDEX": env.get("TRAFFICFLOW_PHYSICAL_GPU_INDEX", ""),
        "TRAFFICFLOW_GPU_UUID": env.get("TRAFFICFLOW_GPU_UUID", ""),
    }
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
            "physical_gpu_index": item["launch_env"]["TRAFFICFLOW_PHYSICAL_GPU_INDEX"],
            "started_at": item["started_at"],
            "attempt": item["attempt"],
        },
    )
    save_state(ops, groups, cap)
    proc = launch(item, gpu, log_path)
    item["pid"] = int(proc.pid)
    item["status"] = "running"
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
        env=child_env(item, gpu, threads),
        stdout=handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc


def wait_for_control(seconds: float) -> None:
    """Sleep until the poll interval elapses or this process receives SIGHUP/SIGTERM."""
    deadline = time.monotonic() + float(seconds)
    while time.monotonic() < deadline:
        if SIGNALS.reload or SIGNALS.stop:
            return
        time.sleep(0.05)


def _roots_problem(training_root: Path, results_root: Path) -> str | None:
    if not Path(results_root).exists():
        return f"results root is not mounted: {results_root}"
    if not Path(training_root).is_dir():
        return f"training root is not a directory: {training_root}"
    return None


def _protect_live_attempts(groups: list[dict]) -> dict[str, dict]:
    protected = {}
    for group in groups:
        for item in group["items"]:
            if item["status"] in {"running", "starting", "completed"}:
                protected[item["item_id"]] = {
                    key: item.get(key)
                    for key in (
                        "status", "run_id", "run_dir", "pid", "command", "log", "gpu_uuid", "gpu_index",
                        "process_start_id", "attempt", "reason", "attempt_history", "worker_config",
                        "started_at", "tracked", "reused_run_dir", "source_experiment_id", "evidence",
                    )
                }
    return protected


def _restore_protected(groups: list[dict], protected: dict[str, dict]) -> None:
    for group in groups:
        for item in group["items"]:
            saved = protected.get(item["item_id"])
            if saved is not None:
                item.update(saved)
        _refresh_group_status(group)


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
    poll_seconds: float = POLL_SECONDS,
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
    sleep = sleep or (lambda seconds: wait_for_control(seconds))

    def current_processes() -> list[dict]:
        if processes is None:
            return live_commands()
        if callable(processes):
            return list(processes())
        return list(processes)

    python = python or LEGACY.training_python()
    saved_payload = read_json(ops / "groups.json")
    saved_cap = None
    if isinstance(saved_payload, dict) and isinstance(saved_payload.get("max_gpus"), int) and not isinstance(saved_payload.get("max_gpus"), bool):
        saved_cap = int(saved_payload["max_gpus"])
    control_path = ops / "control.json"
    if not control_path.is_file():
        atomic_write_json(control_path, {"max_gpus": DEFAULT_MAX_GPUS if saved_cap is None else saved_cap})
    groups = load_state(ops)
    created = groups is None
    root_problem = _roots_problem(training_root, results_root)
    if root_problem:
        log_line(ops, root_problem)
        lock.seek(0)
        lock.truncate()
        lock.close()
        raise SystemExit(root_problem)
    if groups is None:
        groups = build_groups()
        report = apply_inventory(groups, training_root=training_root, results_root=results_root, processes=current_processes())
    else:
        protected = _protect_live_attempts(groups)
        report = apply_inventory(groups, training_root=training_root, results_root=results_root, processes=current_processes())
        _restore_protected(groups, protected)
    adopt_running(groups, current_processes())
    procs: dict[str, object] = {}
    cap = refresh_cap(ops, DEFAULT_MAX_GPUS if saved_cap is None else saved_cap)
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
            for card in cards:
                if card.get("occupancy") == "unknown":
                    log_line(ops, f"skip GPU index={card.get('index')} uuid={card.get('uuid')}: {card.get('reason')}")
            snapshot = current_processes()
            _reap(ops, groups, procs, snapshot)
            _refresh_active(groups, snapshot, results_root)
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
            active = sum(item["status"] == "active" for group in groups for item in group["items"])
            log_line(ops, f"poll max={cap} running_items={running} queued_items={pending} active_items={active}")
            if pending == 0 and running == 0 and active == 0:
                break
            polls += 1
            if max_polls is not None and polls >= max_polls:
                break
            sleep(poll_seconds)
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
        item["error"] = f"launch failed: {exc}"
        item["reason"] = item["error"]
        item["finished_at"] = stamp()
        log_line(ops, f"FAILED {item['item_id']} {exc}")
        return
    if handle is None:
        return
    procs[item["item_id"]] = handle["proc"]
    where = "CPU" if gpu is None else f"GPU {gpu['index']}"
    log_line(ops, f"{where} START {item['item_id']} pid={item['pid']} run_id={item['run_id']}")


def _living_worker(item: dict, rows: list[dict]) -> dict | None:
    worker = str(item.get("worker_config") or "")
    if not worker:
        return None
    for row in rows:
        if worker in _argv(row):
            return row
    return None


def _finish_attempt(item: dict, code: int | None, *, known: bool, reason: str) -> None:
    item["finished_at"] = stamp()
    item["exit_code"] = code if known else None
    if _item_artifacts_complete(item) and (not known or code == 0):
        item["status"] = "completed"
        item["reason"] = None if known else "process ended; exit code unknown"
        return
    item["status"] = "failed"
    item["reason"] = reason if not known else f"exit {code}"


def _reap(ops: Path, groups: list[dict], procs: dict[str, object], rows: list[dict]) -> None:
    """Finish child processes and adopted processes. Adopted exits have no wait status."""
    for group in groups:
        for item in group["items"]:
            if item["status"] not in {"running", "starting"}:
                continue
            proc = procs.get(item["item_id"])
            if proc is not None:
                code = proc.poll()
                if code is None:
                    continue
                worker = _living_worker(item, rows)
                if worker is not None:
                    procs.pop(item["item_id"], None)
                    item["pid"] = int(worker["pid"])
                    item["process_start_id"] = str(worker.get("start_id") or "")
                    item["tracked"] = "worker"
                    item["launcher_exit_code"] = code
                    continue
                _finish_attempt(item, code, known=True, reason=f"exit {code}")
                log_line(ops, f"{item['status'].upper()} {item['item_id']} exit={code}")
                procs.pop(item["item_id"], None)
                continue
            pid = int(item.get("pid") or 0)
            same_pid = next((row for row in rows if int(row.get("pid") or 0) == pid), None) if pid else None
            if same_pid is not None:
                start = str(same_pid.get("start_id") or "")
                recorded = str(item.get("process_start_id") or "")
                if recorded and start and recorded != start:
                    _finish_attempt(item, None, known=False, reason="pid reused")
                    log_line(ops, f"PID REUSED {item['item_id']} pid={pid}")
                    continue
                if _process_matches_attempt(same_pid, item):
                    if start and not recorded:
                        item["process_start_id"] = start
                    continue
                _finish_attempt(item, None, known=False, reason="pid reused")
                log_line(ops, f"PID REUSED {item['item_id']} pid={pid}")
                continue
            _finish_attempt(item, None, known=False, reason="process ended; exit code unknown")
            log_line(ops, f"{item['status'].upper()} {item['item_id']} exit=unknown")
        _refresh_group_status(group)


def _refresh_active(groups: list[dict], rows: list[dict], results_root: Path) -> None:
    """Recheck external same-task processes and convert ended ones from their artifacts."""
    for group in groups:
        changed = False
        for item in group["items"]:
            if item["status"] != "active":
                continue
            matched = None
            for row in rows:
                if int(row.get("pid") or -1) != int(item.get("pid") or -2):
                    continue
                if not command_matches_item(row, item):
                    continue
                start = str(row.get("start_id") or "")
                recorded = str(item.get("process_start_id") or "")
                if recorded and start and recorded != start:
                    continue
                matched = row
                break
            if matched is not None:
                continue
            changed = True
            found = _reusable_for_item(item, results_root)
            item["pid"] = None
            item["finished_at"] = stamp()
            item["exit_code"] = None
            if found is not None:
                item["status"] = "reusable"
                item["run_dir"] = found["run_dir"]
                item["reused_run_dir"] = found["run_dir"]
                item["source_experiment_id"] = found.get("experiment_id")
                item["evidence"] = found.get("reason")
                item["reason"] = "external process ended; existing result is complete"
            else:
                item["status"] = "pending"
                item["reason"] = "external process ended; exit code unknown"
        if changed:
            _refresh_group_status(group)


def _reusable_for_item(item: dict, results_root: Path) -> dict | None:
    base = (
        Path(results_root) / "datasets" / item["dataset_id"] / f"physical_{PHYSICAL_VERSION}" / item["model"]
        / f"in_{item['input_type']}__out_{item['target_type']}" / item["observation_key"]
    )
    if not base.is_dir():
        return None
    identity = (
        item["model"],
        item["dataset_id"],
        item["input_type"],
        item["target_type"],
        int(item["train_sampling_seed"]),
        int(item["model_seed"]),
    )
    try:
        run_dirs = [path for experiment in base.iterdir() if experiment.is_dir() for seed_dir in experiment.glob("seed_*") if seed_dir.is_dir() for path in seed_dir.iterdir() if path.is_dir()]
    except OSError:
        return None
    for run_dir in sorted(run_dirs, key=lambda path: str(path)):
        record = classify_run(run_dir)
        if record.get("status") == "reusable" and _result_identity(record) == identity:
            return record
    return None


def _request_reload(signum, frame) -> None:
    del signum, frame
    SIGNALS.reload = True


def _request_stop(signum, frame) -> None:
    del signum, frame
    SIGNALS.stop = True


def requeue_failed(ops: Path) -> int:
    """Requeue under the same exclusive lock the running scheduler holds."""
    lock = acquire_lock(ops)
    try:
        groups = load_state(ops)
        if groups is None:
            raise SystemExit(f"no queue at {ops / 'groups.json'}")
        changed = 0
        for group in groups:
            for item in group["items"]:
                if item["status"] != "failed":
                    continue
                if item.get("run_dir") and _item_artifacts_complete(item):
                    item["status"] = "completed"
                    item["reason"] = None
                    continue
                if int(item.get("attempt") or 0) >= MAX_ATTEMPTS:
                    continue
                _archive_attempt(item)
                item["status"] = "pending"
                item["run_id"] = None
                item["run_dir"] = None
                item["pid"] = None
                item["command"] = []
                item["log"] = None
                item["exit_code"] = None
                item["finished_at"] = None
                item["worker_config"] = None
                changed += 1
            _refresh_group_status(group)
        control = read_json(ops / "control.json") or {}
        cap = control.get("max_gpus") if isinstance(control, dict) and isinstance(control.get("max_gpus"), int) and not isinstance(control.get("max_gpus"), bool) else DEFAULT_MAX_GPUS
        save_state(ops, groups, int(cap))
        return changed
    finally:
        lock.seek(0)
        lock.truncate()
        lock.close()


def format_plan(groups: list[dict], report: dict) -> str:
    lines = [
        f"campaign={CAMPAIGN_ID}",
        f"train_sampling_seeds={list(TRAIN_SAMPLING_SEEDS)}",
        f"model_seeds={list(MODEL_SEEDS)}",
        "io_pairs=" + ",".join(row["family"] for row in io_pairs()),
        f"inventory={report['inventory_status']}",
        f"groups={len(groups)} items={sum(len(group['items']) for group in groups)}",
        f"counts={report['counts']}",
        f"reusable={report['reusable']}",
        f"to_run={report['to_run']}",
        f"anomalies={len(report['anomalies'])}",
    ]
    for example in report.get("reusable_examples") or []:
        lines.append(f"reused {example}")
    for dataset_id, row in sorted((report.get("datasets") or {}).items()):
        if row.get("status") not in {None, "ready"}:
            lines.append(f"dataset {dataset_id} {row.get('status')} {row.get('detail')}")
    for anomaly in report["anomalies"][:50]:
        lines.append(f"anomaly {anomaly}")
    lines.append("distribution=" + json.dumps(report.get("distribution") or {}, sort_keys=True))
    return "\n".join(lines)


def write_summary_csv(ops: Path, groups: list[dict]) -> None:
    fieldnames = (
        "model",
        "dataset_id",
        "train_sampling_seed",
        "model_seed",
        "input_type",
        "target_type",
        "status",
        "run_dir",
        "reused_run_dir",
        "reason",
    )
    rows = []
    for group in groups:
        for item in group["items"]:
            rows.append(
                {
                    "model": item["model"],
                    "dataset_id": item["dataset_id"],
                    "train_sampling_seed": item["train_sampling_seed"],
                    "model_seed": item["model_seed"],
                    "input_type": item["input_type"],
                    "target_type": item["target_type"],
                    "status": item["status"],
                    "run_dir": item.get("run_dir") or "",
                    "reused_run_dir": item.get("reused_run_dir") or "",
                    "reason": item.get("reason") or "",
                }
            )
    temporary = ops / "campaign_summary.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, ops / "campaign_summary.csv")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan or run the two-band campaign across five sampling seeds and three model seeds."
    )
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
