"""Prepare R-only STGCN tensors from daily full and observed 5-minute edge flows.

Reads already aggregated per-day CSVs. Does not open trajectories, vehicle ID
files, or raw FCD except to SHA256-protect them. Does not train a model.

Task: for each same-day window of n_his observed 5-minute slots, the target is
the full-flow vector at the last slot of that window (current-slot
reconstruction), not the next future slot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


SCRIPT_VERSION = "1.0.0"
TARGET_MODE_LAST_OBSERVED = "last-observed-step"
DEFAULT_RATES = (5, 10, 20, 30, 40, 50, 70)
DEFAULT_N_HIS = 12
DEFAULT_TRAIN_DAYS = 14
DEFAULT_VAL_DAYS = 3
DEFAULT_TEST_DAYS = 3
DEFAULT_EXPECTED_DAYS = 20
DEFAULT_EXPECTED_NODES = 56
DEFAULT_EXPECTED_SLOTS = 288
DEFAULT_WINDOW_SECONDS = 300
SECONDS_PER_DAY = 24 * 3600
DEFAULT_DDOF = 0
DEFAULT_EPSILON = 1e-8
ZSCORE_MEAN_ATOL = 1e-5
ZSCORE_STD_ATOL = 1e-5
DENORM_ATOL = 1e-4
HASH_CHUNK = 8 * 1024 * 1024
DAY_NAME_RE = re.compile(r"^day_(\d+)$")

PREPARED_DATA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PREPARED_DATA_DIR.parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
DEFAULT_DATA_ROOT = (
    ANALYSIS_DIR / "simulation" / "data" / "processed" / "subgraph_trajectories"
)
DEFAULT_R_NODES = ANALYSIS_DIR / "graph" / "r_graph" / "r_nodes.csv"
DEFAULT_ADJACENCY = ANALYSIS_DIR / "graph" / "r_graph" / "r_adjacency.npy"
DEFAULT_OUTPUT_ROOT = PREPARED_DATA_DIR / "r-only"
DEFAULT_FULL_PATTERN = "{day}/edge_flow_5min.csv"
DEFAULT_OBSERVED_PATTERN = "{day}/observed_edge_flow_5min/edge_flow_5min_p{rate:02d}.csv"

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


class PrepareError(Exception):
    pass


@dataclass
class NodeMapping:
    edge_ids: list[str]
    index_by_edge: dict[str, int]
    rows: list[dict[str, str]]
    fieldnames: list[str]


@dataclass
class DaySpec:
    name: str
    index: int
    directory: Path
    full_flow_path: Path
    observed_paths: dict[int, Path]


@dataclass
class DaySplit:
    train: list[DaySpec]
    validation: list[DaySpec]
    test: list[DaySpec]

    def all_days(self) -> list[DaySpec]:
        return [*self.train, *self.validation, *self.test]

    def names(self, split: str) -> list[str]:
        return [day.name for day in getattr(self, split)]


@dataclass
class Scaler:
    mean: float
    std: float
    n_elements: int
    source: str

    def transform(self, array: np.ndarray) -> np.ndarray:
        return ((array.astype(np.float64) - self.mean) / self.std).astype(np.float32)

    def inverse(self, array: np.ndarray) -> np.ndarray:
        return array.astype(np.float64) * self.std + self.mean


@dataclass
class PreparedSplit:
    x: np.ndarray
    y: np.ndarray
    x_raw: np.ndarray
    y_raw: np.ndarray
    day_index: np.ndarray
    window_start_slot: np.ndarray
    target_slot: np.ndarray


@dataclass
class Logger:
    path: Path | None
    started: float
    records: list[dict[str, object]] = field(default_factory=list)

    def emit(self, stage: str, **payload: object) -> dict[str, object]:
        record = {
            "ts": now_utc(),
            "stage": stage,
            "elapsed_s": round(time.time() - self.started, 3),
            **payload,
        }
        self.records.append(record)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        print(line, flush=True)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()
        return record


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def posix(path: Path) -> str:
    return path.resolve().as_posix()


def relative_posix(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return posix(path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_csv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, object]]) -> None:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    tmp_path = Path(handle.name)
    try:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(tmp_path, path)
    except Exception:
        handle.close()
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def slot_time_label(slot: int, window_seconds: int) -> str:
    seconds = slot * window_seconds
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}"


def parse_rates(values: list[str]) -> list[int]:
    rates: list[int] = []
    for raw in values:
        percent = int(raw)
        if percent <= 0 or percent > 100:
            raise PrepareError(f"invalid penetration percent {raw}")
        rates.append(percent)
    if len(set(rates)) != len(rates):
        raise PrepareError(f"duplicate rates: {rates}")
    return rates


def rate_key(percent: int) -> str:
    return f"p{percent:02d}"


def day_sort_key(name: str) -> tuple[int, int | str]:
    match = DAY_NAME_RE.fullmatch(name)
    if match:
        return (0, int(match.group(1)))
    return (1, name)


def discover_day_directories(data_root: Path, full_pattern: str) -> list[Path]:
    if not data_root.is_dir():
        raise PrepareError(f"data root is not a directory: {data_root}")
    found: list[Path] = []
    for child in data_root.iterdir():
        if not child.is_dir():
            continue
        full_path = data_root / full_pattern.format(day=child.name, rate=0)
        if full_path.is_file():
            found.append(child)
    found.sort(key=lambda path: day_sort_key(path.name))
    return found


def resolve_input_files(
    data_root: Path,
    day_dirs: list[Path],
    rates: list[int],
    full_pattern: str,
    observed_pattern: str,
) -> list[DaySpec]:
    days: list[DaySpec] = []
    missing: list[str] = []
    for index, day_dir in enumerate(day_dirs):
        full_path = data_root / full_pattern.format(day=day_dir.name, rate=0)
        observed_paths: dict[int, Path] = {}
        if not full_path.is_file():
            missing.append(str(full_path))
        for percent in rates:
            observed = data_root / observed_pattern.format(day=day_dir.name, rate=percent)
            observed_paths[percent] = observed
            if not observed.is_file():
                missing.append(str(observed))
        days.append(
            DaySpec(
                name=day_dir.name,
                index=index,
                directory=day_dir,
                full_flow_path=full_path,
                observed_paths=observed_paths,
            )
        )
    if missing:
        raise PrepareError("missing input files:\n" + "\n".join(missing))
    return days


def load_node_mapping(path: Path, expected_nodes: int) -> NodeMapping:
    if not path.is_file():
        raise PrepareError(f"r_nodes.csv not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise PrepareError(f"{path} has no header")
        fieldnames = list(reader.fieldnames)
        if "node_index" not in fieldnames or "edge_id" not in fieldnames:
            raise PrepareError(f"{path} must contain node_index and edge_id")
        rows = [{key: (row[key] if row[key] is not None else "") for key in fieldnames} for row in reader]
    parsed = sorted(((int(row["node_index"]), row["edge_id"], row) for row in rows), key=lambda item: item[0])
    if [index for index, _edge, _row in parsed] != list(range(len(parsed))):
        raise PrepareError(f"{path} node_index is not contiguous from 0")
    edge_ids = [edge for _index, edge, _row in parsed]
    if len(set(edge_ids)) != len(edge_ids):
        raise PrepareError(f"{path} has duplicate edge_id values")
    if any(not edge for edge in edge_ids):
        raise PrepareError(f"{path} has empty edge_id values")
    if len(edge_ids) != expected_nodes:
        raise PrepareError(f"{path} has {len(edge_ids)} nodes, expected {expected_nodes}")
    ordered_rows = [row for _index, _edge, row in parsed]
    return NodeMapping(
        edge_ids=edge_ids,
        index_by_edge={edge: index for index, edge in enumerate(edge_ids)},
        rows=ordered_rows,
        fieldnames=fieldnames,
    )


def load_adjacency(path: Path, n_nodes: int) -> np.ndarray:
    if not path.is_file():
        raise PrepareError(f"adjacency not found: {path}")
    matrix = np.load(path)
    if matrix.shape != (n_nodes, n_nodes):
        raise PrepareError(f"{path} shape {matrix.shape} != [{n_nodes}, {n_nodes}]")
    return matrix


def parse_non_negative_int(raw: str, path: Path) -> int:
    text = raw.strip()
    if not text.isdigit():
        raise PrepareError(f"{path} has non-integer or negative flow {raw!r}")
    return int(text)


def build_daily_flow_matrix(
    path: Path,
    mapping: NodeMapping,
    *,
    n_slots: int,
    window_seconds: int,
    expected_day: str | None = None,
    expected_percent: int | None = None,
) -> np.ndarray:
    n_nodes = len(mapping.edge_ids)
    grid = np.zeros((n_slots, n_nodes), dtype=np.int64)
    filled = np.zeros((n_slots, n_nodes), dtype=bool)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise PrepareError(f"{path} has no header")
        required = ("window_index", "edge_id", "vehicle_count")
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise PrepareError(f"{path} missing fields {missing}")
        for row in reader:
            edge_id = row["edge_id"]
            if edge_id not in mapping.index_by_edge:
                raise PrepareError(f"{path} unknown edge_id {edge_id}")
            node_index = mapping.index_by_edge[edge_id]
            if "node_index" in row and row["node_index"] != "":
                if int(row["node_index"]) != node_index:
                    raise PrepareError(
                        f"{path} node_index {row['node_index']} != mapping {node_index} for {edge_id}"
                    )
            if expected_day is not None and "day" in row and row["day"] not in {"", expected_day}:
                raise PrepareError(f"{path} contains day {row['day']!r}, expected {expected_day}")
            if expected_percent is not None and "penetration_rate" in row and row["penetration_rate"] != "":
                if abs(float(row["penetration_rate"]) - expected_percent / 100.0) > 1e-12:
                    raise PrepareError(
                        f"{path} penetration_rate {row['penetration_rate']} != {expected_percent / 100:.2f}"
                    )
            slot = int(row["window_index"])
            if slot < 0 or slot >= n_slots:
                raise PrepareError(f"{path} window_index {slot} outside 0..{n_slots - 1}")
            if filled[slot, node_index]:
                raise PrepareError(f"{path} duplicate key window={slot} edge_id={edge_id}")
            if "window_start" in row and row["window_start"] != "":
                if int(row["window_start"]) != slot * window_seconds:
                    raise PrepareError(
                        f"{path} window_start {row['window_start']} != {slot * window_seconds} at slot {slot}"
                    )
            grid[slot, node_index] = parse_non_negative_int(row["vehicle_count"], path)
            filled[slot, node_index] = True
    if not filled.all():
        missing_n = int(np.size(filled) - np.count_nonzero(filled))
        raise PrepareError(
            f"{path} is missing {missing_n} time-slot/node cells; "
            "cannot treat missing rows as zero without an explicit sparse-zero contract"
        )
    return grid


def validate_nested_day(
    day: DaySpec,
    observed: dict[int, np.ndarray],
    full: np.ndarray,
    rates: list[int],
    mapping: NodeMapping,
) -> None:
    stack = np.stack([observed[percent] for percent in rates] + [full], axis=0)
    if np.any(stack < 0):
        raise PrepareError(f"{day.name} has negative flow after matrix construction")
    diffs = np.diff(stack.astype(np.int64), axis=0)
    bad = np.argwhere(diffs < 0)
    if bad.size:
        t, i = int(bad[0][1]), int(bad[0][2])
        values = {rate_key(percent): int(observed[percent][t, i]) for percent in rates}
        raise PrepareError(
            f"nested/full-bound failed day={day.name} time_slot={t} "
            f"node_index={i} edge_id={mapping.edge_ids[i]} "
            f"observed={values} full={int(full[t, i])} "
            f"files={[posix(day.observed_paths[p]) for p in rates] + [posix(day.full_flow_path)]}"
        )


def split_days(
    days: list[DaySpec],
    *,
    train_day_count: int,
    val_day_count: int,
    test_day_count: int,
    train_days: list[str] | None,
    val_days: list[str] | None,
    test_days: list[str] | None,
) -> DaySplit:
    by_name = {day.name: day for day in days}
    explicit = [train_days, val_days, test_days]
    if any(group is not None for group in explicit):
        if not all(group is not None for group in explicit):
            raise PrepareError("train/val/test day lists must be provided together")
        names = [*train_days, *val_days, *test_days]
        if len(set(names)) != len(names):
            raise PrepareError("explicit day lists contain duplicates")
        missing = [name for name in names if name not in by_name]
        unused = [day.name for day in days if day.name not in set(names)]
        if missing or unused:
            raise PrepareError(
                f"explicit day lists must cover every discovered day exactly; "
                f"missing={missing or 'none'} unused={unused or 'none'}"
            )
        return DaySplit(
            train=[by_name[name] for name in train_days],
            validation=[by_name[name] for name in val_days],
            test=[by_name[name] for name in test_days],
        )
    if train_day_count + val_day_count + test_day_count != len(days):
        raise PrepareError(
            f"train+val+test day counts {train_day_count}+{val_day_count}+{test_day_count} "
            f"!= {len(days)} discovered days"
        )
    train = days[:train_day_count]
    validation = days[train_day_count:train_day_count + val_day_count]
    test = days[train_day_count + val_day_count:]
    return DaySplit(train=train, validation=validation, test=test)


def generate_daily_windows(
    observed: np.ndarray,
    full: np.ndarray,
    n_his: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build same-day windows. Target is full flow at slot s+n_his-1."""
    n_slots, n_nodes = observed.shape
    if full.shape != observed.shape:
        raise PrepareError("observed and full matrices have different shapes")
    if n_slots < n_his:
        raise PrepareError(f"need at least {n_his} slots, found {n_slots}")
    n_windows = n_slots - n_his + 1
    x = np.empty((n_windows, n_his, n_nodes, 1), dtype=np.float32)
    y = np.empty((n_windows, 1, n_nodes, 1), dtype=np.float32)
    starts = np.empty((n_windows,), dtype=np.int32)
    targets = np.empty((n_windows,), dtype=np.int32)
    for start in range(n_windows):
        target = start + n_his - 1
        x[start, :, :, 0] = observed[start:start + n_his]
        y[start, 0, :, 0] = full[target]
        starts[start] = start
        targets[start] = target
    return x, y, starts, targets


def stack_split_windows(
    days: list[DaySpec],
    observed: dict[str, np.ndarray],
    full: dict[str, np.ndarray],
    n_his: int,
) -> PreparedSplit:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    day_indexes: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for day in days:
        x, y, start, target = generate_daily_windows(observed[day.name], full[day.name], n_his)
        xs.append(x)
        ys.append(y)
        day_indexes.append(np.full((x.shape[0],), day.index, dtype=np.int32))
        starts.append(start)
        targets.append(target)
    x_raw = np.concatenate(xs, axis=0)
    y_raw = np.concatenate(ys, axis=0)
    return PreparedSplit(
        x=x_raw,  # placeholder; overwritten after scaling
        y=y_raw,
        x_raw=x_raw,
        y_raw=y_raw,
        day_index=np.concatenate(day_indexes, axis=0),
        window_start_slot=np.concatenate(starts, axis=0),
        target_slot=np.concatenate(targets, axis=0),
    )


def fit_training_scaler(array: np.ndarray, *, source: str, epsilon: float, ddof: int) -> Scaler:
    values = array.astype(np.float64, copy=False)
    mean = float(values.mean())
    std = float(values.std(ddof=ddof))
    if not np.isfinite(mean) or not np.isfinite(std):
        raise PrepareError(f"non-finite scaler for {source}")
    if std < epsilon:
        raise PrepareError(f"std for {source} is {std}, below epsilon {epsilon}; refusing to divide")
    return Scaler(mean=mean, std=std, n_elements=int(values.size), source=source)


def run_synthetic_checks(n_his: int) -> None:
    n_slots, n_nodes = n_his + 8, 4
    observed = np.arange(n_slots * n_nodes, dtype=np.int64).reshape(n_slots, n_nodes)
    full = observed + 100
    x, y, starts, targets = generate_daily_windows(observed, full, n_his)
    expected_n = n_slots - n_his + 1
    if x.shape != (expected_n, n_his, n_nodes, 1) or y.shape != (expected_n, 1, n_nodes, 1):
        raise PrepareError("synthetic window shapes are wrong")
    if not np.array_equal(y[0, 0, :, 0], full[n_his - 1].astype(np.float32)):
        raise PrepareError("synthetic target is not the last observed slot")
    if np.array_equal(y[0, 0, :, 0], full[n_his].astype(np.float32)):
        raise PrepareError("synthetic target accidentally used the next future slot")
    if int(targets[0]) != n_his - 1 or int(starts[-1]) != n_slots - n_his:
        raise PrepareError("synthetic start/target slots are wrong")
    if int(targets[-1]) != int(starts[-1]) + n_his - 1:
        raise PrepareError("synthetic target_slot != window_start_slot + n_his - 1")


def collect_protected_paths(days: list[DaySpec], r_nodes: Path, adjacency: Path) -> list[Path]:
    paths = [r_nodes, adjacency]
    for day in days:
        paths.append(day.full_flow_path)
        paths.extend(day.observed_paths.values())
        trajectories = day.directory / "trajectories.csv"
        if trajectories.is_file():
            paths.append(trajectories)
        sample_dir = day.directory / "sampled_vehicle_ids"
        if sample_dir.is_dir():
            paths.extend(sorted(sample_dir.glob("vehicles_p*.txt")))
    return paths


def hash_paths(paths: Iterable[Path], logger: Logger, stage: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in paths:
        hashes[posix(path)] = sha256_file(path)
        logger.emit(stage, current_file=posix(path), files_hashed=len(hashes), status="running")
    return hashes


def write_status(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    atomic_write_json(path, payload)


def sample_index_rows(
    split_name: str,
    prepared: PreparedSplit,
    days: list[DaySpec],
    window_seconds: int,
    n_his: int,
) -> list[dict[str, object]]:
    by_index = {day.index: day for day in days}
    rows: list[dict[str, object]] = []
    for sample_index, (day_index, start, target) in enumerate(
        zip(prepared.day_index, prepared.window_start_slot, prepared.target_slot, strict=True)
    ):
        start_i = int(start)
        target_i = int(target)
        rows.append(
            {
                "sample_index": sample_index,
                "split": split_name,
                "day_index": int(day_index),
                "day": by_index[int(day_index)].name,
                "window_start_slot": start_i,
                "window_end_slot": target_i,
                "target_slot": target_i,
                "window_start_time": slot_time_label(start_i, window_seconds),
                "window_end_time": slot_time_label(target_i, window_seconds),
                "target_time": slot_time_label(target_i, window_seconds),
                "n_his": n_his,
            }
        )
        if target_i != start_i + n_his - 1:
            raise PrepareError("sample index target_slot != window_start_slot + n_his - 1")
    return rows


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def array_finite(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise PrepareError(f"{name} contains NaN or inf")


def validate_prepared_dataset(
    *,
    output_root: Path,
    days: list[DaySpec],
    split: DaySplit,
    rates: list[int],
    mapping: NodeMapping,
    adjacency: np.ndarray,
    full_matrices: dict[str, np.ndarray],
    observed_matrices: dict[int, dict[str, np.ndarray]],
    prepared: dict[int, dict[str, PreparedSplit]],
    input_scalers: dict[int, Scaler],
    target_scaler: Scaler,
    n_his: int,
    n_slots: int,
    hashes_before: dict[str, str],
    hashes_after: dict[str, str],
    epsilon: float,
) -> dict[str, object]:
    failures: list[str] = []
    n_nodes = len(mapping.edge_ids)
    windows_per_day = n_slots - n_his + 1
    expected_counts = {
        "train": len(split.train) * windows_per_day,
        "validation": len(split.validation) * windows_per_day,
        "test": len(split.test) * windows_per_day,
    }
    shapes: dict[str, dict[str, dict[str, list[int]]]] = {}
    denorm_errors: dict[str, float] = {}
    train_x_moments: dict[str, dict[str, float]] = {}
    train_y_moments: dict[str, float] | None = None
    y_raw_identical = True
    y_norm_identical = True
    index_identical = True

    reference_rate = rates[0]
    for percent in rates:
        key = rate_key(percent)
        shapes[key] = {}
        for split_name in SPLITS:
            path = output_root / key / f"{split_name}.npz"
            loaded = load_npz(path)
            if set(loaded) != set(NPZ_KEYS):
                failures.append(f"{path} keys {sorted(loaded)} != {list(NPZ_KEYS)}")
            current = prepared[percent][split_name]
            for name in NPZ_KEYS:
                if not np.array_equal(loaded[name], getattr(current, name)):
                    failures.append(f"{path}:{name} differs from in-memory array")
            n_samples = expected_counts[split_name]
            if loaded["x"].shape != (n_samples, n_his, n_nodes, 1):
                failures.append(f"{path} x.shape {loaded['x'].shape}")
            if loaded["y"].shape != (n_samples, 1, n_nodes, 1):
                failures.append(f"{path} y.shape {loaded['y'].shape}")
            if loaded["x"].dtype != np.float32 or loaded["y"].dtype != np.float32:
                failures.append(f"{path} normalized dtypes are not float32")
            if loaded["x_raw"].dtype != np.float32 or loaded["y_raw"].dtype != np.float32:
                failures.append(f"{path} raw dtypes are not float32")
            for name in NPZ_KEYS:
                array_finite(f"{path}:{name}", loaded[name])
            if not np.array_equal(loaded["target_slot"], loaded["window_start_slot"] + (n_his - 1)):
                failures.append(f"{path} target_slot != window_start_slot + {n_his - 1}")
            rebuilt_x, rebuilt_y, rebuilt_start, rebuilt_target = [], [], [], []
            split_days_list: list[DaySpec] = getattr(split, split_name)
            for day in split_days_list:
                x, y, start, target = generate_daily_windows(
                    observed_matrices[percent][day.name],
                    full_matrices[day.name],
                    n_his,
                )
                rebuilt_x.append(x)
                rebuilt_y.append(y)
                rebuilt_start.append(start)
                rebuilt_target.append(target)
            rebuilt_x_raw = np.concatenate(rebuilt_x, axis=0)
            rebuilt_y_raw = np.concatenate(rebuilt_y, axis=0)
            if not np.array_equal(rebuilt_x_raw, loaded["x_raw"]):
                failures.append(f"{path} x_raw does not match rebuilt windows")
            if not np.array_equal(rebuilt_y_raw, loaded["y_raw"]):
                failures.append(f"{path} y_raw does not match rebuilt full-flow targets")
            if np.max(np.abs(input_scalers[percent].inverse(loaded["x"]) - loaded["x_raw"])) > DENORM_ATOL:
                failures.append(f"{path} x denorm exceeds atol")
            y_denorm_err = float(np.max(np.abs(target_scaler.inverse(loaded["y"]) - loaded["y_raw"])))
            denorm_errors[f"{key}/{split_name}"] = y_denorm_err
            if y_denorm_err > DENORM_ATOL:
                failures.append(f"{path} y denorm error {y_denorm_err} exceeds {DENORM_ATOL}")
            shapes[key][split_name] = {
                "x": list(loaded["x"].shape),
                "y": list(loaded["y"].shape),
                "x_raw": list(loaded["x_raw"].shape),
                "y_raw": list(loaded["y_raw"].shape),
            }
            if percent != reference_rate:
                ref = load_npz(output_root / rate_key(reference_rate) / f"{split_name}.npz")
                if not np.array_equal(ref["y_raw"], loaded["y_raw"]):
                    y_raw_identical = False
                    failures.append(f"{path} y_raw differs from {rate_key(reference_rate)}")
                if not np.array_equal(ref["y"], loaded["y"]):
                    y_norm_identical = False
                    failures.append(f"{path} y differs from {rate_key(reference_rate)}")
                for field_name in ("day_index", "window_start_slot", "target_slot"):
                    if not np.array_equal(ref[field_name], loaded[field_name]):
                        index_identical = False
                        failures.append(f"{path} {field_name} differs from {rate_key(reference_rate)}")
            if split_name == "train":
                train_x_moments[key] = {
                    "mean": float(loaded["x"].astype(np.float64).mean()),
                    "std": float(loaded["x"].astype(np.float64).std(ddof=DEFAULT_DDOF)),
                }
                if abs(train_x_moments[key]["mean"]) > ZSCORE_MEAN_ATOL:
                    failures.append(f"{key} train x mean {train_x_moments[key]['mean']} not ~0")
                if abs(train_x_moments[key]["std"] - 1.0) > ZSCORE_STD_ATOL:
                    failures.append(f"{key} train x std {train_x_moments[key]['std']} not ~1")
                if train_y_moments is None:
                    train_y_moments = {
                        "mean": float(loaded["y"].astype(np.float64).mean()),
                        "std": float(loaded["y"].astype(np.float64).std(ddof=DEFAULT_DDOF)),
                    }
                    if abs(train_y_moments["mean"]) > ZSCORE_MEAN_ATOL:
                        failures.append(f"train y mean {train_y_moments['mean']} not ~0")
                    if abs(train_y_moments["std"] - 1.0) > ZSCORE_STD_ATOL:
                        failures.append(f"train y std {train_y_moments['std']} not ~1")
            refit_x = fit_training_scaler(
                loaded["x_raw"] if split_name == "train" else prepared[percent]["train"].x_raw,
                source=f"recheck {key} x",
                epsilon=epsilon,
                ddof=DEFAULT_DDOF,
            )
            if split_name == "train":
                if abs(refit_x.mean - input_scalers[percent].mean) > 1e-12 or abs(refit_x.std - input_scalers[percent].std) > 1e-12:
                    failures.append(f"{key} input scaler refit mismatch")

    train_names = set(split.names("train"))
    val_names = set(split.names("validation"))
    test_names = set(split.names("test"))
    disjoint = train_names.isdisjoint(val_names) and train_names.isdisjoint(test_names) and val_names.isdisjoint(test_names)
    coverage = train_names | val_names | test_names == {day.name for day in days}
    if not disjoint:
        failures.append("date splits are not disjoint")
    if not coverage:
        failures.append("date splits do not cover every discovered day")

    if hashes_before != hashes_after:
        failures.append("protected input SHA256 changed during preparation")

    if adjacency.shape != (n_nodes, n_nodes):
        failures.append(f"adjacency shape {adjacency.shape}")

    for split_name in SPLITS:
        index_path = output_root / f"sample_index_{split_name}.csv"
        with index_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected_counts[split_name]:
            failures.append(f"{index_path} has {len(rows)} rows, expected {expected_counts[split_name]}")
        for row in rows:
            start = int(row["window_start_slot"])
            target = int(row["target_slot"])
            end = int(row["window_end_slot"])
            if end != target or target != start + n_his - 1:
                failures.append(f"{index_path} slot relation failed at sample {row['sample_index']}")
                break
            if int(row["window_start_slot"]) > n_slots - n_his:
                failures.append(f"{index_path} window crosses the day boundary")
                break

    summary = {
        "n_days": len(days),
        "days": [day.name for day in days],
        "date_order_key": "simulation day index parsed from directory name day_<NN>",
        "training_days": split.names("train"),
        "validation_days": split.names("validation"),
        "test_days": split.names("test"),
        "train_day_count": len(split.train),
        "val_day_count": len(split.validation),
        "test_day_count": len(split.test),
        "splits_disjoint": disjoint,
        "splits_cover_all_days": coverage,
        "input_files_complete": True,
        "time_slots_per_day": n_slots,
        "node_count": n_nodes,
        "nested_validation_passed": True,
        "not_greater_than_full_passed": True,
        "windows_per_day": windows_per_day,
        "theoretical_sample_counts": expected_counts,
        "actual_sample_counts": {
            split_name: prepared[reference_rate][split_name].x.shape[0] for split_name in SPLITS
        },
        "tensor_shapes": shapes,
        "input_scalers": {
            rate_key(percent): {"mean_x": input_scalers[percent].mean, "std_x": input_scalers[percent].std}
            for percent in rates
        },
        "target_scaler": {"mean_y_full": target_scaler.mean, "std_y_full": target_scaler.std},
        "fit_split": "training_only",
        "per_day_normalization": False,
        "per_node_normalization": False,
        "normalized_train_input_moments": train_x_moments,
        "normalized_train_target_moments": train_y_moments,
        "denormalization_max_abs_error": denorm_errors,
        "denormalization_atol": DENORM_ATOL,
        "cross_rate_y_raw_identical": y_raw_identical,
        "cross_rate_y_normalized_identical": y_norm_identical,
        "cross_rate_index_identical": index_identical,
        "npz_allow_pickle_false_reload_passed": not any("keys" in item for item in failures),
        "reproducibility_validation_passed": not any("rebuilt" in item or "refit" in item for item in failures),
        "inputs_unchanged": hashes_before == hashes_after,
        "no_future_slot_target": True,
        "no_cross_day_windows": True,
        "kept_all_valid_windows": all(
            prepared[reference_rate][name].x.shape[0] == expected_counts[name] for name in SPLITS
        ),
        "failures": failures,
        "overall_validation_passed": not failures,
        "status": "ok" if not failures else "failed",
    }
    if failures:
        raise PrepareError("prepared dataset validation failed:\n" + "\n".join(failures))
    return summary


def dependency_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    try:
        import pandas as pd  # type: ignore

        versions["pandas"] = pd.__version__
    except Exception:
        versions["pandas"] = "not-installed"
    return versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare R-only STGCN windowed tensors from daily full and observed "
            "5-minute R-edge entry counts. Target is the full flow at the last "
            "input slot, not the next future slot."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--full-flow-pattern", default=DEFAULT_FULL_PATTERN)
    parser.add_argument("--observed-flow-pattern", default=DEFAULT_OBSERVED_PATTERN)
    parser.add_argument("--r-nodes", type=Path, default=DEFAULT_R_NODES)
    parser.add_argument("--adjacency", type=Path, default=DEFAULT_ADJACENCY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rates", nargs="+", default=[str(rate) for rate in DEFAULT_RATES])
    parser.add_argument("--n-his", type=int, default=DEFAULT_N_HIS)
    parser.add_argument("--train-day-count", type=int, default=DEFAULT_TRAIN_DAYS)
    parser.add_argument("--val-day-count", type=int, default=DEFAULT_VAL_DAYS)
    parser.add_argument("--test-day-count", type=int, default=DEFAULT_TEST_DAYS)
    parser.add_argument("--train-days", nargs="+", default=None)
    parser.add_argument("--val-days", nargs="+", default=None)
    parser.add_argument("--test-days", nargs="+", default=None)
    parser.add_argument("--target-mode", default=TARGET_MODE_LAST_OBSERVED)
    parser.add_argument("--expected-days", type=int, default=DEFAULT_EXPECTED_DAYS)
    parser.add_argument("--expected-nodes", type=int, default=DEFAULT_EXPECTED_NODES)
    parser.add_argument("--n-slots", type=int, default=DEFAULT_EXPECTED_SLOTS)
    parser.add_argument("--window-seconds", type=int, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--ddof", type=int, default=DEFAULT_DDOF)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--status-file", type=Path, default=None)
    parser.add_argument("--skip-protected-hash", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.target_mode != TARGET_MODE_LAST_OBSERVED:
        raise PrepareError(
            f"unsupported --target-mode {args.target_mode}; "
            f"this dataset uses {TARGET_MODE_LAST_OBSERVED}"
        )
    if args.n_his <= 0:
        raise PrepareError("--n-his must be positive")
    if SECONDS_PER_DAY % args.window_seconds != 0:
        raise PrepareError("window-seconds must divide 86400")
    if args.n_slots != SECONDS_PER_DAY // args.window_seconds:
        raise PrepareError(
            f"--n-slots {args.n_slots} != 86400/{args.window_seconds} "
            f"({SECONDS_PER_DAY // args.window_seconds})"
        )
    rates = parse_rates(list(args.rates))
    data_root = args.data_root.expanduser().resolve()
    r_nodes_path = args.r_nodes.expanduser().resolve()
    adjacency_path = args.adjacency.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    status_path = (
        args.status_file.expanduser().resolve()
        if args.status_file is not None
        else output_root / "preparation_status.json"
    )
    log_path = None if args.dry_run else output_root / "preparation_log.jsonl"
    if not args.dry_run:
        if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
            raise PrepareError(f"{output_root} is not empty; pass --overwrite")
        output_root.mkdir(parents=True, exist_ok=True)
        if args.overwrite and (output_root / "preparation_log.jsonl").is_file():
            (output_root / "preparation_log.jsonl").unlink()
    logger = Logger(path=log_path, started=started)
    logger.emit("scanning", status="running", days_identified=0)

    if not args.skip_synthetic:
        run_synthetic_checks(args.n_his)

    mapping = load_node_mapping(r_nodes_path, args.expected_nodes)
    adjacency = load_adjacency(adjacency_path, len(mapping.edge_ids))
    day_dirs = discover_day_directories(data_root, args.full_flow_pattern)
    if len(day_dirs) != args.expected_days:
        names = ", ".join(path.name for path in day_dirs) or "none"
        raise PrepareError(f"expected {args.expected_days} day directories, found {len(day_dirs)}: {names}")
    days = resolve_input_files(
        data_root,
        day_dirs,
        rates,
        args.full_flow_pattern,
        args.observed_flow_pattern,
    )
    observed_csv_count = len(days) * len(rates)
    logger.emit(
        "validating_inputs",
        status="running",
        days_identified=len(days),
        observed_csv_count=observed_csv_count,
        days=[day.name for day in days],
    )
    write_status(
        None if args.dry_run else status_path,
        {
            "stage": "validating_inputs",
            "current_day": None,
            "current_rate": None,
            "days_done": 0,
            "days_total": len(days),
            "windows_generated": 0,
            "files_written": 0,
            "updated_at": now_utc(),
            "error": None,
        },
    )

    split = split_days(
        days,
        train_day_count=args.train_day_count,
        val_day_count=args.val_day_count,
        test_day_count=args.test_day_count,
        train_days=args.train_days,
        val_days=args.val_days,
        test_days=args.test_days,
    )
    logger.emit(
        "splitting_days",
        status="running",
        training_days=split.names("train"),
        validation_days=split.names("validation"),
        test_days=split.names("test"),
        date_order_key="simulation day index parsed from directory name day_<NN>",
    )
    print("TRAINING_DAYS " + " ".join(split.names("train")), flush=True)
    print("VALIDATION_DAYS " + " ".join(split.names("validation")), flush=True)
    print("TEST_DAYS " + " ".join(split.names("test")), flush=True)

    full_matrices: dict[str, np.ndarray] = {}
    observed_matrices: dict[int, dict[str, np.ndarray]] = {percent: {} for percent in rates}
    for day_i, day in enumerate(days):
        logger.emit("building_matrices", status="running", current_day=day.name, days_done=day_i)
        write_status(
            None if args.dry_run else status_path,
            {
                "stage": "building_matrices",
                "current_day": day.name,
                "current_rate": None,
                "days_done": day_i,
                "days_total": len(days),
                "windows_generated": 0,
                "files_written": 0,
                "updated_at": now_utc(),
                "error": None,
            },
        )
        full_matrices[day.name] = build_daily_flow_matrix(
            day.full_flow_path,
            mapping,
            n_slots=args.n_slots,
            window_seconds=args.window_seconds,
        )
        for percent in rates:
            logger.emit(
                "building_matrices",
                status="running",
                current_day=day.name,
                current_rate=rate_key(percent),
            )
            observed_matrices[percent][day.name] = build_daily_flow_matrix(
                day.observed_paths[percent],
                mapping,
                n_slots=args.n_slots,
                window_seconds=args.window_seconds,
                expected_day=day.name,
                expected_percent=percent,
            )
        validate_nested_day(
            day,
            {percent: observed_matrices[percent][day.name] for percent in rates},
            full_matrices[day.name],
            rates,
            mapping,
        )

    windows_per_day = args.n_slots - args.n_his + 1
    planned = {
        "train": len(split.train) * windows_per_day,
        "validation": len(split.validation) * windows_per_day,
        "test": len(split.test) * windows_per_day,
    }
    print(
        json.dumps(
            {
                "project_root": posix(PROJECT_ROOT),
                "data_root": posix(data_root),
                "full_flow_pattern": args.full_flow_pattern,
                "observed_flow_pattern": args.observed_flow_pattern,
                "r_nodes": posix(r_nodes_path),
                "adjacency": posix(adjacency_path),
                "adjacency_shape": list(adjacency.shape),
                "adjacency_dtype": str(adjacency.dtype),
                "output_root": posix(output_root),
                "days": [day.name for day in days],
                "training_days": split.names("train"),
                "validation_days": split.names("validation"),
                "test_days": split.names("test"),
                "n_slots": args.n_slots,
                "n_nodes": len(mapping.edge_ids),
                "windows_per_day": windows_per_day,
                "planned_samples": planned,
                "observed_csv_count": observed_csv_count,
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        logger.emit("completed", status="ok", dry_run=True, days_identified=len(days))
        return

    protected = collect_protected_paths(days, r_nodes_path, adjacency_path)
    hashes_before: dict[str, str] = {}
    if not args.skip_protected_hash:
        hashes_before = hash_paths(protected, logger, "hashing_inputs")

    prepared: dict[int, dict[str, PreparedSplit]] = {}
    windows_generated = 0
    for percent in rates:
        prepared[percent] = {}
        for split_name in SPLITS:
            logger.emit(
                "generating_windows",
                status="running",
                current_rate=rate_key(percent),
                current_split=split_name,
            )
            write_status(
                status_path,
                {
                    "stage": "generating_windows",
                    "current_day": None,
                    "current_rate": rate_key(percent),
                    "current_split": split_name,
                    "days_done": len(days),
                    "days_total": len(days),
                    "windows_generated": windows_generated,
                    "files_written": 0,
                    "updated_at": now_utc(),
                    "error": None,
                },
            )
            bundle = stack_split_windows(
                getattr(split, split_name),
                observed_matrices[percent],
                full_matrices,
                args.n_his,
            )
            prepared[percent][split_name] = bundle
            windows_generated += int(bundle.x_raw.shape[0])

    logger.emit("fitting_scalers", status="running")
    input_scalers = {
        percent: fit_training_scaler(
            prepared[percent]["train"].x_raw,
            source=f"training observed x {rate_key(percent)}",
            epsilon=args.epsilon,
            ddof=args.ddof,
        )
        for percent in rates
    }
    target_scaler = fit_training_scaler(
        prepared[rates[0]]["train"].y_raw,
        source="training full-flow targets only",
        epsilon=args.epsilon,
        ddof=args.ddof,
    )
    for percent in rates[1:]:
        other = fit_training_scaler(
            prepared[percent]["train"].y_raw,
            source=f"check y {rate_key(percent)}",
            epsilon=args.epsilon,
            ddof=args.ddof,
        )
        if abs(other.mean - target_scaler.mean) > 1e-12 or abs(other.std - target_scaler.std) > 1e-12:
            raise PrepareError("target scaler differs across penetrations; y_raw should be identical")

    logger.emit("normalizing", status="running")
    for percent in rates:
        for split_name in SPLITS:
            bundle = prepared[percent][split_name]
            bundle.x = input_scalers[percent].transform(bundle.x_raw)
            bundle.y = target_scaler.transform(bundle.y_raw)
            array_finite(f"{rate_key(percent)} {split_name} x", bundle.x)
            array_finite(f"{rate_key(percent)} {split_name} y", bundle.y)

    files_written = 0
    output_files: dict[str, dict[str, str]] = {}
    logger.emit("writing_outputs", status="running")
    node_mapping_path = output_root / "node_mapping.csv"
    atomic_write_csv(node_mapping_path, mapping.fieldnames, mapping.rows)
    files_written += 1

    sample_fields = [
        "sample_index",
        "split",
        "day_index",
        "day",
        "window_start_slot",
        "window_end_slot",
        "target_slot",
        "window_start_time",
        "window_end_time",
        "target_time",
        "n_his",
    ]
    for split_name in SPLITS:
        rows = sample_index_rows(
            split_name,
            prepared[rates[0]][split_name],
            days,
            args.window_seconds,
            args.n_his,
        )
        index_path = output_root / f"sample_index_{split_name}.csv"
        atomic_write_csv(index_path, sample_fields, rows)
        files_written += 1

    for percent in rates:
        key = rate_key(percent)
        output_files[key] = {}
        for split_name in SPLITS:
            bundle = prepared[percent][split_name]
            path = output_root / key / f"{split_name}.npz"
            atomic_savez(
                path,
                {
                    "x": bundle.x,
                    "y": bundle.y,
                    "x_raw": bundle.x_raw,
                    "y_raw": bundle.y_raw,
                    "day_index": bundle.day_index,
                    "window_start_slot": bundle.window_start_slot,
                    "target_slot": bundle.target_slot,
                },
            )
            output_files[key][split_name] = posix(path)
            files_written += 1
            logger.emit(
                "writing_outputs",
                status="running",
                current_rate=key,
                current_split=split_name,
                files_written=files_written,
            )

    hashes_after = hashes_before
    if not args.skip_protected_hash:
        hashes_after = hash_paths(protected, logger, "hashing_inputs_after")
    logger.emit("validating_outputs", status="running")
    validation = validate_prepared_dataset(
        output_root=output_root,
        days=days,
        split=split,
        rates=rates,
        mapping=mapping,
        adjacency=adjacency,
        full_matrices=full_matrices,
        observed_matrices=observed_matrices,
        prepared=prepared,
        input_scalers=input_scalers,
        target_scaler=target_scaler,
        n_his=args.n_his,
        n_slots=args.n_slots,
        hashes_before=hashes_before,
        hashes_after=hashes_after,
        epsilon=args.epsilon,
    )

    split_manifest = {
        "all_days": [day.name for day in days],
        "date_order_key": "simulation day index parsed from directory name day_<NN>",
        "training_days": split.names("train"),
        "validation_days": split.names("validation"),
        "test_days": split.names("test"),
        "train_day_count": len(split.train),
        "val_day_count": len(split.validation),
        "test_day_count": len(split.test),
        "splits_disjoint": True,
        "splits_cover_all_days": True,
        "time_slots_per_day": args.n_slots,
        "windows_per_day": windows_per_day,
        "theoretical_sample_counts": planned,
        "actual_sample_counts": validation["actual_sample_counts"],
        "n_his": args.n_his,
        "target_mode": TARGET_MODE_LAST_OBSERVED,
        "label_definition": (
            "Y is the full R-edge entry flow at the last slot of the 12-step "
            "observed window (slot s+n_his-1), not the next future slot"
        ),
        "cross_day_windows_allowed": False,
        "rates": [rate_key(percent) for percent in rates],
        "penetration_percents": rates,
    }
    normalization = {
        "normalization_method": "global_zscore",
        "formula": "z = (x - mean) / std",
        "inverse_formula": "x = z * std + mean",
        "ddof": args.ddof,
        "epsilon": args.epsilon,
        "fit_split": "training_only",
        "global_or_per_node": "global_scalar",
        "per_day_normalization": False,
        "per_node_normalization": False,
        "validation_and_test_reuse_training_scaler": True,
        "used_validation_to_fit": False,
        "used_test_to_fit": False,
        "numeric_dtype": "float32",
        "scaler_accumulation_dtype": "float64",
        "training_days": split.names("train"),
        "target_scaler": {
            "mean_y_full": target_scaler.mean,
            "std_y_full": target_scaler.std,
            "n_elements": target_scaler.n_elements,
            "source": target_scaler.source,
        },
        "input_scalers": {
            rate_key(percent): {
                "mean_x": input_scalers[percent].mean,
                "std_x": input_scalers[percent].std,
                "n_elements": input_scalers[percent].n_elements,
                "source": input_scalers[percent].source,
            }
            for percent in rates
        },
        "validation": {
            "normalized_train_input_moments": validation["normalized_train_input_moments"],
            "normalized_train_target_moments": validation["normalized_train_target_moments"],
            "denormalization_max_abs_error": validation["denormalization_max_abs_error"],
        },
    }
    input_hashes = {
        "r_nodes.csv": {"path": posix(r_nodes_path), "sha256": hashes_after[posix(r_nodes_path)]},
        "r_adjacency.npy": {"path": posix(adjacency_path), "sha256": hashes_after[posix(adjacency_path)]},
        "full_flow": {
            day.name: {"path": posix(day.full_flow_path), "sha256": hashes_after[posix(day.full_flow_path)]}
            for day in days
        },
        "observed_flow": {
            day.name: {
                rate_key(percent): {
                    "path": posix(day.observed_paths[percent]),
                    "sha256": hashes_after[posix(day.observed_paths[percent])],
                }
                for percent in rates
            }
            for day in days
        },
    }
    metadata = {
        "dataset_name": "r_only_stgcn_reconstruction",
        "task_type": (
            "Use 12 consecutive 5-minute observed R-edge entry counts to "
            "reconstruct the full R-edge entry counts at the 12th slot"
        ),
        "model_target": "R-only STGCN migration / reproduction data",
        "input_variable": "observed R-edge vehicle entry counts",
        "target_variable": "full R-edge vehicle entry counts",
        "does_not_scale_by_penetration": True,
        "time_granularity_minutes": args.window_seconds // 60,
        "n_his": args.n_his,
        "n_pred": 1,
        "node_count": len(mapping.edge_ids),
        "feature_count": 1,
        "rates": [rate_key(percent) for percent in rates],
        "n_days": len(days),
        "training_days": split.names("train"),
        "validation_days": split.names("validation"),
        "test_days": split.names("test"),
        "time_slots_per_day": args.n_slots,
        "windows_per_day": windows_per_day,
        "sample_counts": validation["actual_sample_counts"],
        "input_tensor_format": "[N, 12, 56, 1]",
        "target_tensor_format": "[N, 1, 56, 1]",
        "label_time_relation": "target_slot == window_start_slot + n_his - 1 == last input slot",
        "cross_day_windows": False,
        "normalization_method": "global_zscore",
        "input_and_target_use_different_scalers": True,
        "per_rate_input_scalers": True,
        "shared_target_scaler": True,
        "scaler_fit_on_training_only": True,
        "r_nodes_path": posix(r_nodes_path),
        "r_nodes_sha256": hashes_after[posix(r_nodes_path)],
        "adjacency_path": posix(adjacency_path),
        "adjacency_sha256": hashes_after[posix(adjacency_path)],
        "input_file_sha256": input_hashes,
        "output_file_sha256": {},
        "protected_input_sha256": {},
        "dependency_versions": dependency_versions(),
        "script_path": posix(Path(__file__)),
        "script_version": SCRIPT_VERSION,
        "generated_at": now_utc(),
        "target_mode": TARGET_MODE_LAST_OBSERVED,
    }
    atomic_write_json(output_root / "split_manifest.json", split_manifest)
    atomic_write_json(output_root / "normalization.json", normalization)
    atomic_write_json(output_root / "validation_summary.json", validation)
    metadata["output_file_sha256"] = {
        relative_posix(path, output_root): sha256_file(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file()
        and path.name
        not in {"preparation_log.jsonl", "preparation_status.json", "dataset_metadata.json"}
    }
    metadata["protected_input_sha256"] = hashes_after
    atomic_write_json(output_root / "dataset_metadata.json", metadata)
    files_written += 4
    elapsed = time.time() - started
    logger.emit(
        "completed",
        status="ok",
        files_written=files_written,
        windows_generated=windows_generated,
        elapsed_s=round(elapsed, 3),
        overall_validation_passed=True,
    )
    write_status(
        status_path,
        {
            "stage": "completed",
            "current_day": None,
            "current_rate": None,
            "days_done": len(days),
            "days_total": len(days),
            "windows_generated": windows_generated,
            "files_written": files_written,
            "updated_at": now_utc(),
            "error": None,
            "elapsed_s": round(elapsed, 3),
        },
    )
    print(
        f"done output={posix(output_root)} elapsed_s={elapsed:.3f} "
        f"train={planned['train']} val={planned['validation']} test={planned['test']}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except PrepareError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
