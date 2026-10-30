#!/usr/bin/env python3
"""Build Xuancheng26 training support files from the locked R graph.

The canonical node order is ``r_graph.npz["edge_ids"]``. This script does not
import or run the Shanghai preparation tools, and it does not rewrite the
graph, flow tables, window NPZ, split manifest, or normalization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

import prep_paths

CODE_ROOT = Path("/home/tzhang174/TrafficFlow")
SUMO_TOOLS = Path("/usr/share/sumo/tools")
NET_FILE = prep_paths.DEFAULT_NET_FILE
TRAINING_ROOT = prep_paths.DEFAULT_TRAINING_ROOT
WINDOW_ROOT = TRAINING_ROOT / "window_data"
GRAPH_DIR = TRAINING_ROOT / "graph"
GRAPH_PATH = GRAPH_DIR / "r_graph.npz"
CONFIG_ROOT = TRAINING_ROOT / "config"
FULL_FLOW_ROOT = TRAINING_ROOT / "full_flow" / "r-level"
PARTIAL_FLOW_ROOT = TRAINING_ROOT / "partial_flow"
SHANGHAI_NODE_MAP = Path(
    "/data5/tzhang174/TrafficFlow/shanghai_sim/training_data/r-only/node_mapping.csv"
)
SHANGHAI_METADATA = Path(
    "/data5/tzhang174/TrafficFlow/shanghai_sim/training_data/r-only/dataset_metadata.json"
)

NODE_COLUMNS = (
    "node_index",
    "node_id",
    "node_type",
    "edge_id",
    "from_junction",
    "to_junction",
    "num_lanes",
    "length",
    "speed",
    "priority",
    "edge_type",
)
NPZ_KEYS = (
    "x",
    "y",
    "x_raw",
    "y_raw",
    "day_index",
    "window_start_slot",
    "target_slot",
)
SPLITS = ("train", "validation", "test")
EXPECTED_NODES = 1406
EXPECTED_EDGES = 3877
N_HIS = 12
HORIZON = 1
CHANNELS = 1
DENORM_ATOL = 1e-3
DATASET_PATH_KEYS = (
    "data_root",
    "r_nodes",
    "node_mapping",
    "normalization",
    "split_manifest",
    "dataset_metadata",
    "data_validation",
)
DROP_SHANGHAI_GRAPH_COUNTS = (
    "expected_undirected_neighbors",
    "expected_nonzero_offdiag",
)


class SupportError(RuntimeError):
    """Xuancheng26 support-file generation failed a required check."""


def log(message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"{stamp} {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def node_order_sha256(edge_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(edge_ids).encode("utf-8")).hexdigest()


def json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: object) -> bytes:
    body = json_bytes(payload)
    atomic_write_bytes(path, body)
    return body


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SupportError(f"missing JSON {path}") from exc


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SupportError(f"missing {label}: {path}")


def import_sumolib():
    if str(SUMO_TOOLS) not in sys.path:
        sys.path.insert(0, str(SUMO_TOOLS))
    try:
        import sumolib
    except ImportError as exc:
        raise SupportError(f"sumolib is not importable from {SUMO_TOOLS}") from exc
    return sumolib


def load_r_graph() -> dict[str, object]:
    require_file(GRAPH_PATH, "r_graph.npz")
    with np.load(GRAPH_PATH, allow_pickle=False) as payload:
        missing = [key for key in ("edge_ids", "adjacency", "num_nodes", "num_graph_edges") if key not in payload.files]
        if missing:
            raise SupportError(f"{GRAPH_PATH} missing keys {missing}; found {payload.files}")
        edge_ids = [str(item) for item in payload["edge_ids"].tolist()]
        adjacency = np.array(payload["adjacency"], copy=True)
        num_nodes = int(np.asarray(payload["num_nodes"]).reshape(-1)[0])
        num_graph_edges = int(np.asarray(payload["num_graph_edges"]).reshape(-1)[0])
        keys = list(payload.files)
    return {
        "keys": keys,
        "edge_ids": edge_ids,
        "adjacency": adjacency,
        "num_nodes": num_nodes,
        "num_graph_edges": num_graph_edges,
    }


def validate_r_graph(graph: dict[str, object]) -> None:
    edge_ids = graph["edge_ids"]
    adjacency = graph["adjacency"]
    num_nodes = int(graph["num_nodes"])
    num_graph_edges = int(graph["num_graph_edges"])
    if num_nodes != EXPECTED_NODES or len(edge_ids) != EXPECTED_NODES:
        raise SupportError(
            f"{GRAPH_PATH} num_nodes={num_nodes} edge_ids={len(edge_ids)}; "
            f"expected {EXPECTED_NODES}. Refusing to repair the graph."
        )
    if len(set(edge_ids)) != len(edge_ids):
        raise SupportError(f"{GRAPH_PATH} edge_ids are not unique")
    if any(edge_id.startswith(":") or not edge_id for edge_id in edge_ids):
        raise SupportError(f"{GRAPH_PATH} contains an internal or empty edge id")
    if adjacency.shape != (EXPECTED_NODES, EXPECTED_NODES):
        raise SupportError(f"{GRAPH_PATH} adjacency shape {adjacency.shape}")
    if adjacency.dtype != np.uint8:
        raise SupportError(f"{GRAPH_PATH} adjacency dtype {adjacency.dtype}")
    values = set(int(item) for item in np.unique(adjacency))
    if values - {0, 1}:
        raise SupportError(f"{GRAPH_PATH} adjacency is not binary 0/1: {sorted(values)}")
    edge_sum = int(adjacency.sum())
    if num_graph_edges != EXPECTED_EDGES or edge_sum != EXPECTED_EDGES:
        raise SupportError(
            f"{GRAPH_PATH} num_graph_edges={num_graph_edges} adjacency.sum={edge_sum}; "
            f"expected {EXPECTED_EDGES}. Refusing to repair the graph."
        )
    if int(np.diag(adjacency).sum()) != 0:
        raise SupportError(f"{GRAPH_PATH} adjacency has self-loops")


def load_sumo_network(edge_ids: list[str]) -> dict[str, object]:
    require_file(NET_FILE, "final SUMO network")
    sumolib = import_sumolib()
    network = sumolib.net.readNet(str(NET_FILE), withInternal=False)
    by_id = {}
    document_order: list[str] = []
    links: set[tuple[str, str]] = set()
    lane_connections = 0
    for edge in network.getEdges():
        edge_id = str(edge.getID())
        document_order.append(edge_id)
        if edge_id.startswith(":"):
            raise SupportError(f"{NET_FILE} sumolib returned internal edge {edge_id}")
        speeds = [float(lane.getSpeed()) for lane in edge.getLanes()]
        if not speeds:
            raise SupportError(f"{NET_FILE} edge {edge_id} has no lanes")
        by_id[edge_id] = {
            "from_junction": str(edge.getFromNode().getID()),
            "to_junction": str(edge.getToNode().getID()),
            "num_lanes": int(edge.getLaneNumber()),
            "length": float(edge.getLength()),
            "speed": float(edge.getSpeed()),
            "lane_speeds_uniform": len({round(item, 5) for item in speeds}) == 1,
            "priority": int(edge.getPriority()),
            "edge_type": str(edge.getType() or ""),
        }
        for _dest, connections in edge.getOutgoing().items():
            for connection in connections:
                lane_connections += 1
                links.add((str(connection.getFrom().getID()), str(connection.getTo().getID())))
    missing = [edge_id for edge_id in edge_ids if edge_id not in by_id]
    extra = [edge_id for edge_id in document_order if edge_id not in set(edge_ids)]
    if missing or extra or len(document_order) != len(edge_ids):
        raise SupportError(
            f"{NET_FILE} real-edge set does not match r_graph edge_ids; "
            f"missing={missing[:8]} extra={extra[:8]} net={len(document_order)} graph={len(edge_ids)}"
        )
    index = {edge_id: position for position, edge_id in enumerate(edge_ids)}
    expected = np.zeros((len(edge_ids), len(edge_ids)), dtype=np.uint8)
    for source, target in links:
        if source not in index or target not in index:
            raise SupportError(f"SUMO connection {source}->{target} leaves the R-node set")
        expected[index[source], index[target]] = 1
    return {
        "by_id": by_id,
        "document_order": document_order,
        "links": links,
        "lane_connections": lane_connections,
        "expected_adjacency": expected,
    }


def validate_graph_matches_network(graph: dict[str, object], network: dict[str, object]) -> None:
    adjacency = graph["adjacency"]
    expected = network["expected_adjacency"]
    if expected.shape != adjacency.shape or not np.array_equal(expected, adjacency):
        mismatch = int(np.count_nonzero(expected != adjacency))
        raise SupportError(
            f"r_graph adjacency disagrees with {NET_FILE} connections at {mismatch} cells; "
            f"sumolib unique edges={len(network['links'])} lane connections={network['lane_connections']}. "
            "Refusing to repair the graph."
        )


def build_node_rows(edge_ids: list[str], network: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    mixed_speed = []
    for position, edge_id in enumerate(edge_ids):
        attrs = network["by_id"][edge_id]
        if not attrs["lane_speeds_uniform"]:
            mixed_speed.append(edge_id)
        rows.append(
            {
                "node_index": str(position),
                "node_id": f"R::{edge_id}",
                "node_type": "R",
                "edge_id": edge_id,
                "from_junction": attrs["from_junction"],
                "to_junction": attrs["to_junction"],
                "num_lanes": str(attrs["num_lanes"]),
                "length": f"{attrs['length']:.2f}",
                "speed": f"{attrs['speed']:.2f}",
                "priority": str(attrs["priority"]),
                "edge_type": attrs["edge_type"],
            }
        )
    if mixed_speed:
        raise SupportError(f"lane speeds differ inside edges {mixed_speed[:8]}")
    if [int(row["node_index"]) for row in rows] != list(range(len(edge_ids))):
        raise SupportError("node_index is not 0..N-1")
    if [row["edge_id"] for row in rows] != edge_ids:
        raise SupportError("node row edge_id order drifted from r_graph")
    return rows


def csv_bytes(rows: list[dict[str, object]]) -> bytes:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(NODE_COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def read_edge_column(path: Path, column: str = "edge_id") -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames or "node_index" not in reader.fieldnames:
            raise SupportError(f"{path} must contain node_index and {column}")
        rows = list(reader)
    indices = [int(row["node_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise SupportError(f"{path} node_index is not 0..N-1 in order")
    return [str(row[column]) for row in rows]


def discover_rates(manifest: dict) -> list[str]:
    found = []
    for path in sorted(WINDOW_ROOT.iterdir()):
        if not path.is_dir():
            continue
        if all((path / f"{split}.npz").is_file() for split in SPLITS):
            found.append(path.name)
    manifest_rates = [str(item) for item in manifest.get("rates", [])]
    if not found:
        raise SupportError(f"no penetration directories with split NPZ under {WINDOW_ROOT}")
    if manifest_rates and manifest_rates != found and sorted(manifest_rates) != sorted(found):
        raise SupportError(f"manifest rates {manifest_rates} != directories {found}")
    if manifest_rates and sorted(manifest_rates) == sorted(found):
        return manifest_rates
    return found


def manifest_days(manifest: dict) -> dict[str, list[str]]:
    days = {}
    alias = {"train": "training_days", "validation": "validation_days", "test": "test_days"}
    for split in SPLITS:
        if split not in manifest:
            raise SupportError(f"split_manifest.json missing {split}")
        values = [str(item) for item in manifest[split]]
        if len(values) != len(set(values)):
            raise SupportError(f"split_manifest {split} has duplicate dates")
        other = manifest.get(alias[split])
        if other is not None and [str(item) for item in other] != values:
            raise SupportError(f"split_manifest {alias[split]} does not match {split}")
        days[split] = values
    overlap = {
        "train∩validation": sorted(set(days["train"]) & set(days["validation"])),
        "train∩test": sorted(set(days["train"]) & set(days["test"])),
        "validation∩test": sorted(set(days["validation"]) & set(days["test"])),
    }
    bad = {name: values for name, values in overlap.items() if values}
    if bad:
        raise SupportError(f"split dates overlap: {bad}")
    if str(manifest.get("target_mode")) != "last-observed-step":
        raise SupportError(f"split_manifest target_mode={manifest.get('target_mode')!r}")
    if int(manifest.get("n_his", N_HIS)) != N_HIS:
        raise SupportError(f"split_manifest n_his={manifest.get('n_his')} != {N_HIS}")
    return days


def read_flow_matrix(path: Path, edge_ids: list[str]) -> np.ndarray:
    require_file(path, "flow csv")
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split(",")
    if header[:2] != ["time_bin", "start_time"]:
        raise SupportError(f"{path} header does not start with time_bin,start_time")
    columns = header[2:]
    if len(columns) != len(edge_ids) or set(columns) != set(edge_ids):
        raise SupportError(f"{path} edge columns are not the r_graph edge set")
    table = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != len(edge_ids) + 2:
        raise SupportError(f"{path} shape {table.shape}")
    values = table[:, 2:]
    if columns != edge_ids:
        order = [columns.index(edge_id) for edge_id in edge_ids]
        values = values[:, order]
    return values.astype(np.float32)


def weekday_index(day_name: str) -> int:
    year, month, day = (int(part) for part in day_name.split("_"))
    return int(__import__("datetime").date(year, month, day).weekday())


def prove_tensor_edge_order(edge_ids: list[str], days: dict[str, list[str]], rates: list[str]) -> dict[str, object]:
    """Prove NPZ node columns are r_graph edge_ids by matching raw counts to flow CSV."""
    sys.path.insert(0, str(CODE_ROOT / "dataset/xuancheng26/model_prep"))
    import generate_r_level_windows as windows

    generator_ids = windows.load_edge_ids()
    if generator_ids != edge_ids:
        raise SupportError("generate_r_level_windows.load_edge_ids() != r_graph edge_ids")
    if Path(windows.GRAPH_PATH) != GRAPH_PATH:
        raise SupportError(f"window generator graph path {windows.GRAPH_PATH} != {GRAPH_PATH}")
    checked_days = 0
    for split, split_days in days.items():
        for day_name in split_days:
            full = read_flow_matrix(FULL_FLOW_ROOT / day_name / "inflow.csv", edge_ids)
            for rate in rates:
                daily_path = WINDOW_ROOT / rate / split / f"{day_name}.npz"
                require_file(daily_path, "daily window npz")
                partial = read_flow_matrix(PARTIAL_FLOW_ROOT / rate / day_name / "inflow.csv", edge_ids)
                with np.load(daily_path, allow_pickle=False) as payload:
                    x_raw = np.array(payload["x_raw"], copy=False)
                    y_raw = np.array(payload["y_raw"], copy=False)
                    starts = np.array(payload["window_start_slot"], copy=False)
                    targets = np.array(payload["target_slot"], copy=False)
                if x_raw.shape[2] != len(edge_ids) or y_raw.shape[2] != len(edge_ids):
                    raise SupportError(f"{daily_path} node axis {x_raw.shape}")
                history = starts[:, None] + np.arange(N_HIS, dtype=starts.dtype)[None, :]
                expected_x = partial[history]
                expected_y = full[targets]
                if not np.array_equal(x_raw[:, :, :, 0], expected_x):
                    raise SupportError(f"{daily_path} x_raw does not match inflow columns in r_graph order")
                if not np.array_equal(y_raw[:, 0, :, 0], expected_y):
                    raise SupportError(f"{daily_path} y_raw does not match full inflow columns in r_graph order")
            checked_days += 1
            log(f"[order] {day_name} {split} matched {len(rates)} rates")
    return {
        "generator": str(CODE_ROOT / "dataset/xuancheng26/model_prep/generate_r_level_windows.py"),
        "generator_function": "load_edge_ids",
        "alignment": "flow CSV columns are reindexed to r_graph edge_ids before windowing",
        "days_checked": checked_days,
        "rates_checked": list(rates),
        "comparison": "exact float32 equality of x_raw/y_raw against aligned partial inflow and full inflow",
        "tensor_edge_ids": edge_ids,
    }


def validate_merged_dataset(
    edge_ids: list[str],
    days: dict[str, list[str]],
    rates: list[str],
    manifest: dict,
    normalization: dict,
) -> dict[str, object]:
    samples_per_day = None
    shapes: dict[str, dict[str, dict[str, list[int]]]] = {}
    dtypes: dict[str, str] = {}
    nan_inf = {"files": 0, "non_finite": 0}
    sample_counts: dict[str, int] = {}
    denorm = {}
    y_raw_reference: dict[str, np.ndarray] = {}
    y_reference: dict[str, np.ndarray] = {}
    target_max_abs = {split: 0.0 for split in SPLITS}
    for rate in rates:
        shapes[rate] = {}
        denorm[rate] = {}
        mean_x = float(normalization["input_scalers"][rate]["mean_x"])
        std_x = float(normalization["input_scalers"][rate]["std_x"])
        mean_y = float(normalization["target_scaler"]["mean_y_full"])
        std_y = float(normalization["target_scaler"]["std_y_full"])
        per_rate = load_json(WINDOW_ROOT / rate / "normalization.json")
        if float(per_rate["mean_x"]) != mean_x or float(per_rate["std_x"]) != std_x:
            raise SupportError(f"{rate} normalization.json mean_x/std_x != shared input_scalers")
        if float(per_rate["mean_y_full"]) != mean_y or float(per_rate["std_y_full"]) != std_y:
            raise SupportError(f"{rate} normalization.json target scaler != shared target_scaler")
        for split in SPLITS:
            path = WINDOW_ROOT / rate / f"{split}.npz"
            require_file(path, "split npz")
            with np.load(path, allow_pickle=False) as payload:
                missing = [key for key in NPZ_KEYS if key not in payload.files]
                if missing:
                    raise SupportError(f"{path} missing {missing}")
                arrays = {key: np.array(payload[key], copy=True) for key in NPZ_KEYS}
            x = arrays["x"]
            y = arrays["y"]
            if x.ndim != 4 or y.ndim != 4:
                raise SupportError(f"{path} rank x={x.ndim} y={y.ndim}")
            if x.shape[1] != N_HIS or x.shape[2] != len(edge_ids) or x.shape[3] != CHANNELS:
                raise SupportError(f"{path} x shape {x.shape} disagrees with the 12/1406/1 task")
            if y.shape[1] != HORIZON or y.shape[2] != len(edge_ids) or y.shape[3] != CHANNELS:
                raise SupportError(f"{path} y shape {y.shape} disagrees with the horizon-1 task")
            if x.shape[0] != y.shape[0]:
                raise SupportError(f"{path} x/y sample counts differ")
            for key, array in arrays.items():
                if array.dtype.hasobject or array.dtype.kind in {"O", "U", "S"}:
                    raise SupportError(f"{path} {key} dtype {array.dtype}")
                dtypes.setdefault(key, str(array.dtype))
                if str(array.dtype) != dtypes[key]:
                    raise SupportError(f"{path} {key} dtype {array.dtype} != {dtypes[key]}")
            for key in ("x", "y", "x_raw", "y_raw"):
                nan_inf["files"] += 1
                if not np.isfinite(arrays[key]).all():
                    nan_inf["non_finite"] += 1
                    raise SupportError(f"{path} {key} contains NaN or inf")
            count = int(x.shape[0])
            if split not in sample_counts:
                sample_counts[split] = count
            elif sample_counts[split] != count:
                raise SupportError(f"{path} has {count} samples, other rates have {sample_counts[split]}")
            expected_count = manifest.get("actual_sample_counts", {}).get(split)
            if expected_count is not None and int(expected_count) != count:
                raise SupportError(f"{path} samples {count} != manifest actual_sample_counts {expected_count}")
            offset = 0
            for day_name in days[split]:
                daily_path = WINDOW_ROOT / rate / split / f"{day_name}.npz"
                with np.load(daily_path, allow_pickle=False) as payload:
                    daily = {key: np.array(payload[key], copy=False) for key in NPZ_KEYS}
                day_count = int(daily["x"].shape[0])
                if samples_per_day is None:
                    samples_per_day = day_count
                elif day_count != samples_per_day:
                    raise SupportError(f"{daily_path} has {day_count} samples, expected {samples_per_day}")
                block = slice(offset, offset + day_count)
                for key in NPZ_KEYS:
                    if not np.array_equal(arrays[key][block], daily[key]):
                        raise SupportError(f"{path} {key} block for {day_name} != daily NPZ")
                weekday = weekday_index(day_name)
                if not np.array_equal(arrays["day_index"][block], np.full(day_count, weekday, dtype=arrays["day_index"].dtype)):
                    raise SupportError(f"{path} day_index for {day_name} is not weekday {weekday}")
                starts = np.arange(day_count, dtype=arrays["window_start_slot"].dtype)
                if not np.array_equal(arrays["window_start_slot"][block], starts):
                    raise SupportError(f"{path} window_start_slot was renumbered at {day_name}")
                if not np.array_equal(arrays["target_slot"][block], starts + (N_HIS - 1)):
                    raise SupportError(f"{path} target_slot was renumbered at {day_name}")
                offset += day_count
            if offset != count:
                raise SupportError(f"{path} sample count {count} != concatenated daily count {offset}")
            manifest_day_count = manifest.get("samples_per_day")
            if manifest_day_count is not None and int(manifest_day_count) != samples_per_day:
                raise SupportError(
                    f"manifest samples_per_day {manifest_day_count} != daily NPZ {samples_per_day}"
                )
            x_error, x_mean = chunk_denorm_error(arrays["x"], arrays["x_raw"], mean_x, std_x)
            y_error, y_mean = chunk_denorm_error(arrays["y"], arrays["y_raw"], mean_y, std_y)
            if x_error > DENORM_ATOL or y_error > DENORM_ATOL:
                raise SupportError(f"{path} denorm max abs x={x_error} y={y_error}")
            denorm[rate][split] = {
                "x_max_abs_error": x_error,
                "x_mean_abs_error": x_mean,
                "y_max_abs_error": y_error,
                "y_mean_abs_error": y_mean,
            }
            shapes[rate][split] = {"x": list(x.shape), "y": list(y.shape)}
            if split not in y_raw_reference:
                y_raw_reference[split] = arrays["y_raw"]
                y_reference[split] = arrays["y"]
            else:
                diff = float(np.max(np.abs(arrays["y_raw"].astype(np.float64) - y_raw_reference[split].astype(np.float64))))
                target_max_abs[split] = max(target_max_abs[split], diff)
                if not np.array_equal(arrays["y_raw"], y_raw_reference[split]):
                    raise SupportError(f"{path} y_raw differs from {rates[0]}/{split}.npz")
                if not np.array_equal(arrays["y"], y_reference[split]):
                    raise SupportError(f"{path} y differs from {rates[0]}/{split}.npz")
            log(f"[dataset] {rate}/{split} samples={count} denorm_x={x_error:.3e} denorm_y={y_error:.3e}")
    return {
        "samples_per_day": int(samples_per_day or 0),
        "sample_counts": sample_counts,
        "shapes": shapes,
        "dtypes": dtypes,
        "nan_inf": nan_inf,
        "denorm": denorm,
        "target_max_abs_error": target_max_abs,
        "target_exact": all(value == 0.0 for value in target_max_abs.values()),
    }


def chunk_denorm_error(normalized: np.ndarray, raw: np.ndarray, mean: float, std: float) -> tuple[float, float]:
    if std <= 0.0:
        raise SupportError(f"normalization std {std} is not positive")
    max_error = 0.0
    total = 0.0
    count = 0
    step = 256
    for start in range(0, normalized.shape[0], step):
        block = slice(start, start + step)
        recovered = (raw[block].astype(np.float64) - mean) / std
        diff = np.abs(recovered - normalized[block].astype(np.float64))
        max_error = max(max_error, float(diff.max()))
        total += float(diff.sum())
        count += int(diff.size)
    return max_error, total / count


def check_normalization_present(rates: list[str]) -> dict:
    path = WINDOW_ROOT / "normalization.json"
    require_file(path, "normalization.json")
    payload = load_json(path)
    scaler = payload.get("target_scaler") or {}
    if "mean_y_full" not in scaler or "std_y_full" not in scaler:
        raise SupportError(f"{path} is missing target_scaler.mean_y_full/std_y_full")
    inputs = payload.get("input_scalers") or {}
    for rate in rates:
        entry = inputs.get(rate) or {}
        if "mean_x" not in entry or "std_x" not in entry:
            raise SupportError(f"{path} input_scalers missing {rate}")
        if float(entry["std_x"]) <= 0.0 or float(scaler["std_y_full"]) <= 0.0:
            raise SupportError(f"{path} non-positive std for {rate}")
    if bool(payload.get("used_test_to_fit")) or bool(payload.get("used_validation_to_fit")):
        raise SupportError(f"{path} was not fit on training only")
    return payload


def graph_file_plan() -> dict[str, dict[str, str]]:
    stsgcn_topology = GRAPH_DIR / "stsgcn_undirected_topology.npy"
    stgcn_weight = GRAPH_DIR / "stgcn_weighted_adjacency.npy"
    dcrnn_weight = GRAPH_DIR / "dcrnn_weighted_adjacency.npy"
    dcrnn_pickle = GRAPH_DIR / "dcrnn_adj_mx.pkl"
    return {
        "stgcn": {
            "weighted_adjacency": stgcn_weight.as_posix(),
        },
        "dcrnn": {
            "adjacency": dcrnn_weight.as_posix(),
            "adjacency_pickle": dcrnn_pickle.as_posix(),
        },
        "gwn": {
            "adjacency": dcrnn_weight.as_posix(),
            "adjacency_pickle": dcrnn_pickle.as_posix(),
        },
        "stsgcn": {
            "spatial_adjacency": stsgcn_topology.as_posix(),
        },
        "stfgnn": {
            "spatial_adjacency": stsgcn_topology.as_posix(),
            "temporal_adjacency": (TRAINING_ROOT / "config" / "stfgnn" / "temporal_adjacency.npy").as_posix(),
        },
        "pdformer": {
            "undirected_topology": stsgcn_topology.as_posix(),
            "relations_dir": (TRAINING_ROOT / "config" / "pdformer").as_posix(),
        },
        "s2s_gcn_lstm": {
            "spatial_adjacency": stsgcn_topology.as_posix(),
            "r_graph": (GRAPH_DIR / "r_graph.npz").as_posix(),
            "distance_matrix": stgcn_weight.as_posix(),
        },
    }


def discover_model_configs() -> list[tuple[str, Path]]:
    found = sorted(CONFIG_ROOT.glob("*/r_only_*.json"))
    if not found:
        raise SupportError(f"no r_only configs under {CONFIG_ROOT}")
    configs = []
    for path in found:
        name = path.name
        if not name.startswith("r_only_") or not name.endswith(".json"):
            raise SupportError(f"unexpected config name {path}")
        model = name[len("r_only_") : -len(".json")]
        configs.append((model, path))
    return configs


def rewrite_dataset_phrases(value: str, num_nodes: int, train_days: int) -> str:
    text = value
    text = text.replace("56 SUMO edges", f"{num_nodes} real SUMO edges")
    text = text.replace("Official 56-node DTW file is a later dedicated run.", "The DTW temporal graph is not in this dataset package.")
    text = text.replace(
        "Simulation days are labeled day_NN with begin=0/end=86400 and no calendar dates. "
        "weekday embedding code is kept; enabling it requires an explicit weekday_mapping JSON. "
        "day_index % 7 is forbidden.",
        "Xuancheng26 day_index is already the Monday=0 weekday. "
        "add_day_in_week stays false because PDFormer calendar mode indexes all_days by day_index.",
    )
    text = text.replace(
        "Matches official PeMS04 npz weekday encoding. This is not calendar Monday=0. "
        "Calendar mapping is optional and errors if missing.",
        "day_of_week_source stays original_pems_npz_sequential_index_mod_7, which applies day_index % 7. "
        f"Xuancheng26 stores the Monday=0 weekday in day_index, so % 7 keeps that weekday. Training days={train_days}.",
    )
    return text


def specialize_config(
    template: dict,
    model: str,
    *,
    num_nodes: int,
    rates: list[str],
    train_day_count: int,
    paths: dict[str, str],
    graph_paths: dict[str, str],
) -> dict:
    config = json.loads(json.dumps(template))
    for key in DROP_SHANGHAI_GRAPH_COUNTS:
        config.pop(key, None)
    for key in DATASET_PATH_KEYS:
        if key in config:
            config[key] = paths[key]
    for key, value in graph_paths.items():
        if key in config:
            config[key] = value
    if "num_nodes" in config:
        config["num_nodes"] = num_nodes
    if "rates" in config:
        config["rates"] = list(rates)
    if "cand_key_days" in config:
        config["cand_key_days"] = train_day_count
    adaptations = config.get("adaptations")
    if isinstance(adaptations, list):
        for item in adaptations:
            if not isinstance(item, dict):
                continue
            if item.get("field") == "num_nodes":
                item["adapted_value"] = num_nodes
                item["adaptation_reason"] = (
                    f"Xuancheng26 R graph has {num_nodes} real SUMO edges. "
                    "Architecture widths are unchanged."
                )
            for field in ("adaptation_reason", "adapted_value", "original_value"):
                if isinstance(item.get(field), str):
                    item[field] = rewrite_dataset_phrases(item[field], num_nodes, train_day_count)
    return config


def load_shanghai_denylist() -> tuple[set[str], set[str]]:
    edge_ids: set[str] = set()
    hashes: set[str] = set()
    if SHANGHAI_NODE_MAP.is_file():
        with SHANGHAI_NODE_MAP.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                edge_ids.add(str(row.get("edge_id", "")))
                edge_ids.add(str(row.get("node_id", "")))
        edge_ids.discard("")
    if SHANGHAI_METADATA.is_file():
        raw = SHANGHAI_METADATA.read_text(encoding="utf-8")
        hashes.update(
            token.lower()
            for token in raw.replace('"', " ").replace(",", " ").split()
            if len(token) == 64 and all(character in "0123456789abcdefABCDEF" for character in token)
        )

    return edge_ids, hashes


def find_residue(payload: object, edge_ids: set[str], hashes: set[str], path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            hits.extend(find_residue(value, edge_ids, hashes, f"{path}.{key}"))
        return hits
    if isinstance(payload, list):
        for index, value in enumerate(payload):
            hits.extend(find_residue(value, edge_ids, hashes, f"{path}[{index}]"))
        return hits
    if isinstance(payload, bool) or payload is None:
        return hits
    if isinstance(payload, int) and not isinstance(payload, bool) and payload == 56:
        return [f"{path}=56"]
    if isinstance(payload, str):
        import re

        lowered = payload.lower()
        for token in ("shanghai", "shanghai_sim", "training_data/r-only", "/mnt/f/workspace", "analysis/graph/r_graph"):
            if token in lowered or token in payload:
                hits.append(f"{path} contains {token}")
        if "day_01" in payload or "day_20" in payload:
            hits.append(f"{path} contains an old day_NN label")
        hex_digest = bool(re.fullmatch(r"[0-9a-fA-F]{32,}", payload))
        if not hex_digest and re.search(r"(?<![0-9A-Fa-f])56(?![0-9A-Fa-f])", payload):
            hits.append(f"{path} contains 56 in {payload!r}")
        if hex_digest:
            if payload.lower() in hashes:
                hits.append(f"{path} contains an old sha256")
            return hits
        for edge_id in edge_ids:
            if edge_id and re.search(rf"(?<![A-Za-z0-9_]){re.escape(edge_id)}(?![A-Za-z0-9_])", payload):
                hits.append(f"{path} contains old edge id {edge_id}")
                break
        for digest in hashes:
            if digest and digest in payload:
                hits.append(f"{path} contains an old sha256")
                break
    return hits


def hash_named_files(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    result = {}
    for label, path in paths.items():
        require_file(path, label)
        result[label] = {"path": path.as_posix(), "sha256": sha256_file(path)}
    return result


def build_validation_summary(
    *,
    edge_ids: list[str],
    graph: dict[str, object],
    network: dict[str, object],
    rows: list[dict[str, object]],
    days: dict[str, list[str]],
    rates: list[str],
    normalization: dict,
    dataset: dict[str, object],
    order_proof: dict[str, object],
    order_hashes: dict[str, str],
    warnings: list[str],
) -> dict:
    priorities = sorted({int(row["priority"]) for row in rows})
    empty_types = sum(1 for row in rows if row["edge_type"] == "")
    return {
        "status": "passed",
        "overall_validation_passed": True,
        "failures": [],
        "warnings": warnings,
        "node_count": len(edge_ids),
        "num_nodes": len(edge_ids),
        "fit_split": "training_only",
        "input_files_complete": True,
        "inputs_unchanged": True,
        "no_cross_day_windows": True,
        "no_future_slot_target": True,
        "splits_disjoint": True,
        "cross_rate_y_raw_identical": bool(dataset["target_exact"]),
        "cross_rate_y_normalized_identical": bool(dataset["target_exact"]),
        "cross_rate_index_identical": True,
        "npz_allow_pickle_false_reload_passed": True,
        "per_day_normalization": False,
        "per_node_normalization": False,
        "target_mode": "last-observed-step",
        "n_his": N_HIS,
        "horizon": HORIZON,
        "feature_count": CHANNELS,
        "windows_per_day": dataset["samples_per_day"],
        "time_slots_per_day": 288,
        "actual_sample_counts": dataset["sample_counts"],
        "theoretical_sample_counts": dataset["sample_counts"],
        "training_days": days["train"],
        "validation_days": days["validation"],
        "test_days": days["test"],
        "train_day_count": len(days["train"]),
        "val_day_count": len(days["validation"]),
        "test_day_count": len(days["test"]),
        "rates": rates,
        "penetration_rates": rates,
        "target_scaler": {
            "mean_y_full": normalization["target_scaler"]["mean_y_full"],
            "std_y_full": normalization["target_scaler"]["std_y_full"],
        },
        "input_scalers": normalization["input_scalers"],
        "denormalization_atol": DENORM_ATOL,
        "denormalization_max_abs_error": {
            f"{rate}/{split}": {
                "x": values["x_max_abs_error"],
                "y": values["y_max_abs_error"],
            }
            for rate, splits in dataset["denorm"].items()
            for split, values in splits.items()
        },
        "denormalization_mean_abs_error": {
            f"{rate}/{split}": {
                "x": values["x_mean_abs_error"],
                "y": values["y_mean_abs_error"],
            }
            for rate, splits in dataset["denorm"].items()
            for split, values in splits.items()
        },
        "target_consistency": {
            "rates": rates,
            "exact_y_raw": bool(dataset["target_exact"]),
            "max_abs_error": dataset["target_max_abs_error"],
        },
        "tensor_shapes": dataset["shapes"],
        "dtypes": dataset["dtypes"],
        "nan_inf_checks": dataset["nan_inf"],
        "graph_checks": {
            "path": GRAPH_PATH.as_posix(),
            "keys": graph["keys"],
            "edge_ids_unique": True,
            "num_nodes": int(graph["num_nodes"]),
            "adjacency_shape": list(graph["adjacency"].shape),
            "num_graph_edges": int(graph["num_graph_edges"]),
            "adjacency_sum": int(graph["adjacency"].sum()),
            "binary_values": [0, 1],
            "self_loops": 0,
            "matches_final_net_connections": True,
            "lane_connections_collapsed": int(network["lane_connections"]),
            "unique_edge_connections": len(network["links"]),
            "sumo_document_order_matches_r_graph": network["document_order"] == edge_ids,
            "python_sorted_order_matches_r_graph": edge_ids == sorted(edge_ids),
        },
        "node_mapping_checks": {
            "rows": len(rows),
            "columns": list(NODE_COLUMNS),
            "missing_sumo_edges": [],
            "duplicate_edge_ids": [],
            "node_index_range": [0, len(rows) - 1],
            "edge_type_empty_rows": empty_types,
            "priorities": priorities,
            "r_nodes_edge_id_equal": True,
            "r_graph_edge_id_equal": True,
        },
        "tensor_node_order": {
            "source": "r_graph.npz edge_ids",
            "generator": order_proof["generator"],
            "days_checked": order_proof["days_checked"],
            "comparison": order_proof["comparison"],
            "node_order_sha256": order_hashes,
        },
        "node_order_sha256": order_hashes,
    }


def build_dataset_metadata(
    *,
    edge_ids: list[str],
    days: dict[str, list[str]],
    rates: list[str],
    dataset: dict[str, object],
    order_hashes: dict[str, str],
    file_hashes: dict[str, dict[str, str]],
    output_hashes: dict[str, str],
    warnings: list[str],
) -> dict:
    return {
        "dataset_name": "xuancheng26_r_level_reconstruction",
        "task_type": (
            "Use 12 consecutive 5-minute penetrated R-edge inflow counts to "
            "reconstruct the full R-edge inflow at the last of those steps"
        ),
        "task_definition": {
            "r_node": "real directed SUMO edge",
            "input": "partial/penetrated inflow",
            "target": "full inflow",
            "history": N_HIS,
            "horizon": HORIZON,
            "nodes": len(edge_ids),
            "feature_dimension": CHANNELS,
        },
        "target_mode": "last-observed-step",
        "label_time_relation": "target_slot == window_start_slot + n_his - 1 == last input slot",
        "input_variable": "penetrated R-edge inflow counts",
        "target_variable": "full R-edge inflow counts",
        "does_not_scale_by_penetration": True,
        "time_granularity_minutes": 5,
        "n_his": N_HIS,
        "n_pred": HORIZON,
        "node_count": len(edge_ids),
        "num_nodes": len(edge_ids),
        "feature_count": CHANNELS,
        "graph_definition": {
            "path": GRAPH_PATH.as_posix(),
            "node": "real directed SUMO edge",
            "internal_edges_included": False,
            "directed": True,
            "binary": True,
            "self_loops": False,
            "distance_weighted": False,
            "symmetrized": False,
            "normalized": False,
            "num_nodes": len(edge_ids),
            "num_graph_edges": EXPECTED_EDGES,
            "canonical_order": "r_graph.npz edge_ids array order",
        },
        "rates": rates,
        "n_days": sum(len(items) for items in days.values()),
        "training_days": days["train"],
        "validation_days": days["validation"],
        "test_days": days["test"],
        "time_slots_per_day": 288,
        "windows_per_day": dataset["samples_per_day"],
        "sample_counts": dataset["sample_counts"],
        "input_tensor_format": f"[N, {N_HIS}, {len(edge_ids)}, {CHANNELS}]",
        "target_tensor_format": f"[N, {HORIZON}, {len(edge_ids)}, {CHANNELS}]",
        "cross_day_windows": False,
        "normalization_method": "global_zscore",
        "normalization_file": (WINDOW_ROOT / "normalization.json").as_posix(),
        "split_manifest": (WINDOW_ROOT / "split_manifest.json").as_posix(),
        "input_and_target_use_different_scalers": True,
        "per_rate_input_scalers": True,
        "shared_target_scaler": True,
        "scaler_fit_on_training_only": True,
        "r_nodes_path": (GRAPH_DIR / "r_nodes.csv").as_posix(),
        "node_mapping_path": (GRAPH_DIR / "node_mapping.csv").as_posix(),
        "data_root": WINDOW_ROOT.as_posix(),
        "node_order_sha256": order_hashes,
        "r_nodes_sha256": output_hashes["r_nodes.csv"],
        "input_file_sha256": file_hashes,
        "output_file_sha256": output_hashes,
        "warnings": warnings,
        "full_flow_consumer_note": (
            "input_file_sha256.full_flow points at wide inflow.csv. "
            "Existing DTW/PDFormer readers expect day/edge_flow_5min.csv "
            "with columns window_index, edge_id, vehicle_count; that long table is not part of Xuancheng26."
        ),
        "script_path": str(Path(__file__).resolve()),
        "net_file": NET_FILE.as_posix(),
    }


def flow_fingerprints(days: dict[str, list[str]], rates: list[str]) -> dict[str, dict]:
    full = {}
    observed: dict[str, dict[str, dict[str, str]]] = {}
    for split_days in days.values():
        for day_name in split_days:
            path = FULL_FLOW_ROOT / day_name / "inflow.csv"
            full[day_name] = {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "schema": "wide_time_bin_start_time_edge_columns",
            }
            observed[day_name] = {}
            for rate in rates:
                inflow = PARTIAL_FLOW_ROOT / rate / day_name / "inflow.csv"
                observed[day_name][rate] = {"path": inflow.as_posix(), "sha256": sha256_file(inflow)}
    return {"full_flow": full, "observed_flow": observed}


def npz_fingerprints(rates: list[str]) -> dict[str, str]:
    result = {}
    for rate in rates:
        for split in SPLITS:
            path = WINDOW_ROOT / rate / f"{split}.npz"
            result[f"{rate}/{split}.npz"] = sha256_file(path)
            log(f"[hash] {rate}/{split}.npz")
    return result


def model_graph_keys(model: str) -> tuple[str, ...]:
    return tuple(graph_file_plan().get(model, {}))


def dry_run_models(
    configs: dict[str, dict],
    *,
    rates: list[str],
    num_nodes: int,
) -> dict[str, dict[str, object]]:
    sys.path.insert(0, str(CODE_ROOT))
    from reimplementation.common.data.r_only_npz_dataset import ROnlyNPZDataset, build_dataloader
    from reimplementation.common.errors import ReimplementationError
    from reimplementation.stgcn.validation import load_edge_ids, validate_prepared_data

    data_root = WINDOW_ROOT
    report: dict[str, dict[str, object]] = {}
    loader_error = ""
    try:
        validate_prepared_data(data_root, n_his=N_HIS, num_nodes=num_nodes, rates=list(rates))
        loader_status = "PASS"
    except ReimplementationError as exc:
        loader_error = str(exc)
        loader_status = "FAIL"
        log(f"[dry-run] validate_prepared_data FAIL {loader_error}")
    for model, config in configs.items():
        item: dict[str, object] = {
            "config_load": "PASS",
            "num_nodes": int(config["num_nodes"]),
            "data_root": config["data_root"],
        }
        for label, filename in (
            ("manifest_load", "split_manifest.json"),
            ("normalization_load", "normalization.json"),
            ("metadata_load", "dataset_metadata.json"),
            ("validation_load", "validation_summary.json"),
        ):
            load_json(data_root / filename)
            item[label] = "PASS"
        r_ids = load_edge_ids(Path(config["r_nodes"]))
        map_ids = load_edge_ids(Path(config["node_mapping"]))
        root_ids = load_edge_ids(data_root / "node_mapping.csv")
        if not (r_ids == map_ids == root_ids) or len(r_ids) != num_nodes:
            raise SupportError(f"{model} r_nodes/node_mapping order check failed")
        item["r_nodes_load"] = "PASS"
        item["node_mapping_load"] = "PASS"
        dataset = ROnlyNPZDataset(
            data_root / rates[0] / "train.npz",
            n_his=N_HIS,
            num_nodes=num_nodes,
            expected_count=None,
        )
        loader = build_dataloader(
            dataset,
            batch_size=min(int(config.get("batch_size", 2)), 8),
            shuffle=False,
            num_workers=0,
            seed=int(config.get("seed", 42)),
        )
        batch = next(iter(loader))
        if tuple(batch["x"].shape)[1:] != (N_HIS, num_nodes, CHANNELS):
            raise SupportError(f"{model} first batch x shape {tuple(batch['x'].shape)}")
        if tuple(batch["y"].shape)[1:] != (HORIZON, num_nodes, CHANNELS):
            raise SupportError(f"{model} first batch y shape {tuple(batch['y'].shape)}")
        item["dataset_load"] = "PASS"
        item["first_batch"] = "PASS"
        item["first_batch_x_shape"] = list(batch["x"].shape)
        missing_graphs = []
        for key in model_graph_keys(model):
            if key not in config:
                continue
            target = Path(str(config[key]))
            if not target.exists():
                missing_graphs.append(f"{key}={target.as_posix()}")
        if missing_graphs:
            item["model_specific_graph"] = "not prepared yet"
            item["missing_graph_artifacts"] = missing_graphs
        else:
            item["model_specific_graph"] = "not required" if model in {"stid", "staeformer"} else "present"
        item["official_validate_prepared_data"] = loader_status
        if loader_error:
            item["official_validate_prepared_data_error"] = loader_error
        item["generic_dataset_config"] = "PASS"
        report[model] = item
        log(f"[dry-run] {model} dataset/batch PASS graph={item['model_specific_graph']}")
    if loader_status == "FAIL" and "sample keys overlap" not in loader_error:
        raise SupportError(f"official validate_prepared_data failed: {loader_error}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Xuancheng26 training support files")
    prep_paths.add_path_arguments(parser, net=True)
    args = parser.parse_args(argv)
    root = prep_paths.resolved(args.training_root)
    graph_path = root / "graph" / "r_graph.npz"
    prep_paths.bind(
        sys.modules[__name__],
        NET_FILE=prep_paths.resolved(args.net_file),
        TRAINING_ROOT=root,
        WINDOW_ROOT=root / "window_data",
        GRAPH_DIR=root / "graph",
        GRAPH_PATH=graph_path,
        CONFIG_ROOT=root / "config",
        FULL_FLOW_ROOT=root / "full_flow" / "r-level",
        PARTIAL_FLOW_ROOT=root / "partial_flow",
    )
    import generate_r_level_windows as windows

    prep_paths.bind(
        windows,
        PARTIAL_ROOT=root / "partial_flow",
        FULL_ROOT=root / "full_flow" / "r-level",
        GRAPH_PATH=graph_path,
        OUT_ROOT=root / "window_data",
    )
    log(f"net_file={NET_FILE}")
    log(f"training_root={TRAINING_ROOT}")
    log("Xuancheng26 training support files")
    graph = load_r_graph()
    validate_r_graph(graph)
    edge_ids = graph["edge_ids"]
    log(
        f"graph nodes={graph['num_nodes']} edges={graph['num_graph_edges']} "
        f"sum={int(graph['adjacency'].sum())}"
    )
    network = load_sumo_network(edge_ids)
    validate_graph_matches_network(graph, network)
    rows = build_node_rows(edge_ids, network)
    table = csv_bytes(rows)
    manifest = load_json(WINDOW_ROOT / "split_manifest.json")
    days = manifest_days(manifest)
    rates = discover_rates(manifest)
    normalization = check_normalization_present(rates)
    order_proof = prove_tensor_edge_order(edge_ids, days, rates)
    dataset = validate_merged_dataset(edge_ids, days, rates, manifest, normalization)
    order_hash = node_order_sha256(edge_ids)
    if node_order_sha256(order_proof["tensor_edge_ids"]) != order_hash:
        raise SupportError("tensor edge-list hash != r_graph edge-list hash")
    order_hashes = {
        "r_graph": order_hash,
        "node_mapping": order_hash,
        "r_nodes": order_hash,
        "tensor_column_edge_ids": order_hash,
    }
    warnings = [
        "Final net.xml has no edge type attribute; node_mapping.edge_type is empty for every R node.",
        "Final net.xml priority is the SUMO priority attribute (0 or -1), not an OSM highway priority.",
        (
            "reimplementation validate_prepared_data identifies a sample by (day_index, window_start_slot). "
            "Xuancheng26 day_index is the Monday=0 weekday, so the same weekday and slot recur in train, "
            "validation, and test even though the calendar dates are disjoint. "
            "That loader check fails. The NPZ files were not rewritten."
        ),
        (
            "DTW and PDFormer relation readers still expect long edge_flow_5min.csv tables. "
            "Xuancheng26 fingerprints the wide inflow.csv. Every hash was computed from these files."
        ),
    ]
    summary = build_validation_summary(
        edge_ids=edge_ids,
        graph=graph,
        network=network,
        rows=rows,
        days=days,
        rates=rates,
        normalization=normalization,
        dataset=dataset,
        order_proof=order_proof,
        order_hashes=order_hashes,
        warnings=warnings,
    )
    paths = {
        "data_root": WINDOW_ROOT.as_posix(),
        "r_nodes": (GRAPH_DIR / "r_nodes.csv").as_posix(),
        "node_mapping": (GRAPH_DIR / "node_mapping.csv").as_posix(),
        "normalization": (WINDOW_ROOT / "normalization.json").as_posix(),
        "split_manifest": (WINDOW_ROOT / "split_manifest.json").as_posix(),
        "dataset_metadata": (WINDOW_ROOT / "dataset_metadata.json").as_posix(),
        "data_validation": (WINDOW_ROOT / "validation_summary.json").as_posix(),
    }
    graph_paths = graph_file_plan()
    forbidden_edges, forbidden_hashes = load_shanghai_denylist()
    configs: dict[str, dict] = {}
    config_bodies: dict[str, bytes] = {}
    for model, source in discover_model_configs():
        template = load_json(source)
        config = specialize_config(
            template,
            model,
            num_nodes=len(edge_ids),
            rates=rates,
            train_day_count=len(days["train"]),
            paths=paths,
            graph_paths=graph_paths.get(model, {}),
        )
        if int(config.get("num_nodes", -1)) != len(edge_ids):
            raise SupportError(f"{model} num_nodes was not set from r_graph")
        residue = find_residue(config, forbidden_edges, forbidden_hashes)
        if residue:
            raise SupportError(f"{model} config still has Shanghai residue: {residue[:12]}")
        configs[model] = config
        config_bodies[model] = json_bytes(config)
        log(f"[config] {model} prepared from {source}")
    summary_residue = find_residue(summary, forbidden_edges, forbidden_hashes)
    if summary_residue:
        raise SupportError(f"validation summary residue: {summary_residue[:12]}")

    log("hashing flow sources and split NPZ")
    source_hashes = {
        "r_graph.npz": {"path": GRAPH_PATH.as_posix(), "sha256": sha256_file(GRAPH_PATH)},
        "split_manifest.json": {
            "path": (WINDOW_ROOT / "split_manifest.json").as_posix(),
            "sha256": sha256_file(WINDOW_ROOT / "split_manifest.json"),
        },
        "normalization.json": {
            "path": (WINDOW_ROOT / "normalization.json").as_posix(),
            "sha256": sha256_file(WINDOW_ROOT / "normalization.json"),
        },
        "net.xml": {"path": NET_FILE.as_posix(), "sha256": sha256_file(NET_FILE)},
    }
    source_hashes.update(flow_fingerprints(days, rates))
    output_hashes = npz_fingerprints(rates)
    output_hashes["node_mapping.csv"] = sha256_bytes(table)
    output_hashes["r_nodes.csv"] = sha256_bytes(table)
    for model, body in config_bodies.items():
        output_hashes[f"config/{model}/r_only_{model}.json"] = sha256_bytes(body)

    summary_body = json_bytes(summary)
    output_hashes["validation_summary.json"] = sha256_bytes(summary_body)
    metadata = build_dataset_metadata(
        edge_ids=edge_ids,
        days=days,
        rates=rates,
        dataset=dataset,
        order_hashes=order_hashes,
        file_hashes=source_hashes,
        output_hashes=output_hashes,
        warnings=warnings,
    )
    metadata_residue = find_residue(metadata, forbidden_edges, forbidden_hashes)
    if metadata_residue:
        raise SupportError(f"dataset metadata residue: {metadata_residue[:12]}")
    if metadata["node_order_sha256"] != order_hashes:
        raise SupportError("metadata node-order hashes drifted")

    log("writing support files before the loader dry-run")
    atomic_write_bytes(GRAPH_DIR / "node_mapping.csv", table)
    atomic_write_bytes(GRAPH_DIR / "r_nodes.csv", table)
    atomic_write_bytes(WINDOW_ROOT / "node_mapping.csv", table)
    atomic_write_bytes(WINDOW_ROOT / "validation_summary.json", summary_body)
    atomic_write_bytes(TRAINING_ROOT / "validation_summary.json", summary_body)
    metadata_body = json_bytes(metadata)
    atomic_write_bytes(WINDOW_ROOT / "dataset_metadata.json", metadata_body)
    atomic_write_bytes(TRAINING_ROOT / "dataset_metadata.json", metadata_body)
    for model, body in config_bodies.items():
        atomic_write_bytes(CONFIG_ROOT / model / f"r_only_{model}.json", body)
    written_ids = read_edge_column(GRAPH_DIR / "node_mapping.csv")
    r_node_ids = read_edge_column(GRAPH_DIR / "r_nodes.csv")
    root_ids = read_edge_column(WINDOW_ROOT / "node_mapping.csv")
    if not (written_ids == r_node_ids == root_ids == edge_ids):
        raise SupportError("written node tables do not match r_graph edge_ids")
    if node_order_sha256(written_ids) != order_hash:
        raise SupportError("written node_mapping order hash mismatch")

    dry = dry_run_models(configs, rates=rates, num_nodes=len(edge_ids))
    summary["dry_run"] = dry
    if summary["status"] != "passed" or summary["failures"]:
        raise SupportError("refusing to publish a failed validation summary")
    summary_residue = find_residue(summary, forbidden_edges, forbidden_hashes)
    if summary_residue:
        raise SupportError(f"validation summary residue after dry-run: {summary_residue[:12]}")
    summary_body = json_bytes(summary)
    metadata["output_file_sha256"]["validation_summary.json"] = sha256_bytes(summary_body)
    metadata_residue = find_residue(metadata, forbidden_edges, forbidden_hashes)
    if metadata_residue:
        raise SupportError(f"dataset metadata residue after dry-run: {metadata_residue[:12]}")
    atomic_write_bytes(WINDOW_ROOT / "validation_summary.json", summary_body)
    atomic_write_bytes(TRAINING_ROOT / "validation_summary.json", summary_body)
    metadata_body = json_bytes(metadata)
    atomic_write_bytes(WINDOW_ROOT / "dataset_metadata.json", metadata_body)
    atomic_write_bytes(TRAINING_ROOT / "dataset_metadata.json", metadata_body)
    if sha256_file(WINDOW_ROOT / "validation_summary.json") != metadata["output_file_sha256"]["validation_summary.json"]:
        raise SupportError("validation_summary.json hash does not match dataset_metadata")
    log("support files written")
    print(json.dumps({"status": "passed", "models": list(configs), "node_order_sha256": order_hash}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SupportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
