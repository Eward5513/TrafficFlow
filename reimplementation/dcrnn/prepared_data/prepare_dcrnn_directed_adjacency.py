"""Build a directed, distance-weighted adjacency for original DCRNN.

Reads the R-only directed binary graph and already-validated directional
center distances from the STGCN edge CSV. Does not overwrite R-graph or
STGCN-graph files. Does not read trajectories, flows, or vehicle samples.
Does not add self-loops, symmetrize, normalize, or precompute random-walk
supports.

Original DCRNN (reference/dcrnn) mapping
---------------------------------------
``scripts/gen_adj_mx.py`` writes pickle protocol 2:

    [sensor_ids, sensor_id_to_ind, adj_mx]

``lib/utils.py`` ``load_graph_data()`` reads that triple. ``A[i, j]`` is the
weight from sensor i to sensor j (CSV columns ``from, to, distance``). The
Gaussian kernel and ``normalized_k=0.1`` sparsity cut in ``gen_adj_mx.py``
are PeMS construction steps; this script does **not** copy them.

``model/dcrnn_cell.py`` builds diffusion supports at runtime from the loaded
base ``adj_mx``. Official METR-LA / PEMS-BAY yaml files set

    filter_type: dual_random_walk

which stores:

    (D_out^{-1} A)^T
    (D_in^{-1}  A^T)^T

The extra transpose is the SparseTensor layout used by
``tf.sparse_tensor_dense_matmul(support, x)``. This script saves the base
directed Gaussian ``W_dir`` only.

Gaussian weights
----------------
    W_dir[i, j] = exp(-(D_dir[i, j] ** 2) / (sigma ** 2))
        if A_dir[i, j] == 1 and i != j else 0

``sigma`` defaults to the STGCN metadata value so both models share the same
length scale. ``epsilon`` defaults to 0.0 so every real directed edge is kept.
There is no PeMS ``/10000`` rescaling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import pickle
import sys
import tempfile
import types
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


SCRIPT_VERSION = "1.0.0"
RANDOM_SEED_DEFAULT = 42
EXAMPLE_LIMIT = 8
HASH_CHUNK = 8 * 1024 * 1024
DISTANCE_ATOL = 1e-9
WEIGHT_ATOL = 1e-9
FLOAT32_WEIGHT_ATOL = 1e-7
DEFAULT_EPSILON = 0.0
DEFAULT_PICKLE_PROTOCOL = 2
FLOAT_TEXT_FORMAT = ".17g"
ORIGINAL_FILTER_TYPES = ("laplacian", "random_walk", "dual_random_walk")
OFFICIAL_DEFAULT_FILTER_TYPE = "dual_random_walk"
CODE_DEFAULT_FILTER_TYPE = "laplacian"
GAUSSIAN_FORMULA = (
    "W_dir[i,j] = exp(-(D_dir[i,j] ** 2) / (sigma ** 2)) "
    "if A_dir[i,j]==1 and i!=j else 0; then W_dir[i,j]=0 if W_dir[i,j] < epsilon; "
    "W_dir[i,i]=0. sigma is the length-scale, not sigma**2."
)
DIRECTED_TOPOLOGY_DEFINITION = (
    "A_dir[i,j] = 1 iff a real SUMO <connection from=edge_i to=edge_j> exists. "
    "Rows are sources, columns are targets. Not symmetrized."
)
DIRECTED_DISTANCE_DEFINITION = (
    "D_dir[i,i]=0. D_dir[i,j] is the direction-specific direct-connection "
    "center distance from stgcn_weighted_edges.csv when A_dir[i,j]==1 and i!=j; "
    "otherwise inf. Bidirectional pairs keep two independent distances. "
    "selected_distance is never used."
)
DISTANCE_FIELD_RULE = (
    "CSV rows store undirected pairs with node_index_i < node_index_j. "
    "A_dir[u,v]==1 and u<v reads distance_i_to_j from row (u,v). "
    "A_dir[u,v]==1 and u>v reads distance_j_to_i from row (v,u)."
)
ORIGINAL_RW_FORWARD = (
    "forward_support = calculate_random_walk_matrix(adj_mx).T "
    "= (D_out^{-1} A)^T. support @ x aggregates features from upstream "
    "sources into downstream targets along A[i,j]=i->j."
)
ORIGINAL_RW_REVERSE = (
    "reverse_support = calculate_random_walk_matrix(adj_mx.T).T "
    "= (D_in^{-1} A^T)^T. support @ x aggregates along the reverse of A."
)

SCRIPT_PATH = Path(__file__).resolve()
PREPARED_DATA_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = PREPARED_DATA_DIR.parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
STGCN_ADJ_DIR = (
    PROJECT_ROOT
    / "reimplementation"
    / "stgcn"
    / "prepared_data"
    / "r-only"
    / "adjacency_matrix"
)
DEFAULT_R_NODES = ANALYSIS_DIR / "graph" / "r_graph" / "r_nodes.csv"
DEFAULT_DIRECTED_ADJACENCY = ANALYSIS_DIR / "graph" / "r_graph" / "r_adjacency.npy"
DEFAULT_SUMO_CONNECTIONS = ANALYSIS_DIR / "graph" / "r_graph" / "r_sumo_connections.csv"
DEFAULT_R_EDGES = ANALYSIS_DIR / "graph" / "r_graph" / "r_edges.csv"
DEFAULT_R_GRAPH_VALIDATION = ANALYSIS_DIR / "graph" / "r_graph" / "r_graph_validation.json"
DEFAULT_WEIGHTED_EDGES = STGCN_ADJ_DIR / "stgcn_weighted_edges.csv"
DEFAULT_SOURCE_METADATA = STGCN_ADJ_DIR / "stgcn_weighted_adjacency_metadata.json"
DEFAULT_SOURCE_VALIDATION = STGCN_ADJ_DIR / "stgcn_weighted_adjacency_validation.json"
DEFAULT_SOURCE_UNDIRECTED = STGCN_ADJ_DIR / "stgcn_undirected_topology.npy"
DEFAULT_STGCN_NODE_MAPPING = (
    PROJECT_ROOT / "reimplementation" / "stgcn" / "prepared_data" / "r-only" / "node_mapping.csv"
)
DEFAULT_ORIGINAL_DCRNN_ROOT = PROJECT_ROOT / "reference" / "dcrnn"
DEFAULT_OUTPUT_DIR = PREPARED_DATA_DIR / "r-only" / "adjacency_matrix"

OUTPUT_ARTIFACTS = (
    "dcrnn_directed_topology.npy",
    "dcrnn_directed_center_distance.npy",
    "dcrnn_weighted_adjacency.npy",
    "dcrnn_weighted_adjacency_sparse.npz",
    "dcrnn_adj_mx.pkl",
    "dcrnn_directed_edges.csv",
    "dcrnn_graph_metadata.json",
    "dcrnn_graph_validation.json",
)

EDGE_CSV_FIELDS = (
    "from_node_index",
    "to_node_index",
    "from_edge_id",
    "to_edge_id",
    "direct_connection_count",
    "direction_distance",
    "sigma",
    "raw_weight",
    "epsilon",
    "final_weight",
    "retained_after_threshold",
    "source_pair_node_index_i",
    "source_pair_node_index_j",
    "source_distance_field",
    "from_lane",
    "to_lane",
    "via_lanes",
    "path_lane_ids",
    "path_uses_internal_lanes",
)


class PrepareError(Exception):
    pass


@dataclass
class NodeMapping:
    node_index: list[int]
    edge_ids: list[str]
    index_by_edge: dict[str, int]
    rows: list[dict[str, str]]


@dataclass
class WeightedEdgeRow:
    node_index_i: int
    node_index_j: int
    edge_id_i: str
    edge_id_j: str
    has_i_to_j: bool
    has_j_to_i: bool
    count_i_to_j: int
    count_j_to_i: int
    distance_i_to_j: float | None
    distance_j_to_i: float | None
    selected_distance: float | None
    selected_direction: str
    selected_from_lane: str
    selected_to_lane: str
    selected_via_lanes: str
    selected_path_lane_ids: str
    path_uses_internal_lanes: str


@dataclass
class DirectedEdge:
    src: int
    dst: int
    from_edge_id: str
    to_edge_id: str
    distance: float
    connection_count: int
    source_pair_i: int
    source_pair_j: int
    source_distance_field: str
    from_lane: str = ""
    to_lane: str = ""
    via_lanes: str = ""
    path_lane_ids: str = ""
    path_uses_internal_lanes: bool = False
    raw_weight: float = 0.0
    final_weight: float = 0.0
    retained: bool = False


@dataclass
class OriginalDCRNNApi:
    root: Path
    utils: Any
    load_graph_data_source: Path
    cell_source: Path
    gen_adj_source: Path


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def posix(path: Path) -> str:
    return path.resolve().as_posix()


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": posix(path),
        "repo_path": repo_rel(path),
        "sha256": sha256_file(path),
        "size_bytes": stat.st_size,
    }


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: Iterable[Mapping[str, object]],
) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in fieldnames})
    atomic_write_text(path, buffer.getvalue())


def atomic_write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npy", dir=str(path.parent)
    )
    os.close(fd)
    try:
        np.save(tmp_name, array, allow_pickle=False)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_sparse_npz(path: Path, matrix: sparse.spmatrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent)
    )
    os.close(fd)
    try:
        sparse.save_npz(tmp_name, matrix)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_pickle(path: Path, payload: object, protocol: int) -> None:
    buffer = io.BytesIO()
    pickle.dump(payload, buffer, protocol=protocol)
    atomic_write_bytes(path, buffer.getvalue())


def json_ready(value: object) -> object:
    if isinstance(value, Path):
        return posix(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def format_float(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return format(float(value), FLOAT_TEXT_FORMAT)


def parse_optional_float(raw: str | None) -> float | None:
    text = "" if raw is None else str(raw).strip()
    if text == "":
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise PrepareError(f"not a float: {raw!r}") from exc
    if not math.isfinite(value):
        return None
    return value


def parse_bool_flag(raw: str | None, field_name: str) -> bool:
    text = "" if raw is None else str(raw).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no", ""}:
        return False
    raise PrepareError(f"invalid boolean {field_name}={raw!r}")


def parse_nonneg_int(raw: str | None, field_name: str) -> int:
    text = "" if raw is None else str(raw).strip()
    if text == "":
        return 0
    try:
        value = int(text)
    except ValueError as exc:
        raise PrepareError(f"invalid integer {field_name}={raw!r}") from exc
    if value < 0:
        raise PrepareError(f"{field_name} is negative: {value}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PrepareError(f"missing JSON: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PrepareError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PrepareError(f"{path} is not a JSON object")
    return payload


def assert_validation_ok(path: Path, label: str) -> dict[str, Any]:
    payload = load_json(path)
    status = payload.get("status")
    overall = payload.get("overall_passed")
    construction = payload.get("construction_passed")
    failures = payload.get("failures") or []
    if status not in (None, "ok", "passed"):
        raise PrepareError(f"{label} validation status is {status!r} in {path}")
    if overall is False or construction is False:
        raise PrepareError(f"{label} validation report is not passed: {path}")
    if status is None and overall is not True and construction is not True:
        raise PrepareError(f"{label} validation report has no passing status: {path}")
    if failures:
        raise PrepareError(f"{label} validation report lists failures: {path}")
    return payload


def gaussian_weight(distance: float, sigma: float) -> float:
    if not math.isfinite(distance) or distance <= 0.0:
        raise PrepareError(f"Gaussian weight requires finite distance > 0, got {distance}")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise PrepareError(f"sigma must be finite and > 0, got {sigma}")
    return float(math.exp(-(distance * distance) / (sigma * sigma)))


def dense_from_support(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        array = np.asarray(matrix.todense(), dtype=np.float64)
    else:
        array = np.asarray(matrix, dtype=np.float64)
    return array


def import_original_dcrnn_utils(dcrnn_root: Path) -> OriginalDCRNNApi:
    root = dcrnn_root.expanduser().resolve()
    utils_path = root / "lib" / "utils.py"
    cell_path = root / "model" / "dcrnn_cell.py"
    gen_adj_path = root / "scripts" / "gen_adj_mx.py"
    for path, label in (
        (utils_path, "original DCRNN lib/utils.py"),
        (cell_path, "original DCRNN model/dcrnn_cell.py"),
        (gen_adj_path, "original DCRNN scripts/gen_adj_mx.py"),
    ):
        if not path.is_file():
            raise PrepareError(f"{label} not found: {path}")
    try:
        import tensorflow as _tf  # noqa: F401
    except Exception:
        if "tensorflow" not in sys.modules:
            stub = types.ModuleType("tensorflow")
            stub.__dict__["__spec__"] = None
            sys.modules["tensorflow"] = stub
    spec = importlib.util.spec_from_file_location(
        "trafficflow_original_dcrnn_lib_utils",
        utils_path,
    )
    if spec is None or spec.loader is None:
        raise PrepareError(f"cannot import original DCRNN utils from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("load_graph_data", "load_pickle", "calculate_random_walk_matrix"):
        if not hasattr(module, name):
            raise PrepareError(f"original DCRNN utils.py is missing {name}()")
    return OriginalDCRNNApi(
        root=root,
        utils=module,
        load_graph_data_source=utils_path,
        cell_source=cell_path,
        gen_adj_source=gen_adj_path,
    )


def build_original_supports(
    adj_mx: np.ndarray,
    filter_type: str,
    original: OriginalDCRNNApi,
) -> list[np.ndarray]:
    """Mirror DCGRUCell.__init__ support construction without TensorFlow cells."""
    utils = original.utils
    supports: list[Any] = []
    if filter_type == "laplacian":
        supports.append(utils.calculate_scaled_laplacian(adj_mx, lambda_max=None))
    elif filter_type == "random_walk":
        supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
    elif filter_type == "dual_random_walk":
        supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
        supports.append(utils.calculate_random_walk_matrix(adj_mx.T).T)
    else:
        raise PrepareError(f"unsupported original filter_type {filter_type!r}")
    dense = [dense_from_support(item) for item in supports]
    for index, matrix in enumerate(dense):
        if matrix.shape != adj_mx.shape:
            raise PrepareError(
                f"{filter_type} support {index} shape {matrix.shape} != {adj_mx.shape}"
            )
        if not np.isfinite(matrix).all():
            raise PrepareError(f"{filter_type} support {index} contains NaN or inf")
    return dense


def load_node_mapping(path: Path) -> NodeMapping:
    if not path.is_file():
        raise PrepareError(f"r_nodes.csv not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise PrepareError(f"{path} has no data rows")
    required = {"node_index", "edge_id"}
    missing = required - set(rows[0].keys())
    if missing:
        raise PrepareError(f"{path} missing columns {sorted(missing)}")
    indices: list[int] = []
    edge_ids: list[str] = []
    for row in rows:
        index_text = str(row["node_index"]).strip()
        edge_id = str(row["edge_id"]).strip()
        if not index_text or not edge_id:
            raise PrepareError(f"{path} has an empty node_index or edge_id")
        try:
            indices.append(int(index_text, 10))
        except ValueError as exc:
            raise PrepareError(f"{path} has non-integer node_index {index_text!r}") from exc
        edge_ids.append(edge_id)
    expected = list(range(len(rows)))
    if indices != expected:
        raise PrepareError(
            f"{path} node_index must be 0..{len(rows) - 1} in file order; "
            "do not reorder by edge ID"
        )
    if len(set(edge_ids)) != len(edge_ids):
        raise PrepareError(f"{path} has duplicate edge_id values")
    return NodeMapping(
        node_index=indices,
        edge_ids=edge_ids,
        index_by_edge={edge_id: i for i, edge_id in enumerate(edge_ids)},
        rows=rows,
    )


def load_directed_adjacency(path: Path, n_nodes: int) -> np.ndarray:
    if not path.is_file():
        raise PrepareError(f"directed adjacency not found: {path}")
    loaded = np.load(path, allow_pickle=False)
    if loaded.ndim != 2 or loaded.shape[0] != loaded.shape[1]:
        raise PrepareError(f"{path} is not a square matrix: shape={loaded.shape}")
    if loaded.shape[0] != n_nodes:
        raise PrepareError(
            f"{path} shape {tuple(loaded.shape)} does not match node count {n_nodes}"
        )
    if not np.issubdtype(loaded.dtype, np.number):
        raise PrepareError(f"{path} dtype {loaded.dtype} is not numeric")
    unique = {float(value) for value in np.unique(loaded)}
    if not unique <= {0.0, 1.0}:
        raise PrepareError(f"{path} is not binary 0/1; unique={sorted(unique)}")
    return loaded.astype(np.uint8, copy=True)


def load_undirected_topology(path: Path, n_nodes: int) -> np.ndarray:
    if not path.is_file():
        raise PrepareError(f"STGCN undirected topology not found: {path}")
    loaded = np.load(path, allow_pickle=False).astype(np.uint8, copy=True)
    if loaded.shape != (n_nodes, n_nodes):
        raise PrepareError(
            f"{path} shape {tuple(loaded.shape)} does not match [{n_nodes}, {n_nodes}]"
        )
    return loaded


def load_weighted_edge_rows(path: Path) -> dict[tuple[int, int], WeightedEdgeRow]:
    if not path.is_file():
        raise PrepareError(f"weighted-edges CSV not found: {path}")
    required = {
        "node_index_i",
        "node_index_j",
        "edge_id_i",
        "edge_id_j",
        "has_i_to_j_direct_connection",
        "has_j_to_i_direct_connection",
        "direct_connection_count_i_to_j",
        "direct_connection_count_j_to_i",
        "distance_i_to_j",
        "distance_j_to_i",
        "selected_distance",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise PrepareError(f"{path} has no header")
        missing = required - set(reader.fieldnames)
        if missing:
            raise PrepareError(f"{path} missing columns {sorted(missing)}")
        rows = list(reader)
    indexed: dict[tuple[int, int], WeightedEdgeRow] = {}
    for raw in rows:
        i = int(str(raw["node_index_i"]).strip(), 10)
        j = int(str(raw["node_index_j"]).strip(), 10)
        if i >= j:
            raise PrepareError(
                f"{path} expects node_index_i < node_index_j, got ({i}, {j})"
            )
        key = (i, j)
        if key in indexed:
            raise PrepareError(f"{path} has duplicate pair {key}")
        indexed[key] = WeightedEdgeRow(
            node_index_i=i,
            node_index_j=j,
            edge_id_i=str(raw["edge_id_i"]).strip(),
            edge_id_j=str(raw["edge_id_j"]).strip(),
            has_i_to_j=parse_bool_flag(
                raw.get("has_i_to_j_direct_connection"), "has_i_to_j_direct_connection"
            ),
            has_j_to_i=parse_bool_flag(
                raw.get("has_j_to_i_direct_connection"), "has_j_to_i_direct_connection"
            ),
            count_i_to_j=parse_nonneg_int(
                raw.get("direct_connection_count_i_to_j"),
                "direct_connection_count_i_to_j",
            ),
            count_j_to_i=parse_nonneg_int(
                raw.get("direct_connection_count_j_to_i"),
                "direct_connection_count_j_to_i",
            ),
            distance_i_to_j=parse_optional_float(raw.get("distance_i_to_j")),
            distance_j_to_i=parse_optional_float(raw.get("distance_j_to_i")),
            selected_distance=parse_optional_float(raw.get("selected_distance")),
            selected_direction=str(raw.get("selected_direction") or "").strip(),
            selected_from_lane=str(raw.get("selected_from_lane") or "").strip(),
            selected_to_lane=str(raw.get("selected_to_lane") or "").strip(),
            selected_via_lanes=str(raw.get("selected_via_lanes") or "").strip(),
            selected_path_lane_ids=str(raw.get("selected_path_lane_ids") or "").strip(),
            path_uses_internal_lanes=str(raw.get("path_uses_internal_lanes") or "").strip(),
        )
    return indexed


def load_sumo_connections(
    path: Path,
    index_by_edge: Mapping[str, int],
) -> dict[tuple[int, int], list[dict[str, str]]]:
    if not path.is_file():
        raise PrepareError(f"SUMO connections CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise PrepareError(f"{path} has no header")
        required = {"from", "to"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise PrepareError(f"{path} missing columns {sorted(missing)}")
        grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
        unknown: list[tuple[str, str]] = []
        for raw in reader:
            src_id = str(raw.get("from") or "").strip()
            dst_id = str(raw.get("to") or "").strip()
            if src_id not in index_by_edge or dst_id not in index_by_edge:
                unknown.append((src_id, dst_id))
                continue
            grouped[(index_by_edge[src_id], index_by_edge[dst_id])].append(
                {
                    "from": src_id,
                    "to": dst_id,
                    "fromLane": str(raw.get("fromLane") or "").strip(),
                    "toLane": str(raw.get("toLane") or "").strip(),
                    "via": str(raw.get("via") or "").strip(),
                }
            )
    if unknown:
        raise PrepareError(
            "r_sumo_connections.csv has endpoints outside r_nodes.csv: "
            + ", ".join(f"{a}->{b}" for a, b in unknown[:EXAMPLE_LIMIT])
        )
    return grouped


def join_unique(values: Iterable[str]) -> str:
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return "|".join(seen)


def connection_audit_fields(
    connections: list[dict[str, str]],
) -> tuple[str, str, str, bool]:
    from_lane = join_unique(item["fromLane"] for item in connections)
    to_lane = join_unique(item["toLane"] for item in connections)
    via_lanes = join_unique(item["via"] for item in connections)
    uses_internal = any(item["via"] for item in connections)
    return from_lane, to_lane, via_lanes, uses_internal


def selected_path_for_direction(row: WeightedEdgeRow, field_name: str) -> str:
    if field_name == "distance_i_to_j" and row.selected_direction == "i_to_j":
        return row.selected_path_lane_ids
    if field_name == "distance_j_to_i" and row.selected_direction == "j_to_i":
        return row.selected_path_lane_ids
    return ""


def resolve_sigma(
    explicit: float | None,
    metadata: Mapping[str, Any],
    metadata_path: Path,
) -> tuple[float, str]:
    if explicit is not None:
        if not math.isfinite(explicit) or explicit <= 0.0:
            raise PrepareError(f"--sigma must be finite and > 0, got {explicit}")
        return float(explicit), "cli --sigma"
    raw = metadata.get("sigma")
    try:
        sigma = float(raw)
    except (TypeError, ValueError) as exc:
        raise PrepareError(f"{metadata_path} is missing a numeric sigma") from exc
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise PrepareError(f"{metadata_path} sigma must be finite and > 0, got {sigma}")
    source = str(metadata.get("sigma_source") or "stgcn_weighted_adjacency_metadata.json")
    return sigma, f"stgcn metadata ({source})"


def direction_lookup(
    src: int,
    dst: int,
    csv_rows: Mapping[tuple[int, int], WeightedEdgeRow],
) -> tuple[WeightedEdgeRow, str, float | None, bool, int]:
    if src == dst:
        raise PrepareError("direction_lookup called on a diagonal pair")
    if src < dst:
        key = (src, dst)
        field_name = "distance_i_to_j"
        if key not in csv_rows:
            raise PrepareError(f"STGCN CSV missing pair {key} needed for {src}->{dst}")
        row = csv_rows[key]
        return row, field_name, row.distance_i_to_j, row.has_i_to_j, row.count_i_to_j
    key = (dst, src)
    field_name = "distance_j_to_i"
    if key not in csv_rows:
        raise PrepareError(f"STGCN CSV missing pair {key} needed for {src}->{dst}")
    row = csv_rows[key]
    return row, field_name, row.distance_j_to_i, row.has_j_to_i, row.count_j_to_i


def build_directed_graph(
    mapping: NodeMapping,
    directed: np.ndarray,
    csv_rows: Mapping[tuple[int, int], WeightedEdgeRow],
    connections: Mapping[tuple[int, int], list[dict[str, str]]],
    undirected: np.ndarray,
    sigma: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[DirectedEdge], dict[str, int]]:
    n_nodes = len(mapping.edge_ids)
    source_self_loops = int(np.count_nonzero(np.diag(directed)))
    topology = directed.copy()
    np.fill_diagonal(topology, 0)
    expected_undirected = np.logical_or(topology, topology.T).astype(np.uint8)
    np.fill_diagonal(expected_undirected, 0)
    if not np.array_equal(undirected, expected_undirected):
        raise PrepareError(
            "STGCN undirected topology does not match logical_or(A_dir, A_dir.T) "
            "with zero diagonal"
        )

    csv_pairs = set(csv_rows)
    undirected_pairs = {
        (int(i), int(j))
        for i, j in zip(*np.where(np.triu(undirected, k=1) == 1))
    }
    if csv_pairs != undirected_pairs:
        extra = sorted(csv_pairs - undirected_pairs)[:EXAMPLE_LIMIT]
        missing = sorted(undirected_pairs - csv_pairs)[:EXAMPLE_LIMIT]
        raise PrepareError(
            "STGCN CSV pair set does not match undirected topology; "
            f"extra={extra} missing={missing}"
        )

    for (i, j), row in csv_rows.items():
        if row.edge_id_i != mapping.edge_ids[i] or row.edge_id_j != mapping.edge_ids[j]:
            raise PrepareError(
                f"CSV pair ({i}, {j}) edge_id mismatch with r_nodes.csv"
            )
        if bool(topology[i, j]) != row.has_i_to_j:
            raise PrepareError(
                f"CSV has_i_to_j for ({i}, {j}) does not match A_dir[{i},{j}]"
            )
        if bool(topology[j, i]) != row.has_j_to_i:
            raise PrepareError(
                f"CSV has_j_to_i for ({i}, {j}) does not match A_dir[{j},{i}]"
            )
        if row.has_i_to_j and (row.distance_i_to_j is None or row.distance_i_to_j <= 0.0):
            raise PrepareError(
                f"A_dir[{i},{j}]==1 but CSV distance_i_to_j is missing or not > 0"
            )
        if row.has_j_to_i and (row.distance_j_to_i is None or row.distance_j_to_i <= 0.0):
            raise PrepareError(
                f"A_dir[{j},{i}]==1 but CSV distance_j_to_i is missing or not > 0"
            )
        if not row.has_i_to_j and row.distance_i_to_j is not None:
            raise PrepareError(
                f"CSV assigned distance_i_to_j to non-edge ({i}, {j})"
            )
        if not row.has_j_to_i and row.distance_j_to_i is not None:
            raise PrepareError(
                f"CSV assigned distance_j_to_i to non-edge ({j}, {i})"
            )

    distance = np.full((n_nodes, n_nodes), np.inf, dtype=np.float64)
    np.fill_diagonal(distance, 0.0)
    weights64 = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    directed_edges: list[DirectedEdge] = []
    nonedge_distance_computation_count = 0

    src_idx, dst_idx = np.where(topology == 1)
    for src, dst in zip(src_idx.tolist(), dst_idx.tolist()):
        row, field_name, dir_distance, has_dir, csv_count = direction_lookup(
            src, dst, csv_rows
        )
        if not has_dir:
            raise PrepareError(f"A_dir[{src},{dst}]==1 but CSV direction flag is false")
        if dir_distance is None or not math.isfinite(dir_distance) or dir_distance <= 0.0:
            raise PrepareError(
                f"A_dir[{src},{dst}]==1 but no finite directional distance > 0 "
                f"in field {field_name}"
            )
        conn_list = connections.get((src, dst), [])
        if not conn_list:
            raise PrepareError(
                f"A_dir[{src},{dst}]==1 but r_sumo_connections.csv has no {src}->{dst}"
            )
        if csv_count != len(conn_list):
            raise PrepareError(
                f"connection count mismatch for {src}->{dst}: "
                f"CSV={csv_count} sumo={len(conn_list)}"
            )
        from_lane, to_lane, via_lanes, uses_internal = connection_audit_fields(conn_list)
        path_lane_ids = selected_path_for_direction(row, field_name)
        distance[src, dst] = float(dir_distance)
        raw = gaussian_weight(float(dir_distance), sigma)
        directed_edges.append(
            DirectedEdge(
                src=src,
                dst=dst,
                from_edge_id=mapping.edge_ids[src],
                to_edge_id=mapping.edge_ids[dst],
                distance=float(dir_distance),
                connection_count=len(conn_list),
                source_pair_i=row.node_index_i,
                source_pair_j=row.node_index_j,
                source_distance_field=field_name,
                from_lane=from_lane,
                to_lane=to_lane,
                via_lanes=via_lanes,
                path_lane_ids=path_lane_ids,
                path_uses_internal_lanes=uses_internal
                or bool(path_lane_ids and ":" in path_lane_ids),
                raw_weight=raw,
            )
        )

    for src in range(n_nodes):
        for dst in range(n_nodes):
            if src == dst or topology[src, dst] == 1:
                continue
            if math.isfinite(distance[src, dst]):
                nonedge_distance_computation_count += 1

    extra_connections = [
        (src, dst)
        for (src, dst) in connections
        if src != dst and topology[src, dst] != 1
    ]
    if extra_connections:
        raise PrepareError(
            "r_sumo_connections.csv has directed pairs absent from A_dir: "
            + ", ".join(f"{a}->{b}" for a, b in extra_connections[:EXAMPLE_LIMIT])
        )

    for edge in directed_edges:
        final = edge.raw_weight
        retained = True
        if epsilon > 0.0 and final < epsilon:
            final = 0.0
            retained = False
        edge.final_weight = float(final)
        edge.retained = retained
        if retained:
            weights64[edge.src, edge.dst] = edge.final_weight
    np.fill_diagonal(weights64, 0.0)

    weights32 = weights64.astype(np.float32)
    positive64 = weights64 > 0.0
    positive32 = weights32 > 0.0
    if not np.array_equal(positive64, positive32):
        lost = int(np.count_nonzero(positive64 & ~positive32))
        raise PrepareError(
            f"float32 conversion dropped {lost} mathematically nonzero directed edges"
        )

    directed_edges.sort(key=lambda item: (item.src, item.dst))
    stats = {
        "source_self_loops": source_self_loops,
        "nonedge_distance_computation_count": nonedge_distance_computation_count,
        "removed_by_threshold_count": int(sum(1 for edge in directed_edges if not edge.retained)),
        "retained_directed_edge_count": int(sum(1 for edge in directed_edges if edge.retained)),
        "directed_edge_count": int(topology.sum()),
    }
    return topology, distance, weights64, weights32, directed_edges, stats


def component_count(matrix: np.ndarray, connection: str) -> int:
    count, _labels = connected_components(
        csgraph=sparse.csr_matrix((matrix > 0).astype(np.uint8)),
        directed=connection != "weak",
        connection=connection,
    )
    return int(count)


def chain_direction_test(original: OriginalDCRNNApi) -> dict[str, Any]:
    adj = np.array(
        [
            [0.0, 0.8, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    forward, reverse = build_original_supports(adj, "dual_random_walk", original)
    x0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x1 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x2 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward_from_0 = forward @ x0
    forward_from_1 = forward @ x1
    reverse_from_2 = reverse @ x2
    reverse_from_1 = reverse @ x1
    checks = {
        "a_0_1_is_one": bool(adj[0, 1] > 0.0 and adj[1, 0] == 0.0),
        "forward_moves_0_to_1": bool(np.allclose(forward_from_0, [0.0, 1.0, 0.0], atol=1e-12)),
        "forward_moves_1_to_2": bool(np.allclose(forward_from_1, [0.0, 0.0, 1.0], atol=1e-12)),
        "reverse_moves_2_to_1": bool(np.allclose(reverse_from_2, [0.0, 1.0, 0.0], atol=1e-12)),
        "reverse_moves_1_to_0": bool(np.allclose(reverse_from_1, [1.0, 0.0, 0.0], atol=1e-12)),
        "forward_does_not_move_2_to_1": bool(np.allclose(forward @ x2, [0.0, 0.0, 0.0], atol=1e-12)),
        "transpose_is_storage_layout_not_semantic_flip": True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise PrepareError(f"directed chain support test failed: {failed}")
    return {
        "status": "ok",
        "description": "3-node chain 0->1->2 with A[i,j]=i->j",
        "original_forward_definition": ORIGINAL_RW_FORWARD,
        "original_reverse_definition": ORIGINAL_RW_REVERSE,
        "checks": checks,
        "note": (
            "DCGRUCell stores (D^{-1} A).T so sparse_tensor_dense_matmul(support, x) "
            "aggregates from predecessors. The extra transpose is a storage layout "
            "requirement, not a reversal of A[i,j]=i->j."
        ),
    }


def run_synthetic_checks(original: OriginalDCRNNApi) -> None:
    mapping = NodeMapping(
        node_index=[0, 1, 2],
        edge_ids=["A", "B", "C"],
        index_by_edge={"A": 0, "B": 1, "C": 2},
        rows=[],
    )
    directed = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.uint8)
    undirected = np.logical_or(directed, directed.T).astype(np.uint8)
    np.fill_diagonal(undirected, 0)
    csv_rows = {
        (0, 1): WeightedEdgeRow(
            0, 1, "A", "B", True, False, 1, 0, 10.0, None, 10.0, "i_to_j",
            "A_0", "B_0", ":J_0", "A_0|:J_0|B_0", "1",
        ),
        (0, 2): WeightedEdgeRow(
            0, 2, "A", "C", False, True, 0, 1, None, 30.0, 30.0, "j_to_i",
            "C_0", "A_0", ":K_0", "C_0|:K_0|A_0", "1",
        ),
        (1, 2): WeightedEdgeRow(
            1, 2, "B", "C", True, False, 1, 0, 20.0, None, 20.0, "i_to_j",
            "B_0", "C_0", "", "B_0|C_0", "0",
        ),
    }
    connections = {
        (0, 1): [{"from": "A", "to": "B", "fromLane": "0", "toLane": "0", "via": ":J_0"}],
        (1, 2): [{"from": "B", "to": "C", "fromLane": "0", "toLane": "0", "via": ""}],
        (2, 0): [{"from": "C", "to": "A", "fromLane": "0", "toLane": "0", "via": ":K_0"}],
    }
    sigma = 10.0
    topology, distance, _w64, weights32, edges, stats = build_directed_graph(
        mapping, directed, csv_rows, connections, undirected, sigma, 0.0
    )
    if not np.array_equal(topology, directed):
        raise PrepareError("synthetic topology was altered")
    if abs(distance[0, 1] - 10.0) > 1e-12 or abs(distance[1, 2] - 20.0) > 1e-12:
        raise PrepareError("synthetic directional distances were not kept")
    if math.isfinite(distance[1, 0]) or math.isfinite(distance[2, 1]):
        raise PrepareError("synthetic missing reverse distances must stay inf")
    if abs(distance[2, 0] - 30.0) > 1e-12:
        raise PrepareError("synthetic reverse-direction field mapping failed")
    expected_01 = gaussian_weight(10.0, sigma)
    if abs(float(weights32[0, 1]) - expected_01) > 1e-6:
        raise PrepareError("synthetic Gaussian weight mismatch")
    if float(weights32[1, 0]) != 0.0:
        raise PrepareError("synthetic one-way reverse weight must be 0")
    if stats["nonedge_distance_computation_count"] != 0:
        raise PrepareError("synthetic computed a non-neighbour distance")
    if len(edges) != 3:
        raise PrepareError("synthetic directed edge count mismatch")
    both = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    csv_both = {
        (0, 1): WeightedEdgeRow(
            0, 1, "A", "B", True, True, 1, 1, 4.0, 9.0, 4.0, "i_to_j",
            "", "", "", "", "0",
        )
    }
    mapping2 = NodeMapping([0, 1], ["A", "B"], {"A": 0, "B": 1}, [])
    connections_both = {
        (0, 1): [{"from": "A", "to": "B", "fromLane": "0", "toLane": "0", "via": ""}],
        (1, 0): [{"from": "B", "to": "A", "fromLane": "0", "toLane": "0", "via": ""}],
    }
    _t, dist_both, _w64b, w_both, _edges, _stats = build_directed_graph(
        mapping2,
        both,
        csv_both,
        connections_both,
        np.logical_or(both, both.T).astype(np.uint8),
        sigma,
        0.0,
    )
    if abs(dist_both[0, 1] - 4.0) > 1e-12 or abs(dist_both[1, 0] - 9.0) > 1e-12:
        raise PrepareError("synthetic bidirectional distances were collapsed")
    if abs(float(w_both[0, 1]) - float(w_both[1, 0])) < 1e-12:
        raise PrepareError("synthetic bidirectional weights should differ")
    if abs(dist_both[0, 1] - 4.0) == 0 and csv_both[(0, 1)].selected_distance == 4.0:
        if abs(dist_both[1, 0] - csv_both[(0, 1)].selected_distance) < 1e-12:
            raise PrepareError("synthetic reverse used selected_distance")
    chain_direction_test(original)


def edge_to_csv_row(edge: DirectedEdge, sigma: float, epsilon: float) -> dict[str, object]:
    return {
        "from_node_index": edge.src,
        "to_node_index": edge.dst,
        "from_edge_id": edge.from_edge_id,
        "to_edge_id": edge.to_edge_id,
        "direct_connection_count": edge.connection_count,
        "direction_distance": format_float(edge.distance),
        "sigma": format_float(sigma),
        "raw_weight": format_float(edge.raw_weight),
        "epsilon": format_float(epsilon),
        "final_weight": format_float(edge.final_weight),
        "retained_after_threshold": int(edge.retained),
        "source_pair_node_index_i": edge.source_pair_i,
        "source_pair_node_index_j": edge.source_pair_j,
        "source_distance_field": edge.source_distance_field,
        "from_lane": edge.from_lane,
        "to_lane": edge.to_lane,
        "via_lanes": edge.via_lanes,
        "path_lane_ids": edge.path_lane_ids,
        "path_uses_internal_lanes": int(edge.path_uses_internal_lanes),
    }


def validate_outputs(
    *,
    mapping: NodeMapping,
    stgcn_mapping: NodeMapping,
    directed_source: np.ndarray,
    topology: np.ndarray,
    distance: np.ndarray,
    weights64: np.ndarray,
    weights32: np.ndarray,
    sparse_weights: sparse.spmatrix,
    directed_edges: list[DirectedEdge],
    csv_rows_out: list[dict[str, object]],
    sigma: float,
    sigma_source: str,
    epsilon: float,
    stats: Mapping[str, int],
    hashes_before: Mapping[str, str],
    hashes_after: Mapping[str, str],
    original: OriginalDCRNNApi,
    pickle_payload: list[object],
    pickle_path: Path | None,
    pickle_protocol: int,
) -> dict[str, Any]:
    failures: list[str] = []
    n_nodes = len(mapping.edge_ids)
    if mapping.edge_ids != stgcn_mapping.edge_ids:
        failures.append("r_nodes.csv node order does not match STGCN node_mapping.csv")
    if topology.shape != (n_nodes, n_nodes) or topology.dtype != np.uint8:
        failures.append(f"topology dtype/shape invalid: {topology.shape} {topology.dtype}")
    expected_topology = directed_source.copy()
    np.fill_diagonal(expected_topology, 0)
    if not np.array_equal(topology, expected_topology):
        failures.append("output topology != A_dir with zero diagonal")
    if int(np.diag(topology).sum()) != 0:
        failures.append("output topology has self-loops")
    if np.array_equal(topology, topology.T) and int(topology.sum()) > 0:
        # A directed graph may happen to be symmetric; only fail if we symmetrized
        # relative to A_dir. Equality with expected_topology already covers that.
        pass
    extra = int(np.logical_and(topology == 1, expected_topology == 0).sum())
    missing = int(np.logical_and(topology == 0, expected_topology == 1).sum())
    if extra:
        failures.append(f"topology added {extra} edges")
    if missing:
        failures.append(f"topology lost {missing} edges")

    if distance.shape != (n_nodes, n_nodes) or distance.dtype != np.float64:
        failures.append("distance matrix shape/dtype invalid")
    if not np.allclose(np.diag(distance), 0.0, atol=DISTANCE_ATOL):
        failures.append("distance diagonal is not 0")
    if weights32.shape != (n_nodes, n_nodes):
        failures.append("weight matrix shape invalid")
    if not np.allclose(np.diag(weights32), 0.0, atol=FLOAT32_WEIGHT_ATOL):
        failures.append("weight diagonal is not 0")
    if not np.isfinite(weights32).all() or not np.isfinite(weights64).all():
        failures.append("weights contain NaN or inf")
    if np.any(weights64 < -WEIGHT_ATOL) or np.any(weights64 > 1.0 + WEIGHT_ATOL):
        failures.append("weights outside [0, 1]")

    for src in range(n_nodes):
        for dst in range(n_nodes):
            if src == dst:
                continue
            is_edge = topology[src, dst] == 1
            finite = bool(np.isfinite(distance[src, dst]))
            if is_edge and (not finite or distance[src, dst] <= 0.0):
                failures.append(f"edge {src}->{dst} has invalid distance")
                break
            if not is_edge and finite:
                failures.append(f"non-edge {src}->{dst} has a finite distance")
                break
        else:
            continue
        break

    retained_positions = {(edge.src, edge.dst) for edge in directed_edges if edge.retained}
    weight_positions = {
        (int(i), int(j)) for i, j in zip(*np.where(weights32 > 0.0))
    }
    topology_positions = {
        (int(i), int(j)) for i, j in zip(*np.where(topology == 1))
    }
    if epsilon == 0.0 and retained_positions != topology_positions:
        failures.append("epsilon=0 but retained edges != A_dir")
    if weight_positions != retained_positions:
        failures.append("W_dir>0 positions do not match retained directed edges")
    if stats["nonedge_distance_computation_count"] != 0:
        failures.append("non-neighbour distances were computed")
    if epsilon == 0.0 and stats["removed_by_threshold_count"] != 0:
        failures.append("epsilon=0 but edges were removed by threshold")
    if epsilon == 0.0 and stats["retained_directed_edge_count"] != stats["directed_edge_count"]:
        failures.append("epsilon=0 but retained_directed_edge_count != directed_edge_count")

    for edge in directed_edges:
        expected = gaussian_weight(edge.distance, sigma)
        if abs(edge.raw_weight - expected) > WEIGHT_ATOL:
            failures.append(
                f"raw weight mismatch for {edge.src}->{edge.dst}"
            )
            break
        if edge.retained and abs(float(weights64[edge.src, edge.dst]) - edge.final_weight) > WEIGHT_ATOL:
            failures.append(f"dense weight mismatch for {edge.src}->{edge.dst}")
            break
        if edge.source_distance_field not in {"distance_i_to_j", "distance_j_to_i"}:
            failures.append("source_distance_field is not a directional CSV field")
            break
        if edge.source_distance_field == "distance_i_to_j" and not (
            edge.source_pair_i == edge.src and edge.source_pair_j == edge.dst
        ):
            failures.append("distance_i_to_j mapping does not match (src,dst)")
            break
        if edge.source_distance_field == "distance_j_to_i" and not (
            edge.source_pair_i == edge.dst and edge.source_pair_j == edge.src
        ):
            failures.append("distance_j_to_i mapping does not match (src,dst)")
            break

    reconstructed = np.zeros_like(weights32)
    for row in csv_rows_out:
        src = int(row["from_node_index"])
        dst = int(row["to_node_index"])
        retained = int(row["retained_after_threshold"]) == 1
        weight = float(row["final_weight"]) if row["final_weight"] != "" else 0.0
        if retained:
            reconstructed[src, dst] = np.float32(weight)
    if not np.allclose(reconstructed, weights32, atol=FLOAT32_WEIGHT_ATOL):
        failures.append("CSV rows do not reconstruct the dense weight matrix")

    sparse_dense = np.asarray(sparse_weights.todense(), dtype=np.float32)
    if not np.array_equal(sparse_dense, weights32):
        if not np.allclose(sparse_dense, weights32, atol=0.0):
            failures.append("sparse and dense weighted adjacencies differ")

    if np.allclose(weights32, weights32.T, atol=FLOAT32_WEIGHT_ATOL) and int(weights32.sum()) > 0:
        # Allowed only if A_dir itself is symmetric. Flag as a property, not a failure.
        adjacency_is_symmetric = bool(np.array_equal(topology, topology.T))
    else:
        adjacency_is_symmetric = False
    if np.array_equal(topology, topology.T) is False and np.allclose(
        weights32, weights32.T, atol=FLOAT32_WEIGHT_ATOL
    ):
        failures.append("directed topology is asymmetric but W_dir was symmetrized")

    one_way = (topology == 1) & (topology.T == 0)
    if np.any(one_way) and np.any(weights32.T[one_way] > 0):
        failures.append("one-way edges received reverse weights")

    both_ways = (topology == 1) & (topology.T == 1)
    both_src, both_dst = np.where(np.triu(both_ways, k=1))
    for src, dst in zip(both_src.tolist(), both_dst.tolist()):
        if not (np.isfinite(distance[src, dst]) and np.isfinite(distance[dst, src])):
            failures.append(f"bidirectional pair {src}<->{dst} missing a direction distance")
            break
        if abs(distance[src, dst] - distance[dst, src]) > DISTANCE_ATOL:
            if abs(float(weights64[src, dst]) - float(weights64[dst, src])) <= WEIGHT_ATOL:
                failures.append(
                    f"bidirectional pair {src}<->{dst} has different distances but identical weights"
                )
                break

    sensor_ids, sensor_id_to_ind, adj_mx = pickle_payload
    if list(sensor_ids) != mapping.edge_ids:
        failures.append("pickle sensor_ids do not follow r_nodes.csv order")
    if any(not isinstance(item, str) for item in sensor_ids):
        failures.append("pickle sensor_ids are not all strings")
    expected_map = {edge_id: i for i, edge_id in enumerate(mapping.edge_ids)}
    if dict(sensor_id_to_ind) != expected_map:
        failures.append("pickle sensor_id_to_ind does not match node_index")
    adj_loaded = np.asarray(adj_mx)
    if not np.array_equal(adj_loaded.astype(np.float32), weights32):
        if not np.allclose(adj_loaded, weights32, atol=0.0):
            failures.append("pickle adj_mx does not match dcrnn_weighted_adjacency.npy")

    if pickle_path is not None:
        loaded_ids, loaded_map, loaded_adj = original.utils.load_graph_data(str(pickle_path))
    else:
        fd, tmp_pickle = tempfile.mkstemp(prefix=".dcrnn_adj_mx.", suffix=".pkl")
        os.close(fd)
        try:
            with open(tmp_pickle, "wb") as handle:
                pickle.dump(pickle_payload, handle, protocol=pickle_protocol)
            loaded_ids, loaded_map, loaded_adj = original.utils.load_graph_data(tmp_pickle)
        finally:
            try:
                os.unlink(tmp_pickle)
            except OSError:
                pass
    if list(loaded_ids) != mapping.edge_ids:
        failures.append("original load_graph_data() sensor_ids mismatch")
    if dict(loaded_map) != expected_map:
        failures.append("original load_graph_data() mapping mismatch")
    if not np.array_equal(np.asarray(loaded_adj).astype(np.float32), weights32):
        if not np.allclose(np.asarray(loaded_adj), weights32, atol=0.0):
            failures.append("original load_graph_data() adj_mx mismatch")

    supports = {}
    forward_reverse_identical = None
    try:
        chain = chain_direction_test(original)
        for filter_type in ORIGINAL_FILTER_TYPES:
            if filter_type == "laplacian":
                continue
            built = build_original_supports(weights32, filter_type, original)
            supports[filter_type] = [matrix.shape for matrix in built]
            if filter_type == "dual_random_walk":
                forward_reverse_identical = bool(
                    np.allclose(built[0], built[1], atol=1e-12)
                )
                zero_out = np.where(topology.sum(axis=1) == 0)[0]
                if zero_out.size and not np.allclose(built[0][:, zero_out], 0.0, atol=1e-12):
                    # P.T columns of zero-out-degree nodes should be 0 (no outgoing walk)
                    pass
        laplacian_ok = True
        try:
            build_original_supports(weights32, "laplacian", original)
        except Exception as exc:
            laplacian_ok = False
            failures.append(f"original laplacian support construction failed: {exc}")
        del laplacian_ok
    except PrepareError as exc:
        chain = {"status": "failed", "error": str(exc)}
        failures.append(str(exc))

    if hashes_before != hashes_after:
        failures.append("protected input files changed during the run")

    out_degree = topology.sum(axis=1)
    in_degree = topology.sum(axis=0)
    finite_distances = [edge.distance for edge in directed_edges]
    positive_weights = [edge.final_weight for edge in directed_edges if edge.retained]
    single_direction = int(np.logical_and(topology == 1, topology.T == 0).sum())
    bidirectional_directed = int(np.logical_and(topology == 1, topology.T == 1).sum())
    summary = {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "node_count": n_nodes,
        "possible_directed_pair_count": n_nodes * (n_nodes - 1),
        "directed_edge_count": int(topology.sum()),
        "single_direction_undirected_pair_count": single_direction,
        "bidirectional_undirected_pair_count": bidirectional_directed // 2,
        "self_connection_count_in_source": int(stats["source_self_loops"]),
        "retained_directed_edge_count": int(stats["retained_directed_edge_count"]),
        "removed_by_threshold_count": int(stats["removed_by_threshold_count"]),
        "nonedge_distance_computation_count": int(stats["nonedge_distance_computation_count"]),
        "weakly_connected_component_count": component_count(topology, "weak"),
        "strongly_connected_component_count": component_count(topology, "strong"),
        "zero_in_degree_node_count": int(np.count_nonzero(in_degree == 0)),
        "zero_out_degree_node_count": int(np.count_nonzero(out_degree == 0)),
        "minimum_positive_distance": min(finite_distances) if finite_distances else None,
        "median_positive_distance": float(np.median(finite_distances)) if finite_distances else None,
        "maximum_positive_distance": max(finite_distances) if finite_distances else None,
        "minimum_positive_weight": min(positive_weights) if positive_weights else None,
        "median_positive_weight": float(np.median(positive_weights)) if positive_weights else None,
        "maximum_positive_weight": max(positive_weights) if positive_weights else None,
        "sigma": sigma,
        "sigma_source": sigma_source,
        "epsilon": epsilon,
        "adjacency_is_symmetric": bool(adjacency_is_symmetric),
        "forward_reverse_supports_identical": forward_reverse_identical,
        "pickle_protocol": pickle_protocol,
        "original_load_graph_data_ok": not failures,
        "original_support_shapes": supports,
        "directed_chain_test": chain if isinstance(chain, dict) else {"status": "ok"},
        "inputs_unchanged": hashes_before == hashes_after,
        "did_not_read_flow_or_trajectories": True,
        "did_not_instantiate_dcrnn_model": True,
    }
    if failures:
        raise PrepareError("DCRNN directed adjacency validation failed:\n" + "\n".join(failures))
    return summary


def dependency_versions() -> dict[str, str]:
    tf_mod = sys.modules.get("tensorflow")
    tf_file = getattr(tf_mod, "__file__", None) if tf_mod is not None else None
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "tensorflow": "real" if tf_file else "stubbed-for-original-utils-import",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a directed DCRNN adjacency from R-only topology and "
            "STGCN directional center distances. Does not overwrite "
            "r_adjacency.npy or STGCN graph files."
        )
    )
    parser.add_argument("--r-nodes", type=Path, default=DEFAULT_R_NODES)
    parser.add_argument("--directed-adjacency", type=Path, default=DEFAULT_DIRECTED_ADJACENCY)
    parser.add_argument("--sumo-connections", type=Path, default=DEFAULT_SUMO_CONNECTIONS)
    parser.add_argument("--r-edges", type=Path, default=DEFAULT_R_EDGES)
    parser.add_argument("--r-graph-validation", type=Path, default=DEFAULT_R_GRAPH_VALIDATION)
    parser.add_argument("--weighted-edges", type=Path, default=DEFAULT_WEIGHTED_EDGES)
    parser.add_argument("--source-graph-metadata", type=Path, default=DEFAULT_SOURCE_METADATA)
    parser.add_argument(
        "--source-graph-validation", type=Path, default=DEFAULT_SOURCE_VALIDATION
    )
    parser.add_argument(
        "--source-undirected-topology", type=Path, default=DEFAULT_SOURCE_UNDIRECTED
    )
    parser.add_argument(
        "--stgcn-node-mapping", type=Path, default=DEFAULT_STGCN_NODE_MAPPING
    )
    parser.add_argument(
        "--original-dcrnn-root", type=Path, default=DEFAULT_ORIGINAL_DCRNN_ROOT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--pickle-protocol", type=int, default=DEFAULT_PICKLE_PROTOCOL)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED_DEFAULT)
    return parser.parse_args()


def collect_protected(paths: Iterable[Path]) -> tuple[Path, ...]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return tuple(unique)


def main() -> None:
    args = parse_args()
    if args.epsilon < 0.0 or not math.isfinite(args.epsilon):
        raise PrepareError(f"--epsilon must be finite and >= 0, got {args.epsilon}")
    if args.pickle_protocol < 2:
        raise PrepareError("--pickle-protocol must be >= 2 to match original DCRNN")

    original = import_original_dcrnn_utils(args.original_dcrnn_root)
    if not args.skip_synthetic:
        run_synthetic_checks(original)

    r_nodes_path = args.r_nodes.expanduser().resolve()
    adj_path = args.directed_adjacency.expanduser().resolve()
    sumo_path = args.sumo_connections.expanduser().resolve()
    r_edges_path = args.r_edges.expanduser().resolve()
    r_val_path = args.r_graph_validation.expanduser().resolve()
    edges_path = args.weighted_edges.expanduser().resolve()
    meta_path = args.source_graph_metadata.expanduser().resolve()
    stgcn_val_path = args.source_graph_validation.expanduser().resolve()
    undirected_path = args.source_undirected_topology.expanduser().resolve()
    mapping_path = args.stgcn_node_mapping.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    protected = collect_protected(
        [
            r_nodes_path,
            adj_path,
            sumo_path,
            r_edges_path,
            r_val_path,
            edges_path,
            meta_path,
            stgcn_val_path,
            undirected_path,
            mapping_path,
            original.load_graph_data_source,
            original.cell_source,
            original.gen_adj_source,
            STGCN_ADJ_DIR / "stgcn_weighted_adjacency.npy",
            STGCN_ADJ_DIR / "stgcn_center_distance.npy",
        ]
    )
    for path in protected:
        if not path.is_file():
            raise PrepareError(f"missing input: {path}")
        if path == output_dir or output_dir in path.parents:
            raise PrepareError(f"refusing to use protected input inside output-dir: {path}")

    forbidden_dirs = {
        (ANALYSIS_DIR / "graph" / "r_graph").resolve(),
        STGCN_ADJ_DIR.resolve(),
        original.root.resolve(),
    }
    if output_dir in forbidden_dirs:
        raise PrepareError(f"output-dir collides with a protected graph directory: {output_dir}")

    existing = [output_dir / name for name in OUTPUT_ARTIFACTS if (output_dir / name).is_file()]
    if existing and not args.overwrite:
        raise PrepareError(
            "DCRNN graph outputs already exist; pass --overwrite to replace only those files. "
            + ", ".join(path.name for path in existing)
        )

    resolved = {
        "project_root": repo_rel(PROJECT_ROOT),
        "script": repo_rel(SCRIPT_PATH),
        "r_nodes": repo_rel(r_nodes_path),
        "directed_adjacency": repo_rel(adj_path),
        "sumo_connections": repo_rel(sumo_path),
        "weighted_edges": repo_rel(edges_path),
        "source_graph_metadata": repo_rel(meta_path),
        "source_graph_validation": repo_rel(stgcn_val_path),
        "output_dir": repo_rel(output_dir),
        "original_dcrnn_root": repo_rel(original.root),
        "sigma": args.sigma,
        "epsilon": float(args.epsilon),
        "pickle_protocol": int(args.pickle_protocol),
        "overwrite": bool(args.overwrite),
        "filter_type_official_default": OFFICIAL_DEFAULT_FILTER_TYPE,
        "filter_type_code_default": CODE_DEFAULT_FILTER_TYPE,
    }
    print("resolved_config " + json.dumps(resolved, ensure_ascii=False, sort_keys=True), flush=True)

    assert_validation_ok(r_val_path, "R graph")
    assert_validation_ok(stgcn_val_path, "STGCN graph")
    metadata_in = load_json(meta_path)
    mapping = load_node_mapping(r_nodes_path)
    stgcn_mapping = load_node_mapping(mapping_path)
    if mapping.edge_ids != stgcn_mapping.edge_ids:
        raise PrepareError("r_nodes.csv and STGCN node_mapping.csv node order differ")
    directed = load_directed_adjacency(adj_path, len(mapping.edge_ids))
    undirected = load_undirected_topology(undirected_path, len(mapping.edge_ids))
    csv_rows = load_weighted_edge_rows(edges_path)
    connections = load_sumo_connections(sumo_path, mapping.index_by_edge)
    sigma, sigma_source = resolve_sigma(args.sigma, metadata_in, meta_path)
    hashes_before = {posix(path): sha256_file(path) for path in protected}

    topology, distance, weights64, weights32, directed_edges, stats = build_directed_graph(
        mapping,
        directed,
        csv_rows,
        connections,
        undirected,
        sigma,
        float(args.epsilon),
    )
    sparse_weights = sparse.csr_matrix(weights32, dtype=np.float32)
    csv_out = [
        edge_to_csv_row(edge, sigma, float(args.epsilon)) for edge in directed_edges
    ]
    sensor_ids = list(mapping.edge_ids)
    sensor_id_to_ind = {edge_id: int(i) for i, edge_id in enumerate(sensor_ids)}
    pickle_payload = [sensor_ids, sensor_id_to_ind, weights32]
    hashes_after = {posix(path): sha256_file(path) for path in protected}

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for name in OUTPUT_ARTIFACTS:
            path = output_dir / name
            if path.is_file():
                if path.resolve() in set(protected):
                    raise PrepareError(f"--overwrite would replace protected input {path}")
                path.unlink()

    pickle_path = output_dir / "dcrnn_adj_mx.pkl"
    validation = validate_outputs(
        mapping=mapping,
        stgcn_mapping=stgcn_mapping,
        directed_source=directed,
        topology=topology,
        distance=distance,
        weights64=weights64,
        weights32=weights32,
        sparse_weights=sparse_weights,
        directed_edges=directed_edges,
        csv_rows_out=csv_out,
        sigma=sigma,
        sigma_source=sigma_source,
        epsilon=float(args.epsilon),
        stats=stats,
        hashes_before=hashes_before,
        hashes_after=hashes_after,
        original=original,
        pickle_payload=pickle_payload,
        pickle_path=None,
        pickle_protocol=int(args.pickle_protocol),
    )

    atomic_write_npy(output_dir / "dcrnn_directed_topology.npy", topology)
    atomic_write_npy(output_dir / "dcrnn_directed_center_distance.npy", distance)
    atomic_write_npy(output_dir / "dcrnn_weighted_adjacency.npy", weights32)
    atomic_write_sparse_npz(
        output_dir / "dcrnn_weighted_adjacency_sparse.npz", sparse_weights
    )
    atomic_write_pickle(pickle_path, pickle_payload, int(args.pickle_protocol))
    atomic_write_csv(output_dir / "dcrnn_directed_edges.csv", EDGE_CSV_FIELDS, csv_out)

    loaded_ids, loaded_map, loaded_adj = original.utils.load_graph_data(str(pickle_path))
    if list(loaded_ids) != sensor_ids or dict(loaded_map) != sensor_id_to_ind:
        raise PrepareError("original load_graph_data() failed after writing pickle")
    if not np.array_equal(np.asarray(loaded_adj).astype(np.float32), weights32):
        raise PrepareError("original load_graph_data() adj_mx changed after pickle write")
    validation["original_load_graph_data_ok"] = True
    validation["original_load_unmodified_function"] = True
    validation["pickle_edge_id_python_type"] = "str"
    validation["pickle_adj_mx_numpy_dtype"] = str(weights32.dtype)
    hashes_final = {posix(path): sha256_file(path) for path in protected}
    if hashes_final != hashes_before:
        raise PrepareError("protected input files changed while writing outputs")
    validation["inputs_unchanged"] = True

    output_hashes = {
        name: file_fingerprint(output_dir / name)
        for name in OUTPUT_ARTIFACTS
        if name.endswith((".npy", ".npz", ".csv", ".pkl"))
    }
    metadata = {
        "script_path": posix(SCRIPT_PATH),
        "script_version": SCRIPT_VERSION,
        "generated_at": now_utc(),
        "project_root": posix(PROJECT_ROOT),
        "resolved_config": resolved,
        "original_dcrnn": {
            "root": posix(original.root),
            "source": "Yaguang Li et al., ICLR 2018, TensorFlow 1.x (reference/dcrnn)",
            "github": "https://github.com/liyaguang/DCRNN",
            "load_graph_data": posix(original.load_graph_data_source),
            "load_graph_data_sha256": hashes_final[posix(original.load_graph_data_source)],
            "dcrnn_cell": posix(original.cell_source),
            "dcrnn_cell_sha256": hashes_final[posix(original.cell_source)],
            "gen_adj_mx": posix(original.gen_adj_source),
            "gen_adj_mx_sha256": hashes_final[posix(original.gen_adj_source)],
            "graph_file_format": (
                "pickle protocol "
                f"{int(args.pickle_protocol)} list "
                "[sensor_ids: list[str], sensor_id_to_ind: dict[str,int], "
                "adj_mx: numpy.ndarray float32 shape (N,N)]"
            ),
            "code_default_filter_type": CODE_DEFAULT_FILTER_TYPE,
            "official_yaml_filter_type": OFFICIAL_DEFAULT_FILTER_TYPE,
            "supported_filter_types": list(ORIGINAL_FILTER_TYPES),
            "saves_random_walk_supports_in_adj_mx": False,
            "adds_identity_to_base_adjacency": False,
            "applies_gaussian_or_threshold_at_load_time": False,
            "forward_random_walk": ORIGINAL_RW_FORWARD,
            "reverse_random_walk": ORIGINAL_RW_REVERSE,
        },
        "node_count": len(mapping.edge_ids),
        "node_order_source": posix(r_nodes_path),
        "node_order_rule": "node_index 0..N-1 from r_nodes.csv; not lexicographic by edge_id",
        "r_nodes_sha256": hashes_final[posix(r_nodes_path)],
        "directed_adjacency_path": posix(adj_path),
        "directed_adjacency_sha256": hashes_final[posix(adj_path)],
        "r_sumo_connections_path": posix(sumo_path),
        "r_sumo_connections_sha256": hashes_final[posix(sumo_path)],
        "stgcn_weighted_edges_path": posix(edges_path),
        "stgcn_weighted_edges_sha256": hashes_final[posix(edges_path)],
        "stgcn_graph_metadata_path": posix(meta_path),
        "stgcn_graph_metadata_sha256": hashes_final[posix(meta_path)],
        "stgcn_graph_validation_path": posix(stgcn_val_path),
        "stgcn_graph_validation_sha256": hashes_final[posix(stgcn_val_path)],
        "r_graph_validation_path": posix(r_val_path),
        "r_graph_validation_sha256": hashes_final[posix(r_val_path)],
        "directed_topology_definition": DIRECTED_TOPOLOGY_DEFINITION,
        "direction_distance_field_rule": DISTANCE_FIELD_RULE,
        "directed_distance_definition": DIRECTED_DISTANCE_DEFINITION,
        "distance_unit": "meters",
        "gaussian_formula": GAUSSIAN_FORMULA,
        "sigma": sigma,
        "sigma_source": sigma_source,
        "sigma_is_not_sigma_squared": True,
        "epsilon": float(args.epsilon),
        "symmetrized": False,
        "artificial_self_loops_added": False,
        "nonneighbour_distances_computed": False,
        "sumo_routing_recomputed": False,
        "pems_distance_divisor_applied": False,
        "normalized_k_applied": False,
        "selected_distance_used": False,
        "directed_edge_count": int(stats["directed_edge_count"]),
        "single_direction_undirected_pair_count": validation["single_direction_undirected_pair_count"],
        "bidirectional_undirected_pair_count": validation["bidirectional_undirected_pair_count"],
        "weight_statistics": {
            "minimum_positive_weight": validation["minimum_positive_weight"],
            "median_positive_weight": validation["median_positive_weight"],
            "maximum_positive_weight": validation["maximum_positive_weight"],
        },
        "original_graph_load_compatible": True,
        "csv_float_format": FLOAT_TEXT_FORMAT,
        "output_files": output_hashes,
        "dependency_versions": dependency_versions(),
        "random_seed": args.random_seed,
        "did_not_modify": [posix(path) for path in protected],
        "did_not_read_flow_or_trajectories": True,
        "did_not_train_or_port_dcrnn": True,
    }
    atomic_write_json(output_dir / "dcrnn_graph_metadata.json", json_ready(metadata))
    output_hashes["dcrnn_graph_metadata.json"] = file_fingerprint(
        output_dir / "dcrnn_graph_metadata.json"
    )
    atomic_write_json(
        output_dir / "dcrnn_graph_validation.json",
        json_ready(validation),
    )
    print(
        f"done output={repo_rel(output_dir)} nodes={len(mapping.edge_ids)} "
        f"directed_edges={int(stats['directed_edge_count'])} "
        f"sigma={sigma:.6f} epsilon={args.epsilon} "
        f"asymmetric={not validation['adjacency_is_symmetric']} "
        f"validation={validation['status']}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except PrepareError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
