"""Build a symmetric, direct-connection-weighted adjacency for original STGCN.

Reads the R-only directed binary graph and the canonical SUMO net. Does not
overwrite ``r_adjacency.npy``, ``r_nodes.csv``, or ``net_tls.net.xml``. Does
not read trajectories, flows, or vehicle samples. Does not add self-loops,
degree-normalize, or form a Laplacian / Chebyshev kernel.

Original STGCN (reference/STGCN_IJCAI-18/utils/math_graph.py) mapping
-------------------------------------------------------------------
``weight_matrix(file, sigma2=0.1, epsilon=0.5, scaling=True)``:

* If the loaded matrix is already {0, 1}, ``scaling`` is forced False and the
  binary matrix is returned unchanged.
* Otherwise distances are divided by 10000 (PeMS meters → 10 km units), then
  ``W_ij = exp(-(d_ij/10000)^2 / sigma2)`` is kept only when that value is
  ``>= epsilon``, and the diagonal is zeroed with ``1 - I``.
* ``scaled_laplacian(W)`` and ``cheb_poly_approx(L, Ks=3, n)`` run later in
  ``main.py`` when the model graph kernel is built. ``first_approx`` uses
  ``W + I``. Those steps are **not** done here.

This script therefore:

* Uses metre distances with **no** ``/ 10000`` scaling (that constant is
  PeMS-specific and must not be copied).
* Uses ``--sigma`` as σ in ``exp(-d^2 / σ^2)``. That is **not** the original
  ``sigma2=0.1`` (which applies after the 10 km rescaling), and it is not
  ``σ^2`` itself.
* Uses ``--epsilon`` with the same *role* as the original threshold, but the
  default is ``0.0`` so every topological neighbour is kept. Pass a positive
  value only when reproducing a sparsity threshold.
* Writes a weighted ``W`` with zero diagonal. A later STGCN loader should
  feed this matrix in as ``W`` **without** running the distance kernel again
  (treat it like the ``scaling=False`` branch), then build Laplacian /
  Chebyshev at train time.

Direct-connection center distance
---------------------------------
The saved matrices are still shape ``[N_R, N_R]``. Distances are computed
only for undirected topological neighbours:

    i < j and A_sym[i, j] == 1

where ``A_sym = logical_or(A_dir, A_dir.T)`` with a zero diagonal. Non-
neighbours stay ``D = inf`` and ``W = 0``. Full-network shortest paths,
detours through other ordinary edges, and extra neighbours from global
reachability are forbidden.

For each neighbour pair, only directions with a real SUMO ``<connection>``
(``A_dir == 1``) are evaluated. A missing reverse direction is left empty /
unreachable; it is never filled by looping around the rest of ``net_tls``.

Each legal lane-level connection contributes:

    0.5 * source_lane_length
    + sum(internal via-lane lengths of that connection)
    + 0.5 * target_lane_length

The midpoint is 50% of the lane travel length along the lane (XML
``length``, or polyline arc length if the official length is missing). If
the polyline arc length disagrees with the official length, remaining
distances use the 50% arc-length position on the shape. Euclidean
shortcuts are never used. When several lane-level connections exist for
the same road-level direction, every candidate is evaluated and the
shortest finite distance is kept.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


SCRIPT_VERSION = "1.1.0"
RANDOM_SEED_DEFAULT = 42
SAMPLE_PAIR_COUNT = 5
EXAMPLE_LIMIT = 8
HASH_CHUNK = 8 * 1024 * 1024
LENGTH_MISMATCH_TOL_M = 1e-3
DISTANCE_ATOL = 1e-6
WEIGHT_ATOL = 1e-6
DEFAULT_EPSILON = 0.0
DEFAULT_SIGMA_MODE = "median"
SIGMA_MODES = ("median", "mean", "std")
DEFAULT_DISTANCE_MODE = "direct-connection-center"
FLOAT_TEXT_FORMAT = ".17g"
SKIP_EDGE_FUNCTIONS = frozenset({"walkingarea", "crossing"})
ZERO_DIAG = "A_sym[i, i] = 0; no artificial self-loop is added"
TIE_BREAK_RULE = (
    "If both direct directions exist and |d_i_to_j - d_j_to_i| <= "
    f"{DISTANCE_ATOL}, selected_direction = i_to_j. Within a direction, "
    "equal distances keep the lexicographically smaller "
    "(fromLane, toLane, via_lane_ids) tuple."
)

SCRIPT_PATH = Path(__file__).resolve()
PREPARED_DATA_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = PREPARED_DATA_DIR.parents[2]
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
DEFAULT_R_NODES = ANALYSIS_DIR / "graph" / "r_graph" / "r_nodes.csv"
DEFAULT_DIRECTED_ADJACENCY = ANALYSIS_DIR / "graph" / "r_graph" / "r_adjacency.npy"
DEFAULT_NET_FILE = ANALYSIS_DIR / "road_network" / "net_tls.net.xml"
DEFAULT_OUTPUT_DIR = PREPARED_DATA_DIR / "r-only" / "adjacency_matrix"

SYMMETRY_FORMULA = "A_sym = logical_or(A_dir, A_dir.T).astype(uint8); diagonal forced to 0"
CENTER_DEFINITION = (
    "Per-lane travel-distance midpoint: 50% of that lane's official length "
    "along the lane, or 50% of the polyline arc length when the shape and "
    "official length disagree; not the arithmetic mean of endpoints or "
    "coordinates, and not Euclidean distance"
)
DIRECTED_DISTANCE_DEFINITION = (
    "d_dir(i->j) is computed only when A_dir[i,j]==1. It is the minimum "
    "over legal lane-level SUMO <connection from=edge_i to=edge_j> records "
    "of (0.5 * source_lane_length) + sum(via internal lane lengths of that "
    "connection) + (0.5 * target_lane_length). Paths may contain only the "
    "source edge, that connection's internal lanes, and the target edge. "
    "Full-network Dijkstra and detours through other ordinary edges are "
    "not used."
)
SYMMETRIC_DISTANCE_DEFINITION = (
    "If only one direct direction exists, selected_distance is that "
    "direction's direct center distance. If both exist, selected_distance "
    "= min(d_i_to_j, d_j_to_i) with tie-break i_to_j. Missing reverse "
    "directions stay empty/inf and are never filled by a network loop."
)
GAUSSIAN_FORMULA = (
    "W_ij = exp(-(D_ij ** 2) / (sigma ** 2)) if A_sym[i,j]==1 else 0; "
    "then W_ij=0 if W_ij < epsilon; W_ii=0. sigma is the length-scale, "
    "not sigma**2."
)

OUTPUT_ARTIFACTS = (
    "stgcn_undirected_topology.npy",
    "stgcn_center_distance.npy",
    "stgcn_weighted_adjacency.npy",
    "stgcn_weighted_adjacency_sparse.npz",
    "stgcn_weighted_edges.csv",
    "stgcn_weighted_adjacency_metadata.json",
    "stgcn_weighted_adjacency_validation.json",
)

EDGE_CSV_FIELDS = (
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
    "selected_direction",
    "selected_from_lane",
    "selected_to_lane",
    "selected_via_lanes",
    "selected_path_edge_ids",
    "selected_path_lane_ids",
    "path_uses_internal_lanes",
    "sigma",
    "sigma_mode",
    "raw_weight",
    "epsilon",
    "final_weight",
    "retained_after_threshold",
)


class PrepareError(Exception):
    pass


@dataclass
class LaneRecord:
    lane_id: str
    index: str
    length: float
    shape: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class EdgeRecord:
    edge_id: str
    function: str
    shape: list[tuple[float, float]] = field(default_factory=list)
    lanes: list[LaneRecord] = field(default_factory=list)

    @property
    def is_internal(self) -> bool:
        return self.function == "internal" or self.edge_id.startswith(":")

    @property
    def skip(self) -> bool:
        return self.function in SKIP_EDGE_FUNCTIONS


@dataclass
class NodeMapping:
    node_index: list[int]
    edge_ids: list[str]
    index_by_edge: dict[str, int]
    rows: list[dict[str, str]]


@dataclass
class DirectedRoute:
    reachable: bool
    distance: float
    source_lane_id: str
    target_lane_id: str
    via_lane_ids: tuple[str, ...]
    path_lane_ids: tuple[str, ...]
    path_edge_ids: tuple[str, ...]
    uses_internal: bool
    candidate_count: int
    legal_candidate_distances: tuple[float, ...] = ()


@dataclass
class UndirectedPair:
    i: int
    j: int
    edge_i: str
    edge_j: str
    has_i_to_j: bool
    has_j_to_i: bool
    forward: DirectedRoute
    reverse: DirectedRoute
    selected_distance: float
    selected_direction: str
    selected_route: DirectedRoute
    raw_weight: float
    final_weight: float
    retained: bool


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
    writer_holder: list[str] = []

    class _Buf:
        def write(self, text: str) -> int:
            writer_holder.append(text)
            return len(text)

    handle = _Buf()
    writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in fieldnames})
    atomic_write_text(path, "".join(writer_holder))


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


def strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_shape(raw: str | None) -> list[tuple[float, float]]:
    if not raw:
        return []
    points: list[tuple[float, float]] = []
    for token in raw.split():
        if "," not in token:
            continue
        xs, ys = token.split(",", 1)
        try:
            points.append((float(xs), float(ys)))
        except ValueError:
            continue
    return points


def polyline_length(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def parse_lane_length(elem: ET.Element) -> float:
    raw = elem.get("length")
    if raw not in (None, ""):
        try:
            value = float(raw)
            if math.isfinite(value) and value >= 0.0:
                return value
        except ValueError:
            pass
    return polyline_length(parse_shape(elem.get("shape")))


def parse_edge_element(elem: ET.Element) -> EdgeRecord:
    lanes: list[LaneRecord] = []
    for child in elem:
        if strip_namespace(child.tag) != "lane":
            continue
        lane_id = child.get("id") or ""
        if not lane_id:
            continue
        lanes.append(
            LaneRecord(
                lane_id=lane_id,
                index=child.get("index") if child.get("index") is not None else str(len(lanes)),
                length=parse_lane_length(child),
                shape=parse_shape(child.get("shape")),
            )
        )
    return EdgeRecord(
        edge_id=elem.get("id") or "",
        function=elem.get("function") or "",
        shape=parse_shape(elem.get("shape")),
        lanes=lanes,
    )


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
            indices.append(int(index_text))
        except ValueError as exc:
            raise PrepareError(f"{path} has non-integer node_index {index_text!r}") from exc
        edge_ids.append(edge_id)
    expected = list(range(len(rows)))
    if indices != expected:
        raise PrepareError(
            f"{path} node_index must be 0..{len(rows) - 1} in order; "
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
    if not np.isfinite(loaded).all():
        raise PrepareError(f"{path} contains NaN or inf")
    unique = {float(value) for value in np.unique(loaded)}
    if not unique <= {0.0, 1.0}:
        raise PrepareError(f"{path} is not binary 0/1; unique={sorted(unique)}")
    return loaded.astype(np.uint8, copy=True)


def symmetrize_topology(directed: np.ndarray) -> np.ndarray:
    symmetric = np.logical_or(directed, directed.T).astype(np.uint8)
    np.fill_diagonal(symmetric, 0)
    return symmetric


def parse_sumo_net(net_file: Path) -> tuple[dict[str, EdgeRecord], list[dict[str, str]]]:
    if not net_file.is_file():
        raise PrepareError(f"SUMO net file not found: {net_file}")
    edges: dict[str, EdgeRecord] = {}
    connections: list[dict[str, str]] = []
    try:
        context = ET.iterparse(net_file, events=("end",))
        for _event, elem in context:
            tag = strip_namespace(elem.tag)
            if tag == "edge":
                record = parse_edge_element(elem)
                if record.edge_id and record.edge_id not in edges:
                    edges[record.edge_id] = record
                elem.clear()
            elif tag == "connection":
                from_edge = elem.get("from")
                to_edge = elem.get("to")
                if from_edge and to_edge:
                    connections.append({key: value for key, value in elem.attrib.items()})
                elem.clear()
            elif tag == "lane":
                continue
            else:
                elem.clear()
    except ET.ParseError as exc:
        raise PrepareError(f"failed to parse SUMO net XML: {net_file}") from exc
    return edges, connections


def lane_by_index(edge: EdgeRecord, index_text: str) -> LaneRecord | None:
    for lane in edge.lanes:
        if lane.index == index_text:
            return lane
    try:
        position = int(index_text)
    except (TypeError, ValueError):
        return None
    if 0 <= position < len(edge.lanes):
        return edge.lanes[position]
    return None


def build_lane_lookups(
    edges: dict[str, EdgeRecord],
) -> tuple[dict[str, LaneRecord], dict[str, str]]:
    lanes_by_id: dict[str, LaneRecord] = {}
    edge_of_lane: dict[str, str] = {}
    for edge in edges.values():
        if edge.skip:
            continue
        for lane in edge.lanes:
            if not lane.lane_id:
                continue
            lanes_by_id[lane.lane_id] = lane
            edge_of_lane[lane.lane_id] = edge.edge_id
    return lanes_by_id, edge_of_lane


def index_r_to_r_connections(
    connections: list[dict[str, str]],
    r_edge_ids: set[str],
) -> dict[tuple[str, str], list[dict[str, str]]]:
    index: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for attrib in connections:
        source_id = attrib.get("from", "")
        target_id = attrib.get("to", "")
        if source_id in r_edge_ids and target_id in r_edge_ids:
            index[(source_id, target_id)].append(attrib)
    return index


def lane_halves(lane: LaneRecord) -> tuple[float, float]:
    """Return (start_to_mid, mid_to_end) along the lane travel geometry."""
    shape_len = polyline_length(lane.shape)
    official_ok = math.isfinite(lane.length) and lane.length >= 0.0
    if official_ok and shape_len > DISTANCE_ATOL:
        if abs(float(lane.length) - shape_len) > LENGTH_MISMATCH_TOL_M:
            half = 0.5 * shape_len
            return half, half
        half = 0.5 * float(lane.length)
        return half, half
    if official_ok:
        half = 0.5 * float(lane.length)
        return half, half
    half = 0.5 * shape_len
    return half, half


def lane_travel_length(lane: LaneRecord) -> float:
    if math.isfinite(lane.length) and lane.length >= 0.0:
        return float(lane.length)
    return polyline_length(lane.shape)


def consecutive_edge_ids(
    path_lane_ids: tuple[str, ...],
    edge_of_lane: Mapping[str, str],
) -> tuple[str, ...]:
    edge_ids: list[str] = []
    for lane_id in path_lane_ids:
        edge_id = edge_of_lane.get(lane_id, "")
        if not edge_ids or edge_ids[-1] != edge_id:
            edge_ids.append(edge_id)
    return tuple(edge_ids)


def path_has_only_direct_connection_edges(
    path_edge_ids: tuple[str, ...],
    source_edge_id: str,
    target_edge_id: str,
    edges: Mapping[str, EdgeRecord],
) -> bool:
    if not path_edge_ids:
        return False
    if path_edge_ids[0] != source_edge_id or path_edge_ids[-1] != target_edge_id:
        return False
    for edge_id in path_edge_ids[1:-1]:
        record = edges.get(edge_id)
        if record is None or not record.is_internal:
            return False
    return True


def evaluate_direct_connection(
    attrib: dict[str, str],
    source_edge: EdgeRecord,
    target_edge: EdgeRecord,
    edges: Mapping[str, EdgeRecord],
    lanes_by_id: Mapping[str, LaneRecord],
    edge_of_lane: Mapping[str, str],
) -> DirectedRoute | None:
    if attrib.get("from", "") != source_edge.edge_id:
        return None
    if attrib.get("to", "") != target_edge.edge_id:
        return None
    source_lane = lane_by_index(source_edge, attrib.get("fromLane", "0"))
    target_lane = lane_by_index(target_edge, attrib.get("toLane", "0"))
    if source_lane is None or target_lane is None:
        return None
    via_ids = tuple(token for token in (attrib.get("via") or "").split() if token)
    via_lanes: list[LaneRecord] = []
    for via_id in via_ids:
        via_lane = lanes_by_id.get(via_id)
        if via_lane is None:
            return None
        via_edge_id = edge_of_lane.get(via_id, "")
        via_edge = edges.get(via_edge_id)
        if via_edge is None or not via_edge.is_internal:
            return None
        via_length = lane_travel_length(via_lane)
        if not math.isfinite(via_length) or via_length < 0.0:
            return None
        via_lanes.append(via_lane)
    start_to_mid_src, mid_to_end_src = lane_halves(source_lane)
    start_to_mid_tgt, _mid_to_end_tgt = lane_halves(target_lane)
    via_sum = sum(lane_travel_length(lane) for lane in via_lanes)
    distance = mid_to_end_src + via_sum + start_to_mid_tgt
    if not math.isfinite(distance) or distance <= DISTANCE_ATOL:
        return None
    path_lane_ids = (source_lane.lane_id, *via_ids, target_lane.lane_id)
    path_edge_ids = consecutive_edge_ids(path_lane_ids, edge_of_lane)
    if not path_has_only_direct_connection_edges(
        path_edge_ids, source_edge.edge_id, target_edge.edge_id, edges
    ):
        return None
    return DirectedRoute(
        reachable=True,
        distance=float(distance),
        source_lane_id=source_lane.lane_id,
        target_lane_id=target_lane.lane_id,
        via_lane_ids=via_ids,
        path_lane_ids=path_lane_ids,
        path_edge_ids=path_edge_ids,
        uses_internal=bool(via_ids),
        candidate_count=0,
        legal_candidate_distances=(),
    )


def unreachable_route(candidate_count: int = 0) -> DirectedRoute:
    return DirectedRoute(
        reachable=False,
        distance=math.inf,
        source_lane_id="",
        target_lane_id="",
        via_lane_ids=(),
        path_lane_ids=(),
        path_edge_ids=(),
        uses_internal=False,
        candidate_count=candidate_count,
        legal_candidate_distances=(),
    )


def candidate_sort_key(route: DirectedRoute) -> tuple[object, ...]:
    return (
        route.distance,
        route.source_lane_id,
        route.target_lane_id,
        route.via_lane_ids,
    )


def direct_center_distance(
    source_edge: EdgeRecord,
    target_edge: EdgeRecord,
    connection_index: Mapping[tuple[str, str], list[dict[str, str]]],
    edges: Mapping[str, EdgeRecord],
    lanes_by_id: Mapping[str, LaneRecord],
    edge_of_lane: Mapping[str, str],
) -> DirectedRoute:
    attribs = connection_index.get((source_edge.edge_id, target_edge.edge_id), [])
    candidate_count = len(attribs)
    legal: list[DirectedRoute] = []
    for attrib in attribs:
        candidate = evaluate_direct_connection(
            attrib, source_edge, target_edge, edges, lanes_by_id, edge_of_lane
        )
        if candidate is None:
            continue
        legal.append(candidate)
    if not legal:
        return unreachable_route(candidate_count)
    best = min(legal, key=candidate_sort_key)
    return replace(
        best,
        candidate_count=candidate_count,
        legal_candidate_distances=tuple(sorted(item.distance for item in legal)),
    )


def select_undirected_pair(
    i: int,
    j: int,
    edge_i: str,
    edge_j: str,
    has_i_to_j: bool,
    has_j_to_i: bool,
    edges: Mapping[str, EdgeRecord],
    connection_index: Mapping[tuple[str, str], list[dict[str, str]]],
    lanes_by_id: Mapping[str, LaneRecord],
    edge_of_lane: Mapping[str, str],
) -> UndirectedPair:
    if not has_i_to_j and not has_j_to_i:
        raise PrepareError(
            f"A_sym neighbour {edge_i}--{edge_j} has no directed SUMO connection"
        )
    if has_i_to_j:
        forward = direct_center_distance(
            edges[edge_i],
            edges[edge_j],
            connection_index,
            edges,
            lanes_by_id,
            edge_of_lane,
        )
        if not forward.reachable:
            raise PrepareError(
                f"A_dir {edge_i}->{edge_j} has no legal lane-level direct "
                "connection; full-network shortest path rescue is forbidden"
            )
    else:
        forward = unreachable_route(0)
    if has_j_to_i:
        reverse = direct_center_distance(
            edges[edge_j],
            edges[edge_i],
            connection_index,
            edges,
            lanes_by_id,
            edge_of_lane,
        )
        if not reverse.reachable:
            raise PrepareError(
                f"A_dir {edge_j}->{edge_i} has no legal lane-level direct "
                "connection; full-network shortest path rescue is forbidden"
            )
    else:
        reverse = unreachable_route(0)

    if has_i_to_j and has_j_to_i:
        if reverse.distance + DISTANCE_ATOL < forward.distance:
            selected_direction = "j_to_i"
            selected_route = reverse
        else:
            selected_direction = "i_to_j"
            selected_route = forward
    elif has_i_to_j:
        selected_direction = "i_to_j"
        selected_route = forward
    else:
        selected_direction = "j_to_i"
        selected_route = reverse

    selected_distance = float(selected_route.distance)
    if not math.isfinite(selected_distance) or selected_distance <= DISTANCE_ATOL:
        raise PrepareError(
            f"neighbour {edge_i}--{edge_j} selected_distance "
            f"{selected_distance} is not finite and > 0"
        )
    return UndirectedPair(
        i=i,
        j=j,
        edge_i=edge_i,
        edge_j=edge_j,
        has_i_to_j=has_i_to_j,
        has_j_to_i=has_j_to_i,
        forward=forward,
        reverse=reverse,
        selected_distance=selected_distance,
        selected_direction=selected_direction,
        selected_route=selected_route,
        raw_weight=0.0,
        final_weight=0.0,
        retained=True,
    )


def component_summary(labels: np.ndarray) -> dict[str, object]:
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels.tolist()):
        groups.setdefault(int(label), []).append(index)
    sizes = sorted((len(members), members[0]) for members in groups.values())
    isolated = [members[0] for members in groups.values() if len(members) == 1]
    return {
        "component_count": len(groups),
        "component_sizes": [size for size, _start in reversed(sorted(sizes))],
        "isolated_node_count": len(isolated),
        "isolated_node_indices": isolated[:EXAMPLE_LIMIT],
        "largest_component_size": max((len(members) for members in groups.values()), default=0),
    }


def gaussian_weight(distance: float, sigma: float) -> float:
    return math.exp(-(distance * distance) / (sigma * sigma))


def format_float(value: float) -> str:
    return format(float(value), FLOAT_TEXT_FORMAT)


def format_optional_distance(route: DirectedRoute) -> str:
    if not route.reachable or not math.isfinite(route.distance):
        return ""
    return format_float(route.distance)


def join_ids(values: tuple[str, ...]) -> str:
    return "|".join(values)


def run_synthetic_checks() -> None:
    directed = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.uint8)
    symmetric = symmetrize_topology(directed)
    expected = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    if not np.array_equal(symmetric, expected):
        raise PrepareError("synthetic symmetrize failed")
    both = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    if int(np.maximum(both, both.T).max()) != 1:
        raise PrepareError("synthetic bidirectional pair must stay binary")

    edges = {
        "A": EdgeRecord("A", "", lanes=[LaneRecord("A_0", "0", 100.0)]),
        "B": EdgeRecord("B", "", lanes=[LaneRecord("B_0", "0", 50.0)]),
        "C": EdgeRecord("C", "", lanes=[LaneRecord("C_0", "0", 80.0)]),
        ":J_0": EdgeRecord(":J_0", "internal", lanes=[LaneRecord(":J_0_0", "0", 10.0)]),
        ":J_1": EdgeRecord(":J_1", "internal", lanes=[LaneRecord(":J_1_0", "0", 40.0)]),
        ":K_0": EdgeRecord(":K_0", "internal", lanes=[LaneRecord(":K_0_0", "0", 8.0)]),
        ":K_1": EdgeRecord(":K_1", "internal", lanes=[LaneRecord(":K_1_0", "0", 8.0)]),
    }
    lanes_by_id, edge_of_lane = build_lane_lookups(edges)
    connections = [
        {"from": "A", "to": "B", "fromLane": "0", "toLane": "0", "via": ":J_0_0"},
        {"from": "A", "to": "B", "fromLane": "0", "toLane": "0", "via": ":J_1_0"},
        {"from": "B", "to": "C", "fromLane": "0", "toLane": "0", "via": ":K_0_0"},
        {"from": "C", "to": "B", "fromLane": "0", "toLane": "0", "via": ":K_1_0"},
    ]
    connection_index = index_r_to_r_connections(connections, {"A", "B", "C"})
    route = direct_center_distance(
        edges["A"], edges["B"], connection_index, edges, lanes_by_id, edge_of_lane
    )
    expected_distance = 0.5 * 100.0 + 10.0 + 0.5 * 50.0
    longer = 0.5 * 100.0 + 40.0 + 0.5 * 50.0
    if not route.reachable or abs(route.distance - expected_distance) > 1e-9:
        raise PrepareError(
            f"synthetic center distance {route.distance} != {expected_distance}"
        )
    if route.candidate_count != 2:
        raise PrepareError("synthetic A->B should see two lane-level connections")
    if abs(min(route.legal_candidate_distances) - expected_distance) > 1e-9:
        raise PrepareError("synthetic multi-lane min was not selected")
    if abs(max(route.legal_candidate_distances) - longer) > 1e-9:
        raise PrepareError("synthetic longer lane connection was dropped instead of compared")
    if route.via_lane_ids != (":J_0_0",):
        raise PrepareError("synthetic selected via should be the shorter internal lane")
    reverse = direct_center_distance(
        edges["B"], edges["A"], connection_index, edges, lanes_by_id, edge_of_lane
    )
    if reverse.reachable:
        raise PrepareError("synthetic reverse without a direct connection must be unreachable")

    pair = select_undirected_pair(
        0,
        1,
        "B",
        "C",
        True,
        True,
        edges,
        connection_index,
        lanes_by_id,
        edge_of_lane,
    )
    d_bc = 0.5 * 50.0 + 8.0 + 0.5 * 80.0
    d_cb = 0.5 * 80.0 + 8.0 + 0.5 * 50.0
    if abs(d_bc - d_cb) > 1e-9:
        raise PrepareError("synthetic B<->C distances should be equal")
    if pair.selected_direction != "i_to_j":
        raise PrepareError("equal bidirectional distances must tie-break to i_to_j")
    if abs(pair.selected_distance - d_bc) > 1e-9:
        raise PrepareError("synthetic bidirectional selected_distance mismatch")

    one_way = select_undirected_pair(
        0,
        1,
        "A",
        "B",
        True,
        False,
        edges,
        connection_index,
        lanes_by_id,
        edge_of_lane,
    )
    if one_way.reverse.reachable or math.isfinite(one_way.reverse.distance):
        raise PrepareError("one-way reverse must stay unreachable/inf")
    if one_way.selected_direction != "i_to_j":
        raise PrepareError("one-way A->B must select i_to_j")

    sigma, source, mode = resolve_sigma(None, "median", [10.0, 20.0, 30.0])
    if abs(sigma - 20.0) > 1e-12 or mode != "median":
        raise PrepareError("synthetic median sigma failed")
    weight = gaussian_weight(expected_distance, 20.0)
    if not 0.0 < weight <= 1.0:
        raise PrepareError("synthetic gaussian weight out of range")
    tiny = 6.9e-7
    if float(format_float(tiny)) == 0.0:
        raise PrepareError("float format collapsed a nonzero weight to 0")
    del source


def resolve_sigma(
    explicit: float | None,
    sigma_mode: str,
    neighbor_distances: list[float],
) -> tuple[float, str, str]:
    if explicit is not None:
        if not math.isfinite(explicit) or explicit <= 0.0:
            raise PrepareError(f"--sigma must be finite and > 0, got {explicit}")
        return float(explicit), "cli --sigma", sigma_mode
    if sigma_mode not in SIGMA_MODES:
        raise PrepareError(
            f"unsupported --sigma-mode {sigma_mode}; use one of {', '.join(SIGMA_MODES)}"
        )
    positive = [value for value in neighbor_distances if math.isfinite(value) and value > 0.0]
    if not positive:
        raise PrepareError("cannot compute sigma: no positive finite neighbour distances")
    array = np.asarray(positive, dtype=np.float64)
    if sigma_mode == "median":
        sigma = float(np.median(array))
        source = "median of unique undirected positive neighbour distances"
    elif sigma_mode == "mean":
        sigma = float(np.mean(array))
        source = "mean of unique undirected positive neighbour distances"
    else:
        sigma = float(np.std(array, ddof=0))
        source = "std of unique undirected positive neighbour distances (ddof=0)"
    if not math.isfinite(sigma) or sigma <= DISTANCE_ATOL:
        raise PrepareError(f"auto sigma {sigma} is not finite and > {DISTANCE_ATOL}")
    return sigma, source, sigma_mode


def auto_sigma_value(sigma_mode: str, neighbor_distances: list[float]) -> float:
    array = np.asarray(
        [value for value in neighbor_distances if math.isfinite(value) and value > 0.0],
        dtype=np.float64,
    )
    if sigma_mode == "median":
        return float(np.median(array))
    if sigma_mode == "mean":
        return float(np.mean(array))
    return float(np.std(array, ddof=0))


def pair_to_csv_row(
    pair: UndirectedPair,
    sigma: float,
    sigma_mode: str,
    epsilon: float,
    save_path: bool,
) -> dict[str, object]:
    route = pair.selected_route
    return {
        "node_index_i": pair.i,
        "node_index_j": pair.j,
        "edge_id_i": pair.edge_i,
        "edge_id_j": pair.edge_j,
        "has_i_to_j_direct_connection": int(pair.has_i_to_j),
        "has_j_to_i_direct_connection": int(pair.has_j_to_i),
        "direct_connection_count_i_to_j": pair.forward.candidate_count,
        "direct_connection_count_j_to_i": pair.reverse.candidate_count,
        "distance_i_to_j": format_optional_distance(pair.forward),
        "distance_j_to_i": format_optional_distance(pair.reverse),
        "selected_distance": format_float(pair.selected_distance),
        "selected_direction": pair.selected_direction,
        "selected_from_lane": route.source_lane_id if save_path else "",
        "selected_to_lane": route.target_lane_id if save_path else "",
        "selected_via_lanes": join_ids(route.via_lane_ids) if save_path else "",
        "selected_path_edge_ids": join_ids(route.path_edge_ids) if save_path else "",
        "selected_path_lane_ids": join_ids(route.path_lane_ids) if save_path else "",
        "path_uses_internal_lanes": int(route.uses_internal) if save_path else "",
        "sigma": format_float(sigma),
        "sigma_mode": sigma_mode,
        "raw_weight": format_float(pair.raw_weight),
        "epsilon": format_float(epsilon),
        "final_weight": format_float(pair.final_weight),
        "retained_after_threshold": int(pair.retained),
    }


def validate_outputs(
    mapping: NodeMapping,
    directed: np.ndarray,
    symmetric: np.ndarray,
    distance: np.ndarray,
    weights: np.ndarray,
    weights_before: np.ndarray,
    sparse_weights: sparse.spmatrix,
    pairs: list[UndirectedPair],
    csv_rows: list[dict[str, object]],
    sigma: float,
    sigma_mode: str,
    sigma_source: str,
    explicit_sigma: float | None,
    epsilon: float,
    edges: dict[str, EdgeRecord],
    net_path: Path,
    r_nodes_path: Path,
    adj_path: Path,
    hashes_before: dict[str, str],
    hashes_after: dict[str, str],
    random_seed: int,
    distance_computed_pair_count: int,
    nonneighbor_distance_computation_count: int,
) -> dict[str, object]:
    failures: list[str] = []
    n_nodes = len(mapping.edge_ids)
    expected_sym = np.logical_or(directed, directed.T).astype(np.uint8)
    np.fill_diagonal(expected_sym, 0)
    if not np.array_equal(symmetric, expected_sym):
        failures.append("A_sym != logical_or(A_dir, A_dir.T) with zero diagonal")
    if set(np.unique(symmetric).tolist()) - {0, 1}:
        failures.append("A_sym is not binary")
    if not np.array_equal(symmetric, symmetric.T):
        failures.append("A_sym is not symmetric")
    if int(np.diag(symmetric).sum()) != 0:
        failures.append("A_sym has diagonal ones (artificial or retained self-loops)")
    extra = int(np.logical_and(symmetric, np.logical_not(expected_sym)).sum())
    if extra:
        failures.append("A_sym contains neighbours absent from both directed directions")
    if np.maximum(directed, directed.T).max() > 1:
        failures.append("bidirectional roads produced values above 1")

    real_self_loops = int(np.diag(directed).sum())
    undirected_from_dir, labels_weak = connected_components(
        csgraph=sparse.csr_matrix(directed), directed=True, connection="weak"
    )
    n_sym, labels_sym = connected_components(
        csgraph=sparse.csr_matrix(symmetric), directed=False
    )
    if undirected_from_dir != n_sym:
        failures.append(
            f"A_sym components {n_sym} != weak components of A_dir {undirected_from_dir}"
        )

    if mapping.edge_ids != [str(row["edge_id"]).strip() for row in mapping.rows]:
        failures.append("node order diverged from r_nodes.csv")
    if mapping.node_index != list(range(n_nodes)):
        failures.append("node_index is not 0..N-1 from r_nodes.csv")

    if distance.shape != (n_nodes, n_nodes):
        failures.append("distance matrix shape mismatch")
    if not np.array_equal(np.isfinite(distance), np.isfinite(distance.T)):
        failures.append("distance finiteness is not symmetric")
    if not np.allclose(
        np.where(np.isfinite(distance), distance, 0.0),
        np.where(np.isfinite(distance.T), distance.T, 0.0),
        atol=DISTANCE_ATOL,
    ):
        failures.append("finite distances are not symmetric")
    if not np.allclose(np.diag(distance), 0.0, atol=DISTANCE_ATOL):
        failures.append("distance diagonal is not 0")

    finite_offdiag = 0
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            value = float(distance[i, j])
            if symmetric[i, j] == 0:
                if math.isfinite(value):
                    failures.append(f"non-neighbour ({i},{j}) is not +inf")
                    break
            else:
                if not math.isfinite(value) or value <= DISTANCE_ATOL:
                    failures.append(
                        f"neighbour ({i},{j}) has non-finite or non-positive distance"
                    )
                    break
                finite_offdiag += 1
        else:
            continue
        break

    expected_pair_set = {
        (int(i), int(j))
        for i, j in zip(*np.where(np.triu(symmetric, k=1) == 1))
    }
    actual_pair_set = {(pair.i, pair.j) for pair in pairs}
    if any(pair.i >= pair.j for pair in pairs):
        failures.append("CSV pairs are not ordered node_index_i < node_index_j")
    if actual_pair_set != expected_pair_set:
        failures.append("candidate node pairs are not exactly A_sym upper-triangle neighbours")
    expected_pairs = len(expected_pair_set)
    if len(pairs) != expected_pairs:
        failures.append(f"CSV pair count {len(pairs)} != undirected edges {expected_pairs}")
    if len(csv_rows) != expected_pairs:
        failures.append(f"CSV row count {len(csv_rows)} != A_sym upper-triangle nonzero count")
    if distance_computed_pair_count != expected_pairs:
        failures.append(
            "distance_computed_pair_count "
            f"{distance_computed_pair_count} != topology_neighbor_pair_count {expected_pairs}"
        )
    if nonneighbor_distance_computation_count != 0:
        failures.append(
            "non-neighbours were used for distance computation: "
            f"{nonneighbor_distance_computation_count}"
        )

    unreachable_neighbours = [pair for pair in pairs if not math.isfinite(pair.selected_distance)]
    if unreachable_neighbours:
        failures.append(
            "unreachable A_sym neighbour pair "
            f"{unreachable_neighbours[0].edge_i}->{unreachable_neighbours[0].edge_j}"
        )

    single_direction_neighbor_count = 0
    bidirectional_neighbor_count = 0
    direct_lane_connection_candidate_count = 0
    for pair, row in zip(pairs, csv_rows):
        if pair.has_i_to_j and pair.has_j_to_i:
            bidirectional_neighbor_count += 1
        else:
            single_direction_neighbor_count += 1
        if pair.has_i_to_j:
            direct_lane_connection_candidate_count += pair.forward.candidate_count
        if pair.has_j_to_i:
            direct_lane_connection_candidate_count += pair.reverse.candidate_count
        if not pair.has_i_to_j and not pair.has_j_to_i:
            failures.append(f"neighbour {pair.edge_i}-{pair.edge_j} has no direct direction")
        if pair.has_i_to_j:
            if not pair.forward.reachable:
                failures.append(f"missing i->j direct distance for {pair.edge_i}->{pair.edge_j}")
            elif pair.forward.legal_candidate_distances:
                expected_min = min(pair.forward.legal_candidate_distances)
                if abs(pair.forward.distance - expected_min) > DISTANCE_ATOL:
                    failures.append(
                        f"i->j did not keep the shortest legal lane connection for {pair.edge_i}"
                    )
        else:
            if pair.forward.reachable or math.isfinite(pair.forward.distance):
                failures.append(
                    f"direction without A_dir got a distance for {pair.edge_i}->{pair.edge_j}"
                )
            if str(row["distance_i_to_j"]) != "":
                failures.append(
                    f"CSV filled a missing i->j distance for {pair.edge_i}-{pair.edge_j}"
                )
        if pair.has_j_to_i:
            if not pair.reverse.reachable:
                failures.append(f"missing j->i direct distance for {pair.edge_j}->{pair.edge_i}")
            elif pair.reverse.legal_candidate_distances:
                expected_min = min(pair.reverse.legal_candidate_distances)
                if abs(pair.reverse.distance - expected_min) > DISTANCE_ATOL:
                    failures.append(
                        f"j->i did not keep the shortest legal lane connection for {pair.edge_j}"
                    )
        else:
            if pair.reverse.reachable or math.isfinite(pair.reverse.distance):
                failures.append(
                    f"direction without A_dir got a distance for {pair.edge_j}->{pair.edge_i}"
                )
            if str(row["distance_j_to_i"]) != "":
                failures.append(
                    f"CSV filled a missing j->i distance for {pair.edge_i}-{pair.edge_j}"
                )
        if pair.selected_direction == "i_to_j" and directed[pair.i, pair.j] != 1:
            failures.append(f"selected i_to_j but A_dir[{pair.i},{pair.j}] != 1")
        if pair.selected_direction == "j_to_i" and directed[pair.j, pair.i] != 1:
            failures.append(f"selected j_to_i but A_dir[{pair.j},{pair.i}] != 1")
        if pair.has_i_to_j and not pair.has_j_to_i and pair.selected_direction != "i_to_j":
            failures.append(f"one-way neighbour used a reverse detour for {pair.edge_i}-{pair.edge_j}")
        if pair.has_j_to_i and not pair.has_i_to_j and pair.selected_direction != "j_to_i":
            failures.append(f"one-way neighbour used a reverse detour for {pair.edge_i}-{pair.edge_j}")
        if pair.has_i_to_j and pair.has_j_to_i:
            expected_sel = min(pair.forward.distance, pair.reverse.distance)
            if abs(pair.selected_distance - expected_sel) > DISTANCE_ATOL:
                failures.append(
                    f"bidirectional selected_distance is not min of direct distances "
                    f"for {pair.edge_i}-{pair.edge_j}"
                )
            if (
                abs(pair.forward.distance - pair.reverse.distance) <= DISTANCE_ATOL
                and pair.selected_direction != "i_to_j"
            ):
                failures.append(
                    f"equal bidirectional distances did not tie-break to i_to_j "
                    f"for {pair.edge_i}-{pair.edge_j}"
                )
        selected_source = (
            pair.edge_i if pair.selected_direction == "i_to_j" else pair.edge_j
        )
        selected_target = (
            pair.edge_j if pair.selected_direction == "i_to_j" else pair.edge_i
        )
        if not path_has_only_direct_connection_edges(
            pair.selected_route.path_edge_ids, selected_source, selected_target, edges
        ):
            failures.append(
                f"selected path for {pair.edge_i}-{pair.edge_j} is not "
                "source + connection internals + target"
            )
        ordinary_extras = [
            edge_id
            for edge_id in pair.selected_route.path_edge_ids
            if edge_id not in {selected_source, selected_target}
            and not (edges.get(edge_id) is not None and edges[edge_id].is_internal)
        ]
        if ordinary_extras:
            failures.append(
                f"selected path for {pair.edge_i}-{pair.edge_j} contains other ordinary edges"
            )
        if abs(float(distance[pair.i, pair.j]) - pair.selected_distance) > DISTANCE_ATOL:
            failures.append("D matrix does not match selected pair distance")
        expected_raw = gaussian_weight(pair.selected_distance, sigma)
        if abs(pair.raw_weight - expected_raw) > WEIGHT_ATOL:
            failures.append("raw_weight does not match exp(-d^2 / sigma^2)")
        expected_final = 0.0 if expected_raw < epsilon else expected_raw
        if abs(pair.final_weight - expected_final) > WEIGHT_ATOL:
            failures.append("final_weight does not match epsilon rule")
        if pair.retained != (expected_final > 0.0):
            failures.append("retained_after_threshold inconsistent with final_weight")
        csv_raw = float(str(row["raw_weight"]))
        csv_final = float(str(row["final_weight"]))
        if abs(csv_raw - pair.raw_weight) > WEIGHT_ATOL:
            failures.append("CSV raw_weight does not match recomputed Gaussian weight")
        if abs(csv_final - float(weights[pair.i, pair.j])) > WEIGHT_ATOL:
            failures.append("CSV final_weight does not match dense W")
        if pair.final_weight > 0.0 and csv_final == 0.0:
            failures.append(
                f"nonzero weight for {pair.edge_i}-{pair.edge_j} was formatted as 0"
            )
        if pair.raw_weight > 0.0 and csv_raw == 0.0:
            failures.append(
                f"nonzero raw_weight for {pair.edge_i}-{pair.edge_j} was formatted as 0"
            )

    if weights.shape != (n_nodes, n_nodes):
        failures.append("W shape mismatch")
    if not np.array_equal(weights, weights.T):
        failures.append("W is not symmetric")
    if np.isnan(weights).any() or np.isinf(weights).any():
        failures.append("W contains NaN or inf")
    if np.any(weights < -WEIGHT_ATOL) or np.any(weights > 1.0 + WEIGHT_ATOL):
        failures.append("W has values outside [0, 1]")
    if not np.allclose(np.diag(weights), 0.0, atol=WEIGHT_ATOL):
        failures.append("W diagonal is not 0")
    if np.any((symmetric == 0) & (weights > WEIGHT_ATOL)):
        failures.append("W has mass on non-neighbours")
    retained_mask = np.zeros_like(symmetric, dtype=bool)
    for pair in pairs:
        if pair.retained:
            retained_mask[pair.i, pair.j] = True
            retained_mask[pair.j, pair.i] = True
    if np.any((~retained_mask) & (weights > WEIGHT_ATOL)):
        failures.append("W is nonzero off retained adjacency positions")
    dense_from_sparse = np.asarray(sparse_weights.todense())
    if not np.allclose(dense_from_sparse, weights, atol=WEIGHT_ATOL):
        failures.append("sparse W does not match dense W")

    if not math.isfinite(sigma) or sigma <= DISTANCE_ATOL:
        failures.append(f"sigma {sigma} is not finite and > {DISTANCE_ATOL}")
    unique_distances = [pair.selected_distance for pair in pairs]
    if explicit_sigma is None:
        expected_sigma = auto_sigma_value(sigma_mode, unique_distances)
        if abs(sigma - expected_sigma) > DISTANCE_ATOL:
            failures.append(
                f"sigma_mode {sigma_mode} does not match computed sigma {sigma}"
            )
        if not sigma_source.startswith(sigma_mode):
            failures.append("sigma_source does not match sigma_mode")
    elif not sigma_source.startswith("cli"):
        failures.append("explicit --sigma was not recorded as the sigma source")

    threshold_deleted = [
        pair for pair in pairs if pair.raw_weight >= 0.0 and pair.raw_weight < epsilon
    ]
    restored = [pair for pair in threshold_deleted if pair.retained]
    if restored:
        failures.append("threshold-deleted edges were silently retained")

    n_before, labels_before = connected_components(
        csgraph=sparse.csr_matrix((weights_before > 0).astype(np.uint8)), directed=False
    )
    n_after, labels_after = connected_components(
        csgraph=sparse.csr_matrix((weights > 0).astype(np.uint8)), directed=False
    )

    missing_r = [edge_id for edge_id in mapping.edge_ids if edge_id not in edges]
    if missing_r:
        failures.append(f"R edge_id missing from net_tls: {missing_r[:EXAMPLE_LIMIT]}")

    if hashes_before != hashes_after:
        failures.append("protected input files changed during the run")

    rng = random.Random(random_seed)
    sample_source = list(pairs)
    rng.shuffle(sample_source)
    samples = sample_source[: min(SAMPLE_PAIR_COUNT, len(sample_source))]
    sample_records = []
    for pair in samples:
        sample_records.append(
            {
                "node_index_i": pair.i,
                "node_index_j": pair.j,
                "edge_id_i": pair.edge_i,
                "edge_id_j": pair.edge_j,
                "has_i_to_j_direct_connection": pair.has_i_to_j,
                "has_j_to_i_direct_connection": pair.has_j_to_i,
                "forward_reachable": pair.forward.reachable,
                "reverse_reachable": pair.reverse.reachable,
                "distance_i_to_j": pair.forward.distance if pair.forward.reachable else None,
                "distance_j_to_i": pair.reverse.distance if pair.reverse.reachable else None,
                "selected_distance": pair.selected_distance,
                "selected_direction": pair.selected_direction,
                "selected_from_lane": pair.selected_route.source_lane_id,
                "selected_to_lane": pair.selected_route.target_lane_id,
                "selected_via_lanes": list(pair.selected_route.via_lane_ids),
                "path_edge_ids": list(pair.selected_route.path_edge_ids),
                "path_lane_ids": list(pair.selected_route.path_lane_ids),
                "path_uses_internal_lanes": pair.selected_route.uses_internal,
            }
        )

    mismatch_edges = []
    for edge_id in mapping.edge_ids:
        record = edges.get(edge_id)
        if record is None or len(record.lanes) < 2:
            continue
        lane_lengths = [lane.length for lane in record.lanes]
        if max(lane_lengths) - min(lane_lengths) > LENGTH_MISMATCH_TOL_M:
            mismatch_edges.append(edge_id)

    all_possible_undirected_pair_count = n_nodes * (n_nodes - 1) // 2
    summary = {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "node_count": n_nodes,
        "real_directed_self_loops": real_self_loops,
        "inputs_unchanged": hashes_before == hashes_after,
        "protected_input_paths": {
            "r_nodes": posix(r_nodes_path),
            "directed_adjacency": posix(adj_path),
            "net_tls": posix(net_path),
        },
        "symmetry": {
            "formula": SYMMETRY_FORMULA,
            "binary": True,
            "matches_logical_or": extra == 0,
        },
        "connectivity": {
            "directed_weak_components": component_summary(labels_weak),
            "a_sym_components": component_summary(labels_sym),
            "weighted_before_threshold_components": component_summary(labels_before),
            "weighted_after_threshold_components": component_summary(labels_after),
            "a_sym_matches_weak_directed": undirected_from_dir == n_sym,
        },
        "counts": {
            "directed_edges": int(directed.sum()),
            "all_possible_undirected_pair_count": all_possible_undirected_pair_count,
            "topology_neighbor_pair_count": expected_pairs,
            "distance_computed_pair_count": distance_computed_pair_count,
            "nonneighbor_distance_computation_count": nonneighbor_distance_computation_count,
            "single_direction_neighbor_count": single_direction_neighbor_count,
            "bidirectional_neighbor_count": bidirectional_neighbor_count,
            "direct_lane_connection_candidate_count": direct_lane_connection_candidate_count,
            "undirected_edges_before_threshold": expected_pairs,
            "undirected_edges_after_threshold": int(sum(1 for pair in pairs if pair.retained)),
            "threshold_deleted_edges": len(threshold_deleted),
            "finite_offdiag_distance_entries": finite_offdiag,
        },
        "threshold_deleted_examples": [
            {
                "node_index_i": pair.i,
                "node_index_j": pair.j,
                "edge_id_i": pair.edge_i,
                "edge_id_j": pair.edge_j,
                "selected_distance": pair.selected_distance,
                "raw_weight": pair.raw_weight,
            }
            for pair in threshold_deleted[:EXAMPLE_LIMIT]
        ],
        "lane_length_mismatch_edge_ids": mismatch_edges,
        "random_seed": random_seed,
        "sampled_neighbour_pairs": sample_records,
        "sigma": sigma,
        "sigma_mode": sigma_mode,
        "sigma_source": sigma_source,
        "epsilon": epsilon,
        "tie_break_rule": TIE_BREAK_RULE,
        "n_before_threshold_components": n_before,
        "n_after_threshold_components": n_after,
        "euclidean_fallback_used": False,
        "full_network_shortest_path_used": False,
        "npz_allow_pickle_false_reload_expected": True,
    }
    if failures:
        raise PrepareError("weighted adjacency validation failed:\n" + "\n".join(failures))
    return summary


def dependency_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "sumolib": "not-used",
    }
    try:
        import sumolib  # type: ignore

        versions["sumolib"] = getattr(sumolib, "__version__", "imported-unknown-version")
    except Exception:
        versions["sumolib"] = "not-installed"
    return versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a symmetric STGCN adjacency from R-only directed topology and "
            "direct SUMO connection center-to-center distances. Does not "
            "overwrite r_adjacency.npy."
        )
    )
    parser.add_argument("--r-nodes", type=Path, default=DEFAULT_R_NODES)
    parser.add_argument("--directed-adjacency", type=Path, default=DEFAULT_DIRECTED_ADJACENCY)
    parser.add_argument("--net-file", type=Path, default=DEFAULT_NET_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument(
        "--sigma-mode",
        default=DEFAULT_SIGMA_MODE,
        choices=list(SIGMA_MODES),
        help="Used only when --sigma is omitted. Default: median.",
    )
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--distance-mode", default=DEFAULT_DISTANCE_MODE)
    parser.add_argument(
        "--save-path-evidence",
        dest="save_path_evidence",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-save-path-evidence",
        dest="save_path_evidence",
        action="store_false",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED_DEFAULT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.distance_mode != DEFAULT_DISTANCE_MODE:
        raise PrepareError(
            f"unsupported --distance-mode {args.distance_mode}; "
            f"this script only implements {DEFAULT_DISTANCE_MODE} "
            "(full-network shortest paths were removed)"
        )
    if args.epsilon < 0.0 or not math.isfinite(args.epsilon):
        raise PrepareError(f"--epsilon must be finite and >= 0, got {args.epsilon}")
    if not args.skip_synthetic:
        run_synthetic_checks()

    r_nodes_path = args.r_nodes.expanduser().resolve()
    adj_path = args.directed_adjacency.expanduser().resolve()
    net_path = args.net_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    protected = (r_nodes_path, adj_path, net_path)
    for path in protected:
        if not path.is_file():
            raise PrepareError(f"missing input: {path}")
        if path.resolve() == output_dir or output_dir in path.resolve().parents:
            raise PrepareError(f"refusing to use protected input inside output-dir: {path}")

    mapping = load_node_mapping(r_nodes_path)
    directed = load_directed_adjacency(adj_path, len(mapping.edge_ids))
    hashes_before = {posix(path): sha256_file(path) for path in protected}

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise PrepareError(f"{output_dir} is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for name in OUTPUT_ARTIFACTS:
            path = output_dir / name
            if path.is_file():
                path.unlink()

    for path in protected:
        if path.resolve() in {(output_dir / name).resolve() for name in OUTPUT_ARTIFACTS}:
            raise PrepareError("output path collides with a protected input")

    edges, connections = parse_sumo_net(net_path)
    missing = [edge_id for edge_id in mapping.edge_ids if edge_id not in edges]
    if missing:
        raise PrepareError(f"R nodes missing from net_tls: {missing[:EXAMPLE_LIMIT]}")
    for edge_id in mapping.edge_ids:
        record = edges[edge_id]
        if not record.lanes:
            raise PrepareError(f"R edge {edge_id} has no lanes in net_tls")

    lanes_by_id, edge_of_lane = build_lane_lookups(edges)
    r_edge_ids = set(mapping.edge_ids)
    connection_index = index_r_to_r_connections(connections, r_edge_ids)
    symmetric = symmetrize_topology(directed)
    n_nodes = len(mapping.edge_ids)
    distance = np.full((n_nodes, n_nodes), np.inf, dtype=np.float64)
    np.fill_diagonal(distance, 0.0)

    pairs: list[UndirectedPair] = []
    nonneighbor_distance_computation_count = 0
    i_idx, j_idx = np.where(np.triu(symmetric, k=1) == 1)
    for i, j in zip(i_idx.tolist(), j_idx.tolist()):
        if symmetric[i, j] != 1:
            nonneighbor_distance_computation_count += 1
            continue
        edge_i = mapping.edge_ids[i]
        edge_j = mapping.edge_ids[j]
        pair = select_undirected_pair(
            i=i,
            j=j,
            edge_i=edge_i,
            edge_j=edge_j,
            has_i_to_j=bool(directed[i, j]),
            has_j_to_i=bool(directed[j, i]),
            edges=edges,
            connection_index=connection_index,
            lanes_by_id=lanes_by_id,
            edge_of_lane=edge_of_lane,
        )
        distance[i, j] = pair.selected_distance
        distance[j, i] = pair.selected_distance
        pairs.append(pair)
    distance_computed_pair_count = len(pairs)

    unique_distances = [pair.selected_distance for pair in pairs]
    sigma, sigma_source, sigma_mode = resolve_sigma(
        args.sigma, args.sigma_mode, unique_distances
    )
    weights_before = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for pair in pairs:
        raw = gaussian_weight(pair.selected_distance, sigma)
        pair.raw_weight = raw
        weights_before[pair.i, pair.j] = raw
        weights_before[pair.j, pair.i] = raw
    weights = weights_before.copy()
    if args.epsilon > 0.0:
        weights[weights < args.epsilon] = 0.0
    np.fill_diagonal(weights, 0.0)
    weights[symmetric == 0] = 0.0
    for pair in pairs:
        pair.final_weight = float(weights[pair.i, pair.j])
        pair.retained = pair.final_weight > 0.0

    hashes_after = {posix(path): sha256_file(path) for path in protected}
    sparse_weights = sparse.csr_matrix(weights, dtype=np.float32)
    csv_rows = [
        pair_to_csv_row(
            pair, sigma, sigma_mode, float(args.epsilon), args.save_path_evidence
        )
        for pair in pairs
    ]
    validation = validate_outputs(
        mapping=mapping,
        directed=directed,
        symmetric=symmetric,
        distance=distance,
        weights=weights,
        weights_before=weights_before,
        sparse_weights=sparse_weights,
        pairs=pairs,
        csv_rows=csv_rows,
        sigma=sigma,
        sigma_mode=sigma_mode,
        sigma_source=sigma_source,
        explicit_sigma=args.sigma,
        epsilon=float(args.epsilon),
        edges=edges,
        net_path=net_path,
        r_nodes_path=r_nodes_path,
        adj_path=adj_path,
        hashes_before=hashes_before,
        hashes_after=hashes_after,
        random_seed=args.random_seed,
        distance_computed_pair_count=distance_computed_pair_count,
        nonneighbor_distance_computation_count=nonneighbor_distance_computation_count,
    )

    np.save(output_dir / "stgcn_undirected_topology.npy", symmetric, allow_pickle=False)
    np.save(output_dir / "stgcn_center_distance.npy", distance, allow_pickle=False)
    np.save(output_dir / "stgcn_weighted_adjacency.npy", weights, allow_pickle=False)
    sparse.save_npz(output_dir / "stgcn_weighted_adjacency_sparse.npz", sparse_weights)
    atomic_write_csv(
        output_dir / "stgcn_weighted_edges.csv",
        EDGE_CSV_FIELDS,
        csv_rows,
    )

    output_hashes = {
        name: file_fingerprint(output_dir / name)
        for name in OUTPUT_ARTIFACTS
        if name.endswith((".npy", ".npz", ".csv"))
    }
    metadata = {
        "script_path": posix(SCRIPT_PATH),
        "script_version": SCRIPT_VERSION,
        "generated_at": now_utc(),
        "project_root": posix(PROJECT_ROOT),
        "node_count": n_nodes,
        "node_order_source": posix(r_nodes_path),
        "node_order_rule": "node_index 0..N-1 from r_nodes.csv; not lexicographic by edge_id",
        "r_nodes_sha256": hashes_after[posix(r_nodes_path)],
        "directed_adjacency_path": posix(adj_path),
        "directed_adjacency_sha256": hashes_after[posix(adj_path)],
        "net_tls_path": posix(net_path),
        "net_tls_sha256": hashes_after[posix(net_path)],
        "symmetry_formula": SYMMETRY_FORMULA,
        "center_point_definition": CENTER_DEFINITION,
        "multi_lane_center_rule": (
            "Every legal lane-level <connection> for a directed R-R pair is "
            "evaluated. The shortest finite direct center distance is kept. "
            "Equal candidates use lexicographic fromLane/toLane/via tie-break."
        ),
        "distance_unit": "meters",
        "distance_mode": args.distance_mode,
        "directed_center_distance_definition": DIRECTED_DISTANCE_DEFINITION,
        "symmetric_distance_definition": SYMMETRIC_DISTANCE_DEFINITION,
        "tie_break_rule": TIE_BREAK_RULE,
        "full_net_tls_used_for_routing": False,
        "full_network_shortest_path_used": False,
        "internal_lanes_counted_in_distance": True,
        "internal_lanes_are_r_nodes": False,
        "gaussian_formula": GAUSSIAN_FORMULA,
        "sigma": sigma,
        "sigma_mode": sigma_mode,
        "sigma_source": sigma_source,
        "sigma_is_not_sigma_squared": True,
        "epsilon": float(args.epsilon),
        "csv_float_format": FLOAT_TEXT_FORMAT,
        "artificial_self_loops_added": False,
        "diagonal_rule": ZERO_DIAG,
        "normalized": False,
        "laplacian_computed": False,
        "chebyshev_computed": False,
        "euclidean_fallback": False,
        "original_stgcn_notes": {
            "source": "reference/STGCN_IJCAI-18/utils/math_graph.py",
            "weight_matrix_sigma2_default": 0.1,
            "weight_matrix_epsilon_default": 0.5,
            "pems_distance_divisor": 10000,
            "copied_pems_sigma2_or_divisor": False,
            "intended_load": (
                "Use stgcn_weighted_adjacency.npy as W with the scaling=False "
                "branch of weight_matrix (do not apply the distance kernel again). "
                "Build scaled_laplacian and cheb_poly_approx later at train time."
            ),
        },
        "undirected_edge_count_before_threshold": int(np.triu(symmetric, k=1).sum()),
        "undirected_edge_count_after_threshold": int(sum(1 for pair in pairs if pair.retained)),
        "output_files": output_hashes,
        "dependency_versions": dependency_versions(),
        "random_seed": args.random_seed,
        "did_not_modify": [posix(path) for path in protected],
    }
    atomic_write_json(output_dir / "stgcn_weighted_adjacency_metadata.json", json_ready(metadata))
    output_hashes["stgcn_weighted_adjacency_metadata.json"] = file_fingerprint(
        output_dir / "stgcn_weighted_adjacency_metadata.json"
    )
    atomic_write_json(
        output_dir / "stgcn_weighted_adjacency_validation.json",
        json_ready(validation),
    )
    print(
        f"done output={posix(output_dir)} nodes={n_nodes} "
        f"undirected_edges={int(np.triu(symmetric, k=1).sum())} "
        f"sigma={sigma:.6f} sigma_mode={sigma_mode} epsilon={args.epsilon} "
        f"validation={validation['status']}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except PrepareError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
