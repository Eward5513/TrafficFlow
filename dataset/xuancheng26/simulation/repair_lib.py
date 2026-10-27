"""Shared Xuancheng26 trip-repair primitives.

This module is imported by the pipeline scripts and unit tests. It never writes
under the read-only original download tree.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import pickle
import resource
import shutil
import tempfile
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import yaml

CODE_ROOT = Path("/home/tzhang174/TrafficFlow/dataset/xuancheng26/simulation")
DEFAULT_CONFIG = CODE_ROOT / "config" / "repair_trips.yaml"
SOURCE_ROOT = Path("/data5/tzhang174/Xuancheng26")
DATA_ROOT = Path("/data5/tzhang174/TrafficFlow/xuancheng26")

CODE_SUFFIXES = {".py", ".sh", ".ipynb"}
CODE_NAMES = {"requirements.txt", "setup.py", "pyproject.toml"}
LARGE_DATA_SUFFIXES = {
    ".json",
    ".jsonl",
    ".gz",
    ".parquet",
    ".pkl",
    ".pickle",
    ".xml",
    ".csv",
    ".npz",
    ".h5",
    ".rou",
    ".net.xml",
}


def load_config(path: Optional[Path] = None) -> dict:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def data_paths(cfg: Optional[dict] = None) -> dict:
    cfg = cfg or load_config()
    data_root = Path(cfg["data_root"])
    sim = data_root / "simulation_data"
    rn = sim / "road_network"
    td = sim / "trip_data"
    return {
        "data_root": data_root,
        "sim": sim,
        "road_base": rn / "base",
        "road_topology": rn / "topology",
        "road_cache": rn / "cache",
        "repaired": td / "repaired",
        "rejected": td / "rejected",
        "repair_metadata": td / "repair_metadata",
        "manifests": td / "manifests",
        "reports": td / "reports",
        "tmp": td / "tmp",
        "base_net": rn / "base" / "xuancheng.net.xml",
    }


def ensure_data_dirs(cfg: Optional[dict] = None) -> dict:
    paths = data_paths(cfg)
    for key in (
        "road_base",
        "road_topology",
        "road_cache",
        "repaired",
        "rejected",
        "repair_metadata",
        "manifests",
        "reports",
        "tmp",
    ):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def file_sha256_map(paths: Iterable[Path], root: Optional[Path] = None) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for path in paths:
        path = Path(path)
        rel = str(path.relative_to(root)) if root is not None else str(path)
        out[rel] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return out


def code_file_hash(code_root: Path = CODE_ROOT) -> str:
    hasher = hashlib.sha256()
    files = []
    for path in sorted(code_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".pyc" or "__pycache__" in path.parts:
            continue
        files.append(path)
    for path in files:
        rel = path.relative_to(code_root).as_posix().encode("utf-8")
        hasher.update(rel)
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def config_hash(cfg_path: Path = DEFAULT_CONFIG) -> str:
    return sha256_file(cfg_path)


def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_kb = float(usage.ru_maxrss)
    # Linux reports KB; macOS reports bytes. This pipeline runs on Linux.
    return rss_kb / 1024.0


def strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def is_internal_edge_id(edge_id: Optional[str], function: Optional[str] = None) -> bool:
    if not edge_id:
        return True
    if str(edge_id).startswith(":"):
        return True
    if function == "internal":
        return True
    return False


def date_from_trip_name(name: str) -> str:
    parts = Path(name).name.split("_")
    return f"{parts[1]}-{parts[2]}-{parts[3]}"


def trip_id_for(date: str, record_index: int) -> str:
    return f"{date.replace('-', '')}_{record_index}"


def source_trip_files(source_root: Path) -> List[Path]:
    trips_dir = Path(source_root) / "trips"
    return sorted(trips_dir.glob("data_2023_04_*_type_filtered.json"))


def unique_active_dates(cfg: dict) -> List[str]:
    exclude = set((cfg.get("exclude_dates") or {}).keys())
    dates = []
    for path in source_trip_files(Path(cfg["source_root"])):
        date = date_from_trip_name(path.name)
        if date not in exclude:
            dates.append(date)
    return dates


def trip_path_for_date(cfg: dict, date: str) -> Path:
    y, m, d = date.split("-")
    return Path(cfg["source_root"]) / "trips" / f"data_{y}_{m}_{d}_type_filtered.json"


@dataclass(slots=True)
class LaneConn:
    frm: str
    to: str
    from_lane: str
    to_lane: str
    direction: str
    via: str
    tl: str
    link_index: str
    state: str


@dataclass
class SumoGraph:
    net_sha256: str
    max_inserted_edges: int
    normal_ids: List[str]
    normal_set: Set[str]
    internal_ids: List[str]
    length_of: Dict[str, float]
    edge_from: Dict[str, str]
    edge_to: Dict[str, str]
    nlanes_of: Dict[str, int]
    speed_of: Dict[str, float]
    adj: Dict[str, List[str]]
    adj_set: Dict[str, Set[str]]
    reverse_adj: Dict[str, List[str]]
    conn_dir: Dict[Tuple[str, str], str]
    lane_conns: List[LaneConn]
    n_conn_raw: int
    n_conn_edge_pairs: int

    def has_connection(self, a: str, b: str) -> bool:
        return b in self.adj_set.get(a, ())


@dataclass(frozen=True, slots=True)
class Bridge:
    reachable: bool
    k: int
    inserted: Tuple[str, ...]
    insert_len: float
    n_equal: int
    ambiguous: bool
    too_long_or_unreach: bool

    @property
    def ok(self) -> bool:
        return self.reachable and not self.too_long_or_unreach


UNREACHABLE = Bridge(
    reachable=False,
    k=-1,
    inserted=(),
    insert_len=0.0,
    n_equal=0,
    ambiguous=False,
    too_long_or_unreach=True,
)


DIRECT = Bridge(
    reachable=True,
    k=0,
    inserted=(),
    insert_len=0.0,
    n_equal=1,
    ambiguous=False,
    too_long_or_unreach=False,
)


def parse_sumo_net(net_path: Path) -> SumoGraph:
    import xml.etree.ElementTree as ET

    net_path = Path(net_path)
    normal_ids: Set[str] = set()
    internal_ids: Set[str] = set()
    edge_from: Dict[str, str] = {}
    edge_to: Dict[str, str] = {}
    edge_nlanes: Dict[str, int] = defaultdict(int)
    edge_len_sum: Dict[str, float] = defaultdict(float)
    edge_speed: Dict[str, float] = {}
    adj_set: Dict[str, Set[str]] = defaultdict(set)
    conn_dir: Dict[Tuple[str, str], str] = {}
    lane_conns: List[LaneConn] = []
    n_conn_raw = 0
    current_edge: Optional[dict] = None
    current_internal = False

    for event, elem in ET.iterparse(str(net_path), events=("start", "end")):
        tag = strip_ns(elem.tag)
        if event == "start":
            if tag == "edge":
                current_edge = dict(elem.attrib)
                current_internal = is_internal_edge_id(
                    current_edge.get("id"), current_edge.get("function")
                )
            elif tag == "lane" and current_edge is not None:
                eid = current_edge.get("id")
                if eid and not current_internal:
                    edge_nlanes[eid] += 1
                    try:
                        edge_len_sum[eid] += float(elem.attrib.get("length") or 0)
                    except ValueError:
                        pass
                    if eid not in edge_speed:
                        try:
                            edge_speed[eid] = float(elem.attrib.get("speed") or "nan")
                        except ValueError:
                            pass
            continue

        if tag == "edge" and current_edge is not None:
            eid = str(current_edge.get("id") or "")
            if current_internal:
                internal_ids.add(eid)
            else:
                normal_ids.add(eid)
                edge_from[eid] = str(current_edge.get("from") or "")
                edge_to[eid] = str(current_edge.get("to") or "")
            current_edge = None
            current_internal = False
            elem.clear()
        elif tag == "connection":
            n_conn_raw += 1
            frm = str(elem.attrib.get("from") or "")
            to = str(elem.attrib.get("to") or "")
            via = str(elem.attrib.get("via") or "")
            direction = str(elem.attrib.get("dir") or "")
            if frm and to and (not is_internal_edge_id(frm)) and (not is_internal_edge_id(to)):
                adj_set[frm].add(to)
                key = (frm, to)
                if key not in conn_dir:
                    conn_dir[key] = direction
                lane_conns.append(
                    LaneConn(
                        frm=frm,
                        to=to,
                        from_lane=str(elem.attrib.get("fromLane") or ""),
                        to_lane=str(elem.attrib.get("toLane") or ""),
                        direction=direction,
                        via=via,
                        tl=str(elem.attrib.get("tl") or ""),
                        link_index=str(elem.attrib.get("linkIndex") or ""),
                        state=str(elem.attrib.get("state") or ""),
                    )
                )
            elem.clear()
        elif tag not in ("net", "location"):
            elem.clear()

    length_of: Dict[str, float] = {}
    nlanes_of: Dict[str, int] = {}
    for eid in normal_ids:
        n_lanes = max(int(edge_nlanes[eid]), 1)
        nlanes_of[eid] = int(edge_nlanes[eid])
        length_of[eid] = float(edge_len_sum[eid]) / float(n_lanes)

    adj: Dict[str, List[str]] = {eid: [] for eid in normal_ids}
    reverse_adj: Dict[str, List[str]] = {eid: [] for eid in normal_ids}
    adj_set_full: Dict[str, Set[str]] = {eid: set() for eid in normal_ids}
    for frm, dests in adj_set.items():
        if frm not in normal_ids:
            continue
        keep = sorted(d for d in dests if d in normal_ids)
        adj[frm] = keep
        adj_set_full[frm] = set(keep)
        for dst in keep:
            reverse_adj[dst].append(frm)
    for eid in reverse_adj:
        reverse_adj[eid].sort()

    return SumoGraph(
        net_sha256=sha256_file(net_path),
        max_inserted_edges=3,
        normal_ids=sorted(normal_ids),
        normal_set=set(normal_ids),
        internal_ids=sorted(internal_ids),
        length_of=length_of,
        edge_from=edge_from,
        edge_to=edge_to,
        nlanes_of=nlanes_of,
        speed_of=edge_speed,
        adj=adj,
        adj_set=adj_set_full,
        reverse_adj=reverse_adj,
        conn_dir={k: v for k, v in conn_dir.items() if k[0] in normal_ids and k[1] in normal_ids},
        lane_conns=lane_conns,
        n_conn_raw=n_conn_raw,
        n_conn_edge_pairs=len(conn_dir),
    )


def graph_from_synthetic(
    edges: Sequence[str],
    connections: Sequence[Tuple[str, str]],
    lengths: Optional[Dict[str, float]] = None,
    max_inserted_edges: int = 3,
) -> SumoGraph:
    """Build a tiny directed graph for unit tests."""
    normal_ids = list(edges)
    normal_set = set(normal_ids)
    adj_set: Dict[str, Set[str]] = {eid: set() for eid in normal_ids}
    conn_dir: Dict[Tuple[str, str], str] = {}
    for frm, to in connections:
        if frm not in normal_set or to not in normal_set:
            raise ValueError(f"synthetic connection uses unknown edge: {frm}->{to}")
        adj_set[frm].add(to)
        conn_dir[(frm, to)] = "s"
    adj = {eid: sorted(adj_set[eid]) for eid in normal_ids}
    reverse_adj: Dict[str, List[str]] = {eid: [] for eid in normal_ids}
    for frm, dests in adj.items():
        for dst in dests:
            reverse_adj[dst].append(frm)
    for eid in reverse_adj:
        reverse_adj[eid].sort()
    length_of = {eid: float((lengths or {}).get(eid, 1.0)) for eid in normal_ids}
    return SumoGraph(
        net_sha256="synthetic",
        max_inserted_edges=max_inserted_edges,
        normal_ids=list(normal_ids),
        normal_set=normal_set,
        internal_ids=[],
        length_of=length_of,
        edge_from={eid: "" for eid in normal_ids},
        edge_to={eid: "" for eid in normal_ids},
        nlanes_of={eid: 1 for eid in normal_ids},
        speed_of={eid: 13.89 for eid in normal_ids},
        adj=adj,
        adj_set={eid: set(adj[eid]) for eid in normal_ids},
        reverse_adj=reverse_adj,
        conn_dir=conn_dir,
        lane_conns=[],
        n_conn_raw=len(connections),
        n_conn_edge_pairs=len(conn_dir),
    )


@dataclass(slots=True)
class PairRecord:
    """Cached hop-shortest min-length bridges for one (src, dst)."""

    k: int
    insert_len: float
    inserted_list: List[Tuple[str, ...]]
    too_long_or_unreach: bool

    def to_bridge(self, observed: Optional[Sequence[str]] = None) -> Bridge:
        if self.too_long_or_unreach or not self.inserted_list:
            return UNREACHABLE
        if self.k == 0:
            return DIRECT
        observed_set = set(observed) if observed is not None else None

        def key(inserted: Tuple[str, ...]) -> tuple:
            overlap = 0
            if observed_set is not None:
                for edge in inserted:
                    if edge in observed_set:
                        overlap += 1
            return (overlap, inserted)

        chosen = min(self.inserted_list, key=key)
        n_equal = len(self.inserted_list)
        return Bridge(
            reachable=True,
            k=self.k,
            inserted=chosen,
            insert_len=self.insert_len,
            n_equal=n_equal,
            ambiguous=n_equal > 1,
            too_long_or_unreach=False,
        )


MAX_EQUAL_PATHS = 32


class BridgeCache:
    """Persistent (src, dst) lookup, limited to k <= max_inserted_edges."""

    def __init__(self, graph: SumoGraph, max_inserted_edges: int = 3):
        self.graph = graph
        self.max_inserted_edges = int(max_inserted_edges)
        self.net_sha256 = graph.net_sha256
        self._store: Dict[Tuple[str, str], PairRecord] = {}
        self._precomputed_sources: Set[str] = set()
        self.hits = 0
        self.misses = 0

    def cache_key_meta(self) -> dict:
        return {
            "net_sha256": self.net_sha256,
            "max_inserted_edges": self.max_inserted_edges,
        }

    def get(self, src: str, dst: str, observed: Optional[Sequence[str]] = None) -> Bridge:
        rec = self._get_record(src, dst)
        return rec.to_bridge(observed)

    def _get_record(self, src: str, dst: str) -> PairRecord:
        key = (src, dst)
        rec = self._store.get(key)
        if rec is not None:
            self.hits += 1
            return rec
        if src in self._precomputed_sources:
            self.hits += 1
            rec = PairRecord(k=-1, insert_len=0.0, inserted_list=[], too_long_or_unreach=True)
            self._store[key] = rec
            return rec
        self.misses += 1
        rec = self._search_record(src, dst)
        self._store[key] = rec
        return rec

    def _search_record(self, src: str, dst: str) -> PairRecord:
        graph = self.graph
        if src not in graph.normal_set or dst not in graph.normal_set or src == dst:
            return PairRecord(k=-1, insert_len=0.0, inserted_list=[], too_long_or_unreach=True)
        if dst in graph.adj_set.get(src, ()):
            return PairRecord(k=0, insert_len=0.0, inserted_list=[()], too_long_or_unreach=False)
        return _bounded_pair_record(graph, src, dst, self.max_inserted_edges)

    def precompute_all(self) -> int:
        """Fill the cache from every source with one depth-limited BFS."""
        graph = self.graph
        max_k = self.max_inserted_edges
        n_ok = 0
        for src in graph.normal_ids:
            records = _bounded_records_from_source(graph, src, max_k)
            for dst, rec in records.items():
                self._store[(src, dst)] = rec
                if not rec.too_long_or_unreach:
                    n_ok += 1
            self._precomputed_sources.add(src)
        return n_ok

    def dump(self) -> dict:
        payload = {
            "meta": self.cache_key_meta(),
            "n_pairs": len(self._store),
            "precomputed_sources": sorted(self._precomputed_sources),
            "pairs": {},
        }
        pairs = payload["pairs"]
        for (src, dst), rec in self._store.items():
            if rec.too_long_or_unreach:
                continue
            pairs[f"{src}\t{dst}"] = {
                "k": rec.k,
                "insert_len": rec.insert_len,
                "inserted_list": [list(x) for x in rec.inserted_list],
                "too_long_or_unreach": rec.too_long_or_unreach,
            }
        return payload

    def load_pairs(self, payload: dict) -> None:
        meta = payload.get("meta") or {}
        if meta.get("net_sha256") != self.net_sha256:
            raise ValueError("bridge cache net_sha256 mismatch")
        if int(meta.get("max_inserted_edges", -1)) != self.max_inserted_edges:
            raise ValueError("bridge cache max_inserted_edges mismatch")
        self._precomputed_sources = set(payload.get("precomputed_sources") or [])
        for key, rec in (payload.get("pairs") or {}).items():
            src, dst = key.split("\t", 1)
            self._store[(src, dst)] = PairRecord(
                k=int(rec["k"]),
                insert_len=float(rec.get("insert_len") or 0.0),
                inserted_list=[tuple(x) for x in rec.get("inserted_list") or []],
                too_long_or_unreach=bool(rec.get("too_long_or_unreach")),
            )


def _bounded_pair_record(graph: SumoGraph, src: str, dst: str, max_k: int) -> PairRecord:
    records = _bounded_records_from_source(graph, src, max_k, stop_at=dst)
    rec = records.get(dst)
    if rec is None:
        return PairRecord(k=-1, insert_len=0.0, inserted_list=[], too_long_or_unreach=True)
    return rec


def _bounded_records_from_source(
    graph: SumoGraph,
    src: str,
    max_k: int,
    stop_at: Optional[str] = None,
) -> Dict[str, PairRecord]:
    """Depth-limited hop DAG from src. k inserted = hops-1, hops <= max_k+1."""
    adj = graph.adj
    length_of = graph.length_of
    max_hops = max_k + 1
    dist: Dict[str, int] = {src: 0}
    parents: Dict[str, List[str]] = {src: []}
    queue: deque[str] = deque([src])
    while queue:
        u = queue.popleft()
        du = dist[u]
        if du >= max_hops:
            continue
        if stop_at is not None and stop_at in dist and du >= dist[stop_at]:
            continue
        for v in adj.get(u, ()):
            nd = du + 1
            if v == src:
                continue
            if v not in dist:
                dist[v] = nd
                parents[v] = [u]
                if nd < max_hops:
                    queue.append(v)
            elif dist[v] == nd:
                parents[v].append(u)

    targets = [stop_at] if stop_at is not None else [node for node in dist if node != src]
    out: Dict[str, PairRecord] = {}
    for dst in targets:
        if dst is None or dst not in dist:
            continue
        hops = dist[dst]
        k = hops - 1
        if k > max_k:
            continue
        if k == 0:
            out[dst] = PairRecord(k=0, insert_len=0.0, inserted_list=[()], too_long_or_unreach=False)
            continue
        best_len, paths, n_equal = _best_inserted_on_dag(parents, length_of, src, dst, dist)
        if not paths:
            continue
        out[dst] = PairRecord(
            k=k,
            insert_len=best_len,
            inserted_list=paths,
            too_long_or_unreach=False,
        )
    return out


def _best_inserted_on_dag(
    parents: Dict[str, List[str]],
    length_of: Dict[str, float],
    src: str,
    dst: str,
    dist: Dict[str, int],
) -> Tuple[float, List[Tuple[str, ...]], int]:
    """Unique min-length hop-shortest inserted sequences, with a small cap."""
    # DP over nodes by increasing distance. State at v (v != dst): path from src to v
    # including v as an inserted edge. State at dst: inserted = path[1:-1].
    best_len: Dict[str, float] = {src: 0.0}
    best_paths: Dict[str, List[Tuple[str, ...]]] = {src: [()]}
    layers: Dict[int, List[str]] = defaultdict(list)
    for node, d in dist.items():
        layers[d].append(node)
    max_d = dist[dst]
    for d in range(1, max_d + 1):
        for v in layers.get(d, ()):
            cand_len = None
            cand_paths: List[Tuple[str, ...]] = []
            n_equal = 0
            for u in parents.get(v, ()):
                if dist.get(u) != d - 1:
                    continue
                for prefix in best_paths.get(u, ()):
                    if v == dst:
                        inserted = prefix
                        ins_len = best_len[u]
                    else:
                        inserted = prefix + (v,)
                        ins_len = best_len[u] + float(length_of.get(v, 0.0))
                    if cand_len is None or ins_len + 1e-12 < cand_len:
                        cand_len = ins_len
                        cand_paths = [inserted]
                        n_equal = 1
                    elif abs(ins_len - cand_len) <= 1e-12:
                        n_equal += 1
                        if inserted not in cand_paths and len(cand_paths) < MAX_EQUAL_PATHS:
                            cand_paths.append(inserted)
            if cand_len is None:
                continue
            cand_paths.sort()
            best_len[v] = cand_len
            best_paths[v] = cand_paths
    paths = best_paths.get(dst) or []
    n_equal = len(paths)
    return float(best_len.get(dst, 0.0)), paths, n_equal


@dataclass
class RepairResult:
    status: str
    reject_reason: Optional[str]
    repaired_route: Optional[List[str]]
    kept_indices: List[int]
    deleted_indices: List[int]
    deleted_ids: List[str]
    inserted_ids: List[str]
    inserted_len: float
    repair_block_count: int
    max_inserted_single: int
    original_first: Optional[str]
    original_last: Optional[str]
    repaired_first: Optional[str]
    repaired_last: Optional[str]
    endpoint_deleted: bool
    introduced_cycle: bool
    ambiguous_bridge: bool
    quality_flags: List[str]
    unsolved_pair: Optional[List[str]]
    algorithm_status: str
    original_route_length: int
    repaired_route_length: int
    original_edge_count_retained: int
    deleted_observed_edge_count: int
    inserted_edge_count: int
    action: str


def _route_has_cycle(route: Sequence[str]) -> bool:
    seen: Set[str] = set()
    for edge in route:
        if edge in seen:
            return True
        seen.add(edge)
    return False


def _better(a: tuple, b: tuple) -> bool:
    """True if score tuple a is strictly better than b.

    Numeric prefix is maximized (minimization objectives are already negated).
    The last two fields (route ids, kept indices) are lexicographically minimized.
    """
    *a_num, a_route, a_idx = a
    *b_num, b_route, b_idx = b
    if a_num != b_num:
        return a_num > b_num
    if a_route != b_route:
        return a_route < b_route
    if a_idx != b_idx:
        return a_idx < b_idx
    return False


def repair_route(
    route: Sequence[str],
    graph: SumoGraph,
    cache: BridgeCache,
    min_route_edges: int = 2,
) -> RepairResult:
    original = [str(e) for e in route]
    n = len(original)
    original_first = original[0] if original else None
    original_last = original[-1] if original else None

    def fail(reason: str, algo: str, unsolved: Optional[List[str]] = None) -> RepairResult:
        return RepairResult(
            status="rejected",
            reject_reason=reason,
            repaired_route=None,
            kept_indices=[],
            deleted_indices=list(range(n)),
            deleted_ids=list(original),
            inserted_ids=[],
            inserted_len=0.0,
            repair_block_count=0,
            max_inserted_single=0,
            original_first=original_first,
            original_last=original_last,
            repaired_first=None,
            repaired_last=None,
            endpoint_deleted=bool(n),
            introduced_cycle=False,
            ambiguous_bridge=False,
            quality_flags=[],
            unsolved_pair=unsolved,
            algorithm_status=algo,
            original_route_length=n,
            repaired_route_length=0,
            original_edge_count_retained=0,
            deleted_observed_edge_count=n,
            inserted_edge_count=0,
            action="rejected",
        )

    if n == 0:
        return fail("empty_route", "empty_route")

    unknown = [e for e in original if e not in graph.normal_set]
    if unknown:
        return fail("unknown_edge_id", "unknown_edge_id", [unknown[0]])

    # Pair lookup with observed-edge overlap for tie-breaks.
    def bridge_at(i: int, j: int) -> Bridge:
        return cache.get(original[i], original[j], observed=original)

    # DP: best chain ending at index i.
    # pred[i] = previous kept index or -1 if i starts the chain.
    best_retained = [1] * n
    best_keep_first = [0] * n
    best_inserted = [0] * n
    best_ins_len = [0.0] * n
    best_blocks = [0] * n
    best_route: List[Tuple[str, ...]] = [(original[i],) for i in range(n)]
    best_idx: List[Tuple[int, ...]] = [(i,) for i in range(n)]
    pred = [-1] * n
    pair_bridge: List[Optional[Bridge]] = [None] * n
    best_keep_first[0] = 1

    for i in range(n):
        # Starting a new chain at i.
        keep_first_i = 1 if i == 0 else 0
        cand_route_start = (original[i],)
        cand_idx_start = (i,)
        start_score = (
            1,
            keep_first_i,
            0,
            0,
            -0.0,
            0,
            cand_route_start,
            cand_idx_start,
        )
        chosen_score = start_score
        chosen_pred = -1
        chosen_br: Optional[Bridge] = None

        for p in range(i):
            br = bridge_at(p, i)
            if not br.ok:
                continue
            retained = best_retained[p] + 1
            keep_first = best_keep_first[p]
            inserted = best_inserted[p] + br.k
            ins_len = best_ins_len[p] + br.insert_len
            blocks = best_blocks[p] + (1 if br.k > 0 else 0)
            route_ids = best_route[p] + br.inserted + (original[i],)
            idx_ids = best_idx[p] + (i,)
            score = (
                retained,
                keep_first,
                0,
                -inserted,
                -ins_len,
                -blocks,
                route_ids,
                idx_ids,
            )
            if _better(score, chosen_score):
                chosen_score = score
                chosen_pred = p
                chosen_br = br

        if chosen_pred < 0:
            best_retained[i] = 1
            best_keep_first[i] = keep_first_i
            best_inserted[i] = 0
            best_ins_len[i] = 0.0
            best_blocks[i] = 0
            best_route[i] = cand_route_start
            best_idx[i] = cand_idx_start
            pred[i] = -1
            pair_bridge[i] = None
        else:
            br = chosen_br
            assert br is not None
            p = chosen_pred
            best_retained[i] = best_retained[p] + 1
            best_keep_first[i] = best_keep_first[p]
            best_inserted[i] = best_inserted[p] + br.k
            best_ins_len[i] = best_ins_len[p] + br.insert_len
            best_blocks[i] = best_blocks[p] + (1 if br.k > 0 else 0)
            best_route[i] = best_route[p] + br.inserted + (original[i],)
            best_idx[i] = best_idx[p] + (i,)
            pred[i] = p
            pair_bridge[i] = br

    # Choose globally best complete chain (at least min_route_edges).
    best_end = -1
    best_final: Optional[tuple] = None
    for i in range(n):
        if best_retained[i] < min_route_edges:
            continue
        keep_last = 1 if i == n - 1 else 0
        score = (
            best_retained[i],
            best_keep_first[i],
            keep_last,
            -best_inserted[i],
            -best_ins_len[i],
            -best_blocks[i],
            best_route[i],
            best_idx[i],
        )
        if best_final is None or _better(score, best_final):
            best_final = score
            best_end = i

    if best_end < 0:
        # Identify a representative unsolved pair of original consecutive edges.
        unsolved = None
        for i in range(n - 1):
            br = bridge_at(i, i + 1)
            if not br.ok:
                unsolved = [original[i], original[i + 1]]
                break
        return fail("no_repairable_subsequence", "no_repairable_subsequence", unsolved)

    kept_indices = list(best_idx[best_end])
    repaired = list(best_route[best_end])
    kept_set = set(kept_indices)
    deleted_indices = [i for i in range(n) if i not in kept_set]
    deleted_ids = [original[i] for i in deleted_indices]
    inserted_ids: List[str] = []
    max_k = 0
    amb = False
    for t in range(1, len(kept_indices)):
        br = cache.get(
            original[kept_indices[t - 1]],
            original[kept_indices[t]],
            observed=original,
        )
        if br.inserted:
            inserted_ids.extend(br.inserted)
        if br.k > max_k:
            max_k = br.k
        if br.ambiguous:
            amb = True

    if len(repaired) < min_route_edges:
        return fail("fewer_than_two_edges_after_repair", "fewer_than_two_edges_after_repair")

    # Safety: every adjacent pair in the repaired route must be a SUMO connection.
    for a, b in zip(repaired, repaired[1:]):
        if not graph.has_connection(a, b):
            return fail(
                "no_repairable_subsequence",
                "postcheck_illegal_pair",
                [a, b],
            )
        if a.startswith(":") or b.startswith(":"):
            return fail("no_repairable_subsequence", "internal_edge_in_route", [a, b])

    endpoint_deleted = (0 not in kept_set) or ((n - 1) not in kept_set)
    introduced_cycle = _route_has_cycle(repaired)
    deleted_count = len(deleted_indices)
    inserted_count = len(inserted_ids)
    ratio = (deleted_count / n) if n else 0.0
    flags: List[str] = []
    if ratio > 0.2:
        flags.append("deleted_ratio_gt_0.2")
    if ratio > 0.5:
        flags.append("deleted_ratio_gt_0.5")
    if endpoint_deleted:
        flags.append("endpoint_deleted")
    if introduced_cycle:
        flags.append("introduced_cycle")
    if amb:
        flags.append("ambiguous_bridge")

    if deleted_count == 0 and inserted_count == 0:
        action = "direct_legal"
    elif deleted_count == 0 and inserted_count > 0:
        action = "insertion_only"
    elif deleted_count > 0 and inserted_count == 0:
        action = "deletion_only"
    else:
        action = "deletion_insertion"

    return RepairResult(
        status="repaired",
        reject_reason=None,
        repaired_route=repaired,
        kept_indices=kept_indices,
        deleted_indices=deleted_indices,
        deleted_ids=deleted_ids,
        inserted_ids=inserted_ids,
        inserted_len=float(best_ins_len[best_end]),
        repair_block_count=int(best_blocks[best_end]),
        max_inserted_single=int(max_k),
        original_first=original_first,
        original_last=original_last,
        repaired_first=repaired[0],
        repaired_last=repaired[-1],
        endpoint_deleted=endpoint_deleted,
        introduced_cycle=introduced_cycle,
        ambiguous_bridge=amb,
        quality_flags=flags,
        unsolved_pair=None,
        algorithm_status="ok",
        original_route_length=n,
        repaired_route_length=len(repaired),
        original_edge_count_retained=len(kept_indices),
        deleted_observed_edge_count=deleted_count,
        inserted_edge_count=inserted_count,
        action=action,
    )


def depart_time_invalid(record: dict, limit: int = 86400) -> bool:
    try:
        start = record.get("startTime")
        if start is None:
            return True
        return float(start) >= float(limit) or float(start) < 0
    except (TypeError, ValueError):
        return True


def copy_record_with_route(record: dict, route: List[str]) -> dict:
    out = {}
    for key, value in record.items():
        if key == "route":
            out[key] = list(route)
        else:
            out[key] = value
    if "route" not in out:
        out["route"] = list(route)
    return out


def fields_except_route_equal(a: dict, b: dict) -> bool:
    keys = set(a) | set(b)
    for key in keys:
        if key == "route":
            continue
        if a.get(key) != b.get(key):
            return False
    return True


def iter_json_array(path: Path) -> Iterator[dict]:
    """Stream objects from a top-level JSON array without loading the whole file."""
    decoder = json.JSONDecoder()
    with Path(path).open("r", encoding="utf-8") as fh:
        buf = ""
        idx = 0
        in_array = False
        first = True

        def fill() -> bool:
            nonlocal buf, idx
            chunk = fh.read(1 << 20)
            if not chunk:
                return False
            if idx:
                buf = buf[idx:]
                idx = 0
            buf += chunk
            return True

        while True:
            if idx >= len(buf):
                if not fill():
                    return
            ch = buf[idx]
            if ch.isspace():
                idx += 1
                continue
            if not in_array:
                if ch != "[":
                    raise ValueError(f"{path} is not a JSON array")
                in_array = True
                idx += 1
                continue
            if ch == "]":
                return
            if ch == ",":
                idx += 1
                first = False
                continue
            while True:
                try:
                    obj, end = decoder.raw_decode(buf, idx)
                    idx = end
                    if not isinstance(obj, dict):
                        raise ValueError(f"non-object record in {path}")
                    yield obj
                    break
                except json.JSONDecodeError:
                    if not fill():
                        raise


class AtomicJsonArrayWriter:
    def __init__(self, final_path: Path, tmp_dir: Path):
        self.final_path = Path(final_path)
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=self.final_path.name + ".",
            suffix=".partial",
            dir=str(self.tmp_dir),
        )
        os.close(fd)
        self.tmp_path = Path(tmp_name)
        self._fh = self.tmp_path.open("w", encoding="utf-8")
        self._fh.write("[\n")
        self._count = 0
        self._committed = False

    def write(self, obj: dict) -> None:
        if self._count:
            self._fh.write(",\n")
        json.dump(obj, self._fh, ensure_ascii=False, separators=(",", ":"))
        self._count += 1

    @property
    def count(self) -> int:
        return self._count

    def abort(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
        if self.tmp_path.exists():
            self.tmp_path.unlink()

    def commit(self) -> None:
        self._fh.write("\n]\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        os.replace(self.tmp_path, self.final_path)
        self._committed = True


class AtomicGzipJsonlWriter:
    def __init__(self, final_path: Path, tmp_dir: Path):
        self.final_path = Path(final_path)
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=self.final_path.name + ".",
            suffix=".partial",
            dir=str(self.tmp_dir),
        )
        os.close(fd)
        self.tmp_path = Path(tmp_name)
        self._fh = gzip.open(self.tmp_path, "wt", encoding="utf-8")
        self._count = 0

    def write(self, obj: dict) -> None:
        self._fh.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._count += 1

    @property
    def count(self) -> int:
        return self._count

    def abort(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
        if self.tmp_path.exists():
            self.tmp_path.unlink()

    def commit(self) -> None:
        self._fh.flush()
        self._fh.close()
        os.replace(self.tmp_path, self.final_path)


def iter_jsonl_gz(path: Path) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def result_to_metadata(
    *,
    trip_id: str,
    date: str,
    source_file: str,
    source_record_index: int,
    result: RepairResult,
    reject_reason: Optional[str],
    status: str,
) -> dict:
    return {
        "trip_id": trip_id,
        "date": date,
        "source_file": source_file,
        "source_record_index": source_record_index,
        "status": status,
        "original_route_length": result.original_route_length,
        "repaired_route_length": result.repaired_route_length,
        "original_edge_count_retained": result.original_edge_count_retained,
        "deleted_observed_edge_count": result.deleted_observed_edge_count,
        "deleted_observed_edge_indices": result.deleted_indices,
        "deleted_observed_edge_ids": result.deleted_ids,
        "inserted_edge_count": result.inserted_edge_count,
        "inserted_edge_ids": result.inserted_ids,
        "repair_block_count": result.repair_block_count,
        "max_inserted_edges_single_bridge": result.max_inserted_single,
        "inserted_length_m": result.inserted_len,
        "original_first_edge": result.original_first,
        "original_last_edge": result.original_last,
        "repaired_first_edge": result.repaired_first,
        "repaired_last_edge": result.repaired_last,
        "endpoint_deleted": result.endpoint_deleted,
        "introduced_cycle": result.introduced_cycle,
        "ambiguous_bridge": result.ambiguous_bridge,
        "quality_flags": result.quality_flags,
        "reject_reason": reject_reason,
        "action": result.action,
        "algorithm_status": result.algorithm_status,
        "kept_indices": result.kept_indices,
        "unsolved_pair": result.unsolved_pair,
    }


def scan_code_files(root: Path) -> List[str]:
    found = []
    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in CODE_SUFFIXES or path.name in CODE_NAMES:
            found.append(str(path))
    return found


def scan_large_data_in_code(code_root: Path, min_bytes: int = 10 * 1024 * 1024) -> List[str]:
    found = []
    for path in Path(code_root).rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        size = path.stat().st_size
        if size >= min_bytes:
            found.append(f"{path} ({size} bytes)")
            continue
        if path.suffix.lower() in {".parquet"} or path.name.endswith(".net.xml"):
            found.append(str(path))
    return found


def save_pickle_atomic(obj: Any, final_path: Path, tmp_dir: Path) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=final_path.name + ".", suffix=".partial", dir=str(tmp_dir))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, final_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def load_pickle(path: Path) -> Any:
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def write_json_atomic(obj: Any, final_path: Path, tmp_dir: Path) -> None:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=final_path.name + ".", suffix=".partial", dir=str(tmp_dir))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, final_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
