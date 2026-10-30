#!/usr/bin/env python3
"""Build the directed R-level road graph from the final Xuancheng26 SUMO network.

Nodes are real SUMO edges in network-file order. A directed link exists only
when the net file contains at least one real edge-to-edge connection.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from xml.parsers import expat

import numpy as np

import generate_r_level_full_flow as full
import prep_paths

NET_FILE = full.NET_FILE
FLOW_ROOT = full.OUT_ROOT
OUT_PATH = prep_paths.DEFAULT_TRAINING_ROOT / "graph" / "r_graph.npz"


def log(message: str) -> None:
    print(message, flush=True)


def is_internal(edge_id: str) -> bool:
    return edge_id.startswith(":")


def collect_connections(net_path: Path, real_edges: set[str]) -> tuple[int, set[tuple[str, str]]]:
    """Return the lane-connection count and unique real edge-to-edge links."""
    lane_connections = 0
    links: set[tuple[str, str]] = set()

    def start(name: str, attrs) -> None:
        nonlocal lane_connections
        if name != "connection":
            return
        lane_connections += 1
        source = attrs.get("from", "")
        target = attrs.get("to", "")
        if source in real_edges and target in real_edges:
            links.add((source, target))

    parser = expat.ParserCreate()
    parser.StartElementHandler = start
    with net_path.open("rb") as handle:
        parser.ParseFile(handle)
    return lane_connections, links


def weakly_connected(adjacency: np.ndarray) -> tuple[int, list[int], list[list[int]]]:
    size = adjacency.shape[0]
    undirected = np.logical_or(adjacency, adjacency.T)
    visited = np.zeros(size, dtype=bool)
    sizes: list[int] = []
    members: list[list[int]] = []
    for start in range(size):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        group = []
        while stack:
            node = stack.pop()
            group.append(node)
            for nxt in np.flatnonzero(undirected[node]):
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(int(nxt))
        sizes.append(len(group))
        members.append(group)
    return len(sizes), sizes, members


def strongly_connected_count(adjacency: np.ndarray) -> int:
    size = adjacency.shape[0]
    visited = np.zeros(size, dtype=bool)
    finish: list[int] = []
    for start in range(size):
        if visited[start]:
            continue
        stack = [(start, False)]
        visited[start] = True
        while stack:
            node, done = stack.pop()
            if done:
                finish.append(node)
                continue
            stack.append((node, True))
            for nxt in np.flatnonzero(adjacency[node]):
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append((int(nxt), False))
    visited[:] = False
    count = 0
    for start in reversed(finish):
        if visited[start]:
            continue
        count += 1
        stack = [start]
        visited[start] = True
        while stack:
            node = stack.pop()
            for nxt in np.flatnonzero(adjacency[:, node]):
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(int(nxt))
    return count


def check_flow_columns(edge_ids: list[str]) -> tuple[int, list[str]]:
    graph_ids = set(edge_ids)
    missing: list[str] = []
    checked = 0
    for path in sorted(FLOW_ROOT.glob("2023_04_*/inflow.csv")):
        checked += 1
        with path.open(encoding="utf-8") as handle:
            header = handle.readline().rstrip("\n").split(",")[2:]
        for edge_id in header:
            if edge_id not in graph_ids and edge_id not in missing:
                missing.append(edge_id)
    return checked, missing


def build_graph() -> dict[str, object]:
    edge_ids, _lane_to_edge, _internal_lanes = full.load_network(NET_FILE)
    real_edges = set(edge_ids)
    if len(real_edges) != len(edge_ids):
        raise RuntimeError("duplicate real edge ids in the network")
    if any(is_internal(edge_id) for edge_id in edge_ids):
        raise RuntimeError("internal edge entered the node list")
    lane_connections, links = collect_connections(NET_FILE, real_edges)
    index = {edge_id: position for position, edge_id in enumerate(edge_ids)}
    adjacency = np.zeros((len(edge_ids), len(edge_ids)), dtype=np.uint8)
    for source, target in links:
        adjacency[index[source], index[target]] = 1
    if int(adjacency.sum()) != len(links):
        raise RuntimeError("adjacency sum does not match unique real connections")
    return {
        "edge_ids": edge_ids,
        "adjacency": adjacency,
        "lane_connections": lane_connections,
        "links": links,
    }


def save_graph(edge_ids: list[str], adjacency: np.ndarray) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT_PATH.with_name("r_graph.tmp.npz")
    np.savez_compressed(
        temporary,
        edge_ids=np.array(edge_ids, dtype=np.str_),
        adjacency=adjacency,
        num_nodes=np.int64(len(edge_ids)),
        num_graph_edges=np.int64(adjacency.sum()),
    )
    temporary.replace(OUT_PATH)


def reload_and_check(edge_ids: list[str], adjacency: np.ndarray) -> None:
    with np.load(OUT_PATH, allow_pickle=False) as payload:
        loaded_ids = payload["edge_ids"]
        loaded_adj = payload["adjacency"]
        loaded_nodes = int(payload["num_nodes"])
        loaded_edges = int(payload["num_graph_edges"])
    if loaded_ids.shape != (len(edge_ids),):
        raise RuntimeError(f"reloaded edge_ids shape {loaded_ids.shape}")
    if loaded_ids.dtype.kind not in {"U", "S", "O"}:
        raise RuntimeError(f"edge_ids dtype is {loaded_ids.dtype}, expected strings")
    restored = [str(value) for value in loaded_ids.tolist()]
    if restored != edge_ids:
        raise RuntimeError("reloaded edge ids differ from the network ids")
    if any("_" in edge_id and edge_id not in restored for edge_id in edge_ids if "_" in edge_id):
        raise RuntimeError("an underscore edge id was lost")
    if loaded_adj.shape != adjacency.shape or loaded_adj.dtype != np.uint8:
        raise RuntimeError("reloaded adjacency has an unexpected shape or dtype")
    if not np.array_equal(loaded_adj, adjacency):
        raise RuntimeError("reloaded adjacency differs")
    if loaded_nodes != len(edge_ids) or loaded_edges != int(adjacency.sum()):
        raise RuntimeError("reloaded counts differ")
    if int(loaded_adj.sum()) != loaded_edges:
        raise RuntimeError("reloaded adjacency sum differs from num_graph_edges")


def summarize(edge_ids: list[str], adjacency: np.ndarray, lane_connections: int, links: set[tuple[str, str]]) -> None:
    num_nodes = len(edge_ids)
    num_graph_edges = int(adjacency.sum())
    values = set(np.unique(adjacency).tolist())
    if values - {0, 1}:
        raise RuntimeError(f"adjacency contains values other than 0 and 1: {values}")
    if not np.isfinite(adjacency).all():
        raise RuntimeError("adjacency contains non-finite values")
    if num_graph_edges <= 0:
        raise RuntimeError("graph has no directed edges")
    if num_graph_edges != len(links):
        raise RuntimeError(
            f"graph edges {num_graph_edges} != unique real connections {len(links)}"
        )
    out_degree = adjacency.sum(axis=1)
    in_degree = adjacency.sum(axis=0)
    zero_in = np.flatnonzero(in_degree == 0)
    zero_out = np.flatnonzero(out_degree == 0)
    isolated = np.flatnonzero((in_degree == 0) & (out_degree == 0))
    both = np.logical_and(adjacency, adjacency.T)
    reciprocal_pairs = int(np.triu(both, k=1).sum())
    one_way = num_graph_edges - int(both.sum())
    weak_count, weak_sizes, weak_members = weakly_connected(adjacency)
    strong_count = strongly_connected_count(adjacency)
    density = num_graph_edges / (num_nodes * num_nodes)
    checked, missing_flow_edges = check_flow_columns(edge_ids)
    log("R Graph Summary")
    log(f"final net file: {NET_FILE}")
    log(f"num_nodes: {num_nodes}")
    log(f"num_lane_connections: {lane_connections}")
    log(f"num_unique_edge_connections: {len(links)}")
    log(f"num_graph_edges: {num_graph_edges}")
    log(f"graph density: {density:.8f}")
    log(f"reciprocal pair count: {reciprocal_pairs}")
    log(f"one-way connection count: {one_way}")
    log(f"self-loops: {int(np.trace(adjacency))}")
    log(
        "in-degree min/mean/max: "
        f"{int(in_degree.min())}/{float(in_degree.mean()):.3f}/{int(in_degree.max())}"
    )
    log(
        "out-degree min/mean/max: "
        f"{int(out_degree.min())}/{float(out_degree.mean()):.3f}/{int(out_degree.max())}"
    )
    log(f"zero in-degree count: {len(zero_in)}")
    log(f"zero out-degree count: {len(zero_out)}")
    log(f"isolated node count: {len(isolated)}")
    log(f"weakly connected component count: {weak_count}")
    log(f"strongly connected component count: {strong_count}")
    log(f"output path: {OUT_PATH}")
    if len(zero_in):
        log("zero in-degree examples: " + ", ".join(edge_ids[int(i)] for i in zero_in[:8]))
    if len(zero_out):
        log("zero out-degree examples: " + ", ".join(edge_ids[int(i)] for i in zero_out[:8]))
    if len(isolated):
        log("isolated examples: " + ", ".join(edge_ids[int(i)] for i in isolated[:8]))
    if weak_count > 1:
        ordered = sorted(zip(weak_sizes, weak_members), key=lambda item: item[0])
        log(
            "weak component sizes: "
            + ", ".join(str(size) for size in sorted(weak_sizes, reverse=True)[:12])
        )
        for size, members in ordered[:8]:
            if size == max(weak_sizes):
                continue
            log(
                f"small component size={size}: "
                + ", ".join(edge_ids[node] for node in members[:6])
            )
    log(f"full-flow files checked: {checked}")
    if missing_flow_edges:
        log("flow edges missing from graph: " + ", ".join(missing_flow_edges[:20]))
        raise RuntimeError("a flow column is not a graph node")
    log("flow edge ids are a subset of graph edge ids")
    log("R-level directed road graph generated successfully.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Xuancheng26 directed R-level road graph")
    prep_paths.add_path_arguments(parser, net=True)
    args = parser.parse_args(argv)
    root = prep_paths.resolved(args.training_root)
    net_file = prep_paths.resolved(args.net_file)
    flow_root = root / "full_flow" / "r-level"
    prep_paths.bind(
        sys.modules[__name__],
        NET_FILE=net_file,
        FLOW_ROOT=flow_root,
        OUT_PATH=root / "graph" / "r_graph.npz",
    )
    prep_paths.bind(full, NET_FILE=net_file, OUT_ROOT=flow_root)
    log(f"net_file={NET_FILE}")
    log(f"output_path={OUT_PATH}")
    built = build_graph()
    edge_ids = built["edge_ids"]
    adjacency = built["adjacency"]
    save_graph(edge_ids, adjacency)
    reload_and_check(edge_ids, adjacency)
    summarize(edge_ids, adjacency, int(built["lane_connections"]), built["links"])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
