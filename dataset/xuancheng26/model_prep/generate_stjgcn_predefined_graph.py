#!/usr/bin/env python3
"""Build PeMSD7-style predefined graphs for STJGCN on the Xuancheng R graph.

Distances are the directed centre distances already used by DCRNN: the shortest
legal SUMO connection, ``0.5 * source lane + internal via + 0.5 * target lane``.
Non-edges and the diagonal stay infinite. Sigma is the population standard
deviation of those finite distances. The kernel, threshold, and separate
degree normalization of ``A`` and ``A.T`` follow PeMSD7. This does not add
all-pairs shortest paths, and it does not modify the shared graph files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import prep_paths
from generate_dcrnn_directed_adjacency import directional_distances
from generate_stgcn_weighted_adjacency import (
    build_lane_lookups,
    index_connections,
    load_r_graph,
    parse_sumo_net,
    validate_node_order,
)
import generate_stgcn_weighted_adjacency as stgcn
from reimplementation.stjgcn.graph import build_predefined_kernels, node_order_sha256


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Xuancheng26 STJGCN predefined graphs")
    prep_paths.add_3s_path_arguments(parser, net=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--delta-pdf", type=float, default=0.9)
    parser.add_argument("--horizon", type=int, default=12)
    args = parser.parse_args(argv)
    training_root = prep_paths.refuse_code_root(args.training_root)
    graph_dir = training_root / "graph"
    output_dir = prep_paths.refuse_code_root(graph_dir / "stjgcn")
    archive_path = output_dir / "predefined_stjg.npz"
    metadata_path = output_dir / "metadata.json"
    if archive_path.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {archive_path}; pass --overwrite")
    net_file = prep_paths.resolved(args.net_file)
    stgcn.GRAPH_DIR = graph_dir
    stgcn.R_GRAPH = graph_dir / "r_graph.npz"
    stgcn.NODE_MAPPING = graph_dir / "node_mapping.csv"
    stgcn.R_NODES = graph_dir / "r_nodes.csv"
    stgcn.NET_FILE = net_file
    edge_ids, directed = load_r_graph()
    validate_node_order(edge_ids)
    edges, connections = parse_sumo_net(net_file)
    lanes_by_id, edge_of_lane = build_lane_lookups(edges)
    connection_index = index_connections(connections, set(edge_ids))
    directed_edges, _unused_median = directional_distances(
        edge_ids, directed, edges, connection_index, lanes_by_id, edge_of_lane
    )
    distance = np.full((len(edge_ids), len(edge_ids)), np.inf, dtype=np.float64)
    for src, dst, value in directed_edges:
        if not np.isfinite(value) or value <= 0.0:
            raise SystemExit(f"directed distance {edge_ids[src]}->{edge_ids[dst]} is invalid: {value}")
        distance[int(src), int(dst)] = float(value)
    if int(np.isfinite(distance).sum()) != len(directed_edges):
        raise SystemExit("finite distance count does not match the directed R edges")
    forward, backward, stats = build_predefined_kernels(distance, horizon=int(args.horizon), delta_pdf=float(args.delta_pdf))
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_name(f".{archive_path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, A_fw=forward, A_bw=backward, distance=distance.astype(np.float32))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, archive_path)
    metadata = {
        "model": "stjgcn",
        "source": "PeMSD7 utils.load_data kernel on Xuancheng directed centre distances",
        "distance_definition": (
            "Directed centre distance of the shortest legal SUMO connection: "
            "0.5 * source lane length + internal via lengths + 0.5 * target lane length. "
            "This is the existing project definition from direct_center_distance. "
            "Non-edges and the diagonal are inf. No all-pairs shortest path was added."
        ),
        "xuancheng_adaptation": True,
        "num_nodes": len(edge_ids),
        "node_order": "r_graph.npz edge_ids, matching node_mapping.csv and r_nodes.csv",
        "node_order_sha256": node_order_sha256(edge_ids),
        "directed_edges": len(directed_edges),
        "horizon": int(args.horizon),
        "delta_pdf": float(args.delta_pdf),
        "sigma_population": stats["sigma_population"],
        "distance_min_m": stats["distance_min_m"],
        "distance_max_m": stats["distance_max_m"],
        "distance_mean_m": stats["distance_mean_m"],
        "finite_distances": stats["finite_distances"],
        "normalization": stats["normalization"],
        "per_lag": stats["per_lag"],
        "net_file": net_file.as_posix(),
        "output": archive_path.as_posix(),
        "traffic_counts_used": False,
    }
    meta_tmp = metadata_path.with_name(f".{metadata_path.name}.tmp")
    meta_tmp.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(meta_tmp, metadata_path)
    print(json.dumps({"sigma": stats["sigma_population"], "per_lag": stats["per_lag"]}, indent=2))
    print(f"wrote {archive_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
