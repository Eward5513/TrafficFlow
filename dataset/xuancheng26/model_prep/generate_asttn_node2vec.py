#!/usr/bin/env python3
"""Build a frozen 128-d node2vec table for ASTTN from the static R graph.

The reference repository does not ship a dimension header for Xuancheng, so
the width is the explicit adaptation recorded in the metadata. Walks use only
the directed R adjacency and seed 42. This script does not read traffic counts
and does not modify the shared graph files.
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
from reimplementation.asttn.node2vec import NODE2VEC_SETTINGS, out_neighbors, random_walks, train_skipgram
from reimplementation.stjgcn.graph import node_order_sha256

import generate_stgcn_weighted_adjacency as stgcn
from generate_stgcn_weighted_adjacency import load_r_graph, validate_node_order


def _atomic_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Xuancheng26 ASTTN node2vec embedding")
    prep_paths.add_3s_path_arguments(parser)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    training_root = prep_paths.refuse_code_root(args.training_root)
    graph_dir = training_root / "graph"
    output_dir = prep_paths.refuse_code_root(graph_dir / "asttn")
    embedding_path = output_dir / "node2vec_r128.npy"
    metadata_path = output_dir / "metadata.json"
    if embedding_path.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {embedding_path}; pass --overwrite")
    stgcn.GRAPH_DIR = graph_dir
    stgcn.R_GRAPH = graph_dir / "r_graph.npz"
    stgcn.NODE_MAPPING = graph_dir / "node_mapping.csv"
    stgcn.R_NODES = graph_dir / "r_nodes.csv"
    edge_ids, adjacency = load_r_graph()
    validate_node_order(edge_ids)
    neighbors = out_neighbors(adjacency)
    empty = sum(int(item.size == 0) for item in neighbors)
    print(f"nodes={len(edge_ids)} directed_edges={int(adjacency.sum())} nodes_without_out_edge={empty}", flush=True)
    walks = random_walks(
        neighbors,
        num_walks=int(NODE2VEC_SETTINGS["num_walks"]),
        walk_length=int(NODE2VEC_SETTINGS["walk_length"]),
        seed=int(NODE2VEC_SETTINGS["seed"]),
    )
    embedding = train_skipgram(walks, settings=NODE2VEC_SETTINGS)
    if embedding.shape != (len(edge_ids), int(NODE2VEC_SETTINGS["dimensions"])):
        raise SystemExit(f"embedding shape {embedding.shape} is unexpected")
    _atomic_numpy(embedding_path, embedding.astype(np.float32, copy=False))
    metadata = {
        "model": "asttn",
        "embedding": "node2vec",
        "dimensions": int(NODE2VEC_SETTINGS["dimensions"]),
        "dimension_source": "explicit 128-d adaptation; the reference file header is not available for Xuancheng",
        "trainable_in_asttn": False,
        "seed": int(NODE2VEC_SETTINGS["seed"]),
        "settings": NODE2VEC_SETTINGS,
        "num_nodes": len(edge_ids),
        "node_order": "r_graph.npz edge_ids, matching node_mapping.csv and r_nodes.csv",
        "node_order_sha256": node_order_sha256(edge_ids),
        "directed_edges": int(adjacency.sum()),
        "nodes_without_out_edge": int(empty),
        "traffic_counts_used": False,
        "output": embedding_path.as_posix(),
    }
    temporary = metadata_path.with_name(f".{metadata_path.name}.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, metadata_path)
    print(f"wrote {embedding_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
