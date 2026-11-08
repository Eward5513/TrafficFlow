"""Load the frozen R-graph node2vec table in the public node order."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reimplementation.asttn.node2vec import NODE2VEC_SETTINGS
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.utils.hashing import sha256_file
from reimplementation.stjgcn.graph import node_order_sha256


def load_spatial_embedding(path: Path, edge_ids: list[str]) -> dict:
    path = Path(path)
    meta_path = path.with_name("metadata.json")
    if not path.is_file() or not meta_path.is_file():
        raise ReimplementationError(f"missing ASTTN node2vec asset {path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if str(metadata.get("node_order_sha256")) != node_order_sha256(edge_ids):
        raise ReimplementationError("ASTTN node2vec node order does not match the current node mapping")
    if int(metadata.get("dimensions", -1)) != int(NODE2VEC_SETTINGS["dimensions"]):
        raise ReimplementationError("ASTTN node2vec dimension does not match the recorded 128-d adaptation")
    embedding = np.load(path)
    if tuple(embedding.shape) != (len(edge_ids), int(NODE2VEC_SETTINGS["dimensions"])):
        raise ReimplementationError(f"node2vec shape {embedding.shape} != {(len(edge_ids), NODE2VEC_SETTINGS['dimensions'])}")
    if not np.isfinite(embedding).all():
        raise ReimplementationError("node2vec embedding contains NaN or inf")
    return {
        "embedding": embedding.astype(np.float32, copy=False),
        "metadata": metadata,
        "graph_sha256": sha256_file(path),
    }
