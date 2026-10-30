#!/usr/bin/env python3
"""Build Xuancheng26 PDFormer relations from the training-day full inflow.

The numeric definitions are the ports in ``reimplementation/pdformer/graph.py``,
which follow the deleted ``prepare_pdformer_relations.py`` and the reference
PDFormer loader. This script does not restore or call that deleted file.

Geographic and semantic masks are not written. The model builds them at init.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")

CODE_ROOT = Path("/home/tzhang174/TrafficFlow")
import sys

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from reimplementation.pdformer.graph import (
    FASTDTW_RADIUS,
    TEST_ONLY_RELATIONS,
    UNREACHABLE_HOP,
    assert_training_days_only,
    daily_mean_from_days,
    fastdtw,
    hop_shortest_path,
    laplacian_positional_encoding,
    load_binary_topology,
)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from tslearn.clustering import KShape

import prep_paths

TRAINING_ROOT = prep_paths.DEFAULT_TRAINING_ROOT
WINDOW_ROOT = TRAINING_ROOT / "window_data"
FULL_ROOT = TRAINING_ROOT / "full_flow" / "r-level"
GRAPH_DIR = TRAINING_ROOT / "graph"
R_GRAPH = GRAPH_DIR / "r_graph.npz"
TOPOLOGY = GRAPH_DIR / "stsgcn_undirected_topology.npy"
OUTPUT_DIR = TRAINING_ROOT / "config" / "pdformer"
CONFIG_PATH = OUTPUT_DIR / "r_only_pdformer.json"
STEPS_PER_DAY = 288
LAPE_DIM = 8
N_CLUSTER = 16
S_ATTN_SIZE = 3
CLUSTER_MAX_ITER = 5
KSHAPE_N_INIT = 1
SEED = 42
BIDIR = True

_CURVES: np.ndarray | None = None


class RelationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    payload = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(payload.shape).encode("ascii"))
    digest.update(str(payload.dtype).encode("ascii"))
    digest.update(payload.tobytes())
    return digest.hexdigest()


def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npy")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_edge_ids() -> list[str]:
    with np.load(R_GRAPH, allow_pickle=False) as payload:
        edge_ids = [str(item) for item in payload["edge_ids"].tolist()]
        num_nodes = int(np.asarray(payload["num_nodes"]).reshape(-1)[0])
    if len(edge_ids) != num_nodes:
        raise RelationError(f"r_graph has {len(edge_ids)} ids and num_nodes={num_nodes}")
    return edge_ids


def csv_edge_ids(path: Path) -> list[str]:
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "edge_id" not in rows[0]:
        raise RelationError(f"{path} has no edge_id column")
    return [str(row["edge_id"]) for row in rows]


def training_days(edge_count: int) -> list[str]:
    manifest = json.loads((WINDOW_ROOT / "split_manifest.json").read_text(encoding="utf-8"))
    days = assert_training_days_only(manifest["training_days"], manifest)
    if len(days) != 20:
        raise RelationError(f"expected 20 training days, got {len(days)}")
    del edge_count
    return days


def load_training_flow(edge_ids: list[str], days: list[str]) -> tuple[np.ndarray, dict[str, str]]:
    cube = np.empty((len(days), STEPS_PER_DAY, len(edge_ids)), dtype=np.float64)
    hashes: dict[str, str] = {}
    for day_index, day_name in enumerate(days):
        path = FULL_ROOT / day_name / "inflow.csv"
        if not path.is_file():
            raise RelationError(f"missing training full inflow {path}")
        with path.open(encoding="utf-8") as handle:
            header = handle.readline().rstrip("\n").split(",")
        if header[:2] != ["time_bin", "start_time"]:
            raise RelationError(f"{path} header is not time_bin,start_time")
        columns = header[2:]
        if set(columns) != set(edge_ids):
            raise RelationError(f"{path} edge set does not match r_graph")
        order = [columns.index(edge_id) for edge_id in edge_ids]
        table = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
        if table.shape != (STEPS_PER_DAY, len(edge_ids) + 2):
            raise RelationError(f"{path} shape {table.shape}")
        cube[day_index] = table[:, 2:][:, order]
        hashes[day_name] = sha256_file(path)
    if not np.isfinite(cube).all():
        raise RelationError("training full inflow contains NaN or inf")
    return cube, hashes


def _init_worker(curves: np.ndarray) -> None:
    global _CURVES
    _CURVES = curves


def _row_distances(node: int) -> tuple[int, np.ndarray]:
    assert _CURVES is not None
    n_nodes = int(_CURVES.shape[1])
    row = np.empty(n_nodes - node, dtype=np.float64)
    left = _CURVES[:, node, :]
    for offset, other in enumerate(range(node, n_nodes)):
        distance, _path = fastdtw(left, _CURVES[:, other, :], radius=FASTDTW_RADIUS)
        row[offset] = distance
    return node, row


def pairwise_fastdtw(curves: np.ndarray, workers: int) -> np.ndarray:
    n_nodes = int(curves.shape[1])
    distances = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(curves,),
    ) as pool:
        done = 0
        for node, row in pool.map(_row_distances, range(n_nodes), chunksize=1):
            distances[node, node:] = row
            distances[node:, node] = row
            done += 1
            if done % 50 == 0 or done == n_nodes:
                print(f"fastdtw rows {done}/{n_nodes}", flush=True)
    return distances


def component_stats(topology: np.ndarray) -> dict[str, int]:
    graph = csr_matrix((topology > 0).astype(np.int8))
    count, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels, minlength=int(count))
    return {
        "component_count": int(count),
        "largest_component_size": int(sizes.max()) if sizes.size else 0,
        "isolated_node_count": int(np.count_nonzero(sizes == 1)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Xuancheng26 PDFormer relations")
    prep_paths.add_path_arguments(parser)
    args = parser.parse_args(argv)
    root = prep_paths.resolved(args.training_root)
    output_dir = root / "config" / "pdformer"
    prep_paths.bind(
        sys.modules[__name__],
        TRAINING_ROOT=root,
        WINDOW_ROOT=root / "window_data",
        FULL_ROOT=root / "full_flow" / "r-level",
        GRAPH_DIR=root / "graph",
        R_GRAPH=root / "graph" / "r_graph.npz",
        TOPOLOGY=root / "graph" / "stsgcn_undirected_topology.npy",
        OUTPUT_DIR=output_dir,
        CONFIG_PATH=output_dir / "r_only_pdformer.json",
    )
    print(f"training_root={root}", flush=True)
    print(f"output_dir={OUTPUT_DIR}", flush=True)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = {
        "lape_dim": LAPE_DIM,
        "n_cluster": N_CLUSTER,
        "s_attn_size": S_ATTN_SIZE,
        "cluster_max_iter": CLUSTER_MAX_ITER,
        "seed": SEED,
        "far_mask_delta": 7,
        "dtw_delta": 5,
        "bidir": True,
        "cluster_method": "kshape",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RelationError(f"config {key}={config.get(key)!r} != audited {value!r}")

    edge_ids = load_edge_ids()
    mapping_ids = csv_edge_ids(WINDOW_ROOT / "node_mapping.csv")
    graph_mapping_ids = csv_edge_ids(GRAPH_DIR / "node_mapping.csv")
    r_node_ids = csv_edge_ids(GRAPH_DIR / "r_nodes.csv")
    if not (edge_ids == mapping_ids == graph_mapping_ids == r_node_ids):
        raise RelationError("edge_ids, node_mapping, and r_nodes are not the same order")
    n_nodes = len(edge_ids)
    if n_nodes != 1406:
        raise RelationError(f"canonical node count is {n_nodes}, expected the Xuancheng graph size 1406")

    days = training_days(n_nodes)
    topology = load_binary_topology(TOPOLOGY, expected_nodes=n_nodes)
    if not np.array_equal(topology, topology.T):
        raise RelationError("undirected topology is not symmetric")
    if int(np.count_nonzero(np.diag(topology))) != 0:
        raise RelationError("undirected topology diagonal is not 0")

    daily, flow_hashes = load_training_flow(edge_ids, days)
    print(
        f"nodes={n_nodes} days={len(days)} flow={daily.shape} "
        f"flow_mib={daily.nbytes / (1024 ** 2):.1f}",
        flush=True,
    )

    hops = hop_shortest_path(np.asarray(topology, dtype=np.float64), bidir=BIDIR).astype(np.float32)
    if hops.shape != (n_nodes, n_nodes):
        raise RelationError(f"hop shape {hops.shape}")
    if not np.array_equal(hops, hops.T):
        raise RelationError("hop matrix is not symmetric")
    if not np.array_equal(np.diag(hops), np.zeros(n_nodes, dtype=np.float32)):
        raise RelationError("hop diagonal is not 0")
    finite = hops < np.float32(UNREACHABLE_HOP)
    off = finite.copy()
    np.fill_diagonal(off, False)
    unreachable = int(np.count_nonzero(hops == np.float32(UNREACHABLE_HOP)))
    print(
        f"hop max_finite={float(hops[off].max()) if off.any() else 0} "
        f"unreachable_entries={unreachable}",
        flush=True,
    )

    pe_pack = laplacian_positional_encoding(np.asarray(topology, dtype=np.float64), LAPE_DIM)
    pe = np.asarray(pe_pack["laplacian_pe"], dtype=np.float32)
    if pe.shape != (n_nodes, LAPE_DIM):
        raise RelationError(f"laplacian PE shape {pe.shape}")
    if not np.isfinite(pe).all():
        raise RelationError("laplacian PE contains NaN or inf")

    mean_curve = daily_mean_from_days(daily)
    if mean_curve.shape != (STEPS_PER_DAY, n_nodes, 1):
        raise RelationError(f"mean curve shape {mean_curve.shape}")
    window = mean_curve[:S_ATTN_SIZE]
    candidates = np.transpose(window, (1, 0, 2))
    if candidates.shape != (n_nodes, S_ATTN_SIZE, 1):
        raise RelationError(f"pattern candidates {candidates.shape}")
    fitted = KShape(
        n_clusters=N_CLUSTER,
        max_iter=CLUSTER_MAX_ITER,
        n_init=KSHAPE_N_INIT,
        random_state=SEED,
        verbose=False,
    ).fit(candidates)
    keys = np.asarray(fitted.cluster_centers_, dtype=np.float32)
    if keys.shape != (N_CLUSTER, S_ATTN_SIZE, 1):
        raise RelationError(f"pattern keys shape {keys.shape}")
    if not np.isfinite(keys).all():
        raise RelationError("pattern keys contain NaN or inf")
    print(f"pattern_keys {keys.shape} samples={candidates.shape[0]}", flush=True)

    workers = max(
        1,
        min(
            len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1),
            64,
        ),
    )
    print(f"fastdtw workers={workers} radius={FASTDTW_RADIUS} curve={mean_curve.shape}", flush=True)
    distances = pairwise_fastdtw(mean_curve, workers)
    if distances.shape != (n_nodes, n_nodes):
        raise RelationError(f"dtw shape {distances.shape}")
    if not np.isfinite(distances).all():
        raise RelationError("DTW matrix contains NaN or inf")
    if float(np.max(np.abs(distances - distances.T))) != 0.0:
        raise RelationError("DTW matrix is not exactly symmetric")
    if float(np.max(np.abs(np.diag(distances)))) != 0.0:
        raise RelationError("DTW diagonal is not 0")
    if float(distances.min()) < 0.0:
        raise RelationError("DTW matrix has a negative distance")
    dtw = distances.astype(np.float32)

    outputs = {
        "pdformer_shortest_path.npy": hops,
        "pdformer_dtw_distance.npy": dtw,
        "pdformer_laplacian_pe.npy": pe,
        "pdformer_pattern_keys.npy": keys,
    }
    for name, array in outputs.items():
        save_array(OUTPUT_DIR / name, array)

    start, stop = pe_pack["slice"]
    eigenvalues = np.asarray(pe_pack["eigenvalues"], dtype=np.float64)
    components = component_stats(np.asarray(topology))
    metadata = {
        "kind": "official_training_relations",
        "dataset": "xuancheng26",
        "split": "train",
        "days": days,
        "num_nodes": n_nodes,
        "node_order_source": R_GRAPH.as_posix(),
        "node_ids": edge_ids,
        "undirected_topology": TOPOLOGY.as_posix(),
        "undirected_topology_sha256": sha256_file(TOPOLOGY),
        "r_graph_sha256": sha256_file(R_GRAPH),
        "training_day_count": len(days),
        "full_flow_source": (FULL_ROOT / "<date>" / "inflow.csv").as_posix(),
        "full_flow_sha256": flow_hashes,
        "bidir": BIDIR,
        "type_short_path": "hop",
        "shortest_path_algorithm": "floyd_warshall_min_plus",
        "unreachable_hop": UNREACHABLE_HOP,
        "far_mask_delta": 7,
        "geo_mask_inequality": ">=",
        "geo_mask_true_means": "mask_out",
        "geo_mask_stored": False,
        "dtw_algorithm": "fastdtw_numpy_port",
        "dtw_radius": FASTDTW_RADIUS,
        "dtw_distance": "euclidean_on_[T,1]_series",
        "dtw_normalization": "none",
        "dtw_delta": 5,
        "dtw_input": "training_days_mean_daily_full_flow",
        "mean_curve_shape": list(mean_curve.shape),
        "sem_mask_true_means": "mask_out",
        "sem_mask_stored": False,
        "lape_dim": LAPE_DIM,
        "laplacian": "I - D^{-1/2} A^T D^{-1/2}",
        "laplacian_adjacency": "stsgcn_undirected_topology 0/1",
        "eigensolver": "numpy.linalg.eig",
        "eigenvector_slice": [int(start), int(stop)],
        "selected_eigenvalues": [float(item) for item in eigenvalues[start:stop]],
        "isolated_point_num": int(pe_pack["isolated_point_num"]),
        "sign_convention": pe_pack["sign_convention"],
        "pattern_keys_kind": "kshape_training_days",
        "pattern_source": "first_s_attn_size_steps_of_training_day_mean_curve",
        "pattern_sample_count": int(candidates.shape[0]),
        "pattern_length": S_ATTN_SIZE,
        "pattern_feature_dim": 1,
        "n_cluster": N_CLUSTER,
        "cluster_method": "kshape",
        "cluster_max_iter": CLUSTER_MAX_ITER,
        "kshape_n_init": KSHAPE_N_INIT,
        "kshape_random_state": SEED,
        "cand_key_days_config": config.get("cand_key_days"),
        "cand_key_days_applied": False,
        "components": components,
        "hop_max_finite": float(hops[off].max()) if np.any(off) else 0.0,
        "hop_mean_finite_offdiag": float(hops[off].mean()) if np.any(off) else 0.0,
        "hop_unreachable_entries": unreachable,
        "dtw_min": float(dtw.min()),
        "dtw_max": float(dtw.max()),
        "dtw_mean": float(dtw.mean()),
        "dtw_median": float(np.median(dtw)),
        "output_sha256": {name: sha256_array(array) for name, array in outputs.items()},
        "output_shapes": {name: list(array.shape) for name, array in outputs.items()},
        "output_dtypes": {name: str(array.dtype) for name, array in outputs.items()},
        "did_not_use_validation_or_test": True,
        "did_not_use_stgcn_gaussian_weights_as_hop": True,
        "did_not_use_stfgnn_temporal_adjacency": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if metadata["kind"] == TEST_ONLY_RELATIONS or metadata["pattern_keys_kind"] == TEST_ONLY_RELATIONS:
        raise RelationError("refusing to mark official relations as test-only")
    meta_path = OUTPUT_DIR / "pdformer_relations_metadata.json"
    temporary = meta_path.with_name(".pdformer_relations_metadata.json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, meta_path)
    print(f"wrote {OUTPUT_DIR}", flush=True)
    print(
        json.dumps(
            {
                "components": components,
                "hop_max_finite": metadata["hop_max_finite"],
                "hop_mean_finite_offdiag": metadata["hop_mean_finite_offdiag"],
                "hop_unreachable_entries": unreachable,
                "dtw_min": metadata["dtw_min"],
                "dtw_max": metadata["dtw_max"],
                "dtw_mean": metadata["dtw_mean"],
                "dtw_median": metadata["dtw_median"],
                "selected_eigenvalues": metadata["selected_eigenvalues"],
                "isolated_point_num": metadata["isolated_point_num"],
                "sign_convention": metadata["sign_convention"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RelationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
