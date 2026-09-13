"""Training-day DTW temporal graph from official ``data/Temporal_Graph_gen.py``.

This script does **not** install or call ``fastdtw`` / ``dtaidistance``. It ports
the custom Sakoe-Chiba DTW in the original file.

Leakage rule: only ``split_manifest.json`` training days, full R-edge flow
(the prepared-data ``full_flow`` CSVs). Validation and test days are rejected.
The seven penetration rates share one temporal graph.

This session must not write the official 56-node / 288-slot graph. The CLI
refuses that write unless ``--allow-official-r-graph`` is passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from reimplementation.common.data.r_only_npz_dataset import load_json
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.utils.atomic_io import atomic_write_json
from reimplementation.common.utils.hashing import sha256_file, sha256_numpy
from reimplementation.stgcn.validation import load_edge_ids
from reimplementation.stfgnn.model.dtw import (
    adjacency_edge_stats,
    compute_dtw,
    connected_component_stats,
    pairwise_dtw_distance_matrix,
    sparsify_temporal_adjacency,
)

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
OFFICIAL_NODE_COUNT = 56
OFFICIAL_STEPS_PER_DAY = 288


def default_paths() -> dict[str, Path]:
    data_root = PROJECT_ROOT / "reimplementation" / "stgcn" / "prepared_data" / "r-only"
    return {
        "training_flow": PROJECT_ROOT
        / "analysis"
        / "simulation"
        / "data"
        / "processed"
        / "subgraph_trajectories",
        "split_manifest": data_root / "split_manifest.json",
        "r_nodes": PROJECT_ROOT / "analysis" / "graph" / "r_graph" / "r_nodes.csv",
        "dataset_metadata": data_root / "dataset_metadata.json",
        "output_dir": SCRIPT_PATH.parent / "prepared_data" / "r-only" / "temporal_graph",
    }


def training_and_heldout_days(manifest: Mapping[str, Any]) -> tuple[list[str], set[str]]:
    training = [str(item) for item in manifest["training_days"]]
    heldout = {str(item) for item in list(manifest.get("validation_days", [])) + list(manifest.get("test_days", []))}
    overlap = [day for day in training if day in heldout]
    if overlap:
        raise ReimplementationError(f"training_days overlap held-out days: {overlap}")
    return training, heldout


def assert_training_days_only(days: Sequence[str], manifest: Mapping[str, Any]) -> list[str]:
    training, heldout = training_and_heldout_days(manifest)
    requested = [str(item) for item in days]
    bad = [day for day in requested if day in heldout]
    if bad:
        raise ReimplementationError(
            f"temporal graph cannot use validation/test days: {bad}. "
            "Use split_manifest training_days only."
        )
    unknown = [day for day in requested if day not in training]
    if unknown:
        raise ReimplementationError(f"days are not in training_days: {unknown}")
    return requested


def load_day_flow_csv(
    path: Path,
    *,
    node_ids: Sequence[str],
    steps_per_day: int,
    expected_sha256: str | None = None,
) -> np.ndarray:
    if not path.is_file():
        raise ReimplementationError(f"full-flow CSV not found: {path}")
    if expected_sha256 is not None:
        digest = sha256_file(path)
        if digest != expected_sha256:
            raise ReimplementationError(
                f"{path} sha256 {digest} != dataset_metadata {expected_sha256}"
            )
    frame = pd.read_csv(path, dtype={"edge_id": str})
    required = {"window_index", "edge_id", "vehicle_count"}
    missing = required.difference(frame.columns)
    if missing:
        raise ReimplementationError(f"{path} missing columns {sorted(missing)}")
    if frame.empty:
        raise ReimplementationError(f"{path} is empty")
    table = frame.pivot(index="window_index", columns="edge_id", values="vehicle_count")
    table = table.reindex(index=list(range(int(steps_per_day))), columns=list(node_ids))
    if table.shape != (int(steps_per_day), len(node_ids)):
        raise ReimplementationError(f"{path} pivoted shape {table.shape} != ({steps_per_day}, {len(node_ids)})")
    if table.isna().any().any():
        missing_cells = int(table.isna().sum().sum())
        raise ReimplementationError(
            f"{path} has {missing_cells} missing (day, slot, node) cells; "
            "refusing to fill with 0 or interpolate"
        )
    values = np.asarray(table.to_numpy(), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ReimplementationError(f"{path} contains NaN or inf after pivot")
    return values


def expected_full_flow_sha256(metadata: Mapping[str, Any] | None, day: str) -> str | None:
    if not metadata:
        return None
    entry = (
        metadata.get("input_file_sha256", {})
        .get("full_flow", {})
        .get(day, {})
    )
    digest = entry.get("sha256")
    return None if digest is None else str(digest)


def load_training_daily_flow(
    *,
    training_flow: Path,
    split_manifest: Path,
    r_nodes: Path,
    dataset_metadata: Path | None = None,
    steps_per_day: int = 288,
    days: Sequence[str] | None = None,
) -> dict[str, Any]:
    manifest = load_json(split_manifest)
    node_ids = load_edge_ids(r_nodes)
    metadata = load_json(dataset_metadata) if dataset_metadata and dataset_metadata.is_file() else None
    requested = list(days) if days is not None else [str(item) for item in manifest["training_days"]]
    requested = assert_training_days_only(requested, manifest)
    arrays = []
    file_hashes = {}
    if training_flow.is_file() and training_flow.suffix.lower() == ".npy":
        stacked = np.array(np.load(training_flow, allow_pickle=False), dtype=np.float64, copy=True)
        if stacked.ndim != 3:
            raise ReimplementationError(
                f"{training_flow} must be (n_days, steps_per_day, n_nodes), got {stacked.shape}"
            )
        if stacked.shape[0] != len(requested):
            raise ReimplementationError(
                f"{training_flow} has {stacked.shape[0]} days, requested {len(requested)}"
            )
        if stacked.shape[1] != int(steps_per_day) or stacked.shape[2] != len(node_ids):
            raise ReimplementationError(
                f"{training_flow} shape {stacked.shape} != ({len(requested)}, {steps_per_day}, {len(node_ids)})"
            )
        arrays = [stacked[index] for index in range(stacked.shape[0])]
        file_hashes[training_flow.as_posix()] = sha256_file(training_flow)
    else:
        for day in requested:
            csv_path = training_flow / day / "edge_flow_5min.csv"
            values = load_day_flow_csv(
                csv_path,
                node_ids=node_ids,
                steps_per_day=steps_per_day,
                expected_sha256=expected_full_flow_sha256(metadata, day),
            )
            arrays.append(values)
            file_hashes[csv_path.as_posix()] = sha256_file(csv_path)
    daily = np.stack(arrays, axis=0)
    return {
        "daily": daily,
        "days": requested,
        "node_ids": node_ids,
        "file_hashes": file_hashes,
        "steps_per_day": int(steps_per_day),
        "did_not_use_validation_or_test_days": True,
        "did_not_use_observed_penetration_flow": True,
        "shared_across_penetration_rates": True,
    }


def _dtw_pair_job(payload: tuple[np.ndarray, np.ndarray, int, int, bool]) -> float:
    left, right, order, window, normal = payload
    return compute_dtw(left, right, order=order, window=window, normal=normal)


def pairwise_dtw_distance_matrix_parallel(
    daily: np.ndarray,
    *,
    order: int = 1,
    window: int = 12,
    normal: bool = True,
    num_workers: int = 1,
) -> np.ndarray:
    if int(num_workers) <= 1:
        return pairwise_dtw_distance_matrix(daily, order=order, window=window, normal=normal)
    source = np.array(daily, dtype=np.float64, copy=True)
    n_nodes = int(source.shape[2])
    pairs = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    jobs = [
        (source[:, :, i], source[:, :, j], int(order), int(window), bool(normal))
        for i, j in pairs
    ]
    distances = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    with ProcessPoolExecutor(max_workers=int(num_workers)) as pool:
        values = list(pool.map(_dtw_pair_job, jobs))
    for (i, j), value in zip(pairs, values):
        distances[i, j] = value
    return distances + distances.T


def build_temporal_graph(
    daily: np.ndarray,
    *,
    order: int = 1,
    window: int = 12,
    sparsity: float = 0.01,
    top_k: int | None = None,
    num_workers: int = 1,
) -> dict[str, Any]:
    source = np.array(daily, dtype=np.float64, copy=True)
    distances = pairwise_dtw_distance_matrix_parallel(
        source,
        order=order,
        window=window,
        normal=True,
        num_workers=num_workers,
    )
    adjacency = sparsify_temporal_adjacency(distances, sparsity=sparsity, top_k=top_k)
    n_nodes = int(source.shape[2])
    k = int(top_k) if top_k is not None else int(n_nodes * float(sparsity))
    stats = adjacency_edge_stats(adjacency)
    before_self = int(stats["nnz"] - stats["self_loops"])
    return {
        "distances": distances,
        "adjacency": adjacency,
        "top_k": k,
        "sparsity": float(sparsity),
        "order": int(order),
        "dtw_window": int(window),
        "n_days": int(source.shape[0]),
        "steps_per_day": int(source.shape[1]),
        "n_nodes": n_nodes,
        "input_was_modified": False,
        "stats": stats,
        "components": connected_component_stats(adjacency),
        "offdiag_edges_after_sparsity": before_self,
        "self_loops": stats["self_loops"],
        "symmetric": bool(np.allclose(adjacency, adjacency.T)),
        "tie_break": "stable argsort, smaller index first",
        "normalization": "per-day z-score on each node series, not daily mean pattern",
        "did_not_use_fastdtw_package": True,
        "did_not_use_dtaidistance": True,
    }


def write_temporal_graph(
    output_dir: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    distance_path = output_dir / "dtw_distance.npy"
    adj_path = output_dir / "temporal_adjacency.npy"
    csv_path = output_dir / "temporal_adjacency.csv"
    meta_path = output_dir / "temporal_graph_metadata.json"
    for path in (distance_path, adj_path, csv_path, meta_path):
        if path.exists() and not overwrite:
            raise ReimplementationError(f"{path} exists; pass --overwrite")
    np.save(distance_path, np.asarray(payload["distances"]))
    np.save(adj_path, np.asarray(payload["adjacency"]))
    pd.DataFrame(np.asarray(payload["adjacency"])).to_csv(csv_path, index=False, header=None)
    metadata = {
        "graph_kind": "training_days_full_flow_dtw",
        "distance_sha256": sha256_numpy(np.asarray(payload["distances"])),
        "adjacency_sha256": sha256_numpy(np.asarray(payload["adjacency"])),
        **{key: value for key, value in payload.items() if key not in {"distances", "adjacency"}},
    }
    if extra_metadata:
        metadata.update(dict(extra_metadata))
    atomic_write_json(meta_path, metadata)
    return {
        "dtw_distance": distance_path.as_posix(),
        "temporal_adjacency": adj_path.as_posix(),
        "temporal_adjacency_csv": csv_path.as_posix(),
        "metadata": meta_path.as_posix(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = default_paths()
    parser = argparse.ArgumentParser(
        description="Build STFGNN DTW temporal graph from training-day full R-edge flow."
    )
    parser.add_argument("--training-flow", type=Path, default=defaults["training_flow"])
    parser.add_argument("--split-manifest", dest="split_manifest", type=Path, default=defaults["split_manifest"])
    parser.add_argument("--r-nodes", dest="r_nodes", type=Path, default=defaults["r_nodes"])
    parser.add_argument("--dataset-metadata", dest="dataset_metadata", type=Path, default=defaults["dataset_metadata"])
    parser.add_argument("--output-dir", dest="output_dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--steps-per-day", dest="steps_per_day", type=int, default=288)
    parser.add_argument("--dtw-window", dest="dtw_window", type=int, default=12)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--sparsity", type=float, default=0.01)
    parser.add_argument("--top-k", dest="top_k", type=int, default=None, help="Override k=int(N*sparsity). Not the official default.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Not used. Original Temporal_Graph_gen.py ranks by sparsity/top-k, not a distance cutoff.",
    )
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-official-r-graph",
        dest="allow_official_r_graph",
        action="store_true",
        help="Required to write the real 56-node / 288-slot training-day graph.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if args.threshold is not None:
        raise ReimplementationError(
            "original Temporal_Graph_gen.py does not use a distance threshold; "
            "it keeps k=int(N*sparsity) nearest DTW neighbours (optional --top-k override)"
        )
    packed = load_training_daily_flow(
        training_flow=args.training_flow,
        split_manifest=args.split_manifest,
        r_nodes=args.r_nodes,
        dataset_metadata=args.dataset_metadata,
        steps_per_day=int(args.steps_per_day),
    )
    n_nodes = int(packed["daily"].shape[2])
    n_days = int(packed["daily"].shape[0])
    official = (
        n_nodes == OFFICIAL_NODE_COUNT
        and int(args.steps_per_day) == OFFICIAL_STEPS_PER_DAY
        and n_days >= 14
    )
    if official and not args.allow_official_r_graph:
        raise ReimplementationError(
            "refusing to write the official 56-node DTW temporal graph in this session; "
            "pass --allow-official-r-graph for a later dedicated run"
        )
    graph = build_temporal_graph(
        packed["daily"],
        order=int(args.order),
        window=int(args.dtw_window),
        sparsity=float(args.sparsity),
        top_k=args.top_k,
        num_workers=int(args.num_workers),
    )
    extra = {
        "seed": int(args.seed),
        "days": packed["days"],
        "node_ids": packed["node_ids"],
        "file_hashes": packed["file_hashes"],
        "did_not_use_validation_or_test_days": True,
        "shared_across_penetration_rates": True,
        "num_workers": int(args.num_workers),
    }
    written = write_temporal_graph(args.output_dir, graph, overwrite=bool(args.overwrite), extra_metadata=extra)
    return {"written": written, "n_nodes": n_nodes, "n_days": n_days, "top_k": graph["top_k"]}


if __name__ == "__main__":
    try:
        result = main()
        print(json.dumps(result, indent=2), flush=True)
    except ReimplementationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
