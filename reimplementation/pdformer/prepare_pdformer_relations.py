"""Build PDFormer hop / DTW / Laplacian / pattern-key files from training days only.

This session must not write the official 56-node relation set. The CLI refuses
that write unless ``--allow-official-r-graph`` is passed.

DTW uses a NumPy port of ``fastdtw`` radius=6 on the mean daily full-flow
curve of ``split_manifest.json`` training days. Validation and test days are
rejected. Pattern keys from official KShape require ``tslearn``; this script
will not install it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from reimplementation.common.data.r_only_npz_dataset import load_json
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.utils.atomic_io import atomic_write_json
from reimplementation.common.utils.hashing import sha256_file, sha256_numpy
from reimplementation.stgcn.validation import load_edge_ids
from reimplementation.pdformer.graph import (
    FASTDTW_RADIUS,
    TEST_ONLY_RELATIONS,
    UNREACHABLE_HOP,
    assert_training_days_only,
    daily_mean_from_days,
    geographic_mask,
    hop_shortest_path,
    laplacian_positional_encoding,
    load_binary_topology,
    load_day_flow_csv,
    pairwise_fastdtw_distance,
    reject_heldout_relation_input,
    semantic_mask,
    test_only_pattern_keys,
)

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
OFFICIAL_NODE_COUNT = 56
OFFICIAL_STEPS_PER_DAY = 288


def expected_full_flow_sha256(metadata: Mapping[str, Any] | None, day: str) -> str | None:
    if not metadata:
        return None
    entry = metadata.get("input_file_sha256", {}).get("full_flow", {}).get(day, {})
    digest = entry.get("sha256")
    return None if digest is None else str(digest)


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
        "topology": data_root / "adjacency_matrix" / "stgcn_undirected_topology.npy",
        "dataset_metadata": data_root / "dataset_metadata.json",
        "output_dir": SCRIPT_PATH.parent / "prepared_data" / "r-only" / "relations",
    }


def load_training_daily_flow(
    *,
    training_flow: Path,
    split_manifest: Path,
    r_nodes: Path,
    dataset_metadata: Path | None = None,
    steps_per_day: int = 288,
    days: Sequence[str] | None = None,
) -> dict[str, Any]:
    reject_heldout_relation_input(training_flow)
    manifest = load_json(split_manifest)
    node_ids = load_edge_ids(r_nodes)
    metadata = load_json(dataset_metadata) if dataset_metadata and dataset_metadata.is_file() else None
    requested = list(days) if days is not None else [str(item) for item in manifest["training_days"]]
    requested = assert_training_days_only(requested, manifest)
    arrays = []
    file_hashes = {}
    for day in requested:
        path = training_flow / day / "edge_flow_5min.csv"
        values = load_day_flow_csv(
            path,
            node_ids=node_ids,
            steps_per_day=steps_per_day,
            expected_sha256=expected_full_flow_sha256(metadata, day),
        )
        arrays.append(values)
        file_hashes[day] = sha256_file(path)
    stacked = np.stack(arrays, axis=0)
    return {
        "days": requested,
        "node_ids": node_ids,
        "daily_flow": stacked,
        "file_hashes": file_hashes,
        "steps_per_day": int(steps_per_day),
        "split": "train",
    }


def maybe_kshape_pattern_keys(
    candidates: np.ndarray,
    *,
    n_cluster: int,
    cluster_max_iter: int,
    cluster_method: str,
) -> np.ndarray:
    if cluster_method == "synthetic":
        raise ReimplementationError("synthetic pattern keys must be created with test_only_pattern_keys")
    if cluster_method != "kshape":
        raise ReimplementationError(f"unsupported cluster_method {cluster_method}")
    try:
        from tslearn.clustering import KShape  # type: ignore
    except ImportError as exc:
        raise ReimplementationError(
            "tslearn is not installed. Original PDFormer uses tslearn.clustering.KShape "
            "only to cluster training pattern keys. The model forward pass does not need it. "
            "Suggested (not executed): pip install tslearn==0.5.2"
        ) from exc
    fitted = KShape(n_clusters=int(n_cluster), max_iter=int(cluster_max_iter)).fit(candidates)
    return np.asarray(fitted.cluster_centers_, dtype=np.float32)


def build_relations(
    *,
    topology: np.ndarray,
    daily_flow: np.ndarray | None,
    pattern_keys: np.ndarray,
    far_mask_delta: int,
    dtw_delta: int,
    lape_dim: int,
    bidir: bool,
    dtw_radius: int,
    split: str,
) -> dict[str, Any]:
    if split != "train":
        raise ReimplementationError("PDFormer relations only accept --split train")
    hops = hop_shortest_path(topology, bidir=bidir)
    if daily_flow is None:
        raise ReimplementationError("DTW requires training daily full-flow")
    mean_curve = daily_mean_from_days(daily_flow)
    dtw = pairwise_fastdtw_distance(mean_curve, radius=dtw_radius)
    pe = laplacian_positional_encoding(topology, lape_dim)
    geo = geographic_mask(hops, far_mask_delta=far_mask_delta, transpose=True)
    sem = semantic_mask(dtw, dtw_delta=dtw_delta)
    return {
        "hop_matrix": hops.astype(np.float32),
        "dtw_matrix": dtw.astype(np.float32),
        "laplacian_pe": pe["laplacian_pe"],
        "geo_mask": geo,
        "sem_mask": sem,
        "pattern_keys": np.asarray(pattern_keys, dtype=np.float32),
        "isolated_point_num": pe["isolated_point_num"],
        "sign_convention": pe["sign_convention"],
    }


def write_relations(
    output_dir: Path,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    output_dir = Path(output_dir)
    names = {
        "pdformer_shortest_path.npy": payload["hop_matrix"],
        "pdformer_dtw_distance.npy": payload["dtw_matrix"],
        "pdformer_laplacian_pe.npy": payload["laplacian_pe"],
        "pdformer_geo_mask.npy": payload["geo_mask"],
        "pdformer_semantic_mask.npy": payload["sem_mask"],
        "pdformer_pattern_keys.npy": payload["pattern_keys"],
    }
    for name in list(names) + ["pdformer_relations_metadata.json", "pdformer_relations_validation.json"]:
        path = output_dir / name
        if path.exists() and not overwrite:
            raise ReimplementationError(f"refusing to overwrite {path}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, array in names.items():
        np.save(output_dir / name, array)
    atomic_write_json(output_dir / "pdformer_relations_metadata.json", metadata)
    validation = {
        "status": "ok",
        "node_count": int(payload["hop_matrix"].shape[0]),
        "hop_diagonal_zero": bool(np.allclose(np.diag(payload["hop_matrix"]), 0)),
        "geo_mask_true_means": "mask_out",
        "sem_mask_true_means": "mask_out",
        "dtw_algorithm": "fastdtw_radius_6_numpy_port",
        "split": "train",
        "file_sha256": {name: sha256_numpy(np.asarray(array)) for name, array in names.items()},
    }
    atomic_write_json(output_dir / "pdformer_relations_validation.json", validation)


def parse_args() -> argparse.Namespace:
    defaults = default_paths()
    parser = argparse.ArgumentParser(description="Prepare PDFormer hop/DTW/Laplacian relations from training days.")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--training-flow", type=Path, default=defaults["training_flow"])
    parser.add_argument("--split-manifest", type=Path, default=defaults["split_manifest"])
    parser.add_argument("--r-nodes", type=Path, default=defaults["r_nodes"])
    parser.add_argument("--topology", type=Path, default=defaults["topology"])
    parser.add_argument("--dataset-metadata", type=Path, default=defaults["dataset_metadata"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--far-mask-delta", type=int, default=7)
    parser.add_argument("--dtw-delta", type=int, default=5)
    parser.add_argument("--dtw-radius", type=int, default=FASTDTW_RADIUS)
    parser.add_argument("--lape-dim", type=int, default=8)
    parser.add_argument("--n-cluster", type=int, default=16)
    parser.add_argument("--s-attn-size", type=int, default=3)
    parser.add_argument("--cluster-method", type=str, default="kshape")
    parser.add_argument("--cluster-max-iter", type=int, default=5)
    parser.add_argument("--cand-key-days", type=int, default=14)
    parser.add_argument("--bidir", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-official-r-graph", action="store_true")
    parser.add_argument("--pattern-keys-source", type=str, default="kshape", choices=["kshape", "synthetic"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split != "train":
        raise ReimplementationError("PDFormer relations only accept --split train")
    reject_heldout_relation_input(args.training_flow)
    topology = load_binary_topology(args.topology)
    node_count = int(topology.shape[0])
    flow = load_training_daily_flow(
        training_flow=args.training_flow,
        split_manifest=args.split_manifest,
        r_nodes=args.r_nodes,
        dataset_metadata=args.dataset_metadata,
    )
    if (
        node_count == OFFICIAL_NODE_COUNT
        and int(flow["steps_per_day"]) == OFFICIAL_STEPS_PER_DAY
        and not args.allow_official_r_graph
    ):
        raise ReimplementationError(
            "refusing to write the official 56-node PDFormer relation set. "
            "Pass --allow-official-r-graph for a later dedicated run."
        )
    if args.pattern_keys_source == "synthetic":
        keys = test_only_pattern_keys(
            n_cluster=args.n_cluster,
            s_attn_size=args.s_attn_size,
            output_dim=1,
            seed=args.seed,
        )
        key_kind = TEST_ONLY_RELATIONS
    else:
        mean_curve = daily_mean_from_days(flow["daily_flow"])
        window = mean_curve[: args.s_attn_size]
        candidates = np.transpose(window, (1, 0, 2))
        keys = maybe_kshape_pattern_keys(
            candidates,
            n_cluster=args.n_cluster,
            cluster_max_iter=args.cluster_max_iter,
            cluster_method=args.cluster_method,
        )
        key_kind = "kshape_training_days"
    relations = build_relations(
        topology=topology,
        daily_flow=flow["daily_flow"],
        pattern_keys=keys,
        far_mask_delta=args.far_mask_delta,
        dtw_delta=args.dtw_delta,
        lape_dim=args.lape_dim,
        bidir=bool(args.bidir),
        dtw_radius=args.dtw_radius,
        split=args.split,
    )
    metadata = {
        "split": "train",
        "days": flow["days"],
        "node_ids": flow["node_ids"],
        "file_sha256": flow["file_hashes"],
        "bidir": bool(args.bidir),
        "type_short_path": "hop",
        "unreachable_hop": UNREACHABLE_HOP,
        "far_mask_delta": args.far_mask_delta,
        "geo_mask_inequality": ">=",
        "geo_mask_true_means": "mask_out",
        "dtw_algorithm": "fastdtw_numpy_port",
        "dtw_radius": args.dtw_radius,
        "dtw_delta": args.dtw_delta,
        "dtw_input": "training_days_mean_daily_full_flow",
        "sem_mask_true_means": "mask_out",
        "lape_dim": args.lape_dim,
        "laplacian_adjacency": "stgcn_undirected_topology 0/1",
        "pattern_keys_kind": key_kind,
        "did_not_use_validation_or_test": True,
        "did_not_use_stgcn_gaussian_weights_as_hop": True,
    }
    write_relations(args.output_dir, relations, metadata, overwrite=bool(args.overwrite))
    print(f"wrote PDFormer relations to {args.output_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except ReimplementationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
