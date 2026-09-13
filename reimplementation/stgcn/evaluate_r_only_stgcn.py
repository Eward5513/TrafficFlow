"""Evaluate a trained R-only STGCN checkpoint on validation and test splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from reimplementation.common.data.r_only_npz_dataset import rate_tag
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.utils.checkpoint import assert_checkpoint_compatible, load_checkpoint
from reimplementation.common.utils.hashing import sha256_file
from reimplementation.stgcn.engine import (
    build_dataloaders,
    build_model,
    evaluate_one_epoch,
    resolve_device,
    resolve_path,
    _write_split_outputs,
)
from reimplementation.stgcn.losses import STGCNPredictionLoss
from reimplementation.stgcn.validation import validate_graph, validate_prepared_data

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate R-only STGCN from a checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--weighted-adjacency", type=Path, default=None)
    parser.add_argument("--r-nodes", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rate", type=int, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    tag = rate_tag(args.rate)
    data_root = resolve_path(PROJECT_ROOT, args.data_root or config["data_root"])
    adjacency = resolve_path(PROJECT_ROOT, args.weighted_adjacency or config["weighted_adjacency"])
    r_nodes = resolve_path(PROJECT_ROOT, args.r_nodes or config["r_nodes"])
    output_dir = resolve_path(PROJECT_ROOT, args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise ReimplementationError(f"{output_dir} is not empty; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_info = validate_prepared_data(
        data_root,
        n_his=int(config["n_his"]),
        num_nodes=int(config["num_nodes"]),
        rates=[tag],
    )
    graph_info = validate_graph(
        adjacency,
        int(config["Ks"]),
        int(config["num_nodes"]),
        data_info["node_ids"],
        r_nodes,
    )
    device = resolve_device(args.device)
    model = build_model(config, graph_info["cheb_kernel"]).to(device)
    payload = load_checkpoint(args.checkpoint, map_location=device)
    assert_checkpoint_compatible(
        payload,
        penetration_rate=tag,
        graph_sha256=str(graph_info["graph_sha256"]),
        data_file_sha256=data_info["file_hashes"][f"{tag}/train.npz"],
        num_nodes=int(config["num_nodes"]),
        n_his=int(config["n_his"]),
        output_steps=int(config["output_steps"]),
        input_channels=int(config["input_channels"]),
        output_channels=int(config["output_channels"]),
    )
    model.load_state_dict(payload["model_state_dict"])
    loaders = build_dataloaders(data_root, tag, config, data_info["sample_counts"])
    criterion = STGCNPredictionLoss()
    for split in ("validation", "test"):
        result = evaluate_one_epoch(
            model,
            loaders["loaders"][split],
            criterion,
            device,
            float(data_info["mean_y_full"]),
            float(data_info["std_y_full"]),
        )
        _write_split_outputs(
            output_dir,
            split,
            result,
            write_tables=(split == "test"),
            node_ids=list(data_info["node_ids"]),
        )
    print(f"wrote evaluation artifacts to {output_dir.as_posix()}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except ReimplementationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
