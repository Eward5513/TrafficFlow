"""Train R-only STSGCN. Does not write into prepared data, graphs, DCRNN, GWN, or STGCN."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from reimplementation.common.data.r_only_npz_dataset import rate_tag
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.utils.atomic_io import atomic_write_json
from reimplementation.common.utils.hashing import sha256_file
from reimplementation.common.utils.structured_logging import JsonlLogger
from reimplementation.stsgcn.engine import (
    collect_data_info,
    resolve_path,
    run_all_rates,
    run_p70_smoke,
)
from reimplementation.stsgcn.model.stsgcn import CODE_VERSION
from reimplementation.stsgcn.validation import load_node_ids, validate_stsgcn_graph

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = SCRIPT_PATH.parent / "configs" / "r_only_stsgcn.json"

ORIGINAL_TO_MIGRATED = {
    "reference/STSGCN/utils.py::construct_adj": "reimplementation/stsgcn/model/graph.py::construct_localized_adjacency",
    "reference/STSGCN/utils.py::get_adjacency_matrix type_=connectivity": "stgcn_undirected_topology.npy as 0/1 undirected A",
    "reference/STSGCN/models/stsgcn.py::gcn_operation": "reimplementation/stsgcn/model/layers.py::STSGCNGraphConv",
    "reference/STSGCN/models/stsgcn.py::stsgcm": "reimplementation/stsgcn/model/layers.py::STSGCM",
    "reference/STSGCN/models/stsgcn.py::stsgcl / sthgcn_layer_individual": "reimplementation/stsgcn/model/layers.py::STSGCL",
    "reference/STSGCN/models/stsgcn.py::position_embedding": "reimplementation/stsgcn/model/layers.py::PositionEmbedding",
    "reference/STSGCN/utils.py::construct_model first_layer_embedding": "STSGCN.input_projection + ReLU, once",
    "reference/STSGCN/models/stsgcn.py::output_layer": "reimplementation/stsgcn/model/layers.py::OutputLayer",
    "reference/STSGCN/models/stsgcn.py::stsgcn": "reimplementation/stsgcn/model/stsgcn.py::STSGCN",
    "reference/STSGCN/models/stsgcn.py::huber_loss rho=1": "reimplementation/stsgcn/losses.py::HuberRawLoss",
    "horizon": "original num_for_predict=12 -> R-only horizon=1 last-observed-step",
}


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train R-only STSGCN (PyTorch port of original MXNet STSGCN).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--spatial-adjacency", dest="spatial_adjacency", type=Path, default=None)
    parser.add_argument("--graph-metadata", dest="graph_metadata", type=Path, default=None)
    parser.add_argument("--graph-validation", dest="graph_validation", type=Path, default=None)
    parser.add_argument("--r-nodes", dest="r_nodes", type=Path, default=None)
    parser.add_argument("--normalization", type=Path, default=None)
    parser.add_argument("--split-manifest", dest="split_manifest", type=Path, default=None)
    parser.add_argument("--output-root", dest="output_root", type=Path, default=None)
    parser.add_argument("--rates", type=int, nargs="+", default=None)
    parser.add_argument("--rate", type=int, default=None)
    parser.add_argument("--seq-length", dest="seq_length", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--target-mode", dest="target_mode", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test-only", action="store_true")
    parser.add_argument("--module-type", dest="module_type", type=str, default=None)
    parser.add_argument("--activation", dest="act_type", type=str, default=None)
    parser.add_argument("--use-mask", dest="use_mask", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--use-spatial-embedding",
        dest="spatial_emb",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-temporal-embedding",
        dest="temporal_emb",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


def overlay(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    mapping = {
        "data_root": args.data_root,
        "spatial_adjacency": args.spatial_adjacency,
        "graph_metadata": args.graph_metadata,
        "graph_validation": args.graph_validation,
        "r_nodes": args.r_nodes,
        "normalization": args.normalization,
        "split_manifest": args.split_manifest,
        "output_root": args.output_root,
        "horizon": args.horizon,
        "target_mode": args.target_mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "seed": args.seed,
        "device": args.device,
        "num_workers": args.num_workers,
        "module_type": args.module_type,
        "act_type": args.act_type,
    }
    for key, value in mapping.items():
        if value is not None:
            config[key] = value if not isinstance(value, Path) else value.as_posix()
    if args.seq_length is not None:
        config["seq_len"] = int(args.seq_length)
        config["seq_length"] = int(args.seq_length)
        config["n_his"] = int(args.seq_length)
    if args.use_mask is not None:
        config["use_mask"] = bool(args.use_mask)
    if args.spatial_emb is not None:
        config["spatial_emb"] = bool(args.spatial_emb)
        config["use_spatial_embedding"] = bool(args.spatial_emb)
    if args.temporal_emb is not None:
        config["temporal_emb"] = bool(args.temporal_emb)
        config["use_temporal_embedding"] = bool(args.temporal_emb)
    if args.rate is not None:
        config["rates"] = [rate_tag(args.rate)]
    elif args.rates is not None:
        config["rates"] = [rate_tag(item) for item in args.rates]
    return config


def _assert_task(config: dict[str, Any]) -> None:
    if str(config.get("target_mode")) != "last-observed-step":
        raise ReimplementationError("target_mode must be last-observed-step")
    if int(config.get("horizon", 0)) != 1:
        raise ReimplementationError("R-only STSGCN horizon must be 1")
    if int(config.get("seq_len", config.get("seq_length", 0))) != 12:
        raise ReimplementationError("R-only STSGCN seq_len must be 12")
    if str(config.get("module_type")) != "individual":
        raise ReimplementationError("official STSGCN PEMS03 config uses module_type=individual")
    if str(config.get("act_type", config.get("activation"))) != "GLU":
        raise ReimplementationError("official STSGCN PEMS03 config uses act_type=GLU")
    if not bool(config.get("use_mask", True)):
        raise ReimplementationError("official STSGCN PEMS03 config uses use_mask=true")
    if not bool(config.get("temporal_emb", True)) or not bool(config.get("spatial_emb", True)):
        raise ReimplementationError("official STSGCN PEMS03 config enables spatial and temporal embeddings")


def source_manifest(root: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    skip_parts = {"prepared_data", "experiments", "__pycache__", "tests"}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in skip_parts for part in path.parts):
            continue
        if path.suffix not in {".py", ".json", ".md"}:
            continue
        records[path.relative_to(root).as_posix()] = sha256_file(path)
    return records


def original_hashes(project_root: Path) -> dict[str, str]:
    files = [
        "reference/STSGCN/models/stsgcn.py",
        "reference/STSGCN/utils.py",
        "reference/STSGCN/main.py",
        "reference/STSGCN/README.md",
    ]
    return {name: sha256_file(project_root / name) for name in files if (project_root / name).is_file()}


def main() -> None:
    args = parse_args()
    config = overlay(load_config(args.config), args)
    if "seq_len" not in config:
        config["seq_len"] = int(config.get("seq_length", config.get("n_his", 12)))
    _assert_task(config)
    data_root = resolve_path(PROJECT_ROOT, config["data_root"])
    topology = resolve_path(PROJECT_ROOT, config["spatial_adjacency"])
    metadata_path = resolve_path(PROJECT_ROOT, config["graph_metadata"])
    graph_validation = resolve_path(PROJECT_ROOT, config["graph_validation"])
    r_nodes = resolve_path(PROJECT_ROOT, config["r_nodes"])
    output_root = resolve_path(PROJECT_ROOT, config["output_root"])
    protected_dirs = [
        PROJECT_ROOT / "reimplementation" / "dcrnn",
        PROJECT_ROOT / "reimplementation" / "graph_wavenet",
        PROJECT_ROOT / "reimplementation" / "stgcn" / "experiments",
        PROJECT_ROOT / "reference" / "STSGCN",
    ]
    if output_root.resolve() in {path.resolve() for path in protected_dirs}:
        raise ReimplementationError("STSGCN output_root must not be a DCRNN/GWN/STGCN/original directory")
    protected = [
        topology,
        metadata_path,
        graph_validation,
        r_nodes,
        data_root / "normalization.json",
        PROJECT_ROOT / "reference" / "STSGCN" / "models" / "stsgcn.py",
        PROJECT_ROOT / "reference" / "STSGCN" / "utils.py",
        PROJECT_ROOT / "reimplementation" / "dcrnn" / "train_r_only_dcrnn.py",
        PROJECT_ROOT / "reimplementation" / "graph_wavenet" / "train_r_only_gwn.py",
    ]
    before = {str(path): sha256_file(path) for path in protected if path.is_file()}

    if args.smoke_test_only:
        node_ids = load_node_ids(data_root / "node_mapping.csv")
        graph_info = validate_stsgcn_graph(
            topology_path=topology,
            metadata_path=metadata_path,
            validation_path=graph_validation,
            r_nodes_path=r_nodes,
            node_ids=node_ids,
            expected_nodes=int(config["num_nodes"]),
        )
        data_info = collect_data_info(data_root, config)
        with tempfile.TemporaryDirectory(prefix="stsgcn_smoke_") as tmp:
            logger = JsonlLogger(Path(tmp) / "smoke.jsonl")
            try:
                smoke = run_p70_smoke(
                    config,
                    data_root=data_root,
                    graph_info=graph_info,
                    data_info=data_info,
                    logger=logger,
                )
            finally:
                logger.close()
        after = {str(path): sha256_file(path) for path in protected if path.is_file()}
        if before != after:
            raise ReimplementationError("protected input files changed during STSGCN smoke")
        atomic_write_json(SCRIPT_PATH.parent / "smoke_test.json", smoke)
        print("[stsgcn] stage=smoke_testing discarded_model=yes did_not_write_experiments=yes", flush=True)
        return

    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite and args.resume is None:
        raise ReimplementationError(f"{output_root} is not empty; pass --overwrite or --resume")
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output_root / "experiment_manifest.json",
        {
            "model_name": "stsgcn",
            "framework": "pytorch",
            "code_version": CODE_VERSION,
            "target_mode": config.get("target_mode"),
            "seq_len": config.get("seq_len"),
            "horizon": config.get("horizon"),
            "num_nodes": config.get("num_nodes"),
            "seed": config.get("seed"),
            "rates": list(config["rates"]),
            "data_root": data_root.as_posix(),
            "spatial_adjacency": topology.as_posix(),
            "r_nodes": r_nodes.as_posix(),
            "output_root": output_root.as_posix(),
            "smoke_test_only": False,
            "resumed": args.resume is not None,
        },
    )
    atomic_write_json(output_root / "resolved_config.json", config)
    atomic_write_json(output_root / "original_to_migrated_mapping.json", ORIGINAL_TO_MIGRATED)
    atomic_write_json(
        output_root / "environment.json",
        {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "numpy": __import__("numpy").__version__,
            "platform": platform.platform(),
            "code_version": CODE_VERSION,
            "cuda_available": torch.cuda.is_available(),
            "device": str(config.get("device")),
            "command": sys.argv,
            "original_stsgcn_sha256": original_hashes(PROJECT_ROOT),
        },
    )
    atomic_write_json(output_root / "source_code_manifest.json", source_manifest(PROJECT_ROOT / "reimplementation" / "stsgcn"))
    logger = JsonlLogger(output_root / "training_log.jsonl")
    print(
        f"[stsgcn] stage=scanning output_root={output_root.as_posix()} rates={config['rates']} device={config.get('device')}",
        flush=True,
    )
    try:
        result = run_all_rates(
            config,
            project_root=PROJECT_ROOT,
            rates=list(config["rates"]),
            data_root=data_root,
            output_root=output_root,
            topology_path=topology,
            metadata_path=metadata_path,
            graph_validation_path=graph_validation,
            r_nodes_path=r_nodes,
            logger=logger,
            smoke_only=False,
            resume=args.resume,
        )
        after = {str(path): sha256_file(path) for path in protected if path.is_file()}
        if before != after:
            raise ReimplementationError("protected input files changed during STSGCN run")
        atomic_write_json(output_root / "data_runtime_validation.json", result["data_info"])
        atomic_write_json(output_root / "graph_runtime_validation.json", result["graph_info"])
        logger.log({"stage": "completed", "model_name": "stsgcn", "status": "ok"})
        print("[stsgcn] stage=completed status=ok", flush=True)
    except Exception as exc:
        logger.log({"stage": "failed", "model_name": "stsgcn", "status": "failed", "latest_error": str(exc)})
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    try:
        main()
    except ReimplementationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
