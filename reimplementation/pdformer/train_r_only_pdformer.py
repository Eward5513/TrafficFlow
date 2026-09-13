"""Train R-only PDFormer. Does not write into prepared data, graphs, DCRNN, GWN, STSGCN, or STFGNN."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from reimplementation.common.data.r_only_npz_dataset import rate_tag
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.utils.atomic_io import atomic_write_json
from reimplementation.common.utils.hashing import sha256_file
from reimplementation.common.utils.structured_logging import JsonlLogger
from reimplementation.stgcn.engine import resolve_path
from reimplementation.pdformer.engine import (
    collect_data_info,
    collect_graph_info,
    collect_smoke_data_info,
    run_all_rates,
    run_p70_smoke,
)
from reimplementation.pdformer.model.pdformer import CODE_VERSION

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = SCRIPT_PATH.parent / "configs" / "r_only_pdformer.json"

ORIGINAL_TO_MIGRATED = {
    "reference/PDFormer/libcity/model/traffic_flow_prediction/PDFormer.py::DataEmbedding": "reimplementation/pdformer/model/embedding.py::DataEmbedding",
    "reference/PDFormer/libcity/model/traffic_flow_prediction/PDFormer.py::STSelfAttention": "reimplementation/pdformer/model/attention.py::STSelfAttention",
    "reference/PDFormer/libcity/model/traffic_flow_prediction/PDFormer.py::STEncoderBlock": "reimplementation/pdformer/model/layers.py::STEncoderBlock",
    "reference/PDFormer/libcity/model/traffic_flow_prediction/PDFormer.py::PDFormer": "reimplementation/pdformer/model/pdformer.py::PDFormer",
    "reference/PDFormer/libcity/model/traffic_flow_prediction/PDFormer.py::drop_path": "reimplementation/pdformer/model/layers.py::drop_path",
    "reference/PDFormer/libcity/data/dataset/pdformer_dataset.py::_load_rel hop Floyd": "reimplementation/pdformer/graph.py::hop_shortest_path",
    "reference/PDFormer/libcity/data/dataset/pdformer_dataset.py::_get_dtw fastdtw radius=6": "reimplementation/pdformer/graph.py::fastdtw",
    "reference/PDFormer/libcity/executor/pdformer_executor.py::_cal_lape": "reimplementation/pdformer/graph.py::laplacian_positional_encoding",
    "reference/PDFormer/libcity/model/loss.py::huber_loss": "reimplementation/pdformer/losses.py::huber_loss",
    "output_window": "original 12 future steps -> R-only output_window=1 last-observed-step",
}


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train R-only PDFormer (PyTorch port of original LibCity PDFormer).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--r-nodes", dest="r_nodes", type=Path, default=None)
    parser.add_argument("--adjacency", dest="undirected_topology", type=Path, default=None)
    parser.add_argument("--relations-dir", dest="relations_dir", type=Path, default=None)
    parser.add_argument("--normalization", type=Path, default=None)
    parser.add_argument("--split-manifest", dest="split_manifest", type=Path, default=None)
    parser.add_argument("--output-root", dest="output_root", type=Path, default=None)
    parser.add_argument("--rates", type=int, nargs="+", default=None)
    parser.add_argument("--n-his", dest="n_his", type=int, default=None)
    parser.add_argument("--output-window", dest="output_window", type=int, default=None)
    parser.add_argument("--target-mode", dest="target_mode", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
    parser.add_argument("--lr-scheduler", dest="lr_scheduler", type=str, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test-only", action="store_true")
    parser.add_argument("--embed-dim", dest="embed_dim", type=int, default=None)
    parser.add_argument("--lape-dim", dest="lape_dim", type=int, default=None)
    parser.add_argument("--num-heads", dest="num_heads", type=int, default=None)
    parser.add_argument("--geo-num-heads", dest="geo_num_heads", type=int, default=None)
    parser.add_argument("--sem-num-heads", dest="sem_num_heads", type=int, default=None)
    parser.add_argument("--temporal-num-heads", dest="t_num_heads", type=int, default=None)
    parser.add_argument("--enc-depth", dest="enc_depth", type=int, default=None)
    parser.add_argument("--mlp-ratio", dest="mlp_ratio", type=float, default=None)
    parser.add_argument("--drop", type=float, default=None)
    parser.add_argument("--attn-drop", dest="attn_drop", type=float, default=None)
    parser.add_argument("--drop-path", dest="drop_path", type=float, default=None)
    parser.add_argument("--far-mask-delta", dest="far_mask_delta", type=int, default=None)
    parser.add_argument("--dtw-delta", dest="dtw_delta", type=int, default=None)
    parser.add_argument("--pattern-key-count", dest="n_cluster", type=int, default=None)
    parser.add_argument("--s-attn-size", dest="s_attn_size", type=int, default=None)
    parser.add_argument("--type-ln", dest="type_ln", type=str, default=None)
    return parser.parse_args()


def overlay(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    mapping = {
        "data_root": args.data_root,
        "r_nodes": args.r_nodes,
        "undirected_topology": args.undirected_topology,
        "relations_dir": args.relations_dir,
        "normalization": args.normalization,
        "split_manifest": args.split_manifest,
        "output_root": args.output_root,
        "output_window": args.output_window,
        "target_mode": args.target_mode,
        "seed": args.seed,
        "device": args.device,
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "lr_scheduler": args.lr_scheduler,
        "embed_dim": args.embed_dim,
        "lape_dim": args.lape_dim,
        "geo_num_heads": args.geo_num_heads,
        "sem_num_heads": args.sem_num_heads,
        "t_num_heads": args.t_num_heads,
        "enc_depth": args.enc_depth,
        "mlp_ratio": args.mlp_ratio,
        "drop": args.drop,
        "attn_drop": args.attn_drop,
        "drop_path": args.drop_path,
        "far_mask_delta": args.far_mask_delta,
        "dtw_delta": args.dtw_delta,
        "n_cluster": args.n_cluster,
        "s_attn_size": args.s_attn_size,
        "type_ln": args.type_ln,
    }
    for key, value in mapping.items():
        if value is not None:
            config[key] = value if not isinstance(value, Path) else value.as_posix()
    if args.n_his is not None:
        config["n_his"] = int(args.n_his)
        config["input_window"] = int(args.n_his)
    if args.num_heads is not None:
        config["num_heads"] = int(args.num_heads)
    if args.rates is not None:
        config["rates"] = [rate_tag(item) for item in args.rates]
    return config


def _assert_task(config: dict[str, Any]) -> None:
    if str(config.get("target_mode")) != "last-observed-step":
        raise ReimplementationError("target_mode must be last-observed-step")
    if int(config.get("output_window", config.get("horizon", 0))) != 1:
        raise ReimplementationError("R-only PDFormer output_window must be 1")
    if int(config.get("input_window", config.get("n_his", 0))) != 12:
        raise ReimplementationError("R-only PDFormer input_window must be 12")
    geo = int(config.get("geo_num_heads", 4))
    sem = int(config.get("sem_num_heads", 2))
    temporal = int(config.get("t_num_heads", config.get("temporal_num_heads", 2)))
    total = geo + sem + temporal
    if "num_heads" in config and int(config["num_heads"]) != total:
        raise ReimplementationError(
            f"num_heads={config['num_heads']} != geo+sem+t {total}; refusing to retune heads"
        )
    if int(config.get("embed_dim", 64)) % total != 0:
        raise ReimplementationError("embed_dim is not divisible by geo+sem+t heads")
    if str(config.get("type_ln", "pre")) != "pre":
        raise ReimplementationError("PeMS04/08 target config uses type_ln=pre")
    if str(config.get("type_short_path", "hop")) != "hop":
        raise ReimplementationError("PeMS04/08 target config uses type_short_path=hop")
    if str(config.get("set_loss", "huber")) != "huber":
        raise ReimplementationError("PeMS04/08 target config uses set_loss=huber")
    if bool(config.get("add_day_in_week", False)):
        if not config.get("weekday_mapping"):
            raise ReimplementationError(
                "add_day_in_week=true requires a calendar weekday mapping; "
                "simulation day_index % 7 is forbidden"
            )


def source_manifest(root: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    skip_parts = {"prepared_data", "experiments", "__pycache__", "tests", "test_artifacts"}
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
        "reference/PDFormer/libcity/model/traffic_flow_prediction/PDFormer.py",
        "reference/PDFormer/libcity/data/dataset/pdformer_dataset.py",
        "reference/PDFormer/libcity/executor/pdformer_executor.py",
        "reference/PDFormer/PeMS04.json",
        "reference/PDFormer/LICENSE",
    ]
    return {name: sha256_file(project_root / name) for name in files if (project_root / name).is_file()}


def main() -> None:
    args = parse_args()
    config = overlay(load_config(args.config), args)
    if "input_window" not in config:
        config["input_window"] = int(config.get("n_his", 12))
    _assert_task(config)
    data_root = resolve_path(PROJECT_ROOT, config["data_root"])
    output_root = resolve_path(PROJECT_ROOT, config["output_root"])
    protected_dirs = [
        PROJECT_ROOT / "reimplementation" / "dcrnn",
        PROJECT_ROOT / "reimplementation" / "graph_wavenet",
        PROJECT_ROOT / "reimplementation" / "stsgcn",
        PROJECT_ROOT / "reimplementation" / "stfgnn",
        PROJECT_ROOT / "reimplementation" / "stgcn" / "prepared_data",
        PROJECT_ROOT / "analysis" / "graph" / "r_graph",
    ]
    for path in protected_dirs:
        if output_root == path or path in output_root.parents:
            raise ReimplementationError(f"refusing to write PDFormer outputs into {path}")
    if args.smoke_test_only:
        config["device"] = "cpu"
        config["random_flip"] = False
        data_info = collect_smoke_data_info(data_root, config)
        graph_info = collect_graph_info(config, allow_test_only=True, project_root=PROJECT_ROOT)
        payload = run_p70_smoke(
            config,
            data_root=data_root,
            graph_info=graph_info,
            data_info=data_info,
        )
        payload["original_to_migrated"] = ORIGINAL_TO_MIGRATED
        payload["python"] = sys.version
        payload["torch"] = torch.__version__
        payload["platform"] = platform.platform()
        print(json.dumps({k: payload[k] for k in ("status", "training_loss", "prediction_shape")}, indent=2))
        return
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise ReimplementationError(f"{output_root} is not empty; pass --overwrite")
    data_info = collect_data_info(data_root, config)
    graph_info = collect_graph_info(config, allow_test_only=False, project_root=PROJECT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "resolved_config.json", config)
    atomic_write_json(output_root / "source_code_manifest.json", source_manifest(SCRIPT_PATH.parent))
    atomic_write_json(output_root / "original_hashes.json", original_hashes(PROJECT_ROOT))
    logger = JsonlLogger(output_root / "training_log.jsonl")
    try:
        run_all_rates(
            config,
            data_root=data_root,
            output_root=output_root,
            graph_info=graph_info,
            data_info=data_info,
            logger=logger,
            resume_path=args.resume,
        )
    finally:
        logger.close()


if __name__ == "__main__":
    try:
        main()
    except ReimplementationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
