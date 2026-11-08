"""Train Xuancheng26 ASTTN. Does not write into prepared data or other models."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_root import runtime_roots
from reimplementation.asttn.engine import collect_data_info, run_all_rates
from reimplementation.asttn.provenance import CODE_VERSION
from reimplementation.asttn.validation import assert_task_config
from reimplementation.common.data.r_only_npz_dataset import rate_tag
from reimplementation.common.data.state_windows import add_state_type_arguments, apply_state_types
from reimplementation.common.errors import ReimplementationError
from reimplementation.common.utils.atomic_io import atomic_write_json
from reimplementation.common.utils.structured_logging import JsonlLogger
from reimplementation.stgcn.engine import resolve_path, resolve_training_output

DEFAULT_CONFIG = SCRIPT_PATH.parent / "configs" / "r_only_asttn.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Xuancheng26 ASTTN.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", dest="data_root", type=Path, default=None)
    parser.add_argument("--output-root", dest="output_root", type=Path, default=None)
    parser.add_argument("--rates", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    add_state_type_arguments(parser)
    return parser.parse_args()


def overlay(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    for key, value in {
        "data_root": args.data_root,
        "output_root": args.output_root,
        "seed": args.seed,
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }.items():
        if value is not None:
            config[key] = value.as_posix() if isinstance(value, Path) else value
    if args.rates is not None:
        config["rates"] = [rate_tag(item) for item in args.rates]
    return config


def main() -> None:
    args = parse_args()
    config = overlay(json.loads(args.config.read_text(encoding="utf-8")), args)
    config = apply_state_types(config, args, model_name="asttn")
    config["n_his"] = 12
    config["output_steps"] = 1
    assert_task_config(config)
    data_root = resolve_path(PROJECT_ROOT, config["data_root"])
    output_root = resolve_training_output(PROJECT_ROOT, config["output_root"], model_name="asttn")
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / "resolved_config.json", config)
    atomic_write_json(
        output_root / "environment.json",
        {"code_version": CODE_VERSION, **runtime_roots(), "python": sys.version, "torch": torch.__version__, "platform": platform.platform()},
    )
    data_info = collect_data_info(data_root, config, project_root=PROJECT_ROOT)
    logger = JsonlLogger(output_root / "training_log.jsonl")
    try:
        run_all_rates(
            config,
            data_root=data_root,
            output_root=output_root,
            data_info=data_info,
            logger=logger,
            resume_path=args.resume,
            overwrite=bool(args.overwrite),
        )
    finally:
        logger.close()


if __name__ == "__main__":
    try:
        main()
    except ReimplementationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
