"""Train MSTT on one task and one penetration rate.

    python MSTT/train.py --task I2I --penetration p05
    python MSTT/train.py --task All2All --penetration p30 --device cuda:2
    python MSTT/train.py --task All2AE --penetration p70 --batch-size 16 --amp bf16

Every config key is also a flag (``--d-model``, ``--use-gated-attention 1``,
...). ``--max-train-batches`` / ``--max-eval-batches`` make a smoke run that
writes under ``<output_root>/_smoke`` instead of the official tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_root import PathLayoutError, assert_training_results_output  # noqa: E402
from MSTT import MODEL_NAME  # noqa: E402
from MSTT.config import (TASKS, add_config_arguments, load_config, overlay_args, rate_tag,
                         validate_config, validate_conservation_task, task_spec, conservation_output_root)  # noqa: E402


def resolve_output_root(config: dict, *, explicit_conservation: bool = False) -> Path:
    root = conservation_output_root(config, Path(config["output_root"]), explicit=explicit_conservation)
    if (config.get("max_train_batches") is not None or config.get("max_eval_batches") is not None) and "_smoke" not in root.parts:
        root = root / "_smoke"
    try:
        return assert_training_results_output(root, model_name=MODEL_NAME)
    except PathLayoutError as exc:
        raise SystemExit(f"error: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MSTT on one Xuancheng26 task and penetration.")
    parser.add_argument("--task", required=True, choices=list(TASKS))
    parser.add_argument("--penetration", "--rate", dest="penetration", required=True, help="p05 ... p70 (or 5 ... 70)")
    parser.add_argument("--config", type=Path, default=None, help="JSON overriding the defaults in MSTT/config.py")
    parser.add_argument("--overwrite", action="store_true", help="retrain even if the run is complete")
    parser.add_argument("--resume", type=Path, default=None, help="explicit checkpoint to resume from")
    parser.add_argument("--gpu", type=int, default=None, help="shortcut for --device cuda:<gpu>")
    add_config_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    config = overlay_args(load_config(args.config), args)
    if args.gpu is not None:
        config["device"] = f"cuda:{args.gpu}"
    validate_config(config)
    validate_conservation_task(config, task_spec(args.task))
    output_root = resolve_output_root(config, explicit_conservation=args.task == "All2All" and args.conservation_mode is not None)
    config["output_root"] = output_root.as_posix()
    from MSTT.trainer import train_task

    summary = train_task(
        config,
        task=args.task,
        rate=rate_tag(args.penetration),
        output_root=output_root,
        overwrite=bool(args.overwrite),
        resume=args.resume,
    )
    print(json.dumps({key: summary.get(key) for key in ("task", "rate", "best_epoch", "test_mae", "test_rmse", "test_wape", "run_dir")}, indent=2))
    return summary


if __name__ == "__main__":
    main()
