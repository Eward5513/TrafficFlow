"""Optional path overrides for Xuancheng26 model preparation.

Defaults stay on the completed ``fcd_data`` / ``training_data`` batch. Pass
``--fcd-root``, ``--net-file``, and ``--training-root`` to write another batch
with the same subdirectory layout.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import ModuleType

ENV_FCD_ROOT = "XUANCHENG26_FCD_ROOT"
ENV_NET_FILE = "XUANCHENG26_NET_FILE"
ENV_TRAINING_ROOT = "XUANCHENG26_TRAINING_ROOT"

DEFAULT_FCD_ROOT = Path("/data5/tzhang174/TrafficFlow/xuancheng26/simulation_data/fcd_data")
DEFAULT_NET_FILE = Path("/home/tzhang174/TrafficFlow/dataset/xuancheng26/xuancheng_tls_rebuilt.net.xml")
DEFAULT_TRAINING_ROOT = Path("/data5/tzhang174/TrafficFlow/xuancheng26/training_data")


def add_path_arguments(
    parser: argparse.ArgumentParser,
    *,
    fcd: bool = False,
    net: bool = False,
) -> None:
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    if fcd:
        parser.add_argument("--fcd-root", type=Path, default=DEFAULT_FCD_ROOT)
    if net:
        parser.add_argument("--net-file", type=Path, default=DEFAULT_NET_FILE)


def resolved(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def bind(module: ModuleType, **values: object) -> None:
    for name, value in values.items():
        setattr(module, name, value)


def publish_runtime(
    *,
    fcd_root: Path | None = None,
    net_file: Path | None = None,
    training_root: Path | None = None,
) -> None:
    """Expose the selected paths to spawned worker processes."""
    if fcd_root is not None:
        os.environ[ENV_FCD_ROOT] = str(fcd_root)
    if net_file is not None:
        os.environ[ENV_NET_FILE] = str(net_file)
    if training_root is not None:
        os.environ[ENV_TRAINING_ROOT] = str(training_root)


def _env_path(name: str) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"worker is missing {name}")
    return Path(value)


def _bind_running_script(script_name: str, **values: object) -> None:
    """Point both the spawned module and the task function at the selected paths.

    ``runpy`` keeps the function globals separate from ``sys.modules['__main__']``.
    Updating only the module still leaves ``process_day`` on the default paths.
    """
    import sys

    for mod_name in ("__main__", "__mp_main__"):
        module = sys.modules.get(mod_name)
        main_file = getattr(module, "__file__", None) if module is not None else None
        if not main_file or Path(main_file).name != script_name:
            continue
        bind(module, **values)
        for fn_name in ("process_day", "discover_dates"):
            fn = getattr(module, fn_name, None)
            fn_globals = getattr(fn, "__globals__", None)
            if fn_globals is not None:
                fn_globals.update(values)


def init_full_flow_worker() -> None:
    import generate_r_level_full_flow as full

    root = _env_path(ENV_TRAINING_ROOT)
    out_root = root / "full_flow" / "r-level"
    values = dict(
        FCD_ROOT=_env_path(ENV_FCD_ROOT),
        NET_FILE=_env_path(ENV_NET_FILE),
        OUT_ROOT=out_root,
        SUMMARY_PATH=out_root / "generation_summary.csv",
    )
    bind(full, **values)
    _bind_running_script("generate_r_level_full_flow.py", **values)


def init_partial_flow_worker() -> None:
    import generate_r_level_full_flow as full
    import generate_r_level_partial_flow as partial

    root = _env_path(ENV_TRAINING_ROOT)
    fcd_root = _env_path(ENV_FCD_ROOT)
    net_file = _env_path(ENV_NET_FILE)
    full_root = root / "full_flow" / "r-level"
    out_root = root / "partial_flow"
    bind(full, FCD_ROOT=fcd_root, NET_FILE=net_file, OUT_ROOT=full_root)
    values = dict(
        FCD_ROOT=fcd_root,
        NET_FILE=net_file,
        FULL_ROOT=full_root,
        SAMPLED_ROOT=root / "sampled_vehicle",
        OUT_ROOT=out_root,
        SUMMARY_PATH=out_root / "generation_summary.csv",
    )
    bind(partial, **values)
    _bind_running_script("generate_r_level_partial_flow.py", **values)
