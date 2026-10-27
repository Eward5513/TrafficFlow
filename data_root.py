"""Separate the git checkout from generated TrafficFlow data.

``CODE_ROOT`` is ``/home/tzhang174/TrafficFlow`` (scripts, configs, tests, docs).
``DATA_ROOT`` is ``/data5/tzhang174/TrafficFlow`` (OCR, simulation, prepared
training files, and training results). Override the data disk with
``TRAFFICFLOW_DATA_ROOT``.

SUMO ``sumocfg`` paths stay relative to each config file.
"""

from __future__ import annotations

import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("/data5/tzhang174/TrafficFlow")

DATA_PREFIXES = ("ocr_data/", "simulation_data/", "training_data/", "training_results/")
MODEL_NAMES = (
    "stgcn",
    "dcrnn",
    "graph_wavenet",
    "stsgcn",
    "stfgnn",
    "pdformer",
    "stid",
    "staeformer",
    "s2s_gcn_lstm",
)

# Old in-repo relative locations -> current disk layout. Existing files stay put.
_LEGACY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("reimplementation/stgcn/prepared_data/r-only", "training_data/r-only"),
    (
        "reimplementation/dcrnn/prepared_data/r-only/adjacency_matrix",
        "training_data/r-only/dcrnn",
    ),
    ("reimplementation/stfgnn/prepared_data/r-only/temporal_graph", "training_data/stfgnn"),
    ("reimplementation/pdformer/prepared_data/r-only/relations", "training_data/pdformer"),
    ("analysis/simulation/data", "simulation_data"),
    ("analysis/graph/r_graph/r_nodes.csv", "training_data/r-only/node_mapping.csv"),
    ("reimplementation/experiments/r_only/stgcn", "training_results/stgcn"),
    ("reimplementation/stgcn/experiments/r_only", "training_results/stgcn"),
    ("reimplementation/dcrnn/experiments/r_only", "training_results/dcrnn"),
    ("reimplementation/graph_wavenet/experiments/r_only", "training_results/graph_wavenet"),
    ("reimplementation/stsgcn/experiments/r_only", "training_results/stsgcn"),
    ("reimplementation/stfgnn/experiments/r_only", "training_results/stfgnn"),
    ("reimplementation/pdformer/experiments/r_only", "training_results/pdformer"),
    ("reimplementation/stid/experiments/r_only", "training_results/stid"),
    ("reimplementation/staeformer/experiments/r_only", "training_results/staeformer"),
)


class PathLayoutError(RuntimeError):
    """Raised when a path would mix the code checkout with the data disk."""


def code_root() -> Path:
    return CODE_ROOT


def data_root() -> Path:
    env = os.environ.get("TRAFFICFLOW_DATA_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    if DEFAULT_DATA_ROOT.is_dir():
        return DEFAULT_DATA_ROOT.resolve()
    return CODE_ROOT


def ocr_data_dir() -> Path:
    return data_root() / "ocr_data"


def simulation_data_dir() -> Path:
    candidate = data_root() / "simulation_data"
    legacy = CODE_ROOT / "analysis" / "simulation" / "data"
    if candidate.is_dir() or not legacy.is_dir():
        return candidate
    return legacy


def training_data_root() -> Path:
    return data_root() / "training_data"


def shared_training_data_dir() -> Path:
    """Existing shared R-only windows, scalers, and STGCN adjacency."""
    return training_data_root() / "r-only"


def training_data_dir() -> Path:
    """Prepared NPZ package. Keep the historical name used by train scripts."""
    candidate = shared_training_data_dir()
    legacy = CODE_ROOT / "reimplementation" / "stgcn" / "prepared_data" / "r-only"
    if candidate.is_dir() or not legacy.is_dir():
        return candidate
    return legacy


def model_training_data_dir(model_name: str) -> Path:
    """Directory for model-specific preprocessing that is not the shared NPZ pack."""
    name = str(model_name).strip()
    if name not in MODEL_NAMES:
        raise PathLayoutError(f"unknown model name for training_data: {model_name!r}")
    return training_data_root() / name


def dcrnn_adjacency_dir() -> Path:
    """Read path: keep using the existing files under r-only/dcrnn if present."""
    current = shared_training_data_dir() / "dcrnn"
    expected = model_training_data_dir("dcrnn")
    if current.is_dir():
        return current
    return expected


def stgcn_adjacency_dir() -> Path:
    """Shared STGCN/STSGCN/STFGNN/PDFormer topology lives with the NPZ pack."""
    return shared_training_data_dir() / "adjacency_matrix"


def training_results_root() -> Path:
    return data_root() / "training_results"


def model_training_results_dir(model_name: str) -> Path:
    name = str(model_name).strip()
    if name not in MODEL_NAMES:
        raise PathLayoutError(f"unknown model name for training_results: {model_name!r}")
    return training_results_root() / name


def xuancheng26_training_results_root() -> Path:
    """Xuancheng26 checkpoints and metrics. Separate from the shared results root."""
    return data_root() / "xuancheng26" / "training_results"


def xuancheng26_model_results_dir(model_name: str) -> Path:
    name = str(model_name).strip()
    if name not in MODEL_NAMES:
        raise PathLayoutError(f"unknown model name for training_results: {model_name!r}")
    return xuancheng26_training_results_root() / name


def rate_training_results_dir(model_name: str, rate: str, seed: int = 42) -> Path:
    tag = str(rate).strip()
    if not tag.startswith("p"):
        tag = f"p{int(tag):02d}"
    return model_training_results_dir(model_name) / tag / f"seed_{int(seed)}"


def subgraph_trajectories_dir() -> Path:
    return simulation_data_dir() / "processed" / "subgraph_trajectories"


def subgraph_file() -> Path:
    return simulation_data_dir() / "subgraph.txt"


def node_table() -> Path:
    """R-node order. ``node_mapping.csv`` has the same columns as ``r_nodes.csv``."""
    return shared_training_data_dir() / "node_mapping.csv"


def xuancheng26_training_root() -> Path:
    """Xuancheng26 prepared windows, graph tables, and model configs."""
    return data_root() / "xuancheng26" / "training_data"


def xuancheng26_model_config(model_name: str) -> Path:
    name = str(model_name).strip()
    return xuancheng26_training_root() / "config" / name / f"r_only_{name}.json"


def r_graph_dir() -> Path:
    return CODE_ROOT / "analysis" / "graph" / "r_graph"


def runtime_roots() -> dict[str, str]:
    return {
        "code_root": str(code_root()),
        "data_root": str(data_root()),
        "ocr_data": str(ocr_data_dir()),
        "simulation_data": str(simulation_data_dir()),
        "training_data": str(training_data_root()),
        "shared_training_data": str(shared_training_data_dir()),
        "training_results": str(training_results_root()),
    }


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def assert_code_and_data_separated() -> None:
    if code_root().resolve() == data_root().resolve():
        raise PathLayoutError(
            f"CODE_ROOT and DATA_ROOT must differ; both resolved to {code_root()}"
        )


def assert_not_code_root_output(path: Path, *, what: str = "output") -> None:
    resolved = Path(path).resolve()
    if _is_relative_to(resolved, code_root()):
        raise PathLayoutError(
            f"{what} must not be written under CODE_ROOT {code_root()}: {resolved}"
        )


def assert_not_source_data_output(path: Path, *, what: str = "output") -> None:
    resolved = Path(path).resolve()
    for label, parent in (
        ("ocr_data", ocr_data_dir()),
        ("simulation_data", simulation_data_dir()),
    ):
        if parent.exists() and _is_relative_to(resolved, parent):
            raise PathLayoutError(f"{what} must not be written under {label}: {resolved}")


def assert_training_results_output(path: Path, *, model_name: str | None = None) -> Path:
    """Official training/eval outputs belong under DATA_ROOT/training_results/."""
    assert_code_and_data_separated()
    resolved = Path(path).expanduser().resolve()
    assert_not_code_root_output(resolved, what="training_results")
    assert_not_source_data_output(resolved, what="training_results")
    allowed_roots = (
        training_results_root().resolve(),
        xuancheng26_training_results_root().resolve(),
    )
    if not any(resolved == root or _is_relative_to(resolved, root) for root in allowed_roots):
        raise PathLayoutError(
            f"training output must be under {allowed_roots[0]} or {allowed_roots[1]}, got {resolved}"
        )
    if _is_relative_to(resolved, training_data_root()):
        raise PathLayoutError(f"training_results must not be inside training_data: {resolved}")
    xuancheng_data = data_root() / "xuancheng26" / "training_data"
    if xuancheng_data.exists() and _is_relative_to(resolved, xuancheng_data.resolve()):
        raise PathLayoutError(f"training_results must not be inside training_data: {resolved}")
    if model_name is not None:
        expected_dirs = (
            model_training_results_dir(model_name).resolve(),
            xuancheng26_model_results_dir(model_name).resolve(),
        )
        if not any(resolved == expected or _is_relative_to(resolved, expected) for expected in expected_dirs):
            raise PathLayoutError(
                f"{model_name} output must be under {expected_dirs[0]} or {expected_dirs[1]}, got {resolved}"
            )
    return resolved


def assert_model_preprocess_output(path: Path, *, model_name: str) -> Path:
    """New model-specific caches belong under DATA_ROOT/training_data/<model>/."""
    assert_code_and_data_separated()
    resolved = Path(path).expanduser().resolve()
    assert_not_code_root_output(resolved, what="training_data")
    assert_not_source_data_output(resolved, what="training_data")
    expected = model_training_data_dir(model_name).resolve()
    if resolved != expected and not _is_relative_to(resolved, expected):
        raise PathLayoutError(
            f"{model_name} preprocessing must be written under {expected}, got {resolved}"
        )
    shared = shared_training_data_dir().resolve()
    if shared.exists() and _is_relative_to(resolved, shared):
        raise PathLayoutError(
            f"refusing to write {model_name} files into shared training_data/r-only: {resolved}"
        )
    if _is_relative_to(resolved, training_results_root()):
        raise PathLayoutError(
            f"preprocessing must not be written into training_results: {resolved}"
        )
    return resolved


def map_legacy_relative(posix: str) -> str | None:
    for old, new in _LEGACY_PREFIXES:
        if posix == old or posix.startswith(old + "/"):
            return new + posix[len(old) :]
    return None


def resolve_repo_or_data_path(project_root: Path, value: str | Path) -> Path:
    """Resolve a config path against the data disk, with in-repo fallback."""
    path = Path(value)
    if path.is_absolute():
        return path
    posix = path.as_posix()
    mapped = map_legacy_relative(posix)
    if mapped is not None:
        new_path = (data_root() / mapped).resolve()
        old_path = (Path(project_root) / posix).resolve()
        if new_path.exists() or not old_path.exists():
            return new_path
        return old_path
    if posix.startswith(DATA_PREFIXES):
        return (data_root() / posix).resolve()
    return (Path(project_root) / path).resolve()
