"""Separate the git checkout from generated TrafficFlow data.

``CODE_ROOT`` is ``/home/tzhang174/TrafficFlow`` (scripts, configs, tests, docs).
``DATA_ROOT`` is ``/data5/tzhang174/TrafficFlow`` (OCR, simulation, prepared
training files, and training results). Override the data disk with
``TRAFFICFLOW_DATA_ROOT``.

SUMO ``sumocfg`` paths stay relative to each config file.
"""

from __future__ import annotations

import json
import os
import re
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
    "tlast",
    "asttn",
    "stjgcn",
    "MSTT",
    "RF",
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


def xuancheng26_training_results_3s_root() -> Path:
    """Checkpoints for the Xuancheng26 3-second-phase batch."""
    return data_root() / "xuancheng26" / "training_results_3s"


def xuancheng26_model_results_dir(model_name: str) -> Path:
    name = str(model_name).strip()
    if name not in MODEL_NAMES:
        raise PathLayoutError(f"unknown model name for training_results: {model_name!r}")
    return xuancheng26_training_results_root() / name


def model_report_dir(model_name: str) -> Path:
    """Launch-level logs and summaries. They are not a second result root."""
    name = str(model_name).strip()
    if name not in MODEL_NAMES:
        raise PathLayoutError(f"unknown model name for reports: {model_name!r}")
    return xuancheng26_training_results_3s_root() / "reports" / name


def xuancheng26_model_results_3s_dir(model_name: str) -> Path:
    name = str(model_name).strip()
    if name not in MODEL_NAMES:
        raise PathLayoutError(f"unknown model name for training_results: {model_name!r}")
    return xuancheng26_training_results_3s_root() / name


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
    """Retired non-3s Xuancheng package. Callers must use ``xuancheng26_training_3s_root``."""
    raise PathLayoutError(
        "xuancheng26/training_data is retired; use xuancheng26_training_3s_root() "
        f"({xuancheng26_training_3s_root()})"
    )


def xuancheng26_training_3s_root() -> Path:
    """Xuancheng26 3-second-phase package with the four-state windows."""
    return data_root() / "xuancheng26" / "training_data_3s"


def xuancheng26_model_config(model_name: str) -> Path:
    name = str(model_name).strip()
    name = "gwn" if name == "graph_wavenet" else name
    return xuancheng26_training_3s_root() / "config" / name / f"r_only_{name}.json"


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
    if _is_retired_xuancheng_result(resolved):
        raise PathLayoutError(
            "xuancheng26/training_results is retired; write Xuancheng runs under "
            f"{xuancheng26_training_results_3s_root()}"
        )
    shanghai_results = (data_root() / "shanghai_sim" / "training_results").resolve()
    allowed_roots = (
        training_results_root().resolve(),
        shanghai_results,
        xuancheng26_training_results_3s_root().resolve(),
    )
    if not any(resolved == root or _is_relative_to(resolved, root) for root in allowed_roots):
        raise PathLayoutError(
            "training output must be under Shanghai training_results or "
            f"{xuancheng26_training_results_3s_root()}, got {resolved}"
        )
    if _is_relative_to(resolved, training_data_root()):
        raise PathLayoutError(f"training_results must not be inside training_data: {resolved}")
    for prepared in (
        data_root() / "xuancheng26" / "training_data",
        data_root() / "xuancheng26" / "training_data_3s",
    ):
        if prepared.exists() and _is_relative_to(resolved, prepared.resolve()):
            raise PathLayoutError(f"training_results must not be inside training_data: {resolved}")
    if model_name is not None and not _is_shanghai_result(resolved):
        if model_name not in resolved.parts and resolved.name != model_name:
            raise PathLayoutError(
                f"{model_name} output path does not name that model: {resolved}"
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


SAMPLING_SEEDS = (42, 43, 44, 45, 46)
TRAIN_SAMPLING_SEED = 42


def xuancheng26_sampling_root() -> Path:
    """Fixed-uniform sampling tree. The old top-level ``sampling_seeds`` alias is retired."""
    return xuancheng26_training_3s_root() / "datasets" / FIXED_DATASET_ID / "sampling_seeds"


def xuancheng26_shared_root() -> Path:
    return xuancheng26_training_3s_root() / "shared"


def sampling_seed_root(sampling_root: Path | str, sampling_seed: int) -> Path:
    if int(sampling_seed) not in SAMPLING_SEEDS:
        raise PathLayoutError(f"unknown sampling seed: {sampling_seed}")
    return Path(sampling_root) / f"seed_{int(sampling_seed)}"


def sampling_window_root(sampling_root: Path | str, sampling_seed: int = TRAIN_SAMPLING_SEED) -> Path:
    return sampling_seed_root(sampling_root, sampling_seed) / "flow" / "window_data"


def sampling_partial_flow_root(sampling_root: Path | str, sampling_seed: int = TRAIN_SAMPLING_SEED) -> Path:
    return sampling_seed_root(sampling_root, sampling_seed) / "flow" / "partial_flow"


def sampled_vehicle_root(sampling_root: Path | str, sampling_seed: int = TRAIN_SAMPLING_SEED) -> Path:
    return sampling_seed_root(sampling_root, sampling_seed) / "sampled_vehicle"


def sequence_targets_dir(training_root: Path | str | None = None) -> Path:
    """Authoritative 12-step full labels. Dataset shared directories may link here."""
    root = Path(training_root).expanduser().resolve() if training_root else xuancheng26_training_3s_root()
    return root / "shared" / "sequence_targets_12step"


def rf_observation_feature_path(
    training_root: Path | str,
    dataset_id: str,
    sampling_seed: int,
    observation_key: str,
    date: str,
) -> Path:
    """Speed cube for one dataset, sampling seed, observation key, and day."""
    return (
        Path(training_root) / "datasets" / str(dataset_id) / "sampling_seeds" / f"seed_{int(sampling_seed)}"
        / "model_prepared" / "RF" / str(observation_key) / f"{date}.npz"
    )


def traffic_observation_feature_path(
    training_root: Path | str,
    dataset_id: str,
    sampling_seed: int,
    observation_key: str,
    date: str,
) -> Path:
    """Uncapped daily speed mean and snapshot count for one sampled day."""
    return (
        Path(training_root) / "datasets" / str(dataset_id) / "sampling_seeds" / f"seed_{int(sampling_seed)}"
        / "model_prepared" / "traffic_observations" / str(observation_key) / f"{date}.npz"
    )


def all_sp_c_normalization_path(training_root: Path | str, dataset_id: str) -> Path:
    """Dataset-level scaler for the two extra ``all_sp_c`` input channels."""
    return Path(training_root) / "datasets" / str(dataset_id) / "all_sp_c_normalization.json"


def all_sp_c_ready_path(training_root: Path | str, dataset_id: str) -> Path:
    """Separate readiness record. It does not replace the four-state dataset status."""
    return Path(training_root) / "datasets" / str(dataset_id) / "all_sp_c_ready.json"


def sampling_root_from_window(window_root: Path | str) -> Path:
    path = Path(window_root).absolute()
    if path.name != "window_data" or path.parent.name != "flow" or path.parent.parent.name not in {f"seed_{s}" for s in SAMPLING_SEEDS}:
        raise PathLayoutError(f"expected seed_<sampling_seed>/flow/window_data, got {path}")
    return path.parents[2]


def task_results_dir(output_root: Path | str, input_type: str, target_type: str) -> Path:
    """Task directory inside the canonical Xuancheng result tree.

    ``output_root`` is a results root or a legacy ``<results>/<model>`` pointer.
    The returned directory stops at the task leaf; the run id is not included.
    """
    root = Path(output_root)
    if _is_shanghai_result(root):
        leaf = f"in_{input_type}__out_{target_type}"
        return root if root.name == leaf else root / leaf
    base, model = _xuancheng_result_base_and_model(root, None)
    if model is None:
        raise PathLayoutError(f"cannot infer model for task directory from {root}")
    return (
        base / "datasets" / FIXED_DATASET_ID / f"physical_{PHYSICAL_DEFINITION_VERSION}"
        / model / f"in_{input_type}__out_{target_type}"
    )


def _is_shanghai_result(path: Path) -> bool:
    return "shanghai_sim" in Path(path).parts


def _is_retired_xuancheng_result(path: Path) -> bool:
    parts = Path(path).parts
    return "xuancheng26" in parts and "training_results" in parts and "training_results_3s" not in parts


def _xuancheng_result_base_and_model(path: Path, model_name: str | None) -> tuple[Path, str | None]:
    """Split a pointer into the 3s results root and the model name."""
    raw = Path(path)
    if _is_retired_xuancheng_result(raw):
        raise PathLayoutError(
            "xuancheng26/training_results is retired and is not a silent alias of training_results_3s: "
            f"{raw}"
        )
    parts = raw.parts
    model = str(model_name).strip() if model_name else None
    if "training_results_3s" in parts:
        index = parts.index("training_results_3s")
        base = Path(*parts[: index + 1])
        if model is None and index + 1 < len(parts) and parts[index + 1] in MODEL_NAMES:
            model = parts[index + 1]
        if model is None and "datasets" in parts[index + 1 :]:
            tail = parts[index + 1 :]
            for part in tail:
                if part in MODEL_NAMES:
                    model = part
                    break
        return base, model
    if model is None and raw.name in MODEL_NAMES:
        return raw.parent, raw.name
    return raw, model


def sampling_run_results_dir(
    output_root: Path | str,
    input_type: str,
    target_type: str,
    rate: str,
    model_seed: int,
    *,
    physical_version: str | None = None,
    dataset_id: str | None = None,
    model_name: str | None = None,
    experiment_id: str = "baseline",
    run_id: str | None = None,
) -> Path:
    """Return one Xuancheng run directory, including experiment and run id.

    Layout::

        <results_3s>/datasets/<dataset_id>/physical_<version>/<model>/
            in_<input>__out_<target>/<observation_key>/<experiment_id>/
            seed_<model_seed>/<run_id>/

    Shanghai outputs keep the historical ``<model>/in_*__out_*/<rate>/seed_<seed>``
    layout. A path under the retired non-3s Xuancheng result tree raises.
    """
    root = Path(output_root)
    if _is_shanghai_result(root):
        leaf = f"in_{input_type}__out_{target_type}"
        parent = root if root.name == leaf else root / leaf
        return parent / str(rate) / f"seed_{int(model_seed)}"
    if not run_id:
        raise PathLayoutError("Xuancheng result directory requires run_id")
    version = str(physical_version or PHYSICAL_DEFINITION_VERSION)
    dataset = str(dataset_id or FIXED_DATASET_ID)
    base, model = _xuancheng_result_base_and_model(root, model_name)
    if model is None:
        raise PathLayoutError(f"cannot infer model name from {root}")
    if model not in MODEL_NAMES:
        raise PathLayoutError(f"unknown model name for a result run: {model}")
    return (
        base / "datasets" / dataset / f"physical_{version}" / model
        / f"in_{input_type}__out_{target_type}" / str(rate) / str(experiment_id)
        / f"seed_{int(model_seed)}" / str(run_id)
    )


PHYSICAL_DEFINITION_VERSION = "independent_fcd_v1"
FIXED_DATASET_ID = "fixed_uniform"


def parse_canonical_run(path: Path | str) -> dict[str, str]:
    """Read dataset, version, model, experiment, and run id out of a canonical run directory."""
    raw = Path(path)
    parts = raw.parts
    if "datasets" not in parts:
        return {}
    index = parts.index("datasets")
    tail = parts[index + 1 :]
    # dataset / physical_* / model / task / key / experiment / seed_* / run_id
    if len(tail) < 8:
        return {}
    version = tail[1]
    if version.startswith("physical_"):
        version = version[len("physical_") :]
    return {
        "results_root": str(Path(*parts[:index])),
        "dataset_id": tail[0],
        "physical_definition_version": version,
        "model_name": tail[2],
        "experiment_id": tail[5],
        "run_id": tail[7],
    }


def new_run_id(model_name: str, dataset_id: str = FIXED_DATASET_ID) -> str:
    """Unique id for one new training launch. Migration uses ``legacy_<hash>`` instead."""
    import hashlib
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    digest = hashlib.sha256(f"{model_name}|{dataset_id}|{stamp}|{os.getpid()}".encode()).hexdigest()[:8]
    return f"{stamp}_{digest}"
FIXED_OBSERVATION_KEYS = ("p05", "p10", "p20", "p30", "p40", "p50", "p70")
_OBSERVATION_KEY = re.compile(r"^p\d+(?:p\d+)?_p\d+(?:p\d+)?$")


def _percent_token(probability: float) -> str:
    scaled = float(probability) * 100.0
    text = f"{scaled:.8f}".rstrip("0").rstrip(".")
    if not text or text.startswith("-"):
        raise PathLayoutError(f"invalid probability for an observation key: {probability}")
    return text.replace(".", "p")


def observation_key(p_min: float, p_max: float) -> str:
    """Range id such as ``p10_p20``. Decimal bounds use a ``p`` in place of the dot."""
    low = float(p_min)
    high = float(p_max)
    if not (0.0 < low < high <= 1.0):
        raise PathLayoutError(f"expected 0 < p_min < p_max <= 1, got {p_min}, {p_max}")
    return f"p{_percent_token(low)}_p{_percent_token(high)}"


VARIABLE_MODES = ("30min", "1h", "2h", "day")


def variable_dataset_id(mode: str, p_min: float, p_max: float) -> str:
    if mode not in VARIABLE_MODES:
        raise PathLayoutError(f"unknown variable mode: {mode}")
    return f"variable_{mode}_{observation_key(p_min, p_max)}"


def _fixed_sampling_root(training_root: Path) -> Path:
    return training_root / "datasets" / FIXED_DATASET_ID / "sampling_seeds"


def _infer_dataset_id(sampling_root: Path, training_root: Path) -> str | None:
    if (
        sampling_root.name == "sampling_seeds"
        and sampling_root.parent.parent.name == "datasets"
    ):
        return sampling_root.parent.name
    for candidate in (
        training_root / "datasets" / FIXED_DATASET_ID / "sampling_seeds",
        training_root / "sampling_seeds",
    ):
        if candidate.exists() and sampling_root == candidate.resolve():
            return FIXED_DATASET_ID
    return None


def _observation_keys_ok(keys: tuple[str, ...], *, fixed: bool) -> bool:
    if fixed:
        return keys == FIXED_OBSERVATION_KEYS
    if not keys or any(key in FIXED_OBSERVATION_KEYS for key in keys):
        return False
    return all(_OBSERVATION_KEY.fullmatch(key) for key in keys)


def resolve_xuancheng_dataset(
    *,
    dataset_id: str | None = None,
    sampling_root: Path | str | None = None,
    training_root: Path | str | None = None,
) -> dict[str, object]:
    """Resolve one Xuancheng26 3s dataset without mixing two identities.

    An omitted dataset is ``fixed_uniform``. A custom sampling root that is
    not one of the official dataset directories keeps the fixed observation
    keys so existing fixtures and callers continue to work. Official variable
    datasets must name the manifest that declares their observation key.
    """
    root = Path(training_root).expanduser().resolve() if training_root else xuancheng26_training_3s_root()
    sampling_path = Path(sampling_root).expanduser().resolve() if sampling_root else None
    inferred = _infer_dataset_id(sampling_path, root) if sampling_path is not None else None
    selected = str(dataset_id).strip() if dataset_id else None
    if not selected:
        selected = inferred or FIXED_DATASET_ID
    elif inferred and inferred != selected:
        raise PathLayoutError(
            f"dataset_id {selected} contradicts sampling_root {sampling_path} ({inferred})"
        )
    if selected == FIXED_DATASET_ID:
        if not _observation_keys_ok(FIXED_OBSERVATION_KEYS, fixed=True):
            raise PathLayoutError("fixed observation keys are inconsistent")
        official = _fixed_sampling_root(root)
        if sampling_path is None:
            sampling_path = official
        elif inferred not in {None, FIXED_DATASET_ID}:
            raise PathLayoutError(f"sampling_root {sampling_path} is not fixed_uniform")
        manifest_path = root / "datasets" / FIXED_DATASET_ID / "dataset_manifest.json"
        return {
            "dataset_id": FIXED_DATASET_ID,
            "kind": "fixed_uniform",
            "training_root": root,
            "dataset_root": root / "datasets" / FIXED_DATASET_ID,
            "sampling_root": sampling_path,
            "observation_keys": FIXED_OBSERVATION_KEYS,
            "custom_sampling_root": inferred is None and sampling_path != official.resolve(),
            "manifest_path": manifest_path if manifest_path.is_file() else None,
            "schedule_sha256": None,
        }
    dataset_root = root / "datasets" / selected
    manifest_path = dataset_root / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise PathLayoutError(f"missing dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != selected:
        raise PathLayoutError(f"{manifest_path} dataset_id is {manifest.get('dataset_id')!r}")
    keys = tuple(str(item) for item in manifest.get("observation_keys") or ())
    if not _observation_keys_ok(keys, fixed=False):
        raise PathLayoutError(f"{selected} observation keys are not a declared range: {keys}")
    expected = (dataset_root / "sampling_seeds").resolve()
    if sampling_path is None:
        sampling_path = expected
    elif sampling_path != expected:
        raise PathLayoutError(f"sampling_root {sampling_path} is not {expected}")
    schedule_sha = None
    schedule_path = dataset_root / "penetration_schedule.json"
    if schedule_path.is_file():
        schedule_sha = json.loads(schedule_path.read_text(encoding="utf-8")).get("schedule_sha256")
    return {
        "dataset_id": selected,
        "kind": str(manifest.get("kind") or "variable"),
        "training_root": root,
        "dataset_root": dataset_root,
        "sampling_root": sampling_path,
        "observation_keys": keys,
        "custom_sampling_root": False,
        "manifest_path": manifest_path,
        "schedule_sha256": schedule_sha,
    }


def dataset_results_parent(output_root: Path | str, dataset_id: str) -> Path:
    """Model result root for one variable dataset, without the task leaf."""
    root = Path(output_root)
    if root.parent.name == "datasets" and root.name == dataset_id:
        return root
    if root.parent.parent.name == "datasets" and root.parent.name == dataset_id:
        return root.parent
    if root.name.startswith("in_") and "__out_" in root.name:
        root = root.parent
    if root.parent.name == "datasets":
        return root.parent / dataset_id
    return root / "datasets" / dataset_id
