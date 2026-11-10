"""Shared baseline resources and per-dataset RF speeds for one penetration range."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[3]

TASKS = ("I2I", "A2A", "A2I")
PD_TASKS = ("I2I",)
N_NODES = 1406

STATIC = {
    "stgcn": {
        "kind": "static_graph",
        "tasks": TASKS,
        "files": {"weighted_adjacency": "graph/stgcn_weighted_adjacency.npy"},
        "shapes": {"weighted_adjacency": (N_NODES, N_NODES)},
    },
    "dcrnn": {
        "kind": "static_graph",
        "tasks": TASKS,
        "files": {
            "adjacency": "graph/dcrnn_weighted_adjacency.npy",
            "adjacency_pickle": "graph/dcrnn_adj_mx.pkl",
        },
        "shapes": {"adjacency": (N_NODES, N_NODES)},
    },
    "graph_wavenet": {
        "kind": "static_graph",
        "tasks": TASKS,
        "files": {
            "adjacency": "graph/dcrnn_weighted_adjacency.npy",
            "adjacency_pickle": "graph/dcrnn_adj_mx.pkl",
        },
        "shapes": {"adjacency": (N_NODES, N_NODES)},
    },
    "stsgcn": {
        "kind": "static_graph",
        "tasks": TASKS,
        "files": {"spatial_adjacency": "graph/stsgcn_undirected_topology.npy"},
        "shapes": {"spatial_adjacency": (N_NODES, N_NODES)},
    },
    "stfgnn": {
        "kind": "train_full_statistic",
        "tasks": TASKS,
        "files": {
            "spatial_adjacency": "graph/stsgcn_undirected_topology.npy",
            "temporal_adjacency": "config/stfgnn/temporal_adjacency.npy",
        },
        "shapes": {
            "spatial_adjacency": (N_NODES, N_NODES),
            "temporal_adjacency": (N_NODES, N_NODES),
        },
        "note": "Temporal adjacency is the published training-day full-flow graph, shared across datasets and sampling seeds.",
    },
    "pdformer": {
        "kind": "static_topology_and_train_full",
        "tasks": PD_TASKS,
        "unsupported_tasks": ["A2A", "A2I"],
        "files": {
            "shortest_path": "config/pdformer/pdformer_shortest_path.npy",
            "dtw": "config/pdformer/pdformer_dtw_distance.npy",
            "laplacian_pe": "config/pdformer/pdformer_laplacian_pe.npy",
            "pattern_keys": "config/pdformer/pdformer_pattern_keys.npy",
            "undirected_topology": "graph/stsgcn_undirected_topology.npy",
        },
        "shapes": {
            "shortest_path": (N_NODES, N_NODES),
            "dtw": (N_NODES, N_NODES),
            "laplacian_pe": (N_NODES, 8),
            "pattern_keys": (16, 3, 1),
            "undirected_topology": (N_NODES, N_NODES),
        },
        "note": "DTW and pattern keys stay the published training-day full-flow priors. PDFormer accepts one state channel only.",
    },
    "stid": {"kind": "none", "tasks": TASKS, "files": {}, "shapes": {}},
    "staeformer": {"kind": "none", "tasks": TASKS, "files": {}, "shapes": {}},
    "s2s_gcn_lstm": {
        "kind": "static_graph",
        "tasks": TASKS,
        "files": {
            "spatial_adjacency": "graph/stsgcn_undirected_topology.npy",
            "distance_matrix": "graph/stgcn_weighted_adjacency.npy",
        },
        "shapes": {
            "spatial_adjacency": (N_NODES, N_NODES),
            "distance_matrix": (N_NODES, N_NODES),
        },
    },
    "tlast": {"kind": "none", "tasks": TASKS, "files": {}, "shapes": {}},
    "asttn": {
        "kind": "static_graph",
        "tasks": TASKS,
        "files": {"spatial_embedding": "graph/asttn/node2vec_r128.npy"},
        "shapes": {"spatial_embedding": (N_NODES, 128)},
        "note": "Node2vec is built from the road graph. Its training seed is not a vehicle sampling seed.",
    },
    "stjgcn": {
        "kind": "static_graph",
        "tasks": TASKS,
        "files": {"predefined_graph": "graph/stjgcn/predefined_stjg.npz"},
        "shapes": {},
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


RESOURCE_FLAGS = {
    "stgcn": {"weighted_adjacency": "--weighted-adjacency", "r_nodes": "--r-nodes"},
    "dcrnn": {"adjacency": "--adjacency", "adjacency_pickle": "--adjacency-pickle", "r_nodes": "--r-nodes"},
    "graph_wavenet": {"adjacency": "--adjacency", "adjacency_pickle": "--adjacency-pickle", "r_nodes": "--r-nodes"},
    "stsgcn": {"spatial_adjacency": "--spatial-adjacency", "r_nodes": "--r-nodes"},
    "stfgnn": {
        "spatial_adjacency": "--spatial-adjacency",
        "temporal_adjacency": "--temporal-adjacency",
        "r_nodes": "--r-nodes",
    },
    "pdformer": {"undirected_topology": "--adjacency", "relations_dir": "--relations-dir", "r_nodes": "--r-nodes"},
    "s2s_gcn_lstm": {"spatial_adjacency": "--spatial-adjacency", "r_graph": "--r-graph", "r_nodes": "--r-nodes"},
    "stid": {"r_nodes": "--r-nodes"},
    "staeformer": {"r_nodes": "--r-nodes"},
}


def _config_name(model: str) -> str:
    return "gwn" if model == "graph_wavenet" else model


def resolve_model_config(model: str, training_root: Path) -> Path:
    """Config selected for this training root, then the repository fallback."""
    disk_name = _config_name(model)
    root = Path(training_root).expanduser().resolve()
    candidates = (
        root / "config" / disk_name / f"r_only_{disk_name}.json",
        CODE_ROOT / "reimplementation" / model / "configs" / f"r_only_{model}.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"{model} has no config. Looked for {candidates[0]} and {candidates[1]}"
    )


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def rebase_training_path(training_root: Path, value: str) -> Path:
    """Map a config path onto ``training_root``.

    Absolute paths that already live there stay. A path under another
    ``training_data_3s`` tree keeps its tail. Anything else is an error.
    """
    root = Path(training_root).resolve()
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        return (root / raw).resolve()
    if _under(raw, root):
        return raw.resolve()
    parts = raw.parts
    marker = "training_data_3s"
    if marker in parts:
        index = parts.index(marker)
        return root.joinpath(*parts[index + 1 :]).resolve()
    raise RuntimeError(f"{value} does not resolve under {root}")


def resolve_resource(training_root: Path, config: dict, key: str, relative: str, *, flagged: bool) -> Path:
    root = Path(training_root).resolve()
    raw = config.get(key) if isinstance(config, dict) else None
    if raw and not flagged:
        candidate = Path(str(raw)).expanduser()
        if candidate.is_absolute() and not _under(candidate, root):
            raise RuntimeError(
                f"{key} is {raw}, outside {root}, and this training entry has no override flag"
            )
        resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    elif raw:
        resolved = rebase_training_path(root, str(raw))
    else:
        resolved = (root / relative).resolve()
    if not _under(resolved, root):
        raise RuntimeError(f"{key} resolves outside {root}: {resolved}")
    return resolved


def _check_array(path: Path, shape: tuple[int, ...]) -> dict:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            arrays = {name: tuple(int(item) for item in payload[name].shape) for name in payload.files}
        expected = {"A_fw": (12, N_NODES, N_NODES), "A_bw": (12, N_NODES, N_NODES), "distance": (N_NODES, N_NODES)}
        if arrays != expected:
            raise RuntimeError(f"{path} shapes {arrays} != {expected}")
        return {"path": str(path), "shape": arrays, "sha256": _sha256(path), "bytes": path.stat().st_size}
    array = np.load(path, mmap_mode="r")
    got = tuple(int(item) for item in array.shape)
    if got != tuple(shape):
        raise RuntimeError(f"{path} shape {got} != {shape}")
    return {"path": str(path), "shape": list(got), "dtype": str(array.dtype), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _road_attributes(training_root: Path, edges: list[str]) -> dict:
    import csv

    path = training_root / "config" / "RF" / "road_attributes.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        found = [row["edge_id"] for row in csv.DictReader(handle)]
    if found != edges:
        raise RuntimeError("config/RF/road_attributes.csv edge order differs from node_mapping.csv")
    return {"path": str(path), "rows": len(found), "sha256": _sha256(path), "kind": "shared_static"}


def _node_ids(training_root: Path) -> list[str]:
    import csv

    with (training_root / "shared" / "node_mapping.csv").open(encoding="utf-8", newline="") as handle:
        return [row["edge_id"] for row in csv.DictReader(handle)]


def _rf_audit(training_root: Path, dataset_id: str, key: str, seeds: list[int], dates: list[str]) -> tuple[int, list[str]]:
    """Count RF files whose stored sampling, inflow, shape, and loader fields match."""
    from prepare_published_rf_features import groups_for, rf_npz_current, specs_from_ids

    try:
        specs = specs_from_ids(training_root, [dataset_id])
        edges = _node_ids(training_root)
        node_sha = _sha256(training_root / "shared" / "node_mapping.csv")
    except Exception as exc:
        return 0, [f"{dataset_id} RF provenance: {exc}"]
    found = 0
    problems = []
    expected = len(seeds) * len(dates)
    for date in dates:
        try:
            groups = [group for group in groups_for(training_root, date, specs) if group["key"] == key]
        except Exception as exc:
            problems.append(f"{dataset_id} RF {date}: {exc}")
            continue
        seen = {(int(group["sampling_seed"]), group["key"]) for group in groups}
        for seed in seeds:
            if (int(seed), key) not in seen:
                problems.append(f"{dataset_id} RF seed {seed} {date} has no sampling group")
        for group in groups:
            path = Path(group["out"])
            if rf_npz_current(path, group, edges=edges, node_sha=node_sha):
                found += 1
                continue
            state = "unreadable or inconsistent"
            if not path.is_file():
                state = "missing"
            elif path.stat().st_size <= 0:
                state = "empty"
            problems.append(
                f"{dataset_id} RF seed {group['sampling_seed']} {date} is {state} "
                "for the current vehicle list and inflow"
            )
    if found != expected and not any("RF provenance" in item for item in problems):
        problems.append(f"{dataset_id} RF npz {found} != {expected}")
    if len(problems) > 8:
        extra = len(problems) - 8
        problems = problems[:8] + [f"{dataset_id} RF has {extra} more provenance failures"]
    return found, problems


def _load_model_config(model: str, training_root: Path) -> tuple[Path, dict]:
    path = resolve_model_config(model, training_root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return path, payload


def _resolved_resources(model: str, spec: dict, training_root: Path, config: dict) -> dict[str, Path]:
    flags = RESOURCE_FLAGS.get(model, {})
    resolved = {}
    for name, relative in spec["files"].items():
        resolved[name] = resolve_resource(
            training_root, config, name, relative, flagged=name in flags,
        )
    for name in ("distance_matrix", "spatial_embedding", "predefined_graph", "r_graph", "relations_dir"):
        if name in resolved or name not in config:
            continue
        relative = spec["files"].get(name, "")
        resolved[name] = resolve_resource(
            training_root, config, name, relative, flagged=name in flags,
        )
    return resolved


def _check_prior(training_root: Path, model: str, train_days: list[str], edges: list[str], resolved: dict[str, Path]) -> None:
    if model == "stfgnn":
        report_path = training_root / "config" / "stfgnn" / "temporal_activity_generation_report.json"
        if not report_path.is_file():
            raise RuntimeError(f"missing training-only temporal report {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if int(report.get("validation_days_used") or 0) != 0 or int(report.get("test_days_used") or 0) != 0:
            raise RuntimeError("temporal adjacency used validation or test days")
        if [str(day) for day in report.get("training_days") or []] != [str(day) for day in train_days]:
            raise RuntimeError("temporal adjacency was not fit on the shared training split")
    if model == "pdformer":
        relations = resolved.get("relations_dir") or (training_root / "config" / "pdformer")
        meta_path = Path(relations) / "pdformer_relations_metadata.json"
        if not meta_path.is_file():
            raise RuntimeError(f"missing training-only PDFormer metadata {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("split") != "train":
            raise RuntimeError(f"PDFormer relations split is {meta.get('split')!r}")
        if [str(day) for day in meta.get("days") or []] != [str(day) for day in train_days]:
            raise RuntimeError("PDFormer relations were not fit on the shared training split")
        node_ids = [str(item) for item in meta.get("node_ids") or []]
        if node_ids != edges:
            raise RuntimeError("PDFormer node order differs from shared/node_mapping.csv")


def _commands(
    dataset_id: str,
    key: str,
    results_root: Path,
    training_root: Path,
) -> tuple[list[dict], list[str]]:
    commands = []
    errors = []
    sampling_root = training_root / "datasets" / dataset_id / "sampling_seeds"
    experiment = f"variable_{key}_train42_eval5"
    for task, input_type, target_type in (("I2I", "inflow", "inflow"), ("A2A", "all", "all"), ("A2I", "all", "inflow")):
        commands.append({
            "model": "RF",
            "task": task,
            "argv": [
                "python3", "reimplementation/RF/run_rf.py",
                "--dataset-id", dataset_id,
                "--training-root", str(training_root),
                "--penetrations", key,
                "--input-type", input_type,
                "--target-type", target_type,
                "--results-root", str(results_root / "RF"),
                "--experiment-id", experiment,
            ],
        })
    for model, spec in STATIC.items():
        script = f"reimplementation/{model}/train_r_only_{'gwn' if model == 'graph_wavenet' else model}.py"
        try:
            config_path, config = _load_model_config(model, training_root)
            resolved = _resolved_resources(model, spec, training_root, config)
        except Exception as exc:
            errors.append(f"{dataset_id} {model} config: {exc}")
            continue
        flags = RESOURCE_FLAGS.get(model, {})
        for task in spec["tasks"]:
            if task == "I2I":
                input_type, target_type = "inflow", "inflow"
            elif task == "A2A":
                input_type, target_type = "all", "all"
            else:
                input_type, target_type = "all", "inflow"
            argv = [
                "python3", script,
                "--config", str(config_path),
                "--dataset-id", dataset_id,
                "--sampling-root", str(sampling_root),
                "--rates", key,
                "--input-type", input_type,
                "--target-type", target_type,
                "--output-root", str(results_root / model),
                "--experiment-id", experiment,
            ]
            for name, flag in flags.items():
                if name in resolved:
                    argv.extend([flag, str(resolved[name])])
            commands.append({
                "model": model,
                "task": task,
                "argv": argv,
                "config": str(config_path),
                "sampling_root": str(sampling_root),
                "resources": {name: str(path) for name, path in resolved.items()},
            })
    return commands, errors


def _check_resolved(path: Path, shape: tuple[int, ...]) -> dict:
    if path.suffix == ".npz" or shape:
        return _check_array(path, shape)
    if path.suffix == ".pkl":
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"missing {path}")
        return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
    if path.is_dir():
        return {"path": str(path), "kind": "directory"}
    raise RuntimeError(f"no shape check for {path}")


def write_model_manifest(
    training_root: Path,
    dataset_id: str,
    *,
    forced_failures: list[str] | None = None,
    results_root: Path | None = None,
) -> dict:
    from data_root import xuancheng26_training_results_3s_root

    training_root = Path(training_root).expanduser().resolve()
    root = training_root / "datasets" / dataset_id
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    split = json.loads((training_root / "shared" / "split_manifest.json").read_text(encoding="utf-8"))
    dates = [day for name in ("train", "validation", "test") for day in split[name]]
    train_days = [str(day) for day in split["train"]]
    edges = _node_ids(training_root)
    r_nodes = training_root / "graph" / "r_nodes.csv"
    failures = list(forced_failures or [])
    if r_nodes.is_file():
        import csv

        with r_nodes.open(encoding="utf-8", newline="") as handle:
            graph_edges = [row["edge_id"] for row in csv.DictReader(handle)]
        if graph_edges != edges:
            failures.append(f"{dataset_id} graph/r_nodes.csv order differs from shared/node_mapping.csv")
    key = manifest["observation_keys"][0]
    seeds = [int(seed) for seed in manifest["sampling_seeds"]]
    models = []
    try:
        roads = _road_attributes(training_root, edges)
    except Exception as exc:
        roads = None
        failures.append(f"{dataset_id} RF road attributes: {exc}")
    rf_found, rf_problems = _rf_audit(training_root, dataset_id, key, seeds, dates)
    expected_rf = len(seeds) * len(dates)
    if forced_failures or rf_problems or roads is None:
        rf_status = "failed"
        failures.extend(rf_problems)
    else:
        rf_status = "ready"
    models.append({
        "model": "RF",
        "kind": "observed_sampling",
        "tasks": list(TASKS),
        "status": rf_status,
        "generated": {
            "path": f"sampling_seeds/seed_<seed>/model_prepared/RF/{key}/<date>.npz",
            "files": rf_found,
            "expected": expected_rf,
            "provenance": "dataset, observation key, sampling seed, vehicle list, and inflow sha256",
        },
        "shared": roads,
    })
    for model, spec in STATIC.items():
        resources = {}
        status = "ready"
        try:
            _config_path, config = _load_model_config(model, training_root)
            resolved = _resolved_resources(model, spec, training_root, config)
            _check_prior(training_root, model, train_days, edges, resolved)
        except Exception as exc:
            status = "failed"
            resolved = {}
            failures.append(f"{dataset_id} {model} config: {exc}")
        for name, relative in spec["files"].items():
            path = resolved.get(name) or (training_root / relative)
            try:
                if status == "failed" and name not in resolved:
                    raise RuntimeError("config did not resolve this resource")
                checked = _check_resolved(path, spec["shapes"].get(name, ()))
                checked["node_order"] = "shared/node_mapping.csv"
                checked["nodes"] = len(edges)
                resources[name] = checked
            except Exception as exc:
                status = "failed"
                failures.append(f"{dataset_id} {model} {name}: {exc}")
        if not spec["files"]:
            resources = {"required": "window NPZ, node ids, and the existing time-feature interface"}
        models.append({
            "model": model,
            "kind": spec["kind"],
            "tasks": list(spec["tasks"]),
            "unsupported_tasks": list(spec.get("unsupported_tasks") or []),
            "status": status,
            "reuse": "shared" if spec["files"] else "none",
            "resources": resources,
            "note": spec.get("note"),
        })
    result_root = Path(results_root).expanduser().resolve() if results_root else xuancheng26_training_results_3s_root()
    command_list, command_errors = _commands(dataset_id, key, result_root, training_root)
    failures.extend(command_errors)
    ready = not failures and all(item["status"] == "ready" for item in models)
    payload = {
        "dataset_id": dataset_id,
        "observation_key": key,
        "sampling_seeds": seeds,
        "training_root": str(training_root),
        "results_root": str(result_root),
        "models_ready": ready,
        "models": models,
        "train_commands": command_list if ready else [],
        "failures": failures,
    }
    _write_json(root / "model_preparation_manifest.json", payload)
    return payload


def prepare_models(*, training_root: Path, fcd_root: Path, net_file: Path, dataset_ids: list[str], workers: int) -> dict:
    from prepare_published_rf_features import run_prepare

    training_root = Path(training_root)
    forced = {dataset_id: [] for dataset_id in dataset_ids}
    try:
        report = run_prepare([
            "--training-root", str(training_root),
            "--fcd-root", str(fcd_root),
            "--net-file", str(net_file),
            "--workers", str(workers),
            "--dataset-ids", *[str(item) for item in dataset_ids],
        ])
    except Exception as exc:
        report = {"datasets": {}, "error": f"{type(exc).__name__}: {exc}"}
    datasets = report.get("datasets") or {}
    global_error = report.get("error")
    if global_error and not datasets:
        for dataset_id in dataset_ids:
            forced[dataset_id].append(f"{dataset_id} RF preparation failed: {global_error}")
    for dataset_id in dataset_ids:
        if global_error and not datasets:
            continue
        item = datasets.get(dataset_id)
        if item is None:
            detail = f" ({global_error})" if global_error else ""
            forced[dataset_id].append(f"{dataset_id} RF preparation failed: no per-dataset result{detail}")
        elif not item.get("ok"):
            forced[dataset_id].append(
                f"{dataset_id} RF preparation failed: {item.get('error') or 'generation or validation failed'}"
            )
    failures = []
    reports = []
    for dataset_id in dataset_ids:
        try:
            written = write_model_manifest(
                training_root,
                dataset_id,
                forced_failures=forced[dataset_id],
            )
        except Exception as exc:
            written = {
                "dataset_id": dataset_id,
                "models_ready": False,
                "failures": [f"{dataset_id} model manifest failed: {exc}"],
                "train_commands": [],
            }
            failure_path = training_root / "datasets" / dataset_id / "model_preparation_manifest.json"
            try:
                _write_json(failure_path, {
                    "dataset_id": dataset_id,
                    "models_ready": False,
                    "train_commands": [],
                    "failures": written["failures"],
                })
            except Exception:
                pass
        reports.append({
            "dataset_id": dataset_id,
            "models_ready": written["models_ready"],
            "failures": written["failures"],
        })
        failures.extend(written["failures"])
    return {"failures": failures, "datasets": reports}
