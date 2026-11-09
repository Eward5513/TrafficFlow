"""Authoritative Xuancheng26 sampling paths and sequential CPU data loading."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Mapping

from data_root import (
    SAMPLING_SEEDS, TRAIN_SAMPLING_SEED, dataset_results_parent, resolve_xuancheng_dataset,
    sampling_window_root, sampling_root_from_window, sampling_run_results_dir,
)
from reimplementation.common.data.r_only_npz_dataset import (
    bind_observation_keys, build_dataloader, load_json, rate_tag,
)
from reimplementation.common.utils.hashing import sha256_file


def configure_sampling(config: Mapping, *, model_name: str | None = None) -> dict:
    cfg = dict(config)
    explicit = set(cfg.get("explicit_path_keys") or [])
    dataset_id = cfg.get("dataset_id")
    sampling_value = cfg.get("sampling_root")
    if "dataset_id" in explicit and "sampling_root" not in explicit and dataset_id not in (None, "fixed_uniform"):
        sampling_value = None
    if "sampling_root" in explicit and "dataset_id" not in explicit and dataset_id == "fixed_uniform":
        dataset_id = None
    spec = resolve_xuancheng_dataset(dataset_id=dataset_id, sampling_root=sampling_value)
    root = Path(spec["sampling_root"])
    train_seed = int(cfg.get("train_sampling_seed", TRAIN_SAMPLING_SEED))
    seeds = tuple(int(s) for s in cfg.get("evaluation_sampling_seeds", SAMPLING_SEEDS))
    if train_seed != TRAIN_SAMPLING_SEED or seeds != SAMPLING_SEEDS:
        raise ValueError("official protocol requires training seed 42 and evaluation seeds 42–46")
    model_seed = int(cfg.get("model_seed", cfg.get("seed", 42)))
    bind_observation_keys(spec["observation_keys"])
    window = sampling_window_root(root, train_seed)
    cfg.update(model_seed=model_seed, seed=model_seed, train_sampling_seed=train_seed,
               evaluation_sampling_seeds=list(seeds), sampling_root=str(root),
               dataset_id=spec["dataset_id"], observation_keys=list(spec["observation_keys"]),
               data_root=str(window), checkpoint_selection="mean_validation_mae")
    if spec.get("schedule_sha256"):
        cfg["schedule_sha256"] = spec["schedule_sha256"]
    normalization = window / "state_normalization.json"
    if normalization.is_file():
        cfg["normalization_sha256"] = sha256_file(normalization)
        normalization_payload = load_json(normalization)
        if normalization_payload.get("physical_definition_version"):
            cfg["physical_definition_version"] = normalization_payload["physical_definition_version"]
            if normalization_payload.get("definition_sha256"):
                cfg["definition_sha256"] = normalization_payload["definition_sha256"]
    for key, name in (("split_manifest", "split_manifest.json"), ("normalization", "state_normalization.json"),
                      ("data_validation", "validation_summary.json"), ("node_mapping", "node_mapping.csv")):
        cfg[key] = str(window / name)
    if spec["dataset_id"] != "fixed_uniform" and cfg.get("output_root"):
        cfg["output_root"] = str(dataset_results_parent(cfg["output_root"], str(spec["dataset_id"])))
    cfg.pop("explicit_path_keys", None)
    return cfg


def tag_configured_rates(rates, *, dataset_id=None, sampling_root=None) -> list[str]:
    """Validate CLI rates against one dataset. Fixed commands still accept 5 or p05."""
    spec = resolve_xuancheng_dataset(
        dataset_id=dataset_id,
        sampling_root=str(sampling_root) if sampling_root else None,
    )
    bind_observation_keys(spec["observation_keys"])
    return [rate_tag(item, allowed=spec["observation_keys"]) for item in rates]


def is_sampling_loader(loader) -> bool:
    root = getattr(getattr(loader, "dataset", None), "data_root", None)
    if root is None:
        return False
    try:
        sampling_root_from_window(root)
    except ValueError:
        return False
    except RuntimeError:
        return False
    return getattr(loader.dataset, "split", None) in ("validation", "test")


def iter_sampling_loaders(loader):
    """Retain the seed42 loader; create and release each additional CPU dataset."""
    from reimplementation.common.data.state_windows import AlignedStateWindowDataset
    base = loader.dataset
    root = sampling_root_from_window(base.data_root)
    import numpy as np
    for seed in SAMPLING_SEEDS:
        path = sampling_window_root(root, seed)
        if path.absolute() == Path(base.data_root).absolute():
            yield seed, loader
        else:
            dataset = AlignedStateWindowDataset(
                path, base.rate, base.split, input_type=base.input_type, target_type=base.target_type,
                n_his=base.x.shape[1], num_nodes=base.x.shape[2], output_steps=base.y.shape[1],
                expected_count=len(base),
            )
            for key in ("y", "y_raw", "day_index", "window_start_slot", "target_slot"):
                if not np.array_equal(getattr(dataset, key), getattr(base, key)):
                    raise ValueError(f"sampling seed {seed} has different labels or window indices: {key}")
            for name in ("split_manifest.json", "state_normalization.json", "node_mapping.csv"):
                if (Path(base.data_root) / name).exists() and sha256_file(path / name) != sha256_file(Path(base.data_root) / name):
                    raise ValueError(f"sampling metadata mismatch: {seed}/{name}")
            other = build_dataloader(dataset, batch_size=loader.batch_size, shuffle=False,
                                     num_workers=loader.num_workers, seed=42)
            yield seed, other
            del other, dataset


def validate_sampling_package(root: Path, *, n_his: int, num_nodes: int, rates: list[str], splits=("train", "validation", "test")) -> dict:
    """Validate shared metadata and hash all dated inputs; no flat-NPZ fallback."""
    import numpy as np
    from reimplementation.common.data.state_windows import CHANNELS, read_state_target_scaler
    from reimplementation.common.data.r_only_npz_dataset import expected_split_counts
    sampling_root_from_window(root)
    manifest = load_json(root / "split_manifest.json")
    norm = load_json(root / "state_normalization.json")
    state = load_json(root / "state_window_manifest.json")
    report = load_json(root / "validation_summary.json")
    if report.get("failures") or report.get("overall_validation_passed") is False:
        raise ValueError("sampling validation failed")
    if norm.get("fit_split") != "train" or norm["training_days"] != manifest["train"]:
        raise ValueError("scaler must be fitted on training days only")
    if tuple(norm["channels"]) != CHANNELS or tuple(state["channels"]) != CHANNELS:
        raise ValueError("channel order differs")
    if state["num_nodes"] != num_nodes or state["n_his"] != n_his:
        raise ValueError("node/window shape differs")
    with (root / "node_mapping.csv").open() as f:
        nodes = [r["edge_id"] for r in csv.DictReader(f)]
    if len(nodes) != num_nodes or len(set(nodes)) != num_nodes:
        raise ValueError("invalid node mapping")
    counts = expected_split_counts(manifest)
    hashes = {}
    for rate in rates:
        for split in splits:
            digest = hashlib.sha256()
            for day in manifest[split]:
                for kind in (*CHANNELS, "all"):
                    path = root / rate / split / day / f"{kind}.npz"
                    digest.update(f"{day}/{kind}:".encode())
                    digest.update(sha256_file(path).encode())
            # Stable logical key retained in checkpoint metadata, not an input filename.
            hashes[f"{rate}/{split}.npz"] = digest.hexdigest()
    mean, std = read_state_target_scaler(root, "inflow")
    return dict(sample_counts=counts, mean_y_full=mean, std_y_full=std, node_ids=nodes,
                file_hashes=hashes, normalization_sha256=sha256_file(root / "state_normalization.json"),
                split_manifest_sha256=sha256_file(root / "split_manifest.json"),
                target_mode="last-observed-step", did_not_refit_scaler=True,
                prepared_data_report_status="passed", data_validation_status="passed",
                split_manifest_target_mode="last-observed-step", split_sample_keys_disjoint=True,
                denorm_max_abs_error={})


def sampling_rate_dir(output_root, config: Mapping, rate) -> Path:
    """Result directory for one penetration. Never the pre-protocol ``<model>/pXX/seed_YY`` tree."""
    return sampling_run_results_dir(
        output_root,
        str(config.get("input_type") or "inflow"),
        str(config.get("target_type") or "inflow"),
        rate,
        int(config.get("model_seed", config.get("seed", 42))),
        physical_version=config.get("physical_definition_version"),
        dataset_id=str(config.get("dataset_id") or "") or None,
    )


def protect_historical_results(root: Path) -> None:
    """An old task directory is research evidence even when --overwrite is given."""
    root = Path(root)
    for path in [root / 'resolved_config.json', *root.glob('p*/seed_*/resolved_config.json')]:
        if path.is_file():
            cfg = load_json(path)
            if cfg.get('checkpoint_selection') != 'mean_validation_mae':
                raise RuntimeError(f'historical result directory is protected: {root}; use an empty --output-root for the new protocol')
