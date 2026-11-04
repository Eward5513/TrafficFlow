"""Scan completed Xuancheng26 runs and write the experiment inventory.

Does not train, does not use a GPU, and does not modify result directories.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from analysis_lib import (
    FAMILY_FOUR,
    FAMILY_LEGACY,
    FAMILY_PHASE_SINGLE,
    LEGACY_DATA,
    LEGACY_ROOT,
    MODELS,
    PHASE3S_DATA,
    PHASE3S_ROOT,
    RATE_RE,
    RATES,
    SPLIT_TEXT,
    STATES,
    STATS,
    TASK_RE,
    as_float,
    as_int,
    ensure_dirs,
    read_json,
    task_kind,
    traffic_channels,
    write_csv,
    write_json,
)

INVENTORY_FIELDS = [
    "Model",
    "Family",
    "Task Type",
    "Input Type",
    "Output Type",
    "Input Channels",
    "Output Channels",
    "Model Input Dim",
    "Penetration Rate",
    "Random Seed",
    "Train / Val / Test Split",
    "Best Epoch",
    "Epochs Executed",
    "Max Epochs",
    "Early Stopping",
    "Early Stopped",
    "Training Status",
    "Selection Metric",
    "Checkpoint Monitor",
    "Training Loss",
    "Target Mode",
    "Input Window",
    "Did Not Use Test For Selection",
    "Parameter Count",
    "Normalization File",
    "Data Root",
    "Result Directory",
    "Metrics File",
    "Prediction File",
    "MAE",
    "RMSE",
    "MAPE_nonzero",
    "WAPE",
    "Negative Prediction Fraction",
    "Zero Target Fraction",
    "Include In Comparison",
    "Exclusion Reason",
    "MAE-best Epoch",
    "Selection Epoch Matches MAE-best",
    "Code Version",
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = read_json(path)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def history_mae_best(seed_dir: Path) -> int | None:
    history_path = seed_dir / "training_history.json"
    if not history_path.is_file():
        return None
    try:
        history = read_json(history_path)
    except json.JSONDecodeError:
        return None
    if not isinstance(history, list) or not history:
        return None
    best_epoch = None
    best_value = None
    for row in history:
        if not isinstance(row, dict) or "val_mae" not in row or "epoch" not in row:
            return None
        value = as_float(row["val_mae"])
        epoch = as_int(row["epoch"])
        if value is None or epoch is None:
            return None
        if best_value is None or value < best_value or (value == best_value and epoch < best_epoch):
            best_value = value
            best_epoch = epoch
    return best_epoch


def classify(root_name: str, relative: Path) -> dict[str, str] | None:
    parts = relative.parts
    if any(part.startswith("_") or part in {"gaussian_stgcn", "searched_trial018"} for part in parts):
        return {
            "family": "nonstandard",
            "task_type": "nonstandard",
            "input_type": "",
            "output_type": "",
        }
    if len(parts) == 2 and RATE_RE.match(parts[0]) and parts[1].startswith("seed_"):
        family = FAMILY_LEGACY if root_name == "legacy" else FAMILY_PHASE_SINGLE
        return {
            "family": family,
            "task_type": "single_channel_inflow",
            "input_type": "inflow",
            "output_type": "inflow",
        }
    if len(parts) == 3 and parts[2].startswith("seed_") and RATE_RE.match(parts[1]):
        match = TASK_RE.match(parts[0])
        if match is None:
            return {
                "family": "nonstandard",
                "task_type": "nonstandard",
                "input_type": "",
                "output_type": "",
            }
        src = match.group("src")
        dst = match.group("dst")
        return {
            "family": FAMILY_FOUR,
            "task_type": task_kind(src, dst),
            "input_type": src,
            "output_type": dst,
        }
    return {
        "family": "nonstandard",
        "task_type": "nonstandard",
        "input_type": "",
        "output_type": "",
    }


def metric_value(metrics: dict[str, Any] | None, key: str) -> str:
    if metrics is None or key not in metrics:
        return ""
    value = as_float(metrics[key])
    return "" if value is None else f"{value:.10g}"


def scan_seed(model: str, root_name: str, seed_dir: Path, model_dir: Path) -> dict[str, Any]:
    relative = seed_dir.relative_to(model_dir)
    kind = classify(root_name, relative) or {
        "family": "nonstandard",
        "task_type": "nonstandard",
        "input_type": "",
        "output_type": "",
    }
    meta = load_optional(seed_dir / "run_metadata.json")
    config = load_optional(seed_dir / "resolved_config.json")
    metrics = load_optional(seed_dir / "test_metrics.json")
    prediction = seed_dir / "test_predictions.npz"
    metrics_path = seed_dir / "test_metrics.json"
    seed_name = seed_dir.name
    seed = seed_name.split("_", 1)[1] if seed_name.startswith("seed_") else ""
    rate = ""
    for part in relative.parts:
        if RATE_RE.match(part):
            rate = part
    reasons: list[str] = []
    status = "" if meta is None else str(meta.get("status") or "")
    if kind["family"] == "nonstandard":
        reasons.append("path is outside the single-channel and four-state protocol")
    if meta is None:
        reasons.append("run_metadata.json missing or unreadable")
    elif status != "completed":
        reasons.append(f"training status is {status or 'blank'}")
    if metrics is None:
        reasons.append("test_metrics.json missing or unreadable")
    elif any(as_float(metrics.get(key)) is None for key in ("mae", "rmse", "mape_nonzero", "wape")):
        reasons.append("one of MAE/RMSE/MAPE_nonzero/WAPE is missing")
    if not prediction.is_file() or prediction.stat().st_size <= 0:
        reasons.append("test_predictions.npz missing or empty")
    if seed != "42":
        reasons.append(f"seed directory is {seed_name}, not seed_42")
    meta_seed = "" if meta is None or meta.get("seed") is None else str(meta.get("seed"))
    if meta is not None and meta_seed not in {"", "42"}:
        reasons.append(f"metadata seed is {meta_seed}")
    target_mode = ""
    input_window = ""
    if config is not None:
        target_mode = str(config.get("target_mode") or "")
        for key in ("input_window", "input_len", "n_his"):
            if config.get(key) is not None:
                input_window = str(config.get(key))
                break
    if meta is not None and not target_mode:
        target_mode = str(meta.get("target_mode") or "")
    if target_mode and target_mode != "last-observed-step":
        reasons.append(f"target_mode is {target_mode}")
    if input_window and input_window != "12":
        reasons.append(f"input window is {input_window}")
    did_not = "" if meta is None else meta.get("did_not_use_test_for_selection")
    if did_not is False:
        reasons.append("metadata says test was used for selection")
    best_epoch = "" if meta is None or meta.get("best_epoch") is None else str(meta.get("best_epoch"))
    mae_best = history_mae_best(seed_dir)
    selected = as_int(best_epoch)
    matches = ""
    if mae_best is not None and selected is not None:
        matches = "yes" if mae_best == selected else "no"
    input_channels = traffic_channels(kind["input_type"])
    output_channels = traffic_channels(kind["output_type"])
    if config is not None:
        if config.get("input_channels") is not None:
            input_channels = int(config["input_channels"])
        if config.get("output_channels") is not None:
            output_channels = int(config["output_channels"])
    model_input_dim = ""
    if config is not None and config.get("input_dim") is not None:
        model_input_dim = str(config.get("input_dim"))
    data_root = ""
    normalization = ""
    if config is not None:
        data_root = str(config.get("data_root") or "")
        normalization = str(config.get("normalization") or config.get("state_normalization") or "")
    expected_root = LEGACY_DATA.as_posix() if root_name == "legacy" else PHASE3S_DATA.as_posix()
    if data_root and data_root != expected_root:
        reasons.append(f"data_root {data_root} is not {expected_root}")
    include = "yes" if not reasons and kind["family"] != "nonstandard" else "no"
    return {
        "Model": model,
        "Family": kind["family"],
        "Task Type": kind["task_type"],
        "Input Type": kind["input_type"],
        "Output Type": kind["output_type"],
        "Input Channels": "" if input_channels is None else input_channels,
        "Output Channels": "" if output_channels is None else output_channels,
        "Model Input Dim": model_input_dim,
        "Penetration Rate": rate,
        "Random Seed": seed,
        "Train / Val / Test Split": SPLIT_TEXT,
        "Best Epoch": best_epoch,
        "Epochs Executed": "" if meta is None or meta.get("epochs_executed") is None else meta.get("epochs_executed"),
        "Max Epochs": "" if meta is None or meta.get("max_epochs") is None else meta.get("max_epochs"),
        "Early Stopping": "" if meta is None else meta.get("early_stopping"),
        "Early Stopped": "" if meta is None else meta.get("early_stopped"),
        "Training Status": status or "missing",
        "Selection Metric": "" if meta is None else meta.get("selection_metric") or "",
        "Checkpoint Monitor": "" if meta is None else meta.get("checkpoint_monitor") or "",
        "Training Loss": "" if meta is None else meta.get("training_loss") or "",
        "Target Mode": target_mode,
        "Input Window": input_window,
        "Did Not Use Test For Selection": "" if did_not is None else did_not,
        "Parameter Count": "" if meta is None or meta.get("parameter_count") is None else meta.get("parameter_count"),
        "Normalization File": normalization,
        "Data Root": data_root,
        "Result Directory": seed_dir.as_posix(),
        "Metrics File": metrics_path.as_posix() if metrics_path.is_file() else "",
        "Prediction File": prediction.as_posix() if prediction.is_file() else "",
        "MAE": metric_value(metrics, "mae"),
        "RMSE": metric_value(metrics, "rmse"),
        "MAPE_nonzero": metric_value(metrics, "mape_nonzero"),
        "WAPE": metric_value(metrics, "wape"),
        "Negative Prediction Fraction": metric_value(metrics, "negative_prediction_fraction"),
        "Zero Target Fraction": metric_value(metrics, "zero_target_fraction"),
        "Include In Comparison": include,
        "Exclusion Reason": "; ".join(reasons),
        "MAE-best Epoch": "" if mae_best is None else mae_best,
        "Selection Epoch Matches MAE-best": matches,
        "Code Version": "" if meta is None else meta.get("code_version") or "",
        "_metrics": metrics or {},
        "_meta": meta or {},
    }


def iter_seed_dirs(model_dir: Path) -> list[Path]:
    if not model_dir.is_dir():
        return []
    found = [path for path in model_dir.rglob("seed_*") if path.is_dir()]
    return sorted(found)


def coverage_issues(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    comparable = [row for row in rows if row["Include In Comparison"] == "yes"]
    expected = {
        FAMILY_LEGACY: [(model, "single_channel_inflow", "inflow", "inflow", rate) for model in MODELS for rate in RATES],
        FAMILY_PHASE_SINGLE: [(model, "single_channel_inflow", "inflow", "inflow", rate) for model in MODELS for rate in RATES],
        FAMILY_FOUR: [],
    }
    for model in MODELS:
        for state in STATES:
            for rate in RATES:
                expected[FAMILY_FOUR].append((model, "single_to_single", state, state, rate))
                expected[FAMILY_FOUR].append((model, "all_to_single", "all", state, rate))
            for rate in RATES:
                expected[FAMILY_FOUR].append((model, "all_to_all", "all", "all", rate))
    present = {
        (
            row["Family"],
            row["Model"],
            row["Task Type"],
            row["Input Type"],
            row["Output Type"],
            row["Penetration Rate"],
        )
        for row in comparable
    }
    missing_by_family: dict[str, list[str]] = {FAMILY_LEGACY: [], FAMILY_PHASE_SINGLE: [], FAMILY_FOUR: []}
    for family, keys in expected.items():
        for model, task, src, dst, rate in keys:
            if (family, model, task, src, dst, rate) not in present:
                missing_by_family[family].append(f"{model} {task} {src}->{dst} {rate}")
    for family, missing in missing_by_family.items():
        if not missing:
            continue
        severity = "high" if family != FAMILY_PHASE_SINGLE else "medium"
        if family == FAMILY_LEGACY and all(item.startswith("tlast ") or "s2s_gcn_lstm" in item for item in missing):
            severity = "high"
        issues.append(
            {
                "severity": severity,
                "scope": family,
                "issue": f"{len(missing)} expected completed runs are absent from the comparison set",
                "detail": "; ".join(missing[:40]) + ("" if len(missing) <= 40 else f"; ... {len(missing) - 40} more"),
            }
        )
    return issues


def protocol_issues(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    comparable = [row for row in rows if row["Include In Comparison"] == "yes"]
    issues.append(
        {
            "severity": "low",
            "scope": "calendar",
            "issue": "Stored dates are 2023, not 2026-04-01 through 2026-04-29",
            "detail": (
                "Both split manifests contain 29 dates from 2023-04-01 through 2023-04-30 "
                "with 2023-04-09 removed. Test days are 2023-04-26 through 2023-04-30. "
                "The analysis uses these stored dates."
            ),
        }
    )
    issues.append(
        {
            "severity": "low",
            "scope": "location",
            "issue": "Four-state results are not under the path named in the request",
            "detail": (
                f"The request names {LEGACY_ROOT}. That directory holds the earlier "
                "single-channel inflow runs. Four-state runs and the 3-second-phase "
                f"single-channel reruns are under {PHASE3S_ROOT}. "
                "The 3-second label is the signal-timing grid (xuancheng_tls_3s), "
                "not a 3-second aggregation. Both packages use 288 five-minute bins."
            ),
        }
    )
    selectors = sorted({(row["Family"], row["Selection Metric"]) for row in comparable})
    by_family: dict[str, set[str]] = {}
    for family, metric in selectors:
        by_family.setdefault(family, set()).add(metric)
    for family, metrics in sorted(by_family.items()):
        if len(metrics) > 1:
            issues.append(
                {
                    "severity": "medium",
                    "scope": family,
                    "issue": "Checkpoint selection metric is not the same for every model",
                    "detail": ", ".join(sorted(metrics))
                    + ". Test metrics are still raw-scale MAE/RMSE/MAPE_nonzero/WAPE. "
                    "A run stays in the comparison even when the selected epoch is not the validation-MAE epoch; "
                    "those runs are listed separately.",
                }
            )
    mismatched = [
        row
        for row in rows
        if row["Selection Epoch Matches MAE-best"] == "no"
    ]
    if mismatched:
        issues.append(
            {
                "severity": "medium",
                "scope": "checkpoint",
                "issue": f"{len(mismatched)} runs selected a checkpoint that is not the validation-MAE minimum",
                "detail": "; ".join(
                    f"{row['Model']} {row['Family']} {row['Task Type']} {row['Input Type']}->{row['Output Type']} {row['Penetration Rate']} selected {row['Best Epoch']} vs MAE-best {row['MAE-best Epoch']}"
                    for row in mismatched[:25]
                ),
            }
        )
    losses = sorted({(row["Family"], row["Training Loss"] or "unrecorded") for row in comparable})
    loss_by_family: dict[str, set[str]] = {}
    for family, loss in losses:
        loss_by_family.setdefault(family, set()).add(loss)
    for family, values in sorted(loss_by_family.items()):
        if len(values) > 1 or any("huber" in item.lower() for item in values):
            issues.append(
                {
                    "severity": "medium",
                    "scope": family,
                    "issue": "Training loss is not uniform",
                    "detail": ", ".join(sorted(values))
                    + ". TLAST trains Huber delta=2 on the raw scale. The other five models train unmasked raw MAE. "
                    "Published test metrics use the shared raw-count definition either way.",
                }
            )
    epochs = sorted({(row["Family"], row["Model"], str(row["Max Epochs"]), str(row["Early Stopping"])) for row in comparable})
    issues.append(
        {
            "severity": "low",
            "scope": "schedule",
            "issue": "Epoch budget and early-stopping flags differ by model",
            "detail": "; ".join(f"{family} {model} max_epochs={budget} early_stopping={flag}" for family, model, budget, flag in epochs),
        }
    )
    stopped = [row for row in comparable if str(row["Early Stopped"]).lower() == "true"]
    if stopped:
        issues.append(
            {
                "severity": "low",
                "scope": "early_stop",
                "issue": f"{len(stopped)} compared runs stopped before the epoch budget",
                "detail": "; ".join(
                    f"{row['Model']} {row['Penetration Rate']} {row['Task Type']} {row['Input Type']}->{row['Output Type']} epoch {row['Epochs Executed']}/{row['Max Epochs']}"
                    for row in stopped[:20]
                ),
            }
        )
    else:
        issues.append(
            {
                "severity": "low",
                "scope": "early_stop",
                "issue": "No compared run has early_stopped=true",
                "detail": "Budgets still differ. A model that trains for more epochs is not the same optimization schedule.",
            }
        )
    norm_files = sorted({(row["Family"], row["Normalization File"]) for row in comparable})
    issues.append(
        {
            "severity": "medium",
            "scope": "normalization",
            "issue": "Resolved configs name a normalization file; four-state training replaces the target scaler at runtime",
            "detail": (
                "Recorded paths: "
                + "; ".join(f"{family}: {path or 'blank'}" for family, path in norm_files)
                + ". Single-channel runs use normalization.json (global z-score, input scaler per penetration, shared full-inflow target scaler, fit on training days). "
                "Four-state engines call finalize_state_scaler, which reads state_normalization.json and uses one training-set scalar per physical variable. "
                "TLAST norm_flag=none is the published architecture switch, not a skipped scaler. "
                "Test JSON metrics are computed after denormalization, with MAPE only where the raw target is positive and with negative predictions left unclipped."
            ),
        }
    )
    seeds = sorted({row["Random Seed"] for row in comparable})
    issues.append(
        {
            "severity": "medium",
            "scope": "seed",
            "issue": "Every compared run uses a single seed",
            "detail": f"Seeds present: {', '.join(seeds) or 'none'}. No mean or standard deviation across seeds can be reported.",
        }
    )
    nonstandard = [row for row in rows if row["Family"] == "nonstandard"]
    if nonstandard:
        issues.append(
            {
                "severity": "low",
                "scope": "nonstandard",
                "issue": f"{len(nonstandard)} seed directories sit outside the comparison protocol",
                "detail": "; ".join(sorted({row['Result Directory'] for row in nonstandard})),
            }
        )
    zero_groups: dict[tuple[str, str, str, str], set[str]] = {}
    for row in comparable:
        key = (row["Family"], row["Task Type"], row["Output Type"], row["Penetration Rate"])
        zero_groups.setdefault(key, set()).add(row["Zero Target Fraction"])
    conflicting = [key for key, values in zero_groups.items() if len(values) > 1]
    if conflicting:
        issues.append(
            {
                "severity": "high",
                "scope": "labels",
                "issue": "Models that should share a test label report different zero-target fractions",
                "detail": "; ".join(f"{family} {task} {dst} {rate}" for family, task, dst, rate in conflicting[:20]),
            }
        )
    else:
        issues.append(
            {
                "severity": "low",
                "scope": "labels",
                "issue": "Within each family, task, output and penetration, zero-target fractions match across models",
                "detail": "This is consistent with a shared test label. It does not prove the arrays are identical; the prediction pass checks hashes.",
            }
        )
    legacy_zero = {
        row["Penetration Rate"]: row["Zero Target Fraction"]
        for row in comparable
        if row["Family"] == FAMILY_LEGACY and row["Model"] == "stid"
    }
    phase_zero = {
        row["Penetration Rate"]: row["Zero Target Fraction"]
        for row in comparable
        if row["Family"] == FAMILY_PHASE_SINGLE and row["Model"] == "stid"
    }
    four_zero = {
        row["Penetration Rate"]: row["Zero Target Fraction"]
        for row in comparable
        if row["Family"] == FAMILY_FOUR
        and row["Model"] == "stid"
        and row["Task Type"] == "single_to_single"
        and row["Output Type"] == "inflow"
    }
    issues.append(
        {
            "severity": "high",
            "scope": "families",
            "issue": "The two single-channel packages do not share inflow labels with each other",
            "detail": (
                f"STID p05 zero-target fraction legacy={legacy_zero.get('p05', 'absent')}, "
                f"phase3s flat={phase_zero.get('p05', 'absent')}, "
                f"four-state inflow={four_zero.get('p05', 'absent')}. "
                "Legacy uses the original signal timing. Phase3s and four-state use the regenerated "
                "3-second-phase FCD. Their MAE values are not one leaderboard."
            ),
        }
    )
    return issues


def scaler_fingerprint() -> dict[str, Any]:
    paths = {
        "legacy_normalization": LEGACY_DATA / "normalization.json",
        "phase3s_normalization": PHASE3S_DATA / "normalization.json",
        "phase3s_state_normalization": PHASE3S_DATA / "state_normalization.json",
        "legacy_split": LEGACY_DATA / "split_manifest.json",
        "phase3s_split": PHASE3S_DATA / "split_manifest.json",
        "phase3s_state_manifest": PHASE3S_DATA / "state_window_manifest.json",
    }
    payload: dict[str, Any] = {}
    for name, path in paths.items():
        payload[name] = {
            "path": path.as_posix(),
            "exists": path.is_file(),
            "sha256": file_sha256(path) if path.is_file() else "",
        }
    return payload


def main() -> None:
    ensure_dirs()
    rows: list[dict[str, Any]] = []
    roots = (("legacy", LEGACY_ROOT), ("phase3s", PHASE3S_ROOT))
    for root_name, root in roots:
        for model in MODELS:
            model_dir = root / model
            if not model_dir.is_dir():
                continue
            for seed_dir in iter_seed_dirs(model_dir):
                rows.append(scan_seed(model, root_name, seed_dir, model_dir))
    rows.sort(key=lambda row: (row["Family"], row["Model"], row["Task Type"], row["Input Type"], row["Output Type"], row["Penetration Rate"]))
    public = [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows]
    inventory_path = STATS.parent / "experiment_inventory.csv"
    write_csv(inventory_path, public, INVENTORY_FIELDS)
    issues = protocol_issues(public) + coverage_issues(public)
    write_json(STATS / "audit_issues.json", issues)
    write_json(STATS / "scaler_fingerprints.json", scaler_fingerprint())
    counts: dict[str, int] = {}
    for row in public:
        key = f"{row['Family']}|{row['Include In Comparison']}"
        counts[key] = counts.get(key, 0) + 1
    print(f"wrote {inventory_path} rows={len(public)}")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    print(f"issues={len(issues)}")


if __name__ == "__main__":
    main()
