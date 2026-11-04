"""Spatial, temporal, and bias summaries from saved test predictions.

Uses the test NPZ only. Validation predictions were used for checkpoint
selection and are not treated as a held-out error distribution.
"""

from __future__ import annotations

import gzip
import hashlib
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis_lib import (
    FAMILY_FOUR,
    FAMILY_LEGACY,
    FIGURES,
    MODEL_COLOR,
    MODELS,
    NODE_TABLE,
    OUT_ROOT,
    RATE_PERCENT,
    STATE_LABEL,
    STATES,
    TABLES,
    as_float,
    ensure_dirs,
    label,
    read_csv,
    read_json,
    write_csv,
    write_json,
)

PERIODS = (
    ("night_00_06", 0, 72),
    ("morning_peak_07_09", 84, 108),
    ("midday_10_16", 120, 192),
    ("evening_peak_17_19", 204, 228),
)
WEEKDAY = {
    2: "2023-04-26",
    3: "2023-04-27",
    4: "2023-04-28",
    5: "2023-04-29",
    6: "2023-04-30",
}


def load_edges() -> list[dict[str, str]]:
    rows = read_csv(NODE_TABLE)
    if len(rows) != 1406:
        raise RuntimeError(f"expected 1406 nodes, found {len(rows)}")
    return rows


def sha256_array(array: np.ndarray) -> str:
    packed = np.ascontiguousarray(array)
    return hashlib.sha256(packed.view(np.uint8)).hexdigest()


def squeeze_prediction(array: np.ndarray) -> np.ndarray:
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise RuntimeError(f"unexpected prediction shape {array.shape}")
    return array


def pearson(pred: np.ndarray, true: np.ndarray) -> float:
    left = pred.astype(np.float64, copy=False).ravel()
    right = true.astype(np.float64, copy=False).ravel()
    left = left - left.mean()
    right = right - right.mean()
    denom = np.sqrt(np.dot(left, left) * np.dot(right, right))
    if denom == 0:
        return float("nan")
    return float(np.dot(left, right) / denom)


def flow_group(mean_true: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    groups = np.full(mean_true.shape, "inactive", dtype=object)
    active = mean_true > 1e-6
    info = {
        "inactive_roads": int((~active).sum()),
        "active_roads": int(active.sum()),
        "low_max_mean": "",
        "mid_max_mean": "",
    }
    if int(active.sum()) < 30:
        return groups, info
    low_cut, high_cut = np.quantile(mean_true[active], [1.0 / 3.0, 2.0 / 3.0])
    groups[active & (mean_true <= low_cut)] = "low"
    groups[active & (mean_true > low_cut) & (mean_true <= high_cut)] = "medium"
    groups[active & (mean_true > high_cut)] = "high"
    info["low_max_mean"] = float(low_cut)
    info["mid_max_mean"] = float(high_cut)
    return groups, info


def check_alignment(start: np.ndarray, target: np.ndarray, day_index: np.ndarray) -> dict[str, object]:
    start = start.reshape(-1)
    target = target.reshape(-1)
    day_index = day_index.reshape(-1)
    slot_ok = bool(np.array_equal(target, start + 11) and np.all((start >= 0) & (target <= 287)))
    day_counts = {int(day): int(np.sum(day_index == day)) for day in np.unique(day_index)}
    expected_days = day_counts == {day: 277 for day in WEEKDAY}
    blocks_ok = True
    if expected_days:
        for day in WEEKDAY:
            block = start[day_index == day]
            if not np.array_equal(block, np.arange(277)):
                blocks_ok = False
                break
    else:
        blocks_ok = False
    return {
        "target_is_last_input_slot": slot_ok,
        "test_days_match_manifest": expected_days and blocks_ok,
        "day_counts": day_counts,
    }


def magnitude_bins(true: np.ndarray, pred: np.ndarray) -> list[dict[str, float]]:
    true_flat = true.ravel()
    pred_flat = pred.ravel()
    cuts = (
        ("0", true_flat == 0),
        ("(0,2]", (true_flat > 0) & (true_flat <= 2)),
        ("(2,5]", (true_flat > 2) & (true_flat <= 5)),
        ("(5,10]", (true_flat > 5) & (true_flat <= 10)),
        ("(10,20]", (true_flat > 10) & (true_flat <= 20)),
        (">20", true_flat > 20),
    )
    rows = []
    for name, mask in cuts:
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin": name, "count": 0, "mae": "", "bias": "", "mean_true": ""})
            continue
        error = pred_flat[mask].astype(np.float64) - true_flat[mask].astype(np.float64)
        rows.append(
            {
                "bin": name,
                "count": count,
                "mae": float(np.mean(np.abs(error))),
                "bias": float(np.mean(error)),
                "mean_true": float(np.mean(true_flat[mask])),
            }
        )
    return rows


def period_rows(pred: np.ndarray, true: np.ndarray, target_slot: np.ndarray) -> list[dict[str, float]]:
    slots = target_slot.reshape(-1)
    output = []
    for name, left, right in PERIODS:
        mask = (slots >= left) & (slots < right)
        if not np.any(mask):
            continue
        error = pred[mask].astype(np.float64) - true[mask].astype(np.float64)
        output.append(
            {
                "period": name,
                "mae": float(np.mean(np.abs(error))),
                "bias": float(np.mean(error)),
                "samples": int(mask.sum()),
            }
        )
    return output


def hour_rows(pred: np.ndarray, true: np.ndarray, target_slot: np.ndarray) -> list[dict[str, float]]:
    slots = target_slot.reshape(-1)
    hours = slots // 12
    output = []
    for hour in range(24):
        mask = hours == hour
        if not np.any(mask):
            continue
        error = pred[mask].astype(np.float64) - true[mask].astype(np.float64)
        output.append({"hour": hour, "mae": float(np.mean(np.abs(error))), "bias": float(np.mean(error))})
    return output


def day_rows(pred: np.ndarray, true: np.ndarray, day_index: np.ndarray) -> list[dict[str, object]]:
    output = []
    for day, date_name in WEEKDAY.items():
        mask = day_index.reshape(-1) == day
        if not np.any(mask):
            continue
        error = pred[mask].astype(np.float64) - true[mask].astype(np.float64)
        output.append(
            {
                "day_index": day,
                "date": date_name,
                "weekday": date_name and ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][
                    __import__("datetime").date.fromisoformat(date_name).weekday()
                ],
                "mae": float(np.mean(np.abs(error))),
                "bias": float(np.mean(error)),
            }
        )
    return output


def confirm_phase3s_duplicates(inventory: list[dict[str, str]]) -> dict[str, object]:
    pairs = []
    for row in inventory:
        if row["Family"] != "phase3s_single_channel" or row["Include In Comparison"] != "yes":
            continue
        match = next(
            item
            for item in inventory
            if item["Include In Comparison"] == "yes"
            and item["Family"] == FAMILY_FOUR
            and item["Model"] == row["Model"]
            and item["Task Type"] == "single_to_single"
            and item["Output Type"] == "inflow"
            and item["Penetration Rate"] == row["Penetration Rate"]
        )
        pairs.append((row, match))
    same_metrics = 0
    same_pred = 0
    for left, right in pairs:
        if left["MAE"] == right["MAE"] and left["RMSE"] == right["RMSE"] and left["WAPE"] == right["WAPE"]:
            same_metrics += 1
        with np.load(left["Prediction File"]) as loaded:
            left_hash = sha256_array(np.asarray(loaded["pred_raw"]))
        with np.load(right["Prediction File"]) as loaded:
            right_hash = sha256_array(np.asarray(loaded["pred_raw"]))
        if left_hash == right_hash:
            same_pred += 1
    return {
        "pairs": len(pairs),
        "identical_published_metrics": same_metrics,
        "identical_pred_raw_sha256": same_pred,
    }


def plot_hourly(hour_table: list[dict[str, object]], family: str, task: str, channel: str, rate: str, path: Path) -> None:
    selected = [
        row
        for row in hour_table
        if row["family"] == family and row["task_type"] == task and row["channel"] == channel and row["penetration"] == rate
    ]
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for model in MODELS:
        rows = [row for row in selected if row["model"] == model]
        if not rows:
            continue
        rows.sort(key=lambda row: int(row["hour"]))
        ax.plot(
            [int(row["hour"]) for row in rows],
            [float(row["mae"]) for row in rows],
            color=MODEL_COLOR[model],
            linewidth=1.6,
            label=label(model),
        )
    ax.set_xlabel("Hour of test day")
    ax.set_ylabel("Test MAE (vehicles / road / 5 min)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title(f"{label_task(task)} {STATE_LABEL.get(channel, channel)} {rate}")
    ax.axvspan(7, 9, color="#bbbbbb", alpha=0.35, label="07:00-09:00" if family else None)
    ax.axvspan(17, 19, color="#888888", alpha=0.18)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def label_task(task: str) -> str:
    return {
        "single_channel_inflow": "Legacy single-channel",
        "single_to_single": "Single-to-single",
        "all_to_single": "All-to-single",
        "all_to_all": "All-to-all",
    }.get(task, task)


def plot_terciles(spatial: list[dict[str, object]], family: str, task: str, channel: str, rate: str, path: Path) -> None:
    selected = [
        row
        for row in spatial
        if row["family"] == family
        and row["task_type"] == task
        and row["channel"] == channel
        and row["penetration"] == rate
        and row["flow_group"] in {"low", "medium", "high"}
    ]
    models = [model for model in MODELS if any(row["model"] == model for row in selected)]
    if not models:
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    width = 0.25
    xs = np.arange(len(models))
    for offset, group in enumerate(("low", "medium", "high")):
        heights = []
        for model in models:
            match = next(row for row in selected if row["model"] == model and row["flow_group"] == group)
            heights.append(float(match["mae"]))
        ax.bar(xs + (offset - 1) * width, heights, width=width, label=group)
    ax.set_xticks(xs)
    ax.set_xticklabels([label(model) for model in models], rotation=20, ha="right")
    ax.set_ylabel("Test MAE within roads of this group")
    ax.set_title(f"{label_task(task)} {STATE_LABEL.get(channel, channel)} {rate}")
    ax.legend(frameon=False, title="Test-mean tercile")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_bias_bins(bins: list[dict[str, object]], family: str, task: str, channel: str, rate: str, path: Path) -> None:
    selected = [
        row
        for row in bins
        if row["family"] == family and row["task_type"] == task and row["channel"] == channel and row["penetration"] == rate
    ]
    order = ["0", "(0,2]", "(2,5]", "(5,10]", "(10,20]", ">20"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for model in MODELS:
        rows = [row for row in selected if row["model"] == model and row["bias"] != ""]
        if not rows:
            continue
        by_bin = {row["bin"]: float(row["bias"]) for row in rows}
        ax.plot(order, [by_bin.get(name, np.nan) for name in order], marker="o", color=MODEL_COLOR[model], label=label(model))
    ax.axhline(0.0, color="#444444", linewidth=0.8)
    ax.set_xlabel("True count bin")
    ax.set_ylabel("Mean error (prediction - true)")
    ax.set_title(f"{label_task(task)} {STATE_LABEL.get(channel, channel)} {rate}")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    inventory = read_csv(OUT_ROOT / "experiment_inventory.csv")
    edges = load_edges()
    duplicate = confirm_phase3s_duplicates(inventory)
    print("phase3s duplicate check", duplicate, flush=True)
    runs = [
        row
        for row in inventory
        if row["Include In Comparison"] == "yes" and row["Family"] in {FAMILY_LEGACY, FAMILY_FOUR}
    ]
    node_path = TABLES / "node_metrics.csv.gz"
    run_rows = []
    spatial_rows = []
    period_table = []
    hour_table = []
    day_table = []
    bin_table = []
    label_hashes: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)
    node_mae_bank: dict[tuple[str, str, str, str, str], np.ndarray] = {}
    with gzip.open(node_path, "wt", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "family",
                "model",
                "task_type",
                "channel",
                "penetration",
                "node_index",
                "edge_id",
                "mae",
                "rmse",
                "bias",
                "mean_true",
                "flow_group",
            ],
        )
        writer.writeheader()
        for index, row in enumerate(runs, start=1):
            prediction_path = Path(row["Prediction File"])
            published = read_json(Path(row["Metrics File"]))
            with np.load(prediction_path) as loaded:
                pred = squeeze_prediction(np.asarray(loaded["pred_raw"]))
                true = squeeze_prediction(np.asarray(loaded["y_raw"]))
                start = np.asarray(loaded["window_start_slot"])
                target = np.asarray(loaded["target_slot"])
                day_index = np.asarray(loaded["day_index"])
            alignment = check_alignment(start, target, day_index)
            if pred.shape != true.shape:
                raise RuntimeError(f"shape mismatch {prediction_path}")
            channels = [row["Output Type"]] if pred.shape[-1] == 1 else list(STATES)
            if pred.shape[-1] != len(channels):
                raise RuntimeError(f"{prediction_path} channels {pred.shape[-1]} != {channels}")
            recomputed = float(np.mean(np.abs(pred.astype(np.float64) - true.astype(np.float64))))
            published_mae = float(published["mae"])
            max_channel_gap = 0.0
            for channel_index, channel in enumerate(channels):
                channel_pred = pred[:, :, channel_index]
                channel_true = true[:, :, channel_index]
                channel_mae = float(np.mean(np.abs(channel_pred.astype(np.float64) - channel_true.astype(np.float64))))
                published_key = "mae" if len(channels) == 1 else f"{channel}_mae"
                published_channel = float(published[published_key])
                max_channel_gap = max(max_channel_gap, abs(channel_mae - published_channel))
                label_hashes[(row["Family"], row["Task Type"], row["Input Type"], channel, row["Penetration Rate"])].add(
                    sha256_array(channel_true)
                )
                mean_true = channel_true.mean(axis=0)
                abs_error = np.abs(channel_pred.astype(np.float64) - channel_true.astype(np.float64))
                sq_error = (channel_pred.astype(np.float64) - channel_true.astype(np.float64)) ** 2
                node_mae = abs_error.mean(axis=0)
                node_rmse = np.sqrt(sq_error.mean(axis=0))
                node_bias = (channel_pred.astype(np.float64) - channel_true.astype(np.float64)).mean(axis=0)
                groups, group_info = flow_group(mean_true)
                node_mae_bank[(row["Family"], row["Task Type"], channel, row["Penetration Rate"], row["Model"])] = node_mae.astype(np.float32)
                for node_index in range(node_mae.shape[0]):
                    writer.writerow(
                        {
                            "family": row["Family"],
                            "model": row["Model"],
                            "task_type": row["Task Type"],
                            "channel": channel,
                            "penetration": row["Penetration Rate"],
                            "node_index": node_index,
                            "edge_id": edges[node_index]["edge_id"],
                            "mae": f"{node_mae[node_index]:.8g}",
                            "rmse": f"{node_rmse[node_index]:.8g}",
                            "bias": f"{node_bias[node_index]:.8g}",
                            "mean_true": f"{mean_true[node_index]:.8g}",
                            "flow_group": groups[node_index],
                        }
                    )
                for group_name in ("inactive", "low", "medium", "high"):
                    mask = groups == group_name
                    if not np.any(mask):
                        continue
                    spatial_rows.append(
                        {
                            "family": row["Family"],
                            "model": row["Model"],
                            "task_type": row["Task Type"],
                            "channel": channel,
                            "penetration": row["Penetration Rate"],
                            "flow_group": group_name,
                            "roads": int(mask.sum()),
                            "mae": float(node_mae[mask].mean()),
                            "bias": float(node_bias[mask].mean()),
                            "mean_true": float(mean_true[mask].mean()),
                            "low_max_mean": group_info["low_max_mean"],
                            "mid_max_mean": group_info["mid_max_mean"],
                        }
                    )
                for item in period_rows(channel_pred, channel_true, target):
                    period_table.append(
                        {
                            "family": row["Family"],
                            "model": row["Model"],
                            "task_type": row["Task Type"],
                            "channel": channel,
                            **item,
                            "penetration": row["Penetration Rate"],
                        }
                    )
                for item in hour_rows(channel_pred, channel_true, target):
                    hour_table.append(
                        {
                            "family": row["Family"],
                            "model": row["Model"],
                            "task_type": row["Task Type"],
                            "channel": channel,
                            "penetration": row["Penetration Rate"],
                            **item,
                        }
                    )
                for item in day_rows(channel_pred, channel_true, day_index):
                    day_table.append(
                        {
                            "family": row["Family"],
                            "model": row["Model"],
                            "task_type": row["Task Type"],
                            "channel": channel,
                            "penetration": row["Penetration Rate"],
                            **item,
                        }
                    )
                for item in magnitude_bins(channel_true, channel_pred):
                    bin_table.append(
                        {
                            "family": row["Family"],
                            "model": row["Model"],
                            "task_type": row["Task Type"],
                            "channel": channel,
                            "penetration": row["Penetration Rate"],
                            **item,
                        }
                    )
            error = pred.astype(np.float64) - true.astype(np.float64)
            run_rows.append(
                {
                    "family": row["Family"],
                    "model": row["Model"],
                    "task_type": row["Task Type"],
                    "input_type": row["Input Type"],
                    "output_type": row["Output Type"],
                    "penetration": row["Penetration Rate"],
                    "aligned": alignment["target_is_last_input_slot"],
                    "test_days_match_manifest": alignment["test_days_match_manifest"],
                    "published_mae": published_mae,
                    "recomputed_mae": recomputed,
                    "mae_abs_diff": abs(recomputed - published_mae),
                    "max_channel_mae_abs_diff": max_channel_gap,
                    "bias": float(error.mean()),
                    "fraction_overestimate": float(np.mean(error > 0)),
                    "correlation": pearson(pred, true),
                    "prediction_file": row["Prediction File"],
                }
            )
            if index % 20 == 0 or index == len(runs):
                print(f"processed {index}/{len(runs)}", flush=True)
    write_csv(
        TABLES / "prediction_audit.csv",
        run_rows,
        [
            "family",
            "model",
            "task_type",
            "input_type",
            "output_type",
            "penetration",
            "aligned",
            "test_days_match_manifest",
            "published_mae",
            "recomputed_mae",
            "mae_abs_diff",
            "max_channel_mae_abs_diff",
            "bias",
            "fraction_overestimate",
            "correlation",
            "prediction_file",
        ],
    )
    write_csv(
        TABLES / "spatial_by_flow_tercile.csv",
        spatial_rows,
        [
            "family",
            "model",
            "task_type",
            "channel",
            "penetration",
            "flow_group",
            "roads",
            "mae",
            "bias",
            "mean_true",
            "low_max_mean",
            "mid_max_mean",
        ],
    )
    write_csv(
        TABLES / "temporal_by_period.csv",
        period_table,
        ["family", "model", "task_type", "channel", "penetration", "period", "mae", "bias", "samples"],
    )
    write_csv(
        TABLES / "temporal_by_hour.csv",
        hour_table,
        ["family", "model", "task_type", "channel", "penetration", "hour", "mae", "bias"],
    )
    write_csv(
        TABLES / "temporal_by_day.csv",
        day_table,
        ["family", "model", "task_type", "channel", "penetration", "day_index", "date", "weekday", "mae", "bias"],
    )
    write_csv(
        TABLES / "bias_by_magnitude.csv",
        bin_table,
        ["family", "model", "task_type", "channel", "penetration", "bin", "count", "mae", "bias", "mean_true"],
    )
    hash_conflicts = [
        {"family": key[0], "task_type": key[1], "input_type": key[2], "channel": key[3], "penetration": key[4], "distinct_label_hashes": len(values)}
        for key, values in label_hashes.items()
        if len(values) != 1
    ]
    overlap_rows = []
    correlation_rows = []
    for family, task, channel, rate in {
        (key[0], key[1], key[2], key[3]) for key in node_mae_bank
    }:
        models = [model for model in MODELS if (family, task, channel, rate, model) in node_mae_bank]
        if len(models) < 2:
            continue
        hard_sets = {}
        for model in models:
            values = node_mae_bank[(family, task, channel, rate, model)]
            cutoff = np.quantile(values, 0.9)
            hard_sets[model] = set(np.flatnonzero(values >= cutoff).tolist())
        shared = set.intersection(*(hard_sets[model] for model in models)) if hard_sets else set()
        overlap_rows.append(
            {
                "family": family,
                "task_type": task,
                "channel": channel,
                "penetration": rate,
                "models": len(models),
                "top_decile_size": len(next(iter(hard_sets.values()))),
                "intersection_all_models": len(shared),
                "shared_edge_ids": ",".join(edges[index]["edge_id"] for index in sorted(shared)[:30]),
            }
        )
        for left_index, left in enumerate(models):
            for right in models[left_index + 1 :]:
                a = node_mae_bank[(family, task, channel, rate, left)].astype(np.float64)
                b = node_mae_bank[(family, task, channel, rate, right)].astype(np.float64)
                correlation_rows.append(
                    {
                        "family": family,
                        "task_type": task,
                        "channel": channel,
                        "penetration": rate,
                        "model_a": left,
                        "model_b": right,
                        "node_mae_correlation": pearson(a, b),
                    }
                )
    write_csv(
        TABLES / "hard_road_overlap.csv",
        overlap_rows,
        ["family", "task_type", "channel", "penetration", "models", "top_decile_size", "intersection_all_models", "shared_edge_ids"],
    )
    write_csv(
        TABLES / "node_mae_correlation.csv",
        correlation_rows,
        ["family", "task_type", "channel", "penetration", "model_a", "model_b", "node_mae_correlation"],
    )
    max_gap = max(float(row["mae_abs_diff"]) for row in run_rows)
    max_channel = max(float(row["max_channel_mae_abs_diff"]) for row in run_rows)
    aligned = sum(str(row["aligned"]) == "True" for row in run_rows)
    days_ok = sum(str(row["test_days_match_manifest"]) == "True" for row in run_rows)
    summary = {
        "runs": len(run_rows),
        "aligned": aligned,
        "test_days_match_manifest": days_ok,
        "max_mae_abs_diff": max_gap,
        "max_channel_mae_abs_diff": max_channel,
        "label_hash_conflicts": hash_conflicts,
        "phase3s_flat_vs_four_state_inflow": duplicate,
        "node_metrics": node_path.as_posix(),
    }
    write_json(TABLES.parent / "stats" / "error_summary.json", summary)
    plot_hourly(
        hour_table,
        FAMILY_LEGACY,
        "single_channel_inflow",
        "inflow",
        "p05",
        FIGURES / "error" / "legacy_hourly_mae_p05.png",
    )
    plot_hourly(
        hour_table,
        FAMILY_LEGACY,
        "single_channel_inflow",
        "inflow",
        "p70",
        FIGURES / "error" / "legacy_hourly_mae_p70.png",
    )
    for channel in ("inflow", "accumulation_start"):
        for rate in ("p05", "p70"):
            plot_hourly(
                hour_table,
                FAMILY_FOUR,
                "single_to_single",
                channel,
                rate,
                FIGURES / "error" / f"s2s_hourly_{channel}_{rate}.png",
            )
            plot_terciles(
                spatial_rows,
                FAMILY_FOUR,
                "single_to_single",
                channel,
                rate,
                FIGURES / "error" / f"s2s_tercile_{channel}_{rate}.png",
            )
        plot_bias_bins(
            bin_table,
            FAMILY_FOUR,
            "single_to_single",
            channel,
            "p05",
            FIGURES / "error" / f"s2s_bias_bins_{channel}_p05.png",
        )
    plot_terciles(
        spatial_rows,
        FAMILY_LEGACY,
        "single_channel_inflow",
        "inflow",
        "p05",
        FIGURES / "error" / "legacy_tercile_inflow_p05.png",
    )
    plot_bias_bins(
        bin_table,
        FAMILY_LEGACY,
        "single_channel_inflow",
        "inflow",
        "p05",
        FIGURES / "error" / "legacy_bias_bins_inflow_p05.png",
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
