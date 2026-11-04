"""Build the long metric table, rankings, penetration summaries, and curves.

Comparison sets never mix families. All-to-all pooled MAE is stored but is not
used as a cross-state score.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis_lib import (
    FAMILY_FOUR,
    FAMILY_LEGACY,
    FAMILY_PHASE_SINGLE,
    FIGURES,
    METRIC_KEYS,
    MODEL_COLOR,
    MODEL_MARKER,
    MODELS,
    OUT_ROOT,
    PRACTICAL_GAP_PCT,
    RATE_PERCENT,
    RATES,
    STATE_LABEL,
    STATES,
    STATS,
    TABLES,
    TASK_LABEL,
    as_float,
    ensure_dirs,
    label,
    read_csv,
    read_json,
    write_csv,
    write_json,
)

LONG_FIELDS = [
    "family",
    "model",
    "task_type",
    "input_type",
    "output_type",
    "channel",
    "channel_role",
    "penetration",
    "penetration_percent",
    "seed",
    "mae",
    "rmse",
    "mape_nonzero",
    "wape",
    "negative_prediction_fraction",
    "zero_target_fraction",
    "best_epoch",
    "selection_matches_mae_best",
    "result_directory",
    "metrics_file",
]

RANK_FIELDS = [
    "family",
    "task_type",
    "input_type",
    "output_type",
    "channel",
    "penetration",
    "model",
    "mae",
    "rmse",
    "mape_nonzero",
    "wape",
    "rank_mae",
    "rank_rmse",
    "rank_mape_nonzero",
    "rank_wape",
    "best_mae",
    "mae_gap",
    "mae_gap_percent",
    "gap_below_practical_threshold",
    "models_ranked",
]

DEGRADE_FIELDS = [
    "family",
    "task_type",
    "input_type",
    "output_type",
    "channel",
    "model",
    "mae_p70",
    "mae_p05",
    "mae_increase_p70_to_p05",
    "relative_increase_percent",
    "low_range_slope_per_point",
    "high_range_slope_per_point",
    "monotonic_nonincreasing_mae",
    "rates_present",
]


def channel_rows(inventory_row: dict[str, str]) -> list[dict[str, Any]]:
    metrics = read_json(Path(inventory_row["Metrics File"]))
    base = {
        "family": inventory_row["Family"],
        "model": inventory_row["Model"],
        "task_type": inventory_row["Task Type"],
        "input_type": inventory_row["Input Type"],
        "output_type": inventory_row["Output Type"],
        "penetration": inventory_row["Penetration Rate"],
        "penetration_percent": RATE_PERCENT[inventory_row["Penetration Rate"]],
        "seed": inventory_row["Random Seed"],
        "best_epoch": inventory_row["Best Epoch"],
        "selection_matches_mae_best": inventory_row["Selection Epoch Matches MAE-best"],
        "result_directory": inventory_row["Result Directory"],
        "metrics_file": inventory_row["Metrics File"],
    }
    rows: list[dict[str, Any]] = []
    if inventory_row["Task Type"] == "all_to_all":
        rows.append(metric_record(base, metrics, "pooled", "pooled_not_for_state_ranking"))
        for state in STATES:
            rows.append(metric_record(base, metrics, state, "per_state", prefix=f"{state}_"))
        return rows
    channel = inventory_row["Output Type"] or "inflow"
    rows.append(metric_record(base, metrics, channel, "per_state"))
    return rows


def metric_record(
    base: dict[str, Any],
    metrics: dict[str, Any],
    channel: str,
    role: str,
    prefix: str = "",
) -> dict[str, Any]:
    row = dict(base)
    row["channel"] = channel
    row["channel_role"] = role
    for key in (*METRIC_KEYS, "negative_prediction_fraction", "zero_target_fraction"):
        value = metrics.get(f"{prefix}{key}")
        row[key] = "" if value is None else f"{float(value):.10g}"
    return row


def ranked(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["channel_role"] == "pooled_not_for_state_ranking":
            continue
        key = (
            row["family"],
            row["task_type"],
            row["input_type"],
            row["output_type"],
            row["channel"],
            row["penetration"],
        )
        groups[key].append(row)
    output: list[dict[str, Any]] = []
    for key, group in groups.items():
        values = {metric: [as_float(row[metric]) for row in group] for metric in METRIC_KEYS}
        ranks = {metric: rank_map(group, metric) for metric in METRIC_KEYS}
        best_mae = min(value for value in values["mae"] if value is not None)
        for row in group:
            mae = as_float(row["mae"])
            gap = None if mae is None else mae - best_mae
            gap_pct = None if gap is None or best_mae == 0 else 100.0 * gap / best_mae
            output.append(
                {
                    "family": key[0],
                    "task_type": key[1],
                    "input_type": key[2],
                    "output_type": key[3],
                    "channel": key[4],
                    "penetration": key[5],
                    "model": row["model"],
                    "mae": row["mae"],
                    "rmse": row["rmse"],
                    "mape_nonzero": row["mape_nonzero"],
                    "wape": row["wape"],
                    "rank_mae": ranks["mae"][row["model"]],
                    "rank_rmse": ranks["rmse"][row["model"]],
                    "rank_mape_nonzero": ranks["mape_nonzero"][row["model"]],
                    "rank_wape": ranks["wape"][row["model"]],
                    "best_mae": f"{best_mae:.10g}",
                    "mae_gap": "" if gap is None else f"{gap:.10g}",
                    "mae_gap_percent": "" if gap_pct is None else f"{gap_pct:.10g}",
                    "gap_below_practical_threshold": ""
                    if gap_pct is None
                    else ("yes" if gap_pct < PRACTICAL_GAP_PCT else "no"),
                    "models_ranked": len(group),
                }
            )
    output.sort(
        key=lambda row: (
            row["family"],
            row["task_type"],
            row["channel"],
            row["penetration"],
            int(row["rank_mae"]),
            row["model"],
        )
    )
    return output


def rank_map(group: list[dict[str, str]], metric: str) -> dict[str, int]:
    ordered = sorted(group, key=lambda row: (as_float(row[metric]) is None, as_float(row[metric]) or math.inf, row["model"]))
    return {row["model"]: index + 1 for index, row in enumerate(ordered)}


def series_map(rows: list[dict[str, str]]) -> dict[tuple[str, ...], dict[str, float]]:
    found: dict[tuple[str, ...], dict[str, float]] = {}
    for row in rows:
        if row["channel_role"] == "pooled_not_for_state_ranking":
            continue
        value = as_float(row["mae"])
        if value is None:
            continue
        key = (row["family"], row["task_type"], row["input_type"], row["output_type"], row["channel"], row["model"])
        found.setdefault(key, {})[row["penetration"]] = value
    return found


def degradation(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for key, by_rate in series_map(rows).items():
        family, task, src, dst, channel, model = key
        present = [rate for rate in RATES if rate in by_rate]
        mae70 = by_rate.get("p70")
        mae05 = by_rate.get("p05")
        mae20 = by_rate.get("p20")
        mae40 = by_rate.get("p40")
        increase = None if mae70 is None or mae05 is None else mae05 - mae70
        relative = None if increase is None or not mae70 else 100.0 * increase / mae70
        low_slope = None if mae05 is None or mae20 is None else (mae05 - mae20) / (20 - 5)
        high_slope = None if mae40 is None or mae70 is None else (mae40 - mae70) / (70 - 40)
        monotonic = True
        last = None
        for rate in RATES:
            if rate not in by_rate:
                continue
            if last is not None and by_rate[rate] > last + 1e-9:
                monotonic = False
            last = by_rate[rate]
        output.append(
            {
                "family": family,
                "task_type": task,
                "input_type": src,
                "output_type": dst,
                "channel": channel,
                "model": model,
                "mae_p70": "" if mae70 is None else f"{mae70:.10g}",
                "mae_p05": "" if mae05 is None else f"{mae05:.10g}",
                "mae_increase_p70_to_p05": "" if increase is None else f"{increase:.10g}",
                "relative_increase_percent": "" if relative is None else f"{relative:.10g}",
                "low_range_slope_per_point": "" if low_slope is None else f"{low_slope:.10g}",
                "high_range_slope_per_point": "" if high_slope is None else f"{high_slope:.10g}",
                "monotonic_nonincreasing_mae": "yes" if present == list(RATES) and monotonic else ("incomplete" if present != list(RATES) else "no"),
                "rates_present": ",".join(present),
            }
        )
    return output


def plot_curves(
    rows: list[dict[str, str]],
    *,
    family: str,
    task_type: str,
    channel: str,
    metric: str,
    path_name: str,
    ylabel: str,
) -> None:
    selected = [
        row
        for row in rows
        if row["family"] == family
        and row["task_type"] == task_type
        and row["channel"] == channel
        and row["channel_role"] != "pooled_not_for_state_ranking"
    ]
    if not selected:
        return
    models = [model for model in MODELS if any(row["model"] == model for row in selected)]
    if len(models) < 2:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for model in models:
        xs = []
        ys = []
        for rate in RATES:
            match = next((row for row in selected if row["model"] == model and row["penetration"] == rate), None)
            if match is None or as_float(match[metric]) is None:
                continue
            xs.append(RATE_PERCENT[rate])
            ys.append(as_float(match[metric]))
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            color=MODEL_COLOR[model],
            marker=MODEL_MARKER[model],
            linewidth=1.8,
            markersize=6,
            label=label(model),
        )
    ax.set_xlabel("Penetration rate (%)")
    ax.set_ylabel(ylabel)
    title_task = TASK_LABEL.get(task_type, task_type)
    title_channel = STATE_LABEL.get(channel, channel)
    ax.set_title(f"{title_task}, {title_channel}")
    ax.set_xticks([RATE_PERCENT[rate] for rate in RATES])
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    destination = FIGURES / path_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination)
    plt.close(fig)


def plot_degradation(rows: list[dict[str, str]], family: str, task_type: str, channel: str, path_name: str) -> None:
    selected = [
        row
        for row in rows
        if row["family"] == family and row["task_type"] == task_type and row["channel"] == channel and row["relative_increase_percent"] != ""
    ]
    if len(selected) < 2:
        return
    selected = sorted(selected, key=lambda row: MODELS.index(row["model"]) if row["model"] in MODELS else 99)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    xs = range(len(selected))
    ax.bar(
        list(xs),
        [float(row["relative_increase_percent"]) for row in selected],
        color=[MODEL_COLOR.get(row["model"], "#777777") for row in selected],
    )
    ax.set_xticks(list(xs))
    ax.set_xticklabels([label(row["model"]) for row in selected], rotation=20, ha="right")
    ax.set_ylabel("MAE increase from p70 to p05 (%)")
    ax.set_title(f"{TASK_LABEL.get(task_type, task_type)}, {STATE_LABEL.get(channel, channel)}")
    fig.tight_layout()
    destination = FIGURES / path_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination)
    plt.close(fig)


def plot_mechanism(rows: list[dict[str, str]], channel: str) -> None:
    task_order = ("single_to_single", "all_to_single", "all_to_all")
    styles = {"single_to_single": "-", "all_to_single": "--", "all_to_all": ":"}
    models = [
        model
        for model in MODELS
        if any(row["model"] == model and row["channel"] == channel and row["family"] == FAMILY_FOUR for row in rows)
    ]
    if not models:
        return
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.2), sharex=True, sharey=True)
    for ax, model in zip(axes.ravel(), models):
        for task in task_order:
            xs = []
            ys = []
            for rate in RATES:
                match = next(
                    (
                        row
                        for row in rows
                        if row["family"] == FAMILY_FOUR
                        and row["model"] == model
                        and row["task_type"] == task
                        and row["channel"] == channel
                        and row["penetration"] == rate
                        and row["channel_role"] != "pooled_not_for_state_ranking"
                    ),
                    None,
                )
                if match is None:
                    continue
                xs.append(RATE_PERCENT[rate])
                ys.append(float(match["mae"]))
            ax.plot(xs, ys, linestyle=styles[task], color=MODEL_COLOR[model], marker="o", markersize=3.5, label=TASK_LABEL[task])
        ax.set_title(label(model), fontsize=10)
        ax.set_xticks([5, 20, 40, 70])
    for ax in axes[1, :]:
        ax.set_xlabel("Penetration (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Test MAE")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, loc="upper center", ncol=3)
    fig.suptitle(f"{STATE_LABEL[channel]}: single input, joint input, joint output", y=1.02)
    fig.tight_layout()
    destination = FIGURES / "four_state" / f"mechanism_{channel}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def contrast_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    index: dict[tuple[str, str, str], float] = {}
    for row in rows:
        if row["family"] != FAMILY_FOUR or row["channel_role"] == "pooled_not_for_state_ranking":
            continue
        if row["channel"] not in STATES:
            continue
        index[(row["model"], row["task_type"], row["channel"], row["penetration"])] = float(row["mae"])
    output: list[dict[str, Any]] = []
    for model in MODELS:
        for channel in STATES:
            for rate in RATES:
                s2s = index.get((model, "single_to_single", channel, rate))
                a2s = index.get((model, "all_to_single", channel, rate))
                a2a = index.get((model, "all_to_all", channel, rate))
                if s2s is None or a2s is None or a2a is None:
                    continue
                output.append(
                    {
                        "model": model,
                        "channel": channel,
                        "penetration": rate,
                        "s2s_mae": f"{s2s:.10g}",
                        "a2s_mae": f"{a2s:.10g}",
                        "a2a_mae": f"{a2a:.10g}",
                        "fusion_gap_a2s_minus_s2s": f"{(a2s - s2s):.10g}",
                        "fusion_gap_percent": f"{(100.0 * (a2s - s2s) / s2s):.10g}",
                        "multitask_gap_a2a_minus_a2s": f"{(a2a - a2s):.10g}",
                        "multitask_gap_percent": f"{(100.0 * (a2a - a2s) / a2s):.10g}",
                        "confounded_gap_a2a_minus_s2s": f"{(a2a - s2s):.10g}",
                        "confounded_gap_percent": f"{(100.0 * (a2a - s2s) / s2s):.10g}",
                    }
                )
    return output


def main() -> None:
    ensure_dirs()
    inventory = [
        row
        for row in read_csv(OUT_ROOT / "experiment_inventory.csv")
        if row["Include In Comparison"] == "yes"
    ]
    long_rows: list[dict[str, Any]] = []
    for row in inventory:
        long_rows.extend(channel_rows(row))
    write_csv(OUT_ROOT / "metrics_long.csv", long_rows, LONG_FIELDS)
    string_rows = [{key: str(value) for key, value in row.items()} for row in long_rows]
    rank_rows = ranked(string_rows)
    degrade_rows = degradation(string_rows)
    contrasts = contrast_table(string_rows)
    write_csv(TABLES / "rankings.csv", rank_rows, RANK_FIELDS)
    write_csv(TABLES / "penetration_degradation.csv", degrade_rows, DEGRADE_FIELDS)
    write_csv(
        TABLES / "four_state_mechanism_gaps.csv",
        contrasts,
        [
            "model",
            "channel",
            "penetration",
            "s2s_mae",
            "a2s_mae",
            "a2a_mae",
            "fusion_gap_a2s_minus_s2s",
            "fusion_gap_percent",
            "multitask_gap_a2a_minus_a2s",
            "multitask_gap_percent",
            "confounded_gap_a2a_minus_s2s",
            "confounded_gap_percent",
        ],
    )
    metric_ylabel = {
        "mae": "Test MAE (vehicles / road / 5 min)",
        "rmse": "Test RMSE (vehicles / road / 5 min)",
        "wape": "Test WAPE (%)",
        "mape_nonzero": "Test MAPE on positive targets (%)",
    }
    for metric in ("mae", "rmse", "wape"):
        plot_curves(
            string_rows,
            family=FAMILY_LEGACY,
            task_type="single_channel_inflow",
            channel="inflow",
            metric=metric,
            path_name=f"legacy_single_channel/{metric}_vs_penetration.png",
            ylabel=metric_ylabel[metric],
        )
        plot_curves(
            string_rows,
            family=FAMILY_PHASE_SINGLE,
            task_type="single_channel_inflow",
            channel="inflow",
            metric=metric,
            path_name=f"phase3s_single_channel/{metric}_vs_penetration.png",
            ylabel=metric_ylabel[metric],
        )
        for state in STATES:
            plot_curves(
                string_rows,
                family=FAMILY_FOUR,
                task_type="single_to_single",
                channel=state,
                metric=metric,
                path_name=f"four_state/s2s_{metric}_{state}.png",
                ylabel=metric_ylabel[metric],
            )
    plot_degradation(
        degrade_rows,
        FAMILY_LEGACY,
        "single_channel_inflow",
        "inflow",
        "legacy_single_channel/relative_degradation_p70_to_p05.png",
    )
    plot_degradation(
        degrade_rows,
        FAMILY_PHASE_SINGLE,
        "single_channel_inflow",
        "inflow",
        "phase3s_single_channel/relative_degradation_p70_to_p05.png",
    )
    for state in STATES:
        plot_degradation(
            degrade_rows,
            FAMILY_FOUR,
            "single_to_single",
            state,
            f"four_state/s2s_relative_degradation_{state}.png",
        )
        plot_mechanism(string_rows, state)
    print(f"metrics_long={len(long_rows)} rankings={len(rank_rows)} contrasts={len(contrasts)}")


if __name__ == "__main__":
    main()
