"""Plot scale estimate variants and NMAE trends across penetration rates."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


SCALE_DIR = Path(__file__).resolve().parent
FLOW_DIR = SCALE_DIR.parent
SCALE_METRICS_DIR = SCALE_DIR / "metrics"
PLOT_DIR = SCALE_DIR / "plot"
FULL_FLOW_DIR = FLOW_DIR / "full_flow"

DEFAULT_JUNCTION_IDS = (
    "cluster_1262396634_1746662956",
    "cluster_1247897642_2350807770",
    "cluster_1262396675_2350807772",
    "cluster_3476413627_3476413628",
    "3476413738",
    "cluster_1746667327_1746667337",
    "1746667339",
    "1746667341",
    "cluster_3476413732_3476413733",
)
DEFAULT_SEED = 42
NMAE_METHOD = "scale+cap+smooth"
NMAE_PLOT_DIR = PLOT_DIR / "nmae_by_penetration"


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def penetration_sort_key(penetration: str) -> tuple[int, float | str]:
    try:
        return (0, float(penetration.replace("_", ".")))
    except ValueError:
        return (1, penetration)


def available_penetrations() -> list[str]:
    if not FULL_FLOW_DIR.exists():
        raise FileNotFoundError(f"full_flow dir not found: {FULL_FLOW_DIR}")

    penetrations = [
        path.name
        for path in FULL_FLOW_DIR.iterdir()
        if path.is_dir() and any(path.glob("*_edge_departures_by_cycle.json"))
    ]
    if not penetrations:
        raise FileNotFoundError(
            f"no penetration folders with departure JSON found under: {FULL_FLOW_DIR}"
        )

    return sorted(penetrations, key=penetration_sort_key)


def scale_metrics_file(penetration: str, seed: int, junction_id: str) -> Path:
    """Return combined scale metrics path."""
    return (
        SCALE_METRICS_DIR
        / penetration
        / f"{junction_id}_{seed}_scale_metrics.json"
    )


def output_file(
    penetration: str,
    seed: int,
    junction_id: str,
) -> Path:
    """Return output plot path."""
    return (
        PLOT_DIR
        / penetration
        / (
            f"{junction_id}_{penetration}_{seed}_"
            "all_phases_scale_estimates.png"
        )
    )


def nmae_output_file(junction_id: str, seed: int) -> Path:
    """Return output path for one junction's NMAE trend plot."""
    return NMAE_PLOT_DIR / f"{junction_id}_{seed}_{NMAE_METHOD}_nmae_by_penetration.png"


def reset_plot_dir(penetrations: list[str]) -> None:
    """Clear plot outputs and recreate one folder per penetration rate."""
    if PLOT_DIR.exists():
        shutil.rmtree(PLOT_DIR)
    for penetration in penetrations:
        (PLOT_DIR / penetration).mkdir(parents=True, exist_ok=True)
    NMAE_PLOT_DIR.mkdir(parents=True, exist_ok=True)


def sum_period_counts(entry: dict[str, Any], field_name: str) -> float:
    """Sum a field across all directions in periodCounts."""
    return sum(
        float(direction_counts.get(field_name, 0))
        for direction_counts in entry["periodCounts"].values()
    )


def method_counts_by_cycle(
    scale_metrics: dict[str, Any],
    method: str,
    field_name: str = "observedCount",
) -> dict[int, float]:
    """Read combined scale-metrics method counts keyed by cycle index."""
    counts_by_cycle: dict[int, float] = {}
    for entry in scale_metrics["methods"][method]["metricTimeline"]:
        cycle_index = int(entry["cycleIndex"])
        counts_by_cycle[cycle_index] = counts_by_cycle.get(
            cycle_index,
            0.0,
        ) + sum_period_counts(entry, field_name)
    return counts_by_cycle


def overall_nmae(scale_metrics: dict[str, Any], method: str = NMAE_METHOD) -> float | None:
    """Return overall NMAE for one method from a scale metrics JSON."""
    value = scale_metrics["methods"][method]["metrics"]["overall"].get("nmae")
    return None if value is None else float(value)


def metric_cycle_indices(scale_metrics: dict[str, Any]) -> list[int]:
    """Return the cycle range used by scale metrics."""
    metric_cycle_range = scale_metrics.get("metricCycleRange", {})
    min_cycle_index = metric_cycle_range.get("minCycleIndex")
    max_cycle_index = metric_cycle_range.get("maxCycleIndex")
    if min_cycle_index is not None and max_cycle_index is not None:
        return list(range(int(min_cycle_index), int(max_cycle_index) + 1))

    return sorted(
        {
            int(entry["cycleIndex"])
            for entry in scale_metrics["methods"][NMAE_METHOD]["metricTimeline"]
            if "cycleIndex" in entry
        }
    )


def plot_scale_estimates(
    scale_metrics: dict[str, Any],
    plot_file: Path,
    penetration: str,
    junction_id: str,
) -> None:
    """Plot scale estimate variants for one penetration rate."""
    actual_counts = method_counts_by_cycle(scale_metrics, "obsonly", "actualCount")
    obsonly_counts = method_counts_by_cycle(scale_metrics, "obsonly")
    scale_counts = method_counts_by_cycle(scale_metrics, "scale")
    scale_cap_counts = method_counts_by_cycle(scale_metrics, "scale+cap")
    scale_cap_smoothed_counts = method_counts_by_cycle(
        scale_metrics,
        "scale+cap+smooth",
    )

    cycle_indices = metric_cycle_indices(scale_metrics)

    plt.figure(figsize=(12, 6))
    plt.plot(
        cycle_indices,
        [actual_counts.get(cycle_index, 0) for cycle_index in cycle_indices],
        marker="o",
        linewidth=2.2,
        markersize=4,
        label="actual",
    )
    plt.plot(
        cycle_indices,
        [obsonly_counts.get(cycle_index, 0) for cycle_index in cycle_indices],
        marker="o",
        linewidth=1.6,
        markersize=3,
        label="obsonly",
    )
    plt.plot(
        cycle_indices,
        [scale_counts.get(cycle_index, 0) for cycle_index in cycle_indices],
        marker="o",
        linewidth=1.8,
        markersize=3,
        label="scale",
    )
    plt.plot(
        cycle_indices,
        [scale_cap_counts.get(cycle_index, 0) for cycle_index in cycle_indices],
        marker="o",
        linewidth=1.8,
        markersize=3,
        label="scale+cap",
    )
    plt.plot(
        cycle_indices,
        [
            scale_cap_smoothed_counts.get(cycle_index, 0)
            for cycle_index in cycle_indices
        ],
        marker="o",
        linestyle="--",
        linewidth=2.0,
        markersize=4,
        label="scale+cap+smooth",
    )

    plt.title(
        "All Phases Scale Estimate Variants - "
        f"{junction_id} ({penetration}%)"
    )
    plt.xlabel("Cycle index")
    plt.ylabel("Vehicle Count")
    plt.xticks(cycle_indices)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plot_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_file, dpi=200)
    plt.close()


def plot_nmae_by_penetration(
    nmae_points: list[tuple[str, float]],
    plot_file: Path,
    junction_id: str,
) -> None:
    """Plot scale+cap+smooth NMAE trend for one junction."""
    if not nmae_points:
        print(f"skipped NMAE plot with no data: {junction_id}")
        return

    ordered_points = sorted(nmae_points, key=lambda item: penetration_sort_key(item[0]))
    penetrations = [penetration for penetration, _nmae in ordered_points]
    x_values = [float(penetration.replace("_", ".")) for penetration in penetrations]
    y_values = [nmae for _penetration, nmae in ordered_points]

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_values,
        y_values,
        marker="o",
        linewidth=2.0,
        markersize=5,
        label=NMAE_METHOD,
    )
    plt.title(f"{junction_id} {NMAE_METHOD} NMAE by Penetration")
    plt.xlabel("Penetration Rate (%)")
    plt.ylabel("NMAE")
    plt.xticks(x_values, penetrations)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plot_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_file, dpi=200)
    plt.close()


def main() -> None:
    penetrations = available_penetrations()
    junction_ids = list(DEFAULT_JUNCTION_IDS)
    reset_plot_dir([str(penetration) for penetration in penetrations])
    nmae_by_junction: dict[str, list[tuple[str, float]]] = {
        junction_id: []
        for junction_id in junction_ids
    }

    for penetration in penetrations:
        for junction_id in junction_ids:
            metrics_file = scale_metrics_file(penetration, DEFAULT_SEED, junction_id)
            if not metrics_file.exists():
                print(f"skipped missing metrics: {metrics_file.resolve()}")
                continue

            scale_metrics = load_json(metrics_file)
            plot_file = output_file(
                penetration,
                DEFAULT_SEED,
                junction_id,
            )
            plot_scale_estimates(
                scale_metrics,
                plot_file,
                str(penetration),
                junction_id,
            )
            print(f"saved {penetration} {junction_id}: {plot_file.resolve()}")

            nmae = overall_nmae(scale_metrics)
            if nmae is not None:
                nmae_by_junction[junction_id].append((str(penetration), nmae))

    for junction_id, nmae_points in nmae_by_junction.items():
        plot_file = nmae_output_file(junction_id, DEFAULT_SEED)
        plot_nmae_by_penetration(nmae_points, plot_file, junction_id)
        if nmae_points:
            print(f"saved NMAE {junction_id}: {plot_file.resolve()}")


if __name__ == "__main__":
    main()
