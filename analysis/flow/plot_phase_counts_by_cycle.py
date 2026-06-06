from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


FLOW_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = FLOW_DIR.parent
METRICS_DIR = ANALYSIS_DIR / "metrics"
SCALE_METRICS_DIR = FLOW_DIR / "scale" / "metrics"
PLOT_DIR = FLOW_DIR / "plot"
SEED = 42
TARGET_PHASE_INDEXES = (0, 2, 4)
DEFAULT_PENETRATIONS = ("5", "10", "20", "30", "40", "50")
DIRECTION_LABELS = {
    "s": "straight",
    "l": "left",
}
PHASE_DIRECTION_SERIES = (
    (None, "overall"),
    ("l", "left"),
    ("s", "straight"),
)
TARGET_JUNCTION_IDS = (
    "cluster_1262396634_1746662956",
    "cluster_1247897642_2350807770",
    "cluster_1262396675_2350807772",
    "cluster_3476413627_3476413628",
    "cluster_1746667327_1746667337",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot vehicle counts by cycle for each phaseIndex in an "
            "edge_departures_by_cycle JSON file."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=(
            "Input JSON files. If omitted, plots 5, 10, 20, 30, 40, and 50 "
            "for TARGET_JUNCTION_IDS."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output image path. Defaults to "
            "analysis/flow/plot/<penetration>/<input_stem>_phase_counts_by_cycle.png."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure window after saving.",
    )
    return parser.parse_args()


def load_departure_data(input_file: Path) -> dict[str, Any]:
    if not input_file.exists():
        raise FileNotFoundError(f"input JSON not found: {input_file}")
    return json.loads(input_file.read_text(encoding="utf-8"))


def default_input_file(penetration: str, junction_id: str) -> Path:
    return FLOW_DIR / penetration / f"{junction_id}_edge_departures_by_cycle.json"


def input_files_from_args(args: argparse.Namespace) -> list[Path]:
    if args.inputs:
        return args.inputs
    return [
        default_input_file(penetration, junction_id)
        for penetration in DEFAULT_PENETRATIONS
        for junction_id in TARGET_JUNCTION_IDS
    ]


def clear_plot_dir() -> None:
    if not PLOT_DIR.exists():
        return

    for path in PLOT_DIR.iterdir():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def load_json_file(input_file: Path) -> dict[str, Any]:
    if not input_file.exists():
        raise FileNotFoundError(f"JSON file not found: {input_file}")
    return json.loads(input_file.read_text(encoding="utf-8"))


def penetration_from_input(input_file: Path) -> str:
    return input_file.parent.name


def metric_file_for_input(input_file: Path, suffix: str) -> Path:
    penetration = penetration_from_input(input_file)
    junction_id = input_file.stem.removesuffix("_edge_departures_by_cycle")
    metrics_dir = METRICS_DIR / penetration
    candidates = [
        metrics_dir / f"{junction_id}_{SEED}_{suffix}.json",
        metrics_dir / f"{junction_id}_{suffix}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "metric JSON not found, tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def scale_metrics_file_for_input(input_file: Path) -> Path:
    penetration = penetration_from_input(input_file)
    junction_id = input_file.stem.removesuffix("_edge_departures_by_cycle")
    metrics_dir = SCALE_METRICS_DIR / penetration
    candidates = [
        metrics_dir / f"{junction_id}_{SEED}_scale_metrics.json",
        metrics_dir / f"{junction_id}_scale_metrics.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "scale metrics JSON not found, tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def phase_indices(data: dict[str, Any]) -> list[int]:
    if data.get("phaseTimeSlices"):
        return sorted({int(item["phaseIndex"]) for item in data["phaseTimeSlices"]})

    phases: set[int] = set()
    for cycle in data["cycles"]:
        for time_slice in cycle["timeSlices"]:
            phases.add(int(time_slice["phaseIndex"]))
    return sorted(phases)


def counts_by_phase(data: dict[str, Any]) -> tuple[list[int], dict[int, list[int]]]:
    phases = phase_indices(data)
    cycle_indices = [int(cycle["cycleIndex"]) for cycle in data["cycles"]]
    counts = {phase_index: [] for phase_index in phases}

    for cycle in data["cycles"]:
        count_by_phase = {
            int(time_slice["phaseIndex"]): int(time_slice["count"])
            for time_slice in cycle["timeSlices"]
        }
        for phase_index in phases:
            counts[phase_index].append(count_by_phase.get(phase_index, 0))

    return cycle_indices, counts


def default_output_file(input_file: Path) -> Path:
    penetration = penetration_from_input(input_file)
    return (
        PLOT_DIR
        / penetration
        / f"{input_file.stem}_phase_counts_by_cycle.png"
    )


def phase_output_file(
    input_file: Path, phase_index: int, direction: str | None = None
) -> Path:
    penetration = penetration_from_input(input_file)
    direction_part = (
        DIRECTION_LABELS.get(direction, direction)
        if direction is not None
        else "overall"
    )
    return (
        PLOT_DIR
        / penetration
        / f"{input_file.stem}_phase{phase_index}_{direction_part}_counts_by_cycle.png"
    )


def plot_phase_counts(
    data: dict[str, Any],
    output_file: Path,
) -> None:
    cycle_indices, counts = counts_by_phase(data)
    junction_id = data.get("junctionId", "unknown junction")

    plt.figure(figsize=(12, 6))
    for phase_index, phase_counts in counts.items():
        plt.plot(
            cycle_indices,
            phase_counts,
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=f"phaseIndex {phase_index}",
        )

    plt.title(f"Vehicle Counts by Cycle - overall - {junction_id}")
    plt.xlabel("Cycle Index")
    plt.ylabel("Vehicle Count")
    plt.xticks(cycle_indices)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200)
    plt.close()


def sum_period_counts(entry: dict[str, Any], field_name: str) -> float:
    return sum(
        float(direction_counts.get(field_name, 0))
        for direction_counts in entry["periodCounts"].values()
    )


def timeline_counts_for_phase(
    metric_data: dict[str, Any],
    phase_index: int,
    field_name: str,
    *,
    method: str | None = None,
    direction: str | None = None,
) -> dict[int, float]:
    timeline_parent = metric_data["methods"][method] if method else metric_data
    counts_by_cycle: dict[int, float] = {}

    for entry in timeline_parent["metricTimeline"]:
        if int(entry["phaseIndex"]) != phase_index:
            continue
        cycle_index = int(entry["cycleIndex"])
        if direction is None:
            counts_by_cycle[cycle_index] = sum_period_counts(entry, field_name)
        else:
            counts_by_cycle[cycle_index] = float(
                entry["periodCounts"].get(direction, {}).get(field_name, 0)
            )

    return counts_by_cycle


def plot_observed_vs_total_phase_counts(
    data: dict[str, Any],
    total_counts_by_cycle: dict[int, float],
    observed_counts_by_cycle: dict[int, float],
    scale_counts_by_cycle: dict[int, float],
    capped_scale_counts_by_cycle: dict[int, float],
    capped_smoothed_scale_counts_by_cycle: dict[int, float],
    queue_counts_by_cycle: dict[int, float],
    output_file: Path,
    phase_index: int,
    direction: str | None = None,
) -> None:
    cycle_indices = [int(cycle["cycleIndex"]) for cycle in data["cycles"]]
    junction_id = data.get("junctionId", "unknown junction")
    direction_label = (
        DIRECTION_LABELS.get(direction, direction)
        if direction is not None
        else "overall"
    )

    plt.figure(figsize=(12, 6))
    plt.plot(
        cycle_indices,
        [total_counts_by_cycle.get(cycle_index, 0) for cycle_index in cycle_indices],
        marker="o",
        linewidth=1.8,
        markersize=3,
        label="total",
    )
    plt.plot(
        cycle_indices,
        [observed_counts_by_cycle.get(cycle_index, 0) for cycle_index in cycle_indices],
        marker="o",
        linewidth=1.8,
        markersize=3,
        label="observed",
    )
    plt.plot(
        cycle_indices,
        [scale_counts_by_cycle.get(cycle_index, 0) for cycle_index in cycle_indices],
        marker="o",
        linewidth=1.8,
        markersize=3,
        label="scale",
    )
    plt.plot(
        cycle_indices,
        [
            capped_scale_counts_by_cycle.get(cycle_index, 0)
            for cycle_index in cycle_indices
        ],
        marker="o",
        linewidth=1.8,
        markersize=3,
        label="scale capped",
    )
    plt.plot(
        cycle_indices,
        [
            capped_smoothed_scale_counts_by_cycle.get(cycle_index, 0)
            for cycle_index in cycle_indices
        ],
        marker="o",
        linewidth=1.8,
        markersize=3,
        label="scale capped + smoothed",
    )
    plt.plot(
        cycle_indices,
        [queue_counts_by_cycle.get(cycle_index, 0) for cycle_index in cycle_indices],
        marker="o",
        linewidth=1.8,
        markersize=3,
        label="queue evidence",
    )

    plt.title(
        "Observed vs Total Vehicle Counts - "
        f"phaseIndex {phase_index} - {direction_label} - {junction_id}"
    )
    plt.xlabel("Cycle Index")
    plt.ylabel("Vehicle Count")
    plt.xticks(cycle_indices)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=200)
    plt.close()


def plot_input_file(input_file: Path, output_file: Path | None = None) -> None:
    output_file = output_file or default_output_file(input_file)

    data = load_departure_data(input_file)
    plot_phase_counts(data, output_file)
    print(f"saved: {output_file.resolve()}")

    scale_metrics = load_json_file(scale_metrics_file_for_input(input_file))
    queue_metrics = load_json_file(metric_file_for_input(input_file, "queue_estimates"))

    for phase_index in TARGET_PHASE_INDEXES:
        for direction, _label in PHASE_DIRECTION_SERIES:
            phase_counts_file = phase_output_file(input_file, phase_index, direction)
            total_counts_by_cycle = timeline_counts_for_phase(
                scale_metrics,
                phase_index,
                "actualCount",
                method="obsonly",
                direction=direction,
            )
            observed_counts_by_cycle = timeline_counts_for_phase(
                scale_metrics,
                phase_index,
                "observedCount",
                method="obsonly",
                direction=direction,
            )
            scale_counts_by_cycle = timeline_counts_for_phase(
                scale_metrics,
                phase_index,
                "observedCount",
                method="scale",
                direction=direction,
            )
            queue_counts_by_cycle = timeline_counts_for_phase(
                queue_metrics,
                phase_index,
                "estimatedCount",
                direction=direction,
            )
            smoothed_scale_counts_by_cycle = timeline_counts_for_phase(
                scale_metrics,
                phase_index,
                "observedCount",
                method="scale+cap+smooth",
                direction=direction,
            )
            capped_scale_counts_by_cycle = timeline_counts_for_phase(
                scale_metrics,
                phase_index,
                "observedCount",
                method="scale+cap",
                direction=direction,
            )
            plot_observed_vs_total_phase_counts(
                data,
                total_counts_by_cycle,
                observed_counts_by_cycle,
                scale_counts_by_cycle,
                capped_scale_counts_by_cycle,
                smoothed_scale_counts_by_cycle,
                queue_counts_by_cycle,
                phase_counts_file,
                phase_index,
                direction,
            )
            print(f"saved: {phase_counts_file.resolve()}")


def main() -> None:
    args = parse_args()
    input_files = input_files_from_args(args)
    if args.output is not None and len(input_files) != 1:
        raise ValueError("--output can only be used when plotting one input file.")

    clear_plot_dir()

    for input_file in input_files:
        print(f"plotting: {input_file}")
        plot_input_file(input_file, args.output)

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
