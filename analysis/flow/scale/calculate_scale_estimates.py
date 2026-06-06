"""Calculate combined scale estimate metrics for selected junction phases."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from pathlib import Path
from typing import Any


SCALE_DIR = Path(__file__).resolve().parent
FLOW_DIR = SCALE_DIR.parent
ANALYSIS_DIR = FLOW_DIR.parent
SCALE_METRICS_DIR = SCALE_DIR / "metrics"
FULL_FLOW_DIR = FLOW_DIR / "full_flow"
TARGET_JUNCTION_IDS = (
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
DIRECTIONS = ("s", "l")
DEFAULT_SEED = 42
SMOOTH_WINDOW_SIZE = 3
METRIC_START_CYCLE_INDEX = 1
METRIC_END_TIME_SECONDS = 3800.0
METHOD_SOURCES = (
    ("obsonly", "obsonly"),
    ("scale", "scale"),
    ("scale+cap", "scale_upper_bound"),
    ("scale+cap+smooth", "scale_upper_bound_smooth"),
)

TLS_FILE = FLOW_DIR / "junction_tls.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate combined scale metrics for selected penetration rates "
            "and junction phases."
        )
    )
    parser.add_argument(
        "-p",
        "--penetration",
        default=None,
        help=(
            "Penetration-rate folder/name used by sample, departures, and "
            "metrics paths. If omitted, uses all full_flow folders that "
            "contain departure JSON files."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed used to generate the sample file (default: {DEFAULT_SEED}).",
    )
    return parser.parse_args()


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


def sample_file_for_penetration(penetration: str, seed: int) -> Path:
    return ANALYSIS_DIR / "simulation" / "penetration" / f"{penetration}_{seed}_sample.txt"


def departures_file_for_penetration(penetration: str, junction_id: str) -> Path:
    return FULL_FLOW_DIR / penetration / f"{junction_id}_edge_departures_by_cycle.json"


def scale_metrics_file_for_penetration(
    penetration: str, seed: int, junction_id: str
) -> Path:
    return (
        SCALE_METRICS_DIR
        / penetration
        / f"{junction_id}_{seed}_scale_metrics.json"
    )


def scale_metrics_summary_file_for_penetration(
    penetration: str, seed: int, junction_id: str
) -> Path:
    return (
        SCALE_METRICS_DIR
        / penetration
        / f"{junction_id}_{seed}_scale_metrics_summary.json"
    )


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def reset_scale_metrics_dir(penetrations: list[str]) -> None:
    if SCALE_METRICS_DIR.exists():
        shutil.rmtree(SCALE_METRICS_DIR)
    for penetration in penetrations:
        (SCALE_METRICS_DIR / penetration).mkdir(parents=True, exist_ok=True)


def load_sample_vehicle_ids(sample_file: Path) -> set[str]:
    if not sample_file.exists():
        raise FileNotFoundError(f"sample vehicle file not found: {sample_file}")
    return {
        line.strip()
        for line in sample_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def penetration_rate(penetration: str) -> float:
    value = float(penetration.rstrip("%"))
    if value <= 0:
        raise ValueError(f"penetration must be positive: {penetration}")
    return value / 100.0 if value > 1 else value


def find_junction_tls(tls_data: dict[str, Any], junction_id: str) -> dict[str, Any]:
    if tls_data.get("junctionId") == junction_id:
        return tls_data

    for junction in tls_data.get("junctions", []):
        if junction.get("junctionId") == junction_id:
            return junction

    raise ValueError(f"junction not found in TLS file: {junction_id}")


def yellow_phase_indexes(tls: dict[str, Any]) -> set[int]:
    indexes: set[int] = set()
    for phase in tls["phases"]:
        has_allowed_connections = bool(phase.get("allowedConnections"))
        state = str(phase.get("state", "")).lower()
        if not has_allowed_connections and "y" in state:
            indexes.add(int(phase["index"]))
    return indexes


def main_phase_indexes(tls: dict[str, Any]) -> list[int]:
    return [
        int(phase["index"])
        for phase in tls["phases"]
        if phase.get("allowedConnections")
    ]


def cycle_duration_from_tls(tls: dict[str, Any]) -> float:
    return sum(float(phase["duration"]) for phase in tls["phases"])


def max_metric_cycle_index(tls: dict[str, Any]) -> int:
    return int(METRIC_END_TIME_SECONDS // cycle_duration_from_tls(tls))


def count_direction(
    time_slice: dict[str, Any], direction: str, sample_vehicle_ids: set[str]
) -> tuple[int, int]:
    actual_count = 0
    observed_count = 0

    for movement in time_slice.get("movements", []):
        if movement.get("direction") != direction:
            continue

        actual_count += int(movement.get("count", 0))
        observed_count += sum(
            1
            for vehicle_id in movement.get("vehicleIds", [])
            if vehicle_id in sample_vehicle_ids
        )

    return observed_count, actual_count


def count_from_edge(
    time_slice: dict[str, Any], from_edge: str, sample_vehicle_ids: set[str]
) -> tuple[int, int]:
    actual_count = 0
    observed_count = 0

    for movement in time_slice.get("movements", []):
        if movement.get("fromEdge") != from_edge:
            continue

        actual_count += int(movement.get("count", 0))
        observed_count += sum(
            1
            for vehicle_id in movement.get("vehicleIds", [])
            if vehicle_id in sample_vehicle_ids
        )

    return observed_count, actual_count


def scaled_count(sampled_observed_count: int, rate: float) -> int:
    return math.floor(sampled_observed_count / rate + 0.5)


def lane_counts_for_direction(
    time_slice: dict[str, Any],
    direction: str,
    sample_vehicle_ids: set[str],
) -> list[dict[str, Any]]:
    counts_by_lane: dict[str, dict[str, Any]] = {}

    for movement in time_slice.get("movements", []):
        if movement.get("direction") != direction:
            continue

        from_lane = movement.get("fromLane")
        if not from_lane:
            continue

        lane_counts = counts_by_lane.setdefault(
            from_lane,
            {
                "fromLane": from_lane,
                "actualCount": 0,
                "sampledObservedCount": 0,
            },
        )
        lane_counts["actualCount"] += int(movement.get("count", 0))
        lane_counts["sampledObservedCount"] += sum(
            1
            for vehicle_id in movement.get("vehicleIds", [])
            if vehicle_id in sample_vehicle_ids
        )

    return list(counts_by_lane.values())


def lane_counts_for_from_edge(
    time_slice: dict[str, Any],
    from_edge: str,
    sample_vehicle_ids: set[str],
) -> list[dict[str, Any]]:
    counts_by_lane: dict[str, dict[str, Any]] = {}

    for movement in time_slice.get("movements", []):
        if movement.get("fromEdge") != from_edge:
            continue

        from_lane = movement.get("fromLane")
        if not from_lane:
            continue

        lane_counts = counts_by_lane.setdefault(
            from_lane,
            {
                "fromLane": from_lane,
                "actualCount": 0,
                "sampledObservedCount": 0,
            },
        )
        lane_counts["actualCount"] += int(movement.get("count", 0))
        lane_counts["sampledObservedCount"] += sum(
            1
            for vehicle_id in movement.get("vehicleIds", [])
            if vehicle_id in sample_vehicle_ids
        )

    return list(counts_by_lane.values())


def upper_bound_count(
    time_slice: dict[str, Any],
    direction: str,
    sample_vehicle_ids: set[str],
    rate: float,
) -> tuple[float, list[dict[str, Any]]]:
    per_lane_capacity = (float(time_slice["end"]) - float(time_slice["start"])) / 2.0
    lane_counts = lane_counts_for_direction(time_slice, direction, sample_vehicle_ids)
    lane_estimates: list[dict[str, Any]] = []

    for lane_count in lane_counts:
        scaled = scaled_count(lane_count["sampledObservedCount"], rate)
        capped = min(float(scaled), per_lane_capacity)
        lane_estimates.append(
            {
                "fromLane": lane_count["fromLane"],
                "actualCount": lane_count["actualCount"],
                "sampledObservedCount": lane_count["sampledObservedCount"],
                "scaledObservedCount": scaled,
                "perLaneCapacity": per_lane_capacity,
                "cappedObservedCount": capped,
            }
        )

    return sum(item["cappedObservedCount"] for item in lane_estimates), lane_estimates


def upper_bound_count_for_from_edge(
    time_slice: dict[str, Any],
    from_edge: str,
    sample_vehicle_ids: set[str],
    rate: float,
) -> tuple[float, list[dict[str, Any]]]:
    per_lane_capacity = (float(time_slice["end"]) - float(time_slice["start"])) / 2.0
    lane_counts = lane_counts_for_from_edge(time_slice, from_edge, sample_vehicle_ids)
    lane_estimates: list[dict[str, Any]] = []

    for lane_count in lane_counts:
        scaled = scaled_count(lane_count["sampledObservedCount"], rate)
        capped = min(float(scaled), per_lane_capacity)
        lane_estimates.append(
            {
                "fromLane": lane_count["fromLane"],
                "actualCount": lane_count["actualCount"],
                "sampledObservedCount": lane_count["sampledObservedCount"],
                "scaledObservedCount": scaled,
                "perLaneCapacity": per_lane_capacity,
                "cappedObservedCount": capped,
            }
        )

    return sum(item["cappedObservedCount"] for item in lane_estimates), lane_estimates


def build_row(
    method: str,
    cycle: dict[str, Any],
    time_slice: dict[str, Any],
    direction: str,
    sampled_observed_count: int,
    observed_count: float,
    actual_count: int,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    error = observed_count - actual_count
    row = {
        "method": method,
        "cycleIndex": cycle["cycleIndex"],
        "phaseIndex": int(time_slice["phaseIndex"]),
        "direction": direction,
        "start": time_slice["start"],
        "end": time_slice["end"],
        "sampledObservedCount": sampled_observed_count,
        "observedCount": observed_count,
        "actualCount": actual_count,
        "error": error,
        "absoluteError": abs(error),
        "squaredError": error * error,
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def build_edge_row(
    method: str,
    cycle: dict[str, Any],
    time_slice: dict[str, Any],
    from_edge: str,
    sampled_observed_count: int,
    observed_count: float,
    actual_count: int,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    error = observed_count - actual_count
    row = {
        "method": method,
        "cycleIndex": cycle["cycleIndex"],
        "phaseIndex": int(time_slice["phaseIndex"]),
        "fromEdge": from_edge,
        "start": time_slice["start"],
        "end": time_slice["end"],
        "sampledObservedCount": sampled_observed_count,
        "observedCount": observed_count,
        "actualCount": actual_count,
        "error": error,
        "absoluteError": abs(error),
        "squaredError": error * error,
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def smooth_rows(
    rows: list[dict[str, Any]],
    method: str,
    source_field: str = "observedCount",
) -> list[dict[str, Any]]:
    values_by_phase_direction: dict[tuple[int, str], list[tuple[int, float]]] = {}
    for row in rows:
        key = (int(row["phaseIndex"]), row["direction"])
        values_by_phase_direction.setdefault(key, []).append(
            (int(row["cycleIndex"]), float(row[source_field]))
        )

    radius = SMOOTH_WINDOW_SIZE // 2
    smoothed_values: dict[tuple[int, int, str], float] = {}
    for (phase_index, direction), values in values_by_phase_direction.items():
        values.sort(key=lambda item: item[0])
        for index, (cycle_index, _value) in enumerate(values):
            start = max(0, index - radius)
            end = min(len(values), index + radius + 1)
            window_values = [value for _cycle, value in values[start:end]]
            smoothed_values[(cycle_index, phase_index, direction)] = (
                sum(window_values) / len(window_values)
            )

    smoothed_rows: list[dict[str, Any]] = []
    for row in rows:
        cycle_index = int(row["cycleIndex"])
        phase_index = int(row["phaseIndex"])
        direction = row["direction"]
        observed_count = smoothed_values[(cycle_index, phase_index, direction)]
        smoothed_row = build_row(
            method,
            {
                "cycleIndex": row["cycleIndex"],
            },
            {
                "phaseIndex": row["phaseIndex"],
                "start": row["start"],
                "end": row["end"],
            },
            direction,
            row["sampledObservedCount"],
            observed_count,
            row["actualCount"],
            {
                "rawObservedCount": row[source_field],
                "smoothingWindowSize": SMOOTH_WINDOW_SIZE,
            },
        )
        smoothed_rows.append(smoothed_row)

    return smoothed_rows


def smooth_edge_rows(
    rows: list[dict[str, Any]],
    method: str,
    source_field: str = "observedCount",
) -> list[dict[str, Any]]:
    values_by_phase_edge: dict[tuple[int, str], list[tuple[int, float]]] = {}
    for row in rows:
        key = (int(row["phaseIndex"]), str(row["fromEdge"]))
        values_by_phase_edge.setdefault(key, []).append(
            (int(row["cycleIndex"]), float(row[source_field]))
        )

    radius = SMOOTH_WINDOW_SIZE // 2
    smoothed_values: dict[tuple[int, int, str], float] = {}
    for (phase_index, from_edge), values in values_by_phase_edge.items():
        values.sort(key=lambda item: item[0])
        for index, (cycle_index, _value) in enumerate(values):
            start = max(0, index - radius)
            end = min(len(values), index + radius + 1)
            window_values = [value for _cycle, value in values[start:end]]
            smoothed_values[(cycle_index, phase_index, from_edge)] = (
                sum(window_values) / len(window_values)
            )

    smoothed_rows: list[dict[str, Any]] = []
    for row in rows:
        cycle_index = int(row["cycleIndex"])
        phase_index = int(row["phaseIndex"])
        from_edge = str(row["fromEdge"])
        observed_count = smoothed_values[(cycle_index, phase_index, from_edge)]
        smoothed_rows.append(
            build_edge_row(
                method,
                {
                    "cycleIndex": row["cycleIndex"],
                },
                {
                    "phaseIndex": row["phaseIndex"],
                    "start": row["start"],
                    "end": row["end"],
                },
                from_edge,
                row["sampledObservedCount"],
                observed_count,
                row["actualCount"],
                {
                    "rawObservedCount": row[source_field],
                    "smoothingWindowSize": SMOOTH_WINDOW_SIZE,
                },
            )
        )

    return smoothed_rows


def build_method_rows(
    departures: dict[str, Any],
    sample_vehicle_ids: set[str],
    yellow_indexes: set[int],
    rate: float,
) -> dict[str, list[dict[str, Any]]]:
    rows_by_method: dict[str, list[dict[str, Any]]] = {
        "obsonly": [],
        "scale": [],
        "scale_upper_bound": [],
    }

    for cycle in departures["cycles"]:
        for time_slice in cycle["timeSlices"]:
            phase_index = int(time_slice["phaseIndex"])
            if phase_index in yellow_indexes:
                continue

            for direction in DIRECTIONS:
                sampled_observed_count, actual_count = count_direction(
                    time_slice, direction, sample_vehicle_ids
                )
                rows_by_method["obsonly"].append(
                    build_row(
                        "obsonly",
                        cycle,
                        time_slice,
                        direction,
                        sampled_observed_count,
                        sampled_observed_count,
                        actual_count,
                    )
                )
                rows_by_method["scale"].append(
                    build_row(
                        "scale",
                        cycle,
                        time_slice,
                        direction,
                        sampled_observed_count,
                        scaled_count(sampled_observed_count, rate),
                        actual_count,
                    )
                )
                capped_count, lane_estimates = upper_bound_count(
                    time_slice,
                    direction,
                    sample_vehicle_ids,
                    rate,
                )
                rows_by_method["scale_upper_bound"].append(
                    build_row(
                        "scale_upper_bound",
                        cycle,
                        time_slice,
                        direction,
                        sampled_observed_count,
                        capped_count,
                        actual_count,
                        {
                            "laneEstimates": lane_estimates,
                        },
                    )
                )

    rows_by_method["scale_upper_bound_smooth"] = smooth_rows(
        rows_by_method["scale_upper_bound"],
        "scale_upper_bound_smooth",
    )
    return rows_by_method


def incoming_edges_from_departures(departures: dict[str, Any]) -> list[str]:
    if "incomingEdges" in departures:
        return [str(edge) for edge in departures["incomingEdges"]]

    edges = {
        str(movement["fromEdge"])
        for cycle in departures["cycles"]
        for time_slice in cycle["timeSlices"]
        for movement in time_slice.get("movements", [])
        if movement.get("fromEdge")
    }
    return sorted(edges)


def build_edge_method_rows(
    departures: dict[str, Any],
    sample_vehicle_ids: set[str],
    yellow_indexes: set[int],
    rate: float,
) -> dict[str, list[dict[str, Any]]]:
    rows_by_method: dict[str, list[dict[str, Any]]] = {
        "obsonly": [],
        "scale": [],
        "scale_upper_bound": [],
    }
    incoming_edges = incoming_edges_from_departures(departures)

    for cycle in departures["cycles"]:
        for time_slice in cycle["timeSlices"]:
            phase_index = int(time_slice["phaseIndex"])
            if phase_index in yellow_indexes:
                continue

            for from_edge in incoming_edges:
                sampled_observed_count, actual_count = count_from_edge(
                    time_slice,
                    from_edge,
                    sample_vehicle_ids,
                )
                rows_by_method["obsonly"].append(
                    build_edge_row(
                        "obsonly",
                        cycle,
                        time_slice,
                        from_edge,
                        sampled_observed_count,
                        sampled_observed_count,
                        actual_count,
                    )
                )
                rows_by_method["scale"].append(
                    build_edge_row(
                        "scale",
                        cycle,
                        time_slice,
                        from_edge,
                        sampled_observed_count,
                        scaled_count(sampled_observed_count, rate),
                        actual_count,
                    )
                )
                capped_count, lane_estimates = upper_bound_count_for_from_edge(
                    time_slice,
                    from_edge,
                    sample_vehicle_ids,
                    rate,
                )
                rows_by_method["scale_upper_bound"].append(
                    build_edge_row(
                        "scale_upper_bound",
                        cycle,
                        time_slice,
                        from_edge,
                        sampled_observed_count,
                        capped_count,
                        actual_count,
                        {
                            "laneEstimates": lane_estimates,
                        },
                    )
                )

    rows_by_method["scale_upper_bound_smooth"] = smooth_edge_rows(
        rows_by_method["scale_upper_bound"],
        "scale_upper_bound_smooth",
    )
    return rows_by_method


def summarize_sample_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "periodCount": 0,
            "mae": None,
            "rmse": None,
            "observedTotal": 0,
            "actualTotal": 0,
        }

    return {
        "periodCount": len(rows),
        "mae": sum(item["absoluteError"] for item in rows) / len(rows),
        "rmse": math.sqrt(sum(item["squaredError"] for item in rows) / len(rows)),
        "observedTotal": sum(item["observedCount"] for item in rows),
        "actualTotal": sum(item["actualCount"] for item in rows),
    }


def calculate_nmae(rows: list[dict[str, Any]]) -> float | None:
    actual_total = sum(item["actualCount"] for item in rows)
    if actual_total == 0:
        return None
    return sum(item["absoluteError"] for item in rows) / actual_total


def calculate_sample_metrics(
    rows: list[dict[str, Any]], include_overall_nmae: bool = False
) -> dict[str, Any]:
    by_direction = {
        direction: summarize_sample_rows(
            [item for item in rows if item["direction"] == direction]
        )
        for direction in DIRECTIONS
    }
    for direction in DIRECTIONS:
        direction_rows = [item for item in rows if item["direction"] == direction]
        by_direction[direction]["nmae"] = calculate_nmae(direction_rows)

    metrics = {
        "overall": summarize_sample_rows(rows),
        "byDirection": by_direction,
    }
    if include_overall_nmae:
        metrics["overall"]["nmae"] = calculate_nmae(rows)
    return metrics


def empty_estimate_summary() -> dict[str, Any]:
    return {
        "periodCount": 0,
        "mae": None,
        "mse": None,
        "rmse": None,
        "estimatedTotal": 0.0,
        "actualTotal": 0.0,
        "nmae": None,
    }


def summarize_estimate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return empty_estimate_summary()

    period_count = len(rows)
    absolute_error_total = sum(float(item["absoluteError"]) for item in rows)
    squared_error_total = sum(float(item["squaredError"]) for item in rows)
    estimated_total = sum(float(item["observedCount"]) for item in rows)
    actual_total = sum(float(item["actualCount"]) for item in rows)
    mse = squared_error_total / period_count
    return {
        "periodCount": period_count,
        "mae": absolute_error_total / period_count,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "estimatedTotal": estimated_total,
        "actualTotal": actual_total,
        "nmae": absolute_error_total / actual_total if actual_total else None,
    }


def calculate_estimate_metrics(
    rows: list[dict[str, Any]], phase_indexes: list[int]
) -> dict[str, Any]:
    return {
        "overall": summarize_estimate_rows(rows),
        "byDirection": {
            direction: summarize_estimate_rows(
                [item for item in rows if item["direction"] == direction]
            )
            for direction in DIRECTIONS
        },
        "byPhase": {
            f"phase{phase_index}": {
                "overall": summarize_estimate_rows(
                    [
                        item
                        for item in rows
                        if int(item["phaseIndex"]) == phase_index
                    ]
                ),
                "byDirection": {
                    direction: summarize_estimate_rows(
                        [
                            item
                            for item in rows
                            if int(item["phaseIndex"]) == phase_index
                            and item["direction"] == direction
                        ]
                    )
                    for direction in DIRECTIONS
                },
            }
            for phase_index in phase_indexes
        },
    }


def calculate_edge_estimate_metrics(
    rows: list[dict[str, Any]],
    phase_indexes: list[int],
) -> dict[str, Any]:
    from_edges = sorted({str(item["fromEdge"]) for item in rows})
    return {
        "overall": summarize_estimate_rows(rows),
        "byFromEdge": {
            from_edge: summarize_estimate_rows(
                [item for item in rows if str(item["fromEdge"]) == from_edge]
            )
            for from_edge in from_edges
        },
        "byPhase": {
            f"phase{phase_index}": {
                "overall": summarize_estimate_rows(
                    [
                        item
                        for item in rows
                        if int(item["phaseIndex"]) == phase_index
                    ]
                ),
                "byFromEdge": {
                    from_edge: summarize_estimate_rows(
                        [
                            item
                            for item in rows
                            if int(item["phaseIndex"]) == phase_index
                            and str(item["fromEdge"]) == from_edge
                        ]
                    )
                    for from_edge in from_edges
                },
            }
            for phase_index in phase_indexes
        },
    }


def direction_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}
    for direction in DIRECTIONS:
        direction_rows = [item for item in rows if item["direction"] == direction]
        counts[direction] = {
            "observedCount": sum(item["observedCount"] for item in direction_rows),
            "actualCount": sum(item["actualCount"] for item in direction_rows),
            "error": sum(item["error"] for item in direction_rows),
            "absoluteError": sum(item["absoluteError"] for item in direction_rows),
            "squaredError": sum(item["squaredError"] for item in direction_rows),
        }
    return counts


def edge_counts(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "observedCount": row["observedCount"],
        "actualCount": row["actualCount"],
        "error": row["error"],
        "absoluteError": row["absoluteError"],
        "squaredError": row["squaredError"],
    }


def build_sample_timeline_entry(
    period_rows: list[dict[str, Any]], cumulative_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    first = period_rows[0]
    return {
        "cycleIndex": first["cycleIndex"],
        "phaseIndex": first["phaseIndex"],
        "start": first["start"],
        "end": first["end"],
        "periodCounts": direction_counts(period_rows),
        "cumulativeMetrics": calculate_sample_metrics(cumulative_rows),
    }


def build_sample_metric_timeline(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    cumulative_rows: list[dict[str, Any]] = []
    period_rows: list[dict[str, Any]] = []
    current_key: tuple[int, int, float, float] | None = None

    for row in rows:
        key = (
            int(row["cycleIndex"]),
            int(row["phaseIndex"]),
            float(row["start"]),
            float(row["end"]),
        )
        if current_key is not None and key != current_key:
            cumulative_rows.extend(period_rows)
            timeline.append(build_sample_timeline_entry(period_rows, cumulative_rows))
            period_rows = []

        current_key = key
        period_rows.append(row)

    if period_rows:
        cumulative_rows.extend(period_rows)
        timeline.append(build_sample_timeline_entry(period_rows, cumulative_rows))

    return timeline


def build_edge_metric_timeline(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            int(row["cycleIndex"]),
            int(row["phaseIndex"]),
            str(row["fromEdge"]),
        ),
    )
    for row in sorted_rows:
        entry = {
            "cycleIndex": row["cycleIndex"],
            "phaseIndex": row["phaseIndex"],
            "fromEdge": row["fromEdge"],
            "start": row["start"],
            "end": row["end"],
            **edge_counts(row),
        }
        if "rawObservedCount" in row:
            entry["rawObservedCount"] = row["rawObservedCount"]
        if "smoothingWindowSize" in row:
            entry["smoothingWindowSize"] = row["smoothingWindowSize"]
        timeline.append(entry)
    return timeline


def filtered_phase_rows(
    rows: list[dict[str, Any]],
    phase_indexes: list[int],
) -> list[dict[str, Any]]:
    phase_index_set = set(phase_indexes)
    return [
        row
        for row in rows
        if int(row["phaseIndex"]) in phase_index_set
    ]


def metric_rows(
    rows: list[dict[str, Any]],
    max_cycle_index: int,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if METRIC_START_CYCLE_INDEX <= int(row["cycleIndex"]) <= max_cycle_index
    ]


def cycle_range(rows: list[dict[str, Any]]) -> dict[str, int | None]:
    cycle_indices = sorted({int(row["cycleIndex"]) for row in rows})
    if not cycle_indices:
        return {
            "minCycleIndex": None,
            "maxCycleIndex": None,
            "cycleCount": 0,
        }
    return {
        "minCycleIndex": cycle_indices[0],
        "maxCycleIndex": cycle_indices[-1],
        "cycleCount": len(cycle_indices),
    }


def rows_with_method_name(
    rows: list[dict[str, Any]],
    method_name: str,
) -> list[dict[str, Any]]:
    renamed_rows = []
    for row in rows:
        renamed_row = copy.deepcopy(row)
        renamed_row["method"] = method_name
        renamed_rows.append(renamed_row)
    return renamed_rows


def build_combined_method_result(
    rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    phase_indexes: list[int],
    edge_rows: list[dict[str, Any]] | None = None,
    edge_metric_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result = {
        "metrics": calculate_estimate_metrics(metric_rows, phase_indexes),
        "metricTimeline": build_sample_metric_timeline(rows),
        "rows": rows,
    }
    if edge_rows is not None and edge_metric_rows is not None:
        result["edgeMetrics"] = calculate_edge_estimate_metrics(
            edge_metric_rows,
            phase_indexes,
        )
        result["edgeMetricTimeline"] = build_edge_metric_timeline(edge_rows)
        result["edgeRows"] = edge_rows
    return result


def build_scale_metrics_output(
    penetration: str,
    seed: int,
    junction_id: str,
    tls: dict[str, Any],
    sample_vehicle_ids: set[str],
) -> tuple[dict[str, Any], Path]:
    sample_file = sample_file_for_penetration(penetration, seed)
    departures_file = departures_file_for_penetration(penetration, junction_id)
    output_file = scale_metrics_file_for_penetration(penetration, seed, junction_id)
    rate = penetration_rate(penetration)
    departures = load_json(departures_file)
    phase_indexes = main_phase_indexes(tls)
    metric_max_cycle_index = max_metric_cycle_index(tls)

    rows_by_method = build_method_rows(
        departures,
        sample_vehicle_ids,
        yellow_phase_indexes(tls),
        rate,
    )
    edge_rows_by_method = build_edge_method_rows(
        departures,
        sample_vehicle_ids,
        yellow_phase_indexes(tls),
        rate,
    )
    combined_methods = {}
    data_cycle_range: dict[str, int | None] | None = None
    metric_cycle_range: dict[str, int | None] | None = None
    for method_name, source_method in METHOD_SOURCES:
        rows = filtered_phase_rows(rows_by_method[source_method], phase_indexes)
        method_metric_rows = metric_rows(rows, metric_max_cycle_index)
        edge_rows = filtered_phase_rows(
            edge_rows_by_method[source_method],
            phase_indexes,
        )
        edge_metric_rows = metric_rows(edge_rows, metric_max_cycle_index)
        if data_cycle_range is None:
            data_cycle_range = cycle_range(rows)
        if metric_cycle_range is None:
            metric_cycle_range = cycle_range(method_metric_rows)
        combined_methods[method_name] = build_combined_method_result(
            rows_with_method_name(rows, method_name),
            rows_with_method_name(method_metric_rows, method_name),
            phase_indexes,
            rows_with_method_name(edge_rows, method_name),
            rows_with_method_name(edge_metric_rows, method_name),
        )

    output = {
        "junctionId": junction_id,
        "penetration": penetration,
        "seed": seed,
        "penetrationRate": rate,
        "sampleFile": str(sample_file.resolve()),
        "departuresFile": str(departures_file.resolve()),
        "directions": list(DIRECTIONS),
        "phaseIndexes": phase_indexes,
        "cycleRange": data_cycle_range,
        "metricCycleRange": metric_cycle_range,
        "metricEndTimeSeconds": METRIC_END_TIME_SECONDS,
        "metricMaxCycleIndex": metric_max_cycle_index,
        "methodOrder": [method_name for method_name, _source in METHOD_SOURCES],
        "methods": combined_methods,
    }
    return output, output_file


def build_scale_metrics_summary_output(scale_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "junctionId": scale_metrics["junctionId"],
        "penetration": scale_metrics["penetration"],
        "seed": scale_metrics["seed"],
        "penetrationRate": scale_metrics["penetrationRate"],
        "directions": scale_metrics["directions"],
        "phaseIndexes": scale_metrics["phaseIndexes"],
        "cycleRange": scale_metrics["cycleRange"],
        "metricCycleRange": scale_metrics["metricCycleRange"],
        "metricEndTimeSeconds": scale_metrics["metricEndTimeSeconds"],
        "metricMaxCycleIndex": scale_metrics["metricMaxCycleIndex"],
        "methodOrder": scale_metrics["methodOrder"],
        "metrics": {
            method: scale_metrics["methods"][method]["metrics"]
            for method in scale_metrics["methodOrder"]
        },
        "edgeMetrics": {
            method: scale_metrics["methods"][method]["edgeMetrics"]
            for method in scale_metrics["methodOrder"]
            if "edgeMetrics" in scale_metrics["methods"][method]
        },
    }


def process_junction(
    penetration: str,
    seed: int,
    junction_id: str,
    tls_data: dict[str, Any],
    sample_vehicle_ids: set[str],
) -> None:
    print(f"processing junction: {junction_id}")
    tls = find_junction_tls(tls_data, junction_id)
    scale_metrics, scale_metrics_file = build_scale_metrics_output(
        penetration,
        seed,
        junction_id,
        tls,
        sample_vehicle_ids,
    )
    write_json(scale_metrics_file, scale_metrics)
    print(f"saved: {scale_metrics_file.resolve()}")
    summary_file = scale_metrics_summary_file_for_penetration(
        penetration,
        seed,
        junction_id,
    )
    write_json(summary_file, build_scale_metrics_summary_output(scale_metrics))
    print(f"saved: {summary_file.resolve()}")
    print(
        json.dumps(
            build_scale_metrics_summary_output(scale_metrics)["metrics"],
            ensure_ascii=False,
            indent=2,
        )
    )


def process_penetration(
    penetration: str,
    seed: int,
    tls_data: dict[str, Any],
    junction_ids: list[str] | tuple[str, ...],
) -> None:
    sample_file = sample_file_for_penetration(penetration, seed)
    sample_vehicle_ids = load_sample_vehicle_ids(sample_file)

    for junction_id in junction_ids:
        departures_file = departures_file_for_penetration(penetration, junction_id)
        if not departures_file.exists():
            print(f"skipped missing departures: {departures_file.resolve()}")
            continue

        process_junction(
            penetration,
            seed,
            junction_id,
            tls_data,
            sample_vehicle_ids,
        )


def process_generated_metrics(args: argparse.Namespace) -> None:
    tls_data = load_json(TLS_FILE)
    penetrations = [args.penetration] if args.penetration else available_penetrations()
    reset_scale_metrics_dir(penetrations)

    for penetration in penetrations:
        print(f"processing penetration: {penetration}")
        process_penetration(penetration, args.seed, tls_data, TARGET_JUNCTION_IDS)


def main() -> None:
    args = parse_args()
    process_generated_metrics(args)


if __name__ == "__main__":
    main()
