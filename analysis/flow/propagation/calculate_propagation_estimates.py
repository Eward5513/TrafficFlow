"""Calculate traffic flow propagation estimates.

This script propagates configured upstream links to their relevant downstream
service windows. It filters upstream movements to the first edge on each
propagated path and uses all upstream phases that can enter that edge, but
intentionally does not implement full network propagation or downstream turning
splits.
"""

from __future__ import annotations

import json
import math
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import pandas as pd


PROPAGATION_DIR = Path(__file__).resolve().parent
FLOW_DIR = PROPAGATION_DIR.parent
ANALYSIS_DIR = FLOW_DIR.parent
SCALE_METRICS_DIR = FLOW_DIR / "scale" / "metrics"
PROPAGATION_METRICS_DIR = PROPAGATION_DIR / "metrics"
FULL_FLOW_DIR = FLOW_DIR / "full_flow"
TLS_FILE = FLOW_DIR / "junction_tls.json"
ROAD_NETWORK_NET_FILE = ANALYSIS_DIR / "road_network" / "net_tls.net.xml"
DEFAULT_SEED = 42
DEFAULT_ESTIMATE_FIELD = "cappedObservedCount"
SMOOTH_WINDOW_SIZE = 3
METRIC_START_DOWN_CYCLE_INDEX = 1
METRIC_END_TIME_SECONDS = 3800.0
DEFAULT_ZERO_FILTERED_UPSTREAM_WEIGHT = 0.5
WEIGHTED_AVERAGE_COEFFICIENT_STEP = 0.1
ESTIMATE_FIELDS: tuple[tuple[str, str], ...] = (
    ("upstreamEstimate", "upstream_estimate"),
    ("smoothedUpstreamEstimate", "smoothed_upstream_estimate"),
    (
        "zeroFilteredSmoothedUpstreamEstimate",
        "zero_filtered_smoothed_upstream_estimate",
    ),
    (
        "zeroFilteredScaleSmoothedAverageEstimate",
        "zero_filtered_scale_smoothed_average_estimate",
    ),
)
UPSTREAM_REQUIRED_COLUMNS = {
    "junction_id",
    "phase_id",
    "window_start",
    "window_end",
    "estimate",
}


@dataclass(frozen=True)
class EdgeSegment:
    """One road segment used by a propagation link."""

    edge_id: str
    length_meters: float
    speed_limit_mps: float


@dataclass(frozen=True)
class PropagationLink:
    """Configuration for one upstream-to-downstream propagation calculation."""

    upstream_junction_id: str
    downstream_junction_id: str
    path_edges: tuple[EdgeSegment, ...]

    @property
    def upstream_estimate_edge_id(self) -> str:
        """Return the first edge, used to filter upstream departures."""
        return self.path_edges[0].edge_id

    @property
    def downstream_incoming_edge_id(self) -> str:
        """Return the final edge entering the downstream junction."""
        return self.path_edges[-1].edge_id

    @property
    def edge_length_meters(self) -> float:
        """Return total propagated path length."""
        return sum(edge.length_meters for edge in self.path_edges)

    @property
    def speed_limit_mps(self) -> float:
        """Return the first edge speed limit for legacy single-field output."""
        return self.path_edges[0].speed_limit_mps


@dataclass(frozen=True)
class PropagationTarget:
    """Configuration for one downstream junction's propagation calculations."""

    downstream_junction_id: str
    phase_by_incoming_edge: dict[str, int]
    target_downstream_phase_indices: tuple[int, ...]
    links: tuple[PropagationLink, ...]
    skipped_links: tuple[dict[str, Any], ...] = ()
    requires_edge_scale_metrics: bool = False


def load_edge_segments_from_net(edge_ids: tuple[str, ...]) -> dict[str, EdgeSegment]:
    """Load edge lengths and speeds from the SUMO network file."""
    if not ROAD_NETWORK_NET_FILE.exists():
        raise FileNotFoundError(f"net file not found: {ROAD_NETWORK_NET_FILE}")

    pending_edge_ids = set(edge_ids)
    segments: dict[str, EdgeSegment] = {}
    for _event, edge in ET.iterparse(ROAD_NETWORK_NET_FILE, events=("end",)):
        if edge.tag != "edge":
            continue

        edge_id = edge.get("id")
        if edge_id not in pending_edge_ids:
            edge.clear()
            continue

        lane = edge.find("lane")
        if lane is None:
            raise ValueError(f"edge has no lane in net file: {edge_id}")
        segments[edge_id] = EdgeSegment(
            edge_id=edge_id,
            length_meters=float(lane.get("length", "0")),
            speed_limit_mps=float(lane.get("speed", "0")),
        )
        pending_edge_ids.remove(edge_id)
        edge.clear()
        if not pending_edge_ids:
            break

    if pending_edge_ids:
        raise ValueError(
            "edge(s) not found in net file: " + ", ".join(sorted(pending_edge_ids))
        )
    return segments


CLUSTER_174_EDGE_SEGMENTS = load_edge_segments_from_net(
    ("870805596", "1173730629#3", "1301070921", "-1301070928#0", "162853423")
)


PROPAGATION_LINKS = (
    PropagationLink(
        upstream_junction_id="cluster_1262396634_1746662956",
        downstream_junction_id="cluster_1262396675_2350807772",
        path_edges=(EdgeSegment("162851811#1", 292.27, 27.78),),
    ),
    PropagationLink(
        upstream_junction_id="cluster_3476413627_3476413628",
        downstream_junction_id="cluster_1262396675_2350807772",
        path_edges=(EdgeSegment("340418825#2", 181.19, 27.78),),
    ),
    PropagationLink(
        upstream_junction_id="cluster_1247897642_2350807770",
        downstream_junction_id="cluster_1262396675_2350807772",
        path_edges=(EdgeSegment("850154308", 162.11, 27.78),),
    ),
    PropagationLink(
        upstream_junction_id="cluster_1746667327_1746667337",
        downstream_junction_id="cluster_1262396675_2350807772",
        path_edges=(
            EdgeSegment("1301070927", 74.85, 27.78),
            EdgeSegment("870805597", 256.40, 27.78),
        ),
    ),
)

JUNCTION_1746667341_EDGE_SEGMENTS = load_edge_segments_from_net(
    ("870805598#0", "-850154329#0", "-870805598#1", "1301070928#0")
)


CLUSTER_174_PROPAGATION_LINKS = (
    PropagationLink(
        upstream_junction_id="cluster_3476413732_3476413733",
        downstream_junction_id="cluster_1746667327_1746667337",
        path_edges=(CLUSTER_174_EDGE_SEGMENTS["870805596"],),
    ),
    PropagationLink(
        upstream_junction_id="cluster_1262396675_2350807772",
        downstream_junction_id="cluster_1746667327_1746667337",
        path_edges=(
            CLUSTER_174_EDGE_SEGMENTS["1173730629#3"],
            CLUSTER_174_EDGE_SEGMENTS["1301070921"],
        ),
    ),
    PropagationLink(
        upstream_junction_id="1746667341",
        downstream_junction_id="cluster_1746667327_1746667337",
        path_edges=(CLUSTER_174_EDGE_SEGMENTS["-1301070928#0"],),
    ),
    PropagationLink(
        upstream_junction_id="1746667339",
        downstream_junction_id="cluster_1746667327_1746667337",
        path_edges=(CLUSTER_174_EDGE_SEGMENTS["162853423"],),
    ),
)

JUNCTION_1746667341_PROPAGATION_LINKS = (
    PropagationLink(
        upstream_junction_id="3476413738",
        downstream_junction_id="1746667341",
        path_edges=(JUNCTION_1746667341_EDGE_SEGMENTS["870805598#0"],),
    ),
    PropagationLink(
        upstream_junction_id="cluster_3476413627_3476413628",
        downstream_junction_id="1746667341",
        path_edges=(
            JUNCTION_1746667341_EDGE_SEGMENTS["-850154329#0"],
            JUNCTION_1746667341_EDGE_SEGMENTS["-870805598#1"],
        ),
    ),
    PropagationLink(
        upstream_junction_id="cluster_1746667327_1746667337",
        downstream_junction_id="1746667341",
        path_edges=(JUNCTION_1746667341_EDGE_SEGMENTS["1301070928#0"],),
    ),
)

PROPAGATION_TARGETS = (
    PropagationTarget(
        downstream_junction_id="cluster_1262396675_2350807772",
        phase_by_incoming_edge={
            "340418825#2": 0,
            "162851811#1": 2,
            "850154308": 4,
            "870805597": 6,
        },
        target_downstream_phase_indices=(0, 2, 4, 6),
        links=PROPAGATION_LINKS,
    ),
    PropagationTarget(
        downstream_junction_id="cluster_1746667327_1746667337",
        phase_by_incoming_edge={
            "870805596": 0,
            "1301070921": 0,
            "-1301070928#0": 2,
            "162853423": 2,
        },
        target_downstream_phase_indices=(0, 2),
        links=CLUSTER_174_PROPAGATION_LINKS,
        requires_edge_scale_metrics=True,
    ),
    PropagationTarget(
        downstream_junction_id="1746667341",
        phase_by_incoming_edge={
            "870805598#0": 0,
            "-870805598#1": 0,
            "1301070928#0": 2,
        },
        target_downstream_phase_indices=(0, 2),
        links=JUNCTION_1746667341_PROPAGATION_LINKS,
        requires_edge_scale_metrics=True,
    ),
)


def selected_propagation_links(
    target: PropagationTarget,
) -> tuple[PropagationLink, ...]:
    """Return propagation links for the configured downstream service phases."""
    target_phases = set(target.target_downstream_phase_indices)
    return tuple(
        link
        for link in target.links
        if target.phase_by_incoming_edge[link.downstream_incoming_edge_id]
        in target_phases
    )


def penetration_sort_key(penetration: str) -> tuple[int, float | str]:
    try:
        return (0, float(penetration.replace("_", ".")))
    except ValueError:
        return (1, penetration)


def available_penetrations() -> list[str]:
    """Return penetration folders supported by full_flow data."""
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


def compute_delay(
    edge_length: float, speed_limit: float, speed_ratio: float = 0.6
) -> float:
    """Compute propagation delay from edge length and average speed."""
    if speed_limit <= 0:
        raise ValueError(f"speed_limit must be positive: {speed_limit}")
    if speed_ratio <= 0:
        raise ValueError(f"speed_ratio must be positive: {speed_ratio}")

    average_speed = speed_ratio * speed_limit
    return edge_length / average_speed


def compute_link_delay(link: PropagationLink, speed_ratio: float = 0.6) -> float:
    """Compute total propagation delay across all path edges."""
    return sum(
        compute_delay(edge.length_meters, edge.speed_limit_mps, speed_ratio)
        for edge in link.path_edges
    )


def effective_average_speed(link: PropagationLink, delay_seconds: float) -> float:
    """Return path-level average speed implied by total length and delay."""
    if delay_seconds <= 0:
        raise ValueError(f"delay_seconds must be positive: {delay_seconds}")
    return link.edge_length_meters / delay_seconds


def load_tls_data(path: Path = TLS_FILE) -> dict:
    """Load exported SUMO traffic-light phase data."""
    if not path.exists():
        raise FileNotFoundError(f"TLS JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def reset_propagation_metrics_dir(path: Path = PROPAGATION_METRICS_DIR) -> None:
    """Clear and recreate the propagation metrics output directory."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def find_junction_tls(tls_data: dict, junction_id: str) -> dict:
    """Find one junction's TLS data in the exported junction_tls JSON."""
    if tls_data.get("junctionId") == junction_id:
        return tls_data

    for junction in tls_data.get("junctions", []):
        if junction.get("junctionId") == junction_id:
            return junction

    raise ValueError(f"junction not found in TLS data: {junction_id}")


def cycle_duration_from_tls(tls: dict) -> float:
    """Return one full signal cycle duration from TLS phase data."""
    return sum(float(phase["duration"]) for phase in tls["phases"])


def max_metric_cycle_index(cycle_duration: float) -> int:
    """Return the last full downstream cycle index used for metrics."""
    return int(METRIC_END_TIME_SECONDS // cycle_duration)


def downstream_metric_horizon_end_time(
    tls: dict[str, Any],
    phase_indices: set[int],
    max_cycle_index: int,
    start_time: float = 0.0,
) -> float:
    """Return the end time needed to include metric windows through a cycle."""
    phases = tls["phases"]
    cycle_duration = cycle_duration_from_tls(tls)
    latest_end_time = float(start_time)
    for phase_index in phase_indices:
        if phase_index < 0 or phase_index >= len(phases):
            raise ValueError(f"phase_index out of range: {phase_index}")
        phase_start = sum(float(phase["duration"]) for phase in phases[:phase_index])
        phase_duration = float(phases[phase_index]["duration"])
        latest_end_time = max(
            latest_end_time,
            float(start_time)
            + max_cycle_index * cycle_duration
            + phase_start
            + phase_duration,
        )
    return latest_end_time


def downstream_connections_by_phase(
    tls: dict[str, Any],
    incoming_edge_id: str,
) -> dict[int, list[dict[str, Any]]]:
    """Return downstream allowed connections grouped by phase for one incoming edge."""
    connections_by_phase: dict[int, list[dict[str, Any]]] = {}
    for phase in tls["phases"]:
        phase_index = int(phase["index"])
        connections = [
            connection
            for connection in phase.get("allowedConnections", [])
            if connection.get("from") == incoming_edge_id
        ]
        if connections:
            connections_by_phase[phase_index] = connections
    return connections_by_phase


def configured_downstream_connections_by_phase(
    tls: dict[str, Any],
    incoming_edge_id: str,
    phase_by_incoming_edge: dict[str, int],
) -> dict[int, list[dict[str, Any]]]:
    """Return the configured downstream service phase for one incoming edge."""
    if incoming_edge_id not in phase_by_incoming_edge:
        raise ValueError(
            f"no configured downstream phase for incoming edge {incoming_edge_id}"
        )

    phase_index = phase_by_incoming_edge[incoming_edge_id]
    connections_by_phase = downstream_connections_by_phase(tls, incoming_edge_id)
    if phase_index not in connections_by_phase:
        raise ValueError(
            f"configured phase{phase_index} does not use incoming edge "
            f"{incoming_edge_id}"
        )
    return {phase_index: connections_by_phase[phase_index]}


def build_relevant_downstream_windows_from_tls(
    tls: dict[str, Any],
    phase_indices: set[int],
    start_time: float,
    end_time: float,
) -> pd.DataFrame:
    """Build chronological downstream windows for all relevant phase indices."""
    phases = tls["phases"]
    cycle_duration = cycle_duration_from_tls(tls)
    phase_starts = {
        index: sum(float(phase["duration"]) for phase in phases[:index])
        for index in range(len(phases))
    }
    phase_durations = {
        int(phase["index"]): float(phase["duration"])
        for phase in phases
    }
    rows = []
    cycle_index = 0
    cycle_start = float(start_time)
    while cycle_start < end_time:
        for phase_index in sorted(phase_indices):
            if phase_index < 0 or phase_index >= len(phases):
                raise ValueError(f"phase_index out of range: {phase_index}")
            window_start = cycle_start + phase_starts[phase_index]
            if window_start >= end_time:
                continue
            rows.append(
                {
                    "cycle_index": cycle_index,
                    "phase_index": phase_index,
                    "phase_id": f"phase{phase_index}",
                    "window_start": window_start,
                    "window_end": min(
                        window_start + phase_durations[phase_index],
                        float(end_time),
                    ),
                }
            )
        cycle_index += 1
        cycle_start += cycle_duration

    return pd.DataFrame(rows).sort_values(
        ["window_end", "window_start", "phase_index"],
        ignore_index=True,
    )


def validate_upstream_estimates(upstream_df: pd.DataFrame) -> pd.DataFrame:
    """Validate that upstream estimate data has the required columns."""
    missing_columns = UPSTREAM_REQUIRED_COLUMNS - set(upstream_df.columns)
    if missing_columns:
        raise ValueError(
            "upstream estimates missing required columns: "
            f"{sorted(missing_columns)}"
        )
    return upstream_df


def default_upstream_metrics_file(
    penetration: str,
    seed: int,
    upstream_junction_id: str,
) -> Path:
    """Return the upstream combined scale metrics JSON."""
    return (
        SCALE_METRICS_DIR
        / penetration
        / f"{upstream_junction_id}_{seed}_scale_metrics.json"
    )


def downstream_scale_smoothed_metrics_file(
    penetration: str,
    seed: int,
    downstream_junction_id: str,
) -> Path:
    """Return downstream combined scale metrics JSON path."""
    return (
        SCALE_METRICS_DIR
        / penetration
        / f"{downstream_junction_id}_{seed}_scale_metrics.json"
    )


def downstream_departures_file_for_penetration(
    penetration: str,
    downstream_junction_id: str,
) -> Path:
    """Return downstream departures JSON path for actual edge-specific counts."""
    return FULL_FLOW_DIR / penetration / f"{downstream_junction_id}_edge_departures_by_cycle.json"


def upstream_departures_file_for_metrics(
    metrics: dict[str, Any],
    source_file: Path,
    upstream_junction_id: str,
) -> Path:
    """Return upstream departures file in the local workspace."""
    penetration = str(metrics.get("penetration") or source_file.parent.name)
    junction_id = str(metrics.get("junctionId") or upstream_junction_id)
    return FULL_FLOW_DIR / penetration / f"{junction_id}_edge_departures_by_cycle.json"


def upstream_sample_file_for_metrics(metrics: dict[str, Any], source_file: Path) -> Path:
    """Return upstream sample file in the local workspace."""
    penetration = str(metrics.get("penetration") or source_file.parent.name)
    seed = int(metrics.get("seed") or DEFAULT_SEED)
    return ANALYSIS_DIR / "simulation" / "penetration" / f"{penetration}_{seed}_sample.txt"


def penetration_rate_from_metrics(metrics: dict[str, Any], source_file: Path) -> float:
    """Read the sampling penetration rate used by the scale estimate."""
    if "penetrationRate" in metrics:
        rate = float(metrics["penetrationRate"])
    else:
        penetration = str(metrics.get("penetration") or source_file.parent.name)
        rate = float(penetration) / 100.0
    if rate <= 0:
        raise ValueError(f"penetration rate must be positive: {rate}")
    return rate


def load_sample_vehicle_ids(sample_file: Path) -> set[str]:
    """Load sampled vehicle IDs used by the scale estimator."""
    if not sample_file.exists():
        raise FileNotFoundError(f"sample vehicle file not found: {sample_file}")
    return {
        line.strip()
        for line in sample_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def departure_time_slices(
    departures: dict[str, Any],
) -> dict[tuple[int, int], dict[str, Any]]:
    """Index departure time slices by cycle and phase."""
    time_slices: dict[tuple[int, int], dict[str, Any]] = {}
    for cycle in departures["cycles"]:
        cycle_index = int(cycle["cycleIndex"])
        for time_slice in cycle["timeSlices"]:
            phase_index = int(time_slice["phaseIndex"])
            time_slices[(cycle_index, phase_index)] = time_slice
    return time_slices


def target_edge_phase_indices(
    departures: dict[str, Any],
    target_edge_id: str,
) -> set[int]:
    """Return upstream phase indices with movements entering the target edge."""
    phase_indices: set[int] = set()
    for cycle in departures["cycles"]:
        for time_slice in cycle["timeSlices"]:
            if any(
                movement.get("toEdge") == target_edge_id
                for movement in time_slice.get("movements", [])
            ):
                phase_indices.add(int(time_slice["phaseIndex"]))
    return phase_indices


def scaled_count(sampled_observed_count: int, rate: float) -> int:
    """Scale a sampled count back to the estimated full count."""
    return math.floor(sampled_observed_count / rate + 0.5)


def target_edge_counts_for_time_slice(
    time_slice: dict[str, Any],
    sample_vehicle_ids: set[str],
    rate: float,
    target_edge_id: str,
) -> dict[str, float]:
    """Estimate counts for movements entering the propagated edge only."""
    phase_duration = float(time_slice["end"]) - float(time_slice["start"])
    per_lane_capacity = phase_duration / 2.0
    lane_counts: dict[str, dict[str, int | str]] = {}

    for movement in time_slice.get("movements", []):
        if movement.get("toEdge") != target_edge_id:
            continue

        from_lane = movement.get("fromLane")
        if not from_lane:
            continue

        lane_count = lane_counts.setdefault(
            from_lane,
            {
                "fromLane": from_lane,
                "actualCount": 0,
                "sampledObservedCount": 0,
            },
        )
        lane_count["actualCount"] = int(lane_count["actualCount"]) + int(
            movement.get("count", 0)
        )
        lane_count["sampledObservedCount"] = int(
            lane_count["sampledObservedCount"]
        ) + sum(
            1
            for vehicle_id in movement.get("vehicleIds", [])
            if vehicle_id in sample_vehicle_ids
        )

    sampled_total = sum(int(item["sampledObservedCount"]) for item in lane_counts.values())
    actual_total = sum(int(item["actualCount"]) for item in lane_counts.values())
    capped_total = sum(
        min(float(scaled_count(int(item["sampledObservedCount"]), rate)), per_lane_capacity)
        for item in lane_counts.values()
    )
    return {
        "actualCount": float(actual_total),
        "observedCount": float(scaled_count(sampled_total, rate)),
        "cappedObservedCount": float(capped_total),
    }


def target_edge_estimates_from_metrics_json(
    metrics: dict[str, Any],
    source_file: Path,
    timeline: list[dict[str, Any]],
    estimate_field: str,
    upstream_junction_id: str,
    target_edge_id: str,
) -> pd.DataFrame:
    """Build scale+capped upstream estimates for movements entering the target edge."""
    departures = load_json(
        upstream_departures_file_for_metrics(metrics, source_file, upstream_junction_id)
    )
    departure_slices = departure_time_slices(departures)
    target_phases = target_edge_phase_indices(departures, target_edge_id)
    if not target_phases:
        raise ValueError(
            f"no upstream phases enter target edge {target_edge_id} "
            f"for junction {upstream_junction_id}"
        )
    sample_vehicle_ids = load_sample_vehicle_ids(
        upstream_sample_file_for_metrics(metrics, source_file)
    )
    rate = penetration_rate_from_metrics(metrics, source_file)
    junction_id = str(metrics.get("junctionId") or upstream_junction_id)

    edge_rows = []
    for entry in timeline:
        phase_index = int(entry["phaseIndex"])
        if phase_index not in target_phases:
            continue

        cycle_index = int(entry["cycleIndex"])
        departure_slice = departure_slices.get((cycle_index, phase_index))
        counts = (
            target_edge_counts_for_time_slice(
                departure_slice,
                sample_vehicle_ids,
                rate,
                target_edge_id,
            )
            if departure_slice
            else {
                "actualCount": 0.0,
                "observedCount": 0.0,
                "cappedObservedCount": 0.0,
            }
        )
        edge_rows.append(
            {
                "cycleIndex": cycle_index,
                "phaseIndex": phase_index,
                "start": float(entry["start"]),
                "end": float(entry["end"]),
                **counts,
            }
        )

    rows = []
    for edge_row in edge_rows:
        if estimate_field not in edge_row:
            raise ValueError(f"unsupported edge estimate field: {estimate_field}")
        rows.append(
            {
                "junction_id": junction_id,
                "phase_id": f'phase{edge_row["phaseIndex"]}',
                "window_start": float(edge_row["start"]),
                "window_end": float(edge_row["end"]),
                "estimate": float(edge_row[estimate_field]),
                "actual_count": float(edge_row["actualCount"]),
                "scaled_count": float(edge_row["observedCount"]),
                "capped_count": float(edge_row["cappedObservedCount"]),
            }
        )

    upstream_df = pd.DataFrame(
        rows,
        columns=[
            "junction_id",
            "phase_id",
            "window_start",
            "window_end",
            "estimate",
            "actual_count",
            "scaled_count",
            "capped_count",
        ],
    )
    return validate_upstream_estimates(upstream_df)


def upstream_estimates_from_metrics_json(
    metrics: dict[str, Any],
    source_file: Path,
    upstream_junction_id: str,
    target_edge_id: str,
    estimate_field: str = DEFAULT_ESTIMATE_FIELD,
) -> pd.DataFrame:
    """Convert a metrics JSON timeline into target-edge upstream estimate windows."""
    timeline = metrics.get("metricTimeline")
    if timeline is None and "methods" in metrics:
        timeline = metrics["methods"]["scale+cap"]["metricTimeline"]
    if timeline is None:
        raise ValueError(f"metrics JSON has no metricTimeline: {source_file}")

    return target_edge_estimates_from_metrics_json(
        metrics,
        source_file,
        timeline,
        estimate_field,
        upstream_junction_id,
        target_edge_id,
    )


def load_upstream_estimates(
    path: str,
    upstream_junction_id: str,
    target_edge_id: str,
) -> pd.DataFrame:
    """Load upstream estimates from CSV/JSON."""

    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        upstream_df = pd.read_csv(input_path)
    elif suffix == ".json":
        data = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and (
            "metricTimeline" in data or "methods" in data
        ):
            return upstream_estimates_from_metrics_json(
                data,
                input_path,
                upstream_junction_id,
                target_edge_id,
            )
        upstream_df = pd.DataFrame(data)
    else:
        raise ValueError(f"unsupported upstream estimate file type: {suffix}")

    return validate_upstream_estimates(upstream_df)


def downstream_actual_counts_by_window(
    departures: dict[str, Any],
    incoming_edge_id: str,
) -> dict[tuple[int, int], float]:
    """Read actual downstream counts for one incoming edge by cycle and phase."""
    counts_by_window: dict[tuple[int, int], float] = {}
    for cycle in departures["cycles"]:
        cycle_index = int(cycle["cycleIndex"])
        for time_slice in cycle["timeSlices"]:
            phase_index = int(time_slice["phaseIndex"])
            actual_count = sum(
                float(movement.get("count", 0))
                for movement in time_slice.get("movements", [])
                if movement.get("fromEdge") == incoming_edge_id
            )
            counts_by_window[(cycle_index, phase_index)] = actual_count
    return counts_by_window


def sum_period_counts(entry: dict[str, Any], field_name: str) -> float:
    """Sum a metric field across all directions in periodCounts."""
    return sum(
        float(direction_counts.get(field_name, 0))
        for direction_counts in entry["periodCounts"].values()
    )


def downstream_scale_smoothed_counts_by_window(
    metrics: dict[str, Any],
    incoming_edge_id: str | None = None,
    field_name: str = "cappedThenSmoothedObservedCount",
) -> dict[tuple[int, int], float]:
    """Read downstream scale+capped+smoothed counts by cycle and phase."""
    timeline_parent = metrics
    if "methods" in metrics:
        timeline_parent = metrics["methods"]["scale+cap+smooth"]
        field_name = "observedCount"

    counts_by_window: dict[tuple[int, int], float] = {}
    edge_timeline = timeline_parent.get("edgeMetricTimeline")
    if incoming_edge_id is not None and edge_timeline is not None:
        for entry in edge_timeline:
            if entry.get("fromEdge") != incoming_edge_id:
                continue
            cycle_index = int(entry["cycleIndex"])
            phase_index = int(entry["phaseIndex"])
            counts_by_window[(cycle_index, phase_index)] = float(
                entry.get(field_name, 0)
            )
        return counts_by_window

    for entry in timeline_parent["metricTimeline"]:
        cycle_index = int(entry["cycleIndex"])
        phase_index = int(entry["phaseIndex"])
        counts_by_window[(cycle_index, phase_index)] = sum_period_counts(
            entry,
            field_name,
        )
    return counts_by_window


def add_zero_filtered_scale_average_estimate(
    result_df: pd.DataFrame,
    downstream_smoothed_counts_by_window: dict[tuple[int, int], float],
    zero_filtered_upstream_weight: float = DEFAULT_ZERO_FILTERED_UPSTREAM_WEIGHT,
) -> pd.DataFrame:
    """Combine zero-filtered upstream and downstream scale-smoothed estimates."""
    result_df = result_df.copy()
    result_df["downstream_scale_capped_smoothed_estimate"] = result_df.apply(
        lambda row: downstream_smoothed_counts_by_window.get(
            (int(row["down_cycle_index"]), int(row["down_phase_index"]))
        ),
        axis=1,
    )
    result_df["zero_filtered_upstream_average_weight"] = zero_filtered_upstream_weight
    result_df["downstream_scale_average_weight"] = 1.0 - zero_filtered_upstream_weight
    result_df["zero_filtered_scale_smoothed_average_estimate"] = (
        zero_filtered_upstream_weight
        * result_df["zero_filtered_smoothed_upstream_estimate"]
        + (1.0 - zero_filtered_upstream_weight)
        * result_df["downstream_scale_capped_smoothed_estimate"]
    )
    return result_df


def best_zero_filtered_scale_average_weight(result_df: pd.DataFrame) -> float:
    """Find the one-decimal upstream weight that minimizes NMAE."""
    valid_df = result_df.dropna(
        subset=[
            "actual_downstream_count",
            "zero_filtered_smoothed_upstream_estimate",
            "downstream_scale_capped_smoothed_estimate",
        ]
    )
    if valid_df.empty:
        return DEFAULT_ZERO_FILTERED_UPSTREAM_WEIGHT

    actual_total = float(valid_df["actual_downstream_count"].sum())
    if actual_total == 0.0:
        return DEFAULT_ZERO_FILTERED_UPSTREAM_WEIGHT

    best_weight = DEFAULT_ZERO_FILTERED_UPSTREAM_WEIGHT
    best_nmae = math.inf
    candidate_count = int(1 / WEIGHTED_AVERAGE_COEFFICIENT_STEP) + 1
    for index in range(candidate_count):
        weight = round(index * WEIGHTED_AVERAGE_COEFFICIENT_STEP, 1)
        estimate = (
            weight * valid_df["zero_filtered_smoothed_upstream_estimate"]
            + (1.0 - weight) * valid_df["downstream_scale_capped_smoothed_estimate"]
        )
        nmae = float((estimate - valid_df["actual_downstream_count"]).abs().sum())
        nmae /= actual_total
        if nmae < best_nmae:
            best_nmae = nmae
            best_weight = weight

    return best_weight


def add_downstream_actuals(
    result_df: pd.DataFrame,
    actual_counts_by_window: dict[tuple[int, int], float],
) -> pd.DataFrame:
    """Attach downstream actual counts and propagation errors."""
    result_df = result_df.copy()
    result_df["actual_downstream_count"] = result_df.apply(
        lambda row: actual_counts_by_window.get(
            (int(row["down_cycle_index"]), int(row["down_phase_index"]))
        ),
        axis=1,
    )
    result_df["upstream_estimate_error"] = (
        result_df["upstream_estimate"] - result_df["actual_downstream_count"]
    )
    result_df["upstream_estimate_absolute_error"] = result_df[
        "upstream_estimate_error"
    ].abs()
    result_df["upstream_estimate_squared_error"] = (
        result_df["upstream_estimate_error"] ** 2
    )
    if "smoothed_upstream_estimate" in result_df:
        result_df["smoothed_upstream_estimate_error"] = (
            result_df["smoothed_upstream_estimate"]
            - result_df["actual_downstream_count"]
        )
        result_df["smoothed_upstream_estimate_absolute_error"] = result_df[
            "smoothed_upstream_estimate_error"
        ].abs()
        result_df["smoothed_upstream_estimate_squared_error"] = (
            result_df["smoothed_upstream_estimate_error"] ** 2
        )
    if "zero_filtered_smoothed_upstream_estimate" in result_df:
        result_df["zero_filtered_smoothed_upstream_estimate_error"] = (
            result_df["zero_filtered_smoothed_upstream_estimate"]
            - result_df["actual_downstream_count"]
        )
        result_df["zero_filtered_smoothed_upstream_estimate_absolute_error"] = result_df[
            "zero_filtered_smoothed_upstream_estimate_error"
        ].abs()
        result_df["zero_filtered_smoothed_upstream_estimate_squared_error"] = (
            result_df["zero_filtered_smoothed_upstream_estimate_error"] ** 2
        )
    if "zero_filtered_scale_smoothed_average_estimate" in result_df:
        result_df["zero_filtered_scale_smoothed_average_estimate_error"] = (
            result_df["zero_filtered_scale_smoothed_average_estimate"]
            - result_df["actual_downstream_count"]
        )
        result_df["zero_filtered_scale_smoothed_average_estimate_absolute_error"] = (
            result_df["zero_filtered_scale_smoothed_average_estimate_error"].abs()
        )
        result_df["zero_filtered_scale_smoothed_average_estimate_squared_error"] = (
            result_df["zero_filtered_scale_smoothed_average_estimate_error"] ** 2
        )
    return result_df


def smooth_counts(values: list[float]) -> list[float]:
    """Smooth a time-ordered count series with a centered moving window."""
    radius = SMOOTH_WINDOW_SIZE // 2
    smoothed_values: list[float] = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        window_values = values[start:end]
        smoothed_values.append(sum(window_values) / len(window_values))
    return smoothed_values


def smooth_counts_ignoring_zero_observations(values: list[float]) -> list[float]:
    """Smooth non-zero observations, then interpolate zero-valued gaps."""
    observed_points = [
        (index, value)
        for index, value in enumerate(values)
        if value > 0
    ]
    if not observed_points:
        return [0.0 for _ in values]

    observed_indices = [index for index, _ in observed_points]
    observed_values = [value for _, value in observed_points]
    smoothed_observed_values = smooth_counts(observed_values)
    smoothed_by_index = dict(zip(observed_indices, smoothed_observed_values))

    filled_values: list[float] = []
    next_observed_position = 0
    for index in range(len(values)):
        if index in smoothed_by_index:
            filled_values.append(smoothed_by_index[index])
            continue

        while (
            next_observed_position < len(observed_indices)
            and observed_indices[next_observed_position] < index
        ):
            next_observed_position += 1

        if next_observed_position == 0:
            filled_values.append(smoothed_observed_values[0])
        elif next_observed_position >= len(observed_indices):
            filled_values.append(smoothed_observed_values[-1])
        else:
            left_position = next_observed_position - 1
            right_position = next_observed_position
            left_index = observed_indices[left_position]
            right_index = observed_indices[right_position]
            left_value = smoothed_observed_values[left_position]
            right_value = smoothed_observed_values[right_position]
            ratio = (index - left_index) / (right_index - left_index)
            filled_values.append(left_value + (right_value - left_value) * ratio)

    return filled_values


def add_smoothed_upstream_estimates(result_df: pd.DataFrame) -> pd.DataFrame:
    """Add regular smoothing and zero-filtered smoothing estimates."""
    if result_df.empty:
        return result_df.copy()

    result_df = result_df.copy()
    values = result_df["upstream_estimate"].astype(float).tolist()
    result_df["upstream_estimate_missing_for_smoothing"] = [
        value == 0.0 for value in values
    ]
    result_df["smoothed_upstream_estimate"] = smooth_counts(values)
    result_df["zero_filtered_smoothed_upstream_estimate"] = (
        smooth_counts_ignoring_zero_observations(values)
    )
    result_df["upstream_estimate_smooth_window_size"] = SMOOTH_WINDOW_SIZE
    return result_df


def summarize_propagation_metrics(
    result_df: pd.DataFrame,
    estimate_column: str,
) -> dict[str, Any]:
    """Summarize propagated arrivals against downstream actual counts."""
    if "actual_downstream_count" not in result_df or estimate_column not in result_df:
        return {
            "periodCount": 0,
            "mae": None,
            "rmse": None,
            "estimatedTotal": 0.0,
            "actualTotal": 0.0,
            "nmae": None,
        }

    valid_df = result_df.dropna(subset=["actual_downstream_count", estimate_column])
    if valid_df.empty:
        return {
            "periodCount": 0,
            "mae": None,
            "rmse": None,
            "estimatedTotal": 0.0,
            "actualTotal": 0.0,
            "nmae": None,
        }

    errors = valid_df[estimate_column] - valid_df["actual_downstream_count"]
    absolute_error_sum = float(errors.abs().sum())
    squared_error_sum = float((errors ** 2).sum())
    estimated_total = float(valid_df[estimate_column].sum())
    actual_total = float(valid_df["actual_downstream_count"].sum())
    period_count = int(len(valid_df))
    return {
        "periodCount": period_count,
        "mae": absolute_error_sum / period_count,
        "rmse": math.sqrt(squared_error_sum / period_count),
        "estimatedTotal": estimated_total,
        "actualTotal": actual_total,
        "nmae": absolute_error_sum / actual_total if actual_total else None,
    }


def propagation_metric_records(
    propagation_data: dict[str, Any],
    min_down_cycle_index: int,
    max_down_cycle_index: int,
) -> list[dict[str, Any]]:
    """Extract cycle-level records used by overall per-penetration metrics."""
    upstream_junction_id = str(propagation_data["upstreamJunctionId"])
    downstream_junction_id = str(propagation_data["downstreamJunctionId"])
    downstream_incoming_edge = str(propagation_data["downstreamIncomingEdge"])
    phase_index = int(propagation_data["downstreamInfluencePhases"][0]["phaseIndex"])
    records = []
    for index, row in enumerate(propagation_data["results"]):
        cycle_index = int(row.get("down_cycle_index", index + 1))
        if not (
            min_down_cycle_index
            <= cycle_index
            <= max_down_cycle_index
        ):
            continue
        if "actual_downstream_count" not in row:
            continue

        record = {
            "upstreamJunctionId": upstream_junction_id,
            "downstreamJunctionId": downstream_junction_id,
            "downstreamIncomingEdge": downstream_incoming_edge,
            "phaseIndex": phase_index,
            "cycleIndex": cycle_index,
            "actualCount": float(row["actual_downstream_count"]),
        }
        for _, field_name in ESTIMATE_FIELDS:
            if field_name in row and row[field_name] is not None:
                record[field_name] = float(row[field_name])
        records.append(record)

    return records


def build_overall_metrics_output(
    target: PropagationTarget,
    penetration: str,
    seed: int,
    records: list[dict[str, Any]],
    min_down_cycle_index: int,
    max_down_cycle_index: int,
) -> dict[str, Any]:
    """Build one per-penetration metrics JSON from all propagation links."""
    overall_df = pd.DataFrame(records)
    if not overall_df.empty:
        overall_df = overall_df.rename(columns={"actualCount": "actual_downstream_count"})

    metrics = {
        metric_name: {
            **summarize_propagation_metrics(overall_df, field_name),
            "minDownCycleIndex": min_down_cycle_index,
            "maxDownCycleIndex": max_down_cycle_index,
        }
        for metric_name, field_name in ESTIMATE_FIELDS
    }
    metrics_by_incoming_edge = {}
    if not overall_df.empty:
        for incoming_edge, link_df in overall_df.groupby(
            "downstreamIncomingEdge",
            sort=True,
        ):
            first_row = link_df.iloc[0]
            metrics_by_incoming_edge[str(incoming_edge)] = {
                "upstreamJunctionId": str(first_row["upstreamJunctionId"]),
                "phaseIndex": int(first_row["phaseIndex"]),
                "recordCount": int(len(link_df)),
                "metrics": {
                    metric_name: {
                        **summarize_propagation_metrics(link_df, field_name),
                        "minDownCycleIndex": min_down_cycle_index,
                        "maxDownCycleIndex": max_down_cycle_index,
                    }
                    for metric_name, field_name in ESTIMATE_FIELDS
                },
            }

    return {
        "downstreamJunctionId": target.downstream_junction_id,
        "penetration": penetration,
        "seed": seed,
        "cycleIndexRange": {
            "min": min_down_cycle_index,
            "max": max_down_cycle_index,
        },
        "metricEndTimeSeconds": METRIC_END_TIME_SECONDS,
        "linkCount": len(selected_propagation_links(target)),
        "skippedLinks": list(target.skipped_links),
        "recordCount": len(records),
        "metrics": metrics,
        "metricsByIncomingEdge": metrics_by_incoming_edge,
        "records": records,
    }


def map_downstream_window_to_upstream_interval(
    prev_down_end: float, down_end: float, delay_seconds: float
) -> Tuple[float, float]:
    """Map a downstream arrival interval back to an upstream departure interval."""
    upstream_interval_start = prev_down_end - delay_seconds
    upstream_interval_end = down_end - delay_seconds
    return upstream_interval_start, upstream_interval_end


def estimate_upstream_flow_for_interval(
    upstream_df: pd.DataFrame,
    interval_start: float,
    interval_end: float,
) -> Tuple[float, pd.DataFrame]:
    """Estimate upstream flow in an interval with overlap-based allocation."""
    debug_rows = []

    for row in upstream_df.itertuples(index=False):
        window_start = float(row.window_start)
        window_end = float(row.window_end)
        window_duration = window_end - window_start
        if window_duration <= 0:
            continue

        overlap_start = max(window_start, interval_start)
        overlap_end = min(window_end, interval_end)
        overlap_duration = max(0.0, overlap_end - overlap_start)
        if overlap_duration <= 0:
            continue

        overlap_ratio = overlap_duration / window_duration
        contribution = float(row.estimate) * overlap_ratio
        debug_row = {
            "up_junction_id": row.junction_id,
            "up_phase_id": row.phase_id,
            "up_window_start": window_start,
            "up_window_end": window_end,
            "up_estimate": float(row.estimate),
            "interval_start": interval_start,
            "interval_end": interval_end,
            "overlap_start": overlap_start,
            "overlap_end": overlap_end,
            "overlap_duration": overlap_duration,
            "overlap_ratio": overlap_ratio,
            "contribution": contribution,
        }
        if hasattr(row, "actual_count"):
            debug_row["up_actual_count"] = float(row.actual_count)
        if hasattr(row, "scaled_count"):
            debug_row["up_scaled_count"] = float(row.scaled_count)
        if hasattr(row, "capped_count"):
            debug_row["up_capped_count"] = float(row.capped_count)
        debug_rows.append(debug_row)

    debug_df = pd.DataFrame(debug_rows)
    total = float(debug_df["contribution"].sum()) if not debug_df.empty else 0.0
    return total, debug_df


def compute_propagated_arrivals(
    upstream_df: pd.DataFrame,
    downstream_windows: pd.DataFrame,
    delay_seconds: float,
    downstream_junction_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute propagated arrival estimates for relevant downstream windows.

    The first relevant downstream window is skipped because it has no previous
    service end for this incoming edge, so its arrival interval cannot be
    defined as (previous_downstream_service_end, current_service_end].
    """
    result_rows = []
    debug_frames = []

    for index in range(1, len(downstream_windows)):
        prev_down_end = float(downstream_windows.iloc[index - 1]["window_end"])
        down_cycle_index = int(downstream_windows.iloc[index]["cycle_index"])
        down_phase_index = int(downstream_windows.iloc[index]["phase_index"])
        down_phase_id = str(downstream_windows.iloc[index]["phase_id"])
        down_start = float(downstream_windows.iloc[index]["window_start"])
        down_end = float(downstream_windows.iloc[index]["window_end"])
        interval_start, interval_end = map_downstream_window_to_upstream_interval(
            prev_down_end,
            down_end,
            delay_seconds,
        )
        total, debug_df = estimate_upstream_flow_for_interval(
            upstream_df,
            interval_start,
            interval_end,
        )

        num_overlapped = len(debug_df)
        total_overlap_duration = (
            float(debug_df["overlap_duration"].sum()) if not debug_df.empty else 0.0
        )
        result_rows.append(
            {
                "downstream_junction_id": downstream_junction_id,
                "downstream_phase_id": down_phase_id,
                "down_phase_index": down_phase_index,
                "down_cycle_index": down_cycle_index,
                "down_window_sequence": index,
                "down_window_start": down_start,
                "down_window_end": down_end,
                "prev_down_window_end": prev_down_end,
                "upstream_interval_start": interval_start,
                "upstream_interval_end": interval_end,
                "delay_seconds": delay_seconds,
                "upstream_estimate": total,
                "num_overlapped_upstream_windows": num_overlapped,
                "total_overlap_duration": total_overlap_duration,
            }
        )

        if not debug_df.empty:
            debug_df = debug_df.assign(
                downstream_junction_id=downstream_junction_id,
                downstream_phase_id=down_phase_id,
                down_phase_index=down_phase_index,
                down_cycle_index=down_cycle_index,
                down_window_start=down_start,
                down_window_end=down_end,
                prev_down_window_end=prev_down_end,
            )
            debug_frames.append(debug_df)

    result_df = pd.DataFrame(result_rows)
    debug_all_df = (
        pd.concat(debug_frames, ignore_index=True)
        if debug_frames
        else pd.DataFrame()
    )
    return result_df, debug_all_df


def propagation_output_file(
    link: PropagationLink,
    penetration: str,
    seed: int,
) -> Path:
    """Return the JSON output path for this propagation estimate."""
    return (
        PROPAGATION_METRICS_DIR
        / str(penetration)
        / (
            f"{link.upstream_junction_id}_to_{link.downstream_junction_id}_"
            f"{penetration}_{seed}_propagation_estimates.json"
        )
    )


def overall_metrics_file(
    downstream_junction_id: str,
    penetration: str,
    seed: int,
) -> Path:
    """Return per-penetration overall propagation metrics path."""
    return (
        PROPAGATION_METRICS_DIR
        / str(penetration)
        / f"{downstream_junction_id}_{penetration}_{seed}_overall_propagation_metrics.json"
    )


def write_propagation_output(
    output_file: Path,
    *,
    link: PropagationLink,
    penetration: str,
    seed: int,
    average_speed: float,
    delay_seconds: float,
    upstream_estimates_file: Path,
    downstream_cycle_duration: float,
    upstream_phase_ids: list[str],
    downstream_phase_ids: list[str],
    downstream_phase_durations: dict[str, float],
    downstream_influence_phases: list[dict[str, Any]],
    downstream_windows: pd.DataFrame,
    result_df: pd.DataFrame,
    debug_all_df: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    """Write propagation result and overlap debug tables to JSON."""
    output = {
        "upstreamJunctionId": link.upstream_junction_id,
        "downstreamJunctionId": link.downstream_junction_id,
        "penetration": penetration,
        "seed": seed,
        "edgeId": link.upstream_estimate_edge_id,
        "edgeIds": [edge.edge_id for edge in link.path_edges],
        "edgeSegments": [
            {
                "edgeId": edge.edge_id,
                "lengthMeters": edge.length_meters,
                "speedLimitMps": edge.speed_limit_mps,
            }
            for edge in link.path_edges
        ],
        "edgeLengthMeters": link.edge_length_meters,
        "speedLimitMps": link.speed_limit_mps,
        "averageSpeedMps": average_speed,
        "delaySeconds": delay_seconds,
        "phaseId": "multiple",
        "phaseIndex": None,
        "downstreamPhaseIds": downstream_phase_ids,
        "downstreamPhaseDurations": downstream_phase_durations,
        "downstreamInfluencePhases": downstream_influence_phases,
        "upstreamEstimatePhaseIds": upstream_phase_ids,
        "upstreamEstimateField": DEFAULT_ESTIMATE_FIELD,
        "upstreamEstimateToEdge": link.upstream_estimate_edge_id,
        "downstreamIncomingEdge": link.downstream_incoming_edge_id,
        "upstreamEstimatesFile": str(upstream_estimates_file.resolve()),
        "tlsFile": str(TLS_FILE.resolve()),
        "downstreamCycleDuration": downstream_cycle_duration,
        "metrics": metrics,
        "downstreamWindows": downstream_windows.to_dict(orient="records"),
        "results": result_df.to_dict(orient="records"),
        "debug": debug_all_df.to_dict(orient="records"),
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def calculate_propagation_for_penetration(
    penetration: str,
    seed: int,
    *,
    target: PropagationTarget,
    link: PropagationLink,
    speed_ratio: float,
    downstream_tls: dict[str, Any],
    downstream_cycle_duration: float,
) -> Path:
    """Calculate and save one penetration rate's propagation estimates."""
    metric_max_down_cycle_index = max_metric_cycle_index(downstream_cycle_duration)
    delay_seconds = compute_link_delay(link, speed_ratio)
    average_speed = effective_average_speed(link, delay_seconds)
    upstream_estimates_file = default_upstream_metrics_file(
        penetration,
        seed,
        link.upstream_junction_id,
    )
    upstream_df = load_upstream_estimates(
        str(upstream_estimates_file),
        link.upstream_junction_id,
        link.upstream_estimate_edge_id,
    )
    downstream_departures = load_json(
        downstream_departures_file_for_penetration(
            penetration,
            link.downstream_junction_id,
        )
    )
    downstream_scale_smoothed_metrics = load_json(
        downstream_scale_smoothed_metrics_file(
            penetration,
            seed,
            link.downstream_junction_id,
        )
    )
    if (
        target.requires_edge_scale_metrics
        and "edgeMetricTimeline"
        not in downstream_scale_smoothed_metrics["methods"]["scale+cap+smooth"]
    ):
        raise ValueError(
            "cluster_174 propagation requires edge-level scale metrics; "
            "rerun calculate_scale_estimates.py first"
        )
    downstream_connections = configured_downstream_connections_by_phase(
        downstream_tls,
        link.downstream_incoming_edge_id,
        target.phase_by_incoming_edge,
    )
    if not downstream_connections:
        raise ValueError(
            f"no downstream phases use incoming edge "
            f"{link.downstream_incoming_edge_id}"
        )
    downstream_phase_indices = set(downstream_connections)
    upstream_coverage_end_time = (
        float(upstream_df["window_end"].max())
        + delay_seconds
        + downstream_cycle_duration
    )
    metric_horizon_end_time = downstream_metric_horizon_end_time(
        downstream_tls,
        downstream_phase_indices,
        metric_max_down_cycle_index,
    )
    end_time = max(upstream_coverage_end_time, metric_horizon_end_time)
    downstream_windows = build_relevant_downstream_windows_from_tls(
        downstream_tls,
        downstream_phase_indices,
        start_time=0,
        end_time=end_time,
    )
    result_df, debug_all_df = compute_propagated_arrivals(
        upstream_df=upstream_df,
        downstream_windows=downstream_windows,
        delay_seconds=delay_seconds,
        downstream_junction_id=link.downstream_junction_id,
    )
    result_df = add_smoothed_upstream_estimates(result_df)
    result_df = add_zero_filtered_scale_average_estimate(
        result_df,
        downstream_scale_smoothed_counts_by_window(
            downstream_scale_smoothed_metrics,
            link.downstream_incoming_edge_id,
        ),
    )
    result_df = add_downstream_actuals(
        result_df,
        downstream_actual_counts_by_window(
            downstream_departures,
            link.downstream_incoming_edge_id,
        ),
    )
    metric_df = result_df[
        (
            result_df["down_cycle_index"].astype(int)
            >= METRIC_START_DOWN_CYCLE_INDEX
        )
        & (
            result_df["down_cycle_index"].astype(int)
            <= metric_max_down_cycle_index
        )
    ]
    best_zero_filtered_upstream_weight = best_zero_filtered_scale_average_weight(
        metric_df
    )
    print(
        f"{penetration} {link.upstream_junction_id}: "
        f"best_zero_filtered_upstream_weight="
        f"{best_zero_filtered_upstream_weight:.1f}"
    )
    result_df = add_zero_filtered_scale_average_estimate(
        result_df,
        downstream_scale_smoothed_counts_by_window(
            downstream_scale_smoothed_metrics,
            link.downstream_incoming_edge_id,
        ),
        best_zero_filtered_upstream_weight,
    )
    result_df = add_downstream_actuals(
        result_df,
        downstream_actual_counts_by_window(
            downstream_departures,
            link.downstream_incoming_edge_id,
        ),
    )
    metric_df = result_df[
        (
            result_df["down_cycle_index"].astype(int)
            >= METRIC_START_DOWN_CYCLE_INDEX
        )
        & (
            result_df["down_cycle_index"].astype(int)
            <= metric_max_down_cycle_index
        )
    ]
    metrics = {
        "upstreamEstimate": {
            **summarize_propagation_metrics(
                metric_df,
                "upstream_estimate",
            ),
            "minDownCycleIndex": METRIC_START_DOWN_CYCLE_INDEX,
            "maxDownCycleIndex": metric_max_down_cycle_index,
        },
        "smoothedUpstreamEstimate": {
            **summarize_propagation_metrics(
                metric_df,
                "smoothed_upstream_estimate",
            ),
            "minDownCycleIndex": METRIC_START_DOWN_CYCLE_INDEX,
            "maxDownCycleIndex": metric_max_down_cycle_index,
        },
        "zeroFilteredSmoothedUpstreamEstimate": {
            **summarize_propagation_metrics(
                metric_df,
                "zero_filtered_smoothed_upstream_estimate",
            ),
            "minDownCycleIndex": METRIC_START_DOWN_CYCLE_INDEX,
            "maxDownCycleIndex": metric_max_down_cycle_index,
        },
        "zeroFilteredScaleSmoothedAverageEstimate": {
            **summarize_propagation_metrics(
                metric_df,
                "zero_filtered_scale_smoothed_average_estimate",
            ),
            "minDownCycleIndex": METRIC_START_DOWN_CYCLE_INDEX,
            "maxDownCycleIndex": metric_max_down_cycle_index,
            "zeroFilteredUpstreamWeight": best_zero_filtered_upstream_weight,
            "downstreamScaleSmoothedWeight": (
                1.0 - best_zero_filtered_upstream_weight
            ),
            "coefficientStep": WEIGHTED_AVERAGE_COEFFICIENT_STEP,
        },
    }
    output_file = propagation_output_file(link, penetration, seed)
    write_propagation_output(
        output_file,
        link=link,
        penetration=penetration,
        seed=seed,
        average_speed=average_speed,
        delay_seconds=delay_seconds,
        upstream_estimates_file=upstream_estimates_file,
        downstream_cycle_duration=downstream_cycle_duration,
        upstream_phase_ids=sorted(upstream_df["phase_id"].unique().tolist()),
        downstream_phase_ids=[f"phase{index}" for index in sorted(downstream_phase_indices)],
        downstream_phase_durations={
            f"phase{index}": float(downstream_tls["phases"][index]["duration"])
            for index in sorted(downstream_phase_indices)
        },
        downstream_influence_phases=[
            {
                "phaseIndex": phase_index,
                "phaseId": f"phase{phase_index}",
                "connections": [
                    {
                        "from": connection.get("from"),
                        "to": connection.get("to"),
                        "dir": connection.get("dir"),
                        "laneCount": connection.get("laneCount"),
                    }
                    for connection in downstream_connections[phase_index]
                ],
            }
            for phase_index in sorted(downstream_connections)
        ],
        downstream_windows=downstream_windows,
        result_df=result_df,
        debug_all_df=debug_all_df,
        metrics=metrics,
    )
    return output_file


if __name__ == "__main__":
    reset_propagation_metrics_dir()
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)

    speed_ratio = 0.6
    tls_data = load_tls_data()
    penetrations = available_penetrations()

    print(f"downstream_tls_file = {TLS_FILE.resolve()}")
    for target in PROPAGATION_TARGETS:
        downstream_tls = find_junction_tls(tls_data, target.downstream_junction_id)
        downstream_cycle_duration = cycle_duration_from_tls(downstream_tls)
        metric_max_down_cycle_index = max_metric_cycle_index(downstream_cycle_duration)
        links = selected_propagation_links(target)
        if not links:
            raise ValueError(
                "no propagation links configured for downstream phases "
                f"{target.target_downstream_phase_indices}"
            )

        print()
        print(f"downstream_junction_id = {target.downstream_junction_id}")
        print(f"downstream_cycle_duration = {downstream_cycle_duration:.6f} s")
        print(
            "target_downstream_phases = "
            + ",".join(
                f"phase{index}" for index in target.target_downstream_phase_indices
            )
        )
        if target.skipped_links:
            print(f"skipped_links = {json.dumps(list(target.skipped_links))}")
        print("propagation_links:")
        for link in links:
            delay_seconds = compute_link_delay(link, speed_ratio)
            average_speed = effective_average_speed(link, delay_seconds)
            downstream_phase_ids = ",".join(
                f"phase{phase_index}"
                for phase_index in sorted(
                    configured_downstream_connections_by_phase(
                        downstream_tls,
                        link.downstream_incoming_edge_id,
                        target.phase_by_incoming_edge,
                    )
                )
            )
            print(
                f"  {link.upstream_junction_id} -> {link.downstream_junction_id} | "
                f"edges={','.join(edge.edge_id for edge in link.path_edges)} | "
                f"downstream_phases={downstream_phase_ids} | "
                f"length={link.edge_length_meters:.2f} m | "
                f"average_speed={average_speed:.6f} m/s | "
                f"delay={delay_seconds:.6f} s"
            )
        print()

        for penetration in penetrations:
            overall_records = []
            for link in links:
                output_file = calculate_propagation_for_penetration(
                    str(penetration),
                    DEFAULT_SEED,
                    target=target,
                    link=link,
                    speed_ratio=speed_ratio,
                    downstream_tls=downstream_tls,
                    downstream_cycle_duration=downstream_cycle_duration,
                )
                print(
                    f"saved {penetration} {link.upstream_junction_id}: "
                    f"{output_file.resolve()}"
                )
                overall_records.extend(
                    propagation_metric_records(
                        load_json(output_file),
                        METRIC_START_DOWN_CYCLE_INDEX,
                        metric_max_down_cycle_index,
                    )
                )

            metrics_file = overall_metrics_file(
                target.downstream_junction_id,
                str(penetration),
                DEFAULT_SEED,
            )
            write_json(
                metrics_file,
                build_overall_metrics_output(
                    target,
                    str(penetration),
                    DEFAULT_SEED,
                    overall_records,
                    METRIC_START_DOWN_CYCLE_INDEX,
                    metric_max_down_cycle_index,
                ),
            )
            print(f"saved {penetration} overall metrics: {metrics_file.resolve()}")
