from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

try:
    from .build_sumo_edge_graph import DEFAULT_NET_FILE, load_sumo_edge_graph
    from .match_sumo_edges import (
        PathCache,
        build_osm_candidate_index,
        get_candidate_sumo_edges,
        shortest_path_between_edges,
    )
    from .parse_trajectories import DEFAULT_TRAJ_FILE, iter_trajectory_file
except ImportError:
    from build_sumo_edge_graph import DEFAULT_NET_FILE, load_sumo_edge_graph
    from match_sumo_edges import (
        PathCache,
        build_osm_candidate_index,
        get_candidate_sumo_edges,
        shortest_path_between_edges,
    )
    from parse_trajectories import DEFAULT_TRAJ_FILE, iter_trajectory_file


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_FILE = BASE_DIR / "matched_trips.rou.xml"
LOG_DIR = BASE_DIR / "log"
DEFAULT_REACHABILITY_LOG_FILE = LOG_DIR / "reachability_prune_logs.txt"
DEFAULT_VEHICLE_TYPE_ID = "passenger"
DEFAULT_VEHICLE_CLASS = "passenger"
DEFAULT_DEPART_BEGIN = 0.0
DEFAULT_DEPART_END = 3600.0
DEFAULT_PREVIEW_COUNT = 5
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
ROUTES_XSD = "http://sumo.dlr.de/xsd/routes_file.xsd"
ET.register_namespace("xsi", XSI_NS)


@dataclass(slots=True)
class MatchedTrip:
    vin: str
    start_edge: str
    end_edge: str
    via_edges: list[str]
    kept_edges: list[str]
    dropped_osm_edges: list[str]
    line_no: int | None = None


def choose_preferred_candidate(osm_edge_id: str, candidates: list[str]) -> str:
    """
    Deterministically choose one SUMO edge candidate for an OSM edge id.
    Preference order follows the matching rule semantics:
      1) exact: osm
      2) split forward: osm#*
      3) exact reverse: -osm
      4) split reverse: -osm#*
      5) lexical fallback
    """
    if not candidates:
        raise ValueError("candidates must not be empty.")

    neg_osm_edge = f"-{osm_edge_id}"
    ranked = sorted(
        candidates,
        key=lambda edge_id: (
            0
            if edge_id == osm_edge_id
            else 1
            if edge_id.startswith(f"{osm_edge_id}#")
            else 2
            if edge_id == neg_osm_edge
            else 3
            if edge_id.startswith(f"{neg_osm_edge}#")
            else 4,
            edge_id,
        ),
    )
    return ranked[0]


def filter_matchable_sumo_edges(
    osm_edges: list[str],
    candidate_index: dict[str, list[str]],
    valid_edges: dict[str, object],
) -> tuple[list[str], list[str]]:
    """
    Keep only OSM edges that can be mapped to at least one SUMO edge.
    Returns:
      (kept_sumo_edges, dropped_osm_edges)
    """
    kept_sumo_edges: list[str] = []
    dropped_osm_edges: list[str] = []
    for osm_edge_id in osm_edges:
        candidates = get_candidate_sumo_edges(
            osm_edge_id,
            valid_edges,
            candidate_index=candidate_index,
        )
        if not candidates:
            dropped_osm_edges.append(osm_edge_id)
            continue
        kept_sumo_edges.append(choose_preferred_candidate(osm_edge_id, candidates))
    return kept_sumo_edges, dropped_osm_edges


def prune_unreachable_edges_in_order(
    kept_sumo_edges: list[str],
    graph: dict[str, set[str]],
    *,
    nearby_path_cache: dict[tuple[str, str], tuple[int, list[str]]] | None = None,
    path_cache: PathCache | None = None,
) -> tuple[list[str], list[tuple[str, str, str]]]:
    """
    Enforce ordered reachability through the kept edge sequence.

    Starting from the first kept edge, each subsequent edge must be reachable
    from the last accepted edge. Unreachable edges are dropped and returned
    as log entries: (from_edge, dropped_edge, reason).
    """
    if not kept_sumo_edges:
        return [], []

    accepted: list[str] = [kept_sumo_edges[0]]
    pruned_logs: list[tuple[str, str, str]] = []

    for edge_id in kept_sumo_edges[1:]:
        from_edge = accepted[-1]
        shortest = shortest_path_between_edges(
            graph,
            from_edge,
            edge_id,
            nearby_path_cache=nearby_path_cache,
            path_cache=path_cache,
        )
        if shortest is None:
            pruned_logs.append((from_edge, edge_id, "unreachable_from_previous_kept_edge"))
            continue
        accepted.append(edge_id)

    return accepted, pruned_logs


def to_trip_record(
    vin: str,
    kept_sumo_edges: list[str],
    dropped_osm_edges: list[str],
    *,
    line_no: int | None = None,
) -> MatchedTrip | None:
    """
    Convert kept SUMO edge sequence to one SUMO trip.
    A valid trip needs at least start and end edge.
    """
    if len(kept_sumo_edges) < 2:
        return None

    return MatchedTrip(
        vin=vin,
        start_edge=kept_sumo_edges[0],
        end_edge=kept_sumo_edges[-1],
        via_edges=kept_sumo_edges[1:-1],
        kept_edges=kept_sumo_edges,
        dropped_osm_edges=dropped_osm_edges,
        line_no=line_no,
    )


def build_depart_times(
    record_count: int,
    *,
    depart_begin: float = 0.0,
    depart_end: float = 3600.0,
) -> list[float]:
    if record_count <= 0:
        return []
    if depart_end < depart_begin:
        raise ValueError("--depart-end must be greater than or equal to --depart-begin.")
    if record_count == 1:
        return [depart_begin]

    step = (depart_end - depart_begin) / (record_count - 1)
    return [depart_begin + idx * step for idx in range(record_count)]


def format_depart_time(value: float) -> str:
    return f"{value:.2f}"


def build_unique_vehicle_id(vin: str, seen_counts: dict[str, int]) -> str:
    count = seen_counts.get(vin, 0) + 1
    seen_counts[vin] = count
    if count == 1:
        return vin
    return f"{vin}_{count}"


def write_trips_xml(
    trips: list[MatchedTrip],
    output_file: Path,
    *,
    vehicle_type_id: str = "passenger",
    vehicle_class: str = "passenger",
    depart_begin: float = 0.0,
    depart_end: float = 3600.0,
) -> Path:
    output_path = output_file.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element(
        "routes",
        {f"{{{XSI_NS}}}noNamespaceSchemaLocation": ROUTES_XSD},
    )
    ET.SubElement(root, "vType", {"id": vehicle_type_id, "vClass": vehicle_class})

    seen_counts: dict[str, int] = {}
    depart_times = build_depart_times(
        len(trips),
        depart_begin=depart_begin,
        depart_end=depart_end,
    )

    for idx, trip in enumerate(trips):
        vehicle_id = build_unique_vehicle_id(trip.vin, seen_counts)
        attrs = {
            "id": vehicle_id,
            "type": vehicle_type_id,
            "depart": format_depart_time(depart_times[idx]),
            "from": trip.start_edge,
            "to": trip.end_edge,
        }
        if trip.via_edges:
            attrs["via"] = " ".join(trip.via_edges)
        ET.SubElement(root, "trip", attrs)

    if hasattr(ET, "indent"):
        ET.indent(root, space="    ")
    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def process_trajectories(
    *,
    net_file: Path = DEFAULT_NET_FILE,
    traj_file: Path = DEFAULT_TRAJ_FILE,
    output_file: Path = DEFAULT_OUTPUT_FILE,
    reachability_log_file: Path = DEFAULT_REACHABILITY_LOG_FILE,
    vehicle_type_id: str = DEFAULT_VEHICLE_TYPE_ID,
    vehicle_class: str = DEFAULT_VEHICLE_CLASS,
    depart_begin: float = DEFAULT_DEPART_BEGIN,
    depart_end: float = DEFAULT_DEPART_END,
    preview_count: int = DEFAULT_PREVIEW_COUNT,
) -> tuple[Path, int, int, int, int, int, list[str]]:
    if preview_count < 0:
        raise ValueError("--preview must be non-negative.")

    net = load_sumo_edge_graph(
        net_file.expanduser().resolve(),
        vehicle_class=vehicle_class,
    )
    candidate_index = build_osm_candidate_index(net.edges)
    path_cache: PathCache = {}
    reachability_log_path = reachability_log_file.expanduser().resolve()
    reachability_log_path.parent.mkdir(parents=True, exist_ok=True)

    trips: list[MatchedTrip] = []
    total = 0
    kept_any = 0
    skipped_too_short = 0
    trajectories_with_unreachable = 0
    total_unreachable_pruned = 0
    previews: list[str] = []

    with reachability_log_path.open("w", encoding="utf-8") as log_fh:
        for record in iter_trajectory_file(traj_file.expanduser().resolve(), warn=True):
            total += 1
            kept_sumo_edges, dropped_osm_edges = filter_matchable_sumo_edges(
                record.osm_edges,
                candidate_index,
                net.edges,
            )
            if kept_sumo_edges:
                kept_any += 1

            pruned_sumo_edges, prune_logs = prune_unreachable_edges_in_order(
                kept_sumo_edges,
                net.graph,
                nearby_path_cache=net.nearby_path_cache,
                path_cache=path_cache,
            )
            if prune_logs:
                trajectories_with_unreachable += 1
                total_unreachable_pruned += len(prune_logs)
            for from_edge, dropped_edge, reason in prune_logs:
                line_no = "" if record.line_no is None else str(record.line_no)
                log_fh.write(
                    "\t".join(
                        [
                            record.vin,
                            line_no,
                            reason,
                            from_edge,
                            dropped_edge,
                            " ".join(pruned_sumo_edges),
                        ]
                    )
                    + "\n"
                )

            trip = to_trip_record(
                record.vin,
                pruned_sumo_edges,
                dropped_osm_edges,
                line_no=record.line_no,
            )
            if trip is None:
                skipped_too_short += 1
                continue
            trips.append(trip)
            if len(previews) < preview_count:
                previews.append(
                    f"vin={trip.vin} from={trip.start_edge} to={trip.end_edge} "
                    f"via_count={len(trip.via_edges)} dropped={len(trip.dropped_osm_edges)} "
                    f"pruned_unreachable={len(prune_logs)}"
                )

    output_path = write_trips_xml(
        trips,
        output_file,
        vehicle_type_id=vehicle_type_id,
        vehicle_class=vehicle_class,
        depart_begin=depart_begin,
        depart_end=depart_end,
    )
    return (
        output_path,
        total,
        kept_any,
        skipped_too_short,
        trajectories_with_unreachable,
        total_unreachable_pruned,
        previews,
    )


def main() -> None:
    (
        output_path,
        total,
        kept_any,
        skipped_too_short,
        trajectories_with_unreachable,
        total_unreachable_pruned,
        previews,
    ) = process_trajectories(
        net_file=DEFAULT_NET_FILE,
        traj_file=DEFAULT_TRAJ_FILE,
        output_file=DEFAULT_OUTPUT_FILE,
        reachability_log_file=DEFAULT_REACHABILITY_LOG_FILE,
        vehicle_type_id=DEFAULT_VEHICLE_TYPE_ID,
        vehicle_class=DEFAULT_VEHICLE_CLASS,
        depart_begin=DEFAULT_DEPART_BEGIN,
        depart_end=DEFAULT_DEPART_END,
        preview_count=DEFAULT_PREVIEW_COUNT,
    )

    print(f"Trajectories read      : {total}")
    print(f"Keepable trajectories  : {kept_any}")
    print(f"Skipped (<2 kept edges): {skipped_too_short}")
    print(f"Traj with unreachable  : {trajectories_with_unreachable}")
    print(f"Unreachable edges pruned: {total_unreachable_pruned}")
    print(f"Output route file      : {output_path}")
    print(f"Reachability log file  : {DEFAULT_REACHABILITY_LOG_FILE.resolve()}")
    if previews:
        print("")
        print(f"First {len(previews)} generated trips:")
        for text in previews:
            print(f"- {text}")


__all__ = [
    "DEFAULT_DEPART_BEGIN",
    "DEFAULT_DEPART_END",
    "DEFAULT_OUTPUT_FILE",
    "DEFAULT_REACHABILITY_LOG_FILE",
    "DEFAULT_PREVIEW_COUNT",
    "DEFAULT_VEHICLE_CLASS",
    "DEFAULT_VEHICLE_TYPE_ID",
    "LOG_DIR",
    "MatchedTrip",
    "build_depart_times",
    "build_osm_candidate_index",
    "build_unique_vehicle_id",
    "choose_preferred_candidate",
    "filter_matchable_sumo_edges",
    "format_depart_time",
    "prune_unreachable_edges_in_order",
    "process_trajectories",
    "to_trip_record",
    "write_trips_xml",
]


if __name__ == "__main__":
    main()
