from __future__ import annotations

import argparse
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MATCHED_FILE = BASE_DIR / "matched_routes.txt"
DEFAULT_OUTPUT_FILE = BASE_DIR / "matched_routes.rou.xml"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
ROUTES_XSD = "http://sumo.dlr.de/xsd/routes_file.xsd"

ET.register_namespace("xsi", XSI_NS)
LOGGER = logging.getLogger(__name__)

TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")


@dataclass(slots=True)
class MatchedRouteRecord:
    vin: str
    start_time: str
    sumo_edges: list[str]
    line_no: int | None = None


def parse_matched_route_line(
    line: str,
    line_no: int | None = None,
) -> MatchedRouteRecord | None:
    """
    Parse one successful matched trajectory line:
        vin start_time sumo_edge_1 sumo_edge_2 ...
    where start_time is the HH:MM:SS entry time of the first edge.
    """
    text = line.strip()
    if not text:
        return None

    tokens = text.split()
    if len(tokens) < 3:
        raise ValueError(
            f"Invalid matched route line {line_no}: expected vin, start time and at least one SUMO edge."
        )
    if not TIME_RE.fullmatch(tokens[1]):
        raise ValueError(
            f"Invalid matched route line {line_no}: second field must be HH:MM:SS start time, got {tokens[1]!r}."
        )

    return MatchedRouteRecord(
        vin=tokens[0],
        start_time=tokens[1],
        sumo_edges=tokens[2:],
        line_no=line_no,
    )


def iter_matched_route_file(txt_file: str | Path) -> list[MatchedRouteRecord]:
    """
    Load all successful matched trajectories from `matched_routes.txt`.
    """
    matched_path = Path(txt_file)
    if not matched_path.exists():
        raise FileNotFoundError(f"Matched route file not found: {matched_path}")

    records: list[MatchedRouteRecord] = []
    try:
        with matched_path.open("r", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                record = parse_matched_route_line(raw_line, line_no=line_no)
                if record is not None:
                    records.append(record)
    except OSError as exc:
        raise OSError(f"Failed to read matched route file: {matched_path}") from exc

    return records


def build_unique_vehicle_id(vin: str, seen_counts: dict[str, int]) -> str:
    """
    Use VIN as the base vehicle id and suffix duplicates if needed.
    """
    count = seen_counts.get(vin, 0) + 1
    seen_counts[vin] = count
    if count == 1:
        return vin
    return f"{vin}_{count}"


def format_depart_time(value: float) -> str:
    return f"{value:.2f}"


def start_time_to_seconds(start_time: str) -> float:
    hours, minutes, seconds = start_time.split(":")
    return float(int(hours) * 3600 + int(minutes) * 60 + int(seconds))


def build_depart_times(records: list[MatchedRouteRecord]) -> list[float]:
    """
    Use each trajectory's first-edge entry time (start_time) as its depart time,
    converted to seconds since midnight.
    """
    return [start_time_to_seconds(record.start_time) for record in records]


def build_sumo_routes_tree(
    records: list[MatchedRouteRecord],
    *,
    vehicle_type_id: str = "passenger",
    vehicle_class: str = "passenger",
) -> ET.ElementTree:
    """
    Convert successful matched trajectories into a SUMO `.rou.xml` tree.

    Depart times use each trajectory's start_time. Vehicles are sorted
    by depart time because SUMO requires route files sorted by departure.
    """
    root = ET.Element(
        "routes",
        {f"{{{XSI_NS}}}noNamespaceSchemaLocation": ROUTES_XSD},
    )
    ET.SubElement(root, "vType", {"id": vehicle_type_id, "vClass": vehicle_class})

    seen_counts: dict[str, int] = {}
    depart_times = build_depart_times(records)

    record_depart_pairs = sorted(zip(records, depart_times), key=lambda pair: pair[1])

    for record, depart_time in record_depart_pairs:
        vehicle_id = build_unique_vehicle_id(record.vin, seen_counts)
        vehicle_elem = ET.SubElement(
            root,
            "vehicle",
            {
                "id": vehicle_id,
                "type": vehicle_type_id,
                "depart": format_depart_time(depart_time),
            },
        )
        ET.SubElement(vehicle_elem, "route", {"edges": " ".join(record.sumo_edges)})

    if hasattr(ET, "indent"):
        ET.indent(root, space="    ")
    return ET.ElementTree(root)


def write_sumo_routes_file(
    matched_file: str | Path,
    output_file: str | Path,
    *,
    vehicle_type_id: str = "passenger",
    vehicle_class: str = "passenger",
) -> Path:
    records = iter_matched_route_file(matched_file)
    LOGGER.info("Loaded %d matched trajectories from %s", len(records), Path(matched_file))
    output_path = Path(output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tree = build_sumo_routes_tree(
        records,
        vehicle_type_id=vehicle_type_id,
        vehicle_class=vehicle_class,
    )
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    LOGGER.info("Wrote SUMO route file: %s", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert successful matched trajectories into a SUMO route file."
    )
    parser.add_argument(
        "--matched",
        type=Path,
        default=DEFAULT_MATCHED_FILE,
        help="Path to matched trajectory txt file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Path to output SUMO route .rou.xml file.",
    )
    parser.add_argument(
        "--vehicle-type-id",
        default="passenger",
        help="SUMO vehicle type id written into <vType> and <vehicle type=...>.",
    )
    parser.add_argument(
        "--vehicle-class",
        default="passenger",
        help="SUMO vehicle class written into <vType vClass=...>.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    write_sumo_routes_file(
        matched_file=args.matched,
        output_file=args.output,
        vehicle_type_id=args.vehicle_type_id,
        vehicle_class=args.vehicle_class,
    )
    LOGGER.info(
        "Export completed. output=%s",
        Path(args.output).expanduser().resolve(),
    )


__all__ = [
    "DEFAULT_MATCHED_FILE",
    "DEFAULT_OUTPUT_FILE",
    "MatchedRouteRecord",
    "build_depart_times",
    "build_sumo_routes_tree",
    "iter_matched_route_file",
    "parse_matched_route_line",
    "write_sumo_routes_file",
]


if __name__ == "__main__":
    main()
