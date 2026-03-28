from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MATCHED_FILE = BASE_DIR / "matched_routes.txt"
DEFAULT_OUTPUT_FILE = BASE_DIR / "matched_routes.rou.xml"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
ROUTES_XSD = "http://sumo.dlr.de/xsd/routes_file.xsd"

ET.register_namespace("xsi", XSI_NS)


@dataclass(slots=True)
class MatchedRouteRecord:
    vin: str
    sumo_edges: list[str]
    line_no: int | None = None


def parse_matched_route_line(
    line: str,
    line_no: int | None = None,
) -> MatchedRouteRecord | None:
    """
    Parse one successful matched trajectory line:
        vin sumo_edge_1 sumo_edge_2 ...
    """
    text = line.strip()
    if not text:
        return None

    tokens = text.split()
    if len(tokens) < 2:
        raise ValueError(
            f"Invalid matched route line {line_no}: expected vin and at least one SUMO edge."
        )

    return MatchedRouteRecord(
        vin=tokens[0],
        sumo_edges=tokens[1:],
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


def build_depart_times(
    record_count: int,
    *,
    depart_begin: float = 0.0,
    depart_end: float = 3600.0,
) -> list[float]:
    """
    Distribute all vehicles within the inclusive depart time range.
    """
    if record_count <= 0:
        return []
    if depart_end < depart_begin:
        raise ValueError("--depart-end must be greater than or equal to --depart-begin.")
    if record_count == 1:
        return [depart_begin]

    step = (depart_end - depart_begin) / (record_count - 1)
    return [depart_begin + index * step for index in range(record_count)]


def build_sumo_routes_tree(
    records: list[MatchedRouteRecord],
    *,
    vehicle_type_id: str = "passenger",
    vehicle_class: str = "passenger",
    depart_begin: float = 0.0,
    depart_end: float = 3600.0,
) -> ET.ElementTree:
    """
    Convert successful matched trajectories into a SUMO `.rou.xml` tree.
    """
    root = ET.Element(
        "routes",
        {f"{{{XSI_NS}}}noNamespaceSchemaLocation": ROUTES_XSD},
    )
    ET.SubElement(root, "vType", {"id": vehicle_type_id, "vClass": vehicle_class})

    seen_counts: dict[str, int] = {}
    depart_times = build_depart_times(
        len(records),
        depart_begin=depart_begin,
        depart_end=depart_end,
    )
    for index, record in enumerate(records):
        vehicle_id = build_unique_vehicle_id(record.vin, seen_counts)
        vehicle_elem = ET.SubElement(
            root,
            "vehicle",
            {
                "id": vehicle_id,
                "type": vehicle_type_id,
                "depart": format_depart_time(depart_times[index]),
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
    depart_begin: float = 0.0,
    depart_end: float = 3600.0,
) -> Path:
    records = iter_matched_route_file(matched_file)
    output_path = Path(output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tree = build_sumo_routes_tree(
        records,
        vehicle_type_id=vehicle_type_id,
        vehicle_class=vehicle_class,
        depart_begin=depart_begin,
        depart_end=depart_end,
    )
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
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
    parser.add_argument(
        "--depart-begin",
        type=float,
        default=0.0,
        help="Depart time of the first vehicle.",
    )
    parser.add_argument(
        "--depart-end",
        type=float,
        default=3600.0,
        help="Latest allowed depart time; all vehicles are evenly distributed in range.",
    )
    args = parser.parse_args()

    if args.depart_end < args.depart_begin:
        raise ValueError("--depart-end must be greater than or equal to --depart-begin.")

    write_sumo_routes_file(
        matched_file=args.matched,
        output_file=args.output,
        vehicle_type_id=args.vehicle_type_id,
        vehicle_class=args.vehicle_class,
        depart_begin=args.depart_begin,
        depart_end=args.depart_end,
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
