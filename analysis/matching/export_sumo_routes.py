from __future__ import annotations

import argparse
import logging
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MATCHED_FILE = BASE_DIR / "matched_routes.txt"
DEFAULT_OUTPUT_FILE = BASE_DIR / "matched_routes.rou.xml"
DEFAULT_SHUFFLED_OUTPUT_FILE = BASE_DIR / "matched_routes_shuffled_depart.rou.xml"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
ROUTES_XSD = "http://sumo.dlr.de/xsd/routes_file.xsd"

ET.register_namespace("xsi", XSI_NS)
LOGGER = logging.getLogger(__name__)


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


def shuffle_depart_times(
    depart_times: list[float],
    *,
    random_seed: int | None = None,
) -> list[float]:
    shuffled_times = depart_times.copy()
    random.Random(random_seed).shuffle(shuffled_times)
    return shuffled_times


def build_sumo_routes_tree(
    records: list[MatchedRouteRecord],
    *,
    vehicle_type_id: str = "passenger",
    vehicle_class: str = "passenger",
    depart_begin: float = 0.0,
    depart_end: float = 3600.0,
    depart_times: list[float] | None = None,
    allow_reroute: bool = False,
    vehicle_id_prefix: str = "",
    sort_by_depart_time: bool = False,
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
    if depart_times is None:
        depart_times = build_depart_times(
            len(records),
            depart_begin=depart_begin,
            depart_end=depart_end,
        )
    elif len(depart_times) != len(records):
        raise ValueError("Length of depart_times must match record count.")

    record_depart_pairs = list(zip(records, depart_times))
    if sort_by_depart_time:
        record_depart_pairs.sort(key=lambda pair: pair[1])

    for record, depart_time in record_depart_pairs:
        vehicle_id = build_unique_vehicle_id(f"{vehicle_id_prefix}{record.vin}", seen_counts)
        vehicle_attrs = {
            "id": vehicle_id,
            "type": vehicle_type_id,
            "depart": format_depart_time(depart_time),
        }
        if allow_reroute:
            vehicle_attrs["reroute"] = "true"
        vehicle_elem = ET.SubElement(
            root,
            "vehicle",
            vehicle_attrs,
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
    LOGGER.info("Loaded %d matched trajectories from %s", len(records), Path(matched_file))
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
    LOGGER.info("Wrote SUMO route file: %s", output_path)
    return output_path


def write_sumo_routes_files_with_shuffled_depart(
    matched_file: str | Path,
    output_file: str | Path,
    shuffled_output_file: str | Path,
    *,
    vehicle_type_id: str = "passenger",
    shuffled_vehicle_type_id: str | None = None,
    vehicle_class: str = "passenger",
    depart_begin: float = 0.0,
    depart_end: float = 3600.0,
    random_seed: int | None = None,
    shuffled_vehicle_id_prefix: str = "shuffled_",
) -> tuple[Path, Path]:
    records = iter_matched_route_file(matched_file)
    LOGGER.info("Loaded %d matched trajectories from %s", len(records), Path(matched_file))
    normal_output_path = Path(output_file).expanduser().resolve()
    shuffled_output_path = Path(shuffled_output_file).expanduser().resolve()
    normal_output_path.parent.mkdir(parents=True, exist_ok=True)
    shuffled_output_path.parent.mkdir(parents=True, exist_ok=True)

    base_depart_times = build_depart_times(
        len(records),
        depart_begin=depart_begin,
        depart_end=depart_end,
    )
    if shuffled_vehicle_type_id is None:
        shuffled_vehicle_type_id = f"{vehicle_type_id}_shuffled"
    if shuffled_vehicle_type_id == vehicle_type_id:
        shuffled_vehicle_type_id = f"{vehicle_type_id}_shuffled"
        LOGGER.warning(
            "shuffled_vehicle_type_id equals vehicle_type_id; auto-adjusted to %s to avoid duplicate vType IDs",
            shuffled_vehicle_type_id,
        )
    LOGGER.info(
        "Generated base depart times in range [%.2f, %.2f] for %d vehicles",
        depart_begin,
        depart_end,
        len(records),
    )

    normal_tree = build_sumo_routes_tree(
        records,
        vehicle_type_id=vehicle_type_id,
        vehicle_class=vehicle_class,
        depart_times=base_depart_times,
    )
    normal_tree.write(normal_output_path, encoding="utf-8", xml_declaration=True)
    LOGGER.info("Wrote standard SUMO route file: %s", normal_output_path)

    shuffled_tree = build_sumo_routes_tree(
        records,
        vehicle_type_id=shuffled_vehicle_type_id,
        vehicle_class=vehicle_class,
        depart_times=shuffle_depart_times(base_depart_times, random_seed=random_seed),
        allow_reroute=True,
        vehicle_id_prefix=shuffled_vehicle_id_prefix,
        sort_by_depart_time=True,
    )
    shuffled_tree.write(shuffled_output_path, encoding="utf-8", xml_declaration=True)
    LOGGER.info(
        "Wrote shuffled SUMO route file with reroute enabled: %s (shuffle_seed=%s, vType=%s, vehicle_id_prefix=%s)",
        shuffled_output_path,
        random_seed,
        shuffled_vehicle_type_id,
        shuffled_vehicle_id_prefix,
    )

    return normal_output_path, shuffled_output_path


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
        "--shuffled-output",
        type=Path,
        default=DEFAULT_SHUFFLED_OUTPUT_FILE,
        help="Path to additional SUMO route .rou.xml file with shuffled depart times and reroute enabled.",
    )
    parser.add_argument(
        "--vehicle-type-id",
        default="passenger",
        help="SUMO vehicle type id written into <vType> and <vehicle type=...>.",
    )
    parser.add_argument(
        "--shuffled-vehicle-type-id",
        default=None,
        help="Optional vType id for shuffled output. Defaults to '<vehicle-type-id>_shuffled'.",
    )
    parser.add_argument(
        "--shuffled-vehicle-id-prefix",
        default="shuffled_",
        help="Vehicle id prefix for shuffled output to avoid duplicate ids across route files.",
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
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Optional random seed for shuffled depart times.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.depart_end < args.depart_begin:
        raise ValueError("--depart-end must be greater than or equal to --depart-begin.")

    write_sumo_routes_files_with_shuffled_depart(
        matched_file=args.matched,
        output_file=args.output,
        shuffled_output_file=args.shuffled_output,
        vehicle_type_id=args.vehicle_type_id,
        shuffled_vehicle_type_id=args.shuffled_vehicle_type_id,
        vehicle_class=args.vehicle_class,
        depart_begin=args.depart_begin,
        depart_end=args.depart_end,
        random_seed=args.shuffle_seed,
        shuffled_vehicle_id_prefix=args.shuffled_vehicle_id_prefix,
    )
    LOGGER.info(
        "Export completed. standard=%s shuffled=%s",
        Path(args.output).expanduser().resolve(),
        Path(args.shuffled_output).expanduser().resolve(),
    )


__all__ = [
    "DEFAULT_MATCHED_FILE",
    "DEFAULT_OUTPUT_FILE",
    "DEFAULT_SHUFFLED_OUTPUT_FILE",
    "MatchedRouteRecord",
    "build_depart_times",
    "build_sumo_routes_tree",
    "iter_matched_route_file",
    "parse_matched_route_line",
    "shuffle_depart_times",
    "write_sumo_routes_files_with_shuffled_depart",
    "write_sumo_routes_file",
]


if __name__ == "__main__":
    main()
