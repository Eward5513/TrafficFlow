from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = BASE_DIR.parent
DEFAULT_INPUT_FILES = [
    BASE_DIR / "random_routes_1w.rou.xml",
    ANALYSIS_DIR / "matching" / "matched_routes.rou.xml",
    ANALYSIS_DIR / "matching" / "matched_routes_shuffled_depart.rou.xml",
]
DEFAULT_OUTPUT_FILE = BASE_DIR / "basic_data.rou.xml"
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"

ET.register_namespace("xsi", XSI_NS)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def depart_sort_value(vehicle: ET.Element) -> float:
    depart = vehicle.get("depart", "0")
    try:
        return float(depart)
    except ValueError:
        return float("inf")


def load_route_elements(
    source: Path,
) -> tuple[ET.Element, list[ET.Element], list[ET.Element], list[ET.Element]]:
    if not source.exists():
        raise FileNotFoundError(f"Input route file not found: {source}")

    root = ET.parse(source).getroot()
    if local_name(root.tag) != "routes":
        raise ValueError(f"Input route file root is not <routes>: {source}")

    vtypes: list[ET.Element] = []
    vehicles: list[ET.Element] = []
    others: list[ET.Element] = []
    for child in root:
        child_name = local_name(child.tag)
        if child_name == "vType":
            vtypes.append(child)
        elif child_name == "vehicle":
            vehicles.append(child)
        else:
            others.append(child)

    return root, vtypes, others, vehicles


def merge_route_files(input_files: list[Path], output_file: Path) -> None:
    if not input_files:
        raise ValueError("At least one input route file is required.")

    merged_root: ET.Element | None = None
    vtype_by_id: dict[str, ET.Element] = {}
    other_elements: list[ET.Element] = []
    vehicles_with_order: list[tuple[int, ET.Element]] = []

    for input_file in input_files:
        root, vtypes, others, vehicles = load_route_elements(input_file)
        if merged_root is None:
            merged_root = ET.Element(root.tag, root.attrib)

        for vtype in vtypes:
            vtype_id = vtype.get("id")
            if vtype_id is None:
                other_elements.append(copy.deepcopy(vtype))
                continue

            existing = vtype_by_id.get(vtype_id)
            if existing is None:
                vtype_by_id[vtype_id] = copy.deepcopy(vtype)
            elif ET.tostring(existing) != ET.tostring(vtype):
                raise ValueError(f"Conflicting vType definition for id={vtype_id!r}.")

        other_elements.extend(copy.deepcopy(element) for element in others)
        vehicles_with_order.extend(
            (len(vehicles_with_order), copy.deepcopy(vehicle)) for vehicle in vehicles
        )

    if merged_root is None:
        raise ValueError("No route files were loaded.")

    vehicles_with_order.sort(key=lambda item: (depart_sort_value(item[1]), item[0]))
    for element in [*vtype_by_id.values(), *other_elements]:
        merged_root.append(element)
    for _, vehicle in vehicles_with_order:
        merged_root.append(vehicle)

    ET.indent(merged_root, space="    ")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(merged_root).write(
        output_file,
        encoding="utf-8",
        xml_declaration=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge route XML files into analysis/simulation/basic_data.rou.xml.",
    )
    parser.add_argument(
        "input_files",
        nargs="*",
        type=Path,
        default=DEFAULT_INPUT_FILES,
        help="Route XML files to merge. Defaults to the three basic data route files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Output route XML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_route_files(args.input_files, args.output)
    print(f"Merged {len(args.input_files)} files into {args.output}")


if __name__ == "__main__":
    main()
