from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = BASE_DIR.parent
DEFAULT_INPUT_FILE = ANALYSIS_DIR / "basic_data.rou.xml"
DEFAULT_SAMPLED_COLOR = "255,80,80"
DEFAULT_UNSAMPLED_COLOR = "160,160,160"
DEFAULT_SEED = 42
XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"

ET.register_namespace("xsi", XSI_NS)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_penetration(value: float) -> float:
    if value < 0 or value > 100:
        raise ValueError("Penetration rate must be between 0 and 100.")
    return value


def format_number_for_filename(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).replace(".", "_")


def output_name_for_penetration(penetration: float, seed: int | None) -> str:
    penetration_text = format_number_for_filename(penetration)
    seed_text = "none" if seed is None else str(seed)
    return f"{penetration_text}_{seed_text}.rou.xml"


def sample_name_for_penetration(penetration: float, seed: int | None) -> str:
    penetration_text = format_number_for_filename(penetration)
    seed_text = "none" if seed is None else str(seed)
    return f"{penetration_text}_{seed_text}_sample.txt"


def select_vehicle_indices(
    vehicle_count: int,
    penetration: float,
    *,
    seed: int | None = None,
) -> set[int]:
    sample_count = round(vehicle_count * penetration / 100)
    if sample_count <= 0:
        return set()

    rng = random.Random(seed)
    return set(rng.sample(range(vehicle_count), sample_count))


def apply_penetration_sample(
    input_file: Path,
    output_file: Path,
    sample_file: Path,
    penetration: float,
    *,
    seed: int | None = None,
) -> tuple[int, int]:
    if not input_file.exists():
        raise FileNotFoundError(f"Input route file not found: {input_file}")

    tree = ET.parse(input_file)
    root = tree.getroot()
    if local_name(root.tag) != "routes":
        raise ValueError(f"Input route file root is not <routes>: {input_file}")

    vehicles = [element for element in root if local_name(element.tag) == "vehicle"]
    selected_indices = select_vehicle_indices(len(vehicles), penetration, seed=seed)
    selected_vehicle_ids: list[str] = []

    for index, vehicle in enumerate(vehicles):
        if index in selected_indices:
            vehicle.set("color", DEFAULT_SAMPLED_COLOR)
            vehicle_id = vehicle.get("id")
            if vehicle_id is None:
                raise ValueError("Selected vehicle is missing required id attribute.")
            selected_vehicle_ids.append(vehicle_id)
        else:
            vehicle.set("color", DEFAULT_UNSAMPLED_COLOR)

    ET.indent(root, space="    ")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    sample_file.write_text("\n".join(selected_vehicle_ids) + "\n", encoding="utf-8")
    return len(selected_indices), len(vehicles)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample connected vehicles from basic_data.rou.xml by penetration rate.",
    )
    parser.add_argument(
        "-p",
        "--penetration",
        type=float,
        default=30.0,
        help="Penetration rate percentage, for example 30 for 30%%.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Input route XML file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output route XML file. Defaults to "
            "analysis/simulation/penetration/<rate>_<seed>.rou.xml."
        ),
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=None,
        help=(
            "Output text file for sampled vehicle ids. Defaults to "
            "analysis/simulation/penetration/<rate>_<seed>_sample.txt."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducible sampling. Defaults to 42.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    penetration = parse_penetration(args.penetration)
    output_file = args.output or BASE_DIR / output_name_for_penetration(
        penetration,
        args.seed,
    )
    sample_file = args.sample_output or BASE_DIR / sample_name_for_penetration(
        penetration,
        args.seed,
    )
    selected_count, vehicle_count = apply_penetration_sample(
        args.input,
        output_file,
        sample_file,
        penetration,
        seed=args.seed,
    )
    print(
        f"Selected {selected_count}/{vehicle_count} vehicles "
        f"({penetration:g}%) into {output_file}; ids written to {sample_file}"
    )


if __name__ == "__main__":
    main()
