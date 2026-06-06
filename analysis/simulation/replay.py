"""Open SUMO GUI from a saved simulation state.

Usage:
    python replay.py 105 -p 50

This launches:
    sumo-gui -c sumocfg/simulation_replay_50_42.sumocfg \
      --load-state state/50/state_105.00.xml.gz \
      --begin 105
"""

from __future__ import annotations

import argparse
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SUMO_GUI = "sumo-gui"
TEMPLATE_FILE = BASE_DIR / "simulation_replay.sumocfg"
CONFIG_DIR = BASE_DIR / "sumocfg"
DEFAULT_PENETRATION = 50.0
DEFAULT_SEED = 42


def format_number_for_filename(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).replace(".", "_")


def parse_time(value: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid time: {value}") from exc


def parse_penetration(value: str) -> float:
    try:
        penetration = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid penetration rate: {value}") from exc

    if penetration < 0 or penetration > 100:
        raise argparse.ArgumentTypeError(
            f"penetration rate must be between 0 and 100: {value}"
        )
    return penetration


def required_child(root: ET.Element, parent_tag: str, child_tag: str) -> ET.Element:
    parent = root.find(parent_tag)
    if parent is None:
        raise ValueError(f"missing <{parent_tag}> section in replay template")

    child = parent.find(child_tag)
    if child is None:
        raise ValueError(f"missing <{child_tag}> in <{parent_tag}> section")
    return child


def set_config_value(
    root: ET.Element,
    parent_tag: str,
    child_tag: str,
    value: str,
) -> None:
    required_child(root, parent_tag, child_tag).set("value", value)


def replay_config_name(penetration: float, seed: int) -> str:
    penetration_text = format_number_for_filename(penetration)
    return f"simulation_replay_{penetration_text}_{seed}.sumocfg"


def generate_replay_config(penetration: float, seed: int) -> Path:
    if not TEMPLATE_FILE.exists():
        raise FileNotFoundError(f"replay template not found: {TEMPLATE_FILE}")

    penetration_text = format_number_for_filename(penetration)
    route_file = BASE_DIR / "penetration" / f"{penetration_text}_{seed}.rou.xml"
    if not route_file.exists():
        raise FileNotFoundError(f"route file not found: {route_file}")

    tree = ET.parse(TEMPLATE_FILE)
    root = tree.getroot()
    set_config_value(root, "input", "net-file", "../../road_network/net_tls.net.xml")
    set_config_value(
        root,
        "input",
        "route-files",
        f"../penetration/{penetration_text}_{seed}.rou.xml",
    )
    set_config_value(root, "processing", "seed", str(seed))

    ET.indent(tree, space="    ")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_file = CONFIG_DIR / replay_config_name(penetration, seed)
    tree.write(config_file, encoding="utf-8", xml_declaration=False)
    return config_file


def state_file_for_time(time_seconds: float, penetration: float) -> Path:
    penetration_text = format_number_for_filename(penetration)
    return BASE_DIR / "state" / penetration_text / f"state_{time_seconds:.2f}.xml.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open sumo-gui with simulation_replay.sumocfg and a saved state."
    )
    parser.add_argument("time", type=parse_time, help="Replay time in seconds, e.g. 105")
    parser.add_argument(
        "-p",
        "--penetration",
        type=parse_penetration,
        default=DEFAULT_PENETRATION,
        help=(
            "State folder under analysis/simulation/state "
            f"(default: {format_number_for_filename(DEFAULT_PENETRATION)})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed used in penetration route filenames. Defaults to {DEFAULT_SEED}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_file = state_file_for_time(args.time, args.penetration)
    config_file = generate_replay_config(args.penetration, args.seed)

    if not state_file.exists():
        raise FileNotFoundError(f"state file not found: {state_file}")

    command = [
        SUMO_GUI,
        "-c",
        str(config_file.relative_to(BASE_DIR)),
        "--load-state",
        str(state_file.relative_to(BASE_DIR)),
        "--begin",
        f"{args.time:g}",
    ]
    print(f"Generated {config_file.relative_to(BASE_DIR)}")
    subprocess.run(command, cwd=BASE_DIR, check=True)


if __name__ == "__main__":
    main()
