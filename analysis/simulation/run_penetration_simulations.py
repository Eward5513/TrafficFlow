from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TextIO


BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_FILE = BASE_DIR / "simulation_output.sumocfg"
CONFIG_DIR = BASE_DIR / "sumocfg"
ROUTE_GENERATOR = BASE_DIR / "penetration" / "generate_penetration_routes.py"
SEED = 42
PENETRATIONS = tuple(float(value) for value in range(5, 21)) + (30.0, 40.0, 50.0)
SUMO_BINARY = "sumo"


def format_number_for_filename(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).replace(".", "_")


def required_child(root: ET.Element, parent_tag: str, child_tag: str) -> ET.Element:
    parent = root.find(parent_tag)
    if parent is None:
        raise ValueError(f"missing <{parent_tag}> section in template")

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


def config_name_for_penetration(penetration: float, seed: int) -> str:
    penetration_text = format_number_for_filename(penetration)
    return f"simulation_output_{penetration_text}_{seed}.sumocfg"


def log_name_for_penetration(penetration: float, seed: int) -> str:
    penetration_text = format_number_for_filename(penetration)
    return f"simulation_output_{penetration_text}_{seed}.log"


def route_file_for_penetration(penetration: float, seed: int) -> Path:
    penetration_text = format_number_for_filename(penetration)
    return BASE_DIR / "penetration" / f"{penetration_text}_{seed}.rou.xml"


def sample_file_for_penetration(penetration: float, seed: int) -> Path:
    penetration_text = format_number_for_filename(penetration)
    return BASE_DIR / "penetration" / f"{penetration_text}_{seed}_sample.txt"


def output_dir_for_penetration(penetration: float) -> Path:
    penetration_text = format_number_for_filename(penetration)
    return BASE_DIR / "output" / penetration_text


def expected_output_files_for_penetration(penetration: float) -> tuple[Path, ...]:
    output_dir = output_dir_for_penetration(penetration)
    return (
        output_dir / "fcd.csv",
        output_dir / "tripinfo.xml",
        output_dir / "vehroute.xml",
        output_dir / "summary.xml",
        output_dir / "statistics.xml",
    )


def simulation_output_exists(penetration: float) -> bool:
    return all(path.exists() for path in expected_output_files_for_penetration(penetration))


def ensure_penetration_routes(penetrations: tuple[float, ...], seed: int) -> None:
    if not ROUTE_GENERATOR.exists():
        raise FileNotFoundError(f"route generator not found: {ROUTE_GENERATOR}")

    for penetration in penetrations:
        penetration_text = format_number_for_filename(penetration)
        route_file = route_file_for_penetration(penetration, seed)
        sample_file = sample_file_for_penetration(penetration, seed)
        if route_file.exists() and sample_file.exists():
            print(f"Skipped {penetration_text}% sample, files already exist")
            continue

        command = [
            sys.executable,
            str(ROUTE_GENERATOR),
            "--penetration",
            str(penetration),
            "--seed",
            str(seed),
        ]
        print(f"Generating {penetration_text}% sample with seed {seed}")
        subprocess.run(command, cwd=BASE_DIR, check=True)


def generate_config(
    template_file: Path,
    penetration: float,
    seed: int,
) -> Path:
    if not template_file.exists():
        raise FileNotFoundError(f"template config not found: {template_file}")

    penetration_text = format_number_for_filename(penetration)
    route_file = route_file_for_penetration(penetration, seed)
    if not route_file.exists():
        raise FileNotFoundError(
            f"route file not found: {route_file}. "
            "Generate it first with penetration/generate_penetration_routes.py."
        )

    output_dir = output_dir_for_penetration(penetration)
    output_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(template_file)
    root = tree.getroot()

    set_config_value(root, "input", "net-file", "../../road_network/net_tls.net.xml")
    set_config_value(
        root,
        "input",
        "route-files",
        f"../penetration/{penetration_text}_{seed}.rou.xml",
    )
    set_config_value(root, "processing", "seed", str(seed))
    set_config_value(
        root,
        "output",
        "fcd-output",
        f"../output/{penetration_text}/fcd.csv",
    )
    set_config_value(
        root,
        "output",
        "tripinfo-output",
        f"../output/{penetration_text}/tripinfo.xml",
    )
    set_config_value(
        root,
        "output",
        "vehroute-output",
        f"../output/{penetration_text}/vehroute.xml",
    )
    set_config_value(
        root,
        "output",
        "summary-output",
        f"../output/{penetration_text}/summary.xml",
    )
    set_config_value(
        root,
        "output",
        "statistic-output",
        f"../output/{penetration_text}/statistics.xml",
    )

    ET.indent(tree, space="    ")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_file = CONFIG_DIR / config_name_for_penetration(penetration, seed)
    tree.write(config_file, encoding="utf-8", xml_declaration=False)
    return config_file


def start_sumo(
    config_file: Path,
    log_file: Path,
    sumo_binary: str,
) -> tuple[subprocess.Popen, TextIO]:
    command = [sumo_binary, "-c", str(config_file.relative_to(BASE_DIR))]
    log_handle = log_file.open("w", encoding="utf-8")
    log_handle.write(f"Running: {' '.join(command)}\n\n")
    log_handle.flush()
    process = subprocess.Popen(
        command,
        cwd=BASE_DIR,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return process, log_handle


def main() -> None:
    processes: list[tuple[str, Path, subprocess.Popen, TextIO]] = []

    ensure_penetration_routes(PENETRATIONS, SEED)

    for penetration in PENETRATIONS:
        penetration_text = format_number_for_filename(penetration)
        already_simulated = simulation_output_exists(penetration)
        config_file = generate_config(TEMPLATE_FILE, penetration, SEED)
        log_file = CONFIG_DIR / log_name_for_penetration(penetration, SEED)
        print(f"Generated {config_file.name} for {penetration_text}%")

        if already_simulated:
            print(f"Skipped {penetration_text}% simulation, output already exists")
            continue

        process, log_handle = start_sumo(config_file, log_file, SUMO_BINARY)
        processes.append((penetration_text, log_file, process, log_handle))
        print(f"Started {penetration_text}% simulation, log: {log_file.name}")

    failed_rates: list[str] = []
    for penetration_text, log_file, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code == 0:
            print(f"Finished {penetration_text}% simulation, log: {log_file.name}")
        else:
            failed_rates.append(penetration_text)
            print(
                f"Failed {penetration_text}% simulation with code {return_code}, "
                f"log: {log_file.name}"
            )

    if failed_rates:
        raise SystemExit(f"SUMO failed for penetration rates: {', '.join(failed_rates)}")


if __name__ == "__main__":
    main()
