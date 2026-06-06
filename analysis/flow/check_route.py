"""Print the edge sequence for one vehicle from FCD CSV.

Usage:
    python3 check_route.py 121
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
FCD_CSV_FILE = BASE_DIR / "simulation" / "output" / "50" / "fcd.csv"

TIME_FIELDS = ("time", "timestep_time", "timestep")
VEHICLE_ID_FIELDS = ("id", "vehicle_id", "vehicle")
EDGE_FIELDS = ("edge", "vehicle_edge")
LANE_FIELDS = ("lane", "vehicle_lane")


def edge_from_lane(lane_id: str | None) -> str | None:
    if not lane_id or "_" not in lane_id:
        return None

    edge_id, lane_index = lane_id.rsplit("_", 1)
    if not lane_index.isdigit():
        return None
    return edge_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print route observed in FCD for one vehicle.")
    parser.add_argument("vehicle_id", help="Vehicle id, e.g. 121")
    return parser.parse_args()


def field_index(header: list[str], names: tuple[str, ...]) -> int:
    for name in names:
        if name in header:
            return header.index(name)
    raise ValueError(f"missing field, expected one of: {names}")


def optional_field_index(header: list[str], names: tuple[str, ...]) -> int | None:
    for name in names:
        if name in header:
            return header.index(name)
    return None


def print_vehicle_route(vehicle_id: str) -> None:
    if not FCD_CSV_FILE.exists():
        raise FileNotFoundError(f"FCD CSV file not found: {FCD_CSV_FILE}")

    route: list[dict[str, str | float | None]] = []
    last_edge: str | None = None
    seen_vehicle = False

    with FCD_CSV_FILE.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        time_index = field_index(header, TIME_FIELDS)
        vehicle_index = field_index(header, VEHICLE_ID_FIELDS)
        lane_index = field_index(header, LANE_FIELDS)
        edge_index = optional_field_index(header, EDGE_FIELDS)

        for row in reader:
            current_vehicle_id = row[vehicle_index]
            if current_vehicle_id != vehicle_id:
                if seen_vehicle:
                    # FCD is time-ordered, so the same vehicle can appear again later.
                    # Keep scanning until EOF to avoid missing sparse records.
                    pass
                continue

            seen_vehicle = True
            time = float(row[time_index])
            lane_id = row[lane_index]
            edge_id = row[edge_index] if edge_index is not None and row[edge_index] else None
            if not edge_id:
                edge_id = edge_from_lane(lane_id)

            if edge_id is None or edge_id == last_edge:
                continue

            route.append(
                {
                    "time": time,
                    "edge": edge_id,
                    "lane": lane_id,
                }
            )
            last_edge = edge_id

    if not route:
        print(f"vehicle {vehicle_id} not found in {FCD_CSV_FILE}")
        return

    print(f"vehicle_id: {vehicle_id}")
    print(f"fcd_file: {FCD_CSV_FILE}")
    print("route:")
    for index, item in enumerate(route, start=1):
        print(
            f'{index:>4}. time={item["time"]:>8.2f} '
            f'edge={item["edge"]} lane={item["lane"]}'
        )


def main() -> None:
    args = parse_args()
    print_vehicle_route(args.vehicle_id)


if __name__ == "__main__":
    main()
