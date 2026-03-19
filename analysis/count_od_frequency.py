#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count origin-destination (OD) edge-pair frequencies across all vehicles.

Each vehicle contributes one OD pair: (first_edge, last_edge).
The output can be used to sample correlated start/end edges for
synthetic route generation, replacing independent marginal sampling.

Supported input formats (same as count_edge_frequency.py):
  - SUMO .rou.xml
  - route_by_edge.txt  (vin hh:mm:ss edge_id hh:mm:ss edge_id ...)

Output JSON structure:
  {
    "source_file": "...",
    "vehicle_count": N,
    "route_count_with_edges": N,
    "unique_origin_count": N,
    "unique_destination_count": N,
    "unique_od_pair_count": N,
    "total_od_occurrences": N,
    "od_pairs": [
      {"origin": "A", "destination": "B", "count": N},
      ...                                              // sorted by count desc
    ]
  }

Usage:
  python analysis/count_od_frequency.py
  python analysis/count_od_frequency.py --input data/route_by_edge.txt
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

HMS_TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def simplify_edge_id(edge_id: str) -> str:
    token = edge_id.strip()
    if token.startswith("-"):
        token = token[1:]
    return token.split("#", 1)[0]


def od_from_rou_xml(
    xml_path: Path, simplify: bool
) -> tuple[Counter[tuple[str, str]], int, int]:
    od_counter: Counter[tuple[str, str]] = Counter()
    vehicle_count = 0
    route_count = 0

    for _, elem in ET.iterparse(str(xml_path), events=("end",)):
        tag = local_name(elem.tag)

        if tag == "vehicle":
            vehicle_count += 1

        if tag == "route":
            raw = elem.attrib.get("edges", "").strip()
            if raw:
                edges = [simplify_edge_id(t) if simplify else t for t in raw.split()]
                edges = [e for e in edges if e]
                if edges:
                    route_count += 1
                    od_counter[(edges[0], edges[-1])] += 1

        elem.clear()

    return od_counter, vehicle_count, route_count


def od_from_route_by_edge_txt(
    txt_path: Path,
) -> tuple[Counter[tuple[str, str]], int, int]:
    od_counter: Counter[tuple[str, str]] = Counter()
    vehicle_count = 0
    route_count = 0

    for raw_line in txt_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if not parts:
            continue

        vehicle_count += 1
        edges: list[str] = []

        idx = 1
        while idx + 1 < len(parts):
            time_token = parts[idx]
            edge_token = parts[idx + 1].strip()
            idx += 2

            if not HMS_TIME_RE.match(time_token):
                continue
            if edge_token:
                # route_by_edge.txt keeps original edge ids, no simplify.
                edges.append(edge_token)

        if edges:
            route_count += 1
            od_counter[(edges[0], edges[-1])] += 1

    return od_counter, vehicle_count, route_count


def count_od(
    input_path: Path, simplify: bool
) -> tuple[Counter[tuple[str, str]], int, int]:
    if input_path.suffix.lower() == ".xml":
        return od_from_rou_xml(input_path, simplify=simplify)

    if input_path.suffix.lower() == ".txt":
        return od_from_route_by_edge_txt(input_path)

    raise ValueError(
        f"Unsupported input format: {input_path}. "
        "Only .rou.xml and route_by_edge-style .txt are supported."
    )


def write_output(
    output_json: Path,
    od_counter: Counter[tuple[str, str]],
    vehicle_count: int,
    route_count: int,
    source_file: Path,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)

    sorted_pairs = sorted(od_counter.items(), key=lambda x: x[1], reverse=True)

    unique_origins = len({o for (o, _d) in od_counter})
    unique_destinations = len({d for (_o, d) in od_counter})

    result = {
        "source_file": str(source_file),
        "vehicle_count": vehicle_count,
        "route_count_with_edges": route_count,
        "unique_origin_count": unique_origins,
        "unique_destination_count": unique_destinations,
        "unique_od_pair_count": len(od_counter),
        "total_od_occurrences": int(sum(od_counter.values())),
        "od_pairs": [
            {"origin": o, "destination": d, "count": int(cnt)}
            for (o, d), cnt in sorted_pairs
        ],
    }

    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  route_count_with_edges  : {route_count}")
    print(f"  unique_origin_count     : {unique_origins}")
    print(f"  unique_destination_count: {unique_destinations}")
    print(f"  unique_od_pair_count    : {len(od_counter)}")
    if sorted_pairs:
        (o, d), top_cnt = sorted_pairs[0]
        print(f"  most_frequent_od        : {o} -> {d}  ({top_cnt})")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input = repo_root / "data" / "routes_tls_36000.rou.xml"
    default_output = Path(__file__).resolve().parent / "od_frequency.json"

    parser = argparse.ArgumentParser(
        description="Count OD (origin-destination) edge-pair frequencies from SUMO .rou.xml or route_by_edge.txt."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Input path (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output JSON path (default: {default_output})",
    )
    simplify_group = parser.add_mutually_exclusive_group()
    simplify_group.add_argument(
        "--simplify",
        dest="simplify",
        action="store_true",
        help="Enable edge id simplification (default: enabled, .xml only).",
    )
    simplify_group.add_argument(
        "--no-simplify",
        dest="simplify",
        action="store_false",
        help="Disable edge id simplification.",
    )
    parser.set_defaults(simplify=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Counting OD pairs: {input_path}")
    od_counter, vehicle_count, route_count = count_od(input_path, simplify=args.simplify)

    write_output(
        output_json=output_path,
        od_counter=od_counter,
        vehicle_count=vehicle_count,
        route_count=route_count,
        source_file=input_path,
    )

    print(f"Done. JSON result: {output_path}")


if __name__ == "__main__":
    main()
