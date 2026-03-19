#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count route length (number of edges per vehicle) distribution.

Outputs a histogram, percentiles, and the raw length->count mapping
so synthetic route generators can sample directly from the empirical
distribution instead of assuming log-normal.

Supported input formats (same as count_edge_frequency.py):
  - SUMO .rou.xml
  - route_by_edge.txt  (vin hh:mm:ss edge_id hh:mm:ss edge_id ...)

Output JSON structure:
  {
    "source_file": "...",
    "vehicle_count": N,
    "route_count_with_edges": N,
    "length_stats": {
      "min": N, "max": N, "mean": F, "median": F,
      "p10": N, "p25": N, "p50": N, "p75": N,
      "p90": N, "p95": N, "p99": N
    },
    "length_distribution": {
      "5": 120,
      "6": 340,
      ...
    }
  }

Usage:
  python analysis/count_route_length_distribution.py
  python analysis/count_route_length_distribution.py --input data/route_by_edge.txt
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


def lengths_from_rou_xml(
    xml_path: Path, simplify: bool
) -> tuple[list[int], int, int]:
    lengths: list[int] = []
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
                    lengths.append(len(edges))

        elem.clear()

    return lengths, vehicle_count, route_count


def lengths_from_route_by_edge_txt(
    txt_path: Path,
) -> tuple[list[int], int, int]:
    lengths: list[int] = []
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
        edge_count = 0

        idx = 1
        while idx + 1 < len(parts):
            time_token = parts[idx]
            edge_token = parts[idx + 1].strip()
            idx += 2

            if not HMS_TIME_RE.match(time_token):
                continue
            if edge_token:
                edge_count += 1

        if edge_count > 0:
            route_count += 1
            lengths.append(edge_count)

    return lengths, vehicle_count, route_count


def collect_lengths(
    input_path: Path, simplify: bool
) -> tuple[list[int], int, int]:
    if input_path.suffix.lower() == ".xml":
        return lengths_from_rou_xml(input_path, simplify=simplify)

    if input_path.suffix.lower() == ".txt":
        return lengths_from_route_by_edge_txt(input_path)

    raise ValueError(
        f"Unsupported input format: {input_path}. "
        "Only .rou.xml and route_by_edge-style .txt are supported."
    )


def percentile(sorted_data: list[int], p: float) -> float:
    """Linear interpolation percentile (same as numpy default)."""
    n = len(sorted_data)
    if n == 0:
        return 0.0
    idx = (n - 1) * p / 100.0
    lo = int(idx)
    hi = lo + 1
    frac = idx - lo
    if hi >= n:
        return float(sorted_data[-1])
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


def compute_stats(lengths: list[int]) -> dict:
    if not lengths:
        return {}
    s = sorted(lengths)
    total = sum(s)
    mean = total / len(s)
    return {
        "min": s[0],
        "max": s[-1],
        "mean": round(mean, 4),
        "median": round(percentile(s, 50), 4),
        "p10": round(percentile(s, 10), 4),
        "p25": round(percentile(s, 25), 4),
        "p50": round(percentile(s, 50), 4),
        "p75": round(percentile(s, 75), 4),
        "p90": round(percentile(s, 90), 4),
        "p95": round(percentile(s, 95), 4),
        "p99": round(percentile(s, 99), 4),
    }


def write_output(
    output_json: Path,
    lengths: list[int],
    vehicle_count: int,
    route_count: int,
    source_file: Path,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)

    length_counter: Counter[int] = Counter(lengths)
    stats = compute_stats(lengths)

    # Sort by length key for readability
    distribution = {str(k): v for k, v in sorted(length_counter.items())}

    result = {
        "source_file": str(source_file),
        "vehicle_count": vehicle_count,
        "route_count_with_edges": route_count,
        "length_stats": stats,
        "length_distribution": distribution,
    }

    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  route_count  : {route_count}")
    for key in ("min", "mean", "median", "p90", "p99", "max"):
        print(f"  {key:<12} : {stats.get(key)}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input = repo_root / "data" / "routes_tls_36000.rou.xml"
    default_output = Path(__file__).resolve().parent / "route_length_distribution.json"

    parser = argparse.ArgumentParser(
        description="Compute route length distribution from SUMO .rou.xml or route_by_edge.txt."
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

    print(f"Collecting route lengths: {input_path}")
    lengths, vehicle_count, route_count = collect_lengths(input_path, simplify=args.simplify)

    write_output(
        output_json=output_path,
        lengths=lengths,
        vehicle_count=vehicle_count,
        route_count=route_count,
        source_file=input_path,
    )

    print(f"Done. JSON result: {output_path}")


if __name__ == "__main__":
    main()
