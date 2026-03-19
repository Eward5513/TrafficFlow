#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reformat route_by_edge.txt so that each output line carries at most
MAX_PAIRS (default 13) (timestamp, edge_id) pairs.

Input format (route_by_edge.txt):
  vin hh:mm:ss edge_id hh:mm:ss edge_id ...

Output format (wrapped .txt):
  vin hh:mm:ss edge_id hh:mm:ss edge_id ...   <- first line, up to 13 pairs
  hh:mm:ss edge_id hh:mm:ss edge_id ...       <- continuation (same vin)

Continuation lines do not repeat vin; they are still part of the same trajectory.

Usage:
  python analysis/export_routes_json.py
  python analysis/export_routes_json.py --input data/route_by_edge.txt
  python analysis/export_routes_json.py --max-pairs 10
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

HMS_TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")


def reformat(txt_path: Path, out_path: Path, max_pairs: int) -> tuple[int, int, int, int]:
    """Read *txt_path*, wrap at *max_pairs* per line, write to *out_path*.

    Returns
    -------
    (vehicle_count, route_count_with_edges, total_pair_occurrences, output_line_count)
    """
    vehicle_count = 0
    route_count = 0
    total_pairs = 0
    output_lines: list[str] = []

    for raw_line in txt_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if not parts:
            continue

        vehicle_count += 1
        vin = parts[0]

        # Collect all valid (time, edge_id) pairs from this vehicle's record.
        pairs: list[tuple[str, str]] = []
        idx = 1
        while idx + 1 < len(parts):
            time_token = parts[idx]
            edge_token = parts[idx + 1]
            idx += 2
            if HMS_TIME_RE.match(time_token):
                pairs.append((time_token, edge_token))

        if not pairs:
            continue

        route_count += 1
        total_pairs += len(pairs)

        # Emit one output line per chunk of max_pairs.
        # First chunk keeps vin, continuation chunks omit vin.
        for chunk_idx, start in enumerate(range(0, len(pairs), max_pairs)):
            chunk = pairs[start : start + max_pairs]
            tokens = [vin] if chunk_idx == 0 else []
            for time_token, edge_token in chunk:
                tokens.append(time_token)
                tokens.append(edge_token)
            output_lines.append(" ".join(tokens))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    return vehicle_count, route_count, total_pairs, len(output_lines)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input = repo_root / "data" / "route_by_edge.txt"
    default_output = Path(__file__).resolve().parent / "route_by_edge_wrapped.txt"

    parser = argparse.ArgumentParser(
        description=(
            "Reformat route_by_edge.txt so each line has at most "
            "--max-pairs (timestamp, edge_id) pairs."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Input route_by_edge.txt path (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output .txt path (default: {default_output})",
    )
    parser.add_argument(
        "--max-pairs",
        dest="max_pairs",
        type=int,
        default=13,
        help="Maximum (timestamp, edge_id) pairs per output line (default: 13).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if args.max_pairs <= 0:
        raise ValueError("--max-pairs must be > 0")

    vehicle_count, route_count, total_pairs, line_count = reformat(
        input_path, output_path, max_pairs=args.max_pairs
    )

    print(f"Done. Output: {output_path}")
    print(f"  vehicles parsed       : {vehicle_count}")
    print(f"  routes with edges     : {route_count}")
    print(f"  total pairs           : {total_pairs}")
    print(f"  output lines written  : {line_count}  (max {args.max_pairs} pairs each)")


if __name__ == "__main__":
    main()
