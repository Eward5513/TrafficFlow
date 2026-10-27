#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrap route_by_edge.txt into a screenshot-friendly text layout.

The script keeps the data as plain text. It only limits how many
(timestamp, edge_id) pairs appear on each line, so long trajectories are
easier to display, review, and capture in screenshots.

Input format (route_by_edge.txt):
  vin hh:mm:ss edge_id hh:mm:ss edge_id ...

Output format (wrapped .txt):
  vin hh:mm:ss edge_id hh:mm:ss edge_id ...   <- first line, up to 13 pairs
  hh:mm:ss edge_id hh:mm:ss edge_id ...       <- continuation (same vin)

Continuation lines do not repeat vin; they are still part of the same trajectory.

Usage:
  python wrap_routes_for_screenshot.py
"""

from __future__ import annotations

import re
from pathlib import Path

from _paths import OCR_DATA

INPUT_TXT = OCR_DATA / "ocr_output" / "route_by_edge.txt"
OUTPUT_TXT = OCR_DATA / "ocr_output" / "route_by_edge_wrapped.txt"
MAX_PAIRS = 13

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


def main() -> None:
    input_path = INPUT_TXT.resolve()
    output_path = OUTPUT_TXT.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if MAX_PAIRS <= 0:
        raise ValueError("MAX_PAIRS must be > 0")

    vehicle_count, route_count, total_pairs, line_count = reformat(
        input_path, output_path, max_pairs=MAX_PAIRS
    )

    print(f"Done. Output: {output_path}")
    print(f"  vehicles parsed       : {vehicle_count}")
    print(f"  routes with edges     : {route_count}")
    print(f"  total pairs           : {total_pairs}")
    print(f"  output lines written  : {line_count}  (max {MAX_PAIRS} pairs each)")


if __name__ == "__main__":
    main()
