#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count consecutive edge-pair (A -> B) frequencies and compute per-edge
transition probabilities for the full road network.

The output adjacency map can be used directly as a Markov-chain model
for connected route generation.

Supported input formats (same as count_edge_frequency.py):
  - SUMO .rou.xml
  - route_by_edge.txt  (vin hh:mm:ss edge_id hh:mm:ss edge_id ...)

Output JSON structure:
  {
    "source_file": "...",
    "vehicle_count": N,
    "route_count_with_edges": N,
    "unique_from_edge_count": N,
    "unique_pair_count": N,
    "total_pair_occurrences": N,
    "adjacency": {
      "A": [{"to": "B", "count": N, "prob": 0.xx}, ...],
      ...
    }
  }

Usage:
  python analysis/count_edge_pair_frequency.py
  python analysis/count_edge_pair_frequency.py --input data/route_by_edge.txt
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
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


def _count_pairs_from_edge_list(
    edges: list[str],
    pair_counter: Counter[tuple[str, str]],
) -> None:
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        if a and b:
            pair_counter[(a, b)] += 1


def count_pairs_from_rou_xml(
    xml_path: Path, simplify: bool
) -> tuple[Counter[tuple[str, str]], int, int]:
    pair_counter: Counter[tuple[str, str]] = Counter()
    vehicle_count = 0
    route_count = 0

    for _, elem in ET.iterparse(str(xml_path), events=("end",)):
        tag = local_name(elem.tag)

        if tag == "vehicle":
            vehicle_count += 1

        if tag == "route":
            raw = elem.attrib.get("edges", "").strip()
            if raw:
                route_count += 1
                edges = [simplify_edge_id(t) if simplify else t for t in raw.split()]
                _count_pairs_from_edge_list(edges, pair_counter)

        elem.clear()

    return pair_counter, vehicle_count, route_count


def count_pairs_from_route_by_edge_txt(
    txt_path: Path,
) -> tuple[Counter[tuple[str, str]], int, int]:
    pair_counter: Counter[tuple[str, str]] = Counter()
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
            _count_pairs_from_edge_list(edges, pair_counter)

    return pair_counter, vehicle_count, route_count


def count_pairs(
    input_path: Path, simplify: bool
) -> tuple[Counter[tuple[str, str]], int, int]:
    if input_path.suffix.lower() == ".xml":
        return count_pairs_from_rou_xml(input_path, simplify=simplify)

    if input_path.suffix.lower() == ".txt":
        return count_pairs_from_route_by_edge_txt(input_path)

    raise ValueError(
        f"Unsupported input format: {input_path}. "
        "Only .rou.xml and route_by_edge-style .txt are supported."
    )


def build_adjacency(
    pair_counter: Counter[tuple[str, str]],
) -> dict[str, list[dict]]:
    from_totals: dict[str, int] = defaultdict(int)
    for (a, _b), cnt in pair_counter.items():
        from_totals[a] += cnt

    adjacency: dict[str, list[dict]] = defaultdict(list)
    for (a, b), cnt in pair_counter.items():
        prob = cnt / from_totals[a]
        adjacency[a].append({"to": b, "count": int(cnt), "prob": round(prob, 6)})

    for a in adjacency:
        adjacency[a].sort(key=lambda x: x["count"], reverse=True)

    return dict(adjacency)


def _compact_entry(entry: dict) -> str:
    """Serialize a single adjacency entry as a compact one-liner."""
    parts = []
    for k, v in entry.items():
        parts.append(f'"{k}": {json.dumps(v, ensure_ascii=False)}')
    return "{" + ", ".join(parts) + "}"


def _serialize(result: dict) -> str:
    """
    Serialize result to JSON with adjacency list items compacted to one line each.

    Example output for one edge:
      "107838397": [
        {"to": "108780288", "count": 912, "prob": 0.695},
        {"to": "27583008", "count": 401, "prob": 0.305}
      ]
    """
    # Serialize scalars / metadata section normally, then append adjacency manually.
    meta = {k: v for k, v in result.items() if k != "adjacency"}
    # Build JSON for meta block (strip closing brace to append adjacency)
    meta_json = json.dumps(meta, ensure_ascii=False, indent=2)
    meta_json = meta_json.rstrip().rstrip("}")  # remove trailing "}"

    lines = [meta_json.rstrip(), '  "adjacency": {']
    adjacency: dict[str, list[dict]] = result["adjacency"]
    edge_keys = list(adjacency.keys())
    for i, from_edge in enumerate(edge_keys):
        entries = adjacency[from_edge]
        comma_outer = "," if i < len(edge_keys) - 1 else ""
        if not entries:
            lines.append(f'    {json.dumps(from_edge)}: []{comma_outer}')
            continue
        lines.append(f'    {json.dumps(from_edge)}: [')
        for j, entry in enumerate(entries):
            comma_inner = "," if j < len(entries) - 1 else ""
            lines.append(f'      {_compact_entry(entry)}{comma_inner}')
        lines.append(f'    ]{comma_outer}')
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def write_outputs(
    output_json: Path,
    pair_counter: Counter[tuple[str, str]],
    vehicle_count: int,
    route_count: int,
    source_file: Path,
    adjacency: dict[str, list[dict]],
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "source_file": str(source_file),
        "vehicle_count": vehicle_count,
        "route_count_with_edges": route_count,
        "unique_from_edge_count": len(adjacency),
        "unique_pair_count": len(pair_counter),
        "total_pair_occurrences": int(sum(pair_counter.values())),
        "adjacency": adjacency,
    }

    output_json.write_text(_serialize(result), encoding="utf-8")
    print(f"  unique_from_edge_count : {len(adjacency)}")
    print(f"  unique_pair_count      : {len(pair_counter)}")
    print(f"  total_pair_occurrences : {sum(pair_counter.values())}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input = repo_root / "data" / "routes_tls_36000.rou.xml"
    default_output = Path(__file__).resolve().parent / "edge_pair_frequency.json"

    parser = argparse.ArgumentParser(
        description=(
            "Build full edge-transition adjacency from SUMO .rou.xml or route_by_edge.txt. "
            "Outputs per-edge successor counts and transition probabilities."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Input path (.rou.xml or route_by_edge.txt format, default: {default_input})",
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

    print(f"Counting edge pairs: {input_path}")
    pair_counter, vehicle_count, route_count = count_pairs(input_path, simplify=args.simplify)

    adjacency = build_adjacency(pair_counter)

    write_outputs(
        output_json=output_path,
        pair_counter=pair_counter,
        vehicle_count=vehicle_count,
        route_count=route_count,
        source_file=input_path,
        adjacency=adjacency,
    )

    print(f"Done. JSON result: {output_path}")


if __name__ == "__main__":
    main()
