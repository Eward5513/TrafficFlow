#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def simplify_edge_id(edge_id: str) -> str:
    token = edge_id.strip()
    if token.startswith("-"):
        token = token[1:]
    return token.split("#", 1)[0]


def count_edges(xml_path: Path, simplify: bool) -> tuple[Counter[str], int, int]:
    edge_counter: Counter[str] = Counter()
    vehicle_count = 0
    route_count = 0

    # Streaming parse to handle very large .rou.xml files.
    for _, elem in ET.iterparse(str(xml_path), events=("end",)):
        tag = local_name(elem.tag)

        if tag == "vehicle":
            vehicle_count += 1

        if tag == "route":
            edges = elem.attrib.get("edges", "").strip()
            if edges:
                route_count += 1
                for token in edges.split():
                    edge_id = simplify_edge_id(token) if simplify else token
                    if edge_id:
                        edge_counter[edge_id] += 1

        # Free parsed subtree memory as soon as possible.
        elem.clear()

    return edge_counter, vehicle_count, route_count


def write_outputs(
    output_json: Path,
    counter: Counter[str],
    vehicle_count: int,
    route_count: int,
    source_xml: Path,
    min_count: int,
    top_k: Optional[int],
    selected_items: list[tuple[str, int]],
    matched_edge_count_before_top_k: int,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "source_file": str(source_xml),
        "vehicle_count": vehicle_count,
        "route_count_with_edges": route_count,
        "unique_edge_count": len(counter),
        "total_edge_occurrences": int(sum(counter.values())),
        "min_count": min_count,
        "filter_rule": f"count > {min_count}",
        "matched_edge_count_before_top_k": matched_edge_count_before_top_k,
        "top_k": top_k,
        "result_edge_count": len(selected_items),
        "most_frequent_edge": (
            {"edge_id": selected_items[0][0], "count": int(selected_items[0][1])}
            if selected_items
            else None
        ),
        "top_edges": [
            {"rank": idx + 1, "edge_id": edge_id, "count": int(cnt)}
            for idx, (edge_id, cnt) in enumerate(selected_items)
        ],
    }

    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    output_txt = output_json.with_suffix(".txt")
    lines = [
        f"source_file: {source_xml}",
        f"vehicle_count: {vehicle_count}",
        f"route_count_with_edges: {route_count}",
        f"unique_edge_count: {len(counter)}",
        f"total_edge_occurrences: {sum(counter.values())}",
        f"filter_rule: count > {min_count}",
        f"matched_edge_count_before_top_k: {matched_edge_count_before_top_k}",
        f"top_k: {top_k if top_k is not None else 'not applied'}",
        f"result_edge_count: {len(selected_items)}",
        "",
        "Edges in result:",
    ]
    for idx, (edge_id, cnt) in enumerate(selected_items, start=1):
        lines.append(f"{idx:>3}. {edge_id}\t{cnt}")
    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input = repo_root / "data" / "routes_tls_36000.rou.xml"
    default_output = Path(__file__).resolve().parent / "edge_frequency_top.json"

    parser = argparse.ArgumentParser(
        description="Count most frequent edge ids in a SUMO .rou.xml file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Input .rou.xml path (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output JSON path (default: {default_output})",
    )
    parser.add_argument(
        "--min-count",
        "--count",
        dest="min_count",
        type=int,
        default=1000,
        help="Keep edges with count > min_count (default: 1000).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Apply top-k only when explicitly provided (default: not applied).",
    )
    simplify_group = parser.add_mutually_exclusive_group()
    simplify_group.add_argument(
        "--simplify",
        dest="simplify",
        action="store_true",
        help="Enable edge id simplification (default: enabled).",
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
    if args.min_count < 0:
        raise ValueError("--min-count/--count must be >= 0")
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError("--top-k must be > 0")

    counter, vehicle_count, route_count = count_edges(input_path, simplify=args.simplify)
    selected_items = [(edge_id, cnt) for edge_id, cnt in counter.items() if cnt > args.min_count]
    selected_items.sort(key=lambda item: item[1], reverse=True)
    matched_edge_count_before_top_k = len(selected_items)

    if args.top_k is not None:
        selected_items = selected_items[: args.top_k]

    write_outputs(
        output_json=output_path,
        counter=counter,
        vehicle_count=vehicle_count,
        route_count=route_count,
        source_xml=input_path,
        min_count=args.min_count,
        top_k=args.top_k,
        selected_items=selected_items,
        matched_edge_count_before_top_k=matched_edge_count_before_top_k,
    )

    print(f"Done. JSON result: {output_path}")
    print(f"Done. TXT summary: {output_path.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
