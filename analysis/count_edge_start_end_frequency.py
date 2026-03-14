#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
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


def _normalize_edge(edge_id: str, simplify: bool) -> str:
    if simplify:
        return simplify_edge_id(edge_id)
    return edge_id.strip()


def count_start_end_edges(
    xml_path: Path, simplify: bool
) -> tuple[Counter[str], Counter[str], Counter[str], int, int]:
    start_counter: Counter[str] = Counter()
    end_counter: Counter[str] = Counter()
    start_or_end_counter: Counter[str] = Counter()
    vehicle_count = 0
    route_count = 0

    # Streaming parse to handle very large .rou.xml files.
    for _, elem in ET.iterparse(str(xml_path), events=("end",)):
        tag = local_name(elem.tag)

        if tag == "vehicle":
            vehicle_count += 1

        if tag == "route":
            edges_raw = elem.attrib.get("edges", "").strip()
            if edges_raw:
                route_count += 1
                tokens = edges_raw.split()
                if tokens:
                    start_edge = _normalize_edge(tokens[0], simplify=simplify)
                    end_edge = _normalize_edge(tokens[-1], simplify=simplify)

                    if start_edge:
                        start_counter[start_edge] += 1
                        # B 口径：起点或终点按角色累计，同边可重复计数
                        start_or_end_counter[start_edge] += 1
                    if end_edge:
                        end_counter[end_edge] += 1
                        start_or_end_counter[end_edge] += 1

        elem.clear()

    return start_counter, end_counter, start_or_end_counter, vehicle_count, route_count


def build_group_result(counter: Counter[str], top_k: int) -> dict:
    selected_items = list(counter.items())
    selected_items.sort(key=lambda item: item[1], reverse=True)
    matched_edge_count_before_top_k = len(selected_items)

    selected_items = selected_items[:top_k]

    return {
        "unique_edge_count": len(counter),
        "total_edge_occurrences": int(sum(counter.values())),
        "top_k": top_k,
        "matched_edge_count_before_top_k": matched_edge_count_before_top_k,
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


def write_outputs(
    output_json: Path,
    source_xml: Path,
    simplify: bool,
    vehicle_count: int,
    route_count: int,
    start_result: dict,
    end_result: dict,
    start_or_end_result: dict,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "source_file": str(source_xml),
        "simplify": simplify,
        "vehicle_count": vehicle_count,
        "route_count_with_edges": route_count,
        "start": start_result,
        "end": end_result,
        "start_or_end": start_or_end_result,
    }
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    output_txt = output_json.with_suffix(".txt")
    lines = [
        f"source_file: {source_xml}",
        f"simplify: {simplify}",
        f"vehicle_count: {vehicle_count}",
        f"route_count_with_edges: {route_count}",
        "",
    ]

    def _append_group(title: str, group: dict) -> None:
        lines.extend(
            [
                f"[{title}]",
                f"unique_edge_count: {group['unique_edge_count']}",
                f"total_edge_occurrences: {group['total_edge_occurrences']}",
                f"matched_edge_count_before_top_k: {group['matched_edge_count_before_top_k']}",
                f"top_k: {group['top_k']}",
                f"result_edge_count: {group['result_edge_count']}",
                "Edges in result:",
            ]
        )
        for item in group["top_edges"]:
            lines.append(f"{item['rank']:>3}. {item['edge_id']}\t{item['count']}")
        lines.append("")

    _append_group("start", start_result)
    _append_group("end", end_result)
    _append_group("start_or_end", start_or_end_result)

    output_txt.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input = repo_root / "data" / "routes_tls_36000.rou.xml"
    default_output = Path(__file__).resolve().parent / "edge_start_end_frequency_top.json"

    parser = argparse.ArgumentParser(
        description="Count start/end edge distributions in a SUMO .rou.xml file."
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
        "--top-k",
        type=int,
        default=200,
        help="Keep top-k edges by count (default: 200).",
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
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")

    start_counter, end_counter, start_or_end_counter, vehicle_count, route_count = count_start_end_edges(
        input_path, simplify=args.simplify
    )

    start_result = build_group_result(start_counter, top_k=args.top_k)
    end_result = build_group_result(end_counter, top_k=args.top_k)
    start_or_end_result = build_group_result(start_or_end_counter, top_k=args.top_k)

    write_outputs(
        output_json=output_path,
        source_xml=input_path,
        simplify=args.simplify,
        vehicle_count=vehicle_count,
        route_count=route_count,
        start_result=start_result,
        end_result=end_result,
        start_or_end_result=start_or_end_result,
    )

    print(f"Done. JSON result: {output_path}")
    print(f"Done. TXT summary: {output_path.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
