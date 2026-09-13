"""
遍历 SUMO 路网文件，找出互为反向的边对并打印。

判定逻辑：两条普通边（非 internal）A、B，若 A.from == B.to 且 A.to == B.from，
则 (A, B) 互为反向边对。轨迹中相邻出现这样的边对意味着车辆原地绕了个环。

用法：
    python analysis/road_network/find_reverse_edge_pairs.py
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
NET_FILE = BASE_DIR / "net_tls.net.xml"


def find_reverse_pairs(net_file: Path) -> list[tuple[str, str]]:
    # (from, to) -> [edge_id, ...]，普通路网中同向平行边也可能存在，故用列表
    od_to_edges: dict[tuple[str, str], list[str]] = {}

    for _, elem in ET.iterparse(net_file, events=("end",)):
        if elem.tag != "edge":
            continue
        if elem.get("function") == "internal":
            elem.clear()
            continue
        edge_id = elem.get("id")
        from_node = elem.get("from")
        to_node = elem.get("to")
        if edge_id and from_node and to_node:
            od_to_edges.setdefault((from_node, to_node), []).append(edge_id)
        elem.clear()

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for (from_node, to_node), edge_ids in od_to_edges.items():
        if (from_node, to_node) in seen:
            continue
        reverse_ids = od_to_edges.get((to_node, from_node))
        if not reverse_ids:
            continue
        seen.add((from_node, to_node))
        seen.add((to_node, from_node))
        for eid in edge_ids:
            for rid in reverse_ids:
                pairs.append((eid, rid))

    return pairs


def main() -> None:
    pairs = find_reverse_pairs(NET_FILE)
    pairs.sort()
    print(f"# 共找到 {len(pairs)} 对互为反向的边（可直接粘贴为 Python 元组列表）")
    print("REVERSE_EDGE_PAIRS = [")
    for a, b in pairs:
        print(f'    ("{a}", "{b}"),')
    print("]")


if __name__ == "__main__":
    main()
