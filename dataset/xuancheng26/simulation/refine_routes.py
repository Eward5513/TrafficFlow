"""
先删除已从路网移除的 edge，再压缩路口内部连续 edge 子序列，并把受影响车辆转成 trip。

处理顺序：
    原始 route
    → 删除全部 REMOVED_EDGES
    → 在删除后的序列上同一次扫描中分别识别并压缩各路口集合的连续子序列
    → 已发生删除或压缩的车辆转换成 trip
    → 完全没有修改的完整 route 按新路网 <connection> 检查相邻连通性

未命中且相邻 connection 都存在的车辆保持原样。缺口留给 SUMO 按 from/to/via 补全。
写出的每个顶层 <vType> 都带同一组换道属性；同名属性会被覆盖，不重复。

本脚本只改写 XML，不调用 SUMO 或 duarouter。

用法：
    python3 refine_routes.py \
        --input /data5/tzhang174/TrafficFlow/xuancheng26/simulation_data/trip_data/prepared/routes_2023_04_14.rou.xml \
        --net /home/tzhang174/TrafficFlow/dataset/xuancheng26/xuancheng_tls_rebuilt.net.xml

输出写在输入文件旁边，文件名为 refined_<输入文件名>。
全天仿真读取的是 simulation_data/trip_data/refined/ 里的 refined_routes_*.rou.xml。
"""

from __future__ import annotations

import argparse
import errno
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterable, TextIO
from xml.parsers import expat
from xml.sax.saxutils import escape, quoteattr

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent
DEFAULT_INPUT_FILE = Path(
    "/data5/tzhang174/TrafficFlow/xuancheng26/simulation_data/trip_data/prepared/routes_2023_04_14.rou.xml"
)
DEFAULT_NET_FILE = DATASET_DIR / "xuancheng_tls_rebuilt.net.xml"

JUNCTION_EDGE_SETS = {
    "junction_34180200000232": {
        "34180200742",
        "34180200707",
        "34180205350",
        "34180201038",
        "34180205349",
        "34180201041_1",
        "34180200708",
        "34180200751_1",
        "34180200747",
    },
    "junction_34180200000350": {
        "34180201254",
        "34180201251",
        "34180201175",
        "34180201177",
        "34180201253",
        "34180201250",
        "34180201187",
        "34180201189",
    },
    "junction_34180200000215": {
        "34180200631",
        "34180203063",
        "34180200627",
        "34180200630",
        "34180200628",
        "34180200629",
        "34180200632",
        "34180200625",
    },
    "junction_34180200000207": {
        "34180203088",
        "34180200582",
        "34180202962",
        "34180200234",
        "34180200583",
        "34180200584",
        "34180203141",
    },
}
REMOVED_EDGES = {
    "34180200741",
    "34180200837",
    "34180200836",
    "34180201515",
    "34180200705",
    "34180200710",
    "34180200703",
    "34180200704",
    "34180200706",
    "34180200709",
    "34180201180",
    "34180201188",
    "34180203816",
    "34180201176",
    "34180201189_1",
    "34180201252",
    "34180201179",
    "34180203992",
    "34180203064",
    "34180203073",
    "34180203061",
    "E14",
    "34180203099",
    "34180200626",
    "34180202982",
    "34180202980",
    "34180202981",
    "34180203118",
    "34180200501_1",
    "34180201042_2",
    "34180201042",
    "34180201039_3",
    "34180201459_2",
    "34180201039_2",
    "34180201459_1",
    "34180201693",
    "34180201460",

    "34180203084",
    "34180203082",
    "34180200235",
    "34180203089",
    "34180200586",
    "34180200587",
    "34180203140",
    "34180203015",
    "34180202963",
    "34180200588",
    "34180203087",
    "34180200585",

    "34180201596",
    "34180201594",
    "34180201419",
    "34180201417",
    "34180200530",


    "34180200664_2",
    "34180200662_2",
    "34180200661_2",
    "34180200663_2",
    "34180200664_1",
    "34180200663_1",
    "34180200661_1",

    "34180201164",

    "34180204209",
    "34180204208",
    "34180204212",
    "34180204211",

    "34180205028",
    "34180205026",

    "34180205267_2",

    "34180201543_2",
    "34180201138_2",

    "34180202987",
    "34180200666",
    "34180203147",
    "34180202986",
    "34180203071",
    "34180203756",
    "34180203769",
    "34180204216",
    "34180200430",
    "34180200426",

    "34180200429",
    "34180200424",
    "34180200423",
    "34180200257",
    "34180200425",
    "34180200428",
    "34180201309",

    "34180204173",
    "34180203787",
    "34180203723",
    "34180204172",

    "34180200419_2",
    "34180202064",
    "34180200422_2",
    "34180202062",
    "34180200415_1",

    "34180201536_1",

    "34180201535_1",
    "34180202420_1",

    "34180200649_2",
    "34180200650_2",
    "34180200648_2",
    "34180200478_2",
    "34180200478_1",
    "34180200648_1",
    "34180200650_1",
    "34180200649_1",
    "34180200649_3",

    "34180201495",
    "34180201494",

    "34180204412",
    "34180203834",
    "34180201637",
    "34180204462",
    "34180201747_1",
    "34180201636_1",
    "34180203657",
    "34180203655",

    "34180201662",
    "34180201664",

    "34180203045",
    "34180203120",
    "34180200283",
    "34180200341_1",
    "34180200284_1",

    "34180201824_1",
    "34180205035",
    "34180202440",
    "34180200112",
    "34180200111",
    "34180200120_2",
    "34180202906_2",
    "34180202906_1",
    "34180200122_1",
    "34180200122_2",    
    "34180200120_1",
    "34180200119_2",

    "34180204204",
    "34180203654",
    "34180204429",
    "34180203739",
    "34180201365",
    "34180201364",

    "34180202626",
    "34180202640",
    "34180202625",
    "34180202633",
    "34180200221",
    "34180200220",
    "34180200162",
    "34180200164",

    "34180203072",
    "34180203563",
    "34180203465",
    "34180203546",
    "34180203075",
    "34180201282",
    "34180201280",
    "34180201237",
    "34180201159",
    "34180200732",
    "34180200724",
    "34180201078",
    "34180201284",
    "34180201076",
    "34180201075",
    "34180201077",
    "34180201079",

    "34180205267_1",
    "34180200645",

    "34180203016",
    "34180202950",
    "34180202907",
    "34180202908",

    "34180202932",
    "34180202934",

    "34180200190",
    "34180200189",
    "34180200191",

    "34180200038",
    "34180200033",
    "34180200035",
    "34180200034",
    "34180200032",
    "-34180200032",

    "34180200085",
    "34180200082",
    "34180200089",
    "34180200088",
    "E2",
    "-E2.200",
    "34180200179",
    "34180200182",
    "34180200025_1",
    "34180200024_1",

    "34180200120_3",
    "34180200119",

    "34180200419",
    "34180200416",

    "34180200616_1",
    "34180200618",

    "34180202063",
    "34180200417",

    "34180200422",
    "34180200616",

    "34180202851",
    "34180200615",
    "34180200619",

    "34180200401",
    "34180200404",
    "34180200402",
    "34180200403",

    "34180200400",
    "34180202027_1",
    "34180202025_1",

    "34180202026_1",
    "34180201435_1",

    "34180202736",
    "34180202737",

    "34180203044",
    "34180204461",

    "34180201815_1",
    "34180203088_1",
    "34180203086",
    "34180202998",

    "34180203088_1",

    "34180203833",
    "34180204228",

    "34180200267",
    "34180201767",
    "34180201423",
    "34180200581",
    "34180202999",

    "34180202368",
    "34180201143_1",

    "34180200662",
    "34180200659",
    "34180205099_2",
    "34180204061_1",
    "34180201543_1",

    "34180203605",
    "34180205133",

    "34180200342_1",

    "34180200622",

    "34180201068_1",
    "34180201069_1",
    "34180201925_1",
    "34180202075_1",
    "34180202076_1",

    "34180202073_1",

    "34180200538_1",
    "34180200224",
    "34180200223",
    "34180202213_1",
    "34180201958",
    "34180201258",

    "34180202211",
    "34180201822",
    "34180201965",
    "34180201283",
    "34180201281",
    "34180203076",
    "34180201537_1",
    "34180201953",
    "34180201954",
    "34180201070_1",
    "34180201600",
    "34180201658",

    "34180200272",

    "34180201742",
    "34180201743",
    "34180201813",
    "34180202947",
    "34180202933",
    "34180202948",
    "34180201161_1",

    "34180201064",
    "34180200650_3",
    "34180202077_1",

    "34180204449",
    "34180201178",
    "34180205349_1",
    "34180201191",
    "34180201824_3",
    "34180201915",
    "34180201912",
    "34180201819",
    "34180201819_1",
    "34180201821_1",
    "34180201823_1",
    "34180201366",
    "34180201368",
    "34180201363",
    "34180200478_3",
    "34180200483",

    "34180200926_1",
    "34180201906_1",
    "34180201459_3",
    "34180201462_1",
    "34180201039_4",
    "34180201042_3",
    "34180200744",
    "34180200715_1",
    "34180200275",
    "34180200278",
    "34180200716_1",

    "34180203815",
    "34180203814",
    "34180203993",
    "34180200503_1",
    
    "34180201660",
    "34180201042_4",
    "34180201539_1",
    "34180202078_1",
    "34180201926",
    "34180201631_1",
    "34180200279_1",
    "34180200282",
    "34180200669",
    "34180200568",
    "34180200569",

    "34180201553",
    "34180200668",
    "34180200239",
    "34180200241",
    "34180203724",

    "34180200773",
    "34180200770",
    "34180200768",
    "34180201158",
    "34180200775",

    "34180201167",
    "34180201081",

    "34180200532",
    "34180200121_1",
    "34180202214_1",
    "34180202423_1",
    "34180200645_1",
    "34180201065",

    "34180200339",
    "34180200750_1",
    "34180200618_1",
    "34180200617",

}

DEMAND_TAGS = frozenset({"vehicle", "trip", "flow"})
VTYPE_LANECHANGE_ATTRS = (
    ("lcSpeedGain", "0.5"),
    ("lcKeepRight", "0"),
    ("lcStrategic", "1.0"),
    ("lcCooperative", "1.0"),
)
ROUTE_CHILD_TAGS = frozenset({"route"})
ATTRS_REPLACED_ON_CONVERT = frozenset({"route", "from", "to", "via"})
PARSE_CHUNK_SIZE = 8 * 1024 * 1024


class RouteFormatError(Exception):
    """Malformed route XML that cannot be converted reliably."""

    def __init__(self, message: str, line: int | None = None) -> None:
        self.line = line
        if line is None:
            super().__init__(message)
        else:
            super().__init__(f"line {line}: {message}")


@dataclass(slots=True)
class XmlElem:
    tag: str
    attrs: list[tuple[str, str]]
    children: list["XmlElem"] = field(default_factory=list)
    text: str = ""


@dataclass(slots=True)
class ConversionStats:
    total_vehicles: int = 0
    unaffected_vehicles: int = 0
    converted_vehicles: int = 0
    junction_runs: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in JUNCTION_EDGE_SETS}
    )
    compressed_runs: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in JUNCTION_EDGE_SETS}
    )
    removed_internal_edges: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in JUNCTION_EDGE_SETS}
    )
    removed_edges: int = 0
    dropped_vehicles: int = 0
    network_edge_count: int = 0
    direct_connection_count: int = 0
    checked_full_routes: int = 0
    kept_full_routes: int = 0
    converted_by_missing_connection: int = 0
    missing_connection_occurrences: int = 0
    missing_pair_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    missing_connections_report: Path | None = None
    dropped_vehicle_ids: list[str] = field(default_factory=list)


def get_attr(elem: XmlElem, name: str) -> str | None:
    for key, value in elem.attrs:
        if key == name:
            return value
    return None


def demand_id(elem: XmlElem) -> str:
    value = get_attr(elem, "id")
    if value is None or value == "":
        return "<missing-id>"
    return value


def pair_expat_attrs(attrs: object) -> list[tuple[str, str]]:
    if isinstance(attrs, dict):
        return [(str(key), str(value)) for key, value in attrs.items()]
    items = list(attrs)
    if len(items) % 2 != 0:
        raise RouteFormatError(f"Attribute list has odd length: {items!r}")
    return [(str(items[i]), str(items[i + 1])) for i in range(0, len(items), 2)]


def detect_newline(raw: bytes) -> str:
    return "\r\n" if b"\r\n" in raw else "\n"


def skip_xml_tag(fp: BinaryIO, start_idx: int) -> int:
    """Return the byte index immediately after the tag that starts at start_idx."""
    fp.seek(start_idx)
    first = fp.read(1)
    if first != b"<":
        raise RouteFormatError(
            f"Expected '<' at byte {start_idx}, found {first!r}"
        )
    in_squote = False
    in_dquote = False
    pos = start_idx + 1
    while True:
        char = fp.read(1)
        if not char:
            raise RouteFormatError(
                f"Unexpected EOF while scanning XML tag starting at byte {start_idx}"
            )
        pos += 1
        if in_squote:
            if char == b"'":
                in_squote = False
            continue
        if in_dquote:
            if char == b'"':
                in_dquote = False
            continue
        if char == b"'":
            in_squote = True
        elif char == b'"':
            in_dquote = True
        elif char == b">":
            return pos


def exclusive_element_end(fp: BinaryIO, end_event_idx: int) -> int:
    """
    Map expat EndElement CurrentByteIndex to the first byte after the element.

    Empty elements report an index already past `/>`. Elements with a separate
    end tag report the index of `</tag>`, so the closing tag still has to be
    skipped.
    """
    fp.seek(end_event_idx)
    peek = fp.read(2)
    if peek == b"</":
        return skip_xml_tag(fp, end_event_idx)
    return end_event_idx


def read_slice(fp: BinaryIO, start: int, end: int) -> bytes:
    if end < start:
        raise RuntimeError(f"Invalid byte slice {start}:{end}")
    fp.seek(start)
    return fp.read(end - start)


def split_prefix(raw: bytes, tag: str) -> tuple[bytes, bytes]:
    token = b"<" + tag.encode("utf-8")
    start = 0
    while True:
        idx = raw.find(token, start)
        if idx < 0:
            raise RouteFormatError(
                f"Could not locate start tag <{tag}> in copied XML slice"
            )
        after = idx + len(token)
        if after == len(raw) or raw[after] in b" \t\r\n/>":
            return raw[:idx], raw[idx:]
        start = idx + 1


def indent_from_prefix(prefix: bytes, newline: str) -> str:
    prefix_text = prefix.decode("utf-8")
    if newline in prefix_text:
        last_line = prefix_text.rsplit(newline, 1)[-1]
    else:
        last_line = prefix_text
    i = len(last_line)
    while i > 0 and last_line[i - 1] in " \t":
        i -= 1
    return last_line[i:]


def prefix_before_indent(prefix: bytes, newline: str) -> bytes:
    indent = indent_from_prefix(prefix, newline).encode("utf-8")
    if indent and prefix.endswith(indent):
        return prefix[: len(prefix) - len(indent)]
    return prefix


def parse_edge_list(edges_attr: str, *, context: str) -> list[str]:
    edges = edges_attr.split()
    if not edges:
        raise RouteFormatError(f"{context}: edges attribute is empty")
    return edges


def validate_disjoint_junction_sets() -> dict[str, str]:
    """Map each junction edge to exactly one set name; error on overlaps."""
    edge_to_junction: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for name, edges in JUNCTION_EDGE_SETS.items():
        for edge in edges:
            previous = edge_to_junction.get(edge)
            if previous is None:
                edge_to_junction[edge] = name
                continue
            owners = duplicates.setdefault(edge, [previous])
            if name not in owners:
                owners.append(name)
    if duplicates:
        details = "\n".join(
            f"  {edge_id}: {', '.join(names)}"
            for edge_id, names in sorted(duplicates.items())
        )
        raise RouteFormatError(
            "JUNCTION_EDGE_SETS contain overlapping edge IDs:\n" + details
        )
    removed_overlaps: list[tuple[str, str]] = []
    for name, edges in JUNCTION_EDGE_SETS.items():
        for edge_id in sorted(edges & REMOVED_EDGES):
            removed_overlaps.append((edge_id, name))
    if removed_overlaps:
        details = "\n".join(
            f"  {edge_id}: belongs to {junction} and REMOVED_EDGES"
            for edge_id, junction in removed_overlaps
        )
        raise RouteFormatError(
            "REMOVED_EDGES overlap junction edge sets:\n" + details
        )
    return edge_to_junction


def compress_junction_runs(
    edges: list[str],
    edge_to_junction: dict[str, str],
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """
    Compress each maximal consecutive run that belongs to one named junction.

    Length 1 or 2 is kept as-is. Length >= 3 keeps only the first and last
    edges of that run. Adjacent edges from different junctions stay separate.
    """
    per_junction = {
        name: {"runs": 0, "compressed": 0, "removed_internal": 0}
        for name in JUNCTION_EDGE_SETS
    }
    kept: list[str] = []
    index = 0
    length = len(edges)
    while index < length:
        junction = edge_to_junction.get(edges[index])
        if junction is None:
            kept.append(edges[index])
            index += 1
            continue
        start = index
        index += 1
        while index < length and edge_to_junction.get(edges[index]) == junction:
            index += 1
        run = edges[start:index]
        stats = per_junction[junction]
        stats["runs"] += 1
        if len(run) >= 3:
            kept.append(run[0])
            kept.append(run[-1])
            stats["compressed"] += 1
            stats["removed_internal"] += len(run) - 2
        else:
            kept.extend(run)
    return kept, per_junction


def remove_independent_edges(edges: list[str]) -> tuple[list[str], int]:
    """First pass: drop every occurrence of REMOVED_EDGES, preserving order."""
    kept: list[str] = []
    removed_count = 0
    for edge in edges:
        if edge not in REMOVED_EDGES:
            kept.append(edge)
            continue
        removed_count += 1
    return kept, removed_count


def xml_local_name(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def load_sumo_network(net_path: Path) -> tuple[set[str], set[tuple[str, str]]]:
    """Load non-internal edge IDs and direct from/to connections from a SUMO net."""
    network_edges: set[str] = set()
    direct_connections: set[tuple[str, str]] = set()
    try:
        context = ET.iterparse(net_path, events=("end",))
        for _, elem in context:
            tag = xml_local_name(elem.tag)
            if tag == "edge":
                edge_id = elem.get("id")
                if (
                    edge_id
                    and elem.get("function") != "internal"
                    and not edge_id.startswith(":")
                ):
                    network_edges.add(edge_id)
            elif tag == "connection":
                from_edge = elem.get("from")
                to_edge = elem.get("to")
                if (
                    from_edge
                    and to_edge
                    and not from_edge.startswith(":")
                    and not to_edge.startswith(":")
                ):
                    direct_connections.add((from_edge, to_edge))
            elem.clear()
    except ET.ParseError as exc:
        raise RouteFormatError(
            f"Failed to parse network XML {net_path}: {exc}"
        ) from exc
    except OSError as exc:
        raise RouteFormatError(f"Failed to read network file {net_path}: {exc}") from exc
    return network_edges, direct_connections


def is_complete_route_vehicle(elem: XmlElem) -> bool:
    if elem.tag == "trip":
        return False
    if get_attr(elem, "route") is not None:
        return True
    return any(child.tag == "route" for child in elem.children)


def missing_route_edges(edges: list[str], network_edges: set[str]) -> list[str]:
    missing: list[str] = []
    seen: set[str] = set()
    for edge in edges:
        if edge in network_edges or edge in seen:
            continue
        seen.add(edge)
        missing.append(edge)
    return missing


def missing_direct_pairs(
    edges: list[str],
    direct_connections: set[tuple[str, str]],
) -> list[tuple[str, str]]:
    missing: list[tuple[str, str]] = []
    for from_edge, to_edge in zip(edges, edges[1:]):
        if (from_edge, to_edge) not in direct_connections:
            missing.append((from_edge, to_edge))
    return missing


def missing_connections_report_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".missing_connections.txt")


def format_open_tag(
    tag: str,
    attrs: list[tuple[str, str]],
    *,
    indent: str,
    newline: str,
    self_close: bool,
) -> str:
    if not attrs:
        closer = "/>" if self_close else ">"
        return f"{indent}<{tag}{closer}"
    first_name, first_value = attrs[0]
    lines = [f"{indent}<{tag} {first_name}={quoteattr(first_value)}"]
    continuation = indent + (" " * (len(tag) + 2))
    for name, value in attrs[1:]:
        lines.append(f"{continuation}{name}={quoteattr(value)}")
    if self_close:
        lines[-1] += "/>"
    else:
        lines[-1] += ">"
    return newline.join(lines)


def serialize_element(elem: XmlElem, *, indent: str, newline: str) -> str:
    text = elem.text
    children = elem.children
    has_text = bool(text) and bool(text.strip())
    self_close = not children and not has_text
    opening = format_open_tag(
        elem.tag,
        elem.attrs,
        indent=indent,
        newline=newline,
        self_close=self_close,
    )
    if self_close:
        return opening
    child_indent = indent + "    "
    parts = [opening]
    if has_text and not children:
        parts.append(escape(text.strip()))
        parts.append(f"</{elem.tag}>")
        return "".join(parts)
    if has_text:
        parts.append(newline)
        parts.append(child_indent)
        parts.append(escape(text.strip()))
    for child in children:
        parts.append(newline)
        parts.append(serialize_element(child, indent=child_indent, newline=newline))
    parts.append(newline)
    parts.append(f"{indent}</{elem.tag}>")
    return "".join(parts)


def converted_output_tag(original_tag: str) -> str:
    if original_tag == "vehicle":
        return "trip"
    return original_tag


def build_converted_attrs(
    elem: XmlElem,
    kept_edges: list[str],
) -> list[tuple[str, str]]:
    attrs = [
        (name, value)
        for name, value in elem.attrs
        if name not in ATTRS_REPLACED_ON_CONVERT
    ]
    attrs.append(("from", kept_edges[0]))
    attrs.append(("to", kept_edges[-1]))
    if len(kept_edges) > 2:
        attrs.append(("via", " ".join(kept_edges[1:-1])))
    return attrs


def remaining_children(elem: XmlElem) -> list[XmlElem]:
    return [child for child in elem.children if child.tag not in ROUTE_CHILD_TAGS]


def apply_vtype_lanechange_attrs(elem: XmlElem) -> None:
    """Set lane-change attributes on a vType, replacing any existing values."""
    replacements = dict(VTYPE_LANECHANGE_ATTRS)
    seen: set[str] = set()
    updated: list[tuple[str, str]] = []
    for name, value in elem.attrs:
        if name in replacements:
            updated.append((name, replacements[name]))
            seen.add(name)
        else:
            updated.append((name, value))
    for name, value in VTYPE_LANECHANGE_ATTRS:
        if name not in seen:
            updated.append((name, value))
    elem.attrs = updated


def refined_output_path(input_path: Path) -> Path:
    return input_path.parent / f"refined_{input_path.name}"


class RouteRefiner:
    def __init__(
        self,
        input_path: Path,
        output_path: Path,
        edge_to_junction: dict[str, str],
        network_edges: set[str],
        direct_connections: set[tuple[str, str]],
        report_path: Path,
    ) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.edge_to_junction = edge_to_junction
        self.network_edges = network_edges
        self.direct_connections = direct_connections
        self.report_path = report_path
        self.stats = ConversionStats()
        self.stats.network_edge_count = len(network_edges)
        self.stats.direct_connection_count = len(direct_connections)
        self.named_routes: dict[str, list[str] | None] = {}
        self._copy_fp: BinaryIO | None = None
        self._out_fp: BinaryIO | None = None
        self._parser: object | None = None
        self._report_fp: TextIO | None = None
        self._stack: list[XmlElem] = []
        self._depth = 0
        self._prev_end = 0

    def convert(self) -> ConversionStats:
        try:
            with self.input_path.open("rb") as parse_fp, self.input_path.open(
                "rb"
            ) as copy_fp, self.output_path.open("wb") as out_fp:
                self._copy_fp = copy_fp
                self._out_fp = out_fp
                parser = expat.ParserCreate(encoding="UTF-8")
                parser.ordered_attributes = True
                parser.buffer_text = True
                parser.StartElementHandler = self._start_element
                parser.EndElementHandler = self._end_element
                parser.CharacterDataHandler = self._char_data
                self._parser = parser
                try:
                    while True:
                        chunk = parse_fp.read(PARSE_CHUNK_SIZE)
                        if not chunk:
                            parser.Parse(b"", True)
                            break
                        parser.Parse(chunk, False)
                except RouteFormatError:
                    raise
                except expat.ExpatError as exc:
                    raise RouteFormatError(
                        f"XML parse error in {self.input_path}: {exc}",
                        line=exc.lineno,
                    ) from exc
                finally:
                    if self._out_fp is not None:
                        self._out_fp.flush()
                        os.fsync(self._out_fp.fileno())
                    self._parser = None
                    self._copy_fp = None
                    self._out_fp = None
        finally:
            if self._report_fp is not None:
                self._report_fp.close()
                self._report_fp = None
        if self.stats.converted_by_missing_connection:
            self.stats.missing_connections_report = self.report_path
        elif self.report_path.exists():
            self.report_path.unlink()
        return self.stats

    def _start_element(self, name: str, attrs: object) -> None:
        parser = self._parser
        copy_fp = self._copy_fp
        out_fp = self._out_fp
        assert parser is not None and copy_fp is not None and out_fp is not None
        self._depth += 1
        elem = XmlElem(tag=name, attrs=pair_expat_attrs(attrs))
        # Do not attach top-level children to <routes>; that would retain every
        # vehicle in memory. Nested children still hang off the current element.
        if self._depth > 2 and self._stack:
            self._stack[-1].children.append(elem)
        self._stack.append(elem)
        if self._depth != 1:
            return
        if name != "routes":
            raise RouteFormatError(
                f"Expected root element <routes>, found <{name}>",
                line=parser.CurrentLineNumber,
            )
        header_end = skip_xml_tag(copy_fp, parser.CurrentByteIndex)
        out_fp.write(read_slice(copy_fp, 0, header_end))
        self._prev_end = header_end

    def _end_element(self, name: str) -> None:
        parser = self._parser
        copy_fp = self._copy_fp
        out_fp = self._out_fp
        assert parser is not None and copy_fp is not None and out_fp is not None
        if not self._stack or self._stack[-1].tag != name:
            raise RouteFormatError(
                f"Mismatched end tag </{name}>",
                line=parser.CurrentLineNumber,
            )
        elem = self._stack.pop()
        self._register_named_route(elem, line=parser.CurrentLineNumber)
        if self._depth == 2:
            elem_end = exclusive_element_end(copy_fp, parser.CurrentByteIndex)
            raw = read_slice(copy_fp, self._prev_end, elem_end)
            self._handle_top_level(elem, raw, line=parser.CurrentLineNumber)
            self._prev_end = elem_end
        elif self._depth == 1:
            rest_end = exclusive_element_end(copy_fp, parser.CurrentByteIndex)
            out_fp.write(read_slice(copy_fp, self._prev_end, rest_end))
            copy_fp.seek(rest_end)
            trailing = copy_fp.read()
            if trailing:
                out_fp.write(trailing)
        self._depth -= 1

    def _char_data(self, data: str) -> None:
        if self._depth <= 1 or not self._stack:
            return
        current = self._stack[-1]
        if current.children:
            return
        current.text += data

    def _register_named_route(self, elem: XmlElem, *, line: int) -> None:
        if elem.tag != "route":
            return
        route_id = get_attr(elem, "id")
        if route_id is None:
            return
        if route_id in self.named_routes:
            raise RouteFormatError(
                f"Duplicate shared route id {route_id!r}",
                line=line,
            )
        edges_attr = get_attr(elem, "edges")
        if edges_attr is None:
            self.named_routes[route_id] = None
            return
        self.named_routes[route_id] = parse_edge_list(
            edges_attr,
            context=f"shared route id {route_id!r}",
        )

    def _handle_top_level(self, elem: XmlElem, raw: bytes, *, line: int) -> None:
        out_fp = self._out_fp
        assert out_fp is not None
        if elem.tag == "vType":
            apply_vtype_lanechange_attrs(elem)
            newline = detect_newline(raw)
            prefix, _element = split_prefix(raw, elem.tag)
            indent = indent_from_prefix(prefix, newline)
            leading = prefix_before_indent(prefix, newline).decode("utf-8")
            xml_text = leading + serialize_element(
                elem,
                indent=indent,
                newline=newline,
            )
            out_fp.write(xml_text.encode("utf-8"))
            return
        if elem.tag not in DEMAND_TAGS:
            out_fp.write(raw)
            return
        self._process_demand_element(elem, raw, line=line)

    def _process_demand_element(
        self,
        elem: XmlElem,
        raw: bytes,
        *,
        line: int,
    ) -> None:
        self.stats.total_vehicles += 1
        edges = self._resolve_edges(elem, line=line)
        after_remove, removed_count = remove_independent_edges(edges)
        kept_edges, per_junction = compress_junction_runs(
            after_remove,
            self.edge_to_junction,
        )
        compressed_edge_count = 0
        for name, junction_stats in per_junction.items():
            self.stats.junction_runs[name] += junction_stats["runs"]
            compressed_edge_count += junction_stats["removed_internal"]
        modified = removed_count > 0 or compressed_edge_count > 0
        if not modified:
            self._keep_or_fix_complete_route(elem, raw, kept_edges, line=line)
            return

        for name, junction_stats in per_junction.items():
            self.stats.compressed_runs[name] += junction_stats["compressed"]
            self.stats.removed_internal_edges[name] += junction_stats[
                "removed_internal"
            ]
        self.stats.removed_edges += removed_count
        if len(kept_edges) < 2:
            self.stats.dropped_vehicles += 1
            self.stats.dropped_vehicle_ids.append(demand_id(elem))
            return

        self._write_trip(elem, raw, kept_edges)
        self.stats.converted_vehicles += 1

    def _keep_or_fix_complete_route(
        self,
        elem: XmlElem,
        raw: bytes,
        edges: list[str],
        *,
        line: int,
    ) -> None:
        out_fp = self._out_fp
        assert out_fp is not None
        if not is_complete_route_vehicle(elem):
            out_fp.write(raw)
            self.stats.unaffected_vehicles += 1
            return

        self.stats.checked_full_routes += 1
        vehicle_id = demand_id(elem)
        missing_edges = missing_route_edges(edges, self.network_edges)
        if missing_edges:
            raise RouteFormatError(
                f"Vehicle {vehicle_id!r} contains edges missing from the new network: "
                f"{', '.join(missing_edges)}. Add them to REMOVED_EDGES or define a mapping.",
                line=line,
            )
        if len(edges) < 2:
            out_fp.write(raw)
            self.stats.unaffected_vehicles += 1
            self.stats.kept_full_routes += 1
            return
        missing_pairs = missing_direct_pairs(edges, self.direct_connections)
        if not missing_pairs:
            out_fp.write(raw)
            self.stats.unaffected_vehicles += 1
            self.stats.kept_full_routes += 1
            return
        self._record_missing_connections(vehicle_id, missing_pairs)
        self._write_trip(elem, raw, edges)
        self.stats.converted_by_missing_connection += 1

    def _record_missing_connections(
        self,
        vehicle_id: str,
        missing_pairs: list[tuple[str, str]],
    ) -> None:
        self.stats.missing_connection_occurrences += len(missing_pairs)
        self.stats.missing_pair_counts.update(missing_pairs)
        if self._report_fp is None:
            self._report_fp = self.report_path.open("w", encoding="utf-8")
            self._report_fp.write("vehicle_id\tfrom_edge\tto_edge\n")
        for from_edge, to_edge in missing_pairs:
            self._report_fp.write(f"{vehicle_id}\t{from_edge}\t{to_edge}\n")

    def _write_trip(self, elem: XmlElem, raw: bytes, kept_edges: list[str]) -> None:
        out_fp = self._out_fp
        assert out_fp is not None
        newline = detect_newline(raw)
        prefix, _element = split_prefix(raw, elem.tag)
        indent = indent_from_prefix(prefix, newline)
        leading = prefix_before_indent(prefix, newline).decode("utf-8")
        converted = XmlElem(
            tag=converted_output_tag(elem.tag),
            attrs=build_converted_attrs(elem, kept_edges),
            children=remaining_children(elem),
        )
        xml_text = leading + serialize_element(
            converted,
            indent=indent,
            newline=newline,
        )
        out_fp.write(xml_text.encode("utf-8"))

    def _resolve_edges(self, elem: XmlElem, *, line: int) -> list[str]:
        kind = elem.tag
        vid = demand_id(elem)
        context = f"{kind} id {vid!r}"
        if any(child.tag == "routeDistribution" for child in elem.children):
            raise RouteFormatError(
                f"{context}: <routeDistribution> is not supported; "
                "need a nested <route edges=\"...\">, a route= reference, or from/to",
                line=line,
            )
        nested_routes = [child for child in elem.children if child.tag == "route"]
        if len(nested_routes) > 1:
            raise RouteFormatError(
                f"{context}: expected at most one nested <route>, found {len(nested_routes)}",
                line=line,
            )
        if len(nested_routes) == 1:
            nested = nested_routes[0]
            edges_attr = get_attr(nested, "edges")
            if edges_attr is None:
                raise RouteFormatError(
                    f"{context}: nested <route> is missing the edges attribute",
                    line=line,
                )
            return parse_edge_list(edges_attr, context=f"{context} nested <route>")

        route_ref = get_attr(elem, "route")
        if route_ref is not None:
            if route_ref not in self.named_routes:
                raise RouteFormatError(
                    f"{context}: references unknown shared route id {route_ref!r}",
                    line=line,
                )
            edges = self.named_routes[route_ref]
            if edges is None:
                raise RouteFormatError(
                    f"{context}: shared route id {route_ref!r} has no edges attribute",
                    line=line,
                )
            return list(edges)

        from_edge = get_attr(elem, "from")
        to_edge = get_attr(elem, "to")
        if from_edge is not None and to_edge is not None:
            via_attr = get_attr(elem, "via")
            via_edges = via_attr.split() if via_attr else []
            return [from_edge, *via_edges, to_edge]

        raise RouteFormatError(
            f"{context}: missing route edges (no nested <route>, "
            "no route= reference, and no from/to attributes)",
            line=line,
        )


def print_summary(stats: ConversionStats, output_path: Path) -> None:
    print("转换统计")
    print("--------")
    print(f"总车辆数：{stats.total_vehicles}")
    print(f"未受影响车辆数：{stats.unaffected_vehicles}")
    print(f"因路口压缩或固定 edge 删除而转换成 trip 的车辆数量：{stats.converted_vehicles}")
    for name in JUNCTION_EDGE_SETS:
        print(f"删除 REMOVED_EDGES 后，{name} 找到的连续子序列数量：{stats.junction_runs[name]}")
        print(f"{name} 实际压缩的子序列数量：{stats.compressed_runs[name]}")
        print(f"{name} 因压缩而删除的内部 edge 数量：{stats.removed_internal_edges[name]}")
    print(f"REMOVED_EDGES 条目数：{len(REMOVED_EDGES)}")
    print(f"REMOVED_EDGES 的删除次数：{stats.removed_edges}")
    print(f"因最终不足两条 edge 而删除的车辆数：{stats.dropped_vehicles}")
    print(f"从新路网读取的非 internal edge 数量：{stats.network_edge_count}")
    print(f"从新路网读取的直接 edge connection 数量：{stats.direct_connection_count}")
    print(f"接受连接性检查的完整 route 数量：{stats.checked_full_routes}")
    print(f"所有相邻 pair 都合法、保持完整 route 的车辆数量：{stats.kept_full_routes}")
    print(
        "因至少一个相邻 pair 无直接 connection 而转换成 trip 的车辆数量："
        f"{stats.converted_by_missing_connection}"
    )
    print(f"缺失 direct connection 的 edge pair 总出现次数：{stats.missing_connection_occurrences}")
    print(f"缺失 direct connection 的唯一 edge pair 数量：{len(stats.missing_pair_counts)}")
    print(
        "最终 trip 总数："
        f"{stats.converted_vehicles + stats.converted_by_missing_connection}"
    )
    print(f"最终保留完整 route 的车辆总数：{stats.kept_full_routes}")
    if stats.missing_connections_report is not None:
        print(f"缺失 connection 报告路径：{stats.missing_connections_report}")
    print(f"输出文件路径：{output_path}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "先删除已从路网移除的 edge，再按各命名路口集合压缩内部连续 edge 子序列，"
            "然后按新路网 connection 检查未修改完整 route 的相邻连通性。"
            "未受影响车辆保持原样。输出写到 refined_<输入文件名>。"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=f"输入 route XML（例如：{DEFAULT_INPUT_FILE.name}）",
    )
    parser.add_argument(
        "--net",
        type=Path,
        required=True,
        help=f"修改后的 SUMO 路网 XML（例如：{DEFAULT_NET_FILE.name}）",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def resolve_io_paths(input_path: Path, net_path: Path) -> tuple[Path, Path, Path]:
    input_resolved = input_path.expanduser().resolve()
    net_resolved = net_path.expanduser().resolve()
    if not input_resolved.is_file():
        raise RouteFormatError(f"Input file not found: {input_resolved}")
    if not net_resolved.is_file():
        raise RouteFormatError(f"Network file not found: {net_resolved}")
    output_resolved = refined_output_path(input_resolved).resolve()
    if output_resolved == input_resolved:
        raise RouteFormatError(
            "Refusing to overwrite the input file; "
            f"automatic output path is {output_resolved}"
        )
    return input_resolved, net_resolved, output_resolved


def replace_output_file(tmp_path: Path, output_path: Path) -> None:
    """
    Copy tmp_path onto output_path, overwriting in place.

    Do not use os.replace()/rename(). On WSL drvfs (/mnt/f, ...) rename can
    unlink the destination and then fail with ENOENT, after which the same
    name cannot be recreated until Windows releases the deleted file.
    """
    if not tmp_path.is_file():
        raise RouteFormatError(f"Temporary output was not created: {tmp_path}")
    last_error: OSError | None = None
    for attempt in range(15):
        try:
            with tmp_path.open("rb") as source, output_path.open("wb") as dest:
                shutil.copyfileobj(source, dest, length=PARSE_CHUNK_SIZE)
                dest.flush()
                os.fsync(dest.fileno())
            tmp_path.unlink(missing_ok=True)
            return
        except OSError as exc:
            last_error = exc
            if exc.errno != errno.ENOENT or attempt == 14:
                break
            time.sleep(0.2)
    raise RouteFormatError(
        f"Failed to write output file {output_path}: {last_error}. "
        "WSL /mnt/f cannot always recreate a filename after a failed rename; retry the command."
    ) from last_error


def convert_file(input_path: Path, net_path: Path) -> tuple[ConversionStats, Path]:
    edge_to_junction = validate_disjoint_junction_sets()
    input_resolved, net_resolved, output_resolved = resolve_io_paths(
        input_path,
        net_path,
    )
    network_edges, direct_connections = load_sumo_network(net_resolved)
    output_resolved.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_resolved.with_name(
        f"{output_resolved.name}.{os.getpid()}.partial"
    )
    report_path = missing_connections_report_path(output_resolved)
    converter = RouteRefiner(
        input_resolved,
        tmp_path,
        edge_to_junction,
        network_edges,
        direct_connections,
        report_path,
    )
    try:
        stats = converter.convert()
        replace_output_file(tmp_path, output_resolved)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        if report_path.exists():
            report_path.unlink()
        raise
    return stats, output_resolved


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        stats, output_resolved = convert_file(args.input, args.net)
    except (RouteFormatError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    print_summary(stats, output_resolved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
