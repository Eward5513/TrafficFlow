"""
生成 20 天的日需求数据，每天一份 route 文件，全部落在 `data/` 下。

每天的构成：

1. 匹配轨迹翻倍 —— `analysis/matching/matched_routes.rou.xml` 里每条严格匹配
   还原的真实轨迹产出两辆车（原车 id 不变、复制车 id 加 `.dup1`），两辆车各自
   独立地在原 depart 上加 [-300, +300) 秒的抖动（±5 分钟），负值截到 0；
2. 随机背景流 —— 当天对应的 `data/random_trip/<编号>_<种子>.rou.xml`，3 万条，
   既不翻倍也不抖动，depart 原样保留。

编号与种子都来自 `data/random_trip/` 里的文件名（由 `generate_random_routes.py`
生成），本脚本不自己抽随机数：同一个种子既是当天随机流的生成种子，也是当天
depart 抖动的种子，因此重跑结果完全一致。输出文件与随机流共用同一个词干，写到
`data/route_demand/<编号>_<种子>.rou.xml`。

清洗规则对每天的全部车辆生效，与旧版一致：

- 反向边去环：遍历边序列，若相邻两条边互为反向边（起终点相同、方向相反，由
  `analysis/road_network/find_reverse_edge_pairs.py` 提供），视为多余的环并整对
  删除；起点边不参与配对；删除后需保证轨迹仍连通，否则不删；不处理嵌套环。
- 去短轨迹：边序列长度不足 3 的车辆整辆丢弃。

去环和去短只看边序列、与 depart 无关，所以匹配轨迹那部分只清洗一次就被 20 天
共用，结果与「每天各清洗一次」完全等价。

用法：

    python analysis/simulation/merge_basic_routes.py            # 处理发现的全部天
    python analysis/simulation/merge_basic_routes.py --only 1   # 只跑第 1 天
"""

from __future__ import annotations

import argparse
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = BASE_DIR.parent

# ---- 硬编码配置 ----
MATCHED_ROUTES_FILE = ANALYSIS_DIR / "matching" / "matched_routes.rou.xml"
# 由 analysis/road_network/find_reverse_edge_pairs.py 生成的反向边对列表
REVERSE_PAIRS_FILE = ANALYSIS_DIR / "road_network" / "reverse_edge_pairs.txt"
# 用于连通性检测的路网文件
NET_FILE = ANALYSIS_DIR / "road_network" / "net_tls.net.xml"
DATA_DIR = BASE_DIR / "data"
# generate_random_routes.py 的产物目录，同时也是编号与种子的唯一来源
RANDOM_DIR = DATA_DIR / "random_trip"
# 20 份日需求的输出目录
DEMAND_DIR = DATA_DIR / "route_demand"

# 匹配轨迹翻倍倍数（含原车）
DUPLICATE_FACTOR = 2
# 复制车 id 后缀，最终形如 <原 id>.dup1
DUPLICATE_ID_SUFFIX = ".dup"
# 出发时间抖动幅度（秒），双向，即 ±5 分钟
JITTER_SECONDS = 300.0
# 边序列短于该长度的轨迹整辆丢弃
MIN_ROUTE_EDGES = 3
# 随机背景流车辆 type=random_passenger；duarouter 常把 vType 写到单独文件，
# rou.xml 里未必带定义，合并日需求时手动补上。
RANDOM_VTYPE_ID = "random_passenger"
RANDOM_VTYPE_CLASS = "passenger"

XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
ROUTE_SUFFIX = ".rou.xml"
STEM_PATTERN = re.compile(r"^(\d+)_(\d+)$")
# 排序时用来区分来源，保证同一 depart 下的输出顺序稳定
SOURCE_MATCHED = 0
SOURCE_RANDOM = 1

ET.register_namespace("xsi", XSI_NS)


@dataclass(slots=True)
class DaySource:
    """一天的输入：编号、种子、以及当天的随机背景流文件。"""

    index: int
    seed: int
    random_file: Path

    @property
    def output_file(self) -> Path:
        return DEMAND_DIR / f"{self.index}_{self.seed}{ROUTE_SUFFIX}"


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_text_tolerant(path: Path) -> str:
    """读取文本文件，自动尝试常见编码（utf-8 / gbk / latin-1）。"""
    data = path.read_bytes()
    for encoding in ("utf-8", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def load_reverse_pairs(path: Path) -> set[tuple[str, str]]:
    """从 find_reverse_edge_pairs.py 的输出文件中解析反向边对。

    同时存入 (a, b) 和 (b, a)，方便顺序无关地判断。
    """
    pairs: set[tuple[str, str]] = set()
    pattern = re.compile(r'\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)')
    for line in read_text_tolerant(path).splitlines():
        match = pattern.search(line)
        if match:
            a, b = match.group(1), match.group(2)
            pairs.add((a, b))
            pairs.add((b, a))
    return pairs


def load_edge_connections(net_file: Path) -> set[tuple[str, str]]:
    """从路网文件中读取边级连通关系：(from_edge, to_edge) 集合。

    只统计普通边之间的 connection，忽略 internal 边。
    """
    connections: set[tuple[str, str]] = set()
    for _, elem in ET.iterparse(net_file, events=("end",)):
        if elem.tag == "connection":
            from_edge = elem.get("from")
            to_edge = elem.get("to")
            if (
                from_edge
                and to_edge
                and not from_edge.startswith(":")
                and not to_edge.startswith(":")
            ):
                connections.add((from_edge, to_edge))
        elem.clear()
    return connections


def prune_reverse_loops(
    edges: list[str],
    reverse_pairs: set[tuple[str, str]],
    connections: set[tuple[str, str]],
) -> list[str]:
    """去掉轨迹中相邻出现的反向边对（多余的环）。

    规则：
    - edges[i] 与 edges[i+1] 互为反向边即视为多余的环，整对删除；
    - 起点边（索引 0）不参与配对；
    - 删除后必须保证轨迹仍然连通：删除对前面的边要能直接连到删除对
      后面的边（若删除对位于轨迹末尾则无需检查），否则不删；
    - 单向前扫，不回退，不处理删除后新产生的嵌套环。
    """
    result = list(edges)
    i = 1
    while i + 1 < len(result):
        if (result[i], result[i + 1]) not in reverse_pairs:
            i += 1
            continue

        pair_at_end = i + 2 >= len(result)
        still_connected = pair_at_end or (
            (result[i - 1], result[i + 2]) in connections
        )
        if not still_connected:
            i += 1
            continue

        # 删除这一对后，former result[i+2] 移到当前 i，继续向前扫描，
        # 不回退检查 (result[i-1], 新 result[i]) 这个新接缝，即忽略嵌套环。
        del result[i : i + 2]
    return result


def prune_vehicle_routes(
    vehicles: list[ET.Element],
    reverse_pairs: set[tuple[str, str]],
    connections: set[tuple[str, str]],
) -> tuple[int, int]:
    """对每辆车的 route 边序列去环，返回 (受影响车辆数, 删除的边数)。"""
    pruned_vehicles = 0
    removed_edges = 0
    for vehicle in vehicles:
        for child in vehicle:
            if local_name(child.tag) != "route":
                continue
            edges_attr = child.get("edges")
            if not edges_attr:
                continue
            edges = edges_attr.split()
            pruned = prune_reverse_loops(edges, reverse_pairs, connections)
            if len(pruned) != len(edges):
                child.set("edges", " ".join(pruned))
                pruned_vehicles += 1
                removed_edges += len(edges) - len(pruned)
    return pruned_vehicles, removed_edges


def filter_short_routes(
    vehicles: list[ET.Element],
    min_edges: int = MIN_ROUTE_EDGES,
) -> tuple[list[ET.Element], int]:
    """去掉边序列长度不足 min_edges 的短轨迹。

    返回 (保留的车辆列表, 被删除的车辆数)。
    """
    kept: list[ET.Element] = []
    removed = 0
    for vehicle in vehicles:
        route = next(
            (child for child in vehicle if local_name(child.tag) == "route"), None
        )
        edges = (route.get("edges") or "").split() if route is not None else []
        if len(edges) < min_edges:
            removed += 1
            continue
        kept.append(vehicle)
    return kept, removed


def depart_value(vehicle: ET.Element) -> float | None:
    """取车辆的数值型 depart；`triggered` 之类的非数值写法返回 None。"""
    try:
        return float(vehicle.get("depart", "0"))
    except ValueError:
        return None


def load_route_elements(
    source: Path,
) -> tuple[ET.Element, list[ET.Element], list[ET.Element], list[ET.Element]]:
    if not source.exists():
        raise FileNotFoundError(f"Input route file not found: {source}")

    root = ET.parse(source).getroot()
    if local_name(root.tag) != "routes":
        raise ValueError(f"Input route file root is not <routes>: {source}")

    vtypes: list[ET.Element] = []
    vehicles: list[ET.Element] = []
    others: list[ET.Element] = []
    for child in root:
        child_name = local_name(child.tag)
        if child_name == "vType":
            vtypes.append(child)
        elif child_name == "vehicle":
            vehicles.append(child)
        else:
            others.append(child)

    return root, vtypes, others, vehicles


def make_random_passenger_vtype() -> ET.Element:
    return ET.Element(
        "vType", {"id": RANDOM_VTYPE_ID, "vClass": RANDOM_VTYPE_CLASS}
    )


def merge_vtypes(collections: list[list[ET.Element]]) -> list[ET.Element]:
    """按 id 去重合并 vType，定义冲突直接报错。"""
    by_id: dict[str, ET.Element] = {}
    extras: list[ET.Element] = []
    for vtypes in collections:
        for vtype in vtypes:
            vtype_id = vtype.get("id")
            if vtype_id is None:
                extras.append(vtype)
                continue
            existing = by_id.get(vtype_id)
            if existing is None:
                by_id[vtype_id] = vtype
            elif ET.tostring(existing) != ET.tostring(vtype):
                raise ValueError(f"Conflicting vType definition for id={vtype_id!r}.")
    return [*by_id.values(), *extras]


def prepare_vehicle_for_output(vehicle: ET.Element) -> None:
    """把单个 <vehicle> 排版成 4 空格缩进的形状，供逐辆序列化时直接写出。

    车辆本身缩进一级由写出方补上，这里只处理内部的换行与子元素缩进。
    """
    children = list(vehicle)
    if children:
        vehicle.text = "\n        "
        for child in children[:-1]:
            child.tail = "\n        "
        children[-1].tail = "\n    "
    else:
        vehicle.text = None
    vehicle.tail = None


def build_header(
    root_tag: str,
    root_attrib: dict[str, str],
    header_elements: list[ET.Element],
) -> tuple[str, str]:
    """生成输出文件的根标签开头和结尾。

    返回 (含头部元素的开头片段, 闭合标签)。
    """
    header_root = ET.Element(root_tag, root_attrib)
    for element in header_elements:
        header_root.append(element)
    ET.indent(header_root, space="    ")
    header_xml = ET.tostring(header_root, encoding="unicode")

    closing = f"</{root_tag}>"
    prefix, separator, _ = header_xml.rpartition(closing)
    if separator:
        return prefix, closing

    # 没有任何头部元素时 ElementTree 会输出自闭合根标签，手工拆成开闭两半
    stripped = header_xml.rstrip()
    if not stripped.endswith("/>"):
        raise ValueError(f"unexpected root serialization: {header_xml!r}")
    return f"{stripped[:-2].rstrip()}>\n", closing


def discover_days(random_dir: Path) -> list[DaySource]:
    """从 data/random_trip/ 的文件名解析出每天的编号与种子。

    校验编号唯一、从 1 起连续，避免下游拿到不完整的天序列。
    """
    if not random_dir.is_dir():
        raise SystemExit(
            f"random flow directory not found: {random_dir}\n"
            "Run generate_random_routes.py first (needs SUMO_HOME)."
        )

    by_index: dict[int, DaySource] = {}
    for route_file in sorted(random_dir.glob(f"*_*{ROUTE_SUFFIX}")):
        match = STEM_PATTERN.match(route_file.name[: -len(ROUTE_SUFFIX)])
        if match is None:
            continue
        index, seed = int(match.group(1)), int(match.group(2))
        if index in by_index:
            raise SystemExit(
                f"day {index} maps to more than one random flow file in {random_dir}; "
                "keep exactly one file per day index"
            )
        by_index[index] = DaySource(index=index, seed=seed, random_file=route_file)

    if not by_index:
        raise SystemExit(
            f"no random flow files matching <index>_<seed>{ROUTE_SUFFIX} in {random_dir}\n"
            "Run generate_random_routes.py first (needs SUMO_HOME)."
        )

    expected = list(range(1, len(by_index) + 1))
    missing = [index for index in expected if index not in by_index]
    if missing:
        raise SystemExit(
            "random flow day indices must start at 1 and be contiguous; "
            f"missing {missing} while {sorted(by_index)} are present"
        )
    return [by_index[index] for index in expected]


def load_matched_vehicles(
    reverse_pairs: set[tuple[str, str]],
    connections: set[tuple[str, str]],
) -> tuple[ET.Element, list[ET.Element], list[ET.Element], list[ET.Element]]:
    """解析并清洗匹配轨迹，结果被 20 天共用。

    返回 (根元素, vType 列表, 其他头部元素, 清洗后的车辆列表)。
    """
    root, vtypes, others, vehicles = load_route_elements(MATCHED_ROUTES_FILE)
    print(f"Loaded {len(vehicles)} matched trajectories from {MATCHED_ROUTES_FILE}")

    pruned_vehicles, removed_edges = prune_vehicle_routes(
        vehicles, reverse_pairs, connections
    )
    print(
        f"Pruned reverse-edge loops: {pruned_vehicles} vehicles affected, "
        f"{removed_edges} edges removed."
    )

    vehicles, removed_short = filter_short_routes(vehicles)
    print(
        f"Filtered short routes (<{MIN_ROUTE_EDGES} edges): "
        f"{removed_short} vehicles removed, {len(vehicles)} kept."
    )

    for vehicle in vehicles:
        prepare_vehicle_for_output(vehicle)
    return root, vtypes, others, vehicles


def load_random_vehicles(
    day: DaySource,
    reverse_pairs: set[tuple[str, str]],
    connections: set[tuple[str, str]],
) -> tuple[list[ET.Element], list[ET.Element], list[ET.Element]]:
    """解析并清洗当天的随机背景流。

    返回 (vType 列表, 其他头部元素, 清洗后的车辆列表)。
    """
    _, vtypes, others, vehicles = load_route_elements(day.random_file)
    pruned_vehicles, removed_edges = prune_vehicle_routes(
        vehicles, reverse_pairs, connections
    )
    vehicles, removed_short = filter_short_routes(vehicles)
    print(
        f"  random flow {day.random_file.name}: {len(vehicles)} kept "
        f"({pruned_vehicles} pruned, {removed_edges} edges removed, "
        f"{removed_short} short routes dropped)"
    )

    for vehicle in vehicles:
        prepare_vehicle_for_output(vehicle)
    return vtypes, others, vehicles


def build_day(
    day: DaySource,
    matched_root: ET.Element,
    matched_vtypes: list[ET.Element],
    matched_others: list[ET.Element],
    matched_vehicles: list[ET.Element],
    matched_ids: list[str],
    matched_departs: list[float | None],
    reverse_pairs: set[tuple[str, str]],
    connections: set[tuple[str, str]],
) -> int:
    """构建并写出一天的 route 文件，返回车辆总数。"""
    random_vtypes, random_others, random_vehicles = load_random_vehicles(
        day, reverse_pairs, connections
    )

    # 抖动种子沿用当天随机流的生成种子，重跑结果一致
    rng = random.Random(day.seed)
    entries: list[tuple[float, int, int, int]] = []
    for order, base_depart in enumerate(matched_departs):
        for copy_index in range(DUPLICATE_FACTOR):
            # 无论 depart 是否可解析都抽一次，保证随机数流与数据无关
            jitter = rng.uniform(-JITTER_SECONDS, JITTER_SECONDS)
            if base_depart is None:
                depart = float("inf")
            else:
                depart = max(0.0, base_depart + jitter)
            entries.append((depart, SOURCE_MATCHED, order, copy_index))

    for order, vehicle in enumerate(random_vehicles):
        base_depart = depart_value(vehicle)
        depart = float("inf") if base_depart is None else base_depart
        entries.append((depart, SOURCE_RANDOM, order, 0))

    entries.sort()

    header_elements = merge_vtypes(
        [matched_vtypes, random_vtypes, [make_random_passenger_vtype()]]
    )
    header_elements.extend(matched_others)
    header_elements.extend(random_others)
    prefix, closing = build_header(
        local_name(matched_root.tag), dict(matched_root.attrib), header_elements
    )

    DEMAND_DIR.mkdir(parents=True, exist_ok=True)
    with day.output_file.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("<?xml version='1.0' encoding='utf-8'?>\n")
        handle.write(prefix)
        for depart, source, order, copy_index in entries:
            if source == SOURCE_MATCHED:
                vehicle = matched_vehicles[order]
                base_id = matched_ids[order]
                vehicle.set(
                    "id",
                    base_id
                    if copy_index == 0
                    else f"{base_id}{DUPLICATE_ID_SUFFIX}{copy_index}",
                )
                if matched_departs[order] is not None:
                    vehicle.set("depart", f"{depart:.2f}")
            else:
                vehicle = random_vehicles[order]
            handle.write("    ")
            handle.write(ET.tostring(vehicle, encoding="unicode"))
            handle.write("\n")
        handle.write(f"{closing}\n")

    return len(entries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one route file per day from the matched trajectories (doubled, "
            "depart jittered by +/-5 min) plus that day's random background flow."
        ),
    )
    parser.add_argument(
        "--only",
        type=int,
        nargs="+",
        default=None,
        metavar="INDEX",
        help="Only build these day indices, for example --only 1 or --only 1 2 3.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for required in (MATCHED_ROUTES_FILE, REVERSE_PAIRS_FILE, NET_FILE):
        if not required.exists():
            raise SystemExit(f"required input not found: {required}")

    days = discover_days(RANDOM_DIR)
    print(f"Discovered {len(days)} day(s) of random background flow in {RANDOM_DIR}")

    if args.only:
        wanted = set(args.only)
        unknown = sorted(wanted - {day.index for day in days})
        if unknown:
            raise SystemExit(f"no random flow file for day index(es): {unknown}")
        days = [day for day in days if day.index in wanted]

    reverse_pairs = load_reverse_pairs(REVERSE_PAIRS_FILE)
    print(f"Loaded {len(reverse_pairs) // 2} reverse edge pairs from {REVERSE_PAIRS_FILE}")
    connections = load_edge_connections(NET_FILE)
    print(f"Loaded {len(connections)} edge connections from {NET_FILE}")

    print("\n== Loading matched trajectories ==")
    matched_root, matched_vtypes, matched_others, matched_vehicles = (
        load_matched_vehicles(reverse_pairs, connections)
    )
    # id 与 depart 必须在任何改写之前留档，写出时反复覆写这两个属性
    matched_ids = [vehicle.get("id", "") for vehicle in matched_vehicles]
    matched_departs = [depart_value(vehicle) for vehicle in matched_vehicles]

    counts: list[tuple[DaySource, int]] = []
    for day in days:
        print(f"\n== Day {day.index} (seed {day.seed}) ==")
        if day.output_file.exists():
            print(f"  overwriting existing {day.output_file.name}")
        total = build_day(
            day,
            matched_root,
            matched_vtypes,
            matched_others,
            matched_vehicles,
            matched_ids,
            matched_departs,
            reverse_pairs,
            connections,
        )
        counts.append((day, total))
        print(f"  wrote {day.output_file} ({total} vehicles)")

    print("\n== Summary ==")
    print(
        f"matched trajectories per day: {len(matched_vehicles)} x {DUPLICATE_FACTOR} "
        f"= {len(matched_vehicles) * DUPLICATE_FACTOR}"
    )
    for day, total in counts:
        print(f"day {day.index} (seed {day.seed}): {total} vehicles -> {day.output_file.name}")


if __name__ == "__main__":
    main()
