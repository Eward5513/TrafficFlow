from __future__ import annotations

from collections import deque
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DEFAULT_NET_FILE = PROJECT_ROOT / "analysis" / "data" / "net_tls1.net.xml"
DEFAULT_VEHICLE_CLASS = "passenger"
NearbyPathCache = dict[tuple[str, str], tuple[int, list[str]]]
VehicleClassSet = frozenset[str]
PermissionRule = tuple[VehicleClassSet | None, VehicleClassSet | None]


@dataclass(slots=True)
class LaneInfo:
    lane_id: str
    length: float | None
    speed: float | None
    allow_classes: VehicleClassSet | None = None
    disallow_classes: VehicleClassSet | None = None

    def allows_vehicle_class(self, vehicle_class: str) -> bool:
        if self.allow_classes is not None:
            return vehicle_class in self.allow_classes
        if self.disallow_classes is not None:
            return vehicle_class not in self.disallow_classes
        return True


@dataclass(slots=True)
class EdgeInfo:
    edge_id: str
    from_node: str | None
    to_node: str | None
    priority: int | None
    edge_type: str | None
    function: str | None
    lanes: list[LaneInfo] = field(default_factory=list)

    @property
    def lane_ids(self) -> list[str]:
        return [lane.lane_id for lane in self.lanes]

    @property
    def lane_lengths(self) -> list[float | None]:
        return [lane.length for lane in self.lanes]

    @property
    def lane_speeds(self) -> list[float | None]:
        return [lane.speed for lane in self.lanes]

    def allows_vehicle_class(self, vehicle_class: str) -> bool:
        return any(lane.allows_vehicle_class(vehicle_class) for lane in self.lanes)


@dataclass(slots=True)
class SumoEdgeGraph:
    edges: dict[str, EdgeInfo]
    graph: dict[str, set[str]]
    reverse_graph: dict[str, set[str]]
    nearby_path_cache: NearbyPathCache = field(default_factory=dict)


def strip_namespace(tag: str) -> str:
    """Return a tag name without an XML namespace prefix."""
    return tag.rsplit("}", 1)[-1]


def parse_optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def parse_optional_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def parse_vehicle_classes(value: str | None) -> VehicleClassSet | None:
    if value in (None, ""):
        return None
    return frozenset(value.split())


def resolve_permissions(
    *,
    allow: str | None,
    disallow: str | None,
    default_rule: PermissionRule,
) -> PermissionRule:
    allow_classes = parse_vehicle_classes(allow)
    disallow_classes = parse_vehicle_classes(disallow)
    default_allow, default_disallow = default_rule
    return (
        allow_classes if allow_classes is not None else default_allow,
        disallow_classes if disallow_classes is not None else default_disallow,
    )


def parse_edge_element(
    edge_elem: ET.Element,
    edge_type_rules: dict[str, PermissionRule],
) -> EdgeInfo:
    """Extract one normal SUMO edge and its effective lane permissions."""
    edge_type = edge_elem.get("type")
    default_rule = edge_type_rules.get(edge_type or "", (None, None))
    lanes: list[LaneInfo] = []
    for child in edge_elem:
        if strip_namespace(child.tag) != "lane":
            continue
        allow_classes, disallow_classes = resolve_permissions(
            allow=child.get("allow"),
            disallow=child.get("disallow"),
            default_rule=default_rule,
        )
        lanes.append(
            LaneInfo(
                lane_id=child.get("id", ""),
                length=parse_optional_float(child.get("length")),
                speed=parse_optional_float(child.get("speed")),
                allow_classes=allow_classes,
                disallow_classes=disallow_classes,
            )
        )

    return EdgeInfo(
        edge_id=edge_elem.get("id", ""),
        from_node=edge_elem.get("from"),
        to_node=edge_elem.get("to"),
        priority=parse_optional_int(edge_elem.get("priority")),
        edge_type=edge_type,
        function=edge_elem.get("function"),
        lanes=lanes,
    )


def is_normal_edge(edge_id: str | None, function: str | None) -> bool:
    if not edge_id:
        return False
    if edge_id.startswith(":"):
        return False
    if function == "internal":
        return False
    return True


def parse_sumo_net(
    net_file: Path,
    vehicle_class: str = DEFAULT_VEHICLE_CLASS,
) -> tuple[dict[str, EdgeInfo], list[tuple[str, str]]]:
    """
    Parse a SUMO net.xml file in a streaming way.

    Returns:
        valid_edges:
            Mapping from normal edge id to parsed metadata for the requested
            vehicle class.
        raw_connections:
            All (from_edge, to_edge) pairs seen in <connection> elements.
    """
    net_path = Path(net_file)
    if not net_path.exists():
        raise FileNotFoundError(f"SUMO net file not found: {net_path}")

    valid_edges: dict[str, EdgeInfo] = {}
    raw_connections: list[tuple[str, str]] = []
    edge_type_rules: dict[str, PermissionRule] = {}

    try:
        context = ET.iterparse(net_path, events=("start", "end"))
        _, root = next(context)

        for event, elem in context:
            if event != "end":
                continue

            tag = strip_namespace(elem.tag)

            if tag == "type":
                type_id = elem.get("id")
                if type_id:
                    edge_type_rules[type_id] = (
                        parse_vehicle_classes(elem.get("allow")),
                        parse_vehicle_classes(elem.get("disallow")),
                    )
                elem.clear()
                root.clear()

            elif tag == "edge":
                edge_id = elem.get("id")
                function = elem.get("function")

                if is_normal_edge(edge_id, function):
                    edge_info = parse_edge_element(elem, edge_type_rules)
                    if edge_info.allows_vehicle_class(vehicle_class):
                        valid_edges[edge_info.edge_id] = edge_info

                elem.clear()
                root.clear()

            elif tag == "connection":
                from_edge = elem.get("from")
                to_edge = elem.get("to")
                if from_edge and to_edge:
                    raw_connections.append((from_edge, to_edge))
                elem.clear()
                root.clear()

    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse XML file: {net_path}") from exc
    except OSError as exc:
        raise OSError(f"Failed to read SUMO net file: {net_path}") from exc

    return valid_edges, raw_connections


def build_edge_graph(
    raw_connections: list[tuple[str, str]],
    valid_edges: dict[str, EdgeInfo],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """
    Build edge-transition graph using only normal edges as nodes.

    Connections that reference internal or missing edges are skipped.
    """
    graph: dict[str, set[str]] = {edge_id: set() for edge_id in valid_edges}
    reverse_graph: dict[str, set[str]] = {edge_id: set() for edge_id in valid_edges}

    for from_edge, to_edge in raw_connections:
        if from_edge not in valid_edges or to_edge not in valid_edges:
            continue

        graph[from_edge].add(to_edge)
        reverse_graph[to_edge].add(from_edge)

    return graph, reverse_graph


def build_nearby_path_cache(
    graph: dict[str, set[str]],
    max_hops: int = 2,
) -> NearbyPathCache:
    """
    Precompute short local paths for each edge.

    By default this caches all 1-hop and 2-hop reachable neighbors so matching
    can answer common nearby transitions without running a full BFS.
    """
    if max_hops < 1:
        return {}

    nearby_path_cache: NearbyPathCache = {}

    for start in sorted(graph):
        visited: set[str] = {start}
        queue: deque[tuple[str, list[str]]] = deque([(start, [start])])

        while queue:
            current, path = queue.popleft()
            current_hops = len(path) - 1
            if current_hops >= max_hops:
                continue

            for neighbor in sorted(graph.get(current, ())):
                if neighbor in visited:
                    continue

                next_path = path + [neighbor]
                visited.add(neighbor)
                nearby_path_cache[(start, neighbor)] = (len(next_path) - 1, next_path)
                queue.append((neighbor, next_path))

    return nearby_path_cache


def load_sumo_edge_graph(
    net_file: Path = DEFAULT_NET_FILE,
    vehicle_class: str = DEFAULT_VEHICLE_CLASS,
) -> SumoEdgeGraph:
    """
    Parse the SUMO net file and keep all required structures in memory.

    Returns:
        SumoEdgeGraph:
            - edges: normal edge metadata reachable by the requested vehicle class
            - graph: outgoing adjacency
            - reverse_graph: incoming adjacency
            - nearby_path_cache: precomputed 1-hop/2-hop local paths
    """
    valid_edges, raw_connections = parse_sumo_net(net_file, vehicle_class=vehicle_class)
    graph, reverse_graph = build_edge_graph(raw_connections, valid_edges)
    nearby_path_cache = build_nearby_path_cache(graph, max_hops=2)
    return SumoEdgeGraph(
        edges=valid_edges,
        graph=graph,
        reverse_graph=reverse_graph,
        nearby_path_cache=nearby_path_cache,
    )


__all__ = [
    "DEFAULT_NET_FILE",
    "DEFAULT_VEHICLE_CLASS",
    "EdgeInfo",
    "LaneInfo",
    "NearbyPathCache",
    "SumoEdgeGraph",
    "build_edge_graph",
    "build_nearby_path_cache",
    "load_sumo_edge_graph",
    "parse_sumo_net",
]
