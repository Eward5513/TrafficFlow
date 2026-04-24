from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

try:
    from .build_sumo_edge_graph import EdgeInfo, NearbyPathCache
except ImportError:
    from build_sumo_edge_graph import EdgeInfo, NearbyPathCache


PathCache = dict[tuple[str, str], tuple[int, list[str]] | None]
CandidateIndex = dict[str, list[str]]


@dataclass(slots=True)
class MatchResult:
    success: bool
    reason: str | None
    osm_edges: list[str] = field(default_factory=list)
    candidate_sets: list[list[str]] = field(default_factory=list)
    matched_sumo_edges: list[str] = field(default_factory=list)
    total_cost: int | None = None
    missing_osm_edge_id: str | None = None
    fuzzy: bool = False
    dropped_osm_edges: list[str] = field(default_factory=list)
    kept_osm_edges: list[str] = field(default_factory=list)
    chosen_candidates: list[str] = field(default_factory=list)


def normalize_osm_key_from_sumo_edge(edge_id: str) -> str:
    """
    Convert SUMO edge ids like `12345#1` or `-12345#1` to OSM base key `12345`.
    """
    text = edge_id[1:] if edge_id.startswith("-") else edge_id
    return text.split("#", 1)[0]


def candidate_matches_osm_edge(osm_edge_id: str, sumo_edge_id: str) -> bool:
    if sumo_edge_id == osm_edge_id:
        return True
    if sumo_edge_id.startswith(f"{osm_edge_id}#"):
        return True

    neg_osm_edge_id = f"-{osm_edge_id}"
    if sumo_edge_id == neg_osm_edge_id:
        return True
    if sumo_edge_id.startswith(f"{neg_osm_edge_id}#"):
        return True

    return False


def build_osm_candidate_index(valid_edges: dict[str, EdgeInfo]) -> CandidateIndex:
    """
    Build a reusable OSM id -> candidate SUMO edge list index.
    """
    index: CandidateIndex = {}
    for edge_id in valid_edges:
        osm_key = normalize_osm_key_from_sumo_edge(edge_id)
        index.setdefault(osm_key, []).append(edge_id)

    for osm_key in index:
        index[osm_key].sort()

    return index


def get_candidate_sumo_edges(
    osm_edge_id: str,
    valid_edges: dict[str, EdgeInfo],
    candidate_index: CandidateIndex | None = None,
) -> list[str]:
    """
    Map one OSM edge id to candidate SUMO edge ids using the required string rules.
    """
    if candidate_index is not None:
        return list(candidate_index.get(osm_edge_id, []))

    matched = [
        edge_id
        for edge_id in valid_edges
        if candidate_matches_osm_edge(osm_edge_id, edge_id)
    ]
    matched.sort()
    return matched


def build_candidate_sets(
    osm_edges: list[str],
    valid_edges: dict[str, EdgeInfo],
    candidate_index: CandidateIndex | None = None,
) -> list[list[str]]:
    return [
        get_candidate_sumo_edges(osm_edge_id, valid_edges, candidate_index)
        for osm_edge_id in osm_edges
    ]


def shortest_path_between_edges(
    graph: dict[str, set[str]],
    start: str,
    goal: str,
    nearby_path_cache: NearbyPathCache | None = None,
    path_cache: PathCache | None = None,
) -> tuple[int, list[str]] | None:
    """
    Find the shortest path between two SUMO edges.

    Search order:
        1. global path cache
        2. precomputed 1-hop/2-hop nearby cache
        3. fallback BFS

    Returns:
        (distance, path), where distance is the number of graph hops and path
        includes both start and goal. If unreachable, return None.
    """
    cache_key = (start, goal)
    if path_cache is not None and cache_key in path_cache:
        return path_cache[cache_key]

    if start == goal:
        result = (0, [start])
        if path_cache is not None:
            path_cache[cache_key] = result
        return result

    if nearby_path_cache is not None and cache_key in nearby_path_cache:
        result = nearby_path_cache[cache_key]
        if path_cache is not None:
            path_cache[cache_key] = result
        return result

    queue: deque[str] = deque([start])
    parents: dict[str, str | None] = {start: None}

    while queue:
        current = queue.popleft()
        for neighbor in sorted(graph.get(current, ())):
            if neighbor in parents:
                continue

            parents[neighbor] = current
            if neighbor == goal:
                path = [goal]
                node = goal
                while parents[node] is not None:
                    node = parents[node]
                    path.append(node)
                path.reverse()

                result = (len(path) - 1, path)
                if path_cache is not None:
                    path_cache[cache_key] = result
                return result

            queue.append(neighbor)

    if path_cache is not None:
        path_cache[cache_key] = None
    return None


def remove_simple_loops(
    sumo_edges: list[str],
) -> tuple[list[str], int]:
    """
    Collapse simple ``A B A`` loops in a matched SUMO edge sequence.

    A *simple loop* is a contiguous triplet ``(X, Y, X)`` where a single
    intermediate edge ``Y`` separates two occurrences of the same edge ``X``.
    Such triplets are collapsed by dropping ``Y`` (and the duplicate ``X``),
    so the sequence continues with the single ``X``. This removes obviously
    redundant "go out one step and come back" detours while keeping all
    other edges untouched.

    The routine uses a single left-to-right pass with a stack, which also
    correctly handles chained/overlapping loops such as ``A B A C A`` (all
    loops collapse to a single ``A``) without needing repeated passes.

    Returns:
        (cleaned_edges, loops_removed) where ``loops_removed`` is the number
        of simple ``A B A`` triplets that were collapsed.
    """
    cleaned: list[str] = []
    loops_removed = 0
    for edge in sumo_edges:
        if len(cleaned) >= 2 and cleaned[-2] == edge:
            cleaned.pop()
            loops_removed += 1
        else:
            cleaned.append(edge)
    return cleaned, loops_removed


def reconstruct_matched_path(
    end_candidate: str,
    parents_by_layer: list[dict[str, str | None]],
    segment_paths_by_layer: list[dict[str, list[str]]],
) -> tuple[list[str], list[str]]:
    """
    Reconstruct the full SUMO edge sequence from DP back-pointers.
    """
    chosen_candidates: list[str] = []
    current_candidate: str | None = end_candidate

    for layer_idx in range(len(parents_by_layer) - 1, -1, -1):
        if current_candidate is None:
            raise ValueError("Broken DP state while reconstructing matched path.")
        chosen_candidates.append(current_candidate)
        current_candidate = parents_by_layer[layer_idx][current_candidate]

    chosen_candidates.reverse()

    matched_sumo_edges: list[str] = []
    for layer_idx, candidate in enumerate(chosen_candidates):
        segment = segment_paths_by_layer[layer_idx][candidate]
        if not matched_sumo_edges:
            matched_sumo_edges.extend(segment)
        else:
            matched_sumo_edges.extend(segment[1:])

    return chosen_candidates, matched_sumo_edges


def match_trajectory_to_sumo(
    osm_edges: list[str],
    graph: dict[str, set[str]],
    valid_edges: dict[str, EdgeInfo],
    *,
    candidate_index: CandidateIndex | None = None,
    nearby_path_cache: NearbyPathCache | None = None,
    path_cache: PathCache | None = None,
) -> MatchResult:
    """
    Match one OSM edge sequence to the globally shortest SUMO edge sequence
    that hits each candidate set in order.
    """
    if not osm_edges:
        return MatchResult(
            success=False,
            reason="empty_osm_edges",
            osm_edges=list(osm_edges),
        )

    candidate_sets = build_candidate_sets(osm_edges, valid_edges, candidate_index)
    if any(not candidates for candidates in candidate_sets):
        missing_layer_idx = next(
            idx for idx, candidates in enumerate(candidate_sets) if not candidates
        )
        return MatchResult(
            success=False,
            reason="missing_candidate",
            osm_edges=list(osm_edges),
            candidate_sets=candidate_sets,
            missing_osm_edge_id=osm_edges[missing_layer_idx],
        )

    if len(candidate_sets) == 1:
        chosen_edge = candidate_sets[0][0]
        return MatchResult(
            success=True,
            reason=None,
            osm_edges=list(osm_edges),
            candidate_sets=candidate_sets,
            matched_sumo_edges=[chosen_edge],
            total_cost=1,
            chosen_candidates=[chosen_edge],
        )

    current_costs: dict[str, int] = {candidate: 1 for candidate in candidate_sets[0]}
    parents_by_layer: list[dict[str, str | None]] = [
        {candidate: None for candidate in candidate_sets[0]}
    ]
    segment_paths_by_layer: list[dict[str, list[str]]] = [
        {candidate: [candidate] for candidate in candidate_sets[0]}
    ]

    for current_candidates in candidate_sets[1:]:
        previous_candidates = sorted(current_costs)
        next_costs: dict[str, int] = {}
        next_parents: dict[str, str | None] = {}
        next_segment_paths: dict[str, list[str]] = {}

        for goal_candidate in current_candidates:
            best_total_cost: int | None = None
            best_parent: str | None = None
            best_path: list[str] | None = None

            for start_candidate in previous_candidates:
                shortest = shortest_path_between_edges(
                    graph,
                    start_candidate,
                    goal_candidate,
                    nearby_path_cache=nearby_path_cache,
                    path_cache=path_cache,
                )
                if shortest is None:
                    continue

                distance, path = shortest
                total_cost = current_costs[start_candidate] + distance
                if best_total_cost is None or total_cost < best_total_cost:
                    best_total_cost = total_cost
                    best_parent = start_candidate
                    best_path = path
                elif total_cost == best_total_cost and best_path is not None:
                    tie_key = (path, start_candidate)
                    best_tie_key = (best_path, best_parent or "")
                    if tie_key < best_tie_key:
                        best_parent = start_candidate
                        best_path = path

            if best_total_cost is None or best_parent is None or best_path is None:
                continue

            next_costs[goal_candidate] = best_total_cost
            next_parents[goal_candidate] = best_parent
            next_segment_paths[goal_candidate] = best_path

        if not next_costs:
            return MatchResult(
                success=False,
                reason="unreachable_between_candidate_sets",
                osm_edges=list(osm_edges),
                candidate_sets=candidate_sets,
            )

        current_costs = next_costs
        parents_by_layer.append(next_parents)
        segment_paths_by_layer.append(next_segment_paths)

    end_candidate = min(sorted(current_costs), key=lambda edge_id: current_costs[edge_id])
    chosen_candidates, matched_sumo_edges = reconstruct_matched_path(
        end_candidate=end_candidate,
        parents_by_layer=parents_by_layer,
        segment_paths_by_layer=segment_paths_by_layer,
    )

    return MatchResult(
        success=True,
        reason=None,
        osm_edges=list(osm_edges),
        candidate_sets=candidate_sets,
        matched_sumo_edges=matched_sumo_edges,
        total_cost=len(matched_sumo_edges),
        chosen_candidates=chosen_candidates,
    )


def fuzzy_match_trajectory_to_sumo(
    osm_edges: list[str],
    graph: dict[str, set[str]],
    valid_edges: dict[str, EdgeInfo],
    *,
    candidate_index: CandidateIndex | None = None,
    nearby_path_cache: NearbyPathCache | None = None,
    path_cache: PathCache | None = None,
) -> MatchResult:
    """
    Fuzzy-match an OSM edge sequence by dropping every OSM edge that has no
    corresponding SUMO candidate (typically because the underlying SUMO network
    was modified after the trajectory was recorded). The remaining OSM edges
    are then matched with the normal DP, which connects neighbouring candidates
    using the shortest SUMO path — effectively bridging the dropped segments.
    """
    if not osm_edges:
        return MatchResult(
            success=False,
            reason="empty_osm_edges",
            osm_edges=list(osm_edges),
            fuzzy=True,
        )

    candidate_sets = build_candidate_sets(osm_edges, valid_edges, candidate_index)
    kept_pairs: list[tuple[int, str]] = [
        (idx, osm_edges[idx])
        for idx, candidates in enumerate(candidate_sets)
        if candidates
    ]
    dropped_osm_edges: list[str] = [
        osm_edges[idx]
        for idx, candidates in enumerate(candidate_sets)
        if not candidates
    ]

    if not kept_pairs:
        return MatchResult(
            success=False,
            reason="fuzzy_no_surviving_candidates",
            osm_edges=list(osm_edges),
            candidate_sets=candidate_sets,
            fuzzy=True,
            dropped_osm_edges=dropped_osm_edges,
        )

    if not dropped_osm_edges:
        # Nothing to drop — fuzzy mode degenerates into a normal match.
        return match_trajectory_to_sumo(
            osm_edges,
            graph,
            valid_edges,
            candidate_index=candidate_index,
            nearby_path_cache=nearby_path_cache,
            path_cache=path_cache,
        )

    kept_osm_edges = [edge_id for _, edge_id in kept_pairs]
    inner_result = match_trajectory_to_sumo(
        kept_osm_edges,
        graph,
        valid_edges,
        candidate_index=candidate_index,
        nearby_path_cache=nearby_path_cache,
        path_cache=path_cache,
    )

    return MatchResult(
        success=inner_result.success,
        reason=inner_result.reason,
        osm_edges=list(osm_edges),
        candidate_sets=candidate_sets,
        matched_sumo_edges=inner_result.matched_sumo_edges,
        total_cost=inner_result.total_cost,
        missing_osm_edge_id=inner_result.missing_osm_edge_id,
        fuzzy=True,
        dropped_osm_edges=dropped_osm_edges,
        kept_osm_edges=kept_osm_edges,
        chosen_candidates=inner_result.chosen_candidates,
    )


__all__ = [
    "CandidateIndex",
    "MatchResult",
    "PathCache",
    "build_candidate_sets",
    "build_osm_candidate_index",
    "candidate_matches_osm_edge",
    "fuzzy_match_trajectory_to_sumo",
    "get_candidate_sumo_edges",
    "match_trajectory_to_sumo",
    "normalize_osm_key_from_sumo_edge",
    "reconstruct_matched_path",
    "remove_simple_loops",
    "shortest_path_between_edges",
]
