from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

try:
    from .build_sumo_edge_graph import DEFAULT_NET_FILE, SumoEdgeGraph, load_sumo_edge_graph
    from .match_sumo_edges import (
        CandidateIndex,
        MatchResult,
        PathCache,
        build_osm_candidate_index,
        fuzzy_match_trajectory_to_sumo,
        match_trajectory_to_sumo,
        remove_simple_loops,
    )
except ImportError:
    from build_sumo_edge_graph import DEFAULT_NET_FILE, SumoEdgeGraph, load_sumo_edge_graph
    from match_sumo_edges import (
        CandidateIndex,
        MatchResult,
        PathCache,
        build_osm_candidate_index,
        fuzzy_match_trajectory_to_sumo,
        match_trajectory_to_sumo,
        remove_simple_loops,
    )


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TRAJ_FILE = BASE_DIR.parent / "ocr_output" / "route_by_edge_no_merge.txt"
DEFAULT_OUTPUT_FILE = BASE_DIR / "matched_routes.txt"
DEFAULT_FAILED_OUTPUT_FILE = BASE_DIR / "failed_routes.txt"
DEFAULT_FUZZY_LOG_FILE = BASE_DIR / "fuzzy_match_logs.txt"
TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")


@dataclass(slots=True)
class TrajectoryRecord:
    vin: str
    timestamps: list[str] = field(default_factory=list)
    osm_edges: list[str] = field(default_factory=list)
    line_no: int | None = None


@dataclass(slots=True)
class BadLineInfo:
    line_no: int
    reason: str
    raw_line: str


@dataclass(slots=True)
class TrajectoryLoadResult:
    records: list[TrajectoryRecord] = field(default_factory=list)
    bad_lines: list[BadLineInfo] = field(default_factory=list)
    total_lines: int = 0
    empty_lines: int = 0

    @property
    def successful_lines(self) -> int:
        return len(self.records)

    @property
    def skipped_lines(self) -> int:
        return len(self.bad_lines)


@dataclass(slots=True)
class TrajectoryStreamStats:
    total_lines: int = 0
    empty_lines: int = 0
    successful_lines: int = 0
    bad_lines: int = 0
    bad_line_examples: list[BadLineInfo] = field(default_factory=list)
    max_bad_line_examples: int = 5

    def add_bad_line(self, bad_line: BadLineInfo) -> None:
        self.bad_lines += 1
        if len(self.bad_line_examples) < self.max_bad_line_examples:
            self.bad_line_examples.append(bad_line)


@dataclass(slots=True)
class MatchingRunStats:
    matched_trajectories: int = 0
    fuzzy_matched_trajectories: int = 0
    failed_trajectories: int = 0
    loop_trajectories: int = 0
    total_loops_removed: int = 0
    failed_examples: list[str] = field(default_factory=list)
    loop_examples: list[str] = field(default_factory=list)
    max_failed_examples: int = 5
    max_loop_examples: int = 5

    def add_failed_example(self, text: str) -> None:
        if len(self.failed_examples) < self.max_failed_examples:
            self.failed_examples.append(text)

    def add_loop_example(self, text: str) -> None:
        if len(self.loop_examples) < self.max_loop_examples:
            self.loop_examples.append(text)


def is_valid_timestamp(token: str) -> bool:
    return bool(TIME_RE.fullmatch(token))


def build_bad_line_info(line_no: int, raw_line: str, reason: str | None) -> BadLineInfo:
    return BadLineInfo(
        line_no=line_no,
        reason=reason or "unknown parse error",
        raw_line=raw_line.rstrip("\n"),
    )


def parse_trajectory_line(
    line: str,
    line_no: int | None = None,
) -> tuple[TrajectoryRecord | None, str | None]:
    """
    Parse one trajectory line in the format:
        vin timestamp_1 osm_edge_1 timestamp_2 osm_edge_2 ...
    """
    text = line.strip()
    if not text:
        return None, "empty line"

    tokens = text.split()
    if len(tokens) < 3:
        return None, "expected at least 3 fields: vin timestamp edge"

    pair_tokens = tokens[1:]
    if len(pair_tokens) % 2 != 0:
        return None, "timestamp/edge fields must appear in pairs"

    timestamps: list[str] = []
    osm_edges: list[str] = []
    for index in range(0, len(pair_tokens), 2):
        timestamp = pair_tokens[index]
        osm_edge = pair_tokens[index + 1]
        pair_no = index // 2 + 1

        if not is_valid_timestamp(timestamp):
            return None, f"invalid timestamp at pair {pair_no}: {timestamp!r}"
        if not osm_edge:
            return None, f"empty OSM edge id at pair {pair_no}"

        timestamps.append(timestamp)
        osm_edges.append(osm_edge)

    return (
        TrajectoryRecord(
            vin=tokens[0],
            timestamps=timestamps,
            osm_edges=osm_edges,
            line_no=line_no,
        ),
        None,
    )


def iter_trajectory_file(
    txt_file: str | Path,
    *,
    warn: bool = True,
    stats: TrajectoryStreamStats | None = None,
) -> Iterator[TrajectoryRecord]:
    """
    Stream trajectories from file one by one.
    """
    traj_path = Path(txt_file)
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    try:
        with traj_path.open("r", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                if stats is not None:
                    stats.total_lines += 1

                if not raw_line.strip():
                    if stats is not None:
                        stats.empty_lines += 1
                    continue

                record, error = parse_trajectory_line(raw_line, line_no=line_no)
                if record is not None:
                    if stats is not None:
                        stats.successful_lines += 1
                    yield record
                    continue

                bad_line = build_bad_line_info(line_no, raw_line, error)
                if stats is not None:
                    stats.add_bad_line(bad_line)
                if warn:
                    print(
                        f"[warning] skip line {line_no}: {bad_line.reason}",
                        file=sys.stderr,
                    )
    except OSError as exc:
        raise OSError(f"Failed to read trajectory file: {traj_path}") from exc


def load_trajectory_file(
    txt_file: str | Path,
    *,
    warn: bool = True,
) -> TrajectoryLoadResult:
    """
    Load a trajectory txt file into memory.
    """
    result = TrajectoryLoadResult()
    stats = TrajectoryStreamStats(max_bad_line_examples=1_000_000)
    for record in iter_trajectory_file(txt_file, warn=warn, stats=stats):
        result.records.append(record)

    result.total_lines = stats.total_lines
    result.empty_lines = stats.empty_lines
    result.bad_lines = list(stats.bad_line_examples)
    return result


def match_trajectory_record(
    record: TrajectoryRecord,
    net: SumoEdgeGraph,
    *,
    candidate_index: CandidateIndex | None = None,
    path_cache: PathCache | None = None,
    enable_fuzzy_fallback: bool = True,
) -> MatchResult:
    """
    Match one parsed trajectory record against the in-memory SUMO edge graph.

    When the strict match fails because some OSM edges have no SUMO candidate
    (typically due to SUMO network modifications) and ``enable_fuzzy_fallback``
    is True, retry with a fuzzy match that drops the missing OSM edges and
    bridges the remaining segments through shortest SUMO paths.
    """
    result = match_trajectory_to_sumo(
        record.osm_edges,
        net.graph,
        net.edges,
        candidate_index=candidate_index,
        nearby_path_cache=net.nearby_path_cache,
        path_cache=path_cache,
    )

    if (
        not result.success
        and result.reason == "missing_candidate"
        and enable_fuzzy_fallback
    ):
        fuzzy_result = fuzzy_match_trajectory_to_sumo(
            record.osm_edges,
            net.graph,
            net.edges,
            candidate_index=candidate_index,
            nearby_path_cache=net.nearby_path_cache,
            path_cache=path_cache,
        )
        if fuzzy_result.success:
            return fuzzy_result
        # Prefer the original strict failure reason so downstream reporting
        # still points at the first unmatchable OSM edge.
        fuzzy_result.reason = fuzzy_result.reason or result.reason
        fuzzy_result.missing_osm_edge_id = (
            fuzzy_result.missing_osm_edge_id or result.missing_osm_edge_id
        )
        return fuzzy_result

    return result


def format_matched_output(vin: str, matched_sumo_edges: list[str]) -> str:
    return " ".join([vin, *matched_sumo_edges])


def format_failed_output(record: TrajectoryRecord, result: MatchResult) -> str:
    line_no = "" if record.line_no is None else str(record.line_no)
    return "\t".join(
        [
            record.vin,
            result.reason or "unknown_match_failure",
            line_no,
            result.missing_osm_edge_id or "",
            " ".join(record.osm_edges),
        ]
    )


def find_chosen_candidate_positions(
    matched_sumo_edges: list[str],
    chosen_candidates: list[str],
) -> list[int] | None:
    """
    Locate each chosen candidate in the matched edge sequence with an
    order-preserving left-to-right scan.
    """
    positions: list[int] = []
    cursor = 0
    for candidate in chosen_candidates:
        while cursor < len(matched_sumo_edges) and matched_sumo_edges[cursor] != candidate:
            cursor += 1
        if cursor >= len(matched_sumo_edges):
            return None
        positions.append(cursor)
        cursor += 1
    return positions


def build_fuzzy_quoted_edge_indexes(
    result: MatchResult,
    matched_sumo_edges: list[str],
) -> set[int]:
    """
    Return indexes of SUMO edges that were inserted to bridge dropped OSM
    segments during fuzzy matching. These indexes should be quoted in logs.
    """
    if not result.fuzzy or not result.candidate_sets or len(result.kept_osm_edges) < 2:
        return set()

    kept_indices = [
        idx
        for idx, candidates in enumerate(result.candidate_sets)
        if candidates
    ]
    if len(kept_indices) < 2:
        return set()

    positions = find_chosen_candidate_positions(matched_sumo_edges, result.chosen_candidates)
    if positions is None or len(positions) != len(kept_indices):
        return set()

    quoted_indexes: set[int] = set()
    for pair_idx in range(len(kept_indices) - 1):
        left_idx = kept_indices[pair_idx]
        right_idx = kept_indices[pair_idx + 1]
        has_missing_between = any(
            not result.candidate_sets[inner_idx]
            for inner_idx in range(left_idx + 1, right_idx)
        )
        if not has_missing_between:
            continue

        bridge_start = positions[pair_idx]
        bridge_end = positions[pair_idx + 1]
        for edge_pos in range(bridge_start + 1, bridge_end):
            quoted_indexes.add(edge_pos)

    return quoted_indexes


def format_fuzzy_log_output(record: TrajectoryRecord, result: MatchResult) -> str:
    """
    Format one fuzzy-match log line:
        vin edge_a "inserted_1" "inserted_2" edge_b ...
    """
    quoted_indexes = build_fuzzy_quoted_edge_indexes(result, result.matched_sumo_edges)
    rendered_edges: list[str] = []
    for idx, edge_id in enumerate(result.matched_sumo_edges):
        if idx in quoted_indexes:
            rendered_edges.append(f"\"{edge_id}\"")
        else:
            rendered_edges.append(edge_id)
    return " ".join([record.vin, *rendered_edges])


def preview_edges(edges: list[str], limit: int = 10) -> str:
    if len(edges) <= limit:
        return " ".join(edges)
    return " ".join(edges[:limit]) + " ..."


def print_run_summary(
    parse_stats: TrajectoryStreamStats,
    match_stats: MatchingRunStats,
    success_examples: list[str],
    output_file: Path,
    failed_output_file: Path,
    fuzzy_log_output_file: Path,
) -> None:
    print(f"Total lines           : {parse_stats.total_lines}")
    print(f"Empty lines skipped   : {parse_stats.empty_lines}")
    print(f"Parsed trajectories   : {parse_stats.successful_lines}")
    print(f"Bad lines skipped     : {parse_stats.bad_lines}")
    print(f"Matched trajectories  : {match_stats.matched_trajectories}")
    print(f"  of which fuzzy match: {match_stats.fuzzy_matched_trajectories}")
    print(f"  with simple loops   : {match_stats.loop_trajectories}")
    print(f"Simple loops removed  : {match_stats.total_loops_removed}")
    print(f"Failed trajectories   : {match_stats.failed_trajectories}")
    print(f"Matched output        : {output_file}")
    print(f"Failed output         : {failed_output_file}")
    print(f"Fuzzy match logs      : {fuzzy_log_output_file}")

    if parse_stats.bad_line_examples:
        print("")
        print("Bad line examples:")
        for bad_line in parse_stats.bad_line_examples:
            print(f"- line {bad_line.line_no}: {bad_line.reason}")

    if match_stats.failed_examples:
        print("")
        print("Match failure examples:")
        for text in match_stats.failed_examples:
            print(f"- {text}")

    if match_stats.loop_examples:
        print("")
        print("Simple loop examples:")
        for text in match_stats.loop_examples:
            print(f"- {text}")

    if success_examples:
        print("")
        print(f"First {len(success_examples)} matched trajectories:")
        for text in success_examples:
            print(f"- {text}")


def process_trajectory_file(
    net: SumoEdgeGraph,
    traj_file: str | Path,
    output_file: str | Path,
    failed_output_file: str | Path,
    fuzzy_log_output_file: str | Path,
    *,
    preview_count: int = 5,
    warn: bool = True,
    candidate_index: CandidateIndex | None = None,
    path_cache: PathCache | None = None,
) -> tuple[TrajectoryStreamStats, MatchingRunStats, list[str]]:
    """
    Stream the trajectory file, match each trajectory, and write success/failure
    results to separate files.
    """
    if preview_count < 0:
        raise ValueError("preview_count must be non-negative.")

    if candidate_index is None:
        candidate_index = build_osm_candidate_index(net.edges)
    if path_cache is None:
        path_cache = {}

    output_path = Path(output_file).expanduser().resolve()
    failed_output_path = Path(failed_output_file).expanduser().resolve()
    fuzzy_log_output_path = Path(fuzzy_log_output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_output_path.parent.mkdir(parents=True, exist_ok=True)
    fuzzy_log_output_path.parent.mkdir(parents=True, exist_ok=True)

    parse_stats = TrajectoryStreamStats()
    match_stats = MatchingRunStats()
    success_examples: list[str] = []

    with (
        output_path.open("w", encoding="utf-8") as matched_fh,
        failed_output_path.open("w", encoding="utf-8") as failed_fh,
        fuzzy_log_output_path.open("w", encoding="utf-8") as fuzzy_log_fh,
    ):
        for record in iter_trajectory_file(traj_file, warn=warn, stats=parse_stats):
            result = match_trajectory_record(
                record,
                net,
                candidate_index=candidate_index,
                path_cache=path_cache,
            )

            if result.success:
                match_stats.matched_trajectories += 1
                if result.fuzzy:
                    match_stats.fuzzy_matched_trajectories += 1
                    fuzzy_log_fh.write(format_fuzzy_log_output(record, result) + "\n")
                cleaned_edges, loops_removed = remove_simple_loops(
                    result.matched_sumo_edges
                )
                if loops_removed > 0:
                    match_stats.loop_trajectories += 1
                    match_stats.total_loops_removed += loops_removed
                    match_stats.add_loop_example(
                        f"vin={record.vin} loops_removed={loops_removed} "
                        f"before={preview_edges(result.matched_sumo_edges, limit=6)} "
                        f"after={preview_edges(cleaned_edges, limit=6)}"
                    )
                matched_fh.write(
                    format_matched_output(record.vin, cleaned_edges) + "\n"
                )
                if len(success_examples) < preview_count:
                    fuzzy_tag = " [fuzzy]" if result.fuzzy else ""
                    success_examples.append(
                        f"vin={record.vin}{fuzzy_tag} "
                        f"sumo_edges={preview_edges(cleaned_edges)}"
                    )
                continue

            match_stats.failed_trajectories += 1
            failed_fh.write(format_failed_output(record, result) + "\n")
            match_stats.add_failed_example(f"vin={record.vin} reason={result.reason}")

    return parse_stats, match_stats, success_examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match OSM edge trajectories to the SUMO edge graph."
    )
    parser.add_argument(
        "--net",
        type=Path,
        default=DEFAULT_NET_FILE,
        help="Path to SUMO net.net.xml file.",
    )
    parser.add_argument(
        "--traj",
        type=Path,
        default=DEFAULT_TRAJ_FILE,
        help="Path to trajectory txt file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Path to matched trajectory output file.",
    )
    parser.add_argument(
        "--failed-output",
        type=Path,
        default=DEFAULT_FAILED_OUTPUT_FILE,
        help="Path to failed trajectory output file.",
    )
    parser.add_argument(
        "--fuzzy-log-output",
        type=Path,
        default=DEFAULT_FUZZY_LOG_FILE,
        help=(
            "Path to fuzzy-match detail logs. Inserted bridge edges caused by "
            "missing candidates are wrapped in double quotes."
        ),
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=5,
        help="How many successful matched trajectories to preview.",
    )
    args = parser.parse_args()

    if args.preview < 0:
        raise ValueError("--preview must be non-negative.")

    net = load_sumo_edge_graph(args.net.expanduser().resolve())
    output_path = args.output.expanduser().resolve()
    failed_output_path = args.failed_output.expanduser().resolve()
    fuzzy_log_output_path = args.fuzzy_log_output.expanduser().resolve()
    parse_stats, match_stats, success_examples = process_trajectory_file(
        net,
        args.traj,
        output_path,
        failed_output_path,
        fuzzy_log_output_path,
        preview_count=args.preview,
        warn=True,
    )

    print_run_summary(
        parse_stats=parse_stats,
        match_stats=match_stats,
        success_examples=success_examples,
        output_file=output_path,
        failed_output_file=failed_output_path,
        fuzzy_log_output_file=fuzzy_log_output_path,
    )


__all__ = [
    "BadLineInfo",
    "CandidateIndex",
    "DEFAULT_FAILED_OUTPUT_FILE",
    "DEFAULT_FUZZY_LOG_FILE",
    "DEFAULT_OUTPUT_FILE",
    "DEFAULT_TRAJ_FILE",
    "MatchResult",
    "MatchingRunStats",
    "PathCache",
    "TrajectoryLoadResult",
    "TrajectoryRecord",
    "TrajectoryStreamStats",
    "format_failed_output",
    "format_fuzzy_log_output",
    "format_matched_output",
    "fuzzy_match_trajectory_to_sumo",
    "iter_trajectory_file",
    "load_trajectory_file",
    "match_trajectory_record",
    "match_trajectory_to_sumo",
    "parse_trajectory_line",
    "process_trajectory_file",
]


if __name__ == "__main__":
    main()
