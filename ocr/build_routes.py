from pathlib import Path

from _paths import OCR_DATA
from validate_ocr_summary import INT_RE, TIME_RE, is_vin_token


OUT_DIR = OCR_DATA / "ocr_output_all"
LOG_DIR = OUT_DIR / "log"
INPUT_TXT = OUT_DIR / "all_ocr_no_blank_lines_refined.txt"
OUTPUT_TXT = OUT_DIR / "route_by_edge.txt"
OUTPUT_NO_MERGE_TXT = OUT_DIR / "route_by_edge_no_merge.txt"
OUTPUT_SPLIT_TXT = OUT_DIR / "route_by_edge_split.txt"
LOG_TXT = LOG_DIR / "route_by_edge_build_log.txt"

SPLIT_GAP_SECONDS = 3600


def ts_to_seconds(ts: str) -> int:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def parse_pairs_from_tokens(tokens: list[str]) -> list[tuple[str, str]]:
    """
    Extract (timestamp, edge_id) pairs from a flat token list.
    Silently skips malformed pairs.
    """
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(tokens) - 1:
        ts = tokens[i]
        edge_id = tokens[i + 1]
        if TIME_RE.fullmatch(ts) and INT_RE.fullmatch(edge_id):
            pairs.append((ts, edge_id))
            i += 2
        else:
            i += 1
    return pairs


def parse_file(input_txt: Path) -> list[dict]:
    """
    Parse draft file into trajectory groups.

    Each group:
        {
            "vin": str,
            "pairs": [(ts, edge_id), ...],
            "start_line": int,
            "end_line": int,
        }

    A new group begins whenever a vin token is found at the start of a line.
    Continuation lines (no vin) extend the current group's pairs.
    """
    lines = input_txt.read_text(encoding="utf-8").splitlines()
    groups: list[dict] = []
    current_group: dict | None = None

    for line_no, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            continue

        tokens = text.split()
        if not tokens:
            continue

        if is_vin_token(tokens[0]):
            if current_group is not None:
                current_group["end_line"] = line_no - 1
                groups.append(current_group)

            current_group = {
                "vin": tokens[0],
                "pairs": parse_pairs_from_tokens(tokens[1:]),
                "start_line": line_no,
                "end_line": line_no,
            }
        else:
            if current_group is None:
                continue
            current_group["pairs"].extend(parse_pairs_from_tokens(tokens))
            current_group["end_line"] = line_no

    if current_group is not None:
        groups.append(current_group)

    return groups


def check_time_order(pairs: list[tuple[str, str]]) -> tuple[bool, str]:
    """
    Return (True, "") if timestamps are non-decreasing, else (False, reason).
    """
    for i in range(len(pairs) - 1):
        t1 = ts_to_seconds(pairs[i][0])
        t2 = ts_to_seconds(pairs[i + 1][0])
        if t2 < t1:
            return (
                False,
                f"timestamp out of order at pair#{i + 1} "
                f"({pairs[i][0]} -> {pairs[i + 1][0]})",
            )
    return True, ""


def check_vin_order(groups: list[dict], log_lines: list[str]) -> None:
    """
    Log any groups where vin is not strictly greater than the previous vin.
    """
    prev_vin: int | None = None
    for group in groups:
        vin_int = int(group["vin"])
        if prev_vin is not None and vin_int < prev_vin:
            line_range = (
                f"lines {group['start_line']}-{group['end_line']}"
                if group["start_line"] != group["end_line"]
                else f"line {group['start_line']}"
            )
            log_lines.append(f"[vin-order-anomaly] vin={group['vin']} {line_range}")
            log_lines.append(
                f"reason: vin {group['vin']} is not greater than previous vin {prev_vin}"
            )
            log_lines.append("")
        prev_vin = vin_int


def merge_groups_by_vin(groups: list[dict]) -> list[dict]:
    """
    Merge all groups that share the same vin into one, preserving pair order.
    The merged group records all source line ranges.
    """
    merged: dict[str, dict] = {}
    for group in groups:
        vin = group["vin"]
        if vin not in merged:
            merged[vin] = {
                "vin": vin,
                "pairs": list(group["pairs"]),
                "line_ranges": [f"lines {group['start_line']}-{group['end_line']}"
                                if group["start_line"] != group["end_line"]
                                else f"line {group['start_line']}"],
            }
        else:
            merged[vin]["pairs"].extend(group["pairs"])
            merged[vin]["line_ranges"].append(
                f"lines {group['start_line']}-{group['end_line']}"
                if group["start_line"] != group["end_line"]
                else f"line {group['start_line']}"
            )

    return list(merged.values())


def build_routes_no_merge(groups: list[dict]) -> tuple[list[str], list[str]]:
    """
    Same as build_routes but skips merging: each group becomes its own trajectory.
    VIN order check is still performed. Log entries use a separate tag.
    """
    log_lines: list[str] = []
    check_vin_order(groups, log_lines)

    output_lines: list[str] = []
    for group in groups:
        vin = group["vin"]
        pairs = group["pairs"]
        line_range = (
            f"lines {group['start_line']}-{group['end_line']}"
            if group["start_line"] != group["end_line"]
            else f"line {group['start_line']}"
        )

        if not pairs:
            log_lines.append(f"[no-merge-skip] vin={vin} {line_range}")
            log_lines.append("reason: no valid pairs found")
            log_lines.append("")
            continue

        ok, reason = check_time_order(pairs)
        if not ok:
            log_lines.append(f"[no-merge-skip] vin={vin} {line_range}")
            log_lines.append(f"reason: {reason}")
            log_lines.append("")
            continue

        tokens = [vin]
        for ts, edge_id in pairs:
            tokens.append(ts)
            tokens.append(edge_id)
        output_lines.append(" ".join(tokens))

    return output_lines, log_lines


def split_pairs_by_gap(
    pairs: list[tuple[str, str]], gap_seconds: int
) -> list[list[tuple[str, str]]]:
    """
    Split a pair sequence into segments whenever the entry timestamps of two
    adjacent edges differ by more than gap_seconds.
    """
    segments: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = [pairs[0]]
    for prev, cur in zip(pairs, pairs[1:]):
        if ts_to_seconds(cur[0]) - ts_to_seconds(prev[0]) > gap_seconds:
            segments.append(current)
            current = []
        current.append(cur)
    segments.append(current)
    return segments


def build_routes_split(
    route_lines: list[str], gap_seconds: int = SPLIT_GAP_SECONDS
) -> tuple[list[str], list[str]]:
    """
    Take merged route output lines (one trajectory per vin line) and split each
    trajectory where adjacent edge entry timestamps differ by more than
    gap_seconds. Each resulting segment keeps the original vin.
    """
    log_lines: list[str] = []
    output_lines: list[str] = []

    for line in route_lines:
        tokens = line.split()
        vin = tokens[0]
        pairs = parse_pairs_from_tokens(tokens[1:])
        segments = split_pairs_by_gap(pairs, gap_seconds)

        if len(segments) > 1:
            boundaries = []
            idx = 0
            for seg in segments[:-1]:
                idx += len(seg)
                boundaries.append(f"{pairs[idx - 1][0]} -> {pairs[idx][0]}")
            log_lines.append(f"[split] vin={vin} into {len(segments)} segments")
            log_lines.append(f"gaps    : {', '.join(boundaries)}")
            log_lines.append("")

        for seg in segments:
            seg_tokens = [vin]
            for ts, edge_id in seg:
                seg_tokens.append(ts)
                seg_tokens.append(edge_id)
            output_lines.append(" ".join(seg_tokens))

    return output_lines, log_lines


def build_routes(groups: list[dict]) -> tuple[list[str], list[str]]:
    log_lines: list[str] = []

    # Check VIN ordering in original file before any merging.
    check_vin_order(groups, log_lines)

    # Merge groups that share the same VIN.
    merged = merge_groups_by_vin(groups)
    merge_count = len(groups) - len(merged)
    if merge_count > 0:
        for m in merged:
            if len(m["line_ranges"]) > 1:
                log_lines.append(f"[merged] vin={m['vin']}")
                log_lines.append(f"source  : {', '.join(m['line_ranges'])}")
                log_lines.append("")

    output_lines: list[str] = []
    for m in merged:
        vin = m["vin"]
        pairs = m["pairs"]
        line_ranges = ", ".join(m["line_ranges"])

        if not pairs:
            log_lines.append(f"[skip] vin={vin} ({line_ranges})")
            log_lines.append("reason: no valid pairs found")
            log_lines.append("")
            continue

        ok, reason = check_time_order(pairs)
        if not ok:
            log_lines.append(f"[skip] vin={vin} ({line_ranges})")
            log_lines.append(f"reason: {reason}")
            log_lines.append("")
            continue

        tokens = [vin]
        for ts, edge_id in pairs:
            tokens.append(ts)
            tokens.append(edge_id)
        output_lines.append(" ".join(tokens))

    return output_lines, log_lines


def main() -> None:
    input_path = INPUT_TXT.resolve()
    output_path = OUTPUT_TXT.resolve()
    output_no_merge_path = OUTPUT_NO_MERGE_TXT.resolve()
    output_split_path = OUTPUT_SPLIT_TXT.resolve()
    log_path = LOG_TXT.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    groups = parse_file(input_path)
    unique_vins = len({g["vin"] for g in groups})
    merged_count = len(groups) - unique_vins

    # Merged output
    output_lines, log_lines = build_routes(groups)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines), encoding="utf-8")

    # No-merge output
    no_merge_lines, no_merge_log_lines = build_routes_no_merge(groups)
    output_no_merge_path.parent.mkdir(parents=True, exist_ok=True)
    output_no_merge_path.write_text("\n".join(no_merge_lines), encoding="utf-8")

    # Split output (based on merged trajectories)
    split_lines, split_log_lines = build_routes_split(
        output_lines, SPLIT_GAP_SECONDS
    )
    output_split_path.parent.mkdir(parents=True, exist_ok=True)
    output_split_path.write_text("\n".join(split_lines), encoding="utf-8")

    # Combined log
    all_log: list[str] = []
    if log_lines:
        all_log += ["=== merged mode ===", ""] + log_lines
    if no_merge_log_lines:
        all_log += ["=== no-merge mode ===", ""] + no_merge_log_lines
    if split_log_lines:
        all_log += ["=== split mode ===", ""] + split_log_lines
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(all_log) if all_log else "All trajectories processed successfully.",
        encoding="utf-8",
    )

    print(f"Total trajectory groups parsed : {len(groups)}")
    print(f"Unique VINs                    : {unique_vins}")
    print(f"Merged duplicate VIN groups    : {merged_count}")
    print(f"--- merged output ---")
    print(f"Output trajectories            : {len(output_lines)}")
    print(f"Skipped trajectories           : {unique_vins - len(output_lines)}")
    print(f"Output                         : {output_path}")
    print(f"--- no-merge output ---")
    print(f"Output trajectories            : {len(no_merge_lines)}")
    print(f"Skipped trajectories           : {len(groups) - len(no_merge_lines)}")
    print(f"Output                         : {output_no_merge_path}")
    print(f"--- split output (from merged) ---")
    print(f"Output trajectories            : {len(split_lines)}")
    print(f"Split count                    : {len(split_lines) - len(output_lines)}")
    print(f"Output                         : {output_split_path}")
    print(f"Log    : {log_path}")


if __name__ == "__main__":
    main()
