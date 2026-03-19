import argparse
from pathlib import Path

from validate_ocr_summary import INT_RE, TIME_RE, is_vin_token


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_TXT = BASE_DIR / "ocr_output" / "route_by_edge_draft4.txt"
DEFAULT_OUTPUT_TXT = BASE_DIR / "ocr_output" / "route_by_edge.txt"
DEFAULT_LOG_TXT = BASE_DIR / "ocr_output" / "route_by_edge_build_log.txt"


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
    parser = argparse.ArgumentParser(
        description="Build final route file from draft OCR output."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_TXT, help="Input draft txt path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_TXT, help="Output route txt path.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_TXT, help="Log txt path.")
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    log_path = args.log.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    groups = parse_file(input_path)
    output_lines, log_lines = build_routes(groups)

    unique_vins = len({g["vin"] for g in groups})
    merged_count = len(groups) - unique_vins

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines), encoding="utf-8")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(log_lines) if log_lines else "All trajectories processed successfully.",
        encoding="utf-8",
    )

    print(f"Total trajectory groups parsed : {len(groups)}")
    print(f"Unique VINs                    : {unique_vins}")
    print(f"Merged duplicate VIN groups    : {merged_count}")
    print(f"Output trajectories            : {len(output_lines)}")
    print(f"Skipped trajectories           : {unique_vins - len(output_lines)}")
    print(f"Output : {output_path}")
    print(f"Log    : {log_path}")


if __name__ == "__main__":
    main()
