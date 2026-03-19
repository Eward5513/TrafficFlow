import argparse
from pathlib import Path

from validate_ocr_summary import INT_RE, TIME_RE, is_line_valid, is_vin_token, load_valid_edge_ids


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_TXT = BASE_DIR / "ocr_output" / "all_ocr_no_blank_lines.txt"
DEFAULT_GEOJSON = BASE_DIR.parent / "data" / "basemap.geojson"
DEFAULT_OUTPUT_TXT = BASE_DIR / "ocr_output" / "all_ocr_no_blank_lines_refined.txt"
DEFAULT_LOG_TXT = BASE_DIR / "ocr_output" / "all_ocr_no_blank_lines_refine_log.txt"

def is_timestamp(token: str) -> bool:
    return bool(TIME_RE.fullmatch(token))


def is_numeric(token: str) -> bool:
    return bool(INT_RE.fullmatch(token))


def compact_tokens(line: str) -> str:
    return "".join(line.split())


def similarity_type_after_compact(a: str, b: str) -> str | None:
    """
    Return match type after removing spaces:
    - exact: exactly same
    - one-char-diff: same length, exactly one different char
    - one-char-extra: length differs by 1 and can match by one insertion/deletion
    """
    sa = compact_tokens(a)
    sb = compact_tokens(b)

    if sa == sb:
        return "exact"

    if len(sa) == len(sb):
        diff = sum(1 for x, y in zip(sa, sb) if x != y)
        return "one-char-diff" if diff == 1 else None

    if abs(len(sa) - len(sb)) != 1:
        return None

    short, long = (sa, sb) if len(sa) < len(sb) else (sb, sa)
    i = 0
    j = 0
    mismatch_used = False
    while i < len(short) and j < len(long):
        if short[i] == long[j]:
            i += 1
            j += 1
            continue
        if mismatch_used:
            return None
        mismatch_used = True
        j += 1

    return "one-char-extra"


def merge_split_edge_ids(
    line: str, line_no: int, valid_edge_ids: set[str], logs: list[str]
) -> tuple[str, int]:
    """
    Try fixing OCR split edge ids:
    edge_id might be split into two numeric tokens by an extra space.
    """
    tokens = line.split()
    if len(tokens) < 2:
        return line, 0

    start_idx = 1 if is_vin_token(tokens[0]) else 0
    edge_idx = start_idx + 1
    merge_count = 0

    while edge_idx < len(tokens):
        time_idx = edge_idx - 1
        if time_idx < 0 or not is_timestamp(tokens[time_idx]):
            edge_idx += 2
            continue

        edge_token = tokens[edge_idx]
        if edge_token in valid_edge_ids:
            edge_idx += 2
            continue

        if edge_idx + 1 >= len(tokens):
            edge_idx += 2
            continue

        next_token = tokens[edge_idx + 1]
        if not (is_numeric(edge_token) and is_numeric(next_token) and not is_timestamp(next_token)):
            edge_idx += 2
            continue

        combined = edge_token + next_token
        if combined not in valid_edge_ids:
            edge_idx += 2
            continue

        before = " ".join(tokens)
        tokens[edge_idx] = combined
        del tokens[edge_idx + 1]
        after = " ".join(tokens)

        merge_count += 1
        logs.append(f"[merge] line {line_no}")
        logs.append(f"before: {before}")
        logs.append(f"after : {after}")
        logs.append("")

        edge_idx += 2

    return " ".join(tokens), merge_count


def refine_lines(lines: list[str], valid_edge_ids: set[str]) -> tuple[list[str], int, int, int, list[str]]:
    """
    1) Remove full lines that start with '(' (ignoring leading spaces)
    2) If current line and next line are exactly the same, drop current line
    3) Try merging split edge-id tokens per line
    4) If current line is invalid and is same/near-same as next line after removing spaces, drop current line
    """
    filtered_lines: list[tuple[int, str]] = []
    deduped_lines: list[tuple[int, str]] = []
    merged_lines: list[tuple[int, str]] = []
    refined: list[str] = []
    total_merges = 0
    dropped_adjacent_exact_duplicates = 0
    dropped_invalid_duplicates = 0
    logs: list[str] = []

    for line_no, line in enumerate(lines, start=1):
        if line.lstrip().startswith("("):
            continue
        filtered_lines.append((line_no, line))

    i = 0
    while i < len(filtered_lines):
        line_no, line = filtered_lines[i]
        if i + 1 < len(filtered_lines):
            next_line_no, next_line = filtered_lines[i + 1]
            if line == next_line:
                dropped_adjacent_exact_duplicates += 1
                logs.append(f"[drop-adjacent-exact] line {line_no}")
                logs.append(
                    f"reason: current line exactly equals next line {next_line_no} after '(' lines removal"
                )
                logs.append(f"current line: {line}")
                logs.append(f"next line   : {next_line}")
                logs.append("action      : delete current line")
                logs.append("")
                i += 1
                continue

        deduped_lines.append((line_no, line))
        i += 1

    for line_no, line in deduped_lines:
        merged_line, merges = merge_split_edge_ids(line, line_no, valid_edge_ids, logs)
        total_merges += merges
        merged_lines.append((line_no, merged_line))

    i = 0
    while i < len(merged_lines):
        line_no, line = merged_lines[i]
        if i + 1 < len(merged_lines):
            next_line_no, next_line = merged_lines[i + 1]
            match_type = similarity_type_after_compact(line, next_line)
            if (not is_line_valid(line, valid_edge_ids)) and (match_type is not None):
                dropped_invalid_duplicates += 1
                logs.append(f"[drop-invalid-duplicate] line {line_no}")
                logs.append(
                    f"reason: current line invalid and matches next line {next_line_no} after compact "
                    f"(type={match_type})"
                )
                logs.append(f"current line: {line}")
                logs.append(f"next line   : {next_line}")
                logs.append("action      : delete current line")
                logs.append("")
                i += 1
                continue

        refined.append(line)
        i += 1

    return refined, total_merges, dropped_adjacent_exact_duplicates, dropped_invalid_duplicates, logs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refine merged OCR txt by deleting lines and fixing split edge ids."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_TXT, help="Input txt path.")
    parser.add_argument("--geojson", type=Path, default=DEFAULT_GEOJSON, help="Road network geojson path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_TXT, help="Output txt path.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_TXT, help="Merge log txt path.")
    args = parser.parse_args()

    input_path = args.input.resolve()
    geojson_path = args.geojson.resolve()
    output_path = args.output.resolve()
    log_path = args.log.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input txt not found: {input_path}")
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")

    original_lines = input_path.read_text(encoding="utf-8").splitlines()
    valid_edge_ids = load_valid_edge_ids(geojson_path)
    if not valid_edge_ids:
        raise ValueError(f"No edge ids found in geojson: {geojson_path}")

    refined_lines, total_merges, dropped_adjacent_exact_duplicates, dropped_invalid_duplicates, logs = refine_lines(
        original_lines, valid_edge_ids
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(refined_lines), encoding="utf-8")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if logs:
        log_path.write_text("\n".join(logs), encoding="utf-8")
    else:
        log_path.write_text("No refinement actions were applied.", encoding="utf-8")

    print(f"Input lines: {len(original_lines)}")
    print(f"Output lines: {len(refined_lines)}")
    print(f"Removed lines: {len(original_lines) - len(refined_lines)}")
    print(f"Dropped adjacent exact-duplicate lines: {dropped_adjacent_exact_duplicates}")
    print(f"Merged edge-id splits: {total_merges}")
    print(f"Dropped invalid-duplicate lines: {dropped_invalid_duplicates}")
    print(f"Output: {output_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
