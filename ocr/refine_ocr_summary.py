from pathlib import Path

from _paths import GEOJSON, OCR_DATA
from validate_ocr_summary import INT_RE, TIME_RE, is_line_valid, is_vin_token, load_valid_edge_ids


OUT_DIR = OCR_DATA / "ocr_output_all"
LOG_DIR = OUT_DIR / "log"
INPUT_TXT = OUT_DIR / "all_ocr_no_blank_lines.txt"
OUTPUT_TXT = OUT_DIR / "all_ocr_no_blank_lines_refined.txt"
LOG_TXT = LOG_DIR / "all_ocr_no_blank_lines_refine_log.txt"

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


def drop_vin_order_duplicates(
    numbered_lines: list[tuple[int, str]], logs: list[str]
) -> tuple[list[tuple[int, str]], int]:
    """
    When a vin line breaks the non-decreasing vin order, check both endpoints of
    the violation (previous vin line and current vin line) against their adjacent
    lines. If one of them is same/near-same as a neighbor after removing spaces,
    it is an OCR-duplicated line with a misread vin: drop it.
    """
    result = list(numbered_lines)
    dropped = 0

    def leading_vin(line: str) -> int | None:
        tokens = line.split()
        if tokens and is_vin_token(tokens[0]):
            return int(tokens[0])
        return None

    def find_duplicate_neighbor(idx: int) -> tuple[str, int, str] | None:
        for direction, n_idx in (("previous", idx - 1), ("next", idx + 1)):
            if 0 <= n_idx < len(result):
                match_type = similarity_type_after_compact(result[idx][1], result[n_idx][1])
                if match_type is not None:
                    return direction, n_idx, match_type
        return None

    def vin_before(idx: int) -> int | None:
        for i in range(idx - 1, -1, -1):
            vin = leading_vin(result[i][1])
            if vin is not None:
                return vin
        return None

    def vin_after(idx: int) -> int | None:
        for i in range(idx + 1, len(result)):
            vin = leading_vin(result[i][1])
            if vin is not None:
                return vin
        return None

    # 每删一行后重新扫描，避免索引和 vin 顺序状态失效
    restart = True
    while restart:
        restart = False
        prev_idx: int | None = None
        prev_vin: int | None = None
        for idx in range(len(result)):
            vin = leading_vin(result[idx][1])
            if vin is None:
                continue
            if prev_vin is not None and vin < prev_vin:
                for cand_idx in (prev_idx, idx):
                    found = find_duplicate_neighbor(cand_idx)
                    if found is None:
                        continue
                    # 只删「删掉后局部顺序能恢复」的那一端：
                    # 删 prev 行后，要求其前一个 vin <= 当前 vin；
                    # 删当前行后，要求 prev vin <= 其后一个 vin。
                    if cand_idx == prev_idx:
                        before = vin_before(prev_idx)
                        if before is not None and before > vin:
                            continue
                    else:
                        after = vin_after(idx)
                        if after is not None and after < prev_vin:
                            continue
                    direction, n_idx, match_type = found
                    line_no, line = result[cand_idx]
                    other_line_no, other_line = result[n_idx]
                    dropped += 1
                    logs.append(f"[drop-vin-order-duplicate] line {line_no}")
                    logs.append(
                        f"reason: vin order broken between line {result[prev_idx][0]} "
                        f"and line {result[idx][0]}; this line matches {direction} "
                        f"line {other_line_no} after compact (type={match_type})"
                    )
                    logs.append(f"current line : {line}")
                    logs.append(f"{direction} line".ljust(13) + f": {other_line}")
                    logs.append("action       : delete current line")
                    logs.append("")
                    del result[cand_idx]
                    restart = True
                    break
                if restart:
                    break
            prev_idx = idx
            prev_vin = vin

    return result, dropped


def refine_lines(
    lines: list[str], valid_edge_ids: set[str]
) -> tuple[list[str], int, int, int, int, list[str]]:
    """
    1) Remove full lines that start with '(' (ignoring leading spaces)
    2) If current line and next line are exactly the same, drop current line
    3) Try merging split edge-id tokens per line
    4) If current line is invalid and is same/near-same (after removing spaces) as the
       next line or the previous kept line, drop current line
    5) If a vin line breaks non-decreasing vin order and one endpoint of the violation
       is same/near-same as its neighbor, drop that endpoint line
    """
    filtered_lines: list[tuple[int, str]] = []
    deduped_lines: list[tuple[int, str]] = []
    merged_lines: list[tuple[int, str]] = []
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

    refined_with_no: list[tuple[int, str]] = []
    for i, (line_no, line) in enumerate(merged_lines):
        if not is_line_valid(line, valid_edge_ids):
            # 与下一行比较（坏行在前、好行在后）
            neighbors: list[tuple[str, int, str]] = []
            if i + 1 < len(merged_lines):
                next_line_no, next_line = merged_lines[i + 1]
                neighbors.append(("next", next_line_no, next_line))
            # 与上一条已保留的行比较（好行在前、坏行在后），
            # 用已保留行而不是原相邻行，确保输出里仍留有近似副本
            if refined_with_no:
                prev_line_no, prev_line = refined_with_no[-1]
                neighbors.append(("previous", prev_line_no, prev_line))

            dropped = False
            for direction, other_line_no, other_line in neighbors:
                match_type = similarity_type_after_compact(line, other_line)
                if match_type is None:
                    continue
                dropped_invalid_duplicates += 1
                logs.append(f"[drop-invalid-duplicate] line {line_no}")
                logs.append(
                    f"reason: current line invalid and matches {direction} line {other_line_no} "
                    f"after compact (type={match_type})"
                )
                logs.append(f"current line : {line}")
                logs.append(f"{direction} line".ljust(13) + f": {other_line}")
                logs.append("action       : delete current line")
                logs.append("")
                dropped = True
                break

            if dropped:
                continue

        refined_with_no.append((line_no, line))

    refined_with_no, dropped_vin_order_duplicates = drop_vin_order_duplicates(
        refined_with_no, logs
    )

    refined = [line for _, line in refined_with_no]

    return (
        refined,
        total_merges,
        dropped_adjacent_exact_duplicates,
        dropped_invalid_duplicates,
        dropped_vin_order_duplicates,
        logs,
    )


def main() -> None:
    input_path = INPUT_TXT.resolve()
    geojson_path = GEOJSON.resolve()
    output_path = OUTPUT_TXT.resolve()
    log_path = LOG_TXT.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input txt not found: {input_path}")
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")

    original_lines = input_path.read_text(encoding="utf-8").splitlines()
    valid_edge_ids = load_valid_edge_ids(geojson_path)
    if not valid_edge_ids:
        raise ValueError(f"No edge ids found in geojson: {geojson_path}")

    (
        refined_lines,
        total_merges,
        dropped_adjacent_exact_duplicates,
        dropped_invalid_duplicates,
        dropped_vin_order_duplicates,
        logs,
    ) = refine_lines(original_lines, valid_edge_ids)

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
    print(f"Dropped vin-order-duplicate lines: {dropped_vin_order_duplicates}")
    print(f"Output: {output_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
