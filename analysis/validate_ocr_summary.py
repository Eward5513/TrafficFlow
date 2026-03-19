import argparse
import json
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_TXT = BASE_DIR / "ocr_output" / "all_ocr_no_blank_lines.txt"
DEFAULT_GEOJSON = BASE_DIR.parent / "data" / "basemap.geojson"
DEFAULT_OUTPUT_TXT = BASE_DIR / "ocr_output" / "invalid_line_numbers.txt"
DEFAULT_DETAIL_OUTPUT_TXT = BASE_DIR / "ocr_output" / "invalid_line_details.txt"

TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}$")
INT_RE = re.compile(r"^-?\d+$")


def load_valid_edge_ids(geojson_path: Path) -> set[str]:
    """
    Load valid edge ids from the road network GeoJSON.
    Primary source is feature.properties.osm_id.
    """
    data = json.loads(geojson_path.read_text(encoding="utf-8"))
    features = data.get("features", [])

    edge_ids: set[str] = set()
    for feat in features:
        props = feat.get("properties", {}) if isinstance(feat, dict) else {}

        osm_id = props.get("osm_id")
        if osm_id is not None:
            edge_ids.add(str(osm_id))

        # Keep compatibility with possible alternate naming.
        alt_id = props.get("edge_id")
        if alt_id is not None:
            edge_ids.add(str(alt_id))

    return edge_ids


def is_vin_token(token: str) -> bool:
    """
    VIN is defined here as an integer token <= 100000.
    """
    if not INT_RE.fullmatch(token):
        return False
    value = int(token)
    return 0 <= value <= 100000


def is_line_valid(line: str, valid_edge_ids: set[str]) -> bool:
    """
    Valid formats:
      1) vin time edge_id time edge_id ...
      2) time edge_id time edge_id ...
    """
    return explain_invalid_reason(line, valid_edge_ids) is None


def validate_line(line: str, valid_edge_ids: set[str]) -> bool:
    """
    Backward-compatible alias.
    """
    return is_line_valid(line, valid_edge_ids)


def explain_invalid_reason(line: str, valid_edge_ids: set[str]) -> str | None:
    """
    Return None if line is valid; otherwise return a human-readable reason.
    """
    text = line.strip()
    if not text:
        return "empty line"

    tokens = text.split()
    if len(tokens) < 2:
        return f"not enough tokens: {len(tokens)}"

    start_idx = 1 if is_vin_token(tokens[0]) else 0
    seq = tokens[start_idx:]

    # Must be one or more (time, edge_id) pairs.
    if len(seq) < 2:
        return "missing time-edge pairs"
    if len(seq) % 2 != 0:
        return f"odd token count after vin handling: {len(seq)}"

    for i in range(0, len(seq), 2):
        ts = seq[i]
        edge_id = seq[i + 1]
        pair_idx = (i // 2) + 1

        if not TIME_RE.fullmatch(ts):
            return f"pair#{pair_idx} invalid timestamp: '{ts}'"
        if not INT_RE.fullmatch(edge_id):
            return f"pair#{pair_idx} edge id is not integer: '{edge_id}'"
        if edge_id not in valid_edge_ids:
            return f"pair#{pair_idx} edge id not in geojson: '{edge_id}'"

    return None


def find_invalid_lines(input_txt: Path, valid_edge_ids: set[str]) -> tuple[list[int], int]:
    invalid_lines: list[int] = []
    lines = input_txt.read_text(encoding="utf-8").splitlines()
    for line_no, line in enumerate(lines, start=1):
        if not is_line_valid(line, valid_edge_ids):
            invalid_lines.append(line_no)
    return invalid_lines, len(lines)


def find_invalid_line_details(input_txt: Path, valid_edge_ids: set[str]) -> tuple[list[dict[str, str]], int]:
    details: list[dict[str, str]] = []
    lines = input_txt.read_text(encoding="utf-8").splitlines()
    for line_no, line in enumerate(lines, start=1):
        reason = explain_invalid_reason(line, valid_edge_ids)
        if reason is None:
            continue
        details.append(
            {
                "line_no": str(line_no),
                "reason": reason,
                "line": line,
            }
        )
    return details, len(lines)


def find_adjacent_duplicate_lines(input_txt: Path) -> tuple[list[dict[str, str]], int]:
    duplicates: list[dict[str, str]] = []
    lines = input_txt.read_text(encoding="utf-8").splitlines()
    for idx in range(len(lines) - 1):
        current = lines[idx]
        nxt = lines[idx + 1]
        if current.strip() == nxt.strip():
            duplicates.append(
                {
                    "line_no": str(idx + 1),
                    "next_line_no": str(idx + 2),
                    "line_current": current,
                    "line_next": nxt,
                }
            )
    return duplicates, len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate OCR merged txt against road network edges.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_TXT, help="Path to merged OCR txt.")
    parser.add_argument("--geojson", type=Path, default=DEFAULT_GEOJSON, help="Path to basemap geojson.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_TXT, help="Output txt with invalid line numbers.")
    parser.add_argument(
        "--detail-output",
        type=Path,
        default=DEFAULT_DETAIL_OUTPUT_TXT,
        help="Output txt with full invalid line details.",
    )
    args = parser.parse_args()

    input_txt = args.input.resolve()
    geojson_path = args.geojson.resolve()
    output_txt = args.output.resolve()
    detail_output_txt = args.detail_output.resolve()

    if not input_txt.exists():
        raise FileNotFoundError(f"Input txt not found: {input_txt}")
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")

    valid_edge_ids = load_valid_edge_ids(geojson_path)
    if not valid_edge_ids:
        raise ValueError(f"No edge ids found in geojson: {geojson_path}")

    invalid_lines, total_lines = find_invalid_lines(input_txt, valid_edge_ids)
    invalid_details, _ = find_invalid_line_details(input_txt, valid_edge_ids)
    adjacent_duplicates, _ = find_adjacent_duplicate_lines(input_txt)

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("\n".join(str(n) for n in invalid_lines), encoding="utf-8")
    detail_output_txt.parent.mkdir(parents=True, exist_ok=True)
    invalid_detail_blocks = [
        "\n".join(
            [
                f"line {item['line_no']}",
                f"reason: {item['reason']}",
                f"raw   : {item['line']}",
            ]
        )
        for item in invalid_details
    ]
    adjacent_detail_blocks = [
        "\n".join(
            [
                f"adjacent duplicate: line {item['line_no']} == line {item['next_line_no']}",
                f"line {item['line_no']}: {item['line_current']}",
                f"line {item['next_line_no']}: {item['line_next']}",
            ]
        )
        for item in adjacent_duplicates
    ]
    detail_blocks = invalid_detail_blocks + adjacent_detail_blocks
    detail_output_txt.write_text(
        "\n\n".join(detail_blocks) if detail_blocks else "No invalid lines or adjacent duplicate lines.",
        encoding="utf-8",
    )

    print(f"Total lines: {total_lines}")
    print(f"Invalid lines: {len(invalid_lines)}")
    print(f"Adjacent duplicate pairs: {len(adjacent_duplicates)}")
    print(f"Output: {output_txt}")
    print(f"Detail Output: {detail_output_txt}")


if __name__ == "__main__":
    main()
