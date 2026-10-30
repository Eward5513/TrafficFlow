#!/usr/bin/env python3
"""Write a 3-second-grid copy of the Xuancheng26 TLS programs.

The source is the network that completed the 29-day FCD batch:
``xuancheng_tls_rebuilt.net.xml``. Only ``<phase duration>`` values inside
``<tlLogic>`` change. Lanes, connections, link indexes, phase states, program
ids, and offsets stay byte-for-byte. ``night_peak.add.xml`` is copied to
``night_peak_3s.add.xml`` with one switch change: program 1 starts at 54900 s
instead of 55000 s. The return to program 0 stays at 75000 s.

Rules:
- a phase whose state contains ``y`` or ``Y`` gets duration 3;
- every other phase gets ``ceil(duration / 3) * 3``;
- a mixed yellow phase, a non-static program, or an extra phase attribute
  aborts the write.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

CODE_ROOT = Path("/home/tzhang174/TrafficFlow/dataset/xuancheng26")
SOURCE_NET = CODE_ROOT / "xuancheng_tls_rebuilt.net.xml"
SOURCE_ADDITIONAL = CODE_ROOT / "night_peak.add.xml"
OUTPUT_NET = CODE_ROOT / "xuancheng_tls_3s.net.xml"
OUTPUT_ADDITIONAL = CODE_ROOT / "night_peak_3s.add.xml"
AUDIT_JSON = CODE_ROOT / "tls_3s_audit.json"
AUDIT_MD = CODE_ROOT / "tls_3s_audit.md"
AUDIT_CSV = CODE_ROOT / "tls_3s_phase_durations.csv"
# 54900 = 3 * 18300. The source file keeps 55000, which is 1 second off that grid.
PEAK_SWITCH_FROM = "55000"
PEAK_SWITCH_TO = "54900"

PHASE_RE = re.compile(
    r'^(?P<indent>\s*)<phase duration="(?P<duration>[^"]+)"'
    r'(?P<gap>\s+)state="(?P<state>[^"]+)"(?P<tail>\s*/>)\s*$'
)
TLOGIC_OPEN_RE = re.compile(r"^(\s*)<tlLogic\b([^>]*)>\s*$")
TLOGIC_CLOSE_RE = re.compile(r"^\s*</tlLogic>\s*$")
ATTR_RE = re.compile(r'(\w+)="([^"]*)"')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def new_duration(state: str, old: int) -> tuple[int, str]:
    if "y" in state or "Y" in state:
        return 3, "yellow_exact_3"
    return int(math.ceil(old / 3.0) * 3), "ceil_multiple_of_3"


def parse_duration(raw: str) -> int:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"duration must be positive, got {raw!r}")
    if abs(value - round(value)) > 1e-9:
        raise ValueError(f"non-integer duration is not in the source set: {raw!r}")
    return int(round(value))


def phase_flags(state: str) -> dict:
    letters = set(state)
    has_yellow = ("y" in letters) or ("Y" in letters)
    other = letters - set("yYrR")
    return {
        "has_yellow": has_yellow,
        "mixed_yellow": has_yellow and bool(other),
        "letters": "".join(sorted(letters)),
    }


def transform(text: str) -> tuple[str, dict]:
    programs: list[dict] = []
    current: dict | None = None
    out_lines: list[str] = []
    blocked: list[dict] = []
    phase_rows: list[dict] = []

    for line_no, line in enumerate(text.splitlines(keepends=True), start=1):
        body = line[:-1] if line.endswith("\n") else line
        if body.endswith("\r"):
            body = body[:-1]
        opened = TLOGIC_OPEN_RE.match(body)
        if opened:
            if current is not None:
                raise SystemExit(f"nested tlLogic at line {line_no}")
            attrs = dict(ATTR_RE.findall(opened.group(2)))
            expected = {"id", "type", "programID", "offset"}
            if set(attrs) != expected:
                blocked.append(
                    {
                        "line": line_no,
                        "reason": "unexpected tlLogic attributes",
                        "attrs": attrs,
                    }
                )
            current = {
                "id": attrs.get("id", ""),
                "type": attrs.get("type", ""),
                "program_id": attrs.get("programID", ""),
                "offset": attrs.get("offset", ""),
                "line": line_no,
                "phases": [],
                "old_cycle": 0,
                "new_cycle": 0,
            }
            out_lines.append(line)
            continue
        if current is not None and TLOGIC_CLOSE_RE.match(body):
            current["changed"] = any(phase["changed"] for phase in current["phases"])
            programs.append(current)
            current = None
            out_lines.append(line)
            continue
        if current is not None:
            matched = PHASE_RE.match(body)
            if matched is None:
                blocked.append(
                    {
                        "line": line_no,
                        "tls_id": current["id"],
                        "program_id": current["program_id"],
                        "reason": "tlLogic child is not a simple phase",
                        "text": body.strip()[:240],
                    }
                )
                out_lines.append(line)
                continue
            state = matched.group("state")
            flags = phase_flags(state)
            try:
                old = parse_duration(matched.group("duration"))
            except ValueError as exc:
                blocked.append(
                    {
                        "line": line_no,
                        "tls_id": current["id"],
                        "program_id": current["program_id"],
                        "reason": str(exc),
                        "state": state,
                    }
                )
                out_lines.append(line)
                continue
            if current["type"] != "static":
                blocked.append(
                    {
                        "line": line_no,
                        "tls_id": current["id"],
                        "program_id": current["program_id"],
                        "reason": f"non-static controller type={current['type']!r}",
                        "state": state,
                    }
                )
                out_lines.append(line)
                continue
            if flags["mixed_yellow"]:
                blocked.append(
                    {
                        "line": line_no,
                        "tls_id": current["id"],
                        "program_id": current["program_id"],
                        "reason": "mixed yellow phase; duration left unchanged",
                        "state": state,
                        "old_duration": old,
                    }
                )
                out_lines.append(line)
                continue
            updated, rule = new_duration(state, old)
            if updated <= 0 or updated % 3 != 0:
                raise SystemExit(f"rule produced non-positive non-multiple {updated} at line {line_no}")
            changed = updated != old
            phase = {
                "index": len(current["phases"]),
                "line": line_no,
                "state": state,
                "old_duration": old,
                "new_duration": updated,
                "changed": changed,
                "rule": rule,
                "has_yellow": flags["has_yellow"],
            }
            current["phases"].append(phase)
            current["old_cycle"] += old
            current["new_cycle"] += updated
            phase_rows.append(
                {
                    "tls_id": current["id"],
                    "program_id": current["program_id"],
                    "offset": current["offset"],
                    **phase,
                }
            )
            if changed:
                newline = "\n" if line.endswith("\n") else ""
                line = (
                    f'{matched.group("indent")}<phase duration="{updated}"'
                    f'{matched.group("gap")}state="{state}"{matched.group("tail")}{newline}'
                )
            out_lines.append(line)
            continue
        if "<phase " in body or body.strip().startswith("<tlLogic"):
            blocked.append(
                {
                    "line": line_no,
                    "reason": "phase or tlLogic outside the expected one-line form",
                    "text": body.strip()[:240],
                }
            )
        out_lines.append(line)

    if current is not None:
        raise SystemExit("unclosed tlLogic at end of file")

    tls_ids = {item["id"] for item in programs}
    modified_programs = [item for item in programs if item["changed"]]
    modified_tls = {item["id"] for item in modified_programs}
    yellow = [row for row in phase_rows if row["has_yellow"]]
    changed_rows = [row for row in phase_rows if row["changed"]]
    bad_final = [
        row for row in phase_rows if row["new_duration"] <= 0 or row["new_duration"] % 3 != 0
    ]
    audit = {
        "source_net": str(SOURCE_NET),
        "output_net": str(OUTPUT_NET),
        "source_additional": str(SOURCE_ADDITIONAL),
        "output_additional": str(OUTPUT_ADDITIONAL),
        "tls_count": len(tls_ids),
        "program_count": len(programs),
        "phase_count": len(phase_rows),
        "modified_tls_count": len(modified_tls),
        "modified_program_count": len(modified_programs),
        "modified_phase_count": len(changed_rows),
        "yellow_phase_count": len(yellow),
        "yellow_modified_count": sum(1 for row in yellow if row["changed"]),
        "unchanged_phase_count": len(phase_rows) - len(changed_rows),
        "final_phases_not_multiple_of_3": bad_final,
        "blocked": blocked,
        "programs": [
            {
                "id": item["id"],
                "program_id": item["program_id"],
                "type": item["type"],
                "offset": item["offset"],
                "phase_count": len(item["phases"]),
                "old_cycle": item["old_cycle"],
                "new_cycle": item["new_cycle"],
                "cycle_delta": item["new_cycle"] - item["old_cycle"],
                "changed": item["changed"],
                "phases": item["phases"],
            }
            for item in programs
        ],
    }
    return "".join(out_lines), audit


def write_peak_additional(source_text: str) -> str:
    """Copy the source WAUT file and move the evening-peak start to 54900 s."""
    if "<tlLogic" in source_text or "<phase " in source_text:
        raise SystemExit(f"{SOURCE_ADDITIONAL} contains tlLogic or phase definitions")
    old = f'<wautSwitch time="{PEAK_SWITCH_FROM}" to="1"/>'
    new = f'<wautSwitch time="{PEAK_SWITCH_TO}" to="1"/>'
    if source_text.count(old) != 1:
        raise SystemExit(f"expected exactly one switch {old}")
    if source_text.count('<wautSwitch time="75000" to="0"/>') != 1:
        raise SystemExit("expected the 75000 s switch back to program 0 to stay unchanged")
    updated = source_text.replace(old, new, 1)
    if updated.count(new) != 1 or PEAK_SWITCH_FROM in updated:
        raise SystemExit("peak switch rewrite did not apply cleanly")
    return updated


def additional_notes(text: str) -> dict:
    if "<tlLogic" in text or "<phase " in text:
        raise SystemExit("additional file contains tlLogic or phase definitions")
    switches = re.findall(r'<wautSwitch\b([^>]*)/>', text)
    parsed = []
    for raw in switches:
        attrs = dict(ATTR_RE.findall(raw))
        time_raw = attrs.get("time", "")
        time_value = float(time_raw)
        parsed.append(
            {
                "time": time_raw,
                "to": attrs.get("to"),
                "multiple_of_3": abs(time_value / 3.0 - round(time_value / 3.0)) < 1e-9,
                "remainder_mod_3": int(time_value) % 3,
            }
        )
    waut = re.search(r"<WAUT\b([^>]*)>", text, flags=re.S)
    waut_attrs = dict(ATTR_RE.findall(waut.group(1))) if waut else {}
    ref = waut_attrs.get("refTime", "")
    ref_value = float(ref) if ref else None
    junctions = re.findall(r"<wautJunction\b([^>]*)/>", text, flags=re.S)
    junction_ids = []
    for raw in junctions:
        # Skip the commented example. The real elements are outside comments.
        junction_ids.append(dict(ATTR_RE.findall(raw)).get("junctionID"))
    comment_stripped = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    live_ids = [
        dict(ATTR_RE.findall(raw)).get("junctionID")
        for raw in re.findall(r"<wautJunction\b([^>]*)/>", comment_stripped, flags=re.S)
    ]
    return {
        "waut": waut_attrs,
        "refTime_multiple_of_3": ref_value is not None
        and abs(ref_value / 3.0 - round(ref_value / 3.0)) < 1e-9,
        "switches": parsed,
        "commented_or_live_junction_ids": junction_ids,
        "live_junction_ids": live_ids,
        "switch_times_changed": True,
        "source_switch_to_program_1": PEAK_SWITCH_FROM,
        "output_switch_to_program_1": PEAK_SWITCH_TO,
        "note": (
            f"The 3-second additional file starts program 1 at {PEAK_SWITCH_TO} s. "
            f"night_peak.add.xml still starts it at {PEAK_SWITCH_FROM} s. "
            f"{PEAK_SWITCH_TO} = 3*18300, so the evening-peak switch is on the "
            "3-second grid. The switch back to program 0 stays at 75000 s. "
            "The file comment still mentions 58000/76000; the loaded switches "
            f"are {PEAK_SWITCH_TO} and 75000."
        ),
    }


def render_markdown(audit: dict, extra: dict) -> str:
    shortened = [item for item in audit["programs"] if item["cycle_delta"] < 0]
    lengthened = [item for item in audit["programs"] if item["cycle_delta"] > 0]
    unchanged_cycle = [item for item in audit["programs"] if item["cycle_delta"] == 0]
    lines = [
        "# Xuancheng26 TLS 3-second phase audit",
        "",
        "Phase durations were rewritten from the completed 29-day network.",
        "States, program ids, offsets, lanes, and connections were not changed.",
        f"The evening-peak WAUT switch moves from {PEAK_SWITCH_FROM} s to {PEAK_SWITCH_TO} s.",
        "",
        "## Summary",
        "",
        f"- Source net: `{audit['source_net']}`",
        f"- New net: `{audit['output_net']}`",
        f"- Source additional: `{audit['source_additional']}`",
        f"- New additional: `{audit['output_additional']}`",
        f"- Signal lights (`tlLogic` ids): {audit['tls_count']}",
        f"- Programs: {audit['program_count']}",
        f"- Programs modified: {audit['modified_program_count']}",
        f"- Signals with at least one modified program: {audit['modified_tls_count']}",
        f"- Phases: {audit['phase_count']}",
        f"- Phases modified: {audit['modified_phase_count']}",
        f"- Phases unchanged: {audit['unchanged_phase_count']}",
        f"- Yellow phases (`y` or `Y` in state): {audit['yellow_phase_count']}",
        f"- Yellow phases whose duration changed: {audit['yellow_modified_count']}",
        f"- Phases that are still not a positive multiple of 3: {len(audit['final_phases_not_multiple_of_3'])}",
        f"- Blocked / unmodified special cases: {len(audit['blocked'])}",
        f"- Programs with a longer cycle: {len(lengthened)}",
        f"- Programs with the same cycle: {len(unchanged_cycle)}",
        f"- Programs with a shorter cycle: {len(shortened)}",
        "",
        "A shorter cycle happens when yellow phases move from 4 or 5 seconds to 3 seconds and that reduction is larger than the upward rounding of the other phases. The yellow duration rule is applied as specified.",
        "",
        "Rules: yellow phases are set to 3 seconds. Other phases use `ceil(old / 3) * 3`.",
        "Every controller in the source net is `type=\"static\"`. No `minDur`, `maxDur`, or actuated program was present.",
        "No phase mixes yellow with a permissive or off state.",
        "",
        "## Alignment",
        "",
        "Every `tlLogic` offset is `0`, which lies on the 3-second grid.",
        "Making phase durations multiples of 3 does not by itself move the WAUT switch onto that grid.",
        "",
        f"- WAUT id: `{extra['waut'].get('id')}`",
        f"- refTime: `{extra['waut'].get('refTime')}` (multiple of 3: {extra['refTime_multiple_of_3']})",
        f"- startProg: `{extra['waut'].get('startProg')}`",
        f"- Live junctions switched by WAUT: {len(extra['live_junction_ids'])}",
        "",
        "| switch time (s) | to program | multiple of 3 | remainder |",
        "| --- | --- | --- | --- |",
    ]
    for item in extra["switches"]:
        lines.append(
            f"| {item['time']} | {item['to']} | {item['multiple_of_3']} | {item['remainder_mod_3']} |"
        )
    lines.extend(
        [
            "",
            extra["note"],
            "",
            "## Program cycles",
            "",
            "| tls id | program | offset | phases | old cycle | new cycle | delta |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for program in audit["programs"]:
        lines.append(
            "| {id} | {program_id} | {offset} | {phase_count} | {old_cycle} | {new_cycle} | {cycle_delta} |".format(
                **program
            )
        )
    lines.extend(
        [
            "",
            "## Changed phases",
            "",
            "Unchanged phases are omitted here and listed in `tls_3s_phase_durations.csv`.",
            "",
            "| tls id | program | phase | rule | old | new | state |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for program in audit["programs"]:
        for phase in program["phases"]:
            if not phase["changed"]:
                continue
            lines.append(
                f"| {program['id']} | {program['program_id']} | {phase['index']} | {phase['rule']} | "
                f"{phase['old_duration']} | {phase['new_duration']} | `{phase['state']}` |"
            )
    if audit["blocked"]:
        lines.extend(["", "## Blocked items", ""])
        for item in audit["blocked"]:
            lines.append(f"- {json.dumps(item, ensure_ascii=False)}")
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, audit: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tls_id",
                "program_id",
                "offset",
                "phase_index",
                "rule",
                "has_yellow",
                "old_duration",
                "new_duration",
                "changed",
                "state",
            ],
        )
        writer.writeheader()
        for program in audit["programs"]:
            for phase in program["phases"]:
                writer.writerow(
                    {
                        "tls_id": program["id"],
                        "program_id": program["program_id"],
                        "offset": program["offset"],
                        "phase_index": phase["index"],
                        "rule": phase["rule"],
                        "has_yellow": phase["has_yellow"],
                        "old_duration": phase["old_duration"],
                        "new_duration": phase["new_duration"],
                        "changed": phase["changed"],
                        "state": phase["state"],
                    }
                )
    os.replace(tmp, path)


def validate_pair(source: str, output: str, audit: dict) -> None:
    if source.splitlines() == []:
        raise SystemExit("source net is empty")
    src_lines = source.splitlines()
    out_lines = output.splitlines()
    if len(src_lines) != len(out_lines):
        raise SystemExit(f"line count changed: {len(src_lines)} -> {len(out_lines)}")
    changed = 0
    for index, (left, right) in enumerate(zip(src_lines, out_lines), start=1):
        if left == right:
            continue
        changed += 1
        left_mask = re.sub(r'duration="[^"]+"', 'duration="*"', left)
        right_mask = re.sub(r'duration="[^"]+"', 'duration="*"', right)
        if left_mask != right_mask or "<phase " not in left:
            raise SystemExit(f"non-duration change at line {index}:\n{left}\n{right}")
    if changed != audit["modified_phase_count"]:
        raise SystemExit(
            f"rewritten lines {changed} != modified phases {audit['modified_phase_count']}"
        )
    root = ET.fromstring(output)
    found = []
    for logic in root.findall("tlLogic"):
        states = []
        durations = []
        for phase in list(logic):
            if phase.tag != "phase":
                raise SystemExit(f"unexpected child <{phase.tag}> in {logic.get('id')}")
            duration = int(phase.get("duration"))
            if duration <= 0 or duration % 3 != 0:
                raise SystemExit(
                    f"invalid duration {duration} in {logic.get('id')} program {logic.get('programID')}"
                )
            durations.append(duration)
            states.append(phase.get("state"))
        found.append((logic.get("id"), logic.get("programID"), logic.get("offset"), durations, states))
    expected = []
    for program in audit["programs"]:
        expected.append(
            (
                program["id"],
                program["program_id"],
                program["offset"],
                [phase["new_duration"] for phase in program["phases"]],
                [phase["state"] for phase in program["phases"]],
            )
        )
    if found != expected:
        raise SystemExit("reparsed tlLogic sequence does not match the audit")
    source_root = ET.fromstring(source)
    source_states = [
        [phase.get("state") for phase in logic.findall("phase")]
        for logic in source_root.findall("tlLogic")
    ]
    output_states = [item[4] for item in found]
    if source_states != output_states:
        raise SystemExit("phase states changed")
    if len(source_root.findall("connection")) != len(root.findall("connection")):
        raise SystemExit("connection count changed")
    if len(source_root.findall("edge")) != len(root.findall("edge")):
        raise SystemExit("edge count changed")


def main() -> int:
    if not SOURCE_NET.is_file() or not SOURCE_ADDITIONAL.is_file():
        raise SystemExit("source net or night_peak.add.xml is missing")
    source = SOURCE_NET.read_text(encoding="utf-8")
    output, audit = transform(source)
    if audit["blocked"] or audit["final_phases_not_multiple_of_3"]:
        AUDIT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise SystemExit(
            f"refusing to write {OUTPUT_NET}: blocked={len(audit['blocked'])} "
            f"bad_final={len(audit['final_phases_not_multiple_of_3'])}"
        )
    if audit["tls_count"] == 0 or audit["phase_count"] == 0:
        raise SystemExit("no tlLogic phases parsed")
    validate_pair(source, output, audit)
    additional_text = write_peak_additional(SOURCE_ADDITIONAL.read_text(encoding="utf-8"))
    extra = additional_notes(additional_text)
    program1 = {item["id"] for item in audit["programs"] if item["program_id"] == "1"}
    live = set(extra["live_junction_ids"])
    extra["program1_ids"] = sorted(program1)
    extra["waut_missing_program1"] = sorted(program1 - live)
    extra["waut_without_program1"] = sorted(live - program1)
    if extra["waut_missing_program1"] or extra["waut_without_program1"]:
        raise SystemExit(
            "WAUT junctions do not match program 1: "
            f"missing={extra['waut_missing_program1']} extra={extra['waut_without_program1']}"
        )
    audit["additional"] = extra
    audit["source_net_sha256"] = sha256(SOURCE_NET)
    tmp_net = OUTPUT_NET.with_suffix(OUTPUT_NET.suffix + ".tmp")
    tmp_net.write_text(output, encoding="utf-8")
    os.replace(tmp_net, OUTPUT_NET)
    tmp_additional = OUTPUT_ADDITIONAL.with_suffix(OUTPUT_ADDITIONAL.suffix + ".tmp")
    tmp_additional.write_text(additional_text, encoding="utf-8")
    os.replace(tmp_additional, OUTPUT_ADDITIONAL)
    audit["output_net_sha256"] = sha256(OUTPUT_NET)
    audit["source_additional_sha256"] = sha256(SOURCE_ADDITIONAL)
    audit["output_additional_sha256"] = sha256(OUTPUT_ADDITIONAL)
    if OUTPUT_ADDITIONAL.read_text(encoding="utf-8") != additional_text:
        raise SystemExit("night_peak_3s.add.xml does not match the peak-switch rewrite")
    # Drop per-phase state from the JSON programs? Keep them; the file is small.
    AUDIT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    AUDIT_MD.write_text(render_markdown(audit, extra), encoding="utf-8")
    write_csv(AUDIT_CSV, audit)
    print(
        f"tls={audit['tls_count']} programs={audit['program_count']} "
        f"modified_programs={audit['modified_program_count']} "
        f"phases={audit['phase_count']} modified_phases={audit['modified_phase_count']} "
        f"yellow={audit['yellow_phase_count']}"
    )
    print(f"wrote {OUTPUT_NET}")
    print(f"wrote {OUTPUT_ADDITIONAL}")
    print(f"wrote {AUDIT_MD}")
    return 0


if __name__ == "__main__":
    # Rule checks used by the writer. These are the examples from the task.
    examples = {
        (False, 17): 18,
        (False, 20): 21,
        (False, 30): 30,
        (False, 35): 36,
        (False, 40): 42,
        (True, 4): 3,
        (True, 5): 3,
        (True, 3): 3,
    }
    for (yellow, old), expected in examples.items():
        state = "y" if yellow else "G"
        got, _rule = new_duration(state, old)
        if got != expected:
            raise SystemExit(f"rule self-check failed for {old} -> {got}, expected {expected}")
    sys.exit(main())
