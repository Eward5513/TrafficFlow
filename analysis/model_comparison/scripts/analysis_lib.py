"""Shared constants for the completed-experiment audit.

Reads training results only. Does not import torch and does not write outside
``analysis/model_comparison/``.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

CODE_ROOT = Path("/home/tzhang174/TrafficFlow")
OUT_ROOT = CODE_ROOT / "analysis" / "model_comparison"
TABLES = OUT_ROOT / "tables"
FIGURES = OUT_ROOT / "figures"
STATS = OUT_ROOT / "stats"

LEGACY_ROOT = Path("/data5/tzhang174/TrafficFlow/xuancheng26/training_results")
PHASE3S_ROOT = Path("/data5/tzhang174/TrafficFlow/xuancheng26/training_results_3s")
LEGACY_DATA = Path("/data5/tzhang174/TrafficFlow/xuancheng26/training_data/window_data")
PHASE3S_DATA = Path("/data5/tzhang174/TrafficFlow/xuancheng26/training_data_3s/window_data")
NODE_TABLE = Path("/data5/tzhang174/TrafficFlow/xuancheng26/training_data_3s/graph/r_nodes.csv")

MODELS = (
    "stid",
    "stgcn",
    "dcrnn",
    "graph_wavenet",
    "s2s_gcn_lstm",
    "tlast",
)
MODEL_LABEL = {
    "stid": "STID",
    "stgcn": "STGCN",
    "dcrnn": "DCRNN",
    "graph_wavenet": "Graph WaveNet",
    "s2s_gcn_lstm": "S2S-GCN-LSTM",
    "tlast": "TLAST",
}
MODEL_COLOR = {
    "stid": "#0072B2",
    "stgcn": "#E69F00",
    "dcrnn": "#009E73",
    "graph_wavenet": "#CC79A7",
    "s2s_gcn_lstm": "#D55E00",
    "tlast": "#332288",
}
MODEL_MARKER = {
    "stid": "o",
    "stgcn": "s",
    "dcrnn": "D",
    "graph_wavenet": "^",
    "s2s_gcn_lstm": "v",
    "tlast": "P",
}

RATES = ("p05", "p10", "p20", "p30", "p40", "p50", "p70")
RATE_PERCENT = {name: int(name[1:]) for name in RATES}
STATES = ("inflow", "outflow", "accumulation_start", "accumulation_end")
STATE_LABEL = {
    "inflow": "Inflow",
    "outflow": "Outflow",
    "accumulation_start": "Accumulation start",
    "accumulation_end": "Accumulation end",
}
TASK_LABEL = {
    "single_to_single": "Single-to-single",
    "all_to_single": "All-to-single",
    "all_to_all": "All-to-all",
    "single_channel_inflow": "Single-channel inflow",
}

FAMILY_LEGACY = "legacy_single_channel"
FAMILY_PHASE_SINGLE = "phase3s_single_channel"
FAMILY_FOUR = "four_state"

# Stated once and used everywhere a "small gap" is described. This is not a
# statistical test. Every official run uses one seed.
PRACTICAL_GAP_PCT = 2.0

SPLIT_TEXT = (
    "train 20 days (2023-04-01..2023-04-08 and 2023-04-10..2023-04-21); "
    "validation 4 days (2023-04-22..2023-04-25); "
    "test 5 days (2023-04-26..2023-04-30); "
    "2023-04-09 absent; 277 windows/day; no cross-day window"
)

RATE_RE = re.compile(r"^p(?:05|10|20|30|40|50|70)$")
TASK_RE = re.compile(
    r"^in_(?P<src>all|inflow|outflow|accumulation_start|accumulation_end)"
    r"__out_(?P<dst>all|inflow|outflow|accumulation_start|accumulation_end)$"
)
METRIC_KEYS = ("mae", "rmse", "mape_nonzero", "wape")


def ensure_dirs() -> None:
    for path in (OUT_ROOT, TABLES, FIGURES, STATS, OUT_ROOT / "scripts"):
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def as_int(value: Any) -> int | None:
    number = as_float(value)
    if number is None:
        return None
    return int(number)


def label(model: str) -> str:
    return MODEL_LABEL.get(model, model)


def task_kind(src: str, dst: str) -> str:
    if src == "all" and dst == "all":
        return "all_to_all"
    if src == "all" and dst in STATES:
        return "all_to_single"
    if src == dst and src in STATES:
        return "single_to_single"
    return "other"


def traffic_channels(name: str) -> int | None:
    if name == "all":
        return 4
    if name in STATES:
        return 1
    return None
