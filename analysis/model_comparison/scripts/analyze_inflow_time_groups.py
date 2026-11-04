"""Date × hour × road-group error breakdown for four-state inflow.

Reads saved test predictions only. Does not train and does not use a GPU.

    python3 analysis/model_comparison/scripts/analyze_inflow_time_groups.py

Settings, all at seed 42 on the 3-second-phase four-state experiments:

- A Inflow→Inflow
- B All→Inflow
- C All→All, inflow channel only

Road groups are fixed from the shared test-set mean inflow, using the same
rule as ``error_distribution.flow_group``. Hour, period, and day metrics are
recomputed from additive sums, not by averaging cell-level RMSE or WAPE.
"""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import hashlib
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from analysis_lib import (
    FAMILY_FOUR,
    FIGURES,
    LEGACY_DATA,
    MODEL_COLOR,
    MODEL_LABEL,
    MODELS,
    NODE_TABLE,
    OUT_ROOT,
    PHASE3S_DATA,
    RATE_PERCENT,
    RATES,
    STATES,
    STATS,
    TABLES,
    ensure_dirs,
    label,
    read_csv,
    read_json,
    write_json,
)
from error_distribution import WEEKDAY, check_alignment, flow_group, sha256_array, squeeze_prediction

SETTING_IDS = ("inflow_to_inflow", "all_to_inflow", "all_to_all_inflow")
SETTING_LABEL = {
    "inflow_to_inflow": "Inflow→Inflow",
    "all_to_inflow": "All→Inflow",
    "all_to_all_inflow": "All→All inflow",
}
SETTING_TASK = {
    "inflow_to_inflow": "single_to_single",
    "all_to_inflow": "all_to_single",
    "all_to_all_inflow": "all_to_all",
}
SETTING_CHANNELS = {
    "inflow_to_inflow": 1,
    "all_to_inflow": 1,
    "all_to_all_inflow": 4,
}
COMPARISONS = (
    ("input_B_minus_A", "增加输入（B−A）", 0, 1),
    ("joint_C_minus_B", "联合输出（C−B）", 1, 2),
    ("total_C_minus_A", "总变化（C−A，不是输入融合收益）", 0, 2),
)
GROUP_NAMES = ("inactive", "low", "medium", "high")
GROUP_LABEL = {
    "inactive": "零流量",
    "low": "低流量",
    "medium": "中流量",
    "high": "高流量",
    "all": "全部道路",
}
PERIODS = (
    ("night_00_06", "夜间 00:00–06:00", (0, 1, 2, 3, 4, 5)),
    ("morning_peak_07_09", "早高峰 07:00–09:00", (7, 8)),
    ("midday_10_16", "白天 10:00–16:00", (10, 11, 12, 13, 14, 15)),
    ("evening_peak_17_19", "晚高峰 17:00–19:00", (17, 18)),
)
OTHER_HOURS = (6, 9, 16, 19, 20, 21, 22, 23)
TIME_BUCKETS = PERIODS + (("outside_named_periods", "其余时段（06、09、16、19–23 时）", OTHER_HOURS),)
FOCUS_MODELS = ("stid", "graph_wavenet", "tlast")
FIELDS = (
    "n_elements",
    "n_positive_true",
    "n_zero_true",
    "n_negative_true",
    "n_negative_pred",
    "sum_true",
    "sum_abs_true",
    "sum_error",
    "sum_abs_error",
    "sum_sq_error",
    "sum_ape",
    "U",
    "O",
    "n_over",
    "n_under",
    "n_equal",
)
FIELD_INDEX = {name: index for index, name in enumerate(FIELDS)}
COUNT_FIELDS = {
    "n_elements",
    "n_positive_true",
    "n_zero_true",
    "n_negative_true",
    "n_negative_pred",
    "n_over",
    "n_under",
    "n_equal",
    "n_roads",
    "n_slots",
    "day_index",
    "hour",
}
N_DAY = 5
N_HOUR = 24
N_GROUP = 4
EXPECTED_RUNS = len(MODELS) * len(RATES) * len(SETTING_IDS)
FIGURE_DIR = FIGURES / "inflow_time_groups"
ROAD_PATH = TABLES / "inflow_road_groups.csv"
HOUR_PATH = TABLES / "inflow_time_hour_group.csv"
PERIOD_PATH = TABLES / "inflow_time_period_group.csv"
DAY_PATH = TABLES / "inflow_time_day_group.csv"
DELTA_PATH = TABLES / "inflow_time_setting_deltas.csv"
REPORT_PATH = OUT_ROOT / "inflow_time_group_analysis.md"
VALIDATION_PATH = STATS / "inflow_time_group_validation.json"
FONT_PATH = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
ZH_WEEK = "一二三四五六日"


def fmt_float(value: float) -> str:
    return format(float(value), ".16g")


def fmt_cell(value: object) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    number = float(value)
    if not math.isfinite(number):
        return "undefined"
    if abs(number - round(number)) <= 1e-8 and abs(number) < 1e15:
        return str(int(round(number)))
    return fmt_float(number)


def write_table(path: Path, fieldnames: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt_cell(row.get(key, "")) for key in fieldnames})


def md_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def num(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "不可定义"
    return f"{float(value):.{digits}f}"


def signed(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "不可定义"
    return f"{float(value):+.{digits}f}"


def pct(value: float | None, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "不可定义"
    return f"{float(value):.{digits}f}%"


def signed_pct(value: float | None, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "不可定义"
    return f"{float(value):+.{digits}f}%"


def wan(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "不可定义"
    number = float(value)
    sign = "−" if number < 0 else ""
    magnitude = abs(number)
    if magnitude >= 10000:
        return f"{sign}{magnitude / 10000:.2f} 万"
    return f"{sign}{magnitude:.1f}"


def model_name(model: str) -> str:
    return MODEL_LABEL.get(model, model)


def rate_name(rate: str) -> str:
    return f"{RATE_PERCENT[rate]}%"


def clock(slot: int) -> str:
    return f"{slot // 12:02d}:{(slot % 12) * 5:02d}"


def hour_coverage(hour: int, slot_min: int, slot_max: int, n_slots: int) -> str:
    start = clock(slot_min)
    end = clock(slot_max + 1)
    if n_slots == 12 and slot_min == hour * 12 and slot_max == hour * 12 + 11:
        return f"{hour:02d}:00–{hour + 1:02d}:00（目标时间片 {slot_min}–{slot_max}，12 个 5 分钟片）"
    return (
        f"{hour:02d}:00–{hour + 1:02d}:00 仅覆盖 {start}–{end}"
        f"（目标时间片 {slot_min}–{slot_max}，{n_slots} 个 5 分钟片）"
    )


def period_coverage(hours: tuple[int, ...], slots: np.ndarray) -> str:
    parts = []
    total = 0
    for hour in hours:
        n_slots = int(slots[hour])
        total += n_slots
        parts.append(f"{hour:02d}时 {n_slots} 片")
    return f"合计 {total} 个目标时间片（" + "，".join(parts) + "）"


def load_calendar() -> list[dict[str, object]]:
    manifest = read_json(PHASE3S_DATA / "split_manifest.json")
    if manifest.get("day_index_encoding") != "python_weekday_monday_0":
        raise RuntimeError(f"unexpected day_index_encoding: {manifest.get('day_index_encoding')}")
    listed = manifest.get("test") or manifest.get("test_days")
    if listed != manifest.get("test_days", listed):
        raise RuntimeError("split manifest test and test_days disagree")
    calendar = []
    for raw in listed:
        iso = str(raw).replace("_", "-")
        date = dt.date.fromisoformat(iso)
        calendar.append(
            {
                "day_index": date.weekday(),
                "date": iso,
                "weekday_en": date.strftime("%a"),
                "weekday_zh": "星期" + ZH_WEEK[date.weekday()],
            }
        )
    expected = {index: name for index, name in WEEKDAY.items()}
    observed = {int(item["day_index"]): str(item["date"]) for item in calendar}
    if observed != expected:
        raise RuntimeError(f"test calendar {observed} does not match WEEKDAY {expected}")
    if [item["date"] for item in calendar] != ["2023-04-26", "2023-04-27", "2023-04-28", "2023-04-29", "2023-04-30"]:
        raise RuntimeError(f"unexpected test dates: {calendar}")
    return calendar


def load_nodes() -> tuple[list[dict[str, str]], str]:
    metadata = read_json(PHASE3S_DATA / "dataset_metadata.json")
    legacy = read_json(LEGACY_DATA / "dataset_metadata.json")
    if metadata["node_order_sha256"] != legacy["node_order_sha256"]:
        raise RuntimeError("3-second and legacy node_order_sha256 differ")
    edges = read_csv(NODE_TABLE)
    mapping_path = Path(metadata["node_mapping_path"])
    mapping = read_csv(mapping_path)
    edge_ids = [row["edge_id"] for row in edges]
    if len(edges) != 1406 or edge_ids != [row["edge_id"] for row in mapping]:
        raise RuntimeError("r_nodes.csv and node_mapping.csv disagree or do not have 1406 rows")
    if [row["node_index"] for row in edges] != [str(index) for index in range(1406)]:
        raise RuntimeError("r_nodes.csv node_index is not 0..1405 in file order")
    digest = hashlib.sha256("\n".join(edge_ids).encode("utf-8")).hexdigest()
    if digest != metadata["node_order_sha256"]["r_nodes"]:
        raise RuntimeError("r_nodes edge_id hash does not match dataset_metadata node_order_sha256")
    if digest != metadata["node_order_sha256"]["tensor_column_edge_ids"]:
        raise RuntimeError("tensor column edge-id hash does not match r_nodes")
    return edges, digest


def select_runs() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    inventory = read_csv(OUT_ROOT / "experiment_inventory.csv")
    wanted = []
    anomalies = []
    seen: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in inventory:
        if row["Family"] != FAMILY_FOUR or row["Model"] not in MODELS or row["Penetration Rate"] not in RATES:
            continue
        if row["Random Seed"] != "42":
            continue
        task, src, dst = row["Task Type"], row["Input Type"], row["Output Type"]
        if task == "single_to_single" and src == "inflow" and dst == "inflow":
            setting = "inflow_to_inflow"
        elif task == "all_to_single" and src == "all" and dst == "inflow":
            setting = "all_to_inflow"
        elif task == "all_to_all" and src == "all" and dst == "all":
            setting = "all_to_all_inflow"
        else:
            continue
        reasons = []
        if row["Include In Comparison"] != "yes":
            reasons.append(row["Exclusion Reason"] or "inventory excluded this run")
        if row["Training Status"] != "completed":
            reasons.append(f"training status is {row['Training Status']}")
        result_dir = row["Result Directory"]
        prediction = row["Prediction File"]
        if "training_results_3s" not in result_dir or "training_data_3s" not in row["Data Root"]:
            reasons.append("path is not the 3-second-phase result")
        if "smoke" in result_dir.lower() or "smoke" in prediction.lower():
            reasons.append("smoke path")
        meta_path = Path(result_dir) / "run_metadata.json"
        if not meta_path.is_file():
            reasons.append("run_metadata.json missing")
        else:
            meta = read_json(meta_path)
            if str(meta.get("status") or "") != "completed":
                reasons.append(f"metadata status is {meta.get('status')}")
            if str(meta.get("seed")) not in {"42", ""}:
                reasons.append(f"metadata seed is {meta.get('seed')}")
            if meta.get("did_not_use_test_for_selection") is False:
                reasons.append("metadata says test was used for selection")
            target_mode = str(meta.get("target_mode") or "")
            if target_mode and target_mode != "last-observed-step":
                reasons.append(f"target_mode is {target_mode}")
        if not prediction or not Path(prediction).is_file() or Path(prediction).stat().st_size <= 0:
            reasons.append("prediction file missing or empty")
        record = dict(row)
        record["setting"] = setting
        if reasons:
            anomalies.append(
                {
                    "model": row["Model"],
                    "penetration": row["Penetration Rate"],
                    "setting": setting,
                    "result_directory": result_dir,
                    "reasons": reasons,
                }
            )
            continue
        key = (row["Model"], row["Penetration Rate"], setting)
        if key in seen:
            anomalies.append(
                {
                    "model": row["Model"],
                    "penetration": row["Penetration Rate"],
                    "setting": setting,
                    "result_directory": result_dir,
                    "reasons": ["duplicate included run"],
                }
            )
            continue
        seen[key] = record
    for model in MODELS:
        for rate in RATES:
            for setting in SETTING_IDS:
                row = seen.get((model, rate, setting))
                if row is None:
                    if not any(
                        item["model"] == model and item["penetration"] == rate and item["setting"] == setting
                        for item in anomalies
                    ):
                        anomalies.append(
                            {
                                "model": model,
                                "penetration": rate,
                                "setting": setting,
                                "result_directory": "",
                                "reasons": ["no included inventory row"],
                            }
                        )
                    continue
                wanted.append(row)
    return wanted, anomalies


def empty_cube() -> np.ndarray:
    return np.full((len(FIELDS), len(MODELS), len(RATES), len(SETTING_IDS), N_DAY, N_HOUR, N_GROUP), np.nan)


def pool(cube: np.ndarray, model: int, rate: int, setting: int, days, hours, groups) -> np.ndarray:
    part = cube[:, model, rate, setting][:, list(days)][:, :, list(hours)][:, :, :, list(groups)]
    return part.reshape(part.shape[0], -1).sum(axis=1)


def sums_from_vector(vector: np.ndarray) -> dict[str, float]:
    return {name: float(vector[index]) for index, name in enumerate(FIELDS)}


def sums(cube: np.ndarray, model: int, rate: int, setting: int, days, hours, groups) -> dict[str, float]:
    return sums_from_vector(pool(cube, model, rate, setting, days, hours, groups))


def metrics_from_sums(values: dict[str, float]) -> dict[str, float | None]:
    count = values["n_elements"]
    if count <= 0:
        raise RuntimeError("empty cell")
    positive = values["n_positive_true"]
    abs_error = values["sum_abs_error"]
    true_sum = values["sum_true"]
    abs_true = values["sum_abs_true"]
    error_sum = values["sum_error"]
    under = values["U"]
    over = values["O"]
    share_denom = under + over
    output: dict[str, float | None] = {
        "mean_true": true_sum / count,
        "zero_fraction": values["n_zero_true"] / count,
        "mae": abs_error / count,
        "rmse": math.sqrt(values["sum_sq_error"] / count),
        "mape_nonzero": None if positive == 0 else values["sum_ape"] / positive * 100.0,
        "wape": None if abs_true == 0 else abs_error / abs_true * 100.0,
        "bias": error_sum / count,
        "volume_bias_rate_pct": None if true_sum == 0 else error_sum / true_sum * 100.0,
        "over_fraction": values["n_over"] / count,
        "under_fraction": values["n_under"] / count,
        "equal_fraction": values["n_equal"] / count,
        "over_abs_share": None if share_denom == 0 else over / share_denom,
        "under_abs_share": None if share_denom == 0 else under / share_denom,
        "underestimation_gap": under - over,
        "abs_bias": abs(error_sum / count),
        "abs_net_error": abs(error_sum),
    }
    return output


def cell_metrics(cube: np.ndarray, model: int, rate: int, setting: int, days, hours, groups) -> dict[str, float | None]:
    return metrics_from_sums(sums(cube, model, rate, setting, days, hours, groups))


def defined_sub(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def defined_rel(candidate: float | None, reference: float | None) -> float | None:
    if candidate is None or reference is None or reference == 0:
        return None
    return (candidate - reference) / reference * 100.0


def compare_metrics(reference: dict[str, float | None], candidate: dict[str, float | None]) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for metric in ("mae", "rmse", "mape_nonzero", "wape"):
        output[f"{metric}_reference"] = reference[metric]
        output[f"{metric}_candidate"] = candidate[metric]
        output[f"{metric}_diff"] = defined_sub(candidate[metric], reference[metric])
        output[f"{metric}_rel_pct"] = defined_rel(candidate[metric], reference[metric])
    for metric in ("bias", "volume_bias_rate_pct", "under_abs_share"):
        output[f"{metric}_reference"] = reference[metric]
        output[f"{metric}_candidate"] = candidate[metric]
        output[f"{metric}_diff"] = defined_sub(candidate[metric], reference[metric])
    for metric in ("abs_bias", "abs_net_error", "underestimation_gap"):
        output[f"{metric}_reference"] = reference[metric]
        output[f"{metric}_candidate"] = candidate[metric]
        output[f"{metric}_diff"] = defined_sub(candidate[metric], reference[metric])
    return output


def direct_channel_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    error = pred.astype(np.float64) - true.astype(np.float64)
    abs_error = np.abs(error)
    positive = true.astype(np.float64) > 0.0
    ape = np.zeros(error.shape, dtype=np.float64)
    np.divide(abs_error, true.astype(np.float64), out=ape, where=positive)
    true_sum = float(true.astype(np.float64).sum())
    abs_true = float(np.abs(true.astype(np.float64)).sum())
    abs_sum = float(abs_error.sum())
    return {
        "mae": abs_sum / error.size,
        "rmse": math.sqrt(float(np.square(error).sum()) / error.size),
        "mape_nonzero": float(ape[positive].mean() * 100.0),
        "wape": abs_sum / abs_true * 100.0,
        "bias": float(error.mean()),
        "zero_fraction": float(np.mean(true == 0)),
        "sum_true": true_sum,
        "sum_abs_error": abs_sum,
        "sum_error": float(error.sum()),
    }


def accumulate(pred: np.ndarray, true: np.ndarray, day_code: np.ndarray, hours: np.ndarray, groups: np.ndarray) -> np.ndarray:
    pred64 = pred.astype(np.float64, copy=False)
    true64 = true.astype(np.float64, copy=False)
    error = pred64 - true64
    abs_error = np.abs(error)
    positive = true64 > 0.0
    ape = np.zeros(error.shape, dtype=np.float64)
    np.divide(abs_error, true64, out=ape, where=positive)
    key = np.ascontiguousarray((day_code * N_HOUR + hours)[:, None] * N_GROUP + groups[None, :]).ravel()
    n_bins = N_DAY * N_HOUR * N_GROUP

    def bincount(weights: np.ndarray) -> np.ndarray:
        return np.bincount(key, weights=weights.ravel(), minlength=n_bins).astype(np.float64, copy=False)

    packed = {
        "n_elements": np.bincount(key, minlength=n_bins).astype(np.float64),
        "n_positive_true": bincount(positive.astype(np.float64)),
        "n_zero_true": bincount((true64 == 0.0).astype(np.float64)),
        "n_negative_true": bincount((true64 < 0.0).astype(np.float64)),
        "n_negative_pred": bincount((pred64 < 0.0).astype(np.float64)),
        "sum_true": bincount(true64),
        "sum_abs_true": bincount(np.abs(true64)),
        "sum_error": bincount(error),
        "sum_abs_error": bincount(abs_error),
        "sum_sq_error": bincount(np.square(error)),
        "sum_ape": bincount(ape),
        "U": bincount(np.maximum(-error, 0.0)),
        "O": bincount(np.maximum(error, 0.0)),
        "n_over": bincount((error > 0.0).astype(np.float64)),
        "n_under": bincount((error < 0.0).astype(np.float64)),
        "n_equal": bincount((error == 0.0).astype(np.float64)),
    }
    output = np.empty((len(FIELDS), N_DAY, N_HOUR, N_GROUP), dtype=np.float64)
    for name, values in packed.items():
        output[FIELD_INDEX[name]] = values.reshape(N_DAY, N_HOUR, N_GROUP)
    return output


def self_check() -> None:
    cube = np.zeros((len(FIELDS), 1, 1, 1, N_DAY, N_HOUR, N_GROUP))
    cube[FIELD_INDEX["sum_abs_error"], 0, 0, 0, 1, 7, 2] = 3
    cube[FIELD_INDEX["sum_abs_error"], 0, 0, 0, 1, 8, 2] = 4
    cube[FIELD_INDEX["U"], 0, 0, 0, 1, 7, 2] = 10
    got = pool(cube, 0, 0, 0, [1], [7, 8], [2])
    if got[FIELD_INDEX["sum_abs_error"]] != 7 or got[FIELD_INDEX["U"]] != 10:
        raise RuntimeError("pool indexing self-check failed")
    rng = np.random.default_rng(0)
    error = rng.normal(size=(30, 6))
    under = np.maximum(-error, 0.0)
    over = np.maximum(error, 0.0)
    if np.max(np.abs(under + over - np.abs(error))) != 0 or np.max(np.abs(over - under - error)) != 0:
        raise RuntimeError("U/O identity self-check failed")


def load_reference_tables() -> dict[str, dict[tuple, tuple[float, float]]]:
    def pull(path: Path, key_fields: tuple[str, ...], value_fields: tuple[str, ...]) -> dict[tuple, tuple[float, float]]:
        output = {}
        for row in read_csv(path):
            if row["family"] != FAMILY_FOUR or row["channel"] != "inflow":
                continue
            if row["task_type"] not in SETTING_TASK.values():
                continue
            if row["model"] not in MODELS or row["penetration"] not in RATES:
                continue
            key = tuple(int(row[field]) if field in {"day_index", "hour"} else row[field] for field in key_fields)
            output[key] = tuple(float(row[field]) for field in value_fields)
        return output

    return {
        "day": pull(
            TABLES / "temporal_by_day.csv",
            ("model", "task_type", "penetration", "day_index"),
            ("mae", "bias"),
        ),
        "hour": pull(
            TABLES / "temporal_by_hour.csv",
            ("model", "task_type", "penetration", "hour"),
            ("mae", "bias"),
        ),
        "period": pull(
            TABLES / "temporal_by_period.csv",
            ("model", "task_type", "penetration", "period"),
            ("mae", "bias"),
        ),
        "spatial": pull(
            TABLES / "spatial_by_flow_tercile.csv",
            ("model", "task_type", "penetration", "flow_group"),
            ("mae", "bias"),
        ),
    }


def reference_groups(model: str, task: str, penetration: str) -> dict[int, str]:
    found: dict[int, str] = {}
    with gzip.open(TABLES / "node_metrics.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["family"] == FAMILY_FOUR
                and row["channel"] == "inflow"
                and row["model"] == model
                and row["task_type"] == task
                and row["penetration"] == penetration
            ):
                found[int(row["node_index"])] = row["flow_group"]
    if len(found) != 1406:
        raise RuntimeError(f"node_metrics group lookup returned {len(found)} rows for {model} {task} {penetration}")
    return found


def process_runs(runs: list[dict[str, str]], calendar: list[dict[str, object]]) -> dict[str, object]:
    cube = empty_cube()
    day_lookup = np.full(7, -1, dtype=np.int64)
    for position, item in enumerate(calendar):
        day_lookup[int(item["day_index"])] = position
    reference_hash = ""
    reference_groups_codes: np.ndarray | None = None
    group_info: dict[str, object] = {}
    slot_count = np.zeros((N_DAY, N_HOUR), dtype=np.int64)
    slot_min = np.full((N_DAY, N_HOUR), -1, dtype=np.int64)
    slot_max = np.full((N_DAY, N_HOUR), -1, dtype=np.int64)
    day_slot_hash = ""
    max_direct = {"mae": 0.0, "rmse": 0.0, "mape_nonzero": 0.0, "wape": 0.0, "bias": 0.0, "sum_abs_error": 0.0}
    max_published = {"mae": 0.0, "rmse": 0.0, "mape_nonzero": 0.0, "wape": 0.0, "zero_fraction": 0.0}
    max_day_hour = 0.0
    max_group = 0.0
    negative_true_total = 0.0
    processed = []
    for index, row in enumerate(runs, start=1):
        setting = row["setting"]
        model_index = MODELS.index(row["Model"])
        rate_index = RATES.index(row["Penetration Rate"])
        setting_index = SETTING_IDS.index(setting)
        with np.load(row["Prediction File"]) as loaded:
            pred = squeeze_prediction(np.asarray(loaded["pred_raw"]))
            true = squeeze_prediction(np.asarray(loaded["y_raw"]))
            day = np.asarray(loaded["day_index"]).reshape(-1)
            start = np.asarray(loaded["window_start_slot"]).reshape(-1)
            target = np.asarray(loaded["target_slot"]).reshape(-1)
        if pred.shape != true.shape:
            raise RuntimeError(f"shape mismatch {row['Prediction File']}")
        expected_channels = SETTING_CHANNELS[setting]
        if pred.ndim != 3 or pred.shape[-1] != expected_channels or pred.shape[1] != 1406 or pred.shape[0] != 1385:
            raise RuntimeError(f"unexpected prediction shape {pred.shape} for {row['Prediction File']}")
        if STATES[0] != "inflow":
            raise RuntimeError("inflow is no longer channel 0")
        pred = np.ascontiguousarray(pred[..., 0])
        true = np.ascontiguousarray(true[..., 0])
        alignment = check_alignment(start, target, day)
        if not alignment["target_is_last_input_slot"] or not alignment["test_days_match_manifest"]:
            raise RuntimeError(f"alignment failed for {row['Prediction File']}: {alignment}")
        day_code = day_lookup[day]
        hours = np.asarray(target // 12, dtype=np.int64)
        if np.any(day_code < 0) or np.any((hours < 0) | (hours > 23)):
            raise RuntimeError(f"day or hour outside the test calendar: {row['Prediction File']}")
        label_hash = sha256_array(true)
        slot_hash = sha256_array(day) + ":" + sha256_array(target)
        if not reference_hash:
            reference_hash = label_hash
            day_slot_hash = slot_hash
            names, info = flow_group(true.mean(axis=0))
            codes = np.array([GROUP_NAMES.index(str(name)) for name in names], dtype=np.int64)
            counts = {name: int(np.sum(codes == code)) for code, name in enumerate(GROUP_NAMES)}
            if counts != {"inactive": 266, "low": 380, "medium": 380, "high": 380}:
                raise RuntimeError(f"unexpected road-group counts {counts}")
            if len(set(codes.tolist())) != 4 or codes.shape != (1406,):
                raise RuntimeError("road groups are not a complete partition")
            inactive_max = float(true.mean(axis=0)[codes == 0].max())
            if inactive_max != 0.0:
                raise RuntimeError(f"inactive roads are not exactly zero; max mean is {inactive_max}")
            reference_groups_codes = codes
            group_info = {
                "counts": counts,
                "low_max_mean": float(info["low_max_mean"]),
                "mid_max_mean": float(info["mid_max_mean"]),
                "inactive_max_mean": inactive_max,
                "label_sha256": label_hash,
                "mean_true": np.asarray(true.mean(axis=0)),
                "codes": codes,
            }
            for day_position in range(N_DAY):
                for hour in range(N_HOUR):
                    slots = target[(day_code == day_position) & (hours == hour)]
                    slot_count[day_position, hour] = slots.size
                    slot_min[day_position, hour] = int(slots.min())
                    slot_max[day_position, hour] = int(slots.max())
            if not np.array_equal(slot_count, np.tile(slot_count[0], (N_DAY, 1))):
                raise RuntimeError("target-slot coverage differs across test days")
            if int(slot_count[0, 0]) != 1 or int(slot_min[0, 0]) != 11 or int(slot_max[0, 0]) != 11:
                raise RuntimeError(f"hour 0 coverage is not the single 00:55 slot: {slot_count[0, 0]} {slot_min[0, 0]}")
            for hour in range(1, 24):
                if int(slot_count[0, hour]) != 12 or int(slot_min[0, hour]) != hour * 12 or int(slot_max[0, hour]) != hour * 12 + 11:
                    raise RuntimeError(f"hour {hour} is not a complete 12-slot hour")
            if int(slot_count[0].sum()) != 277:
                raise RuntimeError("daily target slots do not sum to 277")
        elif label_hash != reference_hash or slot_hash != day_slot_hash:
            raise RuntimeError(f"inflow labels or time index differ: {row['Prediction File']}")
        else:
            names, _info = flow_group(true.mean(axis=0))
            codes = np.array([GROUP_NAMES.index(str(name)) for name in names], dtype=np.int64)
            if not np.array_equal(codes, reference_groups_codes):
                raise RuntimeError(f"road groups changed for {row['Prediction File']}")
        binned = accumulate(pred, true, day_code, hours, reference_groups_codes)
        direct = direct_channel_metrics(pred, true)
        pooled = metrics_from_sums(sums_from_vector(binned.reshape(len(FIELDS), -1).sum(axis=1)))
        for key in ("mae", "rmse", "mape_nonzero", "wape", "bias"):
            max_direct[key] = max(max_direct[key], abs(float(pooled[key]) - direct[key]))
        max_direct["sum_abs_error"] = max(
            max_direct["sum_abs_error"],
            abs(float(binned[FIELD_INDEX["sum_abs_error"]].sum()) - direct["sum_abs_error"]),
        )
        abs_error = np.abs(pred.astype(np.float64) - true.astype(np.float64))
        error = pred.astype(np.float64) - true.astype(np.float64)
        for day_position in range(N_DAY):
            for hour in range(N_HOUR):
                mask = (day_code == day_position) & (hours == hour)
                max_day_hour = max(max_day_hour, abs(float(abs_error[mask].sum()) - float(binned[FIELD_INDEX["sum_abs_error"], day_position, hour].sum())))
                max_day_hour = max(max_day_hour, abs(float(error[mask].sum()) - float(binned[FIELD_INDEX["sum_error"], day_position, hour].sum())))
        for group in range(N_GROUP):
            mask = reference_groups_codes == group
            max_group = max(max_group, abs(float(abs_error[:, mask].sum()) - float(binned[FIELD_INDEX["sum_abs_error"], :, :, group].sum())))
            max_group = max(max_group, abs(float(error[:, mask].sum()) - float(binned[FIELD_INDEX["sum_error"], :, :, group].sum())))
        published = read_json(Path(row["Metrics File"]))
        prefix = "inflow_" if setting == "all_to_all_inflow" else ""
        for key in ("mae", "rmse", "mape_nonzero", "wape"):
            max_published[key] = max(max_published[key], abs(float(pooled[key]) - float(published[f"{prefix}{key}"])))
        published_zero = float(published[f"{prefix}zero_target_fraction"])
        max_published["zero_fraction"] = max(max_published["zero_fraction"], abs(float(pooled["zero_fraction"]) - published_zero))
        negative_true_total += float(binned[FIELD_INDEX["n_negative_true"]].sum())
        cube[:, model_index, rate_index, setting_index] = binned
        processed.append(
            {
                "model": row["Model"],
                "penetration": row["Penetration Rate"],
                "setting": setting,
                "prediction_file": row["Prediction File"],
                "mae": pooled["mae"],
                "published_mae": float(published[f"{prefix}mae"]),
            }
        )
        print(f"processed {index}/{len(runs)} {row['Model']} {row['Penetration Rate']} {setting}", flush=True)
    if np.isnan(cube).any():
        raise RuntimeError("prediction cube has missing cells")
    return {
        "cube": cube,
        "group_info": group_info,
        "slot_count": slot_count,
        "slot_min": slot_min,
        "slot_max": slot_max,
        "max_direct": max_direct,
        "max_published": max_published,
        "max_day_hour_sum_gap": max_day_hour,
        "max_group_sum_gap": max_group,
        "negative_true_total": negative_true_total,
        "processed": processed,
        "day_slot_hash": day_slot_hash,
    }


def check_existing_tables(cube: np.ndarray, references: dict[str, dict[tuple, tuple[float, float]]]) -> dict[str, float]:
    gaps = {"day_mae": 0.0, "day_bias": 0.0, "hour_mae": 0.0, "hour_bias": 0.0, "period_mae": 0.0, "period_bias": 0.0, "spatial_mae": 0.0, "spatial_bias": 0.0}
    missing = []
    for model_index, model in enumerate(MODELS):
        for rate_index, rate in enumerate(RATES):
            for setting_index, setting in enumerate(SETTING_IDS):
                task = SETTING_TASK[setting]
                for day_position, item in enumerate(load_calendar()):
                    metric = cell_metrics(cube, model_index, rate_index, setting_index, [day_position], range(N_HOUR), range(N_GROUP))
                    key = (model, task, rate, int(item["day_index"]))
                    if key not in references["day"]:
                        missing.append(("day", key))
                        continue
                    published_mae, published_bias = references["day"][key]
                    gaps["day_mae"] = max(gaps["day_mae"], abs(float(metric["mae"]) - published_mae))
                    gaps["day_bias"] = max(gaps["day_bias"], abs(float(metric["bias"]) - published_bias))
                for hour in range(N_HOUR):
                    metric = cell_metrics(cube, model_index, rate_index, setting_index, range(N_DAY), [hour], range(N_GROUP))
                    key = (model, task, rate, hour)
                    if key not in references["hour"]:
                        missing.append(("hour", key))
                        continue
                    published_mae, published_bias = references["hour"][key]
                    gaps["hour_mae"] = max(gaps["hour_mae"], abs(float(metric["mae"]) - published_mae))
                    gaps["hour_bias"] = max(gaps["hour_bias"], abs(float(metric["bias"]) - published_bias))
                for period, _label, hours in PERIODS:
                    metric = cell_metrics(cube, model_index, rate_index, setting_index, range(N_DAY), hours, range(N_GROUP))
                    key = (model, task, rate, period)
                    if key not in references["period"]:
                        missing.append(("period", key))
                        continue
                    published_mae, published_bias = references["period"][key]
                    gaps["period_mae"] = max(gaps["period_mae"], abs(float(metric["mae"]) - published_mae))
                    gaps["period_bias"] = max(gaps["period_bias"], abs(float(metric["bias"]) - published_bias))
                for group_index, group in enumerate(GROUP_NAMES):
                    metric = cell_metrics(cube, model_index, rate_index, setting_index, range(N_DAY), range(N_HOUR), [group_index])
                    key = (model, task, rate, group)
                    if key not in references["spatial"]:
                        missing.append(("spatial", key))
                        continue
                    published_mae, published_bias = references["spatial"][key]
                    gaps["spatial_mae"] = max(gaps["spatial_mae"], abs(float(metric["mae"]) - published_mae))
                    gaps["spatial_bias"] = max(gaps["spatial_bias"], abs(float(metric["bias"]) - published_bias))
    if missing:
        raise RuntimeError(f"existing comparison tables are missing {len(missing)} keys, for example {missing[:3]}")
    return gaps


def identity_gaps(cube: np.ndarray) -> dict[str, float]:
    values = {name: cube[index] for index, name in enumerate(FIELDS)}
    share = values["U"] + values["O"]
    return {
        "U_plus_O_minus_abs": float(np.max(np.abs(share - values["sum_abs_error"]))),
        "O_minus_U_minus_error": float(np.max(np.abs(values["O"] - values["U"] - values["sum_error"]))),
        "direction_count": float(np.max(np.abs(values["n_over"] + values["n_under"] + values["n_equal"] - values["n_elements"]))),
        "true_abs_gap": float(np.max(np.abs(values["sum_abs_true"] - values["sum_true"]))),
        "negative_true": float(values["n_negative_true"].sum()),
    }


def coverage_gaps(cube: np.ndarray, slot_count: np.ndarray, group_counts: dict[str, int]) -> float:
    gap = 0.0
    elements = cube[FIELD_INDEX["n_elements"]]
    for day in range(N_DAY):
        for hour in range(N_HOUR):
            for group_index, name in enumerate(GROUP_NAMES):
                expected = float(slot_count[day, hour] * group_counts[name])
                gap = max(gap, abs(float(elements[:, :, :, day, hour, group_index].max()) - expected))
                gap = max(gap, abs(float(elements[:, :, :, day, hour, group_index].min()) - expected))
    return gap


def base_row(model: str, rate: str, setting: str) -> dict[str, object]:
    return {
        "family": FAMILY_FOUR,
        "channel": "inflow",
        "model": model,
        "penetration": rate,
        "setting": setting,
        "setting_label": SETTING_LABEL[setting],
        "task_type": SETTING_TASK[setting],
    }


def metric_fields(values: dict[str, float], metric: dict[str, float | None], n_roads: int, n_slots: int) -> dict[str, object]:
    row: dict[str, object] = {"n_roads": n_roads, "n_slots": n_slots}
    for name in FIELDS:
        row[name] = values[name]
    for name, value in metric.items():
        if name in {"abs_bias", "abs_net_error"}:
            continue
        row[name] = value
    return row


METRIC_COLUMNS = [
    "n_roads",
    "n_slots",
    "n_elements",
    "n_positive_true",
    "n_zero_true",
    "n_negative_true",
    "n_negative_pred",
    "sum_true",
    "sum_abs_true",
    "sum_error",
    "sum_abs_error",
    "sum_sq_error",
    "sum_ape",
    "under_abs_error",
    "over_abs_error",
    "n_over",
    "n_under",
    "n_equal",
    "mean_true",
    "zero_fraction",
    "mae",
    "rmse",
    "mape_nonzero",
    "wape",
    "bias",
    "volume_bias_rate_pct",
    "over_fraction",
    "under_fraction",
    "equal_fraction",
    "over_abs_share",
    "under_abs_share",
    "underestimation_gap",
]


def iter_hour_rows(cube, calendar, slot_count, slot_min, slot_max, group_counts):
    for model_index, model in enumerate(MODELS):
        for rate_index, rate in enumerate(RATES):
            for setting_index, setting in enumerate(SETTING_IDS):
                for day_position, item in enumerate(calendar):
                    for hour in range(N_HOUR):
                        coverage = hour_coverage(hour, int(slot_min[day_position, hour]), int(slot_max[day_position, hour]), int(slot_count[day_position, hour]))
                        for group_index, group in enumerate(GROUP_NAMES):
                            values = sums(cube, model_index, rate_index, setting_index, [day_position], [hour], [group_index])
                            metric = metrics_from_sums(values)
                            row = base_row(model, rate, setting)
                            row.update(
                                {
                                    "date": item["date"],
                                    "day_index": item["day_index"],
                                    "weekday": item["weekday_en"],
                                    "hour": hour,
                                    "time_coverage": coverage,
                                    "flow_group": group,
                                }
                            )
                            row.update(metric_fields(values, metric, group_counts[group], int(slot_count[day_position, hour])))
                            yield row


def iter_period_rows(cube, calendar, slot_count, group_counts):
    for model_index, model in enumerate(MODELS):
        for rate_index, rate in enumerate(RATES):
            for setting_index, setting in enumerate(SETTING_IDS):
                for day_position, item in enumerate(calendar):
                    for period, period_label, hours in PERIODS:
                        n_slots = int(sum(int(slot_count[day_position, hour]) for hour in hours))
                        for group_index, group in enumerate(GROUP_NAMES):
                            values = sums(cube, model_index, rate_index, setting_index, [day_position], hours, [group_index])
                            metric = metrics_from_sums(values)
                            row = base_row(model, rate, setting)
                            row.update(
                                {
                                    "date": item["date"],
                                    "day_index": item["day_index"],
                                    "weekday": item["weekday_en"],
                                    "period": period,
                                    "period_label": period_label,
                                    "included_hours": ",".join(str(hour) for hour in hours),
                                    "time_coverage": period_coverage(hours, slot_count[day_position]),
                                    "flow_group": group,
                                }
                            )
                            row.update(metric_fields(values, metric, group_counts[group], n_slots))
                            yield row


def iter_day_rows(cube, calendar, group_counts):
    for model_index, model in enumerate(MODELS):
        for rate_index, rate in enumerate(RATES):
            for setting_index, setting in enumerate(SETTING_IDS):
                for day_position, item in enumerate(calendar):
                    for group_index, group in enumerate(GROUP_NAMES):
                        values = sums(cube, model_index, rate_index, setting_index, [day_position], range(N_HOUR), [group_index])
                        metric = metrics_from_sums(values)
                        row = base_row(model, rate, setting)
                        row.update(
                            {
                                "date": item["date"],
                                "day_index": item["day_index"],
                                "weekday": item["weekday_en"],
                                "time_coverage": "全天 277 个目标时间片，从时间片 11（00:55）到时间片 287（23:55）",
                                "flow_group": group,
                            }
                        )
                        row.update(metric_fields(values, metric, group_counts[group], 277))
                        yield row


def delta_row(grain, comparison, comparison_label, model, rate, reference_setting, candidate_setting, location, reference_sums, candidate_sums, n_roads, n_slots):
    reference = metrics_from_sums(reference_sums)
    candidate = metrics_from_sums(candidate_sums)
    compared = compare_metrics(reference, candidate)
    row = {
        "grain": grain,
        "comparison": comparison,
        "comparison_label": comparison_label,
        "model": model,
        "penetration": rate,
        "reference_setting": reference_setting,
        "candidate_setting": candidate_setting,
        "n_elements": reference_sums["n_elements"],
        "n_roads": n_roads,
        "n_slots": n_slots,
        "sum_abs_error_reference": reference_sums["sum_abs_error"],
        "sum_abs_error_candidate": candidate_sums["sum_abs_error"],
        "sum_abs_error_diff": candidate_sums["sum_abs_error"] - reference_sums["sum_abs_error"],
        "U_reference": reference_sums["U"],
        "U_candidate": candidate_sums["U"],
        "U_diff": candidate_sums["U"] - reference_sums["U"],
        "O_reference": reference_sums["O"],
        "O_candidate": candidate_sums["O"],
        "O_diff": candidate_sums["O"] - reference_sums["O"],
        "sum_error_reference": reference_sums["sum_error"],
        "sum_error_candidate": candidate_sums["sum_error"],
        "sum_error_diff": candidate_sums["sum_error"] - reference_sums["sum_error"],
        "n_over_reference": reference_sums["n_over"],
        "n_over_candidate": candidate_sums["n_over"],
        "n_under_reference": reference_sums["n_under"],
        "n_under_candidate": candidate_sums["n_under"],
        "n_equal_reference": reference_sums["n_equal"],
        "n_equal_candidate": candidate_sums["n_equal"],
    }
    row.update(location)
    row.update(compared)
    if reference_sums["n_elements"] != candidate_sums["n_elements"]:
        raise RuntimeError("paired cells have different sample counts")
    return row


def iter_delta_rows(cube, calendar, slot_count, group_counts):
    def emit(grain, model_index, rate_index, ref_index, cand_index, days, hours, groups, location, n_roads, n_slots):
        reference_sums = sums(cube, model_index, rate_index, ref_index, days, hours, groups)
        candidate_sums = sums(cube, model_index, rate_index, cand_index, days, hours, groups)
        comparison, comparison_label, _ref, _cand = next(
            item for item in COMPARISONS if item[2] == ref_index and item[3] == cand_index
        )
        return delta_row(
            grain,
            comparison,
            comparison_label,
            MODELS[model_index],
            RATES[rate_index],
            SETTING_IDS[ref_index],
            SETTING_IDS[cand_index],
            location,
            reference_sums,
            candidate_sums,
            n_roads,
            n_slots,
        )

    for model_index, model in enumerate(MODELS):
        for rate_index, rate in enumerate(RATES):
            for _comparison, _label, ref_index, cand_index in COMPARISONS:
                yield emit(
                    "network",
                    model_index,
                    rate_index,
                    ref_index,
                    cand_index,
                    range(N_DAY),
                    range(N_HOUR),
                    range(N_GROUP),
                    {
                        "date": "all",
                        "day_index": "",
                        "weekday": "",
                        "hour": "",
                        "period": "",
                        "flow_group": "all",
                        "time_coverage": "五个测试日全部 1385 个目标时间片 × 1406 条道路",
                    },
                    1406,
                    1385,
                )
                for day_position, item in enumerate(calendar):
                    for group_index, group in enumerate(GROUP_NAMES):
                        yield emit(
                            "day",
                            model_index,
                            rate_index,
                            ref_index,
                            cand_index,
                            [day_position],
                            range(N_HOUR),
                            [group_index],
                            {
                                "date": item["date"],
                                "day_index": item["day_index"],
                                "weekday": item["weekday_en"],
                                "hour": "",
                                "period": "",
                                "flow_group": group,
                                "time_coverage": "全天 277 个目标时间片",
                            },
                            group_counts[group],
                            277,
                        )
                    for period, _period_label, hours in PERIODS:
                        n_slots = int(sum(int(slot_count[day_position, hour]) for hour in hours))
                        for group_index, group in enumerate(GROUP_NAMES):
                            yield emit(
                                "period",
                                model_index,
                                rate_index,
                                ref_index,
                                cand_index,
                                [day_position],
                                hours,
                                [group_index],
                                {
                                    "date": item["date"],
                                    "day_index": item["day_index"],
                                    "weekday": item["weekday_en"],
                                    "hour": "",
                                    "period": period,
                                    "flow_group": group,
                                    "time_coverage": period_coverage(hours, slot_count[day_position]),
                                },
                                group_counts[group],
                                n_slots,
                            )
                    for hour in range(N_HOUR):
                        for group_index, group in enumerate(GROUP_NAMES):
                            yield emit(
                                "hour",
                                model_index,
                                rate_index,
                                ref_index,
                                cand_index,
                                [day_position],
                                [hour],
                                [group_index],
                                {
                                    "date": item["date"],
                                    "day_index": item["day_index"],
                                    "weekday": item["weekday_en"],
                                    "hour": hour,
                                    "period": "",
                                    "flow_group": group,
                                    "time_coverage": hour_coverage(
                                        hour,
                                        int(slot_count[day_position, hour] and (hour * 12 if hour else 11)),
                                        int((hour * 12 + 11) if hour else 11),
                                        int(slot_count[day_position, hour]),
                                    ),
                                },
                                group_counts[group],
                                int(slot_count[day_position, hour]),
                            )


HOUR_FIELDS = [
    "family",
    "channel",
    "model",
    "penetration",
    "setting",
    "setting_label",
    "task_type",
    "date",
    "day_index",
    "weekday",
    "hour",
    "time_coverage",
    "flow_group",
    *METRIC_COLUMNS,
]
PERIOD_FIELDS = [
    "family",
    "channel",
    "model",
    "penetration",
    "setting",
    "setting_label",
    "task_type",
    "date",
    "day_index",
    "weekday",
    "period",
    "period_label",
    "included_hours",
    "time_coverage",
    "flow_group",
    *METRIC_COLUMNS,
]
DAY_FIELDS = [
    "family",
    "channel",
    "model",
    "penetration",
    "setting",
    "setting_label",
    "task_type",
    "date",
    "day_index",
    "weekday",
    "time_coverage",
    "flow_group",
    *METRIC_COLUMNS,
]
DELTA_FIELDS = [
    "grain",
    "comparison",
    "comparison_label",
    "model",
    "penetration",
    "date",
    "day_index",
    "weekday",
    "hour",
    "period",
    "flow_group",
    "time_coverage",
    "reference_setting",
    "candidate_setting",
    "n_elements",
    "n_roads",
    "n_slots",
    "mae_reference",
    "mae_candidate",
    "mae_diff",
    "mae_rel_pct",
    "rmse_reference",
    "rmse_candidate",
    "rmse_diff",
    "rmse_rel_pct",
    "mape_nonzero_reference",
    "mape_nonzero_candidate",
    "mape_nonzero_diff",
    "mape_nonzero_rel_pct",
    "wape_reference",
    "wape_candidate",
    "wape_diff",
    "wape_rel_pct",
    "bias_reference",
    "bias_candidate",
    "bias_diff",
    "abs_bias_reference",
    "abs_bias_candidate",
    "abs_bias_diff",
    "sum_error_reference",
    "sum_error_candidate",
    "sum_error_diff",
    "abs_net_error_reference",
    "abs_net_error_candidate",
    "abs_net_error_diff",
    "underestimation_gap_reference",
    "underestimation_gap_candidate",
    "underestimation_gap_diff",
    "U_reference",
    "U_candidate",
    "U_diff",
    "O_reference",
    "O_candidate",
    "O_diff",
    "sum_abs_error_reference",
    "sum_abs_error_candidate",
    "sum_abs_error_diff",
    "under_abs_share_reference",
    "under_abs_share_candidate",
    "volume_bias_rate_pct_reference",
    "volume_bias_rate_pct_candidate",
    "volume_bias_rate_pct_diff",
    "n_over_reference",
    "n_over_candidate",
    "n_under_reference",
    "n_under_candidate",
    "n_equal_reference",
    "n_equal_candidate",
]


def write_road_groups(edges, group_info) -> None:
    rows = []
    means = group_info["mean_true"]
    codes = group_info["codes"]
    for index, edge in enumerate(edges):
        rows.append(
            {
                "node_index": index,
                "node_id": edge["node_id"],
                "edge_id": edge["edge_id"],
                "mean_true_inflow": float(means[index]),
                "flow_group": GROUP_NAMES[int(codes[index])],
                "low_max_mean": group_info["low_max_mean"],
                "mid_max_mean": group_info["mid_max_mean"],
                "n_test_slots": 1385,
                "label_sha256": group_info["label_sha256"],
                "group_rule": "error_distribution.flow_group on the full five-day test mean; inactive if mean <= 1e-6",
            }
        )
    path = ROAD_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        writer.writerows(rows)


def use_chinese_font() -> None:
    cjk = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if cjk.is_file():
        font_manager.fontManager.addfont(str(cjk))
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def finish_figure(fig: plt.Figure, path: Path, tight: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_true_volume(cube, calendar, slot_count) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for day_position, item in enumerate(calendar):
        totals = []
        zeros = []
        for hour in range(N_HOUR):
            values = sums(cube, 0, 0, 0, [day_position], [hour], range(N_GROUP))
            totals.append(values["sum_true"])
            zeros.append(values["n_zero_true"] / values["n_elements"])
        axes[0].plot(range(24), totals, linewidth=1.6, label=f"{item['date'][5:]} {item['weekday_zh']}")
        axes[1].plot(range(24), zeros, linewidth=1.6, label=f"{item['date'][5:]} {item['weekday_zh']}")
    axes[0].set_ylabel("真实 inflow 总和（辆）")
    axes[1].set_ylabel("零值比例")
    for axis in axes:
        axis.set_xlabel("小时")
        axis.set_xticks(range(0, 24, 2))
        axis.axvspan(7, 9, color="#bbbbbb", alpha=0.35)
        axis.axvspan(17, 19, color="#888888", alpha=0.18)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("五个测试日的 inflow 真实值。00 时只有 00:55 一个目标时间片。")
    finish_figure(fig, FIGURE_DIR / "true_volume_by_hour.png")


def plot_day_mae_wape(cube, calendar) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.4), sharex=True)
    dates = [item["date"][5:] for item in calendar]
    for column, rate in enumerate(("p05", "p70")):
        rate_index = RATES.index(rate)
        for model in FOCUS_MODELS:
            model_index = MODELS.index(model)
            maes = []
            wapes = []
            for day_position in range(N_DAY):
                metric = cell_metrics(cube, model_index, rate_index, 0, [day_position], range(N_HOUR), range(N_GROUP))
                maes.append(metric["mae"])
                wapes.append(metric["wape"])
            axes[0, column].plot(dates, maes, marker="o", color=MODEL_COLOR[model], label=model_name(model))
            axes[1, column].plot(dates, wapes, marker="o", color=MODEL_COLOR[model], label=model_name(model))
        axes[0, column].set_title(f"{rate_name(rate)}  Inflow→Inflow")
        axes[0, column].set_ylabel("MAE（辆/道路/5分钟）")
        axes[1, column].set_ylabel("WAPE（%）")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("绝对误差和相对误差是否在同一天升高")
    finish_figure(fig, FIGURE_DIR / "day_mae_and_wape.png")


def plot_direction(cube) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(12.8, 10.6), layout="constrained")
    period_labels = ["夜间", "早高峰", "白天", "晚高峰"]
    group_labels = ["低", "中", "高"]
    specs = (
        (0, "under_fraction", "p05 样本低估比例"),
        (1, "under_abs_share", "p05 绝对误差低估占比"),
        (2, "under_fraction", "p70 样本低估比例"),
        (3, "under_abs_share", "p70 绝对误差低估占比"),
    )
    image = None
    for row, (rate_name_index, metric_name, title) in enumerate(specs):
        rate = "p05" if row < 2 else "p70"
        rate_index = RATES.index(rate)
        for column, model in enumerate(FOCUS_MODELS):
            model_index = MODELS.index(model)
            grid = np.zeros((3, 4))
            for group_offset, group in enumerate((1, 2, 3)):
                for period_index, (_period, _label, hours) in enumerate(PERIODS):
                    metric = cell_metrics(cube, model_index, rate_index, 0, range(N_DAY), hours, [group])
                    grid[group_offset, period_index] = metric[metric_name]
            axis = axes[row, column]
            image = axis.imshow(grid, vmin=0.35, vmax=0.85, cmap="RdBu_r", aspect="auto")
            axis.set_xticks(range(4), period_labels, fontsize=8)
            axis.set_yticks(range(3), group_labels)
            if row == 0:
                axis.set_title(model_name(model))
            if column == 0:
                axis.set_ylabel(title, fontsize=8)
            for y in range(3):
                for x in range(4):
                    axis.text(x, y, f"{grid[y, x] * 100:.0f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02, label="比例")
    fig.suptitle("Inflow→Inflow，五日合计。大于 50% 偏低估。")
    finish_figure(fig, FIGURE_DIR / "direction_sample_vs_mass.png", tight=False)


def relative_grid(cube, model_index, rate_index, ref_setting, cand_setting, days, group_indexes, hour_sets) -> np.ndarray:
    grid = np.full((len(group_indexes), len(hour_sets)), np.nan)
    for row, group in enumerate(group_indexes):
        for column, hours in enumerate(hour_sets):
            reference = cell_metrics(cube, model_index, rate_index, ref_setting, days, hours, [group])
            candidate = cell_metrics(cube, model_index, rate_index, cand_setting, days, hours, [group])
            grid[row, column] = defined_rel(candidate["mae"], reference["mae"])
    return grid


def plot_input_heatmaps(cube) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.4), layout="constrained")
    period_labels = ["夜间", "早高峰", "白天", "晚高峰"]
    group_labels = ["低", "中", "高"]
    grids = []
    for column, model in enumerate(FOCUS_MODELS):
        for row, rate in enumerate(("p05", "p70")):
            grid = relative_grid(
                cube,
                MODELS.index(model),
                RATES.index(rate),
                0,
                1,
                range(N_DAY),
                (1, 2, 3),
                [hours for _name, _label, hours in PERIODS],
            )
            grids.append(grid)
            axis = axes[row, column]
            image = axis.imshow(grid, cmap="RdBu_r", vmin=-8, vmax=8, aspect="auto")
            axis.set_xticks(range(4), period_labels, fontsize=8)
            axis.set_yticks(range(3), group_labels)
            if row == 0:
                axis.set_title(model_name(model))
            if column == 0:
                axis.set_ylabel(rate_name(rate))
            for y in range(grid.shape[0]):
                for x in range(grid.shape[1]):
                    axis.text(x, y, f"{grid[y, x]:+.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02, label="MAE 相对变化（%）")
    fig.suptitle("增加输入：All→Inflow 相对 Inflow→Inflow。负值表示 MAE 下降。五日合计，色标 ±8%。")
    finish_figure(fig, FIGURE_DIR / "input_mae_rel_group_period.png", tight=False)
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.4))
    for column, model in enumerate(FOCUS_MODELS):
        for row, rate in enumerate(("p05", "p70")):
            grid = np.full((N_DAY, 4), np.nan)
            for day in range(N_DAY):
                for period_index, (_name, _label, hours) in enumerate(PERIODS):
                    reference = cell_metrics(cube, MODELS.index(model), RATES.index(rate), 0, [day], hours, [3])
                    candidate = cell_metrics(cube, MODELS.index(model), RATES.index(rate), 1, [day], hours, [3])
                    grid[day, period_index] = defined_rel(candidate["mae"], reference["mae"])
            axis = axes[row, column]
            axis.imshow(grid, cmap="RdBu_r", vmin=-8, vmax=8, aspect="auto")
            axis.set_xticks(range(4), period_labels, fontsize=8)
            axis.set_yticks(range(N_DAY), ["04-26", "04-27", "04-28", "04-29", "04-30"])
            if row == 0:
                axis.set_title(model_name(model))
            if column == 0:
                axis.set_ylabel(rate_name(rate))
            for y in range(grid.shape[0]):
                for x in range(grid.shape[1]):
                    axis.text(x, y, f"{grid[y, x]:+.1f}", ha="center", va="center", fontsize=7)
    fig.suptitle("增加输入后，高流量道路 MAE 的相对变化（%）。负值表示下降。")
    finish_figure(fig, FIGURE_DIR / "input_mae_rel_high_by_date.png")


def partition_contribution(cube, model_index, rate_index, ref_setting, cand_setting, days, buckets, groups) -> list[dict[str, object]]:
    rows = []
    for day in days:
        for bucket_name, bucket_label, hours in buckets:
            for group in groups:
                reference = sums(cube, model_index, rate_index, ref_setting, [day], hours, [group])
                candidate = sums(cube, model_index, rate_index, cand_setting, [day], hours, [group])
                reference_metric = metrics_from_sums(reference)
                candidate_metric = metrics_from_sums(candidate)
                rows.append(
                    {
                        "day": day,
                        "bucket": bucket_name,
                        "bucket_label": bucket_label,
                        "group": group,
                        "sum_abs_diff": candidate["sum_abs_error"] - reference["sum_abs_error"],
                        "U_diff": candidate["U"] - reference["U"],
                        "O_diff": candidate["O"] - reference["O"],
                        "sum_error_diff": candidate["sum_error"] - reference["sum_error"],
                        "mae_reference": reference_metric["mae"],
                        "mae_candidate": candidate_metric["mae"],
                        "mae_diff": candidate_metric["mae"] - reference_metric["mae"],
                        "mae_rel": defined_rel(candidate_metric["mae"], reference_metric["mae"]),
                        "wape_reference": reference_metric["wape"],
                        "wape_candidate": candidate_metric["wape"],
                        "wape_diff": defined_sub(candidate_metric["wape"], reference_metric["wape"]),
                        "bias_reference": reference_metric["bias"],
                        "bias_candidate": candidate_metric["bias"],
                        "gap_reference": reference_metric["underestimation_gap"],
                        "gap_candidate": candidate_metric["underestimation_gap"],
                        "n_elements": reference["n_elements"],
                    }
                )
    return rows


def gross_parts(values: list[float]) -> tuple[float, float, float]:
    negative = sum(value for value in values if value < 0)
    positive = sum(value for value in values if value > 0)
    return negative, positive, negative + positive


def plot_cancellation(cube, calendar) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
    x = np.arange(len(GROUP_NAMES))
    width = 0.25
    for axis, rate in zip(axes, ("p05", "p70")):
        for offset, model in enumerate(FOCUS_MODELS):
            heights = []
            for group in range(N_GROUP):
                reference = sums(cube, MODELS.index(model), RATES.index(rate), 0, range(N_DAY), range(N_HOUR), [group])
                candidate = sums(cube, MODELS.index(model), RATES.index(rate), 1, range(N_DAY), range(N_HOUR), [group])
                heights.append((candidate["sum_abs_error"] - reference["sum_abs_error"]) / 10000)
            axis.bar(x + (offset - 1) * width, heights, width=width, color=MODEL_COLOR[model], label=model_name(model))
        axis.axhline(0, color="#444444", linewidth=0.8)
        axis.set_xticks(x, [GROUP_LABEL[name] for name in GROUP_NAMES])
        axis.set_title(rate_name(rate))
        axis.set_ylabel("绝对误差总和变化（万辆）")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("增加输入后，各道路组绝对误差总和的变化。负值是误差减少。")
    finish_figure(fig, FIGURE_DIR / "input_abs_change_by_group.png")


def plot_tlast_uo(cube, calendar) -> None:
    model_index = MODELS.index("tlast")
    rate_index = RATES.index("p05")
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.4))
    specs = (
        ("道路组", [(name, range(N_DAY), range(N_HOUR), [index]) for index, name in enumerate(GROUP_NAMES)]),
        ("日期", [(item["date"][5:], [index], range(N_HOUR), range(N_GROUP)) for index, item in enumerate(calendar)]),
        ("时段", [(label_text, range(N_DAY), hours, range(N_GROUP)) for _name, label_text, hours in TIME_BUCKETS]),
    )
    for axis, (title, items) in zip(axes, specs):
        u_values = []
        o_values = []
        labels = []
        for name, days, hours, groups in items:
            reference = sums(cube, model_index, rate_index, 0, days, hours, groups)
            candidate = sums(cube, model_index, rate_index, 1, days, hours, groups)
            u_values.append((candidate["U"] - reference["U"]) / 10000)
            o_values.append((candidate["O"] - reference["O"]) / 10000)
            labels.append(GROUP_LABEL.get(name, name).replace(" 00:00–06:00", "").replace(" 07:00–09:00", "").replace(" 10:00–16:00", "").replace(" 17:00–19:00", ""))
        x = np.arange(len(labels))
        axis.bar(x - 0.18, u_values, width=0.36, label="ΔU 低估误差", color="#0072B2")
        axis.bar(x + 0.18, o_values, width=0.36, label="ΔO 高估误差", color="#D55E00")
        axis.axhline(0, color="#444444", linewidth=0.8)
        axis.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
        axis.set_title(title)
        axis.set_ylabel("变化（万辆）")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("TLAST 5%，增加输入后的 U 与 O 变化。负值表示该类误差减少。")
    finish_figure(fig, FIGURE_DIR / "tlast_p05_uo_change.png")


def direction_word(under_value: float, over_value: float, close: float) -> str:
    if abs(under_value - over_value) <= close:
        return "接近"
    return "低估" if under_value > over_value else "高估"


def mass_direction(metric: dict[str, float | None]) -> str:
    share = metric["under_abs_share"]
    if share is None:
        return "不可定义"
    return direction_word(share, 1.0 - share, 0.01)


def location_phrase(calendar, day, bucket_label, group: int) -> str:
    return f"{calendar[day]['date']}（{calendar[day]['weekday_zh']}）{bucket_label}、{GROUP_LABEL[GROUP_NAMES[group]]}道路"


def mae_change_phrase(row: dict[str, object]) -> str:
    if row["wape_reference"] is None or row["wape_candidate"] is None or row["wape_diff"] is None:
        wape_bit = "WAPE 不可定义"
    else:
        wape_bit = (
            f"WAPE 从 {pct(row['wape_reference'])} 变为 {pct(row['wape_candidate'])}"
            f"（{signed(row['wape_diff'], 2)} 个百分点）"
        )
    return (
        f"MAE 从 {num(row['mae_reference'])} 变为 {num(row['mae_candidate'])}"
        f"（{signed(row['mae_diff'])}，{signed_pct(row['mae_rel'])}），"
        f"{wape_bit}"
    )


def render_report(cube, calendar, validation, anomalies) -> str:
    lines: list[str] = []
    add = lines.append
    counts = validation["groups"]["counts"]
    add("# Inflow 日期 × 时段 × 道路流量组误差分析")
    add("")
    add("## 数据与定义")
    add("")
    add("本次只读取 3 秒相位四状态实验已经保存的测试预测，在 CPU 上聚合，没有重新训练。旧信号配时、3 秒相位单通道目录和 smoke 结果都没有进入统计。单通道目录里的 inflow 预测与这里的 Inflow→Inflow 重复，不另算一组。")
    add("")
    add("三种设置都只评价 inflow：")
    add("")
    add("- A，Inflow→Inflow：单状态输入、单状态输出。")
    add("- B，All→Inflow：四个状态一起输入，只输出 inflow。这是“增加输入”。")
    add("- C，All→All 的 inflow 通道：四个状态一起输入、一起输出，再取出 inflow。C 相对 B 是“联合输出”。C 相对 A 是输入和输出同时变化的总结果，不把它解释成输入融合的收益。")
    add("")
    add(f"模型是 STID、STGCN、DCRNN、Graph WaveNet、S2S-GCN-LSTM 和 TLAST。渗透率是 5%、10%、20%、30%、40%、50% 和 70%，seed 都是 42。库存里符合条件并完成校验的运行有 {validation['processed_runs']} 组，预期 {EXPECTED_RUNS} 组。")
    add("")
    add("预测和标签用 `test_predictions.npz` 的 `pred_raw`、`y_raw`。误差是 e = pred_raw − y_raw，负预测不截断，真实值为 0 的样本保留。高估是 e > 0，低估是 e < 0，相等是 e = 0。高估误差总量 O = sum(max(e, 0))，低估误差总量 U = sum(max(−e, 0))。净低估量记为 U − O，它等于 −sum(e)。平均偏差 bias = mean(e)。总量偏差率 = sum(e) / sum(y_raw)。MAPE_nonzero 只在真实值大于 0 的位置计算。WAPE、总量偏差率和相对变化在分母为 0 时记为不可定义，不填 0。")
    add("")
    add("时段和道路组的汇总都先把样本数、误差和、绝对误差和、平方误差和、绝对百分比误差和、真实值和、U、O 加起来，再计算 MAE、RMSE、MAPE 和 WAPE。没有把不同格子的 RMSE 或 WAPE 直接平均。")
    add("")
    add("## 日期、时间和道路组")
    add("")
    date_text = "、".join(f"{item['date']}（{item['weekday_zh']}）" for item in calendar)
    add(f"`split_manifest.json` 的测试日是 {date_text}，编码是 `python_weekday_monday_0`。`day_index` 因此是 2、3、4、5、6，分别对应这五天，而不是测试列表里的序号。五天的星期各不相同，所以按 `day_index` 分组不会把不同日期并到一起。这只是 2023 年 4 月 26 日至 30 日这五天，不能把星期五、星期六的高低说成普遍的周末规律。")
    add("")
    add("每个测试日有 277 个目标时间片。`target_slot = window_start_slot + 11`，第一片是时间片 11，覆盖 00:55–01:00。因此 00 时只有 1 个目标时间片；01–23 时各有 12 个时间片，覆盖完整的小时。五个测试日的这个覆盖完全一致。")
    add("")
    add("四个主要时段沿用已有定义，时间片 t 覆盖 `[t×5 分钟, (t+1)×5 分钟)`：")
    add("")
    add("- 夜间 00:00–06:00：时间片 11–71，共 61 片。")
    add("- 早高峰 07:00–09:00：时间片 84–107，共 24 片。")
    add("- 白天 10:00–16:00：时间片 120–191，共 72 片。")
    add("- 晚高峰 17:00–19:00：时间片 204–227，共 24 片。")
    add("")
    add("06、09、16 时和 19–23 时不在这四个时段里，仍保留在逐小时表中。四个时段加这些其余小时正好是 277 片。")
    add("")
    add(f"道路组用全部五个测试日、全部 1385 个目标时间片的 inflow 真实值，按每条道路求平均，再调用与 `error_distribution.py` 相同的 `flow_group`。均值小于等于 1e-6 的道路为 inactive；其余道路按三分位切成 low、medium、high。本次 inactive 的均值恰好全是 0，共 {counts['inactive']} 条。低组上界 {validation['groups']['low_max_mean']:.6f}，中组上界 {validation['groups']['mid_max_mean']:.6f}，低、中、高各 {counts['low']} 条。四组互斥，合计 {sum(counts.values())} 条。126 组运行的 inflow 标签数组哈希相同，所以组别没有按模型、渗透率、日期或小时重算。")
    add("")
    add("## 校验")
    add("")
    if anomalies:
        add("以下运行没有进入统计，表格里也没有用 0 填补：")
        add("")
        for item in anomalies:
            add(f"- {model_name(item['model'])} {item['penetration']} {SETTING_LABEL.get(item['setting'], item['setting'])}：{'；'.join(item['reasons'])}")
        add("")
    else:
        add("库存筛选和元数据复核后，没有缺失、重复、未完成或被排除的目标运行。")
        add("")
    identity = validation["identity"]
    published = validation["published"]
    existing = validation["existing_tables"]
    add(f"节点轴长度为 1406，`r_nodes.csv` 与 `node_mapping.csv` 的 `edge_id` 顺序一致，哈希 `{validation['node_order_sha256']}` 与 3 秒相位和旧数据的 `node_order_sha256` 相同。All→All 取通道 0。通道顺序仍是 inflow、outflow、accumulation_start、accumulation_end，重算的 inflow MAE 与 `test_metrics.json` 的分通道 MAE 最大绝对差是 {published['mae']:.3e}。")
    add("")
    add(f"逐小时、逐日、四个主要时段和四组道路按统计量加总后，与已有表的最大绝对差：逐日 MAE {existing['day_mae']:.3e}、逐日 bias {existing['day_bias']:.3e}；逐小时 MAE {existing['hour_mae']:.3e}、bias {existing['hour_bias']:.3e}；时段 MAE {existing['period_mae']:.3e}、bias {existing['period_bias']:.3e}；道路组 MAE {existing['spatial_mae']:.3e}、bias {existing['spatial_bias']:.3e}。直接对预测数组求和与分箱求和的绝对误差差最大是 {validation['direct_sums']['sum_abs_error']:.3e}。")
    add("")
    add(f"每个格子上都有 U + O = 绝对误差和，最大偏差 {identity['U_plus_O_minus_abs']:.3e}；O − U = 误差和，最大偏差 {identity['O_minus_U_minus_error']:.3e}；高估、低估、相等的数量之和等于元素数，最大偏差 {identity['direction_count']:.3e}。真实值没有负数。每个日期×小时×道路组的元素数等于该小时的时间片数乘以该组道路数，偏差 {validation['coverage_element_gap']:.3e}。")
    add("")
    add("这些校验通过。下面的数值都来自这次聚合。")
    add("")
    lines.extend(section_days(cube, calendar))
    lines.extend(section_direction(cube, calendar))
    lines.extend(section_input(cube, calendar))
    lines.extend(section_joint(cube, calendar))
    lines.extend(section_tlast(cube, calendar))
    lines.extend(section_cancellation(cube, calendar))
    lines.extend(section_rates(cube, calendar))
    lines.extend(section_limits())
    return "\n".join(lines).rstrip() + "\n"


def section_days(cube, calendar) -> list[str]:
    lines = ["## 1. 五个测试日的流量规模、零值比例、误差和偏差", ""]
    truth_rows = []
    for day, item in enumerate(calendar):
        values = sums(cube, 0, 0, 0, [day], range(N_HOUR), range(N_GROUP))
        metric = metrics_from_sums(values)
        truth_rows.append((item, values, metric))
        for model_index in range(len(MODELS)):
            for rate_index in range(len(RATES)):
                for setting_index in range(len(SETTING_IDS)):
                    other = sums(cube, model_index, rate_index, setting_index, [day], range(N_HOUR), range(N_GROUP))
                    if abs(other["sum_true"] - values["sum_true"]) > 1e-6 or other["n_zero_true"] != values["n_zero_true"]:
                        raise RuntimeError("true inflow differs across runs inside a test day")
    lines.append("真实 inflow 与模型无关。五日的规模如下。")
    lines.append("")
    lines.extend(
        md_table(
            ["日期", "星期", "真实值总和（万辆）", "每道路每片均值", "零值比例", "正值样本比例"],
            [
                [
                    item["date"],
                    item["weekday_zh"],
                    f"{values['sum_true'] / 10000:.2f}",
                    num(metric["mean_true"]),
                    pct(metric["zero_fraction"] * 100),
                    pct(values["n_positive_true"] / values["n_elements"] * 100),
                ]
                for item, values, metric in truth_rows
            ],
        )
    )
    lines.append("")
    ordered = sorted(truth_rows, key=lambda item: item[1]["sum_true"], reverse=True)
    zero_ordered = sorted(truth_rows, key=lambda item: item[2]["zero_fraction"])
    lines.append(
        f"真实值总和从高到低是 "
        + "、".join(f"{item['date']} {values['sum_true'] / 10000:.2f} 万辆" for item, values, _metric in ordered)
        + "。"
        f"零值比例从低到高是 "
        + "、".join(f"{item['date']} {pct(metric['zero_fraction'] * 100)}" for item, _values, metric in zero_ordered)
        + "。30 日是星期日，但它的流量和零值比例要和表中另外四天直接比较，不能单独当成周末效应。"
    )
    lines.append("")
    lines.append("下表是 Inflow→Inflow、全部道路合并后的 MAE 和 WAPE。MAE 的单位是辆/道路/5 分钟。")
    lines.append("")
    for rate in ("p05", "p70"):
        rate_index = RATES.index(rate)
        lines.append(f"渗透率 {rate_name(rate)}：")
        lines.append("")
        headers = ["模型"]
        for item, _values, _metric in truth_rows:
            headers.extend([f"{item['date'][5:]} MAE", f"{item['date'][5:]} WAPE"])
        rows = []
        for model_index, model in enumerate(MODELS):
            row = [model_name(model)]
            for day in range(N_DAY):
                metric = cell_metrics(cube, model_index, rate_index, 0, [day], range(N_HOUR), range(N_GROUP))
                row.extend([num(metric["mae"]), pct(metric["wape"])])
            rows.append(row)
        lines.extend(md_table(headers, rows))
        lines.append("")
    lines.extend(day_error_paragraph(cube, "p05", 0))
    lines.append("")
    lines.extend(day_error_paragraph(cube, "p70", 0))
    lines.append("")
    lines.append(other_setting_day_sentence(cube))
    lines.append("")
    lines.append("28 日和 29 日的真实值总和更高，所以 MAE 变大里有流量规模的因素。只有上面点名的模型，这两天的 WAPE 或 MAPE 也同时高于另外三天。不能把“MAE 更高”直接写成“相对误差更高”。这五天也不能推出星期五或星期六普遍更难预测。")
    lines.append("")
    return lines


def pooled_days(cube, model_index: int, rate_index: int, setting: int, days: list[int]) -> dict[str, float | None]:
    return metrics_from_sums(sums(cube, model_index, rate_index, setting, days, range(N_HOUR), range(N_GROUP)))


def day_exceeds_calm(metrics: list[dict[str, float | None]], key: str) -> tuple[bool, bool]:
    calm_max = max(metrics[index][key] for index in (0, 1, 4))
    return metrics[2][key] > calm_max, metrics[3][key] > calm_max


def day_error_paragraph(cube, rate: str, setting: int) -> list[str]:
    rate_index = RATES.index(rate)
    fragments = []
    named = {key: [] for key in ("mae_yes", "mae_no", "wape_yes", "wape_no", "mape_yes", "mape_no")}
    more_negative = 0
    for model_index, model in enumerate(MODELS):
        metrics = [cell_metrics(cube, model_index, rate_index, setting, [day], range(N_HOUR), range(N_GROUP)) for day in range(N_DAY)]
        busy = pooled_days(cube, model_index, rate_index, setting, [2, 3])
        calm = pooled_days(cube, model_index, rate_index, setting, [0, 1, 4])
        flags = {
            "mae": day_exceeds_calm(metrics, "mae"),
            "wape": day_exceeds_calm(metrics, "wape"),
            "mape": day_exceeds_calm(metrics, "mape_nonzero"),
        }
        for key, pair in flags.items():
            named[f"{key}_{'yes' if all(pair) else 'no'}"].append(model_name(model))
        more_negative += int(busy["bias"] < calm["bias"])
        fragments.append(
            f"{model_name(model)}：28 日和 29 日合并后 MAE {num(busy['mae'])}、WAPE {pct(busy['wape'])}、MAPE {pct(busy['mape_nonzero'])}、bias {signed(busy['bias'])}；"
            f"26、27、30 日合并后 MAE {num(calm['mae'])}、WAPE {pct(calm['wape'])}、MAPE {pct(calm['mape_nonzero'])}、bias {signed(calm['bias'])}。"
            f"28 日、29 日相对另外三天最大值：MAE {'高于' if flags['mae'][0] else '不高于'}、{'高于' if flags['mae'][1] else '不高于'}；"
            f"WAPE {'高于' if flags['wape'][0] else '不高于'}、{'高于' if flags['wape'][1] else '不高于'}；"
            f"MAPE {'高于' if flags['mape'][0] else '不高于'}、{'高于' if flags['mape'][1] else '不高于'}。"
        )
    setting_label = SETTING_LABEL[SETTING_IDS[setting]]

    def joined(key: str) -> str:
        return "、".join(named[key]) if named[key] else "没有"

    return [
        f"{rate_name(rate)} 的 {setting_label}：28 日和 29 日的 MAE 都高于另外三天最大值的模型是 {joined('mae_yes')}；不是的是 {joined('mae_no')}。",
        f"WAPE 也两天都更高的模型是 {joined('wape_yes')}；不是的是 {joined('wape_no')}。",
        f"MAPE_nonzero 也两天都更高的模型是 {joined('mape_yes')}；不是的是 {joined('mape_no')}。",
        f"{more_negative} 个模型在 28 日与 29 日合并后的 bias 比另外三天合并后更负。合并用的是误差和与样本数，不是把每天的 WAPE 再平均。",
        *fragments,
    ]


def other_setting_day_sentence(cube) -> str:
    parts = []
    for setting in (1, 2):
        for rate in ("p05", "p70"):
            rate_index = RATES.index(rate)
            mae_both = wape_both = 0
            for model_index in range(len(MODELS)):
                metrics = [cell_metrics(cube, model_index, rate_index, setting, [day], range(N_HOUR), range(N_GROUP)) for day in range(N_DAY)]
                mae_both += int(all(day_exceeds_calm(metrics, "mae")))
                wape_both += int(all(day_exceeds_calm(metrics, "wape")))
            parts.append(f"{SETTING_LABEL[SETTING_IDS[setting]]} {rate_name(rate)}：MAE {mae_both}/6，WAPE {wape_both}/6")
    return "All→Inflow 和 All→All 是否也是 28、29 两日都高于另外三天：" + "；".join(parts) + "。设置之间的差别放在后面。"


def section_direction(cube, calendar) -> list[str]:
    lines = ["## 2. 低、中、高流量道路在各时段更倾向高估还是低估", ""]
    lines.append("这里把“样本数量上的倾向”和“绝对误差总量上的倾向”分开。样本倾向比较低估样本数和高估样本数，相等样本单独计入，但不拿来决定哪一边更多。误差总量倾向比较 U 和 O。五日先按统计量加总，再看每一天是否换向。下表是 Inflow→Inflow、六个模型里的最小到最大范围。")
    lines.append("")
    for rate in ("p05", "p70"):
        lines.append(f"渗透率 {rate_name(rate)}，五日合计：")
        lines.append("")
        rows = []
        for group in (1, 2, 3):
            for period, period_label, hours in PERIODS:
                sample_under = []
                sample_over = []
                mass_under = []
                biases = []
                for model_index in range(len(MODELS)):
                    metric = cell_metrics(cube, model_index, RATES.index(rate), 0, range(N_DAY), hours, [group])
                    sample_under.append(metric["under_fraction"])
                    sample_over.append(metric["over_fraction"])
                    mass_under.append(metric["under_abs_share"])
                    biases.append(metric["bias"])
                rows.append(
                    [
                        GROUP_LABEL[GROUP_NAMES[group]],
                        period_label.split()[0],
                        f"{min(sample_under) * 100:.1f}–{max(sample_under) * 100:.1f}",
                        f"{min(sample_over) * 100:.1f}–{max(sample_over) * 100:.1f}",
                        (
                            "不可定义"
                            if any(value is None for value in mass_under)
                            else f"{min(mass_under) * 100:.1f}–{max(mass_under) * 100:.1f}"
                        ),
                        f"{min(biases):+.3f}–{max(biases):+.3f}",
                    ]
                )
        lines.extend(md_table(["道路组", "时段", "低估样本%", "高估样本%", "U/(U+O)%", "bias"], rows))
        lines.append("")
    lines.extend(direction_prose(cube, calendar, "p05"))
    lines.append("")
    lines.extend(direction_prose(cube, calendar, "p70"))
    lines.append("")
    lines.append("零流量道路的真实值全是 0，WAPE 和总量偏差率不可定义。那里的正误差来自正预测，负误差来自负预测。它不改变低、中、高三组的上述方向，完整数字在逐小时和逐时段表里。")
    lines.append("")
    lines.append("图 `figures/inflow_time_groups/direction_sample_vs_mass.png` 给出 STID、Graph WaveNet 和 TLAST 在两个渗透率上的样本低估比例和绝对误差低估占比。")
    lines.append("")
    return lines


def direction_prose(cube, calendar, rate: str) -> list[str]:
    rate_index = RATES.index(rate)
    lines = [
        f"{rate_name(rate)}，Inflow→Inflow，先把五个测试日加总。样本方向比较低估样本数和高估样本数；误差总量方向比较 U 和 O。"
    ]
    flips = 0
    for group in (1, 2, 3):
        for period, period_label, hours in PERIODS:
            buckets = {
                "sample": {"低估": [], "高估": [], "接近": []},
                "mass": {"低估": [], "高估": [], "接近": [], "不可定义": []},
            }
            under_fracs = []
            shares = []
            biases = []
            for model_index, model in enumerate(MODELS):
                metric = cell_metrics(cube, model_index, rate_index, 0, range(N_DAY), hours, [group])
                sample_word = direction_word(metric["under_fraction"], metric["over_fraction"], 0.01)
                mass_word = mass_direction(metric)
                buckets["sample"][sample_word].append(model_name(model))
                buckets["mass"][mass_word].append(model_name(model))
                under_fracs.append(metric["under_fraction"])
                if metric["under_abs_share"] is not None:
                    shares.append(metric["under_abs_share"])
                biases.append(metric["bias"])
                for day, _item in enumerate(calendar):
                    daily = cell_metrics(cube, model_index, rate_index, 0, [day], hours, [group])
                    if (
                        direction_word(daily["under_fraction"], daily["over_fraction"], 0.01) != sample_word
                        or mass_direction(daily) != mass_word
                    ):
                        flips += 1
            share_text = "不可定义" if not shares else f"{min(shares) * 100:.1f}–{max(shares) * 100:.1f}%"
            lines.append(
                f"{GROUP_LABEL[GROUP_NAMES[group]]}、{period_label.split()[0]}："
                f"样本低估 {'、'.join(buckets['sample']['低估']) or '没有'}；"
                f"样本高估 {'、'.join(buckets['sample']['高估']) or '没有'}；"
                f"样本接近 {'、'.join(buckets['sample']['接近']) or '没有'}。"
                f"误差总量低估 {'、'.join(buckets['mass']['低估']) or '没有'}；"
                f"误差总量高估 {'、'.join(buckets['mass']['高估']) or '没有'}；"
                f"误差总量接近 {'、'.join(buckets['mass']['接近']) or '没有'}。"
                f"低估样本比例 {min(under_fracs) * 100:.1f}–{max(under_fracs) * 100:.1f}%，"
                f"U 占比 {share_text}，bias {min(biases):+.3f} 至 {max(biases):+.3f}。"
            )
    lines.append(
        f"再按单个日期拆开，有 {flips} 个模型×日期×时段×道路组与该模型自己的五日合计标签不同。"
        "接近 50% 的格子换一天就会换标签。五日合计是这五天加在一起的倾向，不是每一天都相同。逐日数字在 `tables/inflow_time_period_group.csv`。"
    )
    return lines


def network_compare(cube, model_index: int, rate_index: int, ref: int, cand: int) -> dict[str, object]:
    reference_sums = sums(cube, model_index, rate_index, ref, range(N_DAY), range(N_HOUR), range(N_GROUP))
    candidate_sums = sums(cube, model_index, rate_index, cand, range(N_DAY), range(N_HOUR), range(N_GROUP))
    reference = metrics_from_sums(reference_sums)
    candidate = metrics_from_sums(candidate_sums)
    output = compare_metrics(reference, candidate)
    output["sum_abs_diff"] = candidate_sums["sum_abs_error"] - reference_sums["sum_abs_error"]
    output["U_diff"] = candidate_sums["U"] - reference_sums["U"]
    output["O_diff"] = candidate_sums["O"] - reference_sums["O"]
    output["sum_error_diff"] = candidate_sums["sum_error"] - reference_sums["sum_error"]
    output["U_reference"] = reference_sums["U"]
    output["O_reference"] = reference_sums["O"]
    output["U_candidate"] = candidate_sums["U"]
    output["O_candidate"] = candidate_sums["O"]
    return output


def section_input(cube, calendar) -> list[str]:
    lines = ["## 3. 增加四状态输入的收益和退化", ""]
    lines.append("增加输入指 B 减 A。MAE 差值为负表示 All→Inflow 的绝对误差更小。全网结果先列出来，再按绝对误差总和的变化看它落在哪些日期、时段和道路组。绝对误差总和的变化可以相加，等于全网变化；相对 MAE 不行。")
    lines.append("")
    rows = []
    for rate in ("p05", "p70"):
        for model_index, model in enumerate(MODELS):
            compared = network_compare(cube, model_index, RATES.index(rate), 0, 1)
            rows.append(
                [
                    model_name(model),
                    rate_name(rate),
                    num(compared["mae_reference"]),
                    num(compared["mae_candidate"]),
                    signed(compared["mae_diff"]),
                    signed_pct(compared["mae_rel_pct"]),
                    signed(compared["wape_diff"], 2),
                    wan(compared["sum_abs_diff"]),
                ]
            )
    lines.extend(md_table(["模型", "渗透率", "A MAE", "B MAE", "MAE 差", "MAE 相对变化", "WAPE 差（百分点）", "绝对误差总和变化"], rows))
    lines.append("")
    for paragraph in input_location_prose(cube, calendar, "p05"):
        lines.append(paragraph)
        lines.append("")
    for paragraph in input_location_prose(cube, calendar, "p70"):
        lines.append(paragraph)
        lines.append("")
    lines.append("")
    lines.append("图 `figures/inflow_time_groups/input_mae_rel_group_period.png` 是五日合计的道路组×时段相对 MAE，`input_mae_rel_high_by_date.png` 是高流量道路按日期拆开的相对 MAE，`input_abs_change_by_group.png` 是绝对误差总和按道路组的变化。色标限制在 ±8%，超出的格子仍标出实际数字。")
    lines.append("")
    return lines


def input_location_prose(cube, calendar, rate: str) -> list[str]:
    rate_index = RATES.index(rate)
    sentences = [f"{rate_name(rate)}："]
    for model_index, model in enumerate(MODELS):
        cells = partition_contribution(cube, model_index, rate_index, 0, 1, range(N_DAY), TIME_BUCKETS, range(N_GROUP))
        ordered = sorted(cells, key=lambda item: item["sum_abs_diff"])
        best = ordered[0]
        worst = ordered[-1]
        by_group = []
        for group in range(N_GROUP):
            value = sum(item["sum_abs_diff"] for item in cells if item["group"] == group)
            by_group.append(f"{GROUP_LABEL[GROUP_NAMES[group]]} {wan(value)}")
        by_bucket = []
        for bucket, bucket_label, _hours in TIME_BUCKETS:
            value = sum(item["sum_abs_diff"] for item in cells if item["bucket"] == bucket)
            by_bucket.append(f"{bucket_label.split('（')[0]} {wan(value)}")
        sentences.append(
            f"{model_name(model)} 的绝对误差总和变化按道路组是 {'、'.join(by_group)}；按时间是 {'、'.join(by_bucket)}。"
            f"减少最多的格子是 {location_phrase(calendar, best['day'], best['bucket_label'], best['group'])}，{mae_change_phrase(best)}，绝对误差总和 {wan(best['sum_abs_diff'])}。"
            f"增加最多的格子是 {location_phrase(calendar, worst['day'], worst['bucket_label'], worst['group'])}，{mae_change_phrase(worst)}，绝对误差总和 {wan(worst['sum_abs_diff'])}。"
        )
    return sentences


def section_joint(cube, calendar) -> list[str]:
    lines = ["## 4. 联合输出是否保留、增强或抵消增加输入的变化", ""]
    lines.append("联合输出指 C 减 B。它和增加输入使用同一批道路和时间。二者的绝对误差总和变化可以直接相加，等于 C 减 A。下面的“抵消”只指数值上的对冲，不是模型内部机制已经查明。")
    lines.append("")
    rows = []
    sentences = []
    for rate in ("p05", "p70"):
        rate_index = RATES.index(rate)
        for model_index, model in enumerate(MODELS):
            added = network_compare(cube, model_index, rate_index, 0, 1)
            joint = network_compare(cube, model_index, rate_index, 1, 2)
            total = network_compare(cube, model_index, rate_index, 0, 2)
            rows.append(
                [
                    model_name(model),
                    rate_name(rate),
                    num(added["mae_reference"]),
                    num(added["mae_candidate"]),
                    num(joint["mae_candidate"]),
                    signed(added["mae_diff"]),
                    signed(joint["mae_diff"]),
                    signed(total["mae_diff"]),
                    wan(added["sum_abs_diff"]),
                    wan(joint["sum_abs_diff"]),
                    wan(total["sum_abs_diff"]),
                ]
            )
            cells = partition_contribution(cube, model_index, rate_index, 0, 1, range(N_DAY), TIME_BUCKETS, range(N_GROUP))
            joint_cells = partition_contribution(cube, model_index, rate_index, 1, 2, range(N_DAY), TIME_BUCKETS, range(N_GROUP))
            gain = [(left, right) for left, right in zip(cells, joint_cells) if left["sum_abs_diff"] < 0]
            loss = [(left, right) for left, right in zip(cells, joint_cells) if left["sum_abs_diff"] > 0]
            gain_input = sum(item[0]["sum_abs_diff"] for item in gain)
            gain_joint = sum(item[1]["sum_abs_diff"] for item in gain)
            loss_input = sum(item[0]["sum_abs_diff"] for item in loss)
            loss_joint = sum(item[1]["sum_abs_diff"] for item in loss)
            sentences.append(
                f"{model_name(model)} {rate_name(rate)}：增加输入改变绝对误差 {wan(added['sum_abs_diff'])}，联合输出再改变 {wan(joint['sum_abs_diff'])}，总变化 {wan(total['sum_abs_diff'])}。"
                f"在增加输入已经降低误差的格子里，输入侧合计 {wan(gain_input)}，这些格子上的联合输出合计 {wan(gain_joint)}。"
                f"在增加输入已经升高误差的格子里，输入侧合计 {wan(loss_input)}，联合输出合计 {wan(loss_joint)}。"
                + joint_words(added["sum_abs_diff"], joint["sum_abs_diff"])
            )
    lines.extend(
        md_table(
            ["模型", "渗透率", "A", "B", "C", "B−A", "C−B", "C−A", "B−A 绝对误差", "C−B 绝对误差", "C−A 绝对误差"],
            rows,
        )
    )
    lines.append("")
    for sentence in sentences:
        lines.append(sentence)
        lines.append("")
    lines.append("C−A 这一列只描述最终预测离 Inflow→Inflow 有多远。例如联合输出把 B 的改善抵消掉时，C 可以接近 A，但这不是“增加输入没有作用”，而是输出端的变化把输入端的变化冲掉了。")
    lines.append("")
    return lines


def joint_words(input_delta: float, joint_delta: float) -> str:
    if abs(input_delta) < 1000 and abs(joint_delta) < 1000:
        return "两项变化相对全网绝对误差都很小。"
    if input_delta < 0 and joint_delta > 0:
        ratio = joint_delta / -input_delta * 100
        if ratio < 25:
            return f"联合输出只抵消了约 {ratio:.0f}% 的输入侧绝对误差降幅，输入侧的大部分收益还在。"
        if ratio < 100:
            return f"联合输出抵消了输入侧绝对误差降幅的约 {ratio:.0f}%，收益只保留了一部分。"
        return f"联合输出的绝对误差增幅是输入侧降幅的约 {ratio:.0f}%，输入侧收益被抵消并转成净变差。"
    if input_delta < 0 and joint_delta < 0:
        return "全网绝对误差继续下降。这只说明两个全网变化同号；原先已经变好的格子是否继续变好，以上一句的格子合计为准。"
    if input_delta > 0 and joint_delta < 0:
        ratio = -joint_delta / input_delta * 100
        return f"增加输入本身使绝对误差上升，联合输出再把它降下来，降幅约为该升幅的 {ratio:.0f}%。"
    if input_delta > 0 and joint_delta > 0:
        return "联合输出和增加输入同向，绝对误差继续上升。"
    return "两项变化的方向不足以归入上面几类。"


def section_tlast(cube, calendar) -> list[str]:
    lines = ["## 5. TLAST 在 5% 渗透率增加输入后，净低估为什么减轻", ""]
    model_index = MODELS.index("tlast")
    rate_index = RATES.index("p05")
    reference = sums(cube, model_index, rate_index, 0, range(N_DAY), range(N_HOUR), range(N_GROUP))
    candidate = sums(cube, model_index, rate_index, 1, range(N_DAY), range(N_HOUR), range(N_GROUP))
    reference_metric = metrics_from_sums(reference)
    candidate_metric = metrics_from_sums(candidate)
    gap_ref = reference_metric["underestimation_gap"]
    gap_cand = candidate_metric["underestimation_gap"]
    gap_drop = gap_ref - gap_cand
    u_diff = candidate["U"] - reference["U"]
    o_diff = candidate["O"] - reference["O"]
    lines.append(
        f"全网、全部道路和时间：Inflow→Inflow 的 bias 是 {signed(reference_metric['bias'])}，All→Inflow 是 {signed(candidate_metric['bias'])}。"
        f"净低估量 U−O 从 {wan(gap_ref)} 变为 {wan(gap_cand)}，减少 {wan(gap_drop)}，相当于原来净低估量的 {gap_drop / gap_ref * 100:.2f}%。"
        f"同一变化里，U 从 {wan(reference['U'])} 变为 {wan(candidate['U'])}（{signed_change(u_diff)}，相对 {u_diff / reference['U'] * 100:+.2f}%），"
        f"O 从 {wan(reference['O'])} 变为 {wan(candidate['O'])}（{signed_change(o_diff)}，相对 {o_diff / reference['O'] * 100:+.2f}%）。"
    )
    lines.append("")
    lines.append(
        f"因为净低估量 = U − O，它的减少量 = −ΔU + ΔO = {wan(-u_diff)} + {wan(o_diff)} = {wan(gap_drop)}。"
        f"U 下降贡献 {wan(-u_diff)}，O 上升贡献 {wan(o_diff)}。"
        f"所以净低估减少 {gap_drop / gap_ref * 100:.2f}% 不等于低估误差总量减少 {gap_drop / gap_ref * 100:.2f}%；"
        f"U 实际只减少了 {-u_diff / reference['U'] * 100:.2f}%。"
        f"绝对平均偏差从 {num(reference_metric['abs_bias'])} 变为 {num(candidate_metric['abs_bias'])}，绝对净误差 |sum(e)| 从 {wan(reference_metric['abs_net_error'])} 变为 {wan(candidate_metric['abs_net_error'])}。"
        "bias 仍然为负，全网没有从净低估变成净高估。"
    )
    lines.append("")
    lines.append("按道路组、日期和时段拆开的 ΔU、ΔO 如下。负值表示该类误差减少。")
    lines.append("")
    rows = []
    for group in range(N_GROUP):
        left = sums(cube, model_index, rate_index, 0, range(N_DAY), range(N_HOUR), [group])
        right = sums(cube, model_index, rate_index, 1, range(N_DAY), range(N_HOUR), [group])
        rows.append([GROUP_LABEL[GROUP_NAMES[group]], wan(right["U"] - left["U"]), wan(right["O"] - left["O"]), wan((left["U"] - left["O"]) - (right["U"] - right["O"]))])
    lines.extend(md_table(["道路组", "ΔU", "ΔO", "净低估减少"], rows))
    lines.append("")
    rows = []
    for bucket, bucket_label, hours in TIME_BUCKETS:
        left = sums(cube, model_index, rate_index, 0, range(N_DAY), hours, range(N_GROUP))
        right = sums(cube, model_index, rate_index, 1, range(N_DAY), hours, range(N_GROUP))
        rows.append([bucket_label.split("（")[0], wan(right["U"] - left["U"]), wan(right["O"] - left["O"]), wan((left["U"] - left["O"]) - (right["U"] - right["O"]))])
    lines.extend(md_table(["时间", "ΔU", "ΔO", "净低估减少"], rows))
    lines.append("")
    o_up = []
    o_down = []
    for bucket, bucket_label, hours in TIME_BUCKETS:
        left = sums(cube, model_index, rate_index, 0, range(N_DAY), hours, range(N_GROUP))
        right = sums(cube, model_index, rate_index, 1, range(N_DAY), hours, range(N_GROUP))
        (o_up if right["O"] - left["O"] > 0 else o_down).append(bucket_label.split("（")[0])
    lines.append(
        f"O 上升的时段是 {'、'.join(o_up) or '没有'}；O 下降的时段是 {'、'.join(o_down) or '没有'}。"
        "净低估如果在 O 下降的时段也减轻，那是 U 下降得更多，不是靠高估误差增加。"
    )
    lines.append("")
    rows = []
    for day, item in enumerate(calendar):
        left = sums(cube, model_index, rate_index, 0, [day], range(N_HOUR), range(N_GROUP))
        right = sums(cube, model_index, rate_index, 1, [day], range(N_HOUR), range(N_GROUP))
        rows.append([item["date"], wan(right["U"] - left["U"]), wan(right["O"] - left["O"]), wan((left["U"] - left["O"]) - (right["U"] - right["O"]))])
    lines.extend(md_table(["日期", "ΔU", "ΔO", "净低估减少"], rows))
    lines.append("")
    cells = partition_contribution(cube, model_index, rate_index, 0, 1, range(N_DAY), TIME_BUCKETS, range(4))
    u_down = sorted(cells, key=lambda item: item["U_diff"])[:3]
    o_up = sorted(cells, key=lambda item: item["O_diff"], reverse=True)[:3]
    lines.append("U 减少最多的三个日期×时段×道路组：" + "；".join(f"{location_phrase(calendar, item['day'], item['bucket_label'], item['group'])} ΔU {wan(item['U_diff'])}" for item in u_down) + "。")
    lines.append("")
    lines.append("O 增加最多的三个格子：" + "；".join(f"{location_phrase(calendar, item['day'], item['bucket_label'], item['group'])} ΔO {wan(item['O_diff'])}" for item in o_up) + "。")
    lines.append("")
    hour_rows = []
    for hour in range(N_HOUR):
        left = sums(cube, model_index, rate_index, 0, range(N_DAY), [hour], range(N_GROUP))
        right = sums(cube, model_index, rate_index, 1, range(N_DAY), [hour], range(N_GROUP))
        hour_rows.append((hour, right["U"] - left["U"], right["O"] - left["O"]))
    u_hour = min(hour_rows, key=lambda item: item[1])
    o_hour = max(hour_rows, key=lambda item: item[2])
    lines.append(
        f"按小时合计五个测试日和全部道路，U 下降最多的是 {u_hour[0]:02d} 时（ΔU {wan(u_hour[1])}），O 上升最多的是 {o_hour[0]:02d} 时（ΔO {wan(o_hour[2])}）。"
        "小时 0 只有一个时间片，不拿它的单片数值和完整小时比规模。"
    )
    lines.append("")
    lines.append("这些是测试预测的分解，不是训练过程的证据。本次没有读取梯度、注意力或隐藏状态，不能判断 U 下降和 O 上升来自哪一个模块。")
    lines.append("")
    lines.append("图 `figures/inflow_time_groups/tlast_p05_uo_change.png` 对应上面三张表。")
    lines.append("")
    p70 = network_compare(cube, model_index, RATES.index("p70"), 0, 1)
    lines.append(
        f"作为对照，TLAST 在 70% 增加输入后，bias 从 {signed(p70['bias_reference'])} 变为 {signed(p70['bias_candidate'])}，"
        f"净低估量 U−O 从 {wan(p70['underestimation_gap_reference'])} 变为 {wan(p70['underestimation_gap_candidate'])}，"
        f"U {signed_change(p70['U_diff'])}，O {signed_change(p70['O_diff'])}，"
        f"MAE 从 {num(p70['mae_reference'])} 变为 {num(p70['mae_candidate'])}（{signed(p70['mae_diff'])}，{signed_pct(p70['mae_rel_pct'])}）。"
        "5% 上的净低估减轻不能直接搬到 70%。第 7 节把两个渗透率放在一起。"
    )
    lines.append("")
    return lines


def signed_change(value: float) -> str:
    if value < 0:
        return f"减少 {wan(-value)}"
    if value > 0:
        return f"增加 {wan(value)}"
    return "没有变化"


def section_cancellation(cube, calendar) -> list[str]:
    lines = ["## 6. 全网接近持平时，内部是否相互抵消", ""]
    lines.append("一个模型的全网 MAE 几乎不变，仍可能是一部分日期、时段或道路组变好、另一部分变差。下面用绝对误差总和而不是 MAE 百分比来核对，因为只有绝对误差总和能加回全网。完整划分是五个日期 ×（四个主要时段加其余时段）× 四个道路组。负的部分是改善，正的部分是退化，两者之和是全网变化。")
    lines.append("")
    rows = []
    notes = []
    for rate in ("p05", "p70"):
        for model_index, model in enumerate(MODELS):
            for comparison, ref, cand, title in (
                ("增加输入", 0, 1, "B−A"),
                ("联合输出", 1, 2, "C−B"),
                ("总变化", 0, 2, "C−A"),
            ):
                cells = partition_contribution(cube, model_index, RATES.index(rate), ref, cand, range(N_DAY), TIME_BUCKETS, range(N_GROUP))
                negative, positive, net = gross_parts([item["sum_abs_diff"] for item in cells])
                network = network_compare(cube, model_index, RATES.index(rate), ref, cand)
                if abs(float(network["mae_rel_pct"] or 0)) > 1 and comparison != "增加输入":
                    continue
                rows.append(
                    [
                        model_name(model),
                        rate_name(rate),
                        title,
                        signed_pct(network["mae_rel_pct"]),
                        wan(negative),
                        wan(positive),
                        wan(net),
                    ]
                )
                opposing = min(-negative, positive)
                if (
                    comparison in {"增加输入", "总变化"}
                    and abs(float(network["mae_rel_pct"] or 0)) <= 1
                    and opposing >= max(abs(net), 1.0)
                    and opposing >= 5000
                ):
                    best = min(cells, key=lambda item: item["sum_abs_diff"])
                    worst = max(cells, key=lambda item: item["sum_abs_diff"])
                    group_parts = []
                    for group in range(N_GROUP):
                        value = sum(item["sum_abs_diff"] for item in cells if item["group"] == group)
                        group_parts.append(f"{GROUP_LABEL[GROUP_NAMES[group]]} {wan(value)}")
                    date_parts = []
                    for day, item in enumerate(calendar):
                        value = sum(cell["sum_abs_diff"] for cell in cells if cell["day"] == day)
                        date_parts.append(f"{item['date'][5:]} {wan(value)}")
                    notes.append(
                        f"{model_name(model)} {rate_name(rate)} {comparison}的全网 MAE 相对变化是 {signed_pct(network['mae_rel_pct'])}。"
                        f"细格子上的改善合计 {wan(negative)}，退化合计 {wan(positive)}，相抵后剩 {wan(net)}。"
                        f"按道路组：{'、'.join(group_parts)}。按日期：{'、'.join(date_parts)}。"
                        f"改善最大的格子是 {location_phrase(calendar, best['day'], best['bucket_label'], best['group'])}，{mae_change_phrase(best)}。"
                        f"退化最大的格子是 {location_phrase(calendar, worst['day'], worst['bucket_label'], worst['group'])}，{mae_change_phrase(worst)}。"
                    )
    lines.extend(md_table(["模型", "渗透率", "比较", "全网 MAE 相对变化", "细格改善", "细格退化", "相抵后"], rows))
    lines.append("")
    for note in notes:
        lines.append(note)
        lines.append("")
    lines.append("上表保留全部增加输入，以及全网 MAE 相对变化不超过 1% 的联合输出和总变化。文字只展开改善和退化都不少于相抵后结果、且各自至少 5000 辆的增加输入或总变化。零流量道路上很大的相对 MAE 往往来自接近 0 的基准，规模以绝对误差总和为准。改善和退化同时存在，是这些预测加总出来的结果，不说明两组道路共享同一个尚未观察到的原因。")
    lines.append("")
    return lines


def section_rates(cube, calendar) -> str:
    lines = ["## 7. 低渗透率和高渗透率是否不同", ""]
    lines.append("前几节的 5% 和 70% 已经分开列出。这里补上七个渗透率的全网符号，用来看两个端点是不是特例。")
    lines.append("")
    headers = ["模型"] + [rate_name(rate) for rate in RATES]
    input_rows = []
    joint_rows = []
    for model_index, model in enumerate(MODELS):
        input_row = [model_name(model)]
        joint_row = [model_name(model)]
        for rate_index, _rate in enumerate(RATES):
            added = network_compare(cube, model_index, rate_index, 0, 1)
            joint = network_compare(cube, model_index, rate_index, 1, 2)
            input_row.append(signed_pct(added["mae_rel_pct"], 2))
            joint_row.append(signed_pct(joint["mae_rel_pct"], 2))
        input_rows.append(input_row)
        joint_rows.append(joint_row)
    lines.append("增加输入的全网 MAE 相对变化：")
    lines.append("")
    lines.extend(md_table(headers, input_rows))
    lines.append("")
    lines.append("联合输出的全网 MAE 相对变化：")
    lines.append("")
    lines.extend(md_table(headers, joint_rows))
    lines.append("")
    lines.append(rate_direction_sentence(cube))
    lines.append("")
    lines.append("低渗透率的绝对误差更大，所以同样的相对变化对应更大的车辆数。第 3 节里 5% 和 70% 的道路组、时段贡献不能按百分比直接对照规模。TLAST 在 5% 增加输入会降低 MAE 并减轻净低估，在 70% 则升高 MAE；这个符号差别在上表里是端点现象，中间渗透率需要逐列看，不能概括成“渗透率越高，增加输入越差”的一条直线。")
    lines.append("")
    lines.append("5% 和 70% 的共同点是第 1 节里的 MAE：两个渗透率上，28 日和 29 日的 MAE 都高于另外三天的最大值。相对误差不是共同点。第 1 节按模型列出了 WAPE 和 MAPE 是否也更高；70% 的 WAPE 名单如果是“没有”，就不能把忙日较高的 MAE 说成较高的相对误差。高渗透率把五天的 MAE 都压低了，但没有抹平这五天之间的 MAE 高低。TLAST 的输入变化从 5% 的负号到 70% 的正号，中间渗透率并不是每一格都单调，不能概括成一条直线。")
    lines.append("")
    return lines


def rate_direction_sentence(cube) -> str:
    fragments = []
    for rate in ("p05", "p70"):
        metric = cell_metrics(cube, MODELS.index("tlast"), RATES.index(rate), 0, range(N_DAY), PERIODS[1][2], [3])
        low = cell_metrics(cube, MODELS.index("tlast"), RATES.index(rate), 0, range(N_DAY), PERIODS[0][2], [1])
        fragments.append(
            f"TLAST {rate_name(rate)} Inflow→Inflow，高流量早高峰的低估样本比例是 {pct(metric['under_fraction'] * 100, 1)}、U 占比 {pct(metric['under_abs_share'] * 100, 1)}；"
            f"低流量夜间的低估样本比例是 {pct(low['under_fraction'] * 100, 1)}、U 占比 {pct(low['under_abs_share'] * 100, 1)}"
        )
    return "方向上的低、高渗透率差别用 TLAST 作例子：" + "。".join(fragments) + "。六个模型的范围已在第 2 节按两个渗透率分别给出，不能把 5% 的样本方向直接说成 70% 也一样。"


def section_limits() -> list[str]:
    return [
        "## 数据直接支持的范围",
        "",
        "上面每一项都注明了模型、渗透率、设置、日期、时段和道路组，并给出变化前后的数值。支持这些句子的是测试预测的加总。它们不包含统计显著性检验，也没有第二个随机种子。",
        "",
        "不能从这五天推出的解释包括：星期五或星期六普遍更难预测；增加输入在所有渗透率上都会减轻净低估；全网 MAE 接近说明各个道路组和时段都没有变化。联合输出对存量通道的影响不在本次 inflow 表内。",
        "",
        "## 输出文件",
        "",
        "- `tables/inflow_road_groups.csv`：固定的 1406 条道路分组。",
        "- `tables/inflow_time_hour_group.csv`：日期 × 小时 × 道路组。",
        "- `tables/inflow_time_period_group.csv`：日期 × 四个主要时段 × 道路组。",
        "- `tables/inflow_time_day_group.csv`：日期 × 道路组。",
        "- `tables/inflow_time_setting_deltas.csv`：网络、日期、时段和小时上的 B−A、C−B、C−A。",
        "- `figures/inflow_time_groups/`：真实流量、逐日 MAE/WAPE、高估低估方向、增加输入的相对 MAE、道路组绝对误差变化，以及 TLAST 5% 的 U/O 变化。",
        "- `stats/inflow_time_group_validation.json`：校验差值。",
        "",
        "复现命令：",
        "",
        "```bash",
        "python3 analysis/model_comparison/scripts/analyze_inflow_time_groups.py",
        "```",
        "",
    ]


def assert_checks(validation: dict[str, object]) -> None:
    problems = []
    if validation["processed_runs"] != EXPECTED_RUNS:
        problems.append(f"processed {validation['processed_runs']} runs, expected {EXPECTED_RUNS}")
    if validation["anomalies"]:
        problems.append(f"{len(validation['anomalies'])} anomalies")
    identity = validation["identity"]
    for key, limit in {
        "U_plus_O_minus_abs": 1e-6,
        "O_minus_U_minus_error": 1e-6,
        "direction_count": 1e-6,
        "true_abs_gap": 1e-6,
    }.items():
        if identity[key] > limit:
            problems.append(f"{key}={identity[key]}")
    if identity["negative_true"] != 0:
        problems.append("negative true values")
    if validation["published"]["mae"] > 1e-6 or validation["direct_sums"]["mae"] > 1e-6:
        problems.append(f"mae gap published {validation['published']['mae']} direct {validation['direct_sums']['mae']}")
    if validation["published"]["rmse"] > 1e-5 or validation["direct_sums"]["rmse"] > 1e-5:
        problems.append("rmse gap")
    if validation["published"]["mape_nonzero"] > 1e-4 or validation["published"]["wape"] > 1e-4:
        problems.append("relative-metric gap versus published metrics")
    for key, value in validation["existing_tables"].items():
        if value > 1e-4:
            problems.append(f"existing table {key}={value}")
    if validation["coverage_element_gap"] > 0:
        problems.append("element coverage gap")
    if validation["groups"]["counts"] != {"inactive": 266, "low": 380, "medium": 380, "high": 380}:
        problems.append("group counts")
    if problems:
        raise RuntimeError("validation failed: " + "; ".join(problems))


def main() -> None:
    self_check()
    ensure_dirs()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    calendar = load_calendar()
    edges, node_hash = load_nodes()
    runs, anomalies = select_runs()
    if anomalies or len(runs) != EXPECTED_RUNS:
        write_json(VALIDATION_PATH, {"processed_runs": len(runs), "anomalies": anomalies})
        raise RuntimeError(f"expected {EXPECTED_RUNS} official runs, found {len(runs)}; anomalies saved to {VALIDATION_PATH}")
    references = load_reference_tables()
    result = process_runs(runs, calendar)
    cube = result["cube"]
    stored_groups = reference_groups("stid", "single_to_single", "p05")
    codes = result["group_info"]["codes"]
    mismatches = [index for index in range(1406) if stored_groups[index] != GROUP_NAMES[int(codes[index])]]
    if mismatches:
        raise RuntimeError(f"road groups differ from node_metrics.csv.gz at {len(mismatches)} nodes")
    existing = check_existing_tables(cube, references)
    identity = identity_gaps(cube)
    coverage_gap = coverage_gaps(cube, result["slot_count"], result["group_info"]["counts"])
    validation = {
        "processed_runs": len(result["processed"]),
        "expected_runs": EXPECTED_RUNS,
        "anomalies": anomalies,
        "node_order_sha256": node_hash,
        "label_sha256": result["group_info"]["label_sha256"],
        "day_slot_hash": result["day_slot_hash"],
        "groups": {
            "counts": result["group_info"]["counts"],
            "low_max_mean": result["group_info"]["low_max_mean"],
            "mid_max_mean": result["group_info"]["mid_max_mean"],
            "inactive_max_mean": result["group_info"]["inactive_max_mean"],
            "node_metric_mismatches": len(mismatches),
        },
        "slots_per_hour": result["slot_count"][0].tolist(),
        "published": result["max_published"],
        "direct_sums": result["max_direct"],
        "max_day_hour_sum_gap": result["max_day_hour_sum_gap"],
        "max_group_sum_gap": result["max_group_sum_gap"],
        "existing_tables": existing,
        "identity": identity,
        "coverage_element_gap": coverage_gap,
        "command": "python3 analysis/model_comparison/scripts/analyze_inflow_time_groups.py",
    }
    assert_checks(validation)
    write_road_groups(edges, result["group_info"])
    write_table(HOUR_PATH, HOUR_FIELDS, iter_hour_rows(cube, calendar, result["slot_count"], result["slot_min"], result["slot_max"], result["group_info"]["counts"]))
    write_table(PERIOD_PATH, PERIOD_FIELDS, iter_period_rows(cube, calendar, result["slot_count"], result["group_info"]["counts"]))
    write_table(DAY_PATH, DAY_FIELDS, iter_day_rows(cube, calendar, result["group_info"]["counts"]))
    write_table(DELTA_PATH, DELTA_FIELDS, iter_delta_rows(cube, calendar, result["slot_count"], result["group_info"]["counts"]))
    use_chinese_font()
    plot_true_volume(cube, calendar, result["slot_count"])
    plot_day_mae_wape(cube, calendar)
    plot_direction(cube)
    plot_input_heatmaps(cube)
    plot_cancellation(cube, calendar)
    plot_tlast_uo(cube, calendar)
    report = render_report(cube, calendar, validation, anomalies)
    REPORT_PATH.write_text(report, encoding="utf-8")
    write_json(VALIDATION_PATH, validation)
    print("validation", validation["published"], validation["existing_tables"], validation["identity"], flush=True)
    print("wrote", REPORT_PATH, flush=True)


if __name__ == "__main__":
    main()
