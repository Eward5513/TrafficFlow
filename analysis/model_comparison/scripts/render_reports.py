"""Write the audit and comparison reports from the tables produced earlier."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from analysis_lib import (
    FAMILY_FOUR,
    FAMILY_LEGACY,
    FAMILY_PHASE_SINGLE,
    MODEL_LABEL,
    MODELS,
    OUT_ROOT,
    PRACTICAL_GAP_PCT,
    RATES,
    STATE_LABEL,
    STATES,
    STATS,
    TABLES,
    as_float,
    label,
    read_csv,
    read_json,
)

FAMILY_TITLE = {
    FAMILY_LEGACY: "原始信号配时，单通道 inflow",
    FAMILY_PHASE_SINGLE: "3 秒相位仿真，单通道 inflow",
    FAMILY_FOUR: "3 秒相位仿真，四状态",
}


def fmt(value: object, digits: int = 3) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def ranked(rows: list[dict[str, str]], family: str, task: str, channel: str, rate: str) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row["family"] == family and row["task_type"] == task and row["channel"] == channel and row["penetration"] == rate
    ]
    return sorted(selected, key=lambda row: int(row["rank_mae"]))


def metric_lookup(long_rows: list[dict[str, str]]) -> dict[tuple[str, ...], dict[str, str]]:
    found = {}
    for row in long_rows:
        if row["channel_role"] == "pooled_not_for_state_ranking":
            continue
        found[(row["family"], row["model"], row["task_type"], row["channel"], row["penetration"])] = row
    return found


def huber_gaps(inventory: list[dict[str, str]]) -> list[dict[str, float]]:
    gaps = []
    for row in inventory:
        if row["Include In Comparison"] != "yes" or row["Selection Epoch Matches MAE-best"] != "no":
            continue
        history_path = Path(row["Result Directory"]) / "training_history.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        by_epoch = {int(item["epoch"]): item for item in history}
        selected = by_epoch[int(float(row["Best Epoch"]))]
        mae_best = by_epoch[int(float(row["MAE-best Epoch"]))]
        selected_mae = float(selected["val_mae"])
        best_mae = float(mae_best["val_mae"])
        gaps.append(
            {
                "model": row["Model"],
                "task": row["Task Type"],
                "output": row["Output Type"],
                "rate": row["Penetration Rate"],
                "selected_epoch": int(float(row["Best Epoch"])),
                "mae_epoch": int(float(row["MAE-best Epoch"])),
                "selected_val_mae": selected_mae,
                "best_val_mae": best_mae,
                "absolute_gap": selected_mae - best_mae,
                "relative_gap_percent": 100.0 * (selected_mae - best_mae) / best_mae if best_mae else float("nan"),
            }
        )
    return gaps


def coverage_table(inventory: list[dict[str, str]]) -> str:
    headers = ["模型", "原始单通道", "3 秒相位单通道", "四状态", "未进入比较的目录"]
    lines = []
    comparable = [row for row in inventory if row["Include In Comparison"] == "yes"]
    for model in MODELS:
        def count(family: str, task: str | None = None) -> int:
            return sum(
                1
                for row in comparable
                if row["Model"] == model and row["Family"] == family and (task is None or row["Task Type"] == task)
            )
        extra = sum(1 for row in inventory if row["Model"] == model and row["Family"] == "nonstandard")
        four = count(FAMILY_FOUR)
        lines.append(
            [
                MODEL_LABEL[model],
                f"{count(FAMILY_LEGACY)}/7",
                f"{count(FAMILY_PHASE_SINGLE)}/7",
                f"{four}/63",
                str(extra),
            ]
        )
    return markdown_table(headers, lines)


def wide_metric_table(rank_rows: list[dict[str, str]], family: str, task: str, channel: str, metric: str) -> str:
    models = []
    for model in MODELS:
        if any(row["family"] == family and row["task_type"] == task and row["channel"] == channel and row["model"] == model for row in rank_rows):
            models.append(model)
    headers = ["渗透率", *[MODEL_LABEL[model] for model in models]]
    body = []
    for rate in RATES:
        cells = [rate]
        present = False
        for model in models:
            match = next(
                (
                    row
                    for row in rank_rows
                    if row["family"] == family
                    and row["task_type"] == task
                    and row["channel"] == channel
                    and row["penetration"] == rate
                    and row["model"] == model
                ),
                None,
            )
            if match is None:
                cells.append("")
            else:
                present = True
                cells.append(f"{fmt(match[metric])} (#{match[f'rank_{metric}']})")
        if present:
            body.append(cells)
    return markdown_table(headers, body)


def winner_sentence(rank_rows: list[dict[str, str]], family: str, task: str, channel: str) -> str:
    pieces = []
    for rate in RATES:
        ordered = ranked(rank_rows, family, task, channel, rate)
        if not ordered:
            continue
        best = ordered[0]
        second = ordered[1] if len(ordered) > 1 else None
        gap = "无对照" if second is None else f"与 {label(second['model'])} 相差 {fmt(second['mae_gap_percent'], 1)}%"
        pieces.append(f"{rate} 为 {label(best['model'])}（MAE {fmt(best['mae'])}，{gap}）")
    return "；".join(pieces) + "。"


def small_gap_note(rank_rows: list[dict[str, str]], family: str, task: str, channel: str, ignore: set[str] | None = None) -> str:
    ignore = ignore or set()
    notes = []
    for rate in RATES:
        ordered = [row for row in ranked(rank_rows, family, task, channel, rate) if row["model"] not in ignore]
        if len(ordered) < 2:
            continue
        gap = as_float(ordered[1]["mae_gap_percent"])
        if gap is None:
            continue
        if gap < PRACTICAL_GAP_PCT:
            notes.append(f"{rate} 前两名差距 {fmt(gap, 1)}%")
        else:
            notes.append(f"{rate} 前两名差距 {fmt(gap, 1)}%，超过 {PRACTICAL_GAP_PCT:.0f}% 的报告阈值")
    return "；".join(notes) + "。"


def degradation_table(rows: list[dict[str, str]], family: str, task: str, channel: str) -> str:
    selected = [
        row
        for row in rows
        if row["family"] == family and row["task_type"] == task and row["channel"] == channel and row["mae_p05"] and row["mae_p70"]
    ]
    selected.sort(key=lambda row: float(row["mae_increase_p70_to_p05"]))
    body = [
        [
            label(row["model"]),
            fmt(row["mae_p70"]),
            fmt(row["mae_p05"]),
            fmt(row["mae_increase_p70_to_p05"]),
            fmt(row["relative_increase_percent"], 1),
            fmt(row["low_range_slope_per_point"], 4),
            fmt(row["high_range_slope_per_point"], 4),
            row["monotonic_nonincreasing_mae"],
        ]
        for row in selected
    ]
    return markdown_table(
        ["模型", "p70 MAE", "p05 MAE", "绝对增幅", "相对增幅 %", "p05→p20 斜率", "p40→p70 斜率", "随渗透率升高 MAE 不升"],
        body,
    )


def mechanism_summary(contrasts: list[dict[str, str]]) -> str:
    lines = []
    for channel in STATES:
        for rate in ("p05", "p30", "p70"):
            part = [row for row in contrasts if row["channel"] == channel and row["penetration"] == rate]
            if not part:
                continue
            fusion = ", ".join(
                f"{label(row['model'])} {fmt(row['fusion_gap_percent'], 1)}%"
                for row in sorted(part, key=lambda row: float(row["fusion_gap_percent"]))
            )
            multi = ", ".join(
                f"{label(row['model'])} {fmt(row['multitask_gap_percent'], 1)}%"
                for row in sorted(part, key=lambda row: float(row["multitask_gap_percent"]))
            )
            lines.append(f"- {STATE_LABEL[channel]} {rate}。联合输入相对单独输入：{fusion}。联合输出相对单输出：{multi}。")
    return "\n".join(lines)


def state_pair_table(long_rows: list[dict[str, str]], left: str, right: str) -> str:
    body = []
    for model in MODELS:
        for rate in ("p05", "p70"):
            a = next(
                (
                    row
                    for row in long_rows
                    if row["family"] == FAMILY_FOUR
                    and row["task_type"] == "single_to_single"
                    and row["model"] == model
                    and row["channel"] == left
                    and row["penetration"] == rate
                ),
                None,
            )
            b = next(
                (
                    row
                    for row in long_rows
                    if row["family"] == FAMILY_FOUR
                    and row["task_type"] == "single_to_single"
                    and row["model"] == model
                    and row["channel"] == right
                    and row["penetration"] == rate
                ),
                None,
            )
            if a is None or b is None:
                continue
            body.append(
                [
                    label(model),
                    rate,
                    fmt(a["mae"]),
                    fmt(b["mae"]),
                    fmt(float(b["mae"]) - float(a["mae"])),
                    fmt(a["wape"], 1),
                    fmt(b["wape"], 1),
                ]
            )
    return markdown_table(
        ["模型", "渗透率", f"{STATE_LABEL[left]} MAE", f"{STATE_LABEL[right]} MAE", "右减左", f"{STATE_LABEL[left]} WAPE", f"{STATE_LABEL[right]} WAPE"],
        body,
    )


def difficulty_table(long_rows: list[dict[str, str]]) -> str:
    body = []
    for channel in STATES:
        for rate in ("p05", "p70"):
            maes = []
            wapes = []
            mapes = []
            zeros = []
            for model in MODELS:
                row = next(
                    item
                    for item in long_rows
                    if item["family"] == FAMILY_FOUR
                    and item["task_type"] == "single_to_single"
                    and item["model"] == model
                    and item["channel"] == channel
                    and item["penetration"] == rate
                )
                maes.append(float(row["mae"]))
                wapes.append(float(row["wape"]))
                mapes.append(float(row["mape_nonzero"]))
                zeros.append(float(row["zero_target_fraction"]))
            body.append(
                [
                    STATE_LABEL[channel],
                    rate,
                    fmt(median(maes)),
                    fmt(min(maes)),
                    fmt(max(maes)),
                    fmt(median(wapes), 1),
                    fmt(median(mapes), 1),
                    fmt(100 * median(zeros), 1),
                ]
            )
    return markdown_table(
        ["状态", "渗透率", "六模型 MAE 中位数", "最小 MAE", "最大 MAE", "WAPE 中位数", "MAPE_nonzero 中位数", "零值比例中位数 %"],
        body,
    )


def error_prose(summary: dict[str, object]) -> list[str]:
    return [
        f"测试预测共复核 {summary['runs']} 组。`target_slot == window_start_slot + 11` 且时间片落在当天 0–287 的有 {summary['aligned']} 组。",
        f"测试日与 manifest 的 5 天、每天 277 个窗口、窗口起点为 0..276 的有 {summary['test_days_match_manifest']} 组。",
        f"由 `pred_raw` 与 `y_raw` 重算的 MAE 相对 `test_metrics.json` 的最大绝对差是 {float(summary['max_mae_abs_diff']):.3e}；分通道最大绝对差是 {float(summary['max_channel_mae_abs_diff']):.3e}。",
        f"标签数组哈希冲突 {len(summary['label_hash_conflicts'])} 组。同一家族、任务、通道和渗透率下，各模型面对的是同一份测试标签。",
        (
            "3 秒相位单通道 inflow 的 35 组测试预测与四状态 single-to-single inflow 逐字节相同"
            f"（{summary['phase3s_flat_vs_four_state_inflow']['identical_pred_raw_sha256']}/"
            f"{summary['phase3s_flat_vs_four_state_inflow']['pairs']}）。"
            "下文不把这 35 组当作另一次独立实验。"
        ),
    ]


def spatial_sentence(spatial: list[dict[str, str]], family: str, task: str, channel: str, rate: str) -> str:
    parts = []
    for model in MODELS:
        groups = {
            row["flow_group"]: row
            for row in spatial
            if row["family"] == family
            and row["task_type"] == task
            and row["channel"] == channel
            and row["penetration"] == rate
            and row["model"] == model
        }
        if "low" not in groups or "high" not in groups:
            continue
        low = float(groups["low"]["mae"])
        high = float(groups["high"]["mae"])
        ratio = high / low if low else float("nan")
        parts.append(f"{label(model)} 高/低 = {fmt(ratio, 2)}（{fmt(high)} / {fmt(low)}）")
    if not parts:
        return "这一组没有足够的活跃道路。"
    example = next(
        row
        for row in spatial
        if row["family"] == family and row["task_type"] == task and row["channel"] == channel and row["penetration"] == rate and row["flow_group"] == "low"
    )
    return (
        f"道路分组按该通道测试集平均真实值的三分位，只在平均值大于 0 的道路上划分。"
        f"低组上界 {fmt(example['low_max_mean'])}，中组上界 {fmt(example['mid_max_mean'])}。"
        + "；".join(parts)
        + "。"
    )


def period_sentence(periods: list[dict[str, str]], family: str, task: str, channel: str, rate: str) -> str:
    pieces = []
    for model in MODELS:
        rows = {
            row["period"]: row
            for row in periods
            if row["family"] == family
            and row["task_type"] == task
            and row["channel"] == channel
            and row["penetration"] == rate
            and row["model"] == model
        }
        if "morning_peak_07_09" not in rows or "midday_10_16" not in rows:
            continue
        morning = float(rows["morning_peak_07_09"]["mae"])
        evening = float(rows["evening_peak_17_19"]["mae"])
        midday = float(rows["midday_10_16"]["mae"])
        night = float(rows["night_00_06"]["mae"])
        pieces.append(
            f"{label(model)} 早高峰 {fmt(morning)}、晚高峰 {fmt(evening)}、10:00–16:00 {fmt(midday)}、0:00–6:00 {fmt(night)}"
        )
    return "；".join(pieces) + "。"


def bias_sentence(audit: list[dict[str, str]], family: str, task: str, channel_is_output: bool = False) -> str:
    pieces = []
    for rate in ("p05", "p70"):
        rows = [
            row
            for row in audit
            if row["family"] == family and row["task_type"] == task and row["penetration"] == rate
        ]
        if channel_is_output:
            rows = [row for row in rows if row["output_type"] in STATES or row["output_type"] == "inflow"]
        if not rows:
            continue
        biases = [float(row["bias"]) for row in rows]
        corrs = [float(row["correlation"]) for row in rows]
        overs = [float(row["fraction_overestimate"]) for row in rows]
        pieces.append(
            f"{rate} 平均偏差 {fmt(min(biases))} 到 {fmt(max(biases))}，"
            f"高估样本比例 {fmt(100 * min(overs), 1)}%–{fmt(100 * max(overs), 1)}%，"
            f"预测与真实值相关系数 {fmt(min(corrs), 3)}–{fmt(max(corrs), 3)}"
        )
    return "；".join(pieces) + "。"


def overlap_sentence(rows: list[dict[str, str]], family: str, task: str, channel: str, rate: str) -> str:
    match = next(
        (
            row
            for row in rows
            if row["family"] == family and row["task_type"] == task and row["channel"] == channel and row["penetration"] == rate
        ),
        None,
    )
    if match is None:
        return "没有重叠统计。"
    return (
        f"{match['models']} 个模型各自 MAE 最高的约 10% 道路中，共同出现 {match['intersection_all_models']} 条"
        f"（每组大约 {match['top_decile_size']} 条）。"
    )


def correlation_sentence(rows: list[dict[str, str]], family: str, task: str, channel: str, rate: str) -> str:
    selected = [
        row
        for row in rows
        if row["family"] == family and row["task_type"] == task and row["channel"] == channel and row["penetration"] == rate
    ]
    if not selected:
        return "没有道路误差相关。"
    values = [float(row["node_mae_correlation"]) for row in selected]
    pair = min(selected, key=lambda row: float(row["node_mae_correlation"]))
    return (
        f"道路 MAE 的两两相关范围是 {fmt(min(values), 2)}–{fmt(max(values), 2)}；"
        f"最低的一对是 {label(pair['model_a'])} 与 {label(pair['model_b'])}（{fmt(pair['node_mae_correlation'], 2)}）。"
    )


def write(name: str, text: str) -> None:
    path = OUT_ROOT / name
    path.write_text(text.strip() + "\n", encoding="utf-8")
    print(f"wrote {path}")


def build() -> None:
    inventory = read_csv(OUT_ROOT / "experiment_inventory.csv")
    long_rows = read_csv(OUT_ROOT / "metrics_long.csv")
    rank_rows = read_csv(TABLES / "rankings.csv")
    degrade_rows = read_csv(TABLES / "penetration_degradation.csv")
    contrasts = read_csv(TABLES / "four_state_mechanism_gaps.csv")
    issues = read_json(STATS / "audit_issues.json")
    error_summary = read_json(STATS / "error_summary.json")
    audit = read_csv(TABLES / "prediction_audit.csv")
    spatial = read_csv(TABLES / "spatial_by_flow_tercile.csv")
    periods = read_csv(TABLES / "temporal_by_period.csv")
    days = read_csv(TABLES / "temporal_by_day.csv")
    bins = read_csv(TABLES / "bias_by_magnitude.csv")
    overlap = read_csv(TABLES / "hard_road_overlap.csv")
    correlations = read_csv(TABLES / "node_mae_correlation.csv")
    gaps = huber_gaps(inventory)
    gap_values = [item["relative_gap_percent"] for item in gaps]
    excluded = [row for row in inventory if row["Family"] == "nonstandard"]
    excluded_lines = [
        f"- {label(row['Model'])} `{row['Result Directory']}`，状态 {row['Training Status']}，MAE {fmt(row['MAE']) if row['MAE'] else '无'}。"
        for row in excluded
    ]

    audit_md = f"""
# 实验配置一致性审计

比较只使用 `run_metadata.json` 中 `status=completed`、带有测试指标和测试预测、且路径符合约定的 seed 42 运行。原始单通道在 `/data5/tzhang174/TrafficFlow/xuancheng26/training_results/`。四状态和后来的单通道目录在 `/data5/tzhang174/TrafficFlow/xuancheng26/training_results_3s/`。请求里写的结果根目录只覆盖前一套。

三套结果分开比较：

- 原始信号配时上的单通道 inflow。标签来自 `training_data/window_data`。
- 3 秒相位仿真上的单通道 inflow。标签来自 `training_data_3s/sampling_seeds/seed_42/flow/window_data`。名称里的 3 秒指信号相位时长被对齐到 3 秒网格，时间聚合仍是 5 分钟、每天 288 个时间片。
- 同一套 3 秒相位仿真上的四状态实验。通道顺序是 inflow、outflow、accumulation_start、accumulation_end。

请求中的日期是 2026-04-01 至 2026-04-29。两份 `split_manifest.json` 实际保存的是 2023-04-01 至 2023-04-30，缺 2023-04-09，共 29 天。训练 20 天、验证 4 天（04-22 至 04-25）、测试 5 天（04-26 至 04-30）。下文使用文件里的日期。

## 完成数量

每个模型的四状态预期是 4 个 single-to-single、4 个 all-to-single、1 个 all-to-all，再乘 7 个渗透率，共 63 组。

{coverage_table(inventory)}

TLAST 没有原始单通道结果，也没有单独的 3 秒相位单通道目录。S2S-GCN-LSTM 的原始单通道只有 p70。其余已进入比较的单元都覆盖了 7 个渗透率。四状态六种模型都是 63/63。

3 秒相位单通道的 35 组测试预测与四状态 single-to-single inflow 逐字节相同，见 `stats/error_summary.json`。它们不提供新的测试结果，下文不再单列排行榜。

未进入比较、但目录里确实有完成记录的运行：

{chr(10).join(excluded_lines)}

`searched_trial018` 是超参搜索，`gaussian_stgcn` 是另一种边权，`_smoke` 是冒烟测试。它们的 MAE 不能补进上面的完成数。原始数据上，搜索试验 p70 的 MAE 是 2.186，Gaussian 边权 p70 是 2.004，都低于协议运行的 4.392，也仍高于同一渗透率下 STID 的 1.124。协议运行本身使用 100 个 epoch；3 秒相位上的 S2S 预算是 200 个 epoch。因此原始单通道那一个 4.392 不能代表“把四状态里的 S2S 曲线换到旧数据上”。

清单：`experiment_inventory.csv`。

## 对齐、划分和指标定义

六个模型的测试指标都经过 `reimplementation/common/evaluation.py`，最终调用 `traffic_metrics`。计算发生在反归一化后的车辆数上，负预测不截断。MAE 和 RMSE 对全部道路和时间片取平均，包括真实值为 0 的位置。MAPE_nonzero 只在真实值大于 0 的位置计算 `|误差| / 真实值`，分母不加 epsilon，结果乘 100。WAPE 是 `sum(|误差|) / sum(|真实值|) × 100`。All-to-all 另外按最后一维的四个通道分别计算；通道顺序与这份函数里的名称一致。重算结果与 JSON 的差异见误差报告。

输入窗口都是 12 个 5 分钟时间片。`target_mode` 都是 `last-observed-step`，也就是 `target_slot == window_start_slot + 11`，不是下一个时间片。窗口不跨天：每个测试日 277 个窗口，起点是 0..276。时间片 0 对应仿真日 00:00，因为流量按 `event_time // 300` 分箱，仿真从 0 秒开始。

检查点由验证集选择，测试集不参与。STID、DCRNN、Graph WaveNet、S2S-GCN-LSTM、STGCN 的选择量是包含零值的验证集原始 MAE。TLAST 的选择量是原始尺度 Huber（delta=2）。TLAST 有 {len(gaps)} 组选出的 epoch 不是验证 MAE 最小的 epoch。这些组的验证 MAE 比 MAE 最优 epoch 高，中位数 {fmt(median(gap_values), 2)}%，最大 {fmt(max(gap_values), 2)}%；其中 {sum(value >= PRACTICAL_GAP_PCT for value in gap_values)} 组达到或超过 {PRACTICAL_GAP_PCT:.0f}%。最大的一组是 accumulation end、p70，验证 MAE 0.329 对 0.313。测试指标仍按原始 MAE 报告，但对应的是 Huber 检查点。因此 TLAST 与其他模型相差约 2% 的测试名次不能当作稳定排序。后文里超过 10% 的存量差距大于这个选模差异，方向不受这一检查点替换的影响；具体会变多少则没有把 MAE 检查点再评一遍测试集，不能改写已有数字。

归一化都是训练集上的全局 z-score。单通道 inflow 使用各渗透率自己的输入 scaler 和所有渗透率共享的全量 inflow scaler。四状态训练在运行时用 `state_normalization.json` 替换目标 scaler，每个物理量一个标量，拟合划分是 train。配置文件里仍写着单通道 `normalization.json` 的路径，这是记录方式，不是又用了 inflow 的均值去缩放 accumulation。TLAST 的 `norm_flag=none` 是原论文的结构开关，损失仍然先反归一化再计算。

## 仍然存在的协议差异

这些差异不阻止报告测试集原始 MAE，但阻止把名次直接解释成结构优劣。

- 只有 seed 42。没有标准差。
- STGCN 的训练损失是归一化空间的 L2，最多 50 个 epoch，没有早停。其余模型的训练损失是原始尺度 MAE，TLAST 是原始尺度 Huber。
- DCRNN 有早停，耐心 50，最多 100 个 epoch。四状态里有多组在 74–93 epoch 停止。S2S 在四状态上的预算是 200，并且会早停。STID、Graph WaveNet、TLAST 固定 100 个 epoch，没有早停。
- 参数量大约是 S2S 7.1 万、STID 15.5 万、Graph WaveNet 32.8 万、DCRNN 37.1 万、TLAST 100 万、STGCN 109 万。输入通道数不同时，同一模型的参数量有几千的差别。
- STID 的 `input_dim` 包含时间通道。单通道任务是 3，四状态 all 输入是 6。交通状态通道仍分别是 1 和 4，清单里分开记录。
- 原始仿真和 3 秒相位仿真的全量标签不同。STID 在两套 inflow 上的零值比例并不相同。不能把两套 MAE 排进一张表。

问题条目也写在 `stats/audit_issues.json`，共 {len(issues)} 条。其中“训练损失”一条的自动说明把 STGCN 也写成了原始 MAE，以上面的列表为准。

## 这一阶段不能做的事

没有第二颗随机种子，不能报告均值和标准差。没有把测试集用于选模的证据，也没有发现跨天窗口。道路没有预先定义的低、中、高流量分组；误差报告里的分组是本次按测试集均值现划的。
"""

    legacy_s2s = next(
        row
        for row in rank_rows
        if row["family"] == FAMILY_LEGACY and row["model"] == "s2s_gcn_lstm" and row["penetration"] == "p70"
    )
    single_md = f"""
# 单通道 inflow 重建

这一节只比较原始信号配时上的单通道实验。3 秒相位单通道与四状态 inflow 的 single-to-single 预测相同，放在四状态报告里，不在这里重复排名。

参与全部 7 个渗透率的是 STID、STGCN、DCRNN、Graph WaveNet。S2S-GCN-LSTM 只有 p70。TLAST 没有这套结果。单位是每条道路、每个 5 分钟时间片的车辆数。MAPE 和 WAPE 已经是百分数。

相对差距小于 {PRACTICAL_GAP_PCT:.0f}% 时，本文只把它记为小差距。这不是显著性检验。

## MAE

{wide_metric_table(rank_rows, FAMILY_LEGACY, "single_channel_inflow", "inflow", "mae")}

各渗透率 MAE 最低的模型：{winner_sentence(rank_rows, FAMILY_LEGACY, "single_channel_inflow", "inflow")}

去掉只有 p70 的 S2S 后，前两名的差距：{small_gap_note(rank_rows, FAMILY_LEGACY, "single_channel_inflow", "inflow", {"s2s_gcn_lstm"})}

没有一个模型在 7 个渗透率上都是 MAE 最低。p05 是 Graph WaveNet，p10–p30 是 STGCN，p40–p70 是 STID。除 p05 的第二名差距为 2.3%、p70 的 STGCN 落后 3.1% 以外，这四个模型之间的差距都低于 2%。p10–p50 上，四者的 MAE 可以看成同一水平。p70 的 S2S MAE 是 {fmt(legacy_s2s['mae'])}，比 STID 高 {fmt(legacy_s2s['mae_gap_percent'], 1)}%。这个数字来自 100 epoch 的协议运行；同目录的搜索试验和 Gaussian 边权运行更低，但仍然明显差于另外四个模型，而且不属于这张表的协议。

## RMSE、MAPE_nonzero 和 WAPE

RMSE：

{wide_metric_table(rank_rows, FAMILY_LEGACY, "single_channel_inflow", "inflow", "rmse")}

MAPE_nonzero：

{wide_metric_table(rank_rows, FAMILY_LEGACY, "single_channel_inflow", "inflow", "mape_nonzero")}

WAPE：

{wide_metric_table(rank_rows, FAMILY_LEGACY, "single_channel_inflow", "inflow", "wape")}

四个完整模型在 RMSE 和 WAPE 上的名次与 MAE 基本一致，差距同样很小。MAPE 只看正的真实值，对小流量更敏感，所以个别渗透率的名次会和 MAE 不同；它没有产生一个在所有渗透率上都稳定领先的模型。S2S 在 p70 的四项指标都远离另外四个模型。

图：`figures/legacy_single_channel/mae_vs_penetration.png`、`rmse_vs_penetration.png`、`wape_vs_penetration.png`。原始排名在 `tables/rankings.csv`。

## 可以保留的判断

在这套单通道 inflow 上，STID、STGCN、DCRNN 和 Graph WaveNet 没有拉开有实际意义的稳定差距。低渗透率下 Graph WaveNet 略低，高渗透率下 STID 略低，中间渗透率几乎重合。S2S-GCN-LSTM 在唯一完成的 p70 上明显更差，但该运行的训练预算和另一次搜索不一致，不能单独用来估计这个结构的全部潜力。TLAST 没有加入这套比较。
"""

    four_md = f"""
# 四状态实验

数据是 3 秒相位仿真。六种模型都完成了 7 个渗透率上的 single-to-single、all-to-single 和 all-to-all。All-to-all 的四个通道分开比较。四个通道直接平均的 MAE 只留在 `metrics_long.csv` 的 `pooled_not_for_state_ranking` 行里，不作为名次依据。

Single-to-single 用同一状态的部分观测重建该状态的全量值。All-to-single 把四种状态一起作为输入，只预测一种状态，用来看输入端多变量。All-to-all 输入和输出都是四种状态，和 all-to-single 对比时输入相同，用来看输出端联合学习。两者相减才把两种机制分开；all-to-all 直接减 single-to-single 会把它们混在一起。

## 四种状态的尺度

{difficulty_table(long_rows)}

Inflow 和 outflow 的 MAE 大约是 accumulation 的三倍，因为流量本身更大。WAPE 和 MAPE 相反：accumulation 的 WAPE 中位数明显高于流量，零值比例也大约是流量的两倍。按相对误差，存量比流量更难重建。Accumulation start 和 end 的中位数几乎一样，不能从这些汇总里把期初和期末分成两种难度。

## Single-to-single

### Inflow

{wide_metric_table(rank_rows, FAMILY_FOUR, "single_to_single", "inflow", "mae")}

{winner_sentence(rank_rows, FAMILY_FOUR, "single_to_single", "inflow")}

前两名差距：{small_gap_note(rank_rows, FAMILY_FOUR, "single_to_single", "inflow", {"s2s_gcn_lstm"})}

去掉 S2S 后，inflow 上没有模型以超过约 3% 的差距稳居第一。TLAST 在 p05 和 p10 的 MAE 最低，但 p05 只比 Graph WaveNet 低 0.5%。p20 以后 STID 与 TLAST 交替领先，差距多在 2% 以内。S2S 在七个渗透率上都是最后一名，p05 比第一名高 41%，p70 高 76%。

### Outflow

{wide_metric_table(rank_rows, FAMILY_FOUR, "single_to_single", "outflow", "mae")}

{winner_sentence(rank_rows, FAMILY_FOUR, "single_to_single", "outflow")}

Outflow 的名次和 inflow 相同方向：低渗透率 TLAST，高渗透率 STID，S2S 始终最后。p05 上 TLAST 与第二名 Graph WaveNet 的差距是 4.0%，比 inflow 上的 0.5% 大。p70 上除 S2S 外，STGCN 落后 STID 3.3%，其余低于 2.2%。

Inflow 与 outflow 的配对：

{state_pair_table(long_rows, "inflow", "outflow")}

同一模型、同一渗透率下，outflow MAE 通常略高于 inflow，p05 的差别大约是 0 到 0.08 辆车，p70 大约是 0 到 0.03。WAPE 也很接近。没有出现一种模型只把其中一个方向做好、另一个方向明显失败的情况。

### Accumulation

Start：

{wide_metric_table(rank_rows, FAMILY_FOUR, "single_to_single", "accumulation_start", "mae")}

End：

{wide_metric_table(rank_rows, FAMILY_FOUR, "single_to_single", "accumulation_end", "mae")}

Start 的领先者：{winner_sentence(rank_rows, FAMILY_FOUR, "single_to_single", "accumulation_start")}

这里的差距和流量不是一个量级。p05 上 TLAST 的 start MAE 是 0.674，STID 高 7.5%，Graph WaveNet 高 28.9%，STGCN 高 38.1%，S2S 高 52.7%。p70 反过来，STID 为 0.290，TLAST 高 14.0%，STGCN 高 17.5%，S2S 高 83.5%。End 的名次和差距与 start 平行。

Start 与 end：

{state_pair_table(long_rows, "accumulation_start", "accumulation_end")}

同一模型在两个存量上的 MAE 差不超过约 0.01。重建难度没有因为“期初”或“期末”分开。

RMSE、MAPE 和 WAPE 的完整名次在 `tables/rankings.csv`。Single-to-single 的曲线在 `figures/four_state/s2s_mae_<state>.png`、`s2s_rmse_<state>.png` 和 `s2s_wape_<state>.png`。

## 输入融合和联合输出

负的百分数表示后一种设置的 MAE 更低。融合差距是 all-to-single 减 single-to-single，再除以后者。多任务差距是 all-to-all 的对应通道减 all-to-single，输入都是四种状态。

{mechanism_summary(contrasts)}

流量上，两种改动大多落在 ±3% 以内，符号也不统一。S2S 从额外输入里得到的相对降幅稍大，大约 2%–10%，但它的绝对 MAE 仍然最高。对 STID、STGCN、DCRNN、Graph WaveNet 和 TLAST，现有结果不支持“把四种状态一起输入就能明显改善流量重建”。

存量上，额外输入的收益仍然小，多数模型在 0 到 −4% 之间。联合输出则系统地变差：p05 的 start 通道，STID 的 MAE 上升 21.9%，TLAST 上升 13.0%，DCRNN 上升 6.5%，Graph WaveNet 上升 4.5%，STGCN 基本不变。p70 上 S2S 的联合输出比单输出高 25.5%。因此，all-to-all 相对 single-to-single 的变化不能解释成“模型利用了额外状态”；存量上的主要变化来自输出端同时预测四个量，而且这个变化是损失。

每个状态的三套曲线画在 `figures/four_state/mechanism_<state>.png`。逐格数字在 `tables/four_state_mechanism_gaps.csv`。

## 这一节不能推出的结构结论

TLAST 在低渗透率存量上的 MAE 更低，STID 在高渗透率存量上更低。这是结果差异，不是已经证明代理节点或身份嵌入是原因。STGCN 只用 50 个 epoch 和归一化 L2，S2S 的参数量和训练预算都不同。要比较结构，需要先把损失、epoch 和选择指标对齐。
"""

    pen_md = f"""
# 渗透率敏感性

渗透率是观测车辆所占比例，不是把标签按比例缩小。全量标签不随渗透率缩放。曲线的横轴使用 5、10、20、30、40、50、70 这些实际百分比，不是等距类别。

相对增幅是 `(MAE_p05 - MAE_p70) / MAE_p70`。一个在 p70 上已经很差的模型，相对增幅可能更小，所以表里同时给出绝对增幅。低区间斜率是 `(MAE_p05 - MAE_p20) / 15`，高区间斜率是 `(MAE_p40 - MAE_p70) / 30`，单位是 MAE 每增加一个渗透率百分点。斜率越大，表示渗透率下降时 MAE 上升越快。

## 原始单通道 inflow

{degradation_table(degrade_rows, FAMILY_LEGACY, "single_channel_inflow", "inflow")}

四个完整模型的 MAE 都随渗透率升高而下降。从 p70 到 p05，MAE 大约增加 1.48–1.61，相对增幅 130%–141%。低区间斜率约 0.024–0.037，高区间约 0.020–0.021。Graph WaveNet 的低区间斜率最低，绝对增幅也最小；DCRNN 的低区间斜率最高。差别存在，但 p05 上四者的 MAE 仍只相差 0.13。S2S 没有 p05，不能计算这条退化。

名次随渗透率变化：Graph WaveNet、STGCN、STID 依次成为 MAE 最低的模型。变化发生在不到 3% 的差距里。图：`figures/legacy_single_channel/relative_degradation_p70_to_p05.png`。

## 四状态 single-to-single

### Inflow

{degradation_table(degrade_rows, FAMILY_FOUR, "single_to_single", "inflow")}

### Outflow

{degradation_table(degrade_rows, FAMILY_FOUR, "single_to_single", "outflow")}

### Accumulation start

{degradation_table(degrade_rows, FAMILY_FOUR, "single_to_single", "accumulation_start")}

### Accumulation end

{degradation_table(degrade_rows, FAMILY_FOUR, "single_to_single", "accumulation_end")}

六条 single-to-single 曲线都是渗透率越高、MAE 越低，没有出现升高渗透率后 MAE 反弹。

流量上，S2S 的相对增幅约 83%，是六者里最小的，但它在 p70 的 MAE 已经是 1.97，绝对增幅 1.65，大于 TLAST 的 1.42 和 Graph WaveNet 的 1.45。低区间绝对斜率也是 S2S 最大（inflow 0.057）。相对增幅小，是因为高渗透率误差没有降下来，不是因为它少依赖观测。TLAST 和 Graph WaveNet 的低区间斜率低于 STID、STGCN 和 DCRNN。到 p40–p70，五个较强模型的斜率都接近 0.020，高渗透率区间的退化速度几乎一样。

存量上这个差别更大。TLAST 从 p70 到 p05 的 start MAE 增加 0.344，STID 增加 0.435，Graph WaveNet 增加 0.569。Graph WaveNet、DCRNN 和 STGCN 的相对增幅约 174%–190%，TLAST 约 104%，STID 约 150%。TLAST 在低渗透率的优势和它在高渗透率落后于 STID 是同一件事的两面：它的曲线更平，两端都不走到 STID 的高渗透率水平。S2S 的存量相对增幅再次较小，绝对误差在两端都最大。

Outflow 与 inflow、accumulation end 与 start 的增幅和斜率成对接近。渗透率变化没有只破坏其中一个方向或其中一个存量端点。

图：`figures/four_state/s2s_relative_degradation_<state>.png` 和 `figures/four_state/s2s_mae_<state>.png`。

## 名次是否随渗透率改变

原始单通道和四状态流量上，名次会变，但较强四个或五个模型的差距经常小于 2%。这种名次变化不能当成稳定的结构排序。

存量上的名次变化更大，而且差距超过报告阈值。p05–p20 是 TLAST，p30 附近开始变成 STID，p70 上 TLAST 比 STID 高 14%–16%。这是现有结果里最清楚的渗透率与模型交互。它仍是一次 seed、一套损失设置下的现象。

All-to-single 和 all-to-all 的渗透率曲线与 single-to-single 同向。联合输出没有改变“观测越少、误差越大”这一形状；它主要把存量曲线整体抬高。见 `figures/four_state/mechanism_<state>.png`。

## 还不能回答的机制问题

低渗透率下，Graph WaveNet 和 TLAST 的流量 MAE 上升较慢，TLAST 的存量 MAE 上升更慢。实验记录里没有把这个差异分离成空间相关、时间相关或全局信息。自适应邻接、扩散卷积、代理节点和时间卷积都同时在变化，而且损失与 epoch 也不相同。可以把现象留给下一轮消融，不能从曲线形状指定原因。
"""

    # Error narrative numbers
    legacy_spatial = spatial_sentence(spatial, FAMILY_LEGACY, "single_channel_inflow", "inflow", "p05")
    inflow_spatial_low = spatial_sentence(spatial, FAMILY_FOUR, "single_to_single", "inflow", "p05")
    inflow_spatial_high = spatial_sentence(spatial, FAMILY_FOUR, "single_to_single", "inflow", "p70")
    acc_spatial = spatial_sentence(spatial, FAMILY_FOUR, "single_to_single", "accumulation_start", "p05")
    legacy_period = period_sentence(periods, FAMILY_LEGACY, "single_channel_inflow", "inflow", "p05")
    inflow_period = period_sentence(periods, FAMILY_FOUR, "single_to_single", "inflow", "p05")
    acc_period = period_sentence(periods, FAMILY_FOUR, "single_to_single", "accumulation_start", "p05")
    legacy_bias = bias_sentence(
        [row for row in audit if row["task_type"] == "single_channel_inflow"],
        FAMILY_LEGACY,
        "single_channel_inflow",
    )
    four_bias = bias_sentence(
        [row for row in audit if row["output_type"] == "inflow" and row["task_type"] == "single_to_single"],
        FAMILY_FOUR,
        "single_to_single",
    )

    def bin_line(family: str, task: str, channel: str, rate: str, model: str) -> str:
        rows = [
            row
            for row in bins
            if row["family"] == family
            and row["task_type"] == task
            and row["channel"] == channel
            and row["penetration"] == rate
            and row["model"] == model
        ]
        return "，".join(f"{row['bin']} MAE {fmt(row['mae'])} / 偏差 {fmt(row['bias'])}" for row in rows if row["mae"] != "")

    day_bits = []
    for model in ("stid", "tlast", "s2s_gcn_lstm"):
        selected = [
            row
            for row in days
            if row["family"] == FAMILY_FOUR
            and row["task_type"] == "single_to_single"
            and row["channel"] == "inflow"
            and row["penetration"] == "p05"
            and row["model"] == model
        ]
        selected.sort(key=lambda row: row["date"])
        day_bits.append(
            label(model)
            + " "
            + "，".join(f"{row['date']} {fmt(row['mae'])}" for row in selected)
        )

    error_md = f"""
# 空间与时间误差

预测文件是测试集 `test_predictions.npz` 里的 `pred_raw` 和 `y_raw`。验证集预测不进入这一节，因为检查点是按验证误差挑选的。没有重新训练。

{chr(10).join('- ' + line for line in error_prose(error_summary))}

道路编号使用 `r_nodes.csv` 的 `edge_id`，节点顺序与两套数据的 `node_order_sha256` 一致。每条道路的 MAE、RMSE、偏差和测试均值在 `tables/node_metrics.csv.gz`。

## 空间

项目里没有现成的低、中、高流量道路分组。这里用该通道在测试集上的道路平均真实值，把平均值大于 0 的道路分成三等分。零均值道路单独称为 inactive，不放进三分位。

原始单通道 inflow，p05：{legacy_spatial}

四状态 single-to-single inflow，p05：{inflow_spatial_low}

同一任务 p70：{inflow_spatial_high}

四状态 single-to-single accumulation start，p05：{acc_spatial}

高流量组的 MAE 明显高于低流量组。p05 的 inflow 上，高组 MAE 大约是低组的 5.8–7.8 倍。1406 条道路里有 266 条在该测试集上的平均 inflow 为 0，没有进入三分位；其余每组 380 条。低组的平均真实值上界约 6.0，高组超过约 19.8。绝对误差随流量尺度上升，不等于相对误差也更差。Inflow 上各模型的高/低比值接近；存量上 TLAST 的比值小于图模型，但高流量道路仍然更难。当前差距不是由“只有某一个模型能重建高流量道路”造成的。

p05 的 single-to-single inflow 上，六种模型误差最高的约 10% 道路：{overlap_sentence(overlap, FAMILY_FOUR, "single_to_single", "inflow", "p05")}
道路 MAE 的模型间相关：{correlation_sentence(correlations, FAMILY_FOUR, "single_to_single", "inflow", "p05")}

p05 的 accumulation start：{overlap_sentence(overlap, FAMILY_FOUR, "single_to_single", "accumulation_start", "p05")}
{correlation_sentence(correlations, FAMILY_FOUR, "single_to_single", "accumulation_start", "p05")}

难重建的道路在模型之间高度重合。换模型没有把误差从一组道路搬到另一组道路。共同出现的边编号写在 `tables/hard_road_overlap.csv`。

图：`figures/error/legacy_tercile_inflow_p05.png`、`figures/error/s2s_tercile_inflow_p05.png`、`figures/error/s2s_tercile_inflow_p70.png`、`figures/error/s2s_tercile_accumulation_start_p05.png`、`figures/error/s2s_tercile_accumulation_start_p70.png`。

## 时间

时段按仿真时钟划分。时间片 `t` 覆盖 `[t×300, (t+1)×300)` 秒。早高峰是 07:00–09:00，晚高峰是 17:00–19:00，平峰取 10:00–16:00，夜间取 00:00–06:00。其余时段留在逐小时表里。

原始单通道 inflow，p05：{legacy_period}

四状态 inflow，p05：{inflow_period}

四状态 accumulation start，p05：{acc_period}

夜间 MAE 明显低于白天。对 STID、TLAST 和 Graph WaveNet，07:00–09:00 只略高于 10:00–16:00，17:00–19:00 并不稳定地更高。S2S 的早高峰则明显高于平峰。共同的形状是夜间低、白天高，不是某一个模型单独在高峰失效。

测试日是 2023-04-26 至 04-30，星期三到星期日。p05 的四状态 inflow：{'; '.join(day_bits)}。04-28 和 04-29 的 MAE 高于另外三天，这两天最低的是 Graph WaveNet；04-26、04-27 和 04-30 最低的是 TLAST 或 STID。S2S 五天都是最高。较强模型的日名次会变，全日汇总不是每一天都成立。周日并没有和周五、周六一样高，所以这不是一个简单的周末效应。逐日表是 `tables/temporal_by_day.csv`，逐小时表是 `tables/temporal_by_hour.csv`。

图：`figures/error/legacy_hourly_mae_p05.png`、`legacy_hourly_mae_p70.png`、`s2s_hourly_inflow_p05.png`、`s2s_hourly_inflow_p70.png`、`s2s_hourly_accumulation_start_p05.png`、`s2s_hourly_accumulation_start_p70.png`。阴影标出 07:00–09:00 和 17:00–19:00。

## 系统性偏差

偏差定义为预测减真实值。正值是高估。

原始单通道，按整张测试张量：{legacy_bias}

四状态 single-to-single inflow：{four_bias}

整体平均偏差为负：p05 的 single-to-single inflow 上，STGCN 约 −0.20，Graph WaveNet 约 −0.39，TLAST 和 S2S 约 −0.88，STID 约 −1.04。平均偏差为负并不等于大多数样本都被低估。Graph WaveNet 在 p05 inflow 上有 63.6% 的样本预测偏高，但偏高的幅度小于偏低的幅度，所以均值仍是 −0.39。TLAST 只有 32.1% 的样本偏高。相关系数在 inflow 上大约 0.93–0.97，在 accumulation 上大约 0.81–0.93，存量预测和真实值的线性关系更弱。

STID 在四状态 inflow、p05 上按真实值分箱：{bin_line(FAMILY_FOUR, 'single_to_single', 'inflow', 'p05', 'stid')}。

TLAST 同一设置：{bin_line(FAMILY_FOUR, 'single_to_single', 'inflow', 'p05', 'tlast')}。

S2S 同一设置：{bin_line(FAMILY_FOUR, 'single_to_single', 'inflow', 'p05', 's2s_gcn_lstm')}。

真实值为 0 时，偏差非负：STID 约 +0.12，TLAST 约 +0.02，S2S 约 +0.26。真实值超过 20 时，三个模型的偏差都在 −3.5 到 −4.6。高流量被低估是共同的；零值和中等流量上的符号则因模型而异，S2S 在 10 以下的箱子里偏差为正。负预测没有在指标里被截成 0；`metrics_long.csv` 里的 `negative_prediction_fraction` 记录了比例。

图：`figures/error/legacy_bias_bins_inflow_p05.png`、`figures/error/s2s_bias_bins_inflow_p05.png`、`figures/error/s2s_bias_bins_accumulation_start_p05.png`。

分箱、时段和道路三分位的全部数字分别在 `tables/bias_by_magnitude.csv`、`tables/temporal_by_period.csv` 和 `tables/spatial_by_flow_tercile.csv`。
"""

    findings = f"""
# 综合发现

所有比较都来自 seed 42 的一次运行。差距小于 {PRACTICAL_GAP_PCT:.0f}% 时，不把它写成某个模型稳定更好。原始仿真和 3 秒相位仿真的标签不同，没有合并排名。四个状态的 MAE 没有平均。

## 重建任务里已经能看见的困难

全量标签不随渗透率缩放，观测却从 70% 降到 5%。在完成的 single-to-single 曲线上，MAE 随渗透率单调上升。较强模型的流量 MAE 从 p70 到 p05 大约增加 1.4–1.6 辆车每道路每 5 分钟，S2S 约增加 1.65；相对各自的 p70，增幅大约是 83%–145%，其中较小的相对增幅来自 p70 误差仍然很高的 S2S。存量的绝对 MAE 较小，WAPE 更高，零值更多。按相对误差，存量比流量更难。

误差集中在测试集平均流量更高的道路上，也集中在白天而不是夜间。早晚高峰并不稳定地高于 10:00–16:00；04-28 和 04-29 两天的误差高于另外三个测试日。六种模型误差最高的道路高度重合，道路 MAE 的相关在 p05 inflow 上约为 0.96–1.00。难样本主要来自数据，不是某一个模型偶然没拟合到的路段。

高流量位置被系统性低估。零值位置的偏差非负，但幅度因模型而差很多。整体平均偏差为负，同时仍有模型在超过一半的样本上预测偏高，因为低估发生在数值更大的位置。负的车辆数预测被保留在指标里，输出没有受非负约束。存量上的预测—真实值相关低于流量。

Inflow 与 outflow、accumulation start 与 end 的误差成对接近。方向和端点本身不是当前的主要分裂。

## 现有模型各自改变了什么

在原始单通道 inflow 上，STID、STGCN、DCRNN 和 Graph WaveNet 的四项指标都很接近。低渗透率 Graph WaveNet 略好，高渗透率 STID 略好，中间几乎重合。复杂的图卷积、扩散卷积和自适应邻接没有在这套单通道流量上留下超过约 3% 的稳定优势。STID 不使用路网邻接，却处于同一水平。就这次实验而言，额外的空间结构没有表现为明确收益。

S2S-GCN-LSTM 是直接针对重建任务的基线，但在所有完成的四状态曲线上都是 MAE 最高的模型。p70 的 inflow MAE 仍比 STID 高大约 76%。它的参数量也最小。原始单通道的协议运行更差，而且那次只有 100 个 epoch；搜索试验更好一些，仍明显落后。参数量少和训练预算不同都可能有贡献，现有设计不能把差距全部记到 Seq2Seq 结构上。可以确定的是，按已经完成的协议运行，它没有成为更强的重建基线。

TLAST 的差异集中在存量，而且依赖渗透率。p05 的 accumulation MAE 比 STID 低约 7%，比图模型低约 30%–38%。p70 则比 STID 高约 14%–16%。流量上它和 Graph WaveNet、STID 的差距经常小于 2%。代理节点如果有作用，现有证据只把它限制在低渗透率存量，不能说它全面改善了重建。

把四种观测一起输入、只预测一种状态，流量 MAE 的变化大多在 ±3% 以内，存量大约 0 到 −4%。输入端的多变量融合收益小且不稳定。输出端同时预测四个状态则让存量变差，STID 在 p05 大约变差 22%，TLAST 大约 13%。联合学习没有表现出对这四个物理量的正迁移。下一轮不应把 all-to-all 的一个平均数当成“利用了多状态信息”。

STGCN 参数量最大，训练却是 50 个 epoch 的归一化 L2。它在流量上仍处在领先集团，在存量上落后。参数量、图结构和最终误差之间没有单调关系。

## 还没有被这些实验解决的问题

1. 低渗透率下的存量重建。即使是 p05 上最好的 TLAST，accumulation 的 WAPE 仍然远高于流量。
2. 高流量样本的低估。真实值超过 20 时，几个模型的平均偏差都在 −3.5 以下。高峰时段的绝对误差高于夜间，但并不稳定地高于 10:00–16:00。
3. 输出的物理约束。负预测和零值上的正偏差还在。
4. 如何使用其他状态。额外输入几乎没有帮助，联合输出对存量有害。缺的是一种不会把四个尺度绑在同一个头里的用法，而不是再增加一个 all-to-all 跑分。
5. 名次的不确定性。一个 seed，加上 TLAST 的 Huber 选模、STGCN 的损失和 epoch、S2S 的预算，都使 2% 左右的流量名次无法解释。

## 下一阶段更有信息量的实验

先固定协议，再改结构。建议用同一损失（原始 MAE 或同一 Huber）、同一 epoch 上限、同一种验证选择量和至少三个种子，只重复 single-to-single 的 inflow 与 accumulation start，渗透率保留 p05、p20、p70。这样可以直接检验 TLAST 与 STID 在存量上的渗透率交叉是否还在。

空间结构用一个最小对照：同一主干上比较“不用邻接”“固定路网”“自适应邻接”，而不是继续横比六个完整模型。TLAST 的代理节点同样只与去掉代理节点的版本比较。

多状态只做两个已经分开的对照，不再增加混合设置：同一单输出下比较单状态输入和四状态输入；同一四状态输入下比较单输出和四输出。存量上的负迁移如果复现，就不要把 all-to-all 作为主结果。

误差分析可以继续用现有预测，不必为了分箱再训练。值得追加的是按道路均值分组后的 WAPE，用来区分“高流量绝对误差大”和“高流量相对误差也大”。这次报告只给了前者。
"""

    write("comparability_audit.md", audit_md)
    write("single_channel_comparison.md", single_md)
    write("four_state_comparison.md", four_md)
    write("penetration_analysis.md", pen_md)
    write("error_distribution.md", error_md)
    write("overall_findings.md", findings)


if __name__ == "__main__":
    build()
