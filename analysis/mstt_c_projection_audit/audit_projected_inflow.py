"""Audit MSTT projected conservation on saved p05 All2All test predictions.

Reads existing NPZ files only. Does not train, reload a checkpoint, or write
into a result directory.

``pred_raw`` is the post-projection prediction on the vehicle-count scale.
``raw_pred_physical`` is the pre-projection prediction on that same scale.
Channel order is inflow, outflow, accumulation_start, accumulation_end.

The inflow update implied by ``MSTT/model/conservation.py`` is

    I_final = I_raw + w_I * r_raw
    r_raw = N_end - N_start - I + O

``raw_conservation_residual`` is not stored in the NPZ, so this script
recomputes it from ``raw_pred_physical``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from data_root import xuancheng26_training_results_3s_root

SEEDS = (42, 43, 44, 45, 46)
INFLOW = 0
CHANNELS = ("inflow", "outflow", "accumulation_start", "accumulation_end")
# A position is unchanged when the absolute-error difference is within this
# many vehicles. Float32 predictions at this magnitude have an ulp near 1e-6.
UNCHANGED_ATOL = 1e-5
ERROR_EDGES = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, np.inf)
PUBLISHED_C_INFLOW = 2.4383
PUBLISHED_A_INFLOW = 2.4540
PUBLISHED_ATOL = 5e-4
WEIGHT_QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)


def project_conservation(physical: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match ``project_conservation``: y' = y - r (a * w), a = (-1, 1, -1, 1)."""
    if physical.shape != weights.shape or physical.shape[-1] != 4:
        raise ValueError("projection inputs must share shape [..., 4]")
    residual = physical[..., 3] - physical[..., 2] - physical[..., 0] + physical[..., 1]
    direction = np.array((-1.0, 1.0, -1.0, 1.0), dtype=np.float64)
    return physical - residual[..., None] * (direction * weights), residual


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size != len(SEEDS):
        raise ValueError(f"expected {len(SEEDS)} seed values, got {array.size}")
    return {"mean": float(array.mean()), "std": float(array.std(ddof=1))}


def _quantile_dict(values: np.ndarray) -> dict[str, float]:
    points = np.quantile(values, WEIGHT_QUANTILES)
    return {f"p{int(q * 100):02d}": float(v) for q, v in zip(WEIGHT_QUANTILES, points)}


def _bin_label(left: float, right: float) -> str:
    left_text = f"{left:g}"
    right_text = "inf" if np.isinf(right) else f"{right:g}"
    return f"[{left_text},{right_text})"


def error_bins(abs_before: np.ndarray, abs_after: np.ndarray) -> list[dict[str, float | str | int]]:
    index = np.digitize(abs_before, ERROR_EDGES[1:-1], right=False)
    rows = []
    for bin_id, (left, right) in enumerate(zip(ERROR_EDGES[:-1], ERROR_EDGES[1:])):
        mask = index == bin_id
        count = int(mask.sum())
        row: dict[str, float | str | int] = {
            "bin": _bin_label(left, right),
            "count": count,
            "fraction": float(count / abs_before.size),
        }
        if count:
            before = float(abs_before[mask].mean())
            after = float(abs_after[mask].mean())
            delta = abs_before[mask] - abs_after[mask]
            row.update({
                "mae_before": before,
                "mae_after": after,
                "delta_mae": before - after,
                "improved_fraction": float(np.mean(delta > UNCHANGED_ATOL)),
                "worsened_fraction": float(np.mean(delta < -UNCHANGED_ATOL)),
                "unchanged_fraction": float(np.mean(np.abs(delta) <= UNCHANGED_ATOL)),
            })
        rows.append(row)
    return rows


def audit_projected_seed(path: Path) -> dict:
    with np.load(path, mmap_mode="r") as archive:
        required = ("raw_pred_physical", "pred_raw", "y_raw", "correction_weights", "y_pred_raw", "y_true_raw")
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise ValueError(f"{path} is missing {missing}")
        raw = np.asarray(archive["raw_pred_physical"], dtype=np.float64)
        final = np.asarray(archive["pred_raw"], dtype=np.float64)
        true = np.asarray(archive["y_raw"], dtype=np.float64)
        weights = np.asarray(archive["correction_weights"], dtype=np.float64)
        stored_final = np.asarray(archive["y_pred_raw"], dtype=np.float64)
        stored_true = np.asarray(archive["y_true_raw"], dtype=np.float64)
        sample_index = np.asarray(archive["sample_index"])
    if raw.shape != final.shape or raw.shape != true.shape or raw.shape != weights.shape:
        raise ValueError(f"{path} prediction arrays do not share a shape")
    if raw.shape[-1] != 4:
        raise ValueError(f"{path} does not have four state channels")
    reconstructed, residual = project_conservation(raw, weights)
    inflow_raw = raw[..., INFLOW]
    inflow_final = final[..., INFLOW]
    inflow_true = true[..., INFLOW]
    inflow_weight = weights[..., INFLOW]
    formula_inflow = inflow_raw + inflow_weight * residual
    abs_before = np.abs(inflow_raw - inflow_true)
    abs_after = np.abs(inflow_final - inflow_true)
    error_delta = abs_before - abs_after
    correction = inflow_final - inflow_raw
    mae_before = float(abs_before.mean())
    mae_after = float(abs_after.mean())
    return {
        "n_positions": int(inflow_raw.size),
        "shape": list(raw.shape),
        "sample_index_is_arange": bool(np.array_equal(sample_index, np.arange(sample_index.shape[0]))),
        "pred_raw_matches_y_pred_raw_max_abs": float(np.max(np.abs(final - stored_final))),
        "y_raw_matches_y_true_raw_max_abs": float(np.max(np.abs(true - stored_true))),
        "projection_reconstruction_max_abs": float(np.max(np.abs(reconstructed - final))),
        "inflow_formula_max_abs": float(np.max(np.abs(formula_inflow - inflow_final))),
        "inflow_formula_mean_abs": float(np.mean(np.abs(formula_inflow - inflow_final))),
        "weight_sum_max_abs_error": float(np.max(np.abs(weights.sum(axis=-1) - 1.0))),
        "mae_before": mae_before,
        "mae_after": mae_after,
        "delta_mae": mae_before - mae_after,
        "improved_fraction": float(np.mean(error_delta > UNCHANGED_ATOL)),
        "worsened_fraction": float(np.mean(error_delta < -UNCHANGED_ATOL)),
        "unchanged_fraction": float(np.mean(np.abs(error_delta) <= UNCHANGED_ATOL)),
        "mean_abs_inflow_correction": float(np.mean(np.abs(correction))),
        "mean_inflow_correction": float(np.mean(correction)),
        "mean_abs_raw_residual": float(np.mean(np.abs(residual))),
        "correction_over_residual": float(np.mean(np.abs(correction)) / np.mean(np.abs(residual))),
        "weight_mean": float(inflow_weight.mean()),
        "weight_std": float(inflow_weight.std(ddof=1)),
        "weight_quantiles": _quantile_dict(inflow_weight),
        "channel_mean_abs_correction": {
            name: float(np.mean(np.abs(final[..., index] - raw[..., index])))
            for index, name in enumerate(CHANNELS)
        },
        "channel_weight_mean": {
            name: float(weights[..., index].mean()) for index, name in enumerate(CHANNELS)
        },
        "error_bins": error_bins(abs_before.reshape(-1), abs_after.reshape(-1)),
    }


def audit_unconstrained_seed(path: Path) -> dict:
    with np.load(path, mmap_mode="r") as archive:
        if "raw_pred_physical" in archive.files:
            raise ValueError(f"{path} contains a projection array; it is not unconstrained A")
        prediction = np.asarray(archive["pred_raw"], dtype=np.float64)
        target = np.asarray(archive["y_raw"], dtype=np.float64)
    if prediction.shape != target.shape or prediction.shape[-1] != 4:
        raise ValueError(f"{path} is not a four-state prediction")
    return {
        "n_positions": int(prediction[..., INFLOW].size),
        "shape": list(prediction.shape),
        "mae": float(np.mean(np.abs(prediction[..., INFLOW] - target[..., INFLOW]))),
    }


def _published_inflow(run_dir: Path) -> dict:
    report = json.loads((run_dir / "test_sampling_metrics.json").read_text())
    per_seed = {int(seed): float(metrics["inflow_mae"]) for seed, metrics in report["per_seed"].items()}
    ordered = [per_seed[seed] for seed in SEEDS]
    return {"per_seed": per_seed, "mean": float(report["mean"]["inflow_mae"]), "std": float(report["std"]["inflow_mae"]),
            "recomputed_mean_std": mean_std(ordered)}


def verify_original_c(config: dict) -> None:
    output_root = str(config.get("output_root", ""))
    layout = config.get("attention_layout")
    supervision = config.get("projection_supervision")
    block_order = (config.get("architecture") or {}).get("block_order")
    problems = []
    if config.get("conservation_mode") != "projected":
        problems.append("conservation_mode is not projected")
    if supervision not in (None, "all"):
        problems.append(f"projection_supervision is {supervision!r}")
    if config.get("task") != "All2All":
        problems.append("task is not All2All")
    if config.get("attention_order") != "TS" or config.get("ffn_placement") != "after_spatial":
        problems.append("stack is not single-layer T then S then FFN")
    if int(config.get("num_layers", -1)) != 1:
        problems.append("num_layers is not 1")
    if layout not in (None, "factorized_ts"):
        problems.append(f"attention_layout is {layout!r}")
    if block_order != ["T", "S", "FFN"]:
        problems.append(f"block_order is {block_order!r}")
    if "joint_st" in output_root or "C_inflow_aux" in output_root:
        problems.append(f"output_root is {output_root}")
    if float(config.get("projection_raw_loss_weight", -1)) != 0.5:
        problems.append("projection_raw_loss_weight is not the original C value 0.5")
    if float(config.get("projection_consistency_weight", -1)) != 0.0:
        problems.append("projection_consistency_weight is not the original C value 0")
    if problems:
        raise ValueError("resolved config is not original projected C: " + "; ".join(problems))


def verify_unconstrained_a(config: dict) -> None:
    output_root = str(config.get("output_root", ""))
    if config.get("conservation_mode") != "none" or config.get("task") != "All2All":
        raise ValueError("A config is not unconstrained All2All")
    if "single_tsf_adaptive128_all2a_100ep" not in output_root:
        raise ValueError(f"A output_root is {output_root}")
    if config.get("projection_supervision") not in (None, "all"):
        raise ValueError("A config carries a non-original projection supervision")


def _aggregate_bins(seed_rows: list[dict]) -> list[dict]:
    aggregated = []
    for bin_id, edge in enumerate(zip(ERROR_EDGES[:-1], ERROR_EDGES[1:])):
        members = [row["error_bins"][bin_id] for row in seed_rows]
        item = {"bin": _bin_label(*edge)}
        for key in ("count", "fraction", "mae_before", "mae_after", "delta_mae",
                    "improved_fraction", "worsened_fraction", "unchanged_fraction"):
            values = [float(member[key]) for member in members if key in member]
            if len(values) == len(SEEDS):
                item[key] = mean_std(values)
        aggregated.append(item)
    return aggregated


def aggregate(seed_rows: list[dict], a_rows: list[dict], published_c: dict, published_a: dict) -> dict:
    scalar_keys = (
        "mae_before", "mae_after", "delta_mae", "improved_fraction", "worsened_fraction",
        "unchanged_fraction", "mean_abs_inflow_correction", "mean_inflow_correction",
        "mean_abs_raw_residual", "correction_over_residual", "weight_mean", "weight_std",
        "inflow_formula_max_abs", "projection_reconstruction_max_abs",
    )
    summary = {key: mean_std([row[key] for row in seed_rows]) for key in scalar_keys}
    summary["weight_quantiles"] = {
        name: mean_std([row["weight_quantiles"][name] for row in seed_rows]) for name in seed_rows[0]["weight_quantiles"]
    }
    summary["channel_mean_abs_correction"] = {
        name: mean_std([row["channel_mean_abs_correction"][name] for row in seed_rows]) for name in CHANNELS
    }
    summary["channel_weight_mean"] = {
        name: mean_std([row["channel_weight_mean"][name] for row in seed_rows]) for name in CHANNELS
    }
    summary["error_bins"] = _aggregate_bins(seed_rows)
    a_mae = mean_std([row["mae"] for row in a_rows])
    c_before = summary["mae_before"]["mean"]
    c_after = summary["mae_after"]["mean"]
    a_mean = a_mae["mean"]
    summary["comparison"] = {
        "A_prediction": a_mae,
        "C_before_projection": summary["mae_before"],
        "C_after_projection": summary["mae_after"],
        "A_minus_C_before": a_mean - c_before,
        "projection_delta_mae": c_before - c_after,
        "A_minus_C_after": a_mean - c_after,
        "projection_share_of_A_to_C_after_gap": (c_before - c_after) / (a_mean - c_after),
    }
    summary["published_check"] = {
        "C_reported_mean": published_c["mean"],
        "C_recomputed_after_mean": c_after,
        "C_abs_difference": abs(published_c["mean"] - c_after),
        "C_matches_published_2_4383": abs(c_after - PUBLISHED_C_INFLOW) <= PUBLISHED_ATOL,
        "A_reported_mean": published_a["mean"],
        "A_recomputed_mean": a_mean,
        "A_abs_difference": abs(published_a["mean"] - a_mean),
        "A_matches_published_2_4540": abs(a_mean - PUBLISHED_A_INFLOW) <= PUBLISHED_ATOL,
    }
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def write_report(path: Path, summary: dict, seed_rows: list[dict]) -> None:
    comparison = summary["comparison"]
    check = summary["published_check"]
    lines = [
        "# 原版 C 投影对 Inflow 的审计",
        "",
        "只读取 p05 All2All 已保存的测试预测。五个采样种子各自计算 MAE，再取种子间均值和样本标准差（ddof=1）。",
        f"绝对误差变化不超过 {UNCHANGED_ATOL:g} 辆车的位置记为不变。",
        "",
        "## 五种子 Inflow MAE",
        "",
        "| 采样种子 | 投影前 | 投影后 | ΔMAE（前−后） | 误差减小 | 误差增大 | 不变 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in seed_rows:
        lines.append(
            f"| {row['seed']} | {_fmt(row['mae_before'])} | {_fmt(row['mae_after'])} | {_fmt(row['delta_mae'], 6)} "
            f"| {_fmt(100 * row['improved_fraction'], 2)}% | {_fmt(100 * row['worsened_fraction'], 2)}% "
            f"| {_fmt(100 * row['unchanged_fraction'], 2)}% |"
        )
    before = summary["mae_before"]
    after = summary["mae_after"]
    delta = summary["delta_mae"]
    lines.extend([
        f"| 均值 | {_fmt(before['mean'])} | {_fmt(after['mean'])} | {_fmt(delta['mean'], 6)} "
        f"| {_fmt(100 * summary['improved_fraction']['mean'], 2)}% | {_fmt(100 * summary['worsened_fraction']['mean'], 2)}% "
        f"| {_fmt(100 * summary['unchanged_fraction']['mean'], 2)}% |",
        f"| 标准差 | {_fmt(before['std'], 6)} | {_fmt(after['std'], 6)} | {_fmt(delta['std'], 6)} "
        f"| {_fmt(100 * summary['improved_fraction']['std'], 3)}% | {_fmt(100 * summary['worsened_fraction']['std'], 3)}% "
        f"| {_fmt(100 * summary['unchanged_fraction']['std'], 3)}% |",
        "",
        "## 与已保存测试报告核对",
        "",
        f"- C 投影后重算均值 {check['C_recomputed_after_mean']:.6f}，报告值 {check['C_reported_mean']:.6f}，"
        f"与 2.4383 的差为 {check['C_recomputed_after_mean'] - PUBLISHED_C_INFLOW:+.6f}。",
        f"- A 重算均值 {check['A_recomputed_mean']:.6f}，报告值 {check['A_reported_mean']:.6f}，"
        f"与 2.4540 的差为 {check['A_recomputed_mean'] - PUBLISHED_A_INFLOW:+.6f}。",
        "",
        "## 投影改了什么",
        "",
        f"- Inflow 修正量平均绝对值 {summary['mean_abs_inflow_correction']['mean']:.6f} ± {summary['mean_abs_inflow_correction']['std']:.6f}。",
        f"- 投影前守恒残差平均绝对值 {summary['mean_abs_raw_residual']['mean']:.6f} ± {summary['mean_abs_raw_residual']['std']:.6f}。",
        f"- 二者之比 {summary['correction_over_residual']['mean']:.6f}。公式 `I_final = I_raw + w_I * r_raw` 的最大绝对偏差，五种子均值是 {summary['inflow_formula_max_abs']['mean']:.3e}。",
        f"- Inflow 权重均值 {summary['weight_mean']['mean']:.6f} ± {summary['weight_std']['mean']:.6f}（后一个数是位置分布的种子内标准差，再对种子取均值）。",
        "",
        "| 误差分组（投影前绝对误差） | 位置比例 | 投影前 MAE | 投影后 MAE | ΔMAE |",
        "|---|---:|---:|---:|---:|",
    ])
    for item in summary["error_bins"]:
        lines.append(
            f"| {item['bin']} | {_fmt(100 * item['fraction']['mean'], 2)}% | {_fmt(item['mae_before']['mean'])} "
            f"| {_fmt(item['mae_after']['mean'])} | {_fmt(item['delta_mae']['mean'], 6)} |"
        )
    lines.extend([
        "",
        "## 与无约束 A 的数值对照",
        "",
        "| 实验 | 测试 Inflow MAE |",
        "|---|---:|",
        f"| A 原始预测 | {comparison['A_prediction']['mean']:.4f} ± {comparison['A_prediction']['std']:.4f} |",
        f"| C 投影前 | {comparison['C_before_projection']['mean']:.4f} ± {comparison['C_before_projection']['std']:.4f} |",
        f"| C 投影后 | {comparison['C_after_projection']['mean']:.4f} ± {comparison['C_after_projection']['std']:.4f} |",
        "",
        f"A 减 C 投影后为 {comparison['A_minus_C_after']:.6f}。"
        f"其中 A 减 C 投影前为 {comparison['A_minus_C_before']:.6f}，投影步骤再贡献 {comparison['projection_delta_mae']:.6f}。"
        f"这两项相加等于 A 与 C 投影后的差，投影步骤占这个差的 {100 * comparison['projection_share_of_A_to_C_after_gap']:.1f}%。"
        "这是同一批指标上的数值分解，不是两个独立训练模型之间的因果效应。",
        "",
        "位置数量和 MAE 的方向不一致：五种子平均有 "
        f"{100 * summary['worsened_fraction']['mean']:.2f}% 的位置绝对误差变大，"
        f"{100 * summary['improved_fraction']['mean']:.2f}% 变小。"
        "小于 0.5 辆车的位置占全部样本约四成，这一组的 MAE 上升；10 辆车以上的位置 MAE 下降更大，所以整体 MAE 仍然下降。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def self_check() -> None:
    physical = np.array([[3.0, 1.0, 4.0, 8.0], [0.0, 2.0, 1.0, 1.0]], dtype=np.float64)
    weights = np.array([[0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25]], dtype=np.float64)
    final, residual = project_conservation(physical, weights)
    expected_residual = np.array([8.0 - 4.0 - 3.0 + 1.0, 1.0 - 1.0 - 0.0 + 2.0])
    if not np.allclose(residual, expected_residual):
        raise AssertionError("residual fixture failed")
    if not np.allclose(final[..., 0], physical[..., 0] + weights[..., 0] * residual):
        raise AssertionError("inflow update fixture failed")
    final_residual = final[..., 3] - final[..., 2] - final[..., 0] + final[..., 1]
    if not np.allclose(final_residual, 0.0, atol=1e-12):
        raise AssertionError("projected residual fixture failed")
    stats = mean_std([1.0, 2.0, 3.0, 4.0, 5.0])
    if not np.isclose(stats["mean"], 3.0) or not np.isclose(stats["std"], np.sqrt(2.5)):
        raise AssertionError("five-seed std fixture failed")


def default_runs() -> tuple[Path, Path]:
    root = xuancheng26_training_results_3s_root() / "MSTT"
    projected = root / "conservation_best_backbone" / "projected" / "in_all__out_all" / "p05" / "seed_42"
    unconstrained = root / "ablation_100ep" / "single_tsf_adaptive128_all2a_100ep" / "in_all__out_all" / "p05" / "seed_42"
    return projected, unconstrained


def run(projected_run: Path, unconstrained_run: Path, output_dir: Path) -> dict:
    self_check()
    projected_config = json.loads((projected_run / "resolved_config.json").read_text())
    unconstrained_config = json.loads((unconstrained_run / "resolved_config.json").read_text())
    verify_original_c(projected_config)
    verify_unconstrained_a(unconstrained_config)
    seed_rows = []
    a_rows = []
    for seed in SEEDS:
        projected_path = projected_run / "predictions" / f"seed_{seed}" / "test.npz"
        unconstrained_path = unconstrained_run / "predictions" / f"seed_{seed}" / "test.npz"
        projected_row = audit_projected_seed(projected_path)
        projected_row["seed"] = seed
        unconstrained_row = audit_unconstrained_seed(unconstrained_path)
        unconstrained_row["seed"] = seed
        seed_rows.append(projected_row)
        a_rows.append(unconstrained_row)
    published_c = _published_inflow(projected_run)
    published_a = _published_inflow(unconstrained_run)
    summary = aggregate(seed_rows, a_rows, published_c, published_a)
    gap = summary["comparison"]
    if abs(gap["A_minus_C_before"] + gap["projection_delta_mae"] - gap["A_minus_C_after"]) > 1e-12:
        raise ValueError("A-to-C gap does not split into pre-projection gap plus projection delta")
    if summary["projection_reconstruction_max_abs"]["mean"] > 1e-3:
        raise ValueError("saved final predictions do not match the projection formula")
    if not summary["published_check"]["C_matches_published_2_4383"]:
        raise ValueError("recomputed C inflow MAE does not match 2.4383")
    if not summary["published_check"]["A_matches_published_2_4540"]:
        raise ValueError("recomputed A inflow MAE does not match 2.4540")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "unchanged_atol_vehicles": UNCHANGED_ATOL,
        "seeds": list(SEEDS),
        "projected_run": str(projected_run),
        "unconstrained_run": str(unconstrained_run),
        "per_seed": seed_rows,
        "unconstrained_per_seed": a_rows,
        "summary": summary,
        "published_c": published_c,
        "published_a": published_a,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_csv(output_dir / "per_seed.csv", [
        {
            "seed": row["seed"],
            "mae_before": row["mae_before"],
            "mae_after": row["mae_after"],
            "delta_mae": row["delta_mae"],
            "improved_fraction": row["improved_fraction"],
            "worsened_fraction": row["worsened_fraction"],
            "unchanged_fraction": row["unchanged_fraction"],
            "mean_abs_inflow_correction": row["mean_abs_inflow_correction"],
            "mean_abs_raw_residual": row["mean_abs_raw_residual"],
            "weight_mean": row["weight_mean"],
            "weight_std": row["weight_std"],
            **row["weight_quantiles"],
            "A_mae": a_row["mae"],
        }
        for row, a_row in zip(seed_rows, a_rows)
    ])
    bin_rows = []
    for row in seed_rows:
        for item in row["error_bins"]:
            bin_rows.append({"seed": row["seed"], **item})
    _write_csv(output_dir / "error_bins.csv", bin_rows)
    write_report(output_dir / "report.md", summary, seed_rows)
    return payload


def main() -> None:
    projected_default, unconstrained_default = default_runs()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projected-run", type=Path, default=projected_default)
    parser.add_argument("--unconstrained-run", type=Path, default=unconstrained_default)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("self-check passed")
        return
    payload = run(args.projected_run, args.unconstrained_run, args.output_dir)
    summary = payload["summary"]
    print(
        "C before {before:.6f} after {after:.6f} delta {delta:.6f} A {a_mae:.6f}".format(
            before=summary["mae_before"]["mean"],
            after=summary["mae_after"]["mean"],
            delta=summary["delta_mae"]["mean"],
            a_mae=summary["comparison"]["A_prediction"]["mean"],
        )
    )
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
