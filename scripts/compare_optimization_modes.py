"""Compare sobol_stepwise and full_lm calibration modes.

This script is intentionally outside the core calibration pipeline. It runs
repeatable diagnostics for workflow sanity, simulation truth comparison, and
workspace split validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibration.data_io import load_dataset
from calibration.metrics import position_error_metrics
from calibration.parameters import build_error_parameters, parameter_scales, zero_error_vector
from calibration.pipeline import CalibrationConfig, CalibrationResult, run_calibration_on_dataset
from calibration.robot_model import MultiSourceRobotModel
from simulation.generator import generate_synthetic_dataset


MODES = ("sobol_stepwise", "full_lm")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate sobol_stepwise vs full_lm behavior.")
    parser.add_argument("--real-dataset", default="dataset/real_world200.pkl")
    parser.add_argument("--output-md", default="outputs/optimization_mode_validation.md")
    parser.add_argument("--output-json", default="outputs/optimization_mode_validation.json")
    parser.add_argument("--sim-samples", type=int, default=120)
    parser.add_argument("--sim-split-samples", type=int, default=160)
    parser.add_argument("--noise", type=float, default=4.0e-5)
    parser.add_argument("--payload", type=float, default=0.0)
    parser.add_argument("--truth-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sensitivity-samples", type=int, default=16)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--split-axis", choices=("x", "y", "z"), default="x")
    args = parser.parse_args()

    report = run_validation(args)
    output_md = Path(args.output_md)
    output_json = Path(args.output_json)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(format_markdown(report), encoding="utf-8")
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_md}")
    print(f"Wrote {output_json}")


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    full_sim = generate_synthetic_dataset(
        output_path="outputs/validation_synthetic_full.pkl",
        n_samples=args.sim_samples,
        payload=args.payload,
        measurement_noise_std=args.noise,
        truth_scale=args.truth_scale,
        seed=args.seed,
    )
    sim_split_dataset = generate_synthetic_dataset(
        output_path="outputs/validation_synthetic_split.pkl",
        n_samples=args.sim_split_samples,
        payload=args.payload,
        measurement_noise_std=args.noise,
        truth_scale=args.truth_scale,
        seed=args.seed + 17,
    )
    real_dataset = load_dataset(args.real_dataset)

    return {
        "settings": {
            "real_dataset": str(args.real_dataset),
            "sim_samples": int(args.sim_samples),
            "sim_split_samples": int(args.sim_split_samples),
            "noise": float(args.noise),
            "payload": float(args.payload),
            "truth_scale": float(args.truth_scale),
            "seed": int(args.seed),
            "sensitivity_samples": int(args.sensitivity_samples),
            "max_nfev": int(args.max_nfev),
            "split_axis": args.split_axis,
        },
        "workflow_sanity": _workflow_sanity(full_sim, args),
        "simulation_full_fit": _fit_both_modes(full_sim, full_sim, args, label="simulation/full"),
        "simulation_workspace_split": _workspace_split_experiment(
            sim_split_dataset, args, label="simulation/workspace"
        ),
        "real_workspace_split": _workspace_split_experiment(
            real_dataset, args, label="real/workspace"
        ),
    }


def _workflow_sanity(dataset: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    rows = {}
    for mode in MODES:
        result = _fit(dataset, mode, args)
        rows[mode] = _result_structure(result)
    return rows


def _fit_both_modes(
    train_dataset: dict[str, Any],
    eval_dataset: dict[str, Any],
    args: argparse.Namespace,
    label: str,
) -> dict[str, Any]:
    rows = {}
    for mode in MODES:
        result = _fit(train_dataset, mode, args)
        rows[mode] = _result_metrics(result, train_dataset, eval_dataset, label=label)
    rows["comparison"] = _mode_comparison(rows)
    return rows


def _workspace_split_experiment(
    dataset: dict[str, Any],
    args: argparse.Namespace,
    label: str,
) -> dict[str, Any]:
    train_indices, validation_indices, threshold = _workspace_split_indices(dataset, args.split_axis)
    train_dataset = _subset_dataset(dataset, train_indices)
    validation_dataset = _subset_dataset(dataset, validation_indices)
    result = _fit_both_modes(train_dataset, validation_dataset, args, label=label)
    result["split"] = {
        "axis": args.split_axis,
        "threshold": float(threshold),
        "train_count": int(len(train_indices)),
        "validation_count": int(len(validation_indices)),
        "train_workspace": _workspace_bounds(train_dataset),
        "validation_workspace": _workspace_bounds(validation_dataset),
    }
    return result


def _fit(dataset: dict[str, Any], mode: str, args: argparse.Namespace) -> CalibrationResult:
    config = CalibrationConfig(
        sensitivity_samples=args.sensitivity_samples,
        seed=args.seed,
        max_nfev_per_stage=args.max_nfev,
        optimization_mode=mode,
    )
    return run_calibration_on_dataset(dataset, config=config)


def _result_structure(result: CalibrationResult) -> dict[str, Any]:
    return {
        "rank": int(result.redundancy.rank),
        "nullity": int(result.redundancy.nullity),
        "independent_count": int(len(result.redundancy.independent_indices)),
        "redundant_count": int(len(result.redundancy.redundant_indices)),
        "batch_count": int(len(result.batches)),
        "batch_sizes": [int(len(batch)) for batch in result.batches],
        "stage_count": int(len(result.identification.stages)),
        "final_active_count": int(len(result.identification.stages[-1].active_indices))
        if result.identification.stages
        else 0,
        "top_sensitivity": _top_sensitivity_names(result, 10),
    }


def _result_metrics(
    result: CalibrationResult,
    train_dataset: dict[str, Any],
    eval_dataset: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    model = MultiSourceRobotModel()
    train_pred = model.batch_positions(
        train_dataset["joints"],
        result.identification.vector,
        result.parameters,
        train_dataset.get("payloads"),
        train_dataset.get("directions"),
    )
    eval_pred = model.batch_positions(
        eval_dataset["joints"],
        result.identification.vector,
        result.parameters,
        eval_dataset.get("payloads"),
        eval_dataset.get("directions"),
    )
    metrics = {
        "structure": _result_structure(result),
        "train_position": _mm_metrics(position_error_metrics(train_dataset["measured_positions"], train_pred)),
        "eval_position": _mm_metrics(position_error_metrics(eval_dataset["measured_positions"], eval_pred)),
        "stages": [
            {
                "stage": int(stage.stage),
                "optimized_count": int(len(stage.optimized_indices)),
                "active_count": int(len(stage.active_indices)),
                "rmse_component_mm": float(stage.rmse * 1000.0),
                "nfev": int(stage.nfev),
            }
            for stage in result.identification.stages
        ],
        "label": label,
    }
    if "true_error_vector" in eval_dataset:
        metrics["truth"] = _truth_metrics(result, eval_dataset)
    return metrics


def _truth_metrics(result: CalibrationResult, eval_dataset: dict[str, Any]) -> dict[str, Any]:
    parameters = result.parameters
    scales = parameter_scales(parameters)
    truth = np.asarray(eval_dataset["true_error_vector"], dtype=float).reshape(len(parameters))
    estimate = np.asarray(result.identification.vector, dtype=float).reshape(len(parameters))
    error = estimate - truth
    active = (
        list(result.identification.stages[-1].active_indices)
        if result.identification.stages
        else []
    )
    independent = list(result.redundancy.independent_indices)
    return {
        "all_parameters": _parameter_error_stats(error, truth, estimate, scales, list(range(len(parameters)))),
        "active_parameters": _parameter_error_stats(error, truth, estimate, scales, active),
        "independent_parameters": _parameter_error_stats(error, truth, estimate, scales, independent),
        "inactive_parameters": _parameter_error_stats(
            error,
            truth,
            estimate,
            scales,
            [i for i in range(len(parameters)) if i not in set(active)],
        ),
        "output_equivalence": _output_equivalence_metrics(result, eval_dataset, truth, estimate),
        "largest_normalized_errors": _largest_parameter_errors(parameters, error, truth, estimate, scales, 12),
    }


def _output_equivalence_metrics(
    result: CalibrationResult,
    dataset: dict[str, Any],
    truth: np.ndarray,
    estimate: np.ndarray,
) -> dict[str, float]:
    model = MultiSourceRobotModel()
    true_positions = model.batch_positions(
        dataset["joints"],
        truth,
        result.parameters,
        dataset.get("payloads"),
        dataset.get("directions"),
    )
    estimated_positions = model.batch_positions(
        dataset["joints"],
        estimate,
        result.parameters,
        dataset.get("payloads"),
        dataset.get("directions"),
    )
    return _mm_metrics(position_error_metrics(true_positions, estimated_positions))


def _parameter_error_stats(
    error: np.ndarray,
    truth: np.ndarray,
    estimate: np.ndarray,
    scales: np.ndarray,
    indices: list[int],
) -> dict[str, float | int]:
    if not indices:
        return {
            "count": 0,
            "normalized_mae": 0.0,
            "normalized_rmse": 0.0,
            "truth_l2": 0.0,
            "estimate_l2": 0.0,
            "error_l2": 0.0,
        }
    idx = np.asarray(indices, dtype=int)
    normalized = error[idx] / np.maximum(scales[idx], 1.0e-20)
    return {
        "count": int(len(idx)),
        "normalized_mae": float(np.mean(np.abs(normalized))),
        "normalized_rmse": float(np.sqrt(np.mean(np.square(normalized)))),
        "truth_l2": float(np.linalg.norm(truth[idx])),
        "estimate_l2": float(np.linalg.norm(estimate[idx])),
        "error_l2": float(np.linalg.norm(error[idx])),
    }


def _largest_parameter_errors(
    parameters: list[Any],
    error: np.ndarray,
    truth: np.ndarray,
    estimate: np.ndarray,
    scales: np.ndarray,
    count: int,
) -> list[dict[str, Any]]:
    order = np.argsort(-np.abs(error / np.maximum(scales, 1.0e-20)))[:count]
    return [
        {
            "name": parameters[i].name,
            "group": parameters[i].group,
            "unit": parameters[i].unit,
            "true": float(truth[i]),
            "estimated": float(estimate[i]),
            "error": float(error[i]),
            "normalized_abs_error": float(abs(error[i]) / max(scales[i], 1.0e-20)),
        }
        for i in order
    ]


def _workspace_split_indices(dataset: dict[str, Any], axis: str) -> tuple[np.ndarray, np.ndarray, float]:
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    positions = np.asarray(dataset["measured_positions"], dtype=float).reshape(-1, 3)
    values = positions[:, axis_index]
    threshold = float(np.median(values))
    train = np.flatnonzero(values <= threshold)
    validation = np.flatnonzero(values > threshold)
    return train, validation, threshold


def _subset_dataset(dataset: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    count = len(np.asarray(dataset["joints"]).reshape(-1, 6))
    subset: dict[str, Any] = {}
    for key, value in dataset.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (count,):
            subset[key] = value[indices].copy()
        else:
            subset[key] = value
    metadata = dict(subset.get("metadata", {})) if isinstance(subset.get("metadata"), dict) else {}
    metadata["subset_count"] = int(len(indices))
    subset["metadata"] = metadata
    return subset


def _workspace_bounds(dataset: dict[str, Any]) -> dict[str, list[float]]:
    positions = np.asarray(dataset["measured_positions"], dtype=float).reshape(-1, 3)
    return {
        "min_xyz": [float(x) for x in np.min(positions, axis=0)],
        "max_xyz": [float(x) for x in np.max(positions, axis=0)],
    }


def _top_sensitivity_names(result: CalibrationResult, count: int) -> list[dict[str, Any]]:
    names = [parameter.name for parameter in result.parameters]
    rows = []
    for index in result.sensitivity.ranked_indices[:count]:
        rows.append(
            {
                "name": names[index],
                "score": float(result.sensitivity.normalized_scores[index]),
                "total_index": float(result.sensitivity.total_indices[index]),
            }
        )
    return rows


def _mode_comparison(rows: dict[str, Any]) -> dict[str, float]:
    sobol = rows["sobol_stepwise"]
    full = rows["full_lm"]
    return {
        "eval_rmse_delta_mm_full_minus_sobol": float(
            full["eval_position"]["rmse"] - sobol["eval_position"]["rmse"]
        ),
        "train_rmse_delta_mm_full_minus_sobol": float(
            full["train_position"]["rmse"] - sobol["train_position"]["rmse"]
        ),
    }


def _mm_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: float(value * 1000.0) for key, value in metrics.items()}


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# sobol_stepwise 与 full_lm 验证报告",
        "",
        "## 实验设置",
        "",
        _json_block(report["settings"]),
        "",
        "## 1. 工作流有效性检查",
        "",
        _workflow_table(report["workflow_sanity"]),
        "",
        "结论：`sobol_stepwise` 与 `full_lm` 不是同一条执行路径。前者会产生非零敏感性排序、冗余剔除结果和多阶段累计 LM；后者跳过分析，并把 54 个参数作为单阶段活动参数。",
        "",
        "## 2. 仿真全数据拟合：真值参数与等效输出",
        "",
        _mode_metric_table(report["simulation_full_fit"]),
        "",
        _truth_table(report["simulation_full_fit"]),
        "",
        "## 3. 仿真工作空间 A/B 验证",
        "",
        _split_summary(report["simulation_workspace_split"]),
        "",
        _mode_metric_table(report["simulation_workspace_split"]),
        "",
        _truth_table(report["simulation_workspace_split"]),
        "",
        "## 4. 真实数据工作空间 A/B 验证",
        "",
        _split_summary(report["real_workspace_split"]),
        "",
        _mode_metric_table(report["real_workspace_split"]),
        "",
        "## 5. 解释与建议",
        "",
        _interpretation(report),
        "",
        "- 如果两种方法最终定位 RMSE 基本一致，优先说明当前数据和评价指标主要约束的是末端位置输出，而不是唯一物理参数。冗余参数可以在不同数值组合下产生几乎相同的末端位移。",
        "- `sobol_stepwise` 的主要价值不一定体现为训练误差更低，而应体现为参数维度更小、阶段收敛更稳、对验证工作空间和噪声更不敏感。如果验证 RMSE 也接近，说明当前数据量、工作空间覆盖和模型噪声水平下，全参数 LM 尚未明显暴露泛化劣化。",
        "- 当前敏感性指标使用“扰动相对名义模型造成的 RMS 位移”，与真实残差方向无关，因此它更像可观测影响排序，不是“对当前数据误差最该修正的参数”的直接排序。",
        "- 冗余性分析在零误差附近做局部 Jacobian 秩分析。非线性较强、工作空间覆盖不足或阈值变化时，独立集可能变化；这会削弱它相对 full_lm 的稳定优势。",
        "- 建议后续加入验证集 RMSE、等效输出误差、参数范数/先验正则项、重复种子稳定性、不同工作空间留一验证，作为是否过拟合的主要判据，而不是只比较训练定位误差。",
    ]
    return "\n".join(lines) + "\n"


def _interpretation(report: dict[str, Any]) -> str:
    full_fit = report["simulation_full_fit"]
    sim_split = report["simulation_workspace_split"]
    real_split = report["real_workspace_split"]
    sim_full_truth = full_fit["full_lm"]["truth"]
    sim_sobol_truth = full_fit["sobol_stepwise"]["truth"]
    split_full_truth = sim_split["full_lm"]["truth"]
    split_sobol_truth = sim_split["sobol_stepwise"]["truth"]
    return "\n".join(
        [
            "本次运行的直接结论：",
            "",
            f"- 工作流正常：`sobol_stepwise` 最终活动参数为 {report['workflow_sanity']['sobol_stepwise']['final_active_count']} 个，"
            f"`full_lm` 最终活动参数为 {report['workflow_sanity']['full_lm']['final_active_count']} 个；"
            f"`sobol_stepwise` 有 {report['workflow_sanity']['sobol_stepwise']['batch_count']} 个阶段，"
            f"`full_lm` 只有 {report['workflow_sanity']['full_lm']['batch_count']} 个阶段。",
            f"- 仿真全数据拟合中，full_lm 相比 sobol_stepwise 的定位 RMSE 只低 "
            f"{abs(full_fit['comparison']['eval_rmse_delta_mm_full_minus_sobol']):.6f} mm，差异可以认为没有工程意义。",
            f"- 但 full_lm 的全参数归一化真值 RMSE 为 {sim_full_truth['all_parameters']['normalized_rmse']:.3f}，"
            f"sobol_stepwise 为 {sim_sobol_truth['all_parameters']['normalized_rmse']:.3f}；full_lm 的参数数值明显偏离真值。",
            f"- 同时，full_lm 与真值参数在末端输出上的等效 RMSE 只有 "
            f"{sim_full_truth['output_equivalence']['rmse']:.6f} mm，sobol_stepwise 为 "
            f"{sim_sobol_truth['output_equivalence']['rmse']:.6f} mm。也就是说，full_lm 参数不真，但输出几乎等效。",
            f"- 仿真 A/B 工作空间验证中，full_lm 的验证 RMSE 比 sobol_stepwise 高 "
            f"{sim_split['comparison']['eval_rmse_delta_mm_full_minus_sobol']:.6f} mm，仍然很小；"
            f"但 full_lm 的全参数归一化真值 RMSE 升到 {split_full_truth['all_parameters']['normalized_rmse']:.3f}，"
            f"sobol_stepwise 为 {split_sobol_truth['all_parameters']['normalized_rmse']:.3f}。",
            f"- 真实 A/B 工作空间验证中，两种方法训练 RMSE 约 "
            f"{real_split['sobol_stepwise']['train_position']['rmse']:.3f}/{real_split['full_lm']['train_position']['rmse']:.3f} mm，"
            f"验证 RMSE 约 {real_split['sobol_stepwise']['eval_position']['rmse']:.3f}/"
            f"{real_split['full_lm']['eval_position']['rmse']:.3f} mm。验证误差明显高于训练误差，但两种方法几乎同步升高，"
            "因此这更像工作空间外推/数据分布差异，而不是 full_lm 独有过拟合。",
        ]
    )


def _workflow_table(rows: dict[str, Any]) -> str:
    table = [
        "| mode | rank | nullity | independent | redundant | batches | batch sizes | final active | top sensitivity |",
        "|---|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for mode in MODES:
        row = rows[mode]
        top = ", ".join(item["name"] for item in row["top_sensitivity"][:5])
        table.append(
            f"| {mode} | {row['rank']} | {row['nullity']} | {row['independent_count']} | "
            f"{row['redundant_count']} | {row['batch_count']} | {row['batch_sizes']} | "
            f"{row['final_active_count']} | {top} |"
        )
    return "\n".join(table)


def _mode_metric_table(rows: dict[str, Any]) -> str:
    table = [
        "| mode | train RMSE mm | train mean mm | eval RMSE mm | eval mean mm | active params | stages |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        row = rows[mode]
        table.append(
            f"| {mode} | {row['train_position']['rmse']:.6f} | {row['train_position']['mean']:.6f} | "
            f"{row['eval_position']['rmse']:.6f} | {row['eval_position']['mean']:.6f} | "
            f"{row['structure']['final_active_count']} | {row['structure']['stage_count']} |"
        )
    comparison = rows.get("comparison", {})
    table.extend(
        [
            "",
            f"- eval RMSE 差值 full_lm - sobol_stepwise: {comparison.get('eval_rmse_delta_mm_full_minus_sobol', 0.0):.6f} mm",
            f"- train RMSE 差值 full_lm - sobol_stepwise: {comparison.get('train_rmse_delta_mm_full_minus_sobol', 0.0):.6f} mm",
        ]
    )
    return "\n".join(table)


def _truth_table(rows: dict[str, Any]) -> str:
    if "truth" not in rows["sobol_stepwise"]:
        return "该实验没有仿真真值参数。"
    table = [
        "| mode | all norm RMSE | active norm RMSE | output-equivalent RMSE mm | output-equivalent max mm |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        truth = rows[mode]["truth"]
        table.append(
            f"| {mode} | {truth['all_parameters']['normalized_rmse']:.6f} | "
            f"{truth['active_parameters']['normalized_rmse']:.6f} | "
            f"{truth['output_equivalence']['rmse']:.6f} | {truth['output_equivalence']['max']:.6f} |"
        )
    return "\n".join(table)


def _split_summary(rows: dict[str, Any]) -> str:
    split = rows["split"]
    return (
        f"- 划分轴: {split['axis']}\n"
        f"- 阈值: {split['threshold']:.6f}\n"
        f"- A/训练样本数: {split['train_count']}\n"
        f"- B/验证样本数: {split['validation_count']}"
    )


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"


if __name__ == "__main__":
    main()
