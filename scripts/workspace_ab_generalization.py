"""A/B workspace generalization experiment for full_lm vs sobol_stepwise.

The script intentionally stays outside the core calibration algorithms. It
generates two synthetic workspaces with the same true error vector, identifies
parameters on workspace A, validates on workspace B, and reports both position
accuracy and parameter deviation from truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calibration.data_io import save_dataset
from calibration.metrics import position_error_metrics
from calibration.parameters import (
    build_error_parameters,
    parameter_scales,
    sample_truth_vector,
    vector_to_named_dict,
    zero_error_vector,
)
from calibration.pipeline import CalibrationConfig, CalibrationResult, run_calibration_on_dataset
from calibration.robot_model import MultiSourceRobotModel
from config.nominal_config import NOMINAL_ROBOT


MODES = ("sobol_stepwise", "full_lm")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train full_lm and sobol_stepwise in workspace A, validate in workspace B."
    )
    parser.add_argument("--output-dir", default="outputs/workspace_ab_generalization")
    parser.add_argument("--n-a", type=int, default=150, help="workspace A sample count")
    parser.add_argument("--n-b", type=int, default=150, help="workspace B sample count")
    parser.add_argument("--noise", type=float, default=4.0e-5, help="measurement noise std, meters")
    parser.add_argument("--payload", type=float, default=0.0)
    parser.add_argument(
        "--truth-scale",
        type=float,
        default=2.0,
        help="scale of sampled true error vector; 2.0 is about twice the default.",
    )
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument("--sensitivity-samples", type=int, default=24)
    parser.add_argument("--max-nfev", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parameters = build_error_parameters()
    rng = np.random.default_rng(args.seed)
    truth = sample_truth_vector(rng, parameters, sigma_scale=args.truth_scale)
    dataset_a = _make_workspace_dataset(
        workspace="A",
        n_samples=args.n_a,
        truth=truth,
        seed=args.seed + 1,
        payload=args.payload,
        noise=args.noise,
    )
    dataset_b = _make_workspace_dataset(
        workspace="B",
        n_samples=args.n_b,
        truth=truth,
        seed=args.seed + 2,
        payload=args.payload,
        noise=args.noise,
    )
    save_dataset(output_dir / "workspace_A.pkl", dataset_a)
    save_dataset(output_dir / "workspace_B.pkl", dataset_b)

    figure_path = output_dir / "workspace_ab_positions.png"
    separation = _plot_workspace_distribution(dataset_a, dataset_b, figure_path)

    rows: dict[str, Any] = {}
    for mode in MODES:
        result = _fit(dataset_a, mode, args)
        rows[mode] = _evaluate_mode(result, dataset_a, dataset_b, truth)
    rows["comparison"] = _comparison(rows)

    report = {
        "settings": {
            "n_a": int(args.n_a),
            "n_b": int(args.n_b),
            "noise_m": float(args.noise),
            "payload_kg": float(args.payload),
            "truth_scale": float(args.truth_scale),
            "seed": int(args.seed),
            "sensitivity_samples": int(args.sensitivity_samples),
            "max_nfev_per_stage": int(args.max_nfev),
            "workspace_definition": _workspace_definitions(),
        },
        "artifacts": {
            "dataset_a": str((output_dir / "workspace_A.pkl").resolve()),
            "dataset_b": str((output_dir / "workspace_B.pkl").resolve()),
            "figure": str(figure_path.resolve()),
            "json": str((output_dir / "report.json").resolve()),
            "markdown": str((output_dir / "report.md").resolve()),
        },
        "workspace_separation": separation,
        "modes": rows,
    }

    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(f"Wrote {(output_dir / 'report.md').resolve()}")
    print(f"Wrote {(output_dir / 'report.json').resolve()}")
    print(f"Wrote {figure_path.resolve()}")


def _workspace_definitions() -> dict[str, list[list[float]]]:
    return {
        "A_joint_ranges_rad": [
            [-2.90, -1.70],
            [-2.40, -1.80],
            [-2.60, -1.80],
            [-3.10, -1.20],
            [-2.50, -0.30],
            [-np.pi, np.pi],
        ],
        "B_joint_ranges_rad": [
            [1.70, 2.90],
            [-0.85, -0.40],
            [-0.45, 0.20],
            [1.20, 3.10],
            [0.30, 2.50],
            [-np.pi, np.pi],
        ],
    }


def _make_workspace_dataset(
    workspace: str,
    n_samples: int,
    truth: np.ndarray,
    seed: int,
    payload: float,
    noise: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    parameters = build_error_parameters()
    joints = _sample_workspace_joints(rng, n_samples, workspace)
    directions = rng.choice([-1.0, 1.0], size=(n_samples, 6))
    payloads = np.full(n_samples, float(payload), dtype=float)
    model = MultiSourceRobotModel()
    true_positions = model.batch_positions(joints, truth, parameters, payloads, directions)
    measured = true_positions + rng.normal(0.0, noise, size=true_positions.shape)
    nominal_positions = model.batch_positions(
        joints, zero_error_vector(parameters), parameters, payloads, directions
    )
    return {
        "joints": joints,
        "measured_positions": measured,
        "payloads": payloads,
        "directions": directions,
        "nominal_positions": nominal_positions,
        "true_positions": true_positions,
        "parameter_names": [p.name for p in parameters],
        "parameter_groups": [p.group for p in parameters],
        "true_error_vector": truth.copy(),
        "true_error_parameters": vector_to_named_dict(truth, parameters),
        "nominal_robot": NOMINAL_ROBOT,
        "metadata": {
            "workspace": workspace,
            "n_samples": int(n_samples),
            "payload": float(payload),
            "measurement_noise_std": float(noise),
            "seed": int(seed),
        },
    }


def _sample_workspace_joints(
    rng: np.random.Generator, n_samples: int, workspace: str
) -> np.ndarray:
    definitions = _workspace_definitions()
    key = "A_joint_ranges_rad" if workspace == "A" else "B_joint_ranges_rad"
    ranges = np.asarray(definitions[key], dtype=float)
    nominal_limits = np.asarray(NOMINAL_ROBOT["joint_limits"], dtype=float)
    lower = np.maximum(ranges[:, 0], nominal_limits[:, 0])
    upper = np.minimum(ranges[:, 1], nominal_limits[:, 1])
    return rng.uniform(lower, upper, size=(n_samples, 6))


def _fit(dataset: dict[str, Any], mode: str, args: argparse.Namespace) -> CalibrationResult:
    config = CalibrationConfig(
        sensitivity_samples=args.sensitivity_samples,
        max_nfev_per_stage=args.max_nfev,
        seed=args.seed,
        optimization_mode=mode,
    )
    return run_calibration_on_dataset(dataset, config=config)


def _evaluate_mode(
    result: CalibrationResult,
    dataset_a: dict[str, Any],
    dataset_b: dict[str, Any],
    truth: np.ndarray,
) -> dict[str, Any]:
    model = MultiSourceRobotModel()
    estimate = np.asarray(result.identification.vector, dtype=float)
    pred_a = model.batch_positions(
        dataset_a["joints"],
        estimate,
        result.parameters,
        dataset_a.get("payloads"),
        dataset_a.get("directions"),
    )
    pred_b = model.batch_positions(
        dataset_b["joints"],
        estimate,
        result.parameters,
        dataset_b.get("payloads"),
        dataset_b.get("directions"),
    )
    true_pred_b = model.batch_positions(
        dataset_b["joints"],
        truth,
        result.parameters,
        dataset_b.get("payloads"),
        dataset_b.get("directions"),
    )
    estimated_true_pred_b = model.batch_positions(
        dataset_b["joints"],
        estimate,
        result.parameters,
        dataset_b.get("payloads"),
        dataset_b.get("directions"),
    )
    return {
        "structure": {
            "rank": int(result.redundancy.rank),
            "nullity": int(result.redundancy.nullity),
            "independent_count": int(len(result.redundancy.independent_indices)),
            "redundant_count": int(len(result.redundancy.redundant_indices)),
            "stage_count": int(len(result.identification.stages)),
            "batch_sizes": [int(len(batch)) for batch in result.batches],
            "final_active_count": int(len(result.identification.stages[-1].active_indices)),
        },
        "train_A_position_mm": _mm(position_error_metrics(dataset_a["measured_positions"], pred_a)),
        "validation_B_position_mm": _mm(position_error_metrics(dataset_b["measured_positions"], pred_b)),
        "B_output_equivalence_to_truth_mm": _mm(
            position_error_metrics(true_pred_b, estimated_true_pred_b)
        ),
        "parameter_error": _parameter_error(result, truth, estimate),
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
    }


def _parameter_error(
    result: CalibrationResult, truth: np.ndarray, estimate: np.ndarray
) -> dict[str, Any]:
    scales = parameter_scales(result.parameters)
    error = estimate - truth
    normalized = error / np.maximum(scales, 1.0e-20)
    active = set(result.identification.stages[-1].active_indices)
    inactive_indices = [i for i in range(len(result.parameters)) if i not in active]
    return {
        "all_normalized_mae": float(np.mean(np.abs(normalized))),
        "all_normalized_rmse": float(np.sqrt(np.mean(np.square(normalized)))),
        "active_normalized_rmse": _indexed_norm_rmse(normalized, list(active)),
        "inactive_normalized_rmse": _indexed_norm_rmse(normalized, inactive_indices),
        "truth_l2": float(np.linalg.norm(truth)),
        "estimate_l2": float(np.linalg.norm(estimate)),
        "error_l2": float(np.linalg.norm(error)),
        "largest_normalized_errors": _largest_errors(result, truth, estimate, normalized),
    }


def _indexed_norm_rmse(values: np.ndarray, indices: list[int]) -> float:
    if not indices:
        return 0.0
    selected = values[np.asarray(indices, dtype=int)]
    return float(np.sqrt(np.mean(np.square(selected))))


def _largest_errors(
    result: CalibrationResult,
    truth: np.ndarray,
    estimate: np.ndarray,
    normalized_error: np.ndarray,
    count: int = 12,
) -> list[dict[str, Any]]:
    order = np.argsort(-np.abs(normalized_error))[:count]
    return [
        {
            "name": result.parameters[i].name,
            "group": result.parameters[i].group,
            "unit": result.parameters[i].unit,
            "true": float(truth[i]),
            "estimated": float(estimate[i]),
            "error": float(estimate[i] - truth[i]),
            "normalized_abs_error": float(abs(normalized_error[i])),
        }
        for i in order
    ]


def _plot_workspace_distribution(
    dataset_a: dict[str, Any], dataset_b: dict[str, Any], path: Path
) -> dict[str, Any]:
    pos_a = np.asarray(dataset_a["measured_positions"], dtype=float)
    pos_b = np.asarray(dataset_b["measured_positions"], dtype=float)
    separation = _workspace_separation(pos_a, pos_b)

    fig = plt.figure(figsize=(12, 10))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.scatter(pos_a[:, 0], pos_a[:, 1], pos_a[:, 2], s=16, alpha=0.75, label="A train")
    ax3d.scatter(pos_b[:, 0], pos_b[:, 1], pos_b[:, 2], s=16, alpha=0.75, label="B validation")
    ax3d.set_xlabel("X m")
    ax3d.set_ylabel("Y m")
    ax3d.set_zlabel("Z m")
    ax3d.legend(loc="best")
    ax3d.set_title("End-effector positions")

    for subplot, axes, title in (
        (2, (0, 1), "XY"),
        (3, (0, 2), "XZ"),
        (4, (1, 2), "YZ"),
    ):
        ax = fig.add_subplot(2, 2, subplot)
        ax.scatter(pos_a[:, axes[0]], pos_a[:, axes[1]], s=16, alpha=0.75, label="A train")
        ax.scatter(pos_b[:, axes[0]], pos_b[:, axes[1]], s=16, alpha=0.75, label="B validation")
        ax.set_xlabel("XYZ"[axes[0]] + " m")
        ax.set_ylabel("XYZ"[axes[1]] + " m")
        ax.set_title(title)
        ax.grid(True, linewidth=0.4, alpha=0.35)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return separation


def _workspace_separation(pos_a: np.ndarray, pos_b: np.ndarray) -> dict[str, Any]:
    min_a, max_a = np.min(pos_a, axis=0), np.max(pos_a, axis=0)
    min_b, max_b = np.min(pos_b, axis=0), np.max(pos_b, axis=0)
    overlap = np.maximum(0.0, np.minimum(max_a, max_b) - np.maximum(min_a, min_b))
    union = np.maximum(max_a, max_b) - np.minimum(min_a, min_b)
    distances = cdist(pos_a, pos_b)
    return {
        "A_bounds_m": {"min_xyz": min_a.tolist(), "max_xyz": max_a.tolist()},
        "B_bounds_m": {"min_xyz": min_b.tolist(), "max_xyz": max_b.tolist()},
        "centroid_A_m": np.mean(pos_a, axis=0).tolist(),
        "centroid_B_m": np.mean(pos_b, axis=0).tolist(),
        "centroid_distance_m": float(np.linalg.norm(np.mean(pos_a, axis=0) - np.mean(pos_b, axis=0))),
        "bbox_overlap_ratio_xyz": (overlap / np.maximum(union, 1.0e-20)).tolist(),
        "nearest_cross_workspace_distance_m": {
            "mean": float(np.mean(np.min(distances, axis=1))),
            "min": float(np.min(distances)),
            "p05": float(np.percentile(np.min(distances, axis=1), 5)),
        },
    }


def _comparison(rows: dict[str, Any]) -> dict[str, float]:
    sobol = rows["sobol_stepwise"]
    full = rows["full_lm"]
    return {
        "B_rmse_delta_full_minus_sobol_mm": float(
            full["validation_B_position_mm"]["rmse"] - sobol["validation_B_position_mm"]["rmse"]
        ),
        "A_rmse_delta_full_minus_sobol_mm": float(
            full["train_A_position_mm"]["rmse"] - sobol["train_A_position_mm"]["rmse"]
        ),
        "truth_norm_rmse_delta_full_minus_sobol": float(
            full["parameter_error"]["all_normalized_rmse"]
            - sobol["parameter_error"]["all_normalized_rmse"]
        ),
        "B_output_equiv_rmse_delta_full_minus_sobol_mm": float(
            full["B_output_equivalence_to_truth_mm"]["rmse"]
            - sobol["B_output_equivalence_to_truth_mm"]["rmse"]
        ),
    }


def _mm(metrics: dict[str, float]) -> dict[str, float]:
    return {key: float(value * 1000.0) for key, value in metrics.items()}


def _markdown(report: dict[str, Any]) -> str:
    modes = report["modes"]
    comparison = modes["comparison"]
    sep = report["workspace_separation"]
    figure = report["artifacts"]["figure"]
    figure_markdown = figure.replace("\\", "/")
    lines = [
        "# A/B 工作空间泛化验证：sobol_stepwise vs full_lm",
        "",
        "## 实验设置",
        "",
        f"- A 训练样本数：{report['settings']['n_a']}",
        f"- B 验证样本数：{report['settings']['n_b']}",
        f"- 观测噪声标准差：{report['settings']['noise_m'] * 1000.0:.6f} mm",
        f"- 真值误差尺度：{report['settings']['truth_scale']:.2f}x",
        f"- Sobol 样本数：{report['settings']['sensitivity_samples']}",
        f"- 每阶段最大函数评估：{report['settings']['max_nfev_per_stage']}",
        "",
        "## A/B 末端位置分布",
        "",
        f"![A/B workspace]({figure_markdown})",
        "",
        f"- A/B 质心距离：{sep['centroid_distance_m']:.6f} m",
        "- 包围盒重叠比例 xyz："
        + ", ".join(f"{x:.3f}" for x in sep["bbox_overlap_ratio_xyz"]),
        f"- 跨空间最近距离均值：{sep['nearest_cross_workspace_distance_m']['mean']:.6f} m",
        f"- 跨空间最近距离最小值：{sep['nearest_cross_workspace_distance_m']['min']:.6f} m",
        "",
        "## 定位精度",
        "",
        "| mode | A train RMSE mm | A train mean mm | B validation RMSE mm | B validation mean mm | active params | stages |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        row = modes[mode]
        lines.append(
            f"| {mode} | {row['train_A_position_mm']['rmse']:.6f} | "
            f"{row['train_A_position_mm']['mean']:.6f} | "
            f"{row['validation_B_position_mm']['rmse']:.6f} | "
            f"{row['validation_B_position_mm']['mean']:.6f} | "
            f"{row['structure']['final_active_count']} | {row['structure']['stage_count']} |"
        )
    lines.extend(
        [
            "",
            f"- B 验证 RMSE 差值 full_lm - sobol_stepwise：{comparison['B_rmse_delta_full_minus_sobol_mm']:.6f} mm",
            f"- A 训练 RMSE 差值 full_lm - sobol_stepwise：{comparison['A_rmse_delta_full_minus_sobol_mm']:.6f} mm",
            "",
            "## 参数与真值差异",
            "",
            "| mode | all norm RMSE | active norm RMSE | inactive norm RMSE | B output-equivalent RMSE mm | B output-equivalent max mm |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        row = modes[mode]
        perr = row["parameter_error"]
        equiv = row["B_output_equivalence_to_truth_mm"]
        lines.append(
            f"| {mode} | {perr['all_normalized_rmse']:.6f} | "
            f"{perr['active_normalized_rmse']:.6f} | {perr['inactive_normalized_rmse']:.6f} | "
            f"{equiv['rmse']:.6f} | {equiv['max']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- 参数归一化 RMSE 差值 full_lm - sobol_stepwise：{comparison['truth_norm_rmse_delta_full_minus_sobol']:.6f}",
            f"- B 输出等效 RMSE 差值 full_lm - sobol_stepwise：{comparison['B_output_equiv_rmse_delta_full_minus_sobol_mm']:.6f} mm",
            "",
            "## 结论",
            "",
            _conclusion(report),
            "",
            "## 产物",
            "",
            f"- JSON：`{report['artifacts']['json']}`",
            f"- Markdown：`{report['artifacts']['markdown']}`",
            f"- A 数据集：`{report['artifacts']['dataset_a']}`",
            f"- B 数据集：`{report['artifacts']['dataset_b']}`",
            f"- 分布图：`{figure}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _conclusion(report: dict[str, Any]) -> str:
    modes = report["modes"]
    delta_b = modes["comparison"]["B_rmse_delta_full_minus_sobol_mm"]
    delta_truth = modes["comparison"]["truth_norm_rmse_delta_full_minus_sobol"]
    sobol_b = modes["sobol_stepwise"]["validation_B_position_mm"]["rmse"]
    full_b = modes["full_lm"]["validation_B_position_mm"]["rmse"]
    sobol_truth = modes["sobol_stepwise"]["parameter_error"]["all_normalized_rmse"]
    full_truth = modes["full_lm"]["parameter_error"]["all_normalized_rmse"]
    if abs(delta_b) < 0.01 and delta_truth > 1.0:
        diagnosis = (
            "两种算法在 B 空间的定位 RMSE 仍然非常接近，但 full_lm 的参数真值偏差显著更大。"
            "这更支持“参数冗余性导致多解”：不同参数组合能产生几乎等效的末端位置输出，"
            "而不是 full_lm 在 A 空间局部过拟合后到 B 空间明显失效。"
        )
    elif abs(delta_b) >= 0.01 and delta_truth > 1.0:
        diagnosis = (
            "full_lm 在 B 空间的定位误差和参数真值偏差都更大，说明当前 A/B 差异下已经出现"
            "可观测的局部区域过拟合或冗余方向漂移。"
        )
    else:
        diagnosis = (
            "本次设置下定位误差和参数误差差距都不够明显，不能把差异明确归因到冗余多解或局部过拟合。"
        )
    return (
        f"B 验证 RMSE：sobol_stepwise={sobol_b:.6f} mm，full_lm={full_b:.6f} mm；"
        f"全参数归一化真值 RMSE：sobol_stepwise={sobol_truth:.6f}，full_lm={full_truth:.6f}。"
        + diagnosis
    )


if __name__ == "__main__":
    main()
