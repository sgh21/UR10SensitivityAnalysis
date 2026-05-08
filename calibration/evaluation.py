"""Detailed evaluation helpers kept separate from calibration logic."""

from __future__ import annotations

from typing import Any

import numpy as np

from calibration.parameters import ErrorParameter, parameter_scales
from calibration.robot_model import MultiSourceRobotModel


def build_evaluation_report(result: Any, top_k: int = 10) -> dict[str, Any]:
    """Return a detailed, JSON-friendly evaluation report."""
    measured = np.asarray(result.dataset["measured_positions"], dtype=float).reshape(-1, 3)
    estimated = np.asarray(result.identification.vector, dtype=float).reshape(-1)
    report: dict[str, Any] = {
        "data": _data_summary(result.dataset),
        "position": position_evaluation(
            measured=measured,
            nominal=result.nominal_positions,
            calibrated=result.calibrated_positions,
            top_k=top_k,
        ),
        "identification": _identification_summary(result),
    }
    truth = result.dataset.get("true_error_vector", None)
    if truth is not None:
        report["parameter_truth"] = parameter_truth_evaluation(
            parameters=result.parameters,
            true_vector=np.asarray(truth, dtype=float).reshape(-1),
            estimated_vector=estimated,
            active_indices=_active_indices(result),
            independent_indices=result.redundancy.independent_indices,
            top_k=top_k,
        )
    return report


def position_evaluation(
    measured: np.ndarray,
    nominal: np.ndarray,
    calibrated: np.ndarray,
    top_k: int = 10,
) -> dict[str, Any]:
    """Evaluate before/after position errors in norm and per-axis residuals."""
    measured_array = np.asarray(measured, dtype=float).reshape(-1, 3)
    nominal_residual = np.asarray(nominal, dtype=float).reshape(-1, 3) - measured_array
    calibrated_residual = np.asarray(calibrated, dtype=float).reshape(-1, 3) - measured_array
    before_norm = np.linalg.norm(nominal_residual, axis=1)
    after_norm = np.linalg.norm(calibrated_residual, axis=1)

    before_stats = _norm_stats(before_norm)
    after_stats = _norm_stats(after_norm)
    return {
        "norm_before": before_stats,
        "norm_after": after_stats,
        "improvement_percent": _improvement(before_stats, after_stats),
        "axis_before": _axis_stats(nominal_residual),
        "axis_after": _axis_stats(calibrated_residual),
        "axis_rmse_improvement_percent": _axis_improvement(
            _axis_stats(nominal_residual), _axis_stats(calibrated_residual), "rmse"
        ),
        "worst_samples_after": _worst_samples(after_norm, calibrated_residual, top_k),
    }


def parameter_truth_evaluation(
    parameters: list[ErrorParameter],
    true_vector: np.ndarray,
    estimated_vector: np.ndarray,
    active_indices: list[int],
    independent_indices: list[int],
    top_k: int = 10,
) -> dict[str, Any]:
    """Compare simulated truth and identified parameters."""
    truth = np.asarray(true_vector, dtype=float).reshape(len(parameters))
    estimate = np.asarray(estimated_vector, dtype=float).reshape(len(parameters))
    error = estimate - truth
    scales = parameter_scales(parameters)

    subsets = {
        "all": list(range(len(parameters))),
        "identified_active": list(active_indices),
        "independent": list(independent_indices),
        "not_identified": [i for i in range(len(parameters)) if i not in set(active_indices)],
    }
    return {
        "subsets": {
            name: _parameter_subset_stats(indices, error, truth, estimate, scales)
            for name, indices in subsets.items()
        },
        "by_group": _group_parameter_stats(parameters, error, truth, estimate, scales),
        "per_parameter": _parameter_differences(parameters, truth, estimate, error, scales),
        "largest_absolute_errors": _top_parameter_errors(
            parameters, truth, estimate, error, scales, top_k=top_k
        ),
    }


def _data_summary(dataset: dict[str, Any]) -> dict[str, Any]:
    joints = np.asarray(dataset["joints"], dtype=float).reshape(-1, 6)
    payloads = np.asarray(dataset.get("payloads", np.zeros(len(joints))), dtype=float).reshape(-1)
    summary = {
        "samples": int(len(joints)),
        "joint_min": np.min(joints, axis=0).tolist(),
        "joint_max": np.max(joints, axis=0).tolist(),
        "payload_min": float(np.min(payloads)) if payloads.size else 0.0,
        "payload_max": float(np.max(payloads)) if payloads.size else 0.0,
        "has_truth": bool("true_error_vector" in dataset),
    }
    metadata = dataset.get("metadata", None)
    if isinstance(metadata, dict):
        summary["metadata"] = metadata
    return summary


def _identification_summary(result: Any) -> dict[str, Any]:
    names = [param.name for param in result.parameters]
    measured = np.asarray(result.dataset["measured_positions"], dtype=float).reshape(-1, 3)
    return {
        "rank": int(result.redundancy.rank),
        "nullity": int(result.redundancy.nullity),
        "condition_number": _finite_number(result.redundancy.condition_number),
        "independent_count": int(len(result.redundancy.independent_indices)),
        "redundant_count": int(len(result.redundancy.redundant_indices)),
        "independent_parameters": [names[i] for i in result.redundancy.independent_indices],
        "redundant_parameters": [names[i] for i in result.redundancy.redundant_indices],
        "used_exhaustive_redundancy_search": bool(result.redundancy.used_exhaustive_search),
        "correlated_sets": [
            [names[i] for i in correlated_set]
            for correlated_set in result.redundancy.correlated_sets
        ],
        "batches": [[names[i] for i in batch] for batch in result.batches],
        "stages": _stage_identification_summaries(result, measured, names),
    }


def _stage_identification_summaries(
    result: Any,
    measured: np.ndarray,
    names: list[str],
) -> list[dict[str, Any]]:
    baseline_stats = _norm_stats(
        np.linalg.norm(np.asarray(result.nominal_positions, dtype=float).reshape(-1, 3) - measured, axis=1)
    )
    model = MultiSourceRobotModel()
    joints = np.asarray(result.dataset["joints"], dtype=float).reshape(-1, 6)
    payloads = result.dataset.get("payloads", None)
    directions = result.dataset.get("directions", None)
    previous_stats = baseline_stats

    summaries: list[dict[str, Any]] = []
    for stage in result.identification.stages:
        positions = model.batch_positions(
            joints,
            stage.vector_snapshot,
            result.parameters,
            payloads,
            directions,
        )
        residual = np.asarray(positions, dtype=float).reshape(-1, 3) - measured
        error_norm = np.linalg.norm(residual, axis=1)
        stats = _norm_stats(error_norm)
        summaries.append(
            {
                "stage": int(stage.stage),
                "optimized_count": int(len(stage.optimized_indices)),
                "active_count": int(len(stage.active_indices)),
                "optimized_parameters": [names[i] for i in stage.optimized_indices],
                "active_parameters": [names[i] for i in stage.active_indices],
                "component_rmse": float(stage.rmse),
                "position_error": stats,
                "improvement_from_nominal_percent": _improvement(baseline_stats, stats),
                "improvement_from_previous_percent": _improvement(previous_stats, stats),
                "cost": float(stage.cost),
                "nfev": int(stage.nfev),
            }
        )
        previous_stats = stats
    return summaries


def _norm_stats(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float).reshape(-1)
    return {
        "mean": float(np.mean(x)),
        "rmse": float(np.sqrt(np.mean(np.square(x)))),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95.0)),
        "max": float(np.max(x)),
        "std": float(np.std(x)),
    }


def _axis_stats(residuals: np.ndarray) -> dict[str, dict[str, float]]:
    axis_names = ("x", "y", "z")
    output: dict[str, dict[str, float]] = {}
    for index, axis in enumerate(axis_names):
        values = np.asarray(residuals, dtype=float)[:, index]
        output[axis] = {
            "bias": float(np.mean(values)),
            "mean_abs": float(np.mean(np.abs(values))),
            "rmse": float(np.sqrt(np.mean(np.square(values)))),
            "max_abs": float(np.max(np.abs(values))),
            "std": float(np.std(values)),
        }
    return output


def _improvement(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {
        key: _safe_improvement(before[key], after[key])
        for key in ("mean", "rmse", "median", "p95", "max", "std")
    }


def _axis_improvement(
    before: dict[str, dict[str, float]],
    after: dict[str, dict[str, float]],
    metric: str,
) -> dict[str, float]:
    return {
        axis: _safe_improvement(before[axis][metric], after[axis][metric])
        for axis in ("x", "y", "z")
    }


def _safe_improvement(before: float, after: float) -> float:
    if abs(before) <= 1.0e-20:
        return 0.0
    return float((before - after) / before * 100.0)


def _worst_samples(
    error_norm: np.ndarray,
    residuals: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    count = min(int(top_k), len(error_norm))
    order = np.argsort(-error_norm)[:count]
    return [
        {
            "index": int(index),
            "error_norm": float(error_norm[index]),
            "residual_xyz": np.asarray(residuals[index], dtype=float).tolist(),
        }
        for index in order
    ]


def _active_indices(result: Any) -> list[int]:
    if not result.identification.stages:
        return []
    return list(result.identification.stages[-1].active_indices)


def _parameter_subset_stats(
    indices: list[int],
    error: np.ndarray,
    truth: np.ndarray,
    estimate: np.ndarray,
    scales: np.ndarray,
) -> dict[str, float | int]:
    if not indices:
        return {
            "count": 0,
            "mae": 0.0,
            "rmse": 0.0,
            "max_abs": 0.0,
            "normalized_mae": 0.0,
            "normalized_rmse": 0.0,
            "truth_l2": 0.0,
            "estimate_l2": 0.0,
            "error_l2": 0.0,
        }
    idx = np.asarray(indices, dtype=int)
    subset_error = error[idx]
    normalized_error = subset_error / np.maximum(scales[idx], 1.0e-20)
    return {
        "count": int(len(idx)),
        "mae": float(np.mean(np.abs(subset_error))),
        "rmse": float(np.sqrt(np.mean(np.square(subset_error)))),
        "max_abs": float(np.max(np.abs(subset_error))),
        "normalized_mae": float(np.mean(np.abs(normalized_error))),
        "normalized_rmse": float(np.sqrt(np.mean(np.square(normalized_error)))),
        "truth_l2": float(np.linalg.norm(truth[idx])),
        "estimate_l2": float(np.linalg.norm(estimate[idx])),
        "error_l2": float(np.linalg.norm(subset_error)),
    }


def _group_parameter_stats(
    parameters: list[ErrorParameter],
    error: np.ndarray,
    truth: np.ndarray,
    estimate: np.ndarray,
    scales: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[int]] = {}
    for index, parameter in enumerate(parameters):
        groups.setdefault(parameter.group, []).append(index)
    return {
        group: _parameter_subset_stats(indices, error, truth, estimate, scales)
        for group, indices in groups.items()
    }


def _top_parameter_errors(
    parameters: list[ErrorParameter],
    truth: np.ndarray,
    estimate: np.ndarray,
    error: np.ndarray,
    scales: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    count = min(int(top_k), len(parameters))
    order = np.argsort(-np.abs(error))[:count]
    return [
        {
            "name": parameters[index].name,
            "group": parameters[index].group,
            "unit": parameters[index].unit,
            "true": float(truth[index]),
            "estimated": float(estimate[index]),
            "error": float(error[index]),
            "abs_error": float(abs(error[index])),
            "normalized_abs_error": float(abs(error[index]) / max(scales[index], 1.0e-20)),
        }
        for index in order
    ]


def _parameter_differences(
    parameters: list[ErrorParameter],
    truth: np.ndarray,
    estimate: np.ndarray,
    error: np.ndarray,
    scales: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "name": parameter.name,
            "group": parameter.group,
            "unit": parameter.unit,
            "true": float(truth[index]),
            "estimated": float(estimate[index]),
            "error": float(error[index]),
            "abs_error": float(abs(error[index])),
            "normalized_abs_error": float(abs(error[index]) / max(scales[index], 1.0e-20)),
        }
        for index, parameter in enumerate(parameters)
    ]


def _finite_number(value: float) -> float | str:
    number = float(value)
    if np.isfinite(number):
        return number
    return "inf" if number > 0.0 else "-inf"
