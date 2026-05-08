"""Stepwise Levenberg-Marquardt identification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from calibration.parameters import ErrorParameter, parameter_scales
from calibration.robot_model import MultiSourceRobotModel


@dataclass
class IdentificationStage:
    """Diagnostics for one cumulative identification stage."""

    stage: int
    optimized_indices: list[int]
    active_indices: list[int]
    cost: float
    rmse: float
    nfev: int
    vector_snapshot: np.ndarray


@dataclass
class IdentificationResult:
    """Final identified vector and stage diagnostics."""

    vector: np.ndarray
    stages: list[IdentificationStage]


def make_paper_batches(
    parameters: list[ErrorParameter],
    ranked_indices: list[int],
    scores: np.ndarray,
    independent_indices: list[int],
    high_cumulative_score: float = 0.80,
) -> list[list[int]]:
    """Build Algorithm-1 batches: high-sensitivity set, then groups by SI sum."""
    independent = set(independent_indices)
    ranked = [index for index in ranked_indices if index in independent]
    if not ranked:
        return []

    cumulative = 0.0
    high: list[int] = []
    total_independent_score = float(np.sum(scores[ranked]))
    threshold = high_cumulative_score * total_independent_score
    for index in ranked:
        high.append(index)
        cumulative += float(scores[index])
        if cumulative >= threshold:
            break

    high_set = set(high)
    grouped: dict[str, list[int]] = {}
    for index in ranked:
        if index in high_set:
            continue
        grouped.setdefault(parameters[index].group, []).append(index)

    ordered_groups = sorted(
        grouped.values(),
        key=lambda group: float(np.sum(scores[group])),
        reverse=True,
    )
    return [high] + [group for group in ordered_groups if group]


def identify_stepwise_lm(
    model: MultiSourceRobotModel,
    joint_configs: np.ndarray,
    measured_positions: np.ndarray,
    parameters: list[ErrorParameter],
    batches: list[list[int]],
    payloads: np.ndarray | float | None = None,
    directions: np.ndarray | None = None,
    initial_vector: np.ndarray | None = None,
    max_nfev_per_stage: int = 120,
) -> IdentificationResult:
    """Run cumulative LM over ordered parameter batches."""
    full = (
        np.zeros(len(parameters), dtype=float)
        if initial_vector is None
        else np.asarray(initial_vector, dtype=float).reshape(len(parameters)).copy()
    )
    measured = np.asarray(measured_positions, dtype=float).reshape(-1, 3)
    active: list[int] = []
    stages: list[IdentificationStage] = []
    scales = parameter_scales(parameters)

    for stage_number, batch in enumerate(batches, start=1):
        active = _unique(active + list(batch))
        x0 = full[active].copy()
        method = "lm" if measured.size >= len(active) else "trf"
        result = least_squares(
            _active_residuals,
            x0=x0,
            args=(full, active, model, joint_configs, measured, parameters, payloads, directions),
            method=method,
            x_scale=np.maximum(scales[active], 1.0e-12),
            max_nfev=max_nfev_per_stage,
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
        )
        full[active] = result.x
        residuals = _residuals(
            model, joint_configs, measured, full, parameters, payloads, directions
        )
        stages.append(
            IdentificationStage(
                stage=stage_number,
                optimized_indices=list(batch),
                active_indices=list(active),
                cost=float(result.cost),
                rmse=float(np.sqrt(np.mean(np.square(residuals)))),
                nfev=int(result.nfev),
                vector_snapshot=full.copy(),
            )
        )
    return IdentificationResult(full, stages)


def _active_residuals(
    active_values: np.ndarray,
    full_vector: np.ndarray,
    active_indices: list[int],
    model: MultiSourceRobotModel,
    joint_configs: np.ndarray,
    measured_positions: np.ndarray,
    parameters: list[ErrorParameter],
    payloads: np.ndarray | float | None,
    directions: np.ndarray | None,
) -> np.ndarray:
    candidate = full_vector.copy()
    candidate[active_indices] = active_values
    return _residuals(
        model, joint_configs, measured_positions, candidate, parameters, payloads, directions
    )


def _residuals(
    model: MultiSourceRobotModel,
    joint_configs: np.ndarray,
    measured_positions: np.ndarray,
    vector: np.ndarray,
    parameters: list[ErrorParameter],
    payloads: np.ndarray | float | None,
    directions: np.ndarray | None,
) -> np.ndarray:
    predicted = model.batch_positions(joint_configs, vector, parameters, payloads, directions)
    return (predicted - measured_positions).reshape(-1)


def _unique(values: list[int]) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output
