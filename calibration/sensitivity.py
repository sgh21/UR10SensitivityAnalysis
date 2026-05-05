"""Latin-hypercube/Sobol total sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from calibration.parameters import ErrorParameter, parameter_bounds
from calibration.robot_model import MultiSourceRobotModel


@dataclass
class SensitivityResult:
    """Total sensitivity scores and descending ranking."""

    total_indices: np.ndarray
    normalized_scores: np.ndarray
    ranked_indices: list[int]


def sobol_total_indices_lhs(
    model: MultiSourceRobotModel,
    joint_configs: np.ndarray,
    parameters: list[ErrorParameter],
    payloads: np.ndarray | float | None = None,
    directions: np.ndarray | None = None,
    n_samples: int = 128,
    seed: int | None = None,
) -> SensitivityResult:
    """Estimate Eq. (22) total sensitivity indices with LHS column swaps."""
    lower, upper = parameter_bounds(parameters)
    dim = len(parameters)
    sampler_a = qmc.LatinHypercube(d=dim, seed=seed)
    sampler_b = qmc.LatinHypercube(d=dim, seed=None if seed is None else seed + 1)
    matrix_a = qmc.scale(sampler_a.random(n_samples), lower, upper)
    matrix_b = qmc.scale(sampler_b.random(n_samples), lower, upper)

    outputs_a = _evaluate(model, joint_configs, matrix_a, parameters, payloads, directions)
    variance = float(np.sum(np.var(outputs_a, axis=0, ddof=1)))
    if variance <= 1.0e-20:
        zeros = np.zeros(dim, dtype=float)
        return SensitivityResult(zeros, zeros, list(range(dim)))

    total = np.zeros(dim, dtype=float)
    for index in range(dim):
        swapped = matrix_a.copy()
        swapped[:, index] = matrix_b[:, index]
        outputs_swapped = _evaluate(
            model, joint_configs, swapped, parameters, payloads, directions
        )
        total[index] = float(
            np.mean(np.sum(np.square(outputs_a - outputs_swapped), axis=1))
            / (2.0 * variance)
        )
    score_sum = float(np.sum(total))
    normalized = total / score_sum if score_sum > 1.0e-20 else np.zeros_like(total)
    return SensitivityResult(total, normalized, [int(i) for i in np.argsort(-normalized)])


def _evaluate(
    model: MultiSourceRobotModel,
    joint_configs: np.ndarray,
    sample_matrix: np.ndarray,
    parameters: list[ErrorParameter],
    payloads: np.ndarray | float | None,
    directions: np.ndarray | None,
) -> np.ndarray:
    values = [
        model.batch_positions(joint_configs, sample, parameters, payloads, directions).reshape(-1)
        for sample in sample_matrix
    ]
    return np.asarray(values, dtype=float)
