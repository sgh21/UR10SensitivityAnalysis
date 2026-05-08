"""Latin-hypercube/Sobol sensitivity analysis aligned with the paper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from calibration.parameters import ErrorParameter, parameter_bounds, zero_error_vector
from calibration.robot_model import MultiSourceRobotModel


@dataclass
class SensitivityResult:
    """First-order and total sensitivity scores with descending total ranking."""

    first_order_indices: np.ndarray
    total_indices: np.ndarray
    normalized_scores: np.ndarray
    ranked_indices: list[int]
    output_variance: float


def sobol_total_indices_lhs(
    model: MultiSourceRobotModel,
    joint_configs: np.ndarray,
    parameters: list[ErrorParameter],
    payloads: np.ndarray | float | None = None,
    directions: np.ndarray | None = None,
    n_samples: int = 128,
    seed: int | None = None,
) -> SensitivityResult:
    """Estimate the paper's Eq. (21) and Eq. (22) with LHS column swaps.

    The paper defines an influence function ``f(Dv)`` for positioning error.
    Because the current dataset contains measured positions rather than full
    poses, this implementation uses a scalar RMS displacement between the
    perturbed and nominal predicted positions across the sampled trajectory.
    The total index from Eq. (22) is used for ordering, matching Algorithm 1.
    """
    lower, upper = parameter_bounds(parameters)
    dim = len(parameters)
    sampler_a = qmc.LatinHypercube(d=dim, seed=seed)
    sampler_b = qmc.LatinHypercube(d=dim, seed=None if seed is None else seed + 1)
    matrix_a = qmc.scale(sampler_a.random(n_samples), lower, upper)
    matrix_b = qmc.scale(sampler_b.random(n_samples), lower, upper)

    nominal_positions = model.batch_positions(
        joint_configs,
        zero_error_vector(parameters),
        parameters,
        payloads,
        directions,
    )
    outputs_a = _evaluate_error_mapping(
        model, joint_configs, matrix_a, parameters, nominal_positions, payloads, directions
    )
    outputs_b = _evaluate_error_mapping(
        model, joint_configs, matrix_b, parameters, nominal_positions, payloads, directions
    )
    variance = float(np.var(np.concatenate([outputs_a, outputs_b]), ddof=1))
    if variance <= 1.0e-20:
        zeros = np.zeros(dim, dtype=float)
        return SensitivityResult(zeros, zeros, zeros, list(range(dim)), 0.0)

    first_order = np.zeros(dim, dtype=float)
    total = np.zeros(dim, dtype=float)
    for index in range(dim):
        swapped = matrix_a.copy()
        swapped[:, index] = matrix_b[:, index]
        outputs_swapped = _evaluate_error_mapping(
            model, joint_configs, swapped, parameters, nominal_positions, payloads, directions
        )
        # Eq. (21): contribution of parameter i alone.
        first_order[index] = float(
            np.mean(outputs_b * (outputs_swapped - outputs_a)) / variance
        )
        # Eq. (22): contribution of parameter i plus interactions.
        total[index] = float(
            np.mean(np.square(outputs_a - outputs_swapped)) / (2.0 * variance)
        )
    score_sum = float(np.sum(total))
    normalized = total / score_sum if score_sum > 1.0e-20 else np.zeros_like(total)
    return SensitivityResult(
        first_order,
        total,
        normalized,
        [int(i) for i in np.argsort(-normalized)],
        variance,
    )


def _evaluate_error_mapping(
    model: MultiSourceRobotModel,
    joint_configs: np.ndarray,
    sample_matrix: np.ndarray,
    parameters: list[ErrorParameter],
    nominal_positions: np.ndarray,
    payloads: np.ndarray | float | None,
    directions: np.ndarray | None,
) -> np.ndarray:
    values = []
    for sample in sample_matrix:
        predicted = model.batch_positions(joint_configs, sample, parameters, payloads, directions)
        displacement = predicted - nominal_positions
        values.append(float(np.sqrt(np.mean(np.sum(np.square(displacement), axis=1)))))
    return np.asarray(values, dtype=float)
