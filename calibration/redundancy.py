"""Correlation analysis for multisource calibration parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr

from calibration.parameters import ErrorParameter, parameter_scales
from calibration.robot_model import MultiSourceRobotModel


@dataclass
class RedundancyResult:
    """Result of the Jacobian rank/correlation analysis."""

    jacobian: np.ndarray
    independent_indices: list[int]
    redundant_indices: list[int]
    rank: int
    condition_number: float
    singular_values: np.ndarray


def output_jacobian(
    model: MultiSourceRobotModel,
    joint_configs: np.ndarray,
    error_vector: np.ndarray,
    parameters: list[ErrorParameter],
    payloads: np.ndarray | float | None = None,
    directions: np.ndarray | None = None,
    step_ratio: float = 1.0e-4,
) -> np.ndarray:
    """Finite-difference Jacobian of stacked xyz outputs wrt parameters."""
    x0 = np.asarray(error_vector, dtype=float).reshape(len(parameters))
    y0 = model.batch_positions(joint_configs, x0, parameters, payloads, directions).reshape(-1)
    scales = parameter_scales(parameters)
    jacobian = np.zeros((y0.size, x0.size), dtype=float)
    for index in range(x0.size):
        step = max(scales[index] * step_ratio, 1.0e-8)
        x_step = x0.copy()
        x_step[index] += step
        y_step = model.batch_positions(
            joint_configs, x_step, parameters, payloads, directions
        ).reshape(-1)
        jacobian[:, index] = (y_step - y0) / step
    return jacobian


def analyze_redundancy(
    model: MultiSourceRobotModel,
    joint_configs: np.ndarray,
    error_vector: np.ndarray,
    parameters: list[ErrorParameter],
    payloads: np.ndarray | float | None = None,
    directions: np.ndarray | None = None,
    tolerance: float = 1.0e-7,
) -> RedundancyResult:
    """Identify independent and correlated parameters from ``J.T @ J``.

    The paper describes SVD of the normal matrix.  QR with column pivoting is
    used here to choose a stable independent subset without enumerating all
    combinations; the returned singular values and condition number are still
    computed from the normalized Jacobian.
    """
    jacobian = output_jacobian(
        model, joint_configs, error_vector, parameters, payloads, directions
    )
    norms = np.linalg.norm(jacobian, axis=0)
    normalized = jacobian / np.maximum(norms, 1.0e-15)
    _, r_matrix, pivots = qr(normalized, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r_matrix))
    if diag.size == 0 or diag[0] <= 1.0e-15:
        rank = 0
    else:
        rank = int(np.sum(diag > tolerance * diag[0]))
    independent = sorted(int(i) for i in pivots[:rank] if norms[i] > 1.0e-14)
    redundant = [i for i in range(len(parameters)) if i not in independent]
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    if singular_values.size == 0 or singular_values[-1] <= 1.0e-15:
        condition = float("inf")
    else:
        condition = float(singular_values[0] / singular_values[-1])
    return RedundancyResult(
        jacobian=jacobian,
        independent_indices=independent,
        redundant_indices=redundant,
        rank=len(independent),
        condition_number=condition,
        singular_values=singular_values,
    )
