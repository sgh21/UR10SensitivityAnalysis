"""Position-error metrics used by examples and reports."""

from __future__ import annotations

import numpy as np


def position_error_metrics(measured: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return common Euclidean position error statistics in meters."""
    measured_array = np.asarray(measured, dtype=float).reshape(-1, 3)
    predicted_array = np.asarray(predicted, dtype=float).reshape(-1, 3)
    errors = np.linalg.norm(predicted_array - measured_array, axis=1)
    return {
        "mean": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "max": float(np.max(errors)),
        "std": float(np.std(errors)),
    }
