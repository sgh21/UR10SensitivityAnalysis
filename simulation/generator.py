"""Decoupled synthetic dataset generation for calibration validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from calibration.data_io import save_dataset
from calibration.parameters import (
    build_error_parameters,
    sample_truth_vector,
    vector_to_named_dict,
    zero_error_vector,
)
from calibration.robot_model import MultiSourceRobotModel
from config.nominal_config import NOMINAL_ROBOT


def generate_synthetic_dataset(
    output_path: str | Path,
    n_samples: int = 120,
    payload: float = 0.0,
    measurement_noise_std: float = 2.0e-5,
    truth_scale: float = 1.0,
    seed: int = 42,
) -> dict:
    """Generate and save a pkl dataset compatible with real-data identification."""
    rng = np.random.default_rng(seed)
    parameters = build_error_parameters()
    truth = sample_truth_vector(rng, parameters, sigma_scale=truth_scale)
    joints = _sample_joints(rng, n_samples)
    directions = rng.choice([-1.0, 1.0], size=(n_samples, 6))
    payloads = np.full(n_samples, float(payload), dtype=float)
    model = MultiSourceRobotModel()

    true_positions = model.batch_positions(joints, truth, parameters, payloads, directions)
    measured = true_positions + rng.normal(0.0, measurement_noise_std, size=true_positions.shape)
    nominal_positions = model.batch_positions(
        joints, zero_error_vector(parameters), parameters, payloads, directions
    )

    dataset = {
        "joints": joints,
        "measured_positions": measured,
        "payloads": payloads,
        "directions": directions,
        "nominal_positions": nominal_positions,
        "true_positions": true_positions,
        "parameter_names": [p.name for p in parameters],
        "parameter_groups": [p.group for p in parameters],
        "true_error_vector": truth,
        "true_error_parameters": vector_to_named_dict(truth, parameters),
        "nominal_robot": NOMINAL_ROBOT,
        "metadata": {
            "n_samples": int(n_samples),
            "payload": float(payload),
            "measurement_noise_std": float(measurement_noise_std),
            "truth_scale": float(truth_scale),
            "seed": int(seed),
        },
    }
    save_dataset(output_path, dataset)
    return dataset


def _sample_joints(rng: np.random.Generator, n_samples: int) -> np.ndarray:
    limits = np.asarray(NOMINAL_ROBOT["joint_limits"], dtype=float)
    lower, upper = limits[:, 0], limits[:, 1]
    return rng.uniform(lower, upper, size=(n_samples, 6))
