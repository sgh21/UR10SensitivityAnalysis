"""High-level interfaces for simulation validation and real-data identification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from calibration.data_io import load_dataset
from calibration.metrics import position_error_metrics
from calibration.parameters import (
    ErrorParameter,
    build_error_parameters,
    vector_to_named_dict,
    zero_error_vector,
)
from calibration.redundancy import RedundancyResult, analyze_redundancy
from calibration.robot_model import MultiSourceRobotModel
from calibration.sensitivity import SensitivityResult, sobol_total_indices_lhs
from calibration.stepwise_lm import IdentificationResult, identify_stepwise_lm, make_paper_batches
from simulation.generator import generate_synthetic_dataset


@dataclass
class CalibrationConfig:
    """Numerical settings for the paper baseline."""

    redundancy_tolerance: float = 1.0e-7
    sensitivity_samples: int = 128
    high_cumulative_score: float = 0.80
    max_nfev_per_stage: int = 120
    seed: int = 123


@dataclass
class CalibrationResult:
    """Complete result object returned by both main interfaces."""

    parameters: list[ErrorParameter]
    redundancy: RedundancyResult
    sensitivity: SensitivityResult
    batches: list[list[int]]
    identification: IdentificationResult
    nominal_metrics: dict[str, float]
    calibrated_metrics: dict[str, float]
    nominal_positions: np.ndarray
    calibrated_positions: np.ndarray
    dataset: dict[str, Any]


def run_real_identification(
    dataset_path: str | Path,
    config: CalibrationConfig | None = None,
) -> CalibrationResult:
    """Main interface 1: identify parameters from a real measured pkl."""
    dataset = load_dataset(dataset_path)
    return run_calibration_on_dataset(dataset, config=config)


def run_simulation_validation(
    output_dataset_path: str | Path = "outputs/synthetic_dataset.pkl",
    n_samples: int = 120,
    payload: float = 0.0,
    measurement_noise_std: float = 2.0e-5,
    truth_scale: float = 1.0,
    config: CalibrationConfig | None = None,
) -> CalibrationResult:
    """Main interface 2: generate synthetic data, save it, then identify."""
    cfg = config if config is not None else CalibrationConfig()
    dataset = generate_synthetic_dataset(
        output_path=output_dataset_path,
        n_samples=n_samples,
        payload=payload,
        measurement_noise_std=measurement_noise_std,
        truth_scale=truth_scale,
        seed=cfg.seed,
    )
    return run_calibration_on_dataset(dataset, config=cfg)


def run_calibration_on_dataset(
    dataset: dict[str, Any],
    config: CalibrationConfig | None = None,
) -> CalibrationResult:
    """Shared paper-baseline pipeline used by real and simulated data."""
    cfg = config if config is not None else CalibrationConfig()
    parameters = build_error_parameters()
    model = MultiSourceRobotModel()
    joints = np.asarray(dataset["joints"], dtype=float).reshape(-1, 6)
    measured = np.asarray(dataset["measured_positions"], dtype=float).reshape(-1, 3)
    payloads = dataset.get("payloads", None)
    directions = dataset.get("directions", None)
    zero = zero_error_vector(parameters)

    nominal_positions = model.batch_positions(joints, zero, parameters, payloads, directions)
    nominal_metrics = position_error_metrics(measured, nominal_positions)

    redundancy = analyze_redundancy(
        model,
        joints,
        zero,
        parameters,
        payloads=payloads,
        directions=directions,
        tolerance=cfg.redundancy_tolerance,
    )
    sensitivity = sobol_total_indices_lhs(
        model,
        joints,
        parameters,
        payloads=payloads,
        directions=directions,
        n_samples=cfg.sensitivity_samples,
        seed=cfg.seed,
    )
    batches = make_paper_batches(
        parameters=parameters,
        ranked_indices=sensitivity.ranked_indices,
        scores=sensitivity.normalized_scores,
        independent_indices=redundancy.independent_indices,
        high_cumulative_score=cfg.high_cumulative_score,
    )
    identification = identify_stepwise_lm(
        model,
        joints,
        measured,
        parameters,
        batches,
        payloads=payloads,
        directions=directions,
        initial_vector=zero,
        max_nfev_per_stage=cfg.max_nfev_per_stage,
    )
    calibrated_positions = model.batch_positions(
        joints, identification.vector, parameters, payloads, directions
    )
    calibrated_metrics = position_error_metrics(measured, calibrated_positions)

    return CalibrationResult(
        parameters=parameters,
        redundancy=redundancy,
        sensitivity=sensitivity,
        batches=batches,
        identification=identification,
        nominal_metrics=nominal_metrics,
        calibrated_metrics=calibrated_metrics,
        nominal_positions=nominal_positions,
        calibrated_positions=calibrated_positions,
        dataset=dataset,
    )


def summarize_result(result: CalibrationResult, top_k: int = 10) -> dict[str, Any]:
    """Build a compact serializable summary for CLI output or notebooks."""
    names = [p.name for p in result.parameters]
    top_indices = result.sensitivity.ranked_indices[:top_k]
    return {
        "nominal_metrics": result.nominal_metrics,
        "calibrated_metrics": result.calibrated_metrics,
        "rank": result.redundancy.rank,
        "redundant_parameters": [names[i] for i in result.redundancy.redundant_indices],
        "top_sensitivity": [
            {
                "name": names[i],
                "group": result.parameters[i].group,
                "score": float(result.sensitivity.normalized_scores[i]),
            }
            for i in top_indices
        ],
        "batches": [[names[i] for i in batch] for batch in result.batches],
        "identified_parameters": vector_to_named_dict(
            result.identification.vector, result.parameters
        ),
        "stages": [
            {
                "stage": stage.stage,
                "optimized_count": len(stage.optimized_indices),
                "active_count": len(stage.active_indices),
                "rmse": stage.rmse,
                "nfev": stage.nfev,
            }
            for stage in result.identification.stages
        ],
    }
