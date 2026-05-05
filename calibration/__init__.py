"""Sensitivity-analysis baseline for multisource robot calibration."""

from calibration.pipeline import CalibrationConfig, run_real_identification, run_simulation_validation

__all__ = [
    "CalibrationConfig",
    "run_real_identification",
    "run_simulation_validation",
]
