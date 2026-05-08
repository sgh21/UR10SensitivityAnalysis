"""Sensitivity-analysis baseline for multisource robot calibration."""

__all__ = [
    "CalibrationConfig",
    "build_evaluation_report",
    "generate_evaluation_plots",
    "run_real_identification",
    "run_simulation_validation",
]


def __getattr__(name: str):
    """Lazily expose public helpers without creating import cycles."""
    if name == "build_evaluation_report":
        from calibration.evaluation import build_evaluation_report

        return build_evaluation_report
    if name == "generate_evaluation_plots":
        from calibration.visualization import generate_evaluation_plots

        return generate_evaluation_plots
    if name in {"CalibrationConfig", "run_real_identification", "run_simulation_validation"}:
        from calibration.pipeline import (
            CalibrationConfig,
            run_real_identification,
            run_simulation_validation,
        )

        return {
            "CalibrationConfig": CalibrationConfig,
            "run_real_identification": run_real_identification,
            "run_simulation_validation": run_simulation_validation,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
