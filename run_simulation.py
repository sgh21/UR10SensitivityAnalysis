"""Run the synthetic validation interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from calibration.pipeline import CalibrationConfig, run_simulation_validation, summarize_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic data and run baseline calibration.")
    parser.add_argument("--output", default="outputs/synthetic_dataset.pkl", help="where to save generated pkl data")
    parser.add_argument("--samples", type=int, default=120, help="number of sampled poses")
    parser.add_argument("--payload", type=float, default=0.0, help="payload mass in kg")
    parser.add_argument("--noise", type=float, default=2.0e-5, help="measurement noise std in meters")
    parser.add_argument("--truth-scale", type=float, default=1.0, help="scale of simulated truth errors")
    parser.add_argument("--sensitivity-samples", type=int, default=128, help="LHS sample count")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    args = parser.parse_args()

    config = CalibrationConfig(sensitivity_samples=args.sensitivity_samples, seed=args.seed)
    result = run_simulation_validation(
        output_dataset_path=Path(args.output),
        n_samples=args.samples,
        payload=args.payload,
        measurement_noise_std=args.noise,
        truth_scale=args.truth_scale,
        config=config,
    )
    print(json.dumps(summarize_result(result), ensure_ascii=False, indent=2, default=_json_default))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


if __name__ == "__main__":
    main()
