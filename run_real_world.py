"""Run the real-data identification interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from calibration.pipeline import CalibrationConfig, run_real_identification, summarize_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline calibration on a measured pkl dataset.")
    parser.add_argument("dataset", help="path to a measured pkl")
    parser.add_argument("--sensitivity-samples", type=int, default=128, help="LHS sample count")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    args = parser.parse_args()

    result = run_real_identification(
        Path(args.dataset),
        config=CalibrationConfig(sensitivity_samples=args.sensitivity_samples, seed=args.seed),
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
