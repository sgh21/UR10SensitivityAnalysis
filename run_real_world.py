"""Run the real-data identification interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from calibration.pipeline import CalibrationConfig, run_real_identification, summarize_result
from calibration.reporting import format_result_report
from calibration.visualization import generate_evaluation_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline calibration on a measured pkl dataset.")
    parser.add_argument("dataset", help="path to a measured pkl")
    parser.add_argument("--sensitivity-samples", type=int, default=128, help="LHS sample count")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--top-k", type=int, default=10, help="number of detailed rows in evaluation tables")
    parser.add_argument("--plots-dir", default="outputs/figures", help="directory for generated evaluation plots")
    parser.add_argument("--plot-prefix", default="real_world", help="filename prefix for generated plots")
    parser.add_argument("--no-plots", action="store_true", help="disable automatic plot generation")
    parser.add_argument("--json", action="store_true", help="print the raw JSON summary")
    parser.add_argument("--optimization-mode",choices=("sobol_stepwise", "full_lm"),default="sobol_stepwise",help="calibration mode")
    args = parser.parse_args()

    result = run_real_identification(
        Path(args.dataset),
        config=CalibrationConfig(sensitivity_samples=args.sensitivity_samples, seed=args.seed, optimization_mode=args.optimization_mode),
    )
    summary = summarize_result(result, top_k=args.top_k)
    if not args.no_plots:
        summary["plots"] = generate_evaluation_plots(
            result,
            output_dir=args.plots_dir,
            prefix=args.plot_prefix,
            top_k=args.top_k,
        )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(format_result_report(summary, title="Real-World Calibration Report"))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


if __name__ == "__main__":
    main()
