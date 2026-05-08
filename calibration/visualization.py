"""Plot evaluation figures without coupling them to calibration internals."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from calibration.parameters import parameter_scales
from calibration.robot_model import MultiSourceRobotModel


def generate_evaluation_plots(
    result: Any,
    output_dir: str | Path = "outputs/figures",
    prefix: str = "calibration",
    top_k: int = 20,
    dpi: int = 160,
) -> dict[str, str]:
    """Generate the standard calibration evaluation plots.

    The function accepts a completed ``CalibrationResult`` and only reads from
    it.  No optimizer, sensitivity, or model-selection logic is run here.
    """
    plt = _pyplot()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    safe_prefix = _safe_prefix(prefix)

    files: dict[str, str] = {}
    files["position_accuracy"] = str(
        _plot_position_accuracy(result, output_path, safe_prefix, dpi, plt)
    )
    files["sensitivity_distribution"] = str(
        _plot_sensitivity_distribution(result, output_path, safe_prefix, top_k, dpi, plt)
    )
    files["calibration_trend"] = str(
        _plot_calibration_trend(result, output_path, safe_prefix, dpi, plt)
    )
    if result.dataset.get("true_error_vector", None) is not None:
        files["parameter_truth"] = str(
            _plot_parameter_truth(result, output_path, safe_prefix, top_k, dpi, plt)
        )

    manifest = output_path / f"{safe_prefix}_plots.json"
    manifest.write_text(json.dumps(files, indent=2, ensure_ascii=False), encoding="utf-8")
    files["manifest"] = str(manifest)
    return files


def _plot_position_accuracy(
    result: Any,
    output_path: Path,
    prefix: str,
    dpi: int,
    plt: Any,
) -> Path:
    measured = _measured_positions(result)
    before_mm = _position_error_norm_mm(result.nominal_positions, measured)
    after_mm = _position_error_norm_mm(result.calibrated_positions, measured)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    bins = _hist_bins(np.concatenate([before_mm, after_mm]))
    axes[0].hist(before_mm, bins=bins, alpha=0.58, label="Before", color="#6b7280")
    axes[0].hist(after_mm, bins=bins, alpha=0.72, label="After", color="#2563eb")
    axes[0].set_title("Position error distribution")
    axes[0].set_xlabel("Euclidean error (mm)")
    axes[0].set_ylabel("Sample count")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    metrics = ("Mean", "RMSE", "P95", "Max")
    x = np.arange(len(metrics))
    width = 0.36
    before_stats = _error_stats(before_mm)
    after_stats = _error_stats(after_mm)
    before_bars = axes[1].bar(
        x - width / 2.0, before_stats, width, label="Before", color="#6b7280"
    )
    after_bars = axes[1].bar(
        x + width / 2.0, after_stats, width, label="After", color="#2563eb"
    )
    axes[1].set_title("Position accuracy metrics")
    axes[1].set_ylabel("Error (mm)")
    axes[1].set_xticks(x, metrics)
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.25)
    _scale_metric_axis_for_contrast(axes[1], before_stats + after_stats)
    _annotate_bars(axes[1], before_bars)
    _annotate_bars(axes[1], after_bars)

    path = output_path / f"{prefix}_position_accuracy.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _plot_parameter_truth(
    result: Any,
    output_path: Path,
    prefix: str,
    top_k: int,
    dpi: int,
    plt: Any,
) -> Path:
    truth = np.asarray(result.dataset["true_error_vector"], dtype=float).reshape(-1)
    estimate = np.asarray(result.identification.vector, dtype=float).reshape(-1)
    scales = parameter_scales(result.parameters)
    truth_scaled = truth / np.maximum(scales, 1.0e-20)
    estimate_scaled = estimate / np.maximum(scales, 1.0e-20)
    error_scaled = estimate_scaled - truth_scaled

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True)
    groups = _ordered_groups(result.parameters)
    colors = _group_colors(groups, plt)
    for group in groups:
        indices = [i for i, p in enumerate(result.parameters) if p.group == group]
        axes[0].scatter(
            truth_scaled[indices],
            estimate_scaled[indices],
            s=32,
            alpha=0.82,
            label=group,
            color=colors[group],
        )
    limit = float(max(np.max(np.abs(truth_scaled)), np.max(np.abs(estimate_scaled)), 1.0))
    axes[0].plot([-limit, limit], [-limit, limit], color="#111827", linewidth=1.0)
    axes[0].set_xlim(-limit * 1.08, limit * 1.08)
    axes[0].set_ylim(-limit * 1.08, limit * 1.08)
    axes[0].set_title("Identified vs true parameters")
    axes[0].set_xlabel("True value / prior scale")
    axes[0].set_ylabel("Estimated value / prior scale")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)

    count = min(int(top_k), len(result.parameters))
    order = np.argsort(-np.abs(error_scaled))[:count][::-1]
    names = [result.parameters[i].name for i in order]
    axes[1].barh(
        np.arange(count),
        np.abs(error_scaled[order]),
        color=[colors[result.parameters[i].group] for i in order],
    )
    axes[1].set_title(f"Top {count} normalized parameter errors")
    axes[1].set_xlabel("Absolute error / prior scale")
    axes[1].set_yticks(np.arange(count), names, fontsize=8)
    axes[1].grid(True, axis="x", alpha=0.25)

    path = output_path / f"{prefix}_parameter_truth.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _plot_sensitivity_distribution(
    result: Any,
    output_path: Path,
    prefix: str,
    top_k: int,
    dpi: int,
    plt: Any,
) -> Path:
    scores = np.asarray(result.sensitivity.normalized_scores, dtype=float).reshape(-1)
    total = np.asarray(result.sensitivity.total_indices, dtype=float).reshape(-1)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)
    axes[0].hist(scores, bins=_hist_bins(scores), color="#059669", alpha=0.78)
    axes[0].set_title("Sensitivity score histogram")
    axes[0].set_xlabel("Normalized total sensitivity score")
    axes[0].set_ylabel("Parameter count")
    axes[0].grid(True, alpha=0.25)

    count = min(int(top_k), len(result.parameters))
    order = list(result.sensitivity.ranked_indices[:count])[::-1]
    names = [result.parameters[i].name for i in order]
    axes[1].barh(np.arange(count), total[order], color="#0f766e")
    axes[1].set_title(f"Top {count} total sensitivity indices")
    axes[1].set_xlabel("Total sensitivity index")
    axes[1].set_yticks(np.arange(count), names, fontsize=8)
    axes[1].grid(True, axis="x", alpha=0.25)
    _annotate_horizontal_bars(axes[1])

    path = output_path / f"{prefix}_sensitivity_distribution.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _plot_calibration_trend(
    result: Any,
    output_path: Path,
    prefix: str,
    dpi: int,
    plt: Any,
) -> Path:
    series = _stage_error_series(result)
    stages = np.asarray([item["stage"] for item in series], dtype=float)
    labels = [item["label"] for item in series]

    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    for key, color, marker in (
        ("rmse_mm", "#2563eb", "o"),
        ("mean_mm", "#475569", "s"),
        ("p95_mm", "#dc2626", "^"),
    ):
        ax.plot(
            stages,
            [item[key] for item in series],
            marker=marker,
            linewidth=1.8,
            color=color,
            label=key.replace("_mm", "").upper(),
        )
        for x_value, item in zip(stages, series):
            ax.annotate(
                _format_mm(float(item[key])),
                (x_value, float(item[key])),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=7,
                color=color,
            )
    ax.set_title("Position error trend during calibration")
    ax.set_xlabel("Calibration stage")
    ax.set_ylabel("Euclidean error (mm)")
    ax.set_xticks(stages, labels)
    ax.legend()
    ax.grid(True, alpha=0.25)
    _scale_metric_axis_for_contrast(
        ax,
        [
            float(item[key])
            for item in series
            for key in ("mean_mm", "rmse_mm", "p95_mm")
        ],
    )

    path = output_path / f"{prefix}_calibration_trend.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _stage_error_series(result: Any) -> list[dict[str, float | str]]:
    measured = _measured_positions(result)
    series: list[dict[str, float | str]] = [
        _stage_stats(stage=0, label="0", positions=result.nominal_positions, measured=measured)
    ]
    model = MultiSourceRobotModel()
    joints = np.asarray(result.dataset["joints"], dtype=float).reshape(-1, 6)
    payloads = result.dataset.get("payloads", None)
    directions = result.dataset.get("directions", None)

    for stage in result.identification.stages:
        vector = getattr(stage, "vector_snapshot", None)
        if vector is None:
            series.append(
                {
                    "stage": float(stage.stage),
                    "label": str(stage.stage),
                    "mean_mm": float(stage.rmse * 1000.0),
                    "rmse_mm": float(stage.rmse * 1000.0),
                    "p95_mm": float(stage.rmse * 1000.0),
                }
            )
            continue
        positions = model.batch_positions(joints, vector, result.parameters, payloads, directions)
        series.append(
            _stage_stats(stage=stage.stage, label=str(stage.stage), positions=positions, measured=measured)
        )

    if not result.identification.stages:
        series.append(
            _stage_stats(
                stage=1,
                label="final",
                positions=result.calibrated_positions,
                measured=measured,
            )
        )
    return series


def _stage_stats(
    stage: int,
    label: str,
    positions: np.ndarray,
    measured: np.ndarray,
) -> dict[str, float | str]:
    errors = _position_error_norm_mm(positions, measured)
    return {
        "stage": float(stage),
        "label": label,
        "mean_mm": float(np.mean(errors)),
        "rmse_mm": float(np.sqrt(np.mean(np.square(errors)))),
        "p95_mm": float(np.percentile(errors, 95.0)),
    }


def _position_error_norm_mm(predicted: np.ndarray, measured: np.ndarray) -> np.ndarray:
    predicted_array = np.asarray(predicted, dtype=float).reshape(-1, 3)
    measured_array = np.asarray(measured, dtype=float).reshape(-1, 3)
    return np.linalg.norm(predicted_array - measured_array, axis=1) * 1000.0


def _measured_positions(result: Any) -> np.ndarray:
    return np.asarray(result.dataset["measured_positions"], dtype=float).reshape(-1, 3)


def _error_stats(values_mm: np.ndarray) -> list[float]:
    values = np.asarray(values_mm, dtype=float).reshape(-1)
    return [
        float(np.mean(values)),
        float(np.sqrt(np.mean(np.square(values)))),
        float(np.percentile(values, 95.0)),
        float(np.max(values)),
    ]


def _hist_bins(values: np.ndarray) -> int | np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2 or float(np.max(finite) - np.min(finite)) <= 1.0e-12:
        center = float(finite[0]) if finite.size else 0.0
        return np.linspace(center - 0.5, center + 0.5, 12)
    return min(24, max(8, int(np.sqrt(finite.size))))


def _scale_metric_axis_for_contrast(ax: Any, values: list[float]) -> None:
    positive = np.asarray([value for value in values if value > 0.0], dtype=float)
    if positive.size == 0:
        return
    min_value = float(np.min(positive))
    max_value = float(np.max(positive))
    if max_value / max(min_value, 1.0e-12) >= 25.0:
        ax.set_yscale("log")
        ax.set_ylim(min_value * 0.55, max_value * 1.8)
    else:
        ax.set_ylim(0.0, max_value * 1.18)


def _annotate_bars(ax: Any, bars: Any) -> None:
    is_log = ax.get_yscale() == "log"
    for bar in bars:
        value = float(bar.get_height())
        if value <= 0.0:
            continue
        y = value * 1.10 if is_log else value + ax.get_ylim()[1] * 0.015
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            _format_mm(value),
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )


def _annotate_horizontal_bars(ax: Any) -> None:
    xmax = float(ax.get_xlim()[1])
    for patch in ax.patches:
        value = float(patch.get_width())
        ax.text(
            value + xmax * 0.012,
            patch.get_y() + patch.get_height() / 2.0,
            _format_sensitivity(value),
            va="center",
            fontsize=7,
        )
    ax.set_xlim(0.0, xmax * 1.15 if xmax > 0.0 else 1.0)


def _format_mm(value: float) -> str:
    if value >= 10.0:
        return f"{value:.1f}"
    if value >= 1.0:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _format_sensitivity(value: float) -> str:
    if value >= 0.01:
        return f"{value:.3f}"
    return f"{value:.1e}"


def _ordered_groups(parameters: list[Any]) -> list[str]:
    groups: list[str] = []
    for parameter in parameters:
        if parameter.group not in groups:
            groups.append(parameter.group)
    return groups


def _group_colors(groups: list[str], plt: Any) -> dict[str, Any]:
    cmap = plt.get_cmap("tab10")
    return {group: cmap(index % 10) for index, group in enumerate(groups)}


def _pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for visualization. Install requirements.txt first."
        ) from exc
    return plt


def _safe_prefix(prefix: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(prefix).strip())
    return cleaned or "calibration"
