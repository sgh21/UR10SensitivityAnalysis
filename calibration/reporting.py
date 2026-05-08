"""Human-readable CLI reports for calibration runs."""

from __future__ import annotations

from typing import Any


def format_result_report(summary: dict[str, Any], title: str = "Calibration Report") -> str:
    """Format a pipeline summary as a compact text report."""
    lines: list[str] = []
    lines.extend([title, "=" * len(title), ""])
    lines.extend(_accuracy_section(summary))
    lines.extend(_redundancy_section(summary))
    lines.extend(_stage_section(summary))
    lines.extend(_sensitivity_section(summary))
    lines.extend(_parameter_truth_section(summary))
    lines.extend(_plots_section(summary))
    return "\n".join(lines).rstrip()


def _accuracy_section(summary: dict[str, Any]) -> list[str]:
    nominal = summary.get("nominal_metrics", {})
    calibrated = summary.get("calibrated_metrics", {})
    evaluation = summary.get("evaluation", {})
    improvement = (
        evaluation.get("position", {})
        .get("improvement_percent", {})
    )
    rows = [
        (
            "Mean",
            _m_to_mm(nominal.get("mean")),
            _m_to_mm(calibrated.get("mean")),
            improvement.get("mean"),
        ),
        (
            "RMSE",
            _m_to_mm(nominal.get("rmse")),
            _m_to_mm(calibrated.get("rmse")),
            improvement.get("rmse"),
        ),
        (
            "Max",
            _m_to_mm(nominal.get("max")),
            _m_to_mm(calibrated.get("max")),
            improvement.get("max"),
        ),
        (
            "Std",
            _m_to_mm(nominal.get("std")),
            _m_to_mm(calibrated.get("std")),
            improvement.get("std"),
        ),
    ]
    return [
        "Position Accuracy",
        _table(
            ["Metric", "Before mm", "After mm", "Improve %"],
            [[name, _fmt(before), _fmt(after), _fmt(percent)] for name, before, after, percent in rows],
        ),
        "",
    ]


def _redundancy_section(summary: dict[str, Any]) -> list[str]:
    evaluation = summary.get("evaluation", {})
    identification = evaluation.get("identification", {})
    independent = summary.get("independent_parameters", [])
    redundant = summary.get("redundant_parameters", [])
    rows = [
        ["Rank", summary.get("rank", identification.get("rank", ""))],
        ["Nullity", identification.get("nullity", "")],
        ["Independent", len(independent)],
        ["Redundant", len(redundant)],
        ["Condition", identification.get("condition_number", "")],
    ]
    lines = [
        "Redundancy",
        _table(["Item", "Value"], [[str(k), _fmt(v)] for k, v in rows]),
    ]
    if independent:
        lines.extend(["", "Independent parameters:", _wrap_names(independent)])
    if redundant:
        lines.extend(["", "Redundant parameters:", _wrap_names(redundant)])
    lines.append("")
    return lines


def _stage_section(summary: dict[str, Any]) -> list[str]:
    stages = (
        summary.get("evaluation", {})
        .get("identification", {})
        .get("stages", [])
    )
    if not stages:
        stages = summary.get("stages", [])
    rows: list[list[str]] = []
    detail_lines: list[str] = []
    for stage in stages:
        pos = stage.get("position_error", {})
        rows.append(
            [
                str(stage.get("stage", "")),
                str(stage.get("optimized_count", "")),
                str(stage.get("active_count", "")),
                _fmt(_m_to_mm(pos.get("rmse")) if pos else _m_to_mm(stage.get("component_rmse"))),
                _fmt(_m_to_mm(pos.get("mean")) if pos else ""),
                _fmt(_m_to_mm(pos.get("p95")) if pos else ""),
                str(stage.get("nfev", "")),
            ]
        )
        params = stage.get("optimized_parameters", [])
        if params:
            detail_lines.extend(
                [
                    f"Stage {stage.get('stage', '')} optimized parameters:",
                    _wrap_names(params),
                ]
            )
    lines = [
        "Identification Stages",
        _table(
            ["Stage", "New", "Active", "RMSE mm", "Mean mm", "P95 mm", "NFEV"],
            rows,
        ),
    ]
    if detail_lines:
        lines.extend(["", *detail_lines])
    lines.append("")
    return lines


def _sensitivity_section(summary: dict[str, Any]) -> list[str]:
    rows = []
    for item in summary.get("top_identified_sensitivity", summary.get("top_sensitivity", [])):
        rows.append(
            [
                item.get("name", ""),
                item.get("group", ""),
                _fmt(item.get("score")),
                _fmt(item.get("total_index")),
                _fmt(item.get("first_order_index")),
            ]
        )
    if not rows:
        return []
    return [
        "Top Identified Sensitivity",
        _table(["Parameter", "Group", "Score", "Total", "First"], rows),
        "",
    ]


def _parameter_truth_section(summary: dict[str, Any]) -> list[str]:
    truth_report = (
        summary.get("evaluation", {})
        .get("parameter_truth", {})
    )
    if not truth_report:
        return []

    lines = ["True vs Identified Parameters"]
    subset_rows = []
    for name, stats in truth_report.get("subsets", {}).items():
        subset_rows.append(
            [
                name,
                stats.get("count", ""),
                _fmt(stats.get("normalized_mae")),
                _fmt(stats.get("normalized_rmse")),
                _fmt(stats.get("error_l2")),
            ]
        )
    if subset_rows:
        lines.extend(
            [
                _table(
                    ["Subset", "Count", "Norm MAE", "Norm RMSE", "Error L2"],
                    subset_rows,
                ),
                "",
            ]
        )

    group_rows = []
    for group, stats in truth_report.get("by_group", {}).items():
        group_rows.append(
            [
                group,
                stats.get("count", ""),
                _fmt(stats.get("normalized_mae")),
                _fmt(stats.get("normalized_rmse")),
                _fmt(stats.get("error_l2")),
            ]
        )
    if group_rows:
        lines.extend(
            [
                "By parameter group:",
                _table(["Group", "Count", "Norm MAE", "Norm RMSE", "Error L2"], group_rows),
                "",
            ]
        )

    error_rows = []
    for item in truth_report.get("largest_absolute_errors", []):
        error_rows.append(
            [
                item.get("name", ""),
                item.get("group", ""),
                item.get("unit", ""),
                _fmt(item.get("true")),
                _fmt(item.get("estimated")),
                _fmt(item.get("error")),
                _fmt(item.get("normalized_abs_error")),
            ]
        )
    if error_rows:
        lines.extend(
            [
                "Largest parameter errors:",
                _table(
                    ["Parameter", "Group", "Unit", "True", "Estimated", "Error", "Norm Abs"],
                    error_rows,
                ),
                "",
            ]
        )
    return lines


def _plots_section(summary: dict[str, Any]) -> list[str]:
    plots = summary.get("plots", {})
    if not plots:
        return []
    rows = [[name, path] for name, path in plots.items()]
    return ["Generated Plots", _table(["Name", "Path"], rows), ""]


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "(none)"
    text_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in text_rows))
        for index in range(len(headers))
    ]
    header = "  ".join(headers[index].ljust(widths[index]) for index in range(len(headers)))
    rule = "  ".join("-" * widths[index] for index in range(len(headers)))
    body = [
        "  ".join(row[index].ljust(widths[index]) for index in range(len(headers)))
        for row in text_rows
    ]
    return "\n".join([header, rule, *body])


def _wrap_names(names: list[str], width: int = 96) -> str:
    lines: list[str] = []
    current = ""
    for name in names:
        token = name if not current else f", {name}"
        if current and len(current) + len(token) > width:
            lines.append(current)
            current = name
        else:
            current += token
    if current:
        lines.append(current)
    return "\n".join(lines)


def _m_to_mm(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value) * 1000.0


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    number = float(value)
    if abs(number) >= 1000.0:
        return f"{number:.3e}"
    if abs(number) >= 10.0:
        return f"{number:.3f}"
    if abs(number) >= 1.0:
        return f"{number:.4f}"
    if abs(number) >= 0.001:
        return f"{number:.6f}"
    return f"{number:.3e}"
