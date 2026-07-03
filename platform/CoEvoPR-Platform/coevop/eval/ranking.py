"""Rank-based objective scoring shared by Tier-2 and Tier-3 evaluators."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable


STRUCTURAL_FAILURE_STAGES = {
    "timeout",
    "solver_init",
    "config_parse",
    "placement_iter",
    "legalization",
    "postprocess",
    "metric_extraction",
    "output_artifact",
    "openroad",
}


@dataclass(frozen=True)
class RankSummary:
    objective_id: str
    average_rank: float
    design_count: int
    seed_count: int
    cell_count: int
    structural_failure_count: int
    metric_regression_count: int
    severe_regression_count: int
    mean_runtime_seconds: float | None
    metric_ranks: dict[str, float]
    design_ranks: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_result(row: dict[str, Any], severe_threshold_pct: float = 10.0) -> str:
    """Classify a completed comparison row without throwing away regressions."""

    status = str(row.get("status") or "").lower()
    failure_stage = str(row.get("failure_stage") or "")
    if status != "success" or failure_stage in STRUCTURAL_FAILURE_STAGES:
        return "structural_failure"
    deltas = [
        _float_or_none(row.get("hpwl_delta_pct")),
        _float_or_none(row.get("overflow_delta_pct")),
        _float_or_none(row.get("routed_wirelength_delta_pct")),
        _float_or_none(row.get("grt_overflow_delta_pct")),
        _float_or_none(row.get("drc_count_delta_pct")),
    ]
    finite = [value for value in deltas if value is not None]
    if any(value > severe_threshold_pct for value in finite):
        return "severe_regression"
    if any(value > 0.0 for value in finite):
        return "metric_regression"
    return "success_or_improvement"


def add_outcome_labels(
    rows: Iterable[dict[str, Any]],
    *,
    severe_threshold_pct: float = 10.0,
) -> list[dict[str, Any]]:
    labeled = []
    for row in rows:
        enriched = dict(row)
        enriched["outcome_label"] = classify_result(
            enriched,
            severe_threshold_pct=severe_threshold_pct,
        )
        labeled.append(enriched)
    return labeled


def aggregate_rank_scores(
    rows: Iterable[dict[str, Any]],
    *,
    metric_names: tuple[str, ...] = ("hpwl_delta_pct", "overflow_delta_pct"),
    severe_threshold_pct: float = 10.0,
) -> list[RankSummary]:
    """Aggregate coefficient-free ranks.

    Rows are ranked separately for each design/seed and metric. Lower metric
    deltas are better. Failed/non-finite cells receive the worst rank for the
    design/seed group. The aggregate averages metrics, then seeds within each
    design, then designs with equal design weight.
    """

    labeled_rows = add_outcome_labels(rows, severe_threshold_pct=severe_threshold_pct)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in labeled_rows:
        grouped[(str(row.get("design")), int(row.get("seed") or 0))].append(row)

    objective_design_seed_scores: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    objective_metric_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    objective_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for (design, seed), group_rows in grouped.items():
        objectives = [str(row.get("objective_id")) for row in group_rows]
        group_size = max(1, len(set(objectives)))
        for row in group_rows:
            objective_rows[str(row.get("objective_id"))].append(row)
        for metric_name in metric_names:
            ranks = _rank_group(group_rows, metric_name, worst_rank=float(group_size))
            for row in group_rows:
                objective_id = str(row.get("objective_id"))
                rank = ranks[id(row)]
                objective_design_seed_scores[objective_id][design][seed].append(rank)
                objective_metric_scores[objective_id][metric_name].append(rank)

    summaries: list[RankSummary] = []
    for objective_id, by_design in objective_design_seed_scores.items():
        design_ranks: dict[str, float] = {}
        cell_count = 0
        seed_ids: set[int] = set()
        for design, by_seed in by_design.items():
            seed_means = []
            for seed, ranks in by_seed.items():
                if ranks:
                    seed_means.append(sum(ranks) / len(ranks))
                    seed_ids.add(seed)
                    cell_count += 1
            if seed_means:
                design_ranks[design] = sum(seed_means) / len(seed_means)
        if not design_ranks:
            continue
        rows_for_objective = objective_rows[objective_id]
        runtime_values = [
            runtime
            for runtime in (_float_or_none(row.get("runtime_seconds")) for row in rows_for_objective)
            if runtime is not None
        ]
        labels = [
            classify_result(row, severe_threshold_pct=severe_threshold_pct)
            for row in rows_for_objective
        ]
        metric_ranks = {
            metric_name: sum(values) / len(values)
            for metric_name, values in objective_metric_scores[objective_id].items()
            if values
        }
        summaries.append(
            RankSummary(
                objective_id=objective_id,
                average_rank=sum(design_ranks.values()) / len(design_ranks),
                design_count=len(design_ranks),
                seed_count=len(seed_ids),
                cell_count=cell_count,
                structural_failure_count=labels.count("structural_failure"),
                metric_regression_count=labels.count("metric_regression"),
                severe_regression_count=labels.count("severe_regression"),
                mean_runtime_seconds=(
                    sum(runtime_values) / len(runtime_values) if runtime_values else None
                ),
                metric_ranks=metric_ranks,
                design_ranks=design_ranks,
            )
        )

    summaries.sort(
        key=lambda summary: (
            summary.average_rank,
            summary.structural_failure_count,
            summary.severe_regression_count,
            summary.mean_runtime_seconds if summary.mean_runtime_seconds is not None else math.inf,
            summary.objective_id,
        )
    )
    return summaries


def _rank_group(
    rows: list[dict[str, Any]],
    metric_name: str,
    *,
    worst_rank: float,
) -> dict[int, float]:
    finite_items = []
    non_finite_ids = []
    for row in rows:
        value = _float_or_none(row.get(metric_name))
        if str(row.get("status") or "").lower() != "success" or value is None:
            non_finite_ids.append(id(row))
        else:
            finite_items.append((id(row), value))

    finite_items.sort(key=lambda item: item[1])
    ranks: dict[int, float] = {}
    position = 1
    index = 0
    while index < len(finite_items):
        value = finite_items[index][1]
        end = index + 1
        while end < len(finite_items) and finite_items[end][1] == value:
            end += 1
        average_rank = (position + position + (end - index) - 1) / 2.0
        for row_id, _ in finite_items[index:end]:
            ranks[row_id] = average_rank
        position += end - index
        index = end
    for row_id in non_finite_ids:
        ranks[row_id] = worst_rank
    return ranks


def _float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None
