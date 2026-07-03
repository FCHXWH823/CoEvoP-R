"""Tier-1 offline scoring against CircuitNet scalar labels."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from coevop.eval.correlations import finite_pair, kendall_tau_b, spearman
from coevop.objectives.baselines import BASELINE_OBJECTIVES, ObjectiveDefinition, score_objective
from coevop.objectives.dsl import score_components, score_spec
from coevop.objectives.spec import ObjectiveSpec, objective_spec_to_dict


DEFAULT_TARGETS = (
    "target.congestion_egr_overflow_mean",
    "target.congestion_gr_overflow_mean",
    "target.congestion_gr_util_p95",
    "target.drc_hotspot_sum",
)

TARGET_BADNESS_DIRECTIONS = {
    "target.congestion_egr_overflow_mean": 1,
    "target.congestion_gr_overflow_mean": 1,
    "target.congestion_gr_util_p95": 1,
    "target.drc_hotspot_sum": 1,
}


@dataclass(frozen=True)
class TargetCorrelation:
    target: str
    spearman: float
    kendall: float
    n: int


@dataclass(frozen=True)
class ObjectiveRanking:
    objective_id: str
    description: str
    formula: str
    score: float
    correlations: list[TargetCorrelation]


def _target_vector(rows: list[dict[str, object]], target: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            values.append(float(row.get(target, math.nan)))
        except (TypeError, ValueError):
            values.append(math.nan)
    return np.asarray(values, dtype=np.float64)


def evaluate_objective(
    rows: list[dict[str, object]],
    objective: ObjectiveDefinition,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    score_mode: str = "signed_badness_mean",
) -> ObjectiveRanking:
    objective_scores = score_objective(rows, objective)
    correlations = []
    for target in targets:
        target_values = _target_vector(rows, target)
        x, y = finite_pair(objective_scores, target_values)
        correlations.append(
            TargetCorrelation(
                target=target,
                spearman=spearman(x, y),
                kendall=kendall_tau_b(x, y),
                n=len(x),
            )
        )

    score = aggregate_correlations(correlations, score_mode=score_mode)
    return ObjectiveRanking(
        objective_id=objective.objective_id,
        description=objective.description,
        formula=objective.formula,
        score=score,
        correlations=correlations,
    )


def evaluate_scores(
    rows: list[dict[str, object]],
    objective_id: str,
    description: str,
    formula: str,
    scores: np.ndarray,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    score_mode: str = "signed_badness_mean",
) -> ObjectiveRanking:
    correlations = []
    for target in targets:
        target_values = _target_vector(rows, target)
        x, y = finite_pair(scores, target_values)
        correlations.append(
            TargetCorrelation(
                target=target,
                spearman=spearman(x, y),
                kendall=kendall_tau_b(x, y),
                n=len(x),
            )
        )

    score = aggregate_correlations(correlations, score_mode=score_mode)
    return ObjectiveRanking(
        objective_id=objective_id,
        description=description,
        formula=formula,
        score=score,
        correlations=correlations,
    )


def evaluate_spec(
    rows: list[dict[str, object]],
    spec: ObjectiveSpec,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    score_mode: str = "signed_badness_mean",
) -> ObjectiveRanking:
    scores = score_spec(rows, spec)
    return evaluate_scores(
        rows=rows,
        objective_id=spec.id,
        description=spec.rationale,
        formula=json.dumps(objective_spec_to_dict(spec)["ast"], sort_keys=True),
        scores=scores,
        targets=targets,
        score_mode=score_mode,
    )


def component_diagnostics(
    rows: list[dict[str, object]],
    spec: ObjectiveSpec,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    score_mode: str = "signed_badness_mean",
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for component_name, scores in score_components(rows, spec).items():
        ranking = evaluate_scores(
            rows=rows,
            objective_id=f"{spec.id}:{component_name}",
            description=f"component {component_name}",
            formula=json.dumps((spec.components or {}).get(component_name, {}), sort_keys=True),
            scores=scores,
            targets=targets,
            score_mode=score_mode,
        )
        diagnostics[component_name] = {
            "score": ranking.score,
            "stats": _score_stats(scores),
            "correlations": [asdict(correlation) for correlation in ranking.correlations],
        }
    return diagnostics


def rank_objectives(
    rows: list[dict[str, object]],
    objectives: tuple[ObjectiveDefinition, ...] = BASELINE_OBJECTIVES,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    score_mode: str = "signed_badness_mean",
) -> list[ObjectiveRanking]:
    rankings = [
        evaluate_objective(rows, objective, targets, score_mode=score_mode)
        for objective in objectives
    ]
    return sorted(
        rankings,
        key=lambda item: item.score if math.isfinite(item.score) else -math.inf,
        reverse=True,
    )


def rankings_to_dict(rankings: list[ObjectiveRanking], row_count: int) -> dict:
    return {
        "row_count": row_count,
        "targets": list(DEFAULT_TARGETS),
        "target_badness_directions": TARGET_BADNESS_DIRECTIONS,
        "rankings": [asdict(ranking) for ranking in rankings],
    }


def aggregate_correlations(
    correlations: list[TargetCorrelation],
    score_mode: str = "signed_badness_mean",
) -> float:
    values = []
    for correlation in correlations:
        if not math.isfinite(correlation.spearman):
            continue
        direction = TARGET_BADNESS_DIRECTIONS.get(correlation.target, 1)
        aligned = float(direction) * correlation.spearman
        if score_mode == "signed_badness_mean":
            values.append(aligned)
        elif score_mode == "absolute_mean":
            values.append(abs(aligned))
        else:
            raise ValueError(f"unknown Tier-1 score mode: {score_mode}")
    return float(np.mean(values)) if values else math.nan


def _score_stats(scores: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(scores, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "finite_count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "finite_count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def write_rankings_json(
    rankings: list[ObjectiveRanking],
    row_count: int,
    output: str | Path,
) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rankings_to_dict(rankings, row_count), f, indent=2, sort_keys=True)
        f.write("\n")
    return output_path


def write_rankings_csv(rankings: list[ObjectiveRanking], output: str | Path) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "objective_id",
                "score",
                "target",
                "spearman",
                "kendall",
                "n",
                "formula",
            ],
        )
        writer.writeheader()
        for rank, ranking in enumerate(rankings, start=1):
            for correlation in ranking.correlations:
                writer.writerow(
                    {
                        "rank": rank,
                        "objective_id": ranking.objective_id,
                        "score": ranking.score,
                        "target": correlation.target,
                        "spearman": correlation.spearman,
                        "kendall": correlation.kendall,
                        "n": correlation.n,
                        "formula": ranking.formula,
                    }
                )
    return output_path
