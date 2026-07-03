"""Handwritten offline baselines for v0 Tier-1 scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObjectiveDefinition:
    objective_id: str
    description: str
    terms: dict[str, float]

    @property
    def formula(self) -> str:
        chunks = []
        for term, weight in self.terms.items():
            chunks.append(f"{weight:g}*z({term})")
        return " + ".join(chunks)


BASELINE_OBJECTIVES = (
    ObjectiveDefinition(
        objective_id="rudy_mean_proxy",
        description="RUDY mean as a cheap routability pressure proxy.",
        terms={"feature.rudy.mean": 1.0},
    ),
    ObjectiveDefinition(
        objective_id="rudy_p95_proxy",
        description="RUDY p95 as a hotspot-sensitive routability pressure proxy.",
        terms={"feature.rudy.p95": 1.0},
    ),
    ObjectiveDefinition(
        objective_id="rudy_pin_p95_proxy",
        description="Pin-aware RUDY p95 proxy.",
        terms={"feature.rudy_pin.p95": 1.0},
    ),
    ObjectiveDefinition(
        objective_id="density_p95_proxy",
        description="Cell-density p95 proxy.",
        terms={"feature.cell_density.p95": 1.0},
    ),
    ObjectiveDefinition(
        objective_id="rudy_density_proxy",
        description="RUDY hotspot plus cell-density hotspot proxy.",
        terms={"feature.rudy.p95": 1.0, "feature.cell_density.p95": 0.35},
    ),
    ObjectiveDefinition(
        objective_id="rudy_pin_density_proxy",
        description="Pin-aware RUDY hotspot plus density hotspot proxy.",
        terms={"feature.rudy_pin.p95": 1.0, "feature.cell_density.p95": 0.35},
    ),
    ObjectiveDefinition(
        objective_id="rudy_long_short_mix",
        description="Blend of short, long, and pin-long RUDY pressure.",
        terms={
            "feature.rudy_short.p95": 0.5,
            "feature.rudy_long.p95": 0.75,
            "feature.rudy_pin_long.p95": 0.75,
        },
    ),
)


def _column(rows: list[dict[str, object]], name: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(name, math.nan)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(math.nan)
    return np.asarray(values, dtype=np.float64)


def _zscore(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.full(values.shape, math.nan, dtype=np.float64)
    mean = np.mean(values[finite])
    std = np.std(values[finite])
    if not math.isfinite(float(std)) or std == 0:
        out = np.zeros(values.shape, dtype=np.float64)
        out[~finite] = math.nan
        return out
    out = (values - mean) / std
    out[~finite] = math.nan
    return out


def score_objective(
    rows: list[dict[str, object]],
    objective: ObjectiveDefinition,
) -> np.ndarray:
    score = np.zeros(len(rows), dtype=np.float64)
    used = 0
    for term, weight in objective.terms.items():
        values = _zscore(_column(rows, term))
        if not np.any(np.isfinite(values)):
            continue
        score += float(weight) * np.nan_to_num(values, nan=0.0)
        used += 1

    if used == 0:
        return np.full(len(rows), math.nan, dtype=np.float64)
    return score
