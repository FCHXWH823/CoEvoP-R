"""Lightweight candidate memory for offline objective evolution."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any


DEFAULT_OPERATOR_WEIGHTS: dict[str, float] = {
    "crossover": 0.35,
    "mutate": 0.30,
    "param_tune": 0.20,
    "simplify": 0.15,
}

ALLOWED_OPERATORS = ("init", "mutate", "crossover", "param_tune", "simplify")


@dataclass(frozen=True)
class CandidateCluster:
    signature: tuple[Any, ...]
    candidates: list[dict[str, Any]]

    @property
    def best_score(self) -> float:
        return max(_finite_score(candidate.get("score")) for candidate in self.candidates)

    def best_candidate(self) -> dict[str, Any]:
        return max(self.candidates, key=lambda candidate: _finite_score(candidate.get("score")))


class CandidateMemory:
    def __init__(self, candidates: list[dict[str, Any]], *, islands: int = 5) -> None:
        self.islands = max(1, int(islands))
        self._clusters_by_island: list[dict[tuple[Any, ...], list[dict[str, Any]]]] = [
            {} for _ in range(self.islands)
        ]
        for candidate in candidates:
            signature = candidate_signature(candidate)
            island_id = _stable_island(signature, self.islands)
            self._clusters_by_island[island_id].setdefault(signature, []).append(candidate)

    @property
    def is_empty(self) -> bool:
        return not any(self._clusters_by_island)

    def sample_parents(
        self,
        count: int,
        *,
        rng: random.Random,
    ) -> list[dict[str, Any]]:
        clusters = self._sample_island_clusters(rng)
        if not clusters:
            return []
        selected: list[dict[str, Any]] = []
        used_signatures: set[tuple[Any, ...]] = set()
        for _ in range(max(1, count)):
            available = [cluster for cluster in clusters if cluster.signature not in used_signatures]
            if not available:
                available = clusters
            cluster = _weighted_cluster_choice(available, rng)
            selected.append(cluster.best_candidate())
            used_signatures.add(cluster.signature)
        return selected

    def _sample_island_clusters(self, rng: random.Random) -> list[CandidateCluster]:
        non_empty = [island for island in self._clusters_by_island if island]
        if not non_empty:
            return []
        island = rng.choice(non_empty)
        return [
            CandidateCluster(signature=signature, candidates=candidates)
            for signature, candidates in island.items()
        ]


def candidate_signature(candidate: dict[str, Any]) -> tuple[Any, ...]:
    metrics = candidate.get("metrics", {})
    correlations = (
        metrics.get("split_correlations", {})
        .get("validation")
        or metrics.get("correlations", [])
    )
    values = []
    for correlation in correlations:
        try:
            value = float(correlation.get("spearman", math.nan))
        except (TypeError, ValueError, AttributeError):
            value = math.nan
        values.append(round(value, 2) if math.isfinite(value) else "nan")
    deployable = bool(metrics.get("dreamplace_deployable", False))
    return (*values, f"deployable:{str(deployable).lower()}")


def choose_operator(
    generation: int,
    *,
    memory_empty: bool,
    rng: random.Random,
    weights: dict[str, float] | None = None,
) -> str:
    if generation == 0 or memory_empty:
        return "init"
    weights = normalize_operator_weights(weights or DEFAULT_OPERATOR_WEIGHTS)
    operators = list(weights)
    return rng.choices(operators, weights=[weights[operator] for operator in operators], k=1)[0]


def normalize_operator_weights(weights: dict[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for operator, value in weights.items():
        if operator not in DEFAULT_OPERATOR_WEIGHTS:
            raise ValueError(f"unknown evolution operator: {operator}")
        numeric = float(value)
        if numeric < 0 or not math.isfinite(numeric):
            raise ValueError(f"operator weight must be finite and non-negative: {operator}")
        if numeric > 0:
            normalized[operator] = numeric
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("at least one evolution operator weight must be positive")
    return {operator: round(value / total, 12) for operator, value in normalized.items()}


def parse_operator_weights(text: str | None) -> dict[str, float]:
    if not text:
        return dict(DEFAULT_OPERATOR_WEIGHTS)
    weights: dict[str, float] = {}
    for chunk in text.split(","):
        if not chunk.strip():
            continue
        if "=" not in chunk:
            raise ValueError("operator weights must use name=value entries")
        name, value = chunk.split("=", maxsplit=1)
        weights[name.strip()] = float(value.strip())
    return normalize_operator_weights(weights)


def _weighted_cluster_choice(
    clusters: list[CandidateCluster],
    rng: random.Random,
) -> CandidateCluster:
    best_scores = [cluster.best_score for cluster in clusters]
    finite_scores = [score for score in best_scores if math.isfinite(score)]
    if not finite_scores:
        return rng.choice(clusters)
    floor = min(finite_scores)
    weights = [
        (score - floor + 1e-3) if math.isfinite(score) else 1e-3
        for score in best_scores
    ]
    return rng.choices(clusters, weights=weights, k=1)[0]


def _stable_island(signature: tuple[Any, ...], islands: int) -> int:
    encoded = "|".join(str(part) for part in signature)
    return sum(ord(char) for char in encoded) % islands


def _finite_score(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return -math.inf
    return numeric if math.isfinite(numeric) else -math.inf
