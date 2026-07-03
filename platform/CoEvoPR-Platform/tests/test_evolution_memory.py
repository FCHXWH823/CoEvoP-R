import random

import pytest

from coevop.evolution.memory import (
    CandidateMemory,
    candidate_signature,
    choose_operator,
    parse_operator_weights,
)


def _candidate(objective_id: str, spearman: float, score: float) -> dict:
    return {
        "objective_id": objective_id,
        "score": score,
        "metrics": {
            "dreamplace_deployable": False,
            "split_correlations": {
                "validation": [
                    {"target": "t0", "spearman": spearman},
                    {"target": "t1", "spearman": spearman / 2.0},
                ]
            },
        },
    }


def test_candidate_memory_samples_diverse_parent_signatures() -> None:
    candidates = [
        _candidate("a", 0.9, 0.9),
        _candidate("b", 0.9, 0.8),
        _candidate("c", 0.1, 0.7),
    ]
    memory = CandidateMemory(candidates, islands=1)

    parents = memory.sample_parents(2, rng=random.Random(0))
    signatures = {candidate_signature(parent) for parent in parents}

    assert len(parents) == 2
    assert len(signatures) == 2


def test_operator_scheduler_and_weight_parser() -> None:
    rng = random.Random(0)

    assert choose_operator(0, memory_empty=False, rng=rng) == "init"
    assert choose_operator(3, memory_empty=True, rng=rng) == "init"
    weights = parse_operator_weights("mutate=2,crossover=1")
    assert weights["mutate"] == pytest.approx(2 / 3)
    assert weights["crossover"] == pytest.approx(1 / 3)

    with pytest.raises(ValueError, match="unknown evolution operator"):
        parse_operator_weights("bad=1")
