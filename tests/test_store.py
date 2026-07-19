from pathlib import Path

import pytest

from coevop.objectives.spec import parse_objective_spec
from coevop.store.sqlite import SQLiteStore


def _spec(term: str = "rudy_p95"):
    return parse_objective_spec(
        {
            "id": "",
            "rationale": f"Use {term} as a simple routability proxy.",
            "parent_ids": [],
            "declared_term_usage": [term],
            "ast": {"op": "term", "name": term},
        },
        created_by="test",
    )


def test_sqlite_store_records_candidates_evaluations_and_failures(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "evolution.sqlite")
    spec = _spec()

    try:
        store.upsert_candidate(
            spec,
            generation=0,
            provider="mock",
            prompt_id="prompt_0",
            status="scored",
            artifact_path=str(tmp_path / "objective.json"),
        )
        store.record_evaluation(
            objective_id=spec.id,
            tier="tier1",
            dataset="fixture",
            score=0.75,
            metrics={"score": 0.75},
            artifact_path=str(tmp_path / "objective.json"),
            runtime_seconds=0.01,
        )
        store.record_failure(
            generation=0,
            provider="mock",
            prompt_id="prompt_1",
            failure_reason="duplicate objective",
        )

        top = store.top_candidates(limit=1)
        assert store.has_candidate(spec.id)
        assert store.count_candidates() == 1
        assert store.count_failures() == 1
        assert top[0]["objective_id"] == spec.id
        assert top[0]["score"] == pytest.approx(0.75)
        assert top[0]["term_set"] == ["rudy_p95"]
        assert top[0]["metrics"]["score"] == pytest.approx(0.75)
        assert store.scored_candidates(limit=None)[0]["objective_id"] == spec.id
    finally:
        store.close()
