"""SQLite-backed candidate, metric, and failure store."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coevop.objectives.spec import ObjectiveSpec, objective_spec_to_dict


@dataclass(frozen=True)
class CandidateRecord:
    objective_id: str
    generation: int
    provider: str
    prompt_id: str
    status: str
    failure_reason: str | None
    objective_json: str
    artifact_path: str
    parent_ids: list[str]
    complexity: int
    term_set: list[str]


@dataclass(frozen=True)
class EvaluationRecord:
    objective_id: str
    tier: str
    dataset: str
    score: float
    metrics_json: str
    artifact_path: str | None
    runtime_seconds: float | None


@dataclass(frozen=True)
class FailureRecord:
    generation: int
    provider: str
    prompt_id: str
    failure_reason: str
    payload_json: str | None


class SQLiteStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.init_schema()

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            create table if not exists candidates (
                objective_id text primary key,
                generation integer not null,
                provider text not null,
                prompt_id text not null,
                status text not null,
                failure_reason text,
                objective_json text not null,
                artifact_path text not null,
                parent_ids_json text not null,
                complexity integer not null,
                term_set_json text not null,
                created_at text not null default current_timestamp
            );

            create table if not exists evaluations (
                id integer primary key autoincrement,
                objective_id text not null,
                tier text not null,
                dataset text not null,
                score real not null,
                metrics_json text not null,
                artifact_path text,
                runtime_seconds real,
                created_at text not null default current_timestamp,
                unique(objective_id, tier, dataset)
            );

            create table if not exists failures (
                id integer primary key autoincrement,
                generation integer not null,
                provider text not null,
                prompt_id text not null,
                failure_reason text not null,
                payload_json text,
                created_at text not null default current_timestamp
            );
            """
        )
        self.conn.commit()

    def has_candidate(self, objective_id: str) -> bool:
        row = self.conn.execute(
            "select 1 from candidates where objective_id = ?",
            (objective_id,),
        ).fetchone()
        return row is not None

    def upsert_candidate(
        self,
        spec: ObjectiveSpec,
        *,
        generation: int,
        provider: str,
        prompt_id: str,
        status: str,
        artifact_path: str,
        failure_reason: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            insert into candidates (
                objective_id, generation, provider, prompt_id, status, failure_reason,
                objective_json, artifact_path, parent_ids_json, complexity, term_set_json
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(objective_id) do update set
                status = excluded.status,
                failure_reason = excluded.failure_reason,
                artifact_path = excluded.artifact_path
            """,
            (
                spec.id,
                generation,
                provider,
                prompt_id,
                status,
                failure_reason,
                json.dumps(objective_spec_to_dict(spec), sort_keys=True),
                artifact_path,
                json.dumps(spec.parent_ids, sort_keys=True),
                spec.complexity,
                json.dumps(spec.term_set, sort_keys=True),
            ),
        )
        self.conn.commit()

    def record_evaluation(
        self,
        *,
        objective_id: str,
        tier: str,
        dataset: str,
        score: float,
        metrics: dict[str, Any],
        artifact_path: str | None = None,
        runtime_seconds: float | None = None,
    ) -> None:
        self.conn.execute(
            """
            insert into evaluations (
                objective_id, tier, dataset, score, metrics_json, artifact_path, runtime_seconds
            )
            values (?, ?, ?, ?, ?, ?, ?)
            on conflict(objective_id, tier, dataset) do update set
                score = excluded.score,
                metrics_json = excluded.metrics_json,
                artifact_path = excluded.artifact_path,
                runtime_seconds = excluded.runtime_seconds
            """,
            (
                objective_id,
                tier,
                dataset,
                float(score),
                json.dumps(metrics, sort_keys=True),
                artifact_path,
                runtime_seconds,
            ),
        )
        self.conn.commit()

    def record_failure(
        self,
        *,
        generation: int,
        provider: str,
        prompt_id: str,
        failure_reason: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.conn.execute(
            """
            insert into failures (generation, provider, prompt_id, failure_reason, payload_json)
            values (?, ?, ?, ?, ?)
            """,
            (
                generation,
                provider,
                prompt_id,
                failure_reason,
                json.dumps(payload, sort_keys=True) if payload is not None else None,
            ),
        )
        self.conn.commit()

    def top_candidates(self, limit: int = 10, tier: str = "tier1") -> list[dict[str, Any]]:
        return self.scored_candidates(limit=limit, tier=tier)

    def scored_candidates(
        self,
        limit: int | None = 10,
        tier: str = "tier1",
    ) -> list[dict[str, Any]]:
        limit_clause = "" if limit is None else "limit ?"
        params: tuple[Any, ...] = (tier,) if limit is None else (tier, limit)
        rows = self.conn.execute(
            f"""
            select
                c.objective_id,
                c.generation,
                c.provider,
                c.prompt_id,
                c.status,
                c.artifact_path,
                c.parent_ids_json,
                c.complexity,
                c.term_set_json,
                e.dataset,
                e.score,
                e.metrics_json
            from candidates c
            join evaluations e on e.objective_id = c.objective_id
            where e.tier = ? and c.status = 'scored'
            order by e.score desc
            {limit_clause}
            """,
            params,
        ).fetchall()
        return [_decode_row(row) for row in rows]

    def count_candidates(self) -> int:
        row = self.conn.execute("select count(*) as n from candidates").fetchone()
        return int(row["n"])

    def count_failures(self) -> int:
        row = self.conn.execute("select count(*) as n from failures").fetchone()
        return int(row["n"])


def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    for key in ("parent_ids_json", "term_set_json", "metrics_json"):
        if key in payload and payload[key] is not None:
            payload[key.removesuffix("_json")] = json.loads(payload.pop(key))
    return payload
