"""Cheap inner-loop evolution using CircuitNet Tier-1 labels."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from coevop.datasets.circuitnet import load_manifest
from coevop.datasets.splits import (
    SplitSet,
    selected_rows,
    split_rows,
    split_rows_by_design_holdout,
)
from coevop.datasets.summaries import compute_summary_rows, write_summary_csv
from coevop.eval.tier1 import (
    ObjectiveRanking,
    component_diagnostics,
    evaluate_spec,
    rank_objectives,
    write_rankings_json,
)
from coevop.llm.providers import LLMProvider, provider_from_name
from coevop.objectives.dsl import score_spec
from coevop.objectives.fitting import fit_objective_constants
from coevop.objectives.program import PROGRAM_SYSTEM_CONTRACT, program_example
from coevop.objectives.spec import ObjectiveSpec, objective_spec_to_dict, write_objective_spec
from coevop.objectives.terms import term_descriptions, term_names, unsupported_terms_for_scope
from coevop.evolution.memory import (
    DEFAULT_OPERATOR_WEIGHTS,
    CandidateMemory,
    candidate_signature,
    choose_operator,
    normalize_operator_weights,
)
from coevop.store.sqlite import SQLiteStore


@dataclass(frozen=True)
class EvolutionSummary:
    run_dir: str
    db_path: str
    provider: str
    dataset: str
    row_count: int
    split_strategy: str
    split_row_counts: dict[str, int]
    selection_split: str
    term_scope: str
    reflection_mode: str
    diversity_lambda: float
    operator_weights: dict[str, float]
    islands: int
    constant_fit_enabled: bool
    requested_candidates: int
    generations: int
    accepted_candidates: int
    rejected_candidates: int
    top_candidates: list[dict[str, Any]]


def run_offline_evolution(
    *,
    manifest_path: str | Path,
    provider_name: str,
    run_dir: str | Path,
    population_size: int = 20,
    generations: int = 1,
    elite_count: int = 5,
    max_attempts_per_candidate: int = 3,
    seed: int = 0,
    db_path: str | Path | None = None,
    max_samples: int | None = None,
    split_strategy: str = "auto",
    selection_split: str = "validation",
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    heldout_fraction: float = 0.2,
    reflection_mode: str = "accepted",
    diversity_lambda: float = 0.05,
    operator_weights: dict[str, float] | None = None,
    islands: int = 5,
    enable_constant_fit: bool = True,
    term_scope: str = "tier1",
    validation_design: str | None = None,
    heldout_design: str | None = None,
    external_feedback: dict[str, Any] | None = None,
) -> EvolutionSummary:
    rng = random.Random(seed)
    operator_weights = normalize_operator_weights(operator_weights or DEFAULT_OPERATOR_WEIGHTS)
    islands = max(1, int(islands))
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    objective_dir = run_dir / "objectives"
    objective_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(db_path) if db_path is not None else run_dir / "evolution.sqlite"
    store = SQLiteStore(db_path)
    provider = provider_from_name(provider_name)

    manifest = load_manifest(manifest_path)
    rows = compute_summary_rows(manifest, max_samples=max_samples)
    write_summary_csv(rows, run_dir / "scalars.csv")
    if validation_design or heldout_design:
        split_set = split_rows_by_design_holdout(
            rows,
            validation_design=validation_design,
            heldout_design=heldout_design,
            seed=seed,
        )
    else:
        split_set = split_rows(
            rows,
            strategy=split_strategy,
            seed=seed,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            heldout_fraction=heldout_fraction,
        )
    if selection_split not in split_set.splits:
        raise ValueError(f"unknown selection split: {selection_split}")
    if selection_split == "heldout":
        raise ValueError("heldout is reporting-only and cannot be used for selection")
    effective_selection_split = selection_split if split_set.splits[selection_split].rows else "train"
    selection_rows = selected_rows(split_set, selection_split)
    train_rows = selected_rows(split_set, "train")
    validation_rows = selected_rows(split_set, "validation")
    _write_split_manifest(split_set, run_dir / "split_manifest.json")

    baseline_rankings = rank_objectives(selection_rows)
    baseline_rankings_by_split = _baseline_rankings_by_split(split_set)
    write_rankings_json(
        baseline_rankings,
        len(selection_rows),
        run_dir / "baseline_rankings.json",
    )
    _write_rankings_by_split(
        baseline_rankings_by_split,
        split_set,
        run_dir / "baseline_rankings_by_split.json",
    )

    accepted = 0
    rejected = 0
    elites: list[ObjectiveSpec] = []
    dataset = _dataset_name(manifest_path)
    context = _initial_context(
        baseline_rankings,
        split_set,
        effective_selection_split,
        term_scope,
        external_feedback=external_feedback,
    )
    long_term_reflection_path = run_dir / "long_term_reflection.txt"
    long_term_reflection_path.write_text("", encoding="utf-8")

    try:
        for generation in range(generations):
            for index in range(population_size):
                memory = CandidateMemory(
                    store.scored_candidates(limit=None),
                    islands=islands,
                )
                operator = choose_operator(
                    generation,
                    memory_empty=memory.is_empty,
                    rng=rng,
                    weights=operator_weights,
                )
                prompt_id = f"generation_{generation:03d}_{operator}_{index:04d}"
                spec, producer_error, fit_report = _produce_candidate(
                    provider=provider,
                    operator=operator,
                    memory=memory,
                    context=context,
                    rng=rng,
                    train_rows=train_rows,
                    validation_rows=validation_rows,
                    term_scope=term_scope,
                )
                attempt = 1
                while attempt <= max_attempts_per_candidate:
                    if spec is None:
                        store.record_failure(
                            generation=generation,
                            provider=provider_name,
                            prompt_id=f"{prompt_id}_attempt_{attempt}",
                            failure_reason=producer_error or "provider did not produce a candidate",
                        )
                        rejected += 1
                    elif store.has_candidate(spec.id):
                        store.record_failure(
                            generation=generation,
                            provider=provider_name,
                            prompt_id=f"{prompt_id}_attempt_{attempt}",
                            failure_reason=f"duplicate objective {spec.id}",
                            payload=objective_spec_to_dict(spec),
                        )
                        rejected += 1
                    else:
                        reason = _rejection_reason(rows, selection_rows, spec)
                        if reason is None:
                            accepted += 1
                            _score_and_store_candidate(
                                split_set=split_set,
                                selection_split=effective_selection_split,
                                spec=spec,
                                generation=generation,
                                provider_name=provider_name,
                                prompt_id=prompt_id,
                                objective_dir=objective_dir,
                                store=store,
                                dataset=dataset,
                                provider=provider,
                                reflection_mode=reflection_mode,
                                operator=operator,
                                fit_report=fit_report,
                                long_term_reflection=_read_text(long_term_reflection_path),
                            )
                            if enable_constant_fit and operator != "param_tune":
                                fitted_spec, fitted_report = fit_objective_constants(
                                    spec,
                                    train_rows=train_rows,
                                    validation_rows=validation_rows,
                                    term_scope=term_scope,
                                )
                                if fitted_spec is not None and not store.has_candidate(fitted_spec.id):
                                    fitted_reason = _rejection_reason(
                                        rows,
                                        selection_rows,
                                        fitted_spec,
                                    )
                                    if fitted_reason is None:
                                        accepted += 1
                                        _score_and_store_candidate(
                                            split_set=split_set,
                                            selection_split=effective_selection_split,
                                            spec=fitted_spec,
                                            generation=generation,
                                            provider_name=provider_name,
                                            prompt_id=f"{prompt_id}_fit",
                                            objective_dir=objective_dir,
                                            store=store,
                                            dataset=dataset,
                                            provider=provider,
                                            reflection_mode=reflection_mode,
                                            operator="param_tune",
                                            fit_report=fitted_report,
                                            long_term_reflection=_read_text(
                                                long_term_reflection_path,
                                            ),
                                        )
                            selected_candidates = select_candidates(
                                store.scored_candidates(limit=None),
                                limit=elite_count,
                                diversity_lambda=diversity_lambda,
                            )
                            elites = _load_specs_from_candidates(selected_candidates)
                            _write_long_term_reflection(
                                store,
                                long_term_reflection_path,
                                effective_selection_split,
                                diversity_lambda,
                            )
                            context = _context_from_store(
                                store,
                                baseline_rankings,
                                split_set,
                                effective_selection_split,
                                diversity_lambda,
                                long_term_reflection_path,
                                term_scope,
                                external_feedback=external_feedback,
                            )
                            break
                        store.record_failure(
                            generation=generation,
                            provider=provider_name,
                            prompt_id=f"{prompt_id}_attempt_{attempt}",
                            failure_reason=reason,
                            payload=objective_spec_to_dict(spec),
                        )
                        rejected += 1
                    attempt += 1
                    spec, producer_error, fit_report = _produce_candidate(
                        provider=provider,
                        operator=operator,
                        memory=memory,
                        context=context,
                        rng=rng,
                        train_rows=train_rows,
                        validation_rows=validation_rows,
                        term_scope=term_scope,
                    )
    finally:
        top_candidates = select_candidates(
            store.scored_candidates(limit=None),
            limit=elite_count,
            diversity_lambda=diversity_lambda,
        )
        write_top_candidates(top_candidates, run_dir / "top_candidates.json", run_dir / "top_candidates.csv")
        write_comparison_table(
            top_candidates,
            baseline_rankings_by_split,
            run_dir / "comparison_table.csv",
        )
        summary = EvolutionSummary(
            run_dir=str(run_dir),
            db_path=str(db_path),
            provider=provider_name,
            dataset=dataset,
            row_count=len(rows),
            split_strategy=split_set.strategy,
            split_row_counts=split_set.row_counts,
            selection_split=effective_selection_split,
            term_scope=term_scope,
            reflection_mode=reflection_mode,
            diversity_lambda=diversity_lambda,
            operator_weights=operator_weights,
            islands=islands,
            constant_fit_enabled=enable_constant_fit,
            requested_candidates=population_size * generations,
            generations=generations,
            accepted_candidates=accepted,
            rejected_candidates=rejected,
            top_candidates=top_candidates,
        )
        (run_dir / "summary.json").write_text(
            json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        store.close()

    return summary


def write_top_candidates(
    top_candidates: list[dict[str, Any]],
    json_path: str | Path,
    csv_path: str | Path,
) -> tuple[Path, Path]:
    json_output = Path(json_path)
    csv_output = Path(csv_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps({"top_candidates": top_candidates}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with csv_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "objective_id",
                "score",
                "generation",
                "provider",
                "complexity",
                "term_set",
                "train_score",
                "validation_score",
                "heldout_score",
                "artifact_path",
            ],
        )
        writer.writeheader()
        for rank, candidate in enumerate(top_candidates, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "objective_id": candidate["objective_id"],
                    "score": candidate["score"],
                    "generation": candidate["generation"],
                    "provider": candidate["provider"],
                    "complexity": candidate["complexity"],
                    "term_set": ",".join(candidate.get("term_set", [])),
                    "train_score": _split_score(candidate, "train"),
                    "validation_score": _split_score(candidate, "validation"),
                    "heldout_score": _split_score(candidate, "heldout"),
                    "artifact_path": candidate["artifact_path"],
                }
            )
    return json_output, csv_output


def write_comparison_table(
    top_candidates: list[dict[str, Any]],
    baseline_rankings_by_split: dict[str, list[ObjectiveRanking]],
    csv_path: str | Path,
) -> Path:
    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "kind",
        "rank",
        "objective_id",
        "selection_score",
        "train_score",
        "validation_score",
        "heldout_score",
        "complexity",
        "term_set",
        "formula",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in _baseline_comparison_rows(baseline_rankings_by_split):
            writer.writerow(row)
        for rank, candidate in enumerate(top_candidates, start=1):
            writer.writerow(
                {
                    "kind": "generated",
                    "rank": rank,
                    "objective_id": candidate["objective_id"],
                    "selection_score": candidate["score"],
                    "train_score": _split_score(candidate, "train"),
                    "validation_score": _split_score(candidate, "validation"),
                    "heldout_score": _split_score(candidate, "heldout"),
                    "complexity": candidate["complexity"],
                    "term_set": ",".join(candidate.get("term_set", [])),
                    "formula": json.dumps(candidate.get("metrics", {}).get("ast", {}), sort_keys=True),
                }
            )
    return output_path


def select_candidates(
    candidates: list[dict[str, Any]],
    *,
    limit: int,
    diversity_lambda: float = 0.05,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining = sorted(
        candidates,
        key=lambda candidate: _finite_score(candidate.get("score")),
        reverse=True,
    )
    while remaining and len(selected) < limit:
        best_index = 0
        best_value = -math.inf
        for index, candidate in enumerate(remaining):
            score = _finite_score(candidate.get("score"))
            diversity_penalty = diversity_lambda * _max_term_jaccard(candidate, selected)
            complexity_penalty = 0.002 * float(candidate.get("complexity", 0))
            adjusted = score - diversity_penalty - complexity_penalty
            if adjusted > best_value:
                best_value = adjusted
                best_index = index
        selected.append(remaining.pop(best_index))
    return selected


def _score_and_store_candidate(
    *,
    split_set: SplitSet,
    selection_split: str,
    spec: ObjectiveSpec,
    generation: int,
    provider_name: str,
    prompt_id: str,
    objective_dir: Path,
    store: SQLiteStore,
    dataset: str,
    provider: LLMProvider,
    reflection_mode: str,
    operator: str,
    fit_report: dict[str, Any] | None,
    long_term_reflection: str,
) -> None:
    start = time.time()
    objective_path = write_objective_spec(spec, objective_dir / f"{spec.id}.json")
    split_rankings = _evaluate_spec_by_split(split_set, spec)
    ranking = split_rankings[selection_split]
    runtime = time.time() - start
    undeployable_terms = unsupported_terms_for_scope(spec.term_set, "dreamplace")
    signature = candidate_signature(
        {
            "metrics": {
                "split_correlations": {
                    "validation": [
                        asdict(correlation)
                        for correlation in split_rankings["validation"].correlations
                    ]
                },
                "dreamplace_deployable": not undeployable_terms,
            }
        }
    )
    metrics = {
        "score": ranking.score,
        "selection_split": selection_split,
        "operator": operator,
        "generation": generation,
        "signature": list(signature),
        "dreamplace_deployable": not undeployable_terms,
        "dreamplace_undeployable_terms": undeployable_terms,
        "split_scores": {
            name: split_ranking.score for name, split_ranking in split_rankings.items()
        },
        "split_correlations": {
            name: [asdict(correlation) for correlation in split_ranking.correlations]
            for name, split_ranking in split_rankings.items()
        },
        "correlations": [asdict(correlation) for correlation in ranking.correlations],
        "rationale": spec.rationale,
        "ast": spec.ast,
        "source_program": spec.source_program,
        "components": spec.components,
        "component_diagnostics": {
            name: component_diagnostics(split.rows, spec) if split.rows else {}
            for name, split in split_set.splits.items()
        },
        "term_set": spec.term_set,
        "complexity": spec.complexity,
        "program_validation": {
            "component_count": len(spec.components or {}),
            "score_complexity": spec.complexity,
        },
        "long_term_reflection": long_term_reflection,
    }
    if fit_report is not None:
        metrics["fit_report"] = fit_report
    if reflection_mode == "accepted":
        metrics["reflection"] = _safe_reflection(provider, spec, metrics)
    store.upsert_candidate(
        spec,
        generation=generation,
        provider=provider_name,
        prompt_id=prompt_id,
        status="scored",
        artifact_path=str(objective_path),
    )
    store.record_evaluation(
        objective_id=spec.id,
        tier="tier1",
        dataset=dataset,
        score=ranking.score,
        metrics=metrics,
        artifact_path=str(objective_path),
        runtime_seconds=runtime,
    )


def _produce_candidate(
    *,
    provider: LLMProvider,
    operator: str,
    memory: CandidateMemory,
    context: dict[str, Any],
    rng: random.Random,
    train_rows: list[dict[str, object]],
    validation_rows: list[dict[str, object]],
    term_scope: str,
) -> tuple[ObjectiveSpec | None, str | None, dict[str, Any] | None]:
    try:
        if operator == "init" or memory.is_empty:
            return provider.generate(context=context, term_scope=term_scope), None, None

        parent_count = 2 if operator == "crossover" else 1
        parent_candidates = memory.sample_parents(parent_count, rng=rng)
        parents = _load_specs_from_candidates(parent_candidates)
        if not parents:
            return provider.generate(context=context, term_scope=term_scope), None, None

        if operator == "param_tune":
            fitted_spec, fit_report = fit_objective_constants(
                parents[0],
                train_rows=train_rows,
                validation_rows=validation_rows,
                term_scope=term_scope,
            )
            if fitted_spec is None:
                return (
                    None,
                    f"constant fitting did not produce a sibling: {fit_report.get('status')}",
                    fit_report,
                )
            return fitted_spec, None, fit_report

        if operator in {"mutate", "crossover", "simplify"}:
            feedback = {
                "operator": operator,
                "instruction": _operator_instruction(operator),
                "top_candidates": context.get("top_candidates", []),
                "reflections": context.get("reflections", []),
                "long_term_reflection": context.get("long_term_reflection", ""),
                "selection_split": context.get("selection_split", "validation"),
            }
            return provider.mutate(parents, feedback=feedback, term_scope=term_scope), None, None
        return None, f"unsupported evolution operator: {operator}", None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}", None


def _operator_instruction(operator: str) -> str:
    if operator == "mutate":
        return "Modify one parent to improve validation correlation while keeping the formula simple."
    if operator == "crossover":
        return (
            "Recombine two parents by keeping complementary useful components and avoiding "
            "duplicate terms."
        )
    if operator == "simplify":
        return (
            "Simplify the parent for better cross-design generalization; remove weak, flat, "
            "or overfit components."
        )
    return "Generate a stable routing-risk objective."


def _rejection_reason(
    all_rows: list[dict[str, object]],
    selection_rows: list[dict[str, object]],
    spec: ObjectiveSpec,
) -> str | None:
    scores = score_spec(all_rows, spec)
    finite = np.isfinite(scores)
    if not np.any(finite):
        return "objective produced no finite scores"
    if np.any(np.abs(scores[finite]) > 1e12):
        return "objective produced unstable large scores"
    if float(np.std(scores[finite])) == 0.0:
        return "objective produced a constant score"
    if spec.complexity > 12:
        return "objective exceeds complexity budget"
    ranking = evaluate_spec(selection_rows, spec)
    if not math.isfinite(ranking.score):
        return "objective produced no finite Tier-1 correlation"
    return None


def _load_specs_from_candidates(candidates: list[dict[str, Any]]) -> list[ObjectiveSpec]:
    specs = []
    for candidate in candidates:
        payload = json.loads(Path(candidate["artifact_path"]).read_text(encoding="utf-8"))
        specs.append(
            ObjectiveSpec(
                id=payload["id"],
                ast=payload["ast"],
                constants={str(k): float(v) for k, v in payload.get("constants", {}).items()},
                term_set=[str(term) for term in payload.get("term_set", [])],
                complexity=int(payload["complexity"]),
                created_by=str(payload["created_by"]),
                parent_ids=[str(parent) for parent in payload.get("parent_ids", [])],
                rationale=str(payload.get("rationale", "")),
                source_program=payload.get("source_program"),
                components=payload.get("components"),
            )
        )
    return specs


def _initial_context(
    baseline_rankings: list[Any],
    split_set: SplitSet,
    selection_split: str,
    term_scope: str,
    external_feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if term_scope == "deployable":
        environment_goal = (
            "Find a simple symbolic objective over placement-state observables that "
            "has useful Tier-1 proxy feedback and remains eligible for DREAMPlace "
            "Tier-2 deployment after gradient checks."
        )
    else:
        environment_goal = (
            "Find a simple symbolic objective with high validation Spearman correlation "
            "against routing badness labels while preserving held-out generalization."
        )
    context = {
        "dataset": {
            "row_count": sum(split_set.row_counts.values()),
            "split_row_counts": split_set.row_counts,
            "split_group_counts": split_set.group_counts,
            "split_strategy": split_set.strategy,
            "selection_split": selection_split,
            "source": "CircuitNet Tier-1 scalar summaries",
        },
        "selection_split": selection_split,
        "placement_environment": {
            "term_scope": term_scope,
            "available_terms": term_names(term_scope),
            "term_descriptions": term_descriptions(term_scope),
            "objective_program_contract": PROGRAM_SYSTEM_CONTRACT,
            "example_program": program_example(term_scope),
            "label_policy": (
                "CircuitNet labels are feedback targets only; do not use label or "
                "target fields in objective programs."
            ),
        },
        "label_targets": {
            "target.congestion_egr_overflow_mean": "badness: early global-routing overflow mean",
            "target.congestion_gr_overflow_mean": "badness: global-routing overflow mean",
            "target.congestion_gr_util_p95": "badness: global-routing utilization hotspot p95",
            "target.drc_hotspot_sum": "badness: DRC hotspot sum",
        },
        "baseline_top": [
            {
                "objective_id": ranking.objective_id,
                "score": ranking.score,
                "formula": ranking.formula,
            }
            for ranking in baseline_rankings[:5]
        ],
        "goal": environment_goal,
        "long_term_reflection": "",
    }
    if external_feedback:
        context["external_feedback"] = external_feedback
    return context


def _context_from_store(
    store: SQLiteStore,
    baseline_rankings: list[Any],
    split_set: SplitSet,
    selection_split: str,
    diversity_lambda: float,
    long_term_reflection_path: Path,
    term_scope: str,
    external_feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = _initial_context(
        baseline_rankings,
        split_set,
        selection_split,
        term_scope,
        external_feedback=external_feedback,
    )
    selected_candidates = select_candidates(
        store.scored_candidates(limit=None),
        limit=5,
        diversity_lambda=diversity_lambda,
    )
    context["top_candidates"] = [
        {
            "objective_id": candidate["objective_id"],
            "score": candidate["score"],
            "operator": candidate.get("metrics", {}).get("operator"),
            "signature": candidate.get("metrics", {}).get("signature"),
            "train_validation_split_scores": {
                name: candidate.get("metrics", {}).get("split_scores", {}).get(name)
                for name in ("train", "validation")
            },
            "validation_component_diagnostics": candidate.get("metrics", {})
            .get("component_diagnostics", {})
            .get(selection_split, {}),
            "term_set": candidate.get("term_set", []),
            "complexity": candidate["complexity"],
        }
        for candidate in selected_candidates
    ]
    context["reflections"] = [
        {
            "objective_id": candidate["objective_id"],
            "reflection": candidate.get("metrics", {}).get("reflection", ""),
        }
        for candidate in selected_candidates
        if candidate.get("metrics", {}).get("reflection")
    ]
    context["long_term_reflection"] = _read_text(long_term_reflection_path)
    return context


def _dataset_name(manifest_path: str | Path) -> str:
    path = Path(manifest_path)
    return path.stem.replace(".local", "")


def _baseline_rankings_by_split(split_set: SplitSet) -> dict[str, list[ObjectiveRanking]]:
    return {
        split_name: rank_objectives(split.rows) if split.rows else []
        for split_name, split in split_set.splits.items()
    }


def _evaluate_spec_by_split(
    split_set: SplitSet,
    spec: ObjectiveSpec,
) -> dict[str, ObjectiveRanking]:
    return {
        split_name: evaluate_spec(split.rows, spec) if split.rows else _empty_ranking(spec)
        for split_name, split in split_set.splits.items()
    }


def _empty_ranking(spec: ObjectiveSpec) -> ObjectiveRanking:
    return ObjectiveRanking(
        objective_id=spec.id,
        description=spec.rationale,
        formula=json.dumps(objective_spec_to_dict(spec)["ast"], sort_keys=True),
        score=math.nan,
        correlations=[],
    )


def _write_split_manifest(split_set: SplitSet, output: str | Path) -> Path:
    payload = {
        "strategy": split_set.strategy,
        "group_field": split_set.group_field,
        "seed": split_set.seed,
        "row_counts": split_set.row_counts,
        "group_counts": split_set.group_counts,
        "groups": {name: split.group_keys for name, split in split_set.splits.items()},
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _write_rankings_by_split(
    rankings_by_split: dict[str, list[ObjectiveRanking]],
    split_set: SplitSet,
    output: str | Path,
) -> Path:
    payload = {
        "split_strategy": split_set.strategy,
        "row_counts": split_set.row_counts,
        "rankings_by_split": {
            split_name: [asdict(ranking) for ranking in rankings]
            for split_name, rankings in rankings_by_split.items()
        },
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _split_score(candidate: dict[str, Any], split_name: str) -> float | str:
    split_scores = candidate.get("metrics", {}).get("split_scores", {})
    value = split_scores.get(split_name)
    return value if value is not None else ""


def _safe_reflection(provider: LLMProvider, spec: ObjectiveSpec, metrics: dict[str, Any]) -> str:
    try:
        return provider.reflect(spec, _reflection_metrics(metrics))
    except Exception as exc:
        return f"reflection_failed: {type(exc).__name__}: {exc}"


def _reflection_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    split_scores = metrics.get("split_scores", {})
    split_correlations = metrics.get("split_correlations", {})
    component_diagnostics = metrics.get("component_diagnostics", {})
    return {
        "score": metrics.get("score"),
        "selection_split": metrics.get("selection_split"),
        "operator": metrics.get("operator"),
        "signature": metrics.get("signature"),
        "dreamplace_deployable": metrics.get("dreamplace_deployable"),
        "dreamplace_undeployable_terms": metrics.get("dreamplace_undeployable_terms"),
        "train_validation_split_scores": {
            name: split_scores.get(name) for name in ("train", "validation")
        },
        "train_validation_split_correlations": {
            name: split_correlations.get(name, []) for name in ("train", "validation")
        },
        "train_validation_component_diagnostics": {
            name: component_diagnostics.get(name, {}) for name in ("train", "validation")
        },
        "term_set": metrics.get("term_set"),
        "complexity": metrics.get("complexity"),
        "fit_report": metrics.get("fit_report"),
        "long_term_reflection": metrics.get("long_term_reflection"),
    }


def _write_long_term_reflection(
    store: SQLiteStore,
    output_path: Path,
    selection_split: str,
    diversity_lambda: float,
) -> Path:
    candidates = select_candidates(
        store.scored_candidates(limit=None),
        limit=8,
        diversity_lambda=diversity_lambda,
    )
    lines = [
        "Long-term reflection memory for CoEvoP&R Tier-1 evolution.",
        f"Selection split: {selection_split}. Reporting-only split metrics are omitted.",
    ]
    for index, candidate in enumerate(candidates, start=1):
        metrics = candidate.get("metrics", {})
        split_scores = metrics.get("split_scores", {})
        reflection = str(metrics.get("reflection", "")).strip()
        if len(reflection) > 220:
            reflection = reflection[:217].rstrip() + "..."
        score = _finite_score(candidate.get("score"))
        score_text = f"{score:.6g}" if math.isfinite(score) else "nan"
        lines.append(
            (
                f"{index}. {candidate['objective_id']} score={score_text} "
                f"operator={metrics.get('operator')} "
                f"train={split_scores.get('train')} validation={split_scores.get('validation')} "
                f"deployable={metrics.get('dreamplace_deployable')} "
                f"terms={','.join(candidate.get('term_set', []))} "
                f"reflection={reflection or 'none'}"
            )
        )
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _finite_score(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return -math.inf
    return numeric if math.isfinite(numeric) else -math.inf


def _max_term_jaccard(candidate: dict[str, Any], selected: list[dict[str, Any]]) -> float:
    if not selected:
        return 0.0
    candidate_terms = set(candidate.get("term_set", []))
    if not candidate_terms:
        return 0.0
    overlaps = []
    for other in selected:
        other_terms = set(other.get("term_set", []))
        union = candidate_terms | other_terms
        overlaps.append(len(candidate_terms & other_terms) / len(union) if union else 0.0)
    return max(overlaps) if overlaps else 0.0


def _baseline_comparison_rows(
    baseline_rankings_by_split: dict[str, list[ObjectiveRanking]],
) -> list[dict[str, Any]]:
    by_objective: dict[str, dict[str, ObjectiveRanking]] = {}
    for split_name, rankings in baseline_rankings_by_split.items():
        for ranking in rankings:
            by_objective.setdefault(ranking.objective_id, {})[split_name] = ranking

    rows = []
    validation_rankings = baseline_rankings_by_split.get("validation") or baseline_rankings_by_split.get(
        "train", []
    )
    ordered_ids = [ranking.objective_id for ranking in validation_rankings]
    ordered_ids.extend(sorted(set(by_objective).difference(ordered_ids)))
    for rank, objective_id in enumerate(ordered_ids, start=1):
        split_rankings = by_objective[objective_id]
        primary = (
            split_rankings.get("validation")
            or split_rankings.get("train")
            or next(iter(split_rankings.values()))
        )
        rows.append(
            {
                "kind": "baseline",
                "rank": rank,
                "objective_id": objective_id,
                "selection_score": primary.score,
                "train_score": _ranking_score(split_rankings, "train"),
                "validation_score": _ranking_score(split_rankings, "validation"),
                "heldout_score": _ranking_score(split_rankings, "heldout"),
                "complexity": "",
                "term_set": "",
                "formula": primary.formula,
            }
        )
    return rows


def _ranking_score(split_rankings: dict[str, ObjectiveRanking], split_name: str) -> float | str:
    ranking = split_rankings.get(split_name)
    return ranking.score if ranking is not None else ""
