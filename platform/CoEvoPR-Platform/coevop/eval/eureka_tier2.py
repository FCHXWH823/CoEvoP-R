"""Eureka-style Tier-2 DREAMPlace objective discovery."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised by py310 via tomli
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from coevop.eval.ranking import add_outcome_labels, aggregate_rank_scores
from coevop.eval.shared_panel import load_shared_panel, write_dreamplace_panel
from coevop.eval.tier2_dreamplace import run_tier2_dreamplace
from coevop.llm.providers import LLMProvider, provider_from_name
from coevop.objectives.presets import objective_preset
from coevop.objectives.program import PROGRAM_SYSTEM_CONTRACT
from coevop.objectives.spec import ObjectiveSpec, load_objective_spec, write_objective_spec
from coevop.objectives.terms import term_descriptions, term_names, unsupported_terms_for_scope


@dataclass(frozen=True)
class EurekaTier2Config:
    provider: str
    eureka_iterations: int
    samples_per_iteration: int
    candidates_per_iteration: int
    term_scope: str
    severe_regression_pct: float
    temperature: float
    search_panel: str
    search_baseline_presets: list[str]
    final_enabled: bool
    final_panel: str | None
    final_top_k: int
    final_baseline_presets: list[str]
    prior_feedback: str | None


@dataclass(frozen=True)
class CandidateArtifact:
    objective_id: str | None
    iteration: int
    sample_index: int
    status: str
    artifact_path: str | None
    prompt_messages_path: str
    prompt_context_path: str
    raw_response_path: str | None
    failure_reason: str | None
    term_set: list[str]
    complexity: int | None
    provider_metadata: dict[str, Any]


def load_eureka_tier2_config(path: str | Path) -> EurekaTier2Config:
    config_path = Path(path)
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    search = dict(payload.get("search", {}))
    final = dict(payload.get("final", {}))
    feedback = dict(payload.get("feedback", {}))
    return EurekaTier2Config(
        provider=str(payload.get("provider", "mock")),
        eureka_iterations=int(payload.get("eureka_iterations", 5)),
        samples_per_iteration=int(payload.get("samples_per_iteration", 16)),
        candidates_per_iteration=int(payload.get("candidates_per_iteration", 4)),
        term_scope=str(payload.get("term_scope", "dreamplace_replacement")),
        severe_regression_pct=float(payload.get("severe_regression_pct", 10.0)),
        temperature=float(payload.get("temperature", 1.0)),
        search_panel=str(_resolve_config_path(config_path, str(search["panel"]))),
        search_baseline_presets=[str(name) for name in search.get("baseline_presets", [])],
        final_enabled=bool(final.get("enabled", False)),
        final_panel=(
            str(_resolve_config_path(config_path, str(final["panel"])))
            if final.get("panel")
            else None
        ),
        final_top_k=int(final.get("top_k", 8)),
        final_baseline_presets=[str(name) for name in final.get("baseline_presets", [])],
        prior_feedback=(
            str(_resolve_config_path(config_path, str(feedback["prior_feedback"])))
            if feedback.get("prior_feedback")
            else None
        ),
    )


def run_eureka_tier2(
    *,
    config_path: str | Path,
    run_dir: str | Path,
    dreamplace_root: str | Path,
    resume: bool = False,
    retry_failed: bool = False,
) -> dict[str, Any]:
    config = load_eureka_tier2_config(config_path)
    if config.term_scope != "dreamplace_replacement":
        raise ValueError("eureka-tier2 currently requires term_scope='dreamplace_replacement'")
    run_root = Path(run_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(asdict(config), run_root / "config_resolved.json")
    provider = provider_from_name(config.provider)
    provider_metadata = provider.metadata()
    prompt_pack = build_prompt_context(
        config=config,
        previous_feedback=_load_json(config.prior_feedback),
        iteration_index=0,
        iteration_feedback=None,
    )
    (run_root / "prompt_pack.md").write_text(
        _prompt_context_markdown(prompt_pack),
        encoding="utf-8",
    )

    iteration_summaries = []
    all_candidates: list[CandidateArtifact] = []
    all_search_rows: list[dict[str, Any]] = []
    iteration_feedback: dict[str, Any] | None = _load_json(config.prior_feedback)

    for iteration_index in range(config.eureka_iterations):
        iteration_dir = run_root / f"iteration_{iteration_index:03d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        context = build_prompt_context(
            config=config,
            previous_feedback=_load_json(config.prior_feedback),
            iteration_index=iteration_index,
            iteration_feedback=iteration_feedback,
        )
        candidates = generate_iteration_candidates(
            provider=provider,
            config=config,
            context=context,
            iteration_dir=iteration_dir,
            iteration_index=iteration_index,
        )
        all_candidates.extend(candidates)
        selected = select_static_candidates(candidates, top_k=config.candidates_per_iteration)
        selected_paths = copy_selected_objectives(selected, iteration_dir / "selected_objectives")
        baseline_paths = write_baseline_presets(
            config.search_baseline_presets,
            iteration_dir / "baselines",
        )
        search_panel_path = write_dreamplace_panel(
            load_shared_panel(config.search_panel),
            iteration_dir / "search_panel.toml",
        )
        tier2_summary = run_tier2_dreamplace(
            panel_path=search_panel_path,
            objective_paths=baseline_paths + selected_paths,
            dreamplace_root=dreamplace_root,
            run_dir=iteration_dir / "tier2_search",
            store_path=iteration_dir / "tier2_search" / "tier2.sqlite",
            resume=resume,
            retry_failed=retry_failed,
            include_default=True,
            include_custom_default=True,
            require_output_artifact=True,
            require_def_output=True,
            disable_legalization=False,
        )
        tier2_rows = add_outcome_labels(
            _read_csv(Path(tier2_summary["comparison_csv"])),
            severe_threshold_pct=config.severe_regression_pct,
        )
        all_search_rows.extend(tier2_rows)
        rankings = [ranking.to_dict() for ranking in aggregate_rank_scores(
            tier2_rows,
            severe_threshold_pct=config.severe_regression_pct,
        )]
        generated_ids = {candidate.objective_id for candidate in selected if candidate.objective_id}
        iteration_feedback = build_iteration_feedback(
            candidates=candidates,
            selected=selected,
            tier2_rows=tier2_rows,
            rankings=rankings,
            generated_ids=generated_ids,
            provider=provider,
        )
        _write_json(iteration_feedback, iteration_dir / "feedback.json")
        iteration_summary = {
            "iteration": iteration_index,
            "candidate_count": len(candidates),
            "valid_candidate_count": sum(1 for candidate in candidates if candidate.status == "valid"),
            "selected_objective_ids": [candidate.objective_id for candidate in selected],
            "tier2": tier2_summary,
            "rankings": rankings,
            "feedback": str(iteration_dir / "feedback.json"),
        }
        _write_json(iteration_summary, iteration_dir / "iteration_summary.json")
        iteration_summaries.append(iteration_summary)

    final_summary = None
    if config.final_enabled and config.final_panel:
        final_candidates = select_final_candidates(
            candidates=all_candidates,
            search_rows=all_search_rows,
            top_k=config.final_top_k,
            severe_regression_pct=config.severe_regression_pct,
        )
        final_objective_paths = copy_selected_objectives(
            final_candidates,
            run_root / "final_candidates",
        )
        final_baseline_paths = write_baseline_presets(
            config.final_baseline_presets,
            run_root / "final_baselines",
        )
        final_panel_path = write_dreamplace_panel(
            load_shared_panel(config.final_panel),
            run_root / "final_tier2" / "final_panel.toml",
        )
        final_summary = run_tier2_dreamplace(
            panel_path=final_panel_path,
            objective_paths=final_baseline_paths + final_objective_paths,
            dreamplace_root=dreamplace_root,
            run_dir=run_root / "final_tier2",
            store_path=run_root / "final_tier2" / "tier2.sqlite",
            resume=resume,
            retry_failed=retry_failed,
            include_default=True,
            include_custom_default=True,
            require_output_artifact=True,
            require_def_output=True,
            disable_legalization=False,
        )

    summary = {
        "run_dir": str(run_root),
        "config": str(config_path),
        "provider_metadata": provider_metadata,
        "iterations": iteration_summaries,
        "candidate_attempt_count": len(all_candidates),
        "valid_candidate_count": sum(1 for candidate in all_candidates if candidate.status == "valid"),
        "search_success_count": sum(1 for row in all_search_rows if row.get("status") == "success"),
        "final_tier2": final_summary,
    }
    _write_json(summary, run_root / "summary.json")
    (run_root / "eureka_tier2_report.md").write_text(
        build_report(summary, all_search_rows),
        encoding="utf-8",
    )
    return summary


def build_prompt_context(
    *,
    config: EurekaTier2Config,
    previous_feedback: dict[str, Any] | None,
    iteration_index: int,
    iteration_feedback: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "research_goal": (
            "CoEvoP&R targets DATE 2027 and searches for symbolic, interpretable, "
            "differentiable placement objectives that improve routing-aware placement."
        ),
        "method_background": {
            "eureka": (
                "Use code-writing LLMs, environment code context, many samples per "
                "iteration, execution feedback, and in-context improvement."
            ),
            "llm_sr_reevo_funsearch": (
                "Use structured candidate memory, mutation/reflection, component "
                "diagnostics, and explicit feedback from evaluated programs."
            ),
        },
        "task": (
            "Generate one complete replacement objective for DREAMPlace. The objective "
            "may differ from wirelength+density and will be judged by measured "
            "DREAMPlace placement behavior after execution."
        ),
        "term_scope": config.term_scope,
        "generation_policy": {
            "samples_per_iteration": config.samples_per_iteration,
            "eureka_iterations": config.eureka_iterations,
            "candidates_per_iteration": config.candidates_per_iteration,
            "temperature": config.temperature,
            "api_memory": "disabled; all memory is explicit in saved prompt context",
        },
        "allowed_terms": term_names(config.term_scope),
        "term_descriptions": term_descriptions(config.term_scope),
        "objective_program_contract": PROGRAM_SYSTEM_CONTRACT,
        "dreamplace_environment_code_excerpts": {
            "custom_objective_path": (
                "DREAMPlace PlaceObj.obj_fn loads custom_objective_spec, evaluates "
                "the ObjectiveSpec AST, normalizes each term by its first observed "
                "scale, rescales by output scale, and fails on non-finite objective "
                "or gradient."
            ),
            "deployable_terms": (
                "wirelength, wirelength_wawl, wirelength_lse, density, "
                "density_electric, density_bell, soft_rudy_mean, "
                "soft_rudy_pnorm, route_pressure_long, pin_density_pnorm, "
                "pin_count_weighted_wl"
            ),
            "feedback_metrics": (
                "DREAMPlace Tier-2 reports HPWL, overflow, custom objective call "
                "count, custom total, gradient norm, runtime, and output DEF path."
            ),
        },
        "previous_failed_run_summary": {
            "lesson": (
                "A prior pin_density_pnorm-only candidate reduced overflow by about "
                "47% but increased HPWL by about 300% and badly degraded partial "
                "timing metrics. This is negative feedback showing that a pure "
                "proxy-pressure mechanism can spread placement too aggressively."
            ),
            "previous_feedback": previous_feedback,
        },
        "iteration_index": iteration_index,
        "iteration_feedback": iteration_feedback,
        "selection_policy": {
            "primary": (
                "DREAMPlace Tier-2 execution feedback over wirelength-related "
                "placement quality, congestion/overflow behavior, runtime, and "
                "failure status"
            ),
            "structural_failures": "worst rank",
            "runtime": "tie breaker",
            "threshold_policy": "exact thresholds are evaluator-side and are not prompt targets",
        },
    }


def generate_iteration_candidates(
    *,
    provider: LLMProvider,
    config: EurekaTier2Config,
    context: dict[str, Any],
    iteration_dir: Path,
    iteration_index: int,
) -> list[CandidateArtifact]:
    candidates = []
    for sample_index in range(config.samples_per_iteration):
        sample_dir = iteration_dir / "samples" / f"sample_{sample_index:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_context = dict(context)
        sample_context["sample_index"] = sample_index
        prompt_context_path = sample_dir / "prompt_context.md"
        prompt_context_path.write_text(_prompt_context_markdown(sample_context), encoding="utf-8")
        prompt_messages_path = sample_dir / "prompt_messages.json"
        raw_response_path = sample_dir / "raw_response.json"
        try:
            trace = provider.generate_traced(
                context=sample_context,
                term_scope=config.term_scope,
            )
            _write_json(trace.messages, prompt_messages_path)
            _write_json(trace.raw_response or {}, raw_response_path)
            spec = trace.spec
            failure = static_rejection_reason(spec)
            if failure:
                candidate = CandidateArtifact(
                    objective_id=spec.id,
                    iteration=iteration_index,
                    sample_index=sample_index,
                    status="rejected",
                    artifact_path=None,
                    prompt_messages_path=str(prompt_messages_path),
                    prompt_context_path=str(prompt_context_path),
                    raw_response_path=str(raw_response_path),
                    failure_reason=failure,
                    term_set=spec.term_set,
                    complexity=spec.complexity,
                    provider_metadata=trace.metadata,
                )
            else:
                objective_path = write_objective_spec(spec, sample_dir / "objective.json")
                candidate = CandidateArtifact(
                    objective_id=spec.id,
                    iteration=iteration_index,
                    sample_index=sample_index,
                    status="valid",
                    artifact_path=str(objective_path),
                    prompt_messages_path=str(prompt_messages_path),
                    prompt_context_path=str(prompt_context_path),
                    raw_response_path=str(raw_response_path),
                    failure_reason=None,
                    term_set=spec.term_set,
                    complexity=spec.complexity,
                    provider_metadata=trace.metadata,
                )
        except Exception as exc:
            _write_json({"error": f"{type(exc).__name__}: {exc}"}, raw_response_path)
            candidate = CandidateArtifact(
                objective_id=None,
                iteration=iteration_index,
                sample_index=sample_index,
                status="provider_error",
                artifact_path=None,
                prompt_messages_path=str(prompt_messages_path),
                prompt_context_path=str(prompt_context_path),
                raw_response_path=str(raw_response_path),
                failure_reason=f"{type(exc).__name__}: {exc}",
                term_set=[],
                complexity=None,
                provider_metadata=provider.metadata(),
            )
        _write_json(asdict(candidate), sample_dir / "candidate_record.json")
        candidates.append(candidate)
    _write_json([asdict(candidate) for candidate in candidates], iteration_dir / "candidates.json")
    return candidates


def static_rejection_reason(spec: ObjectiveSpec) -> str | None:
    unsupported = unsupported_terms_for_scope(spec.term_set, "dreamplace_replacement")
    if unsupported:
        return f"unsupported dreamplace_replacement terms: {unsupported}"
    if len(spec.term_set) < 2:
        return "replacement objective must use at least two term families"
    return None


def select_static_candidates(
    candidates: list[CandidateArtifact],
    *,
    top_k: int,
) -> list[CandidateArtifact]:
    valid = [candidate for candidate in candidates if candidate.status == "valid" and candidate.artifact_path]
    valid.sort(key=lambda candidate: (-(len(set(candidate.term_set))), candidate.complexity or 9999))
    selected: list[CandidateArtifact] = []
    seen_ids: set[str] = set()
    seen_asts: set[str] = set()
    for candidate in valid:
        if not candidate.objective_id or candidate.objective_id in seen_ids:
            continue
        ast_signature = json.dumps(load_objective_spec(candidate.artifact_path).ast, sort_keys=True)
        if ast_signature in seen_asts:
            continue
        selected.append(candidate)
        seen_ids.add(candidate.objective_id)
        seen_asts.add(ast_signature)
        if len(selected) >= top_k:
            break
    return selected


def copy_selected_objectives(candidates: list[CandidateArtifact], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for candidate in candidates:
        if not candidate.artifact_path or not candidate.objective_id:
            continue
        output = output_dir / f"{candidate.objective_id}.json"
        shutil.copyfile(candidate.artifact_path, output)
        paths.append(str(output))
    _write_json([asdict(candidate) for candidate in candidates], output_dir / "manifest.json")
    return paths


def write_baseline_presets(names: list[str], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        path = write_objective_spec(objective_preset(name), output_dir / f"{_safe_name(name)}.json")
        paths.append(str(path))
    return paths


def build_iteration_feedback(
    *,
    candidates: list[CandidateArtifact],
    selected: list[CandidateArtifact],
    tier2_rows: list[dict[str, Any]],
    rankings: list[dict[str, Any]],
    generated_ids: set[str | None],
    provider: LLMProvider,
) -> dict[str, Any]:
    best_generated = next(
        (
            ranking
            for ranking in rankings
            if ranking.get("objective_id") in generated_ids
        ),
        None,
    )
    reflection = ""
    if best_generated:
        best_id = str(best_generated["objective_id"])
        best_candidate = next((candidate for candidate in selected if candidate.objective_id == best_id), None)
        if best_candidate and best_candidate.artifact_path:
            try:
                reflection = provider.reflect(
                    load_objective_spec(best_candidate.artifact_path),
                    {"tier2_rank": best_generated, "tier2_rows": tier2_rows},
                )
            except Exception as exc:
                reflection = f"reflection_failed: {type(exc).__name__}: {exc}"
    return {
        "candidate_attempt_count": len(candidates),
        "valid_candidate_count": sum(1 for candidate in candidates if candidate.status == "valid"),
        "selected_objectives": [asdict(candidate) for candidate in selected],
        "rankings": rankings,
        "best_generated": best_generated,
        "structural_failures": [
            row for row in tier2_rows if row.get("outcome_label") == "structural_failure"
        ][:20],
        "severe_regressions": [
            row for row in tier2_rows if row.get("outcome_label") == "severe_regression"
        ][:20],
        "reflection": reflection,
    }


def select_final_candidates(
    *,
    candidates: list[CandidateArtifact],
    search_rows: list[dict[str, Any]],
    top_k: int,
    severe_regression_pct: float,
) -> list[CandidateArtifact]:
    generated_by_id = {
        candidate.objective_id: candidate
        for candidate in candidates
        if candidate.status == "valid" and candidate.objective_id and candidate.artifact_path
    }
    rankings = aggregate_rank_scores(search_rows, severe_threshold_pct=severe_regression_pct)
    selected = []
    for ranking in rankings:
        candidate = generated_by_id.get(ranking.objective_id)
        if candidate is None:
            continue
        if ranking.structural_failure_count or ranking.severe_regression_count:
            continue
        selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected


def build_report(summary: dict[str, Any], search_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Eureka Tier-2 DREAMPlace Report",
        "",
        f"- Run directory: {summary['run_dir']}",
        f"- Candidate attempts: {summary['candidate_attempt_count']}",
        f"- Valid candidates: {summary['valid_candidate_count']}",
        f"- Search successful DREAMPlace rows: {summary['search_success_count']}",
        f"- Final Tier-2 enabled: {summary.get('final_tier2') is not None}",
        "",
        "## Iterations",
    ]
    for item in summary["iterations"]:
        lines.append(
            "- "
            f"iteration_{int(item['iteration']):03d}: "
            f"valid={item['valid_candidate_count']}/{item['candidate_count']}, "
            f"selected={item['selected_objective_ids']}, "
            f"tier2_success={item['tier2'].get('success_count')}"
        )
    if search_rows:
        lines.extend(["", "## Search Outcome Labels"])
        counts: dict[str, int] = {}
        for row in search_rows:
            label = str(row.get("outcome_label") or "unknown")
            counts[label] = counts.get(label, 0) + 1
        for label, count in sorted(counts.items()):
            lines.append(f"- {label}: {count}")
    return "\n".join(lines).rstrip() + "\n"


def _prompt_context_markdown(context: dict[str, Any]) -> str:
    return (
        "# Eureka-Style CoEvoP&R Prompt Context\n\n"
        "```json\n"
        f"{json.dumps(context, indent=2, sort_keys=True)}\n"
        "```\n"
    )


def _load_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return {"warning": "feedback file not found", "path": str(candidate)}
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"payload": payload}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _write_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
