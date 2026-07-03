"""End-to-end CoEvoP&R orchestration over a shared validation panel."""

from __future__ import annotations

import csv
import json
import shutil
import sqlite3
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
from coevop.eval.tier3_openroad import run_tier3_openroad
from coevop.evolution.memory import parse_operator_weights
from coevop.evolution.offline import EvolutionSummary, run_offline_evolution
from coevop.objectives.presets import objective_preset
from coevop.objectives.spec import load_objective_spec, write_objective_spec
from coevop.objectives.terms import unsupported_terms_for_scope


@dataclass(frozen=True)
class EndToEndConfig:
    manifest: str
    provider: str
    rounds: int
    shared_panel: str
    dreamplace_root: str
    chipbench_root: str
    tier3_enabled: bool
    top_k_tier2: int
    min_terms_tier2: int
    top_k_tier3: int
    severe_regression_pct: float
    prior_feedback: str | None
    tier1: dict[str, Any]
    baseline_presets: list[str]


def load_end_to_end_config(path: str | Path) -> EndToEndConfig:
    config_path = Path(path)
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    tier1 = dict(payload.get("tier1", {}))
    promotion = dict(payload.get("promotion", {}))
    tier2 = dict(payload.get("tier2", {}))
    tier3 = dict(payload.get("tier3", {}))
    feedback = dict(payload.get("feedback", {}))
    return EndToEndConfig(
        manifest=str(_resolve_config_path(config_path, str(payload["manifest"]))),
        provider=str(payload.get("provider", "mock")),
        rounds=int(payload.get("rounds", 1)),
        shared_panel=str(_resolve_config_path(config_path, str(tier2["panel"]))),
        dreamplace_root=str(payload.get("dreamplace_root", "")),
        chipbench_root=str(payload.get("chipbench_root", "")),
        tier3_enabled=bool(tier3.get("enabled", False)),
        top_k_tier2=int(promotion.get("top_k_tier2", 5)),
        min_terms_tier2=int(promotion.get("min_terms_tier2", 1)),
        top_k_tier3=int(tier3.get("top_k", 5)),
        severe_regression_pct=float(feedback.get("severe_regression_pct", 10.0)),
        prior_feedback=(
            str(_resolve_config_path(config_path, str(feedback["prior_feedback"])))
            if feedback.get("prior_feedback")
            else None
        ),
        tier1=tier1,
        baseline_presets=[str(name) for name in tier2.get("baseline_presets", [])],
    )


def run_end_to_end(
    *,
    config_path: str | Path,
    run_dir: str | Path,
    dreamplace_root: str | Path | None = None,
    chipbench_root: str | Path | None = None,
    resume: bool = False,
    retry_failed: bool = False,
) -> dict[str, Any]:
    config = load_end_to_end_config(config_path)
    run_root = Path(run_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    dreamplace_root = Path(dreamplace_root or config.dreamplace_root)
    chipbench_root = Path(chipbench_root or config.chipbench_root)
    _write_json(asdict(config), run_root / "config_resolved.json")

    conn = _open_store(run_root / "e2e.sqlite")
    feedback_context = _load_prior_feedback(config.prior_feedback)
    round_summaries = []
    try:
        for round_index in range(config.rounds):
            round_dir = run_root / f"round_{round_index:03d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            round_summary_path = round_dir / "round_summary.json"
            tier1_summary_path = round_dir / "tier1" / "summary.json"
            if resume and tier1_summary_path.is_file():
                tier1_summary = EvolutionSummary(
                    **json.loads(tier1_summary_path.read_text(encoding="utf-8"))
                )
            else:
                tier1_summary = run_offline_evolution(
                    manifest_path=config.manifest,
                    provider_name=config.provider,
                    run_dir=round_dir / "tier1",
                    population_size=int(config.tier1.get("population_size", 20)),
                    generations=int(config.tier1.get("generations", 1)),
                    elite_count=int(config.tier1.get("elite_count", 5)),
                    max_attempts_per_candidate=int(
                        config.tier1.get("max_attempts_per_candidate", 3)
                    ),
                    seed=int(config.tier1.get("seed", round_index)),
                    max_samples=(
                        int(config.tier1["max_samples"])
                        if config.tier1.get("max_samples") is not None
                        else None
                    ),
                    split_strategy=str(config.tier1.get("split_strategy", "auto")),
                    selection_split=str(config.tier1.get("selection_split", "validation")),
                    reflection_mode=str(config.tier1.get("reflection_mode", "accepted")),
                    diversity_lambda=float(config.tier1.get("diversity_lambda", 0.05)),
                    operator_weights=parse_operator_weights(config.tier1.get("operator_weights")),
                    islands=int(config.tier1.get("islands", 5)),
                    enable_constant_fit=bool(config.tier1.get("enable_constant_fit", True)),
                    term_scope="deployable",
                    validation_design=config.tier1.get("validation_design"),
                    heldout_design=config.tier1.get("heldout_design"),
                    external_feedback=feedback_context,
                )
            promoted = promote_candidates(
                tier1_summary.top_candidates,
                output_dir=round_dir / "promoted_objectives",
                top_k=config.top_k_tier2,
                min_terms=config.min_terms_tier2,
            )
            baseline_paths = _write_baseline_presets(
                config.baseline_presets,
                round_dir / "promoted_objectives" / "baselines",
            )
            shared_panel = load_shared_panel(config.shared_panel)
            dreamplace_panel_path = write_dreamplace_panel(
                shared_panel,
                round_dir / "tier2_dreamplace" / "shared_dreamplace_panel.toml",
            )
            tier2_summary = run_tier2_dreamplace(
                panel_path=dreamplace_panel_path,
                objective_paths=baseline_paths + [item["artifact_path"] for item in promoted],
                dreamplace_root=dreamplace_root,
                run_dir=round_dir / "tier2_dreamplace",
                store_path=round_dir / "tier2_dreamplace" / "tier2.sqlite",
                resume=resume,
                retry_failed=retry_failed,
                include_default=True,
                include_custom_default=True,
                require_output_artifact=True,
                require_def_output=True,
                disable_legalization=False,
            )
            tier2_rows = _read_csv(Path(tier2_summary["comparison_csv"]))
            tier2_labeled = add_outcome_labels(
                tier2_rows,
                severe_threshold_pct=config.severe_regression_pct,
            )
            tier2_rankings = aggregate_rank_scores(
                tier2_rows,
                severe_threshold_pct=config.severe_regression_pct,
            )
            _write_json(
                {"rankings": [ranking.to_dict() for ranking in tier2_rankings]},
                round_dir / "tier2_dreamplace" / "rankings.json",
            )
            placements_path = write_placements_manifest(
                tier2_labeled,
                round_dir / "tier2_dreamplace" / "placements.json",
            )
            tier3_summary = None
            tier3_rows: list[dict[str, Any]] = []
            tier3_rankings = []
            if config.tier3_enabled and _placement_count(placements_path) > 0:
                selected_placements_path = write_selected_tier3_placements(
                    tier2_labeled,
                    tier2_rankings=[ranking.to_dict() for ranking in tier2_rankings],
                    output=round_dir / "tier2_dreamplace" / "placements_tier3_selected.json",
                    top_k=config.top_k_tier3,
                )
                tier3_summary = run_tier3_openroad(
                    panel_path=config.shared_panel,
                    placements_path=selected_placements_path,
                    chipbench_root=chipbench_root,
                    run_dir=round_dir / "tier3_openroad",
                    store_path=round_dir / "tier3_openroad" / "tier3.sqlite",
                    resume=resume,
                    retry_failed=retry_failed,
                )
                tier3_rows = add_outcome_labels(
                    _read_csv(Path(tier3_summary["comparison_csv"])),
                    severe_threshold_pct=config.severe_regression_pct,
                )
                tier3_rankings = aggregate_rank_scores(
                    tier3_rows,
                    metric_names=(
                        "routed_wirelength_delta_pct",
                        "grt_overflow_delta_pct",
                        "drc_count_delta_pct",
                    ),
                    severe_threshold_pct=config.severe_regression_pct,
                )

            feedback = build_feedback(
                tier1_summary=asdict(tier1_summary),
                promoted=promoted,
                tier2_rows=tier2_labeled,
                tier2_rankings=[ranking.to_dict() for ranking in tier2_rankings],
                tier3_rows=tier3_rows,
                tier3_rankings=[ranking.to_dict() for ranking in tier3_rankings],
            )
            feedback_path = _write_json(feedback, round_dir / "feedback.json")
            feedback_context = feedback
            round_summary = {
                "round": round_index,
                "tier1": asdict(tier1_summary),
                "promoted_count": len(promoted),
                "tier2": tier2_summary,
                "tier3": tier3_summary,
                "feedback": str(feedback_path),
            }
            _write_json(round_summary, round_summary_path)
            _record_round(conn, round_summary)
            round_summaries.append(round_summary)
    finally:
        conn.close()

    summary = {
        "run_dir": str(run_root),
        "config": str(config_path),
        "round_count": len(round_summaries),
        "rounds": round_summaries,
    }
    _write_json(summary, run_root / "summary.json")
    (run_root / "e2e_report.md").write_text(_build_report(summary), encoding="utf-8")
    return summary


def promote_candidates(
    top_candidates: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    top_k: int,
    min_terms: int = 1,
) -> list[dict[str, Any]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    promoted = []
    seen_ids: set[str] = set()
    seen_asts: set[str] = set()
    for candidate in top_candidates:
        artifact_path = candidate.get("artifact_path")
        if not artifact_path:
            continue
        spec = load_objective_spec(artifact_path)
        if unsupported_terms_for_scope(spec.term_set, "dreamplace"):
            continue
        if len(spec.term_set) < min_terms:
            continue
        ast_signature = json.dumps(spec.ast, sort_keys=True)
        if spec.id in seen_ids or ast_signature in seen_asts:
            continue
        copied = output / f"{_safe_name(spec.id)}.json"
        shutil.copyfile(artifact_path, copied)
        seen_ids.add(spec.id)
        seen_asts.add(ast_signature)
        promoted.append(
            {
                "objective_id": spec.id,
                "artifact_path": str(copied),
                "source_artifact_path": str(artifact_path),
                "tier1_score": candidate.get("score"),
                "term_set": spec.term_set,
                "complexity": spec.complexity,
                "dreamplace_deployable": True,
            }
        )
        if len(promoted) >= top_k:
            break
    _write_json({"promoted": promoted}, output / "manifest.json")
    return promoted


def _load_prior_feedback(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    feedback_path = Path(path)
    if not feedback_path.is_file():
        return {
            "warning": "configured prior feedback file was not found",
            "prior_feedback_path": str(feedback_path),
        }
    return json.loads(feedback_path.read_text(encoding="utf-8"))


def write_placements_manifest(rows: list[dict[str, Any]], output: str | Path) -> Path:
    placements = []
    for row in rows:
        artifact = str(row.get("output_artifact") or "")
        if row.get("status") != "success" or not artifact.lower().endswith(".def"):
            continue
        placements.append(
            {
                "design": row["design"],
                "objective_id": row["objective_id"],
                "seed": int(row["seed"]),
                "def_path": artifact,
                "source": "tier2_dreamplace",
            }
        )
    return _write_json({"placements": placements}, output)


def write_selected_tier3_placements(
    rows: list[dict[str, Any]],
    *,
    tier2_rankings: list[dict[str, Any]],
    output: str | Path,
    top_k: int,
) -> Path:
    selected_ids = {"default"}
    for ranking in tier2_rankings:
        objective_id = str(ranking.get("objective_id") or "")
        if not objective_id or objective_id in selected_ids or objective_id == "custom_default":
            continue
        selected_ids.add(objective_id)
        if len(selected_ids) >= top_k + 1:
            break
    selected_rows = [row for row in rows if str(row.get("objective_id")) in selected_ids]
    return write_placements_manifest(selected_rows, output)


def build_feedback(
    *,
    tier1_summary: dict[str, Any],
    promoted: list[dict[str, Any]],
    tier2_rows: list[dict[str, Any]],
    tier2_rankings: list[dict[str, Any]],
    tier3_rows: list[dict[str, Any]],
    tier3_rankings: list[dict[str, Any]],
) -> dict[str, Any]:
    tier1_candidates = []
    for candidate in tier1_summary.get("top_candidates", [])[:10]:
        metrics = candidate.get("metrics", {})
        split_scores = metrics.get("split_scores", {})
        tier1_candidates.append(
            {
                "objective_id": candidate.get("objective_id"),
                "validation_score": _split_score_value(split_scores, "validation"),
                "train_score": _split_score_value(split_scores, "train"),
                "score": candidate.get("score"),
                "term_set": candidate.get("term_set"),
                "complexity": candidate.get("complexity"),
                "operator": metrics.get("operator"),
                "signature": metrics.get("signature"),
            }
        )
    return {
        "tier1_validation_summary": {
            "provider": tier1_summary.get("provider"),
            "accepted_candidates": tier1_summary.get("accepted_candidates"),
            "rejected_candidates": tier1_summary.get("rejected_candidates"),
            "selection_split": tier1_summary.get("selection_split"),
            "top_candidates": tier1_candidates,
        },
        "promoted_objectives": promoted,
        "tier2_rank_summary": tier2_rankings[:10],
        "tier3_rank_summary": tier3_rankings[:10],
        "best_candidates": _best_candidates(tier2_rankings, tier3_rankings),
        "regressed_candidates": _rows_by_label(tier2_rows + tier3_rows, "metric_regression"),
        "severe_regressions": _rows_by_label(tier2_rows + tier3_rows, "severe_regression"),
        "structural_failures": _rows_by_label(tier2_rows + tier3_rows, "structural_failure"),
        "feedback_policy": (
            "Held-out final panel metrics are excluded. Structural failures are "
            "negative feasibility feedback; completed regressions remain useful "
            "behavioral feedback."
        ),
    }


def _split_score_value(split_scores: dict[str, Any], split_name: str) -> Any:
    value = split_scores.get(split_name)
    if isinstance(value, dict):
        return value.get("score")
    return value


def _write_baseline_presets(names: list[str], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        spec = objective_preset(name)
        path = write_objective_spec(spec, output_dir / f"{_safe_name(name)}.json")
        paths.append(str(path))
    return paths


def _best_candidates(
    tier2_rankings: list[dict[str, Any]],
    tier3_rankings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source = tier3_rankings if tier3_rankings else tier2_rankings
    return [
        {
            "objective_id": item.get("objective_id"),
            "average_rank": item.get("average_rank"),
            "source": "tier3" if tier3_rankings else "tier2",
        }
        for item in source[:5]
        if item.get("objective_id") not in {"default", "custom_default"}
    ]


def _rows_by_label(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    compact = []
    for row in rows:
        if row.get("outcome_label") != label:
            continue
        compact.append(
            {
                "design": row.get("design"),
                "objective_id": row.get("objective_id"),
                "seed": row.get("seed"),
                "failure_stage": row.get("failure_stage"),
                "metrics_stage": row.get("metrics_stage"),
                "hpwl_delta_pct": row.get("hpwl_delta_pct"),
                "overflow_delta_pct": row.get("overflow_delta_pct"),
                "estimated_wirelength_delta_pct": row.get("estimated_wirelength_delta_pct"),
                "routed_wirelength_delta_pct": row.get("routed_wirelength_delta_pct"),
                "grt_overflow_delta_pct": row.get("grt_overflow_delta_pct"),
                "drc_count_delta_pct": row.get("drc_count_delta_pct"),
            }
        )
    return compact[:20]


def _placement_count(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return len(payload.get("placements", []))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _open_store(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        create table if not exists e2e_rounds (
            round integer primary key,
            summary_json text not null,
            updated_at text not null default current_timestamp
        )
        """
    )
    return conn


def _record_round(conn: sqlite3.Connection, summary: dict[str, Any]) -> None:
    conn.execute(
        """
        insert into e2e_rounds (round, summary_json)
        values (?, ?)
        on conflict(round) do update set
            summary_json=excluded.summary_json,
            updated_at=current_timestamp
        """,
        (summary["round"], json.dumps(summary, sort_keys=True)),
    )
    conn.commit()


def _build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# End-To-End CoEvoP&R Report",
        "",
        f"- Run directory: {summary['run_dir']}",
        f"- Rounds completed: {summary['round_count']}",
        "",
        "## Rounds",
    ]
    for round_summary in summary["rounds"]:
        tier2 = round_summary.get("tier2") or {}
        tier3 = round_summary.get("tier3") or {}
        lines.append(
            "- "
            f"round_{int(round_summary['round']):03d}: "
            f"promoted={round_summary.get('promoted_count')}, "
            f"tier2_success={tier2.get('success_count')}, "
            f"tier3_success={tier3.get('success_count') if tier3 else 'disabled'}"
        )
    return "\n".join(lines).rstrip() + "\n"


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
