"""Tier-2 DREAMPlace placement evaluation runner."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sqlite3
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised by py310 via tomli
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from coevop.backends.dreamplace import (
    make_run_config,
    parse_dreamplace_log,
    run_dreamplace,
    write_run_summary,
)
from coevop.eval.macro_preflight import (
    audit_output_macro_coordinates,
    load_hard_macro_components,
)
from coevop.objectives.presets import objective_preset
from coevop.objectives.spec import ObjectiveSpec, load_objective_spec, write_objective_spec
from coevop.objectives.terms import unsupported_terms_for_scope


DEFAULT_OBJECTIVE_ID = "default"
CUSTOM_DEFAULT_OBJECTIVE_ID = "custom_default"
CUSTOM_DEFAULT_PRESET = "dreamplace_native_default"
TIMING_MODES = {"none", "compatibility_only", "claim"}
MINIMUM_SCORING_ITERATIONS = 1000
TIER2_TABLE_COLUMNS = [
    "design",
    "objective_id",
    "seed",
    "status",
    "failure_stage",
    "hpwl",
    "overflow",
    "wns",
    "tns",
    "requested_iterations",
    "completed_iterations",
    "iteration_budget_satisfied",
    "custom_objective_calls",
    "custom_total",
    "custom_grad_norm",
    "component_summary_json",
    "timing_net_coverage",
    "timing_policy_update_count",
    "timing_policy_summary_json",
    "runtime_seconds",
    "output_artifact",
    "coevop_commit",
    "dreamplace_commit",
]


@dataclass(frozen=True)
class DreamPlaceDesign:
    name: str
    base_config: str
    timing_mode: str


@dataclass(frozen=True)
class DreamPlacePanel:
    seeds: list[int]
    iterations: int
    gpu: int | None
    timeout_seconds: int
    log_interval: int
    timing_mode: str
    designs: list[DreamPlaceDesign]
    stop_overflow: float | None = None
    driver: str | None = None


@dataclass(frozen=True)
class Tier2Objective:
    objective_id: str
    objective_path: str | None
    source: str
    spec_id: str | None
    term_set: list[str]
    # DSL v2: "raw" runs the custom objective without hidden per-term value
    # normalization (controller mode); None keeps the legacy normalized path.
    objective_semantics: str | None = None


@dataclass(frozen=True)
class Tier2Result:
    design: str
    objective_id: str
    seed: int
    status: str
    failure_stage: str | None
    hpwl: float | None
    overflow: float | None
    wns: float | None
    tns: float | None
    requested_iterations: int
    completed_iterations: int
    iteration_budget_satisfied: bool
    custom_objective_calls: int
    custom_total: float | None
    custom_grad_norm: float | None
    component_summary_json: str
    runtime_seconds: float | None
    output_artifact: str | None
    coevop_commit: str | None
    dreamplace_commit: str | None
    run_dir: str
    config_path: str | None
    log_path: str | None
    returncode: int | None
    metrics: dict[str, Any]
    timing_net_coverage: float | None = None
    timing_policy_update_count: int = 0
    timing_policy_summary_json: str = "{}"


def load_panel(path: str | Path) -> DreamPlacePanel:
    panel_path = Path(path)
    payload = tomllib.loads(panel_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in payload.get("seeds", [1000])]
    if not seeds:
        raise ValueError("DREAMPlace panel must contain at least one seed")
    timing_mode = str(payload.get("timing_mode", "compatibility_only"))
    _validate_timing_mode(timing_mode)
    designs = []
    for item in payload.get("designs", []):
        name = str(item["name"])
        base_config_value = item.get("base_config", item.get("dreamplace_config"))
        if base_config_value is None:
            raise ValueError(
                f"design {name} must define base_config or dreamplace_config"
            )
        base_config = _resolve_panel_path(panel_path, str(base_config_value))
        design_timing_mode = str(item.get("timing_mode", timing_mode))
        _validate_timing_mode(design_timing_mode)
        designs.append(
            DreamPlaceDesign(
                name=name,
                base_config=str(base_config),
                timing_mode=design_timing_mode,
            )
        )
    if not designs:
        raise ValueError("DREAMPlace panel must contain at least one [[designs]] entry")
    return DreamPlacePanel(
        seeds=seeds,
        iterations=int(payload.get("iterations", 1000)),
        gpu=int(payload["gpu"]) if payload.get("gpu") is not None else None,
        timeout_seconds=int(payload.get("timeout_seconds", 900)),
        log_interval=int(payload.get("log_interval", 10)),
        timing_mode=timing_mode,
        designs=designs,
        stop_overflow=(
            float(payload["stop_overflow"])
            if payload.get("stop_overflow") is not None
            else None
        ),
        driver=str(payload["driver"]) if payload.get("driver") else None,
    )


def run_tier2_dreamplace(
    *,
    panel_path: str | Path,
    objective_paths: list[str | Path],
    dreamplace_root: str | Path,
    run_dir: str | Path,
    store_path: str | Path | None = None,
    resume: bool = False,
    retry_failed: bool = False,
    include_default: bool = True,
    include_custom_default: bool = True,
    require_output_artifact: bool = False,
    require_def_output: bool = False,
    disable_legalization: bool = True,
    objective_semantics: str | None = None,
) -> dict[str, Any]:
    run_root = Path(run_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    store_path = Path(store_path) if store_path is not None else run_root / "tier2.sqlite"
    panel = load_panel(panel_path)
    objectives = prepare_objectives(
        objective_paths,
        run_root=run_root,
        include_default=include_default,
        include_custom_default=include_custom_default,
        objective_semantics=objective_semantics,
    )
    tool_versions = {
        "coevop": _git_metadata(Path.cwd()),
        "dreamplace": _git_metadata(Path(dreamplace_root)),
    }
    _write_json(asdict(panel), run_root / "panel_manifest.json")
    _write_json([asdict(objective) for objective in objectives], run_root / "objective_manifest.json")
    _write_json(tool_versions, run_root / "tool_versions.json")

    conn = _open_store(store_path)
    results = []
    try:
        for design in panel.designs:
            for objective in objectives:
                for seed in panel.seeds:
                    result = _run_one_cell(
                        dreamplace_root=Path(dreamplace_root),
                        panel=panel,
                        design=design,
                        objective=objective,
                        seed=seed,
                        run_root=run_root,
                        resume=resume,
                        retry_failed=retry_failed,
                        tool_versions=tool_versions,
                        require_output_artifact=require_output_artifact,
                        require_def_output=require_def_output,
                        disable_legalization=disable_legalization,
                    )
                    _upsert_result(conn, result)
                    results.append(result)
    finally:
        conn.close()

    metrics_csv = write_metrics_csv(results, run_root / "metrics.csv")
    comparison = build_comparison_rows(results)
    comparison_csv = write_comparison_csv(comparison, run_root / "comparison_table.csv")
    report = build_report(
        panel=panel,
        objectives=objectives,
        results=results,
        comparison_rows=comparison,
    )
    report_path = run_root / "tier2_report.md"
    report_path.write_text(report, encoding="utf-8")
    summary = {
        "run_dir": str(run_root),
        "store": str(store_path),
        "panel_manifest": str(run_root / "panel_manifest.json"),
        "objective_manifest": str(run_root / "objective_manifest.json"),
        "tool_versions": str(run_root / "tool_versions.json"),
        "metrics_csv": str(metrics_csv),
        "comparison_csv": str(comparison_csv),
        "tier2_report": str(report_path),
        "result_count": len(results),
        "success_count": sum(1 for result in results if result.status == "success"),
        "failure_count": sum(1 for result in results if result.status != "success"),
        "minimum_scoring_iterations": MINIMUM_SCORING_ITERATIONS,
        "claim_valid_iteration_budget": (
            panel.iterations >= MINIMUM_SCORING_ITERATIONS
            and all(result.iteration_budget_satisfied for result in results)
        ),
    }
    _write_json(summary, run_root / "summary.json")
    return summary


def prepare_objectives(
    objective_paths: list[str | Path],
    *,
    run_root: Path,
    include_default: bool,
    include_custom_default: bool,
    objective_semantics: str | None = None,
) -> list[Tier2Objective]:
    objectives: list[Tier2Objective] = []
    if include_default:
        objectives.append(
            Tier2Objective(
                objective_id=DEFAULT_OBJECTIVE_ID,
                objective_path=None,
                source="native_dreamplace",
                spec_id=None,
                term_set=[],
            )
        )
    if include_custom_default:
        spec = objective_preset(CUSTOM_DEFAULT_PRESET)
        path = write_objective_spec(spec, run_root / "objectives" / "custom_default.json")
        objectives.append(
            _objective_from_spec(
                CUSTOM_DEFAULT_OBJECTIVE_ID,
                path,
                spec,
                "native_identity_preset",
            )
        )

    used_ids = {objective.objective_id for objective in objectives}
    for raw_path in objective_paths:
        path = Path(raw_path)
        spec = load_objective_spec(path)
        unsupported = unsupported_terms_for_scope(spec.term_set, "dreamplace")
        if unsupported:
            raise ValueError(
                f"{path} contains DREAMPlace-unsupported terms {unsupported}; "
                "Tier-2 accepts deployable terms only"
            )
        objective_id = _unique_objective_id(spec.id, path, used_ids)
        used_ids.add(objective_id)
        objectives.append(
            _objective_from_spec(
                objective_id,
                path,
                spec,
                "file",
                objective_semantics=objective_semantics,
            )
        )
    return objectives


def build_comparison_rows(results: list[Tier2Result]) -> list[dict[str, Any]]:
    baseline_id = _comparison_baseline_id(results)
    by_key = {(r.design, r.seed, r.objective_id): r for r in results}
    rows = []
    for result in results:
        baseline = by_key.get((result.design, result.seed, baseline_id))
        row = _result_csv_row(result)
        row["baseline_objective_id"] = baseline_id
        row["hpwl_delta_pct"] = _delta_pct(result.hpwl, baseline.hpwl if baseline else None)
        row["overflow_delta_pct"] = _delta_pct(result.overflow, baseline.overflow if baseline else None)
        row["wns_delta"] = _delta_value(result.wns, baseline.wns if baseline else None)
        row["tns_delta"] = _delta_value(result.tns, baseline.tns if baseline else None)
        row["tier2_score"] = _tier2_score(row, result.status)
        row["seed_variance_penalty"] = _seed_variance_penalty(results, result.design, result.objective_id)
        rows.append(row)
    return rows


def write_metrics_csv(results: list[Tier2Result], output: str | Path) -> Path:
    rows = [_result_csv_row(result) for result in results]
    return _write_csv(rows, output, TIER2_TABLE_COLUMNS)


def write_comparison_csv(rows: list[dict[str, Any]], output: str | Path) -> Path:
    fieldnames = TIER2_TABLE_COLUMNS + [
        "baseline_objective_id",
        "hpwl_delta_pct",
        "overflow_delta_pct",
        "wns_delta",
        "tns_delta",
        "tier2_score",
        "seed_variance_penalty",
    ]
    return _write_csv(rows, output, fieldnames)


def build_report(
    *,
    panel: DreamPlacePanel,
    objectives: list[Tier2Objective],
    results: list[Tier2Result],
    comparison_rows: list[dict[str, Any]],
) -> str:
    successes = [result for result in results if result.status == "success"]
    normalization = _normalization_summary(results)
    lines = [
        "# Tier-2 DREAMPlace Report",
        "",
        "## Panel",
        f"- Designs: {len(panel.designs)}",
        f"- Seeds: {panel.seeds}",
        f"- Objectives: {len(objectives)}",
        f"- Iterations: {panel.iterations}",
        f"- Minimum scoring budget: {MINIMUM_SCORING_ITERATIONS}",
        (
            "- Claim-valid iteration budget: "
            f"{panel.iterations >= MINIMUM_SCORING_ITERATIONS and all(result.iteration_budget_satisfied for result in results)}"
        ),
        "- Runs below the minimum are infrastructure tests only and cannot enter evolution or timing feedback.",
        f"- Timing mode: {panel.timing_mode}",
        "",
        "## Normalization Regression",
        f"- Baseline used for comparison: {_comparison_baseline_id(results)}",
        (
            "- Custom-default control: internal native-objective identity path "
            f"({CUSTOM_DEFAULT_PRESET}), not an LLM-facing term."
        ),
        (
            "- Normalization coverage: "
            f"{normalization['compared_pairs']}/{normalization['expected_pairs']}"
        ),
        (
            "- Custom-default executed through custom path: "
            f"{normalization['custom_default_called_pairs']}/{normalization['expected_pairs']}"
        ),
        f"- Max custom-default HPWL delta vs native default: {normalization['max_hpwl_delta_pct']}",
        f"- Max custom-default overflow delta vs native default: {normalization['max_overflow_delta_pct']}",
        "",
        "## Run Status",
        f"- Total cells: {len(results)}",
        f"- Successes: {len(successes)}",
        f"- Failures: {len(results) - len(successes)}",
        "",
        "## Best Completed Rows",
    ]
    completed = [
        row for row in comparison_rows if row.get("status") == "success" and _is_finite(row.get("tier2_score"))
    ]
    completed.sort(key=lambda row: float(row["tier2_score"]), reverse=True)
    if completed:
        for row in completed[:10]:
            lines.append(
                "- "
                f"{row['design']} {row['objective_id']} seed={row['seed']} "
                f"score={float(row['tier2_score']):.6g} "
                f"hpwl_delta={row['hpwl_delta_pct']} overflow_delta={row['overflow_delta_pct']}"
            )
    else:
        lines.append("- No successful comparable rows.")
    lines.extend(
        [
            "",
            "## Timing",
            "- Timing configs are compatibility-only in Tier-2 v1.",
            "- WNS/TNS values, when present, are exploratory and not used for claims.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _run_one_cell(
    *,
    dreamplace_root: Path,
    panel: DreamPlacePanel,
    design: DreamPlaceDesign,
    objective: Tier2Objective,
    seed: int,
    run_root: Path,
    resume: bool,
    retry_failed: bool,
    tool_versions: dict[str, Any],
    require_output_artifact: bool = False,
    require_def_output: bool = False,
    disable_legalization: bool = True,
) -> Tier2Result:
    run_dir = run_root / _safe_name(design.name) / _safe_name(objective.objective_id) / f"seed_{seed}"
    summary_path = run_dir / "run_summary.json"
    if resume and summary_path.is_file():
        previous = _load_result_from_summary(
            summary_path,
            design.name,
            objective.objective_id,
            seed,
            tool_versions=tool_versions,
            require_output_artifact=require_output_artifact,
            require_def_output=require_def_output,
            expected_iterations=panel.iterations,
        )
        if previous.status == "success" or not retry_failed:
            # Resume revalidates legacy summaries against the current artifact
            # and movable-macro contract. Persist that normalized result so
            # downstream audits do not observe stale pre-fix status fields.
            _write_json(asdict(previous), summary_path)
            return previous
    start = time.time()
    try:
        config_path = make_run_config(
            design.base_config,
            objective.objective_path,
            run_dir,
            iterations=panel.iterations,
            gpu=panel.gpu,
            seed=seed,
            custom_objective_log_interval=panel.log_interval,
            disable_legalization=disable_legalization,
            stop_overflow=panel.stop_overflow,
            objective_semantics=objective.objective_semantics,
        )
    except Exception as exc:
        return _failed_result(
            design=design.name,
            objective_id=objective.objective_id,
            seed=seed,
            run_dir=run_dir,
            failure_stage="config_parse",
            message=str(exc),
            runtime_seconds=time.time() - start,
            requested_iterations=panel.iterations,
        )
    run = run_dreamplace(
        dreamplace_root=dreamplace_root,
        config_path=config_path,
        run_name=f"{design.name}_{objective.objective_id}_seed_{seed}",
        run_dir=run_dir,
        timeout_seconds=panel.timeout_seconds,
        driver=panel.driver or "dreamplace/Placer.py",
    )
    write_run_summary(run, run_dir / "dreamplace_run.json")
    macro_runtime_requirement = _macro_runtime_requirement(design.base_config)
    result = _result_from_run(
        design=design.name,
        objective_id=objective.objective_id,
        seed=seed,
        run_dir=run_dir,
        run=run,
        runtime_seconds=time.time() - start,
        tool_versions=tool_versions,
        require_output_artifact=require_output_artifact,
        require_def_output=require_def_output,
        macro_runtime_requirement=macro_runtime_requirement,
        requested_iterations=panel.iterations,
    )
    _write_json(asdict(result), summary_path)
    return result


def _result_from_run(
    *,
    design: str,
    objective_id: str,
    seed: int,
    run_dir: Path,
    run: Any,
    runtime_seconds: float,
    tool_versions: dict[str, Any],
    require_output_artifact: bool = False,
    require_def_output: bool = False,
    macro_runtime_requirement: dict[str, Any] | None = None,
    requested_iterations: int = 0,
) -> Tier2Result:
    metrics = run.metrics or {}
    last = metrics.get("last", {})
    custom = metrics.get("custom_objective", {})
    timing_policy_trajectory = custom.get("timing_policy_trajectory") or []
    failure_stage = _failure_stage(run.returncode, metrics, last)
    completed_iterations = int(metrics.get("global_place_iterations_completed") or 0)
    iteration_budget_satisfied = (
        requested_iterations <= 0 or completed_iterations >= requested_iterations
    )
    metrics["requested_iterations"] = int(requested_iterations)
    metrics["completed_iterations"] = completed_iterations
    metrics["iteration_budget_satisfied"] = iteration_budget_satisfied
    if failure_stage is None and not iteration_budget_satisfied:
        failure_stage = "iteration_budget"
    output_artifact = _find_output_artifact(run_dir)
    if failure_stage is None and require_output_artifact and output_artifact is None:
        failure_stage = "output_artifact"
    if (
        failure_stage is None
        and require_def_output
        and (output_artifact is None or Path(output_artifact).suffix.lower() != ".def")
    ):
        failure_stage = "output_artifact"
    macro_validation = _validate_macro_runtime(metrics, macro_runtime_requirement)
    macro_output_validation = _validate_macro_output(
        output_artifact,
        macro_runtime_requirement,
    )
    macro_validation = _reconcile_macro_runtime_validation(
        macro_validation,
        macro_output_validation,
        macro_runtime_requirement,
    )
    metrics["macro_runtime_validation"] = macro_validation
    metrics["macro_output_validation"] = macro_output_validation
    if failure_stage is None and not macro_validation["ok"]:
        failure_stage = "macro_runtime"
    if failure_stage is None and not macro_output_validation["ok"]:
        failure_stage = "macro_output"
    status = "success" if failure_stage is None else "failed"
    return Tier2Result(
        design=design,
        objective_id=objective_id,
        seed=seed,
        status=status,
        failure_stage=failure_stage,
        hpwl=_float_or_none(last.get("hpwl")),
        overflow=_float_or_none(last.get("overflow")),
        wns=_float_or_none(last.get("wns")),
        tns=_float_or_none(last.get("tns")),
        requested_iterations=int(requested_iterations),
        completed_iterations=completed_iterations,
        iteration_budget_satisfied=iteration_budget_satisfied,
        custom_objective_calls=int(custom.get("calls") or 0),
        custom_total=_float_or_none(custom.get("last_total")),
        custom_grad_norm=_float_or_none(custom.get("last_grad_norm")),
        component_summary_json=json.dumps(
            custom.get("component_summary") or {},
            sort_keys=True,
        ),
        timing_net_coverage=_float_or_none(metrics.get("timing_net_coverage")),
        timing_policy_update_count=max(
            [
                int(sample.get("update_count") or 0)
                for sample in timing_policy_trajectory
                if isinstance(sample, dict)
            ]
            or [0]
        ),
        timing_policy_summary_json=json.dumps(
            {
                "trajectory": timing_policy_trajectory,
                "last": timing_policy_trajectory[-1] if timing_policy_trajectory else None,
            },
            sort_keys=True,
        ),
        runtime_seconds=runtime_seconds,
        output_artifact=output_artifact,
        coevop_commit=_metadata_value(tool_versions, "coevop", "commit"),
        dreamplace_commit=_metadata_value(tool_versions, "dreamplace", "commit"),
        run_dir=str(run_dir),
        config_path=run.config_path,
        log_path=run.log_path,
        returncode=run.returncode,
        metrics=metrics,
    )


def _failed_result(
    *,
    design: str,
    objective_id: str,
    seed: int,
    run_dir: Path,
    failure_stage: str,
    message: str,
    runtime_seconds: float,
    tool_versions: dict[str, Any] | None = None,
    requested_iterations: int = 0,
) -> Tier2Result:
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"error": message, "last": {}, "custom_objective": {}}
    result = Tier2Result(
        design=design,
        objective_id=objective_id,
        seed=seed,
        status="failed",
        failure_stage=failure_stage,
        hpwl=None,
        overflow=None,
        wns=None,
        tns=None,
        requested_iterations=int(requested_iterations),
        completed_iterations=0,
        iteration_budget_satisfied=False,
        custom_objective_calls=0,
        custom_total=None,
        custom_grad_norm=None,
        component_summary_json="{}",
        runtime_seconds=runtime_seconds,
        output_artifact=None,
        coevop_commit=_metadata_value(tool_versions or {}, "coevop", "commit"),
        dreamplace_commit=_metadata_value(tool_versions or {}, "dreamplace", "commit"),
        run_dir=str(run_dir),
        config_path=None,
        log_path=None,
        returncode=None,
        metrics=metrics,
        timing_net_coverage=None,
        timing_policy_update_count=0,
        timing_policy_summary_json="{}",
    )
    _write_json(asdict(result), run_dir / "run_summary.json")
    return result


def _load_result_from_summary(
    summary_path: Path,
    design: str,
    objective_id: str,
    seed: int,
    *,
    tool_versions: dict[str, Any] | None = None,
    require_output_artifact: bool = False,
    require_def_output: bool = False,
    expected_iterations: int | None = None,
) -> Tier2Result:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if "metrics" in payload and "returncode" in payload:
        return _result_from_run(
            design=design,
            objective_id=objective_id,
            seed=seed,
            run_dir=summary_path.parent,
            run=type("Run", (), payload),
            runtime_seconds=_float_or_none(payload.get("runtime_seconds")),
            tool_versions=tool_versions or {},
            require_output_artifact=require_output_artifact,
            require_def_output=require_def_output,
            requested_iterations=int(expected_iterations or 0),
            macro_runtime_requirement=_macro_runtime_requirement(
                payload.get("config_path") or ""
            ),
        )
    payload.setdefault("coevop_commit", _metadata_value(tool_versions or {}, "coevop", "commit"))
    payload.setdefault(
        "dreamplace_commit",
        _metadata_value(tool_versions or {}, "dreamplace", "commit"),
    )
    payload.setdefault("component_summary_json", "{}")
    payload.setdefault(
        "timing_net_coverage",
        _float_or_none(payload.get("metrics", {}).get("timing_net_coverage")),
    )
    timing_trajectory = (
        payload.get("metrics", {})
        .get("custom_objective", {})
        .get("timing_policy_trajectory", [])
    )
    payload.setdefault(
        "timing_policy_update_count",
        max(
            [
                int(sample.get("update_count") or 0)
                for sample in timing_trajectory
                if isinstance(sample, dict)
            ]
            or [0]
        ),
    )
    payload.setdefault(
        "timing_policy_summary_json",
        json.dumps(
            {
                "trajectory": timing_trajectory,
                "last": timing_trajectory[-1] if timing_trajectory else None,
            },
            sort_keys=True,
        ),
    )
    requested_iterations = int(
        expected_iterations
        if expected_iterations is not None
        else payload.get("requested_iterations", 0)
    )
    metrics = payload.setdefault("metrics", {})
    completed_iterations = int(
        metrics.get(
            "global_place_iterations_completed",
            payload.get("completed_iterations", 0),
        )
        or 0
    )
    if completed_iterations == 0 and payload.get("log_path"):
        reparsed = parse_dreamplace_log(payload["log_path"])
        completed_iterations = int(
            reparsed.get("global_place_iterations_completed") or 0
        )
        metrics.update(reparsed)
    iteration_budget_satisfied = (
        requested_iterations <= 0 or completed_iterations >= requested_iterations
    )
    payload["requested_iterations"] = requested_iterations
    payload["completed_iterations"] = completed_iterations
    payload["iteration_budget_satisfied"] = iteration_budget_satisfied
    metrics["requested_iterations"] = requested_iterations
    metrics["completed_iterations"] = completed_iterations
    metrics["iteration_budget_satisfied"] = iteration_budget_satisfied
    if payload.get("status") == "success" and not iteration_budget_satisfied:
        payload["status"] = "failed"
        payload["failure_stage"] = "iteration_budget"
    result = Tier2Result(**payload)
    macro_validation = _validate_macro_runtime(
        result.metrics,
        _macro_runtime_requirement(result.config_path or ""),
    )
    missing_required_artifact = require_output_artifact and result.output_artifact is None
    missing_def_artifact = (
        require_def_output
        and (result.output_artifact is None or Path(result.output_artifact).suffix.lower() != ".def")
    )
    if result.status == "success" and not macro_validation["ok"]:
        return Tier2Result(
            **{
                **asdict(result),
                "status": "failed",
                "failure_stage": "macro_runtime",
                "metrics": {
                    **result.metrics,
                    "macro_runtime_validation": macro_validation,
                },
            }
        )
    if result.status == "success" and (missing_required_artifact or missing_def_artifact):
        return Tier2Result(
            **{
                **asdict(result),
                "status": "failed",
                "failure_stage": "output_artifact",
            }
        )
    return result


def _macro_runtime_requirement(base_config: str | Path) -> dict[str, Any]:
    """Read the fail-closed movable-macro contract embedded in a base config."""

    path = Path(base_config).expanduser()
    if not path.is_file():
        return {"required": False, "base_config": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"required": False, "base_config": str(path)}
    audit = payload.get("coevop_macro_preflight") or {}
    movable_count = int(audit.get("movable_macro_count") or 0)
    input_def_path = audit.get("def_path")
    lef_paths = [str(value) for value in audit.get("lef_paths", [])]
    input_components = (
        load_hard_macro_components(input_def_path, lef_paths)
        if input_def_path and lef_paths
        else []
    )
    movable_components = [
        item
        for item in input_components
        if str(item.get("status")) in {"PLACED", "UNPLACED"}
    ]
    unplaced_count = sum(
        str(item.get("status")) == "UNPLACED" for item in movable_components
    )
    return {
        # Movability is defined by DEF component status, not by DREAMPlace's
        # macro_place_flag. Search runs may disable the expensive macro
        # legalization stage while still optimizing movable macro coordinates.
        "required": bool(movable_count > 0),
        "macro_place_flag": int(payload.get("macro_place_flag") or 0),
        "base_config": str(path),
        "expected_movable_macro_count": movable_count,
        "expected_fixed_hard_macro_count": int(
            audit.get("fixed_macro_count", audit.get("fixed_hard_macro_count", 0)) or 0
        ),
        "input_def_sha256": audit.get("def_sha256", audit.get("input_def_sha256")),
        "input_def_path": input_def_path,
        "input_hard_macro_count": int(audit.get("hard_macro_count") or movable_count),
        "input_components": movable_components,
        "input_unplaced_macro_count": unplaced_count,
        "input_fixed_hard_macro_count": len(input_components) - len(movable_components),
    }


def _validate_macro_runtime(
    metrics: dict[str, Any],
    requirement: dict[str, Any] | None,
) -> dict[str, Any]:
    contract = dict(requirement or {"required": False})
    public_contract = {
        key: value for key, value in contract.items() if key != "input_components"
    }
    area = _float_or_none(metrics.get("movable_macro_area"))
    required = bool(contract.get("required"))
    ok = not required or (area is not None and area > 0.0)
    return {
        **public_contract,
        "reported_movable_macro_area": area,
        "ok": ok,
        "reason": None if ok else "DREAMPlace reported zero or missing movable macro area",
    }


def _reconcile_macro_runtime_validation(
    runtime_validation: dict[str, Any],
    output_validation: dict[str, Any],
    requirement: dict[str, Any] | None,
) -> dict[str, Any]:
    """Allow old DREAMPlace logs to use complete UNPLACED-macro DEF evidence.

    DREAMPlace 4.0 predates the ``total_movable_macro_area`` log field. Its
    parser compatibility path accepts selected hard macros only as UNPLACED,
    so a complete emitted DEF is the available runtime evidence. PLACED-input
    experiments retain the stricter positive-area requirement.
    """

    result = dict(runtime_validation)
    if result.get("ok") or not result.get("required"):
        return result
    contract = dict(requirement or {})
    expected = int(contract.get("expected_movable_macro_count") or 0)
    unplaced = int(contract.get("input_unplaced_macro_count") or 0)
    if expected > 0 and unplaced == expected and output_validation.get("ok"):
        result.update(
            {
                "ok": True,
                "reason": None,
                "validation_mode": "unplaced_input_output_coordinates",
                "runtime_area_log_available": False,
            }
        )
    return result


def _validate_macro_output(
    output_artifact: str | None,
    requirement: dict[str, Any] | None,
) -> dict[str, Any]:
    contract = dict(requirement or {"required": False})
    required = bool(contract.get("required"))
    if not required:
        return {"required": False, "ok": True, "reason": None}
    components = contract.get("input_components") or []
    if not output_artifact or Path(output_artifact).suffix.lower() != ".def":
        return {
            "required": True,
            "ok": False,
            "reason": "missing output DEF for macro-coordinate audit",
        }
    if not components:
        return {
            "required": True,
            "ok": False,
            "reason": "input hard-macro components were unavailable for output audit",
        }
    result = audit_output_macro_coordinates(
        input_components=components,
        output_def=output_artifact,
    )
    movement_required = bool(result.get("comparable_coordinate_count"))
    movement_observed = bool(result.get("changed_coordinate_count"))
    if movement_required and not movement_observed:
        result["ok"] = False
        result["movement_failure"] = "no selected movable macro changed coordinates"
    result["movement_required"] = movement_required
    result["movement_observed"] = movement_observed
    result["required"] = True
    result["reason"] = None if result["ok"] else "output macro-coordinate audit failed"
    return result


def _open_store(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        create table if not exists tier2_results (
            design text not null,
            objective_id text not null,
            seed integer not null,
            status text not null,
            failure_stage text,
            hpwl real,
            overflow real,
            wns real,
            tns real,
            requested_iterations integer not null default 0,
            completed_iterations integer not null default 0,
            iteration_budget_satisfied integer not null default 0,
            custom_objective_calls integer not null,
            custom_total real,
            custom_grad_norm real,
            timing_net_coverage real,
            timing_policy_update_count integer not null default 0,
            timing_policy_summary_json text not null default '{}',
            runtime_seconds real,
            output_artifact text,
            coevop_commit text,
            dreamplace_commit text,
            run_dir text not null,
            config_path text,
            log_path text,
            returncode integer,
            metrics_json text not null,
            updated_at text not null default current_timestamp,
            primary key (design, objective_id, seed)
        )
        """
    )
    _ensure_column(conn, "tier2_results", "coevop_commit", "text")
    _ensure_column(conn, "tier2_results", "dreamplace_commit", "text")
    _ensure_column(conn, "tier2_results", "requested_iterations", "integer not null default 0")
    _ensure_column(conn, "tier2_results", "completed_iterations", "integer not null default 0")
    _ensure_column(
        conn,
        "tier2_results",
        "iteration_budget_satisfied",
        "integer not null default 0",
    )
    _ensure_column(conn, "tier2_results", "timing_net_coverage", "real")
    _ensure_column(
        conn,
        "tier2_results",
        "timing_policy_update_count",
        "integer not null default 0",
    )
    _ensure_column(
        conn,
        "tier2_results",
        "timing_policy_summary_json",
        "text not null default '{}'",
    )
    conn.commit()
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, declaration: str) -> None:
    columns = {row[1] for row in conn.execute(f"pragma table_info({table})")}
    if column not in columns:
        conn.execute(f"alter table {table} add column {column} {declaration}")


def _upsert_result(conn: sqlite3.Connection, result: Tier2Result) -> None:
    conn.execute(
        """
        insert into tier2_results (
            design, objective_id, seed, status, failure_stage, hpwl, overflow,
            wns, tns, requested_iterations, completed_iterations,
            iteration_budget_satisfied, custom_objective_calls, custom_total, custom_grad_norm,
            timing_net_coverage, timing_policy_update_count, timing_policy_summary_json,
            runtime_seconds, output_artifact, coevop_commit, dreamplace_commit,
            run_dir, config_path, log_path,
            returncode, metrics_json
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(design, objective_id, seed) do update set
            status=excluded.status,
            failure_stage=excluded.failure_stage,
            hpwl=excluded.hpwl,
            overflow=excluded.overflow,
            wns=excluded.wns,
            tns=excluded.tns,
            requested_iterations=excluded.requested_iterations,
            completed_iterations=excluded.completed_iterations,
            iteration_budget_satisfied=excluded.iteration_budget_satisfied,
            custom_objective_calls=excluded.custom_objective_calls,
            custom_total=excluded.custom_total,
            custom_grad_norm=excluded.custom_grad_norm,
            timing_net_coverage=excluded.timing_net_coverage,
            timing_policy_update_count=excluded.timing_policy_update_count,
            timing_policy_summary_json=excluded.timing_policy_summary_json,
            runtime_seconds=excluded.runtime_seconds,
            output_artifact=excluded.output_artifact,
            coevop_commit=excluded.coevop_commit,
            dreamplace_commit=excluded.dreamplace_commit,
            run_dir=excluded.run_dir,
            config_path=excluded.config_path,
            log_path=excluded.log_path,
            returncode=excluded.returncode,
            metrics_json=excluded.metrics_json,
            updated_at=current_timestamp
        """,
        (
            result.design,
            result.objective_id,
            result.seed,
            result.status,
            result.failure_stage,
            result.hpwl,
            result.overflow,
            result.wns,
            result.tns,
            result.requested_iterations,
            result.completed_iterations,
            int(result.iteration_budget_satisfied),
            result.custom_objective_calls,
            result.custom_total,
            result.custom_grad_norm,
            result.timing_net_coverage,
            result.timing_policy_update_count,
            result.timing_policy_summary_json,
            result.runtime_seconds,
            result.output_artifact,
            result.coevop_commit,
            result.dreamplace_commit,
            result.run_dir,
            result.config_path,
            result.log_path,
            result.returncode,
            json.dumps(result.metrics, sort_keys=True),
        ),
    )
    conn.commit()


def _comparison_baseline_id(results: list[Tier2Result]) -> str:
    normalization = _normalization_summary(results)
    if not normalization["coverage_complete"]:
        return DEFAULT_OBJECTIVE_ID
    max_hpwl = abs(float(normalization["max_hpwl_delta_pct"] or 0.0))
    max_overflow = abs(float(normalization["max_overflow_delta_pct"] or 0.0))
    if max(max_hpwl, max_overflow) > 1.0:
        return CUSTOM_DEFAULT_OBJECTIVE_ID
    return DEFAULT_OBJECTIVE_ID


def _normalization_summary(results: list[Tier2Result]) -> dict[str, Any]:
    by_key = {(r.design, r.seed, r.objective_id): r for r in results}
    expected_pairs = {
        (result.design, result.seed)
        for result in results
        if result.objective_id in {DEFAULT_OBJECTIVE_ID, CUSTOM_DEFAULT_OBJECTIVE_ID}
    }
    hpwl_deltas = []
    overflow_deltas = []
    compared = 0
    for (design, seed, objective_id), custom in by_key.items():
        if objective_id != CUSTOM_DEFAULT_OBJECTIVE_ID:
            continue
        default = by_key.get((design, seed, DEFAULT_OBJECTIVE_ID))
        if default is None:
            continue
        hpwl_delta = _delta_pct(custom.hpwl, default.hpwl)
        overflow_delta = _delta_pct(custom.overflow, default.overflow)
        if (
            custom.status == "success"
            and default.status == "success"
            and (_is_finite(hpwl_delta) or _is_finite(overflow_delta))
        ):
            compared += 1
        if _is_finite(hpwl_delta):
            hpwl_deltas.append(abs(float(hpwl_delta)))
        if _is_finite(overflow_delta):
            overflow_deltas.append(abs(float(overflow_delta)))
    return {
        "max_hpwl_delta_pct": max(hpwl_deltas) if hpwl_deltas else None,
        "max_overflow_delta_pct": max(overflow_deltas) if overflow_deltas else None,
        "compared_pairs": compared,
        "expected_pairs": len(expected_pairs),
        "custom_default_called_pairs": sum(
            1
            for result in results
            if result.objective_id == CUSTOM_DEFAULT_OBJECTIVE_ID
            and result.status == "success"
            and int(result.custom_objective_calls or 0) > 0
        ),
        "coverage_complete": compared == len(expected_pairs) and bool(expected_pairs),
    }


def _failure_stage(returncode: int, metrics: dict[str, Any], last: dict[str, Any]) -> str | None:
    if metrics.get("timed_out") or returncode == -9:
        return "timeout"
    if metrics.get("missing_input_file") or metrics.get("assertion"):
        return "solver_init"
    if returncode != 0:
        return "placement_iter"
    if not last:
        return "metric_extraction"
    if not (_is_finite(last.get("hpwl")) or _is_finite(last.get("overflow"))):
        return "metric_extraction"
    custom = metrics.get("custom_objective", {})
    if custom.get("calls") and not _is_finite(custom.get("last_grad_norm")):
        return "metric_extraction"
    return None


def _tier2_score(row: dict[str, Any], status: str) -> float:
    if status != "success":
        return -100.0
    score = 0.0
    hpwl_delta = row.get("hpwl_delta_pct")
    overflow_delta = row.get("overflow_delta_pct")
    if _is_finite(hpwl_delta):
        score -= float(hpwl_delta)
    if _is_finite(overflow_delta):
        score -= 0.5 * float(overflow_delta)
    runtime = row.get("runtime_seconds")
    if _is_finite(runtime):
        score -= min(float(runtime), 3600.0) / 3600.0
    return score


def _seed_variance_penalty(
    results: list[Tier2Result],
    design: str,
    objective_id: str,
) -> float | None:
    hpwls = [
        float(result.hpwl)
        for result in results
        if result.design == design
        and result.objective_id == objective_id
        and result.status == "success"
        and _is_finite(result.hpwl)
    ]
    if len(hpwls) < 2:
        return None
    mean = sum(hpwls) / len(hpwls)
    if mean == 0:
        return None
    variance = sum((value - mean) ** 2 for value in hpwls) / (len(hpwls) - 1)
    return 100.0 * math.sqrt(variance) / abs(mean)


def _result_csv_row(result: Tier2Result) -> dict[str, Any]:
    return {column: getattr(result, column) for column in TIER2_TABLE_COLUMNS}


def _write_csv(rows: list[dict[str, Any]], output: str | Path, fieldnames: list[str]) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def _write_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _validate_timing_mode(value: str) -> None:
    if value not in TIMING_MODES:
        raise ValueError(f"invalid timing_mode {value!r}; expected one of {sorted(TIMING_MODES)}")


def _git_metadata(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=resolved,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=resolved,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=resolved,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"root": git_root, "commit": commit, "dirty": dirty}
    except Exception as exc:
        return {"root": str(resolved), "commit": None, "dirty": None, "error": str(exc)}


def _metadata_value(payload: dict[str, Any], section: str, key: str) -> Any:
    section_payload = payload.get(section, {})
    if not isinstance(section_payload, dict):
        return None
    return section_payload.get(key)


def _objective_from_spec(
    objective_id: str,
    path: Path,
    spec: ObjectiveSpec,
    source: str,
    objective_semantics: str | None = None,
) -> Tier2Objective:
    # Stateful controller specs are only meaningful over raw terms: their
    # registers carry the scale, and hidden value normalization would silently
    # re-pin the tradeoff the controller is supposed to schedule.
    semantics = objective_semantics
    if getattr(spec, "is_stateful", False):
        if semantics == "normalized":
            raise ValueError(
                f"stateful objective {spec.id} cannot run with normalized semantics"
            )
        semantics = "raw"
    return Tier2Objective(
        objective_id=objective_id,
        objective_path=str(path),
        source=source,
        spec_id=spec.id,
        term_set=spec.term_set,
        objective_semantics=semantics,
    )


def _unique_objective_id(spec_id: str, path: Path, used_ids: set[str]) -> str:
    if spec_id not in used_ids:
        return spec_id
    stem = _safe_name(path.stem)
    candidate = f"{spec_id}_{stem}"
    suffix = 2
    while candidate in used_ids:
        candidate = f"{spec_id}_{stem}_{suffix}"
        suffix += 1
    return candidate


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _resolve_panel_path(panel_path: Path, value: str) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    if path.is_absolute():
        return path
    return (panel_path.parent / path).resolve()


def _find_output_artifact(run_dir: Path) -> str | None:
    result_dir = run_dir / "results"
    if not result_dir.is_dir():
        return None
    candidates = []
    for suffix in ("*.gp.def", "*.gp.pl", "*.def", "*.pl"):
        candidates.extend(result_dir.rglob(suffix))
    if not candidates:
        return None
    candidates.sort(key=lambda path: (path.suffix != ".def", str(path)))
    return str(candidates[0])


def _delta_pct(value: Any, baseline: Any) -> float | None:
    if not (_is_finite(value) and _is_finite(baseline)):
        return None
    baseline_value = float(baseline)
    if baseline_value == 0.0:
        return None
    return 100.0 * (float(value) - baseline_value) / abs(baseline_value)


def _delta_value(value: Any, baseline: Any) -> float | None:
    """Return a signed absolute delta; positive WNS/TNS means improvement."""

    if not (_is_finite(value) and _is_finite(baseline)):
        return None
    return float(value) - float(baseline)


def _float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
