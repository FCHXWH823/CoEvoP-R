"""OpenTimer/DREAMPlace timing-proxy feedback.

This evaluator intentionally uses DREAMPlace's timing-enabled OpenTimer path
instead of calling ``ot-shell`` directly. DREAMPlace constructs placement-based
FLUTE/Elmore RC trees from DEF coordinates and per-micron R/C, then reports STA
metrics through OpenTimer. The signal is therefore a timing proxy, not routed
parasitic signoff.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised by py310 via tomli
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from coevop.backends.dreamplace import run_dreamplace, write_run_summary

TIMING_PROXY_RC_MODEL = "flute_elmore_per_micron"
TIMING_PROXY_MODES = {"diagnostic", "tie_breaker", "gate"}
TIMING_PROXY_STATUSES = {
    "success",
    "partial",
    "failed",
    "missing_timing_collateral",
    "not_run",
    "unavailable",
}
TIMING_PROXY_FAILURE_STAGES = {
    "missing_timing_collateral",
    "missing_placement_def",
    "timer_init",
    "rc_tree_construction",
    "sta_update",
    "metric_parse",
    "timing_coverage",
    "timeout",
    "dreamplace",
}
TIMING_PROXY_COLUMNS = [
    "design",
    "objective_id",
    "seed",
    "variant",
    "status",
    "failure_stage",
    "timing_proxy_rc_model",
    "timing_proxy_hpwl",
    "placement_overflow",
    "placement_max_density",
    "timing_proxy_total_nets",
    "timing_proxy_skipped_nets",
    "timing_proxy_pin_incomplete_nets",
    "timing_proxy_net_coverage",
    "timing_proxy_wns",
    "timing_proxy_tns",
    "timing_proxy_num_violating_paths",
    "timing_proxy_runtime_seconds",
    "source_placement_iterations",
    "source_placement_completed_iterations",
    "source_placement_budget_satisfied",
    "def_path",
    "artifacts",
]


@dataclass(frozen=True)
class TimingProxyDesign:
    name: str
    base_config: str
    lib: str | list[str] | None = None
    sdc: str | None = None
    verilog: str | None = None
    top_module: str | None = None
    lef: list[str] | None = None
    rc_estimate: str = TIMING_PROXY_RC_MODEL


@dataclass(frozen=True)
class TimingProxyPanel:
    ot_shell: str
    timeout_seconds: int
    iterations: int
    gpu: int | None
    threads: int
    designs: list[TimingProxyDesign]
    legalize: bool = True
    learning_rate: float | None = None
    fixed_position: bool = False
    position_jitter: float = 0.0
    min_net_coverage: float = 0.95
    scalarize_vector_nets: bool = False
    derive_verilog_from_def: bool = False


@dataclass(frozen=True)
class TimingProxyResult:
    design: str
    objective_id: str
    seed: int
    variant: str
    status: str
    failure_stage: str | None
    timing_proxy_rc_model: str
    timing_proxy_hpwl: float | None
    timing_proxy_wns: float | None
    timing_proxy_tns: float | None
    timing_proxy_num_violating_paths: int | None
    timing_proxy_runtime_seconds: float | None
    def_path: str
    run_dir: str
    config_path: str | None
    log_path: str | None
    returncode: int | None
    artifacts: dict[str, Any]
    placement_overflow: float | None = None
    placement_max_density: float | None = None
    timing_proxy_total_nets: int | None = None
    timing_proxy_skipped_nets: int | None = None
    timing_proxy_pin_incomplete_nets: int | None = None
    timing_proxy_net_coverage: float | None = None
    source_placement_iterations: int = 0
    source_placement_completed_iterations: int = 0
    source_placement_budget_satisfied: bool = False


@dataclass(frozen=True)
class TimingProxyPlacement:
    design: str
    objective_id: str
    seed: int
    def_path: str
    source: str | None = None
    variant: str = "original"
    requested_iterations: int = 0
    completed_iterations: int = 0
    iteration_budget_satisfied: bool = False


def load_timing_proxy_placements(path: str | Path) -> list[TimingProxyPlacement]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    raw_items = payload.get("placements", payload if isinstance(payload, list) else [])
    if not isinstance(raw_items, list):
        raise ValueError("placements JSON must contain a 'placements' list")
    placements = []
    for item in raw_items:
        placements.append(
            TimingProxyPlacement(
                design=str(item["design"]),
                objective_id=str(item["objective_id"]),
                seed=int(item["seed"]),
                def_path=str(item["def_path"]),
                source=item.get("source"),
                variant=str(item.get("variant", "original")),
                requested_iterations=int(item.get("requested_iterations") or 0),
                completed_iterations=int(item.get("completed_iterations") or 0),
                iteration_budget_satisfied=bool(
                    item.get("iteration_budget_satisfied", False)
                ),
            )
        )
    return placements


def load_timing_proxy_panel(path: str | Path) -> TimingProxyPanel:
    panel_path = Path(path)
    payload = tomllib.loads(panel_path.read_text(encoding="utf-8"))
    designs = []
    for item in payload.get("designs", []):
        name = str(item["name"])
        base_value = item.get("base_config", item.get("dreamplace_config"))
        if base_value is None:
            raise ValueError(f"timing proxy design {name} must define base_config")
        designs.append(
            TimingProxyDesign(
                name=name,
                base_config=str(_resolve_panel_path(panel_path, str(base_value))),
                lib=_optional_paths(panel_path, item.get("lib", payload.get("lib"))),
                sdc=_optional_path(panel_path, item.get("sdc")),
                verilog=_optional_path(panel_path, item.get("verilog")),
                top_module=str(item["top_module"]) if item.get("top_module") else None,
                lef=[
                    str(_resolve_panel_path(panel_path, str(value)))
                    for value in item.get("lef", [])
                ]
                if item.get("lef") is not None
                else None,
                rc_estimate=str(item.get("rc_estimate", TIMING_PROXY_RC_MODEL)),
            )
        )
    if not designs:
        raise ValueError("timing proxy panel must contain at least one [[designs]] entry")
    return TimingProxyPanel(
        ot_shell=str(
            _resolve_panel_path(
                panel_path,
                str(payload.get("ot_shell", "${DREAMPLACE_ROOT}/bin/ot-shell")),
            )
        ),
        timeout_seconds=int(payload.get("timeout_seconds", 600)),
        iterations=int(payload.get("iterations", 1)),
        gpu=int(payload["gpu"]) if payload.get("gpu") is not None else None,
        threads=int(payload.get("threads", 8)),
        designs=designs,
        legalize=bool(payload.get("legalize", True)),
        learning_rate=(
            float(payload["learning_rate"])
            if payload.get("learning_rate") is not None
            else None
        ),
        fixed_position=bool(payload.get("fixed_position", False)),
        position_jitter=float(payload.get("position_jitter", 0.0)),
        min_net_coverage=float(payload.get("min_net_coverage", 0.95)),
        scalarize_vector_nets=bool(payload.get("scalarize_vector_nets", False)),
        derive_verilog_from_def=bool(payload.get("derive_verilog_from_def", False)),
    )


def run_timing_proxy_eval(
    *,
    panel_path: str | Path,
    placements_path: str | Path,
    dreamplace_root: str | Path,
    run_dir: str | Path,
    resume: bool = False,
    retry_failed: bool = False,
) -> dict[str, Any]:
    panel = load_timing_proxy_panel(panel_path)
    placements = load_timing_proxy_placements(placements_path)
    run_root = Path(run_dir).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    design_by_name = {design.name: design for design in panel.designs}
    results = []
    for placement in placements:
        design = design_by_name.get(placement.design)
        result = _run_one_timing_proxy_cell(
            panel=panel,
            design=design,
            placement=placement,
            dreamplace_root=Path(dreamplace_root),
            run_root=run_root,
            resume=resume,
            retry_failed=retry_failed,
        )
        results.append(result)

    metrics_csv = write_timing_proxy_metrics_csv(results, run_root / "metrics.csv")
    comparison_rows = build_timing_proxy_comparison_rows(results)
    comparison_csv = write_timing_proxy_comparison_csv(
        comparison_rows,
        run_root / "comparison_table.csv",
    )
    report_path = run_root / "timing_proxy_report.md"
    report_path.write_text(
        build_timing_proxy_report(panel=panel, results=results),
        encoding="utf-8",
    )
    summary = {
        "run_dir": str(run_root),
        "panel": str(panel_path),
        "placements": str(placements_path),
        "metrics_csv": str(metrics_csv),
        "comparison_csv": str(comparison_csv),
        "timing_proxy_report": str(report_path),
        "result_count": len(results),
        "success_count": sum(1 for item in results if item.status == "success"),
        "failure_count": sum(1 for item in results if item.status != "success"),
        "timing_proxy_rc_model": TIMING_PROXY_RC_MODEL,
    }
    _write_json([asdict(result) for result in results], run_root / "results.json")
    _write_json(summary, run_root / "summary.json")
    return summary


def run_timing_proxy_audit(
    *,
    design: str,
    base_config: str | Path,
    timing_panel_path: str | Path | None = None,
    placements_path: str | Path,
    dreamplace_root: str | Path,
    run_dir: str | Path,
    perturbations: int = 10,
    timeout_seconds: int = 600,
    gpu: int | None = None,
    seed: int = 1000,
) -> dict[str, Any]:
    run_root = Path(run_dir).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    placements = [item for item in load_timing_proxy_placements(placements_path) if item.design == design]
    if not placements:
        raise ValueError(f"placements file does not contain design {design!r}")
    selected = _audit_seed_placements(placements)
    perturbed_payload = {"placements": []}
    perturbation_records = []
    magnitudes = _perturbation_magnitudes(max(1, perturbations))
    for source_index, placement in enumerate(selected):
        source_def = _resolve_existing_path(placement.def_path, base=Path(placements_path).parent)
        perturbed_payload["placements"].append(
            {
                "design": placement.design,
                "objective_id": placement.objective_id,
                "seed": placement.seed,
                "def_path": str(source_def),
                "source": "timing_proxy_audit_original",
                "variant": f"source_{source_index:02d}_original",
            }
        )
        perturbation_records.append(
            {
                "source_objective_id": placement.objective_id,
                "source_seed": placement.seed,
                "variant": f"source_{source_index:02d}_original",
                "magnitude_dbu": 0,
                "def_path": str(source_def),
            }
        )
        for index, magnitude in enumerate(magnitudes, start=1):
            output_def = (
                run_root
                / "perturbed_defs"
                / _safe_name(placement.objective_id)
                / f"seed_{placement.seed}"
                / f"perturb_{index:03d}_mag_{magnitude}.def"
            )
            perturb_def(
                source_def,
                output_def,
                magnitude_dbu=magnitude,
                seed=seed + source_index * 1000 + index,
            )
            variant = f"source_{source_index:02d}_perturb_{index:03d}"
            perturbed_payload["placements"].append(
                {
                    "design": placement.design,
                    "objective_id": f"{placement.objective_id}__{variant}",
                    "seed": placement.seed,
                    "def_path": str(output_def),
                    "source": "timing_proxy_audit_perturbation",
                    "variant": variant,
                }
            )
            perturbation_records.append(
                {
                    "source_objective_id": placement.objective_id,
                    "source_seed": placement.seed,
                    "variant": variant,
                    "magnitude_dbu": magnitude,
                    "def_path": str(output_def),
                }
            )

    panel_path = _write_audit_panel(
        run_root / "timing_proxy_panel.toml",
        base_config=base_config,
        source_panel=timing_panel_path,
        design=design,
        dreamplace_root=Path(dreamplace_root),
        timeout_seconds=timeout_seconds,
        gpu=gpu,
    )
    perturbed_placements = _write_json(
        perturbed_payload,
        run_root / "audit_placements.json",
    )
    timing_summary = run_timing_proxy_eval(
        panel_path=panel_path,
        placements_path=perturbed_placements,
        dreamplace_root=dreamplace_root,
        run_dir=run_root / "timing_proxy_eval",
        resume=True,
        retry_failed=False,
    )
    results = _load_json(Path(timing_summary["run_dir"]) / "results.json")
    audit_rows = _audit_rows(results, perturbation_records)
    correlations = audit_correlations(audit_rows)
    recommendation = timing_proxy_mode_from_correlations(
        correlations,
        gate_threshold=0.70,
        tiebreaker_threshold=0.95,
    )
    rows_csv = _write_csv(
        audit_rows,
        run_root / "audit_rows.csv",
        [
            "source_objective_id",
            "source_seed",
            "variant",
            "magnitude_dbu",
            "hpwl",
            "wns",
            "tns",
            "hpwl_delta",
            "wns_delta",
            "tns_delta",
            "status",
        ],
    )
    summary = {
        "run_dir": str(run_root),
        "design": design,
        "base_config": str(base_config),
        "timing_panel": str(timing_panel_path) if timing_panel_path else None,
        "placements": str(placements_path),
        "perturbations": perturbations,
        "timing_proxy_eval": timing_summary,
        "audit_rows_csv": str(rows_csv),
        "correlations": correlations,
        "recommended_mode": recommendation["mode"],
        "recommendation_reason": recommendation["reason"],
        "timing_proxy_rc_model": TIMING_PROXY_RC_MODEL,
    }
    _write_json(perturbation_records, run_root / "perturbations.json")
    _write_json(summary, run_root / "summary.json")
    (run_root / "timing_proxy_audit_report.md").write_text(
        build_timing_proxy_audit_report(summary, audit_rows),
        encoding="utf-8",
    )
    return summary


def perturb_def(
    input_def: str | Path,
    output_def: str | Path,
    *,
    magnitude_dbu: int,
    seed: int,
) -> Path:
    """Perturb non-fixed DEF component coordinates by bounded integer noise."""

    input_path = Path(input_def)
    output_path = Path(output_def)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    pattern = re.compile(r"(?P<prefix>\+\s+PLACED\s+\(\s*)(?P<x>-?\d+)(?P<mid>\s+)(?P<y>-?\d+)(?P<suffix>\s*\))")
    changed = 0
    lines = []
    for line in input_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match or magnitude_dbu <= 0:
            lines.append(line)
            continue
        x = max(0, int(match.group("x")) + rng.randint(-magnitude_dbu, magnitude_dbu))
        y = max(0, int(match.group("y")) + rng.randint(-magnitude_dbu, magnitude_dbu))
        line = (
            line[: match.start()]
            + f"{match.group('prefix')}{x}{match.group('mid')}{y}{match.group('suffix')}"
            + line[match.end() :]
        )
        changed += 1
        lines.append(line)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(
        {
            "input_def": str(input_path),
            "output_def": str(output_path),
            "magnitude_dbu": magnitude_dbu,
            "seed": seed,
            "changed_component_count": changed,
        },
        output_path.with_suffix(output_path.suffix + ".perturbation.json"),
    )
    return output_path


def audit_correlations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hpwl = [_float_or_none(row.get("hpwl_delta")) for row in rows]
    wns = [_float_or_none(row.get("wns_delta")) for row in rows]
    tns = [_float_or_none(row.get("tns_delta")) for row in rows]
    hpwl_wns_pairs = _finite_pairs(hpwl, wns)
    hpwl_tns_pairs = _finite_pairs(hpwl, tns)
    return {
        "hpwl_wns_pearson": pearson([a for a, _ in hpwl_wns_pairs], [b for _, b in hpwl_wns_pairs]),
        "hpwl_wns_spearman": spearman([a for a, _ in hpwl_wns_pairs], [b for _, b in hpwl_wns_pairs]),
        "hpwl_tns_pearson": pearson([a for a, _ in hpwl_tns_pairs], [b for _, b in hpwl_tns_pairs]),
        "hpwl_tns_spearman": spearman([a for a, _ in hpwl_tns_pairs], [b for _, b in hpwl_tns_pairs]),
        "pair_count_wns": len(hpwl_wns_pairs),
        "pair_count_tns": len(hpwl_tns_pairs),
    }


def timing_proxy_mode_from_correlations(
    correlations: dict[str, Any],
    *,
    gate_threshold: float,
    tiebreaker_threshold: float,
) -> dict[str, str]:
    values = [
        abs(value)
        for value in (
            _float_or_none(correlations.get("hpwl_wns_pearson")),
            _float_or_none(correlations.get("hpwl_wns_spearman")),
            _float_or_none(correlations.get("hpwl_tns_pearson")),
            _float_or_none(correlations.get("hpwl_tns_spearman")),
        )
        if value is not None
    ]
    if not values:
        return {
            "mode": "diagnostic",
            "reason": "insufficient finite perturbation pairs",
        }
    max_abs = max(values)
    if max_abs > tiebreaker_threshold:
        return {
            "mode": "diagnostic",
            "reason": f"timing proxy is highly HPWL-correlated (max |corr|={max_abs:.3g})",
        }
    if max_abs >= gate_threshold:
        return {
            "mode": "tie_breaker",
            "reason": f"timing proxy is partially HPWL-correlated (max |corr|={max_abs:.3g})",
        }
    return {
        "mode": "gate",
        "reason": f"timing proxy has useful independence from HPWL (max |corr|={max_abs:.3g})",
    }


def apply_timing_proxy_policy(
    metrics: dict[str, Any],
    *,
    requested_mode: str,
    wns_regression_gate_ns: float,
    tns_regression_pct_gate: float,
    tns_regression_min_abs_ns: float = 0.05,
    wns_delta_min_abs_ns: float = 0.05,
    selection_weight: float = 0.05,
) -> None:
    """Apply optional OpenTimer parent-selection policy.

    Sign convention: positive WNS/TNS deltas are better than baseline, and
    negative WNS/TNS deltas indicate timing regression.
    """

    mode = requested_mode if requested_mode in TIMING_PROXY_MODES else "diagnostic"
    metrics["timing_proxy_mode"] = mode
    status = normalize_timing_proxy_status(metrics.get("timing_proxy_status"))
    metrics["timing_proxy_status"] = status
    metrics["timing_proxy_parent_signal"] = "not_used"
    source_budget_satisfied = bool(
        metrics.get("timing_proxy_source_placement_budget_satisfied", False)
    )
    if mode != "diagnostic" and not source_budget_satisfied:
        _block_timing_parent(
            metrics,
            "Timing proxy source placement did not complete the required full "
            "iteration budget; timing feedback cannot affect parent selection.",
        )
        metrics["timing_proxy_parent_signal"] = "source_placement_incomplete"
        return
    if mode == "diagnostic" or status in {"", "unavailable", "missing_timing_collateral"}:
        return
    if status not in {"success", "partial"}:
        _block_timing_parent(metrics, "Timing proxy failed for this candidate.")
        return
    wns_delta = _float_or_none(metrics.get("timing_proxy_wns_delta"))
    tns_delta = _float_or_none(metrics.get("timing_proxy_tns_delta"))
    tns_delta_pct = _float_or_none(metrics.get("timing_proxy_tns_delta_pct"))
    wns_bad = wns_delta is not None and wns_delta < -abs(wns_regression_gate_ns)
    tns_bad = (
        tns_delta is not None
        and tns_delta_pct is not None
        and tns_delta < -abs(tns_regression_min_abs_ns)
        and tns_delta_pct < -abs(tns_regression_pct_gate)
    )
    if mode == "gate" and (wns_bad or tns_bad):
        _block_timing_parent(
            metrics,
            "Timing proxy regressed versus baseline; keep as feedback but do not promote as parent.",
        )
        return
    if mode in {"tie_breaker", "gate"} and status in {"success", "partial"}:
        if metrics.get("timing_proxy_selection_applied"):
            metrics["timing_proxy_parent_signal"] = (
                "tie_breaker" if mode == "tie_breaker" else "soft_gate"
            )
            return
        quality_terms = []
        if wns_delta is not None:
            quality_terms.append(
                math.tanh(wns_delta / max(abs(wns_delta_min_abs_ns), 1e-12))
            )
        if tns_delta_pct is not None:
            quality_terms.append(
                math.tanh(tns_delta_pct / max(abs(tns_regression_pct_gate), 1e-12))
            )
        timing_quality = (
            sum(quality_terms) / len(quality_terms) if quality_terms else 0.0
        )
        admission_weight = 0.25 if mode == "tie_breaker" else 1.0
        adjustment = float(selection_weight) * admission_weight * timing_quality
        metrics["combined_score"] = max(
            0.0,
            float(metrics.get("combined_score") or 0.0) + adjustment,
        )
        metrics["timing_proxy_quality"] = timing_quality
        metrics["timing_proxy_selection_adjustment"] = adjustment
        metrics["timing_proxy_selection_weight"] = float(selection_weight)
        metrics["timing_proxy_selection_applied"] = True
        # Preserve the old field for artifact compatibility.
        metrics["timing_proxy_tie_breaker_bonus"] = adjustment
        metrics["timing_proxy_tie_breaker_applied"] = mode == "tie_breaker"
        metrics["timing_proxy_parent_signal"] = (
            "tie_breaker" if mode == "tie_breaker" else "soft_gate"
        )


def _block_timing_parent(metrics: dict[str, Any], lesson: str) -> None:
    metrics["parent_eligible"] = False
    metrics["negative_memory_only"] = True
    metrics["timing_proxy_parent_signal"] = "blocked"
    existing = str(metrics.get("feedback_lesson") or "").strip()
    metrics["feedback_lesson"] = f"{existing} {lesson}".strip()


def normalize_timing_proxy_status(value: Any) -> str:
    status = str(value or "unavailable").strip()
    if status in TIMING_PROXY_STATUSES:
        return status
    return "unavailable"


def timing_proxy_metrics_by_objective(
    summary: dict[str, Any],
    *,
    baseline_objective_id: str = "default",
) -> dict[str, dict[str, Any]]:
    comparison_csv = summary.get("comparison_csv")
    if not comparison_csv or not Path(str(comparison_csv)).is_file():
        return {}
    with Path(str(comparison_csv)).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("objective_id")), []).append(row)
    result: dict[str, dict[str, Any]] = {}
    for objective_id, objective_rows in grouped.items():
        if objective_id == baseline_objective_id:
            continue
        final_def_per_design = []
        for design_name in sorted(
            {str(row.get("design") or "unknown") for row in objective_rows}
        ):
            design_rows = [
                row
                for row in objective_rows
                if str(row.get("design") or "unknown") == design_name
            ]
            final_def_per_design.append(
                {
                    "design": design_name,
                    "hpwl_delta_pct": _mean_finite(
                        row.get("timing_proxy_hpwl_delta_pct") for row in design_rows
                    ),
                    "overflow_delta_pct": _mean_finite(
                        row.get("placement_overflow_delta_pct") for row in design_rows
                    ),
                    "cell_count": len(design_rows),
                }
            )
        result[objective_id] = {
            "timing_proxy_status": _aggregate_status(objective_rows),
            "timing_proxy_rc_model": TIMING_PROXY_RC_MODEL,
            "timing_proxy_hpwl": _mean_finite(row.get("timing_proxy_hpwl") for row in objective_rows),
            "placement_overflow": _mean_finite(
                row.get("placement_overflow") for row in objective_rows
            ),
            "placement_max_density": _mean_finite(
                row.get("placement_max_density") for row in objective_rows
            ),
            "timing_proxy_wns": _mean_finite(row.get("timing_proxy_wns") for row in objective_rows),
            "timing_proxy_tns": _mean_finite(row.get("timing_proxy_tns") for row in objective_rows),
            "timing_proxy_net_coverage": _mean_finite(
                row.get("timing_proxy_net_coverage") for row in objective_rows
            ),
            "timing_proxy_total_nets": _min_finite_int(
                row.get("timing_proxy_total_nets") for row in objective_rows
            ),
            "timing_proxy_skipped_nets": _min_finite_int(
                row.get("timing_proxy_skipped_nets") for row in objective_rows
            ),
            "timing_proxy_pin_incomplete_nets": _min_finite_int(
                row.get("timing_proxy_pin_incomplete_nets") for row in objective_rows
            ),
            "timing_proxy_hpwl_delta_pct": _mean_finite(
                row.get("timing_proxy_hpwl_delta_pct") for row in objective_rows
            ),
            "final_def_hpwl": _mean_finite(
                row.get("timing_proxy_hpwl") for row in objective_rows
            ),
            "final_def_overflow": _mean_finite(
                row.get("placement_overflow") for row in objective_rows
            ),
            "final_def_hpwl_delta_pct": _mean_finite(
                row.get("timing_proxy_hpwl_delta_pct") for row in objective_rows
            ),
            "final_def_overflow_delta_pct": _mean_finite(
                row.get("placement_overflow_delta_pct") for row in objective_rows
            ),
            "final_def_per_design_deltas": final_def_per_design,
            "timing_proxy_wns_delta": _mean_finite(
                row.get("timing_proxy_wns_delta") for row in objective_rows
            ),
            "timing_proxy_tns_delta_pct": _mean_finite(
                row.get("timing_proxy_tns_delta_pct") for row in objective_rows
            ),
            "timing_proxy_tns_delta": _mean_finite(
                row.get("timing_proxy_tns_delta") for row in objective_rows
            ),
            "timing_proxy_runtime_seconds": _mean_finite(
                row.get("timing_proxy_runtime_seconds") for row in objective_rows
            ),
            "timing_proxy_source_placement_iterations": _min_finite_int(
                row.get("source_placement_iterations") for row in objective_rows
            ),
            "timing_proxy_source_placement_completed_iterations": _min_finite_int(
                row.get("source_placement_completed_iterations")
                for row in objective_rows
            ),
            "timing_proxy_source_placement_budget_satisfied": all(
                _csv_bool(row.get("source_placement_budget_satisfied"))
                for row in objective_rows
            ),
        }
    return result


def placements_from_tier2_comparison(
    comparison_csv: str | Path,
    output: str | Path,
    *,
    objective_ids: set[str] | None = None,
    design_names: set[str] | None = None,
    include_default: bool = True,
    include_custom_default: bool = True,
    minimum_source_iterations: int = 1000,
) -> Path:
    selected_ids = set(objective_ids or set())
    selected_designs = set(design_names or set())
    if include_default:
        selected_ids.add("default")
    if include_custom_default:
        selected_ids.add("custom_default")
    rows = []
    with Path(comparison_csv).open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            artifact = str(row.get("output_artifact") or "")
            if row.get("status") != "success" or not artifact.lower().endswith(".def"):
                continue
            requested_iterations = int(float(row.get("requested_iterations") or 0))
            completed_iterations = int(float(row.get("completed_iterations") or 0))
            budget_satisfied = _csv_bool(row.get("iteration_budget_satisfied"))
            if (
                not budget_satisfied
                or requested_iterations < int(minimum_source_iterations)
                or completed_iterations < requested_iterations
            ):
                continue
            if selected_ids and str(row.get("objective_id")) not in selected_ids:
                continue
            if selected_designs and str(row.get("design")) not in selected_designs:
                continue
            rows.append(
                {
                    "design": row["design"],
                    "objective_id": row["objective_id"],
                    "seed": int(row["seed"]),
                    "def_path": artifact,
                    "source": "tier2_search_for_timing_proxy",
                    "requested_iterations": requested_iterations,
                    "completed_iterations": completed_iterations,
                    "iteration_budget_satisfied": budget_satisfied,
                }
            )
    return _write_json({"placements": rows}, Path(output))


def build_timing_proxy_comparison_rows(
    results: list[TimingProxyResult],
    *,
    baseline_objective_id: str = "default",
) -> list[dict[str, Any]]:
    by_key = {(item.design, item.seed, item.objective_id, item.variant): item for item in results}
    rows = []
    for result in results:
        baseline = by_key.get((result.design, result.seed, baseline_objective_id, result.variant))
        if baseline is None:
            baseline = _baseline_for_design_seed(results, result.design, result.seed, baseline_objective_id)
        row = _result_row(result)
        row["baseline_objective_id"] = baseline_objective_id
        row["timing_proxy_hpwl_delta_pct"] = _delta_pct(
            result.timing_proxy_hpwl,
            baseline.timing_proxy_hpwl if baseline else None,
        )
        row["placement_overflow_delta_pct"] = _delta_pct(
            result.placement_overflow,
            baseline.placement_overflow if baseline else None,
        )
        row["placement_max_density_delta_pct"] = _delta_pct(
            result.placement_max_density,
            baseline.placement_max_density if baseline else None,
        )
        row["timing_proxy_wns_delta"] = _delta_abs(
            result.timing_proxy_wns,
            baseline.timing_proxy_wns if baseline else None,
        )
        row["timing_proxy_tns_delta_pct"] = _delta_pct(
            result.timing_proxy_tns,
            baseline.timing_proxy_tns if baseline else None,
        )
        row["timing_proxy_tns_delta"] = _delta_abs(
            result.timing_proxy_tns,
            baseline.timing_proxy_tns if baseline else None,
        )
        rows.append(row)
    return rows


def write_timing_proxy_metrics_csv(results: list[TimingProxyResult], output: str | Path) -> Path:
    return _write_csv([_result_row(result) for result in results], output, TIMING_PROXY_COLUMNS)


def write_timing_proxy_comparison_csv(rows: list[dict[str, Any]], output: str | Path) -> Path:
    return _write_csv(
        rows,
        output,
        TIMING_PROXY_COLUMNS
        + [
            "baseline_objective_id",
            "timing_proxy_hpwl_delta_pct",
            "placement_overflow_delta_pct",
            "placement_max_density_delta_pct",
            "timing_proxy_wns_delta",
            "timing_proxy_tns_delta_pct",
            "timing_proxy_tns_delta",
        ],
    )


def build_timing_proxy_report(*, panel: TimingProxyPanel, results: list[TimingProxyResult]) -> str:
    lines = [
        "# Timing Proxy Report",
        "",
        f"- RC model: `{TIMING_PROXY_RC_MODEL}`",
        f"- Designs: {len(panel.designs)}",
        f"- Cells: {len(results)}",
        f"- Successes: {sum(1 for item in results if item.status == 'success')}",
        f"- Failures/skips: {sum(1 for item in results if item.status != 'success')}",
        "",
        "OpenTimer is used through DREAMPlace's placement-based FLUTE/Elmore RC path. These metrics are proxy feedback, not routed signoff.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def build_timing_proxy_audit_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    correlations = summary.get("correlations", {})
    lines = [
        "# Timing Proxy Orthogonality Audit",
        "",
        f"- Design: `{summary.get('design')}`",
        f"- RC model: `{summary.get('timing_proxy_rc_model')}`",
        f"- Perturbation rows: {len(rows)}",
        f"- Recommended mode: `{summary.get('recommended_mode')}`",
        f"- Reason: {summary.get('recommendation_reason')}",
        "",
        "## Correlations",
        f"- HPWL vs WNS Pearson: {correlations.get('hpwl_wns_pearson')}",
        f"- HPWL vs WNS Spearman: {correlations.get('hpwl_wns_spearman')}",
        f"- HPWL vs TNS Pearson: {correlations.get('hpwl_tns_pearson')}",
        f"- HPWL vs TNS Spearman: {correlations.get('hpwl_tns_spearman')}",
    ]
    return "\n".join(lines).rstrip() + "\n"


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson(_rank(xs), _rank(ys))


def _run_one_timing_proxy_cell(
    *,
    panel: TimingProxyPanel,
    design: TimingProxyDesign | None,
    placement: TimingProxyPlacement,
    dreamplace_root: Path,
    run_root: Path,
    resume: bool,
    retry_failed: bool,
) -> TimingProxyResult:
    variant = placement.variant or "original"
    cell_dir = (
        run_root
        / _safe_name(placement.design)
        / _safe_name(placement.objective_id)
        / f"seed_{placement.seed}"
        / _safe_name(variant)
    )
    summary_path = cell_dir / "timing_proxy_result.json"
    if resume and summary_path.is_file():
        previous_payload = _load_json(summary_path)
        previous_payload.setdefault("source_placement_iterations", 0)
        previous_payload.setdefault("source_placement_completed_iterations", 0)
        previous_payload.setdefault("source_placement_budget_satisfied", False)
        previous = TimingProxyResult(**previous_payload)
        if previous.status == "success" or not retry_failed:
            return previous
    start = time.time()
    if design is None:
        return _failed_result(placement, cell_dir, "missing_timing_collateral", "design not in timing proxy panel", variant, start)
    missing = _missing_collateral(design, panel)
    if missing:
        return _failed_result(
            placement,
            cell_dir,
            "missing_timing_collateral",
            "missing timing collateral: " + ", ".join(missing),
            variant,
            start,
        )
    def_path = _resolve_existing_path(placement.def_path, base=Path.cwd())
    if not def_path.is_file():
        return _failed_result(
            placement,
            cell_dir,
            "missing_placement_def",
            f"DEF does not exist: {placement.def_path}",
            variant,
            start,
        )
    try:
        config_path = _make_timing_proxy_config(
            design=design,
            panel=panel,
            placement_def=def_path,
            run_dir=cell_dir,
        )
    except Exception as exc:
        return _failed_result(placement, cell_dir, "timer_init", str(exc), variant, start)
    fixed_driver = Path(__file__).resolve().parents[2] / "scripts" / "dreamplace_opentimer_fixed_def.py"
    run = run_dreamplace(
        dreamplace_root=dreamplace_root,
        config_path=config_path,
        run_name=f"timing_proxy_{placement.design}_{placement.objective_id}_s{placement.seed}_{variant}",
        run_dir=cell_dir,
        timeout_seconds=panel.timeout_seconds,
        driver=fixed_driver if panel.fixed_position else "dreamplace/Placer.py",
    )
    write_run_summary(run, cell_dir / "dreamplace_run.json")
    last = run.metrics.get("last", {}) if run.metrics else {}
    failure_stage = _timing_failure_stage(run.returncode, run.metrics, last)
    net_coverage = _float_or_none(run.metrics.get("timing_net_coverage"))
    if (
        failure_stage is None
        and net_coverage is not None
        and net_coverage < panel.min_net_coverage
    ):
        failure_stage = "timing_coverage"
        status = "partial"
    else:
        status = "success" if failure_stage is None else "failed"
    result = TimingProxyResult(
        design=placement.design,
        objective_id=placement.objective_id,
        seed=placement.seed,
        variant=variant,
        status=status,
        failure_stage=failure_stage,
        timing_proxy_rc_model=TIMING_PROXY_RC_MODEL,
        timing_proxy_hpwl=_float_or_none(last.get("hpwl")),
        placement_overflow=_float_or_none(last.get("overflow")),
        placement_max_density=_float_or_none(last.get("max_density")),
        timing_proxy_total_nets=_int_or_none(run.metrics.get("timing_total_nets")),
        timing_proxy_skipped_nets=_int_or_none(run.metrics.get("timing_skipped_nets")),
        timing_proxy_pin_incomplete_nets=_int_or_none(
            run.metrics.get("timing_pin_incomplete_nets")
        ),
        timing_proxy_net_coverage=net_coverage,
        timing_proxy_wns=_float_or_none(last.get("wns")),
        timing_proxy_tns=_float_or_none(last.get("tns")),
        timing_proxy_num_violating_paths=_violating_path_count(run.metrics),
        timing_proxy_runtime_seconds=time.time() - start,
        def_path=str(def_path),
        run_dir=str(cell_dir),
        config_path=str(config_path),
        log_path=run.log_path,
        returncode=run.returncode,
        source_placement_iterations=placement.requested_iterations,
        source_placement_completed_iterations=placement.completed_iterations,
        source_placement_budget_satisfied=placement.iteration_budget_satisfied,
        artifacts={
            "dreamplace_run": str(cell_dir / "dreamplace_run.json"),
            **(
                {
                    "def_timing_netlist": str(
                        cell_dir / "def_timing_netlist" / "placement_flat.v"
                    ),
                    "def_timing_netlist_manifest": str(
                        cell_dir / "def_timing_netlist" / "manifest.json"
                    ),
                }
                if panel.derive_verilog_from_def
                else {}
            ),
            "def_normalization": str(
                cell_dir / "timing_collateral" / "def_normalization.json"
            ),
            "verilog_normalization": str(
                cell_dir / "timing_collateral" / "verilog_normalization.json"
            ),
            "sdc_normalization": str(cell_dir / "timing_collateral" / "sdc_normalization.json"),
            "source": placement.source,
            "error": run.metrics.get("assertion") or run.metrics.get("missing_input_file"),
        },
    )
    _write_json(asdict(result), summary_path)
    return result


def _make_timing_proxy_config(
    *,
    design: TimingProxyDesign,
    panel: TimingProxyPanel,
    placement_def: Path,
    run_dir: Path,
) -> Path:
    base_config = Path(design.base_config)
    config = json.loads(base_config.read_text(encoding="utf-8-sig"))
    result_dir = run_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    prepared_def, instance_name_map = _prepare_timing_def(
        def_file=placement_def,
        run_dir=run_dir,
        scalarize_vector_nets=panel.scalarize_vector_nets,
    )
    config["def_input"] = str(prepared_def)
    config["result_dir"] = str(result_dir)
    config["timer_engine"] = "opentimer"
    config["timing_opt_flag"] = 1
    # A timing-proxy run evaluates the supplied placement.  ChipBench base
    # configs commonly randomize movable cells before global placement, which
    # would erase method/seed differences and make every fixed-DEF comparison
    # converge to the same proxy result.
    config["random_center_init_flag"] = 0
    config["gp_noise_ratio"] = 0.0
    config["timing_proxy_position_jitter"] = float(panel.position_jitter)
    if not _positive_float(config.get("wire_resistance_per_micron")):
        config["wire_resistance_per_micron"] = 2.535
    if not _positive_float(config.get("wire_capacitance_per_micron")):
        config["wire_capacitance_per_micron"] = 1.6e-16
    # DREAMPlace reports OpenTimer WNS/TNS either during late timing-driven
    # global placement iterations (>500) or in the post-legalization STA block.
    # Timing-proxy cells intentionally use very short placement runs, so enable
    # legalization here to exercise the cheap final STA path.
    config["legalize_flag"] = int(panel.legalize)
    if not panel.legalize:
        config["detailed_place_flag"] = 0
    config["num_threads"] = int(panel.threads)
    config["plot_flag"] = 0
    if panel.gpu is not None:
        config["gpu"] = int(panel.gpu)
    if not config.get("global_place_stages"):
        config["global_place_stages"] = [
            {
                "iteration": int(panel.iterations),
                "learning_rate": 0.01,
                "optimizer": "nesterov",
                "wirelength": "weighted_average",
            }
        ]
    for stage in config.get("global_place_stages", []):
        stage["iteration"] = int(panel.iterations)
        if panel.learning_rate is not None:
            stage["learning_rate"] = float(panel.learning_rate)
    if design.lib:
        config["lib_input"] = str(_prepare_timing_lib(lib=design.lib, run_dir=run_dir))
    source_verilog = None
    if panel.derive_verilog_from_def:
        source_verilog = _write_flat_verilog_from_def(
            def_file=placement_def,
            run_dir=run_dir,
            top_module=_timing_top_module(design, config),
        )
    elif design.verilog:
        source_verilog = Path(design.verilog)
    prepared_verilog = (
        _prepare_timing_verilog(
            verilog=source_verilog,
            run_dir=run_dir,
            instance_name_map=instance_name_map,
            top_module=_timing_top_module(design, config),
            scalarize_vector_nets=(
                panel.scalarize_vector_nets and not panel.derive_verilog_from_def
            ),
        )
        if source_verilog is not None
        else None
    )
    if design.sdc:
        config["sdc_input"] = str(
            _prepare_timing_sdc(
                sdc=Path(design.sdc),
                verilog=prepared_verilog,
                run_dir=run_dir,
            )
        )
    if prepared_verilog is not None:
        config["verilog_input"] = str(prepared_verilog)
    if design.lef:
        config["lef_input"] = design.lef
    output = run_dir / "timing_proxy_config.json"
    output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _write_flat_verilog_from_def(
    *,
    def_file: Path,
    run_dir: Path,
    top_module: str | None,
) -> Path:
    """Build a flat gate-level timing netlist from the evaluated placement.

    A routed or placed DEF already contains the exact component and net names
    used by DREAMPlace.  Reconstructing the named-port Verilog view from that
    DEF keeps OpenTimer's timing graph aligned with every evaluated candidate,
    including net names introduced or rewritten by the physical-design flow.
    """

    text = def_file.read_text(encoding="utf-8", errors="replace")
    component_entries = _def_section_entries(text, "COMPONENTS")
    pin_entries = _def_section_entries(text, "PINS")
    net_entries = _def_section_entries(text, "NETS")
    components: list[tuple[str, str]] = []
    component_names: set[str] = set()
    for entry in component_entries:
        match = re.match(r"^\s*-\s+(\S+)\s+(\S+)", entry)
        if not match:
            continue
        instance = _canonical_def_name(match.group(1))
        master = _canonical_def_name(match.group(2))
        components.append((instance, master))
        component_names.add(instance)

    ports: list[tuple[str, str, str]] = []
    for entry in pin_entries:
        name_match = re.match(r"^\s*-\s+(\S+)", entry)
        net_match = re.search(r"\+\s+NET\s+(\S+)", entry)
        direction_match = re.search(r"\+\s+DIRECTION\s+(INPUT|OUTPUT|INOUT)", entry)
        if not name_match or not direction_match:
            continue
        name = _canonical_def_name(name_match.group(1))
        net = _canonical_def_name(net_match.group(1)) if net_match else name
        ports.append((name, direction_match.group(1).lower(), net))

    instance_pins: dict[str, list[tuple[str, str]]] = {
        instance: [] for instance, _ in components
    }
    net_names: list[str] = []
    seen_nets: set[str] = set()
    for entry in net_entries:
        name_match = re.match(r"^\s*-\s+(\S+)", entry)
        if not name_match:
            continue
        net = _canonical_def_name(name_match.group(1))
        if net not in seen_nets:
            seen_nets.add(net)
            net_names.append(net)
        for raw_instance, raw_pin in re.findall(r"\(\s*(\S+)\s+(\S+)\s*\)", entry):
            instance = _canonical_def_name(raw_instance)
            if instance == "PIN" or instance not in component_names:
                continue
            instance_pins[instance].append((_canonical_def_name(raw_pin), net))

    if not components or not net_names:
        raise ValueError(
            f"DEF timing-netlist export found no components or nets in {def_file}"
        )

    module = _verilog_identifier(top_module or def_file.stem)
    lines = ["// Generated from the evaluated placement DEF by CoEvoP&R."]
    lines.append(f"module {module} (")
    for index, (name, _, _) in enumerate(ports):
        comma = "," if index + 1 < len(ports) else ""
        lines.append(f"  {_verilog_identifier(name)}{comma}")
    lines.append(");")
    for name, direction, _ in ports:
        lines.append(f"  {direction} {_verilog_identifier(name)};")

    port_names = {name for name, _, _ in ports}
    for net in net_names:
        if net not in port_names:
            lines.append(f"  wire {_verilog_identifier(net)};")
    for name, direction, net in ports:
        if name == net:
            continue
        if direction == "output":
            lines.append(
                f"  assign {_verilog_identifier(name)} = {_verilog_identifier(net)};"
            )
        else:
            lines.append(
                f"  assign {_verilog_identifier(net)} = {_verilog_identifier(name)};"
            )

    for instance, master in components:
        connections = instance_pins.get(instance, [])
        lines.append(f"  {_verilog_identifier(master)} {_verilog_identifier(instance)} (")
        for index, (pin, net) in enumerate(connections):
            comma = "," if index + 1 < len(connections) else ""
            lines.append(
                f"    .{_verilog_identifier(pin)}({_verilog_identifier(net)}){comma}"
            )
        lines.append("  );")
    lines.append("endmodule")
    lines.append("")

    output_dir = run_dir.resolve() / "def_timing_netlist"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "placement_flat.v"
    output.write_text("\n".join(lines), encoding="utf-8")
    _write_json(
        {
            "source_def": str(def_file),
            "output_verilog": str(output),
            "top_module": top_module,
            "component_count": len(components),
            "port_count": len(ports),
            "net_count": len(net_names),
            "connected_instance_pin_count": sum(
                len(value) for value in instance_pins.values()
            ),
        },
        output_dir / "manifest.json",
    )
    return output


def _def_section_entries(text: str, section: str) -> list[str]:
    """Return semicolon-terminated records from one DEF section."""

    entries: list[str] = []
    active = False
    current: list[str] = []
    end_marker = f"END {section}"
    for line in text.splitlines():
        stripped = line.strip()
        if not active:
            if stripped.startswith(f"{section} "):
                active = True
            continue
        if stripped.startswith(end_marker):
            break
        if stripped.startswith("-"):
            current = [stripped]
        elif current:
            current.append(stripped)
        if current and stripped.endswith(";"):
            entries.append(" ".join(current))
            current = []
    return entries


def _verilog_identifier(name: str) -> str:
    canonical = _canonical_def_name(name)
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", canonical):
        return canonical
    return f"\\{canonical} "


def _prepare_timing_def(
    *,
    def_file: Path,
    run_dir: Path,
    scalarize_vector_nets: bool = False,
) -> tuple[Path, dict[str, str]]:
    """Create a run-local DEF with OpenTimer-safe component names.

    Yosys/ORFS can emit escaped component names such as
    ``dpath.a_reg.out\\[0\\]$_DFFE_PP_``. DREAMPlace can read those names from
    DEF, but the installed OpenTimer binding normalizes escaped Verilog
    instance names too aggressively and collapses bit-indexed registers.  The
    timing proxy therefore sanitizes DEF component names and the matching
    Verilog instance names with the same deterministic map.
    """

    prepared_dir = run_dir.resolve() / "timing_collateral"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    output = prepared_dir / def_file.name
    text = def_file.read_text(encoding="utf-8", errors="replace")
    component_tokens = _def_component_tokens(text)
    net_tokens = _def_net_tokens(text)
    name_map: dict[str, str] = {}
    used: set[str] = set()
    for token in component_tokens:
        canonical = _canonical_def_name(token)
        safe = _open_timer_safe_identifier(canonical, used)
        if safe != token or canonical != token:
            name_map[canonical] = safe
    for token in net_tokens:
        canonical = _canonical_def_name(token)
        # Ordinary DEF/Verilog identifiers already agree and must retain their
        # names. Verilog vector bits such as ``data[0]`` are also native
        # OpenTimer identifiers: renaming them in DEF while leaving the vector
        # declaration intact in Verilog disconnects every affected timing net.
        # Only escaped or otherwise punctuation-bearing nets need the shared
        # OpenTimer-safe mapping used by _prepare_timing_verilog().
        if canonical == token and (
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", canonical)
            or (
                not scalarize_vector_nets
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*(?:\[\d+\])+", canonical)
            )
        ):
            continue
        safe = _open_timer_safe_identifier(f"v_{canonical}", used)
        if safe != token or canonical != token:
            name_map[canonical] = safe
    token_map: dict[str, str] = {}
    for token in component_tokens + net_tokens:
        canonical = _canonical_def_name(token)
        safe = name_map.get(canonical)
        if safe:
            token_map[token] = safe
    sanitized = _rewrite_def_tokens(text, token_map)
    output.write_text(sanitized, encoding="utf-8")
    _write_json(
        {
            "source_def": str(def_file),
            "output_def": str(output),
            "component_count": len(component_tokens),
            "net_count": len(net_tokens),
            "renamed_identifier_count": len(name_map),
            "scalarize_vector_nets": scalarize_vector_nets,
            "name_map": name_map,
        },
        prepared_dir / "def_normalization.json",
    )
    return output, name_map


def _rewrite_def_tokens(text: str, token_map: dict[str, str]) -> str:
    """Rewrite exact DEF tokens in one pass.

    Component names can also appear in NETS records, so the rewrite must cover
    the whole DEF.  Repeated full-string replacement is quadratic on large
    designs such as swerv_wrapper; a token pass keeps timing collateral
    preparation linear in the DEF size.
    """

    if not token_map:
        return text

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        return token_map.get(token, token)

    return re.sub(r"\S+", replace, text)


def _prepare_timing_sdc(*, sdc: Path, verilog: Path | None, run_dir: Path) -> Path:
    """Copy SDC into the run directory and repair unambiguous scalar port names.

    Some generated ChipBench timing collateral names scalar ports as ``foo_0`` in
    SDC while the Verilog top-level port is ``foo``. OpenTimer then silently loses
    IO timing constraints, which can leave WNS/TNS as NaN. We only rewrite
    ``foo_0`` to ``foo`` when ``foo_0`` is absent and ``foo`` is present as a
    Verilog port; all edits are run-local and recorded next to the copied SDC.
    """

    prepared_dir = run_dir.resolve() / "timing_collateral"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    output = prepared_dir / sdc.name
    text = sdc.read_text(encoding="utf-8", errors="replace")
    replacements: dict[str, str] = {}
    added_input_transitions: list[str] = []
    added_output_loads: list[str] = []
    simplified_sdc = False
    if verilog is not None and verilog.is_file():
        port_directions = _verilog_top_port_directions(verilog)
        if _sdc_needs_plain_opentimer_subset(text):
            text = _plain_opentimer_sdc(text, port_directions)
            simplified_sdc = True
        ports = set(port_directions)
        sdc_ports = set(_sdc_get_ports(text))
        for port in sorted(sdc_ports):
            if port in ports or not port.endswith("_0"):
                continue
            scalar = port[:-2]
            if scalar in ports and scalar not in sdc_ports:
                replacements[port] = scalar
        for old, new in replacements.items():
            text = _replace_sdc_port(text, old, new)
        if "set_input_transition" not in text:
            input_ports = sorted(
                name
                for name, direction in port_directions.items()
                if direction == "input" and name != ""
            )
            added_input_transitions = input_ports
            text = (
                text.rstrip()
                + "\n\n# CoEvoP&R timing-proxy interface defaults\n"
                + "\n".join(f"set_input_transition 0.05 [get_ports {name}]" for name in input_ports)
                + "\n"
            )
        if "set_load" not in text:
            output_ports = sorted(
                name
                for name, direction in port_directions.items()
                if direction == "output" and name != ""
            )
            added_output_loads = output_ports
            text = (
                text.rstrip()
                + "\n"
                + "\n".join(f"set_load -pin_load 4 [get_ports {name}]" for name in output_ports)
                + "\n"
            )
    output.write_text(text, encoding="utf-8")
    _write_json(
        {
            "source_sdc": str(sdc),
            "source_verilog": str(verilog) if verilog else None,
            "output_sdc": str(output),
            "replacements": replacements,
            "added_input_transition_count": len(added_input_transitions),
            "added_output_load_count": len(added_output_loads),
            "input_transition_value": 0.05 if added_input_transitions else None,
            "output_load_pin_load": 4.0 if added_output_loads else None,
            "simplified_sdc": simplified_sdc,
        },
        prepared_dir / "sdc_normalization.json",
    )
    return output


def _def_component_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    in_components = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("COMPONENTS "):
            in_components = True
            continue
        if in_components and stripped.startswith("END COMPONENTS"):
            break
        if not in_components:
            continue
        match = re.match(r"^-\s+(\S+)\s+\S+", stripped)
        if match:
            tokens.append(match.group(1))
    return tokens


def _def_net_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    in_nets = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("NETS "):
            in_nets = True
            continue
        if in_nets and stripped.startswith("END NETS"):
            break
        if not in_nets:
            continue
        match = re.match(r"^-\s+(\S+)", stripped)
        if match:
            tokens.append(match.group(1))
    return tokens


def _canonical_def_name(name: str) -> str:
    return re.sub(r"\\(.)", r"\1", name)


def _open_timer_safe_identifier(name: str, used: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_]", "_", name)
    base = re.sub(r"_+", "_", base)
    if not base:
        base = "u"
    if base[0].isdigit():
        base = f"u_{base}"
    candidate = base
    index = 1
    while candidate in used:
        index += 1
        candidate = f"{base}_{index}"
    used.add(candidate)
    return candidate


def _prepare_timing_verilog(
    *,
    verilog: Path,
    run_dir: Path,
    instance_name_map: dict[str, str] | None = None,
    top_module: str | None = None,
    scalarize_vector_nets: bool = False,
) -> Path:
    prepared_dir = run_dir.resolve() / "timing_collateral"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    output = prepared_dir / verilog.name
    source_verilog = verilog
    vector_scalarization = None
    if scalarize_vector_nets:
        source_verilog, vector_scalarization = _scalarize_verilog_vector_nets(
            verilog=verilog,
            prepared_dir=prepared_dir,
        )
    text = source_verilog.read_text(encoding="utf-8", errors="replace")
    attribute_matches = re.findall(r"\(\*.*?\*\)", text, flags=re.DOTALL)
    sanitized = re.sub(r"\(\*.*?\*\)", "", text, flags=re.DOTALL)
    top_module_requested = top_module
    top_module_found = None
    top_module_moved = False
    if top_module_requested:
        sanitized, top_module_found, top_module_moved = _move_verilog_top_module_first(
            sanitized,
            top_module_requested,
        )
    instance_name_map = instance_name_map or {}
    used_instance_names = set(instance_name_map.values())
    escaped_instance_count = 0

    def replace_escaped_instance(match: re.Match[str]) -> str:
        nonlocal escaped_instance_count
        escaped_instance_count += 1
        raw_name = match.group("name")
        safe = instance_name_map.get(raw_name)
        if safe is None:
            safe = _open_timer_safe_identifier(raw_name, used_instance_names)
            instance_name_map[raw_name] = safe
        return f"{match.group('prefix')}{safe}{match.group('space')}"

    sanitized = re.sub(
        r"(?P<prefix>^\s*[A-Za-z_][A-Za-z0-9_$]*\s+)\\(?P<name>\S+)(?P<space>\s+)(?=\()",
        replace_escaped_instance,
        sanitized,
        flags=re.MULTILINE,
    )
    # OpenTimer's Verilog reader truncates escaped identifiers at punctuation.
    # For example, ``\X0[0]`` and ``\X0[1]`` can both be inserted as ``X0``.
    # Sanitize every remaining escaped identifier, not only instance names, so
    # vector ports and hierarchical nets remain distinct in the timing graph.
    escaped_identifier_map: dict[str, str] = {}
    used_verilog_names = set(used_instance_names)

    def replace_escaped_identifier(match: re.Match[str]) -> str:
        raw_name = match.group("name")
        safe = instance_name_map.get(raw_name)
        if safe is None:
            safe = escaped_identifier_map.get(raw_name)
        if safe is None:
            safe = _open_timer_safe_identifier(f"v_{raw_name}", used_verilog_names)
            escaped_identifier_map[raw_name] = safe
        return f"{safe}{match.group('space')}"

    sanitized = re.sub(
        r"\\(?P<name>\S+)(?P<space>\s)",
        replace_escaped_identifier,
        sanitized,
    )
    output.write_text(sanitized, encoding="utf-8")
    _write_json(
        {
            "source_verilog": str(verilog),
            "normalized_source_verilog": str(source_verilog),
            "output_verilog": str(output),
            "vector_scalarization": vector_scalarization,
            "removed_yosys_attribute_count": len(attribute_matches),
            "renamed_escaped_instance_count": escaped_instance_count,
            "instance_name_map": instance_name_map,
            "renamed_escaped_identifier_count": len(escaped_identifier_map),
            "escaped_identifier_map": escaped_identifier_map,
            "top_module_requested": top_module_requested,
            "top_module_found": top_module_found,
            "top_module_moved_first": top_module_moved,
        },
        prepared_dir / "verilog_normalization.json",
    )
    return output


def _scalarize_verilog_vector_nets(
    *,
    verilog: Path,
    prepared_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Use Yosys to turn vector bits into distinct scalar timing nets."""

    yosys = shutil.which("yosys")
    if yosys is None:
        raise RuntimeError(
            "scalarize_vector_nets=true requires yosys on PATH; "
            "OpenTimer otherwise collapses vector bits into one timing net"
        )
    output = prepared_dir / f"{verilog.stem}.splitnets.v"
    script = prepared_dir / "split_vector_nets.ys"
    source_arg = json.dumps(verilog.resolve().as_posix())
    output_arg = json.dumps(output.resolve().as_posix())
    script.write_text(
        "\n".join(
            [
                f"read_verilog {source_arg}",
                "splitnets -ports",
                # The physical DEF and synthesized netlist share internal net
                # names.  Preserve them while scalarizing vector ports so the
                # timing graph remains aligned with the placement database.
                f"write_verilog -noattr -norename {output_arg}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [yosys, "-q", "-s", str(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not output.is_file():
        detail = (completed.stderr or completed.stdout or "unknown yosys error").strip()
        raise RuntimeError(f"Yosys vector-net scalarization failed: {detail}")
    return output, {
        "enabled": True,
        "tool": yosys,
        "source": str(verilog),
        "output": str(output),
        "script": str(script),
        "stderr": completed.stderr.strip() or None,
    }


def _timing_top_module(design: TimingProxyDesign, config: dict[str, Any]) -> str | None:
    if design.top_module:
        return design.top_module
    audit = config.get("asap7_generation_audit")
    if isinstance(audit, dict):
        variables = audit.get("variables")
        if isinstance(variables, dict):
            design_name = variables.get("DESIGN_NAME")
            if isinstance(design_name, str) and design_name.strip():
                return design_name.strip()
        design_name = audit.get("design_name")
        if isinstance(design_name, str) and design_name.strip():
            return design_name.strip()
    return design.name


def _move_verilog_top_module_first(text: str, top_module: str) -> tuple[str, str | None, bool]:
    modules = list(_iter_verilog_module_blocks(text))
    if not modules:
        return text, None, False
    requested = _verilog_module_name_key(top_module)
    selected_index = None
    for index, module in enumerate(modules):
        if _verilog_module_name_key(module["name"]) == requested:
            selected_index = index
            break
    if selected_index is None:
        return text, None, False
    if selected_index == 0:
        return text, modules[selected_index]["name"], False
    prefix = text[: modules[0]["start"]]
    selected = modules[selected_index]
    ordered_blocks = [selected["text"]] + [
        module["text"] for index, module in enumerate(modules) if index != selected_index
    ]
    return prefix + "\n\n".join(block.strip() for block in ordered_blocks) + "\n", selected["name"], True


def _iter_verilog_module_blocks(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    module_start = re.compile(r"^\s*module\s+(?P<name>\\\S+|[A-Za-z_][A-Za-z0-9_$]*)\b")
    offset = 0
    current_name: str | None = None
    current_start: int | None = None
    current_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        if current_name is None:
            match = module_start.match(line)
            if match:
                current_name = match.group("name")
                current_start = offset
                current_lines = [line]
                if re.search(r"\bendmodule\b", line):
                    block_text = "".join(current_lines)
                    blocks.append(
                        {
                            "name": current_name,
                            "start": current_start,
                            "end": offset + len(line),
                            "text": block_text,
                        }
                    )
                    current_name = None
                    current_start = None
                    current_lines = []
        else:
            current_lines.append(line)
            if re.search(r"\bendmodule\b", line):
                block_text = "".join(current_lines)
                blocks.append(
                    {
                        "name": current_name,
                        "start": current_start,
                        "end": offset + len(line),
                        "text": block_text,
                    }
                )
                current_name = None
                current_start = None
                current_lines = []
        offset += len(line)
    return blocks


def _verilog_module_name_key(name: str) -> str:
    return name.strip().lstrip("\\")


def _sdc_needs_plain_opentimer_subset(text: str) -> bool:
    unsupported_markers = (
        "lsearch",
        "all_inputs",
        "all_outputs",
        "[expr",
        "$",
        "current_design",
        "[*]",
    )
    return any(marker in text for marker in unsupported_markers)


def _plain_opentimer_sdc(text: str, port_directions: dict[str, str]) -> str:
    explicit_clock = _first_sdc_clock(text)
    clk_name = _sdc_variable(
        text,
        "clk_name",
        explicit_clock[0] if explicit_clock else "CLK",
    )
    clk_port = _sdc_variable(
        text,
        "clk_port_name",
        explicit_clock[1] if explicit_clock else _infer_clock_port(port_directions),
    )
    clk_period = _sdc_float_variable(
        text,
        "clk_period",
        explicit_clock[2] if explicit_clock else 1.0,
    )
    clk_io_pct = _sdc_float_variable(text, "clk_io_pct", 0.2)
    io_delay = clk_period * clk_io_pct
    input_ports = sorted(
        name
        for name, direction in port_directions.items()
        if direction == "input" and name and name != clk_port
    )
    output_ports = sorted(
        name for name, direction in port_directions.items() if direction == "output" and name
    )
    lines = [
        "# CoEvoP&R generated OpenTimer-compatible SDC",
        f"create_clock [get_ports {{{clk_port}}}] -name {clk_name} -period {clk_period:g}",
        f"set_input_delay -clock {clk_name} 0 [get_ports {{{clk_port}}}]",
    ]
    lines.extend(
        f"set_input_delay -clock {clk_name} {io_delay:g} [get_ports {{{name}}}]"
        for name in input_ports
    )
    lines.extend(
        f"set_output_delay -clock {clk_name} {io_delay:g} [get_ports {{{name}}}]"
        for name in output_ports
    )
    return "\n".join(lines) + "\n"


def _first_sdc_clock(text: str) -> tuple[str, str, float] | None:
    for line in text.splitlines():
        if "create_clock" not in line or "get_ports" not in line:
            continue
        name_match = re.search(r"-name\s+([^\s\]]+)", line)
        period_match = re.search(r"-period\s+([0-9.eE+-]+)", line)
        port_match = re.search(r"get_ports\s+\{?([^\s\}\]]+)", line)
        if not period_match or not port_match:
            continue
        try:
            period = float(period_match.group(1))
        except ValueError:
            continue
        return (
            name_match.group(1) if name_match else "CLK",
            port_match.group(1),
            period,
        )
    return None


def _sdc_variable(text: str, name: str, default: str) -> str:
    match = re.search(rf"^\s*set\s+{re.escape(name)}\s+([^\s#]+)", text, flags=re.MULTILINE)
    return match.group(1) if match else default


def _sdc_float_variable(text: str, name: str, default: float) -> float:
    raw = _sdc_variable(text, name, str(default))
    try:
        return float(raw)
    except ValueError:
        return default


def _infer_clock_port(port_directions: dict[str, str]) -> str:
    inputs = [name for name, direction in port_directions.items() if direction == "input"]
    for name in inputs:
        lowered = name.lower()
        if "clk" in lowered or "clock" in lowered:
            return name
    return inputs[0] if inputs else "clk"


def _positive_float(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _prepare_timing_lib(*, lib: str | list[str], run_dir: Path) -> Path:
    prepared_dir = run_dir / "timing_collateral"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(lib, str):
        return Path(lib)
    output = prepared_dir / "merged.lib"
    first_text = Path(lib[0]).read_text(encoding="utf-8", errors="replace")
    first_body, first_close = _liberty_without_final_close(first_text)
    with output.open("w", encoding="utf-8", errors="replace") as out:
        out.write(first_body.rstrip())
        for item in lib[1:]:
            source = Path(item)
            out.write(f"\n\n/* BEGIN merged inner Liberty content from {source} */\n")
            out.write(_liberty_inner_text(source.read_text(encoding="utf-8", errors="replace")).strip())
            out.write(f"\n/* END merged inner Liberty content from {source} */\n")
        out.write("\n")
        out.write(first_close)
        out.write("\n")
    (prepared_dir / "lib_merge_manifest.json").write_text(
        json.dumps({"sources": [str(item) for item in lib], "output": str(output)}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return output


def _liberty_without_final_close(text: str) -> tuple[str, str]:
    close_index = text.rfind("}")
    if close_index < 0:
        raise ValueError("Liberty file has no closing brace")
    return text[:close_index], text[close_index:]


def _liberty_inner_text(text: str) -> str:
    match = re.search(r"library\s*\([^)]*\)\s*\{", text)
    close_index = text.rfind("}")
    if not match or close_index < match.end():
        raise ValueError("Could not locate Liberty library body")
    return text[match.end() : close_index]


def _sdc_get_ports(text: str) -> list[str]:
    ports = []
    for match in re.finditer(r"get_ports\s+(?:\{(?P<braced>[^}]+)\}|(?P<plain>[^\]\s]+))", text):
        raw = match.group("braced") if match.group("braced") is not None else match.group("plain")
        for item in str(raw).split():
            item = item.strip()
            if item:
                ports.append(item)
    return ports


def _replace_sdc_port(text: str, old: str, new: str) -> str:
    escaped = re.escape(old)
    text = re.sub(rf"(get_ports\s+\{{){escaped}(\}})", rf"\1{new}\2", text)
    text = re.sub(rf"(get_ports\s+){escaped}(\]|\s)", rf"\1{new}\2", text)
    return text


def _verilog_top_ports(path: Path) -> set[str]:
    return set(_verilog_top_port_directions(path))


def _verilog_top_port_directions(path: Path) -> dict[str, str]:
    text = _strip_verilog_comments(path.read_text(encoding="utf-8", errors="replace"))
    modules = _iter_verilog_module_blocks(text)
    if modules:
        text = modules[0]["text"]
    ports: dict[str, str] = {}
    for match in re.finditer(
        r"\b(?:input|output|inout)\b\s+(?:wire|reg|logic|signed|unsigned|\[[^\]]+\]\s*)*(?P<names>[^;]+);",
        text,
    ):
        direction = match.group(0).split(None, 1)[0]
        for raw_name in match.group("names").split(","):
            name = raw_name.strip()
            name = re.sub(r"=.*$", "", name).strip()
            name = re.sub(r"\[[^\]]+\]", "", name).strip()
            if name:
                ports[name] = direction
    header = re.search(
        r"\bmodule\s+(?:\\\S+|[A-Za-z_][A-Za-z0-9_$]*)\s*\((?P<ports>.*?)\)\s*;",
        text,
        flags=re.S,
    )
    if header:
        for raw_name in header.group("ports").split(","):
            name = raw_name.strip()
            name = re.sub(r"\[[^\]]+\]", "", name).strip()
            if name and name not in ports:
                ports[name] = "unknown"
    return ports


def _strip_verilog_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//.*", "", text)


def _timing_failure_stage(returncode: int, metrics: dict[str, Any], last: dict[str, Any]) -> str | None:
    if metrics.get("timed_out") or returncode == -9:
        return "timeout"
    if metrics.get("missing_input_file"):
        return "timer_init"
    if metrics.get("assertion"):
        return "dreamplace"
    if returncode != 0:
        return "dreamplace"
    if _float_or_none(last.get("wns")) is None and _float_or_none(last.get("tns")) is None:
        return "metric_parse"
    return None


def _violating_path_count(metrics: dict[str, Any]) -> int | None:
    last = metrics.get("last", {}) if isinstance(metrics, dict) else {}
    if _float_or_none(last.get("tns")) is None:
        return None
    return None


def _failed_result(
    placement: TimingProxyPlacement,
    run_dir: Path,
    failure_stage: str,
    message: str,
    variant: str,
    start_time: float,
) -> TimingProxyResult:
    run_dir.mkdir(parents=True, exist_ok=True)
    result = TimingProxyResult(
        design=placement.design,
        objective_id=placement.objective_id,
        seed=placement.seed,
        variant=variant,
        status="failed",
        failure_stage=failure_stage,
        timing_proxy_rc_model=TIMING_PROXY_RC_MODEL,
        timing_proxy_hpwl=None,
        timing_proxy_wns=None,
        timing_proxy_tns=None,
        timing_proxy_num_violating_paths=None,
        timing_proxy_runtime_seconds=time.time() - start_time,
        def_path=placement.def_path,
        run_dir=str(run_dir),
        config_path=None,
        log_path=None,
        returncode=None,
        source_placement_iterations=placement.requested_iterations,
        source_placement_completed_iterations=placement.completed_iterations,
        source_placement_budget_satisfied=placement.iteration_budget_satisfied,
        artifacts={"error": message, "source": placement.source},
    )
    _write_json(asdict(result), run_dir / "timing_proxy_result.json")
    return result


def _missing_collateral(design: TimingProxyDesign, panel: TimingProxyPanel) -> list[str]:
    missing = []
    for label, value in (
        ("base_config", design.base_config),
        ("sdc", design.sdc),
        ("ot_shell", panel.ot_shell),
    ):
        if value and not Path(value).expanduser().is_file():
            missing.append(label)
        if label in {"base_config", "ot_shell"} and not value:
            missing.append(label)
    if not panel.derive_verilog_from_def and not design.verilog:
        missing.append("verilog")
    elif not panel.derive_verilog_from_def and not Path(design.verilog).expanduser().is_file():
        missing.append("verilog")
    if isinstance(design.lib, list):
        for index, value in enumerate(design.lib):
            if not Path(value).expanduser().is_file():
                missing.append(f"lib[{index}]")
    elif design.lib and not Path(design.lib).expanduser().is_file():
        missing.append("lib")
    return missing


def _audit_seed_placements(placements: list[TimingProxyPlacement]) -> list[TimingProxyPlacement]:
    by_objective = {}
    for placement in placements:
        by_objective.setdefault(placement.objective_id, placement)
    selected = []
    if "default" in by_objective:
        selected.append(by_objective["default"])
    for objective_id, placement in by_objective.items():
        if objective_id != "default":
            selected.append(placement)
            break
    return selected or placements[:1]


def _perturbation_magnitudes(count: int) -> list[int]:
    base = [100, 250, 500, 1000, 2000]
    return [base[index % len(base)] for index in range(count)]


def _write_audit_panel(
    output: Path,
    *,
    base_config: str | Path,
    source_panel: str | Path | None,
    design: str,
    dreamplace_root: Path,
    timeout_seconds: int,
    gpu: int | None,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    resolved_base_config = _resolve_existing_path(base_config, base=Path.cwd())
    source = load_timing_proxy_panel(source_panel) if source_panel else None
    source_design = None
    if source is not None:
        source_design = next((item for item in source.designs if item.name == design), None)
        if source_design is None:
            raise ValueError(
                f"timing proxy panel {source_panel} does not contain design {design!r}"
            )
    base_payload = json.loads(resolved_base_config.read_text(encoding="utf-8-sig"))
    lib_input = (
        source_design.lib
        if source_design is not None
        else _resolve_dreamplace_config_path(base_payload.get("lib_input"), dreamplace_root)
    )
    sdc_input = (
        source_design.sdc
        if source_design is not None
        else _resolve_dreamplace_config_path(base_payload.get("sdc_input"), dreamplace_root)
    )
    verilog_input = (
        source_design.verilog
        if source_design is not None
        else _resolve_dreamplace_config_path(base_payload.get("verilog_input"), dreamplace_root)
    )
    lines = [
        f'ot_shell = "{Path(source.ot_shell).as_posix() if source else (dreamplace_root / "bin" / "ot-shell").as_posix()}"',
        f"timeout_seconds = {int(timeout_seconds)}",
        "iterations = 1",
        f"threads = {source.threads if source else 8}",
        f"legalize = {str(source.legalize if source else False).lower()}",
        f"fixed_position = {str(source.fixed_position if source else True).lower()}",
        f"position_jitter = {source.position_jitter if source else 0.0}",
        f"min_net_coverage = {source.min_net_coverage if source else 0.95}",
        f"scalarize_vector_nets = {str(source.scalarize_vector_nets if source else False).lower()}",
        f"derive_verilog_from_def = {str(source.derive_verilog_from_def if source else False).lower()}",
    ]
    if gpu is not None:
        lines.append(f"gpu = {int(gpu)}")
    lines.extend(
        [
            "",
            "[[designs]]",
            f'name = "{design}"',
            f'base_config = "{resolved_base_config.as_posix()}"',
            'rc_estimate = "flute_elmore_per_micron"',
        ]
    )
    if lib_input is not None:
        if isinstance(lib_input, list):
            quoted = ", ".join(f'"{Path(item).as_posix()}"' for item in lib_input)
            lines.append(f"lib = [{quoted}]")
        else:
            lines.append(f'lib = "{Path(lib_input).as_posix()}"')
    if sdc_input is not None:
        lines.append(f'sdc = "{Path(sdc_input).as_posix()}"')
    if verilog_input is not None and not (source and source.derive_verilog_from_def):
        lines.append(f'verilog = "{Path(verilog_input).as_posix()}"')
    if source_design is not None and source_design.top_module:
        lines.append(f'top_module = "{source_design.top_module}"')
    if source_design is not None and source_design.lef:
        quoted = ", ".join(f'"{Path(item).as_posix()}"' for item in source_design.lef)
        lines.append(f"lef = [{quoted}]")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _resolve_dreamplace_config_path(value: Any, dreamplace_root: Path) -> Path | None:
    if not value:
        return None
    if isinstance(value, list):
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidate = dreamplace_root / path
    if candidate.exists():
        return candidate.resolve()
    return path


def _audit_rows(
    results_payload: list[dict[str, Any]],
    perturbation_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records_by_variant = {str(item["variant"]): item for item in perturbation_records}
    rows = []
    baselines: dict[tuple[str, int], dict[str, Any]] = {}
    for result in results_payload:
        variant = str(result.get("variant"))
        record = records_by_variant.get(variant, {})
        if record.get("magnitude_dbu") == 0:
            baselines[(str(record.get("source_objective_id")), int(record.get("source_seed", 0)))] = result
    for result in results_payload:
        variant = str(result.get("variant"))
        record = records_by_variant.get(variant, {})
        key = (str(record.get("source_objective_id")), int(record.get("source_seed", 0)))
        baseline = baselines.get(key, {})
        rows.append(
            {
                "source_objective_id": record.get("source_objective_id"),
                "source_seed": record.get("source_seed"),
                "variant": variant,
                "magnitude_dbu": record.get("magnitude_dbu"),
                "hpwl": result.get("timing_proxy_hpwl"),
                "wns": result.get("timing_proxy_wns"),
                "tns": result.get("timing_proxy_tns"),
                "hpwl_delta": _delta_abs(
                    result.get("timing_proxy_hpwl"),
                    baseline.get("timing_proxy_hpwl"),
                ),
                "wns_delta": _delta_abs(
                    result.get("timing_proxy_wns"),
                    baseline.get("timing_proxy_wns"),
                ),
                "tns_delta": _delta_abs(
                    result.get("timing_proxy_tns"),
                    baseline.get("timing_proxy_tns"),
                ),
                "status": result.get("status"),
            }
        )
    return rows


def _result_row(result: TimingProxyResult) -> dict[str, Any]:
    return {
        "design": result.design,
        "objective_id": result.objective_id,
        "seed": result.seed,
        "variant": result.variant,
        "status": result.status,
        "failure_stage": result.failure_stage,
        "timing_proxy_rc_model": result.timing_proxy_rc_model,
        "timing_proxy_hpwl": result.timing_proxy_hpwl,
        "placement_overflow": result.placement_overflow,
        "placement_max_density": result.placement_max_density,
        "timing_proxy_total_nets": result.timing_proxy_total_nets,
        "timing_proxy_skipped_nets": result.timing_proxy_skipped_nets,
        "timing_proxy_pin_incomplete_nets": result.timing_proxy_pin_incomplete_nets,
        "timing_proxy_net_coverage": result.timing_proxy_net_coverage,
        "timing_proxy_wns": result.timing_proxy_wns,
        "timing_proxy_tns": result.timing_proxy_tns,
        "timing_proxy_num_violating_paths": result.timing_proxy_num_violating_paths,
        "timing_proxy_runtime_seconds": result.timing_proxy_runtime_seconds,
        "source_placement_iterations": result.source_placement_iterations,
        "source_placement_completed_iterations": result.source_placement_completed_iterations,
        "source_placement_budget_satisfied": result.source_placement_budget_satisfied,
        "def_path": result.def_path,
        "artifacts": json.dumps(result.artifacts, sort_keys=True),
    }


def _baseline_for_design_seed(
    results: list[TimingProxyResult],
    design: str,
    seed: int,
    baseline_objective_id: str,
) -> TimingProxyResult | None:
    for result in results:
        if (
            result.design == design
            and result.seed == seed
            and result.objective_id == baseline_objective_id
        ):
            return result
    return None


def _aggregate_status(rows: list[dict[str, Any]]) -> str:
    statuses = [normalize_timing_proxy_status(row.get("status")) for row in rows]
    if statuses and all(status == "success" for status in statuses):
        return "success"
    if any(status == "success" for status in statuses):
        return "partial"
    failures = [str(row.get("failure_stage") or row.get("status") or "failed") for row in rows]
    return failures[0] if failures else "unavailable"


def _resolve_panel_path(panel_path: Path, value: str) -> Path:
    expanded = Path(os.path.expandvars(value).replace("${HOME}", str(Path.home()))).expanduser()
    if expanded.is_absolute():
        return expanded
    return (panel_path.parent / expanded).resolve()


def _optional_path(panel_path: Path, value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(_resolve_panel_path(panel_path, str(value)))


def _optional_paths(panel_path: Path, value: Any) -> str | list[str] | None:
    if value in (None, ""):
        return None
    if isinstance(value, list):
        return [str(_resolve_panel_path(panel_path, str(item))) for item in value]
    return str(_resolve_panel_path(panel_path, str(value)))


def _resolve_existing_path(value: str | Path, *, base: Path) -> Path:
    path = Path(os.path.expandvars(str(value)).replace("${HOME}", str(Path.home()))).expanduser()
    if path.is_absolute():
        return path
    candidates = [
        (Path.cwd() / path).resolve(),
        (base / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        avg = (index + 1 + end) / 2.0
        for item in order[index:end]:
            ranks[item] = avg
        index = end
    return ranks


def _finite_pairs(xs: list[float | None], ys: list[float | None]) -> list[tuple[float, float]]:
    return [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)
    ]


def _mean_finite(values: Any) -> float | None:
    finite = []
    for value in values:
        numeric = _float_or_none(value)
        if numeric is not None:
            finite.append(numeric)
    return sum(finite) / len(finite) if finite else None


def _min_finite_int(values: Any) -> int | None:
    finite = []
    for value in values:
        numeric = _float_or_none(value)
        if numeric is not None:
            finite.append(int(numeric))
    return min(finite) if finite else None


def _csv_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _delta_pct(value: Any, baseline: Any) -> float | None:
    value_float = _float_or_none(value)
    baseline_float = _float_or_none(baseline)
    if value_float is None or baseline_float is None or baseline_float == 0:
        return None
    return 100.0 * (value_float - baseline_float) / abs(baseline_float)


def _delta_abs(value: Any, baseline: Any) -> float | None:
    value_float = _float_or_none(value)
    baseline_float = _float_or_none(baseline)
    if value_float is None or baseline_float is None:
        return None
    return value_float - baseline_float


def _float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _write_csv(rows: list[dict[str, Any]], output: str | Path, fieldnames: list[str]) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _write_json(payload: Any, output: str | Path) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
