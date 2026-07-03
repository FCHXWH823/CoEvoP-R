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
    "timing_proxy_wns",
    "timing_proxy_tns",
    "timing_proxy_num_violating_paths",
    "timing_proxy_runtime_seconds",
    "def_path",
    "artifacts",
]


@dataclass(frozen=True)
class TimingProxyDesign:
    name: str
    base_config: str
    lib: str | None = None
    sdc: str | None = None
    verilog: str | None = None
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


@dataclass(frozen=True)
class TimingProxyPlacement:
    design: str
    objective_id: str
    seed: int
    def_path: str
    source: str | None = None
    variant: str = "original"


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
                lib=_optional_path(panel_path, item.get("lib")),
                sdc=_optional_path(panel_path, item.get("sdc")),
                verilog=_optional_path(panel_path, item.get("verilog")),
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
    if mode == "tie_breaker" and status in {"success", "partial"}:
        if metrics.get("timing_proxy_tie_breaker_applied"):
            metrics["timing_proxy_parent_signal"] = "tie_breaker"
            return
        bonus = 0.0
        if wns_delta is not None:
            bonus += max(-0.05, min(0.05, wns_delta * 0.01))
        if tns_delta_pct is not None:
            bonus += max(-0.05, min(0.05, tns_delta_pct * 0.001))
        metrics["combined_score"] = float(metrics.get("combined_score") or 0.0) + bonus
        metrics["timing_proxy_tie_breaker_bonus"] = bonus
        metrics["timing_proxy_tie_breaker_applied"] = True
        metrics["timing_proxy_parent_signal"] = "tie_breaker"


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
        result[objective_id] = {
            "timing_proxy_status": _aggregate_status(objective_rows),
            "timing_proxy_rc_model": TIMING_PROXY_RC_MODEL,
            "timing_proxy_hpwl": _mean_finite(row.get("timing_proxy_hpwl") for row in objective_rows),
            "timing_proxy_wns": _mean_finite(row.get("timing_proxy_wns") for row in objective_rows),
            "timing_proxy_tns": _mean_finite(row.get("timing_proxy_tns") for row in objective_rows),
            "timing_proxy_hpwl_delta_pct": _mean_finite(
                row.get("timing_proxy_hpwl_delta_pct") for row in objective_rows
            ),
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
        previous = TimingProxyResult(**_load_json(summary_path))
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
    run = run_dreamplace(
        dreamplace_root=dreamplace_root,
        config_path=config_path,
        run_name=f"timing_proxy_{placement.design}_{placement.objective_id}_s{placement.seed}_{variant}",
        run_dir=cell_dir,
        timeout_seconds=panel.timeout_seconds,
    )
    write_run_summary(run, cell_dir / "dreamplace_run.json")
    last = run.metrics.get("last", {}) if run.metrics else {}
    failure_stage = _timing_failure_stage(run.returncode, run.metrics, last)
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
        timing_proxy_wns=_float_or_none(last.get("wns")),
        timing_proxy_tns=_float_or_none(last.get("tns")),
        timing_proxy_num_violating_paths=_violating_path_count(run.metrics),
        timing_proxy_runtime_seconds=time.time() - start,
        def_path=str(def_path),
        run_dir=str(cell_dir),
        config_path=str(config_path),
        log_path=run.log_path,
        returncode=run.returncode,
        artifacts={
            "dreamplace_run": str(cell_dir / "dreamplace_run.json"),
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
    config = json.loads(base_config.read_text(encoding="utf-8"))
    result_dir = run_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    config["def_input"] = str(placement_def)
    config["result_dir"] = str(result_dir)
    config["timer_engine"] = "opentimer"
    config["timing_opt_flag"] = 1
    # DREAMPlace reports OpenTimer WNS/TNS either during late timing-driven
    # global placement iterations (>500) or in the post-legalization STA block.
    # Timing-proxy cells intentionally use very short placement runs, so enable
    # legalization here to exercise the cheap final STA path.
    config["legalize_flag"] = 1
    config["num_threads"] = int(panel.threads)
    config["plot_flag"] = 0
    if panel.gpu is not None:
        config["gpu"] = int(panel.gpu)
    for stage in config.get("global_place_stages", []):
        stage["iteration"] = int(panel.iterations)
    if design.lib:
        config["lib_input"] = design.lib
    if design.sdc:
        config["sdc_input"] = str(
            _prepare_timing_sdc(
                sdc=Path(design.sdc),
                verilog=Path(design.verilog) if design.verilog else None,
                run_dir=run_dir,
            )
        )
    if design.verilog:
        config["verilog_input"] = design.verilog
    if design.lef:
        config["lef_input"] = design.lef
    output = run_dir / "timing_proxy_config.json"
    output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


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
    if verilog is not None and verilog.is_file():
        port_directions = _verilog_top_port_directions(verilog)
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
        },
        prepared_dir / "sdc_normalization.json",
    )
    return output


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
    header = re.search(r"\bmodule\s+\w+\s*\((?P<ports>.*?)\)\s*;", text, flags=re.S)
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
        artifacts={"error": message, "source": placement.source},
    )
    _write_json(asdict(result), run_dir / "timing_proxy_result.json")
    return result


def _missing_collateral(design: TimingProxyDesign, panel: TimingProxyPanel) -> list[str]:
    missing = []
    for label, value in (
        ("base_config", design.base_config),
        ("lib", design.lib),
        ("sdc", design.sdc),
        ("verilog", design.verilog),
        ("ot_shell", panel.ot_shell),
    ):
        if value and not Path(value).expanduser().is_file():
            missing.append(label)
        if label in {"base_config", "ot_shell"} and not value:
            missing.append(label)
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
    design: str,
    dreamplace_root: Path,
    timeout_seconds: int,
    gpu: int | None,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    resolved_base_config = _resolve_existing_path(base_config, base=Path.cwd())
    base_payload = json.loads(resolved_base_config.read_text(encoding="utf-8"))
    lib_input = _resolve_dreamplace_config_path(base_payload.get("lib_input"), dreamplace_root)
    sdc_input = _resolve_dreamplace_config_path(base_payload.get("sdc_input"), dreamplace_root)
    verilog_input = _resolve_dreamplace_config_path(base_payload.get("verilog_input"), dreamplace_root)
    lines = [
        f'ot_shell = "{(dreamplace_root / "bin" / "ot-shell").as_posix()}"',
        f"timeout_seconds = {int(timeout_seconds)}",
        "iterations = 1",
        "threads = 8",
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
        lines.append(f'lib = "{lib_input.as_posix()}"')
    if sdc_input is not None:
        lines.append(f'sdc = "{sdc_input.as_posix()}"')
    if verilog_input is not None:
        lines.append(f'verilog = "{verilog_input.as_posix()}"')
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
        "timing_proxy_wns": result.timing_proxy_wns,
        "timing_proxy_tns": result.timing_proxy_tns,
        "timing_proxy_num_violating_paths": result.timing_proxy_num_violating_paths,
        "timing_proxy_runtime_seconds": result.timing_proxy_runtime_seconds,
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
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
