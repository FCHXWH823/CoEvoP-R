"""Tier-3 OpenROAD/ChiPBench routed evaluation."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sqlite3
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from coevop.eval.chipbench_utils import prepare_chipbench_runtime_dirs
from coevop.eval.macro_preflight import audit_dreamplace_macro_config
from coevop.eval.ranking import aggregate_rank_scores
from coevop.eval.shared_panel import SharedPanelDesign, load_shared_panel


TIER3_TABLE_COLUMNS = [
    "design",
    "objective_id",
    "seed",
    "status",
    "failure_stage",
    "metrics_stage",
    "partial_metrics_available",
    "routed_wirelength",
    "estimated_wirelength",
    "grt_overflow",
    "drc_count",
    "wns",
    "tns",
    "power",
    "area",
    "runtime_seconds",
    "def_path",
    "metrics_path",
]


@dataclass(frozen=True)
class Tier3Placement:
    design: str
    objective_id: str
    seed: int
    def_path: str
    source: str | None = None


@dataclass(frozen=True)
class Tier3Result:
    design: str
    objective_id: str
    seed: int
    status: str
    failure_stage: str | None
    routed_wirelength: float | None
    grt_overflow: float | None
    drc_count: float | None
    wns: float | None
    tns: float | None
    power: float | None
    area: float | None
    runtime_seconds: float | None
    def_path: str
    metrics_path: str | None
    run_dir: str
    log_path: str | None
    returncode: int | None
    raw_metrics: dict[str, Any]


def load_placements(path: str | Path) -> list[Tier3Placement]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    raw_items = payload.get("placements", payload if isinstance(payload, list) else [])
    if not isinstance(raw_items, list):
        raise ValueError("placements JSON must contain a 'placements' list")
    placements = []
    for item in raw_items:
        placements.append(
            Tier3Placement(
                design=str(item["design"]),
                objective_id=str(item["objective_id"]),
                seed=int(item["seed"]),
                def_path=str(item["def_path"]),
                source=item.get("source"),
            )
        )
    return placements


def run_tier3_openroad(
    *,
    panel_path: str | Path,
    placements_path: str | Path,
    chipbench_root: str | Path,
    run_dir: str | Path,
    store_path: str | Path | None = None,
    baseline_objective_id: str = "default",
    resume: bool = False,
    retry_failed: bool = False,
) -> dict[str, Any]:
    panel = load_shared_panel(panel_path)
    placements = load_placements(placements_path)
    chipbench_root = Path(chipbench_root)
    run_root = Path(run_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    store_path = Path(store_path) if store_path else run_root / "tier3.sqlite"
    design_by_name = {design.name: design for design in panel.designs}

    conn = _open_store(store_path)
    results = []
    try:
        for placement in placements:
            design = design_by_name.get(placement.design)
            result = _run_one_tier3_cell(
                placement=placement,
                design=design,
                chipbench_root=chipbench_root,
                timeout_seconds=panel.timeout_seconds,
                run_root=run_root,
                resume=resume,
                retry_failed=retry_failed,
            )
            _upsert_result(conn, result)
            results.append(result)
    finally:
        conn.close()

    metrics_csv = write_metrics_csv(results, run_root / "metrics.csv")
    comparison_rows = build_comparison_rows(results, baseline_objective_id=baseline_objective_id)
    comparison_csv = write_comparison_csv(comparison_rows, run_root / "comparison_table.csv")
    rankings = aggregate_rank_scores(
        comparison_rows,
        metric_names=(
            "routed_wirelength_delta_pct",
            "grt_overflow_delta_pct",
            "drc_count_delta_pct",
            "wns_cost_delta_ns",
            "tns_cost_delta_ns",
        ),
    )
    rankings_path = _write_json(
        {"rankings": [ranking.to_dict() for ranking in rankings]},
        run_root / "rankings.json",
    )
    report_path = run_root / "tier3_report.md"
    report_path.write_text(_build_report(results, rankings), encoding="utf-8")
    summary = {
        "run_dir": str(run_root),
        "store": str(store_path),
        "metrics_csv": str(metrics_csv),
        "comparison_csv": str(comparison_csv),
        "rankings": str(rankings_path),
        "tier3_report": str(report_path),
        "result_count": len(results),
        "success_count": sum(1 for result in results if result.status == "success"),
        "partial_count": sum(1 for result in results if result.status == "partial"),
        "failure_count": sum(1 for result in results if result.status != "success"),
        "baseline_objective_id": baseline_objective_id,
    }
    _write_json(summary, run_root / "summary.json")
    return summary


def recover_tier3_partial_run(
    *,
    panel_path: str | Path,
    placements_path: str | Path,
    chipbench_root: str | Path,
    run_dir: str | Path,
    store_path: str | Path | None = None,
    baseline_objective_id: str = "default",
    failure_stage: str = "interrupted",
    message: str = "Tier-3 run was interrupted before final ChiPBench metrics were written.",
) -> dict[str, Any]:
    """Recover partial Tier-3 artifacts after an interrupted ChiPBench run."""

    panel = load_shared_panel(panel_path)
    placements = load_placements(placements_path)
    chipbench_root = Path(chipbench_root)
    run_root = Path(run_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    store_path = Path(store_path) if store_path else run_root / "tier3.sqlite"
    design_by_name = {design.name: design for design in panel.designs}

    conn = _open_store(store_path)
    results = []
    try:
        for placement in placements:
            design = design_by_name.get(placement.design)
            cell_dir = (
                run_root
                / _safe_name(placement.design)
                / _safe_name(placement.objective_id)
                / f"seed_{placement.seed}"
            )
            if design is None:
                result = _failed_result(
                    placement,
                    cell_dir,
                    "config_parse",
                    f"design {placement.design!r} is not in the shared panel",
                )
            else:
                evaluate_name = (
                    f"coevop_{_safe_name(placement.design)}_"
                    f"{_safe_name(placement.objective_id)}_s{placement.seed}"
                )
                result = _failed_or_partial_result(
                    placement,
                    cell_dir,
                    failure_stage,
                    message,
                    chipbench_root=chipbench_root,
                    evaluate_name=evaluate_name,
                    runtime_seconds=None,
                    returncode=None,
                    log_path=cell_dir / "openroad.log",
                )
            _upsert_result(conn, result)
            results.append(result)
    finally:
        conn.close()

    metrics_csv = write_metrics_csv(results, run_root / "metrics.csv")
    comparison_rows = build_comparison_rows(results, baseline_objective_id=baseline_objective_id)
    comparison_csv = write_comparison_csv(comparison_rows, run_root / "comparison_table.csv")
    rankings = aggregate_rank_scores(
        comparison_rows,
        metric_names=("routed_wirelength_delta_pct", "grt_overflow_delta_pct", "drc_count_delta_pct"),
    )
    rankings_path = _write_json(
        {"rankings": [ranking.to_dict() for ranking in rankings]},
        run_root / "rankings.json",
    )
    report_path = run_root / "tier3_report.md"
    report_path.write_text(_build_report(results, rankings), encoding="utf-8")
    summary = {
        "run_dir": str(run_root),
        "store": str(store_path),
        "metrics_csv": str(metrics_csv),
        "comparison_csv": str(comparison_csv),
        "rankings": str(rankings_path),
        "tier3_report": str(report_path),
        "result_count": len(results),
        "success_count": sum(1 for result in results if result.status == "success"),
        "partial_count": sum(1 for result in results if result.status == "partial"),
        "failure_count": sum(1 for result in results if result.status != "success"),
        "baseline_objective_id": baseline_objective_id,
        "recovered": True,
    }
    _write_json(summary, run_root / "summary.json")
    return summary


def parse_chipbench_metrics(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    flattened = _flatten_metrics(raw)
    return {
        "raw": raw,
        "routed_wirelength": _find_metric(flattened, ["routed_wirelength", "wirelength"]),
        "grt_overflow": _find_metric(flattened, ["grt_overflow", "overflow"]),
        "drc_count": _find_metric(flattened, ["drc_count", "drv_count", "drc"]),
        "wns": _find_metric(flattened, ["wns"]),
        "tns": _find_metric(flattened, ["tns"]),
        "power": _find_metric(flattened, ["total_power", "power"]),
        "area": _find_metric(flattened, ["design_area", "area"]),
    }


def parse_chipbench_partial_metrics(stage_dir: str | Path) -> dict[str, Any] | None:
    """Recover PPA-stage metrics from an incomplete ChiPBench run.

    ChiPBench writes final benchmarking metrics only after detailed routing and
    finish complete. On larger smoke designs, OpenROAD can time out in global
    route after detailed placement and CTS have already produced JSON metrics.
    These values are useful for debugging and comparison, but they are marked
    as partial because they are not final routed PPA.
    """

    stage_root = Path(stage_dir)
    if not stage_root.is_dir():
        return None

    stage_specs = [
        ("grt", "5_1_grt.json", "globalroute"),
        ("cts", "4_1_cts.json", "cts"),
        ("detailedplace", "3_5_place_dp.json", "detailedplace"),
        ("placeopt", "3_4_place_resized.json", "placeopt"),
    ]
    stages: dict[str, Any] = {}
    selected: tuple[str, str, dict[str, Any]] | None = None
    for stage_name, filename, prefix in stage_specs:
        path = stage_root / filename
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        stages[stage_name] = data
        if selected is None:
            selected = (stage_name, prefix, data)

    if selected is None:
        return None

    stage_name, prefix, data = selected
    grt_log = stage_root / "5_1_grt.log"
    grt_summary = _parse_grt_log_summary(grt_log) if grt_log.is_file() else {}
    congestion_report = _parse_grt_congestion_report(
        _find_sibling_report_dir(stage_root) / "congestion.rpt"
    )
    flattened = _flatten_metrics(data)
    stage_metrics = {
        "estimated_wirelength": _first_float(
            data.get(f"{prefix}__route__wirelength__estimated"),
            _find_metric(
                flattened,
                [
                    "route_wirelength_estimated",
                    "estimated_wirelength",
                    "wirelength_estimated",
                    "wirelength",
                ],
            ),
            grt_summary.get("total_wirelength"),
        ),
        "grt_overflow": _first_float(
            _find_metric(
                flattened,
                [
                    "grt_overflow",
                    "globalroute_overflow",
                    "global_route_overflow",
                    "route_overflow",
                    "overflow",
                    "total_overflow",
                ],
            ),
            grt_summary.get("final_total_overflow"),
            congestion_report.get("total_overflow"),
        ),
        "drc_count": _find_metric(
            flattened,
            ["drc_count", "drv_count", "design_violations", "violations"],
        ),
        "wns": _first_float(
            data.get(f"{prefix}__timing__setup__ws"),
            _find_metric(flattened, ["wns", "setup_ws", "worst_slack"]),
        ),
        "tns": _first_float(
            data.get(f"{prefix}__timing__setup__tns"),
            _find_metric(flattened, ["tns", "setup_tns"]),
        ),
        "power": _first_float(
            data.get(f"{prefix}__power__total"),
            _find_metric(flattened, ["total_power", "power_total", "power"]),
        ),
        "area": _first_float(
            data.get(f"{prefix}__design__instance__area__stdcell"),
            _find_metric(flattened, ["stdcell_area", "instance_area_stdcell"]),
        ),
        "total_instance_area": _first_float(
            data.get(f"{prefix}__design__instance__area"),
            _find_metric(flattened, ["instance_area", "design_area", "area"]),
        ),
        "design_violations": _first_float(
            data.get(f"{prefix}__design__violations"),
            _find_metric(flattened, ["design_violations", "violations"]),
        ),
        "setup_violation_count": _first_float(
            data.get(f"{prefix}__timing__drv__setup_violation_count"),
            _find_metric(flattened, ["setup_violation_count"]),
        ),
        "max_slew_violations": _first_float(
            data.get(f"{prefix}__timing__drv__max_slew"),
            _find_metric(flattened, ["max_slew"]),
        ),
        "max_cap_violations": _first_float(
            data.get(f"{prefix}__timing__drv__max_cap"),
            _find_metric(flattened, ["max_cap"]),
        ),
    }
    raw = {
        "metrics_stage": f"partial_{stage_name}",
        "metric_kind": "incomplete_openroad_stage_metrics",
        "stage_dir": str(stage_root),
        "selected_stage": stage_name,
        "note": (
            "Recovered from ChiPBench stage JSON because final routed "
            "metrics.json was not produced."
        ),
        "coevop_partial_metrics": stage_metrics,
        "global_route_log": grt_summary,
        "global_route_congestion_report": congestion_report,
        "stages": stages,
    }
    return {
        "raw": raw,
        "metrics_stage": raw["metrics_stage"],
        "estimated_wirelength": stage_metrics["estimated_wirelength"],
        "routed_wirelength": None,
        "grt_overflow": stage_metrics["grt_overflow"],
        "drc_count": stage_metrics["drc_count"],
        "wns": stage_metrics["wns"],
        "tns": stage_metrics["tns"],
        "power": stage_metrics["power"],
        "area": stage_metrics["area"],
    }


def build_comparison_rows(
    results: list[Tier3Result],
    *,
    baseline_objective_id: str = "default",
) -> list[dict[str, Any]]:
    baseline_id = baseline_objective_id or "default"
    by_key = {(r.design, r.seed, r.objective_id): r for r in results}
    rows = []
    for result in results:
        baseline = by_key.get((result.design, result.seed, baseline_id))
        row = _result_csv_row(result)
        row["baseline_objective_id"] = baseline_id
        row["estimated_wirelength_delta_pct"] = _delta_pct(
            _estimated_wirelength(result),
            _estimated_wirelength(baseline) if baseline else None,
        )
        row["routed_wirelength_delta_pct"] = _delta_pct(
            result.routed_wirelength,
            baseline.routed_wirelength if baseline else None,
        )
        row["grt_overflow_delta_pct"] = _delta_pct(
            result.grt_overflow,
            baseline.grt_overflow if baseline else None,
        )
        row["drc_count_delta_pct"] = _delta_pct(
            result.drc_count,
            baseline.drc_count if baseline else None,
        )
        row["wns_gain_ns"] = _gain(
            result.wns,
            baseline.wns if baseline else None,
        )
        row["tns_gain_ns"] = _gain(
            result.tns,
            baseline.tns if baseline else None,
        )
        row["wns_cost_delta_ns"] = _negate(row["wns_gain_ns"])
        row["tns_cost_delta_ns"] = _negate(row["tns_gain_ns"])
        rows.append(row)
    return rows


def write_metrics_csv(results: list[Tier3Result], output: str | Path) -> Path:
    return _write_csv([_result_csv_row(result) for result in results], output, TIER3_TABLE_COLUMNS)


def write_comparison_csv(rows: list[dict[str, Any]], output: str | Path) -> Path:
    return _write_csv(
        rows,
        output,
        TIER3_TABLE_COLUMNS
        + [
            "baseline_objective_id",
            "estimated_wirelength_delta_pct",
            "routed_wirelength_delta_pct",
            "grt_overflow_delta_pct",
            "drc_count_delta_pct",
            "wns_gain_ns",
            "tns_gain_ns",
            "wns_cost_delta_ns",
            "tns_cost_delta_ns",
        ],
    )


def _run_one_tier3_cell(
    *,
    placement: Tier3Placement,
    design: SharedPanelDesign | None,
    chipbench_root: Path,
    timeout_seconds: int,
    run_root: Path,
    resume: bool,
    retry_failed: bool,
) -> Tier3Result:
    run_dir = (
        run_root
        / _safe_name(placement.design)
        / _safe_name(placement.objective_id)
        / f"seed_{placement.seed}"
    )
    summary_path = run_dir / "run_summary.json"
    if resume and summary_path.is_file():
        previous = _load_result_summary(summary_path)
        if previous.status == "success":
            return previous
        if not retry_failed:
            recovered = _recover_existing_partial_result(
                previous=previous,
                placement=placement,
                design=design,
                chipbench_root=chipbench_root,
                run_dir=run_dir,
            )
            return recovered if recovered is not None else previous
    if design is None:
        return _failed_result(placement, run_dir, "panel_design", "design is not in shared panel")
    def_path = Path(placement.def_path).resolve()
    if not def_path.is_file():
        return _failed_result(placement, run_dir, "missing_def", "placement DEF does not exist")
    if not Path(design.chipbench_config).is_file():
        return _failed_result(placement, run_dir, "missing_config", "ChiPBench config does not exist")

    run_dir.mkdir(parents=True, exist_ok=True)
    prepare_chipbench_runtime_dirs(chipbench_root)
    # ChiPBench executes from its own repository root. Macro freezing and DEF
    # repair may create a run-local relative path, so resolve it first.
    def_path = _prepare_openroad_input_def(def_path, design, run_dir).resolve()
    evaluate_name = (
        f"coevop_{_safe_name(placement.design)}_"
        f"{_safe_name(placement.objective_id)}_s{placement.seed}"
    )
    log_path = run_dir / "openroad.log"
    command = [
        "python3",
        "benchmarking/benchmarking.py",
        f"--mode={design.mode}",
        f"--config_setting={design.chipbench_config}",
        f"--def_path={def_path}",
        f"--evaluate_name={evaluate_name}",
    ]
    start = time.time()
    run_timeout = None if timeout_seconds <= 0 else timeout_seconds
    with log_path.open("w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(
                command,
                cwd=chipbench_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=run_timeout,
                check=False,
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            log.write(f"\nCoEvoP&R ChiPBench timeout after {timeout_seconds} seconds\n")
            _terminate_chipbench_processes(evaluate_name)
            returncode = -9
    runtime = time.time() - start
    metrics_path = chipbench_root / "benchmarking_result" / evaluate_name / "metrics.json"
    if returncode == -9:
        result = _failed_or_partial_result(
            placement,
            run_dir,
            "timeout",
            "ChiPBench timed out",
            chipbench_root=chipbench_root,
            evaluate_name=evaluate_name,
            runtime_seconds=runtime,
            returncode=returncode,
            log_path=log_path,
        )
    elif returncode != 0:
        result = _failed_or_partial_result(
            placement,
            run_dir,
            "openroad",
            "ChiPBench failed",
            chipbench_root=chipbench_root,
            evaluate_name=evaluate_name,
            runtime_seconds=runtime,
            returncode=returncode,
            log_path=log_path,
        )
    elif not metrics_path.is_file():
        result = _failed_or_partial_result(
            placement,
            run_dir,
            "metric_extraction",
            "ChiPBench metrics.json was not produced",
            chipbench_root=chipbench_root,
            evaluate_name=evaluate_name,
            runtime_seconds=runtime,
            returncode=returncode,
            log_path=log_path,
        )
    else:
        parsed = parse_chipbench_metrics(metrics_path)
        copied_metrics_path = run_dir / "metrics.json"
        copied_metrics_path.write_text(
            json.dumps(parsed["raw"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result = Tier3Result(
            design=placement.design,
            objective_id=placement.objective_id,
            seed=placement.seed,
            status="success",
            failure_stage=None,
            routed_wirelength=_float_or_none(parsed.get("routed_wirelength")),
            grt_overflow=_float_or_none(parsed.get("grt_overflow")),
            drc_count=_float_or_none(parsed.get("drc_count")),
            wns=_float_or_none(parsed.get("wns")),
            tns=_float_or_none(parsed.get("tns")),
            power=_float_or_none(parsed.get("power")),
            area=_float_or_none(parsed.get("area")),
            runtime_seconds=runtime,
            def_path=str(def_path),
            metrics_path=str(copied_metrics_path),
            run_dir=str(run_dir),
            log_path=str(log_path),
            returncode=returncode,
            raw_metrics=parsed["raw"],
        )
    _write_json(asdict(result), summary_path)
    return result


def _prepare_openroad_input_def(
    def_path: Path,
    design: SharedPanelDesign,
    run_dir: Path,
) -> Path:
    source_def = _resolve_source_def(design)
    prepared_def = def_path
    if source_def is not None:
        report = _repair_def_physical_components(
            source_def=source_def,
            target_def=prepared_def,
            output_def=run_dir / "openroad_input.def",
        )
        if report["repaired_components"] > 0:
            _write_json(report, run_dir / "def_repair_report.json")
            prepared_def = Path(report["output_def"])
        pin_report = _repair_def_pin_geometry(
            source_def=source_def,
            target_def=prepared_def,
            output_def=run_dir / "openroad_input_with_pins.def",
        )
        _write_json(pin_report, run_dir / "pin_geometry_repair_report.json")
        if pin_report["unresolved_pins"]:
            raise ValueError(
                "cannot restore routable geometry for all OpenROAD top-level pins: "
                f"{pin_report}"
            )
        if pin_report["repaired_pins"] > 0:
            prepared_def = Path(pin_report["output_def"])
    if design.freeze_placed_macros_for_routing:
        macro_report = _freeze_placed_hard_macros_for_routing(
            design_config=Path(design.dreamplace_config),
            target_def=prepared_def,
            output_def=run_dir / "openroad_input_frozen_macros.def",
        )
        _write_json(macro_report, run_dir / "macro_freeze_report.json")
        if not macro_report["ok"]:
            raise ValueError(
                "cannot freeze all DREAMPlace hard macros for OpenROAD: "
                f"{macro_report}"
            )
        prepared_def = Path(macro_report["output_def"])
    return prepared_def


def _freeze_placed_hard_macros_for_routing(
    *,
    design_config: Path,
    target_def: Path,
    output_def: Path,
) -> dict[str, Any]:
    """Freeze DREAMPlace-generated macro coordinates only for routing handoff."""

    dreamplace_root = Path(os.environ.get("DREAMPLACE_ROOT", ""))
    audit = audit_dreamplace_macro_config(
        design_config,
        dreamplace_root=dreamplace_root,
        require_movable_macros=True,
        reject_fixed_hard_macros=True,
    )
    macro_names = {
        str(item["instance"])
        for item in audit.get("components", [])
        if item.get("instance")
    }
    target_text = target_def.read_text(encoding="utf-8", errors="replace")
    components = _parse_def_components(target_text)
    component_names_by_key: dict[str, str] = {}
    ambiguous_keys: set[str] = set()
    for component_name in components:
        key = _canonical_def_instance_name(component_name)
        if key in component_names_by_key and component_names_by_key[key] != component_name:
            ambiguous_keys.add(key)
        else:
            component_names_by_key[key] = component_name
    matched_names: dict[str, str] = {}
    missing: list[str] = []
    for macro_name in sorted(macro_names):
        key = _canonical_def_instance_name(macro_name)
        target_name = component_names_by_key.get(key)
        if target_name is None or key in ambiguous_keys:
            missing.append(macro_name)
        else:
            matched_names[macro_name] = target_name
    replacements: dict[str, str] = {}
    frozen: list[str] = []
    already_fixed: list[str] = []
    without_coordinates: list[str] = []
    for source_name, target_name in sorted(matched_names.items()):
        block = components[target_name]
        if re.search(r"\+\s+FIXED\s*\(", block, flags=re.IGNORECASE):
            already_fixed.append(source_name)
            continue
        if not re.search(r"\+\s+PLACED\s*\(", block, flags=re.IGNORECASE):
            without_coordinates.append(source_name)
            continue
        replacements[target_name] = re.sub(
            r"(\+\s+)PLACED(\s*\()",
            r"\1FIXED\2",
            block,
            count=1,
            flags=re.IGNORECASE,
        )
        frozen.append(source_name)
    ok = (
        bool(audit.get("ok"))
        and bool(macro_names)
        and not missing
        and not without_coordinates
        and len(frozen) + len(already_fixed) == len(macro_names)
    )
    if ok:
        output_def.parent.mkdir(parents=True, exist_ok=True)
        output_def.write_text(
            _replace_def_components(target_text, replacements), encoding="utf-8"
        )
    return {
        "design_config": str(design_config),
        "target_def": str(target_def),
        "output_def": str(output_def),
        "input_macro_preflight_ok": bool(audit.get("ok")),
        "expected_hard_macros": len(macro_names),
        "frozen_hard_macros": len(frozen),
        "already_fixed_hard_macros": len(already_fixed),
        "missing_hard_macros": missing,
        "matched_instance_names": matched_names,
        "hard_macros_without_placed_coordinates": without_coordinates,
        "ok": ok,
    }


def _canonical_def_instance_name(name: str) -> str:
    """Normalize equivalent hierarchical separators used by ODB and DREAMPlace."""

    normalized = str(name).replace("\\", "")
    return re.sub(r"[./]+", "/", normalized).strip("/")


def _resolve_source_def(design: SharedPanelDesign) -> Path | None:
    config_path = Path(design.dreamplace_config)
    if not config_path.is_file():
        return _existing_path(design.reference_def)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _existing_path(design.reference_def)

    candidates: list[str | Path | None] = [
        design.reference_def,
        payload.get("coevop_chipbench_generation_audit", {}).get("source_def"),
        payload.get("def_input"),
    ]
    for candidate in candidates:
        resolved = _resolve_def_candidate(candidate, config_path)
        if resolved is not None:
            return resolved
    return None


def _resolve_def_candidate(candidate: str | Path | None, config_path: Path) -> Path | None:
    if candidate is None:
        return None
    raw = Path(os.path.expandvars(os.path.expanduser(str(candidate))))
    if raw.is_absolute():
        return raw if raw.is_file() else None
    roots = [
        Path.cwd(),
        config_path.parent,
        Path(os.environ.get("DREAMPLACE_ROOT", "")),
        Path.home() / "DREAMPlace" / "install",
    ]
    for root in roots:
        if not str(root):
            continue
        path = (root / raw).resolve()
        if path.is_file():
            return path
    return None


def _existing_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path if path.is_file() else None


def _repair_def_physical_components(
    *,
    source_def: Path,
    target_def: Path,
    output_def: Path,
) -> dict[str, Any]:
    """Restore physical-only component declarations after DREAMPlace DEF export.

    DREAMPlace may emit fixed tap/endcap cells as ordinary library cells on some
    ChiPBench designs. OpenROAD then rejects those boundary cells before GRT.
    The repair keeps DREAMPlace's placement for regular cells but restores
    physical-only component blocks from the source DEF by instance name.
    """

    source_text = source_def.read_text(encoding="utf-8", errors="replace")
    target_text = target_def.read_text(encoding="utf-8", errors="replace")
    source_components = _parse_def_components(source_text)
    target_components = _parse_def_components(target_text)
    source_physical = {
        name: block
        for name, block in source_components.items()
        if _is_physical_only_component(name, block)
    }
    repaired_names = [
        name
        for name, block in target_components.items()
        if name in source_physical and block != source_physical[name]
    ]
    if not repaired_names:
        return {
            "source_def": str(source_def),
            "target_def": str(target_def),
            "output_def": str(target_def),
            "source_physical_components": len(source_physical),
            "target_components": len(target_components),
            "repaired_components": 0,
            "repaired_component_examples": [],
        }

    repaired_text = _replace_def_components(target_text, source_physical)
    output_def.parent.mkdir(parents=True, exist_ok=True)
    output_def.write_text(repaired_text, encoding="utf-8")
    return {
        "source_def": str(source_def),
        "target_def": str(target_def),
        "output_def": str(output_def),
        "source_physical_components": len(source_physical),
        "target_components": len(target_components),
        "repaired_components": len(repaired_names),
        "repaired_component_examples": repaired_names[:20],
    }


def _repair_def_pin_geometry(
    *,
    source_def: Path,
    target_def: Path,
    output_def: Path,
) -> dict[str, Any]:
    """Restore routable top-level pin geometry lost by DREAMPlace DEF export.

    The replacement is deliberately limited to target pins that have no layer
    and placement geometry. Cell and macro component blocks remain untouched.
    """

    source_text = source_def.read_text(encoding="utf-8", errors="replace")
    target_text = target_def.read_text(encoding="utf-8", errors="replace")
    source_pins = _parse_def_pins(source_text)
    target_pins = _parse_def_pins(target_text)
    missing_geometry = sorted(
        name for name, block in target_pins.items() if not _pin_has_routing_geometry(block)
    )
    replacements = {
        name: source_pins[name]
        for name in missing_geometry
        if name in source_pins and _pin_has_routing_geometry(source_pins[name])
    }
    unresolved = sorted(set(missing_geometry) - set(replacements))
    if replacements:
        output_def.parent.mkdir(parents=True, exist_ok=True)
        output_def.write_text(
            _replace_def_pins(target_text, replacements),
            encoding="utf-8",
        )
        result_path = output_def
    else:
        result_path = target_def
    return {
        "source_def": str(source_def),
        "target_def": str(target_def),
        "output_def": str(result_path),
        "source_pins": len(source_pins),
        "target_pins": len(target_pins),
        "pins_missing_geometry": len(missing_geometry),
        "repaired_pins": len(replacements),
        "repaired_pin_examples": sorted(replacements)[:20],
        "unresolved_pins": unresolved,
        "ok": not unresolved,
    }


def _parse_def_components(def_text: str) -> dict[str, str]:
    components: dict[str, str] = {}
    in_components = False
    current: list[str] = []
    current_name: str | None = None
    for line in def_text.splitlines(keepends=True):
        stripped = line.strip()
        if not in_components:
            if stripped.startswith("COMPONENTS "):
                in_components = True
            continue
        if stripped == "END COMPONENTS":
            break
        if stripped.startswith("- "):
            if current_name is not None:
                components[current_name] = "".join(current)
            parts = stripped.split()
            current_name = parts[1] if len(parts) > 1 else None
            current = [line]
            if ";" in stripped and current_name is not None:
                components[current_name] = "".join(current)
                current_name = None
                current = []
            continue
        if current_name is not None:
            current.append(line)
            if ";" in stripped:
                components[current_name] = "".join(current)
                current_name = None
                current = []
    if current_name is not None:
        components[current_name] = "".join(current)
    return components


def _parse_def_pins(def_text: str) -> dict[str, str]:
    return _parse_def_named_section(def_text, section="PINS")


def _parse_def_named_section(def_text: str, *, section: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    in_section = False
    current: list[str] = []
    current_name: str | None = None
    section_header = f"{section} "
    section_end = f"END {section}"
    for line in def_text.splitlines(keepends=True):
        stripped = line.strip()
        if not in_section:
            if stripped.startswith(section_header):
                in_section = True
            continue
        if stripped == section_end:
            break
        if stripped.startswith("- "):
            if current_name is not None:
                entries[current_name] = "".join(current)
            parts = stripped.split()
            current_name = parts[1] if len(parts) > 1 else None
            current = [line]
            if ";" in stripped and current_name is not None:
                entries[current_name] = "".join(current)
                current_name = None
                current = []
            continue
        if current_name is not None:
            current.append(line)
            if ";" in stripped:
                entries[current_name] = "".join(current)
                current_name = None
                current = []
    if current_name is not None:
        entries[current_name] = "".join(current)
    return entries


def _pin_has_routing_geometry(block: str) -> bool:
    return bool(
        re.search(r"\+\s+LAYER\s+\S+", block, flags=re.IGNORECASE)
        and re.search(
            r"\+\s+(?:PLACED|FIXED|COVER)\s*\(",
            block,
            flags=re.IGNORECASE,
        )
    )


def _replace_def_components(def_text: str, replacements: dict[str, str]) -> str:
    output: list[str] = []
    in_components = False
    current: list[str] = []
    current_name: str | None = None
    for line in def_text.splitlines(keepends=True):
        stripped = line.strip()
        if not in_components:
            output.append(line)
            if stripped.startswith("COMPONENTS "):
                in_components = True
            continue
        if stripped == "END COMPONENTS":
            if current_name is not None:
                output.append(replacements.get(current_name, "".join(current)))
                current_name = None
                current = []
            output.append(line)
            in_components = False
            continue
        if stripped.startswith("- "):
            if current_name is not None:
                output.append(replacements.get(current_name, "".join(current)))
            parts = stripped.split()
            current_name = parts[1] if len(parts) > 1 else None
            current = [line]
            if ";" in stripped and current_name is not None:
                output.append(replacements.get(current_name, "".join(current)))
                current_name = None
                current = []
            continue
        if current_name is not None:
            current.append(line)
            if ";" in stripped:
                output.append(replacements.get(current_name, "".join(current)))
                current_name = None
                current = []
        else:
            output.append(line)
    return "".join(output)


def _replace_def_pins(def_text: str, replacements: dict[str, str]) -> str:
    output: list[str] = []
    in_pins = False
    current: list[str] = []
    current_name: str | None = None
    for line in def_text.splitlines(keepends=True):
        stripped = line.strip()
        if not in_pins:
            output.append(line)
            if stripped.startswith("PINS "):
                in_pins = True
            continue
        if stripped == "END PINS":
            if current_name is not None:
                output.append(replacements.get(current_name, "".join(current)))
                current_name = None
                current = []
            output.append(line)
            in_pins = False
            continue
        if stripped.startswith("- "):
            if current_name is not None:
                output.append(replacements.get(current_name, "".join(current)))
            parts = stripped.split()
            current_name = parts[1] if len(parts) > 1 else None
            current = [line]
            if ";" in stripped and current_name is not None:
                output.append(replacements.get(current_name, "".join(current)))
                current_name = None
                current = []
            continue
        if current_name is not None:
            current.append(line)
            if ";" in stripped:
                output.append(replacements.get(current_name, "".join(current)))
                current_name = None
                current = []
        else:
            output.append(line)
    return "".join(output)


def _is_physical_only_component(name: str, block: str) -> bool:
    upper_name = name.upper()
    first_line = block.strip().splitlines()[0].upper() if block.strip() else ""
    tokens = first_line.split()
    master = tokens[2] if len(tokens) >= 3 and tokens[0] == "-" else ""
    physical_name_markers = ("PHY_", "TAP", "ENDCAP", "FILL", "DECAP", "WELLTAP")
    physical_master_markers = ("TAP", "ENDCAP", "FILL", "DECAP", "WELLTAP")
    return upper_name.startswith(physical_name_markers) or any(
        marker in master for marker in physical_master_markers
    )


def _terminate_chipbench_processes(evaluate_name: str) -> None:
    """Best-effort cleanup of ChiPBench/OpenROAD descendants after timeout."""

    if not evaluate_name or os.name != "posix":
        return
    for signal_name in ("TERM", "KILL"):
        subprocess.call(
            ["pkill", f"-{signal_name}", "-f", evaluate_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1)


def _failed_or_partial_result(
    placement: Tier3Placement,
    run_dir: Path,
    failure_stage: str,
    message: str,
    *,
    chipbench_root: Path,
    evaluate_name: str,
    runtime_seconds: float | None = None,
    returncode: int | None = None,
    log_path: str | Path | None = None,
) -> Tier3Result:
    stage_dir = _find_chipbench_stage_dir(chipbench_root, evaluate_name)
    partial = parse_chipbench_partial_metrics(stage_dir) if stage_dir is not None else None
    if partial is None:
        return _failed_result(
            placement,
            run_dir,
            failure_stage,
            message,
            runtime_seconds=runtime_seconds,
            returncode=returncode,
            log_path=log_path,
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    partial_raw = dict(partial["raw"])
    partial_raw["error"] = message
    partial_raw["failure_stage"] = failure_stage
    partial_metrics_path = run_dir / "partial_metrics.json"
    partial_metrics_path.write_text(
        json.dumps(partial_raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = Tier3Result(
        design=placement.design,
        objective_id=placement.objective_id,
        seed=placement.seed,
        status="partial",
        failure_stage=failure_stage,
        routed_wirelength=_float_or_none(partial.get("routed_wirelength")),
        grt_overflow=_float_or_none(partial.get("grt_overflow")),
        drc_count=_float_or_none(partial.get("drc_count")),
        wns=_float_or_none(partial.get("wns")),
        tns=_float_or_none(partial.get("tns")),
        power=_float_or_none(partial.get("power")),
        area=_float_or_none(partial.get("area")),
        runtime_seconds=runtime_seconds,
        def_path=str(Path(placement.def_path).resolve()),
        metrics_path=str(partial_metrics_path),
        run_dir=str(run_dir),
        log_path=str(log_path) if log_path is not None else None,
        returncode=returncode,
        raw_metrics=partial_raw,
    )
    _write_json(asdict(result), run_dir / "run_summary.json")
    return result


def _failed_result(
    placement: Tier3Placement,
    run_dir: Path,
    failure_stage: str,
    message: str,
    runtime_seconds: float | None = None,
    returncode: int | None = None,
    log_path: str | Path | None = None,
) -> Tier3Result:
    run_dir.mkdir(parents=True, exist_ok=True)
    result = Tier3Result(
        design=placement.design,
        objective_id=placement.objective_id,
        seed=placement.seed,
        status="failed",
        failure_stage=failure_stage,
        routed_wirelength=None,
        grt_overflow=None,
        drc_count=None,
        wns=None,
        tns=None,
        power=None,
        area=None,
        runtime_seconds=runtime_seconds,
        def_path=str(Path(placement.def_path).resolve()),
        metrics_path=None,
        run_dir=str(run_dir),
        log_path=str(log_path) if log_path is not None else None,
        returncode=returncode,
        raw_metrics={"error": message},
    )
    _write_json(asdict(result), run_dir / "run_summary.json")
    return result


def _recover_existing_partial_result(
    *,
    previous: Tier3Result,
    placement: Tier3Placement,
    design: SharedPanelDesign | None,
    chipbench_root: Path,
    run_dir: Path,
) -> Tier3Result | None:
    if design is None:
        return None
    evaluate_name = (
        f"coevop_{_safe_name(placement.design)}_"
        f"{_safe_name(placement.objective_id)}_s{placement.seed}"
    )
    return _failed_or_partial_result(
        placement,
        run_dir,
        previous.failure_stage or "metric_extraction",
        str(previous.raw_metrics.get("error") or "previous run failed"),
        chipbench_root=chipbench_root,
        evaluate_name=evaluate_name,
        runtime_seconds=previous.runtime_seconds,
        returncode=previous.returncode,
        log_path=previous.log_path,
    )


def _open_store(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        create table if not exists tier3_results (
            design text not null,
            objective_id text not null,
            seed integer not null,
            status text not null,
            failure_stage text,
            routed_wirelength real,
            grt_overflow real,
            drc_count real,
            wns real,
            tns real,
            power real,
            area real,
            runtime_seconds real,
            def_path text not null,
            metrics_path text,
            run_dir text not null,
            log_path text,
            returncode integer,
            raw_metrics_json text not null,
            updated_at text not null default current_timestamp,
            primary key (design, objective_id, seed)
        )
        """
    )
    return conn


def _upsert_result(conn: sqlite3.Connection, result: Tier3Result) -> None:
    conn.execute(
        """
        insert into tier3_results (
            design, objective_id, seed, status, failure_stage,
            routed_wirelength, grt_overflow, drc_count, wns, tns,
            power, area, runtime_seconds, def_path, metrics_path, run_dir,
            log_path, returncode, raw_metrics_json
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(design, objective_id, seed) do update set
            status=excluded.status,
            failure_stage=excluded.failure_stage,
            routed_wirelength=excluded.routed_wirelength,
            grt_overflow=excluded.grt_overflow,
            drc_count=excluded.drc_count,
            wns=excluded.wns,
            tns=excluded.tns,
            power=excluded.power,
            area=excluded.area,
            runtime_seconds=excluded.runtime_seconds,
            def_path=excluded.def_path,
            metrics_path=excluded.metrics_path,
            run_dir=excluded.run_dir,
            log_path=excluded.log_path,
            returncode=excluded.returncode,
            raw_metrics_json=excluded.raw_metrics_json,
            updated_at=current_timestamp
        """,
        (
            result.design,
            result.objective_id,
            result.seed,
            result.status,
            result.failure_stage,
            result.routed_wirelength,
            result.grt_overflow,
            result.drc_count,
            result.wns,
            result.tns,
            result.power,
            result.area,
            result.runtime_seconds,
            result.def_path,
            result.metrics_path,
            result.run_dir,
            result.log_path,
            result.returncode,
            json.dumps(result.raw_metrics, sort_keys=True),
        ),
    )
    conn.commit()


def _build_report(results: list[Tier3Result], rankings: list[Any]) -> str:
    lines = [
        "# Tier-3 OpenROAD/ChiPBench Report",
        "",
        "## Run Status",
        f"- Total cells: {len(results)}",
        f"- Successes: {sum(1 for result in results if result.status == 'success')}",
        f"- Partial metrics: {sum(1 for result in results if result.status == 'partial')}",
        f"- Failures: {sum(1 for result in results if result.status != 'success')}",
        "",
        "Partial metrics are recovered from the latest completed OpenROAD stage and are not final routed PPA.",
        "",
        "## Rank-Based Objective Summary",
    ]
    if rankings:
        for ranking in rankings[:10]:
            lines.append(
                "- "
                f"{ranking.objective_id}: average_rank={ranking.average_rank:.4g}, "
                f"structural_failures={ranking.structural_failure_count}"
            )
    else:
        lines.append("- No comparable routed results.")
    return "\n".join(lines).rstrip() + "\n"


def _result_csv_row(result: Tier3Result) -> dict[str, Any]:
    row = {column: getattr(result, column, None) for column in TIER3_TABLE_COLUMNS}
    row["metrics_stage"] = _metrics_stage(result)
    row["partial_metrics_available"] = result.status == "partial"
    row["estimated_wirelength"] = _estimated_wirelength(result)
    return row


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


def _flatten_metrics(payload: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(payload, dict):
        flattened: dict[str, Any] = {}
        for key, value in payload.items():
            nested_key = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_metrics(value, nested_key))
        return flattened
    return {prefix: payload}


def _find_metric(flattened: dict[str, Any], aliases: list[str]) -> float | None:
    normalized_aliases = [_normalize_key(alias) for alias in aliases]
    exact_candidates = []
    fuzzy_candidates = []
    for key, value in flattened.items():
        numeric = _float_or_none(value)
        if numeric is None:
            continue
        normalized = _normalize_key(key)
        if normalized in normalized_aliases:
            exact_candidates.append(numeric)
        elif any(alias in normalized for alias in normalized_aliases):
            fuzzy_candidates.append(numeric)
    if exact_candidates:
        return exact_candidates[0]
    return fuzzy_candidates[0] if fuzzy_candidates else None


def _find_chipbench_stage_dir(chipbench_root: Path, evaluate_name: str) -> Path | None:
    logs_root = chipbench_root / "flow" / "logs"
    if not logs_root.is_dir():
        return None
    matches = [path for path in logs_root.rglob(evaluate_name) if path.is_dir()]
    if not matches:
        return None
    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0]


def _find_sibling_report_dir(stage_dir: Path) -> Path:
    parts = list(stage_dir.parts)
    try:
        logs_index = parts.index("logs")
    except ValueError:
        return stage_dir.parent
    parts[logs_index] = "reports"
    return Path(*parts)


def _parse_grt_log_summary(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "path": str(path),
        "global_route_started": "global_route" in text,
        "final_congestion_report_available": "Final congestion report" in text,
        "extra_iterations_started": "Running extra iterations to remove overflow" in text,
        "clock_net_count": _regex_float(text, r"\[INFO GRT-0019\] Found\s+([0-9]+)\s+clock nets"),
        "max_net_degree": _regex_float(text, r"\[INFO GRT-0002\] Maximum degree:\s+([0-9]+)"),
        "macro_count": _regex_float(text, r"\[INFO GRT-0003\] Macros:\s+([0-9]+)"),
        "blockage_count": _regex_float(text, r"\[INFO GRT-0004\] Blockages:\s+([0-9]+)"),
        "total_wirelength": _regex_float(
            text, r"\[INFO GRT-0018\] Total wirelength:\s+([0-9.]+)"
        ),
        "final_max_horizontal_overflow": _regex_float(
            text,
            r"^Total\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+%\s+([0-9.]+)\s*/\s*[0-9.]+\s*/\s*[0-9.]+",
        ),
        "final_max_vertical_overflow": _regex_float(
            text,
            r"^Total\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+%\s+[0-9.]+\s*/\s*([0-9.]+)\s*/\s*[0-9.]+",
        ),
        "final_total_overflow": _regex_float(
            text,
            r"^Total\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+%\s+[0-9.]+\s*/\s*[0-9.]+\s*/\s*([0-9.]+)",
        ),
    }


def _parse_grt_congestion_report(path: Path) -> dict[str, Any]:
    import re

    if not path.is_file():
        return {"path": str(path), "available": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    overflow_values = [
        float(match.group(1))
        for match in re.finditer(r"\boverflow:([0-9]+(?:\.[0-9]+)?)", text)
    ]
    horizontal = len(re.findall(r"violation type:\s*Horizontal congestion", text))
    vertical = len(re.findall(r"violation type:\s*Vertical congestion", text))
    return {
        "path": str(path),
        "available": True,
        "violation_count": len(overflow_values),
        "horizontal_violation_count": horizontal,
        "vertical_violation_count": vertical,
        "total_overflow": sum(overflow_values) if overflow_values else None,
        "max_overflow": max(overflow_values) if overflow_values else None,
    }


def _regex_float(text: str, pattern: str) -> float | None:
    import re

    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return None
    return _float_or_none(match.group(1))


def _metrics_stage(result: Tier3Result) -> str | None:
    raw = result.raw_metrics or {}
    if isinstance(raw, dict):
        stage = raw.get("metrics_stage")
        if isinstance(stage, str):
            return stage
    return "final_route" if result.status == "success" else None


def _estimated_wirelength(result: Tier3Result | None) -> float | None:
    if result is None:
        return None
    raw = result.raw_metrics or {}
    if not isinstance(raw, dict):
        return None
    partial = raw.get("coevop_partial_metrics")
    if isinstance(partial, dict):
        return _float_or_none(partial.get("estimated_wirelength"))
    return None


def _load_result_summary(path: Path) -> Tier3Result:
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed = set(Tier3Result.__dataclass_fields__)
    sanitized = {key: value for key, value in payload.items() if key in allowed}
    defaults = {
        "routed_wirelength": None,
        "grt_overflow": None,
        "drc_count": None,
        "wns": None,
        "tns": None,
        "power": None,
        "area": None,
        "runtime_seconds": None,
        "metrics_path": None,
        "log_path": None,
        "returncode": None,
        "raw_metrics": {},
    }
    for key, value in defaults.items():
        sanitized.setdefault(key, value)
    return Tier3Result(**sanitized)


def _normalize_key(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _delta_pct(value: Any, baseline: Any) -> float | None:
    if not (_is_finite(value) and _is_finite(baseline)):
        return None
    baseline_value = float(baseline)
    if baseline_value == 0.0:
        return None
    return 100.0 * (float(value) - baseline_value) / abs(baseline_value)


def _gain(value: Any, baseline: Any) -> float | None:
    if not (_is_finite(value) and _is_finite(baseline)):
        return None
    return float(value) - float(baseline)


def _negate(value: Any) -> float | None:
    numeric = _float_or_none(value)
    return -numeric if numeric is not None else None


def _float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _first_float(*values: Any) -> float | None:
    for value in values:
        numeric = _float_or_none(value)
        if numeric is not None:
            return numeric
    return None


def _is_finite(value: Any) -> bool:
    return _float_or_none(value) is not None


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
