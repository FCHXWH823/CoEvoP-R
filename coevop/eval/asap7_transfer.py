"""ASAP7 cross-node transfer evaluation helpers.

This module keeps ASAP7 as an external OpenROAD/ChiPBench platform dependency.
The code validates that dependency, prepares DREAMPlace panel files from
verified base configs, and runs a zero-shot transfer objective through
DREAMPlace and OpenROAD.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from coevop.backends.dreamplace import is_patch_applied
from coevop.eval.shared_panel import load_shared_panel
from coevop.eval.tier2_dreamplace import run_tier2_dreamplace
from coevop.eval.tier3_openroad import run_tier3_openroad


ASAP7_PLATFORM_REL = Path("flow") / "platforms" / "asap7"
ASAP7_PLATFORM_CONFIG_REL = ASAP7_PLATFORM_REL / "config.mk"
ASAP7_DESIGN_NAMES = ("gcd", "ibex", "ariane")
MANIFEST_ROW_FIELDS = [
    "family",
    "design",
    "library",
    "method",
    "method_group",
    "seed",
    "source",
    "placement_hpwl",
    "placement_overflow",
    "detailed_route_wirelength",
    "grt_overflow",
    "drc_count",
    "wns",
    "tns",
    "runtime_min",
    "gpu_hours",
    "status",
]


@dataclass(frozen=True)
class ASAP7Design:
    name: str
    chipbench_config: str
    dreamplace_config: str


@dataclass(frozen=True)
class SourceObjectiveResolution:
    objective_path: str | None
    status: str
    search_root: str
    design: str
    candidates: list[dict[str, Any]]
    message: str | None = None


def asap7_designs(
    *,
    chipbench_root: str | Path,
    dreamplace_config_dir: str | Path,
    names: list[str] | None = None,
) -> list[ASAP7Design]:
    selected = names or list(ASAP7_DESIGN_NAMES)
    unknown = sorted(set(selected) - set(ASAP7_DESIGN_NAMES))
    if unknown:
        raise ValueError(f"unknown ASAP7 design(s): {', '.join(unknown)}")
    chipbench_root = Path(chipbench_root)
    dreamplace_config_dir = Path(dreamplace_config_dir)
    return [
        ASAP7Design(
            name=name,
            chipbench_config=str(
                chipbench_root / "flow" / "designs" / "asap7" / name / "config.mk"
            ),
            dreamplace_config=str(dreamplace_config_dir / f"{name}.json"),
        )
        for name in selected
    ]


def check_asap7_panel(
    *,
    chipbench_root: str | Path,
    dreamplace_root: str | Path,
    dreamplace_config_dir: str | Path,
    run_dir: str | Path,
    designs: list[str] | None = None,
    skip_make_dry_run: bool = False,
) -> dict[str, Any]:
    chipbench_root = Path(chipbench_root)
    dreamplace_root = Path(dreamplace_root)
    run_root = Path(run_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    selected = asap7_designs(
        chipbench_root=chipbench_root,
        dreamplace_config_dir=dreamplace_config_dir,
        names=designs,
    )
    platform_config = chipbench_root / ASAP7_PLATFORM_CONFIG_REL
    bootstrap = {
        "platform_config": str(platform_config),
        "exists": platform_config.is_file(),
        "required_manual_fix": None,
    }
    if not platform_config.is_file():
        bootstrap["required_manual_fix"] = (
            "Copy or symlink OpenROAD-flow-scripts/flow/platforms/asap7/ "
            f"to {chipbench_root / ASAP7_PLATFORM_REL} before running ASAP7."
        )

    design_checks = []
    for design in selected:
        chipbench_config = Path(design.chipbench_config)
        dreamplace_config = Path(design.dreamplace_config)
        item: dict[str, Any] = {
            "design": design.name,
            "chipbench_config": str(chipbench_config),
            "chipbench_config_exists": chipbench_config.is_file(),
            "dreamplace_config": str(dreamplace_config),
            "dreamplace_config_exists": dreamplace_config.is_file(),
            "make_dry_run": None,
        }
        if (
            not skip_make_dry_run
            and platform_config.is_file()
            and chipbench_config.is_file()
        ):
            item["make_dry_run"] = _run_make_dry_run(chipbench_root, chipbench_config)
        design_checks.append(item)

    openroad = _command_metadata(["openroad", "-version"])
    payload = {
        "kind": "asap7_panel_check",
        "chipbench_root": str(chipbench_root),
        "dreamplace_root": str(dreamplace_root),
        "dreamplace_config_dir": str(Path(dreamplace_config_dir)),
        "bootstrap": bootstrap,
        "openroad": openroad,
        "dreamplace_patch_applied": is_patch_applied(dreamplace_root),
        "designs": design_checks,
    }
    payload["ok"] = (
        bool(bootstrap["exists"])
        and openroad.get("returncode") == 0
        and bool(payload["dreamplace_patch_applied"])
        and all(item["chipbench_config_exists"] for item in design_checks)
        and all(item["dreamplace_config_exists"] for item in design_checks)
        and all(
            item["make_dry_run"] is None or item["make_dry_run"].get("returncode") == 0
            for item in design_checks
        )
    )
    _write_json(payload, run_root / "asap7_panel_check.json")
    return payload


def generate_asap7_base_configs(
    *,
    chipbench_root: str | Path,
    output_dir: str | Path,
    designs: list[str] | None = None,
    flow_variant: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    chipbench_root = Path(chipbench_root)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    selected = asap7_designs(
        chipbench_root=chipbench_root,
        dreamplace_config_dir=output_root,
        names=designs,
    )
    platform_root = chipbench_root / ASAP7_PLATFORM_REL
    platform_config = chipbench_root / ASAP7_PLATFORM_CONFIG_REL
    results = []
    for design in selected:
        config_mk = Path(design.chipbench_config)
        variables = _parse_make_config(config_mk) if config_mk.is_file() else {}
        raw_def_path = _discover_asap7_def(
            chipbench_root=chipbench_root,
            design_name=design.name,
            variables=variables,
            flow_variant=flow_variant,
        )
        def_path = (
            _sanitize_asap7_def_for_dreamplace(raw_def_path)
            if raw_def_path is not None
            else None
        )
        lef_paths = _discover_asap7_lefs(
            chipbench_root=chipbench_root,
            design_name=design.name,
            platform_root=platform_root,
        )
        output_path = output_root / f"{design.name}.json"
        item: dict[str, Any] = {
            "design": design.name,
            "chipbench_config": str(config_mk),
            "platform_config": str(platform_config),
            "platform_config_exists": platform_config.is_file(),
            "raw_def_path": str(raw_def_path) if raw_def_path else None,
            "def_path": str(def_path) if def_path else None,
            "lef_count": len(lef_paths),
            "lef_examples": [str(path) for path in lef_paths[:10]],
            "output": str(output_path),
            "status": "pending",
        }
        if not platform_config.is_file():
            item["status"] = "blocked"
            item["reason"] = "ASAP7 platform config is missing"
        elif def_path is None:
            item["status"] = "blocked"
            item["reason"] = "No ORFS/ChiPBench DEF output found for base config generation"
        elif not lef_paths:
            item["status"] = "blocked"
            item["reason"] = "No ASAP7 LEF files found"
        elif output_path.exists() and not overwrite:
            item["status"] = "exists"
        else:
            payload = _dreamplace_base_payload(
                design_name=design.name,
                def_path=def_path,
                lef_paths=lef_paths,
                variables=variables,
                chipbench_root=chipbench_root,
                platform_root=platform_root,
            )
            _write_json(payload, output_path)
            item["status"] = "written"
        results.append(item)
    summary = {
        "kind": "asap7_base_config_generation",
        "chipbench_root": str(chipbench_root),
        "output_dir": str(output_root),
        "flow_variant": flow_variant,
        "designs": results,
        "ok": all(item["status"] in {"written", "exists"} for item in results),
    }
    _write_json(summary, output_root / "base_config_generation_summary.json")
    return summary


def resolve_source_objective(
    *,
    source_root: str | Path,
) -> SourceObjectiveResolution:
    root = Path(source_root)
    candidates = []
    if root.exists():
        named = list(root.rglob("frozen_nangate45_champion.json"))
        if not named:
            named = [
                path for path in root.rglob("robust_best_objective.json")
                if "chipbench_controller" in str(path).lower()
            ]
        for path in named:
            stat = path.stat()
            candidates.append(
                {
                    "path": str(path),
                    "mtime_ns": stat.st_mtime_ns,
                    "size": stat.st_size,
                    "sha256": _file_sha256(path),
                }
            )
    candidates.sort(key=lambda item: (int(item["mtime_ns"]), str(item["path"])), reverse=True)
    if not candidates:
        return SourceObjectiveResolution(
            objective_path=None,
            status="missing",
            search_root=str(root),
            design="nangate45",
            candidates=[],
            message=f"No frozen Nangate45 champion was found under {root}",
        )
    latest_mtime = candidates[0]["mtime_ns"]
    latest = [item for item in candidates if item["mtime_ns"] == latest_mtime]
    if len(latest) != 1:
        hashes = {str(item.get("sha256")) for item in latest}
        if len(hashes) == 1:
            return SourceObjectiveResolution(
                objective_path=str(latest[0]["path"]),
                status="resolved",
                search_root=str(root),
                design="nangate45",
                candidates=candidates,
                message=(
                    "Multiple latest robust objective copies were found, but "
                    "their contents are identical; resolved to the first "
                    "deterministic path."
                ),
            )
        return SourceObjectiveResolution(
            objective_path=None,
            status="ambiguous",
            search_root=str(root),
            design="nangate45",
            candidates=candidates,
            message="Latest robust objective is ambiguous because multiple files share the newest timestamp",
        )
    return SourceObjectiveResolution(
        objective_path=str(latest[0]["path"]),
        status="resolved",
        search_root=str(root),
        design="nangate45",
        candidates=candidates,
    )


def resolve_ethernet_source_objective(
    *,
    source_root: str | Path,
    design: str = "ethernet",
) -> SourceObjectiveResolution:
    """Compatibility alias for older callers."""

    return resolve_source_objective(source_root=source_root)


def check_asap7_source_objective(
    *,
    source_root: str | Path,
    output: str | Path | None = None,
) -> dict[str, Any]:
    resolution = resolve_source_objective(source_root=source_root)
    payload = asdict(resolution)
    if output:
        _write_json(payload, Path(output))
    return payload


def run_asap7_transfer(
    *,
    run_dir: str | Path,
    dreamplace_root: str | Path,
    chipbench_root: str | Path,
    dreamplace_config_dir: str | Path,
    source_objective: str | Path | None = None,
    source_root: str | Path = "runs",
    designs: list[str] | None = None,
    seeds: list[int] | None = None,
    iterations: int = 150,
    gpu: int | None = 1,
    timeout_seconds: int = 7200,
    openroad_timeout_seconds: int | None = None,
    resume: bool = False,
    retry_failed: bool = False,
    dry_run: bool = False,
    adaptation_objectives: list[str | Path] | None = None,
) -> dict[str, Any]:
    run_root = Path(run_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    seeds = seeds or [1000, 1001, 1002]
    chipbench_root = Path(chipbench_root)
    dreamplace_root = Path(dreamplace_root)
    dreamplace_config_dir = Path(dreamplace_config_dir)
    selected = asap7_designs(
        chipbench_root=chipbench_root,
        dreamplace_config_dir=dreamplace_config_dir,
        names=designs,
    )

    if source_objective:
        provided_source = Path(source_objective)
        source_resolution = SourceObjectiveResolution(
            objective_path=str(provided_source) if provided_source.is_file() else None,
            status="provided" if provided_source.is_file() else "missing",
            search_root=str(source_root),
            design="nangate45",
            candidates=[],
            message=None if provided_source.is_file() else f"{provided_source} does not exist",
        )
    else:
        source_resolution = resolve_source_objective(source_root=source_root)
    _write_json(asdict(source_resolution), run_root / "source_objective_resolution.json")

    panel_check = check_asap7_panel(
        chipbench_root=chipbench_root,
        dreamplace_root=dreamplace_root,
        dreamplace_config_dir=dreamplace_config_dir,
        run_dir=run_root / "checks",
        designs=[design.name for design in selected],
        skip_make_dry_run=True,
    )
    if source_resolution.objective_path is None:
        return _blocked_transfer_summary(
            run_root,
            "source_objective",
            source_resolution.message or "source objective could not be resolved",
            panel_check,
            source_resolution,
        )
    if not panel_check["bootstrap"]["exists"]:
        return _blocked_transfer_summary(
            run_root,
            "asap7_platform_bootstrap",
            str(panel_check["bootstrap"]["required_manual_fix"]),
            panel_check,
            source_resolution,
        )
    if not dry_run and not panel_check.get("ok"):
        return _blocked_transfer_summary(
            run_root,
            "asap7_panel_check",
            "ASAP7 panel check failed; inspect checks/asap7_panel_check.json",
            panel_check,
            source_resolution,
        )

    prepared_objectives = _prepare_transfer_objectives(
        source_objective=Path(source_resolution.objective_path),
        adaptation_objectives=adaptation_objectives or [],
        output_dir=run_root / "objectives",
    )
    panel_path = _write_asap7_shared_panel(
        selected,
        run_root / "asap7_panel.toml",
        seeds=seeds,
        iterations=iterations,
        gpu=gpu,
        timeout_seconds=timeout_seconds,
        openroad_timeout_seconds=openroad_timeout_seconds,
    )
    post_route_panel_path = _write_asap7_shared_panel(
        selected,
        run_root / "asap7_post_route_panel.toml",
        seeds=seeds,
        iterations=iterations,
        gpu=gpu,
        timeout_seconds=(
            openroad_timeout_seconds if openroad_timeout_seconds is not None else timeout_seconds
        ),
        openroad_timeout_seconds=openroad_timeout_seconds,
    )
    panel_manifest = {
        "library": "asap7",
        "family": "asap7_transfer",
        "designs": [asdict(design) for design in selected],
        "seeds": seeds,
        "iterations": iterations,
        "source_objective": asdict(source_resolution),
    }
    _write_json(panel_manifest, run_root / "panel_manifest.json")

    if dry_run:
        manifest_path = write_manifest_rows([], run_root / "manifest_rows.jsonl")
        summary = {
            "status": "prepared",
            "dry_run": True,
            "run_dir": str(run_root),
            "panel": str(panel_path),
            "post_route_panel": str(post_route_panel_path),
            "source_objective": source_resolution.objective_path,
            "objectives": prepared_objectives,
            "manifest_rows": str(manifest_path),
            "panel_check": panel_check,
        }
        _write_json(summary, run_root / "summary.json")
        (run_root / "asap7_transfer_report.md").write_text(
            _build_asap7_report(summary, [], [], {}),
            encoding="utf-8",
        )
        return summary

    tier2_dir = run_root / "post_placement"
    tier2_summary = run_tier2_dreamplace(
        panel_path=panel_path,
        objective_paths=[item["path"] for item in prepared_objectives],
        dreamplace_root=dreamplace_root,
        run_dir=tier2_dir,
        store_path=tier2_dir / "tier2.sqlite",
        resume=resume,
        retry_failed=retry_failed,
        include_default=True,
        include_custom_default=False,
        require_output_artifact=True,
        require_def_output=True,
        disable_legalization=False,
    )
    placements_path = _placements_from_tier2_comparison(
        tier2_summary=tier2_summary,
        output=run_root / "post_route" / "placements.json",
    )
    tier3_summary = run_tier3_openroad(
        panel_path=post_route_panel_path,
        placements_path=placements_path,
        chipbench_root=chipbench_root,
        run_dir=run_root / "post_route",
        store_path=run_root / "post_route" / "tier3.sqlite",
        baseline_objective_id="default",
        resume=resume,
        retry_failed=retry_failed,
    )
    placement_rows = _read_csv_dicts(Path(str(tier2_summary["comparison_csv"])))
    post_route_rows = _read_csv_dicts(Path(str(tier3_summary["comparison_csv"])))
    manifest_rows = build_manifest_rows(
        placement_rows=placement_rows,
        post_route_rows=post_route_rows,
        source="our_run",
    )
    manifest_path = write_manifest_rows(manifest_rows, run_root / "manifest_rows.jsonl")
    success = classify_zero_shot_success(
        placement_rows=placement_rows,
        post_route_rows=post_route_rows,
        objective_id="ours_zero_shot",
    )
    rankings_path = _write_json(success, run_root / "rankings.json")
    summary = {
        "status": "completed",
        "dry_run": False,
        "run_dir": str(run_root),
        "panel": str(panel_path),
        "post_route_panel": str(post_route_panel_path),
        "source_objective": source_resolution.objective_path,
        "objectives": prepared_objectives,
        "post_placement": tier2_summary,
        "post_route": tier3_summary,
        "manifest_rows": str(manifest_path),
        "rankings": str(rankings_path),
        "zero_shot_success": success,
        "panel_check": panel_check,
    }
    _write_json(summary, run_root / "summary.json")
    (run_root / "asap7_transfer_report.md").write_text(
        _build_asap7_report(summary, placement_rows, post_route_rows, success),
        encoding="utf-8",
    )
    return summary


def build_manifest_rows(
    *,
    placement_rows: list[dict[str, Any]],
    post_route_rows: list[dict[str, Any]],
    source: str,
) -> list[dict[str, Any]]:
    route_by_key = {
        (row.get("design"), row.get("objective_id"), str(row.get("seed"))): row
        for row in post_route_rows
    }
    rows = []
    for placement in placement_rows:
        key = (
            placement.get("design"),
            placement.get("objective_id"),
            str(placement.get("seed")),
        )
        grt = route_by_key.get(key, {})
        runtime_seconds = _sum_optional_numbers(
            _float_or_none(placement.get("runtime_seconds")),
            _float_or_none(grt.get("runtime_seconds")),
        )
        rows.append(
            {
                "family": "asap7_transfer",
                "design": placement.get("design"),
                "library": "asap7",
                "method": placement.get("objective_id"),
                "method_group": _method_group(str(placement.get("objective_id"))),
                "seed": _int_or_none(placement.get("seed")),
                "source": source,
                "placement_hpwl": _float_or_none(placement.get("hpwl")),
                "placement_overflow": _float_or_none(placement.get("overflow")),
                "detailed_route_wirelength": _float_or_none(grt.get("routed_wirelength")),
                "grt_overflow": _float_or_none(grt.get("grt_overflow")),
                "drc_count": _float_or_none(grt.get("drc_count")),
                "wns": _float_or_none(grt.get("wns")),
                "tns": _float_or_none(grt.get("tns")),
                "runtime_min": (
                    runtime_seconds / 60.0 if runtime_seconds is not None else None
                ),
                "gpu_hours": (
                    runtime_seconds / 3600.0 if runtime_seconds is not None else None
                ),
                "status": _combined_status(
                    str(placement.get("status") or ""),
                    str(grt.get("status") or ""),
                ),
            }
        )
    return rows


def write_manifest_rows(rows: list[dict[str, Any]], output: str | Path) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            normalized = {field: row.get(field) for field in MANIFEST_ROW_FIELDS}
            handle.write(json.dumps(normalized, sort_keys=True) + "\n")
    return output_path


def classify_zero_shot_success(
    *,
    placement_rows: list[dict[str, Any]],
    post_route_rows: list[dict[str, Any]],
    objective_id: str = "ours_zero_shot",
) -> dict[str, Any]:
    metrics = {
        "placement_hpwl": (placement_rows, "hpwl_delta_pct"),
        "detailed_route_wirelength": (post_route_rows, "routed_wirelength_delta_pct"),
        "grt_overflow": (post_route_rows, "grt_overflow_delta_pct"),
    }
    designs = sorted(
        {
            str(row.get("design"))
            for rows, _ in metrics.values()
            for row in rows
            if row.get("design")
        }
    )
    design_results = []
    passed_designs = []
    for design in designs:
        metric_summaries = {}
        design_pass = False
        for metric_name, (rows, field) in metrics.items():
            values = [
                _float_or_none(row.get(field))
                for row in rows
                if row.get("design") == design and row.get("objective_id") == objective_id
            ]
            values = [value for value in values if value is not None]
            summary = _mean_ci(values)
            metric_summaries[metric_name] = summary
            if summary.get("n", 0) >= 2 and summary.get("ci_high") is not None:
                if float(summary["ci_high"]) < 0.0:
                    design_pass = True
        if design_pass:
            passed_designs.append(design)
        design_results.append(
            {
                "design": design,
                "passed": design_pass,
                "metrics": metric_summaries,
            }
        )
    return {
        "objective_id": objective_id,
        "success": len(passed_designs) >= 2,
        "passed_design_count": len(passed_designs),
        "required_passed_design_count": 2,
        "passed_designs": passed_designs,
        "designs": design_results,
        "rule": (
            "Zero-shot succeeds if at least two ASAP7 designs improve with 95% CI "
            "not crossing zero on placement HPWL, routed wirelength, or routing "
            "overflow versus native DREAMPlace default."
        ),
    }


def _run_make_dry_run(chipbench_root: Path, config_mk: Path) -> dict[str, Any]:
    command = [
        "bash",
        "-lc",
        (
            "cd flow && "
            f"make -n DESIGN_CONFIG={_sh_quote(str(config_mk.resolve()))} "
            "FLOW_VARIANT=coevop_asap7_check do-floorplan do-place-global"
        ),
    ]
    return _command_metadata(command, cwd=chipbench_root, timeout=120)


def _parse_make_config(path: Path) -> dict[str, str]:
    variables: dict[str, str] = {}
    if not path.is_file():
        return variables
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        if ":=" in line:
            key, value = line.split(":=", 1)
        elif "?=" in line:
            key, value = line.split("?=", 1)
        else:
            key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key.removeprefix("export ").strip()
        variables[key] = value.strip()
    return variables


def _discover_asap7_def(
    *,
    chipbench_root: Path,
    design_name: str,
    variables: dict[str, str],
    flow_variant: str | None,
) -> Path | None:
    design_name_from_config = variables.get("DESIGN_NAME", design_name)
    roots = []
    results_root = chipbench_root / "flow" / "results" / "asap7"
    if flow_variant:
        roots.append(results_root / design_name / flow_variant)
        roots.append(results_root / design_name_from_config / flow_variant)
    roots.extend([results_root / design_name, results_root / design_name_from_config])
    preferred_names = [
        "2_1_floorplan.def",
        "2_floorplan.def",
        "3_1_place_gp.def",
        "3_3_place_gp.def",
        "floorplan.def",
    ]
    for root in roots:
        for name in preferred_names:
            candidate = root / name
            if candidate.is_file():
                return candidate.resolve()
    candidates = []
    for root in roots:
        if root.exists():
            candidates.extend(path for path in root.rglob("*.def") if path.is_file())
    if not candidates:
        return None
    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, str(path)), reverse=True)
    return candidates[0].resolve()


_TIE_COMPONENT_RE = re.compile(r"^(\s*-\s+\S+\s+TIE(?:HI|LO)\S+)(\s*;\s*)$")


def _sanitize_asap7_def_for_dreamplace(def_path: Path) -> Path:
    """Add fixed placement to unplaced ASAP7 tie cells for DREAMPlace parsing.

    OpenROAD can leave tie-high/tie-low components without placement status in
    early floorplan DEFs. DREAMPlace treats these masters as non-CORE cells and
    requires them to be FIXED or PLACED. Keep the raw DEF intact and write a
    deterministic sibling used only by generated DREAMPlace configs.
    """

    output = def_path.with_suffix(def_path.suffix + ".coevop_dreamplace.def")
    text = def_path.read_text(encoding="utf-8", errors="replace")
    changed = 0
    lines = []
    for line in text.splitlines():
        match = _TIE_COMPONENT_RE.match(line)
        if match:
            line = f"{match.group(1)} + FIXED ( 0 0 ) N{match.group(2)}"
            changed += 1
        lines.append(line)
    if changed:
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output.resolve()
    return def_path.resolve()


def _discover_asap7_lefs(
    *,
    chipbench_root: Path,
    design_name: str,
    platform_root: Path,
) -> list[Path]:
    roots = [
        platform_root,
        chipbench_root / "flow" / "designs" / "asap7" / design_name,
    ]
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(path.resolve() for path in root.rglob("*.lef") if path.is_file())
    return sorted(dict.fromkeys(paths), key=lambda path: str(path))


def _dreamplace_base_payload(
    *,
    design_name: str,
    def_path: Path,
    lef_paths: list[Path],
    variables: dict[str, str],
    chipbench_root: Path | None = None,
    platform_root: Path | None = None,
) -> dict[str, Any]:
    target_density = _float_or_none(variables.get("PLACE_DENSITY"))
    portable_roots = _portable_roots(chipbench_root=chipbench_root, platform_root=platform_root)
    return {
        "lef_input": [_portable_external_path(path, portable_roots) for path in lef_paths],
        "def_input": _portable_external_path(def_path, portable_roots),
        "gpu": 1,
        "num_bins_x": 512,
        "num_bins_y": 512,
        "global_place_flag": 1,
        "legalize_flag": 1,
        "detailed_place_flag": 1,
        "routability_opt_flag": 0,
        "timing_opt_flag": 0,
        "target_density": target_density if target_density is not None else 0.70,
        "density_weight": 8e-5,
        "gamma": 8.0,
        "random_seed": 1000,
        "result_dir": "results",
        "asap7_generation_audit": {
            "design_name": design_name,
            "source": "ChiPBench/OpenROAD-flow-scripts ASAP7 DEF + LEF discovery",
            "dbu_unit_note": (
                "Generated config must be smoke-tested; ASAP7 DBU/site/row scale "
                "is validated by asap7-panel-check and first DREAMPlace run."
            ),
            "variables": variables,
        },
    }


def _portable_roots(
    *,
    chipbench_root: Path | None,
    platform_root: Path | None,
) -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = []
    env_flow = os.environ.get("OPENROAD_FLOW_ROOT")
    if env_flow:
        roots.append(("OPENROAD_FLOW_ROOT", Path(env_flow).expanduser().resolve()))
    if chipbench_root is not None:
        roots.append(("CHIPBENCH_ROOT", Path(chipbench_root).expanduser().resolve()))
    if platform_root is not None:
        platform = Path(platform_root).expanduser().resolve()
        if platform.name == "asap7" and platform.parent.name == "platforms":
            flow_root = platform.parents[2] if len(platform.parents) >= 3 else None
            if flow_root is not None:
                roots.append(("OPENROAD_FLOW_ROOT", flow_root.resolve()))
    deduped: list[tuple[str, Path]] = []
    seen: set[tuple[str, str]] = set()
    for env_name, root in roots:
        key = (env_name, str(root))
        if key not in seen:
            deduped.append((env_name, root))
            seen.add(key)
    return sorted(deduped, key=lambda item: len(str(item[1])), reverse=True)


def _portable_external_path(path: Path, roots: list[tuple[str, Path]]) -> str:
    resolved = Path(path).expanduser().resolve()
    for env_name, root in roots:
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        return "${" + env_name + "}/" + relative.as_posix()
    return str(resolved)


def _prepare_transfer_objectives(
    *,
    source_objective: Path,
    adaptation_objectives: list[str | Path],
    output_dir: Path,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = []
    prepared.append(
        _copy_objective_with_id(
            source_objective,
            output_dir / "ours_zero_shot.json",
            "ours_zero_shot",
            "zero_shot_transfer_from_frozen_nangate45_champion",
        )
    )
    for index, raw_path in enumerate(adaptation_objectives, start=1):
        prepared.append(
            _copy_objective_with_id(
                Path(raw_path),
                output_dir / f"ours_adapted_{index}.json",
                f"ours_adapted_{index}",
                "asap7_adapted_objective",
            )
        )
    return prepared


def _copy_objective_with_id(
    source: Path,
    target: Path,
    objective_id: str,
    method_group: str,
) -> dict[str, Any]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["id"] = objective_id
    payload["created_by"] = str(payload.get("created_by") or "asap7_transfer")
    payload["rationale"] = (
        f"{method_group}: {payload.get('rationale', '')}".strip()
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "objective_id": objective_id,
        "path": str(target),
        "source_path": str(source),
        "method_group": method_group,
    }


def _write_asap7_shared_panel(
    designs: list[ASAP7Design],
    output: Path,
    *,
    seeds: list[int],
    iterations: int,
    gpu: int | None,
    timeout_seconds: int,
    openroad_timeout_seconds: int | None,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "seeds = [" + ", ".join(str(seed) for seed in seeds) + "]",
        f"iterations = {iterations}",
        f"timeout_seconds = {timeout_seconds}",
        "log_interval = 10",
        'timing_mode = "none"',
    ]
    if gpu is not None:
        lines.append(f"gpu = {int(gpu)}")
    if openroad_timeout_seconds is not None:
        lines.append(f"# openroad_timeout_seconds = {openroad_timeout_seconds}")
    lines.append("")
    for design in designs:
        lines.extend(
            [
                "[[designs]]",
                f'name = "{design.name}"',
                f'dreamplace_config = "{Path(design.dreamplace_config).resolve().as_posix()}"',
                f'chipbench_config = "{Path(design.chipbench_config).resolve().as_posix()}"',
                'mode = "global"',
                'timing_mode = "none"',
                "",
            ]
        )
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    load_shared_panel(output)
    return output


def _placements_from_tier2_comparison(
    *,
    tier2_summary: dict[str, Any],
    output: Path,
) -> Path:
    comparison_csv = tier2_summary.get("comparison_csv")
    if not comparison_csv:
        raise ValueError("Tier-2 summary does not contain comparison_csv")
    rows = []
    with Path(str(comparison_csv)).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            artifact = str(row.get("output_artifact") or "")
            if row.get("status") == "success" and artifact.lower().endswith(".def"):
                rows.append(
                    {
                        "design": row["design"],
                        "objective_id": row["objective_id"],
                        "seed": int(row["seed"]),
                        "def_path": artifact,
                        "source": "asap7_transfer",
                    }
                )
    return _write_json({"placements": rows}, output)


def _blocked_transfer_summary(
    run_root: Path,
    stage: str,
    message: str,
    panel_check: dict[str, Any],
    source_resolution: SourceObjectiveResolution,
) -> dict[str, Any]:
    summary = {
        "status": "blocked",
        "blocked_stage": stage,
        "message": message,
        "run_dir": str(run_root),
        "panel_check": panel_check,
        "source_objective": asdict(source_resolution),
    }
    write_manifest_rows([], run_root / "manifest_rows.jsonl")
    _write_json(summary, run_root / "summary.json")
    (run_root / "asap7_transfer_report.md").write_text(
        _build_asap7_report(summary, [], [], {}),
        encoding="utf-8",
    )
    return summary


def _build_asap7_report(
    summary: dict[str, Any],
    placement_rows: list[dict[str, Any]],
    post_route_rows: list[dict[str, Any]],
    success: dict[str, Any],
) -> str:
    lines = [
        "# ASAP7 Cross-Node Transfer Experiment",
        "",
        f"- Status: {summary.get('status')}",
        f"- Source objective: `{summary.get('source_objective')}`",
        f"- Custom-default semantics: `term(\"native_objective\")` through the ObjectiveSpec path.",
        "- Designs: gcd, ibex, and ariane in the ASAP7 standard-cell library.",
        "- Evaluation: DREAMPlace placement followed by the OpenROAD post-route flow.",
        "",
    ]
    if summary.get("status") == "blocked":
        lines.extend(
            [
                "## Blocker",
                "",
                f"- Stage: `{summary.get('blocked_stage')}`",
                f"- Message: {summary.get('message')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Artifacts",
            "",
            f"- Panel: `{summary.get('panel')}`",
            f"- Post-route panel: `{summary.get('post_route_panel')}`",
            f"- Manifest rows: `{summary.get('manifest_rows')}`",
            f"- Rankings/success rule: `{summary.get('rankings')}`",
            "",
            "## Zero-Shot Success Rule",
            "",
            "The run succeeds only if `ours_zero_shot` improves placement HPWL, routed wirelength, or routing overflow on at least two of the three ASAP7 designs with 95% CI not crossing zero.",
            "",
        ]
    )
    if success:
        lines.extend(
            [
                f"- Success: {success.get('success')}",
                f"- Passed designs: {success.get('passed_designs')}",
                "",
            ]
        )
    if placement_rows or post_route_rows:
        lines.extend(
            [
                "## Result Tables",
                "",
                f"- Post-placement row count: {len(placement_rows)}",
                f"- Post-route row count: {len(post_route_rows)}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _method_group(objective_id: str) -> str:
    if objective_id == "default":
        return "native_baseline"
    if objective_id == "ours_zero_shot":
        return "ours_zero_shot"
    if objective_id.startswith("ours_adapted"):
        return "ours_adapted"
    return "other"


def _combined_status(placement_status: str, route_status: str) -> str:
    if placement_status != "success":
        return placement_status or "placement_missing"
    if not route_status:
        return "post_route_missing"
    if route_status != "success":
        return route_status
    return "success"


def _mean_ci(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    mean = sum(values) / len(values)
    if len(values) < 2:
        return {"n": len(values), "mean": mean, "ci_low": None, "ci_high": None}
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    margin = 1.96 * math.sqrt(variance) / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "ci_low": mean - margin,
        "ci_high": mean + margin,
    }


def _read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sum_optional_numbers(*values: float | None) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present)


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _command_metadata(
    command: list[str],
    *,
    cwd: str | Path | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"command": command, "error": f"{type(exc).__name__}: {exc}"}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _write_json(payload: Any, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
