"""ChipBench-wide OpenEvolve/DREAMPlace plus post-GRT OpenROAD orchestration."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coevop.backends.dreamplace import is_patch_applied
from coevop.eval.openevolve_tier2 import run_openevolve_tier2
from coevop.eval.shared_panel import load_shared_panel
from coevop.eval.tier2_dreamplace import run_tier2_dreamplace
from coevop.eval.tier3_openroad import run_tier3_openroad


@dataclass(frozen=True)
class ChipBenchSignatureDesign:
    name: str
    dreamplace_config: str
    chipbench_config: str
    reference_def: str | None = None


SIGNATURE_DESIGNS: tuple[ChipBenchSignatureDesign, ...] = (
    ChipBenchSignatureDesign(
        name="bp_fe",
        dreamplace_config="${HOME}/DREAMPlace/install/benchmarks/chipbench/bp_fe/bp_fe.json",
        chipbench_config="${HOME}/ChiPBench/flow/designs/nangate45/bp_fe_top/config.mk",
        reference_def="${HOME}/ChiPBench/bp_fe_placed.def",
    ),
    ChipBenchSignatureDesign(
        name="bp_be",
        dreamplace_config="${HOME}/DREAMPlace/install/benchmarks/chipbench/bp_be/cfg_000.json",
        chipbench_config="${HOME}/ChiPBench/flow/designs/nangate45/bp_be_top/config.mk",
    ),
    ChipBenchSignatureDesign(
        name="swerv_wrapper",
        dreamplace_config="${HOME}/DREAMPlace/install/benchmarks/chipbench/swerv_wrapper/cfg_default.json",
        chipbench_config="${HOME}/ChiPBench/flow/designs/nangate45/swerv_wrapper/config.mk",
    ),
    ChipBenchSignatureDesign(
        name="ethernet",
        dreamplace_config="${HOME}/DREAMPlace/install/benchmarks/chipbench/ethernet/cfg_000.json",
        chipbench_config="${HOME}/ChiPBench/flow/designs/nangate45/ethernet/config.mk",
    ),
    ChipBenchSignatureDesign(
        name="isa_npu",
        dreamplace_config="${HOME}/DREAMPlace/install/benchmarks/chipbench/isa_npu/train_cfg_1446_mixed.json",
        chipbench_config="${HOME}/ChiPBench/flow/designs/nangate45/isa_npu/config.mk",
    ),
)

PRIMARY_BASELINES = ("default", "dreamplace4_default")
SECONDARY_BASELINE_PRESETS = (
    "dreamplace_explicit_wl_density",
    "dreamplace_wawl_electric",
    "dreamplace_wawl_density_bell",
    "dreamplace_lse_density_bell",
    "dreamplace_routing_aware_smoke",
)


def run_chipbench_generalization(
    *,
    run_dir: str | Path,
    dreamplace_root: str | Path,
    chipbench_root: str | Path,
    provider: str = "openai",
    model: str | None = None,
    designs: list[str] | None = None,
    max_iterations: int = 8,
    samples_per_iteration: int = 3,
    search_seeds: list[int] | None = None,
    final_seeds: list[int] | None = None,
    search_iterations: int = 50,
    final_iterations: int = 150,
    gpu: int | None = 1,
    timeout_seconds: int = 3600,
    global_route_args: str = "-allow_congestion -verbose -congestion_iterations 5",
    resume: bool = False,
    retry_failed: bool = False,
    dry_run: bool = False,
    skip_evolution: bool = False,
) -> dict[str, Any]:
    """Run or prepare the ChipBench signature-circuit generalization experiment.

    The function writes per-design OpenEvolve configs and panels under the run
    directory. When not in dry-run mode it executes each design-specific
    OpenEvolve Tier-2 run; each generated config enables GRT-only Tier-3 for
    selected final placements.
    """

    run_root = Path(run_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    if model:
        os.environ["OPENAI_MODEL"] = model

    selected = _select_designs(designs)
    search_seeds = search_seeds or [1000]
    final_seeds = final_seeds or [1000, 1001, 1002]

    audit = _environment_audit(
        dreamplace_root=Path(dreamplace_root),
        chipbench_root=Path(chipbench_root),
        provider=provider,
        model=model or os.environ.get("OPENAI_MODEL"),
        seeds={"search": search_seeds, "final": final_seeds},
    )
    _write_json(audit, run_root / "environment_audit.json")

    configs_root = run_root / "configs"
    finalists_root = run_root / "finalists"
    finalists_root.mkdir(parents=True, exist_ok=True)

    all_post_grt_panel = _write_shared_panel(
        selected,
        configs_root / "chipbench_signature_post_grt.toml",
        seeds=final_seeds,
        iterations=final_iterations,
        gpu=gpu,
        timeout_seconds=timeout_seconds,
        mode="grt_only",
        global_route_args=global_route_args,
    )
    panel_manifest = _panel_manifest(selected, all_post_grt_panel)
    _write_json(panel_manifest, run_root / "panel_manifest.json")

    per_design_summaries: list[dict[str, Any]] = []
    for design in selected:
        design_root = run_root / design.name / "openevolve_tier2"
        design_configs = configs_root / design.name
        search_panel = _write_shared_panel(
            [design],
            design_configs / "search_panel.toml",
            seeds=search_seeds,
            iterations=search_iterations,
            gpu=gpu,
            timeout_seconds=timeout_seconds,
            mode="global",
            global_route_args=global_route_args,
        )
        final_panel = _write_shared_panel(
            [design],
            design_configs / "final_panel.toml",
            seeds=final_seeds,
            iterations=final_iterations,
            gpu=gpu,
            timeout_seconds=timeout_seconds,
            mode="global",
            global_route_args=global_route_args,
        )
        post_grt_panel = _write_shared_panel(
            [design],
            design_configs / "post_grt_panel.toml",
            seeds=final_seeds,
            iterations=final_iterations,
            gpu=gpu,
            timeout_seconds=timeout_seconds,
            mode="grt_only",
            global_route_args=global_route_args,
        )
        config_path = _write_openevolve_config(
            design=design,
            output=design_configs / "openevolve_tier2.toml",
            provider=provider,
            max_iterations=max_iterations,
            samples_per_iteration=samples_per_iteration,
            search_panel=search_panel,
            final_panel=final_panel,
            tier3_panel=post_grt_panel,
        )
        summary: dict[str, Any]
        existing_summary = design_root / "summary.json"
        if skip_evolution and existing_summary.is_file() and not dry_run:
            summary = json.loads(existing_summary.read_text(encoding="utf-8"))
            summary["status"] = summary.get("status", "completed_existing")
            summary["loaded_existing_summary"] = str(existing_summary)
        elif dry_run or skip_evolution:
            summary = {
                "status": "prepared",
                "design": design.name,
                "config": str(config_path),
                "run_dir": str(design_root),
                "dry_run": dry_run,
                "skip_evolution": skip_evolution,
            }
        else:
            start = time.time()
            try:
                summary = run_openevolve_tier2(
                    config_path=config_path,
                    run_dir=design_root,
                    dreamplace_root=dreamplace_root,
                    chipbench_root=chipbench_root,
                    resume=resume,
                    retry_failed=retry_failed,
                )
                summary["status"] = "completed"
            except Exception as exc:  # Keep the panel moving across designs.
                summary = {
                    "status": "failed",
                    "design": design.name,
                    "config": str(config_path),
                    "run_dir": str(design_root),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            summary["runtime_seconds"] = time.time() - start
        summary["design"] = design.name
        per_design_summaries.append(summary)
        _export_design_finalists(
            design_name=design.name,
            design_run_dir=design_root,
            finalists_root=finalists_root,
        )

    _write_json({"designs": per_design_summaries}, run_root / "per_design_run_summary.json")
    dreamplace_metrics = _aggregate_csvs(
        [Path(item["run_dir"]) / "final_tier2" / "comparison_table.csv" for item in per_design_summaries],
        run_root / "dreamplace_metrics.csv",
    )
    post_grt_metrics = _aggregate_csvs(
        [Path(item["run_dir"]) / "tier3_openroad" / "comparison_table.csv" for item in per_design_summaries],
        run_root / "post_grt_metrics.csv",
    )
    comparison_table = _write_combined_comparison(
        dreamplace_metrics,
        post_grt_metrics,
        run_root / "comparison_table.csv",
    )
    summary = {
        "run_dir": str(run_root),
        "environment_audit": str(run_root / "environment_audit.json"),
        "panel_manifest": str(run_root / "panel_manifest.json"),
        "per_design_run_summary": str(run_root / "per_design_run_summary.json"),
        "finalists_dir": str(finalists_root),
        "dreamplace_metrics_csv": str(dreamplace_metrics) if dreamplace_metrics else None,
        "post_grt_metrics_csv": str(post_grt_metrics) if post_grt_metrics else None,
        "comparison_table_csv": str(comparison_table) if comparison_table else None,
        "designs": per_design_summaries,
    }
    report = _build_generalization_report(
        summary=summary,
        audit=audit,
        panel_manifest=panel_manifest,
        run_root=run_root,
    )
    report_path = run_root / "generalization_report.md"
    report_path.write_text(report, encoding="utf-8")
    summary["generalization_report"] = str(report_path)
    _write_json(summary, run_root / "summary.json")
    return summary


def run_chipbench_final_eval_from_existing(
    *,
    source_run: str | Path,
    run_dir: str | Path,
    dreamplace_root: str | Path,
    chipbench_root: str | Path,
    designs: list[str] | None = None,
    seeds: list[int] | None = None,
    iterations: int = 150,
    gpu: int | None = 1,
    timeout_seconds: int = 7200,
    openroad_timeout_seconds: int | None = None,
    global_route_args: str = "-allow_congestion -verbose -congestion_iterations 5",
    resume: bool = False,
    retry_failed: bool = False,
    allow_best_fallback: bool = False,
) -> dict[str, Any]:
    """Run serious final DREAMPlace + GRT-only validation from prior finalists."""

    source_root = Path(source_run)
    run_root = Path(run_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    selected = _select_designs(designs)
    seeds = seeds or [1000, 1001, 1002]
    audit = _environment_audit(
        dreamplace_root=Path(dreamplace_root),
        chipbench_root=Path(chipbench_root),
        provider="reused_finalists",
        model=None,
        seeds={"final": seeds},
    )
    if openroad_timeout_seconds is not None:
        audit["openroad_timeout_seconds"] = openroad_timeout_seconds
    _write_json(audit, run_root / "environment_audit.json")

    design_summaries = []
    for design in selected:
        candidate_path = _candidate_objective_from_source(
            source_root=source_root,
            design_name=design.name,
            allow_best_fallback=allow_best_fallback,
        )
        design_root = run_root / design.name
        if candidate_path is None:
            summary = {
                "design": design.name,
                "status": "skipped",
                "reason": "no robust/pareto finalist objective found in source run",
            }
            _write_json(summary, design_root / "summary.json")
            design_summaries.append(summary)
            continue

        dreamplace_panel = _write_shared_panel(
            [design],
            design_root / "final_panel.toml",
            seeds=seeds,
            iterations=iterations,
            gpu=gpu,
            timeout_seconds=timeout_seconds,
            mode="global",
            global_route_args=global_route_args,
        )
        post_grt_panel = _write_shared_panel(
            [design],
            design_root / "post_grt_panel.toml",
            seeds=seeds,
            iterations=iterations,
            gpu=gpu,
            timeout_seconds=openroad_timeout_seconds
            if openroad_timeout_seconds is not None
            else timeout_seconds,
            mode="grt_only",
            global_route_args=global_route_args,
        )
        tier2_dir = design_root / "tier2_final"
        try:
            tier2_summary = run_tier2_dreamplace(
                panel_path=dreamplace_panel,
                objective_paths=[candidate_path],
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
                output=design_root / "post_grt" / "placements.json",
            )
            tier3_summary = run_tier3_openroad(
                panel_path=post_grt_panel,
                placements_path=placements_path,
                chipbench_root=chipbench_root,
                run_dir=design_root / "post_grt",
                store_path=design_root / "post_grt" / "tier3.sqlite",
                baseline_objective_id="default",
                resume=resume,
                retry_failed=retry_failed,
            )
            summary = {
                "design": design.name,
                "status": "completed",
                "candidate_objective": str(candidate_path),
                "tier2": tier2_summary,
                "post_grt": tier3_summary,
            }
        except Exception as exc:
            summary = {
                "design": design.name,
                "status": "failed",
                "candidate_objective": str(candidate_path),
                "error": f"{type(exc).__name__}: {exc}",
            }
        _write_json(summary, design_root / "summary.json")
        design_summaries.append(summary)

    dreamplace_csv = _aggregate_csvs(
        [
            Path(item.get("tier2", {}).get("comparison_csv", ""))
            for item in design_summaries
            if isinstance(item.get("tier2"), dict)
        ],
        run_root / "dreamplace_metrics.csv",
    )
    post_grt_csv = _aggregate_csvs(
        [
            Path(item.get("post_grt", {}).get("comparison_csv", ""))
            for item in design_summaries
            if isinstance(item.get("post_grt"), dict)
        ],
        run_root / "post_grt" / "metrics.csv",
    )
    comparison_csv = _write_combined_comparison(
        dreamplace_csv,
        post_grt_csv,
        run_root / "comparison_table.csv",
    )
    summary = {
        "run_dir": str(run_root),
        "source_run": str(source_root),
        "environment_audit": str(run_root / "environment_audit.json"),
        "dreamplace_metrics_csv": str(dreamplace_csv) if dreamplace_csv else None,
        "post_grt_metrics_csv": str(post_grt_csv) if post_grt_csv else None,
        "comparison_table_csv": str(comparison_csv) if comparison_csv else None,
        "designs": design_summaries,
        "seeds": seeds,
        "iterations": iterations,
    }
    _write_json(summary, run_root / "summary.json")
    (run_root / "generalization_report.md").write_text(
        _build_final_eval_report(summary),
        encoding="utf-8",
    )
    return summary


def _select_designs(names: list[str] | None) -> list[ChipBenchSignatureDesign]:
    by_name = {design.name: design for design in SIGNATURE_DESIGNS}
    if not names:
        return list(SIGNATURE_DESIGNS)
    selected = []
    for name in names:
        if name not in by_name:
            raise ValueError(f"unknown ChipBench signature design: {name}")
        selected.append(by_name[name])
    return selected


def _candidate_objective_from_source(
    *,
    source_root: Path,
    design_name: str,
    allow_best_fallback: bool,
) -> Path | None:
    design_root = source_root / design_name / "openevolve_tier2"
    candidates = [
        design_root / "robust_best_objective.json",
        design_root / "pareto_best_objective.json",
    ]
    if allow_best_fallback:
        candidates.append(design_root / "best_objective.json")
    for path in candidates:
        if path.is_file():
            return path
    finalists = source_root / "finalists"
    name_candidates = [
        finalists / f"{design_name}_robust_best_objective.json",
        finalists / f"{design_name}_pareto_best_objective.json",
    ]
    if allow_best_fallback:
        name_candidates.append(finalists / f"{design_name}_best_objective.json")
    for path in name_candidates:
        if path.is_file():
            return path
    return None


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
                        "source": "chipbench_final_eval",
                    }
                )
    payload = {"placements": rows}
    return _write_json(payload, output)


def _build_final_eval_report(summary: dict[str, Any]) -> str:
    lines = [
        "# ChipBench Serious Final Validation",
        "",
        f"- Source run: `{summary.get('source_run')}`",
        f"- Seeds: {summary.get('seeds')}",
        f"- DREAMPlace iterations: {summary.get('iterations')}",
        f"- DREAMPlace metrics: `{summary.get('dreamplace_metrics_csv')}`",
        f"- Post-GRT metrics: `{summary.get('post_grt_metrics_csv')}`",
        "",
        "## Results",
        "",
        "| Design | Status | Candidate objective | Post-GRT comparison |",
        "|---|---|---|---|",
    ]
    for item in summary.get("designs", []):
        post_grt = item.get("post_grt") if isinstance(item, dict) else None
        comparison = post_grt.get("comparison_csv") if isinstance(post_grt, dict) else None
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("design")),
                    str(item.get("status")),
                    str(item.get("candidate_objective", "-")),
                    str(comparison or item.get("reason") or item.get("error") or "-"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "This validation reuses previously selected finalist objectives and reruns only native default plus the selected finalist. It does not spend additional LLM calls and does not route every generated candidate.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _write_shared_panel(
    designs: list[ChipBenchSignatureDesign],
    output: Path,
    *,
    seeds: list[int],
    iterations: int,
    gpu: int | None,
    timeout_seconds: int,
    mode: str,
    global_route_args: str,
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
    lines.append("")
    for design in designs:
        lines.extend(
            [
                "[[designs]]",
                f'name = "{design.name}"',
                f'dreamplace_config = "{design.dreamplace_config}"',
                f'chipbench_config = "{design.chipbench_config}"',
                f'mode = "{mode}"',
                'timing_mode = "none"',
                f'global_route_args = "{global_route_args}"',
            ]
        )
        if design.reference_def:
            lines.append(f'reference_def = "{design.reference_def}"')
        lines.append("")
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    # Validate immediately; this catches malformed TOML during tests.
    load_shared_panel(output)
    return output


def _write_openevolve_config(
    *,
    design: ChipBenchSignatureDesign,
    output: Path,
    provider: str,
    max_iterations: int,
    samples_per_iteration: int,
    search_panel: Path,
    final_panel: Path,
    tier3_panel: Path,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    secondary = ", ".join(f'"{name}"' for name in SECONDARY_BASELINE_PRESETS)
    router_background = (Path.cwd() / "prompts" / "router_objective_background.md").as_posix()
    text = f"""# Auto-generated design-specific ChipBench generalization config.
provider = "{provider}"
term_scope = "dreamplace_replacement"
objective_mode = "replacement"
mutation_mode = "diff"

max_iterations = {max_iterations}
samples_per_iteration = {samples_per_iteration}
population_size = 100
archive_size = 20
num_islands = 5
map_elites_enabled = true
feature_bins = 8
feature_dimensions = ["complexity", "mechanism_family", "hpwl_delta_bucket", "overflow_delta_bucket", "opentimer_wns_delta_bucket"]
migration_interval = 20
migration_rate = 0.10
migration_topology = "ring"
checkpoint_interval = 10
num_top_programs = 3
num_diverse_programs = 2
severe_regression_pct = 10.0
seed = 42

target_designs = ["{design.name}"]
evaluate_initial_presets = true
primary_baseline = "default"
parent_policy = "hpwl_safe_only"
structured_feedback = true
router_background = "{router_background}"

hpwl_gate_search_pct = 5.0
hpwl_gate_elite_pct = 3.0
overflow_severe_regression_pct = 10.0
native_default_selection_policy = "prefer_pareto"
native_default_pareto_bonus = 0.02

require_nonzero_routing_term = true
min_replacement_nonbaseline_terms = 2
min_abs_routing_coeff = 0.00000001
route_coeff_cap = 0.01
route_coeff_grid = [-0.00001, -0.000003, -0.000001, -0.0000003, -0.0000001, -0.00000003, -0.00000001, 0.00000001, 0.00000003, 0.0000001, 0.0000003, 0.000001, 0.000003, 0.00001]
density_coeff_grid = [1.0]
wirelength_coeff_grid = [1.0]
max_calibrated_siblings = 8
max_generation_attempts = 4
reject_baseline_mechanism_clones = true
min_baseline_mechanism_distance = 0.35
reject_repeated_negative_mechanisms = true
min_negative_mechanism_distance = 0.55

initial_presets = [{secondary}]

[search]
panel = "{search_panel.resolve().as_posix()}"
baseline_presets = [{secondary}]

[final]
enabled = true
panel = "{final_panel.resolve().as_posix()}"
top_k = 3
baseline_presets = [{secondary}]

[tier3]
enabled = true
panel = "{tier3_panel.resolve().as_posix()}"
top_k = 1
baseline_objective_id = "default"
candidate_policy = "elite_or_aggregate"
baseline_presets = []
"""
    output.write_text(text, encoding="utf-8")
    return output


def _panel_manifest(designs: list[ChipBenchSignatureDesign], post_grt_panel: Path) -> dict[str, Any]:
    return {
        "designs": [design.__dict__ for design in designs],
        "post_grt_panel": str(post_grt_panel),
        "primary_baselines": list(PRIMARY_BASELINES),
        "secondary_baseline_presets": list(SECONDARY_BASELINE_PRESETS),
        "dreamplace4_baseline_policy": (
            "The native default entry is the installed DREAMPlace default flow. "
            "A separate DREAMPlace 4.0 baseline is only counted if the local "
            "installation exposes a distinct runnable 4.0/timing-driven config."
        ),
    }


def _environment_audit(
    *,
    dreamplace_root: Path,
    chipbench_root: Path,
    provider: str,
    model: str | None,
    seeds: dict[str, list[int]],
) -> dict[str, Any]:
    repo_root = Path.cwd()
    windows_repo = Path("/mnt/d/Codes/New Direction/CoEvoPR-Platform")
    audit = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cwd": str(repo_root),
        "provider": provider,
        "resolved_model_requested": model or "provider default",
        "seeds": seeds,
        "coevop_repo": _git_metadata(repo_root),
        "dreamplace_root": str(dreamplace_root),
        "dreamplace_git": _git_metadata(dreamplace_root),
        "dreamplace_patch_applied": is_patch_applied(dreamplace_root),
        "chipbench_root": str(chipbench_root),
        "chipbench_git": _git_metadata(chipbench_root),
        "openroad": _command_metadata(["openroad", "-version"]),
        "gpu": _command_metadata(["bash", "-lc", "nvidia-smi --query-gpu=name --format=csv,noheader"]),
        "platform": _command_metadata(["bash", "-lc", "uname -a"]),
        "windows_wsl_sync": None,
        "dreamplace4_baseline": _dreamplace4_baseline_audit(dreamplace_root),
        "dreamplace_design_flags": _dreamplace_design_flags(selected_designs=SIGNATURE_DESIGNS),
    }
    if windows_repo.exists() and windows_repo.resolve() != repo_root.resolve():
        audit["windows_wsl_sync"] = {
            "windows_repo": str(windows_repo),
            "windows_git": _git_metadata(windows_repo),
            "head_matches": (
                _git_metadata(windows_repo).get("head")
                == audit["coevop_repo"].get("head")
            ),
            "note": "Uncommitted changes must be compared by status/diff, not only HEAD.",
        }
    return audit


def _dreamplace4_baseline_audit(dreamplace_root: Path) -> dict[str, Any]:
    git = _git_metadata(dreamplace_root)
    candidates = [
        dreamplace_root / "dreamplace" / "NonLinearPlace.py",
        dreamplace_root / "dreamplace" / "Timer.py",
        dreamplace_root / "benchmarks" / "chipbench",
    ]
    return {
        "installed_default_is_authoritative": True,
        "separate_4_0_config_found": False,
        "distinct_runnable_baseline": False,
        "git": git,
        "checked_paths": [
            {"path": str(path), "exists": path.exists()} for path in candidates
        ],
        "note": (
            "CoEvoPR treats objective_id=default as the installed DREAMPlace "
            "default flow. If this installation is DREAMPlace 4.0 or newer, "
            "default is also the local DREAMPlace 4.0 default unless a separate "
            "legacy/4.0 config is supplied."
        ),
    }


def _dreamplace_design_flags(
    *, selected_designs: tuple[ChipBenchSignatureDesign, ...]
) -> list[dict[str, Any]]:
    flags = []
    interesting = [
        "timing_opt_flag",
        "routability_opt_flag",
        "target_density",
        "global_place_flag",
        "legalize_flag",
        "detailed_place_flag",
        "gpu",
    ]
    for design in selected_designs:
        path = Path(os.path.expandvars(design.dreamplace_config)).expanduser()
        item: dict[str, Any] = {
            "design": design.name,
            "dreamplace_config": str(path),
            "exists": path.is_file(),
        }
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                item["error"] = f"JSONDecodeError: {exc}"
            else:
                item["flags"] = {key: payload.get(key) for key in interesting if key in payload}
        flags.append(item)
    return flags


def _export_design_finalists(*, design_name: str, design_run_dir: Path, finalists_root: Path) -> None:
    candidates = [
        ("robust_best_objective", design_run_dir / "robust_best_objective.json"),
        ("pareto_best_objective", design_run_dir / "pareto_best_objective.json"),
        ("best_objective", design_run_dir / "best_objective.json"),
    ]
    for label, path in candidates:
        if path.is_file():
            target = finalists_root / f"{design_name}_{label}.json"
            shutil.copy2(path, target)
            code_path = path.with_name(path.name.replace("_objective.json", "_program.py"))
            if code_path.is_file():
                shutil.copy2(code_path, finalists_root / f"{design_name}_{label}.py")


def _aggregate_csvs(paths: list[Path], output: Path) -> Path | None:
    rows: list[dict[str, Any]] = []
    fieldnames: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row = dict(row)
                row["source_csv"] = str(path)
                rows.append(row)
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
    if not rows:
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def _write_combined_comparison(
    dreamplace_csv: Path | None,
    post_grt_csv: Path | None,
    output: Path,
) -> Path | None:
    rows: list[dict[str, Any]] = []
    for stage, path in (("post_placement", dreamplace_csv), ("post_grt", post_grt_csv)):
        if path is None or not path.is_file():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                payload = dict(row)
                payload["evaluation_stage"] = stage
                rows.append(payload)
    if not rows:
        return None
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def _build_generalization_report(
    *,
    summary: dict[str, Any],
    audit: dict[str, Any],
    panel_manifest: dict[str, Any],
    run_root: Path,
) -> str:
    lines = [
        "# ChipBench Generalization Experiment",
        "",
        "## Scope",
        "",
        "This run evolves design-specific DREAMPlace objective functions with the current OpenEvolve-style local memory loop, then evaluates selected final placements with OpenROAD stopped after global routing.",
        "",
        "## Environment",
        "",
        f"- CoEvoPR HEAD: {audit.get('coevop_repo', {}).get('head')}",
        f"- DREAMPlace HEAD: {audit.get('dreamplace_git', {}).get('head')}",
        f"- DREAMPlace patch applied: {audit.get('dreamplace_patch_applied')}",
        f"- OpenROAD version: {audit.get('openroad', {}).get('stdout', '').strip() or audit.get('openroad', {}).get('error')}",
        f"- GPU: {audit.get('gpu', {}).get('stdout', '').strip() or audit.get('gpu', {}).get('error')}",
        f"- Provider/model: {audit.get('provider')} / {audit.get('resolved_model_requested')}",
        f"- Seeds: {audit.get('seeds')}",
        f"- Windows/WSL HEAD match: {bool((audit.get('windows_wsl_sync') or {}).get('head_matches'))}",
        "- Sync note: if HEADs differ, use the dirty working-tree status in `environment_audit.json`; the WSL run tree was overlaid from the Windows repo before execution.",
        "",
        "## Baselines",
        "",
        "- Primary baseline: `default`, the native installed DREAMPlace objective/flow.",
        "- DREAMPlace 4.0 default: counted separately only if a distinct runnable local 4.0/timing-driven configuration exists.",
        f"- Local 4.0 audit: {audit.get('dreamplace4_baseline', {}).get('note')}",
        f"- Local DREAMPlace describe: {audit.get('dreamplace4_baseline', {}).get('git', {}).get('describe')}",
        "- Fixed symbolic objectives are secondary references only, not replacements for the primary native default comparison.",
        "",
        "### DREAMPlace Config Flags",
        "",
        "| Design | timing_opt_flag | routability_opt_flag | target_density | gpu |",
        "|---|---:|---:|---:|---:|",
    ]
    for flag_item in audit.get("dreamplace_design_flags", []):
        flags = flag_item.get("flags") if isinstance(flag_item, dict) else {}
        flags = flags if isinstance(flags, dict) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(flag_item.get("design")),
                    str(flags.get("timing_opt_flag", "-")),
                    str(flags.get("routability_opt_flag", "-")),
                    str(flags.get("target_density", "-")),
                    str(flags.get("gpu", "-")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The local installed `default` is therefore the only primary DREAMPlace baseline unless a separate runnable DREAMPlace 4.0/timing-driven config is provided.",
            "",
            "## Panel",
            "",
        ]
    )
    for design in panel_manifest.get("designs", []):
        lines.append(f"- {design['name']}")
    lines.extend(["", "## Quantitative Summary", ""])
    lines.extend(_summary_table_lines(summary=summary, run_root=run_root))
    lines.extend(["", "## Per-Design Runs", ""])
    for item in summary.get("designs", []):
        lines.extend(
            [
                f"### {item.get('design')}",
                "",
                f"- Status: {item.get('status')}",
                f"- Run directory: `{item.get('run_dir')}`",
                f"- Config: `{item.get('config')}`",
            ]
        )
        if item.get("error"):
            lines.append(f"- Error: `{item.get('error')}`")
        run_dir = Path(str(item.get("run_dir", "")))
        best_code = _read_first_existing(
            [
                run_dir / "robust_best_program.py",
                run_dir / "pareto_best_program.py",
                run_dir / "best_program.py",
            ]
        )
        if best_code:
            lines.extend(["", "Best available generated objective code:", "", "```python", best_code.strip(), "```"])
        tier3 = item.get("final_tier3") if isinstance(item, dict) else None
        if tier3:
            lines.append(f"- Post-GRT summary: `{tier3.get('comparison_csv')}`")
        metrics = item.get("robust_best_program_metrics") or item.get(
            "pareto_best_generated_program_metrics"
        )
        if isinstance(metrics, dict) and metrics:
            lines.extend(
                [
                    f"- Search-panel HPWL delta vs native default: {metrics.get('hpwl_delta_pct')}",
                    f"- Search-panel overflow delta vs native default: {metrics.get('overflow_delta_pct')}",
                    f"- Active routing terms: {metrics.get('active_routing_terms')}",
                    f"- Structural failures in selected metrics: {metrics.get('structural_failure_count')}",
                    f"- Severe regressions in selected metrics: {metrics.get('severe_regression_count')}",
                    f"- Component feedback: {metrics.get('component_feedback') or 'none recorded'}",
                ]
            )
        map_elites = item.get("map_elites") if isinstance(item, dict) else None
        if isinstance(map_elites, dict):
            lines.extend(
                [
                    f"- MAP-Elites occupied cells: {map_elites.get('occupied_cells_total')}",
                    f"- Island cell coverage: {map_elites.get('occupied_cells_by_island')}",
                    f"- Migration events recorded: {len(map_elites.get('migration_events') or [])}",
                ]
            )
        lines.append("")
    lines.extend(
        [
            "## Cross-Circuit Transfer",
            "",
            "Cross-circuit transfer was not run in this pass. The current evidence is per-circuit objective discovery and per-circuit post-GRT validation only. A transfer matrix should route each circuit's best objective on the other ChipBench circuits after the serious per-circuit validation is stable.",
            "",
            "## Current Serious-Run Gap",
            "",
            "This report currently reflects the smoke-scale all-circuit run unless a later serious run overwrites these artifacts. The remaining serious configuration is three seeds `[1000, 1001, 1002]`, 50-iteration search placement, 150-iteration final placement, and GRT-only OpenROAD for the selected finalists.",
            "",
            "## Aggregated Artifacts",
            "",
            f"- DREAMPlace metrics: `{summary.get('dreamplace_metrics_csv')}`",
            f"- Post-GRT metrics: `{summary.get('post_grt_metrics_csv')}`",
            f"- Combined comparison table: `{summary.get('comparison_table_csv')}`",
            f"- Finalist objectives: `{summary.get('finalists_dir')}`",
            "",
            "## Interpretation Rules",
            "",
            "- Post-GRT rows come from OpenROAD global route only; detailed route is intentionally not run.",
            "- Missing or failed designs are retained as failures and should not be hidden from aggregate interpretation.",
            "- The native `default` objective is the key baseline unless a separate DREAMPlace 4.0 config is explicitly available and listed in the audit.",
        ]
    )
    report_path = run_root / "generalization_report.md"
    lines.append("")
    lines.append(f"Report path: `{report_path}`")
    return "\n".join(lines).rstrip() + "\n"


def _summary_table_lines(*, summary: dict[str, Any], run_root: Path) -> list[str]:
    post_grt = _post_grt_by_design(run_root / "post_grt_metrics.csv")
    lines = [
        "| Design | Search-selected program | Routed objective ID | Post-placement HPWL delta | Post-placement overflow delta | Post-GRT status | Post-GRT overflow delta |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for item in summary.get("designs", []):
        metrics = item.get("robust_best_program_metrics") or item.get(
            "pareto_best_generated_program_metrics"
        )
        objective_id = "-"
        hpwl = "-"
        overflow = "-"
        if isinstance(metrics, dict) and metrics:
            objective_id = str(
                item.get("robust_best_program_id")
                or item.get("pareto_best_generated_program_id")
                or "-"
            )
            hpwl = _format_metric(metrics.get("hpwl_delta_pct"))
            overflow = _format_metric(metrics.get("overflow_delta_pct"))
        grt = post_grt.get(str(item.get("design")), {})
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("design")),
                    objective_id,
                    str(grt.get("objective_id", "-")),
                    hpwl,
                    overflow,
                    str(grt.get("status", "-")),
                    _format_metric(grt.get("grt_overflow_delta_pct")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Negative deltas mean improvement against the native installed DREAMPlace `default` baseline. Post-GRT rows are global-route-only results; detailed route is intentionally not run.",
        ]
    )
    return lines


def _post_grt_by_design(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_design: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("objective_id") == "default":
            continue
        by_design[str(row.get("design"))] = row
    return by_design


def _format_metric(value: Any) -> str:
    if value in (None, ""):
        return "-"
    try:
        return f"{float(value):.3f}%"
    except (TypeError, ValueError):
        return str(value)


def _read_first_existing(paths: list[Path]) -> str | None:
    for path in paths:
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
    return None


def _git_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "head": _run_text(["git", "-C", str(path), "rev-parse", "HEAD"]),
        "status_short": _run_text(["git", "-C", str(path), "status", "--short"]),
        "describe": _run_text(["git", "-C", str(path), "describe", "--always", "--dirty"]),
    }


def _command_metadata(command: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
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


def _run_text(command: list[str]) -> str | None:
    result = _command_metadata(command)
    if result.get("returncode") == 0:
        return str(result.get("stdout", "")).strip()
    return None


def _write_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
