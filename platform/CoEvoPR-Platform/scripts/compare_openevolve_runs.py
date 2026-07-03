"""Compare OpenEvolve Tier-2 runs.

This is intentionally artifact-only: it reads saved run directories and does
not call DREAMPlace, OpenTimer, OpenROAD, or an LLM provider.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _fmt(value: Any, suffix: str = "") -> str:
    numeric = _finite(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:+.4f}{suffix}"


def _programs(run_dir: Path) -> list[dict[str, Any]]:
    program_dir = run_dir / "program_db" / "programs"
    if not program_dir.is_dir():
        return []
    programs = []
    for path in sorted(program_dir.glob("*.json")):
        try:
            payload = _load_json(path)
        except Exception:
            continue
        payload["_path"] = str(path)
        programs.append(payload)
    return programs


def _generated(program: dict[str, Any]) -> bool:
    pid = str(program.get("id") or "")
    return not (
        pid.startswith("seed_")
        or pid.startswith("provider_error")
        or (program.get("metrics") or {}).get("seed_baseline")
    )


def _accepted(program: dict[str, Any]) -> bool:
    return str(program.get("status") or "") == "accepted"


def _score(program: dict[str, Any]) -> float:
    return _finite((program.get("metrics") or {}).get("combined_score")) or -1e30


def _best(programs: list[dict[str, Any]], *, require_generated: bool = True) -> dict[str, Any] | None:
    candidates = [
        program
        for program in programs
        if _accepted(program) and (not require_generated or _generated(program))
    ]
    return max(candidates, key=_score) if candidates else None


def _metric(program: dict[str, Any] | None, key: str) -> Any:
    if not program:
        return None
    return (program.get("metrics") or {}).get(key)


def summarize_run(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    summary = _load_json(summary_path) if summary_path.is_file() else {}
    config_path = run_dir / "config_resolved.json"
    config = _load_json(config_path) if config_path.is_file() else {}
    programs = _programs(run_dir)
    generated = [program for program in programs if _generated(program)]
    accepted_generated = [program for program in generated if _accepted(program)]
    parent_eligible_generated = [
        program
        for program in accepted_generated
        if (program.get("metrics") or {}).get("parent_eligible")
    ]
    timing_programs = [
        program
        for program in accepted_generated
        if "timing_proxy_status" in (program.get("metrics") or {})
    ]
    best_generated = _best(programs, require_generated=True)
    best_any = _best(programs, require_generated=False)
    iterations = summary.get("iterations") or []
    return {
        "run_dir": str(run_dir),
        "summary_exists": summary_path.is_file(),
        "config_provider": config.get("provider"),
        "config_iterations": config.get("max_iterations"),
        "config_samples_per_iteration": config.get("samples_per_iteration"),
        "timing_proxy_enabled": config.get("timing_proxy_enabled"),
        "timing_proxy_mode": config.get("timing_proxy_mode"),
        "summary_best_program_id": summary.get("best_program_id"),
        "summary_best_program_metrics": summary.get("best_program_metrics"),
        "completed_iteration_count": len(iterations),
        "program_count": len(programs),
        "generated_program_count": len(generated),
        "accepted_generated_count": len(accepted_generated),
        "parent_eligible_generated_count": len(parent_eligible_generated),
        "timing_proxy_program_count": len(timing_programs),
        "aggregate_routing_winner_count": sum(
            1 for program in accepted_generated if (program.get("metrics") or {}).get("aggregate_tier2_winner")
        ),
        "robust_routing_winner_count": sum(
            1 for program in accepted_generated if (program.get("metrics") or {}).get("all_design_robust_tier2_winner")
        ),
        "best_generated": _program_summary(best_generated),
        "best_any": _program_summary(best_any),
    }


def _program_summary(program: dict[str, Any] | None) -> dict[str, Any] | None:
    if not program:
        return None
    metrics = program.get("metrics") or {}
    return {
        "id": program.get("id"),
        "objective_id": program.get("objective_id"),
        "status": program.get("status"),
        "combined_score": metrics.get("combined_score"),
        "hpwl_delta_pct": metrics.get("hpwl_delta_pct"),
        "overflow_delta_pct": metrics.get("overflow_delta_pct"),
        "worst_design_hpwl_delta_pct": metrics.get("worst_design_hpwl_delta_pct"),
        "worst_design_overflow_delta_pct": metrics.get("worst_design_overflow_delta_pct"),
        "parent_eligible": metrics.get("parent_eligible"),
        "negative_memory_only": metrics.get("negative_memory_only"),
        "aggregate_tier2_winner": metrics.get("aggregate_tier2_winner"),
        "all_design_robust_tier2_winner": metrics.get("all_design_robust_tier2_winner"),
        "timing_proxy_status": metrics.get("timing_proxy_status"),
        "timing_proxy_wns_delta": metrics.get("timing_proxy_wns_delta"),
        "timing_proxy_tns_delta_pct": metrics.get("timing_proxy_tns_delta_pct"),
        "mechanism_terms": metrics.get("mechanism_terms") or program.get("term_set"),
        "path": program.get("_path"),
    }


def compare_runs(previous: Path, current: Path) -> dict[str, Any]:
    return {
        "previous": summarize_run(previous),
        "current": summarize_run(current),
    }


def write_markdown(payload: dict[str, Any], output: Path) -> None:
    previous = payload["previous"]
    current = payload["current"]
    rows = [
        ("Previous", previous),
        ("OpenTimer feedback", current),
    ]
    lines = [
        "# OpenEvolve Run Comparison",
        "",
        "| Run | Iterations | Generated | Accepted generated | Parent-eligible generated | Timing-proxy programs | Best generated score | Best generated HPWL | Best generated overflow | Best generated WNS delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in rows:
        best = item.get("best_generated") or {}
        lines.append(
            "| {label} | {iters} | {gen} | {accepted} | {eligible} | {timing} | `{score}` | `{hpwl}` | `{overflow}` | `{wns}` |".format(
                label=label,
                iters=item.get("completed_iteration_count"),
                gen=item.get("generated_program_count"),
                accepted=item.get("accepted_generated_count"),
                eligible=item.get("parent_eligible_generated_count"),
                timing=item.get("timing_proxy_program_count"),
                score=_fmt(best.get("combined_score")),
                hpwl=_fmt(best.get("hpwl_delta_pct"), "%"),
                overflow=_fmt(best.get("overflow_delta_pct"), "%"),
                wns=_fmt(best.get("timing_proxy_wns_delta")),
            )
        )
    lines.extend(
        [
            "",
            "## Best Generated Candidate Details",
            "",
        ]
    )
    for label, item in rows:
        best = item.get("best_generated") or {}
        lines.extend(
            [
                f"### {label}",
                "",
                f"- Run: `{item.get('run_dir')}`",
                f"- Program: `{best.get('id')}`",
                f"- Objective: `{best.get('objective_id')}`",
                f"- Terms: `{best.get('mechanism_terms')}`",
                f"- Parent eligible: `{best.get('parent_eligible')}`",
                f"- HPWL delta: `{_fmt(best.get('hpwl_delta_pct'), '%')}`",
                f"- Overflow delta: `{_fmt(best.get('overflow_delta_pct'), '%')}`",
                f"- Timing proxy status: `{best.get('timing_proxy_status')}`",
                f"- Timing WNS delta: `{_fmt(best.get('timing_proxy_wns_delta'))}`",
                f"- Timing TNS delta: `{_fmt(best.get('timing_proxy_tns_delta_pct'), '%')}`",
                "",
            ]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    payload = compare_runs(args.previous, args.current)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output:
        write_markdown(payload, args.markdown_output)
    if not args.json_output and not args.markdown_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
