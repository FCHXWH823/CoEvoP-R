"""Summarize an OpenEvolve Tier-2 run from saved iteration artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):+.6f}%"
    except (TypeError, ValueError):
        return "n/a"


def _term_summary(metrics: dict[str, Any]) -> str:
    coeffs = metrics.get("active_routing_coefficients") or []
    if not coeffs:
        terms = metrics.get("active_routing_terms") or []
        return ", ".join(str(term) for term in terms) if terms else "none"
    parts = []
    for item in coeffs:
        if isinstance(item, dict):
            parts.append(f"{item.get('term')}={item.get('coefficient')}")
    return ", ".join(parts) if parts else "none"


def summarize_run(run_dir: Path) -> dict[str, Any]:
    iterations: list[dict[str, Any]] = []
    for summary_path in sorted(run_dir.glob("iteration_*/iteration_summary.json")):
        data = _load_json(summary_path)
        metrics = data.get("metrics") or {}
        iterations.append(
            {
                "iteration": data.get("iteration"),
                "program_id": data.get("program_id"),
                "status": data.get("status"),
                "combined_score": data.get("combined_score"),
                "routing_terms": metrics.get("active_routing_terms") or [],
                "routing_coefficients": metrics.get("active_routing_coefficients") or [],
                "aggregate_tier2_winner": metrics.get("aggregate_tier2_winner"),
                "all_design_robust_tier2_winner": metrics.get("all_design_robust_tier2_winner"),
                "routing_aware_tier2_candidate": metrics.get("routing_aware_tier2_candidate"),
                "parent_eligible": metrics.get("parent_eligible"),
                "negative_memory_only": metrics.get("negative_memory_only"),
                "hpwl_delta_pct": metrics.get("hpwl_delta_pct"),
                "overflow_delta_pct": metrics.get("overflow_delta_pct"),
                "worst_design_hpwl_delta_pct": metrics.get("worst_design_hpwl_delta_pct"),
                "worst_design_overflow_delta_pct": metrics.get("worst_design_overflow_delta_pct"),
                "feedback_lesson": metrics.get("feedback_lesson"),
                "per_design_deltas": metrics.get("per_design_deltas") or [],
                "summary_path": str(summary_path),
            }
        )

    finalists = [
        item
        for item in iterations
        if item.get("routing_aware_tier2_candidate")
        and item.get("all_design_robust_tier2_winner")
    ]
    aggregate_winners = [
        item
        for item in iterations
        if item.get("routing_aware_tier2_candidate")
        and item.get("aggregate_tier2_winner")
    ]
    final_summary_path = run_dir / "summary.json"
    return {
        "run_dir": str(run_dir),
        "final_summary_exists": final_summary_path.exists(),
        "iteration_count": len(iterations),
        "iterations": iterations,
        "aggregate_routing_winner_count": len(aggregate_winners),
        "robust_routing_finalist_count": len(finalists),
        "robust_routing_finalists": finalists,
    }


def write_markdown(summary: dict[str, Any], output: Path) -> None:
    lines = [
        "# OpenEvolve Tier-2 Run Summary",
        "",
        f"Run directory: `{summary['run_dir']}`",
        f"Completed iterations: `{summary['iteration_count']}`",
        f"Final summary exists: `{summary['final_summary_exists']}`",
        f"Aggregate routing winners: `{summary['aggregate_routing_winner_count']}`",
        f"Robust routing finalists: `{summary['robust_routing_finalist_count']}`",
        "",
        "| Iteration | Status | Routing term | Aggregate winner | Robust winner | HPWL delta | Overflow delta | Worst-design HPWL |",
        "|---:|---|---|---|---|---:|---:|---:|",
    ]
    for item in summary["iterations"]:
        lines.append(
            "| {iteration} | `{status}` | `{terms}` | {aggregate} | {robust} | `{hpwl}` | `{overflow}` | `{worst}` |".format(
                iteration=item.get("iteration"),
                status=item.get("status"),
                terms=_term_summary(
                    {
                        "active_routing_terms": item.get("routing_terms"),
                        "active_routing_coefficients": item.get("routing_coefficients"),
                    }
                ),
                aggregate="yes" if item.get("aggregate_tier2_winner") else "no",
                robust="yes" if item.get("all_design_robust_tier2_winner") else "no",
                hpwl=_fmt_pct(item.get("hpwl_delta_pct")),
                overflow=_fmt_pct(item.get("overflow_delta_pct")),
                worst=_fmt_pct(item.get("worst_design_hpwl_delta_pct")),
            )
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    summary = summarize_run(args.run_dir)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(summary, args.markdown_output)
    if not args.json_output and not args.markdown_output:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
