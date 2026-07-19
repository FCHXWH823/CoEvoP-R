#!/usr/bin/env python3
"""Aggregate the post-route metrics reported in the paper tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


METRICS = {
    "routed_wirelength_reduction_pct": ("routed_wirelength_delta_pct", -1.0),
    "congestion_reduction_pct": ("grt_overflow_delta_pct", -1.0),
    "wns_gain_ns": ("wns_gain_ns", 1.0),
    "tns_gain_ns": ("tns_gain_ns", 1.0),
}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def aggregate(
    comparison_csv: str | Path,
    *,
    objective_id: str,
    expected_seeds: int = 3,
) -> dict[str, Any]:
    with Path(comparison_csv).open(newline="", encoding="utf-8") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if str(row.get("objective_id") or "") == objective_id
        ]
    by_design: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_design[str(row.get("design") or "unknown")].append(row)

    design_results = {}
    for design, design_rows in sorted(by_design.items()):
        successful = [row for row in design_rows if row.get("status") == "success"]
        seed_count = len({int(row["seed"]) for row in successful})
        qualified = seed_count >= expected_seeds
        metrics = {}
        for output_name, (source_name, sign) in METRICS.items():
            values = [
                sign * value
                for row in successful
                if (value := _finite(row.get(source_name))) is not None
            ]
            metrics[output_name] = fmean(values) if qualified and values else None
        design_results[design] = {
            "successful_seeds": seed_count,
            "expected_seeds": expected_seeds,
            "qualified": qualified,
            **metrics,
        }

    qualified_designs = [
        result for result in design_results.values() if result["qualified"]
    ]
    means = {
        metric: (
            fmean(
                float(result[metric])
                for result in qualified_designs
                if result[metric] is not None
            )
            if any(result[metric] is not None for result in qualified_designs)
            else None
        )
        for metric in METRICS
    }
    return {
        "objective_id": objective_id,
        "source": str(Path(comparison_csv).resolve()),
        "metric_stage": "post_route",
        "expected_seeds": expected_seeds,
        "all_designs_qualified": bool(design_results)
        and all(result["qualified"] for result in design_results.values()),
        "by_design": design_results,
        "design_means": means,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-csv", required=True, type=Path)
    parser.add_argument("--objective-id", required=True)
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = aggregate(
        args.comparison_csv,
        objective_id=args.objective_id,
        expected_seeds=args.expected_seeds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["all_designs_qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
