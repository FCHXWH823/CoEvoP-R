#!/usr/bin/env python3
"""Build a timing-evidence placement manifest from placement result tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        nargs=3,
        metavar=("DESIGN", "OBJECTIVE_ID", "TIER2_CSV"),
        required=True,
        help="Select default plus OBJECTIVE_ID rows for DESIGN from TIER2_CSV.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default="table1_timing_correlation")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="write available placements and report missing cells without failing",
    )
    parser.add_argument(
        "--relabel-objective",
        action="append",
        default=[],
        metavar="FROM=TO",
        help="Relabel selected objective IDs in the output manifest, e.g. default=evoplace.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        dest="seeds",
        help="Include only this seed; repeat for multiple seeds. Defaults to 1000,1001,1002.",
    )
    return parser.parse_args()


def build_placements(
    inputs: list[list[str]],
    *,
    seeds: set[int],
    relabel: dict[str, str],
    source: str,
) -> tuple[list[dict[str, object]], list[str]]:
    placements: list[dict[str, object]] = []
    missing: list[str] = []
    for design, objective_id, csv_path in inputs:
        with Path(csv_path).open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        source_objectives = {"default", objective_id}
        selected = [
            row
            for row in rows
            if row.get("design") == design
            and row.get("objective_id") in source_objectives
            and row.get("status") == "success"
            and row.get("output_artifact")
            and int(row["seed"]) in seeds
        ]
        for row in selected:
            placements.append(
                {
                    "design": design,
                    "objective_id": relabel.get(row["objective_id"], row["objective_id"]),
                    "seed": int(row["seed"]),
                    "def_path": row["output_artifact"],
                    "source": source,
                    "variant": "original",
                }
            )
        expected_outputs = {relabel.get(item, item) for item in source_objectives}
        for expected_objective in sorted(expected_outputs):
            for seed in sorted(seeds):
                if not any(
                    item["design"] == design
                    and item["objective_id"] == expected_objective
                    and item["seed"] == seed
                    for item in placements
                ):
                    missing.append(f"{design}/{expected_objective}/seed_{seed}")
    return placements, missing


def main() -> int:
    args = _parse_args()
    relabel: dict[str, str] = {}
    for item in args.relabel_objective:
        if "=" not in item:
            raise ValueError(f"invalid --relabel-objective value: {item}")
        source_id, target_id = item.split("=", 1)
        if not source_id or not target_id:
            raise ValueError(f"invalid --relabel-objective value: {item}")
        relabel[source_id] = target_id
    seeds = set(args.seeds or (1000, 1001, 1002))
    placements, missing = build_placements(
        args.input,
        seeds=seeds,
        relabel=relabel,
        source=args.source,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "placements": sorted(
            placements,
            key=lambda item: (str(item["design"]), str(item["objective_id"]), int(item["seed"])),
        ),
        "missing_expected_cells": missing,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "placements": len(placements), "missing": missing}, indent=2))
    return 1 if missing and not args.allow_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
