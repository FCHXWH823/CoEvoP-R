from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from scripts.build_timing_placements import build_placements


def test_relabelled_default_is_checked_under_target_id(tmp_path: Path):
    table = tmp_path / "comparison.csv"
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["design", "objective_id", "seed", "status", "output_artifact"],
        )
        writer.writeheader()
        for seed in (1000, 1001, 1002):
            writer.writerow(
                {
                    "design": "bp_fe",
                    "objective_id": "default",
                    "seed": seed,
                    "status": "success",
                    "output_artifact": tmp_path / f"seed_{seed}.def",
                }
            )
    placements, missing = build_placements(
        [["bp_fe", "default", str(table)]],
        seeds={1000, 1001, 1002},
        relabel={"default": "evoplace"},
        source="test",
    )
    assert missing == []
    assert len(placements) == 3
    assert {row["objective_id"] for row in placements} == {"evoplace"}


def test_allow_missing_writes_partial_manifest_without_failure(tmp_path: Path):
    table = tmp_path / "comparison.csv"
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["design", "objective_id", "seed", "status", "output_artifact"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "design": "bp_fe",
                "objective_id": "default",
                "seed": 1000,
                "status": "success",
                "output_artifact": tmp_path / "seed_1000.def",
            }
        )
    output = tmp_path / "placements.json"
    script = Path(__file__).parents[1] / "scripts" / "build_timing_placements.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            "bp_fe",
            "default",
            str(table),
            "--output",
            str(output),
            "--allow-missing",
        ],
        check=False,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result.returncode == 0
    assert len(payload["placements"]) == 1
    assert payload["missing_expected_cells"] == [
        "bp_fe/default/seed_1001",
        "bp_fe/default/seed_1002",
    ]
