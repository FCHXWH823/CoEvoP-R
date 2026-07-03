"""Verify the 2026-07-03 progress-report numbers from packaged artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


EXPECTED = {
    "bp_fe": {
        "objective_id": "obj_34cad18eb2fc5f98",
        "placement_hpwl_change_pct": 13.56,
        "placement_overflow_change_pct": 72.81,
        "post_grt_wirelength_change_pct": 45.88,
        "routing_congestion_change_pct": 83.47,
        "wns_delta": 0.7927,
        "tns_delta": 187.80,
    },
    "bp_be": {
        "objective_id": "obj_1c623761f3ce9552",
        "placement_hpwl_change_pct": 53.86,
        "placement_overflow_change_pct": 63.48,
        "post_grt_wirelength_change_pct": 68.34,
        "routing_congestion_change_pct": 91.39,
        "wns_delta": 0.9414,
        "tns_delta": 1549.32,
    },
    "ethernet": {
        "objective_id": "obj_705dbe679dcdb0f7",
        "placement_hpwl_change_pct": 47.27,
        "placement_overflow_change_pct": 20.73,
        "post_grt_wirelength_change_pct": 36.27,
        "routing_congestion_change_pct": 83.46,
        "wns_delta": 0.1901,
        "tns_delta": 15.80,
    },
    "swerv_wrapper": {
        "objective_id": "obj_4fd265b50589cb27",
        "placement_hpwl_change_pct": 24.59,
        "placement_overflow_change_pct": 70.09,
        "post_grt_wirelength_change_pct": 30.35,
        "routing_congestion_change_pct": 48.33,
        "wns_delta": 0.1782,
        "tns_delta": 2116.62,
    },
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def value(row: dict[str, str], column: str) -> float | None:
    raw = row.get(column, "")
    if raw == "" or raw is None:
        return None
    return float(raw)


def grouped_mean(rows: list[dict[str, str]], design: str, objective: str, column: str) -> float:
    vals = [
        value(row, column)
        for row in rows
        if row["design"] == design and row["objective_id"] == objective
    ]
    vals = [val for val in vals if val is not None]
    if len(vals) != 3:
        raise ValueError(f"Expected 3 values for {design}/{objective}/{column}, got {len(vals)}")
    return mean(vals)


def lower_pct(default: float, candidate: float) -> float:
    return (default - candidate) / default * 100.0


def compute(artifact_root: Path) -> list[dict[str, float | str]]:
    tier2_rows = read_rows(artifact_root / "metrics" / "selected_tier2_rows_all_designs.csv")
    post_rows = read_rows(artifact_root / "metrics" / "selected_post_grt_rows_all_designs.csv")
    summary: list[dict[str, float | str]] = []

    for design, expected in EXPECTED.items():
        objective = expected["objective_id"]
        default_hpwl = grouped_mean(tier2_rows, design, "default", "hpwl")
        candidate_hpwl = grouped_mean(tier2_rows, design, objective, "hpwl")
        default_overflow = grouped_mean(tier2_rows, design, "default", "overflow")
        candidate_overflow = grouped_mean(tier2_rows, design, objective, "overflow")
        default_wl = grouped_mean(post_rows, design, "default", "estimated_wirelength")
        candidate_wl = grouped_mean(post_rows, design, objective, "estimated_wirelength")
        default_grt = grouped_mean(post_rows, design, "default", "grt_overflow")
        candidate_grt = grouped_mean(post_rows, design, objective, "grt_overflow")
        default_wns = grouped_mean(post_rows, design, "default", "wns")
        candidate_wns = grouped_mean(post_rows, design, objective, "wns")
        default_tns = grouped_mean(post_rows, design, "default", "tns")
        candidate_tns = grouped_mean(post_rows, design, objective, "tns")
        default_drc = grouped_mean(post_rows, design, "default", "drc_count")
        candidate_drc = grouped_mean(post_rows, design, objective, "drc_count")

        summary.append(
            {
                "design": design,
                "objective_id": objective,
                "placement_hpwl_default": default_hpwl,
                "placement_hpwl_coevop": candidate_hpwl,
                "placement_hpwl_change_pct": lower_pct(default_hpwl, candidate_hpwl),
                "placement_overflow_default": default_overflow,
                "placement_overflow_coevop": candidate_overflow,
                "placement_overflow_change_pct": lower_pct(default_overflow, candidate_overflow),
                "post_grt_wirelength_default": default_wl,
                "post_grt_wirelength_coevop": candidate_wl,
                "post_grt_wirelength_change_pct": lower_pct(default_wl, candidate_wl),
                "routing_congestion_default": default_grt,
                "routing_congestion_coevop": candidate_grt,
                "routing_congestion_change_pct": lower_pct(default_grt, candidate_grt),
                "wns_delta": candidate_wns - default_wns,
                "tns_delta": candidate_tns - default_tns,
                "drc_default": default_drc,
                "drc_coevop": candidate_drc,
            }
        )
    return summary


def write_outputs(artifact_root: Path, summary: list[dict[str, float | str]]) -> None:
    fields = list(summary[0].keys())
    with (artifact_root / "verification_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    checks = []
    for row in summary:
        expected = EXPECTED[str(row["design"])]
        for metric in (
            "placement_hpwl_change_pct",
            "placement_overflow_change_pct",
            "post_grt_wirelength_change_pct",
            "routing_congestion_change_pct",
            "wns_delta",
            "tns_delta",
        ):
            tolerance = 0.01 if metric != "wns_delta" else 0.0001
            diff = abs(float(row[metric]) - float(expected[metric]))
            checks.append(
                {
                    "design": row["design"],
                    "metric": metric,
                    "computed": row[metric],
                    "reported": expected[metric],
                    "abs_diff": diff,
                    "pass": diff <= tolerance + 1e-12,
                }
            )

    (artifact_root / "verification_summary.json").write_text(
        json.dumps({"summary": summary, "checks": checks}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 2026-07-03 Report Verification",
        "",
        "Computed from the packaged artifact CSV files only.",
        "",
        "| Circuit | Objective | HPWL lower | Overflow lower | Post-GRT WL lower | GRT overflow lower | WNS delta | TNS delta | DRC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['design']}` | `{row['objective_id']}` | "
            f"{float(row['placement_hpwl_change_pct']):.2f}% | "
            f"{float(row['placement_overflow_change_pct']):.2f}% | "
            f"{float(row['post_grt_wirelength_change_pct']):.2f}% | "
            f"{float(row['routing_congestion_change_pct']):.2f}% | "
            f"{float(row['wns_delta']):+.4f} ns | "
            f"{float(row['tns_delta']):+.2f} ns | "
            f"{float(row['drc_default']):.0f}->{float(row['drc_coevop']):.0f} |"
        )
    failed = [check for check in checks if not check["pass"]]
    lines += ["", f"Verification status: {'PASS' if not failed else 'FAIL'}", f"Failed checks: {len(failed)}"]
    (artifact_root / "verification_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/2026_07_03_report_placements"),
    )
    args = parser.parse_args()
    artifact_root = args.artifact_root.resolve()
    summary = compute(artifact_root)
    write_outputs(artifact_root, summary)
    for row in summary:
        print(
            f"{row['design']}: {row['objective_id']} "
            f"HPWL {float(row['placement_hpwl_change_pct']):.2f}% "
            f"overflow {float(row['placement_overflow_change_pct']):.2f}% "
            f"postGRT_WL {float(row['post_grt_wirelength_change_pct']):.2f}% "
            f"GRT_overflow {float(row['routing_congestion_change_pct']):.2f}% "
            f"WNS {float(row['wns_delta']):+.4f} "
            f"TNS {float(row['tns_delta']):+.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
