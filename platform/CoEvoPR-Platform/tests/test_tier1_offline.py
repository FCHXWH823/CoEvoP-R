from pathlib import Path

import numpy as np
import pytest

from coevop.datasets.circuitnet import build_manifest
from coevop.datasets.summaries import compute_summary_rows, write_summary_csv
from coevop.eval.correlations import kendall_tau_b, spearman
from coevop.eval.tier1 import (
    TargetCorrelation,
    aggregate_correlations,
    component_diagnostics,
    evaluate_spec,
    rank_objectives,
    write_rankings_csv,
    write_rankings_json,
)
from coevop.objectives.spec import parse_objective_spec


def _write_npz(root: Path, relative: str, sample_id: str, value: float) -> None:
    path = root / relative / f"{sample_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, data=np.full((4, 4), value, dtype=np.float64))


def _build_fixture(root: Path) -> Path:
    n14_root = root / "CircuitNet-N14"
    design = n14_root / "routability_features" / "Vortex-small"
    for index, value in enumerate([1.0, 2.0, 3.0, 4.0]):
        sample_id = f"sample_{index}"
        _write_npz(design, "RUDY/RUDY", sample_id, value)
        _write_npz(design, "RUDY/RUDY_pin", sample_id, value + 0.5)
        _write_npz(design, "RUDY/RUDY_short", sample_id, value)
        _write_npz(design, "RUDY/RUDY_long", sample_id, value)
        _write_npz(design, "RUDY/RUDY_pin_long", sample_id, value)
        _write_npz(design, "cell_density", sample_id, value * 2.0)
        _write_npz(design, "macro_region", sample_id, 0.0)
        _write_npz(design, "DRC/DRC_all", sample_id, value)
        _write_npz(
            design,
            "congestion/congestion_early_global_routing/overflow_based/"
            "congestion_eGR_horizontal_overflow",
            sample_id,
            value,
        )
        _write_npz(
            design,
            "congestion/congestion_early_global_routing/overflow_based/"
            "congestion_eGR_vertical_overflow",
            sample_id,
            value,
        )
        _write_npz(
            design,
            "congestion/congestion_global_routing/overflow_based/"
            "congestion_GR_horizontal_overflow",
            sample_id,
            value,
        )
        _write_npz(
            design,
            "congestion/congestion_global_routing/overflow_based/"
            "congestion_GR_vertical_overflow",
            sample_id,
            value,
        )
        _write_npz(
            design,
            "congestion/congestion_global_routing/utilization_based/"
            "congestion_GR_horizontal_util",
            sample_id,
            value / 10.0,
        )
        _write_npz(
            design,
            "congestion/congestion_global_routing/utilization_based/"
            "congestion_GR_vertical_util",
            sample_id,
            value / 10.0,
        )
    return root


def test_rank_correlations_are_perfect_for_monotonic_vectors() -> None:
    x = np.asarray([1.0, 2.0, 3.0, 4.0])
    y = np.asarray([10.0, 20.0, 30.0, 40.0])

    assert spearman(x, y) == pytest.approx(1.0)
    assert kendall_tau_b(x, y) == pytest.approx(1.0)


def test_tier1_score_modes_are_explicit_about_badness_direction() -> None:
    correlations = [
        TargetCorrelation("target.congestion_gr_overflow_mean", spearman=-0.5, kendall=0.0, n=4),
        TargetCorrelation("target.congestion_gr_util_p95", spearman=0.25, kendall=0.0, n=4),
    ]

    assert aggregate_correlations(correlations) == pytest.approx(-0.125)
    assert aggregate_correlations(correlations, score_mode="absolute_mean") == pytest.approx(0.375)


def test_tier1_offline_summarizes_and_ranks_objectives(tmp_path: Path) -> None:
    manifest = build_manifest(_build_fixture(tmp_path))
    rows = compute_summary_rows(manifest)

    assert len(rows) == 4
    assert rows[0]["feature.rudy.mean"] == 1.0
    assert rows[-1]["target.drc_hotspot_sum"] == 64.0

    rankings = rank_objectives(rows)
    assert rankings
    assert rankings[0].score > 0.99

    summary_path = write_summary_csv(rows, tmp_path / "scalars.csv")
    ranking_json_path = write_rankings_json(rankings, len(rows), tmp_path / "rankings.json")
    ranking_csv_path = write_rankings_csv(rankings, tmp_path / "rankings.csv")

    assert summary_path.exists()
    assert ranking_json_path.exists()
    assert ranking_csv_path.exists()


def test_tier1_evaluates_generated_objective_spec(tmp_path: Path) -> None:
    manifest = build_manifest(_build_fixture(tmp_path))
    rows = compute_summary_rows(manifest)
    spec = parse_objective_spec(
        {
            "id": "",
            "rationale": "RUDY p95 should track congestion badness.",
            "parent_ids": [],
            "declared_term_usage": ["rudy_p95"],
            "ast": {"op": "term", "name": "rudy_p95"},
        },
        created_by="test",
    )

    ranking = evaluate_spec(rows, spec)
    diagnostics = component_diagnostics(rows, spec)

    assert ranking.objective_id == spec.id
    assert ranking.score > 0.99
    assert diagnostics["rudy_p95"]["score"] > 0.99
