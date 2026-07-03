from pathlib import Path

import numpy as np

from coevop.datasets.circuitnet import build_manifest, write_manifest
from coevop.evolution.offline import _reflection_metrics, run_offline_evolution
from coevop.store.sqlite import SQLiteStore


def _write_npz(root: Path, relative: str, sample_id: str, value: float) -> None:
    path = root / relative / f"{sample_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, data=np.full((4, 4), value, dtype=np.float64))


def _build_fixture(root: Path) -> Path:
    n14_root = root / "CircuitNet-N14"
    design = n14_root / "routability_features" / "Vortex-small"
    for index, value in enumerate(float(i) for i in range(1, 13)):
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


def test_offline_evolution_scores_and_persists_mock_candidates(tmp_path: Path) -> None:
    manifest = build_manifest(_build_fixture(tmp_path / "dataset"))
    manifest_path = write_manifest(manifest, tmp_path / "manifest.json")
    run_dir = tmp_path / "run"

    summary = run_offline_evolution(
        manifest_path=manifest_path,
        provider_name="mock",
        run_dir=run_dir,
        population_size=2,
        generations=1,
        elite_count=2,
        seed=0,
    )

    assert summary.accepted_candidates >= 2
    assert summary.rejected_candidates == 0
    assert summary.split_strategy == "sample"
    assert summary.split_row_counts == {"train": 8, "validation": 2, "heldout": 2}
    assert summary.selection_split == "validation"
    assert summary.reflection_mode == "accepted"
    assert summary.diversity_lambda == 0.05
    assert summary.operator_weights == {
        "crossover": 0.35,
        "mutate": 0.3,
        "param_tune": 0.2,
        "simplify": 0.15,
    }
    assert summary.islands == 5
    assert summary.constant_fit_enabled is True
    assert len(summary.top_candidates) == 2
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "split_manifest.json").exists()
    assert (run_dir / "top_candidates.json").exists()
    assert (run_dir / "top_candidates.csv").exists()
    assert (run_dir / "comparison_table.csv").exists()
    assert (run_dir / "scalars.csv").exists()
    assert (run_dir / "baseline_rankings.json").exists()
    assert (run_dir / "baseline_rankings_by_split.json").exists()
    assert (run_dir / "long_term_reflection.txt").exists()

    store = SQLiteStore(run_dir / "evolution.sqlite")
    try:
        assert store.count_candidates() >= 2
        top = store.top_candidates(limit=1)[0]
        assert top["score"] > 0.99
        assert top["metrics"]["selection_split"] == "validation"
        assert top["metrics"]["operator"] in {
            "init",
            "mutate",
            "crossover",
            "param_tune",
            "simplify",
        }
        assert top["metrics"]["signature"]
        assert top["metrics"]["program_validation"]
        assert set(top["metrics"]["split_scores"]) == {"train", "validation", "heldout"}
        assert top["metrics"]["reflection"].startswith("Mock reflection")
        assert top["metrics"]["dreamplace_deployable"] is False
        assert top["metrics"]["dreamplace_undeployable_terms"]
        assert top["metrics"]["source_program"]
        assert top["metrics"]["component_diagnostics"]["validation"]
    finally:
        store.close()


def test_reflection_metrics_exclude_heldout() -> None:
    metrics = {
        "split_scores": {"train": 1.0, "validation": 2.0, "heldout": 3.0},
        "split_correlations": {"train": [], "validation": [], "heldout": [{"target": "x"}]},
        "component_diagnostics": {"train": {}, "validation": {}, "heldout": {"x": {}}},
    }

    sanitized = _reflection_metrics(metrics)

    assert "heldout" not in str(sanitized).lower()
