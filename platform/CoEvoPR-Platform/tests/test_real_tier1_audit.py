from pathlib import Path

import numpy as np

from coevop.datasets.circuitnet import build_manifest, write_manifest
from coevop.eval.real_tier1 import write_real_tier1_audit


def _write_npz(root: Path, design: str, relative: str, sample_id: str, value: float) -> None:
    path = root / "CircuitNet-N14" / "routability_features" / design / relative / f"{sample_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, data=np.full((3, 3), value, dtype=np.float64))


def _fixture(root: Path) -> Path:
    for design_index, design in enumerate(["A-small", "B-small", "C-small"]):
        for sample_index in range(4):
            value = float(design_index * 10 + sample_index + 1)
            sample_id = f"sample_{sample_index}"
            _write_npz(root, design, "RUDY/RUDY", sample_id, value)
            _write_npz(root, design, "RUDY/RUDY_pin", sample_id, value)
            _write_npz(root, design, "RUDY/RUDY_short", sample_id, value)
            _write_npz(root, design, "RUDY/RUDY_long", sample_id, value)
            _write_npz(root, design, "RUDY/RUDY_pin_long", sample_id, value)
            _write_npz(root, design, "cell_density", sample_id, value)
            _write_npz(root, design, "macro_region", sample_id, 0.0)
            if not (design == "B-small" and sample_index == 0):
                _write_npz(root, design, "DRC/DRC_all", sample_id, value)
            _write_npz(
                root,
                design,
                "congestion/congestion_early_global_routing/overflow_based/"
                "congestion_eGR_horizontal_overflow",
                sample_id,
                value,
            )
            _write_npz(
                root,
                design,
                "congestion/congestion_early_global_routing/overflow_based/"
                "congestion_eGR_vertical_overflow",
                sample_id,
                value,
            )
            _write_npz(
                root,
                design,
                "congestion/congestion_global_routing/overflow_based/"
                "congestion_GR_horizontal_overflow",
                sample_id,
                value,
            )
            _write_npz(
                root,
                design,
                "congestion/congestion_global_routing/overflow_based/"
                "congestion_GR_vertical_overflow",
                sample_id,
                value,
            )
            _write_npz(
                root,
                design,
                "congestion/congestion_global_routing/utilization_based/"
                "congestion_GR_horizontal_util",
                sample_id,
                value / 10.0,
            )
            _write_npz(
                root,
                design,
                "congestion/congestion_global_routing/utilization_based/"
                "congestion_GR_vertical_util",
                sample_id,
                value / 10.0,
            )
    return root


def test_write_real_tier1_audit_outputs_controls_and_report(tmp_path: Path) -> None:
    manifest = build_manifest(_fixture(tmp_path / "dataset"))
    manifest_path = write_manifest(manifest, tmp_path / "manifest.json")
    output_dir = tmp_path / "real_tier1"

    payload = write_real_tier1_audit(
        manifest_path=manifest_path,
        output_dir=output_dir,
        shuffle_trials=3,
        random_formula_count=4,
        validation_design="C-small",
    )

    assert payload["row_count"] == 12
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "scalars.csv").exists()
    assert (output_dir / "label_audit.json").exists()
    assert (output_dir / "baseline_rankings_by_split.json").exists()
    assert (output_dir / "negative_controls.json").exists()
    assert (output_dir / "real_tier1_report.md").exists()
    assert "Real Tier-1 CircuitNet Report" in (output_dir / "real_tier1_report.md").read_text(
        encoding="utf-8"
    )
