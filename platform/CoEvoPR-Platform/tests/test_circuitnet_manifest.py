from pathlib import Path

import pytest

from coevop.datasets.circuitnet import build_manifest, resolve_n14_root


def _touch_npz(root: Path, relative: str, sample_id: str) -> None:
    path = root / relative / f"{sample_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_resolve_n14_root_accepts_repo_root(tmp_path: Path) -> None:
    n14_root = tmp_path / "CircuitNet-N14"
    (n14_root / "routability_features").mkdir(parents=True)

    assert resolve_n14_root(tmp_path) == n14_root.resolve()


def test_build_manifest_for_routability_design(tmp_path: Path) -> None:
    n14_root = tmp_path / "CircuitNet-N14"
    design = n14_root / "routability_features" / "Vortex-small"
    sample_id = "Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ap"

    _touch_npz(design, "RUDY/RUDY", sample_id)
    _touch_npz(design, "cell_density", sample_id)
    _touch_npz(design, "DRC/DRC_all", sample_id)
    _touch_npz(
        design,
        "congestion/congestion_global_routing/utilization_based/congestion_GR_vertical_util",
        sample_id,
    )

    manifest = build_manifest(tmp_path)

    assert manifest.version == 1
    assert len(manifest.designs) == 1
    assert manifest.designs[0].name == "Vortex-small"
    assert manifest.designs[0].family == "Vortex"
    assert manifest.designs[0].sample_count == 1

    sample = manifest.designs[0].samples[0]
    assert sample.sample_id == sample_id
    assert "rudy" in sample.features
    assert "cell_density" in sample.features
    assert "drc_all" in sample.labels
    assert "congestion_gr_vertical_util" in sample.labels
    assert sample.features["rudy"].startswith("routability_features/Vortex-small/")


def test_build_manifest_rejects_missing_requested_design(tmp_path: Path) -> None:
    (tmp_path / "CircuitNet-N14" / "routability_features").mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        build_manifest(tmp_path, designs=["missing-design"])
