"""CoEvoP&R-E and CoEvoP&R-L configurations share one method and one budget."""

from dataclasses import asdict
from pathlib import Path

import pytest

from coevop.eval.exposure import FAMILIES, write_lodo_config, write_target_evolution_config
from coevop.eval.openevolve_tier2 import (
    _panel_design_names,
    _validate_generalization_split,
    load_openevolve_tier2_config,
)
from coevop.eval.shared_panel import load_shared_panel

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIMARY = load_openevolve_tier2_config(
    REPO_ROOT / "configs" / "openevolve_tier2" / "chipbench_controller_tier2.toml"
)

# Everything except the designs that supply feedback and their collateral.
EXPOSURE_FIELDS = {
    "search_panel",
    "target_designs",
    "final_enabled",
    "final_panel",
    "final_top_k",
    "final_baseline_presets",
    "generalization_enabled",
    "generalization_panel",
    "generalization_top_k",
    "generalization_baseline_presets",
    "generalization_blind",
    "generalization_strict_leakage_guard",
    "tier3_panel",
    "timing_proxy_enabled",
    "timing_proxy_mode",
    "timing_proxy_panel",
    "timing_proxy_audit_required",
    "timing_proxy_audit_interval",
    "timing_proxy_audit_perturbations",
    "timing_proxy_wns_regression_gate_ns",
    "timing_proxy_wns_delta_min_abs_ns",
    "timing_proxy_tns_regression_min_abs_ns",
    "timing_proxy_selection_weight",
    "timing_controller_panel",
    "router_background_path",
}


def _method(config) -> dict:
    return {key: value for key, value in asdict(config).items() if key not in EXPOSURE_FIELDS}


@pytest.mark.parametrize(
    ("family", "design", "panel_name"),
    [
        ("chipbench", "bp_fe", "bp_fe"),
        ("chipbench", "mor1kx", "mor1kx"),
        ("superblue", "16", "superblue16_ot_notiming"),
        ("asap7", "ibex", "ibex"),
    ],
)
def test_target_evolution_uses_the_primary_method_on_one_design(
    tmp_path: Path, family: str, design: str, panel_name: str
) -> None:
    config = load_openevolve_tier2_config(
        write_target_evolution_config(
            family=family, design=design, output_dir=tmp_path, repo_root=REPO_ROOT
        )
    )

    assert _method(config) == _method(PRIMARY)
    # Feedback comes from the same design the candidate is evaluated on.
    assert _panel_design_names(config.search_panel) == [panel_name]
    assert _panel_design_names(config.final_panel) == [panel_name]
    assert config.target_designs == [panel_name]
    for panel in ("search_panel", "final_panel"):
        assert (
            load_shared_panel(getattr(config, panel)).seeds
            == load_shared_panel(getattr(PRIMARY, panel)).seeds
        )
    assert Path(config.tier3_panel) == REPO_ROOT / FAMILIES[family].post_route_panel


def test_tier_b_follows_the_available_timing_collateral(tmp_path: Path) -> None:
    def config_for(family: str, design: str):
        return load_openevolve_tier2_config(
            write_target_evolution_config(
                family=family,
                design=design,
                output_dir=tmp_path / f"{family}_{design}",
                repo_root=REPO_ROOT,
            )
        )

    audited = config_for("chipbench", "or1200")
    assert audited.timing_proxy_enabled is True
    assert audited.timing_proxy_audit_interval == PRIMARY.timing_proxy_audit_interval
    assert audited.timing_controller_panel is not None

    # vga_lcd is outside the four-design timing panel; Superblue ships none.
    assert config_for("chipbench", "vga_lcd").timing_proxy_enabled is False
    superblue = config_for("superblue", "1")
    assert superblue.timing_proxy_enabled is False
    assert superblue.timing_controller_panel is None


def test_leave_one_design_out_only_removes_the_target_from_feedback(tmp_path: Path) -> None:
    config = load_openevolve_tier2_config(
        write_lodo_config(heldout="mor1kx", output_dir=tmp_path, repo_root=REPO_ROOT)
    )

    assert _method(config) == _method(PRIMARY)
    training = _panel_design_names(config.search_panel)
    assert "mor1kx" not in training
    assert sorted([*training, "mor1kx"]) == sorted(_panel_design_names(PRIMARY.search_panel))
    assert config.target_designs == training
    assert config.final_enabled is False
    assert _panel_design_names(config.generalization_panel) == ["mor1kx"]
    assert config.generalization_blind is True
    assert config.generalization_strict_leakage_guard is True
    assert config.generalization_top_k == PRIMARY.final_top_k
    _validate_generalization_split(config)


def test_unknown_targets_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown superblue design"):
        write_target_evolution_config(
            family="superblue", design="2", output_dir=tmp_path, repo_root=REPO_ROOT
        )
    with pytest.raises(ValueError, match="unknown benchmark family"):
        write_target_evolution_config(
            family="ispd", design="adaptec1", output_dir=tmp_path, repo_root=REPO_ROOT
        )
