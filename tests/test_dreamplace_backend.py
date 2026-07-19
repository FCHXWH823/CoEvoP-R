import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from coevop.backends.dreamplace import (
    is_patch_applied,
    make_run_config,
    parse_dreamplace_log,
    run_dreamplace,
)
from coevop.objectives.presets import objective_preset
from coevop.objectives.spec import parse_objective_spec, write_objective_spec


def test_objective_preset_contains_dreamplace_terms() -> None:
    spec = objective_preset("dreamplace_wl_density")

    assert spec.term_set == ["density", "wirelength"]
    assert spec.created_by == "preset"


def test_make_run_config_adds_custom_objective(tmp_path: Path) -> None:
    base_config = tmp_path / "base.json"
    base_config.write_text(
        json.dumps(
            {
                "gpu": 0,
                "global_place_stages": [{"iteration": 1000, "optimizer": "nesterov"}],
                "legalize_flag": 1,
                "detailed_place_flag": 1,
            }
        ),
        encoding="utf-8",
    )
    objective_path = write_objective_spec(
        objective_preset("dreamplace_wl_density"),
        tmp_path / "objective.json",
    )

    output = make_run_config(
        base_config,
        objective_path,
        tmp_path / "run",
        iterations=7,
        gpu=0,
        seed=1001,
        custom_objective_log_interval=3,
        stop_overflow=0.0,
    )
    config = json.loads(output.read_text(encoding="utf-8"))

    assert config["custom_objective_spec"] == str(objective_path.resolve())
    assert config["custom_objective_log_interval"] == 3
    assert config["global_place_stages"][0]["iteration"] == 7
    assert config["gpu"] == 0
    assert config["random_seed"] == 1001
    assert config["stop_overflow"] == 0.0
    assert config["legalize_flag"] == 0
    assert config["detailed_place_flag"] == 0
    assert Path(config["result_dir"]).name == "results"


def test_make_run_config_can_leave_custom_objective_disabled(tmp_path: Path) -> None:
    base_config = tmp_path / "base.json"
    base_config.write_text(
        json.dumps({"global_place_stages": [{"iteration": 1000}]}),
        encoding="utf-8",
    )

    output = make_run_config(base_config, None, tmp_path / "run", iterations=3)
    config = json.loads(output.read_text(encoding="utf-8"))

    assert "custom_objective_spec" not in config
    assert config["global_place_stages"][0]["iteration"] == 3


def test_make_run_config_infers_raw_semantics_for_stateful_policy(tmp_path: Path) -> None:
    base_config = tmp_path / "base.json"
    base_config.write_text(
        json.dumps({"global_place_stages": [{"iteration": 1000}]}),
        encoding="utf-8",
    )
    objective_path = write_objective_spec(
        objective_preset("dreamplace_controller_native_identity"),
        tmp_path / "controller.json",
    )

    output = make_run_config(base_config, objective_path, tmp_path / "run")
    config = json.loads(output.read_text(encoding="utf-8"))

    assert config["custom_objective_semantics"] == "raw"


def test_run_dreamplace_uses_active_python_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dreamplace_root = tmp_path / "dreamplace_root"
    dreamplace_root.mkdir()
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("PYTHONPATH", "existing-dependency-path")
    monkeypatch.setattr("coevop.backends.dreamplace.subprocess.run", fake_run)
    run = run_dreamplace(
        dreamplace_root=dreamplace_root,
        config_path=config,
        run_name="environment-test",
        run_dir=tmp_path / "run",
    )

    assert captured["command"][0] == sys.executable
    assert captured["env"]["PYTHONPATH"] == os.pathsep.join(
        (str(dreamplace_root.resolve()), "existing-dependency-path")
    )
    assert run.command[0] == sys.executable


def test_make_run_config_accepts_utf8_bom_base_config(tmp_path: Path) -> None:
    base_config = tmp_path / "base.json"
    base_config.write_text(
        json.dumps({"global_place_stages": [{"iteration": 1000}]}),
        encoding="utf-8-sig",
    )

    output = make_run_config(base_config, None, tmp_path / "run", iterations=5)
    config = json.loads(output.read_text(encoding="utf-8"))

    assert config["global_place_stages"][0]["iteration"] == 5


def test_is_patch_applied_detects_marker(tmp_path: Path) -> None:
    placeobj = tmp_path / "dreamplace" / "PlaceObj.py"
    placeobj.parent.mkdir(parents=True)
    placeobj.write_text("class PlaceObj:\n    def _coevop_build_extra_ops(self):\n        pass\n", encoding="utf-8")

    assert is_patch_applied(tmp_path)


def test_find_placeobj_backup_falls_back_to_source_tree(tmp_path: Path) -> None:
    from coevop.backends.dreamplace import _find_placeobj_backup

    install_placeobj = tmp_path / "install" / "dreamplace" / "PlaceObj.py"
    install_placeobj.parent.mkdir(parents=True)
    install_placeobj.write_text("_COEVOP_DEPLOYABLE_TERMS = set()\n", encoding="utf-8")
    old_backup = install_placeobj.with_name("PlaceObj.py.coevop_bak_")
    old_backup.write_text("_COEVOP_DEPLOYABLE_TERMS = set()\n", encoding="utf-8")
    source_placeobj = tmp_path / "dreamplace" / "PlaceObj.py"
    source_placeobj.parent.mkdir(parents=True)
    source_placeobj.write_text("class PlaceObj:\n    pass\n", encoding="utf-8")

    assert _find_placeobj_backup(install_placeobj) == source_placeobj


def test_make_run_config_rejects_tier1_only_objective(tmp_path: Path) -> None:
    base_config = tmp_path / "base.json"
    base_config.write_text(json.dumps({"global_place_stages": [{"iteration": 1000}]}), encoding="utf-8")
    spec = parse_objective_spec(
        {
            "id": "",
            "rationale": "RUDY is Tier-1 only until differentiable DREAMPlace support exists.",
            "parent_ids": [],
            "declared_term_usage": ["rudy_p95"],
            "ast": {"op": "term", "name": "rudy_p95"},
        },
        created_by="test",
    )
    objective_path = write_objective_spec(spec, tmp_path / "objective.json")

    with pytest.raises(ValueError, match="unsupported terms"):
        make_run_config(base_config, objective_path, tmp_path / "run")


def test_parse_dreamplace_log_extracts_validation_metrics(tmp_path: Path) -> None:
    log_path = tmp_path / "dreamplace.log"
    log_path.write_text(
        "\n".join(
            [
                "total_movable_cell_area = 1.43187e+06, total_movable_macro_area = 5.39505e+06",
                (
                    "DREAMPlace - iteration 999, ( 399, 0, 0), "
                    "Obj 1.893280E+07, wHPWL 6.056268E+07, Overflow 8.788136E-01"
                ),
                (
                    "CoEvoP&R fixed-position OpenTimer HPWL 5.5E+07, "
                    "Overflow 2.5E-01, MaxDensity 1.4E+00, TNS -2.0, WNS -0.1"
                ),
                "#nodes = 100, #terminals = 0, #movable = 90, #nets = 1000",
                (
                    "[WARNING] CoEvoP&R timing proxy skipped 20 nets and "
                    "30 pin-incomplete nets because placement names were absent"
                ),
                (
                    "INFO CoEvoP&R custom objective id=obj_x call=3 total=2.5E+00 "
                    "wirelength_raw=6.0E+07 weighted_density_raw=1.2E+05"
                ),
                "INFO CoEvoP&R custom objective term_scale wirelength=6.0E+07",
                "INFO CoEvoP&R custom objective output_scale=6.0E+07",
                (
                    'INFO CoEvoP&R custom objective terms {"wirelength": '
                    '{"raw": 6.0E+07, "normalized": 1.0}}'
                ),
                "INFO CoEvoP&R custom objective grad_norm=4.2E+01",
                "Could not open input file missing.lef",
                "[ASSERT ] failed to read input LEF files",
            ]
        ),
        encoding="utf-8",
    )

    metrics = parse_dreamplace_log(log_path)

    assert metrics["last"]["objective"] == pytest.approx(1.893280e7)
    assert metrics["last"]["hpwl"] == pytest.approx(5.5e7)
    assert metrics["last"]["overflow"] == pytest.approx(0.25)
    assert metrics["last"]["max_density"] == pytest.approx(1.4)
    assert metrics["last"]["tns"] == pytest.approx(-2.0)
    assert metrics["last"]["wns"] == pytest.approx(-0.1)
    assert metrics["timing_total_nets"] == 1000
    assert metrics["timing_skipped_nets"] == 20
    assert metrics["timing_pin_incomplete_nets"] == 30
    assert metrics["timing_net_coverage"] == pytest.approx(0.95)
    assert metrics["global_place_iterations_completed"] == 1000
    assert metrics["movable_cell_area"] == pytest.approx(1.43187e6)
    assert metrics["movable_macro_area"] == pytest.approx(5.39505e6)
    assert metrics["custom_objective"]["objective_id"] == "obj_x"
    assert metrics["custom_objective"]["calls"] == 3
    assert metrics["custom_objective"]["last_grad_norm"] == pytest.approx(42.0)
    assert metrics["custom_objective"]["term_scales"]["wirelength"] == pytest.approx(6.0e7)
    assert metrics["custom_objective"]["output_scale"] == pytest.approx(6.0e7)
    assert metrics["custom_objective"]["last_terms"]["wirelength"]["normalized"] == pytest.approx(1.0)
    assert metrics["missing_input_file"] == "Could not open input file missing.lef"
    assert metrics["assertion"] == "[ASSERT ] failed to read input LEF files"


def test_parse_dreamplace_log_uses_monotonic_global_iteration_across_restarts(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "dreamplace.log"
    log_path.write_text(
        "\n".join(
            [
                "DREAMPlace - iteration 599, ( 599, 0, 0), Obj 1.0E+00",
                "DREAMPlace - iteration 600, ( 0, 1, 0), Obj 9.0E-01",
                "DREAMPlace - iteration 974, ( 374, 1, 0), Obj 8.0E-01",
            ]
        ),
        encoding="utf-8",
    )

    metrics = parse_dreamplace_log(log_path)

    assert metrics["global_place_iterations_completed"] == 975


def test_parse_dreamplace_log_recovers_interleaved_timing_skip_warning(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "dreamplace.log"
    log_path.write_text(
        "\n".join(
            [
                "#nodes = 100, #nets = 81398",
                "[WARNING] CoEvoP&R timing proxy skipped 681 CoEvoP&R fixed-position stage=rc_tree_ready",
                "INFO unrelated Python log record",
                "nets and 685 pin-incomplete nets because placement names were absent",
            ]
        ),
        encoding="utf-8",
    )

    metrics = parse_dreamplace_log(log_path)

    assert metrics["timing_skipped_nets"] == 681
    assert metrics["timing_pin_incomplete_nets"] == 685
    assert metrics["timing_net_coverage"] == pytest.approx((81398 - 1366) / 81398)


def test_parse_dreamplace_log_recovers_interleave_after_pin_incomplete(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "dreamplace.log"
    log_path.write_text(
        "\n".join(
            [
                "#nodes = 100, #nets = 32175",
                (
                    "[WARNING] CoEvoP&R timing proxy skipped 2500 nets and "
                    "1273 pin-incompleteCoEvoP&R fixed-position stage=rc_tree_ready"
                ),
                " nets because placement names were absent from the OpenTimer netlist",
            ]
        ),
        encoding="utf-8",
    )

    metrics = parse_dreamplace_log(log_path)

    assert metrics["timing_skipped_nets"] == 2500
    assert metrics["timing_pin_incomplete_nets"] == 1273
    assert metrics["timing_net_coverage"] == pytest.approx((32175 - 3773) / 32175)


def test_parse_dreamplace_log_recovers_interleave_inside_nets_word(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "dreamplace.log"
    log_path.write_text(
        "\n".join(
            [
                "#nodes = 100, #nets = 32175",
                (
                    "[WARNING] CoEvoP&R timing proxy skipp"
                    "CoEvoP&R fixed-position stage=rc_tree_ready"
                ),
                "ed 2500 nets and 1273 pin-incomplete nets because placement names were absent",
            ]
        ),
        encoding="utf-8",
    )

    metrics = parse_dreamplace_log(log_path)

    assert metrics["timing_skipped_nets"] == 2500
    assert metrics["timing_pin_incomplete_nets"] == 1273


def test_parse_dreamplace_log_summarizes_component_history(tmp_path: Path) -> None:
    log_path = tmp_path / "dreamplace.log"
    log_path.write_text(
        "\n".join(
            [
                (
                    "INFO CoEvoP&R custom objective id=obj_x call=1 total=1.0E+00 "
                    "wirelength_raw=1.0E+02 weighted_density_raw=2.0E+00"
                ),
                (
                    'INFO CoEvoP&R custom objective components {"call": 1, '
                    '"components": {"route": {"value": 0.1}, "density": {"value": 1.0}, '
                    '"routing_correction": {"value": 0.01, "grad_norm": 0.1, "value_share": 0.01}}}'
                ),
                "INFO CoEvoP&R custom objective grad_norm=1.0E+01",
                (
                    "INFO CoEvoP&R custom objective id=obj_x call=2 total=1.1E+00 "
                    "wirelength_raw=1.0E+02 weighted_density_raw=2.0E+00"
                ),
                (
                    'INFO CoEvoP&R custom objective components {"call": 2, '
                    '"components": {"route": {"value": 0.2}, "density": {"value": 1.0}, '
                    '"routing_correction": {"value": 0.02, "grad_norm": 0.2, "value_share": 0.02}}}'
                ),
                "INFO CoEvoP&R custom objective grad_norm=1.0E+01",
                (
                    "INFO CoEvoP&R custom objective id=obj_x call=3 total=1.2E+00 "
                    "wirelength_raw=1.0E+02 weighted_density_raw=2.0E+00"
                ),
                (
                    'INFO CoEvoP&R custom objective components {"call": 3, '
                    '"components": {"route": {"value": 0.3}, "density": {"value": 1.0}, '
                    '"routing_correction": {"value": 0.03, "grad_norm": 0.3, "value_share": 0.03}}}'
                ),
                "INFO CoEvoP&R custom objective grad_norm=1.0E+01",
            ]
        ),
        encoding="utf-8",
    )

    custom = parse_dreamplace_log(log_path)["custom_objective"]

    assert len(custom["component_history"]) == 3
    assert custom["last_components"]["route"]["value"] == pytest.approx(0.3)
    assert custom["component_summary"]["route"]["start"] == pytest.approx(0.1)
    assert custom["component_summary"]["route"]["mid"] == pytest.approx(0.2)
    assert custom["component_summary"]["route"]["end"] == pytest.approx(0.3)
    assert custom["component_summary"]["route"]["trajectory"] == [0.1, 0.2, 0.3]
    assert custom["component_summary"]["route"]["min"] == pytest.approx(0.1)
    assert custom["component_summary"]["route"]["max"] == pytest.approx(0.3)
    assert custom["component_summary"]["route"]["trend"] == "up"
    assert custom["component_summary"]["density"]["flat_or_saturated"] is True
    correction = custom["component_summary"]["routing_correction"]
    assert correction["grad_ratio_count"] == 3
    assert correction["grad_ratio_mean"] == pytest.approx(0.02)
    assert correction["grad_ratio_max"] == pytest.approx(0.03)
    assert correction["grad_ratio_trajectory"] == [0.01, 0.02, 0.03]
    assert correction["value_share_mean"] == pytest.approx(0.02)


def test_regular_placement_log_does_not_claim_timing_coverage(tmp_path: Path) -> None:
    log_path = tmp_path / "dreamplace.log"
    log_path.write_text(
        "#nodes = 100, #terminals = 0, #movable = 90, #nets = 1000\n",
        encoding="utf-8",
    )

    metrics = parse_dreamplace_log(log_path)

    assert metrics["timing_total_nets"] == 1000
    assert metrics["timing_proxy_executed"] is False
    assert metrics["timing_net_coverage"] is None


def test_dreamplace_patch_anchors_custom_output_scale_to_wirelength_terms() -> None:
    patch = Path("patches/dreamplace/custom_objective_placeobj.patch").read_text(
        encoding="utf-8"
    )

    assert '("wirelength", "wirelength_wawl", "wirelength_lse", "pin_count_weighted_wl")' in patch
    assert "max(max(self.custom_objective_term_scales.values()), wirelength_scale)" not in patch


def test_dreamplace_patch_preserves_exact_native_identity_control() -> None:
    patch = Path("patches/dreamplace/custom_objective_placeobj.patch").read_text(
        encoding="utf-8"
    )

    assert 'custom_ast == {"op": "term", "name": "native_objective"}' in patch
    assert "def _coevop_typed_wl_density_fastpath" in patch
    assert "def coevop_sync_native_identity_state" in patch
    assert "model.coevop_sync_native_identity_state()" in patch
    assert "output_scale = 1.0" in patch
