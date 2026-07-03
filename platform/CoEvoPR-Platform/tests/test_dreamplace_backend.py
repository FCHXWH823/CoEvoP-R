import json
from pathlib import Path

import pytest

from coevop.backends.dreamplace import is_patch_applied, make_run_config, parse_dreamplace_log
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
    )
    config = json.loads(output.read_text(encoding="utf-8"))

    assert config["custom_objective_spec"] == str(objective_path.resolve())
    assert config["custom_objective_log_interval"] == 3
    assert config["global_place_stages"][0]["iteration"] == 7
    assert config["gpu"] == 0
    assert config["random_seed"] == 1001
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
                "iteration 1 Obj 1.893280E+07, wHPWL 6.056268E+07, Overflow 8.788136E-01",
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
    assert metrics["last"]["hpwl"] == pytest.approx(6.056268e7)
    assert metrics["last"]["overflow"] == pytest.approx(8.788136e-1)
    assert metrics["custom_objective"]["objective_id"] == "obj_x"
    assert metrics["custom_objective"]["calls"] == 3
    assert metrics["custom_objective"]["last_grad_norm"] == pytest.approx(42.0)
    assert metrics["custom_objective"]["term_scales"]["wirelength"] == pytest.approx(6.0e7)
    assert metrics["custom_objective"]["output_scale"] == pytest.approx(6.0e7)
    assert metrics["custom_objective"]["last_terms"]["wirelength"]["normalized"] == pytest.approx(1.0)
    assert metrics["missing_input_file"] == "Could not open input file missing.lef"
    assert metrics["assertion"] == "[ASSERT ] failed to read input LEF files"


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
                    '"components": {"route": {"value": 0.1}, "density": {"value": 1.0}}}'
                ),
                (
                    "INFO CoEvoP&R custom objective id=obj_x call=2 total=1.1E+00 "
                    "wirelength_raw=1.0E+02 weighted_density_raw=2.0E+00"
                ),
                (
                    'INFO CoEvoP&R custom objective components {"call": 2, '
                    '"components": {"route": {"value": 0.2}, "density": {"value": 1.0}}}'
                ),
                (
                    "INFO CoEvoP&R custom objective id=obj_x call=3 total=1.2E+00 "
                    "wirelength_raw=1.0E+02 weighted_density_raw=2.0E+00"
                ),
                (
                    'INFO CoEvoP&R custom objective components {"call": 3, '
                    '"components": {"route": {"value": 0.3}, "density": {"value": 1.0}}}'
                ),
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


def test_dreamplace_patch_anchors_custom_output_scale_to_wirelength_terms() -> None:
    patch = Path("patches/dreamplace/custom_objective_placeobj.patch").read_text(
        encoding="utf-8"
    )

    assert '("wirelength", "wirelength_wawl", "wirelength_lse", "pin_count_weighted_wl")' in patch
    assert "max(max(self.custom_objective_term_scales.values()), wirelength_scale)" not in patch
