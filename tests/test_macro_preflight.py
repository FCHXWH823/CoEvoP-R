import json
from pathlib import Path

from coevop.eval.macro_preflight import (
    audit_dreamplace_macro_config,
    audit_output_macro_coordinates,
)
from coevop.eval.openevolve_tier2 import run_openevolve_tier2
import pytest


def _write_config(tmp_path: Path, *, macro_status: str) -> Path:
    dreamplace_root = tmp_path / "DREAMPlace" / "install"
    bench = dreamplace_root / "benchmarks" / "test"
    bench.mkdir(parents=True)
    (bench / "macros.lef").write_text(
        "MACRO SRAM\n  CLASS BLOCK ;\n  SIZE 10 BY 20 ;\nEND SRAM\n",
        encoding="utf-8",
    )
    (bench / "input.def").write_text(
        "VERSION 5.8 ;\n"
        "COMPONENTS 1 ;\n"
        f"- macro0 SRAM + {macro_status} ( 10 20 ) N ;\n"
        "END COMPONENTS\nEND DESIGN\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "lef_input": ["benchmarks/test/macros.lef"],
                "def_input": "benchmarks/test/input.def",
                "macro_place_flag": 1,
            }
        ),
        encoding="utf-8",
    )
    return config


def test_macro_preflight_accepts_unplaced_hard_macro(tmp_path: Path) -> None:
    config = _write_config(tmp_path, macro_status="UNPLACED")
    audit = audit_dreamplace_macro_config(
        config,
        dreamplace_root=tmp_path / "DREAMPlace" / "install",
        require_movable_macros=True,
        reject_fixed_hard_macros=True,
    )

    assert audit["ok"] is True
    assert audit["movable_macro_count"] == 1
    assert audit["fixed_macro_count"] == 0


def test_macro_preflight_rejects_fixed_hard_macro(tmp_path: Path) -> None:
    config = _write_config(tmp_path, macro_status="FIXED")
    audit = audit_dreamplace_macro_config(
        config,
        dreamplace_root=tmp_path / "DREAMPlace" / "install",
        require_movable_macros=True,
        reject_fixed_hard_macros=True,
    )

    assert audit["ok"] is False
    assert audit["movable_macro_count"] == 0
    assert audit["fixed_macro_count"] == 1
    assert any("FIXED/COVER" in reason for reason in audit["failures"])


def test_macro_preflight_enforces_exact_selection_manifest(tmp_path: Path) -> None:
    config = _write_config(tmp_path, macro_status="PLACED")
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["macro_place_flag"] = 0
    payload["coevop_superblue_macro_selection"] = {
        "expected_movable_macro_count": 1,
        "movable_instances": ["different_macro"],
    }
    config.write_text(json.dumps(payload), encoding="utf-8")

    audit = audit_dreamplace_macro_config(
        config,
        dreamplace_root=tmp_path / "DREAMPlace" / "install",
        require_movable_macros=True,
        reject_fixed_hard_macros=False,
    )

    assert audit["ok"] is False
    assert audit["movable_macro_count"] == 1
    assert audit["macro_place_flag"] == 0
    assert audit["movable_instance_set_matches"] is False
    assert any("instance set" in reason for reason in audit["failures"])


def test_output_macro_audit_requires_coordinates_for_every_macro(tmp_path: Path) -> None:
    config = _write_config(tmp_path, macro_status="UNPLACED")
    audit = audit_dreamplace_macro_config(
        config,
        dreamplace_root=tmp_path / "DREAMPlace" / "install",
        require_movable_macros=True,
        reject_fixed_hard_macros=True,
    )
    output_def = tmp_path / "placed.def"
    output_def.write_text(
        "VERSION 5.8 ;\n"
        "COMPONENTS 1 ;\n"
        "- macro0 SRAM + PLACED ( 100 200 ) FS ;\n"
        "END COMPONENTS\nEND DESIGN\n",
        encoding="utf-8",
    )

    result = audit_output_macro_coordinates(
        input_components=audit["components"],
        output_def=output_def,
    )

    assert result["ok"] is True
    assert result["expected_macro_count"] == 1
    assert result["coordinate_count"] == 1
    assert result["input_unplaced_count"] == 1
    assert result["coordinate_sha256"]


def test_output_macro_audit_ignores_fixed_instances_of_selected_master(
    tmp_path: Path,
) -> None:
    output_def = tmp_path / "placed.def"
    output_def.write_text(
        "VERSION 5.8 ;\n"
        "COMPONENTS 2 ;\n"
        "- selected_macro SRAM + PLACED ( 100 200 ) N ;\n"
        "- fixed_obstacle SRAM + FIXED ( 300 400 ) N ;\n"
        "END COMPONENTS\nEND DESIGN\n",
        encoding="utf-8",
    )

    result = audit_output_macro_coordinates(
        input_components=[
            {
                "instance": "selected_macro",
                "master": "SRAM",
                "status": "PLACED",
                "x": 10,
                "y": 20,
                "orientation": "N",
            }
        ],
        output_def=output_def,
    )

    assert result["ok"] is True
    assert result["expected_macro_count"] == 1
    assert result["observed_macro_count"] == 1
    assert result["changed_coordinate_count"] == 1
    assert result["unexpected_instances"] == []
    assert result["nonselected_same_master_instance_count"] == 1
    assert result["nonselected_same_master_instances_sample"] == ["fixed_obstacle"]


def test_openevolve_fails_before_llm_on_fixed_macro_panel(tmp_path: Path) -> None:
    config = _write_config(tmp_path, macro_status="FIXED")
    panel = tmp_path / "panel.toml"
    panel.write_text(
        "seeds = [1000]\niterations = 1\n"
        "[[designs]]\n"
        "name = \"fixed_design\"\n"
        f'dreamplace_config = "{config.as_posix()}"\n',
        encoding="utf-8",
    )
    evolution = tmp_path / "evolution.toml"
    evolution.write_text(
        "provider = \"mock\"\n"
        "term_scope = \"dreamplace_replacement\"\n"
            "objective_mode = \"replacement\"\n"
            "max_iterations = 1\n"
            "allow_short_search_for_tests = true\n"
            "[macro_preflight]\n"
        "enabled = true\n"
        "require_movable_macros = true\n"
        "reject_fixed_hard_macros = true\n"
        "[search]\n"
        f'panel = "{panel.as_posix()}"\n',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Movable-macro preflight failed"):
        run_openevolve_tier2(
            config_path=evolution,
            run_dir=tmp_path / "run",
            dreamplace_root=tmp_path / "DREAMPlace" / "install",
        )
    assert (tmp_path / "run" / "macro_preflight.json").is_file()
    assert not (tmp_path / "run" / "term_scale_audit").exists()
