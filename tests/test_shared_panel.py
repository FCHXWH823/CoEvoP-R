from pathlib import Path

from coevop.eval import shared_panel
from coevop.eval.shared_panel import check_shared_panel, load_shared_panel, write_dreamplace_panel


def _shared_panel_file(tmp_path: Path) -> Path:
    dreamplace_config = tmp_path / "bp_fe.json"
    chipbench_config = tmp_path / "config.mk"
    reference_def = tmp_path / "bp_fe.def"
    dreamplace_config.write_text("{}", encoding="utf-8")
    chipbench_config.write_text("DESIGN_NAME=bp_fe_top\n", encoding="utf-8")
    reference_def.write_text("VERSION 5.8 ;\n", encoding="utf-8")
    panel = tmp_path / "panel.toml"
    panel.write_text(
        "\n".join(
            [
                "seeds = [1000, 1001]",
                "iterations = 7",
                "gpu = 0",
                "timeout_seconds = 11",
                "log_interval = 3",
                'timing_mode = "none"',
                "stop_overflow = 0.0",
                "",
                "[[designs]]",
                'name = "bp_fe"',
                f'dreamplace_config = "{dreamplace_config.as_posix()}"',
                f'chipbench_config = "{chipbench_config.as_posix()}"',
                f'reference_def = "{reference_def.as_posix()}"',
                'mode = "global"',
                "freeze_placed_macros_for_routing = true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return panel


def test_load_shared_panel_and_write_dreamplace_panel(tmp_path: Path) -> None:
    panel_path = _shared_panel_file(tmp_path)
    panel = load_shared_panel(panel_path)

    assert panel.seeds == [1000, 1001]
    assert panel.designs[0].name == "bp_fe"
    assert panel.designs[0].mode == "global"
    assert panel.designs[0].freeze_placed_macros_for_routing is True
    assert panel.stop_overflow == 0.0

    dreamplace_panel = write_dreamplace_panel(panel, tmp_path / "dreamplace_panel.toml")
    text = dreamplace_panel.read_text(encoding="utf-8")
    assert 'base_config = "' in text
    assert 'name = "bp_fe"' in text
    assert "stop_overflow = 0.0" in text


def test_load_dreamplace_only_shared_panel(tmp_path: Path) -> None:
    dreamplace_config = tmp_path / "superblue1.json"
    dreamplace_config.write_text("{}", encoding="utf-8")
    panel_path = tmp_path / "dreamplace_only_panel.toml"
    panel_path.write_text(
        "\n".join(
            [
                "seeds = [1000]",
                "iterations = 1",
                "",
                "[[designs]]",
                'name = "superblue1"',
                f'dreamplace_config = "{dreamplace_config.as_posix()}"',
                'timing_mode = "none"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    panel = load_shared_panel(panel_path)

    assert Path(panel.designs[0].dreamplace_config) == dreamplace_config
    assert panel.designs[0].chipbench_config is None
    dreamplace_panel = write_dreamplace_panel(panel, tmp_path / "dreamplace_panel.toml")
    text = dreamplace_panel.read_text(encoding="utf-8")
    assert 'name = "superblue1"' in text
    assert "chipbench_config" not in text


def test_shared_panel_check_static(monkeypatch, tmp_path: Path) -> None:
    panel_path = _shared_panel_file(tmp_path)
    dreamplace_root = tmp_path / "dreamplace_root"
    placeobj = dreamplace_root / "dreamplace" / "PlaceObj.py"
    placeobj.parent.mkdir(parents=True)
    placeobj.write_text("def _coevop_build_extra_ops(self):\n    pass\n", encoding="utf-8")
    chipbench_root = tmp_path / "chipbench"
    chipbench_root.mkdir()

    monkeypatch.setattr(shared_panel.shutil, "which", lambda name: "/usr/bin/openroad")

    result = check_shared_panel(
        panel_path=panel_path,
        dreamplace_root=dreamplace_root,
        chipbench_root=chipbench_root,
        run_dir=tmp_path / "check",
        skip_runs=True,
    )

    assert result.ok is True
    assert result.openroad_path == "/usr/bin/openroad"
    assert all(item["exists"] for item in result.file_checks)
    assert (tmp_path / "check" / "shared_panel_check.json").exists()
