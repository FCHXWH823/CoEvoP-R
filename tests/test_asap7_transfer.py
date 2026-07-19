import json
from pathlib import Path

from coevop.eval import asap7_transfer
from coevop.eval.asap7_transfer import (
    MANIFEST_ROW_FIELDS,
    build_manifest_rows,
    check_asap7_panel,
    classify_zero_shot_success,
    generate_asap7_base_configs,
    resolve_ethernet_source_objective,
    run_asap7_transfer,
)


def _objective(path: Path, *, objective_id: str = "obj_source") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "id": objective_id,
                "ast": {"op": "term", "name": "wirelength_wawl"},
                "constants": {},
                "term_set": ["wirelength_wawl"],
                "complexity": 1,
                "created_by": "test",
                "parent_ids": [],
                "rationale": "test objective",
            }
        ),
        encoding="utf-8",
    )
    return path


def _asap7_tree(tmp_path: Path, *, platform: bool = True, dreamplace_configs: bool = True) -> tuple[Path, Path, Path]:
    chipbench_root = tmp_path / "ChiPBench"
    dreamplace_root = tmp_path / "DREAMPlace" / "install"
    dreamplace_config_dir = tmp_path / "configs" / "dreamplace_base" / "asap7"
    for design in ("gcd", "ibex", "ariane"):
        design_dir = chipbench_root / "flow" / "designs" / "asap7" / design
        design_dir.mkdir(parents=True, exist_ok=True)
        (design_dir / "config.mk").write_text(
            f"export PLATFORM = asap7\nexport DESIGN_NAME = {design}\nexport PLACE_DENSITY = 0.35\n",
            encoding="utf-8",
        )
        if dreamplace_configs:
            dreamplace_config_dir.mkdir(parents=True, exist_ok=True)
            (dreamplace_config_dir / f"{design}.json").write_text("{}", encoding="utf-8")
    if platform:
        platform_dir = chipbench_root / "flow" / "platforms" / "asap7"
        platform_dir.mkdir(parents=True, exist_ok=True)
        (platform_dir / "config.mk").write_text("PLATFORM=asap7\n", encoding="utf-8")
        (platform_dir / "asap7.lef").write_text("VERSION 5.8 ;\n", encoding="utf-8")
    dreamplace_root.mkdir(parents=True, exist_ok=True)
    return chipbench_root, dreamplace_root, dreamplace_config_dir


def test_asap7_panel_check_reports_missing_platform(monkeypatch, tmp_path: Path) -> None:
    chipbench_root, dreamplace_root, dreamplace_config_dir = _asap7_tree(
        tmp_path,
        platform=False,
    )
    monkeypatch.setattr(asap7_transfer, "is_patch_applied", lambda root: True)
    monkeypatch.setattr(
        asap7_transfer,
        "_command_metadata",
        lambda *args, **kwargs: {"returncode": 0, "stdout": "OpenROAD test"},
    )

    payload = check_asap7_panel(
        chipbench_root=chipbench_root,
        dreamplace_root=dreamplace_root,
        dreamplace_config_dir=dreamplace_config_dir,
        run_dir=tmp_path / "check",
        skip_make_dry_run=True,
    )

    assert payload["ok"] is False
    assert payload["bootstrap"]["exists"] is False
    assert "Copy or symlink" in payload["bootstrap"]["required_manual_fix"]


def test_resolve_ethernet_source_objective_selects_latest(tmp_path: Path) -> None:
    old = _objective(tmp_path / "runs" / "old" / "frozen_nangate45_champion.json")
    new = _objective(tmp_path / "runs" / "new" / "frozen_nangate45_champion.json")
    old_time = 1_700_000_000
    new_time = old_time + 100
    old.touch()
    new.touch()
    import os

    os.utime(old, (old_time, old_time))
    os.utime(new, (new_time, new_time))

    resolution = resolve_ethernet_source_objective(source_root=tmp_path / "runs")

    assert resolution.status == "resolved"
    assert resolution.objective_path == str(new)


def test_asap7_base_config_generation_blocks_without_orfs_def(tmp_path: Path) -> None:
    chipbench_root, _, _ = _asap7_tree(tmp_path, platform=True)

    summary = generate_asap7_base_configs(
        chipbench_root=chipbench_root,
        output_dir=tmp_path / "generated",
        designs=["gcd"],
    )

    assert summary["ok"] is False
    assert summary["designs"][0]["status"] == "blocked"
    assert "DEF" in summary["designs"][0]["reason"]


def test_asap7_base_config_generation_writes_config_when_def_and_lef_exist(tmp_path: Path) -> None:
    chipbench_root, _, _ = _asap7_tree(tmp_path, platform=True)
    def_dir = chipbench_root / "flow" / "results" / "asap7" / "gcd" / "base"
    def_dir.mkdir(parents=True, exist_ok=True)
    (def_dir / "2_1_floorplan.def").write_text(
        "VERSION 5.8 ;\nCOMPONENTS 1 ;\n    - tie0 TIEHIx1_ASAP7_75t_R ;\nEND COMPONENTS\n",
        encoding="utf-8",
    )

    summary = generate_asap7_base_configs(
        chipbench_root=chipbench_root,
        output_dir=tmp_path / "generated",
        designs=["gcd"],
        flow_variant="base",
    )

    assert summary["ok"] is True
    payload = json.loads((tmp_path / "generated" / "gcd.json").read_text(encoding="utf-8"))
    assert "design_name" not in payload
    assert payload["asap7_generation_audit"]["design_name"] == "gcd"
    assert payload["lef_input"]
    assert payload["def_input"].startswith("${CHIPBENCH_ROOT}/")
    assert payload["def_input"].endswith("2_1_floorplan.def.coevop_dreamplace.def")
    expanded_def = Path(payload["def_input"].replace("${CHIPBENCH_ROOT}", str(chipbench_root)))
    assert "+ FIXED ( 0 0 ) N" in expanded_def.read_text(encoding="utf-8")
    assert payload["target_density"] == 0.35


def test_build_manifest_rows_uses_paper_schema() -> None:
    placement_rows = [
        {
            "design": "gcd",
            "objective_id": "ours_zero_shot",
            "seed": "1000",
            "status": "success",
            "hpwl": "10",
            "overflow": "0.1",
            "runtime_seconds": "60",
        }
    ]
    post_route_rows = [
        {
            "design": "gcd",
            "objective_id": "ours_zero_shot",
            "seed": "1000",
            "status": "success",
            "routed_wirelength": "20",
            "grt_overflow": "3",
            "runtime_seconds": "120",
        }
    ]

    rows = build_manifest_rows(
        placement_rows=placement_rows,
        post_route_rows=post_route_rows,
        source="our_run",
    )

    assert list(rows[0].keys()) == MANIFEST_ROW_FIELDS
    assert rows[0]["library"] == "asap7"
    assert rows[0]["method_group"] == "ours_zero_shot"
    assert rows[0]["detailed_route_wirelength"] == 20.0
    assert rows[0]["runtime_min"] == 3.0


def test_zero_shot_success_requires_two_designs_with_ci_below_zero() -> None:
    placement_rows = []
    post_route_rows = []
    for design, delta in (("gcd", -5.0), ("ibex", -4.0), ("ariane", 2.0)):
        for seed in (1000, 1001, 1002):
            placement_rows.append(
                {
                    "design": design,
                    "objective_id": "ours_zero_shot",
                    "seed": seed,
                    "hpwl_delta_pct": delta,
                }
            )

    summary = classify_zero_shot_success(
        placement_rows=placement_rows,
        post_route_rows=post_route_rows,
    )

    assert summary["success"] is True
    assert summary["passed_designs"] == ["gcd", "ibex"]


def test_asap7_transfer_dry_run_prepares_panels_and_zero_shot_objective(
    monkeypatch,
    tmp_path: Path,
) -> None:
    chipbench_root, dreamplace_root, dreamplace_config_dir = _asap7_tree(tmp_path, platform=True)
    source = _objective(tmp_path / "source" / "robust_best_objective.json")
    monkeypatch.setattr(asap7_transfer, "is_patch_applied", lambda root: True)
    monkeypatch.setattr(
        asap7_transfer,
        "_command_metadata",
        lambda *args, **kwargs: {"returncode": 0, "stdout": "OpenROAD test"},
    )

    summary = run_asap7_transfer(
        run_dir=tmp_path / "run",
        dreamplace_root=dreamplace_root,
        chipbench_root=chipbench_root,
        dreamplace_config_dir=dreamplace_config_dir,
        source_objective=source,
        designs=["gcd"],
        seeds=[1000],
        dry_run=True,
    )

    assert summary["status"] == "prepared"
    assert Path(summary["panel"]).is_file()
    assert Path(summary["post_route_panel"]).is_file()
    post_route_panel_text = Path(summary["post_route_panel"]).read_text(encoding="utf-8")
    assert 'mode = "global"' in post_route_panel_text
    prepared = json.loads((tmp_path / "run" / "objectives" / "ours_zero_shot.json").read_text(encoding="utf-8"))
    assert prepared["id"] == "ours_zero_shot"
    assert Path(summary["manifest_rows"]).is_file()
