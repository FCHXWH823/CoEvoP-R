import json
from pathlib import Path

import pytest

from coevop.backends.dreamplace import DreamPlaceRun
from coevop.eval import tier2_dreamplace
from coevop.eval.tier2_dreamplace import load_panel, prepare_objectives, run_tier2_dreamplace
from coevop.objectives.presets import objective_preset
from coevop.objectives.spec import parse_objective_spec, write_objective_spec


def _base_config(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "gpu": 0,
                "random_seed": 1000,
                "global_place_stages": [{"iteration": 1000}],
                "legalize_flag": 1,
                "detailed_place_flag": 1,
            }
        ),
        encoding="utf-8",
    )
    return path


def _panel(path: Path, base_config: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "seeds = [1000, 1001]",
                "iterations = 7",
                "gpu = 0",
                "timeout_seconds = 11",
                "log_interval = 3",
                'timing_mode = "compatibility_only"',
                "stop_overflow = 0.0",
                'driver = "/tmp/report_only.py"',
                "",
                "[[designs]]",
                'name = "toy"',
                f'base_config = "{base_config.as_posix()}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_load_panel_supports_required_fields(tmp_path: Path) -> None:
    base_config = _base_config(tmp_path / "base.json")
    panel = load_panel(_panel(tmp_path / "panel.toml", base_config))

    assert panel.seeds == [1000, 1001]
    assert panel.iterations == 7
    assert panel.gpu == 0
    assert panel.timeout_seconds == 11
    assert panel.log_interval == 3
    assert panel.timing_mode == "compatibility_only"
    assert panel.stop_overflow == 0.0
    assert panel.driver == "/tmp/report_only.py"
    assert panel.designs[0].name == "toy"


def test_load_panel_accepts_shared_panel_dreamplace_config_alias(tmp_path: Path) -> None:
    base_config = _base_config(tmp_path / "base.json")
    panel_path = _panel(tmp_path / "panel.toml", base_config)
    panel_path.write_text(
        panel_path.read_text(encoding="utf-8").replace("base_config", "dreamplace_config"),
        encoding="utf-8",
    )

    panel = load_panel(panel_path)

    assert Path(panel.designs[0].base_config) == base_config


def test_load_panel_rejects_invalid_timing_mode(tmp_path: Path) -> None:
    base_config = _base_config(tmp_path / "base.json")
    panel = _panel(tmp_path / "panel.toml", base_config)
    panel.write_text(panel.read_text(encoding="utf-8").replace("compatibility_only", "maybe"))

    with pytest.raises(ValueError, match="invalid timing_mode"):
        load_panel(panel)


def test_tier2_runner_writes_expected_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base_config = _base_config(tmp_path / "base.json")
    panel_path = _panel(tmp_path / "panel.toml", base_config)
    objective_path = write_objective_spec(
        objective_preset("dreamplace_density_heavy"),
        tmp_path / "density_heavy.json",
    )
    calls = []

    def fake_run_dreamplace(**kwargs):
        calls.append(kwargs["run_dir"])
        run_dir = Path(kwargs["run_dir"])
        results = run_dir / "results" / "toy"
        results.mkdir(parents=True)
        (results / "toy.gp.pl").write_text("placement", encoding="utf-8")
        objective_id = "obj_x"
        if "custom_default" in str(run_dir):
            objective_id = "custom_default"
        elif "default" in str(run_dir):
            objective_id = None
        return DreamPlaceRun(
            run_name=kwargs["run_name"],
            run_dir=str(run_dir),
            config_path=str(kwargs["config_path"]),
            log_path=str(run_dir / "dreamplace.log"),
            returncode=0,
            command=["fake"],
            metrics={
                "last": {"hpwl": 100.0, "overflow": 0.5},
                "global_place_iterations_completed": 7,
                "custom_objective": {
                    "objective_id": objective_id,
                    "calls": 2 if objective_id else 0,
                    "last_total": 10.0 if objective_id else None,
                    "last_grad_norm": 1.5 if objective_id else None,
                },
            },
        )

    monkeypatch.setattr(tier2_dreamplace, "run_dreamplace", fake_run_dreamplace)
    summary = run_tier2_dreamplace(
        panel_path=panel_path,
        objective_paths=[objective_path],
        dreamplace_root=tmp_path,
        run_dir=tmp_path / "tier2",
        resume=False,
    )

    assert summary["result_count"] == 6
    assert len(calls) == 6
    for filename in (
        "panel_manifest.json",
        "objective_manifest.json",
        "metrics.csv",
        "comparison_table.csv",
        "tier2_report.md",
        "tier2.sqlite",
        "summary.json",
    ):
        assert (tmp_path / "tier2" / filename).exists()
    run_summary_path = tmp_path / "tier2" / "toy" / "default" / "seed_1000" / "run_summary.json"
    assert run_summary_path.exists()
    assert (tmp_path / "tier2" / "toy" / "default" / "seed_1000" / "dreamplace_run.json").exists()
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    assert run_summary["status"] == "success"
    assert "runtime_seconds" in run_summary


def test_tier2_runtime_guard_rejects_zero_movable_macro_area(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_config = _base_config(tmp_path / "base.json")
    payload = json.loads(base_config.read_text(encoding="utf-8"))
    # Search runs may disable DREAMPlace's macro legalizer while keeping DEF
    # macros movable. The runtime guard must remain active in that mode.
    payload["macro_place_flag"] = 0
    payload["coevop_macro_preflight"] = {
        "movable_macro_count": 2,
        "fixed_hard_macro_count": 0,
        "input_def_sha256": "abc123",
    }
    base_config.write_text(json.dumps(payload), encoding="utf-8")
    panel_path = _panel(tmp_path / "panel.toml", base_config)

    def fake_run_dreamplace(**kwargs):
        run_dir = Path(kwargs["run_dir"])
        return DreamPlaceRun(
            run_name=kwargs["run_name"],
            run_dir=str(run_dir),
            config_path=str(kwargs["config_path"]),
            log_path=str(run_dir / "dreamplace.log"),
            returncode=0,
            command=["fake"],
            metrics={
                "last": {"hpwl": 100.0, "overflow": 0.5},
                "custom_objective": {},
                "movable_macro_area": 0.0,
                "global_place_iterations_completed": 7,
            },
        )

    monkeypatch.setattr(tier2_dreamplace, "run_dreamplace", fake_run_dreamplace)
    summary = run_tier2_dreamplace(
        panel_path=panel_path,
        objective_paths=[],
        dreamplace_root=tmp_path,
        run_dir=tmp_path / "tier2",
        resume=False,
    )

    assert summary["result_count"] == 4
    run_summary = json.loads(
        (tmp_path / "tier2" / "toy" / "default" / "seed_1000" / "run_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_summary["status"] == "failed"
    assert run_summary["failure_stage"] == "macro_runtime"
    assert not run_summary["metrics"]["macro_runtime_validation"]["ok"]
    assert run_summary["metrics"]["macro_runtime_validation"]["macro_place_flag"] == 0


def test_unplaced_macro_compatibility_accepts_complete_output_coordinates() -> None:
    reconciled = tier2_dreamplace._reconcile_macro_runtime_validation(
        {
            "required": True,
            "ok": False,
            "reason": "DREAMPlace reported zero or missing movable macro area",
        },
        {"required": True, "ok": True, "coordinate_count": 2},
        {
            "expected_movable_macro_count": 2,
            "input_unplaced_macro_count": 2,
        },
    )

    assert reconciled["ok"] is True
    assert reconciled["validation_mode"] == "unplaced_input_output_coordinates"


def test_tier2_rejects_metrics_before_requested_iteration_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_config = _base_config(tmp_path / "base.json")
    panel_path = _panel(tmp_path / "panel.toml", base_config)

    def fake_run_dreamplace(**kwargs):
        run_dir = Path(kwargs["run_dir"])
        return DreamPlaceRun(
            run_name=kwargs["run_name"],
            run_dir=str(run_dir),
            config_path=str(kwargs["config_path"]),
            log_path=str(run_dir / "dreamplace.log"),
            returncode=0,
            command=["fake"],
            metrics={
                "last": {"hpwl": 1.0, "overflow": 0.0},
                "custom_objective": {},
                "global_place_iterations_completed": 6,
            },
        )

    monkeypatch.setattr(tier2_dreamplace, "run_dreamplace", fake_run_dreamplace)
    summary = run_tier2_dreamplace(
        panel_path=panel_path,
        objective_paths=[],
        dreamplace_root=tmp_path,
        run_dir=tmp_path / "tier2",
        include_custom_default=False,
        resume=False,
    )

    assert summary["result_count"] == 2
    run_summary = json.loads(
        (tmp_path / "tier2" / "toy" / "default" / "seed_1000" / "run_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_summary["status"] == "failed"
    assert run_summary["failure_stage"] == "iteration_budget"
    assert run_summary["requested_iterations"] == 7
    assert run_summary["completed_iterations"] == 6
    assert run_summary["iteration_budget_satisfied"] is False


def test_custom_default_is_native_identity_control(tmp_path: Path) -> None:
    objectives = prepare_objectives(
        [],
        run_root=tmp_path / "tier2",
        include_default=True,
        include_custom_default=True,
    )

    custom_default = next(item for item in objectives if item.objective_id == "custom_default")
    assert custom_default.source == "native_identity_preset"
    assert custom_default.term_set == ["native_objective"]
    payload = json.loads(Path(custom_default.objective_path).read_text(encoding="utf-8"))
    assert payload["term_set"] == ["native_objective"]
    assert payload["ast"] == {"op": "term", "name": "native_objective"}


def test_tier2_runner_rejects_unsupported_tier1_terms(tmp_path: Path) -> None:
    base_config = _base_config(tmp_path / "base.json")
    panel_path = _panel(tmp_path / "panel.toml", base_config)
    spec = parse_objective_spec(
        {
            "id": "",
            "rationale": "Tier-1-only RUDY term.",
            "parent_ids": [],
            "declared_term_usage": ["rudy_p95"],
            "ast": {"op": "term", "name": "rudy_p95"},
        },
        created_by="test",
    )
    objective_path = write_objective_spec(spec, tmp_path / "rudy.json")

    with pytest.raises(ValueError, match="unsupported"):
        run_tier2_dreamplace(
            panel_path=panel_path,
            objective_paths=[objective_path],
            dreamplace_root=tmp_path,
            run_dir=tmp_path / "tier2",
        )


def test_resume_reuses_failed_cells_unless_retry_requested_then_skips_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_config = _base_config(tmp_path / "base.json")
    panel_path = _panel(tmp_path / "panel.toml", base_config)
    calls = []
    state = {"returncode": 1}

    def fake_run_dreamplace(**kwargs):
        calls.append(kwargs["run_dir"])
        run_dir = Path(kwargs["run_dir"])
        metrics = {"last": {}, "custom_objective": {}}
        if state["returncode"] == 0:
            metrics = {
                "last": {"hpwl": 1.0, "overflow": 0.0},
                "custom_objective": {},
                "global_place_iterations_completed": 7,
            }
        return DreamPlaceRun(
            run_name=kwargs["run_name"],
            run_dir=str(run_dir),
            config_path=str(kwargs["config_path"]),
            log_path=str(run_dir / "dreamplace.log"),
            returncode=state["returncode"],
            command=["fake"],
            metrics=metrics,
        )

    monkeypatch.setattr(tier2_dreamplace, "run_dreamplace", fake_run_dreamplace)
    run_tier2_dreamplace(
        panel_path=panel_path,
        objective_paths=[],
        dreamplace_root=tmp_path,
        run_dir=tmp_path / "tier2",
        include_custom_default=False,
        resume=False,
    )
    first_call_count = len(calls)
    assert first_call_count == 2

    state["returncode"] = 0
    run_tier2_dreamplace(
        panel_path=panel_path,
        objective_paths=[],
        dreamplace_root=tmp_path,
        run_dir=tmp_path / "tier2",
        include_custom_default=False,
        resume=True,
    )
    second_call_count = len(calls)
    assert second_call_count == first_call_count

    run_tier2_dreamplace(
        panel_path=panel_path,
        objective_paths=[],
        dreamplace_root=tmp_path,
        run_dir=tmp_path / "tier2",
        include_custom_default=False,
        resume=True,
        retry_failed=True,
    )
    third_call_count = len(calls)
    assert third_call_count == 4

    run_tier2_dreamplace(
        panel_path=panel_path,
        objective_paths=[],
        dreamplace_root=tmp_path,
        run_dir=tmp_path / "tier2",
        include_custom_default=False,
        resume=True,
    )
    assert len(calls) == third_call_count


def test_legacy_raw_summary_resume_preserves_tool_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "toy" / "default" / "seed_1000"
    run_dir.mkdir(parents=True)
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "run_name": "toy_default_seed_1000",
                "run_dir": str(run_dir),
                "config_path": str(run_dir / "dreamplace_config.json"),
                "log_path": str(run_dir / "dreamplace.log"),
                "returncode": 0,
                "command": ["fake"],
                "metrics": {
                    "last": {"hpwl": 10.0, "overflow": 0.1},
                    "custom_objective": {"calls": 0},
                },
            }
        ),
        encoding="utf-8",
    )

    result = tier2_dreamplace._load_result_from_summary(
        summary_path,
        "toy",
        "default",
        1000,
        tool_versions={
            "coevop": {"commit": "coevop-test"},
            "dreamplace": {"commit": "dreamplace-test"},
        },
    )

    assert result.status == "success"
    assert result.coevop_commit == "coevop-test"
    assert result.dreamplace_commit == "dreamplace-test"


def test_resume_persists_revalidated_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_config = _base_config(tmp_path / "base.json")
    panel = load_panel(_panel(tmp_path / "panel.toml", base_config))
    design = panel.designs[0]
    objective = prepare_objectives(
        [],
        run_root=tmp_path / "tier2",
        include_default=True,
        include_custom_default=False,
    )[0]
    run_dir = tmp_path / "tier2" / "toy" / "default" / "seed_1000"
    run_dir.mkdir(parents=True)
    summary_path = run_dir / "run_summary.json"
    summary_path.write_text('{"status":"failed"}', encoding="utf-8")
    revalidated = tier2_dreamplace.Tier2Result(
        design="toy",
        objective_id="default",
        seed=1000,
        status="success",
        failure_stage=None,
        hpwl=10.0,
        overflow=0.1,
        wns=None,
        tns=None,
        requested_iterations=7,
        completed_iterations=7,
        iteration_budget_satisfied=True,
        custom_objective_calls=0,
        custom_total=None,
        custom_grad_norm=None,
        component_summary_json="{}",
        runtime_seconds=12.5,
        output_artifact=str(run_dir / "toy.gp.def"),
        coevop_commit="coevop-test",
        dreamplace_commit="dreamplace-test",
        run_dir=str(run_dir),
        config_path=str(base_config),
        log_path=str(run_dir / "dreamplace.log"),
        returncode=0,
        metrics={"last": {"hpwl": 10.0, "overflow": 0.1}},
    )
    monkeypatch.setattr(
        tier2_dreamplace,
        "_load_result_from_summary",
        lambda *args, **kwargs: revalidated,
    )
    monkeypatch.setattr(
        tier2_dreamplace,
        "run_dreamplace",
        lambda **kwargs: pytest.fail("successful resumed cell must not rerun"),
    )

    result = tier2_dreamplace._run_one_cell(
        dreamplace_root=tmp_path,
        panel=panel,
        design=design,
        objective=objective,
        seed=1000,
        run_root=tmp_path / "tier2",
        resume=True,
        retry_failed=True,
        tool_versions={},
    )

    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    assert result.status == "success"
    assert persisted["status"] == "success"
    assert persisted["runtime_seconds"] == 12.5
