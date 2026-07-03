import csv
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from coevop.eval import tier3_openroad
from coevop.eval.tier3_openroad import (
    load_placements,
    parse_chipbench_metrics,
    parse_chipbench_partial_metrics,
    recover_tier3_partial_run,
    run_tier3_openroad,
)


def _panel(tmp_path: Path, *, mode: str = "global", timeout_seconds: int = 9) -> Path:
    dreamplace_config = tmp_path / "bp_fe.json"
    chipbench_config = tmp_path / "config.mk"
    dreamplace_config.write_text("{}", encoding="utf-8")
    chipbench_config.write_text("DESIGN_NAME=bp_fe_top\n", encoding="utf-8")
    panel = tmp_path / "panel.toml"
    panel.write_text(
        "\n".join(
            [
                "seeds = [1000]",
                "iterations = 1",
                f"timeout_seconds = {timeout_seconds}",
                "",
                "[[designs]]",
                'name = "bp_fe"',
                f'dreamplace_config = "{dreamplace_config.as_posix()}"',
                f'chipbench_config = "{chipbench_config.as_posix()}"',
                f'mode = "{mode}"',
                'global_route_args = "-allow_congestion -verbose -congestion_iterations 5"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return panel


def test_parse_chipbench_metrics_handles_nested_payload(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "route": {"routed_wirelength": 123.0, "grt_overflow": 4},
                "checks": {"drc_count": 2},
                "timing": {"wns": -0.1, "tns": -3.0},
            }
        ),
        encoding="utf-8",
    )

    parsed = parse_chipbench_metrics(metrics_path)

    assert parsed["routed_wirelength"] == 123.0
    assert parsed["grt_overflow"] == 4.0
    assert parsed["drc_count"] == 2.0
    assert parsed["wns"] == -0.1


def test_load_placements_accepts_utf8_bom(tmp_path: Path) -> None:
    def_path = tmp_path / "placement.def"
    def_path.write_text("VERSION 5.8 ;\n", encoding="utf-8")
    placements_path = tmp_path / "placements.json"
    placements_path.write_text(
        json.dumps(
            {
                "placements": [
                    {
                        "design": "bp_fe",
                        "objective_id": "candidate",
                        "seed": 1000,
                        "def_path": str(def_path),
                    }
                ]
            }
        ),
        encoding="utf-8-sig",
    )

    placements = load_placements(placements_path)

    assert len(placements) == 1
    assert placements[0].objective_id == "candidate"


def test_parse_chipbench_partial_metrics_uses_latest_completed_stage(tmp_path: Path) -> None:
    stage_dir = tmp_path / "logs" / "nangate45" / "bp_fe" / "run_a"
    stage_dir.mkdir(parents=True)
    (stage_dir / "3_5_place_dp.json").write_text(
        json.dumps(
            {
                "detailedplace__route__wirelength__estimated": 200.0,
                "detailedplace__timing__setup__ws": -0.3,
                "detailedplace__timing__setup__tns": -5.0,
                "detailedplace__power__total": 0.2,
                "detailedplace__design__instance__area__stdcell": 20.0,
            }
        ),
        encoding="utf-8",
    )
    (stage_dir / "4_1_cts.json").write_text(
        json.dumps(
            {
                "cts__route__wirelength__estimated": 180.0,
                "cts__timing__setup__ws": -0.1,
                "cts__timing__setup__tns": -3.0,
                "cts__power__total": 0.25,
                "cts__design__instance__area__stdcell": 25.0,
                "cts__design__violations": 0,
            }
        ),
        encoding="utf-8",
    )
    (stage_dir / "5_1_grt.log").write_text(
        "[INFO GRT-0019] Found 12 clock nets.\n"
        "[INFO GRT-0101] Running extra iterations to remove overflow.\n",
        encoding="utf-8",
    )

    parsed = parse_chipbench_partial_metrics(stage_dir)

    assert parsed is not None
    assert parsed["metrics_stage"] == "partial_cts"
    assert parsed["estimated_wirelength"] == 180.0
    assert parsed["routed_wirelength"] is None
    assert parsed["wns"] == -0.1
    assert parsed["tns"] == -3.0
    assert parsed["power"] == 0.25
    assert parsed["area"] == 25.0
    assert parsed["raw"]["global_route_log"]["clock_net_count"] == 12.0


def test_parse_chipbench_partial_metrics_prefers_completed_grt_json(tmp_path: Path) -> None:
    stage_dir = tmp_path / "flow" / "logs" / "nangate45" / "bp_fe" / "run_a"
    stage_dir.mkdir(parents=True)
    report_dir = tmp_path / "flow" / "reports" / "nangate45" / "bp_fe" / "run_a"
    report_dir.mkdir(parents=True)
    (stage_dir / "4_1_cts.json").write_text(
        json.dumps(
            {
                "cts__route__wirelength__estimated": 180.0,
                "cts__timing__setup__ws": -0.1,
                "cts__timing__setup__tns": -3.0,
            }
        ),
        encoding="utf-8",
    )
    (stage_dir / "5_1_grt.json").write_text(
        json.dumps(
            {
                "globalroute__drc_count": 0,
            }
        ),
        encoding="utf-8",
    )
    (stage_dir / "5_1_grt.log").write_text(
        "[INFO GRT-0096] Final congestion report:\n"
        "Layer         Resource        Demand        Usage (%)    Max H / Max V / Total Overflow\n"
        "---------------------------------------------------------------------------------------\n"
        "Total          2714911       2947256          108.56%            100 / 87 / 1251012\n"
        "[INFO GRT-0018] Total wirelength: 7036253 um\n",
        encoding="utf-8",
    )
    (report_dir / "congestion.rpt").write_text(
        "violation type: Vertical congestion\n"
        "\tcomment: capacity:17 usage:36 overflow:19\n"
        "violation type: Horizontal congestion\n"
        "\tcomment: capacity:20 usage:24 overflow:4\n",
        encoding="utf-8",
    )

    parsed = parse_chipbench_partial_metrics(stage_dir)

    assert parsed is not None
    assert parsed["metrics_stage"] == "partial_grt"
    assert parsed["estimated_wirelength"] == 7036253.0
    assert parsed["grt_overflow"] == 1251012.0
    assert parsed["drc_count"] == 0.0
    report = parsed["raw"]["global_route_congestion_report"]
    assert report["violation_count"] == 2
    assert report["max_overflow"] == 19.0
    log = parsed["raw"]["global_route_log"]
    assert log["final_max_horizontal_overflow"] == 100.0
    assert log["final_max_vertical_overflow"] == 87.0


def test_tier3_runner_writes_artifacts(monkeypatch, tmp_path: Path) -> None:
    panel_path = _panel(tmp_path)
    def_path = tmp_path / "placement.def"
    def_path.write_text("VERSION 5.8 ;\n", encoding="utf-8")
    placements = tmp_path / "placements.json"
    placements.write_text(
        json.dumps(
            {
                "placements": [
                    {
                        "design": "bp_fe",
                        "objective_id": "default",
                        "seed": 1000,
                        "def_path": str(def_path),
                    },
                    {
                        "design": "bp_fe",
                        "objective_id": "candidate",
                        "seed": 1000,
                        "def_path": str(def_path),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    chipbench_root = tmp_path / "chipbench"
    (chipbench_root / "benchmarking").mkdir(parents=True)

    def fake_run(command, cwd, stdout, stderr, timeout, check):
        evaluate_name = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--evaluate_name="))
        metrics_dir = Path(cwd) / "benchmarking_result" / evaluate_name
        metrics_dir.mkdir(parents=True)
        if "candidate" in evaluate_name:
            wirelength = 90.0
            overflow = 2.0
        else:
            wirelength = 100.0
            overflow = 3.0
        (metrics_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "routed_wirelength": wirelength,
                    "grt_overflow": overflow,
                    "drc_count": 0,
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tier3_openroad.subprocess, "run", fake_run)

    summary = run_tier3_openroad(
        panel_path=panel_path,
        placements_path=placements,
        chipbench_root=chipbench_root,
        run_dir=tmp_path / "tier3",
    )

    assert summary["result_count"] == 2
    assert summary["success_count"] == 2
    assert (tmp_path / "tier3" / "metrics.csv").exists()
    rankings = json.loads((tmp_path / "tier3" / "rankings.json").read_text(encoding="utf-8"))
    assert rankings["rankings"][0]["objective_id"] == "candidate"


def test_tier3_runner_disables_subprocess_timeout_when_configured_zero(
    monkeypatch,
    tmp_path: Path,
) -> None:
    panel_path = _panel(tmp_path, timeout_seconds=0)
    def_path = tmp_path / "placement.def"
    def_path.write_text("VERSION 5.8 ;\n", encoding="utf-8")
    placements = tmp_path / "placements.json"
    placements.write_text(
        json.dumps(
            {
                "placements": [
                    {
                        "design": "bp_fe",
                        "objective_id": "default",
                        "seed": 1000,
                        "def_path": str(def_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    chipbench_root = tmp_path / "chipbench"
    (chipbench_root / "benchmarking").mkdir(parents=True)
    seen_timeouts: list[object] = []

    def fake_run(command, cwd, stdout, stderr, timeout, check):
        seen_timeouts.append(timeout)
        evaluate_name = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--evaluate_name="))
        metrics_dir = Path(cwd) / "benchmarking_result" / evaluate_name
        metrics_dir.mkdir(parents=True)
        (metrics_dir / "metrics.json").write_text(
            json.dumps({"routed_wirelength": 100.0, "grt_overflow": 0.0, "drc_count": 0}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tier3_openroad.subprocess, "run", fake_run)

    summary = run_tier3_openroad(
        panel_path=panel_path,
        placements_path=placements,
        chipbench_root=chipbench_root,
        run_dir=tmp_path / "tier3",
    )

    assert summary["success_count"] == 1
    assert seen_timeouts == [None]


def test_tier3_runner_uses_configured_baseline(monkeypatch, tmp_path: Path) -> None:
    panel_path = _panel(tmp_path)
    def_path = tmp_path / "placement.def"
    def_path.write_text("VERSION 5.8 ;\n", encoding="utf-8")
    placements = tmp_path / "placements.json"
    placements.write_text(
        json.dumps(
            {
                "placements": [
                    {
                        "design": "bp_fe",
                        "objective_id": "density_135",
                        "seed": 1000,
                        "def_path": str(def_path),
                    },
                    {
                        "design": "bp_fe",
                        "objective_id": "candidate",
                        "seed": 1000,
                        "def_path": str(def_path),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    chipbench_root = tmp_path / "chipbench"
    (chipbench_root / "benchmarking").mkdir(parents=True)

    def fake_run(command, cwd, stdout, stderr, timeout, check):
        evaluate_name = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--evaluate_name="))
        metrics_dir = Path(cwd) / "benchmarking_result" / evaluate_name
        metrics_dir.mkdir(parents=True)
        wirelength = 90.0 if "candidate" in evaluate_name else 100.0
        overflow = 2.0 if "candidate" in evaluate_name else 3.0
        (metrics_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "routed_wirelength": wirelength,
                    "grt_overflow": overflow,
                    "drc_count": 0,
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tier3_openroad.subprocess, "run", fake_run)

    summary = run_tier3_openroad(
        panel_path=panel_path,
        placements_path=placements,
        chipbench_root=chipbench_root,
        run_dir=tmp_path / "tier3",
        baseline_objective_id="density_135",
    )

    rows = list(
        csv.DictReader((tmp_path / "tier3" / "comparison_table.csv").open(encoding="utf-8"))
    )
    candidate = next(row for row in rows if row["objective_id"] == "candidate")

    assert summary["baseline_objective_id"] == "density_135"
    assert candidate["baseline_objective_id"] == "density_135"
    assert float(candidate["routed_wirelength_delta_pct"]) == -10.0


def test_tier3_runner_recovers_partial_metrics_on_timeout(monkeypatch, tmp_path: Path) -> None:
    panel_path = _panel(tmp_path)
    def_path = tmp_path / "placement.def"
    def_path.write_text("VERSION 5.8 ;\n", encoding="utf-8")
    placements = tmp_path / "placements.json"
    placements.write_text(
        json.dumps(
            {
                "placements": [
                    {
                        "design": "bp_fe",
                        "objective_id": "default",
                        "seed": 1000,
                        "def_path": str(def_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    chipbench_root = tmp_path / "chipbench"
    (chipbench_root / "benchmarking").mkdir(parents=True)

    def fake_run(command, cwd, stdout, stderr, timeout, check):
        evaluate_name = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--evaluate_name="))
        stage_dir = Path(cwd) / "flow" / "logs" / "nangate45" / "bp_fe" / evaluate_name
        stage_dir.mkdir(parents=True)
        (stage_dir / "4_1_cts.json").write_text(
            json.dumps(
                {
                    "cts__route__wirelength__estimated": 123.0,
                    "cts__timing__setup__ws": -0.2,
                    "cts__timing__setup__tns": -4.0,
                    "cts__power__total": 0.3,
                    "cts__design__instance__area__stdcell": 33.0,
                    "cts__design__violations": 0,
                }
            ),
            encoding="utf-8",
        )
        (stage_dir / "5_1_grt.log").write_text(
            "[INFO GRT-0002] Maximum degree: 7\n",
            encoding="utf-8",
        )
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(tier3_openroad.subprocess, "run", fake_run)

    summary = run_tier3_openroad(
        panel_path=panel_path,
        placements_path=placements,
        chipbench_root=chipbench_root,
        run_dir=tmp_path / "tier3",
    )

    assert summary["success_count"] == 0
    assert summary["partial_count"] == 1
    rows = (tmp_path / "tier3" / "metrics.csv").read_text(encoding="utf-8").splitlines()
    assert "partial_cts" in rows[1]
    assert "123.0" in rows[1]
    result = json.loads(
        (
            tmp_path
            / "tier3"
            / "bp_fe"
            / "default"
            / "seed_1000"
            / "run_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert result["status"] == "partial"
    assert result["metrics_path"].endswith("partial_metrics.json")


def test_tier3_recover_partial_run_writes_standard_artifacts(tmp_path: Path) -> None:
    panel_path = _panel(tmp_path)
    def_path = tmp_path / "placement.def"
    def_path.write_text("VERSION 5.8 ;\n", encoding="utf-8")
    placements = tmp_path / "placements.json"
    placements.write_text(
        json.dumps(
            {
                "placements": [
                    {
                        "design": "bp_fe",
                        "objective_id": "default",
                        "seed": 1000,
                        "def_path": str(def_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    chipbench_root = tmp_path / "chipbench"
    stage_dir = chipbench_root / "flow" / "logs" / "nangate45" / "bp_fe" / "coevop_bp_fe_default_s1000"
    stage_dir.mkdir(parents=True)
    (stage_dir / "4_1_cts.json").write_text(
        json.dumps(
            {
                "cts__route__wirelength__estimated": 321.0,
                "cts__timing__setup__ws": -0.1,
                "cts__timing__setup__tns": -2.0,
                "cts__power__total": 0.2,
                "cts__design__instance__area__stdcell": 44.0,
            }
        ),
        encoding="utf-8",
    )

    summary = recover_tier3_partial_run(
        panel_path=panel_path,
        placements_path=placements,
        chipbench_root=chipbench_root,
        run_dir=tmp_path / "tier3",
        baseline_objective_id="default",
        failure_stage="interrupted",
    )

    assert summary["recovered"] is True
    assert summary["partial_count"] == 1
    assert (tmp_path / "tier3" / "metrics.csv").exists()
    assert (tmp_path / "tier3" / "comparison_table.csv").exists()
    assert (tmp_path / "tier3" / "tier3_report.md").exists()
    result = json.loads(
        (
            tmp_path
            / "tier3"
            / "bp_fe"
            / "default"
            / "seed_1000"
            / "run_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert result["status"] == "partial"
    assert result["failure_stage"] == "interrupted"


def test_tier3_grt_only_mode_runs_global_route_target(monkeypatch, tmp_path: Path) -> None:
    panel_path = _panel(tmp_path, mode="grt_only")
    def_path = tmp_path / "placement.def"
    def_path.write_text("VERSION 5.8 ;\n", encoding="utf-8")
    placements = tmp_path / "placements.json"
    placements.write_text(
        json.dumps(
            {
                "placements": [
                    {
                        "design": "bp_fe",
                        "objective_id": "candidate",
                        "seed": 1000,
                        "def_path": str(def_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    chipbench_root = tmp_path / "chipbench"
    (chipbench_root / "benchmarking").mkdir(parents=True)
    seen_commands: list[list[str]] = []

    def fake_run(command, cwd, stdout, stderr, timeout, check):
        seen_commands.append(command)
        assert command[:2] == ["bash", "-lc"]
        assert "do-grt" in command[2]
        assert "do-5_2_route" not in command[2]
        assert "GLOBAL_ROUTE_ARGS=" in command[2]
        assert "-congestion_iterations 5" in command[2]
        evaluate_name = "coevop_bp_fe_candidate_s1000"
        stage_dir = Path(cwd) / "flow" / "logs" / "nangate45" / "bp_fe" / evaluate_name
        stage_dir.mkdir(parents=True)
        (stage_dir / "5_1_grt.json").write_text(
            json.dumps(
                {
                    "globalroute__route__wirelength__estimated": 95.0,
                    "globalroute__overflow__total": 1.5,
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tier3_openroad.subprocess, "run", fake_run)

    summary = run_tier3_openroad(
        panel_path=panel_path,
        placements_path=placements,
        chipbench_root=chipbench_root,
        run_dir=tmp_path / "tier3",
    )

    assert seen_commands
    assert summary["success_count"] == 1
    rows = list(csv.DictReader((tmp_path / "tier3" / "metrics.csv").open(encoding="utf-8")))
    assert rows[0]["metrics_stage"] == "partial_grt"
    assert float(rows[0]["estimated_wirelength"]) == 95.0
    assert float(rows[0]["grt_overflow"]) == 1.5
