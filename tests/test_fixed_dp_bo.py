"""Fixed-DP schedule BO and the evolution loop's evidence scheduling."""

import csv
import json
from pathlib import Path

import pytest

from coevop.eval import fixed_dp_bo, openevolve_tier2, tier2_dreamplace
from coevop.eval.fixed_dp_bo import (
    SCHEDULE_SPACE,
    fixed_dp_schedule_spec,
    run_fixed_dp_schedule_bo,
)
from coevop.eval.openevolve_tier2 import run_openevolve_tier2
from coevop.eval.timing_proxy import (
    TimingProxyDesign,
    TimingProxyPanel,
    restore_def_identifiers,
    timing_proxy_metric_admission,
    write_timing_driven_config,
)
from coevop.backends.dreamplace import DreamPlaceRun
from coevop.eval.tier2_dreamplace import run_tier2_dreamplace
from coevop.objectives.presets import objective_preset
from coevop.objectives.program import parse_objective_program
from coevop.objectives.spec import load_objective_spec, write_objective_spec

NATIVE_SCHEDULE = {
    "density_init": 8e-5,
    "density_ramp_floor": 0.95,
    "density_ramp_span": 0.10,
    "density_ramp_gain": 2000.0,
    "gamma_floor": 0.06,
    "gamma_ceiling": 10.0,
    "gamma_slope": 8.0,
    "gamma_center": 0.5,
}


def _midpoint(index: int) -> dict[str, float]:
    return {
        name: low + (high - low) * (0.25 + 0.2 * index)
        for name, (low, high, _log) in SCHEDULE_SPACE.items()
    }


def _panel_and_config(tmp_path: Path, *, extra: list[str]) -> Path:
    base = tmp_path / "bp_fe.json"
    base.write_text("{}", encoding="utf-8")
    panel = tmp_path / "panel.toml"
    panel.write_text(
        "\n".join(
            [
                "seeds = [1000]",
                "iterations = 1000",
                "timeout_seconds = 9",
                "",
                "[[designs]]",
                'name = "bp_fe"',
                f'dreamplace_config = "{base.as_posix()}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "evolution.toml"
    config.write_text(
        "\n".join(
            [
                'provider = "mock"',
                'term_scope = "dreamplace_controller"',
                'objective_mode = "controller"',
                "max_iterations = 2",
                "samples_per_iteration = 2",
                "num_islands = 2",
                "migration_interval = 1",
                "checkpoint_interval = 0",
                'primary_baseline = "default"',
                'parent_policy = "pareto_multiobjective"',
                'selection_policy = "pareto_multiobjective"',
                "min_state_registers = 1",
                "max_calibrated_siblings = 0",
                'initial_presets = ["dreamplace_controller_native_identity", '
                '"dreamplace_controller_eplace_smooth"]',
                "",
                *extra,
                "[search]",
                f'panel = "{panel.as_posix()}"',
                "disable_legalization = true",
                "",
                "[final]",
                "enabled = false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config


def _fake_tier2(tmp_path: Path, hpwl_by_objective=None):
    """Tier A stand-in: every objective completes with a routable DEF."""

    def run(**kwargs):
        run_dir = Path(kwargs["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        ids = ["default", "custom_default"] + [
            load_objective_spec(path).id for path in kwargs["objective_paths"]
        ]
        rows = [
            {
                "design": "bp_fe",
                "objective_id": objective_id,
                "seed": 1000,
                "status": "success",
                "failure_stage": "",
                "hpwl_delta_pct": 0.0
                if objective_id in {"default", "custom_default"}
                else (hpwl_by_objective or {}).get(objective_id, -1.0 - 0.1 * index),
                "overflow_delta_pct": 0.0 if index < 2 else -2.0,
                "runtime_seconds": 10 + index,
                "custom_grad_norm": 1.0,
                "output_artifact": str(tmp_path / f"{objective_id}.def"),
                "requested_iterations": 1000,
                "completed_iterations": 1000,
                "iteration_budget_satisfied": True,
            }
            for index, objective_id in enumerate(ids)
        ]
        comparison = run_dir / "comparison_table.csv"
        with comparison.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return {"comparison_csv": str(comparison), "success_count": len(rows)}

    return run


# --- Fixed-DP schedule BO ----------------------------------------------------


def test_fixed_dp_spec_keeps_the_native_objective_form() -> None:
    spec = fixed_dp_schedule_spec(NATIVE_SCHEDULE)
    reference = objective_preset("dreamplace_controller_eplace_smooth")

    assert spec.is_typed_policy
    # Same loss and components as the native-style control: only schedule
    # coefficients are free.
    assert spec.ast == reference.ast
    assert spec.components == reference.components
    assert set(spec.state["registers"]) == {"density_weight", "gamma_scale"}
    assert spec.term_set == ["density_electric", "wirelength_wawl"]

    other = fixed_dp_schedule_spec({**NATIVE_SCHEDULE, "gamma_slope": 12.0})
    assert other.id != spec.id
    assert other.ast == spec.ast


def test_fixed_dp_bo_matches_the_evolution_budget(monkeypatch, tmp_path: Path) -> None:
    config = _panel_and_config(tmp_path, extra=[])
    monkeypatch.setattr(openevolve_tier2, "run_tier2_dreamplace", _fake_tier2(tmp_path))
    asked: list[tuple[int, int, int]] = []

    def proposer(history, count, seed):
        asked.append((len(history), count, seed))
        return [_midpoint(len(history) + index) for index in range(count)]

    summary = run_fixed_dp_schedule_bo(
        config_path=config,
        run_dir=tmp_path / "run",
        dreamplace_root=tmp_path,
        proposer=proposer,
    )

    # Two generations of two proposals, as the evolution configuration asks.
    assert [(size, count) for size, count, _seed in asked] == [(0, 2), (2, 2)]
    assert asked[0][2] != asked[1][2]
    assert summary["generations"] == 2
    assert summary["proposals_per_generation"] == 2
    assert summary["proposal_count"] == 4
    assert summary["valid_count"] == 4
    assert summary["best_schedule_parameters"] in [_midpoint(index) for index in range(4)]
    best = json.loads((tmp_path / "run" / "best_objective.json").read_text(encoding="utf-8"))
    assert best["ast"] == fixed_dp_schedule_spec(NATIVE_SCHEDULE).ast

    # Resuming a finished run proposes nothing more.
    asked.clear()
    run_fixed_dp_schedule_bo(
        config_path=config,
        run_dir=tmp_path / "run",
        dreamplace_root=tmp_path,
        resume=True,
        proposer=proposer,
    )
    assert asked == []


def test_tpe_proposer_requires_the_baselines_extra(monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "optuna", None)
    with pytest.raises(RuntimeError, match="baselines"):
        fixed_dp_bo.propose_with_tpe([], 1, seed=0)


# --- evolution loop: Tier B, periodic audit, migration -----------------------


def test_evolution_loop_audits_the_proxy_and_migrates_each_interval(
    monkeypatch, tmp_path: Path
) -> None:
    timing_panel = tmp_path / "timing.toml"
    timing_panel.write_text(
        "\n".join(
            [
                f'ot_shell = "{(tmp_path / "ot-shell").as_posix()}"',
                "[[designs]]",
                'name = "bp_fe"',
                f'base_config = "{(tmp_path / "bp_fe.json").as_posix()}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = _panel_and_config(
        tmp_path,
        extra=[
            "allow_short_search_for_tests = true",
            "[timing_proxy]",
            "enabled = true",
            'mode = "gate"',
            f'panel = "{timing_panel.as_posix()}"',
            "audit_required = true",
            "audit_interval = 1",
            "",
        ],
    )
    monkeypatch.setattr(openevolve_tier2, "run_tier2_dreamplace", _fake_tier2(tmp_path))
    timing_calls: list[set[str]] = []

    def fake_timing_eval(**kwargs):
        placements = json.loads(Path(kwargs["placements_path"]).read_text(encoding="utf-8"))
        ids = [item["objective_id"] for item in placements["placements"]]
        timing_calls.append(set(ids))
        run_dir = Path(kwargs["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        comparison = run_dir / "comparison_table.csv"
        with comparison.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "design",
                    "objective_id",
                    "seed",
                    "status",
                    "timing_proxy_hpwl_delta_pct",
                    "placement_overflow_delta_pct",
                    "timing_proxy_wns_delta",
                    "timing_proxy_tns_delta",
                    "source_placement_budget_satisfied",
                ],
            )
            writer.writeheader()
            for index, objective_id in enumerate(ids):
                writer.writerow(
                    {
                        "design": "bp_fe",
                        "objective_id": objective_id,
                        "seed": 1000,
                        "status": "success",
                        "timing_proxy_hpwl_delta_pct": -1.0 - 0.1 * index,
                        "placement_overflow_delta_pct": -2.0,
                        "timing_proxy_wns_delta": 0.1 * index,
                        "timing_proxy_tns_delta": 1.0 * index,
                        "source_placement_budget_satisfied": True,
                    }
                )
        return {"comparison_csv": str(comparison), "success_count": len(ids)}

    audits: list[str] = []

    def fake_audit(**kwargs):
        audits.append(Path(kwargs["run_dir"]).parent.name)
        correlations = {
            "hpwl_wns_pearson": 0.5,
            "hpwl_wns_spearman": 0.5,
            "hpwl_tns_pearson": 0.99,
            "hpwl_tns_spearman": 0.99,
            "pair_count_wns": 10,
            "pair_count_tns": 10,
        }
        return {
            "correlations": correlations,
            "metric_admission": timing_proxy_metric_admission(
                correlations,
                gate_threshold=kwargs["gate_threshold"],
                tiebreaker_threshold=kwargs["tiebreaker_threshold"],
            ),
        }

    monkeypatch.setattr(openevolve_tier2, "run_timing_proxy_eval", fake_timing_eval)
    monkeypatch.setattr(openevolve_tier2, "run_timing_proxy_audit", fake_audit)

    summary = run_openevolve_tier2(
        config_path=config,
        run_dir=tmp_path / "run",
        dreamplace_root=tmp_path,
    )

    # The proxy is audited on the seed placements, then after every generation.
    assert audits == ["iteration_0000", "iteration_0001", "iteration_0002"]
    admission = json.loads(
        (tmp_path / "run" / "timing_proxy_audit" / "admission.json").read_text(encoding="utf-8")
    )
    assert admission["iteration"] == 2
    assert admission["authority"] == {"wns": "admit", "tns": "reject"}
    assert all(item["scheduled_timing_audit"] is not None for item in summary["iterations"])

    database = json.loads(
        (tmp_path / "run" / "program_db" / "metadata.json").read_text(encoding="utf-8")
    )
    assert database["last_migration_iteration"] == 2
    programs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "run" / "program_db" / "programs").glob("*.json")
    ]
    evaluated = [program for program in programs if program["metrics"].get("tier_b_evaluated")]
    assert evaluated
    for program in evaluated:
        # Admitted WNS becomes selection evidence; rejected TNS does not.
        assert program["metrics"]["timing_proxy_admission"] == {"wns": "admit", "tns": "reject"}
        assert program["metrics"]["timing_evidence_tns_delta"] is None
    # Tier B never runs without a candidate that passed Tier A.
    assert all(ids - {"default", "custom_default"} for ids in timing_calls)


# --- per-net timing policy: in-loop collateral -------------------------------

NET_POLICY_PROGRAM = """
def init_policy(obs):
    return {"density_weight": 0.00008 * obs("grad_ratio_density_electric"), "gamma_scale": 1.0}

def update_policy(policy, obs):
    return {
        "density_weight": policy("density_weight") * (1.0 + 0.05 * sigmoid(obs("overflow"))),
        "gamma_scale": 0.1 + 0.9 * sigmoid(8.0 * (obs("overflow") - 0.5)),
    }

def update_net_weights(net, policy, obs):
    return 1.0 + sigmoid(4.0 * net("criticality"))

def objective(features, policy):
    wl = term("wirelength_wawl")
    den = term("density_electric")
    score = wl + policy("density_weight") * den
    return score, {"wirelength": wl, "density": den}
"""

TOY_DEF = "\n".join(
    [
        "VERSION 5.8 ;",
        "DESIGN toy_top ;",
        "COMPONENTS 2 ;",
        "  - u/path INV_X1 + UNPLACED ;",
        "  - _1_ NAND2_X1 + UNPLACED ;",
        "END COMPONENTS",
        "PINS 2 ;",
        "  - clk + NET clk + DIRECTION INPUT + USE SIGNAL ;",
        "  - out + NET result + DIRECTION OUTPUT + USE SIGNAL ;",
        "END PINS",
        "NETS 3 ;",
        "  - clk ( PIN clk ) ( u/path A ) + USE CLOCK ;",
        "  - mid ( u/path ZN ) ( _1_ A2 ) + USE SIGNAL ;",
        "  - result ( PIN out ) ( _1_ ZN ) + USE SIGNAL ;",
        "END NETS",
        "END DESIGN",
    ]
) + "\n"


def _timing_collateral(tmp_path: Path) -> tuple[Path, TimingProxyPanel]:
    (tmp_path / "toy.def").write_text(TOY_DEF, encoding="utf-8")
    base = tmp_path / "base.json"
    base.write_text(
        json.dumps(
            {
                "def_input": (tmp_path / "toy.def").as_posix(),
                "timing_opt_flag": 0,
                "global_place_stages": [{"iteration": 1000}],
            }
        ),
        encoding="utf-8",
    )
    sdc = tmp_path / "floorplan.sdc"
    sdc.write_text(
        "current_design toy_top\ncreate_clock -name CLK -period 4 [get_ports {clk}]\n",
        encoding="utf-8",
    )
    (tmp_path / "ot-shell").write_text("", encoding="utf-8")
    panel = TimingProxyPanel(
        ot_shell=str(tmp_path / "ot-shell"),
        timeout_seconds=60,
        iterations=1,
        gpu=None,
        threads=2,
        designs=[
            TimingProxyDesign(
                name="toy", base_config=str(base), sdc=str(sdc), top_module="toy_top"
            )
        ],
        scalarize_vector_nets=True,
        derive_verilog_from_def=True,
    )
    return base, panel


def test_timing_driven_config_enables_in_loop_timing_and_restores_names(tmp_path: Path) -> None:
    base, panel = _timing_collateral(tmp_path)

    config_path, restore_map = write_timing_driven_config(
        design=panel.designs[0],
        panel=panel,
        base_config=base,
        dreamplace_root=tmp_path,
        run_dir=tmp_path / "run",
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["timing_opt_flag"] == 1
    assert config["enable_net_weighting"] == 1
    assert config["timer_engine"] == "opentimer"
    assert "module toy_top" in Path(config["verilog_input"]).read_text(encoding="utf-8")
    assert Path(config["sdc_input"]).is_file()
    prepared_def = Path(config["def_input"])
    assert "u_path" in prepared_def.read_text(encoding="utf-8")
    assert restore_map == {"u_path": "u/path"}

    # The placed DEF leaves with the identifiers the routing flow knows.
    assert restore_def_identifiers(prepared_def, restore_map) == 3
    assert prepared_def.read_text(encoding="utf-8") == TOY_DEF


def test_tier2_runs_net_weight_candidates_timing_driven(monkeypatch, tmp_path: Path) -> None:
    base, panel = _timing_collateral(tmp_path)
    placement_panel = tmp_path / "panel.toml"
    placement_panel.write_text(
        "seeds = [1000]\niterations = 1000\n\n[[designs]]\n"
        f'name = "toy"\nbase_config = "{base.as_posix()}"\n',
        encoding="utf-8",
    )
    timing_panel = tmp_path / "timing_controller.toml"
    timing_panel.write_text(
        f'ot_shell = "{(tmp_path / "ot-shell").as_posix()}"\n'
        "scalarize_vector_nets = true\nderive_verilog_from_def = true\n\n[[designs]]\n"
        f'name = "toy"\nbase_config = "{base.as_posix()}"\n'
        f'sdc = "{(tmp_path / "floorplan.sdc").as_posix()}"\ntop_module = "toy_top"\n',
        encoding="utf-8",
    )
    objective = write_objective_spec(
        parse_objective_program(
            NET_POLICY_PROGRAM, created_by="test", term_scope="dreamplace_controller"
        ),
        tmp_path / "net_policy.json",
    )
    seen: dict[str, dict] = {}

    def fake_run_dreamplace(**kwargs):
        run_dir = Path(kwargs["run_dir"])
        config = json.loads(Path(kwargs["config_path"]).read_text(encoding="utf-8"))
        seen[run_dir.parent.name] = config
        results = run_dir / "results" / "toy"
        results.mkdir(parents=True)
        (results / "toy.gp.def").write_text(
            Path(config["def_input"]).read_text(encoding="utf-8"), encoding="utf-8"
        )
        return DreamPlaceRun(
            run_name=kwargs["run_name"],
            run_dir=str(run_dir),
            config_path=str(kwargs["config_path"]),
            log_path=str(run_dir / "dreamplace.log"),
            returncode=0,
            command=["fake"],
            metrics={
                "last": {"hpwl": 100.0, "overflow": 0.5},
                "global_place_iterations_completed": 1000,
                "custom_objective": {"calls": 3, "last_grad_norm": 1.5},
            },
        )

    monkeypatch.setattr(tier2_dreamplace, "run_dreamplace", fake_run_dreamplace)

    def run(run_dir: str, **kwargs):
        return run_tier2_dreamplace(
            panel_path=placement_panel,
            objective_paths=[objective],
            dreamplace_root=tmp_path,
            run_dir=tmp_path / run_dir,
            include_custom_default=False,
            **kwargs,
        )

    summary = run("with_collateral", timing_controller_panel=timing_panel)

    assert summary["failure_count"] == 0
    candidate = next(config for name, config in seen.items() if name != "default")
    assert candidate["timing_opt_flag"] == 1
    assert candidate["timing_controller_enabled"] == 1
    # The native baseline stays on the untouched, non-timing-driven config.
    assert seen["default"].get("timing_opt_flag") == 0
    assert "timing_controller_enabled" not in seen["default"]
    placed = next((tmp_path / "with_collateral").rglob("obj_*/seed_1000/results/toy/toy.gp.def"))
    assert "u/path" in placed.read_text(encoding="utf-8")
    assert "u_path" not in placed.read_text(encoding="utf-8")

    # Without collateral the candidate fails closed before placement starts.
    seen.clear()
    summary = run("without_collateral")
    assert summary["failure_count"] == 1
    assert list(seen) == ["default"]
