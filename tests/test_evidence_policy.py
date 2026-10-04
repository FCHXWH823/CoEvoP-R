"""Cost-scaled evidence policy: Tier B selection, timing audit, Pareto order."""

import csv
import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from coevop.eval import openevolve_tier2
from coevop.eval.openevolve_tier2 import (
    _dominates,
    _load_iteration_feedback,
    _load_timing_admission,
    _pareto_evidence,
    _prepare_pareto_candidate,
    _refresh_pareto_selection,
    _run_scheduled_tier3_feedback,
    _run_scheduled_timing_proxy_audit,
    _static_rejection_reason,
    _tier_b_selected,
    _timing_proxy_feedback_for_tier2_summary,
    load_openevolve_tier2_config,
)
from coevop.eval.timing_proxy import (
    apply_timing_proxy_admission,
    cap_unstable_timing_proxy_admission,
    default_timing_proxy_admission,
    summarize_timing_proxy_admission,
    timing_proxy_downstream_agreement,
    timing_proxy_metric_admission,
)
from coevop.evolution.openevolve_core import (
    ObjectiveDatabaseConfig,
    ObjectiveProgram,
    ObjectiveProgramDatabase,
    _compact_metrics,
    _feedback_row,
)
from coevop.objectives.presets import objective_preset
from coevop.objectives.program import parse_objective_program

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_CONFIG = REPO_ROOT / "configs" / "openevolve_tier2" / "chipbench_controller_tier2.toml"


def _correlations(wns: tuple[float, float], tns: tuple[float, float], pairs: int = 10) -> dict:
    return {
        "hpwl_wns_pearson": wns[0],
        "hpwl_wns_spearman": wns[1],
        "hpwl_tns_pearson": tns[0],
        "hpwl_tns_spearman": tns[1],
        "pair_count_wns": pairs,
        "pair_count_tns": pairs,
    }


def _admit(metric_outcomes: dict[str, str]) -> dict:
    return {metric: {"outcome": outcome} for metric, outcome in metric_outcomes.items()}


# --- timing-proxy audit ------------------------------------------------------


def test_audit_admits_ties_and_rejects_each_metric_separately() -> None:
    admission = timing_proxy_metric_admission(
        _correlations(wns=(0.58, -0.54), tns=(0.82, 0.78)),
        gate_threshold=0.70,
        tiebreaker_threshold=0.95,
    )
    assert admission["wns"]["outcome"] == "admit"
    assert admission["wns"]["hpwl_redundancy"] == pytest.approx(0.58)
    assert admission["tns"]["outcome"] == "tie"

    redundant = timing_proxy_metric_admission(
        _correlations(wns=(0.97, 0.90), tns=(0.10, 0.20), pairs=2),
        gate_threshold=0.70,
        tiebreaker_threshold=0.95,
    )
    assert redundant["wns"]["outcome"] == "reject"
    # Two finite pairs are not enough to trust a low correlation.
    assert redundant["tns"]["outcome"] == "reject"


def test_audit_summary_counts_outcomes_and_takes_the_majority() -> None:
    summary = summarize_timing_proxy_admission(
        {
            "bp_fe": {
                "wns": {"outcome": "admit", "hpwl_pearson": 0.5, "hpwl_spearman": 0.4},
                "tns": {"outcome": "admit", "hpwl_pearson": 0.6, "hpwl_spearman": 0.6},
            },
            "swerv_wrapper": {
                "wns": {"outcome": "admit", "hpwl_pearson": 0.6, "hpwl_spearman": 0.5},
                "tns": {"outcome": "tie", "hpwl_pearson": 0.8, "hpwl_spearman": 0.8},
            },
            "ethernet": {
                "wns": {"outcome": "admit", "hpwl_pearson": 0.6, "hpwl_spearman": 0.6},
                "tns": {"outcome": "tie", "hpwl_pearson": 0.9, "hpwl_spearman": 0.8},
            },
            "or1200": {
                "wns": {"outcome": "tie", "hpwl_pearson": 0.8, "hpwl_spearman": 0.7},
                "tns": {"outcome": "reject", "hpwl_pearson": 0.99, "hpwl_spearman": 0.9},
            },
        }
    )
    assert summary["counts"]["wns"] == {"reject": 0, "tie": 1, "admit": 3}
    assert summary["counts"]["tns"] == {"reject": 1, "tie": 2, "admit": 1}
    assert summary["authority"] == {"wns": "admit", "tns": "tie"}
    assert summary["median_hpwl_correlation"]["wns"]["hpwl_pearson"] == pytest.approx(0.6)


def test_admission_splits_selection_evidence_from_tie_break_evidence() -> None:
    admission = summarize_timing_proxy_admission(
        {
            "bp_fe": _admit({"wns": "admit", "tns": "tie"}),
            "or1200": _admit({"wns": "reject", "tns": "tie"}),
        }
    )
    metrics = {
        "timing_proxy_wns_delta": 0.3,
        "timing_proxy_per_design_deltas": [
            {"design": "bp_fe", "wns_delta": 0.2, "tns_delta": 4.0},
            {"design": "or1200", "wns_delta": 0.4, "tns_delta": 6.0},
        ],
    }
    apply_timing_proxy_admission(metrics, admission)

    # Only the admitted design supplies the WNS selection coordinate.
    assert metrics["timing_evidence_wns_delta"] == pytest.approx(0.2)
    assert metrics["timing_tiebreak_wns_delta"] is None
    assert metrics["timing_evidence_tns_delta"] is None
    assert metrics["timing_tiebreak_tns_delta"] == pytest.approx(5.0)
    # The unsplit proxy delta stays available as an archive feature.
    assert metrics["timing_proxy_wns_delta"] == 0.3


def test_configured_mode_is_the_admission_until_an_audit_runs(tmp_path: Path) -> None:
    assert default_timing_proxy_admission("gate")["authority"] == {"wns": "admit", "tns": "admit"}
    assert default_timing_proxy_admission("tie_breaker")["authority"]["wns"] == "tie"
    assert default_timing_proxy_admission("diagnostic")["authority"]["tns"] == "reject"

    config = load_openevolve_tier2_config(PRIMARY_CONFIG)
    assert _load_timing_admission(config, tmp_path)["source"] == "configured_mode"


def test_downstream_disagreement_caps_a_metric_at_tie_breaking() -> None:
    agreement = timing_proxy_downstream_agreement(
        {
            "wns": [(0.1, -0.2), (0.2, -0.4), (0.3, -0.5), (0.4, -0.9)],
            "tns": [(1.0, 2.0), (2.0, 5.0), (3.0, 7.0)],
        }
    )
    assert agreement["wns"]["status"] == "unstable"
    assert agreement["tns"]["status"] == "stable"
    assert timing_proxy_downstream_agreement({"wns": [(0.1, 0.2)]})["wns"]["status"] == "unknown"

    capped = cap_unstable_timing_proxy_admission(
        summarize_timing_proxy_admission({"bp_fe": _admit({"wns": "admit", "tns": "admit"})}),
        agreement,
    )
    assert capped["by_design"]["bp_fe"]["wns"]["outcome"] == "tie"
    assert capped["by_design"]["bp_fe"]["tns"]["outcome"] == "admit"
    assert capped["authority"] == {"wns": "tie", "tns": "admit"}


def test_scheduled_audit_runs_per_design_and_records_admission(monkeypatch, tmp_path) -> None:
    config = replace(
        load_openevolve_tier2_config(PRIMARY_CONFIG),
        timing_proxy_audit_interval=20,
    )
    comparison = tmp_path / "tier2.csv"
    with comparison.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "design",
                "objective_id",
                "seed",
                "status",
                "output_artifact",
                "requested_iterations",
                "completed_iterations",
                "iteration_budget_satisfied",
            ],
        )
        writer.writeheader()
        for design in ("bp_fe", "or1200", "dft68"):
            for objective_id in ("default", "obj_best"):
                writer.writerow(
                    {
                        "design": design,
                        "objective_id": objective_id,
                        "seed": 1000,
                        "status": "success",
                        "output_artifact": f"{design}_{objective_id}.def",
                        "requested_iterations": 1000,
                        "completed_iterations": 1000,
                        "iteration_budget_satisfied": True,
                    }
                )
    database = ObjectiveProgramDatabase(tmp_path / "db")
    database.add(
        ObjectiveProgram(
            id="best",
            code="best",
            objective_spec={"id": "obj_best"},
            status="accepted",
            metrics={"combined_score": 2.0, "parent_eligible": True},
            artifacts={"tier2_summary": {"comparison_csv": str(comparison)}},
        )
    )
    audited = []

    def fake_audit(**kwargs):
        audited.append((kwargs["design"], kwargs["gate_threshold"], kwargs["perturbations"]))
        correlations = (
            _correlations(wns=(0.5, 0.5), tns=(0.8, 0.8))
            if kwargs["design"] == "bp_fe"
            else _correlations(wns=(0.98, 0.98), tns=(0.8, 0.8))
        )
        return {
            "correlations": correlations,
            "metric_admission": timing_proxy_metric_admission(
                correlations,
                gate_threshold=kwargs["gate_threshold"],
                tiebreaker_threshold=kwargs["tiebreaker_threshold"],
            ),
        }

    monkeypatch.setattr(openevolve_tier2, "run_timing_proxy_audit", fake_audit)
    kwargs = dict(
        config=config, database=database, run_root=tmp_path / "run", dreamplace_root=tmp_path
    )

    assert _run_scheduled_timing_proxy_audit(iteration=7, **kwargs) is None
    admission = _run_scheduled_timing_proxy_audit(iteration=20, **kwargs)

    # dft68 is outside the four-design timing panel and is never audited.
    assert [design for design, _gate, _count in audited] == ["bp_fe", "or1200"]
    assert audited[0][1:] == (0.70, 5)
    assert admission["by_design"]["bp_fe"]["wns"]["outcome"] == "admit"
    assert admission["by_design"]["or1200"]["wns"]["outcome"] == "reject"
    assert admission["source_program_id"] == "best"
    assert _load_timing_admission(config, tmp_path / "run")["iteration"] == 20


# --- Tier B selection --------------------------------------------------------


def test_tier_b_is_reserved_for_usable_tier_a_evidence() -> None:
    config = SimpleNamespace(timing_proxy_enabled=True)
    usable = {
        "structural_failure_count": 0,
        "iteration_budget_satisfied": True,
        "hpwl_delta_pct": -1.0,
        "overflow_delta_pct": 2.0,
    }
    assert _tier_b_selected(usable, config) is True
    assert _tier_b_selected({**usable, "structural_failure_count": 1}, config) is False
    assert _tier_b_selected({**usable, "iteration_budget_satisfied": False}, config) is False
    assert _tier_b_selected({**usable, "hpwl_delta_pct": None}, config) is False
    assert _tier_b_selected(usable, SimpleNamespace(timing_proxy_enabled=False)) is False


def test_unselected_candidate_does_not_launch_the_timing_proxy(monkeypatch, tmp_path) -> None:
    config = load_openevolve_tier2_config(PRIMARY_CONFIG)

    def fail(**_kwargs):
        raise AssertionError("timing proxy must not run")

    monkeypatch.setattr(openevolve_tier2, "run_timing_proxy_eval", fail)
    metrics, artifacts = _timing_proxy_feedback_for_tier2_summary(
        tier2_summary={},
        config=config,
        iteration_dir=tmp_path,
        dreamplace_root=tmp_path,
        objective_ids=set(),
        resume=False,
        retry_failed=False,
    )
    assert metrics == {}
    assert artifacts["timing_proxy"]["status"] == "not_run"


# --- Pareto evidence order ---------------------------------------------------


def test_pareto_order_has_four_coordinates_with_tiered_evidence() -> None:
    evidence = _pareto_evidence(
        {
            "hpwl_delta_pct": -1.0,
            "overflow_delta_pct": -2.0,
            "timing_proxy_wns_delta": 0.1,
            "routed_wirelength_delta_pct": -3.0,
            "post_route_tns_gain_ns": 5.0,
        }
    )
    assert list(evidence) == ["wirelength", "overflow", "wns", "tns"]
    assert evidence["wirelength"] == {"A": -1.0, "C": -3.0}
    assert evidence["overflow"] == {"A": -2.0}
    assert evidence["wns"] == {"B": -0.1}
    assert evidence["tns"] == {"C": -5.0}


def test_wns_only_and_tns_only_candidates_do_not_dominate_each_other() -> None:
    wns_only = _pareto_evidence(
        {"hpwl_delta_pct": 0.0, "overflow_delta_pct": 0.0, "timing_proxy_wns_delta": 0.1}
    )
    tns_only = _pareto_evidence(
        {"hpwl_delta_pct": 0.0, "overflow_delta_pct": 0.0, "timing_proxy_tns_delta": 1.0}
    )
    assert _dominates(wns_only, tns_only) is False
    assert _dominates(tns_only, wns_only) is False


def test_routed_evidence_is_only_compared_with_routed_evidence() -> None:
    placement = {"hpwl_delta_pct": -5.0, "overflow_delta_pct": -5.0}
    routed_good = _pareto_evidence(
        {**placement, "routed_wirelength_delta_pct": -8.0, "routed_overflow_delta_pct": -20.0}
    )
    routed_bad = _pareto_evidence(
        {
            "hpwl_delta_pct": -6.0,
            "overflow_delta_pct": -6.0,
            "routed_wirelength_delta_pct": 4.0,
            "routed_overflow_delta_pct": 9.0,
        }
    )
    unrouted = _pareto_evidence({"hpwl_delta_pct": -5.5, "overflow_delta_pct": -5.5})

    # Both were routed: the routed measurements decide, not the placement ones.
    assert _dominates(routed_good, routed_bad) is True
    assert _dominates(routed_bad, routed_good) is False
    # Against an unrouted candidate only the shared placement tier is compared.
    assert _dominates(unrouted, routed_good) is True
    assert _dominates(routed_bad, unrouted) is True


def test_tie_level_timing_orders_a_front_but_cannot_dominate(tmp_path: Path) -> None:
    config = SimpleNamespace(selection_policy="pareto_multiobjective", timing_proxy_mode="gate")
    spec = objective_preset("dreamplace_controller_eplace_smooth")

    def database_with(admission: dict) -> ObjectiveProgramDatabase:
        database = ObjectiveProgramDatabase(
            tmp_path / admission["authority"]["wns"],
            ObjectiveDatabaseConfig(num_islands=1, population_size=20, archive_size=20),
        )
        for program_id, wns in (("better_timing", 0.5), ("worse_timing", -0.5)):
            metrics = {
                "hpwl_delta_pct": -1.0,
                "overflow_delta_pct": -1.0,
                "timing_proxy_wns_delta": wns,
                "iteration_budget_satisfied": True,
                "structural_failure_count": 0,
                "runtime_seconds": 1.0,
            }
            _prepare_pareto_candidate(metrics, config, allow_seed_baseline=False)
            database.add(ObjectiveProgram.from_spec(spec, program_id=program_id, metrics=metrics))
        _refresh_pareto_selection(database, config, admission)
        return database

    admitted = database_with(default_timing_proxy_admission("gate"))
    assert admitted.get("better_timing").metrics["pareto_front"] == 0
    assert admitted.get("worse_timing").metrics["pareto_front"] == 1

    tied = database_with(default_timing_proxy_admission("tie_breaker"))
    assert tied.get("better_timing").metrics["pareto_front"] == 0
    assert tied.get("worse_timing").metrics["pareto_front"] == 0
    assert tied.get("better_timing").combined_score > tied.get("worse_timing").combined_score

    rejected = database_with(default_timing_proxy_admission("diagnostic"))
    assert rejected.get("worse_timing").metrics["pareto_front"] == 0
    assert rejected.get("better_timing").metrics["timing_evidence_wns_delta"] is None


# --- routed evidence in the archive and the prompt ---------------------------


def test_routed_feedback_accumulates_every_round_for_the_prompt(monkeypatch, tmp_path) -> None:
    config = load_openevolve_tier2_config(PRIMARY_CONFIG)
    panel = tmp_path / "post_route.toml"
    panel.write_text("seeds = [1000]\n", encoding="utf-8")
    config = replace(config, tier3_panel=str(panel), tier3_top_k=1)
    database = ObjectiveProgramDatabase(tmp_path / "db")
    gains = {20: ("obj_first", -10.0), 40: ("obj_second", -12.5)}

    def fake_route(**kwargs):
        placements = json.loads(Path(kwargs["placements_path"]).read_text())
        objective_id = placements["selected_objective_ids"][0]
        output = Path(kwargs["run_dir"]) / "comparison_table.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "design,objective_id,seed,status,routed_wirelength_delta_pct,"
            "grt_overflow_delta_pct,wns_gain_ns,tns_gain_ns\n"
            f"bp_fe,{objective_id},1000,success,{gains[placements['iteration']][1]},-20,0.25,4.0\n",
            encoding="utf-8",
        )
        return {"comparison_csv": str(output), "success_count": 1}

    monkeypatch.setattr(openevolve_tier2, "run_tier3_openroad", fake_route)
    for iteration, (objective_id, _gain) in gains.items():
        comparison = tmp_path / f"tier2_{iteration}.csv"
        comparison.write_text(
            "design,objective_id,seed,status,output_artifact\n"
            "bp_fe,default,1000,success,default.def\n"
            f"bp_fe,{objective_id},1000,success,candidate.def\n",
            encoding="utf-8",
        )
        program = ObjectiveProgram(
            id=f"program_{objective_id}",
            code="candidate",
            objective_spec={"id": objective_id},
            status="accepted",
            metrics={"combined_score": 1.0, "pareto_admission_passed": True},
            artifacts={"tier2_summary": {"comparison_csv": str(comparison)}},
        )
        database.add(program)
        _run_scheduled_tier3_feedback(
            config=config,
            database=database,
            programs=[program],
            iteration=iteration,
            run_root=tmp_path / "run",
            chipbench_root=tmp_path / "chipbench",
            resume=False,
            retry_failed=False,
        )
        assert program.metrics["tier_c_evaluated"] is True

    stored = json.loads((tmp_path / "run" / "routed_feedback.json").read_text())
    assert [item["iteration"] for item in stored["rounds"]] == [20, 40]
    assert stored["iteration"] == 40

    prompt_view = _load_iteration_feedback(
        replace(config, prior_feedback=None), run_root=tmp_path / "run"
    )
    assert prompt_view["routed_round_count"] == 2
    assert prompt_view["latest_generation"] == 40
    assert prompt_view["routed_candidates"] == [
        "generation=40 program=program_obj_second rWL=-12.5 rOvf=-20 WNS=+0.25 TNS=+4",
        "generation=20 program=program_obj_first rWL=-10 rOvf=-20 WNS=+0.25 TNS=+4",
    ]
    assert prompt_view["latest_round_per_design"] == [
        "program=program_obj_second design=bp_fe rWL=-12.5 rOvf=-20 WNS=+0.25 TNS=+4"
    ]


def test_program_summaries_show_routed_evidence_to_the_proposer() -> None:
    program = ObjectiveProgram(
        id="routed",
        code="candidate",
        status="accepted",
        metrics={
            "hpwl_delta_pct": -1.0,
            "routed_wirelength_delta_pct": -9.0,
            "routed_overflow_delta_pct": -30.0,
            "post_route_wns_gain_ns": 0.4,
            "post_route_tns_gain_ns": 50.0,
            "tier_c_evaluated": True,
        },
    )
    compact = _compact_metrics(program.metrics)
    row = _feedback_row(program)
    for key in (
        "routed_wirelength_delta_pct",
        "routed_overflow_delta_pct",
        "post_route_wns_gain_ns",
        "post_route_tns_gain_ns",
    ):
        assert compact[key] == program.metrics[key]
        assert row[key] == program.metrics[key]
    assert compact["tier_c_evaluated"] is True


# --- island migration --------------------------------------------------------


def _migration_database(tmp_path: Path, clock: str) -> ObjectiveProgramDatabase:
    database = ObjectiveProgramDatabase(
        tmp_path / clock,
        ObjectiveDatabaseConfig(
            population_size=50,
            archive_size=20,
            num_islands=3,
            migration_interval=20,
            migration_rate=1.0,
            migration_clock=clock,
        ),
    )
    for island, preset in enumerate(
        ("dreamplace_route_pressure_long", "dreamplace_expert_rudy", "dreamplace_expert_pin")
    ):
        database.add(
            ObjectiveProgram.from_spec(
                objective_preset(preset),
                program_id=f"elite_{island}",
                metrics={"combined_score": 1.0, "hpwl_delta_pct": -1.0, "overflow_delta_pct": -6.0},
                status="accepted",
            ),
            target_island=island,
        )
    return database


def test_migration_follows_the_evolution_generation_clock(tmp_path: Path) -> None:
    database = _migration_database(tmp_path, "generation")
    rng = random.Random(1)

    assert database.maybe_migrate(iteration=19, rng=rng) == []
    first = database.maybe_migrate(iteration=20, rng=rng)
    # Every populated island sends its elite to both ring neighbours, once.
    assert sorted((m.metrics["source_island"], m.metrics["target_island"]) for m in first) == [
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
    ]
    assert database.maybe_migrate(iteration=20, rng=rng) == []
    assert database.last_migration_iteration == 20

    # The exchange repeats at every interval, not once per island.
    database.add(
        ObjectiveProgram.from_spec(
            objective_preset("dreamplace_expert_rudy_pin"),
            program_id="late_elite",
            metrics={"combined_score": 3.0, "hpwl_delta_pct": -2.0, "overflow_delta_pct": -7.0},
            status="accepted",
        ),
        target_island=0,
    )
    second = database.maybe_migrate(iteration=40, rng=rng)
    assert {m.metrics["source_program_id"] for m in second} == {"late_elite"}
    assert {m.metrics["target_island"] for m in second} == {1, 2}


def test_island_generation_clock_remains_available(tmp_path: Path) -> None:
    database = _migration_database(tmp_path, "island_generation")
    rng = random.Random(1)
    for _ in range(19):
        database.increment_island_generation(0)
    assert database.maybe_migrate(iteration=95, rng=rng) == []
    database.increment_island_generation(0)
    migrants = database.maybe_migrate(iteration=96, rng=rng)
    assert {m.metrics["source_island"] for m in migrants} == {0}


# --- per-net timing policies -------------------------------------------------

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


def test_net_weight_policy_needs_in_loop_timing_collateral(tmp_path: Path) -> None:
    spec = parse_objective_program(
        NET_POLICY_PROGRAM, created_by="test", term_scope="dreamplace_controller"
    )
    config = load_openevolve_tier2_config(PRIMARY_CONFIG)
    database = ObjectiveProgramDatabase(tmp_path / "db")

    # The primary configuration ships collateral for all eight search designs.
    assert _static_rejection_reason(spec, database, config) is None

    without_collateral = replace(config, timing_controller_panel=None)
    reason = _static_rejection_reason(spec, database, without_collateral)
    assert reason is not None
    assert "update_net_weights needs timing analysis during placement" in reason
    assert "bp_fe" in reason
