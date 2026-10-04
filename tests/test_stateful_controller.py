"""DSL v2 stateful controller tests: parsing, validation, scoping, feedback."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from coevop.backends.dreamplace import (
    parse_dreamplace_log,
    summarize_state_trajectory,
)
from coevop.llm.prompts import generation_messages
from coevop.objectives.presets import objective_preset
from coevop.objectives.program import (
    parse_objective_program,
    program_example,
)
from coevop.objectives.spec import (
    ObjectiveSpecError,
    load_objective_spec,
    write_objective_spec,
)
from coevop.objectives.terms import observable_names, privileged_observable_names


CONTROLLER_SCOPE = "dreamplace_controller"


def _parse(source: str):
    return parse_objective_program(
        source, created_by="test", term_scope=CONTROLLER_SCOPE
    )


def test_controller_example_lowers_to_stateful_spec():
    spec = _parse(program_example(CONTROLLER_SCOPE))
    assert spec.is_stateful
    assert spec.is_typed_policy
    assert spec.register_names == ["density_weight", "gamma_scale", "route_weight"]
    assert "overflow" in spec.observable_set
    assert "grad_ratio_density_electric" in spec.observable_set
    assert spec.id.startswith("obj_")


def test_stateless_program_still_parses_with_v1_id():
    source = "\n".join(
        [
            "def objective(features):",
            '    wl = term("wirelength_wawl")',
            '    den = term("density_electric")',
            "    score = wl + 1.35 * den",
            '    return score, {"wirelength": wl, "density": den}',
        ]
    )
    spec = _parse(source)
    assert not spec.is_stateful
    assert spec.state is None
    assert spec.observable_set == []


def test_init_only_registers_get_identity_updates():
    source = "\n".join(
        [
            "def init_state(obs):",
            '    return {"lam": 0.00008 * obs("grad_ratio_density_electric")}',
            "def objective(features, state):",
            '    return term("wirelength_wawl") + state("lam") * term("density_electric")',
        ]
    )
    spec = _parse(source)
    assert spec.state["registers"]["lam"]["update"] == {"op": "state", "name": "lam"}


def test_state_spec_round_trips_through_json(tmp_path):
    spec = _parse(program_example(CONTROLLER_SCOPE))
    path = write_objective_spec(spec, tmp_path / "spec.json")
    loaded = load_objective_spec(path)
    assert loaded.state == spec.state
    assert loaded.id == spec.id
    assert loaded.observable_set == spec.observable_set


@pytest.mark.parametrize(
    "label,source,fragment",
    [
        (
            "obs_in_objective",
            "def objective(features):\n"
            '    return term("wirelength_wawl") + obs("overflow")',
            "obs() is only allowed",
        ),
        (
            "term_in_update",
            "def init_state(obs):\n"
            '    return {"lam": 1.0}\n'
            "def update_state(state, obs):\n"
            '    return {"lam": state("lam") + term("density_electric")}\n'
            "def objective(features, state):\n"
            '    return term("wirelength_wawl") + state("lam") * term("density_electric")',
            "not allowed in state_update",
        ),
        (
            "update_without_init",
            "def update_state(state, obs):\n"
            '    return {"lam": 1.0}\n'
            "def objective(features, state):\n"
            '    return term("wirelength_wawl")',
            "requires init_state",
        ),
        (
            "mismatched_registers",
            "def init_state(obs):\n"
            '    return {"lam": 1.0}\n'
            "def update_state(state, obs):\n"
            '    return {"other": 1.0}\n'
            "def objective(features, state):\n"
            '    return term("wirelength_wawl") + state("lam")',
            "must return exactly the registers",
        ),
        (
            "too_many_registers",
            "def init_state(obs):\n"
            '    return {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0, "e": 1.0}\n'
            "def objective(features, state):\n"
            '    return term("wirelength_wawl") + state("a")',
            "exceed limit",
        ),
        (
            "unknown_observable",
            "def init_state(obs):\n"
            '    return {"lam": obs("magic")}\n'
            "def objective(features, state):\n"
            '    return term("wirelength_wawl") + state("lam")',
            "unknown observable",
        ),
        (
            "stateful_legacy_signature",
            "def init_state(obs):\n"
            '    return {"lam": 1.0}\n'
            "def objective(features):\n"
            '    return term("wirelength_wawl")',
            "objective(features, state)",
        ),
        (
            "privileged_observable",
            "def init_state(obs):\n"
            '    return {"lam": obs("native_density_weight")}\n'
            "def objective(features, state):\n"
            '    return term("wirelength_wawl") + state("lam") * term("density_electric")',
            "unknown observable",
        ),
        (
            "net_outside_policy",
            "def init_state(obs):\n"
            '    return {"lam": net("criticality")}\n'
            "def objective(features, state):\n"
            '    return term("wirelength_wawl") + state("lam")',
            "net() is only allowed",
        ),
    ],
)
def test_rejections(label, source, fragment):
    with pytest.raises(ObjectiveSpecError) as excinfo:
        _parse(source)
    assert fragment in str(excinfo.value), label


def test_privileged_observables_hidden_from_llm_scope():
    visible = set(observable_names(CONTROLLER_SCOPE))
    privileged = set(privileged_observable_names())
    assert privileged
    assert not (visible & privileged)
    assert privileged <= set(observable_names("all"))


def test_identity_preset_uses_privileged_observable():
    spec = objective_preset("dreamplace_controller_native_identity")
    assert spec.is_stateful
    assert spec.observable_set == ["native_density_weight"]
    assert spec.policy_interface == "typed_policy_v1"
    assert set(spec.register_names) == {"density_weight", "gamma_scale"}
    assert set(spec.components or {}) == {"wirelength", "density"}


def test_eplace_smooth_preset_is_self_contained():
    spec = objective_preset("dreamplace_controller_eplace_smooth")
    assert spec.is_stateful
    assert "native_density_weight" not in spec.observable_set
    assert "hpwl_delta_rate" in spec.observable_set


def test_net_weight_policy_parses_with_bounds():
    source = "\n".join(
        [
            "def init_state(obs):",
            '    return {"boost": 1.0}',
            "def update_net_weights(net, state, obs):",
            '    return 1.0 + 0.2 * sigmoid(4.0 * net("criticality")) * state("boost")',
            "def objective(features, state):",
            '    return term("wirelength_wawl") + state("boost") * 0.0 + term("density_electric")',
        ]
    )
    spec = _parse(source)
    policy = spec.net_weight_policy
    assert policy is not None
    assert policy["cadence"] == 1
    assert policy["control"] == "generated"
    assert "multiplier_clip" not in policy
    assert "weight_clip" not in policy
    # Policy participates in identity: same program without policy differs.
    without = _parse(
        "\n".join(
            [
                "def init_state(obs):",
                '    return {"boost": 1.0}',
                "def objective(features, state):",
                '    return term("wirelength_wawl") + state("boost") * 0.0 + term("density_electric")',
            ]
        )
    )
    assert without.id != spec.id


def test_typed_net_weight_policy_uses_policy_accessor():
    source = """
def init_policy(obs):
    return {
        "density_weight": obs("grad_ratio_component_density"),
        "gamma_scale": 1.0,
    }

def update_policy(policy, obs):
    return {
        "density_weight": policy("density_weight"),
        "gamma_scale": policy("gamma_scale"),
    }

def update_net_weights(net, policy, obs):
    return 1.0 + policy("density_weight") * sigmoid(net("criticality"))

def objective(features, policy):
    wl = term("wirelength_wawl")
    den = term("density_electric")
    score = wl + policy("density_weight") * den
    return score, {"wirelength": wl, "density": den}
"""
    spec = _parse(source)
    assert spec.net_weight_policy is not None
    assert spec.net_weight_policy["cadence"] == 1


def test_timing_controller_requires_high_coverage_and_executed_update():
    from coevop.eval.openevolve_tier2 import _apply_timing_controller_admission

    spec = objective_preset("dreamplace_controller_timing_identity")
    config = SimpleNamespace(timing_controller_min_net_coverage=0.95)
    partial = {
        "iteration_budget_satisfied": True,
        "structural_failure_count": 0,
        "in_loop_timing_net_coverage": 0.485,
        "in_loop_timing_policy_updates": 1.0,
        "in_loop_wns_delta": 0.0,
        "in_loop_tns_delta": 0.0,
    }
    _apply_timing_controller_admission(partial, spec, config)
    assert partial["timing_controller_admission_passed"] is False
    assert partial["timing_controller_failure"] == "insufficient_net_coverage"
    assert partial["parent_eligible"] is False
    assert partial["structural_failure_count"] == 1

    complete = {
        "iteration_budget_satisfied": True,
        "structural_failure_count": 0,
        "in_loop_timing_net_coverage": 0.99,
        "in_loop_timing_policy_updates": 1.0,
        "in_loop_wns_delta": 0.01,
        "in_loop_tns_delta": 0.1,
    }
    _apply_timing_controller_admission(complete, spec, config)
    assert complete["timing_controller_admission_passed"] is True
    assert complete["timing_proxy_wns_delta"] == pytest.approx(0.01)
    assert complete["timing_proxy_tns_delta"] == pytest.approx(0.1)

    # The matched native placement is not timing-driven, so a candidate may
    # report in-loop WNS/TNS without a baseline to difference against.
    without_baseline = {
        "iteration_budget_satisfied": True,
        "structural_failure_count": 0,
        "in_loop_timing_net_coverage": 0.99,
        "in_loop_timing_policy_updates": 2.0,
        "in_loop_wns": -0.4,
        "in_loop_tns": -12.0,
    }
    _apply_timing_controller_admission(without_baseline, spec, config)
    assert without_baseline["timing_controller_admission_passed"] is True
    assert "timing_proxy_wns_delta" not in without_baseline


def test_controller_prompt_lists_observables_but_not_privileged():
    messages = generation_messages(
        context={"objective_mode": "controller"},
        term_scope=CONTROLLER_SCOPE,
        mutation_mode="diff",
    )
    user_payload = json.loads(messages[1]["content"])
    assert "allowed_observables" in user_payload
    assert "overflow" in user_payload["allowed_observables"]
    assert "native_density_weight" not in user_payload["allowed_observables"]
    assert "controller_contract" in user_payload["mutation_guidance"]


def test_typed_policy_rejects_fixed_top_level_density_coefficient():
    source = """
def init_policy(obs):
    return {"density_weight": 1.0, "gamma_scale": 1.0}

def update_policy(policy, obs):
    return {
        "density_weight": policy("density_weight"),
        "gamma_scale": policy("gamma_scale"),
    }

def objective(features, policy):
    wl = term("wirelength_wawl")
    den = term("density_electric")
    return wl + 0.5 * den, {"wirelength": wl, "density": den}
"""
    with pytest.raises(ObjectiveSpecError, match="policy slot"):
        _parse(source)


def test_timing_identity_preset_is_privileged_and_bounded():
    spec = objective_preset("dreamplace_controller_timing_identity")
    assert spec.is_typed_policy
    assert spec.net_weight_policy is not None
    assert spec.net_weight_policy["control"] == "native_identity"
    assert spec.net_weight_policy["update"] == {"op": "const", "value": 1.0}


def test_typed_policy_can_calibrate_a_transformed_named_component():
    source = """
def init_policy(obs):
    return {
        "density_weight": obs("grad_ratio_component_density"),
        "gamma_scale": 1.0,
        "route_weight": obs("grad_ratio_component_routing"),
    }

def update_policy(policy, obs):
    return {
        "density_weight": policy("density_weight"),
        "gamma_scale": policy("gamma_scale"),
        "route_weight": policy("route_weight"),
    }

def objective(features, policy):
    wl = term("wirelength_wawl")
    den = term("density_electric")
    route = log1p(term("soft_rudy_pnorm"))
    score = wl + policy("density_weight") * den + policy("route_weight") * route
    return score, {"wirelength": wl, "density": den, "routing": route}
"""
    spec = _parse(source)
    assert "grad_ratio_component_density" in spec.observable_set
    assert "grad_ratio_component_routing" in spec.observable_set


def test_component_calibration_must_reference_a_returned_component():
    source = program_example(CONTROLLER_SCOPE).replace(
        'obs("grad_ratio_density_electric")',
        'obs("grad_ratio_component_missing")',
    )
    with pytest.raises(ObjectiveSpecError, match="not returned"):
        _parse(source)


def test_parse_dreamplace_log_collects_state_trajectory(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "[INFO   ] DREAMPlace - CoEvoP&R custom objective enabled: obj_x semantics=raw stateful=True",
                "[INFO   ] DREAMPlace - CoEvoP&R observable grad_ratio_density_electric=2.786302E-06 (wl_grad=8.4E+01 term_grad=3.0E+07)",
                "[INFO   ] DREAMPlace - CoEvoP&R observable init_value_density_electric=1.234000E+04",
                '[INFO   ] DREAMPlace - CoEvoP&R state sample {"density_weight": 1e-10, "iteration": 0, "lam": 1e-10, "overflow": 1.0}',
                '[INFO   ] DREAMPlace - CoEvoP&R state sample {"density_weight": 2e-10, "iteration": 10, "lam": 2e-10, "overflow": 0.8}',
                '[INFO   ] DREAMPlace - CoEvoP&R timing policy {"iteration": 510, "multiplier_mean": 1.1, "update_count": 1}',
            ]
        ),
        encoding="utf-8",
    )
    metrics = parse_dreamplace_log(log)
    custom = metrics["custom_objective"]
    assert len(custom["state_trajectory"]) == 2
    assert custom["calibration_observables"]["grad_ratio_density_electric"] == pytest.approx(
        2.786302e-06
    )
    assert custom["calibration_observables"]["init_value_density_electric"] == pytest.approx(
        1.234e04
    )
    assert custom["state_summary"]["lam"]["first"] == pytest.approx(1e-10)
    assert custom["state_summary"]["lam"]["last"] == pytest.approx(2e-10)
    assert custom["timing_policy_trajectory"][0]["multiplier_mean"] == pytest.approx(1.1)


def test_mechanism_view_distinguishes_schedules():
    from coevop.eval.openevolve_tier2 import (
        _mechanism_distance,
        _mechanism_signature,
        _spec_mechanism_view,
    )

    identity = objective_preset("dreamplace_controller_native_identity")
    eplace = objective_preset("dreamplace_controller_eplace_smooth")
    generated = _parse(program_example(CONTROLLER_SCOPE))

    views = {
        "identity": _spec_mechanism_view(identity),
        "eplace": _spec_mechanism_view(eplace),
        "generated": _spec_mechanism_view(generated),
    }
    signatures = {name: _mechanism_signature(view) for name, view in views.items()}
    assert len(set(signatures.values())) == 3

    # Same score shape, different schedules: must clear the clone threshold.
    assert _mechanism_distance(views["identity"], views["eplace"]) >= 0.35
    assert _mechanism_distance(views["generated"], views["identity"]) >= 0.35


def test_mechanism_view_stateless_passthrough():
    from coevop.eval.openevolve_tier2 import _mechanism_view

    ast = {"op": "term", "name": "wirelength_wawl"}
    assert _mechanism_view(ast, None) is ast
    assert _mechanism_view(ast, {"registers": {}}) is ast


def test_summarize_state_trajectory_downsamples():
    trajectory = [
        {"iteration": i, "lam": float(i)} for i in range(0, 1000, 10)
    ]
    summary = summarize_state_trajectory(trajectory, max_points=10)
    assert summary["lam"]["min"] == 0.0
    assert summary["lam"]["last"] == 990.0
    assert len(summary["lam"]["samples"]) <= 11


def test_pareto_selection_has_no_hpwl_or_overflow_admission_gate(tmp_path):
    from coevop.eval.openevolve_tier2 import (
        _prepare_pareto_candidate,
        _refresh_pareto_selection,
    )
    from coevop.evolution.openevolve_core import (
        ObjectiveDatabaseConfig,
        ObjectiveProgram,
        ObjectiveProgramDatabase,
    )

    config = SimpleNamespace(selection_policy="pareto_multiobjective")
    database = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(num_islands=1, population_size=20, archive_size=20),
    )
    spec = objective_preset("dreamplace_controller_eplace_smooth")
    rows = [
        ("hpwl_tradeoff", -2.0, 1.0, 0.1, 1.0),
        ("hpwl_tradeoff_slow", -2.0, 1.0, 0.1, 2.0),
        ("overflow_tradeoff", 0.0, -5.0, 0.0, 1.0),
        ("dominated", 5.0, 5.0, -0.1, 1.0),
    ]
    for program_id, hpwl, overflow, wns, runtime in rows:
        metrics = {
            "hpwl_delta_pct": hpwl,
            "overflow_delta_pct": overflow,
            "timing_proxy_wns_delta": wns,
            "iteration_budget_satisfied": True,
            "structural_failure_count": 0,
            "runtime_seconds": runtime,
            "per_design_deltas": [
                {
                    "design": "bp_fe",
                    "hpwl_delta_pct": hpwl,
                    "overflow_delta_pct": overflow,
                }
            ],
        }
        _prepare_pareto_candidate(metrics, config, allow_seed_baseline=False)
        assert metrics["parent_eligible"] is True
        database.add(
            ObjectiveProgram.from_spec(spec, program_id=program_id, metrics=metrics)
        )
    _refresh_pareto_selection(database, config)

    assert database.get("hpwl_tradeoff").metrics["pareto_front"] == 0
    assert database.get("hpwl_tradeoff_slow").metrics["pareto_front"] == 0
    assert database.get("overflow_tradeoff").metrics["pareto_front"] == 0
    assert database.get("dominated").metrics["pareto_front"] > 0
    assert database.get("dominated").is_parent_eligible is False
    assert database.get("dominated").metrics["negative_memory_only"] is True
    assert database.get("dominated").is_elite_eligible is False
    assert (
        database.get("hpwl_tradeoff").combined_score
        > database.get("hpwl_tradeoff_slow").combined_score
    )


def test_pareto_evidence_keeps_wns_and_tns_in_distinct_coordinates():
    from coevop.eval.openevolve_tier2 import _dominates, _pareto_evidence

    wns_only = _pareto_evidence(
        {
            "hpwl_delta_pct": 0.0,
            "overflow_delta_pct": 0.0,
            "timing_proxy_wns_delta": 0.1,
        }
    )
    tns_only = _pareto_evidence(
        {
            "hpwl_delta_pct": 0.0,
            "overflow_delta_pct": 0.0,
            "timing_proxy_tns_delta": 1.0,
        }
    )

    assert wns_only == {
        "wirelength": {"A": 0.0},
        "overflow": {"A": 0.0},
        "wns": {"B": -0.1},
        "tns": {},
    }
    assert tns_only["wns"] == {}
    assert tns_only["tns"] == {"B": -1.0}
    assert _dominates(wns_only, tns_only) is False
    assert _dominates(tns_only, wns_only) is False
