import csv
import json
import random
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

import pytest

from coevop.eval import openevolve_tier2
from coevop.eval.openevolve_tier2 import (
    _apply_baseline_portfolio_gate,
    _apply_final_def_selection_metrics,
    _apply_manuscript_multimetric_selection,
    _apply_measured_routing_influence_gate,
    _apply_near_miss_parent_policy,
    _apply_native_default_selection_policy,
    _apply_routing_gate,
    _ast_signature,
    _baseline_portfolio_summary,
    _calibration_combinations,
    _calibrated_spec,
    _database_objective_identity,
    _derive_calibrated_route_coeff_grid,
    _design_profile_context,
    _generate_and_evaluate_children,
    _evaluate_calibrated_siblings,
    _final_tier2_candidate_objective_ids,
    _iteration_mutation_mode,
    _load_iteration_feedback,
    _mechanism_distance,
    _mechanism_signature,
    _metrics_from_rows,
    _append_retry_feedback_message,
    _promote_bootstrap_seed_parent,
    _refresh_database_scores,
    _retry_feedback_payload,
    _retry_repair_instructions,
    _route_coefficients,
    _rows_with_primary_baseline,
    _run_scheduled_tier3_feedback,
    _sample_iteration_parent,
    _select_best_aggregate_generated_program,
    _select_best_pareto_generated_program,
    _select_best_robust_generated_program,
    _seed_initial_programs,
    _static_rejection_reason,
    _tier3_candidate_programs,
    _target_design_context,
    _term_scale_audit_result,
    _term_scale_prompt_context,
    _validate_generalization_split,
    audit_openevolve_prompt_artifacts,
    audit_prompt_messages,
    build_openevolve_prompt_dry_run,
    load_openevolve_tier2_config,
    run_openevolve_tier2,
)
from coevop.eval.ranking import aggregate_rank_scores
from coevop.eval.design_profile import _macro_summary
from coevop.evolution.openevolve_core import (
    ObjectiveDatabaseConfig,
    ObjectiveProgram,
    ObjectiveProgramDatabase,
    _prompt_visible_policy,
)
from coevop.llm.prompts import generation_messages
from coevop.objectives.presets import objective_preset
from coevop.objectives.program import parse_objective_program
from coevop.objectives.spec import load_objective_spec


def test_scheduled_routed_evidence_updates_program_and_feedback(monkeypatch, tmp_path):
    config = load_openevolve_tier2_config(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "openevolve_tier2"
        / "chipbench_controller_tier2.toml"
    )
    panel = tmp_path / "post_route.toml"
    panel.write_text("seeds = [1000]\n", encoding="utf-8")
    config = replace(
        config,
        tier3_panel=str(panel),
        tier3_top_k=1,
        tier3_interval=20,
        tier3_start_iteration=20,
    )
    comparison = tmp_path / "tier2.csv"
    comparison.write_text(
        "design,objective_id,seed,status,output_artifact\n"
        "bp_fe,default,1000,success,default.def\n"
        "bp_fe,obj_candidate,1000,success,candidate.def\n",
        encoding="utf-8",
    )
    program = ObjectiveProgram(
        id="program_candidate",
        code="candidate",
        objective_spec={"id": "obj_candidate"},
        status="accepted",
        metrics={"combined_score": 1.0, "pareto_admission_passed": True},
        artifacts={"tier2_summary": {"comparison_csv": str(comparison)}},
    )
    database = ObjectiveProgramDatabase(tmp_path / "db")
    database.add(program)

    def fake_route(**kwargs):
        output = Path(kwargs["run_dir"]) / "comparison_table.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "design,objective_id,seed,status,routed_wirelength_delta_pct,"
            "grt_overflow_delta_pct,wns_gain_ns,tns_gain_ns\n"
            "bp_fe,obj_candidate,1000,success,-10,-20,0.25,4.0\n",
            encoding="utf-8",
        )
        return {"comparison_csv": str(output), "success_count": 1}

    monkeypatch.setattr(openevolve_tier2, "run_tier3_openroad", fake_route)
    summary = _run_scheduled_tier3_feedback(
        config=config,
        database=database,
        programs=[program],
        iteration=20,
        run_root=tmp_path / "run",
        chipbench_root=tmp_path / "chipbench",
        resume=False,
        retry_failed=False,
    )

    assert summary["status"] == "success"
    assert program.metrics["routed_wirelength_delta_pct"] == -10.0
    assert program.metrics["routed_overflow_delta_pct"] == -20.0
    assert program.metrics["post_route_wns_gain_ns"] == 0.25
    assert program.metrics["post_route_tns_gain_ns"] == 4.0
    feedback = json.loads((tmp_path / "run" / "routed_feedback.json").read_text())
    assert feedback["evidence"][0]["objective_id"] == "obj_candidate"


def _shared_panel(tmp_path: Path) -> Path:
    dreamplace_config = tmp_path / "bp_fe.json"
    chipbench_config = tmp_path / "config.mk"
    dreamplace_config.write_text(
        json.dumps(
            {
                "target_density": 0.72,
                "num_bins_x": 256,
                "num_bins_y": 256,
                "routability_opt_flag": 1,
                "timing_opt_flag": 0,
                "macro_place_flag": 1,
                "macro_halo_x": 10.0,
                "macro_halo_y": 10.0,
                "coevop_chipbench_generation_audit": {"input_stage": "pre_macro"},
                "coevop_macro_preflight": {
                    "hard_macro_count": 11,
                    "movable_macro_count": 11,
                    "fixed_macro_count": 0,
                    "status_counts": {"UNPLACED": 11},
                    "def_sha256": "abc123",
                },
            }
        ),
        encoding="utf-8",
    )
    chipbench_config.write_text(
        "\n".join(
            [
                "DESIGN_NAME=bp_fe_top",
                "PLATFORM=nangate45",
                "CORE_AREA=0 0 1200 800",
                "PLACE_DENSITY=0.72",
                "CLOCK_PERIOD=1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    panel = tmp_path / "shared_panel.toml"
    panel.write_text(
        "\n".join(
            [
                "seeds = [1000]",
                "iterations = 1000",
                "timeout_seconds = 9",
                "",
                "[[designs]]",
                'name = "bp_fe"',
                f'dreamplace_config = "{dreamplace_config.as_posix()}"',
                f'chipbench_config = "{chipbench_config.as_posix()}"',
                'mode = "global"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return panel


def test_term_scale_audit_result_flags_failed_or_flat_terms():
    run = SimpleNamespace(
        returncode=0,
        runtime_seconds=1.5,
        config_path="config.json",
        log_path="dreamplace.log",
        metrics={
            "custom_objective": {
                "calls": 3,
                "last_grad_norm": 0.0,
                "output_scale": 1.0,
                "term_scales": {"soft_rudy_pnorm": 2.0},
                "last_terms": {
                    "soft_rudy_pnorm": {"raw": 4.0, "normalized": 2.0}
                },
            }
        },
    )

    result = _term_scale_audit_result(
        design="gcd",
        term="soft_rudy_pnorm",
        run=run,
        min_grad_norm=1e-12,
    )

    assert result["status"] == "failed"
    assert "near_zero_gradient" in result["warnings"]
    assert result["raw_value"] == 4.0


def test_term_scale_audit_result_blocks_excessive_runtime():
    run = SimpleNamespace(
        returncode=0,
        runtime_seconds=301.0,
        config_path="config.json",
        log_path="dreamplace.log",
        metrics={
            "custom_objective": {
                "calls": 2,
                "last_grad_norm": 1.0,
                "output_scale": 1.0,
                "term_scales": {"density_bell": 1.0},
                "last_terms": {
                    "density_bell": {"raw": 1.0, "normalized": 1.0}
                },
            }
        },
    )

    result = _term_scale_audit_result(
        design="superblue3",
        term="density_bell",
        run=run,
        min_grad_norm=1e-12,
        max_runtime_seconds=300.0,
    )

    assert result["status"] == "failed"
    assert "excessive_runtime" in result["warnings"]
    assert "exceeds term audit limit" in result["reason"]


def test_term_scale_prompt_context_is_compact_and_prompt_visible():
    payload = {
        "status": "completed",
        "blocked_terms": ["density_bell"],
        "results": [
            {
                "design": "gcd",
                "term": "soft_rudy_pnorm",
                "status": "passed",
                "raw_value": 1000000.0,
                "normalized_value": 0.5,
                "gradient_norm": 0.01,
                "reference_gradient_ratios": {
                    "gradient_ratio_to_wirelength_wawl": 0.2,
                    "gradient_ratio_to_density_electric": 0.5,
                },
                "warnings": [],
            }
        ],
    }

    context = _term_scale_prompt_context(payload)
    visible = _prompt_visible_policy({"term_scale_audit": context})

    assert visible["term_scale_audit"]["blocked_terms"] == ["density_bell"]
    row = visible["term_scale_audit"]["by_design"]["gcd"][0]
    assert row["raw_value"] == "1.000e+06"
    assert row["gradient_ratio_to_wirelength_wawl"] == 0.2


def test_static_validation_rejects_term_scale_audit_blocked_terms(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        term_scale_audit_blocked_terms=["density_bell"],
    )
    database = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength_wawl")',
                '    den = term("density_bell")',
                '    route = term("route_pressure_long")',
                '    pins = term("pin_density_pnorm")',
                "    score = wl + den + 1e-8 * log1p(route * pins)",
                (
                    '    return score, {"wirelength": wl, "density": den, '
                    '"route": route, "pins": pins}'
                ),
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )

    reason = _static_rejection_reason(spec, database, config)

    assert reason is not None
    assert "blocked by the deterministic term-scale audit" in reason


def test_initial_presets_do_not_launch_blocked_terms(tmp_path, monkeypatch):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        initial_presets=[
            "dreamplace_wawl_density_bell",
            "dreamplace_wawl_electric",
        ],
        term_scale_audit_blocked_terms=["density_bell"],
    )
    database = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    captured = {}

    def fail_after_capture(**kwargs):
        captured["objective_paths"] = list(kwargs["objective_paths"])
        raise RuntimeError("stop after objective-path capture")

    monkeypatch.setattr(
        openevolve_tier2,
        "run_tier2_dreamplace",
        fail_after_capture,
    )
    openevolve_tier2._evaluate_initial_presets(
        database=database,
        config=config,
        run_root=tmp_path / "run",
        dreamplace_root=tmp_path / "DREAMPlace" / "install",
        resume=False,
        retry_failed=False,
    )

    launched = [load_objective_spec(path) for path in captured["objective_paths"]]
    assert len(launched) == 1
    assert set(launched[0].term_set) == {"wirelength_wawl", "density_electric"}
    blocked = json.loads(
        (tmp_path / "run" / "initial_presets" / "blocked_presets.json").read_text(
            encoding="utf-8"
        )
    )
    assert blocked[0]["preset"] == "dreamplace_wawl_density_bell"
    assert blocked[0]["blocked_terms"] == ["density_bell"]
    blocked_program = next(
        program
        for program in database.programs.values()
        if program.metrics.get("term_scale_audit_blocked_terms")
    )
    assert blocked_program.status == "rejected"
    assert blocked_program.is_parent_eligible is False


def _generalization_panel(tmp_path: Path) -> Path:
    ethernet_config = tmp_path / "ethernet.json"
    isa_config = tmp_path / "isa_npu.json"
    chipbench_config = tmp_path / "heldout_config.mk"
    ethernet_config.write_text("{}", encoding="utf-8")
    isa_config.write_text("{}", encoding="utf-8")
    chipbench_config.write_text("DESIGN_NAME=heldout\n", encoding="utf-8")
    panel = tmp_path / "generalization_panel.toml"
    panel.write_text(
        "\n".join(
            [
                "seeds = [1000]",
                "iterations = 1000",
                "timeout_seconds = 9",
                "",
                "[[designs]]",
                'name = "ethernet"',
                f'dreamplace_config = "{ethernet_config.as_posix()}"',
                f'chipbench_config = "{chipbench_config.as_posix()}"',
                'mode = "global"',
                "",
                "[[designs]]",
                'name = "isa_npu"',
                f'dreamplace_config = "{isa_config.as_posix()}"',
                f'chipbench_config = "{chipbench_config.as_posix()}"',
                'mode = "global"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return panel


def _config(tmp_path: Path, panel: Path) -> Path:
    config = tmp_path / "openevolve.toml"
    config.write_text(
        "\n".join(
            [
                'provider = "mock"',
                'term_scope = "dreamplace_replacement"',
                "max_iterations = 2",
                "population_size = 20",
                "archive_size = 5",
                "num_islands = 2",
                "checkpoint_interval = 1",
                "num_top_programs = 2",
                "num_diverse_programs = 1",
                "severe_regression_pct = 10.0",
                "seed = 123",
                'initial_presets = ["dreamplace_explicit_wl_density", "dreamplace_density_heavy"]',
                "",
                "[search]",
                f'panel = "{panel.as_posix()}"',
                'baseline_presets = ["dreamplace_routing_aware_smoke"]',
                "",
                "[final]",
                "enabled = false",
                "",
                "[feedback]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config


def test_openevolve_config_loads_defaults(tmp_path):
    config = load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path)))

    assert config.provider == "mock"
    assert config.term_scope == "dreamplace_replacement"
    assert config.objective_mode == "replacement"
    assert config.mutation_mode == "diff"
    assert config.parent_policy == "hpwl_safe_only"
    assert config.primary_baseline == "custom_default"
    assert config.density_coeff_grid == [1.0]
    assert config.wirelength_coeff_grid == [1.0]
    assert config.native_default_selection_policy == "report"
    assert config.native_default_pareto_bonus == 0.0
    assert config.require_robust_parent_gate is True
    assert config.max_iterations == 2
    assert config.num_islands == 2
    assert config.samples_per_iteration == 1
    assert config.map_elites_enabled is True
    assert config.feature_bins == 8
    assert config.migration_interval == 20
    assert config.search_disable_legalization is False
    assert config.final_disable_legalization is False
    assert config.term_scale_audit_max_runtime_seconds == 0.0
    assert config.minimum_scoring_iterations == 1000
    assert config.allow_short_search_for_tests is False


def test_scoring_budget_rejects_short_real_provider_panel(tmp_path):
    panel = _shared_panel(tmp_path)
    panel.write_text(
        panel.read_text(encoding="utf-8").replace("iterations = 1000", "iterations = 50"),
        encoding="utf-8",
    )
    config = load_openevolve_tier2_config(_config(tmp_path, panel))
    config = replace(config, provider="openai")

    with pytest.raises(ValueError, match="at least 1000 iterations"):
        openevolve_tier2._validate_scoring_iteration_budgets(config)


def test_scoring_budget_short_override_is_mock_test_only(tmp_path):
    panel = _shared_panel(tmp_path)
    panel.write_text(
        panel.read_text(encoding="utf-8").replace("iterations = 1000", "iterations = 50"),
        encoding="utf-8",
    )
    config = load_openevolve_tier2_config(_config(tmp_path, panel))
    config = replace(config, allow_short_search_for_tests=True)

    policy = openevolve_tier2._validate_scoring_iteration_budgets(config)

    assert policy["claim_valid"] is False
    assert policy["test_only_override"] is True
    assert policy["panels"][0]["budget_satisfied"] is False


def test_openevolve_config_loads_stage_legalization_controls(tmp_path):
    panel = _shared_panel(tmp_path)
    config_path = tmp_path / "openevolve_legalization.toml"
    config_path.write_text(
        "\n".join(
            [
                'provider = "mock"',
                'term_scope = "dreamplace_replacement"',
                "max_iterations = 1",
                "population_size = 10",
                "archive_size = 3",
                "num_islands = 1",
                "checkpoint_interval = 1",
                "num_top_programs = 1",
                "num_diverse_programs = 1",
                "severe_regression_pct = 10.0",
                "seed = 123",
                "",
                "[search]",
                f'panel = "{panel.as_posix()}"',
                "disable_legalization = true",
                "",
                "[final]",
                "enabled = true",
                f'panel = "{panel.as_posix()}"',
                "disable_legalization = false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_openevolve_tier2_config(config_path)

    assert config.search_disable_legalization is True
    assert config.final_disable_legalization is False


def test_openevolve_config_and_prompt_support_design_specific_generalization(tmp_path):
    search_panel = _shared_panel(tmp_path)
    heldout_panel = _generalization_panel(tmp_path)
    config_path = tmp_path / "openevolve_generalization.toml"
    config_path.write_text(
        "\n".join(
            [
                'provider = "mock"',
                'term_scope = "dreamplace_replacement"',
                'target_designs = ["bp_fe"]',
                "",
                "[search]",
                f'panel = "{search_panel.as_posix()}"',
                "",
                "[generalization]",
                "enabled = true",
                f'panel = "{heldout_panel.as_posix()}"',
                "top_k = 2",
                'baseline_presets = ["dreamplace_explicit_wl_density"]',
                "",
                "[final]",
                "enabled = false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_openevolve_tier2_config(config_path)
    output_dir = tmp_path / "prompt_dry_generalization"
    payload = build_openevolve_prompt_dry_run(
        config_path=config_path,
        output_dir=output_dir,
        iteration=1,
    )
    prompt_messages = json.loads((output_dir / "prompt_messages.json").read_text(encoding="utf-8"))
    prompt_text = "\n".join(message["content"] for message in prompt_messages)

    assert config.target_designs == ["bp_fe"]
    assert config.generalization_enabled is True
    assert config.generalization_top_k == 2
    assert payload["prompt_audit"]["passed"] is True
    assert "optimization_designs" in prompt_text
    assert "bp_fe" in prompt_text
    assert "generalization_designs_withheld_from_feedback" in prompt_text
    assert "ethernet" in prompt_text
    assert "held out from parent selection" in prompt_text
    assert "chip_design_profiles" in prompt_text
    assert "PLACE_DENSITY" in prompt_text
    assert "CORE_AREA" in prompt_text
    assert "heldout_generalization_profiles" in prompt_text
    assert "search_baseline_behavior" not in prompt_text


def test_blind_generalization_hides_heldout_from_prompt_context(tmp_path):
    search_panel = _shared_panel(tmp_path)
    heldout_panel = _generalization_panel(tmp_path)
    config_path = tmp_path / "blind_generalization.toml"
    config_path.write_text(
        "\n".join(
            [
                'provider = "mock"',
                'term_scope = "dreamplace_replacement"',
                'target_designs = ["bp_fe"]',
                "",
                "[search]",
                f'panel = "{search_panel.as_posix()}"',
                "",
                "[generalization]",
                "enabled = true",
                "blind = true",
                "strict_leakage_guard = true",
                f'panel = "{heldout_panel.as_posix()}"',
                "",
                "[final]",
                "enabled = false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_openevolve_tier2_config(config_path)
    context = _target_design_context(config)
    profile = _design_profile_context(config)
    output_dir = tmp_path / "prompt_dry_blind_generalization"
    build_openevolve_prompt_dry_run(
        config_path=config_path,
        output_dir=output_dir,
        iteration=1,
    )
    prompt_messages = json.loads((output_dir / "prompt_messages.json").read_text(encoding="utf-8"))
    prompt_text = "\n".join(message["content"] for message in prompt_messages)

    assert context["blind_generalization"] is True
    assert context["heldout_design_count"] == 2
    assert context["generalization_designs_withheld_from_feedback"] == []
    assert profile["heldout_generalization_profiles"] == []
    assert "ethernet" not in prompt_text
    assert "isa_npu" not in prompt_text


def test_strict_generalization_guard_rejects_overlap(tmp_path):
    panel = _shared_panel(tmp_path)
    config_path = tmp_path / "bad_generalization.toml"
    config_path.write_text(
        "\n".join(
            [
                'provider = "mock"',
                'term_scope = "dreamplace_replacement"',
                'target_designs = ["bp_fe"]',
                "",
                "[search]",
                f'panel = "{panel.as_posix()}"',
                "",
                "[generalization]",
                "enabled = true",
                "blind = true",
                "strict_leakage_guard = true",
                f'panel = "{panel.as_posix()}"',
                "",
                "[final]",
                "enabled = false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_openevolve_tier2_config(config_path)
    with pytest.raises(ValueError, match="heldout generalization leakage"):
        _validate_generalization_split(config)


def test_design_profile_context_adds_measured_search_baseline_behavior(tmp_path):
    panel = _shared_panel(tmp_path)
    config_path = tmp_path / "openevolve_profile.toml"
    config_path.write_text(
        "\n".join(
            [
                'provider = "mock"',
                'term_scope = "dreamplace_replacement"',
                'target_designs = ["bp_fe"]',
                'initial_presets = ["dreamplace_explicit_wl_density"]',
                "",
                "[search]",
                f'panel = "{panel.as_posix()}"',
                'baseline_presets = ["dreamplace_explicit_wl_density"]',
                "",
                "[design_profile]",
                "enabled = true",
                "include_file_stats = true",
                "",
                "[final]",
                "enabled = false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = load_openevolve_tier2_config(config_path)
    comparison_csv = tmp_path / "comparison_table.csv"
    rows = [
        {
            "design": "bp_fe",
            "objective_id": "default",
            "seed": 1000,
            "status": "success",
            "hpwl_delta_pct": 0.0,
            "overflow_delta_pct": 0.0,
            "runtime_seconds": 10.0,
            "outcome_label": "success_or_improvement",
        },
        {
            "design": "bp_fe",
            "objective_id": objective_preset("dreamplace_explicit_wl_density").id,
            "seed": 1000,
            "status": "success",
            "hpwl_delta_pct": -1.2,
            "overflow_delta_pct": -3.4,
            "runtime_seconds": 11.0,
            "outcome_label": "success_or_improvement",
        },
    ]
    with comparison_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    database = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=5, archive_size=2, num_islands=1),
    )
    spec = objective_preset("dreamplace_explicit_wl_density")
    database.add(
        ObjectiveProgram.from_spec(
            spec,
            program_id=f"seed_00_{spec.id}",
            generation=0,
            iteration_found=0,
            metrics={"seed_baseline": True},
            artifacts={"tier2_summary": {"comparison_csv": str(comparison_csv)}},
            status="accepted",
        ),
        target_island=0,
    )

    context = _design_profile_context(config, database=database)

    assert context["search_design_profiles"][0]["name"] == "bp_fe"
    assert context["search_design_profiles"][0]["chipbench_config_summary"]["PLACE_DENSITY"] == "0.72"
    macro = context["search_design_profiles"][0]["macro_placement_summary"]
    assert macro["movable_macro_count"] == 11
    assert macro["fixed_macro_count"] == 0
    assert macro["joint_macro_cell_global_placement"] is True
    behavior = context["search_baseline_behavior"]["by_design"]["bp_fe"]
    assert {item["objective_id"] for item in behavior} >= {
        "default",
        spec.id,
    }
    assert context["search_baseline_behavior"]["source"] == "measured_search_panel_rows"


def test_design_profile_distinguishes_movability_from_macro_legalizer() -> None:
    summary = _macro_summary(
        {
            "macro_place_flag": 0,
            "coevop_macro_preflight": {
                "hard_macro_count": 3787,
                "movable_macro_count": 128,
                "fixed_macro_count": 3659,
                "status_counts": {"PLACED": 128, "FIXED": 3659},
                "def_sha256": "mixed-size-def",
            },
        }
    )

    assert summary["joint_macro_cell_global_placement"] is True
    assert summary["specialized_macro_legalizer_enabled"] is False
    assert summary["fixed_hard_macros_are_obstacles"] is True
    assert "jointly optimizes" in summary["objective_scope_note"]
    assert "legality is audited separately" in summary["objective_scope_note"]


def test_openevolve_config_loads_router_background(tmp_path):
    panel = _shared_panel(tmp_path)
    background = tmp_path / "router_background.md"
    background.write_text("global route overflow = max(0, demand - capacity)\n", encoding="utf-8")
    config_path = tmp_path / "openevolve_background.toml"
    config_path.write_text(
        "\n".join(
            [
                'provider = "mock"',
                'term_scope = "dreamplace_replacement"',
                f'router_background = "{background.as_posix()}"',
                "",
                "[search]",
                f'panel = "{panel.as_posix()}"',
                "",
                "[final]",
                "enabled = false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_openevolve_tier2_config(config_path)

    assert config.router_background_path == str(background)
    assert "global route overflow" in config.router_background


def test_openevolve_prompt_dry_run_writes_audited_prompt(tmp_path):
    config = _config(tmp_path, _shared_panel(tmp_path))
    output_dir = tmp_path / "prompt_dry_run"

    payload = build_openevolve_prompt_dry_run(
        config_path=config,
        output_dir=output_dir,
        iteration=3,
    )

    prompt_messages = json.loads((output_dir / "prompt_messages.json").read_text(encoding="utf-8"))
    prompt_text = "\n".join(message["content"] for message in prompt_messages)

    assert payload["prompt_audit"]["passed"] is True
    assert payload["parent_id"]
    assert (output_dir / "prompt_context.json").is_file()
    assert (output_dir / "prompt_audit.json").is_file()
    assert "wirelength_lse" in prompt_text
    assert "density_bell" in prompt_text
    assert "native_objective" not in prompt_text
    assert "hpwl_gate_search_pct" not in prompt_text


def test_openevolve_prompt_dry_run_includes_configured_prior_feedback(tmp_path):
    panel = _shared_panel(tmp_path)
    feedback = tmp_path / "prior_feedback.json"
    feedback.write_text(
        json.dumps(
            {
                "feedback_version": 1,
                "summary": {
                    "best_overall_objective": "dreamplace_wawl_density_bell",
                    "promotion_worthy_generated_pareto_wins": 0,
                },
                "measured_lessons": [
                    {
                        "mechanism_terms": ["wirelength_wawl", "density_bell"],
                        "observation": "prior measured lesson visible in prompt",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "openevolve_prior_feedback.toml"
    config_path.write_text(
        "\n".join(
            [
                'provider = "mock"',
                'term_scope = "dreamplace_replacement"',
                'objective_mode = "replacement"',
                'primary_baseline = "default"',
                'require_nonzero_routing_term = true',
                'min_replacement_nonbaseline_terms = 2',
                'require_baseline_portfolio_gate = true',
                'require_baseline_portfolio_pareto = true',
                "",
                "[search]",
                f'panel = "{panel.as_posix()}"',
                "",
                "[feedback]",
                f'prior_feedback = "{feedback.as_posix()}"',
                "",
                "[final]",
                "enabled = false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "prompt_dry_feedback"

    payload = build_openevolve_prompt_dry_run(
        config_path=config_path,
        output_dir=output_dir,
        iteration=1,
    )

    prompt_messages = json.loads((output_dir / "prompt_messages.json").read_text(encoding="utf-8"))
    prompt_text = "\n".join(message["content"] for message in prompt_messages)
    context = json.loads((output_dir / "prompt_context.json").read_text(encoding="utf-8"))

    assert payload["prompt_audit"]["passed"] is True
    assert context["run_feedback"]["summary"]["best_overall_objective"] == (
        "dreamplace_wawl_density_bell"
    )
    assert "prior measured lesson visible in prompt" in prompt_text
    assert "require_baseline_portfolio_pareto" not in prompt_text
    assert "hpwl_gate_search_pct" not in prompt_text


def test_route_density_interaction_is_supported_and_calibrated(tmp_path):
    source = "\n".join(
        [
            "def objective(features):",
            '    wl = term("wirelength_wawl")',
            '    den = term("density_bell")',
            '    route = term("route_pressure_long")',
            "    interaction = den * sigmoid(route)",
            "    score = wl + den + 0.001 * interaction",
            '    return score, {"wl": wl, "den": den, "interaction": interaction}',
        ]
    )
    spec = parse_objective_program(
        source,
        created_by="test",
        rationale="route-density interaction test",
        term_scope="dreamplace_replacement",
    )

    coeffs, unsupported = _route_coefficients(spec.ast)

    assert unsupported == set()
    assert coeffs == [("interaction:route_pressure_long", 0.001)]

    calibrated = _calibrated_spec(
        spec,
        route_coeff=1e-8,
        density_coeff=None,
        wirelength_coeff=None,
    )
    calibrated_coeffs, calibrated_unsupported = _route_coefficients(calibrated.ast)

    assert calibrated_unsupported == set()
    assert calibrated_coeffs == [("interaction:route_pressure_long", 1e-8)]


def test_static_rejection_allows_safe_route_anchor_interaction(tmp_path):
    source = "\n".join(
        [
            "def objective(features):",
            '    wl = term("wirelength_wawl")',
            '    den = term("density_bell")',
            '    route = term("route_pressure_long")',
            "    route_density = den * sigmoid(route)",
            "    score = wl + den + 1e-8 * route_density",
            '    return score, {"wl": wl, "den": den, "route_density": route_density}',
        ]
    )
    spec = parse_objective_program(
        source,
        created_by="test",
        rationale="static interaction validation test",
        term_scope="dreamplace_replacement",
    )
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        route_coeff_cap=1e-6,
        require_nonzero_routing_term=True,
        reject_baseline_mechanism_clones=False,
        reject_repeated_negative_mechanisms=False,
    )
    database = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )

    failure = _static_rejection_reason(spec, database, config)

    assert failure is None


def test_iteration_feedback_reloads_configured_file(tmp_path):
    panel = _shared_panel(tmp_path)
    feedback = tmp_path / "routed_feedback.json"
    feedback.write_text(json.dumps({"version": 1, "lesson": "initial"}), encoding="utf-8")
    config_path = tmp_path / "openevolve_feedback.toml"
    config_path.write_text(
        "\n".join(
            [
                'provider = "mock"',
                'term_scope = "dreamplace_replacement"',
                "",
                "[search]",
                f'panel = "{panel.as_posix()}"',
                "",
                "[final]",
                "enabled = false",
                "",
                "[feedback]",
                f'prior_feedback = "{feedback.as_posix()}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config = load_openevolve_tier2_config(config_path)

    assert _load_iteration_feedback(config)["version"] == 1

    feedback.write_text(json.dumps({"version": 2, "lesson": "routed"}), encoding="utf-8")

    assert _load_iteration_feedback(config)["version"] == 2

    fallback_config = replace(config, prior_feedback=None)
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "routed_feedback.json").write_text(
        json.dumps({"version": 3, "lesson": "run-local"}),
        encoding="utf-8",
    )

    assert _load_iteration_feedback(fallback_config, run_root=run_root)["version"] == 3


def test_hpwl_constrained_score_blocks_overflow_only_regression(tmp_path):
    config = load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path)))
    ranking = SimpleNamespace(
        average_rank=1.0,
        structural_failure_count=0,
        metric_regression_count=1,
        severe_regression_count=1,
        design_count=1,
        seed_count=1,
        cell_count=1,
        metric_ranks={"hpwl_delta_pct": 2.0, "overflow_delta_pct": 1.0},
        design_ranks={"bp_fe": 1.5},
    )
    metrics = _metrics_from_rows(
        [
            {
                "objective_id": "bad_route_pressure",
                "hpwl_delta_pct": 100.0,
                "overflow_delta_pct": -40.0,
                "runtime_seconds": 10.0,
                "custom_grad_norm": 1.0,
                "outcome_label": "severe_regression",
            }
        ],
        ranking,
        config,
    )

    assert metrics["combined_score"] == 0.0
    assert metrics["hpwl_gate_passed"] is False
    assert metrics["parent_eligible"] is False
    assert metrics["negative_memory_only"] is True


def test_hpwl_safe_overflow_improver_is_parent_eligible(tmp_path):
    config = load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path)))
    ranking = SimpleNamespace(
        average_rank=1.0,
        structural_failure_count=0,
        metric_regression_count=1,
        severe_regression_count=0,
        design_count=1,
        seed_count=1,
        cell_count=1,
        metric_ranks={"hpwl_delta_pct": 2.0, "overflow_delta_pct": 1.0},
        design_ranks={"bp_fe": 1.5},
    )
    metrics = _metrics_from_rows(
        [
            {
                "objective_id": "safe_route_pressure",
                "hpwl_delta_pct": 2.0,
                "overflow_delta_pct": -5.0,
                "runtime_seconds": 10.0,
                "custom_grad_norm": 1.0,
                "outcome_label": "metric_regression",
            }
        ],
        ranking,
        config,
    )

    assert metrics["combined_score"] > 0.0
    assert metrics["hpwl_gate_passed"] is True
    assert metrics["parent_eligible"] is True
    assert metrics["negative_memory_only"] is False


def test_manuscript_selection_can_promote_timing_gain_with_bounded_placement_tradeoff(
    tmp_path,
):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        selection_policy="manuscript_multiobjective",
        max_parent_hpwl_regression_pct=10.0,
        max_parent_overflow_regression_pct=10.0,
        require_timing_for_generated_parents=True,
        require_timing_improvement_for_elite=True,
        timing_proxy_mode="gate",
        timing_proxy_wns_delta_min_abs_ns=0.000005,
        timing_proxy_wns_regression_gate_ns=0.00005,
        timing_proxy_tns_regression_min_abs_ns=0.0005,
        timing_proxy_tns_regression_pct_gate=10.0,
    )
    metrics = {
        "combined_score": 0.0,
        "constrained_score": 0.0,
        "parent_eligible": False,
        "negative_memory_only": True,
        "hpwl_gate_passed": False,
        "overflow_gate_passed": False,
        "structural_failure_count": 0,
        "routing_term_gate_passed": True,
        "hpwl_delta_pct": 2.0,
        "overflow_delta_pct": 3.0,
        "worst_design_hpwl_delta_pct": 2.0,
        "worst_design_overflow_delta_pct": 3.0,
        "timing_proxy_mode": "gate",
        "timing_proxy_status": "success",
        "timing_proxy_source_placement_budget_satisfied": True,
        "timing_proxy_wns_delta": 0.00002,
        "timing_proxy_tns_delta": 0.001,
        "timing_proxy_tns_delta_pct": 2.0,
    }

    _apply_manuscript_multimetric_selection(
        metrics,
        config,
        allow_seed_baseline=False,
    )

    assert metrics["legacy_hpwl_overflow_parent_eligible"] is False
    assert metrics["manuscript_timing_improved"] is True
    assert metrics["hpwl_gate_passed"] is True
    assert metrics["overflow_gate_passed"] is True
    assert metrics["parent_eligible"] is True
    assert metrics["elite_gate_passed"] is True
    assert metrics["combined_score"] > 0.0


def test_manuscript_selection_blocks_timing_regression_and_incomplete_timing(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        selection_policy="manuscript_multiobjective",
        require_timing_for_generated_parents=True,
        timing_proxy_mode="gate",
        timing_proxy_wns_regression_gate_ns=0.00005,
    )
    base = {
        "structural_failure_count": 0,
        "routing_term_gate_passed": True,
        "hpwl_delta_pct": -2.0,
        "overflow_delta_pct": -3.0,
        "worst_design_hpwl_delta_pct": -2.0,
        "worst_design_overflow_delta_pct": -3.0,
        "timing_proxy_mode": "gate",
        "timing_proxy_status": "success",
        "timing_proxy_source_placement_budget_satisfied": True,
        "timing_proxy_wns_delta": -0.001,
        "timing_proxy_tns_delta": 0.0,
        "timing_proxy_tns_delta_pct": 0.0,
    }
    regressed = dict(base)
    _apply_manuscript_multimetric_selection(
        regressed,
        config,
        allow_seed_baseline=False,
    )
    assert regressed["manuscript_timing_nonregressed"] is False
    assert regressed["parent_eligible"] is False

    incomplete = dict(base)
    incomplete["timing_proxy_wns_delta"] = 0.001
    incomplete["timing_proxy_source_placement_budget_satisfied"] = False
    _apply_manuscript_multimetric_selection(
        incomplete,
        config,
        allow_seed_baseline=False,
    )
    assert incomplete["manuscript_timing_ready"] is False
    assert incomplete["parent_eligible"] is False


def test_measured_routing_influence_gate_rejects_inert_correction(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        require_measured_routing_influence=True,
        min_routing_component_samples=3,
        min_routing_gradient_ratio_mean=1e-4,
        min_routing_gradient_ratio_peak=1e-3,
    )
    metrics = {
        "routing_term_gate_passed": True,
        "routing_aware_tier2_candidate": True,
        "parent_eligible": True,
        "elite_gate_passed": True,
        "component_summary": {
            "routing_correction": {
                "grad_ratio_count": 5,
                "grad_ratio_mean": 1e-10,
                "grad_ratio_max": 1e-8,
            }
        },
    }

    _apply_measured_routing_influence_gate(
        metrics,
        config,
        allow_seed_baseline=False,
    )

    assert metrics["measured_routing_influence_gate_passed"] is False
    assert metrics["routing_term_gate_passed"] is False
    assert metrics["parent_eligible"] is False
    assert metrics["elite_gate_passed"] is False
    assert metrics["negative_memory_only"] is True


def test_measured_routing_influence_gate_accepts_material_correction(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        require_measured_routing_influence=True,
        min_routing_component_samples=3,
        min_routing_gradient_ratio_mean=1e-4,
        min_routing_gradient_ratio_peak=1e-3,
    )
    metrics = {
        "routing_term_gate_passed": True,
        "component_summary": {
            "routing_correction": {
                "grad_ratio_count": 8,
                "grad_ratio_mean": 0.002,
                "grad_ratio_max": 0.01,
            }
        },
    }

    _apply_measured_routing_influence_gate(
        metrics,
        config,
        allow_seed_baseline=False,
    )

    assert metrics["measured_routing_influence_gate_passed"] is True
    assert metrics["routing_term_gate_passed"] is True


def test_final_def_metrics_replace_mixed_stage_selection_values(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        primary_baseline="default",
        timing_proxy_enabled=True,
        hpwl_gate_search_pct=5.0,
        hpwl_gate_elite_pct=3.0,
        per_design_hpwl_gate_search_pct=5.0,
        per_design_overflow_gate_search_pct=5.0,
        min_design_both_improvement_fraction=0.0,
    )
    metrics = {
        "tier2_average_rank": 2.0,
        "structural_failure_count": 0,
        "hpwl_delta_pct": -8.2,
        "overflow_delta_pct": 161.0,
        "per_design_deltas": [
            {
                "design": "mor1kx",
                "hpwl_delta_pct": -8.2,
                "overflow_delta_pct": 161.0,
            }
        ],
        "final_def_hpwl_delta_pct": -8.0,
        "final_def_overflow_delta_pct": -4.0,
        "final_def_per_design_deltas": [
            {
                "design": "mor1kx",
                "hpwl_delta_pct": -8.0,
                "overflow_delta_pct": -4.0,
                "cell_count": 1,
            }
        ],
        "timing_proxy_source_placement_budget_satisfied": True,
    }

    _apply_final_def_selection_metrics(metrics, config)

    assert metrics["optimizer_log_overflow_delta_pct"] == pytest.approx(161.0)
    assert metrics["hpwl_delta_pct"] == pytest.approx(-8.0)
    assert metrics["overflow_delta_pct"] == pytest.approx(-4.0)
    assert metrics["native_default_hpwl_delta_pct"] == pytest.approx(-8.0)
    assert metrics["native_default_overflow_delta_pct"] == pytest.approx(-4.0)
    assert metrics["selection_metric_stage"] == "final_def_post_detailed_placement"
    assert metrics["final_def_selection_metrics_applied"] is True
    assert metrics["parent_eligible"] is True
    assert metrics["combined_score"] > 0.0


def test_final_def_metrics_require_full_source_budget(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        primary_baseline="default",
        timing_proxy_enabled=True,
    )
    metrics = {
        "hpwl_delta_pct": -1.0,
        "overflow_delta_pct": -1.0,
        "final_def_hpwl_delta_pct": -2.0,
        "final_def_overflow_delta_pct": -2.0,
        "final_def_per_design_deltas": [],
        "timing_proxy_source_placement_budget_satisfied": False,
    }

    _apply_final_def_selection_metrics(metrics, config)

    assert metrics["hpwl_delta_pct"] == pytest.approx(-1.0)
    assert metrics["overflow_delta_pct"] == pytest.approx(-1.0)
    assert metrics["final_def_selection_metrics_applied"] is False
    assert "iteration budget" in metrics["final_def_selection_metrics_reason"]


def test_metrics_from_rows_aggregates_component_feedback(tmp_path):
    config = load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path)))
    ranking = SimpleNamespace(
        average_rank=1.0,
        structural_failure_count=0,
        metric_regression_count=0,
        severe_regression_count=0,
        design_count=1,
        seed_count=1,
        cell_count=1,
        metric_ranks=[],
        design_ranks=[],
    )
    rows = [
        {
            "objective_id": "candidate",
            "design": "bp_fe",
            "seed": 1000,
            "hpwl_delta_pct": "1.0",
            "overflow_delta_pct": "-2.0",
            "native_default_hpwl_delta_pct": "1.0",
            "native_default_overflow_delta_pct": "-2.0",
            "custom_default_hpwl_delta_pct": "1.0",
            "custom_default_overflow_delta_pct": "-2.0",
            "runtime_seconds": "5.0",
            "custom_grad_norm": "10.0",
            "status": "success",
            "component_summary_json": json.dumps(
                {
                    "route": {
                        "start": 0.1,
                        "mid": 0.2,
                        "end": 0.3,
                        "mean": 0.2,
                        "trend": "up",
                        "flat_or_saturated": False,
                    }
                }
            ),
        }
    ]

    metrics = _metrics_from_rows(rows, ranking, config)

    assert metrics["component_summary"]["route"]["mean"] == pytest.approx(0.2)
    assert metrics["component_summary"]["route"]["trend"] == "up"
    assert metrics["component_feedback"][0].startswith("route: trend=up")


def test_nonzero_routing_gate_blocks_density_only_generated_candidate(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        require_nonzero_routing_term=True,
        min_abs_routing_coeff=1e-18,
    )
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength")',
                '    den = term("density")',
                "    score = wl + 1.35 * den",
                '    return score, {"wirelength": wl, "density": den}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace",
    )
    metrics = {
        "combined_score": 0.5,
        "constrained_score": 0.5,
        "structural_failure_count": 0,
        "hpwl_gate_passed": True,
        "overflow_gate_passed": True,
        "robust_gate_passed": True,
        "elite_gate_passed": True,
        "parent_eligible": True,
        "negative_memory_only": False,
    }

    _apply_routing_gate(metrics, spec, config, allow_nonrouting_parent=False)

    assert metrics["routing_term_required"] is True
    assert metrics["routing_term_gate_passed"] is False
    assert metrics["routing_aware_tier2_candidate"] is False
    assert metrics["combined_score"] == 0.0
    assert metrics["parent_eligible"] is False
    assert metrics["negative_memory_only"] is True


def test_static_validation_rejects_baseline_like_generated_candidate_before_dreamplace(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        require_nonzero_routing_term=True,
        min_abs_routing_coeff=1e-6,
    )
    database = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength")',
                '    den = term("density")',
                "    score = wl + den",
                '    return score, {"wirelength": wl, "density": den}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace",
    )

    reason = _static_rejection_reason(spec, database, config)

    assert reason is not None
    assert "requires at least one nonzero deployable routing mechanism" in reason


def test_static_validation_rejects_simple_baseline_plus_one_route_term(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        objective_mode="replacement",
        require_nonzero_routing_term=True,
        min_abs_routing_coeff=1e-12,
        route_coeff_cap=0.1,
        min_replacement_nonbaseline_terms=2,
    )
    database = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    simple_spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength")',
                '    den = term("density")',
                '    route = term("soft_rudy_pnorm")',
                "    score = wl + den + 1e-8 * log1p(route)",
                '    return score, {"wirelength": wl, "density": den, "route": route}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )
    rich_spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength")',
                '    den = term("density")',
                '    route = term("soft_rudy_pnorm")',
                '    pins = term("pin_density_pnorm")',
                "    score = wl + den + 1e-8 * log1p(route) + 1e-8 * sqrt(pins)",
                (
                    '    return score, {"wirelength": wl, "density": den, '
                    '"route": route, "pins": pins}'
                ),
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )

    simple_reason = _static_rejection_reason(simple_spec, database, config)
    rich_reason = _static_rejection_reason(rich_spec, database, config)

    assert simple_reason is not None
    assert "too close to the explicit wirelength+density baseline" in simple_reason
    assert rich_reason is None


def test_static_validation_rejects_baseline_mechanism_clone(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        reject_baseline_mechanism_clones=True,
        require_nonzero_routing_term=False,
    )
    database = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    baseline = objective_preset("dreamplace_density_135")
    database.add(
        ObjectiveProgram.from_spec(
            baseline,
            program_id="seed_density_135",
            metrics={
                "seed_baseline": True,
                "mechanism_signature": _mechanism_signature(baseline.ast),
            },
            status="accepted",
        )
    )
    candidate = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength")',
                '    den = term("density")',
                "    score = wl + 1.38 * den",
                '    return score, {"wirelength": wl, "density": den}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )

    reason = _static_rejection_reason(candidate, database, config)

    assert reason is not None
    assert "repeats a fixed baseline mechanism structure" in reason


def test_static_validation_rejects_near_baseline_mechanism_clone(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        objective_mode="replacement",
        require_nonzero_routing_term=True,
        min_abs_routing_coeff=1e-5,
        route_coeff_cap=0.1,
        min_replacement_nonbaseline_terms=2,
        reject_baseline_mechanism_clones=True,
        min_baseline_mechanism_distance=0.35,
    )
    database = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    baseline = objective_preset("dreamplace_routing_aware_smoke")
    database.add(
        ObjectiveProgram.from_spec(
            baseline,
            program_id="seed_route_smoke",
            metrics={
                "seed_baseline": True,
                "mechanism_signature": _mechanism_signature(baseline.ast),
            },
            status="accepted",
        )
    )
    near_clone = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength_wawl")',
                '    den = term("density_electric")',
                '    route = term("soft_rudy_pnorm")',
                '    long_route = term("route_pressure_long")',
                "    score = wl + den + 0.0001 * sqrt(route) + 0.0001 * long_route",
                (
                    '    return score, {"wirelength": wl, "density": den, '
                    '"route": route, "long_route": long_route}'
                ),
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )
    novel = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength_lse")',
                '    den = term("density_bell")',
                '    route = term("route_pressure_long")',
                '    pins = term("pin_density_pnorm")',
                "    access_pressure = log1p(route * pins)",
                "    score = wl + den + 0.0001 * access_pressure",
                (
                    '    return score, {"wirelength": wl, "density": den, '
                    '"route": route, "pins": pins}'
                ),
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )

    near_distance = _mechanism_distance(near_clone.ast, baseline.ast)
    novel_distance = _mechanism_distance(novel.ast, baseline.ast)
    near_reason = _static_rejection_reason(near_clone, database, config)
    novel_reason = _static_rejection_reason(novel, database, config)

    assert near_distance < config.min_baseline_mechanism_distance
    assert novel_distance > config.min_baseline_mechanism_distance
    assert near_reason is not None
    assert "too close to a fixed baseline mechanism" in near_reason
    assert novel_reason is None


def test_static_validation_allows_compact_route_only_interaction(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        objective_mode="replacement",
        require_nonzero_routing_term=True,
        min_abs_routing_coeff=1e-12,
        route_coeff_cap=0.1,
        min_replacement_nonbaseline_terms=2,
    )
    database = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength_lse")',
                '    den = term("density_bell")',
                '    route = term("route_pressure_long")',
                '    pins = term("pin_density_pnorm")',
                "    access_pressure = log1p(route * pins)",
                "    score = wl + den + 1e-8 * access_pressure",
                (
                    '    return score, {"wirelength": wl, "density": den, '
                    '"route": route, "pins": pins}'
                ),
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )

    reason = _static_rejection_reason(spec, database, config)
    coeffs, unsupported = _route_coefficients(spec.ast)

    assert reason is None
    assert unsupported == set()
    assert ("route_pressure_long", 1e-8) in coeffs
    assert ("pin_density_pnorm", 1e-8) in coeffs


def test_static_validation_rejects_route_density_interaction(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        objective_mode="replacement",
        require_nonzero_routing_term=True,
        min_abs_routing_coeff=1e-12,
        route_coeff_cap=0.1,
        min_replacement_nonbaseline_terms=2,
    )
    database = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength_lse")',
                '    den = term("density_bell")',
                '    route = term("route_pressure_long")',
                "    mixed_pressure = log1p(route * den)",
                "    score = wl + den + 1e-8 * mixed_pressure",
                (
                    '    return score, {"wirelength": wl, "density": den, '
                    '"route": route}'
                ),
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )

    reason = _static_rejection_reason(spec, database, config)
    coeffs, unsupported = _route_coefficients(spec.ast)

    assert reason is not None
    assert "unsupported route expression" in reason
    assert coeffs == []
    assert unsupported == {"route_pressure_long"}


def test_retry_feedback_is_appended_to_actual_provider_messages(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        objective_mode="replacement",
        min_replacement_nonbaseline_terms=2,
        route_coeff_cap=0.01,
    )
    context = {
        "prompt_messages": [
            {
                "role": "system",
                "content": "system",
            },
            {
                "role": "user",
                "content": "# Current Program\nbase prompt",
            },
        ]
    }
    rejection_history = [
        {
            "attempt": 1,
            "objective_id": "obj_bad",
            "term_set": ["wirelength", "density", "route_pressure_long"],
            "reason": (
                "replacement objective is too close to the explicit "
                "wirelength+density baseline; use at least 2 non-baseline "
                "observable families"
            ),
        },
        {
            "attempt": 2,
            "objective_id": "obj_repeat",
            "term_set": ["wirelength", "density", "route_pressure_long", "pin_density_pnorm"],
            "reason": (
                "candidate repeats a mechanism structure that is already negative "
                "memory from iter_0001_bad"
            ),
        },
        {
            "attempt": 3,
            "objective_id": "obj_big_coeff",
            "term_set": ["wirelength_lse", "density_bell", "soft_rudy_pnorm"],
            "reason": "replacement route coefficient exceeds cap 0.01",
        },
    ]

    retry_feedback = _retry_feedback_payload(
        attempt=4,
        rejection_history=rejection_history,
        config=config,
    )
    _append_retry_feedback_message(context, retry_feedback)
    messages = generation_messages(
        context,
        term_scope="dreamplace_replacement",
        mutation_mode="diff",
    )
    combined = "\n".join(message["content"] for message in messages)

    assert len(messages) == 3
    assert "Static Validation Feedback For Retry" in combined
    assert "baseline_like_objective" in combined
    assert "coefficient_scale_out_of_bounds" in combined
    assert "Use at least 2 non-default observable families" not in combined
    assert "replacement route coefficient exceeds cap" not in combined
    assert "cap" not in combined.lower()
    assert "abs(c)" not in combined
    assert "0.01" not in combined
    assert "Use a materially different objective mechanism" in combined
    assert "Use a gentler routing-term scale" in combined
    assert "Change the mechanism structure" in combined
    assert "HPWL must not regress" not in combined
    assert audit_prompt_messages(
        messages,
        term_scope="dreamplace_replacement",
        objective_mode="replacement",
    )["passed"]


def test_prompt_audit_accepts_clean_openevolve_messages(tmp_path):
    context = {
        "prompt_messages": [
            {
                "role": "system",
                "content": "You are evolving DREAMPlace objectives.",
            },
            {
                "role": "user",
                "content": (
                    "Use explicit placement observables including wirelength_lse, "
                    "density_bell, route_pressure_long, and pin_density_pnorm. "
                    "Selection uses measured DREAMPlace behavior after execution."
                ),
            },
        ]
    }
    messages = generation_messages(
        context,
        term_scope="dreamplace_replacement",
        mutation_mode="diff",
    )

    audit = audit_prompt_messages(
        messages,
        term_scope="dreamplace_replacement",
        objective_mode="replacement",
    )

    assert audit["passed"] is True
    assert audit["violations"] == []


def test_prompt_audit_rejects_hidden_gates_and_native_shortcuts():
    messages = [
        {
            "role": "user",
            "content": (
                "Use native_objective and obey hpwl_gate_search_pct. "
                "HPWL must not regress."
            ),
        }
    ]

    audit = audit_prompt_messages(
        messages,
        term_scope="dreamplace_replacement",
        objective_mode="replacement",
    )
    patterns = {item["pattern"] for item in audit["violations"]}

    assert audit["passed"] is False
    assert "native_objective" in patterns
    assert "hpwl_gate_search_pct" in patterns
    assert "HPWL must not regress" in patterns


def test_openevolve_prompt_artifact_audit_writes_clean_summary(tmp_path):
    run_dir = tmp_path / "run"
    attempt_dir = run_dir / "iteration_0001" / "attempt_01"
    attempt_dir.mkdir(parents=True)
    messages = [
        {"role": "system", "content": "Evolve placement objectives."},
        {
            "role": "user",
            "content": (
                "Use wirelength_lse, density_bell, route_pressure_long, and "
                "pin_density_pnorm. The evaluator measures placement behavior."
            ),
        },
    ]
    (attempt_dir / "prompt_messages.json").write_text(
        json.dumps(messages),
        encoding="utf-8",
    )
    (attempt_dir / "prompt_context.json").write_text(
        json.dumps(
            {
                "term_scope": "dreamplace_replacement",
                "objective_mode": "replacement",
            }
        ),
        encoding="utf-8",
    )

    summary = audit_openevolve_prompt_artifacts(run_dir)

    assert summary["passed"] is True
    assert summary["prompt_file_count"] == 1
    assert summary["failed_prompt_count"] == 0
    assert (attempt_dir / "prompt_audit.json").exists()
    assert (run_dir / "prompt_audit_summary.json").exists()


def test_openevolve_prompt_artifact_audit_detects_contaminated_prompt(tmp_path):
    run_dir = tmp_path / "run"
    attempt_dir = run_dir / "iteration_0001" / "attempt_01"
    attempt_dir.mkdir(parents=True)
    messages = [
        {
            "role": "user",
            "content": (
                "Use native_objective and route_coeff_grid. HPWL must not regress."
            ),
        }
    ]
    (attempt_dir / "prompt_messages.json").write_text(
        json.dumps(messages),
        encoding="utf-8",
    )
    (attempt_dir / "prompt_context.json").write_text(
        json.dumps(
            {
                "term_scope": "dreamplace_replacement",
                "objective_mode": "replacement",
            }
        ),
        encoding="utf-8",
    )

    summary = audit_openevolve_prompt_artifacts(run_dir)
    patterns = {
        violation["pattern"]
        for result in summary["results"]
        for violation in result["violations"]
    }

    assert summary["passed"] is False
    assert summary["failed_prompt_count"] == 1
    assert "native_objective" in patterns
    assert "route_coeff_grid" in patterns
    assert "HPWL must not regress" in patterns


def test_static_validation_rejects_repeated_negative_mechanism_with_new_coefficients(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        objective_mode="replacement",
        require_nonzero_routing_term=True,
        min_abs_routing_coeff=1e-6,
        route_coeff_cap=0.1,
        reject_repeated_negative_mechanisms=True,
    )
    database = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    failed_spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength")',
                '    den = term("density")',
                '    route = term("route_pressure_long")',
                '    pins = term("pin_density_pnorm")',
                "    score = wl + den + 0.005 * log1p(route) + 0.001 * sqrt(pins)",
                '    return score, {"wirelength": wl, "density": den, "route": route, "pins": pins}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace",
    )
    failed_program = ObjectiveProgram.from_spec(
        failed_spec,
        program_id="failed_route_pressure",
        metrics={
            "negative_memory_only": True,
            "severe_regression_count": 2,
            "mechanism_signature": _mechanism_signature(failed_spec.ast),
        },
        status="accepted",
    )
    database.add(failed_program)
    retry_spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength")',
                '    den = term("density")',
                '    route = term("route_pressure_long")',
                '    pins = term("pin_density_pnorm")',
                "    score = wl + den + 0.01 * log1p(route) + 0.005 * sqrt(pins)",
                '    return score, {"wirelength": wl, "density": den, "route": route, "pins": pins}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace",
    )

    reason = _static_rejection_reason(retry_spec, database, config)

    assert reason is not None
    assert "repeats a mechanism structure" in reason


def test_static_validation_rejects_near_negative_mechanism_repeat(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        objective_mode="replacement",
        require_nonzero_routing_term=True,
        min_abs_routing_coeff=1e-8,
        route_coeff_cap=0.1,
        reject_repeated_negative_mechanisms=True,
        min_negative_mechanism_distance=0.6,
    )
    database = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    failed_spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength_wawl")',
                '    den = term("density_electric")',
                '    route = term("route_pressure_long")',
                '    pins = term("pin_density_pnorm")',
                "    score = wl + den + 1e-7 * sigmoid(route * pins)",
                '    return score, {"wirelength": wl, "density": den, "route": route, "pins": pins}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )
    failed_program = ObjectiveProgram.from_spec(
        failed_spec,
        program_id="failed_route_pin_product",
        metrics={
            "negative_memory_only": True,
            "severe_regression_count": 1,
            "mechanism_signature": _mechanism_signature(failed_spec.ast),
        },
        status="accepted",
    )
    database.add(failed_program)
    retry_spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength_wawl")',
                '    den = term("density_electric")',
                '    route = term("route_pressure_long")',
                '    pins = term("pin_density_pnorm")',
                "    score = wl + den + 3e-7 * softplus(route + pins)",
                '    return score, {"wirelength": wl, "density": den, "route": route, "pins": pins}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )

    assert _mechanism_signature(failed_spec.ast) != _mechanism_signature(retry_spec.ast)
    assert _mechanism_distance(failed_spec.ast, retry_spec.ast) < config.min_negative_mechanism_distance

    reason = _static_rejection_reason(retry_spec, database, config)

    assert reason is not None
    assert "too close to a measured negative" in reason


def test_iteration_mutation_mode_switches_to_full_after_static_rejection_streak(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        mutation_mode="diff",
        full_rewrite_after_static_rejections=2,
    )
    database = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    parent = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_wawl_electric"),
        program_id="safe_parent",
        metrics={"combined_score": 1.0, "parent_eligible": True},
        status="accepted",
    )
    database.add(parent)
    for index in range(2):
        rejected = ObjectiveProgram.from_spec(
            objective_preset("dreamplace_routing_aware_smoke"),
            program_id=f"rejected_mechanism_{index}",
            metrics={
                "combined_score": 0.0,
                "parent_eligible": False,
                "negative_memory_only": True,
                "feedback_lesson": (
                    "Static rejection before DREAMPlace: candidate is too "
                    "close to a measured negative or near-miss mechanism"
                ),
            },
            failure_reason=(
                "candidate is too close to a measured negative or near-miss "
                "mechanism"
            ),
            status="rejected",
        )
        database.add(rejected)

    assert _iteration_mutation_mode(config, database) == "full"

    relaxed = replace(config, full_rewrite_after_static_rejections=3)
    assert _iteration_mutation_mode(relaxed, database) == "diff"


def test_iteration_parent_escapes_nonrobust_mechanism_after_static_rejections(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        full_rewrite_after_static_rejections=2,
        escape_parent_after_static_rejections=2,
        min_negative_mechanism_distance=0.6,
        require_robust_parent_gate=False,
    )
    database = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    safe_baseline = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_wawl_electric"),
        program_id="safe_baseline",
        metrics={
            "combined_score": 0.1,
            "parent_eligible": True,
            "seed_baseline": True,
            "hpwl_gate_passed": True,
            "overflow_gate_passed": True,
        },
        status="accepted",
    )
    route_spec = objective_preset("dreamplace_routing_aware_smoke")
    nonrobust_parent = ObjectiveProgram.from_spec(
        route_spec,
        program_id="nonrobust_generated_parent",
        parent_id=safe_baseline.id,
        metrics={
            "combined_score": 2.0,
            "parent_eligible": True,
            "hpwl_gate_passed": True,
            "overflow_gate_passed": True,
            "near_miss_parent_due_to_robustness": True,
            "mechanism_signature": _mechanism_signature(route_spec.ast),
        },
        status="accepted",
    )
    database.add(safe_baseline)
    database.add(nonrobust_parent)
    for index in range(2):
        rejected = ObjectiveProgram.from_spec(
            route_spec,
            program_id=f"near_negative_{index}",
            metrics={
                "combined_score": 0.0,
                "parent_eligible": False,
                "negative_memory_only": True,
                "mechanism_signature": _mechanism_signature(route_spec.ast),
                "feedback_lesson": (
                    "Static rejection before DREAMPlace: candidate is too close "
                    "to a measured negative or near-miss mechanism"
                ),
            },
            failure_reason="candidate is too close to a measured negative or near-miss mechanism",
            status="rejected",
        )
        database.add(rejected)

    selected = _sample_iteration_parent(
        config=config,
        database=database,
        rng=random.Random(7),
        island_id=0,
    )

    assert nonrobust_parent.is_parent_eligible is True
    assert selected.id == "safe_baseline"


def test_iteration_parent_prefers_different_discovered_mechanism_during_escape(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        full_rewrite_after_static_rejections=2,
        escape_parent_after_static_rejections=2,
        min_negative_mechanism_distance=0.6,
        require_robust_parent_gate=False,
    )
    database = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    baseline = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_wawl_electric"),
        program_id="safe_baseline",
        metrics={
            "combined_score": 0.1,
            "parent_eligible": True,
            "seed_baseline": True,
            "hpwl_gate_passed": True,
            "overflow_gate_passed": True,
        },
        status="accepted",
    )
    different_spec = objective_preset("dreamplace_lse_density_bell")
    different_parent = ObjectiveProgram.from_spec(
        different_spec,
        program_id="different_generated_parent",
        parent_id=baseline.id,
        metrics={
            "combined_score": 0.6,
            "parent_eligible": True,
            "hpwl_gate_passed": True,
            "overflow_gate_passed": True,
            "mechanism_signature": _mechanism_signature(different_spec.ast),
        },
        status="accepted",
    )
    route_spec = objective_preset("dreamplace_routing_aware_smoke")
    database.add(baseline)
    database.add(different_parent)
    for index in range(2):
        database.add(
            ObjectiveProgram.from_spec(
                route_spec,
                program_id=f"near_negative_{index}",
                metrics={
                    "combined_score": 0.0,
                    "parent_eligible": False,
                    "negative_memory_only": True,
                    "mechanism_signature": _mechanism_signature(route_spec.ast),
                    "feedback_lesson": (
                        "Static rejection before DREAMPlace: candidate is too "
                        "close to a measured negative or near-miss mechanism"
                    ),
                },
                failure_reason=(
                    "candidate is too close to a measured negative or near-miss "
                    "mechanism"
                ),
                status="rejected",
            )
        )

    selected = _sample_iteration_parent(
        config=config,
        database=database,
        rng=random.Random(11),
        island_id=0,
    )

    assert selected.id == "different_generated_parent"


def test_robust_best_is_separate_from_aggregate_best(tmp_path):
    database = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    aggregate_spec = objective_preset("dreamplace_routing_aware_smoke")
    aggregate = ObjectiveProgram.from_spec(
        aggregate_spec,
        program_id="aggregate_nonrobust",
        metrics={
            "combined_score": 0.7,
            "parent_eligible": False,
            "negative_memory_only": True,
            "hpwl_gate_passed": True,
            "overflow_gate_passed": True,
            "robust_gate_passed": False,
            "severe_regression_count": 1,
            "structural_failure_count": 0,
            "beats_baseline_portfolio": True,
            "native_default_comparison_label": "dominates_native_default",
            "tier2_average_rank": 1.0,
            "native_default_hpwl_delta_pct": -4.0,
            "native_default_overflow_delta_pct": -60.0,
            "worst_design_hpwl_delta_pct": 65.0,
            "routing_aware_tier2_candidate": True,
        },
        status="accepted",
    )
    robust_spec = objective_preset("dreamplace_route_pressure_long")
    robust = ObjectiveProgram.from_spec(
        robust_spec,
        program_id="robust_candidate",
        metrics={
            "combined_score": 0.2,
            "parent_eligible": True,
            "negative_memory_only": False,
            "hpwl_gate_passed": True,
            "overflow_gate_passed": True,
            "robust_gate_passed": True,
            "severe_regression_count": 0,
            "structural_failure_count": 0,
            "beats_baseline_portfolio": False,
            "native_default_comparison_label": "dominates_native_default",
            "tier2_average_rank": 3.0,
            "native_default_hpwl_delta_pct": -1.0,
            "native_default_overflow_delta_pct": -2.0,
            "worst_design_hpwl_delta_pct": -0.5,
            "worst_design_overflow_delta_pct": -0.2,
            "routing_aware_tier2_candidate": True,
        },
        status="accepted",
    )
    portfolio_losing_near_miss = ObjectiveProgram.from_spec(
        robust_spec,
        program_id="portfolio_losing_near_miss",
        metrics={
            "combined_score": 0.9,
            "parent_eligible": True,
            "negative_memory_only": False,
            "exploration_parent_only": True,
            "hpwl_gate_passed": True,
            "overflow_gate_passed": True,
            "robust_gate_passed": True,
            "severe_regression_count": 0,
            "structural_failure_count": 0,
            "baseline_portfolio_compared": True,
            "beats_baseline_portfolio": False,
            "native_default_comparison_label": "dominates_native_default",
            "tier2_average_rank": 1.0,
            "native_default_hpwl_delta_pct": -2.0,
            "native_default_overflow_delta_pct": -2.0,
            "worst_design_hpwl_delta_pct": -0.5,
            "worst_design_overflow_delta_pct": -0.2,
            "routing_aware_tier2_candidate": True,
        },
        status="accepted",
    )
    rank_only_tradeoff = ObjectiveProgram.from_spec(
        robust_spec,
        program_id="rank_only_tradeoff",
        metrics={
            "combined_score": 1.2,
            "parent_eligible": True,
            "negative_memory_only": False,
            "hpwl_gate_passed": True,
            "overflow_gate_passed": True,
            "robust_gate_passed": True,
            "severe_regression_count": 0,
            "structural_failure_count": 0,
            "baseline_portfolio_compared": True,
            "beats_baseline_portfolio": True,
            "baseline_portfolio_pareto_dominates_best": False,
            "baseline_portfolio_pareto_label_vs_best": "baseline_tradeoff_overflow_for_hpwl",
            "native_default_comparison_label": "dominates_native_default",
            "tier2_average_rank": 0.5,
            "native_default_hpwl_delta_pct": -3.0,
            "native_default_overflow_delta_pct": -3.0,
            "worst_design_hpwl_delta_pct": -0.5,
            "worst_design_overflow_delta_pct": -0.2,
            "routing_aware_tier2_candidate": True,
        },
        status="accepted",
    )
    pareto_too_small = ObjectiveProgram.from_spec(
        robust_spec,
        program_id="pareto_too_small",
        metrics={
            "combined_score": 0.0,
            "parent_eligible": False,
            "negative_memory_only": True,
            "hpwl_gate_passed": True,
            "overflow_gate_passed": True,
            "robust_gate_passed": True,
            "severe_regression_count": 0,
            "structural_failure_count": 0,
            "baseline_portfolio_compared": True,
            "beats_baseline_portfolio": False,
            "baseline_portfolio_pareto_dominates_best": True,
            "baseline_portfolio_effect_pct": 0.11,
            "native_default_comparison_label": "dominates_native_default",
            "tier2_average_rank": 2.0,
            "native_default_hpwl_delta_pct": -1.2,
            "native_default_overflow_delta_pct": -2.1,
            "worst_design_hpwl_delta_pct": -0.4,
            "worst_design_overflow_delta_pct": -0.1,
            "routing_aware_tier2_candidate": True,
        },
        status="accepted",
    )
    promotion_pareto = ObjectiveProgram.from_spec(
        robust_spec,
        program_id="promotion_pareto",
        metrics={
            "combined_score": 0.4,
            "parent_eligible": True,
            "negative_memory_only": False,
            "hpwl_gate_passed": True,
            "overflow_gate_passed": True,
            "robust_gate_passed": True,
            "severe_regression_count": 0,
            "structural_failure_count": 0,
            "baseline_portfolio_compared": True,
            "beats_baseline_portfolio": True,
            "baseline_portfolio_pareto_dominates_best": True,
            "baseline_portfolio_effect_pct": 0.31,
            "native_default_comparison_label": "dominates_native_default",
            "tier2_average_rank": 1.5,
            "native_default_hpwl_delta_pct": -1.3,
            "native_default_overflow_delta_pct": -2.2,
            "worst_design_hpwl_delta_pct": -0.5,
            "worst_design_overflow_delta_pct": -0.2,
            "routing_aware_tier2_candidate": True,
        },
        status="accepted",
    )
    database.add(aggregate)
    database.add(robust)
    database.add(portfolio_losing_near_miss)
    database.add(rank_only_tradeoff)
    database.add(pareto_too_small)
    database.add(promotion_pareto)

    assert _select_best_robust_generated_program(database).id == "promotion_pareto"
    assert _select_best_aggregate_generated_program(database).id == "promotion_pareto"
    assert _select_best_pareto_generated_program(database).id == "promotion_pareto"


def test_retry_feedback_names_near_negative_and_complexity_repairs(tmp_path):
    config = load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path)))
    instructions = _retry_repair_instructions(
        [
            {
                "reason": (
                    "candidate is too close to a measured negative or near-miss "
                    "mechanism from previous candidate"
                )
            },
            {"reason": "ObjectiveSpecError: AST depth exceeds 5"},
            {
                "reason": (
                    "replacement objective must include a density/utilization "
                    "anchor (density, density_electric, or density_bell)"
                )
            },
            {
                "reason": (
                    "replacement objective must include a canonical wirelength "
                    "anchor (wirelength, wirelength_wawl, or wirelength_lse)"
                )
            },
        ],
        config,
    )
    text = "\n".join(instructions)

    assert "observable family or nonlinear composition" in text
    assert "coefficient-only" in text
    assert "shallow sum" in text
    assert "Avoid nested helper calls" in text
    assert "one density/utilization anchor" in text
    assert "one wirelength-family anchor" in text

    payload = _retry_feedback_payload(
        attempt=3,
        rejection_history=[
            {
                "attempt": 1,
                "objective_id": "obj_bad",
                "term_set": ["wirelength_wawl", "density_electric"],
                "reason": (
                    "candidate is too close to a measured negative or near-miss "
                    "mechanism"
                ),
            }
        ],
        config=config,
    )

    assert payload["previous_rejections"][0]["reason_category"] == "near_negative_mechanism"


def test_nonzero_routing_gate_allows_tiny_active_route_term(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        require_nonzero_routing_term=True,
        min_abs_routing_coeff=1e-18,
    )
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength")',
                '    den = term("density")',
                '    route = term("soft_rudy_pnorm")',
                "    score = wl + 1.35 * den + 1e-12 * route",
                '    return score, {"wirelength": wl, "density": den, "route": route}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace",
    )
    metrics = {
        "combined_score": 0.5,
        "constrained_score": 0.5,
        "structural_failure_count": 0,
        "hpwl_gate_passed": True,
        "overflow_gate_passed": True,
        "robust_gate_passed": True,
        "elite_gate_passed": True,
        "parent_eligible": True,
        "negative_memory_only": False,
    }

    _apply_routing_gate(metrics, spec, config, allow_nonrouting_parent=False)

    assert metrics["routing_term_gate_passed"] is True
    assert metrics["routing_aware_tier2_candidate"] is True
    assert metrics["active_routing_terms"] == ["soft_rudy_pnorm"]
    assert metrics["parent_eligible"] is True


def test_custom_default_gate_blocks_baseline_like_routing_candidate(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        require_nonzero_routing_term=True,
        min_abs_routing_coeff=1e-12,
        require_custom_default_gate=True,
        custom_default_hpwl_gate_pct=5.0,
        custom_default_overflow_gate_pct=0.0,
        min_custom_default_effect_pct=0.25,
    )
    ranking = SimpleNamespace(
        average_rank=1.0,
        structural_failure_count=0,
        metric_regression_count=0,
        severe_regression_count=0,
        design_count=1,
        seed_count=1,
        cell_count=1,
        metric_ranks={"hpwl_delta_pct": 1.0, "overflow_delta_pct": 1.0},
        design_ranks={"bp_fe": 1.0},
    )
    metrics = _metrics_from_rows(
        [
            {
                "design": "bp_fe",
                "objective_id": "candidate",
                "hpwl_delta_pct": -20.0,
                "overflow_delta_pct": -60.0,
                "custom_default_hpwl_delta_pct": 0.1,
                "custom_default_overflow_delta_pct": 0.5,
                "runtime_seconds": 10.0,
                "custom_grad_norm": 1.0,
                "outcome_label": "success_or_improvement",
            }
        ],
        ranking,
        config,
    )
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength")',
                '    den = term("density")',
                '    route = term("soft_rudy_pnorm")',
                "    score = wl + den + 1e-8 * route",
                '    return score, {"wirelength": wl, "density": den, "route": route}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )

    _apply_routing_gate(metrics, spec, config, allow_nonrouting_parent=False)

    assert metrics["custom_default_gate_passed"] is False
    assert metrics["parent_eligible"] is False
    assert metrics["negative_memory_only"] is True
    assert "custom_default" in metrics["feedback_lesson"]


def test_custom_default_gate_allows_measurable_nondefault_candidate(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        require_nonzero_routing_term=True,
        min_abs_routing_coeff=1e-12,
        require_custom_default_gate=True,
        custom_default_hpwl_gate_pct=5.0,
        custom_default_overflow_gate_pct=0.0,
        min_custom_default_effect_pct=0.25,
    )
    ranking = SimpleNamespace(
        average_rank=1.0,
        structural_failure_count=0,
        metric_regression_count=0,
        severe_regression_count=0,
        design_count=1,
        seed_count=1,
        cell_count=1,
        metric_ranks={"hpwl_delta_pct": 1.0, "overflow_delta_pct": 1.0},
        design_ranks={"bp_fe": 1.0},
    )
    metrics = _metrics_from_rows(
        [
            {
                "design": "bp_fe",
                "objective_id": "candidate",
                "hpwl_delta_pct": -20.0,
                "overflow_delta_pct": -60.0,
                "custom_default_hpwl_delta_pct": -0.1,
                "custom_default_overflow_delta_pct": -0.8,
                "runtime_seconds": 10.0,
                "custom_grad_norm": 1.0,
                "outcome_label": "success_or_improvement",
            }
        ],
        ranking,
        config,
    )
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength")',
                '    den = term("density")',
                '    route = term("soft_rudy_pnorm")',
                "    score = wl + den + 1e-8 * route",
                '    return score, {"wirelength": wl, "density": den, "route": route}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )

    _apply_routing_gate(metrics, spec, config, allow_nonrouting_parent=False)

    assert metrics["custom_default_gate_passed"] is True
    assert metrics["parent_eligible"] is True
    assert metrics["negative_memory_only"] is False


def test_native_residual_rejects_fixed_density_tuning(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        objective_mode="native_residual",
        route_coeff_cap=0.03,
    )
    database = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    native = term("native_objective")',
                '    wl = term("wirelength")',
                '    den = term("density")',
                '    route = term("soft_rudy_mean")',
                "    score = native + 0.01 * wl + 1.35 * den + 0.003 * route",
                '    return score, {"native": native, "wirelength": wl, "density": den, "route": route}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace",
    )

    reason = _static_rejection_reason(spec, database, config)

    assert reason is not None
    assert "unsupported dreamplace_native_residual terms" in reason


def test_native_residual_allows_native_objective_route_correction(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        objective_mode="native_residual",
        route_coeff_cap=0.03,
    )
    database = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    native = term("native_objective")',
                '    route = term("soft_rudy_pnorm")',
                "    score = native + 0.003 * log1p(route)",
                '    return score, {"native": native, "route": route}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace",
    )

    reason = _static_rejection_reason(spec, database, config)
    coeffs, unsupported = _route_coefficients(spec.ast)

    assert reason is None
    assert unsupported == set()
    assert coeffs == [("soft_rudy_pnorm", 0.003)]


def test_native_residual_prompt_requires_native_objective_base():
    messages = generation_messages(
        {
            "objective_mode": "native_residual",
            "search_policy": {"term_calibration": {"enabled": True}},
        },
        term_scope="dreamplace_native_residual",
        mutation_mode="diff",
    )
    text = "\n".join(message["content"] for message in messages)

    assert 'term(\\"native_objective\\")' in text
    assert 'term(\\"wirelength_wawl\\") + term(\\"density_electric\\")' not in text


def test_native_residual_rejects_replacement_observable_base(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        objective_mode="native_residual",
        route_coeff_cap=0.03,
    )
    database = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength_wawl")',
                '    den = term("density_electric")',
                '    route = term("soft_rudy_mean")',
                "    score = wl + den + 0.001 * route",
                '    return score, {"wirelength": wl, "density": den, "route": route}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace",
    )

    reason = _static_rejection_reason(spec, database, config)

    assert reason is not None
    assert "unsupported dreamplace_native_residual terms" in reason


def test_replacement_static_validation_rejects_internal_native_objective(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        objective_mode="replacement",
        require_nonzero_routing_term=True,
        min_replacement_nonbaseline_terms=2,
    )
    database = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    native = term("native_objective")',
                '    route = term("route_pressure_long")',
                '    pins = term("pin_density_pnorm")',
                "    score = native + 1e-8 * log1p(route * pins)",
                '    return score, {"native": native, "route": route, "pins": pins}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace",
    )

    reason = _static_rejection_reason(spec, database, config)

    assert reason is not None
    assert "unsupported dreamplace_replacement terms" in reason
    assert "native_objective" in reason


def test_replacement_static_validation_rejects_mixed_wirelength_models(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        objective_mode="replacement",
    )
    database = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wawl = term("wirelength_wawl")',
                '    lse = term("wirelength_lse")',
                '    density = term("density_electric")',
                "    score = wawl + lse + density",
                '    return score, {"wawl": wawl, "lse": lse, "density": density}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace",
    )

    reason = _static_rejection_reason(spec, database, config)

    assert reason is not None
    assert "one smooth wirelength formulation" in reason


def test_replacement_static_validation_rejects_mixed_density_models(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        objective_mode="replacement",
    )
    database = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength_wawl")',
                '    electric = term("density_electric")',
                '    bell = term("density_bell")',
                "    score = wl + electric + bell",
                '    return score, {"wirelength": wl, "electric": electric, "bell": bell}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace",
    )

    reason = _static_rejection_reason(spec, database, config)

    assert reason is not None


def test_replacement_static_validation_rejects_missing_wirelength_anchor(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        objective_mode="replacement",
        require_nonzero_routing_term=True,
        min_replacement_nonbaseline_terms=2,
    )
    database = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    den = term("density_electric")',
                '    pin_wl = term("pin_count_weighted_wl")',
                '    route = term("route_pressure_long")',
                '    pins = term("pin_density_pnorm")',
                "    score = den + pin_wl + 1e-8 * softplus(route * pins)",
                '    return score, {"density": den, "pin_wl": pin_wl, "route": route, "pins": pins}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )

    reason = _static_rejection_reason(spec, database, config)

    assert reason is not None
    assert "canonical wirelength anchor" in reason
    assert "pin_count_weighted_wl is a routing-aware modifier" in reason


def test_replacement_static_validation_rejects_missing_density_anchor(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        objective_mode="replacement",
        require_nonzero_routing_term=True,
        min_replacement_nonbaseline_terms=2,
    )
    database = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength_wawl")',
                '    route = term("route_pressure_long")',
                '    pins = term("pin_density_pnorm")',
                "    score = wl + 1e-8 * softplus(route * pins)",
                '    return score, {"wirelength": wl, "route": route, "pins": pins}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )

    reason = _static_rejection_reason(spec, database, config)

    assert reason is not None
    assert "density/utilization anchor" in reason


def test_replacement_validation_rejects_route_coeff_above_cap(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        objective_mode="replacement",
        route_coeff_cap=0.01,
    )
    database = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength_wawl")',
                '    den = term("density_electric")',
                '    route = term("soft_rudy_pnorm")',
                "    score = wl + den + 0.02 * route",
                '    return score, {"wirelength": wl, "density": den, "route": route}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )

    reason = _static_rejection_reason(spec, database, config)

    assert reason is not None
    assert "replacement route coefficient exceeds cap" in reason


def test_mean_improvement_with_design_regression_is_not_parent_eligible(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        hpwl_gate_search_pct=0.0,
        hpwl_gate_elite_pct=0.0,
        per_design_hpwl_gate_search_pct=0.0,
        per_design_overflow_gate_search_pct=0.0,
        min_design_both_improvement_fraction=1.0,
    )
    ranking = SimpleNamespace(
        average_rank=1.0,
        structural_failure_count=0,
        metric_regression_count=0,
        severe_regression_count=0,
        design_count=2,
        seed_count=1,
        cell_count=2,
        metric_ranks={"hpwl_delta_pct": 1.0, "overflow_delta_pct": 1.0},
        design_ranks={"good": 1.0, "bad": 2.0},
    )
    metrics = _metrics_from_rows(
        [
            {
                "design": "good",
                "objective_id": "mean_good_design_bad",
                "hpwl_delta_pct": -3.0,
                "overflow_delta_pct": -3.0,
                "runtime_seconds": 10.0,
                "custom_grad_norm": 1.0,
                "outcome_label": "success_or_improvement",
            },
            {
                "design": "bad",
                "objective_id": "mean_good_design_bad",
                "hpwl_delta_pct": 1.0,
                "overflow_delta_pct": 1.0,
                "runtime_seconds": 10.0,
                "custom_grad_norm": 1.0,
                "outcome_label": "metric_regression",
            },
        ],
        ranking,
        config,
    )

    assert metrics["hpwl_delta_pct"] == -1.0
    assert metrics["overflow_delta_pct"] == -1.0
    assert metrics["hpwl_gate_passed"] is True
    assert metrics["overflow_gate_passed"] is True
    assert metrics["robust_gate_passed"] is False
    assert metrics["parent_eligible"] is False
    assert metrics["negative_memory_only"] is True
    assert metrics["combined_score"] == 0.0
    assert metrics["worst_design_hpwl_delta_pct"] == 1.0
    assert metrics["worst_design_overflow_delta_pct"] == 1.0


def test_aggregate_near_miss_can_be_search_parent_when_robust_parent_gate_disabled(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        hpwl_gate_search_pct=0.0,
        hpwl_gate_elite_pct=0.0,
        per_design_hpwl_gate_search_pct=0.0,
        per_design_overflow_gate_search_pct=0.0,
        min_design_both_improvement_fraction=1.0,
        require_robust_parent_gate=False,
    )
    ranking = SimpleNamespace(
        average_rank=1.0,
        structural_failure_count=0,
        metric_regression_count=0,
        severe_regression_count=0,
        design_count=2,
        seed_count=1,
        cell_count=2,
        metric_ranks={"hpwl_delta_pct": 1.0, "overflow_delta_pct": 1.0},
        design_ranks={"good": 1.0, "bad": 2.0},
    )
    metrics = _metrics_from_rows(
        [
            {
                "design": "good",
                "objective_id": "aggregate_good_design_bad",
                "hpwl_delta_pct": -3.0,
                "overflow_delta_pct": -3.0,
                "runtime_seconds": 10.0,
                "custom_grad_norm": 1.0,
                "outcome_label": "success_or_improvement",
            },
            {
                "design": "bad",
                "objective_id": "aggregate_good_design_bad",
                "hpwl_delta_pct": 1.0,
                "overflow_delta_pct": 1.0,
                "runtime_seconds": 10.0,
                "custom_grad_norm": 1.0,
                "outcome_label": "metric_regression",
            },
        ],
        ranking,
        config,
    )

    assert metrics["robust_gate_passed"] is False
    assert metrics["parent_eligible"] is True
    assert metrics["negative_memory_only"] is False
    assert metrics["near_miss_parent_due_to_robustness"] is True
    assert metrics["elite_gate_passed"] is False
    assert metrics["combined_score"] > 0.0


def test_constrained_score_prefers_actual_improvement_over_zero_delta(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        hpwl_gate_search_pct=0.0,
        hpwl_gate_elite_pct=0.0,
    )
    ranking = SimpleNamespace(
        average_rank=2.0,
        structural_failure_count=0,
        metric_regression_count=0,
        severe_regression_count=0,
        design_count=1,
        seed_count=1,
        cell_count=1,
        metric_ranks={"hpwl_delta_pct": 1.0, "overflow_delta_pct": 1.0},
        design_ranks={"bp_fe": 1.0},
    )
    zero = _metrics_from_rows(
        [
            {
                "objective_id": "baseline",
                "hpwl_delta_pct": 0.0,
                "overflow_delta_pct": 0.0,
                "runtime_seconds": 10.0,
                "custom_grad_norm": 1.0,
                "outcome_label": "success_or_improvement",
            }
        ],
        ranking,
        config,
    )
    improved = _metrics_from_rows(
        [
            {
                "objective_id": "improved",
                "hpwl_delta_pct": -0.1,
                "overflow_delta_pct": -0.1,
                "runtime_seconds": 10.0,
                "custom_grad_norm": 1.0,
                "outcome_label": "success_or_improvement",
            }
        ],
        ranking,
        config,
    )

    assert improved["combined_score"] > zero["combined_score"]


def test_hpwl_safe_overflow_regression_is_negative_memory(tmp_path):
    config = load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path)))
    ranking = SimpleNamespace(
        average_rank=1.0,
        structural_failure_count=0,
        metric_regression_count=1,
        severe_regression_count=0,
        design_count=1,
        seed_count=1,
        cell_count=1,
        metric_ranks={"hpwl_delta_pct": 1.0, "overflow_delta_pct": 2.0},
        design_ranks={"bp_fe": 1.5},
    )
    metrics = _metrics_from_rows(
        [
            {
                "objective_id": "safe_hpwl_bad_overflow",
                "hpwl_delta_pct": -2.0,
                "overflow_delta_pct": 1.0,
                "runtime_seconds": 10.0,
                "custom_grad_norm": 1.0,
                "outcome_label": "metric_regression",
            }
        ],
        ranking,
        config,
    )

    assert metrics["combined_score"] == 0.0
    assert metrics["hpwl_gate_passed"] is True
    assert metrics["overflow_gate_passed"] is False
    assert metrics["parent_eligible"] is False
    assert metrics["negative_memory_only"] is True


def test_rows_track_native_and_custom_default_deltas_independently():
    rows = [
        {
            "design": "isa_npu",
            "seed": "1000",
            "objective_id": "default",
            "hpwl": "64.0",
            "overflow": "0.98",
        },
        {
            "design": "isa_npu",
            "seed": "1000",
            "objective_id": "custom_default",
            "hpwl": "100.0",
            "overflow": "0.07",
        },
        {
            "design": "isa_npu",
            "seed": "1000",
            "objective_id": "candidate",
            "hpwl": "101.0",
            "overflow": "0.069",
        },
    ]

    adjusted = _rows_with_primary_baseline(rows, primary_baseline="custom_default")
    candidate = next(row for row in adjusted if row["objective_id"] == "candidate")

    assert candidate["baseline_objective_id"] == "custom_default"
    assert abs(candidate["hpwl_delta_pct"] - 1.0) < 1e-12
    assert abs(candidate["overflow_delta_pct"] - (-1.4285714285714286)) < 1e-12
    assert abs(candidate["native_default_hpwl_delta_pct"] - 57.8125) < 1e-12
    assert abs(candidate["custom_default_hpwl_delta_pct"] - 1.0) < 1e-12


def test_custom_default_primary_selection_keeps_native_default_reporting(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        primary_baseline="custom_default",
        require_nonzero_routing_term=True,
        require_custom_default_gate=True,
        min_abs_routing_coeff=1e-12,
        min_custom_default_effect_pct=0.25,
        per_design_hpwl_gate_search_pct=5.0,
        per_design_overflow_gate_search_pct=0.0,
        min_design_both_improvement_fraction=1.0,
    )
    rows = _rows_with_primary_baseline(
        [
            {
                "design": "isa_npu",
                "seed": "1000",
                "objective_id": "default",
                "hpwl": "64.0",
                "overflow": "0.98",
                "status": "success",
            },
            {
                "design": "isa_npu",
                "seed": "1000",
                "objective_id": "custom_default",
                "hpwl": "100.0",
                "overflow": "0.07",
                "status": "success",
            },
            {
                "design": "isa_npu",
                "seed": "1000",
                "objective_id": "candidate",
                "hpwl": "99.0",
                "overflow": "0.069",
                "status": "success",
            },
        ],
        primary_baseline="custom_default",
    )
    ranking = SimpleNamespace(
        average_rank=1.0,
        structural_failure_count=0,
        metric_regression_count=0,
        severe_regression_count=0,
        design_count=1,
        seed_count=1,
        cell_count=1,
        metric_ranks={"hpwl_delta_pct": 1.0, "overflow_delta_pct": 1.0},
        design_ranks={"isa_npu": 1.0},
    )
    metrics = _metrics_from_rows(
        [row for row in rows if row["objective_id"] == "candidate"],
        ranking,
        config,
    )
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength")',
                '    den = term("density")',
                '    route = term("soft_rudy_pnorm")',
                "    score = wl + den + 1e-8 * route",
                '    return score, {"wirelength": wl, "density": den, "route": route}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )

    _apply_routing_gate(metrics, spec, config, allow_nonrouting_parent=False)

    assert metrics["hpwl_gate_passed"] is True
    assert metrics["custom_default_gate_passed"] is True
    assert metrics["parent_eligible"] is True
    assert metrics["native_default_hpwl_delta_pct"] > 50.0
    assert metrics["native_default_comparison_label"] == "native_tradeoff_overflow_for_hpwl"
    assert metrics["beats_native_default_pareto"] is False


def test_native_default_comparison_labels_are_explicit():
    assert (
        openevolve_tier2._native_default_comparison_label(-0.1, -1.0)
        == "dominates_native_default"
    )
    assert (
        openevolve_tier2._native_default_comparison_label(4.8, -63.0)
        == "native_tradeoff_overflow_for_hpwl"
    )
    assert (
        openevolve_tier2._native_default_comparison_label(-2.0, 0.5)
        == "native_tradeoff_hpwl_for_overflow"
    )
    assert (
        openevolve_tier2._native_default_comparison_label(1.0, 0.5)
        == "regresses_native_default"
    )
    assert (
        openevolve_tier2._native_default_comparison_label(None, -1.0)
        == "native_default_unknown"
    )


def test_cumulative_iteration_summaries_include_resumed_history(tmp_path):
    first = tmp_path / "iteration_0001"
    second = tmp_path / "iteration_0002"
    first.mkdir()
    second.mkdir()
    (first / "iteration_summary.json").write_text(
        json.dumps({"iteration": 1, "program_id": "p1"}),
        encoding="utf-8",
    )
    (second / "iteration_summary.json").write_text(
        json.dumps({"iteration": 2, "program_id": "p2"}),
        encoding="utf-8",
    )

    summaries = openevolve_tier2._load_cumulative_iteration_summaries(tmp_path)

    assert [item["iteration"] for item in summaries] == [1, 2]
    assert [item["program_id"] for item in summaries] == ["p1", "p2"]


def test_baseline_portfolio_summary_detects_candidate_win(tmp_path):
    rows = _rows_with_primary_baseline(
        [
            {
                "design": "bp_fe",
                "seed": "1000",
                "objective_id": "default",
                "hpwl": "100.0",
                "overflow": "0.100",
                "status": "success",
            },
            {
                "design": "bp_fe",
                "seed": "1000",
                "objective_id": "custom_default",
                "hpwl": "100.0",
                "overflow": "0.100",
                "status": "success",
            },
            {
                "design": "bp_fe",
                "seed": "1000",
                "objective_id": "fixed_density_baseline",
                "hpwl": "99.0",
                "overflow": "0.090",
                "status": "success",
            },
            {
                "design": "bp_fe",
                "seed": "1000",
                "objective_id": "candidate",
                "hpwl": "98.0",
                "overflow": "0.080",
                "status": "success",
            },
        ],
        primary_baseline="custom_default",
    )
    rankings = aggregate_rank_scores(rows)

    summary = _baseline_portfolio_summary(
        rows=rows,
        rankings=rankings,
        candidate_objective_id="candidate",
        baseline_objective_ids={"default", "custom_default", "fixed_density_baseline"},
    )

    assert summary["baseline_portfolio_compared"] is True
    assert summary["beats_baseline_portfolio"] is True
    assert summary["baseline_portfolio_candidate_rank"] == 1
    assert summary["baseline_portfolio_best_objective_id"] == "fixed_density_baseline"
    assert summary["baseline_portfolio_hpwl_delta_vs_best_pct"] == -1.0
    assert summary["baseline_portfolio_overflow_delta_vs_best_pct"] == pytest.approx(-10.0)
    assert summary["baseline_portfolio_pareto_label_vs_best"] == "dominates_best_baseline"
    assert summary["baseline_portfolio_pareto_dominates_best"] is True
    assert summary["baseline_portfolio_pareto_tradeoff_vs_best"] is False


def test_baseline_portfolio_summary_detects_candidate_losing_to_fixed_baseline(tmp_path):
    rows = _rows_with_primary_baseline(
        [
            {
                "design": "bp_fe",
                "seed": "1000",
                "objective_id": "default",
                "hpwl": "100.0",
                "overflow": "0.100",
                "status": "success",
            },
            {
                "design": "bp_fe",
                "seed": "1000",
                "objective_id": "custom_default",
                "hpwl": "100.0",
                "overflow": "0.100",
                "status": "success",
            },
            {
                "design": "bp_fe",
                "seed": "1000",
                "objective_id": "fixed_density_baseline",
                "hpwl": "98.0",
                "overflow": "0.090",
                "status": "success",
            },
            {
                "design": "bp_fe",
                "seed": "1000",
                "objective_id": "candidate",
                "hpwl": "101.0",
                "overflow": "0.080",
                "status": "success",
            },
        ],
        primary_baseline="custom_default",
    )
    rankings = aggregate_rank_scores(rows)

    summary = _baseline_portfolio_summary(
        rows=rows,
        rankings=rankings,
        candidate_objective_id="candidate",
        baseline_objective_ids={"default", "custom_default", "fixed_density_baseline"},
    )

    assert summary["baseline_portfolio_compared"] is True
    assert summary["beats_baseline_portfolio"] is False
    assert summary["baseline_portfolio_candidate_rank"] > 1
    assert summary["baseline_portfolio_best_objective_id"] == "fixed_density_baseline"
    assert summary["baseline_portfolio_hpwl_delta_vs_best_pct"] == 3.0
    assert summary["baseline_portfolio_overflow_delta_vs_best_pct"] == pytest.approx(-10.0)
    assert (
        summary["baseline_portfolio_pareto_label_vs_best"]
        == "baseline_tradeoff_overflow_for_hpwl"
    )
    assert summary["baseline_portfolio_pareto_dominates_best"] is False
    assert summary["baseline_portfolio_pareto_tradeoff_vs_best"] is True


def test_baseline_portfolio_summary_does_not_let_severe_baseline_block_safe_candidate(tmp_path):
    rows = _rows_with_primary_baseline(
        [
            {
                "design": "bp_fe",
                "seed": "1000",
                "objective_id": "default",
                "hpwl": "100.0",
                "overflow": "0.100",
                "status": "success",
            },
            {
                "design": "bp_fe",
                "seed": "1000",
                "objective_id": "severe_overflow_baseline",
                "hpwl": "90.0",
                "overflow": "0.010",
                "status": "success",
            },
            {
                "design": "bp_fe",
                "seed": "1000",
                "objective_id": "candidate",
                "hpwl": "98.0",
                "overflow": "0.090",
                "status": "success",
            },
            {
                "design": "isa_npu",
                "seed": "1000",
                "objective_id": "default",
                "hpwl": "100.0",
                "overflow": "0.100",
                "status": "success",
            },
            {
                "design": "isa_npu",
                "seed": "1000",
                "objective_id": "severe_overflow_baseline",
                "hpwl": "165.0",
                "overflow": "0.010",
                "status": "success",
            },
            {
                "design": "isa_npu",
                "seed": "1000",
                "objective_id": "candidate",
                "hpwl": "99.0",
                "overflow": "0.095",
                "status": "success",
            },
        ],
        primary_baseline="default",
    )
    rankings = aggregate_rank_scores(rows, severe_threshold_pct=10.0)

    summary = _baseline_portfolio_summary(
        rows=rows,
        rankings=rankings,
        candidate_objective_id="candidate",
        baseline_objective_ids={"default", "severe_overflow_baseline"},
    )

    assert summary["baseline_portfolio_candidate_rank"] == 1
    assert summary["baseline_portfolio_best_objective_id"] == "default"
    assert summary["beats_baseline_portfolio"] is True


def test_baseline_portfolio_summary_requires_measurable_effect(tmp_path):
    rows = _rows_with_primary_baseline(
        [
            {
                "design": "bp_fe",
                "seed": "1000",
                "objective_id": "default",
                "hpwl": "100.0",
                "overflow": "0.100",
                "status": "success",
            },
            {
                "design": "bp_fe",
                "seed": "1000",
                "objective_id": "fixed_density_baseline",
                "hpwl": "99.0000",
                "overflow": "0.09000",
                "status": "success",
            },
            {
                "design": "bp_fe",
                "seed": "1000",
                "objective_id": "candidate",
                "hpwl": "98.9999",
                "overflow": "0.08999",
                "status": "success",
            },
        ],
        primary_baseline="default",
    )
    rankings = aggregate_rank_scores(rows)

    summary = _baseline_portfolio_summary(
        rows=rows,
        rankings=rankings,
        candidate_objective_id="candidate",
        baseline_objective_ids={"default", "fixed_density_baseline"},
        min_effect_pct=0.25,
    )

    assert summary["baseline_portfolio_candidate_rank"] == 1
    assert summary["baseline_portfolio_best_objective_id"] == "fixed_density_baseline"
    assert summary["baseline_portfolio_effect_pct"] < 0.25
    assert summary["baseline_portfolio_effect_gate_passed"] is False
    assert summary["beats_baseline_portfolio"] is False


def test_baseline_portfolio_summary_rejects_behavior_clone_of_fixed_baseline(tmp_path):
    rows = _rows_with_primary_baseline(
        [
            {
                "design": "bp_fe",
                "seed": "1000",
                "objective_id": "default",
                "hpwl": "100.0",
                "overflow": "0.100",
                "status": "success",
            },
            {
                "design": "bp_fe",
                "seed": "1000",
                "objective_id": "wawl_electric_baseline",
                "hpwl": "200.0000",
                "overflow": "0.05000",
                "status": "success",
            },
            {
                "design": "bp_fe",
                "seed": "1000",
                "objective_id": "candidate",
                "hpwl": "199.9000",
                "overflow": "0.04999",
                "status": "success",
            },
        ],
        primary_baseline="default",
    )
    rankings = aggregate_rank_scores(rows)

    summary = _baseline_portfolio_summary(
        rows=rows,
        rankings=rankings,
        candidate_objective_id="candidate",
        baseline_objective_ids={"default", "wawl_electric_baseline"},
        min_effect_pct=0.0,
        min_behavior_distance_pct=0.5,
    )

    assert summary["baseline_portfolio_candidate_rank"] > 1
    assert summary["baseline_portfolio_closest_behavior_objective_id"] == "wawl_electric_baseline"
    assert summary["baseline_portfolio_behavior_distance_pct"] < 0.5
    assert summary["baseline_portfolio_behavior_distance_gate_passed"] is False
    assert summary["beats_baseline_portfolio"] is False


def test_baseline_portfolio_gate_blocks_parent_selection_when_required(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        require_baseline_portfolio_gate=True,
    )
    metrics = {
        "combined_score": 1.0,
        "constrained_score": 1.0,
        "elite_gate_passed": True,
        "parent_eligible": True,
        "negative_memory_only": False,
        "baseline_portfolio_gate_passed": False,
        "baseline_portfolio_best_objective_id": "fixed_density_baseline",
    }

    _apply_baseline_portfolio_gate(metrics, config)

    assert metrics["combined_score"] == 0.0
    assert metrics["parent_eligible"] is False
    assert metrics["negative_memory_only"] is True
    assert "fixed_density_baseline" in metrics["feedback_lesson"]


def test_baseline_portfolio_pareto_gate_blocks_rank_only_tradeoff(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        require_baseline_portfolio_gate=True,
        require_baseline_portfolio_pareto=True,
    )
    metrics = {
        "combined_score": 1.0,
        "constrained_score": 1.0,
        "elite_gate_passed": True,
        "parent_eligible": True,
        "negative_memory_only": False,
        "baseline_portfolio_compared": True,
        "baseline_portfolio_gate_passed": True,
        "baseline_portfolio_pareto_dominates_best": False,
        "baseline_portfolio_pareto_label_vs_best": "baseline_tradeoff_overflow_for_hpwl",
        "baseline_portfolio_best_objective_id": "dreamplace_wawl_density_bell",
        "baseline_portfolio_hpwl_delta_vs_best_pct": 0.4,
        "baseline_portfolio_overflow_delta_vs_best_pct": -0.01,
    }

    _apply_baseline_portfolio_gate(metrics, config)

    assert metrics["combined_score"] == 0.0
    assert metrics["parent_eligible"] is False
    assert metrics["negative_memory_only"] is True
    assert "not a clean Pareto improvement" in metrics["feedback_lesson"]
    assert "baseline_tradeoff_overflow_for_hpwl" in metrics["feedback_lesson"]


def test_baseline_portfolio_gate_reports_behavior_clone_reason(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        require_baseline_portfolio_gate=True,
    )
    metrics = {
        "combined_score": 1.0,
        "constrained_score": 1.0,
        "elite_gate_passed": True,
        "parent_eligible": True,
        "negative_memory_only": False,
        "baseline_portfolio_gate_passed": False,
        "baseline_portfolio_best_objective_id": "default",
        "baseline_portfolio_closest_behavior_objective_id": "wawl_electric_baseline",
        "baseline_portfolio_behavior_distance_gate_passed": False,
    }

    _apply_baseline_portfolio_gate(metrics, config)

    assert metrics["combined_score"] == 0.0
    assert metrics["parent_eligible"] is False
    assert "nearly indistinguishable" in metrics["feedback_lesson"]
    assert "wawl_electric_baseline" in metrics["feedback_lesson"]


def test_near_miss_parent_policy_allows_bounded_distinct_tradeoff(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        allow_near_miss_parents=True,
        near_miss_max_worst_hpwl_pct=25.0,
        near_miss_max_worst_overflow_pct=10.0,
        near_miss_parent_score_floor=0.02,
    )
    metrics = {
        "combined_score": 0.0,
        "constrained_score": 0.0,
        "structural_failure_count": 0,
        "severe_regression_count": 0,
        "routing_aware_tier2_candidate": True,
        "baseline_portfolio_behavior_distance_gate_passed": True,
        "hpwl_delta_pct": -8.0,
        "overflow_delta_pct": 4.0,
        "worst_design_hpwl_delta_pct": 12.0,
        "worst_design_overflow_delta_pct": 7.0,
        "parent_eligible": False,
        "negative_memory_only": True,
        "elite_gate_passed": False,
        "feedback_lesson": "Measured tradeoff.",
    }

    _apply_near_miss_parent_policy(metrics, config, allow_seed_baseline=False)

    assert metrics["parent_eligible"] is True
    assert metrics["negative_memory_only"] is False
    assert metrics["elite_gate_passed"] is False
    assert metrics["exploration_parent_only"] is True
    assert metrics["combined_score"] >= 0.02
    assert "stepping stone" in metrics["feedback_lesson"]


def test_near_miss_parent_policy_rejects_behavior_clone(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        allow_near_miss_parents=True,
    )
    metrics = {
        "combined_score": 0.0,
        "constrained_score": 0.0,
        "structural_failure_count": 0,
        "severe_regression_count": 0,
        "routing_aware_tier2_candidate": True,
        "baseline_portfolio_behavior_distance_gate_passed": False,
        "hpwl_delta_pct": -8.0,
        "overflow_delta_pct": 4.0,
        "worst_design_hpwl_delta_pct": 12.0,
        "worst_design_overflow_delta_pct": 7.0,
        "parent_eligible": False,
        "negative_memory_only": True,
    }

    _apply_near_miss_parent_policy(metrics, config, allow_seed_baseline=False)

    assert metrics["parent_eligible"] is False
    assert metrics["negative_memory_only"] is True
    assert "exploration_parent_only" not in metrics


def test_near_miss_parent_policy_ignores_pareto_but_respects_portfolio_gate(tmp_path):
    # Stepping-stone parenting does not require Pareto domination of the
    # single best baseline (that stays on elite promotion), but it still
    # requires outranking the fixed baseline portfolio when that gate is on.
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        allow_near_miss_parents=True,
        require_baseline_portfolio_gate=True,
        require_baseline_portfolio_pareto=True,
    )
    base_metrics = {
        "combined_score": 0.0,
        "constrained_score": 0.0,
        "structural_failure_count": 0,
        "severe_regression_count": 0,
        "routing_aware_tier2_candidate": True,
        "baseline_portfolio_compared": True,
        "baseline_portfolio_pareto_dominates_best": False,
        "baseline_portfolio_pareto_label_vs_best": "baseline_tradeoff_overflow_for_hpwl",
        "baseline_portfolio_behavior_distance_gate_passed": True,
        "hpwl_delta_pct": -8.0,
        "overflow_delta_pct": 4.0,
        "worst_design_hpwl_delta_pct": 12.0,
        "worst_design_overflow_delta_pct": 7.0,
        "parent_eligible": False,
        "negative_memory_only": True,
    }

    blocked = dict(base_metrics, beats_baseline_portfolio=False)
    _apply_near_miss_parent_policy(blocked, config, allow_seed_baseline=False)
    assert blocked["parent_eligible"] is False
    assert blocked["negative_memory_only"] is True
    assert "exploration_parent_only" not in blocked

    promoted = dict(base_metrics, beats_baseline_portfolio=True)
    _apply_near_miss_parent_policy(promoted, config, allow_seed_baseline=False)
    assert promoted["parent_eligible"] is True
    assert promoted["negative_memory_only"] is False
    assert promoted["exploration_parent_only"] is True
    assert promoted["elite_gate_passed"] is False


def test_near_miss_parent_policy_rejects_too_small_fixed_baseline_pareto(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        allow_near_miss_parents=True,
        require_baseline_portfolio_gate=True,
        require_baseline_portfolio_pareto=True,
    )
    metrics = {
        "combined_score": 0.0,
        "constrained_score": 0.0,
        "structural_failure_count": 0,
        "severe_regression_count": 0,
        "routing_aware_tier2_candidate": True,
        "baseline_portfolio_compared": True,
        "baseline_portfolio_pareto_dominates_best": True,
        "beats_baseline_portfolio": False,
        "baseline_portfolio_effect_pct": 0.05,
        "baseline_portfolio_behavior_distance_gate_passed": True,
        "hpwl_delta_pct": -8.0,
        "overflow_delta_pct": -1.0,
        "worst_design_hpwl_delta_pct": 1.0,
        "worst_design_overflow_delta_pct": 0.5,
        "parent_eligible": False,
        "negative_memory_only": True,
    }

    _apply_near_miss_parent_policy(metrics, config, allow_seed_baseline=False)

    assert metrics["parent_eligible"] is False
    assert metrics["negative_memory_only"] is True
    assert "exploration_parent_only" not in metrics


def test_near_miss_parent_policy_rejects_unbounded_tradeoff(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        allow_near_miss_parents=True,
        near_miss_max_worst_hpwl_pct=25.0,
        near_miss_max_worst_overflow_pct=10.0,
    )
    metrics = {
        "combined_score": 0.0,
        "constrained_score": 0.0,
        "structural_failure_count": 0,
        "severe_regression_count": 0,
        "routing_aware_tier2_candidate": True,
        "baseline_portfolio_behavior_distance_gate_passed": True,
        "hpwl_delta_pct": -8.0,
        "overflow_delta_pct": 4.0,
        "worst_design_hpwl_delta_pct": 40.0,
        "worst_design_overflow_delta_pct": 7.0,
        "parent_eligible": False,
        "negative_memory_only": True,
    }

    _apply_near_miss_parent_policy(metrics, config, allow_seed_baseline=False)

    assert metrics["parent_eligible"] is False
    assert metrics["negative_memory_only"] is True
    assert "exploration_parent_only" not in metrics


def test_native_default_prefer_policy_rewards_pareto_wins(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        native_default_selection_policy="prefer_pareto",
        native_default_pareto_bonus=0.25,
    )
    metrics = {
        "combined_score": 1.0,
        "constrained_score": 1.0,
        "parent_eligible": True,
        "native_default_hpwl_delta_pct": -0.2,
        "native_default_overflow_delta_pct": -1.5,
    }

    _apply_native_default_selection_policy(metrics, config, allow_seed_baseline=False)

    assert metrics["native_default_comparison_label"] == "dominates_native_default"
    assert metrics["beats_native_default_pareto"] is True
    assert metrics["combined_score"] == 1.25
    assert metrics["native_default_pareto_bonus_applied"] == 0.25


def test_native_default_require_policy_blocks_generated_tradeoff(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        native_default_selection_policy="require_pareto",
    )
    metrics = {
        "combined_score": 1.0,
        "constrained_score": 1.0,
        "elite_gate_passed": True,
        "parent_eligible": True,
        "negative_memory_only": False,
        "native_default_hpwl_delta_pct": 4.8,
        "native_default_overflow_delta_pct": -63.0,
    }

    _apply_native_default_selection_policy(metrics, config, allow_seed_baseline=False)

    assert metrics["native_default_comparison_label"] == "native_tradeoff_overflow_for_hpwl"
    assert metrics["combined_score"] == 0.0
    assert metrics["parent_eligible"] is False
    assert metrics["negative_memory_only"] is True
    assert "Pareto improvement" in metrics["feedback_lesson"]


def test_native_default_require_policy_keeps_seed_baselines_available(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        native_default_selection_policy="require_pareto",
    )
    metrics = {
        "combined_score": 1.0,
        "constrained_score": 1.0,
        "parent_eligible": True,
        "negative_memory_only": False,
        "seed_baseline": True,
        "native_default_hpwl_delta_pct": 4.8,
        "native_default_overflow_delta_pct": -63.0,
    }

    _apply_native_default_selection_policy(metrics, config, allow_seed_baseline=True)

    assert metrics["native_default_comparison_label"] == "native_tradeoff_overflow_for_hpwl"
    assert metrics["combined_score"] == 1.0
    assert metrics["parent_eligible"] is True
    assert "Seed baseline remains available" in metrics["native_default_gate_note"]


def test_baseline_portfolio_gate_is_not_prompt_visible_policy():
    visible = _prompt_visible_policy(
        {
            "require_baseline_portfolio_gate": True,
            "require_baseline_portfolio_pareto": True,
            "native_default_selection_policy": "require_pareto",
            "native_default_pareto_bonus": 0.25,
            "reject_baseline_mechanism_clones": True,
            "active_routing_terms": ["soft_rudy_pnorm"],
            "parent_policy": "hpwl_safe_only",
        }
    )

    serialized = json.dumps(visible, sort_keys=True)
    assert "require_baseline_portfolio_gate" not in serialized
    assert "require_baseline_portfolio_pareto" not in serialized
    assert "native_default_selection_policy" not in serialized
    assert "native_default_pareto_bonus" not in serialized
    assert "reject_baseline_mechanism_clones" not in serialized
    assert "baseline_portfolio" not in serialized
    assert "soft_rudy_pnorm" in serialized
    assert "physical_anchor_contract" in serialized
    assert "pin_count_weighted_wl is not a standalone substitute" in serialized


def test_openevolve_report_includes_nonbaseline_audit(tmp_path):
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    baseline = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_density_heavy"),
        program_id="fixed_density_heavy",
        metrics={
            "combined_score": 2.0,
            "seed_baseline": True,
            "parent_eligible": True,
            "mechanism_signature": "fixed_density_shape",
        },
        status="accepted",
    )
    generated = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_routing_aware_smoke"),
        program_id="generated_route_objective",
        parent_id=baseline.id,
        generation=1,
        metrics={
            "combined_score": 3.0,
            "hpwl_delta_pct": -0.4,
            "overflow_delta_pct": -2.5,
            "native_default_hpwl_delta_pct": 4.8,
            "native_default_overflow_delta_pct": -63.0,
            "native_default_comparison_label": "native_tradeoff_overflow_for_hpwl",
            "beats_native_default_pareto": False,
            "baseline_portfolio_compared": True,
            "baseline_portfolio_candidate_rank": 1,
            "baseline_portfolio_best_objective_id": "fixed_density_heavy",
            "baseline_portfolio_best_hpwl_delta_pct": -0.1,
            "baseline_portfolio_best_overflow_delta_pct": -1.0,
            "baseline_portfolio_rank_beats_best": True,
            "beats_baseline_portfolio": True,
            "mechanism_signature": "route_interaction_shape",
            "mechanism_terms": ["soft_rudy_pnorm", "pin_density_pnorm"],
            "parent_eligible": True,
            "outcome_label": "success_or_improvement",
        },
        status="accepted",
    )
    db.add(baseline, target_island=0)
    db.add(generated, target_island=0)

    report = openevolve_tier2._build_report(
        {
            "run_dir": str(tmp_path),
            "program_count": len(db.programs),
            "archive_size": len(db.archive),
            "best_program_id": db.best_program_id,
            "final_tier2": None,
            "final_tier3": None,
        },
        db,
    )

    assert "## Methodology Audit" in report
    assert "Generated programs beating the fixed baseline portfolio: 1" in report
    assert "## Best Non-Baseline Candidates" in report
    assert "generated_route_objective" in report
    assert "rank_vs_fixed_baselines=1" in report
    assert "rank_win_fixed_baseline=True" in report
    assert "promotion_win_fixed_baselines=True" in report
    assert "## Baseline Portfolio Comparison" in report
    assert "rank_win=True" in report
    assert "promotion_win=True" in report
    assert "## Native DREAMPlace Default Comparison" in report
    assert "native_label=native_tradeoff_overflow_for_hpwl" in report
    assert "native_hpwl_delta=4.8" in report
    assert "native_overflow_delta=-63.0" in report


def test_refresh_scores_reclassifies_old_native_baseline_metrics(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        primary_baseline="custom_default",
        require_nonzero_routing_term=True,
        require_custom_default_gate=True,
        min_abs_routing_coeff=1e-12,
        min_custom_default_effect_pct=0.25,
        min_design_both_improvement_fraction=0.0,
    )
    db = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength")',
                '    den = term("density")',
                '    route = term("soft_rudy_pnorm")',
                "    score = wl + den + 1e-8 * route",
                '    return score, {"wirelength": wl, "density": den, "route": route}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )
    program = ObjectiveProgram.from_spec(
        spec,
        program_id="old_native_record",
        status="accepted",
        metrics={
            "primary_baseline": "default",
            "hpwl_delta_pct": 100.0,
            "overflow_delta_pct": -60.0,
            "native_default_hpwl_delta_pct": 100.0,
            "native_default_overflow_delta_pct": -60.0,
            "custom_default_hpwl_delta_pct": 0.2,
            "custom_default_overflow_delta_pct": -0.5,
            "tier2_average_rank": 1.0,
            "structural_failure_count": 0,
            "severe_regression_count": 1,
            "worst_design_hpwl_delta_pct": 100.0,
            "worst_design_overflow_delta_pct": -30.0,
        },
        artifacts={
            "candidate_rows": [
                {
                    "design": "bp_fe",
                    "seed": "1000",
                    "objective_id": spec.id,
                    "baseline_objective_id": "default",
                    "hpwl_delta_pct": 100.0,
                    "overflow_delta_pct": -60.0,
                    "custom_default_hpwl_delta_pct": 0.2,
                    "custom_default_overflow_delta_pct": -0.5,
                },
                {
                    "design": "ethernet",
                    "seed": "1000",
                    "objective_id": spec.id,
                    "baseline_objective_id": "default",
                    "hpwl_delta_pct": -10.0,
                    "overflow_delta_pct": -20.0,
                    "custom_default_hpwl_delta_pct": 0.1,
                    "custom_default_overflow_delta_pct": -0.4,
                },
            ]
        },
    )
    db.add(program)

    _refresh_database_scores(db, config)
    refreshed = db.get("old_native_record")

    assert refreshed.metrics["hpwl_delta_pct"] == 0.2
    assert refreshed.metrics["overflow_delta_pct"] == -0.5
    assert refreshed.metrics["native_default_hpwl_delta_pct"] == 100.0
    assert refreshed.metrics["severe_regression_count"] == 0
    assert refreshed.metrics["per_design_delta_baseline"] == "custom_default"
    assert refreshed.metrics["per_design_deltas_stale"] is False
    assert refreshed.metrics["worst_design_hpwl_delta_pct"] == 0.2
    assert refreshed.metrics["worst_design_overflow_delta_pct"] == -0.4
    assert refreshed.metrics["parent_eligible"] is True
    assert refreshed.metrics["elite_gate_passed"] is True

    _refresh_database_scores(db, config)
    refreshed_again = db.get("old_native_record")

    assert refreshed_again.metrics["hpwl_delta_pct"] == 0.2
    assert refreshed_again.metrics["overflow_delta_pct"] == -0.5
    assert refreshed_again.metrics["parent_eligible"] is True
    assert refreshed_again.metrics["elite_gate_passed"] is True


def test_refresh_scores_recomputes_baseline_portfolio_from_comparison_csv(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        primary_baseline="default",
        require_baseline_portfolio_gate=True,
        require_nonzero_routing_term=True,
        min_abs_routing_coeff=1e-12,
        min_baseline_behavior_distance_pct=0.0,
        min_baseline_portfolio_effect_pct=0.0,
        min_design_both_improvement_fraction=0.0,
    )
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength_wawl")',
                '    den = term("density_bell")',
                '    route = term("soft_rudy_pnorm")',
                "    score = wl + den + 1e-8 * route",
                '    return score, {"wirelength": wl, "density": den, "route": route}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )
    comparison_csv = tmp_path / "comparison_table.csv"
    rows = [
        {"design": "bp_fe", "seed": "1000", "objective_id": "default", "status": "success", "hpwl": "100.0", "overflow": "0.100"},
        {"design": "bp_fe", "seed": "1000", "objective_id": "unsafe_baseline", "status": "success", "hpwl": "90.0", "overflow": "0.010"},
        {"design": "bp_fe", "seed": "1000", "objective_id": spec.id, "status": "success", "hpwl": "98.0", "overflow": "0.090"},
        {"design": "isa_npu", "seed": "1000", "objective_id": "default", "status": "success", "hpwl": "100.0", "overflow": "0.100"},
        {"design": "isa_npu", "seed": "1000", "objective_id": "unsafe_baseline", "status": "success", "hpwl": "165.0", "overflow": "0.010"},
        {"design": "isa_npu", "seed": "1000", "objective_id": spec.id, "status": "success", "hpwl": "99.0", "overflow": "0.095"},
    ]
    with comparison_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["design", "seed", "objective_id", "status", "hpwl", "overflow"],
        )
        writer.writeheader()
        writer.writerows(rows)
    db = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    program = ObjectiveProgram.from_spec(
        spec,
        program_id="stale_portfolio_record",
        status="accepted",
        metrics={
            "combined_score": 0.0,
            "primary_baseline": "default",
            "hpwl_delta_pct": -1.5,
            "overflow_delta_pct": -7.5,
            "tier2_average_rank": 2.0,
            "structural_failure_count": 0,
            "severe_regression_count": 0,
            "worst_design_hpwl_delta_pct": -1.0,
            "worst_design_overflow_delta_pct": -5.0,
            "baseline_portfolio_gate_passed": False,
            "beats_baseline_portfolio": False,
            "baseline_portfolio_best_objective_id": "unsafe_baseline",
            "baseline_portfolio_gate_note": "stale failure note",
            "exploration_parent_only": True,
            "near_miss_parent_score": 0.01,
            "near_miss_parent_reason": "stale result from a previous scoring policy",
        },
        artifacts={
            "candidate_rows": [
                {
                    "design": "bp_fe",
                    "seed": "1000",
                    "objective_id": spec.id,
                    "hpwl_delta_pct": -2.0,
                    "overflow_delta_pct": -10.0,
                },
                {
                    "design": "isa_npu",
                    "seed": "1000",
                    "objective_id": spec.id,
                    "hpwl_delta_pct": -1.0,
                    "overflow_delta_pct": -5.0,
                },
            ],
            "tier2_summary": {"comparison_csv": str(comparison_csv)},
            "baseline_portfolio_ids": ["default", "unsafe_baseline"],
        },
    )
    db.add(program)

    _refresh_database_scores(db, config)
    refreshed = db.get("stale_portfolio_record")

    assert refreshed.metrics["baseline_portfolio_best_objective_id"] == "default"
    assert refreshed.metrics["baseline_portfolio_gate_passed"] is True
    assert refreshed.metrics["beats_baseline_portfolio"] is True
    assert "baseline_portfolio_gate_note" not in refreshed.metrics
    assert refreshed.metrics["parent_eligible"] is True
    assert "exploration_parent_only" not in refreshed.metrics
    assert "near_miss_parent_score" not in refreshed.metrics
    assert "near_miss_parent_reason" not in refreshed.metrics


def test_residual_validation_requires_base_and_route_coeff_cap(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        objective_mode="residual",
    )
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    missing_base = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    route = term("soft_rudy_pnorm")',
                "    score = 0.02 * route",
                '    return score, {"route": route}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )
    assert "missing required base terms" in _static_rejection_reason(missing_base, db, config)

    high_coeff = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength")',
                '    den = term("density")',
                '    route = term("soft_rudy_pnorm")',
                "    score = wl + den + 0.30 * route",
                '    return score, {"wirelength": wl, "density": den, "route": route}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )
    assert "coefficient exceeds cap" in _static_rejection_reason(high_coeff, db, config)

    negative_high_coeff = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength")',
                '    den = term("density")',
                '    route = term("soft_rudy_pnorm")',
                "    score = wl + den - 0.30 * route",
                '    return score, {"wirelength": wl, "density": den, "route": route}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )
    assert "coefficient exceeds cap" in _static_rejection_reason(
        negative_high_coeff,
        db,
        config,
    )


def test_route_coefficient_subtraction_preserves_sign():
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength")',
                '    den = term("density")',
                '    route = term("soft_rudy_pnorm")',
                "    score = wl + den - 0.003 * route",
                '    return score, {"wirelength": wl, "density": den, "route": route}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )

    coeffs, unsupported = _route_coefficients(spec.ast)

    assert unsupported == set()
    assert coeffs == [("soft_rudy_pnorm", -0.003)]


def test_seed_fallback_does_not_create_fake_parent(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        evaluate_initial_presets=False,
    )
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )

    _seed_initial_programs(
        db,
        config=config,
        run_root=tmp_path / "run",
        dreamplace_root=tmp_path,
        resume=False,
        retry_failed=False,
    )

    assert db.programs
    assert all(program.metrics["combined_score"] == 0.0 for program in db.programs.values())
    assert not any(program.is_parent_eligible for program in db.programs.values())
    assert all(program.metrics["negative_memory_only"] for program in db.programs.values())


def test_bootstrap_seed_parent_uses_evaluated_nonstructural_seed(tmp_path):
    db = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    db.add(
        ObjectiveProgram.from_spec(
            objective_preset("dreamplace_route_pressure_long"),
            program_id="bad_seed",
            metrics={
                "seed_baseline": True,
                "structural_failure_count": 0,
                "severe_regression_count": 1,
                "tier2_average_rank": 4.0,
                "hpwl_delta_pct": 30.0,
                "overflow_delta_pct": -20.0,
                "hpwl_gate_passed": False,
                "overflow_gate_passed": True,
                "negative_memory_only": True,
                "parent_eligible": False,
            },
        )
    )
    db.add(
        ObjectiveProgram.from_spec(
            objective_preset("dreamplace_wawl_electric"),
            program_id="better_seed",
            metrics={
                "seed_baseline": True,
                "structural_failure_count": 0,
                "severe_regression_count": 0,
                "tier2_average_rank": 2.0,
                "hpwl_delta_pct": 8.0,
                "overflow_delta_pct": -10.0,
                "hpwl_gate_passed": False,
                "overflow_gate_passed": True,
                "negative_memory_only": True,
                "parent_eligible": False,
            },
        )
    )

    selected = _promote_bootstrap_seed_parent(db)

    assert selected is not None
    assert selected.id == "better_seed"
    assert selected.metrics["bootstrap_parent"] is True
    assert selected.metrics["manual_safe_baseline"] is True
    assert selected.is_parent_eligible is True
    assert db.get("bad_seed").is_parent_eligible is False


def test_bootstrap_seed_parent_prefers_measured_quality_before_routing_scaffold(tmp_path):
    db = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    db.add(
        ObjectiveProgram.from_spec(
            objective_preset("dreamplace_wawl_electric"),
            program_id="plain_seed",
            metrics={
                "seed_baseline": True,
                "structural_failure_count": 0,
                "severe_regression_count": 1,
                "tier2_average_rank": 1.0,
                "worst_design_hpwl_delta_pct": 6.0,
                "hpwl_delta_pct": 6.0,
                "overflow_delta_pct": -10.0,
                "hpwl_gate_passed": False,
                "overflow_gate_passed": True,
                "negative_memory_only": True,
                "parent_eligible": False,
            },
        )
    )
    db.add(
        ObjectiveProgram.from_spec(
            objective_preset("dreamplace_routing_aware_smoke"),
            program_id="routing_seed",
            metrics={
                "seed_baseline": True,
                "structural_failure_count": 0,
                "severe_regression_count": 1,
                "tier2_average_rank": 5.0,
                "worst_design_hpwl_delta_pct": 20.0,
                "hpwl_delta_pct": 20.0,
                "overflow_delta_pct": -30.0,
                "hpwl_gate_passed": False,
                "overflow_gate_passed": True,
                "negative_memory_only": True,
                "parent_eligible": False,
            },
        )
    )

    selected = _promote_bootstrap_seed_parent(db)

    assert selected is not None
    assert selected.id == "plain_seed"
    assert selected.metrics["bootstrap_parent"] is True
    assert selected.metrics["manual_safe_baseline"] is True
    assert selected.is_parent_eligible is True


def test_bootstrap_seed_parent_uses_routing_scaffold_only_as_tiebreaker(tmp_path):
    db = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    common_metrics = {
        "seed_baseline": True,
        "structural_failure_count": 0,
        "severe_regression_count": 1,
        "tier2_average_rank": 4.0,
        "worst_design_hpwl_delta_pct": 20.0,
        "hpwl_delta_pct": 10.0,
        "overflow_delta_pct": -10.0,
        "hpwl_gate_passed": False,
        "overflow_gate_passed": True,
        "negative_memory_only": True,
        "parent_eligible": False,
    }
    db.add(
        ObjectiveProgram.from_spec(
            objective_preset("dreamplace_wawl_electric"),
            program_id="plain_seed",
            metrics=dict(common_metrics),
        )
    )
    db.add(
        ObjectiveProgram.from_spec(
            objective_preset("dreamplace_routing_aware_smoke"),
            program_id="routing_seed",
            metrics=dict(common_metrics),
        )
    )

    selected = _promote_bootstrap_seed_parent(db)

    assert selected is not None
    assert selected.id == "routing_seed"
    assert selected.metrics["bootstrap_parent"] is True
    assert selected.metrics["manual_safe_baseline"] is True
    assert selected.is_parent_eligible is True


def test_bootstrap_seed_parent_does_not_promote_structural_failure(tmp_path):
    db = ObjectiveProgramDatabase(
        tmp_path / "db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    db.add(
        ObjectiveProgram.from_spec(
            objective_preset("dreamplace_wawl_electric"),
            program_id="failed_seed",
            status="failed",
            metrics={
                "seed_baseline": True,
                "structural_failure_count": 1,
                "hpwl_delta_pct": -1.0,
                "overflow_delta_pct": -1.0,
                "negative_memory_only": True,
                "parent_eligible": False,
            },
        )
    )

    assert _promote_bootstrap_seed_parent(db) is None
    assert db.get("failed_seed").is_parent_eligible is False


def test_calibrated_spec_rewrites_route_coefficients():
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength")',
                '    den = term("density")',
                '    route = term("soft_rudy_pnorm")',
                "    score = wl + den + 0.02 * route",
                '    return score, {"wirelength": wl, "density": den, "route": route}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )

    calibrated = _calibrated_spec(
        spec,
        route_coeff=0.005,
        density_coeff=1.25,
        wirelength_coeff=1.02,
    )

    assert calibrated.id != spec.id
    assert "soft_rudy_pnorm" in calibrated.term_set
    assert any(abs(value - 0.005) < 1e-12 for value in calibrated.constants.values())
    assert any(abs(value - 1.25) < 1e-12 for value in calibrated.constants.values())
    assert any(abs(value - 1.02) < 1e-12 for value in calibrated.constants.values())
    assert set(calibrated.components or {}) == {
        "wirelength",
        "density",
        "route",
        "routing_correction",
    }
    assert (calibrated.components or {})["routing_correction"] == {
        "op": "mul",
        "args": [
            {"op": "const", "value": 0.005},
            {"op": "term", "name": "soft_rudy_pnorm"},
        ],
    }


def test_calibrated_spec_preserves_native_residual_unary_route_shape():
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    native = term("native_objective")',
                '    route = term("soft_rudy_pnorm")',
                "    score = native + 0.003 * log1p(route)",
                '    return score, {"native": native, "route_hotspot": route}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace",
    )

    calibrated = _calibrated_spec(
        spec,
        route_coeff=0.00001,
        density_coeff=None,
        wirelength_coeff=None,
    )
    coeffs, unsupported = _route_coefficients(calibrated.ast)

    assert unsupported == set()
    assert coeffs == [("soft_rudy_pnorm", 0.00001)]
    assert calibrated.ast == {
        "op": "add",
        "args": [
            {"op": "term", "name": "native_objective"},
            {
                "op": "mul",
                "args": [
                    {"op": "const", "value": 0.00001},
                    {
                        "op": "log1p",
                        "args": [{"op": "term", "name": "soft_rudy_pnorm"}],
                    },
                ],
            },
        ],
    }


def test_calibrated_spec_keeps_identity_base_coefficients_unwrapped():
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength")',
                '    den = term("density")',
                '    route = term("route_pressure_long")',
                '    pins = term("pin_density_pnorm")',
                "    score = wl + den + 0.005 * log1p(route) + 0.003 * softplus(pins)",
                '    return score, {"wirelength": wl, "density": den, "route": route, "pins": pins}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )

    calibrated = _calibrated_spec(
        spec,
        route_coeff=1e-8,
        density_coeff=1.0,
        wirelength_coeff=1.0,
    )
    coeffs, unsupported = _route_coefficients(calibrated.ast)

    assert unsupported == set()
    assert calibrated.complexity <= spec.complexity
    assert coeffs == [("route_pressure_long", 1e-8), ("pin_density_pnorm", 1e-8)]
    assert calibrated.ast["args"][0] == {"op": "term", "name": "wirelength"}
    assert calibrated.ast["args"][1] == {"op": "term", "name": "density"}


def test_calibration_combinations_preserve_route_diversity_under_budget():
    pairs = _calibration_combinations(
        route_coeffs=[-1e-8, -1e-10, -1e-12, 0.0, 1e-12, 1e-10, 1e-8],
        density_coeffs=[1.2, 1.25, 1.26, 1.28, 1.3, 1.35],
        wirelength_coeffs=[1.0, 1.01, 1.02],
        max_count=6,
    )

    route_values = {route for route, _density, _wirelength in pairs}
    wirelength_values = {wirelength for _route, _density, wirelength in pairs}

    assert len(pairs) == 6
    assert 0.0 in route_values
    assert any((route or 0.0) > 0.0 for route in route_values)
    assert any((route or 0.0) < 0.0 for route in route_values)
    assert 1.0 in wirelength_values


def test_calibration_combinations_try_route_only_before_density_retuning():
    pairs = _calibration_combinations(
        route_coeffs=[-1e-6, -1e-8, 1e-8, 1e-6],
        density_coeffs=[1.0, 1.1, 1.25, 1.35],
        wirelength_coeffs=[1.0],
        max_count=6,
    )

    assert pairs[:4] == [
        (1e-8, 1.0, 1.0),
        (-1e-8, 1.0, 1.0),
        (1e-6, 1.0, 1.0),
        (-1e-6, 1.0, 1.0),
    ]
    assert any(density != 1.0 for _route, density, _wirelength in pairs[4:])


def test_calibration_combinations_order_small_coefficients_even_when_all_fit():
    pairs = _calibration_combinations(
        route_coeffs=sorted({-1e-6, -1e-8, -1e-10, -1e-12, -1e-14, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6}),
        density_coeffs=[1.0],
        wirelength_coeffs=[1.0],
        max_count=10,
    )

    assert pairs[:6] == [
        (1e-14, 1.0, 1.0),
        (-1e-14, 1.0, 1.0),
        (1e-12, 1.0, 1.0),
        (-1e-12, 1.0, 1.0),
        (1e-10, 1.0, 1.0),
        (-1e-10, 1.0, 1.0),
    ]


def test_term_calibration_grid_uses_gradient_ratios(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        term_calibration_gradient_ratios=[0.001, 0.01],
        route_coeff_cap=1.0,
        term_calibration_min_grad_norm=1e-20,
    )
    payload = {
        "results": [
            {
                "term": "native_objective",
                "status": "passed",
                "gradient_norm": 100.0,
            },
            {
                "term": "soft_rudy_mean",
                "status": "passed",
                "gradient_norm": 10.0,
            },
            {
                "term": "route_pressure_long",
                "status": "passed",
                "gradient_norm": 20.0,
            },
        ]
    }

    grid = _derive_calibrated_route_coeff_grid(payload, config)

    assert grid == [-0.005, 0.005, -0.05, 0.05]
    assert payload["per_term_suggested_positive_coefficients"]["soft_rudy_mean"] == [
        0.01,
        0.1,
    ]
    assert payload["per_term_suggested_positive_coefficients"]["route_pressure_long"] == [
        0.005,
        0.05,
    ]


def test_replacement_mode_evaluates_calibrated_siblings(monkeypatch, tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        objective_mode="replacement",
        route_coeff_grid=[-1e-10, 1e-10],
        density_coeff_grid=[1.0],
        wirelength_coeff_grid=[1.0],
        max_calibrated_siblings=2,
        route_coeff_cap=0.01,
    )
    parent_spec = objective_preset("dreamplace_wawl_electric")
    parent = ObjectiveProgram.from_spec(
        parent_spec,
        program_id="parent",
        metrics={"combined_score": 1.0, "parent_eligible": True},
    )
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength_wawl")',
                '    den = term("density_electric")',
                '    route = term("soft_rudy_pnorm")',
                "    score = wl + den + 0.005 * route",
                '    return score, {"wirelength": wl, "density": den, "route": route}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )
    raw_program = ObjectiveProgram.from_spec(
        spec,
        program_id="raw",
        parent_id=parent.id,
        metrics={"hpwl_delta_pct": 700.0, "overflow_delta_pct": -40.0},
    )
    database = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )

    def fake_evaluate_many(**kwargs):
        result = {}
        for candidate in kwargs["specs"]:
            result[candidate.id] = (
                {
                    "combined_score": 0.5,
                    "constrained_score": 0.5,
                    "structural_failure_count": 0,
                    "hpwl_delta_pct": 1.0,
                    "overflow_delta_pct": -2.0,
                    "hpwl_gate_passed": True,
                    "overflow_gate_passed": True,
                    "elite_gate_passed": True,
                    "parent_eligible": True,
                    "negative_memory_only": False,
                },
                {"candidate_rows": []},
                "accepted",
                None,
            )
        return result

    monkeypatch.setattr(openevolve_tier2, "_evaluate_search_panel_many", fake_evaluate_many)

    siblings = _evaluate_calibrated_siblings(
        spec=spec,
        raw_program=raw_program,
        provider_trace=SimpleNamespace(
            metadata={"provider": "mock"},
            usage={},
            messages=[],
            raw_response={},
        ),
        config=config,
        parent=parent,
        iteration=1,
        iteration_dir=tmp_path / "iteration_0001",
        dreamplace_root=tmp_path,
        resume=False,
        retry_failed=False,
        database=database,
        prompt_context_path=tmp_path / "prompt_context.json",
    )

    route_coefficients = {
        program.metrics["calibration_report"]["route_coefficient"]
        for program in siblings
    }

    assert len(siblings) == 2
    assert route_coefficients == {-1e-10, 1e-10}
    assert all(program.metrics["operator"] == "param_tune" for program in siblings)
    assert all(program.metrics["calibration_report"]["parent_program_id"] == "raw" for program in siblings)


def test_scale_rejected_candidate_still_runs_calibrated_siblings(monkeypatch, tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        objective_mode="replacement",
        require_nonzero_routing_term=True,
        min_replacement_nonbaseline_terms=2,
        route_coeff_cap=0.01,
        route_coeff_grid=[1e-8],
        density_coeff_grid=[1.0],
        wirelength_coeff_grid=[1.0],
        max_generation_attempts=1,
    )
    parent = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_wawl_electric"),
        program_id="parent",
        metrics={"combined_score": 1.0, "parent_eligible": True},
        status="accepted",
    )
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength_wawl")',
                '    den = term("density_electric")',
                '    route = term("route_pressure_long")',
                '    pins = term("pin_density_pnorm")',
                "    score = wl + den + 0.05 * log1p(route * pins)",
                '    return score, {"wirelength": wl, "density": den, "route": route, "pins": pins}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )
    database = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )

    class FakeProvider:
        def generate_traced(self, *, context, term_scope, mutation_mode):
            return SimpleNamespace(
                spec=spec,
                metadata={"provider": "fake"},
                usage={},
                messages=context["prompt_messages"],
                raw_response={"objective_program": spec.source_program},
            )

    def fake_calibration(**kwargs):
        sibling = ObjectiveProgram.from_spec(
            spec,
            program_id="calibrated_child",
            parent_id=parent.id,
            metrics={
                "combined_score": 0.1,
                "parent_eligible": True,
                "hpwl_gate_passed": True,
                "overflow_gate_passed": True,
                "negative_memory_only": False,
            },
            status="accepted",
        )
        sibling.metrics["calibration_report"] = {"route_coefficient": 1e-8}
        return [sibling]

    monkeypatch.setattr(openevolve_tier2, "_evaluate_calibrated_siblings", fake_calibration)

    programs = _generate_and_evaluate_children(
        provider=FakeProvider(),
        config=config,
        context={
            "prompt_messages": [
                {"role": "system", "content": "You are evolving objectives."},
                {"role": "user", "content": "Use route_pressure_long and pin_density_pnorm."},
            ]
        },
        parent=parent,
        iteration=1,
        iteration_dir=tmp_path / "iteration_0001",
        dreamplace_root=tmp_path,
        resume=False,
        retry_failed=False,
        database=database,
    )

    assert len(programs) == 2
    assert programs[0].status == "rejected"
    assert programs[0].metrics["scale_rejected_but_calibrated"] is True
    assert programs[0].metrics["calibrated_sibling_count"] == 1
    assert programs[1].id == "calibrated_child"


def test_generate_and_evaluate_children_creates_k_independent_samples(monkeypatch, tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        samples_per_iteration=3,
        max_generation_attempts=1,
        max_calibrated_siblings=0,
        require_nonzero_routing_term=False,
        reject_baseline_mechanism_clones=False,
        reject_repeated_negative_mechanisms=False,
        min_replacement_nonbaseline_terms=0,
        route_coeff_cap=1.0,
    )
    parent = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_wl_density"),
        program_id="parent",
        metrics={"combined_score": 1.0, "parent_eligible": True},
        status="accepted",
    )
    database = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=20, archive_size=5, num_islands=1),
    )
    database.add(parent, target_island=0)

    class FakeProvider:
        def __init__(self):
            self.calls = 0

        def generate_traced(self, *, context, term_scope, mutation_mode):
            self.calls += 1
            coeff = 0.01 * self.calls
            spec = parse_objective_program(
                "\n".join(
                    [
                        "def objective(features):",
                        '    wl = term("wirelength_wawl")',
                        '    den = term("density_bell")',
                        '    route = term("route_pressure_long")',
                        f"    score = wl + den + {coeff:.6f} * log1p(route)",
                        '    return score, {"wirelength": wl, "density": den, "route": route}',
                    ]
                ),
                created_by="test",
                parent_ids=[parent.id],
                term_scope="dreamplace_replacement",
            )
            return SimpleNamespace(
                spec=spec,
                messages=context["prompt_messages"],
                raw_response={"sample": self.calls},
                usage={},
                metadata={"provider": "fake", "resolved_model": "fake"},
            )

    def fake_evaluate(**kwargs):
        return (
            {
                "combined_score": 1.0,
                "constrained_score": 1.0,
                "hpwl_delta_pct": -0.1,
                "overflow_delta_pct": -0.2,
                "parent_eligible": True,
                "negative_memory_only": False,
                "structural_failure_count": 0,
                "severe_regression_count": 0,
                "component_summary": {"route": {"trend": "up"}},
            },
            {"objective_path": str(kwargs["objective_path"])},
            "accepted",
            None,
        )

    monkeypatch.setattr(openevolve_tier2, "_evaluate_search_panel", fake_evaluate)

    programs = _generate_and_evaluate_children(
        provider=FakeProvider(),
        config=config,
        context={
            "prompt_messages": [
                {"role": "system", "content": "You are evolving objectives."},
                {"role": "user", "content": "Return one valid objective."},
            ],
            "mutation_mode": "full",
        },
        parent=parent,
        iteration=1,
        iteration_dir=tmp_path / "iteration_0001",
        dreamplace_root=tmp_path,
        resume=False,
        retry_failed=False,
        database=database,
    )

    assert len(programs) == 3
    assert {program.metrics["sample_index"] for program in programs} == {0, 1, 2}
    assert (tmp_path / "iteration_0001" / "sample_0000" / "prompt_context.json").is_file()
    assert (tmp_path / "iteration_0001" / "sample_0002" / "raw_response.json").is_file()


def test_database_objective_identity_tracks_spec_id_and_ast(tmp_path):
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength")',
                '    den = term("density")',
                "    score = wl + 1.25 * den",
                '    return score, {"wirelength": wl, "density": den}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    db.add(
        openevolve_tier2.ObjectiveProgram.from_spec(
            spec,
            program_id=f"iter_0001_{spec.id}_calib_route_0_density_1p25",
            metrics={"combined_score": 1.0, "parent_eligible": True},
        )
    )

    objective_ids, ast_signatures = _database_objective_identity(db)

    assert spec.id in objective_ids
    assert f"iter_0001_{spec.id}_calib_route_0_density_1p25" in objective_ids
    assert _ast_signature(spec.ast) in ast_signatures


def test_tier3_selection_prefers_robust_elites(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        tier3_top_k=1,
    )
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    aggregate_spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength")',
                '    den = term("density")',
                '    route = term("soft_rudy_mean")',
                "    score = wl + 1.35 * den - 1e-10 * route",
                '    return score, {"wirelength": wl, "density": den, "route": route}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )
    elite_spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength")',
                '    den = term("density")',
                '    route = term("soft_rudy_pnorm")',
                "    score = wl + 1.35 * den + 1e-10 * route",
                '    return score, {"wirelength": wl, "density": den, "route": route}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )
    db.add(
        openevolve_tier2.ObjectiveProgram.from_spec(
            aggregate_spec,
            program_id="aggregate",
            metrics={
                "combined_score": 0.0,
                "aggregate_tier2_winner": True,
                "routing_aware_tier2_candidate": True,
                "hpwl_gate_passed": True,
                "overflow_gate_passed": True,
                "elite_gate_passed": False,
                "negative_memory_only": True,
                "tier2_average_rank": 1.0,
            },
        )
    )
    db.add(
        openevolve_tier2.ObjectiveProgram.from_spec(
            elite_spec,
            program_id="elite",
            metrics={
                "combined_score": 0.1,
                "aggregate_tier2_winner": True,
                "routing_aware_tier2_candidate": True,
                "hpwl_gate_passed": True,
                "overflow_gate_passed": True,
                "elite_gate_passed": True,
                "negative_memory_only": False,
                "tier2_average_rank": 2.0,
            },
        )
    )

    selected, policy = _tier3_candidate_programs(db, config)

    assert [program.id for program in selected] == ["elite"]
    assert "robust" in policy


def test_tier3_selection_falls_back_to_aggregate_routing_winners(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        tier3_top_k=2,
        tier3_candidate_policy="elite_or_aggregate",
    )
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    aggregate_spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength")',
                '    den = term("density")',
                '    route = term("soft_rudy_mean")',
                "    score = wl + 1.35 * den - 1e-10 * route",
                '    return score, {"wirelength": wl, "density": den, "route": route}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )
    rejected_spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength")',
                '    den = term("density")',
                '    route = term("soft_rudy_pnorm")',
                "    score = wl + 1.35 * den + 1e-10 * route",
                '    return score, {"wirelength": wl, "density": den, "route": route}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )
    db.add(
        openevolve_tier2.ObjectiveProgram.from_spec(
            aggregate_spec,
            program_id="aggregate",
            metrics={
                "combined_score": 0.0,
                "aggregate_tier2_winner": True,
                "routing_aware_tier2_candidate": True,
                "hpwl_gate_passed": True,
                "overflow_gate_passed": True,
                "elite_gate_passed": False,
                "negative_memory_only": True,
                "tier2_average_rank": 1.0,
                "worst_design_hpwl_delta_pct": 0.0001,
                "worst_design_overflow_delta_pct": 0.0002,
            },
        )
    )
    db.add(
        openevolve_tier2.ObjectiveProgram.from_spec(
            rejected_spec,
            program_id="failed",
            metrics={
                "combined_score": 0.0,
                "aggregate_tier2_winner": False,
                "routing_aware_tier2_candidate": True,
                "hpwl_gate_passed": True,
                "overflow_gate_passed": False,
                "elite_gate_passed": False,
                "negative_memory_only": True,
                "tier2_average_rank": 0.5,
            },
        )
    )

    selected, policy = _tier3_candidate_programs(db, config)

    assert [program.id for program in selected] == ["aggregate"]
    assert "aggregate routing-aware" in policy


def test_tier3_selection_can_require_elites_only(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        tier3_top_k=2,
        tier3_candidate_policy="elite_only",
    )
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength")',
                '    den = term("density")',
                '    route = term("soft_rudy_mean")',
                "    score = wl + 1.35 * den - 1e-10 * route",
                '    return score, {"wirelength": wl, "density": den, "route": route}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )
    db.add(
        openevolve_tier2.ObjectiveProgram.from_spec(
            spec,
            program_id="aggregate",
            metrics={
                "combined_score": 0.0,
                "aggregate_tier2_winner": True,
                "routing_aware_tier2_candidate": True,
                "hpwl_gate_passed": True,
                "overflow_gate_passed": True,
                "elite_gate_passed": False,
                "negative_memory_only": True,
                "tier2_average_rank": 1.0,
            },
        )
    )

    selected, policy = _tier3_candidate_programs(db, config)

    assert selected == []
    assert "robust" in policy


def test_final_tier2_candidate_ids_include_manifest_candidates(tmp_path: Path) -> None:
    manifest = tmp_path / "objective_manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "objective_id": "default",
                    "objective_path": None,
                    "source": "native_dreamplace",
                },
                {
                    "objective_id": "obj_final_candidate",
                    "objective_path": str(tmp_path / "final_candidates" / "program.json"),
                    "source": "file",
                },
                {
                    "objective_id": "obj_fixed_reference",
                    "objective_path": str(tmp_path / "final_baselines" / "fixed.json"),
                    "source": "file",
                },
            ]
        ),
        encoding="utf-8",
    )

    selected = _final_tier2_candidate_objective_ids({"objective_manifest": str(manifest)})

    assert selected == {"obj_final_candidate"}


def test_refresh_database_scores_rebuilds_best_under_current_policy(tmp_path):
    config = replace(
        load_openevolve_tier2_config(_config(tmp_path, _shared_panel(tmp_path))),
        hpwl_gate_search_pct=0.0,
        hpwl_gate_elite_pct=0.0,
    )
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    unsafe = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength")',
                '    den = term("density")',
                "    score = wl + 1.35 * den",
                '    return score, {"wirelength": wl, "density": den}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )
    safe = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength")',
                '    den = term("density")',
                "    score = wl + 1.25 * den",
                '    return score, {"wirelength": wl, "density": den}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace_replacement",
    )
    db.add(
        openevolve_tier2.ObjectiveProgram.from_spec(
            unsafe,
            program_id="unsafe_old_best",
            metrics={
                "combined_score": 99.0,
                "tier2_average_rank": 1.0,
                "hpwl_delta_pct": 0.1,
                "overflow_delta_pct": -1.0,
                "parent_eligible": True,
            },
        )
    )
    db.add(
        openevolve_tier2.ObjectiveProgram.from_spec(
            safe,
            program_id="safe_improver",
            metrics={
                "combined_score": 1.0,
                "tier2_average_rank": 2.0,
                "hpwl_delta_pct": -0.1,
                "overflow_delta_pct": -0.1,
                "parent_eligible": True,
            },
        )
    )

    _refresh_database_scores(db, config)

    assert db.get("unsafe_old_best").is_parent_eligible is False
    assert db.get("safe_improver").is_parent_eligible is True
    assert db.best_program_id == "safe_improver"


def test_openevolve_tier2_mock_loop_creates_memory_artifacts(monkeypatch, tmp_path):
    panel = _shared_panel(tmp_path)
    config = _config(tmp_path, panel)

    def fake_tier2(**kwargs):
        run_dir = Path(kwargs["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        comparison_csv = run_dir / "comparison_table.csv"
        rows = [
            {
                "design": "bp_fe",
                "objective_id": "default",
                "seed": 1000,
                "status": "success",
                "failure_stage": "",
                "hpwl_delta_pct": 0.0,
                "overflow_delta_pct": 0.0,
                "runtime_seconds": 10,
                "custom_grad_norm": "",
                "output_artifact": str(tmp_path / "default.def"),
            },
            {
                "design": "bp_fe",
                "objective_id": "custom_default",
                "seed": 1000,
                "status": "success",
                "failure_stage": "",
                "hpwl_delta_pct": 0.0,
                "overflow_delta_pct": 0.0,
                "runtime_seconds": 10,
                "custom_grad_norm": 1.0,
                "output_artifact": str(tmp_path / "custom.def"),
            },
        ]
        for objective_path in kwargs["objective_paths"]:
            spec = load_objective_spec(objective_path)
            rows.append(
                {
                    "design": "bp_fe",
                    "objective_id": spec.id,
                    "seed": 1000,
                    "status": "success",
                    "failure_stage": "",
                    "hpwl_delta_pct": -1.5,
                    "overflow_delta_pct": -2.5,
                    "runtime_seconds": 11,
                    "custom_grad_norm": 2.0,
                    "output_artifact": str(tmp_path / f"{spec.id}.def"),
                }
            )
        with comparison_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return {
            "comparison_csv": str(comparison_csv),
            "success_count": len(rows),
            "failure_count": 0,
            "result_count": len(rows),
        }

    monkeypatch.setattr(openevolve_tier2, "run_tier2_dreamplace", fake_tier2)

    summary = run_openevolve_tier2(
        config_path=config,
        run_dir=tmp_path / "run",
        dreamplace_root=tmp_path,
    )

    assert summary["program_count"] >= 3
    assert summary["best_program_id"] is not None
    assert summary["final_tier2"] is None
    assert (tmp_path / "run" / "program_db" / "metadata.json").exists()
    assert (tmp_path / "run" / "evolution_trace.jsonl").exists()
    assert (tmp_path / "run" / "best_program.py").exists()
    assert (tmp_path / "run" / "checkpoints" / "checkpoint_0001").exists()

    second_context = json.loads(
        (tmp_path / "run" / "iteration_0002" / "prompt_context.json").read_text(
            encoding="utf-8"
        )
    )
    assert second_context["parent_program"]["code"]
    assert second_context["top_safe_programs"]
    assert second_context["structured_feedback_table"]
    assert second_context["objective_mode"] == "replacement"
    assert second_context["memory_source"] == "explicit_archive_context"


def test_openevolve_tier2_runs_heldout_generalization_after_evolution(monkeypatch, tmp_path):
    search_panel = _shared_panel(tmp_path)
    heldout_panel = _generalization_panel(tmp_path)
    config = tmp_path / "openevolve_generalization.toml"
    config.write_text(
        "\n".join(
            [
                'provider = "mock"',
                'term_scope = "dreamplace_replacement"',
                "max_iterations = 1",
                "population_size = 20",
                "archive_size = 5",
                "num_islands = 2",
                "checkpoint_interval = 1",
                'target_designs = ["bp_fe"]',
                'initial_presets = ["dreamplace_explicit_wl_density", "dreamplace_routing_aware_smoke"]',
                'primary_baseline = "default"',
                "require_nonzero_routing_term = true",
                "min_replacement_nonbaseline_terms = 2",
                "route_coeff_cap = 0.01",
                "",
                "[search]",
                f'panel = "{search_panel.as_posix()}"',
                'baseline_presets = ["dreamplace_explicit_wl_density"]',
                "",
                "[generalization]",
                "enabled = true",
                f'panel = "{heldout_panel.as_posix()}"',
                "top_k = 2",
                'baseline_presets = ["dreamplace_explicit_wl_density"]',
                "",
                "[final]",
                "enabled = false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    calls = []

    def fake_tier2(**kwargs):
        run_dir = Path(kwargs["run_dir"])
        calls.append(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        comparison_csv = run_dir / "comparison_table.csv"
        rows = [
            {
                "design": "bp_fe",
                "objective_id": "default",
                "seed": 1000,
                "status": "success",
                "failure_stage": "",
                "hpwl_delta_pct": 0.0,
                "overflow_delta_pct": 0.0,
                "runtime_seconds": 10,
                "custom_grad_norm": "",
                "output_artifact": str(tmp_path / "default.def"),
            },
            {
                "design": "bp_fe",
                "objective_id": "custom_default",
                "seed": 1000,
                "status": "success",
                "failure_stage": "",
                "hpwl_delta_pct": 0.0,
                "overflow_delta_pct": 0.0,
                "runtime_seconds": 10,
                "custom_grad_norm": 1.0,
                "output_artifact": str(tmp_path / "custom.def"),
            },
        ]
        for objective_path in kwargs["objective_paths"]:
            spec = load_objective_spec(objective_path)
            rows.append(
                {
                    "design": "bp_fe",
                    "objective_id": spec.id,
                    "seed": 1000,
                    "status": "success",
                    "failure_stage": "",
                    "hpwl_delta_pct": -1.0,
                    "overflow_delta_pct": -1.0,
                    "runtime_seconds": 11,
                    "custom_grad_norm": 2.0,
                    "output_artifact": str(tmp_path / f"{spec.id}.def"),
                }
            )
        with comparison_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return {
            "comparison_csv": str(comparison_csv),
            "metrics_csv": str(comparison_csv),
            "success_count": len(rows),
            "failure_count": 0,
            "result_count": len(rows),
        }

    monkeypatch.setattr(openevolve_tier2, "run_tier2_dreamplace", fake_tier2)

    summary = run_openevolve_tier2(
        config_path=config,
        run_dir=tmp_path / "run",
        dreamplace_root=tmp_path,
    )

    assert summary["generalization_tier2"] is not None
    assert summary["generalization_tier2"]["held_out_from_evolution"] is True
    assert summary["design_profiles"]
    assert (tmp_path / "run" / "design_profiles.json").is_file()
    assert any(path.name == "generalization_tier2" for path in calls)
    assert (tmp_path / "run" / "generalization_tier2" / "generalization_report.md").exists()
    prompt_context = json.loads(
        (tmp_path / "run" / "iteration_0001" / "prompt_context.json").read_text(
            encoding="utf-8"
        )
    )
    prompt_text = "\n".join(message["content"] for message in prompt_context["prompt_messages"])
    assert "generalization_designs_withheld_from_feedback" in prompt_text
    assert "held out from parent selection" in prompt_text
