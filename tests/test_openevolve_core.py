import json
import random

import pytest

from coevop.evolution.openevolve_core import (
    DEFAULT_FEATURE_DIMENSIONS,
    ObjectiveDatabaseConfig,
    ObjectiveProgram,
    ObjectiveProgramDatabase,
    ObjectivePromptSampler,
    _hpwl_delta_bucket,
    _mechanism_family,
    _opentimer_wns_delta_bucket,
    _overflow_delta_bucket,
    _parent_sample_weights,
)
from coevop.objectives.presets import objective_preset


def test_objective_program_database_persists_memory(tmp_path):
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=2),
    )
    seed = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_wl_density"),
        metrics={"combined_score": 0.2},
        status="accepted",
    )
    child = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_density_heavy"),
        parent_id=seed.id,
        generation=1,
        iteration_found=1,
        metrics={"combined_score": 0.8, "hpwl_delta_pct": -1.0, "overflow_delta_pct": -2.0},
        artifacts={"dreamplace_log": "finite gradient"},
        status="accepted",
    )

    db.add(seed, target_island=0)
    db.add(child, target_island=0)
    db.save(iteration=1)

    loaded = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=2),
    )
    loaded.load()

    assert loaded.best_program_id == child.id
    assert loaded.get(child.id).parent_id == seed.id
    assert loaded.get(child.id).artifacts["dreamplace_log"] == "finite gradient"
    assert loaded.last_iteration == 1


def test_archive_preserves_mechanism_diversity_before_score_clones(tmp_path):
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=2, num_islands=1),
    )

    def candidate(program_id: str, score: float, preset: str) -> ObjectiveProgram:
        return ObjectiveProgram.from_spec(
            objective_preset(preset),
            program_id=program_id,
            generation=1,
            metrics={
                "combined_score": score,
                "elite_gate_passed": True,
                "negative_memory_only": False,
                "hpwl_delta_pct": -1.0,
                "overflow_delta_pct": -2.0,
            },
            status="accepted",
        )

    db.add(candidate("clone_a", 10.0, "dreamplace_wawl_density_bell"), target_island=0)
    db.add(candidate("clone_b", 9.0, "dreamplace_lse_density_bell"), target_island=0)
    db.add(candidate("diverse", 1.0, "dreamplace_route_pressure_long"), target_island=0)

    assert db.archive == {"clone_a", "diverse"}


def test_map_elites_schema_v2_default_dimensions_and_buckets():
    config = ObjectiveDatabaseConfig()

    assert config.feature_schema_version == 2
    assert config.feature_dimensions == DEFAULT_FEATURE_DIMENSIONS
    assert _hpwl_delta_bucket(-1.0) == "hpwl_improved"
    assert _hpwl_delta_bucket(0.5) == "hpwl_neutral"
    assert _hpwl_delta_bucket(1.0) == "hpwl_regressed"
    assert _overflow_delta_bucket(-5.0) == "overflow_improved"
    assert _overflow_delta_bucket(2.0) == "overflow_neutral"
    assert _overflow_delta_bucket(5.0) == "overflow_regressed"
    assert (
        _opentimer_wns_delta_bucket(
            {"timing_proxy_status": "success", "timing_proxy_wns_delta": 0.05},
            min_abs_ns=0.05,
        )
        == "wns_improved"
    )
    assert (
        _opentimer_wns_delta_bucket(
            {"timing_proxy_status": "success", "timing_proxy_wns_delta": -0.05},
            min_abs_ns=0.05,
        )
        == "wns_regressed"
    )
    assert (
        _opentimer_wns_delta_bucket(
            {"timing_proxy_status": "partial"},
            min_abs_ns=0.05,
        )
        == "timing_unknown"
    )


def test_mechanism_family_uses_low_cardinality_term_buckets():
    assert _mechanism_family({"soft_rudy_pnorm", "pin_density_pnorm"}) == "route_pin_hybrid"
    assert _mechanism_family({"route_pressure_long"}) == "long_route_pressure"
    assert _mechanism_family({"soft_rudy_pnorm"}) == "route_hotspot"
    assert _mechanism_family({"soft_rudy_mean"}) == "route_mean"
    assert _mechanism_family({"pin_density_pnorm"}) == "pin_pressure"
    assert _mechanism_family({"pin_count_weighted_wl"}) == "fanout_weighted_wl"
    assert _mechanism_family({"wirelength_lse", "density_bell"}) == "wl_density_variant"
    assert _mechanism_family({"wirelength_wawl", "density_electric"}) == "wl_density"


def test_map_elites_keeps_best_program_per_island_cell(tmp_path):
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=5, num_islands=1),
    )

    def candidate(program_id: str, score: float) -> ObjectiveProgram:
        return ObjectiveProgram.from_spec(
            objective_preset("dreamplace_routing_aware_smoke"),
            program_id=program_id,
            metrics={
                "combined_score": score,
                "hpwl_delta_pct": -1.0,
                "overflow_delta_pct": -2.0,
                "mechanism_signature": "same_cell",
            },
            status="accepted",
        )

    db.add(candidate("lower", 0.2), target_island=0)
    db.add(candidate("higher", 0.8), target_island=0)

    assert list(db.island_feature_maps[0].values()) == ["higher"]
    assert db.archive == {"higher"}


def test_map_elites_schema_v2_spreads_across_mechanism_and_timing_cells(tmp_path):
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=20, archive_size=10, num_islands=1),
    )

    cases = [
        ("wl", "dreamplace_wawl_density_bell", 0.5, "success", 0.0),
        ("route", "dreamplace_route_pressure_long", 0.6, "success", 0.08),
        ("fanout", "dreamplace_pin_count_weighted_wl", 0.7, "success", -0.08),
    ]
    for program_id, preset, score, timing_status, wns_delta in cases:
        db.add(
            ObjectiveProgram.from_spec(
                objective_preset(preset),
                program_id=program_id,
                metrics={
                    "combined_score": score,
                    "hpwl_delta_pct": -2.0,
                    "overflow_delta_pct": -6.0,
                    "timing_proxy_status": timing_status,
                    "timing_proxy_wns_delta": wns_delta,
                },
                status="accepted",
            ),
            target_island=0,
        )

    occupied = [db.get(program_id).feature_coords for program_id in db.archive]
    assert len(db.island_feature_maps[0]) >= 3
    assert len({item["mechanism_family"] for item in occupied}) >= 3
    assert len({item["opentimer_wns_delta_bucket"] for item in occupied}) >= 2


def test_map_elites_persists_feature_maps_and_migrations(tmp_path):
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(
            population_size=10,
            archive_size=5,
            num_islands=2,
            migration_interval=1,
            migration_rate=1.0,
        ),
    )
    parent = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_routing_aware_smoke"),
        program_id="route_parent",
        metrics={
            "combined_score": 1.0,
            "hpwl_delta_pct": -1.0,
            "overflow_delta_pct": -2.0,
            "mechanism_signature": "route_parent",
        },
        status="accepted",
    )
    db.add(parent, target_island=0)
    db.increment_island_generation(0)
    migrants = db.maybe_migrate(iteration=1, rng=random.Random(1))
    db.save(iteration=1)

    loaded = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(
            population_size=10,
            archive_size=5,
            num_islands=2,
            migration_interval=1,
            migration_rate=1.0,
        ),
    )
    loaded.load()

    assert migrants
    assert loaded.island_feature_maps[0]
    assert loaded.island_feature_maps[1]
    assert loaded.migration_events[-1]["migrated"] == 1


def test_bidirectional_ring_migration_targets_both_neighbors(tmp_path):
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(
            population_size=20,
            archive_size=10,
            num_islands=3,
            migration_interval=1,
            migration_rate=1.0,
        ),
    )
    parent = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_route_pressure_long"),
        program_id="route_parent",
        metrics={"combined_score": 1.0, "hpwl_delta_pct": -2.0, "overflow_delta_pct": -6.0},
        status="accepted",
    )
    db.add(parent, target_island=0)
    db.increment_island_generation(0)

    migrants = db.maybe_migrate(iteration=1, rng=random.Random(1))

    assert {migrant.metrics["target_island"] for migrant in migrants} == {1, 2}
    assert all(migrant.parent_id == "route_parent" for migrant in migrants)


def test_schema_recode_recovers_old_programs_and_excludes_unrecoverable(tmp_path):
    root = tmp_path / "program_db"
    old_db = ObjectiveProgramDatabase(
        root,
        ObjectiveDatabaseConfig(
            population_size=20,
            archive_size=10,
            num_islands=1,
            feature_dimensions=(
                "complexity",
                "term_signature",
                "hpwl_delta_pct",
                "overflow_delta_pct",
                "gradient_stability",
            ),
            feature_schema_version=1,
        ),
    )
    recoverable = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_route_pressure_long"),
        program_id="recoverable",
        metrics={"combined_score": 1.0, "hpwl_delta_pct": -2.0, "overflow_delta_pct": -6.0},
        status="accepted",
    )
    ast_recoverable = ObjectiveProgram(
        id="ast_recoverable",
        code="",
        objective_spec=recoverable.objective_spec,
        metrics={"combined_score": 0.8, "hpwl_delta_pct": 2.0, "overflow_delta_pct": 6.0},
        status="accepted",
        complexity=recoverable.complexity,
        term_set=[],
    )
    unrecoverable = ObjectiveProgram(
        id="unrecoverable",
        code="",
        metrics={"combined_score": 0.9},
        status="accepted",
    )
    old_db.add(recoverable, target_island=0)
    old_db.add(ast_recoverable, target_island=0)
    old_db.add(unrecoverable, target_island=0)
    old_db.save(iteration=1)

    new_db = ObjectiveProgramDatabase(
        root,
        ObjectiveDatabaseConfig(population_size=20, archive_size=10, num_islands=1),
    )
    new_db.load()

    assert new_db.get("recoverable").feature_coords["mechanism_family"] == "long_route_pressure"
    assert new_db.get("ast_recoverable").term_set
    assert new_db.get("unrecoverable").metrics["feature_recode_failed"] is True
    assert "unrecoverable" not in set(new_db.island_feature_maps[0].values())
    assert new_db.feature_recode_summary["programs_recoded"] == 2
    assert new_db.feature_recode_summary["programs_excluded_from_feature_map"] == 1


def test_prompt_sampler_includes_top_diverse_failure_and_artifacts(tmp_path):
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=2),
    )
    parent = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_wl_density"),
        metrics={"combined_score": 0.4},
        artifacts={"summary": "baseline"},
        status="accepted",
    )
    top = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_density_heavy"),
        parent_id=parent.id,
        generation=1,
        metrics={"combined_score": 0.9, "hpwl_delta_pct": -3.0},
        artifacts={"dreamplace": "best run"},
        status="accepted",
    )
    failed = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_routing_aware_smoke"),
        parent_id=parent.id,
        generation=1,
        metrics={"combined_score": 0.0, "structural_failure_count": 1},
        artifacts={"error": "non-finite gradient"},
        failure_reason="non-finite gradient",
        status="failed",
    )
    db.add(parent, target_island=0)
    db.add(top, target_island=1)
    db.add(failed, target_island=1)

    context = ObjectivePromptSampler(
        term_scope="dreamplace_replacement",
        num_top_programs=2,
        num_diverse_programs=1,
    ).build_context(
        parent=parent,
        database=db,
        iteration=2,
        rng=random.Random(7),
        run_feedback={"previous": "feedback"},
    )
    encoded = json.dumps(context)

    assert context["memory_source"] == "explicit_archive_context"
    assert context["prompt_style"] == "restricted_program_evolution"
    assert context["prompt_messages"][0]["role"] == "system"
    assert "# Current Program Information" in context["prompt_messages"][1]["content"]
    assert "# Program Evolution History" in context["prompt_messages"][1]["content"]
    assert parent.id in encoded
    assert top.id in encoded
    assert "non-finite gradient" in encoded
    assert "best run" in encoded
    assert "HPWL must not regress" not in context["prompt_messages"][1]["content"]


def test_prompt_sampler_includes_component_and_map_elites_memory(tmp_path):
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    parent = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_routing_aware_smoke"),
        metrics={
            "combined_score": 0.9,
            "hpwl_delta_pct": -0.5,
            "overflow_delta_pct": -1.5,
            "component_summary": {
                "route_hotspot": {
                    "start": 0.1,
                    "mid": 0.2,
                    "end": 0.25,
                    "mean": 0.18,
                    "trend": "up",
                    "flat_or_saturated": False,
                }
            },
        },
        status="accepted",
    )
    db.add(parent, target_island=0)

    context = ObjectivePromptSampler(
        term_scope="dreamplace_replacement",
        num_top_programs=1,
        num_diverse_programs=1,
    ).build_context(
        parent=parent,
        database=db,
        iteration=2,
        rng=random.Random(3),
    )
    encoded = json.dumps(context)

    assert "component_feedback_memory" in encoded
    assert "route_hotspot" in encoded
    assert "map_elites_memory" in encoded
    assert "occupied_cells_total" in encoded


def test_prompt_sampler_hides_exact_evaluator_thresholds_and_grids(tmp_path):
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    parent = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_wl_density"),
        metrics={"combined_score": 0.4, "parent_eligible": True},
        status="accepted",
    )
    db.add(parent, target_island=0)

    context = ObjectivePromptSampler(
        term_scope="dreamplace_replacement",
        objective_mode="replacement",
        mutation_mode="diff",
        policy={
            "hpwl_gate_search_pct": 5.0,
            "hpwl_gate_elite_pct": 3.0,
            "per_design_hpwl_gate_search_pct": 5.0,
            "per_design_overflow_gate_search_pct": 0.0,
            "route_coeff_grid": [-1e-10, 1e-10],
            "density_coeff_grid": [1.3, 1.35],
            "wirelength_coeff_grid": [1.0, 1.02],
            "require_nonzero_routing_term": True,
            "min_replacement_nonbaseline_terms": 2,
            "reject_baseline_mechanism_clones": True,
            "active_routing_terms": ["route_pressure_long", "pin_density_pnorm"],
        },
    ).build_context(
        parent=parent,
        database=db,
        iteration=2,
        rng=random.Random(7),
    )
    prompt_text = "\n".join(message["content"] for message in context["prompt_messages"])
    context_text = json.dumps(context)

    assert "hpwl_gate_search_pct" not in prompt_text
    assert "per_design_hpwl_gate_search_pct" not in prompt_text
    assert "route_coeff_grid" not in prompt_text
    assert "wirelength_coeff_grid" not in prompt_text
    assert "density_coeff_grid" not in prompt_text
    assert "HPWL must not regress" not in prompt_text
    assert "route_pressure_long" in prompt_text
    assert "pin_density_pnorm" in prompt_text
    assert "min_replacement_nonbaseline_terms" not in prompt_text
    assert "reject_baseline_mechanism_clones" not in prompt_text
    assert "require_nonzero_routing_term" not in prompt_text
    assert "routing_mechanism_policy" in prompt_text
    assert "baseline_similarity" in prompt_text
    assert "coefficient-only edits" in prompt_text
    assert "Syntax-only objective program example" in prompt_text
    assert "route * pins" not in prompt_text
    assert "pin_access_pressure" in prompt_text
    assert "hpwl_gate_search_pct" not in context_text
    assert "route_coeff_grid" not in context_text


def test_prompt_sampler_hides_internal_gate_metrics_from_memory(tmp_path):
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    parent = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_routing_aware_smoke"),
        program_id="generated_route_parent",
        metrics={
            "combined_score": 0.7,
            "parent_eligible": True,
            "negative_memory_only": False,
            "hpwl_gate_passed": True,
            "overflow_gate_passed": True,
            "custom_default_gate_passed": True,
            "custom_default_hpwl_gate_passed": True,
            "baseline_portfolio_gate_passed": True,
            "baseline_portfolio_behavior_distance_gate_passed": True,
            "baseline_portfolio_min_behavior_distance_pct": 0.5,
            "elite_gate_passed": True,
            "hpwl_delta_pct": -0.2,
            "overflow_delta_pct": -1.0,
            "custom_default_hpwl_delta_pct": -0.1,
            "custom_default_overflow_delta_pct": -0.5,
            "baseline_portfolio_candidate_rank": 1,
            "baseline_portfolio_best_objective_id": "dreamplace_density_heavy",
            "baseline_portfolio_hpwl_delta_vs_best_pct": -0.2,
            "baseline_portfolio_overflow_delta_vs_best_pct": -0.4,
            "baseline_portfolio_pareto_label_vs_best": "dominates_best_baseline",
            "baseline_portfolio_pareto_dominates_best": True,
            "feedback_lesson": "Measured improvement came from a route-aware interaction.",
        },
        artifacts={
            "objective_path": "runs/objectives/generated_route_parent.json",
            "calibration_report": {"hpwl_gate_passed": True, "route_coeff_grid": [1e-8]},
            "candidate_rows": [{"hpwl_gate_passed": True, "overflow_delta_pct": -1.0}],
            "provider_metadata": {"provider": "mock", "resolved_model": "mock"},
        },
        status="accepted",
    )
    negative = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_density_heavy"),
        program_id="unsafe_negative",
        metrics={
            "combined_score": 0.0,
            "parent_eligible": False,
            "negative_memory_only": True,
            "hpwl_gate_passed": False,
            "overflow_gate_passed": True,
            "custom_default_gate_passed": False,
            "baseline_portfolio_gate_passed": False,
            "elite_gate_passed": False,
            "hpwl_delta_pct": 20.0,
            "overflow_delta_pct": -8.0,
            "feedback_lesson": "Overflow improved, but placement quality regressed badly.",
        },
        failure_reason="prompt audit failed before provider call: hpwl_gate_passed",
        status="accepted",
    )
    db.add(parent, target_island=0)
    db.add(negative, target_island=0)

    context = ObjectivePromptSampler(
        term_scope="dreamplace_replacement",
        objective_mode="replacement",
        mutation_mode="diff",
        num_top_programs=1,
        num_diverse_programs=1,
    ).build_context(
        parent=parent,
        database=db,
        iteration=3,
        rng=random.Random(5),
    )

    prompt_text = "\n".join(message["content"] for message in context["prompt_messages"])
    context_text = json.dumps(context)
    assert "hpwl_delta_pct" in prompt_text
    assert "overflow_delta_pct" in prompt_text
    assert "candidate_rank_against_fixed_baselines" in prompt_text
    assert "hpwl_delta_vs_best_fixed_baseline_pct" in prompt_text
    assert "pareto_label_vs_best_fixed_baseline" in prompt_text
    assert "dominates_best_baseline" in prompt_text
    assert "Measured improvement came from a route-aware interaction." in prompt_text
    assert "runs/objectives/generated_route_parent.json" in context_text
    assert "resolved_model" in context_text
    assert "candidate_rows" not in prompt_text
    assert "calibration_report" not in prompt_text
    assert "prompt audit failed before provider call" in prompt_text
    for hidden_key in (
        "hpwl_gate_passed",
        "overflow_gate_passed",
        "custom_default_gate_passed",
        "custom_default_hpwl_gate_passed",
        "baseline_portfolio_gate_passed",
        "require_baseline_portfolio_pareto",
        "baseline_portfolio_behavior_distance_gate_passed",
        "baseline_portfolio_min_behavior_distance_pct",
        "elite_gate_passed",
        "parent_eligible",
        "negative_memory_only",
    ):
        assert hidden_key not in prompt_text


def test_prompt_sampler_scrubs_legacy_route_coeff_failure_memory(tmp_path):
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    parent = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_routing_aware_smoke"),
        program_id="safe_route_parent",
        metrics={
            "combined_score": 0.8,
            "hpwl_delta_pct": -1.0,
            "overflow_delta_pct": -10.0,
            "hpwl_gate_passed": True,
            "overflow_gate_passed": True,
            "parent_eligible": True,
            "negative_memory_only": False,
        },
        status="accepted",
    )
    legacy_failure = ObjectiveProgram(
        id="legacy_provider_error",
        code="",
        status="provider_error",
        failure_reason="RuntimeError: prompt audit failed before provider call: coefficient exceeds cap",
        metrics={
            "combined_score": 0.0,
            "feedback_lesson": (
                "Static rejection before DREAMPlace: replacement route coefficient "
                "exceeds cap 0.01"
            ),
            "negative_memory_only": True,
            "structural_failure_count": 1,
        },
        artifacts={
            "provider_error": (
                "RuntimeError: prompt audit failed before provider call: coefficient exceeds cap"
            )
        },
        iteration_found=2,
    )
    nonbaseline_failure = ObjectiveProgram(
        id="legacy_nonbaseline_error",
        code="",
        status="rejected",
        failure_reason=(
            "replacement objective is too close to the explicit wirelength+density "
            "baseline; use at least 2 non-baseline observable families"
        ),
        metrics={
            "combined_score": 0.0,
            "feedback_lesson": (
                "Static rejection before DREAMPlace: replacement objective is too "
                "close to the explicit wirelength+density baseline; use at least "
                "2 non-baseline observable families"
            ),
            "negative_memory_only": True,
            "structural_failure_count": 1,
        },
        iteration_found=2,
    )
    db.add(parent, target_island=0)
    db.add(legacy_failure, target_island=0)
    db.add(nonbaseline_failure, target_island=0)

    context = ObjectivePromptSampler(
        term_scope="dreamplace_replacement",
        objective_mode="replacement",
        mutation_mode="diff",
        num_top_programs=1,
        num_diverse_programs=1,
    ).build_context(
        parent=parent,
        database=db,
        iteration=3,
        rng=random.Random(5),
    )

    prompt_text = "\n".join(message["content"] for message in context["prompt_messages"])
    assert "legacy_provider_error" in prompt_text
    assert "coefficient exceeds cap" not in prompt_text
    assert "route coefficient exceeds cap" not in prompt_text
    assert "cap 0.01" not in prompt_text
    assert "use at least 2 non-baseline" not in prompt_text


def test_prompt_sampler_separates_baseline_references_from_discovered_top(tmp_path):
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    baseline = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_density_135"),
        program_id="seed_density_135",
        metrics={
            "combined_score": 100.0,
            "seed_baseline": True,
            "parent_eligible": True,
            "hpwl_gate_passed": True,
            "overflow_gate_passed": True,
            "negative_memory_only": False,
        },
        status="accepted",
    )
    discovered = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_routing_aware_smoke"),
        program_id="generated_route_candidate",
        parent_id=baseline.id,
        generation=1,
        iteration_found=1,
        metrics={
            "combined_score": 1.0,
            "parent_eligible": True,
            "hpwl_gate_passed": True,
            "overflow_gate_passed": True,
            "negative_memory_only": False,
        },
        status="accepted",
    )
    db.add(baseline, target_island=0)
    db.add(discovered, target_island=0)

    context = ObjectivePromptSampler(
        term_scope="dreamplace_replacement",
        objective_mode="replacement",
        mutation_mode="diff",
        num_top_programs=2,
        num_diverse_programs=1,
    ).build_context(
        parent=baseline,
        database=db,
        iteration=2,
        rng=random.Random(11),
    )
    prompt_text = "\n".join(message["content"] for message in context["prompt_messages"])

    assert [item["id"] for item in context["top_safe_programs"]] == [discovered.id]
    assert context["top_safe_programs"][0]["program_role"] == "discovered_candidate"
    assert [item["id"] for item in context["baseline_reference_programs"]] == [baseline.id]
    assert context["baseline_reference_programs"][0]["program_role"] == "baseline_reference"
    assert "code" not in context["baseline_reference_programs"][0]
    assert "provided as reference behavior, not as discovered objectives to copy" in prompt_text
    feedback_roles = {row["program_role"] for row in context["structured_feedback_table"]}
    assert "baseline_reference" in feedback_roles
    assert "discovered_candidate" in feedback_roles


def test_prompt_sampler_includes_router_background(tmp_path):
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    parent = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_wl_density"),
        metrics={"combined_score": 0.4, "parent_eligible": True},
        status="accepted",
    )
    db.add(parent, target_island=0)

    context = ObjectivePromptSampler(
        term_scope="dreamplace_replacement",
        policy={
            "require_nonzero_routing_term": True,
            "active_routing_terms": ["soft_rudy_mean", "soft_rudy_pnorm", "pin_density_pnorm"],
        },
        router_background="overflow = max(0, demand - capacity)",
    ).build_context(
        parent=parent,
        database=db,
        iteration=1,
        rng=random.Random(1),
    )

    assert "overflow = max(0, demand - capacity)" in context["router_objective_background"]
    assert "overflow = max(0, demand - capacity)" in context["prompt_messages"][1]["content"]
    assert context["output_guidance"]["require_nonzero_routing_term"] is True
    assert "soft_rudy_pnorm" in context["output_guidance"]["active_routing_terms"]


def test_prompt_sampler_includes_near_miss_memory_without_parent_eligibility(tmp_path):
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    parent = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_wl_density"),
        metrics={
            "combined_score": 0.4,
            "parent_eligible": True,
            "hpwl_gate_passed": True,
            "overflow_gate_passed": True,
            "negative_memory_only": False,
        },
        status="accepted",
    )
    near_miss = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_routing_aware_smoke"),
        program_id="near_miss_route",
        parent_id=parent.id,
        generation=1,
        metrics={
            "combined_score": 0.0,
            "parent_eligible": False,
            "negative_memory_only": True,
            "hpwl_gate_passed": True,
            "overflow_gate_passed": False,
            "custom_default_hpwl_delta_pct": 0.1,
            "custom_default_overflow_delta_pct": -0.2,
            "custom_default_effect_pct": 0.2,
            "custom_default_hpwl_gate_passed": True,
            "custom_default_overflow_gate_passed": True,
            "custom_default_effect_gate_passed": False,
            "custom_default_gate_passed": False,
            "per_design_deltas": [
                {
                    "design": "bp_fe",
                    "hpwl_delta_pct": -20.0,
                    "overflow_delta_pct": -60.0,
                },
                {
                    "design": "isa_npu",
                    "hpwl_delta_pct": 170.0,
                    "overflow_delta_pct": -90.0,
                },
            ],
            "feedback_lesson": "Near miss: stable but effect was too small.",
        },
        status="accepted",
    )
    db.add(parent, target_island=0)
    db.add(near_miss, target_island=0)

    context = ObjectivePromptSampler(
        term_scope="dreamplace_replacement",
        num_top_programs=1,
        num_diverse_programs=0,
    ).build_context(
        parent=parent,
        database=db,
        iteration=2,
        rng=random.Random(3),
    )
    encoded = json.dumps(context)

    assert near_miss.is_parent_eligible is False
    assert context["near_miss_programs"]
    assert "near_miss_route" in encoded
    assert "per_design_deltas" in encoded
    assert "isa_npu" in encoded
    assert "worst_hpwl_design" in encoded
    assert "Near miss: stable but effect was too small." in encoded


def test_prompt_sampler_exposes_numeric_opentimer_feedback(tmp_path):
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    parent = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_routing_aware_smoke"),
        program_id="timing_measured_parent",
        metrics={
            "combined_score": 0.4,
            "parent_eligible": True,
            "negative_memory_only": False,
            "hpwl_delta_pct": -2.0,
            "overflow_delta_pct": -3.0,
            "timing_proxy_status": "success",
            "timing_proxy_wns": -0.4,
            "timing_proxy_tns": -12.0,
            "timing_proxy_wns_delta": 0.0125,
            "timing_proxy_tns_delta": 1.25,
            "timing_proxy_tns_delta_pct": 9.5,
            "timing_proxy_parent_signal": "soft_gate",
            "timing_proxy_net_coverage": 0.983,
            "timing_proxy_source_placement_budget_satisfied": True,
            "opentimer_wns_delta_bucket": "wns_improved",
            "final_def_hpwl_delta_pct": -2.0,
            "final_def_overflow_delta_pct": -3.0,
            "selection_metric_stage": "final_def_post_detailed_placement",
        },
        status="accepted",
    )
    db.add(parent, target_island=0)

    context = ObjectivePromptSampler(
        term_scope="dreamplace_replacement",
        num_top_programs=1,
        num_diverse_programs=0,
    ).build_context(
        parent=parent,
        database=db,
        iteration=2,
        rng=random.Random(7),
    )

    parent_metrics = context["parent_program"]["metrics"]
    feedback = next(
        row
        for row in context["structured_feedback_table"]
        if row["objective_id"] == parent.id
    )
    assert parent_metrics["timing_proxy_wns_delta"] == pytest.approx(0.0125)
    assert parent_metrics["timing_proxy_tns_delta"] == pytest.approx(1.25)
    assert parent_metrics["timing_proxy_tns_delta_pct"] == pytest.approx(9.5)
    assert parent_metrics["timing_proxy_net_coverage"] == pytest.approx(0.983)
    assert feedback["timing_proxy_wns_delta"] == pytest.approx(0.0125)
    assert feedback["timing_proxy_tns_delta"] == pytest.approx(1.25)
    assert feedback["timing_proxy_tns_delta_pct"] == pytest.approx(9.5)
    assert feedback["timing_proxy_parent_signal"] == "soft_gate"


def test_prompt_sampler_exposes_compact_mechanism_cluster_memory(tmp_path):
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    parent = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_wawl_electric"),
        metrics={
            "combined_score": 0.3,
            "parent_eligible": True,
            "negative_memory_only": False,
        },
        status="accepted",
    )
    for index, hpwl in enumerate((65.0, 72.0), start=1):
        near_miss = ObjectiveProgram.from_spec(
            objective_preset("dreamplace_routing_aware_smoke"),
            program_id=f"route_pin_near_miss_{index}",
            parent_id=parent.id,
            generation=1,
            metrics={
                "combined_score": 0.0,
                "parent_eligible": False,
                "negative_memory_only": True,
                "native_default_hpwl_delta_pct": -4.0,
                "native_default_overflow_delta_pct": -63.0,
                "mechanism_terms": [
                    "wirelength_wawl",
                    "density_electric",
                    "route_pressure_long",
                    "pin_density_pnorm",
                ],
                "per_design_deltas": [
                    {
                        "design": "bp_fe",
                        "hpwl_delta_pct": -15.0,
                        "overflow_delta_pct": -60.0,
                    },
                    {
                        "design": "isa_npu",
                        "hpwl_delta_pct": hpwl,
                        "overflow_delta_pct": -88.0,
                    },
                ],
                "feedback_lesson": "Repeated route-pin pressure helped overflow but hurt isa_npu.",
            },
            status="accepted",
        )
        db.add(near_miss, target_island=0)
    db.add(parent, target_island=0)

    context = ObjectivePromptSampler(
        term_scope="dreamplace_replacement",
        num_top_programs=1,
        num_diverse_programs=0,
    ).build_context(
        parent=parent,
        database=db,
        iteration=4,
        rng=random.Random(4),
    )
    memory = context["mechanism_cluster_memory"]
    prompt_text = "\n".join(message["content"] for message in context["prompt_messages"])

    clusters = memory["clusters"]
    assert clusters
    route_cluster = next(
        item
        for item in clusters
        if "route_pressure_long" in item["terms"]
        and "pin_density_pnorm" in item["terms"]
    )
    assert route_cluster["program_count"] == 2
    assert route_cluster["worst_hpwl_design"] == {
        "design": "isa_npu",
        "hpwl_delta_pct": 72.0,
    }
    assert route_cluster["cluster_role"] == "mixed_or_negative"
    assert "mechanism_cluster_memory" in prompt_text
    assert "route_pin_near_miss_1" in prompt_text
    assert "isa_npu" in prompt_text


def test_prompt_sampler_exposes_mechanism_archetype_memory(tmp_path):
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    parent = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_wawl_density_bell"),
        metrics={
            "combined_score": 0.3,
            "parent_eligible": True,
            "negative_memory_only": False,
            "seed_baseline": True,
        },
        status="accepted",
    )
    repeated_additive_route = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_routing_aware_smoke"),
        program_id="additive_route_near_miss",
        parent_id=parent.id,
        generation=1,
        metrics={
            "combined_score": 0.0,
            "parent_eligible": False,
            "negative_memory_only": True,
            "mechanism_terms": [
                "wirelength_wawl",
                "density_electric",
                "route_pressure_long",
                "soft_rudy_pnorm",
            ],
            "feedback_lesson": "Additive route term was a measured near miss.",
        },
        status="accepted",
    )
    db.add(parent, target_island=0)
    db.add(repeated_additive_route, target_island=0)

    context = ObjectivePromptSampler(
        term_scope="dreamplace_replacement",
        num_top_programs=1,
        num_diverse_programs=0,
    ).build_context(
        parent=parent,
        database=db,
        iteration=5,
        rng=random.Random(5),
    )
    archetypes = context["mechanism_archetype_memory"]["archetypes"]
    prompt_text = "\n".join(message["content"] for message in context["prompt_messages"])

    assert "mechanism_archetype_memory" in prompt_text
    assert "route_density_interaction" in prompt_text
    route_density = next(item for item in archetypes if item["name"] == "route_density_interaction")
    long_net = next(item for item in archetypes if item["name"] == "long_net_route_pressure")
    assert route_density["status"] == "unexplored"
    assert long_net["observed_count"] == 1
    assert long_net["status"] == "mostly_negative"
    assert "hpwl_gate" not in json.dumps(context["mechanism_archetype_memory"])


@pytest.mark.parametrize(
    "parent_policy",
    ["hpwl_safe_only", "manuscript_multiobjective"],
)
def test_eligible_parent_policies_exclude_negative_memory(tmp_path, parent_policy):
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    safe = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_wl_density"),
        metrics={
            "combined_score": 0.5,
            "hpwl_gate_passed": True,
            "parent_eligible": True,
            "negative_memory_only": False,
        },
        status="accepted",
    )
    unsafe = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_density_heavy"),
        metrics={
            "combined_score": 10.0,
            "hpwl_gate_passed": False,
            "parent_eligible": False,
            "negative_memory_only": True,
        },
        status="accepted",
    )
    db.add(safe, target_island=0)
    db.add(unsafe, target_island=0)

    selected = {
        db.sample_parent(
            rng=random.Random(seed),
            island_id=0,
            parent_policy=parent_policy,
        ).id
        for seed in range(10)
    }

    assert selected == {safe.id}


def test_hpwl_safe_parent_policy_uses_baseline_only_for_bootstrap(tmp_path):
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    baseline = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_density_135"),
        program_id="seed_density_135",
        metrics={
            "combined_score": 10.0,
            "seed_baseline": True,
            "parent_eligible": True,
            "hpwl_gate_passed": True,
            "overflow_gate_passed": True,
            "negative_memory_only": False,
        },
        status="accepted",
    )
    db.add(baseline, target_island=0)

    selected = db.sample_parent(
        rng=random.Random(1),
        island_id=0,
        parent_policy="hpwl_safe_only",
    )

    assert selected.id == baseline.id


def test_hpwl_safe_parent_policy_prefers_generated_candidate_over_baseline(tmp_path):
    db = ObjectiveProgramDatabase(
        tmp_path / "program_db",
        ObjectiveDatabaseConfig(population_size=10, archive_size=3, num_islands=1),
    )
    baseline = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_density_135"),
        program_id="seed_density_135",
        metrics={
            "combined_score": 100.0,
            "seed_baseline": True,
            "parent_eligible": True,
            "hpwl_gate_passed": True,
            "overflow_gate_passed": True,
            "negative_memory_only": False,
        },
        status="accepted",
    )
    generated = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_routing_aware_smoke"),
        program_id="generated_route_candidate",
        parent_id=baseline.id,
        generation=1,
        metrics={
            "combined_score": 1.0,
            "parent_eligible": True,
            "hpwl_gate_passed": True,
            "overflow_gate_passed": True,
            "negative_memory_only": False,
        },
        status="accepted",
    )
    db.add(baseline, target_island=0)
    db.add(generated, target_island=0)

    selected = {
        db.sample_parent(
            rng=random.Random(seed),
            island_id=0,
            parent_policy="hpwl_safe_only",
        ).id
        for seed in range(10)
    }

    assert selected == {generated.id}


def test_hpwl_safe_parent_weights_are_novelty_aware():
    high_score_near_baseline = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_routing_aware_smoke"),
        program_id="high_score_near_baseline",
        metrics={
            "combined_score": 100.0,
            "parent_eligible": True,
            "baseline_mechanism_distance": 0.35,
            "mechanism_signature": "common_route_shape",
        },
        status="accepted",
    )
    low_score_novel = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_routing_aware_smoke"),
        program_id="low_score_novel",
        metrics={
            "combined_score": 1.0,
            "parent_eligible": True,
            "baseline_mechanism_distance": 0.95,
            "mechanism_signature": "novel_route_shape",
        },
        status="accepted",
    )

    raw_weights = _parent_sample_weights(
        [high_score_near_baseline, low_score_novel],
        novelty_aware=False,
    )
    novelty_weights = _parent_sample_weights(
        [high_score_near_baseline, low_score_novel],
        novelty_aware=True,
    )

    raw_ratio = raw_weights[0] / raw_weights[1]
    novelty_ratio = novelty_weights[0] / novelty_weights[1]
    assert raw_ratio == 100.0
    assert novelty_ratio < 6.0
    assert novelty_weights[1] > 0.0


def test_hpwl_safe_parent_weights_penalize_mechanism_crowding():
    clone_a = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_routing_aware_smoke"),
        program_id="clone_a",
        metrics={
            "combined_score": 3.0,
            "parent_eligible": True,
            "baseline_mechanism_distance": 0.8,
            "mechanism_signature": "shared_shape",
        },
        status="accepted",
    )
    clone_b = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_routing_aware_smoke"),
        program_id="clone_b",
        metrics={
            "combined_score": 3.0,
            "parent_eligible": True,
            "baseline_mechanism_distance": 0.8,
            "mechanism_signature": "shared_shape",
        },
        status="accepted",
    )
    unique = ObjectiveProgram.from_spec(
        objective_preset("dreamplace_routing_aware_smoke"),
        program_id="unique",
        metrics={
            "combined_score": 3.0,
            "parent_eligible": True,
            "baseline_mechanism_distance": 0.8,
            "mechanism_signature": "unique_shape",
        },
        status="accepted",
    )

    weights = _parent_sample_weights([clone_a, clone_b, unique], novelty_aware=True)

    assert weights[0] == weights[1]
    assert weights[2] > weights[0]
