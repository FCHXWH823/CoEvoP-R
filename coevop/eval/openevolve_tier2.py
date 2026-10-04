"""OpenEvolve-style Tier-2 objective evolution for DREAMPlace."""

from __future__ import annotations

import csv
import copy
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised by py310 via tomli
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from coevop.eval.design_profile import (
    build_design_profiles,
    merge_baseline_behavior,
    summarize_baseline_behavior,
)
from coevop.eval.macro_preflight import audit_shared_panel_macros
from coevop.eval.ranking import add_outcome_labels, aggregate_rank_scores
from coevop.eval.shared_panel import load_shared_panel, write_dreamplace_panel
from coevop.eval.tier3_openroad import run_tier3_openroad
from coevop.eval.tier2_dreamplace import load_panel, run_tier2_dreamplace
from coevop.eval.timing_proxy import (
    apply_timing_proxy_admission,
    apply_timing_proxy_policy,
    cap_unstable_timing_proxy_admission,
    default_timing_proxy_admission,
    load_timing_proxy_panel,
    placements_from_tier2_comparison,
    run_timing_proxy_audit,
    run_timing_proxy_eval,
    summarize_timing_proxy_admission,
    timing_proxy_downstream_agreement,
    timing_proxy_metrics_by_objective,
)
from coevop.backends.dreamplace import make_run_config, run_dreamplace, write_run_summary
from coevop.evolution.openevolve_core import (
    ROUTED_EVIDENCE_KEYS,
    ObjectiveDatabaseConfig,
    ObjectiveEvolutionTrace,
    ObjectiveProgram,
    ObjectiveProgramDatabase,
    ObjectivePromptSampler,
    objective_code_from_spec,
)
from coevop.llm.providers import LLMProvider, ProviderTrace, provider_from_name
from coevop.objectives.presets import objective_preset
from coevop.objectives.spec import (
    ObjectiveSpec,
    ObjectiveSpecError,
    load_objective_spec,
    parse_objective_spec,
    write_objective_spec,
)
from coevop.objectives.terms import (
    privileged_observable_names,
    term_names,
    unsupported_terms_for_scope,
)


DEFAULT_INITIAL_PRESETS = [
    "dreamplace_explicit_wl_density",
    "dreamplace_wl_density",
    "dreamplace_wawl_electric",
    "dreamplace_wawl_density_bell",
    "dreamplace_lse_density_bell",
    "dreamplace_density_heavy",
    "dreamplace_density_135",
    "dreamplace_routing_aware_smoke",
    "dreamplace_route_pressure_long",
    "dreamplace_pin_count_weighted_wl",
]

ROUTE_CORRECTION_TERMS = {
    "soft_rudy_mean",
    "soft_rudy_pnorm",
    "route_pressure_long",
    "pin_density_pnorm",
    "pin_count_weighted_wl",
}
RESIDUAL_BASE_TERMS = {"wirelength", "density"}
NATIVE_RESIDUAL_BASE_TERMS = {"native_objective"}
NATIVE_RESIDUAL_ALLOWED_TERMS = NATIVE_RESIDUAL_BASE_TERMS | ROUTE_CORRECTION_TERMS
NATIVE_RESIDUAL_FORBIDDEN_SCORE_TERMS = {
    "wirelength",
    "wirelength_wawl",
    "wirelength_lse",
    "density",
    "density_electric",
    "density_bell",
}
WIRELENGTH_FAMILY_TERMS = {"wirelength", "wirelength_wawl", "wirelength_lse"}
DENSITY_FAMILY_TERMS = {"density", "density_electric", "density_bell"}
DEFAULT_EQUIVALENT_BASE_TERMS = {
    "wirelength",
    "wirelength_wawl",
    "density",
    "density_electric",
}
DEFAULT_ROUTE_COEFF_GRID = [-1e-10, -1e-12, 0.0, 1e-12, 1e-10]
DEFAULT_DENSITY_COEFF_GRID = [1.0]
DEFAULT_WIRELENGTH_COEFF_GRID = [1.0]
DEFAULT_TERM_SCALE_AUDIT_TERMS = [
    "wirelength_wawl",
    "wirelength_lse",
    "density_electric",
    "density_bell",
    "soft_rudy_mean",
    "soft_rudy_pnorm",
    "route_pressure_long",
    "pin_density_pnorm",
    "pin_count_weighted_wl",
]
TERM_SCALE_AUDIT_REFERENCE_TERMS = ["wirelength_wawl", "density_electric", "native_objective"]
TERM_SCALE_CORE_TERMS = {"wirelength_wawl", "density_electric"}


@dataclass(frozen=True)
class OpenEvolveTier2Config:
    provider: str
    term_scope: str
    max_iterations: int
    samples_per_iteration: int
    population_size: int
    archive_size: int
    num_islands: int
    map_elites_enabled: bool
    feature_bins: int
    feature_dimensions: list[str]
    migration_interval: int
    migration_rate: float
    migration_topology: str
    migration_clock: str
    checkpoint_interval: int
    num_top_programs: int
    num_diverse_programs: int
    severe_regression_pct: float
    seed: int
    objective_mode: str
    mutation_mode: str
    evaluate_initial_presets: bool
    primary_baseline: str
    hpwl_gate_search_pct: float
    hpwl_gate_elite_pct: float
    per_design_hpwl_gate_search_pct: float
    per_design_overflow_gate_search_pct: float
    min_design_both_improvement_fraction: float
    require_robust_parent_gate: bool
    allow_near_miss_parents: bool
    near_miss_max_worst_hpwl_pct: float
    near_miss_max_worst_overflow_pct: float
    near_miss_parent_score_floor: float
    overflow_severe_regression_pct: float
    require_custom_default_gate: bool
    require_baseline_portfolio_gate: bool
    require_baseline_portfolio_pareto: bool
    min_baseline_portfolio_effect_pct: float
    min_baseline_behavior_distance_pct: float
    native_default_selection_policy: str
    native_default_pareto_bonus: float
    custom_default_hpwl_gate_pct: float
    custom_default_overflow_gate_pct: float
    min_custom_default_effect_pct: float
    require_nonzero_routing_term: bool
    min_replacement_nonbaseline_terms: int
    # Controller mode: minimum state registers a candidate must define; at
    # least one register must read an observable when this is positive.
    min_state_registers: int
    min_abs_routing_coeff: float
    require_measured_routing_influence: bool
    min_routing_component_samples: int
    min_routing_gradient_ratio_mean: float
    min_routing_gradient_ratio_peak: float
    route_coeff_cap: float
    route_coeff_grid: list[float]
    density_coeff_grid: list[float]
    wirelength_coeff_grid: list[float]
    term_calibration_enabled: bool
    term_calibration_terms: list[str]
    term_calibration_gradient_ratios: list[float]
    term_calibration_iterations: int
    term_calibration_timeout_seconds: int
    term_calibration_min_grad_norm: float
    term_scale_audit_enabled: bool
    term_scale_audit_terms: list[str]
    term_scale_audit_reuse_existing: bool
    term_scale_audit_fail_on_broken_core_terms: bool
    term_scale_audit_iterations: int
    term_scale_audit_timeout_seconds: int
    term_scale_audit_max_runtime_seconds: float
    term_scale_audit_min_grad_norm: float
    term_scale_audit_blocked_terms: list[str]
    term_scale_audit_context: dict[str, Any]
    standardization_official_mode: str | None
    standardization_allow_native_residual_fallback: bool
    max_calibrated_siblings: int
    max_generation_attempts: int
    full_rewrite_after_static_rejections: int
    escape_parent_after_static_rejections: int
    reject_baseline_mechanism_clones: bool
    min_baseline_mechanism_distance: float
    reject_repeated_negative_mechanisms: bool
    min_negative_mechanism_distance: float
    parent_policy: str
    selection_policy: str
    selection_hpwl_weight: float
    selection_overflow_weight: float
    selection_wns_weight: float
    selection_tns_weight: float
    max_parent_hpwl_regression_pct: float
    max_parent_overflow_regression_pct: float
    require_timing_for_generated_parents: bool
    require_timing_improvement_for_elite: bool
    structured_feedback: bool
    initial_presets: list[str]
    search_panel: str
    target_designs: list[str]
    search_baseline_presets: list[str]
    search_disable_legalization: bool
    final_enabled: bool
    final_panel: str | None
    final_top_k: int
    final_baseline_presets: list[str]
    final_disable_legalization: bool
    generalization_enabled: bool
    generalization_panel: str | None
    generalization_top_k: int
    generalization_baseline_presets: list[str]
    generalization_candidate_policy: str
    generalization_blind: bool
    generalization_strict_leakage_guard: bool
    tier3_enabled: bool
    tier3_panel: str | None
    tier3_top_k: int
    tier3_interval: int
    tier3_start_iteration: int
    tier3_baseline_objective_id: str
    tier3_baseline_presets: list[str]
    tier3_candidate_policy: str
    macro_preflight_enabled: bool
    macro_preflight_require_movable_macros: bool
    macro_preflight_reject_fixed_hard_macros: bool
    macro_preflight_minimum_movable_macro_count: int
    timing_proxy_enabled: bool
    timing_proxy_mode: str
    timing_proxy_panel: str | None
    timing_proxy_audit_required: bool
    timing_proxy_max_hpwl_corr_for_gate: float
    timing_proxy_max_hpwl_corr_for_tiebreaker: float
    timing_proxy_wns_regression_gate_ns: float
    timing_proxy_wns_delta_min_abs_ns: float
    timing_proxy_tns_regression_pct_gate: float
    timing_proxy_tns_regression_min_abs_ns: float
    timing_proxy_selection_weight: float
    # Generations between in-loop timing-proxy audits; 0 keeps the configured
    # mode as the admission for every design and metric.
    timing_proxy_audit_interval: int
    timing_proxy_audit_perturbations: int
    timing_proxy_audit_min_pairs: int
    timing_controller_min_net_coverage: float
    # Timing collateral for candidates that define update_net_weights.
    timing_controller_panel: str | None
    design_profile_enabled: bool
    design_profile_include_file_stats: bool
    prior_feedback: str | None
    router_background_path: str | None
    router_background: str | None
    minimum_scoring_iterations: int
    allow_short_search_for_tests: bool


def load_openevolve_tier2_config(path: str | Path) -> OpenEvolveTier2Config:
    config_path = Path(path)
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    search = dict(payload.get("search", {}))
    final = dict(payload.get("final", {}))
    generalization = dict(payload.get("generalization", {}))
    tier3 = dict(payload.get("tier3", {}))
    macro_preflight = dict(payload.get("macro_preflight", {}))
    timing_proxy = dict(payload.get("timing_proxy", {}))
    timing_controller = dict(payload.get("timing_controller", {}))
    routing_influence = dict(payload.get("routing_influence", {}))
    term_calibration = dict(payload.get("term_calibration", {}))
    term_scale_audit = dict(payload.get("term_scale_audit", {}))
    standardization = dict(payload.get("standardization", {}))
    design_profile = dict(payload.get("design_profile", {}))
    feedback = dict(payload.get("feedback", {}))
    prompt = dict(payload.get("prompt", {}))
    evaluation = dict(payload.get("evaluation", {}))
    selection = dict(payload.get("selection", {}))
    timing_proxy_mode = str(
        timing_proxy.get("mode", payload.get("timing_proxy_mode", "diagnostic"))
    )
    if timing_proxy_mode not in {"diagnostic", "tie_breaker", "gate"}:
        raise ValueError(
            "timing_proxy_mode must be one of: diagnostic, tie_breaker, gate"
        )
    timing_controller_min_net_coverage = float(
        timing_controller.get(
            "min_net_coverage",
            payload.get("timing_controller_min_net_coverage", 0.95),
        )
    )
    if not 0.0 <= timing_controller_min_net_coverage <= 1.0:
        raise ValueError("timing_controller min_net_coverage must be within [0, 1]")
    router_background_path = (
        str(payload.get("router_background") or prompt.get("router_background"))
        if payload.get("router_background") or prompt.get("router_background")
        else None
    )
    resolved_router_background_path = (
        _resolve_config_path(config_path, router_background_path)
        if router_background_path
        else None
    )
    router_background = None
    if resolved_router_background_path is not None:
        router_background = resolved_router_background_path.read_text(encoding="utf-8")
    return OpenEvolveTier2Config(
        provider=str(payload.get("provider", "mock")),
        term_scope=str(payload.get("term_scope", "dreamplace_replacement")),
        max_iterations=int(payload.get("max_iterations", payload.get("iterations", 80))),
        samples_per_iteration=int(payload.get("samples_per_iteration", 1)),
        population_size=int(payload.get("population_size", 100)),
        archive_size=int(payload.get("archive_size", 20)),
        num_islands=int(payload.get("num_islands", 5)),
        map_elites_enabled=bool(payload.get("map_elites_enabled", True)),
        feature_bins=int(payload.get("feature_bins", 8)),
        feature_dimensions=[
            str(name)
            for name in payload.get(
                "feature_dimensions",
                [
                    "complexity",
                    "mechanism_family",
                    "hpwl_delta_bucket",
                    "overflow_delta_bucket",
                    "opentimer_wns_delta_bucket",
                ],
            )
        ],
        migration_interval=int(payload.get("migration_interval", 20)),
        migration_rate=float(payload.get("migration_rate", 0.10)),
        migration_topology=str(payload.get("migration_topology", "ring")),
        migration_clock=str(payload.get("migration_clock", "generation")),
        checkpoint_interval=int(payload.get("checkpoint_interval", 10)),
        num_top_programs=int(payload.get("num_top_programs", 3)),
        num_diverse_programs=int(payload.get("num_diverse_programs", 2)),
        severe_regression_pct=float(
            payload.get("severe_regression_pct", payload.get("overflow_severe_regression_pct", 10.0))
        ),
        seed=int(payload.get("seed", 42)),
        objective_mode=str(payload.get("objective_mode", "replacement")),
        mutation_mode=str(payload.get("mutation_mode", "diff")),
        evaluate_initial_presets=bool(payload.get("evaluate_initial_presets", True)),
        primary_baseline=str(payload.get("primary_baseline", "custom_default")),
        hpwl_gate_search_pct=float(payload.get("hpwl_gate_search_pct", 5.0)),
        hpwl_gate_elite_pct=float(payload.get("hpwl_gate_elite_pct", 3.0)),
        per_design_hpwl_gate_search_pct=float(
            payload.get(
                "per_design_hpwl_gate_search_pct",
                payload.get("hpwl_gate_search_pct", 5.0),
            )
        ),
        per_design_overflow_gate_search_pct=float(
            payload.get("per_design_overflow_gate_search_pct", 0.0)
        ),
        min_design_both_improvement_fraction=float(
            payload.get("min_design_both_improvement_fraction", 0.0)
        ),
        require_robust_parent_gate=bool(payload.get("require_robust_parent_gate", True)),
        allow_near_miss_parents=bool(payload.get("allow_near_miss_parents", False)),
        near_miss_max_worst_hpwl_pct=float(
            payload.get("near_miss_max_worst_hpwl_pct", 25.0)
        ),
        near_miss_max_worst_overflow_pct=float(
            payload.get("near_miss_max_worst_overflow_pct", 10.0)
        ),
        near_miss_parent_score_floor=float(
            payload.get("near_miss_parent_score_floor", 0.01)
        ),
        overflow_severe_regression_pct=float(payload.get("overflow_severe_regression_pct", 10.0)),
        require_custom_default_gate=bool(payload.get("require_custom_default_gate", False)),
        require_baseline_portfolio_gate=bool(
            payload.get("require_baseline_portfolio_gate", False)
        ),
        require_baseline_portfolio_pareto=bool(
            payload.get("require_baseline_portfolio_pareto", False)
        ),
        min_baseline_portfolio_effect_pct=float(
            payload.get("min_baseline_portfolio_effect_pct", 0.0)
        ),
        min_baseline_behavior_distance_pct=float(
            payload.get("min_baseline_behavior_distance_pct", 0.0)
        ),
        native_default_selection_policy=str(
            payload.get("native_default_selection_policy", "report")
        ),
        native_default_pareto_bonus=float(payload.get("native_default_pareto_bonus", 0.0)),
        custom_default_hpwl_gate_pct=float(
            payload.get("custom_default_hpwl_gate_pct", payload.get("hpwl_gate_search_pct", 5.0))
        ),
        custom_default_overflow_gate_pct=float(
            payload.get("custom_default_overflow_gate_pct", 0.0)
        ),
        min_custom_default_effect_pct=float(payload.get("min_custom_default_effect_pct", 0.0)),
        require_nonzero_routing_term=bool(payload.get("require_nonzero_routing_term", False)),
        min_replacement_nonbaseline_terms=int(
            payload.get("min_replacement_nonbaseline_terms", 0)
        ),
        min_state_registers=int(payload.get("min_state_registers", 0)),
        min_abs_routing_coeff=float(payload.get("min_abs_routing_coeff", 1e-18)),
        require_measured_routing_influence=bool(
            routing_influence.get(
                "required",
                payload.get("require_measured_routing_influence", False),
            )
        ),
        min_routing_component_samples=int(
            routing_influence.get(
                "min_samples",
                payload.get("min_routing_component_samples", 3),
            )
        ),
        min_routing_gradient_ratio_mean=float(
            routing_influence.get(
                "min_gradient_ratio_mean",
                payload.get("min_routing_gradient_ratio_mean", 1e-4),
            )
        ),
        min_routing_gradient_ratio_peak=float(
            routing_influence.get(
                "min_gradient_ratio_peak",
                payload.get("min_routing_gradient_ratio_peak", 1e-3),
            )
        ),
        route_coeff_cap=float(payload.get("route_coeff_cap", 1e-10)),
        route_coeff_grid=[
            float(value)
            for value in payload.get("route_coeff_grid", DEFAULT_ROUTE_COEFF_GRID)
        ],
        density_coeff_grid=[
            float(value)
            for value in payload.get("density_coeff_grid", DEFAULT_DENSITY_COEFF_GRID)
        ],
        wirelength_coeff_grid=[
            float(value)
            for value in payload.get("wirelength_coeff_grid", DEFAULT_WIRELENGTH_COEFF_GRID)
        ],
        term_calibration_enabled=bool(
            term_calibration.get(
                "enabled",
                payload.get("term_calibration_enabled", False),
            )
        ),
        term_calibration_terms=[
            str(term)
            for term in term_calibration.get(
                "terms",
                payload.get(
                    "term_calibration_terms",
                    [
                        "soft_rudy_mean",
                        "soft_rudy_pnorm",
                        "route_pressure_long",
                        "pin_density_pnorm",
                        "pin_count_weighted_wl",
                    ],
                ),
            )
        ],
        term_calibration_gradient_ratios=[
            float(value)
            for value in term_calibration.get(
                "gradient_ratios",
                payload.get(
                    "term_calibration_gradient_ratios",
                    [0.001, 0.003, 0.01, 0.03, 0.05],
                ),
            )
        ],
        term_calibration_iterations=int(
            term_calibration.get(
                "iterations",
                payload.get("term_calibration_iterations", 3),
            )
        ),
        term_calibration_timeout_seconds=int(
            term_calibration.get(
                "timeout_seconds",
                payload.get("term_calibration_timeout_seconds", 900),
            )
        ),
        term_calibration_min_grad_norm=float(
            term_calibration.get(
                "min_grad_norm",
                payload.get("term_calibration_min_grad_norm", 1e-20),
            )
        ),
        term_scale_audit_enabled=bool(
            term_scale_audit.get(
                "enabled",
                payload.get("term_scale_audit_enabled", False),
            )
        ),
        term_scale_audit_terms=[
            str(term)
            for term in term_scale_audit.get(
                "terms",
                payload.get("term_scale_audit_terms", DEFAULT_TERM_SCALE_AUDIT_TERMS),
            )
        ],
        term_scale_audit_reuse_existing=bool(
            term_scale_audit.get(
                "reuse_existing",
                payload.get("term_scale_audit_reuse_existing", True),
            )
        ),
        term_scale_audit_fail_on_broken_core_terms=bool(
            term_scale_audit.get(
                "fail_on_broken_core_terms",
                payload.get("term_scale_audit_fail_on_broken_core_terms", True),
            )
        ),
        term_scale_audit_iterations=int(
            term_scale_audit.get(
                "iterations",
                payload.get("term_scale_audit_iterations", 1),
            )
        ),
        term_scale_audit_timeout_seconds=int(
            term_scale_audit.get(
                "timeout_seconds",
                payload.get("term_scale_audit_timeout_seconds", 900),
            )
        ),
        term_scale_audit_max_runtime_seconds=float(
            term_scale_audit.get(
                "max_runtime_seconds",
                payload.get("term_scale_audit_max_runtime_seconds", 0.0),
            )
        ),
        term_scale_audit_min_grad_norm=float(
            term_scale_audit.get(
                "min_grad_norm",
                payload.get("term_scale_audit_min_grad_norm", 1e-20),
            )
        ),
        term_scale_audit_blocked_terms=[
            str(term)
            for term in payload.get("term_scale_audit_blocked_terms", [])
        ],
        term_scale_audit_context=dict(payload.get("term_scale_audit_context", {})),
        standardization_official_mode=(
            str(standardization.get("official_mode"))
            if standardization.get("official_mode") is not None
            else (
                str(payload.get("standardization_official_mode"))
                if payload.get("standardization_official_mode") is not None
                else None
            )
        ),
        standardization_allow_native_residual_fallback=bool(
            standardization.get(
                "allow_native_residual_fallback",
                payload.get("standardization_allow_native_residual_fallback", True),
            )
        ),
        max_calibrated_siblings=int(payload.get("max_calibrated_siblings", 8)),
        max_generation_attempts=int(payload.get("max_generation_attempts", 3)),
        full_rewrite_after_static_rejections=int(
            payload.get("full_rewrite_after_static_rejections", 0)
        ),
        escape_parent_after_static_rejections=int(
            payload.get(
                "escape_parent_after_static_rejections",
                payload.get("full_rewrite_after_static_rejections", 0),
            )
        ),
        reject_baseline_mechanism_clones=bool(
            payload.get("reject_baseline_mechanism_clones", False)
        ),
        min_baseline_mechanism_distance=float(
            payload.get("min_baseline_mechanism_distance", 0.0)
        ),
        reject_repeated_negative_mechanisms=bool(
            payload.get("reject_repeated_negative_mechanisms", True)
        ),
        min_negative_mechanism_distance=float(
            payload.get("min_negative_mechanism_distance", 0.0)
        ),
        parent_policy=str(payload.get("parent_policy", "hpwl_safe_only")),
        selection_policy=str(
            selection.get(
                "policy",
                payload.get("selection_policy", "legacy_hpwl_overflow"),
            )
        ),
        selection_hpwl_weight=float(
            selection.get("hpwl_weight", payload.get("selection_hpwl_weight", 0.25))
        ),
        selection_overflow_weight=float(
            selection.get(
                "overflow_weight",
                payload.get("selection_overflow_weight", 0.25),
            )
        ),
        selection_wns_weight=float(
            selection.get("wns_weight", payload.get("selection_wns_weight", 0.25))
        ),
        selection_tns_weight=float(
            selection.get("tns_weight", payload.get("selection_tns_weight", 0.25))
        ),
        max_parent_hpwl_regression_pct=float(
            selection.get(
                "max_parent_hpwl_regression_pct",
                payload.get(
                    "max_parent_hpwl_regression_pct",
                    payload.get("severe_regression_pct", 10.0),
                ),
            )
        ),
        max_parent_overflow_regression_pct=float(
            selection.get(
                "max_parent_overflow_regression_pct",
                payload.get(
                    "max_parent_overflow_regression_pct",
                    payload.get("overflow_severe_regression_pct", 10.0),
                ),
            )
        ),
        require_timing_for_generated_parents=bool(
            selection.get(
                "require_timing_for_generated_parents",
                payload.get("require_timing_for_generated_parents", False),
            )
        ),
        require_timing_improvement_for_elite=bool(
            selection.get(
                "require_timing_improvement_for_elite",
                payload.get("require_timing_improvement_for_elite", False),
            )
        ),
        structured_feedback=bool(payload.get("structured_feedback", True)),
        initial_presets=[str(name) for name in payload.get("initial_presets", DEFAULT_INITIAL_PRESETS)],
        search_panel=str(_resolve_config_path(config_path, str(search["panel"]))),
        target_designs=[str(name) for name in payload.get("target_designs", search.get("target_designs", []))],
        search_baseline_presets=[str(name) for name in search.get("baseline_presets", [])],
        search_disable_legalization=bool(search.get("disable_legalization", False)),
        final_enabled=bool(final.get("enabled", False)),
        final_panel=(
            str(_resolve_config_path(config_path, str(final["panel"])))
            if final.get("panel")
            else None
        ),
        final_top_k=int(final.get("top_k", 8)),
        final_baseline_presets=[str(name) for name in final.get("baseline_presets", [])],
        final_disable_legalization=bool(final.get("disable_legalization", False)),
        generalization_enabled=bool(generalization.get("enabled", False)),
        generalization_panel=(
            str(_resolve_config_path(config_path, str(generalization["panel"])))
            if generalization.get("panel")
            else None
        ),
        generalization_top_k=int(generalization.get("top_k", 5)),
        generalization_baseline_presets=[
            str(name) for name in generalization.get("baseline_presets", [])
        ],
        generalization_candidate_policy=str(
            generalization.get("candidate_policy", "elite_or_aggregate")
        ),
        generalization_blind=bool(generalization.get("blind", False)),
        generalization_strict_leakage_guard=bool(
            generalization.get("strict_leakage_guard", False)
        ),
        tier3_enabled=bool(tier3.get("enabled", False)),
        tier3_panel=(
            str(_resolve_config_path(config_path, str(tier3["panel"])))
            if tier3.get("panel")
            else None
        ),
        tier3_top_k=int(tier3.get("top_k", 3)),
        tier3_interval=int(tier3.get("interval", 0)),
        tier3_start_iteration=int(tier3.get("start_iteration", 1)),
        tier3_baseline_objective_id=str(
            tier3.get("baseline_objective_id", payload.get("primary_baseline", "default"))
        ),
        tier3_baseline_presets=[str(name) for name in tier3.get("baseline_presets", [])],
        tier3_candidate_policy=str(
            tier3.get("candidate_policy", payload.get("tier3_candidate_policy", "elite_or_aggregate"))
        ),
        macro_preflight_enabled=bool(
            macro_preflight.get(
                "enabled",
                payload.get("macro_preflight_enabled", False),
            )
        ),
        macro_preflight_require_movable_macros=bool(
            macro_preflight.get(
                "require_movable_macros",
                payload.get("macro_preflight_require_movable_macros", False),
            )
        ),
        macro_preflight_reject_fixed_hard_macros=bool(
            macro_preflight.get(
                "reject_fixed_hard_macros",
                payload.get("macro_preflight_reject_fixed_hard_macros", False),
            )
        ),
        macro_preflight_minimum_movable_macro_count=int(
            macro_preflight.get(
                "minimum_movable_macro_count",
                payload.get("macro_preflight_minimum_movable_macro_count", 1),
            )
        ),
        timing_proxy_enabled=bool(
            timing_proxy.get("enabled", payload.get("timing_proxy_enabled", False))
        ),
        timing_proxy_mode=timing_proxy_mode,
        timing_proxy_panel=(
            str(
                _resolve_config_path(
                    config_path,
                    str(timing_proxy.get("panel", payload.get("timing_proxy_panel"))),
                )
            )
            if timing_proxy.get("panel", payload.get("timing_proxy_panel"))
            else None
        ),
        timing_proxy_audit_required=bool(
            timing_proxy.get(
                "audit_required",
                payload.get("timing_proxy_audit_required", True),
            )
        ),
        timing_proxy_max_hpwl_corr_for_gate=float(
            timing_proxy.get(
                "max_hpwl_corr_for_gate",
                payload.get("timing_proxy_max_hpwl_corr_for_gate", 0.70),
            )
        ),
        timing_proxy_max_hpwl_corr_for_tiebreaker=float(
            timing_proxy.get(
                "max_hpwl_corr_for_tiebreaker",
                payload.get("timing_proxy_max_hpwl_corr_for_tiebreaker", 0.95),
            )
        ),
        timing_proxy_wns_regression_gate_ns=float(
            timing_proxy.get(
                "wns_regression_gate_ns",
                timing_proxy.get(
                    "wns_regression_gate",
                    payload.get(
                        "timing_proxy_wns_regression_gate_ns",
                        payload.get("timing_proxy_wns_regression_gate", 0.20),
                    ),
                ),
            )
        ),
        timing_proxy_wns_delta_min_abs_ns=float(
            timing_proxy.get(
                "wns_delta_min_abs_ns",
                payload.get("timing_proxy_wns_delta_min_abs_ns", 0.05),
            )
        ),
        timing_proxy_tns_regression_pct_gate=float(
            timing_proxy.get(
                "tns_regression_pct_gate",
                payload.get("timing_proxy_tns_regression_pct_gate", 10.0),
            )
        ),
        timing_proxy_tns_regression_min_abs_ns=float(
            timing_proxy.get(
                "tns_regression_min_abs_ns",
                payload.get("timing_proxy_tns_regression_min_abs_ns", 0.05),
            )
        ),
        timing_proxy_selection_weight=float(
            timing_proxy.get(
                "selection_weight",
                payload.get("timing_proxy_selection_weight", 0.05),
            )
        ),
        timing_proxy_audit_interval=int(
            timing_proxy.get(
                "audit_interval",
                payload.get("timing_proxy_audit_interval", 0),
            )
        ),
        timing_proxy_audit_perturbations=int(
            timing_proxy.get(
                "audit_perturbations",
                payload.get("timing_proxy_audit_perturbations", 5),
            )
        ),
        timing_proxy_audit_min_pairs=int(
            timing_proxy.get(
                "audit_min_pairs",
                payload.get("timing_proxy_audit_min_pairs", 3),
            )
        ),
        timing_controller_min_net_coverage=timing_controller_min_net_coverage,
        timing_controller_panel=(
            str(
                _resolve_config_path(
                    config_path,
                    str(
                        timing_controller.get(
                            "panel", payload.get("timing_controller_panel")
                        )
                    ),
                )
            )
            if timing_controller.get("panel", payload.get("timing_controller_panel"))
            else None
        ),
        design_profile_enabled=bool(
            design_profile.get("enabled", payload.get("design_profile_enabled", True))
        ),
        design_profile_include_file_stats=bool(
            design_profile.get(
                "include_file_stats",
                payload.get("design_profile_include_file_stats", True),
            )
        ),
        prior_feedback=(
            str(_resolve_config_path(config_path, str(feedback["prior_feedback"])))
            if feedback.get("prior_feedback")
            else None
        ),
        router_background_path=(
            str(resolved_router_background_path) if resolved_router_background_path else None
        ),
        router_background=router_background,
        minimum_scoring_iterations=int(
            evaluation.get(
                "minimum_iterations",
                payload.get("minimum_scoring_iterations", 1000),
            )
        ),
        allow_short_search_for_tests=bool(
            evaluation.get(
                "allow_short_test_runs",
                payload.get("allow_short_search_for_tests", False),
            )
        ),
    )


def _prompt_policy_from_config(
    config: OpenEvolveTier2Config,
    *,
    database: ObjectiveProgramDatabase | None = None,
) -> dict[str, Any]:
    policy = {
        "objective_mode": config.objective_mode,
        "primary_baseline": config.primary_baseline,
        "hpwl_gate_search_pct": config.hpwl_gate_search_pct,
        "hpwl_gate_elite_pct": config.hpwl_gate_elite_pct,
        "per_design_hpwl_gate_search_pct": config.per_design_hpwl_gate_search_pct,
        "per_design_overflow_gate_search_pct": config.per_design_overflow_gate_search_pct,
        "min_design_both_improvement_fraction": config.min_design_both_improvement_fraction,
        "require_robust_parent_gate": config.require_robust_parent_gate,
        "allow_near_miss_parents": config.allow_near_miss_parents,
        "near_miss_max_worst_hpwl_pct": config.near_miss_max_worst_hpwl_pct,
        "near_miss_max_worst_overflow_pct": config.near_miss_max_worst_overflow_pct,
        "near_miss_parent_score_floor": config.near_miss_parent_score_floor,
        "route_coeff_cap": config.route_coeff_cap,
        "route_coeff_grid": config.route_coeff_grid,
        "density_coeff_grid": config.density_coeff_grid,
        "wirelength_coeff_grid": config.wirelength_coeff_grid,
        "max_calibrated_siblings": config.max_calibrated_siblings,
        "require_custom_default_gate": config.require_custom_default_gate,
        "require_baseline_portfolio_gate": config.require_baseline_portfolio_gate,
        "require_baseline_portfolio_pareto": config.require_baseline_portfolio_pareto,
        "min_baseline_portfolio_effect_pct": config.min_baseline_portfolio_effect_pct,
        "min_baseline_behavior_distance_pct": config.min_baseline_behavior_distance_pct,
        "native_default_selection_policy": config.native_default_selection_policy,
        "native_default_pareto_bonus": config.native_default_pareto_bonus,
        "custom_default_hpwl_gate_pct": config.custom_default_hpwl_gate_pct,
        "custom_default_overflow_gate_pct": config.custom_default_overflow_gate_pct,
        "min_custom_default_effect_pct": config.min_custom_default_effect_pct,
        "max_generation_attempts": config.max_generation_attempts,
        "samples_per_iteration": config.samples_per_iteration,
        "map_elites_enabled": config.map_elites_enabled,
        "migration_topology": config.migration_topology,
        "full_rewrite_after_static_rejections": config.full_rewrite_after_static_rejections,
        "reject_baseline_mechanism_clones": config.reject_baseline_mechanism_clones,
        "min_baseline_mechanism_distance": config.min_baseline_mechanism_distance,
        "reject_repeated_negative_mechanisms": config.reject_repeated_negative_mechanisms,
        "min_negative_mechanism_distance": config.min_negative_mechanism_distance,
        "parent_policy": config.parent_policy,
        "selection_policy": config.selection_policy,
        "selection_metrics": {
            "hpwl_weight": config.selection_hpwl_weight,
            "overflow_weight": config.selection_overflow_weight,
            "wns_weight": config.selection_wns_weight,
            "tns_weight": config.selection_tns_weight,
            "max_parent_hpwl_regression_pct": config.max_parent_hpwl_regression_pct,
            "max_parent_overflow_regression_pct": config.max_parent_overflow_regression_pct,
            "require_timing_for_generated_parents": (
                config.require_timing_for_generated_parents
            ),
            "require_timing_improvement_for_elite": (
                config.require_timing_improvement_for_elite
            ),
        },
        "require_nonzero_routing_term": config.require_nonzero_routing_term,
        "min_replacement_nonbaseline_terms": config.min_replacement_nonbaseline_terms,
        "min_state_registers": config.min_state_registers,
        "min_abs_routing_coeff": config.min_abs_routing_coeff,
        "routing_influence": {
            "required": config.require_measured_routing_influence,
            "component": "routing_correction",
            "min_samples": config.min_routing_component_samples,
            "min_gradient_ratio_mean": config.min_routing_gradient_ratio_mean,
            "min_gradient_ratio_peak": config.min_routing_gradient_ratio_peak,
            "role": (
                "A generated routing mechanism must exert a measured nontrivial "
                "gradient contribution during converged DREAMPlace placement. "
                "Static term presence alone is not sufficient."
            ),
        },
        "term_calibration": {
            "enabled": config.term_calibration_enabled,
            "role": (
                "For native-residual runs, the evaluator measures per-design "
                "native and route-term gradients before search and may evaluate "
                "calibrated siblings at comparable gradient ratios."
            ),
            "terms": list(config.term_calibration_terms),
        },
        "active_routing_terms": sorted(ROUTE_CORRECTION_TERMS),
        "timing_proxy_feedback": {
            "enabled": config.timing_proxy_enabled,
            "mode": config.timing_proxy_mode,
            "rc_model": "DREAMPlace/OpenTimer FLUTE-Elmore placement RC",
            "role": (
                "diagnostic timing proxy unless the orthogonality audit supports "
                "stronger use as tie-breaking or gating"
            ),
        },
        "timing_controller": {
            "min_net_coverage": config.timing_controller_min_net_coverage,
            "available": not _timing_controller_uncovered_designs(config),
            "role": (
                "Per-net timing policies are admitted only when DREAMPlace/OpenTimer "
                "reports finite timing and sufficient matched-net coverage."
            ),
        },
        "router_background_path": config.router_background_path,
        "target_design_context": _target_design_context(config),
    }
    if config.term_scale_audit_context:
        policy["term_scale_audit"] = config.term_scale_audit_context
    if config.term_scale_audit_blocked_terms:
        policy["term_scale_blocked_terms"] = list(config.term_scale_audit_blocked_terms)
    design_profile_context = _design_profile_context(config, database=database)
    if design_profile_context:
        policy["chip_design_profiles"] = design_profile_context
    return policy


def _design_profile_context(
    config: OpenEvolveTier2Config,
    *,
    database: ObjectiveProgramDatabase | None = None,
) -> dict[str, Any]:
    if not config.design_profile_enabled:
        return {}
    profile = build_design_profiles(
        search_panel_path=config.search_panel,
        target_designs=config.target_designs,
        generalization_panel_path=(
            config.generalization_panel
            if (
                config.generalization_enabled
                and config.generalization_panel
                and not config.generalization_blind
            )
            else None
        ),
        include_file_stats=config.design_profile_include_file_stats,
    )
    baseline_behavior = (
        _search_baseline_behavior_from_database(database, config)
        if database is not None
        else None
    )
    return merge_baseline_behavior(profile, baseline_behavior)


def _search_baseline_behavior_from_database(
    database: ObjectiveProgramDatabase | None,
    config: OpenEvolveTier2Config,
) -> dict[str, Any] | None:
    if database is None:
        return None
    comparison_csv = None
    for program in database.programs.values():
        if not program.metrics.get("seed_baseline"):
            continue
        tier2_summary = program.artifacts.get("tier2_summary")
        if not isinstance(tier2_summary, dict):
            continue
        candidate = tier2_summary.get("comparison_csv")
        if candidate and Path(str(candidate)).is_file():
            comparison_csv = Path(str(candidate))
            break
    if comparison_csv is None:
        return None
    baseline_ids = {"default", "custom_default"}
    for name in config.search_baseline_presets or config.initial_presets:
        try:
            baseline_ids.add(objective_preset(name).id)
        except KeyError:
            baseline_ids.add(str(name))
    return summarize_baseline_behavior(
        _read_csv(comparison_csv),
        objective_ids=baseline_ids,
    )


def _target_design_context(config: OpenEvolveTier2Config) -> dict[str, Any]:
    """Summarize the design-specific optimization split for prompt context.

    This is intentionally descriptive rather than threshold-like. Search-panel
    designs are visible because they define the current environment. Held-out
    generalization designs are named as withheld validation targets, but their
    metrics are never inserted into iteration prompts.
    """

    search_designs = _panel_design_names(config.search_panel)
    target_designs = list(config.target_designs or search_designs)
    generalization_designs = (
        _panel_design_names(config.generalization_panel)
        if config.generalization_enabled and config.generalization_panel
        else []
    )
    visible_generalization_designs = [] if config.generalization_blind else generalization_designs
    return {
        "optimization_designs": target_designs,
        "search_panel_designs": search_designs,
        "mode": (
            "design_specific"
            if len(target_designs) == 1
            else "multi_design_search"
        ),
        "multi_design_objective_contract": (
            "Generate one shared objective that is evaluated across all "
            "optimization designs. Prefer mechanisms that survive per-design "
            "variation rather than a circuit-specific trick."
            if len(target_designs) > 1
            else "Generate an objective for the configured optimization design."
        ),
        "generalization_designs_withheld_from_feedback": visible_generalization_designs,
        "blind_generalization": config.generalization_blind,
        "heldout_design_count": len(generalization_designs),
        "generalization_policy": (
            "Generalization designs are evaluated after evolution. Their names, "
            "profiles, and metrics are hidden from prompt context and parent "
            "selection in blind mode."
            if generalization_designs and config.generalization_blind
            else "Generalization designs are evaluated after evolution. Their metrics "
            "are held out from parent selection and LLM prompt feedback."
            if generalization_designs
            else "No separate held-out generalization panel is configured."
        ),
    }


def _timing_controller_uncovered_designs(config: OpenEvolveTier2Config) -> list[str]:
    """Search-panel designs on which update_net_weights cannot run.

    A design is covered when the timing-controller panel supplies its timing
    collateral or its DREAMPlace base config is already timing-driven.
    """

    covered: set[str] = set()
    if config.timing_controller_panel:
        try:
            covered = {
                design.name
                for design in load_timing_proxy_panel(config.timing_controller_panel).designs
            }
        except (OSError, ValueError):
            covered = set()
    uncovered = []
    try:
        designs = load_shared_panel(config.search_panel).designs
    except Exception:
        return []
    for design in designs:
        if design.name in covered:
            continue
        try:
            base = json.loads(Path(design.dreamplace_config).read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            base = {}
        if not (base.get("timing_opt_flag") and base.get("enable_net_weighting")):
            uncovered.append(design.name)
    return uncovered


def _panel_design_names(panel_path: str | Path | None) -> list[str]:
    if not panel_path:
        return []
    try:
        return [design.name for design in load_shared_panel(panel_path).designs]
    except Exception:
        return []


def _validate_generalization_split(config: OpenEvolveTier2Config) -> None:
    if not (
        config.generalization_enabled
        and config.generalization_panel
        and config.generalization_strict_leakage_guard
    ):
        return
    search_designs = set(_panel_design_names(config.search_panel))
    target_designs = set(config.target_designs or search_designs)
    heldout_designs = set(_panel_design_names(config.generalization_panel))
    final_designs = set(_panel_design_names(config.final_panel)) if config.final_panel else set()
    if not heldout_designs:
        raise ValueError(
            "strict generalization leakage guard requires a non-empty heldout panel"
        )
    overlaps = {
        "target_designs": sorted(target_designs & heldout_designs),
        "search_panel": sorted(search_designs & heldout_designs),
        "final_panel": sorted(final_designs & heldout_designs),
    }
    active_overlaps = {name: values for name, values in overlaps.items() if values}
    if active_overlaps:
        raise ValueError(
            "heldout generalization leakage detected: "
            + json.dumps(active_overlaps, sort_keys=True)
        )


def _iteration_mutation_mode(
    config: OpenEvolveTier2Config,
    database: ObjectiveProgramDatabase,
) -> str:
    if config.mutation_mode != "diff":
        return config.mutation_mode
    threshold = max(0, config.full_rewrite_after_static_rejections)
    if threshold <= 0:
        return config.mutation_mode
    if _recent_static_mechanism_rejection_count(database) >= threshold:
        return "full"
    return config.mutation_mode


def _recent_static_mechanism_rejection_count(
    database: ObjectiveProgramDatabase,
    *,
    window: int = 5,
) -> int:
    programs = sorted(
        database.programs.values(),
        key=lambda item: (item.iteration_found, item.timestamp),
        reverse=True,
    )
    count = 0
    for program in programs[:window]:
        text = " ".join(
            str(value or "")
            for value in (
                program.failure_reason,
                program.metrics.get("feedback_lesson"),
            )
        )
        if (
            "too close to a measured negative" in text
            or "repeats a mechanism structure" in text
            or "near-miss mechanism" in text
        ):
            count += 1
    return count


def _sample_iteration_parent(
    *,
    config: OpenEvolveTier2Config,
    database: ObjectiveProgramDatabase,
    rng: random.Random,
    island_id: int,
) -> ObjectiveProgram:
    if not _parent_escape_active(config, database):
        return database.sample_parent(
            rng=rng,
            island_id=island_id,
            parent_policy=config.parent_policy,
        )
    candidates = _mechanism_escape_parent_candidates(
        config=config,
        database=database,
        island_id=island_id,
    )
    if not candidates:
        return database.sample_parent(
            rng=rng,
            island_id=island_id,
            parent_policy=config.parent_policy,
        )
    return rng.choices(
        candidates,
        weights=[_escape_parent_weight(candidate) for candidate in candidates],
        k=1,
    )[0]


def _parent_escape_active(
    config: OpenEvolveTier2Config,
    database: ObjectiveProgramDatabase,
) -> bool:
    threshold = max(0, config.escape_parent_after_static_rejections)
    if threshold <= 0:
        return False
    return _recent_static_mechanism_rejection_count(database) >= threshold


def _mechanism_escape_parent_candidates(
    *,
    config: OpenEvolveTier2Config,
    database: ObjectiveProgramDatabase,
    island_id: int | None,
) -> list[ObjectiveProgram]:
    pool = _parent_candidate_pool(database, island_id=island_id)
    eligible = [
        candidate
        for candidate in pool
        if candidate.is_parent_eligible
        and not _candidate_should_escape_as_parent(candidate, database, config)
    ]
    discovered = [candidate for candidate in eligible if not _is_reference_parent(candidate)]
    if discovered:
        return discovered
    if eligible:
        return eligible
    all_eligible = [
        candidate
        for candidate in database.programs.values()
        if candidate.is_parent_eligible
        and not _candidate_should_escape_as_parent(candidate, database, config)
    ]
    discovered = [candidate for candidate in all_eligible if not _is_reference_parent(candidate)]
    if discovered:
        return discovered
    if all_eligible:
        return all_eligible
    return [
        candidate
        for candidate in database.programs.values()
        if candidate.is_parent_eligible and _is_reference_parent(candidate)
    ]


def _parent_candidate_pool(
    database: ObjectiveProgramDatabase,
    *,
    island_id: int | None,
) -> list[ObjectiveProgram]:
    if island_id is None or island_id < 0 or island_id >= len(database.islands):
        return list(database.programs.values())
    pool = [
        database.programs[program_id]
        for program_id in database.islands[island_id]
        if program_id in database.programs
    ]
    return pool or list(database.programs.values())


def _candidate_should_escape_as_parent(
    candidate: ObjectiveProgram,
    database: ObjectiveProgramDatabase,
    config: OpenEvolveTier2Config,
) -> bool:
    if candidate.metrics.get("near_miss_parent_due_to_robustness"):
        return True
    if not (candidate.objective_spec or {}).get("ast"):
        return False
    candidate_ast = _payload_mechanism_view(candidate.objective_spec)
    candidate_signature = candidate.metrics.get("mechanism_signature") or _mechanism_signature(candidate_ast)
    threshold = max(0.0, config.min_negative_mechanism_distance)
    for memory in database.programs.values():
        if not _is_negative_or_near_miss_memory(memory):
            continue
        if not (memory.objective_spec or {}).get("ast"):
            continue
        memory_ast = _payload_mechanism_view(memory.objective_spec)
        memory_signature = memory.metrics.get("mechanism_signature") or _mechanism_signature(memory_ast)
        if candidate_signature == memory_signature:
            return True
        if (
            threshold > 0.0
            and not _is_unmeasured_negative_memory(memory)
            and _mechanism_distance(candidate_ast, memory_ast) < threshold
        ):
            return True
    return False


def _is_negative_or_near_miss_memory(program: ObjectiveProgram) -> bool:
    if program.metrics.get("negative_memory_only"):
        return True
    if program.metrics.get("near_miss_parent_due_to_robustness"):
        return True
    if program.metrics.get("severe_regression_count", 0):
        return True
    text = " ".join(
        str(value or "")
        for value in (
            program.failure_reason,
            program.metrics.get("feedback_lesson"),
        )
    )
    return (
        "too close to a measured negative" in text
        or "repeats a mechanism structure" in text
        or "near-miss mechanism" in text
    )


def _is_reference_parent(program: ObjectiveProgram) -> bool:
    return bool(
        program.metrics.get("seed_baseline")
        or program.metrics.get("manual_safe_baseline")
        or program.metrics.get("bootstrap_parent")
    )


def _escape_parent_weight(candidate: ObjectiveProgram) -> float:
    baseline_distance = _finite_number(candidate.metrics.get("baseline_mechanism_distance"))
    if baseline_distance is None:
        baseline_distance = 0.5
    novelty = 0.5 + min(1.0, max(0.0, baseline_distance))
    score = _finite_number(candidate.metrics.get("combined_score"))
    if score is None:
        score = 0.0
    score = max(score, 0.0)
    role = 0.35 if _is_reference_parent(candidate) else 1.0
    return max((0.05 + math.log1p(score)) * novelty * role, 1e-6)


def _prepare_term_calibration(
    *,
    config: OpenEvolveTier2Config,
    run_root: Path,
    dreamplace_root: str | Path,
    resume: bool,
    retry_failed: bool,
) -> dict[str, Any]:
    if not config.term_calibration_enabled:
        return {"status": "disabled"}
    calibration_dir = run_root / "term_calibration"
    summary_path = calibration_dir / "term_calibration.json"
    if resume and summary_path.is_file() and not retry_failed:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            payload.get("status") == "failed"
            and payload.get("derived_route_coeff_grid")
            and "FrozenInstanceError" in str(payload.get("reason", ""))
        ):
            payload = dict(payload)
            payload["status"] = "completed_recovered"
            payload["recovered_from_reason"] = payload.get("reason")
            values = [
                float(value)
                for value in payload.get("derived_route_coeff_grid", [])
                if _finite_number(value) is not None and float(value) != 0.0
            ]
            if values:
                values = sorted(set(values), key=lambda value: (abs(value), value))
                payload["effective_route_coeff_grid"] = values
                payload["effective_route_coeff_cap"] = max(
                    config.route_coeff_cap,
                    max(abs(value) for value in values),
                )
                payload["effective_min_abs_routing_coeff"] = min(
                    config.min_abs_routing_coeff,
                    min(abs(value) for value in values),
                )
            _write_json(payload, summary_path)
        return payload

    calibration_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "status": "started",
        "objective_mode": config.objective_mode,
        "configured_route_coeff_grid": list(config.route_coeff_grid),
        "gradient_ratios": list(config.term_calibration_gradient_ratios),
        "terms": list(config.term_calibration_terms),
        "results": [],
        "derived_route_coeff_grid": [],
        "fallback_used": True,
    }
    try:
        panel = load_panel(config.search_panel)
        design = _calibration_design(panel, config)
        payload["design"] = design.name
        payload["base_config"] = design.base_config
        payload["seed"] = panel.seeds[0]
        objective_dir = calibration_dir / "objectives"
        objective_dir.mkdir(parents=True, exist_ok=True)
        terms = ["native_objective", *config.term_calibration_terms]
        seen_terms: set[str] = set()
        for term in terms:
            if term in seen_terms:
                continue
            seen_terms.add(term)
            if term != "native_objective" and term not in ROUTE_CORRECTION_TERMS:
                payload["results"].append(
                    {
                        "term": term,
                        "status": "skipped",
                        "reason": "not a native-residual routing correction term",
                    }
                )
                continue
            spec = _term_calibration_spec(term)
            objective_path = write_objective_spec(
                spec,
                objective_dir / f"{_safe_name(term)}.json",
            )
            term_dir = calibration_dir / _safe_name(term)
            config_path = make_run_config(
                design.base_config,
                objective_path,
                term_dir,
                iterations=max(1, config.term_calibration_iterations),
                gpu=panel.gpu,
                seed=panel.seeds[0],
                custom_objective_log_interval=1,
                disable_legalization=True,
            )
            run = run_dreamplace(
                dreamplace_root=dreamplace_root,
                config_path=config_path,
                run_name=f"term_calibration_{design.name}_{term}",
                run_dir=term_dir,
                timeout_seconds=config.term_calibration_timeout_seconds,
            )
            write_run_summary(run, term_dir / "run_summary.json")
            payload["results"].append(_term_calibration_result(term, run))
        derived = _derive_calibrated_route_coeff_grid(payload, config)
        if derived:
            values = sorted(set(derived), key=lambda value: (abs(value), value))
            payload["derived_route_coeff_grid"] = values
            payload["effective_route_coeff_grid"] = values
            payload["effective_route_coeff_cap"] = max(
                config.route_coeff_cap,
                max(abs(value) for value in values),
            )
            payload["effective_min_abs_routing_coeff"] = min(
                config.min_abs_routing_coeff,
                min(abs(value) for value in values if value != 0.0),
            )
            payload["fallback_used"] = False
            payload["status"] = "completed"
        else:
            payload["status"] = "fallback"
            payload["reason"] = (
                "Could not derive finite calibrated route coefficients; using "
                "configured route_coeff_grid."
            )
    except Exception as exc:
        payload["status"] = "failed"
        payload["reason"] = f"{type(exc).__name__}: {exc}"
    _write_json(payload, summary_path)
    return payload


def _apply_calibrated_route_grid(
    config: OpenEvolveTier2Config,
    payload: dict[str, Any],
) -> OpenEvolveTier2Config:
    values = [
        float(value)
        for value in payload.get("derived_route_coeff_grid", [])
        if _finite_number(value) is not None and float(value) != 0.0
    ]
    if not values:
        return config
    grid = sorted(set(values), key=lambda value: (abs(value), value))
    return replace(
        config,
        route_coeff_grid=grid,
        route_coeff_cap=max(config.route_coeff_cap, max(abs(value) for value in grid)),
        min_abs_routing_coeff=min(
            config.min_abs_routing_coeff,
            min(abs(value) for value in grid),
        ),
    )


def _calibration_design(panel: Any, config: OpenEvolveTier2Config) -> Any:
    if config.target_designs:
        wanted = set(config.target_designs)
        for design in panel.designs:
            if design.name in wanted:
                return design
    return panel.designs[0]


def _term_calibration_spec(term: str) -> ObjectiveSpec:
    if term == "native_objective":
        return objective_preset("dreamplace_native_default")
    return parse_objective_spec(
        {
            "id": "",
            "rationale": f"Single-term calibration objective for {term}.",
            "parent_ids": [],
            "declared_term_usage": [term],
            "ast": {"op": "term", "name": term},
        },
        created_by="term_calibration",
        term_scope="dreamplace",
    )


def _term_calibration_result(term: str, run: Any) -> dict[str, Any]:
    custom = dict((run.metrics or {}).get("custom_objective", {}))
    grad = _finite_number(custom.get("last_grad_norm"))
    status = "passed"
    reason = ""
    if int(getattr(run, "returncode", 1) or 0) != 0:
        status = "failed"
        reason = f"DREAMPlace return code {getattr(run, 'returncode', None)}"
    elif int(custom.get("calls") or 0) <= 0:
        status = "failed"
        reason = "custom objective did not execute"
    elif grad is None or grad <= 0.0:
        status = "failed"
        reason = "missing or nonpositive custom objective gradient"
    last_terms = custom.get("last_terms", {})
    term_payload = last_terms.get(term, {}) if isinstance(last_terms, dict) else {}
    return {
        "term": term,
        "status": status,
        "reason": reason,
        "returncode": getattr(run, "returncode", None),
        "calls": custom.get("calls"),
        "gradient_norm": grad,
        "output_scale": custom.get("output_scale"),
        "term_scale": dict(custom.get("term_scales", {})).get(term),
        "raw_value": term_payload.get("raw") if isinstance(term_payload, dict) else None,
        "normalized_value": (
            term_payload.get("normalized") if isinstance(term_payload, dict) else None
        ),
        "config_path": getattr(run, "config_path", None),
        "log_path": getattr(run, "log_path", None),
    }


def _derive_calibrated_route_coeff_grid(
    payload: dict[str, Any],
    config: OpenEvolveTier2Config,
) -> list[float]:
    results = [item for item in payload.get("results", []) if isinstance(item, dict)]
    native = next((item for item in results if item.get("term") == "native_objective"), None)
    native_grad = _finite_number(native.get("gradient_norm") if native else None)
    if native_grad is None or native_grad <= config.term_calibration_min_grad_norm:
        payload["native_gradient_reason"] = "native objective gradient unavailable"
        return []
    route_results = [
        item
        for item in results
        if item.get("term") in ROUTE_CORRECTION_TERMS
        and item.get("status") == "passed"
        and _finite_number(item.get("gradient_norm")) is not None
        and float(item.get("gradient_norm")) > config.term_calibration_min_grad_norm
    ]
    if not route_results:
        payload["route_gradient_reason"] = "no finite route-term gradients"
        return []
    max_route_grad = max(float(item["gradient_norm"]) for item in route_results)
    per_term: dict[str, list[float]] = {}
    for item in route_results:
        grad = float(item["gradient_norm"])
        suggestions = []
        for ratio in config.term_calibration_gradient_ratios:
            coefficient = abs(float(ratio)) * native_grad / grad
            coefficient = min(coefficient, config.route_coeff_cap)
            if coefficient > 0.0:
                suggestions.append(coefficient)
        per_term[str(item["term"])] = suggestions
    payload["per_term_suggested_positive_coefficients"] = per_term
    conservative = []
    for ratio in config.term_calibration_gradient_ratios:
        coefficient = abs(float(ratio)) * native_grad / max_route_grad
        coefficient = min(coefficient, config.route_coeff_cap)
        if coefficient <= 0.0:
            continue
        conservative.extend([coefficient, -coefficient])
    return sorted(set(conservative), key=lambda value: (abs(value), value))


def _prepare_term_scale_audit(
    *,
    config: OpenEvolveTier2Config,
    run_root: Path,
    dreamplace_root: str | Path,
    resume: bool,
    retry_failed: bool,
) -> dict[str, Any]:
    if not config.term_scale_audit_enabled:
        return {"status": "disabled"}
    audit_dir = run_root / "term_scale_audit"
    summary_path = audit_dir / "term_scale_audit.json"
    if (
        resume
        and config.term_scale_audit_reuse_existing
        and summary_path.is_file()
        and not retry_failed
    ):
        return json.loads(summary_path.read_text(encoding="utf-8"))

    audit_dir.mkdir(parents=True, exist_ok=True)
    configured_terms = list(dict.fromkeys(config.term_scale_audit_terms))
    replacement_terms = set(term_names("dreamplace_replacement"))
    audit_terms = list(
        dict.fromkeys(
            [
                *TERM_SCALE_AUDIT_REFERENCE_TERMS,
                *[term for term in configured_terms if term in replacement_terms],
            ]
        )
    )
    payload: dict[str, Any] = {
        "status": "started",
        "objective_mode": config.objective_mode,
        "terms": audit_terms,
        "configured_terms": configured_terms,
        "reference_terms": TERM_SCALE_AUDIT_REFERENCE_TERMS,
        "results": [],
        "blocked_terms": [],
        "max_runtime_seconds": config.term_scale_audit_max_runtime_seconds,
        "warnings": [],
    }
    try:
        panel = load_panel(config.search_panel)
        wanted = set(config.target_designs or [])
        designs = [
            design
            for design in panel.designs
            if not wanted or design.name in wanted
        ]
        if not designs:
            designs = list(panel.designs)
        payload["designs"] = [design.name for design in designs]
        objective_dir = audit_dir / "objectives"
        objective_dir.mkdir(parents=True, exist_ok=True)
        for design in designs:
            design_results = []
            for term in audit_terms:
                spec = _term_scale_audit_spec(term)
                objective_path = write_objective_spec(
                    spec,
                    objective_dir / f"{_safe_name(design.name)}_{_safe_name(term)}.json",
                )
                term_dir = audit_dir / _safe_name(design.name) / _safe_name(term)
                config_path = make_run_config(
                    design.base_config,
                    objective_path,
                    term_dir,
                    iterations=max(1, config.term_scale_audit_iterations),
                    gpu=panel.gpu,
                    seed=panel.seeds[0],
                    custom_objective_log_interval=1,
                    disable_legalization=True,
                )
                run = run_dreamplace(
                    dreamplace_root=dreamplace_root,
                    config_path=config_path,
                    run_name=f"term_scale_audit_{design.name}_{term}",
                    run_dir=term_dir,
                    timeout_seconds=(
                        min(
                            config.term_scale_audit_timeout_seconds,
                            max(1, math.ceil(config.term_scale_audit_max_runtime_seconds)),
                        )
                        if config.term_scale_audit_max_runtime_seconds > 0.0
                        else config.term_scale_audit_timeout_seconds
                    ),
                )
                write_run_summary(run, term_dir / "run_summary.json")
                result = _term_scale_audit_result(
                    design=design.name,
                    term=term,
                    run=run,
                    min_grad_norm=config.term_scale_audit_min_grad_norm,
                    max_runtime_seconds=config.term_scale_audit_max_runtime_seconds,
                )
                payload["results"].append(result)
                design_results.append(result)
            _add_term_scale_reference_ratios(design_results)
        _finalize_term_scale_audit(payload, config)
    except Exception as exc:
        payload["status"] = "failed"
        payload["reason"] = f"{type(exc).__name__}: {exc}"
        payload["warnings"].append("term scale audit failed before completion")
    _write_json(payload, summary_path)
    (audit_dir / "term_scale_audit.md").write_text(
        _term_scale_audit_markdown(payload),
        encoding="utf-8",
    )
    return payload


def _term_scale_audit_spec(term: str) -> ObjectiveSpec:
    if term == "native_objective":
        return objective_preset("dreamplace_native_default")
    return parse_objective_spec(
        {
            "id": "",
            "rationale": f"Single-term scale audit objective for {term}.",
            "parent_ids": [],
            "declared_term_usage": [term],
            "ast": {"op": "term", "name": term},
        },
        created_by="term_scale_audit",
        term_scope="dreamplace",
    )


def _term_scale_audit_result(
    *,
    design: str,
    term: str,
    run: Any,
    min_grad_norm: float,
    max_runtime_seconds: float = 0.0,
) -> dict[str, Any]:
    custom = dict((run.metrics or {}).get("custom_objective", {}))
    grad = _finite_number(custom.get("last_grad_norm"))
    custom_calls = int(custom.get("calls") or 0)
    last_terms = custom.get("last_terms", {})
    term_payload = last_terms.get(term, {}) if isinstance(last_terms, dict) else {}
    raw = term_payload.get("raw") if isinstance(term_payload, dict) else None
    normalized = term_payload.get("normalized") if isinstance(term_payload, dict) else None
    warnings = []
    status = "passed"
    reason = ""
    if int(getattr(run, "returncode", 1) or 0) != 0:
        status = "failed"
        reason = f"DREAMPlace return code {getattr(run, 'returncode', None)}"
        warnings.append("dreamplace_failed")
    if custom_calls <= 0:
        status = "failed"
        reason = reason or "custom objective did not execute"
        warnings.append("custom_not_called")
    if _finite_number(raw) is None:
        warnings.append("nonfinite_or_missing_raw_value")
    if _finite_number(normalized) is None:
        warnings.append("nonfinite_or_missing_normalized_value")
    if grad is None:
        status = "failed"
        reason = reason or "missing gradient"
        warnings.append("missing_gradient")
    elif grad <= min_grad_norm:
        status = "failed"
        reason = reason or "near-zero gradient"
        warnings.append("near_zero_gradient")
    normalized_value = _finite_number(normalized)
    if normalized_value is not None and abs(normalized_value) > 1e6:
        warnings.append("oversized_normalized_value")
    runtime_seconds = _finite_number(getattr(run, "runtime_seconds", None))
    if (
        max_runtime_seconds > 0.0
        and runtime_seconds is not None
        and runtime_seconds > max_runtime_seconds
    ):
        status = "failed"
        reason = reason or (
            f"runtime {runtime_seconds:.3f}s exceeds term audit limit "
            f"{max_runtime_seconds:.3f}s"
        )
        warnings.append("excessive_runtime")
    return {
        "design": design,
        "term": term,
        "status": status,
        "reason": reason,
        "returncode": getattr(run, "returncode", None),
        "calls": custom_calls,
        "raw_value": raw,
        "normalized_value": normalized,
        "gradient_norm": grad,
        "output_scale": custom.get("output_scale"),
        "term_scale": dict(custom.get("term_scales", {})).get(term),
        "runtime_seconds": getattr(run, "runtime_seconds", None),
        "max_runtime_seconds": max_runtime_seconds,
        "warnings": warnings,
        "config_path": getattr(run, "config_path", None),
        "log_path": getattr(run, "log_path", None),
    }


def _add_term_scale_reference_ratios(results: list[dict[str, Any]]) -> None:
    by_term = {str(item.get("term")): item for item in results}
    references = {
        "wirelength_wawl": _finite_number(
            by_term.get("wirelength_wawl", {}).get("gradient_norm")
        ),
        "density_electric": _finite_number(
            by_term.get("density_electric", {}).get("gradient_norm")
        ),
        "native_objective": _finite_number(
            by_term.get("native_objective", {}).get("gradient_norm")
        ),
    }
    for item in results:
        grad = _finite_number(item.get("gradient_norm"))
        ratios = {}
        for name, ref_grad in references.items():
            if grad is not None and ref_grad is not None and ref_grad > 0.0:
                ratios[f"gradient_ratio_to_{name}"] = grad / ref_grad
        item["reference_gradient_ratios"] = ratios


def _finalize_term_scale_audit(
    payload: dict[str, Any],
    config: OpenEvolveTier2Config,
) -> None:
    results = [item for item in payload.get("results", []) if isinstance(item, dict)]
    visible_terms = set(config.term_scale_audit_terms)
    blocked_terms = sorted(
        {
            str(item.get("term"))
            for item in results
            if item.get("term") in visible_terms and item.get("status") != "passed"
        }
    )
    core_failed = sorted(TERM_SCALE_CORE_TERMS.intersection(blocked_terms))
    payload["blocked_terms"] = blocked_terms
    payload["core_failed_terms"] = core_failed
    payload["prompt_context"] = _term_scale_prompt_context(payload)
    if core_failed and config.term_scale_audit_fail_on_broken_core_terms:
        payload["status"] = "failed_core_terms"
        payload["warnings"].append(f"core terms failed audit: {core_failed}")
    elif any(item.get("status") == "passed" for item in results):
        payload["status"] = "completed"
    else:
        payload["status"] = "failed"
        payload["warnings"].append("no audited term passed")


def _term_scale_prompt_context(payload: dict[str, Any]) -> dict[str, Any]:
    results = [item for item in payload.get("results", []) if isinstance(item, dict)]
    by_design: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        if item.get("term") == "native_objective":
            continue
        by_design.setdefault(str(item.get("design") or "unknown"), []).append(
            {
                "term": item.get("term"),
                "status": item.get("status"),
                "raw_value": _compact_number(item.get("raw_value")),
                "normalized_value": _compact_number(item.get("normalized_value")),
                "gradient_norm": _compact_number(item.get("gradient_norm")),
                "gradient_ratio_to_wirelength_wawl": _compact_number(
                    dict(item.get("reference_gradient_ratios") or {}).get(
                        "gradient_ratio_to_wirelength_wawl"
                    )
                ),
                "gradient_ratio_to_density_electric": _compact_number(
                    dict(item.get("reference_gradient_ratios") or {}).get(
                        "gradient_ratio_to_density_electric"
                    )
                ),
                "warnings": item.get("warnings") or [],
            }
        )
    return {
        "source": "deterministic_pre_search_dreamplace_single_term_audit",
        "status": payload.get("status"),
        "blocked_terms": payload.get("blocked_terms", []),
        "interpretation": (
            "These values describe single-term DREAMPlace scale and gradient "
            "behavior on the configured search designs. They are environment "
            "context, not final quality labels."
        ),
        "by_design": by_design,
    }


def _term_scale_audit_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Term-Scale Audit",
        "",
        f"- Status: {payload.get('status')}",
        f"- Designs: {', '.join(str(item) for item in payload.get('designs', [])) or 'none'}",
        f"- Blocked prompt-visible terms: {', '.join(payload.get('blocked_terms', [])) or 'none'}",
        "",
        "| design | term | status | raw | normalized | grad_norm | warnings |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for item in payload.get("results", []):
        if not isinstance(item, dict):
            continue
        lines.append(
            "| "
            f"{item.get('design')} | {item.get('term')} | {item.get('status')} | "
            f"{_compact_number(item.get('raw_value'))} | "
            f"{_compact_number(item.get('normalized_value'))} | "
            f"{_compact_number(item.get('gradient_norm'))} | "
            f"{', '.join(item.get('warnings') or []) or '-'} |"
        )
    if payload.get("reason"):
        lines.extend(["", f"- Failure reason: {payload.get('reason')}"])
    return "\n".join(lines).rstrip() + "\n"


def _compact_number(value: Any) -> Any:
    numeric = _finite_number(value)
    if numeric is None:
        return None
    if numeric == 0.0:
        return 0.0
    if abs(numeric) >= 1e4 or abs(numeric) < 1e-3:
        return f"{numeric:.3e}"
    return round(numeric, 6)


def _validate_scoring_iteration_budgets(
    config: OpenEvolveTier2Config,
) -> dict[str, Any]:
    minimum = max(1, int(config.minimum_scoring_iterations))
    panel_paths: list[tuple[str, str]] = [("search", config.search_panel)]
    if config.final_enabled and config.final_panel:
        panel_paths.append(("final", config.final_panel))
    if config.generalization_enabled and config.generalization_panel:
        panel_paths.append(("generalization", config.generalization_panel))

    panels = []
    insufficient = []
    for role, path in panel_paths:
        panel = load_panel(path)
        row = {
            "role": role,
            "panel": str(path),
            "iterations": int(panel.iterations),
            "minimum_iterations": minimum,
            "budget_satisfied": int(panel.iterations) >= minimum,
        }
        panels.append(row)
        if not row["budget_satisfied"]:
            insufficient.append(row)

    test_override = bool(
        insufficient
        and config.provider == "mock"
        and config.allow_short_search_for_tests
    )
    if insufficient and not test_override:
        details = ", ".join(
            f"{item['role']}={item['iterations']}" for item in insufficient
        )
        raise ValueError(
            "OpenEvolve scoring requires completed DREAMPlace placement budgets "
            f"of at least {minimum} iterations for native and generated objectives; "
            f"insufficient panels: {details}. Short runs are non-scoring tests only."
        )
    return {
        "minimum_scoring_iterations": minimum,
        "claim_valid": not insufficient,
        "test_only_override": test_override,
        "panels": panels,
    }


def run_openevolve_tier2(
    *,
    config_path: str | Path,
    run_dir: str | Path,
    dreamplace_root: str | Path,
    chipbench_root: str | Path | None = None,
    resume: bool = False,
    retry_failed: bool = False,
) -> dict[str, Any]:
    config = load_openevolve_tier2_config(config_path)
    _validate_generalization_split(config)
    if config.standardization_official_mode:
        if config.objective_mode != config.standardization_official_mode:
            if not (
                config.objective_mode == "native_residual"
                and config.standardization_allow_native_residual_fallback
            ):
                raise ValueError(
                    "configured objective_mode does not match official "
                    f"standardization mode {config.standardization_official_mode!r}: "
                    f"{config.objective_mode!r}"
                )
    if config.objective_mode == "controller":
        if config.term_scope != "dreamplace_controller":
            raise ValueError(
                "controller evolution requires term_scope='dreamplace_controller'"
            )
    elif config.objective_mode == "native_residual":
        if config.term_scope != "dreamplace_native_residual":
            raise ValueError(
                "native_residual openevolve-tier2 runs require "
                "term_scope='dreamplace_native_residual' so only "
                "term(\"native_objective\") plus routing corrections are available "
                "to the provider schema"
            )
    elif config.term_scope != "dreamplace_replacement":
        raise ValueError(
            "replacement/residual openevolve-tier2 runs require "
            "term_scope='dreamplace_replacement'"
        )

    scoring_iteration_policy = _validate_scoring_iteration_budgets(config)
    if config.tier3_enabled and config.tier3_interval > 0 and chipbench_root is None:
        raise RuntimeError(
            "scheduled routed evaluation requires chipbench_root"
        )

    run_root = Path(run_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        scoring_iteration_policy,
        run_root / "scoring_iteration_policy.json",
    )
    macro_preflight = {
        "status": "disabled",
        "ok": True,
        "designs": [],
    }
    if config.macro_preflight_enabled:
        macro_preflight = audit_shared_panel_macros(
            config.search_panel,
            dreamplace_root=dreamplace_root,
            require_movable_macros=config.macro_preflight_require_movable_macros,
            reject_fixed_hard_macros=config.macro_preflight_reject_fixed_hard_macros,
            minimum_movable_macro_count=(
                config.macro_preflight_minimum_movable_macro_count
            ),
        )
        macro_preflight["status"] = "passed" if macro_preflight["ok"] else "failed"
        _write_json(macro_preflight, run_root / "macro_preflight.json")
        if not macro_preflight["ok"]:
            failures = {
                item["design"]: item["failures"]
                for item in macro_preflight["designs"]
                if not item["ok"]
            }
            raise RuntimeError(
                "Movable-macro preflight failed before term evaluation or LLM calls: "
                + json.dumps(failures, sort_keys=True)
            )
    term_scale_audit = _prepare_term_scale_audit(
        config=config,
        run_root=run_root,
        dreamplace_root=dreamplace_root,
        resume=resume,
        retry_failed=retry_failed,
    )
    blocked_terms = [
        str(term)
        for term in term_scale_audit.get("blocked_terms", [])
        if term in term_names("dreamplace_replacement")
    ]
    term_scale_context = dict(term_scale_audit.get("prompt_context") or {})
    if blocked_terms or term_scale_context:
        config = replace(
            config,
            term_scale_audit_blocked_terms=blocked_terms,
            term_scale_audit_context=term_scale_context,
        )
    if (
        config.term_scale_audit_enabled
        and config.term_scale_audit_fail_on_broken_core_terms
        and term_scale_audit.get("status") == "failed_core_terms"
    ):
        _write_json(term_scale_audit, run_root / "term_scale_audit.json")
        raise RuntimeError(
            "Term-scale audit failed for core replacement terms: "
            f"{term_scale_audit.get('core_failed_terms')}"
        )
    term_calibration = _prepare_term_calibration(
        config=config,
        run_root=run_root,
        dreamplace_root=dreamplace_root,
        resume=resume,
        retry_failed=retry_failed,
    )
    config = _apply_calibrated_route_grid(config, term_calibration)
    _write_json(asdict(config), run_root / "config_resolved.json")
    if term_scale_audit and term_scale_audit.get("status") != "disabled":
        _write_json(term_scale_audit, run_root / "term_scale_audit.json")
        (run_root / "term_scale_audit.md").write_text(
            _term_scale_audit_markdown(term_scale_audit),
            encoding="utf-8",
        )
    if term_calibration:
        _write_json(term_calibration, run_root / "term_calibration.json")
    if config.router_background:
        (run_root / "router_objective_background.md").write_text(
            config.router_background,
            encoding="utf-8",
        )

    rng = random.Random(config.seed)
    provider = provider_from_name(config.provider)
    database = ObjectiveProgramDatabase(
        run_root / "program_db",
        ObjectiveDatabaseConfig(
            population_size=config.population_size,
            archive_size=config.archive_size,
            num_islands=config.num_islands,
            map_elites_enabled=config.map_elites_enabled,
            feature_bins=config.feature_bins,
            feature_dimensions=tuple(config.feature_dimensions),
            timing_proxy_wns_delta_min_abs_ns=config.timing_proxy_wns_delta_min_abs_ns,
            migration_interval=config.migration_interval,
            migration_rate=config.migration_rate,
            migration_topology=config.migration_topology,
            migration_clock=config.migration_clock,
        ),
    )
    timing_admission = _load_timing_admission(config, run_root)
    if resume:
        database.load()
        if not database.is_empty:
            _refresh_database_scores(database, config)
            _refresh_pareto_selection(database, config, timing_admission)
            database.save(iteration=database.last_iteration)
    if database.is_empty:
        _seed_initial_programs(
            database,
            config=config,
            run_root=run_root,
            dreamplace_root=dreamplace_root,
            resume=resume,
            retry_failed=retry_failed,
        )
        _refresh_pareto_selection(database, config, timing_admission)
        database.save(iteration=0)
    if not _timing_admission_path(run_root).is_file():
        # Audit the timing proxy on the seed placements before it can
        # influence the first generation.
        initial_audit = _run_scheduled_timing_proxy_audit(
            config=config,
            database=database,
            iteration=0,
            run_root=run_root,
            dreamplace_root=dreamplace_root,
        )
        if initial_audit is not None:
            timing_admission = initial_audit
            _refresh_pareto_selection(database, config, timing_admission)
            database.save(iteration=database.last_iteration)
    if not any(program.is_parent_eligible for program in database.programs.values()):
        promoted = _promote_bootstrap_seed_parent(database)
        if promoted is not None:
            database.add(promoted, target_island=promoted.island)
            database.save(iteration=database.last_iteration)
    if not any(program.is_parent_eligible for program in database.programs.values()):
        raise RuntimeError(
            "No parent-eligible seed baseline is available. Initial presets must "
            "complete real DREAMPlace evaluation and pass the configured selection "
            "gates before LLM evolution can start."
        )
    design_profile_context = _design_profile_context(config, database=database)
    if design_profile_context:
        _write_json(design_profile_context, run_root / "design_profiles.json")
        _write_json(design_profile_context, run_root / "training_design_profiles.json")

    trace = ObjectiveEvolutionTrace(run_root / "evolution_trace.jsonl")
    iteration_summaries = []
    start_iteration = database.last_iteration + 1 if resume and database.last_iteration else 1

    for iteration in range(start_iteration, config.max_iterations + 1):
        prior_feedback = _load_iteration_feedback(config, run_root=run_root)
        active_island = (iteration - 1) % config.num_islands
        parent = _sample_iteration_parent(
            config=config,
            database=database,
            rng=rng,
            island_id=active_island,
        )
        mutation_mode = _iteration_mutation_mode(config, database)
        sampler = ObjectivePromptSampler(
            term_scope=config.term_scope,
            num_top_programs=config.num_top_programs,
            num_diverse_programs=config.num_diverse_programs,
            objective_mode=config.objective_mode,
            mutation_mode=mutation_mode,
            policy=_prompt_policy_from_config(config, database=database),
            router_background=config.router_background,
        )
        iteration_dir = run_root / f"iteration_{iteration:04d}"
        context = sampler.build_context(
            parent=parent,
            database=database,
            iteration=iteration,
            rng=rng,
            run_feedback=prior_feedback,
        )
        if mutation_mode != config.mutation_mode:
            context["adaptive_mutation_reason"] = (
                "Switched from diff to full rewrite because recent measured "
                "static rejections repeated negative mechanism families."
            )
        programs = _generate_and_evaluate_children(
            provider=provider,
            config=config,
            context=context,
            parent=parent,
            iteration=iteration,
            iteration_dir=iteration_dir,
            dreamplace_root=dreamplace_root,
            resume=resume,
            retry_failed=retry_failed,
            database=database,
        )
        for program in programs:
            database.add(program, target_island=parent.island)
        _refresh_pareto_selection(database, config, timing_admission)
        scheduled_tier3 = _run_scheduled_tier3_feedback(
            config=config,
            database=database,
            programs=programs,
            iteration=iteration,
            run_root=run_root,
            chipbench_root=chipbench_root,
            resume=resume,
            retry_failed=retry_failed,
        )
        scheduled_audit = _run_scheduled_timing_proxy_audit(
            config=config,
            database=database,
            iteration=iteration,
            run_root=run_root,
            dreamplace_root=dreamplace_root,
        )
        if scheduled_audit is not None:
            timing_admission = scheduled_audit
        if scheduled_tier3 is not None or scheduled_audit is not None:
            _refresh_pareto_selection(database, config, timing_admission)
        database.increment_island_generation(active_island)
        migrants = database.maybe_migrate(iteration=iteration, rng=rng)
        if migrants:
            # Algorithm 1: the archive is Pareto-refreshed after migration.
            _refresh_pareto_selection(database, config, timing_admission)
        database.save(iteration=iteration)
        for program in programs:
            trace.append(iteration=iteration, parent=parent, child=program)
        for migrant in migrants:
            trace.append(iteration=iteration, parent=database.get(migrant.parent_id), child=migrant)
        primary_program = _best_iteration_program(programs)
        _write_json([program.to_dict() for program in programs], iteration_dir / "program_records.json")
        _write_json(primary_program.to_dict(), iteration_dir / "program_record.json")
        summary = {
            "iteration": iteration,
            "island": active_island,
            "parent_id": parent.id,
            "program_id": primary_program.id,
            "program_ids": [program.id for program in programs],
            "status": primary_program.status,
            "failure_reason": primary_program.failure_reason,
            "combined_score": primary_program.metrics.get("combined_score"),
            "metrics": primary_program.metrics,
            "generated_sample_count": max(1, config.samples_per_iteration),
            "valid_sample_count": sum(1 for program in programs if program.objective_spec),
            "calibrated_program_count": sum(
                1 for program in programs if program.metrics.get("calibration_report")
            ),
            "best_child_id": primary_program.id,
            "migration_count": len(migrants),
            "scheduled_tier3": scheduled_tier3,
            "scheduled_timing_audit": (
                {
                    "authority": scheduled_audit.get("authority"),
                    "counts": scheduled_audit.get("counts"),
                }
                if scheduled_audit is not None
                else None
            ),
            "failed_reasons": [
                program.failure_reason
                for program in programs
                if program.failure_reason
            ],
        }
        _write_json(summary, iteration_dir / "iteration_summary.json")
        iteration_summaries.append(summary)
        if config.checkpoint_interval and iteration % config.checkpoint_interval == 0:
            checkpoint = database.checkpoint(run_root / "checkpoints", iteration)
            _write_json({"checkpoint": str(checkpoint), "iteration": iteration}, checkpoint / "summary.json")

    final_summary = None
    generalization_summary = None
    tier3_summary = None
    if config.final_enabled and config.final_panel:
        final_summary = _run_final_panel(
            config=config,
            database=database,
            run_root=run_root,
            dreamplace_root=dreamplace_root,
            resume=resume,
            retry_failed=retry_failed,
        )
        if final_summary and config.tier3_enabled:
            if chipbench_root is None:
                raise RuntimeError("Tier-3 is enabled but chipbench_root was not provided")
            tier3_summary = _run_tier3_for_finalists(
                config=config,
                database=database,
                final_summary=final_summary,
                run_root=run_root,
                chipbench_root=chipbench_root,
                resume=resume,
                retry_failed=retry_failed,
            )
    if config.generalization_enabled and config.generalization_panel:
        generalization_summary = _run_generalization_panel(
            config=config,
            database=database,
            run_root=run_root,
            dreamplace_root=dreamplace_root,
            resume=resume,
            retry_failed=retry_failed,
        )
        if generalization_summary and config.tier3_enabled:
            if chipbench_root is None:
                raise RuntimeError("Tier-3 is enabled but chipbench_root was not provided")
            generalization_summary["post_route"] = _run_tier3_for_finalists(
                config=config,
                database=database,
                final_summary=generalization_summary,
                run_root=run_root,
                chipbench_root=chipbench_root,
                resume=resume,
                retry_failed=retry_failed,
                output_name="generalization_post_route",
            )

    best_program = database.get(database.best_program_id)
    if best_program:
        (run_root / "best_program.py").write_text(best_program.code, encoding="utf-8")
        (run_root / "best_objective.py").write_text(best_program.code, encoding="utf-8")
        if best_program.objective_spec:
            _write_json(best_program.objective_spec, run_root / "best_objective.json")
    robust_best_program = _select_best_robust_generated_program(database)
    if robust_best_program:
        (run_root / "robust_best_program.py").write_text(
            robust_best_program.code,
            encoding="utf-8",
        )
        if robust_best_program.objective_spec:
            _write_json(
                robust_best_program.objective_spec,
                run_root / "robust_best_objective.json",
            )
            if "chipbench" in str(config.search_panel).lower():
                _write_json(
                    robust_best_program.objective_spec,
                    run_root / "frozen_nangate45_champion.json",
                )
    aggregate_best_generated = _select_best_aggregate_generated_program(database)
    pareto_best_generated = _select_best_pareto_generated_program(database)
    if pareto_best_generated:
        (run_root / "pareto_best_program.py").write_text(
            pareto_best_generated.code,
            encoding="utf-8",
        )
        if pareto_best_generated.objective_spec:
            _write_json(
                pareto_best_generated.objective_spec,
                run_root / "pareto_best_objective.json",
            )

    cumulative_iteration_summaries = _load_cumulative_iteration_summaries(run_root)
    if not cumulative_iteration_summaries:
        cumulative_iteration_summaries = iteration_summaries

    summary = {
        "run_dir": str(run_root),
        "config": str(config_path),
        "provider_metadata": provider.metadata(),
        "scoring_iteration_policy": scoring_iteration_policy,
        "iterations": cumulative_iteration_summaries,
        "program_count": len(database.programs),
        "archive_size": len(database.archive),
        "map_elites": database.map_elites_summary(),
        "best_program_id": database.best_program_id,
        "best_program_metrics": best_program.metrics if best_program else None,
        "robust_best_program_id": robust_best_program.id if robust_best_program else None,
        "robust_best_program_metrics": robust_best_program.metrics if robust_best_program else None,
        "aggregate_best_generated_program_id": (
            aggregate_best_generated.id if aggregate_best_generated else None
        ),
        "aggregate_best_generated_program_metrics": (
            aggregate_best_generated.metrics if aggregate_best_generated else None
        ),
        "pareto_best_generated_program_id": (
            pareto_best_generated.id if pareto_best_generated else None
        ),
        "pareto_best_generated_program_metrics": (
            pareto_best_generated.metrics if pareto_best_generated else None
        ),
        "term_calibration": (
            str(run_root / "term_calibration.json")
            if (run_root / "term_calibration.json").is_file()
            else None
        ),
        "term_scale_audit": (
            str(run_root / "term_scale_audit.json")
            if (run_root / "term_scale_audit.json").is_file()
            else None
        ),
        "macro_preflight": (
            str(run_root / "macro_preflight.json")
            if (run_root / "macro_preflight.json").is_file()
            else None
        ),
        "final_tier2": final_summary,
        "generalization_tier2": generalization_summary,
        "final_tier3": tier3_summary,
        "design_profiles": (
            str(run_root / "design_profiles.json")
            if (run_root / "design_profiles.json").is_file()
            else None
        ),
        "program_db": str(database.root),
        "trace": str(run_root / "evolution_trace.jsonl"),
    }
    _write_json(summary, run_root / "summary.json")
    (run_root / "openevolve_tier2_report.md").write_text(
        _build_report(summary, database),
        encoding="utf-8",
    )
    (run_root / "train_vs_heldout_report.md").write_text(
        _build_train_vs_heldout_report(summary, database),
        encoding="utf-8",
    )
    return summary


def _select_best_robust_generated_program(
    database: ObjectiveProgramDatabase,
) -> ObjectiveProgram | None:
    candidates = [
        program
        for program in database.programs.values()
        if _is_generated_candidate(program)
        and program.status == "accepted"
        and program.is_elite_eligible
        and bool(program.metrics.get("robust_gate_passed"))
        and bool(program.metrics.get("hpwl_gate_passed"))
        and bool(program.metrics.get("overflow_gate_passed"))
        and bool(program.metrics.get("routing_aware_tier2_candidate"))
        and not program.metrics.get("exploration_parent_only")
        and (
            bool(program.metrics.get("manuscript_multimetric_selection"))
            or
            not program.metrics.get("baseline_portfolio_compared")
            or (
                bool(program.metrics.get("beats_baseline_portfolio"))
                and bool(program.metrics.get("baseline_portfolio_pareto_dominates_best"))
            )
        )
        and not program.metrics.get("structural_failure_count", 0)
        and not program.metrics.get("severe_regression_count", 0)
    ]
    if not candidates:
        return None
    return max(candidates, key=_robust_program_selection_key)


def _best_iteration_program(programs: list[ObjectiveProgram]) -> ObjectiveProgram:
    if not programs:
        raise RuntimeError("evolution iteration produced no program records")
    return max(
        programs,
        key=lambda program: (
            _finite_for_sort(program.metrics.get("combined_score"), default=0.0),
            -int(program.metrics.get("structural_failure_count", 0) or 0),
            -int(program.metrics.get("severe_regression_count", 0) or 0),
            program.status == "accepted",
            str(program.id),
        ),
    )


def _select_best_aggregate_generated_program(
    database: ObjectiveProgramDatabase,
) -> ObjectiveProgram | None:
    candidates = [
        program
        for program in database.programs.values()
        if _is_generated_candidate(program)
        and program.status == "accepted"
        and program.is_elite_eligible
        and bool(program.metrics.get("routing_aware_tier2_candidate"))
        and (
            bool(program.metrics.get("manuscript_multimetric_selection"))
            or
            not program.metrics.get("baseline_portfolio_compared")
            or (
                bool(program.metrics.get("beats_baseline_portfolio"))
                and bool(program.metrics.get("baseline_portfolio_pareto_dominates_best"))
            )
        )
        and not program.metrics.get("structural_failure_count", 0)
    ]
    if not candidates:
        return None
    return max(candidates, key=_aggregate_program_selection_key)


def _select_best_pareto_generated_program(
    database: ObjectiveProgramDatabase,
) -> ObjectiveProgram | None:
    candidates = [
        program
        for program in database.programs.values()
        if _is_generated_candidate(program)
        and program.status == "accepted"
        and bool(program.metrics.get("routing_aware_tier2_candidate"))
        and bool(program.metrics.get("beats_baseline_portfolio"))
        and bool(program.metrics.get("baseline_portfolio_pareto_dominates_best"))
        and not program.metrics.get("structural_failure_count", 0)
        and not program.metrics.get("severe_regression_count", 0)
    ]
    if not candidates:
        return None
    return max(candidates, key=_pareto_program_selection_key)


def _is_generated_candidate(program: ObjectiveProgram) -> bool:
    return not (
        program.metrics.get("seed_baseline")
        or program.metrics.get("manual_safe_baseline")
        or program.metrics.get("bootstrap_parent")
    )


def _robust_program_selection_key(program: ObjectiveProgram) -> tuple[Any, ...]:
    metrics = program.metrics
    return (
        (
            _finite_for_sort(metrics.get("combined_score"), default=0.0)
            if metrics.get("manuscript_multimetric_selection")
            else 0.0
        ),
        bool(metrics.get("manuscript_timing_improved")),
        bool(metrics.get("beats_baseline_portfolio")),
        _native_default_label_from_metrics(metrics) == "dominates_native_default",
        _finite_for_sort(metrics.get("combined_score"), default=0.0),
        -_finite_for_sort(metrics.get("tier2_average_rank"), default=999.0),
        -_finite_for_sort(metrics.get("hpwl_delta_pct"), default=999.0),
        -_finite_for_sort(metrics.get("overflow_delta_pct"), default=999.0),
        -_finite_for_sort(metrics.get("runtime_seconds"), default=999999.0),
        str(program.id),
    )


def _aggregate_program_selection_key(program: ObjectiveProgram) -> tuple[Any, ...]:
    metrics = program.metrics
    return (
        bool(metrics.get("beats_baseline_portfolio")),
        _native_default_label_from_metrics(metrics) == "dominates_native_default",
        bool(metrics.get("robust_gate_passed")),
        bool(metrics.get("hpwl_gate_passed")),
        bool(metrics.get("overflow_gate_passed")),
        -int(metrics.get("severe_regression_count", 0) or 0),
        -_finite_for_sort(metrics.get("tier2_average_rank"), default=999.0),
        -_finite_for_sort(metrics.get("native_default_hpwl_delta_pct"), default=999.0),
        -_finite_for_sort(metrics.get("native_default_overflow_delta_pct"), default=999.0),
        str(program.id),
    )


def _pareto_program_selection_key(program: ObjectiveProgram) -> tuple[Any, ...]:
    metrics = program.metrics
    return (
        bool(metrics.get("robust_gate_passed")),
        bool(metrics.get("hpwl_gate_passed")),
        bool(metrics.get("overflow_gate_passed")),
        bool(metrics.get("beats_baseline_portfolio")),
        bool(metrics.get("parent_eligible")),
        _finite_for_sort(metrics.get("baseline_portfolio_effect_pct"), default=0.0),
        _finite_for_sort(metrics.get("combined_score"), default=0.0),
        -_finite_for_sort(metrics.get("tier2_average_rank"), default=999.0),
        -_finite_for_sort(metrics.get("runtime_seconds"), default=999999.0),
        str(program.id),
    )


def _finite_for_sort(value: Any, *, default: float) -> float:
    numeric = _finite_number(value)
    return default if numeric is None else numeric


def _load_cumulative_iteration_summaries(run_root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(run_root.glob("iteration_*/iteration_summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            summaries.append(payload)
    return summaries


def build_openevolve_prompt_dry_run(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    from_run_dir: str | Path | None = None,
    parent_preset: str | None = None,
    iteration: int = 1,
) -> dict[str, Any]:
    """Build and audit one API-bound OpenEvolve prompt without provider calls.

    This is a cheap methodology check: it verifies the exact prompt shape, term
    environment, and hidden-policy audit before spending LLM or DREAMPlace time.
    When ``from_run_dir`` contains an existing ``program_db``, the prompt uses
    that real local memory. Otherwise it creates reference seed programs from
    the configured presets with clearly marked synthetic prompt-only metrics.
    """

    config = load_openevolve_tier2_config(config_path)
    if config.objective_mode == "controller":
        if config.term_scope != "dreamplace_controller":
            raise ValueError(
                "controller prompt dry-run requires term_scope='dreamplace_controller'"
            )
    elif config.objective_mode == "native_residual":
        if config.term_scope != "dreamplace_native_residual":
            raise ValueError(
                "native_residual prompt dry-run requires "
                "term_scope='dreamplace_native_residual'"
            )
    elif config.term_scope != "dreamplace_replacement":
        raise ValueError(
            "openevolve prompt dry-run requires term_scope='dreamplace_replacement'"
        )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    database = ObjectiveProgramDatabase(
        output_root / "program_db",
        ObjectiveDatabaseConfig(
            population_size=config.population_size,
            archive_size=config.archive_size,
            num_islands=config.num_islands,
            map_elites_enabled=config.map_elites_enabled,
            feature_bins=config.feature_bins,
            feature_dimensions=tuple(config.feature_dimensions),
            timing_proxy_wns_delta_min_abs_ns=config.timing_proxy_wns_delta_min_abs_ns,
            migration_interval=config.migration_interval,
            migration_rate=config.migration_rate,
            migration_topology=config.migration_topology,
            migration_clock=config.migration_clock,
        ),
    )

    if from_run_dir:
        database.root = Path(from_run_dir) / "program_db"
        database.load()
        database.root = output_root / "program_db"
        if not database.is_empty:
            _refresh_database_scores(database, config)

    if database.is_empty:
        for index, name in enumerate(config.initial_presets):
            spec = objective_preset(name)
            eligible_parent = (
                name != "dreamplace_controller_native_identity"
                and (
                    config.objective_mode != "native_residual"
                    or set(spec.term_set) <= NATIVE_RESIDUAL_ALLOWED_TERMS
                )
            )
            program = ObjectiveProgram.from_spec(
                spec,
                program_id=f"dry_seed_{index:02d}_{spec.id}",
                code=objective_code_from_spec(spec),
                generation=0,
                iteration_found=0,
                metrics={
                    "combined_score": 1e-6 if eligible_parent else 0.0,
                    "constrained_score": 1e-6 if eligible_parent else 0.0,
                    "seed_baseline": True,
                    "manual_safe_baseline": eligible_parent,
                    "parent_eligible": eligible_parent,
                    "diagnostic_baseline_only": not eligible_parent,
                    "hpwl_gate_passed": True,
                    "overflow_gate_passed": True,
                    "elite_gate_passed": False,
                    "structural_failure_count": 0,
                    "severe_regression_count": 0,
                    "negative_memory_only": False,
                    "hpwl_delta_pct": 0.0,
                    "overflow_delta_pct": 0.0,
                    "feedback_lesson": (
                        (
                            "Synthetic parent context for prompt structure inspection."
                        )
                        if eligible_parent
                        else (
                            "Synthetic diagnostic context excluded from parent selection."
                        )
                    ),
                },
                artifacts={"preset": name, "prompt_dry_run": True},
                status="accepted",
            )
            database.add(program, target_island=index % len(database.islands))

    parent = database.get(parent_preset) if parent_preset else None
    if parent is None and parent_preset:
        matching = [
            program
            for program in database.programs.values()
            if program.id == parent_preset
            or program.artifacts.get("preset") == parent_preset
            or str(program.id).endswith(parent_preset)
        ]
        parent = matching[0] if matching else None
    if parent is None:
        parent = database.sample_parent(
            rng=rng,
            island_id=0,
            parent_policy=config.parent_policy,
        )

    sampler = ObjectivePromptSampler(
        term_scope=config.term_scope,
        num_top_programs=config.num_top_programs,
        num_diverse_programs=config.num_diverse_programs,
        objective_mode=config.objective_mode,
        mutation_mode=config.mutation_mode,
        policy=_prompt_policy_from_config(config, database=database),
        router_background=config.router_background,
    )
    dry_run_feedback = dict(_load_iteration_feedback(config) or {})
    dry_run_feedback.update(
        {
            "prompt_dry_run": True,
            "note": (
                "No provider or DREAMPlace call was made. This artifact is for "
                "prompt inspection and audit only."
            ),
        }
    )
    context = sampler.build_context(
        parent=parent,
        database=database,
        iteration=iteration,
        rng=rng,
        run_feedback=dry_run_feedback,
    )
    prompt_audit = audit_prompt_messages(
        context.get("prompt_messages", []),
        term_scope=config.term_scope,
        objective_mode=config.objective_mode,
    )

    _write_json(context, output_root / "prompt_context.json")
    _write_json(context.get("prompt_messages", []), output_root / "prompt_messages.json")
    _write_json(prompt_audit, output_root / "prompt_audit.json")
    database.save(iteration=0)

    payload = {
        "output_dir": str(output_root),
        "config": str(config_path),
        "from_run_dir": str(from_run_dir) if from_run_dir else None,
        "parent_id": parent.id,
        "message_count": len(context.get("prompt_messages", [])),
        "prompt_audit": prompt_audit,
        "prompt_context": str(output_root / "prompt_context.json"),
        "prompt_messages": str(output_root / "prompt_messages.json"),
        "prompt_audit_path": str(output_root / "prompt_audit.json"),
    }
    _write_json(payload, output_root / "summary.json")
    return payload


def _load_iteration_feedback(
    config: OpenEvolveTier2Config,
    *,
    run_root: str | Path | None = None,
) -> dict[str, Any] | None:
    """Reload external feedback before each LLM prompt is built."""
    if config.prior_feedback:
        return _load_json(config.prior_feedback)
    if run_root is not None:
        run_feedback = Path(run_root) / "routed_feedback.json"
        if run_feedback.exists():
            return _routed_feedback_prompt_view(_load_json(str(run_feedback)))
    return None


ROUTED_FEEDBACK_PROMPT_CANDIDATES = 24
ROUTED_FEEDBACK_PROMPT_DESIGN_ROWS = 24


def _routed_values_text(values: dict[str, Any]) -> str:
    labels = dict(zip(ROUTED_EVIDENCE_KEYS, ("rWL", "rOvf", "WNS", "TNS")))
    parts = []
    for key, label in labels.items():
        value = _finite_number(values.get(key))
        parts.append(f"{label}={value:+.4g}" if value is not None else f"{label}=n/a")
    return " ".join(parts)


def _routed_feedback_prompt_view(payload: Any) -> Any:
    """Condense the cumulative routed record into prompt-sized evidence.

    Every routed candidate from every round stays visible with its aggregate
    post-route evidence; per-design detail is kept for the latest round.
    """

    if not isinstance(payload, dict) or not isinstance(payload.get("rounds"), list):
        return payload
    rounds = sorted(
        (item for item in payload["rounds"] if isinstance(item, dict)),
        key=lambda item: int(item.get("iteration") or 0),
        reverse=True,
    )
    if not rounds:
        return payload
    candidates = [
        f"generation={round_record.get('iteration')} program={item.get('program_id')} "
        + _routed_values_text(item)
        for round_record in rounds
        for item in round_record.get("evidence", [])
        if isinstance(item, dict)
    ]
    per_design = []
    for item in rounds[0].get("evidence", []):
        by_design: dict[str, list[dict[str, Any]]] = {}
        for row in item.get("per_design_routed_evidence") or []:
            if isinstance(row, dict):
                by_design.setdefault(str(row.get("design")), []).append(row)
        for design, rows in sorted(by_design.items()):
            per_design.append(
                f"program={item.get('program_id')} design={design} "
                + _routed_values_text(
                    {
                        "routed_wirelength_delta_pct": _mean_finite(
                            row.get("routed_wirelength_delta_pct") for row in rows
                        ),
                        "routed_overflow_delta_pct": _mean_finite(
                            row.get("grt_overflow_delta_pct") for row in rows
                        ),
                        "post_route_wns_gain_ns": _mean_finite(
                            row.get("wns_gain_ns") for row in rows
                        ),
                        "post_route_tns_gain_ns": _mean_finite(
                            row.get("tns_gain_ns") for row in rows
                        ),
                    }
                )
            )
    return {
        "stage": "scheduled_routed_evaluation",
        "note": (
            "Post-route evidence of every candidate routed so far, newest first. "
            "rWL and rOvf are routed wirelength and routing overflow changes in "
            "percent (negative is better); WNS and TNS are post-route gains in "
            "ns (positive is better). All values are relative to DREAMPlace."
        ),
        "latest_generation": rounds[0].get("iteration"),
        "routed_round_count": len(rounds),
        "routed_candidates": candidates[:ROUTED_FEEDBACK_PROMPT_CANDIDATES],
        "latest_round_per_design": per_design[:ROUTED_FEEDBACK_PROMPT_DESIGN_ROWS],
    }


def _generate_and_evaluate_children(
    *,
    provider: LLMProvider,
    config: OpenEvolveTier2Config,
    context: dict[str, Any],
    parent: ObjectiveProgram,
    iteration: int,
    iteration_dir: Path,
    dreamplace_root: str | Path,
    resume: bool,
    retry_failed: bool,
    database: ObjectiveProgramDatabase,
) -> list[ObjectiveProgram]:
    iteration_dir.mkdir(parents=True, exist_ok=True)
    _write_json(context, iteration_dir / "prompt_context.json")
    programs: list[ObjectiveProgram] = []
    sample_count = max(1, int(config.samples_per_iteration))
    for sample_index in range(sample_count):
        sample_dir = iteration_dir / f"sample_{sample_index:04d}"
        sample_context = copy.deepcopy(context)
        sample_context["sample_index"] = sample_index
        sample_context["samples_per_iteration"] = sample_count
        sample_programs = _generate_and_evaluate_one_child(
            provider=provider,
            config=config,
            context=sample_context,
            parent=parent,
            iteration=iteration,
            sample_index=sample_index,
            iteration_dir=sample_dir,
            dreamplace_root=dreamplace_root,
            resume=resume,
            retry_failed=retry_failed,
            database=database,
        )
        for program in sample_programs:
            program.metrics["sample_index"] = sample_index
            program.artifacts["sample_index"] = sample_index
        programs.extend(sample_programs)
    return programs


def _generate_and_evaluate_one_child(
    *,
    provider: LLMProvider,
    config: OpenEvolveTier2Config,
    context: dict[str, Any],
    parent: ObjectiveProgram,
    iteration: int,
    sample_index: int,
    iteration_dir: Path,
    dreamplace_root: str | Path,
    resume: bool,
    retry_failed: bool,
    database: ObjectiveProgramDatabase,
) -> list[ObjectiveProgram]:
    iteration_dir.mkdir(parents=True, exist_ok=True)
    _write_json(context, iteration_dir / "prompt_context.json")
    rejection_history: list[dict[str, Any]] = []
    rejected_program: ObjectiveProgram | None = None
    prompt_context_path = iteration_dir / "prompt_context.json"
    provider_trace: ProviderTrace | None = None
    spec: ObjectiveSpec | None = None
    scale_rejected_program: ObjectiveProgram | None = None
    mutation_mode = str(context.get("mutation_mode") or config.mutation_mode)
    sample_prompt_id = f"iteration_{iteration:04d}_sample_{sample_index:04d}"

    for attempt in range(1, max(1, config.max_generation_attempts) + 1):
        attempt_dir = iteration_dir / f"attempt_{attempt:02d}"
        attempt_context = copy.deepcopy(context)
        if rejection_history:
            retry_feedback = _retry_feedback_payload(
                attempt=attempt,
                rejection_history=rejection_history,
                config=config,
            )
            attempt_context["retry_feedback"] = retry_feedback
            _append_retry_feedback_message(attempt_context, retry_feedback)
        prompt_context_path = attempt_dir / "prompt_context.json"
        _write_json(attempt_context, prompt_context_path)
        try:
            prompt_audit = audit_prompt_messages(
                attempt_context.get("prompt_messages", []),
                term_scope=config.term_scope,
                objective_mode=config.objective_mode,
            )
            _write_json(prompt_audit, attempt_dir / "prompt_audit.json")
            if not prompt_audit["passed"]:
                raise RuntimeError(
                    "prompt audit failed before provider call: "
                    + ", ".join(item["pattern"] for item in prompt_audit["violations"])
                )
            provider_trace = provider.generate_traced(
                context=attempt_context,
                term_scope=config.term_scope,
                mutation_mode=mutation_mode,
            )
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            _write_json(
                {
                    "provider_error": failure,
                    "iteration_index": iteration,
                    "sample_index": sample_index,
                    "attempt": attempt,
                    "prompt_id": f"{sample_prompt_id}_attempt_{attempt:02d}",
                },
                attempt_dir / "provider_error.json",
            )
            rejection_history.append(
                {
                    "attempt": attempt,
                    "objective_id": None,
                    "term_set": [],
                    "reason": f"provider or parse error before DREAMPlace: {failure}",
                }
            )
            rejected_program = ObjectiveProgram(
                id=f"provider_error_{iteration:04d}_sample_{sample_index:04d}_attempt_{attempt:02d}",
                code="",
                parent_id=parent.id,
                generation=parent.generation + 1,
                iteration_found=iteration,
                metrics={
                    "combined_score": 0.0,
                    "constrained_score": 0.0,
                    "structural_failure_count": 1,
                    "parent_eligible": False,
                    "negative_memory_only": True,
                    "feedback_lesson": (
                        "Provider output could not be parsed or validated before "
                        "DREAMPlace; retry with a valid objective program."
                    ),
                },
                artifacts={
                    "provider_error": failure,
                    "attempt": attempt,
                    "prompt_context_path": str(prompt_context_path),
                    "prompt_id": f"{sample_prompt_id}_attempt_{attempt:02d}",
                },
                failure_reason=failure,
                status="provider_error",
            )
            continue

        _write_json(provider_trace.messages, attempt_dir / "prompt_messages.json")
        _write_json(provider_trace.raw_response or {}, attempt_dir / "raw_response.json")
        _write_json(
            {
                "provider": provider_trace.metadata.get("provider"),
                "resolved_model": provider_trace.metadata.get("resolved_model"),
                "base_url": provider_trace.metadata.get("base_url"),
                "usage": provider_trace.usage or {},
                "iteration_index": iteration,
                "sample_index": sample_index,
                "attempt": attempt,
                "prompt_id": f"{sample_prompt_id}_attempt_{attempt:02d}",
            },
            attempt_dir / "provider_trace.json",
        )
        spec = provider_trace.spec
        failure = _static_rejection_reason(spec, database, config)
        if not failure:
            _write_json(attempt_context, iteration_dir / "prompt_context.json")
            _write_json(provider_trace.messages, iteration_dir / "prompt_messages.json")
            _write_json(provider_trace.raw_response or {}, iteration_dir / "raw_response.json")
            break
        rejected_program = _static_rejected_program(
            spec=spec,
            parent=parent,
            iteration=iteration,
            sample_index=sample_index,
            attempt=attempt,
            failure=failure,
            provider_trace=provider_trace,
            prompt_context_path=prompt_context_path,
            raw_response_path=attempt_dir / "raw_response.json",
            database=database,
        )
        if _is_route_coefficient_scale_rejection(failure):
            _write_json(attempt_context, iteration_dir / "prompt_context.json")
            _write_json(provider_trace.messages, iteration_dir / "prompt_messages.json")
            _write_json(provider_trace.raw_response or {}, iteration_dir / "raw_response.json")
            scale_rejected_program = rejected_program
            break
        rejection_history.append(
            {
                "attempt": attempt,
                "objective_id": spec.id,
                "term_set": spec.term_set,
                "reason": failure,
            }
        )
    else:
        return [rejected_program] if rejected_program is not None else []

    if provider_trace is None or spec is None:
        return []

    if scale_rejected_program is not None:
        calibrated = _evaluate_calibrated_siblings(
            spec=spec,
            raw_program=scale_rejected_program,
            provider_trace=provider_trace,
            config=config,
            parent=parent,
            iteration=iteration,
            iteration_dir=iteration_dir,
            dreamplace_root=dreamplace_root,
            resume=resume,
            retry_failed=retry_failed,
            database=database,
            prompt_context_path=prompt_context_path,
        )
        scale_rejected_program.metrics["calibrated_sibling_count"] = len(calibrated)
        scale_rejected_program.metrics["scale_rejected_but_calibrated"] = True
        return [scale_rejected_program, *calibrated]

    objective_path = write_objective_spec(spec, iteration_dir / "objective.json")
    metrics, artifacts, status, failure_reason = _evaluate_search_panel(
        spec=spec,
        objective_path=objective_path,
        config=config,
        iteration_dir=iteration_dir,
        dreamplace_root=dreamplace_root,
        resume=resume,
        retry_failed=retry_failed,
    )
    raw_program = ObjectiveProgram.from_spec(
        spec,
        program_id=f"iter_{iteration:04d}_sample_{sample_index:04d}_{spec.id}",
        parent_id=parent.id,
        generation=parent.generation + 1,
        iteration_found=iteration,
        metrics=metrics,
        artifacts={
            **artifacts,
            "provider_metadata": provider_trace.metadata,
            "provider_usage": provider_trace.usage or {},
            "prompt_id": f"iteration_{iteration:04d}_sample_{sample_index:04d}",
            "prompt_context_path": str(prompt_context_path),
            "raw_response_path": str(iteration_dir / "raw_response.json"),
        },
        prompt={"context_path": str(prompt_context_path), "messages": provider_trace.messages},
        raw_response=provider_trace.raw_response,
        failure_reason=failure_reason,
        status=status,
    )
    calibrated = _evaluate_calibrated_siblings(
        spec=spec,
        raw_program=raw_program,
        provider_trace=provider_trace,
        config=config,
        parent=parent,
        iteration=iteration,
        iteration_dir=iteration_dir,
        dreamplace_root=dreamplace_root,
        resume=resume,
        retry_failed=retry_failed,
        database=database,
        prompt_context_path=prompt_context_path,
    )
    raw_program.metrics["calibrated_sibling_count"] = len(calibrated)
    return [raw_program, *calibrated]


def _timing_proxy_feedback_for_tier2_summary(
    *,
    tier2_summary: dict[str, Any],
    config: OpenEvolveTier2Config,
    iteration_dir: Path,
    dreamplace_root: str | Path,
    objective_ids: set[str],
    resume: bool,
    retry_failed: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not config.timing_proxy_enabled:
        return {}, {}
    if not objective_ids:
        return {}, {
            "timing_proxy": {
                "status": "not_run",
                "reason": "No candidate was selected for Tier B after Tier A evaluation.",
            }
        }
    if not config.timing_proxy_panel:
        if config.timing_proxy_audit_required:
            raise RuntimeError(
                "timing_proxy_audit_required is true, but no timing_proxy_panel was configured."
            )
        return {}, {
            "timing_proxy": {
                "status": "not_run",
                "reason": "timing_proxy_enabled is true but no timing_proxy_panel was configured.",
            }
        }
    comparison_csv = tier2_summary.get("comparison_csv")
    if not comparison_csv or not Path(str(comparison_csv)).is_file():
        if config.timing_proxy_audit_required:
            raise RuntimeError(
                "timing_proxy_audit_required is true, but the Tier-2 comparison CSV is missing."
            )
        return {}, {
            "timing_proxy": {
                "status": "not_run",
                "reason": "Tier-2 comparison CSV is missing, so placement DEFs cannot be collected.",
            }
        }
    timing_dir = iteration_dir / "timing_proxy"
    try:
        timing_panel = load_timing_proxy_panel(config.timing_proxy_panel)
        timing_design_names = {design.name for design in timing_panel.designs}
        placements_path = placements_from_tier2_comparison(
            comparison_csv,
            timing_dir / "placements.json",
            objective_ids=set(objective_ids),
            design_names=timing_design_names,
            include_default=True,
            include_custom_default=True,
            minimum_source_iterations=config.minimum_scoring_iterations,
        )
        placement_count = _placement_count(placements_path)
        if placement_count == 0:
            if config.timing_proxy_audit_required:
                raise RuntimeError(
                    "timing_proxy_audit_required is true, but no successful Tier-2 DEF "
                    "placements matched the timing proxy panel."
                )
            return {}, {
                "timing_proxy": {
                    "status": "not_run",
                    "reason": "No successful Tier-2 DEF placements were available for timing proxy.",
                    "placements": str(placements_path),
                }
            }
        summary = run_timing_proxy_eval(
            panel_path=config.timing_proxy_panel,
            placements_path=placements_path,
            dreamplace_root=dreamplace_root,
            run_dir=timing_dir / "eval",
            resume=resume,
            retry_failed=retry_failed,
        )
        if config.timing_proxy_audit_required and int(summary.get("success_count", 0)) == 0:
            raise RuntimeError(
                "timing_proxy_audit_required is true, but the timing proxy produced "
                "zero successful WNS/TNS measurements. Check timing collateral before "
                "running OpenTimer-feedback evolution."
            )
        metrics = timing_proxy_metrics_by_objective(
            summary,
            baseline_objective_id="default",
        )
        return metrics, {
            "timing_proxy": {
                "status": "completed",
                "mode": config.timing_proxy_mode,
                "panel": config.timing_proxy_panel,
                "placements": str(placements_path),
                "summary": summary,
                "metrics_by_objective": metrics,
            }
        }
    except Exception as exc:
        if config.timing_proxy_audit_required:
            raise
        return {}, {
            "timing_proxy": {
                "status": "failed",
                "mode": config.timing_proxy_mode,
                "panel": config.timing_proxy_panel,
                "error": f"{type(exc).__name__}: {exc}",
            }
        }


def _tier_b_selected(metrics: dict[str, Any], config: OpenEvolveTier2Config) -> bool:
    """Whether a validated candidate receives the Tier B timing proxy.

    Tier B is reserved for candidates whose Tier A placement is usable
    evidence: it completed the iteration budget without a structural failure
    and produced finite HPWL and overflow.
    """

    return bool(
        config.timing_proxy_enabled
        and not metrics.get("structural_failure_count")
        and metrics.get("iteration_budget_satisfied")
        and _finite_number(metrics.get("hpwl_delta_pct")) is not None
        and _finite_number(metrics.get("overflow_delta_pct")) is not None
    )


def _record_tier_b_evidence(metrics: dict[str, Any], *, selected: bool) -> None:
    metrics["tier_b_selected"] = selected
    metrics["tier_b_evaluated"] = str(metrics.get("timing_proxy_status")) in {
        "success",
        "partial",
    }


def _placement_count(path: str | Path) -> int:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except Exception:
        return 0
    placements = payload.get("placements", []) if isinstance(payload, dict) else payload
    return len(placements) if isinstance(placements, list) else 0


def _evaluate_search_panel(
    *,
    spec: ObjectiveSpec,
    objective_path: Path,
    config: OpenEvolveTier2Config,
    iteration_dir: Path,
    dreamplace_root: str | Path,
    resume: bool,
    retry_failed: bool,
) -> tuple[dict[str, Any], dict[str, Any], str, str | None]:
    baseline_paths = _write_baseline_presets(config.search_baseline_presets, iteration_dir / "baselines")
    panel_path = write_dreamplace_panel(
        load_shared_panel(config.search_panel),
        iteration_dir / "search_panel.toml",
    )
    try:
        tier2_summary = run_tier2_dreamplace(
            panel_path=panel_path,
            objective_paths=baseline_paths + [objective_path],
            dreamplace_root=dreamplace_root,
            run_dir=iteration_dir / "tier2_search",
            store_path=iteration_dir / "tier2_search" / "tier2.sqlite",
            resume=resume,
            retry_failed=retry_failed,
            include_default=True,
            include_custom_default=True,
            require_output_artifact=True,
            require_def_output=True,
            disable_legalization=config.search_disable_legalization,
            objective_semantics=(
                "raw" if config.objective_mode == "controller" else None
            ),
            timing_controller_panel=config.timing_controller_panel,
        )
    except Exception as exc:
        return (
            {
                "combined_score": 0.0,
                "constrained_score": 0.0,
                "structural_failure_count": 1,
                "hpwl_gate_passed": False,
                "overflow_gate_passed": False,
                "elite_gate_passed": False,
                "parent_eligible": False,
                "negative_memory_only": True,
                "primary_baseline": config.primary_baseline,
                "feedback_lesson": "DREAMPlace search-panel evaluation raised an exception.",
            },
            {"tier2_exception": f"{type(exc).__name__}: {exc}"},
            "failed",
            f"tier2_exception: {type(exc).__name__}: {exc}",
        )
    rows = add_outcome_labels(
        _rows_with_primary_baseline(
            _read_csv(Path(tier2_summary["comparison_csv"])),
            primary_baseline=config.primary_baseline,
        ),
        severe_threshold_pct=config.severe_regression_pct,
    )
    rankings = aggregate_rank_scores(rows, severe_threshold_pct=config.severe_regression_pct)
    ranking = next((item for item in rankings if item.objective_id == spec.id), None)
    candidate_rows = [row for row in rows if row.get("objective_id") == spec.id]
    metrics = _metrics_from_rows(candidate_rows, ranking, config)
    baseline_ids = _runtime_baseline_objective_ids(
        tier2_summary=tier2_summary,
        baseline_paths=baseline_paths,
        baseline_preset_names=config.search_baseline_presets,
    )
    metrics.update(
        _baseline_portfolio_summary(
            rows=rows,
            rankings=rankings,
            candidate_objective_id=spec.id,
            baseline_objective_ids=baseline_ids,
            min_effect_pct=config.min_baseline_portfolio_effect_pct,
            min_behavior_distance_pct=config.min_baseline_behavior_distance_pct,
        )
    )
    metrics["mechanism_signature"] = _mechanism_signature(_spec_mechanism_view(spec))
    metrics["mechanism_terms"] = sorted(_terms_in_ast(spec.ast))
    metrics.update(
        _baseline_mechanism_summary_from_presets(_spec_mechanism_view(spec), config)
    )
    tier_b_selected = _tier_b_selected(metrics, config)
    timing_metrics, timing_artifacts = _timing_proxy_feedback_for_tier2_summary(
        tier2_summary=tier2_summary,
        config=config,
        iteration_dir=iteration_dir,
        dreamplace_root=dreamplace_root,
        objective_ids={spec.id} if tier_b_selected else set(),
        resume=resume,
        retry_failed=retry_failed,
    )
    metrics.update(timing_metrics.get(spec.id, {}))
    _record_tier_b_evidence(metrics, selected=tier_b_selected)
    _apply_timing_controller_admission(metrics, spec, config)
    _apply_final_def_selection_metrics(metrics, config)
    _apply_routing_gate(metrics, spec, config, allow_nonrouting_parent=False)
    _apply_measured_routing_influence_gate(
        metrics, config, allow_seed_baseline=False
    )
    _apply_baseline_portfolio_gate(metrics, config)
    _apply_native_default_selection_policy(metrics, config, allow_seed_baseline=False)
    _apply_near_miss_parent_policy(metrics, config, allow_seed_baseline=False)
    apply_timing_proxy_policy(
        metrics,
        requested_mode=config.timing_proxy_mode,
        wns_regression_gate_ns=config.timing_proxy_wns_regression_gate_ns,
        tns_regression_pct_gate=config.timing_proxy_tns_regression_pct_gate,
        tns_regression_min_abs_ns=config.timing_proxy_tns_regression_min_abs_ns,
        wns_delta_min_abs_ns=config.timing_proxy_wns_delta_min_abs_ns,
        selection_weight=config.timing_proxy_selection_weight,
    )
    _apply_manuscript_multimetric_selection(
        metrics,
        config,
        allow_seed_baseline=False,
    )
    artifacts = {
        "tier2_summary": tier2_summary,
        "candidate_rows": candidate_rows[:10],
        "rankings": [item.to_dict() for item in rankings[:10]],
        "baseline_portfolio_ids": sorted(baseline_ids),
        **timing_artifacts,
    }
    status = "accepted" if not metrics.get("structural_failure_count") else "failed"
    failure_reason = None if status == "accepted" else "DREAMPlace structural failure"
    return metrics, artifacts, status, failure_reason


def _evaluate_search_panel_many(
    *,
    specs: list[ObjectiveSpec],
    objective_paths: list[Path],
    config: OpenEvolveTier2Config,
    iteration_dir: Path,
    dreamplace_root: str | Path,
    resume: bool,
    retry_failed: bool,
) -> dict[str, tuple[dict[str, Any], dict[str, Any], str, str | None]]:
    if not specs:
        return {}
    baseline_paths = _write_baseline_presets(config.search_baseline_presets, iteration_dir / "baselines")
    panel_path = write_dreamplace_panel(
        load_shared_panel(config.search_panel),
        iteration_dir / "search_panel.toml",
    )
    try:
        tier2_summary = run_tier2_dreamplace(
            panel_path=panel_path,
            objective_paths=baseline_paths + objective_paths,
            dreamplace_root=dreamplace_root,
            run_dir=iteration_dir / "tier2_search",
            store_path=iteration_dir / "tier2_search" / "tier2.sqlite",
            resume=resume,
            retry_failed=retry_failed,
            include_default=True,
            include_custom_default=True,
            require_output_artifact=True,
            require_def_output=True,
            disable_legalization=config.search_disable_legalization,
            objective_semantics=(
                "raw" if config.objective_mode == "controller" else None
            ),
            timing_controller_panel=config.timing_controller_panel,
        )
    except Exception as exc:
        failure_metrics = {
            "combined_score": 0.0,
            "constrained_score": 0.0,
            "structural_failure_count": 1,
            "hpwl_gate_passed": False,
            "overflow_gate_passed": False,
            "elite_gate_passed": False,
            "parent_eligible": False,
            "negative_memory_only": True,
            "primary_baseline": config.primary_baseline,
            "feedback_lesson": "Calibration DREAMPlace batch raised an exception.",
        }
        return {
            spec.id: (
                dict(failure_metrics),
                {"tier2_exception": f"{type(exc).__name__}: {exc}"},
                "failed",
                f"tier2_exception: {type(exc).__name__}: {exc}",
            )
            for spec in specs
        }

    rows = add_outcome_labels(
        _rows_with_primary_baseline(
            _read_csv(Path(tier2_summary["comparison_csv"])),
            primary_baseline=config.primary_baseline,
        ),
        severe_threshold_pct=config.severe_regression_pct,
    )
    rankings = aggregate_rank_scores(rows, severe_threshold_pct=config.severe_regression_pct)
    ranking_by_id = {item.objective_id: item for item in rankings}
    baseline_ids = _runtime_baseline_objective_ids(
        tier2_summary=tier2_summary,
        baseline_paths=baseline_paths,
        baseline_preset_names=config.search_baseline_presets,
    )
    tier_a_metrics = {
        spec.id: _metrics_from_rows(
            [row for row in rows if row.get("objective_id") == spec.id],
            ranking_by_id.get(spec.id),
            config,
        )
        for spec in specs
    }
    tier_b_ids = {
        spec_id
        for spec_id, spec_metrics in tier_a_metrics.items()
        if _tier_b_selected(spec_metrics, config)
    }
    timing_metrics, timing_artifacts = _timing_proxy_feedback_for_tier2_summary(
        tier2_summary=tier2_summary,
        config=config,
        iteration_dir=iteration_dir,
        dreamplace_root=dreamplace_root,
        objective_ids=tier_b_ids,
        resume=resume,
        retry_failed=retry_failed,
    )
    results: dict[str, tuple[dict[str, Any], dict[str, Any], str, str | None]] = {}
    for spec in specs:
        candidate_rows = [row for row in rows if row.get("objective_id") == spec.id]
        metrics = tier_a_metrics[spec.id]
        metrics.update(
                _baseline_portfolio_summary(
                    rows=rows,
                    rankings=rankings,
                    candidate_objective_id=spec.id,
                    baseline_objective_ids=baseline_ids,
                    min_effect_pct=config.min_baseline_portfolio_effect_pct,
                    min_behavior_distance_pct=config.min_baseline_behavior_distance_pct,
                )
            )
        metrics["mechanism_signature"] = _mechanism_signature(_spec_mechanism_view(spec))
        metrics["mechanism_terms"] = sorted(_terms_in_ast(spec.ast))
        metrics.update(
        _baseline_mechanism_summary_from_presets(_spec_mechanism_view(spec), config)
    )
        metrics.update(timing_metrics.get(spec.id, {}))
        _record_tier_b_evidence(metrics, selected=spec.id in tier_b_ids)
        _apply_timing_controller_admission(metrics, spec, config)
        _apply_final_def_selection_metrics(metrics, config)
        _apply_routing_gate(metrics, spec, config, allow_nonrouting_parent=False)
        _apply_measured_routing_influence_gate(
            metrics, config, allow_seed_baseline=False
        )
        _apply_baseline_portfolio_gate(metrics, config)
        _apply_native_default_selection_policy(metrics, config, allow_seed_baseline=False)
        _apply_near_miss_parent_policy(metrics, config, allow_seed_baseline=False)
        apply_timing_proxy_policy(
            metrics,
            requested_mode=config.timing_proxy_mode,
            wns_regression_gate_ns=config.timing_proxy_wns_regression_gate_ns,
            tns_regression_pct_gate=config.timing_proxy_tns_regression_pct_gate,
            tns_regression_min_abs_ns=config.timing_proxy_tns_regression_min_abs_ns,
            wns_delta_min_abs_ns=config.timing_proxy_wns_delta_min_abs_ns,
            selection_weight=config.timing_proxy_selection_weight,
        )
        _apply_manuscript_multimetric_selection(
            metrics,
            config,
            allow_seed_baseline=False,
        )
        artifacts = {
            "tier2_summary": tier2_summary,
            "candidate_rows": candidate_rows[:10],
            "rankings": [item.to_dict() for item in rankings[:10]],
            "calibration_batch": True,
            "baseline_portfolio_ids": sorted(baseline_ids),
            **timing_artifacts,
        }
        status = "accepted" if not metrics.get("structural_failure_count") else "failed"
        failure_reason = None if status == "accepted" else "DREAMPlace structural failure"
        results[spec.id] = (metrics, artifacts, status, failure_reason)
    return results


def _evaluate_calibrated_siblings(
    *,
    spec: ObjectiveSpec,
    raw_program: ObjectiveProgram,
    provider_trace: ProviderTrace,
    config: OpenEvolveTier2Config,
    parent: ObjectiveProgram,
    iteration: int,
    iteration_dir: Path,
    dreamplace_root: str | Path,
    resume: bool,
    retry_failed: bool,
    database: ObjectiveProgramDatabase,
    prompt_context_path: Path,
) -> list[ObjectiveProgram]:
    if config.objective_mode not in {"replacement", "residual", "native_residual"}:
        return []
    coeffs, unsupported = _route_coefficients(spec.ast)
    spec_terms = set(spec.term_set)
    has_density = bool(DENSITY_FAMILY_TERMS & spec_terms)
    has_wirelength = bool(WIRELENGTH_FAMILY_TERMS & spec_terms)
    if unsupported or (not coeffs and not has_density and not has_wirelength):
        return []

    sibling_records: list[
        tuple[ObjectiveSpec, float | None, float | None, float | None, str, Path]
    ] = []
    seen_ids, seen_ast_signatures = _database_objective_identity(database)
    seen_ids.add(spec.id)
    seen_ast_signatures.add(_ast_signature(spec.ast))
    combinations = _calibration_combinations(
        route_coeffs=sorted(set(config.route_coeff_grid)) if coeffs else [None],
        density_coeffs=sorted(set(config.density_coeff_grid)) if has_density else [None],
        wirelength_coeffs=(
            sorted(set(config.wirelength_coeff_grid)) if has_wirelength else [None]
        ),
        max_count=config.max_calibrated_siblings,
    )
    for route_coeff, density_coeff, wirelength_coeff in combinations:
        if route_coeff is not None and abs(route_coeff) > config.route_coeff_cap:
            continue
        try:
            sibling_spec = _calibrated_spec(
                spec,
                route_coeff=route_coeff,
                density_coeff=density_coeff,
                wirelength_coeff=wirelength_coeff,
            )
        except ObjectiveSpecError:
            continue
        sibling_ast_signature = _ast_signature(sibling_spec.ast)
        if sibling_spec.id in seen_ids or sibling_ast_signature in seen_ast_signatures:
            continue
        seen_ids.add(sibling_spec.id)
        seen_ast_signatures.add(sibling_ast_signature)
        suffix = _safe_name(
            f"route_{route_coeff if route_coeff is not None else 'keep'}_"
            f"density_{density_coeff if density_coeff is not None else 'keep'}_"
            f"wirelength_{wirelength_coeff if wirelength_coeff is not None else 'keep'}"
        )
        objective_path = write_objective_spec(
            sibling_spec,
            iteration_dir / "calibration" / "objectives" / f"{suffix}.json",
        )
        sibling_records.append(
            (sibling_spec, route_coeff, density_coeff, wirelength_coeff, suffix, objective_path)
        )

    if not sibling_records:
        return []

    batch_metrics = _evaluate_search_panel_many(
        specs=[record[0] for record in sibling_records],
        objective_paths=[record[5] for record in sibling_records],
        config=config,
        iteration_dir=iteration_dir / "calibration" / "batch",
        dreamplace_root=dreamplace_root,
        resume=resume,
        retry_failed=retry_failed,
    )

    calibrated: list[ObjectiveProgram] = []
    for (
        sibling_spec,
        route_coeff,
        density_coeff,
        wirelength_coeff,
        suffix,
        objective_path,
    ) in sibling_records:
        metrics, artifacts, status, failure_reason = batch_metrics.get(
            sibling_spec.id,
            (
                {
                    "combined_score": 0.0,
                    "constrained_score": 0.0,
                    "structural_failure_count": 1,
                    "hpwl_gate_passed": False,
                    "overflow_gate_passed": False,
                    "elite_gate_passed": False,
                    "parent_eligible": False,
                    "negative_memory_only": True,
                    "feedback_lesson": "Calibration batch did not produce metrics for this sibling.",
                },
                {"objective_path": str(objective_path)},
                "failed",
                "missing calibration batch metrics",
            ),
        )
        calibration_report = {
            "parent_program_id": raw_program.id,
            "parent_objective_id": spec.id,
            "route_coefficient": route_coeff,
            "density_coefficient": density_coeff,
            "wirelength_coefficient": wirelength_coeff,
            "raw_hpwl_delta_pct": raw_program.metrics.get("hpwl_delta_pct"),
            "raw_overflow_delta_pct": raw_program.metrics.get("overflow_delta_pct"),
            "calibrated_hpwl_delta_pct": metrics.get("hpwl_delta_pct"),
            "calibrated_overflow_delta_pct": metrics.get("overflow_delta_pct"),
            "hpwl_gate_passed": metrics.get("hpwl_gate_passed"),
        }
        metrics["calibration_report"] = calibration_report
        metrics["operator"] = "param_tune"
        artifacts["calibration_report"] = calibration_report
        artifacts["objective_path"] = str(objective_path)
        artifacts["provider_metadata"] = provider_trace.metadata
        artifacts["provider_usage"] = provider_trace.usage or {}
        artifacts["prompt_id"] = f"iteration_{iteration:04d}_calibration_{suffix}"
        artifacts["prompt_context_path"] = str(prompt_context_path)
        artifacts["raw_response_path"] = str(iteration_dir / "raw_response.json")
        calibrated.append(
            ObjectiveProgram.from_spec(
                sibling_spec,
                program_id=f"iter_{iteration:04d}_{sibling_spec.id}_calib_{suffix}",
                parent_id=parent.id,
                generation=parent.generation + 1,
                iteration_found=iteration,
                metrics=metrics,
                artifacts=artifacts,
                prompt={"context_path": str(prompt_context_path), "messages": provider_trace.messages},
                raw_response=provider_trace.raw_response,
                failure_reason=failure_reason,
                status=status,
            )
        )
    return calibrated


def _retry_feedback_payload(
    *,
    attempt: int,
    rejection_history: list[dict[str, Any]],
    config: OpenEvolveTier2Config,
) -> dict[str, Any]:
    repair_instructions = _retry_repair_instructions(rejection_history, config)
    return {
        "attempt": attempt,
        "previous_rejection_count": len(rejection_history),
        "previous_rejections": _sanitized_rejection_history(rejection_history),
        "instruction": (
            "The previous objective was rejected before DREAMPlace. Return a "
            "different complete objective_program that satisfies the restricted "
            "objective contract and addresses the repair instructions below."
        ),
        "repair_instructions": repair_instructions,
        "api_memory": (
            "The API is stateless; this retry message is the only additional "
            "memory for the failed attempts in this request."
        ),
    }


def _static_rejected_program(
    *,
    spec: ObjectiveSpec,
    parent: ObjectiveProgram,
    iteration: int,
    sample_index: int = 0,
    attempt: int,
    failure: str,
    provider_trace: ProviderTrace,
    prompt_context_path: Path,
    raw_response_path: Path,
    database: ObjectiveProgramDatabase,
) -> ObjectiveProgram:
    return ObjectiveProgram.from_spec(
        spec,
        program_id=(
            f"iter_{iteration:04d}_sample_{sample_index:04d}_"
            f"attempt_{attempt:02d}_{spec.id}"
        ),
        parent_id=parent.id,
        generation=parent.generation + 1,
        iteration_found=iteration,
        metrics={
            "combined_score": 0.0,
            "structural_failure_count": 1,
            "parent_eligible": False,
            "negative_memory_only": True,
            # Unmeasured: never ran DREAMPlace. Signature-equality dedup still
            # applies, but distance-based negative rejection must not consult
            # these records (they otherwise snowball into an exclusion ball).
            "static_rejection": True,
            "mechanism_signature": _mechanism_signature(_spec_mechanism_view(spec)),
            "mechanism_terms": sorted(_terms_in_ast(spec.ast)),
            **(_closest_database_baseline_mechanism(spec.ast, database) or {}),
            "feedback_lesson": f"Static rejection before DREAMPlace: {failure}",
        },
        artifacts={
            "static_rejection": failure,
            "attempt": attempt,
            "sample_index": sample_index,
            "provider_metadata": provider_trace.metadata,
            "provider_usage": provider_trace.usage or {},
            "prompt_id": (
                f"iteration_{iteration:04d}_sample_{sample_index:04d}_"
                f"attempt_{attempt:02d}"
            ),
            "prompt_context_path": str(prompt_context_path),
            "raw_response_path": str(raw_response_path),
        },
        prompt={"context_path": str(prompt_context_path), "messages": provider_trace.messages},
        raw_response=provider_trace.raw_response,
        failure_reason=failure,
        status="rejected",
    )


def _is_route_coefficient_scale_rejection(reason: str | None) -> bool:
    return "route coefficient exceeds cap" in str(reason or "")


def _retry_repair_instructions(
    rejection_history: list[dict[str, Any]],
    config: OpenEvolveTier2Config,
) -> list[str]:
    instructions: list[str] = []
    reasons = [str(item.get("reason", "")) for item in rejection_history]
    joined = "\n".join(reasons)
    if "too close to the explicit wirelength+density baseline" in joined:
        instructions.append(
            "Use a materially different objective mechanism beyond explicit "
            "wirelength/density equivalents. Prefer a physical route, pin-access, "
            "alternative smoothing, or nonlinear interaction mechanism rather than "
            "adding a no-op correction."
        )
    if "repeats a mechanism structure" in joined:
        instructions.append(
            "Change the mechanism structure, not only constants. Replace at least "
            "one observable family or nonlinear transform from the repeated "
            "negative-memory candidate."
        )
    if "too close to a measured negative" in joined or "near-miss mechanism" in joined:
        instructions.append(
            "The objective is too close to a measured negative or near-miss "
            "mechanism. Change the observable family or nonlinear composition; "
            "do not retry another coefficient-only or algebraically similar "
            "variant of the same mechanism family."
        )
    if "coefficient exceeds cap" in joined:
        instructions.append(
            "Use a gentler routing-term scale and rely on evaluator-side "
            "calibration to test magnitude. The next objective should change "
            "the mechanism or scale without targeting a hidden numeric boundary."
        )
    if "canonical wirelength anchor" in joined:
        instructions.append(
            "Repair the objective contract by including one wirelength-family "
            "anchor such as wirelength_wawl or wirelength_lse. Keep the routing "
            "mechanism as an additional component rather than replacing the "
            "placement anchor entirely."
        )
    if "density/utilization anchor" in joined:
        instructions.append(
            "Repair the objective contract by including one density/utilization "
            "anchor such as density_electric or density_bell. This is required "
            "for DREAMPlace optimizer stability; the routing-aware mechanism "
            "should still be materially nontrivial and separate from that anchor."
        )
    if "AST primitive count exceeds" in joined or "AST depth" in joined:
        instructions.append(
            "Simplify the formula to fit the restricted DSL: keep the final "
            "score as a shallow sum of two to four components; each component "
            "should be a term, const*term, or const*helper(term or term*term). "
            "Avoid nested helper calls and chained products of transformed "
            "subexpressions."
        )
    if "duplicate objective AST" in joined:
        instructions.append("Return a structurally different objective AST.")
    if not instructions:
        instructions.append(
            "Return a different objective program that satisfies the schema, "
            "uses supported DREAMPlace terms, and is not a duplicate."
        )
    return instructions


def _sanitized_rejection_history(
    rejection_history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sanitized = []
    for item in rejection_history:
        reason = str(item.get("reason", ""))
        sanitized.append(
            {
                "attempt": item.get("attempt"),
                "objective_id": item.get("objective_id"),
                "term_set": item.get("term_set"),
                "reason_category": _rejection_reason_category(reason),
            }
        )
    return sanitized


def _rejection_reason_category(reason: str) -> str:
    if "too close to the explicit wirelength+density baseline" in reason:
        return "baseline_like_objective"
    if "coefficient exceeds cap" in reason:
        return "coefficient_scale_out_of_bounds"
    if "repeats a mechanism structure" in reason:
        return "repeated_mechanism"
    if "too close to a measured negative" in reason or "near-miss mechanism" in reason:
        return "near_negative_mechanism"
    if "duplicate objective AST" in reason:
        return "duplicate_ast"
    if "unsupported" in reason and "terms" in reason:
        return "unsupported_terms"
    if "AST primitive count exceeds" in reason or "AST depth" in reason:
        return "complexity_limit"
    if "missing required base terms" in reason:
        return "missing_required_terms"
    return "static_validation_rejection"


def _append_retry_feedback_message(
    context: dict[str, Any],
    retry_feedback: dict[str, Any],
) -> None:
    messages = context.get("prompt_messages")
    if not isinstance(messages, list):
        return
    messages.append(
        {
            "role": "user",
            "content": "\n".join(
                [
                    "# Static Validation Feedback For Retry",
                    "",
                    "The last objective was rejected before DREAMPlace execution.",
                    "Use this local feedback to repair the next objective. Do not rely on API memory.",
                    "",
                    "```json",
                    json.dumps(retry_feedback, indent=2, sort_keys=True),
                    "```",
                ]
            ),
        }
    )


PROMPT_AUDIT_FORBIDDEN_PATTERNS = (
    "hpwl_gate_search_pct",
    "hpwl_gate_elite_pct",
    "per_design_hpwl_gate_search_pct",
    "per_design_overflow_gate_search_pct",
    "custom_default_hpwl_gate_pct",
    "custom_default_overflow_gate_pct",
    "min_custom_default_effect_pct",
    "require_baseline_portfolio_gate",
    "baseline_portfolio_gate_passed",
    "min_baseline_portfolio_effect_pct",
    "baseline_portfolio_effect_gate_passed",
    "require_baseline_portfolio_pareto",
    "min_baseline_behavior_distance_pct",
    "baseline_portfolio_behavior_distance_gate_passed",
    "baseline_portfolio_min_behavior_distance_pct",
    "allow_near_miss_parents",
    "near_miss_max_worst_hpwl_pct",
    "near_miss_max_worst_overflow_pct",
    "near_miss_parent_score_floor",
    "reject_baseline_mechanism_clones",
    "min_baseline_mechanism_distance",
    "reject_repeated_negative_mechanisms",
    "min_replacement_nonbaseline_terms",
    "min_abs_routing_coeff",
    "route_coeff_cap",
    "route_coeff_grid",
    "density_coeff_grid",
    "wirelength_coeff_grid",
    "require_nonzero_routing_term",
    "parent_policy",
    "hpwl_safe_only",
    "hpwl_gate_passed",
    "overflow_gate_passed",
    "custom_default_gate_passed",
    "custom_default_hpwl_gate_passed",
    "custom_default_overflow_gate_passed",
    "elite_gate_passed",
    "parent_eligible",
    "negative_memory_only",
    "HPWL must not regress",
    "must preserve HPWL",
    "preserve HPWL",
    "do not regress HPWL",
    "Use at least 2 non-default",
    "use at least 2 non-baseline",
    "coefficient exceeds cap",
    "abs(c)",
)


def audit_prompt_messages(
    messages: Any,
    *,
    term_scope: str = "dreamplace_replacement",
    objective_mode: str = "replacement",
) -> dict[str, Any]:
    """Check API-bound prompt messages for hidden policy leakage.

    This audit is about the prompt, not the persisted run config. The evaluator
    can use hard gates internally, but the LLM should see an OpenEvolve/Eureka
    style environment plus empirical feedback rather than exact thresholds or
    native-objective shortcuts.
    """

    if not isinstance(messages, list):
        messages = []
    text_parts = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        text_parts.append(str(message.get("content", "")))
    prompt_text = "\n".join(text_parts)
    prompt_lower = prompt_text.lower()
    violations = []
    for pattern in PROMPT_AUDIT_FORBIDDEN_PATTERNS:
        if pattern.lower() in prompt_lower:
            violations.append(
                {
                    "pattern": pattern,
                    "category": _prompt_audit_category(pattern),
                }
            )
    if term_scope == "dreamplace_replacement" and objective_mode == "replacement":
        if "native_objective" in prompt_text:
            violations.append(
                {
                    "pattern": "native_objective",
                    "category": "forbidden_replacement_term",
                }
            )
    return {
        "passed": not violations,
        "message_count": len(messages),
        "term_scope": term_scope,
        "objective_mode": objective_mode,
        "violations": violations,
    }


def audit_openevolve_prompt_artifacts(
    run_dir: str | Path,
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Audit saved OpenEvolve Tier-2 prompt artifacts in a run directory."""

    root = Path(run_dir)
    prompt_paths = sorted(root.rglob("prompt_messages.json"))
    results = []
    failed = 0
    for prompt_path in prompt_paths:
        messages = json.loads(prompt_path.read_text(encoding="utf-8"))
        context = _load_prompt_context_for_messages(prompt_path)
        audit = audit_prompt_messages(
            messages,
            term_scope=str(context.get("term_scope", "dreamplace_replacement")),
            objective_mode=str(context.get("objective_mode", "replacement")),
        )
        relative_path = str(prompt_path.relative_to(root)) if prompt_path.is_relative_to(root) else str(prompt_path)
        record = {
            "prompt_messages": relative_path,
            "prompt_context": context.get("prompt_context_path"),
            **audit,
        }
        if not audit["passed"]:
            failed += 1
        results.append(record)
        if write:
            _write_json(record, prompt_path.with_name("prompt_audit.json"))
    summary = {
        "run_dir": str(root),
        "prompt_file_count": len(prompt_paths),
        "failed_prompt_count": failed,
        "passed": failed == 0,
        "results": results,
    }
    if write:
        _write_json(summary, root / "prompt_audit_summary.json")
    return summary


def _load_prompt_context_for_messages(prompt_path: Path) -> dict[str, Any]:
    context_path = prompt_path.with_name("prompt_context.json")
    context: dict[str, Any] = {}
    if context_path.is_file():
        try:
            loaded = json.loads(context_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                context = loaded
        except json.JSONDecodeError:
            context = {}
    context["prompt_context_path"] = str(context_path) if context_path.is_file() else None
    return context


def _prompt_audit_category(pattern: str) -> str:
    lowered = pattern.lower()
    if "gate" in lowered or "policy" in lowered or lowered == "hpwl_safe_only":
        return "hidden_selection_policy"
    if "coeff" in lowered or "cap" in lowered or "abs(c)" in lowered:
        return "hidden_coefficient_policy"
    if "hpwl" in lowered and ("must" in lowered or "preserve" in lowered or "regress" in lowered):
        return "over_leading_instruction"
    if "baseline" in lowered or "clone" in lowered:
        return "hidden_baseline_guard"
    return "hidden_evaluator_detail"


def _metrics_from_rows(
    rows: list[dict[str, Any]],
    ranking: Any,
    config: OpenEvolveTier2Config,
) -> dict[str, Any]:
    if ranking is None:
        return {
            "combined_score": 0.0,
            "constrained_score": 0.0,
            "structural_failure_count": 1,
            "severe_regression_count": 0,
            "hpwl_gate_passed": False,
            "overflow_gate_passed": False,
            "elite_gate_passed": False,
            "parent_eligible": False,
            "negative_memory_only": True,
            "primary_baseline": config.primary_baseline,
            "native_default_comparison_label": "native_default_unknown",
            "beats_native_default_pareto": False,
            "native_default_tradeoff": False,
            "native_default_regression": False,
            "feedback_lesson": "No DREAMPlace comparison rows were produced for this objective.",
        }
    hpwl = _mean_finite(row.get("hpwl_delta_pct") for row in rows)
    overflow = _mean_finite(row.get("overflow_delta_pct") for row in rows)
    native_default_hpwl = _mean_finite(row.get("native_default_hpwl_delta_pct") for row in rows)
    native_default_overflow = _mean_finite(
        row.get("native_default_overflow_delta_pct") for row in rows
    )
    custom_default_hpwl = _mean_finite(row.get("custom_default_hpwl_delta_pct") for row in rows)
    custom_default_overflow = _mean_finite(
        row.get("custom_default_overflow_delta_pct") for row in rows
    )
    native_default_label = _native_default_comparison_label(
        native_default_hpwl,
        native_default_overflow,
    )
    runtime = _mean_finite(row.get("runtime_seconds") for row in rows)
    grad_norm = _mean_finite(row.get("custom_grad_norm") for row in rows)
    in_loop_wns_delta = _mean_finite(row.get("wns_delta") for row in rows)
    in_loop_tns_delta = _mean_finite(row.get("tns_delta") for row in rows)
    in_loop_wns = _mean_finite(row.get("wns") for row in rows)
    in_loop_tns = _mean_finite(row.get("tns") for row in rows)
    in_loop_timing_net_coverage = _mean_finite(
        row.get("timing_net_coverage") for row in rows
    )
    in_loop_timing_policy_updates = _mean_finite(
        row.get("timing_policy_update_count") for row in rows
    )
    iteration_budget_satisfied = bool(rows) and all(
        str(row.get("iteration_budget_satisfied", "")).strip().lower()
        in {"1", "true", "yes"}
        for row in rows
    )
    component_summary = _aggregate_component_summaries(rows)
    design_summary = _design_delta_summary(rows)
    structural_failure = bool(ranking.structural_failure_count)
    hpwl_gate_passed = (
        not structural_failure
        and hpwl is not None
        and hpwl <= config.hpwl_gate_search_pct
    )
    overflow_improved = overflow is not None and overflow <= 0.0
    per_design_hpwl_gate_passed = (
        not structural_failure
        and design_summary["worst_design_hpwl_delta_pct"] is not None
        and design_summary["worst_design_hpwl_delta_pct"]
        <= config.per_design_hpwl_gate_search_pct
    )
    per_design_overflow_gate_passed = (
        not structural_failure
        and design_summary["worst_design_overflow_delta_pct"] is not None
        and design_summary["worst_design_overflow_delta_pct"]
        <= config.per_design_overflow_gate_search_pct
    )
    design_both_fraction_passed = (
        design_summary["design_both_improvement_fraction"]
        >= config.min_design_both_improvement_fraction
    )
    robust_gate_passed = (
        per_design_hpwl_gate_passed
        and per_design_overflow_gate_passed
        and design_both_fraction_passed
    )
    elite_gate_passed = (
        hpwl_gate_passed
        and hpwl is not None
        and hpwl <= config.hpwl_gate_elite_pct
        and overflow_improved
        and robust_gate_passed
        and not ranking.severe_regression_count
    )
    constrained_score = _constrained_score(
        average_rank=float(ranking.average_rank),
        hpwl_delta_pct=hpwl,
        overflow_delta_pct=overflow,
        hpwl_gate_passed=hpwl_gate_passed,
        overflow_gate_passed=overflow_improved,
        robust_gate_passed=robust_gate_passed,
        require_robust_gate=config.require_robust_parent_gate,
        design_both_improvement_fraction=design_summary["design_both_improvement_fraction"],
        structural_failure=structural_failure,
        hpwl_gate_search_pct=config.hpwl_gate_search_pct,
    )
    combined_score = constrained_score
    parent_eligible = (
        hpwl_gate_passed
        and overflow_improved
        and (robust_gate_passed or not config.require_robust_parent_gate)
        and not structural_failure
    )
    negative_memory_only = (
        structural_failure
        or not hpwl_gate_passed
        or not overflow_improved
        or (config.require_robust_parent_gate and not robust_gate_passed)
    )
    custom_default_hpwl_gate_passed = (
        custom_default_hpwl is not None
        and custom_default_hpwl <= config.custom_default_hpwl_gate_pct
    )
    custom_default_overflow_gate_passed = (
        custom_default_overflow is not None
        and custom_default_overflow <= config.custom_default_overflow_gate_pct
    )
    custom_default_effect_values = [
        abs(value)
        for value in (custom_default_hpwl, custom_default_overflow)
        if value is not None and math.isfinite(value)
    ]
    custom_default_effect_pct = (
        max(custom_default_effect_values) if custom_default_effect_values else None
    )
    custom_default_effect_gate_passed = (
        config.min_custom_default_effect_pct <= 0.0
        or (
            custom_default_effect_pct is not None
            and custom_default_effect_pct >= config.min_custom_default_effect_pct
        )
    )
    custom_default_gate_passed = (
        custom_default_hpwl_gate_passed
        and custom_default_overflow_gate_passed
        and custom_default_effect_gate_passed
    )
    return {
        "combined_score": combined_score,
        "constrained_score": constrained_score,
        "tier2_average_rank": ranking.average_rank,
        "hpwl_delta_pct": hpwl,
        "overflow_delta_pct": overflow,
        "native_default_hpwl_delta_pct": native_default_hpwl,
        "native_default_overflow_delta_pct": native_default_overflow,
        "native_default_comparison_label": native_default_label,
        "beats_native_default_pareto": native_default_label == "dominates_native_default",
        "native_default_tradeoff": native_default_label
        in {
            "native_tradeoff_overflow_for_hpwl",
            "native_tradeoff_hpwl_for_overflow",
        },
        "native_default_regression": native_default_label == "regresses_native_default",
        "custom_default_hpwl_delta_pct": custom_default_hpwl,
        "custom_default_overflow_delta_pct": custom_default_overflow,
        "custom_default_hpwl_gate_passed": custom_default_hpwl_gate_passed,
        "custom_default_overflow_gate_passed": custom_default_overflow_gate_passed,
        "custom_default_effect_pct": custom_default_effect_pct,
        "custom_default_effect_gate_passed": custom_default_effect_gate_passed,
        "custom_default_gate_passed": custom_default_gate_passed,
        "custom_grad_norm": grad_norm,
        "in_loop_wns_delta": in_loop_wns_delta,
        "in_loop_tns_delta": in_loop_tns_delta,
        "in_loop_wns": in_loop_wns,
        "in_loop_tns": in_loop_tns,
        "in_loop_timing_net_coverage": in_loop_timing_net_coverage,
        "in_loop_timing_policy_updates": in_loop_timing_policy_updates,
        "component_summary": component_summary,
        "component_feedback": _component_feedback_lesson(component_summary),
        "runtime_seconds": runtime,
        "iteration_budget_satisfied": iteration_budget_satisfied,
        "structural_failure_count": ranking.structural_failure_count,
        "metric_regression_count": ranking.metric_regression_count,
        "severe_regression_count": ranking.severe_regression_count,
        "design_count": ranking.design_count,
        "seed_count": ranking.seed_count,
        "cell_count": ranking.cell_count,
        "metric_ranks": ranking.metric_ranks,
        "design_ranks": ranking.design_ranks,
        **design_summary,
        "hpwl_gate_passed": hpwl_gate_passed,
        "overflow_gate_passed": overflow_improved,
        "per_design_hpwl_gate_passed": per_design_hpwl_gate_passed,
        "per_design_overflow_gate_passed": per_design_overflow_gate_passed,
        "design_both_fraction_passed": design_both_fraction_passed,
        "robust_gate_passed": robust_gate_passed,
        "require_robust_parent_gate": config.require_robust_parent_gate,
        "near_miss_parent_due_to_robustness": (
            parent_eligible and not robust_gate_passed
        ),
        "elite_gate_passed": elite_gate_passed,
        "primary_baseline": config.primary_baseline,
        "parent_eligible": parent_eligible,
        "negative_memory_only": negative_memory_only,
        "feedback_lesson": _feedback_lesson(
            structural_failure=structural_failure,
            hpwl_gate_passed=hpwl_gate_passed,
            overflow_gate_passed=overflow_improved,
            robust_gate_passed=robust_gate_passed,
            design_summary=design_summary,
            hpwl_delta_pct=hpwl,
            overflow_delta_pct=overflow,
            elite_gate_passed=elite_gate_passed,
        ),
        "outcome_label": _mean_outcome_label(rows),
        "active_delta_baseline": config.primary_baseline,
        "per_design_delta_baseline": config.primary_baseline,
    }


def _apply_final_def_selection_metrics(
    metrics: dict[str, Any],
    config: OpenEvolveTier2Config,
) -> None:
    """Rebase placement selection onto metrics measured from the emitted DEF.

    DREAMPlace's normal log reports final HPWL after detailed placement but its
    last overflow value belongs to global placement. The fixed-DEF evaluator
    used by the timing proxy recomputes HPWL and density overflow from the same
    emitted placement that OpenTimer sees. Keeping all four selection signals
    on this placement state avoids mixing optimizer stages.
    """

    metrics["final_def_selection_metrics_applied"] = False
    if not config.timing_proxy_enabled:
        metrics["final_def_selection_metrics_reason"] = "timing_proxy_disabled"
        return
    if config.primary_baseline != "default":
        metrics["final_def_selection_metrics_reason"] = (
            "final-DEF timing comparison is currently referenced to native default"
        )
        return
    if not bool(metrics.get("timing_proxy_source_placement_budget_satisfied", False)):
        metrics["final_def_selection_metrics_reason"] = (
            "source placement did not satisfy the configured iteration budget"
        )
        return

    hpwl = _finite_number(metrics.get("final_def_hpwl_delta_pct"))
    overflow = _finite_number(metrics.get("final_def_overflow_delta_pct"))
    per_design_rows = metrics.get("final_def_per_design_deltas")
    if hpwl is None or overflow is None or not isinstance(per_design_rows, list):
        metrics["final_def_selection_metrics_reason"] = (
            "final-DEF HPWL/overflow comparison is incomplete"
        )
        return

    metrics.setdefault("optimizer_log_hpwl_delta_pct", metrics.get("hpwl_delta_pct"))
    metrics.setdefault("optimizer_log_overflow_delta_pct", metrics.get("overflow_delta_pct"))
    metrics.setdefault("optimizer_log_per_design_deltas", metrics.get("per_design_deltas"))

    design_summary = _design_delta_summary(per_design_rows)
    structural_failure = bool(metrics.get("structural_failure_count", 0))
    worst_hpwl = _finite_number(design_summary.get("worst_design_hpwl_delta_pct"))
    worst_overflow = _finite_number(
        design_summary.get("worst_design_overflow_delta_pct")
    )
    design_both_fraction = _finite_number(
        design_summary.get("design_both_improvement_fraction")
    )
    if design_both_fraction is None:
        design_both_fraction = 0.0

    hpwl_gate_passed = (
        not structural_failure and hpwl <= config.hpwl_gate_search_pct
    )
    overflow_gate_passed = not structural_failure and overflow <= 0.0
    per_design_hpwl_gate_passed = (
        not structural_failure
        and worst_hpwl is not None
        and worst_hpwl <= config.per_design_hpwl_gate_search_pct
    )
    per_design_overflow_gate_passed = (
        not structural_failure
        and worst_overflow is not None
        and worst_overflow <= config.per_design_overflow_gate_search_pct
    )
    design_both_fraction_passed = (
        design_both_fraction >= config.min_design_both_improvement_fraction
    )
    robust_gate_passed = (
        per_design_hpwl_gate_passed
        and per_design_overflow_gate_passed
        and design_both_fraction_passed
    )
    severe_regression_count = _active_severe_regression_count(
        hpwl_delta_pct=hpwl,
        overflow_delta_pct=overflow,
        worst_design_hpwl_delta_pct=worst_hpwl,
        worst_design_overflow_delta_pct=worst_overflow,
        severe_threshold_pct=config.severe_regression_pct,
        overflow_severe_threshold_pct=config.overflow_severe_regression_pct,
        structural_failure=structural_failure,
    )
    elite_gate_passed = (
        hpwl_gate_passed
        and hpwl <= config.hpwl_gate_elite_pct
        and overflow_gate_passed
        and robust_gate_passed
        and not severe_regression_count
    )
    constrained_score = _constrained_score(
        average_rank=float(metrics.get("tier2_average_rank") or 999.0),
        hpwl_delta_pct=hpwl,
        overflow_delta_pct=overflow,
        hpwl_gate_passed=hpwl_gate_passed,
        overflow_gate_passed=overflow_gate_passed,
        robust_gate_passed=robust_gate_passed,
        require_robust_gate=config.require_robust_parent_gate,
        design_both_improvement_fraction=design_both_fraction,
        structural_failure=structural_failure,
        hpwl_gate_search_pct=config.hpwl_gate_search_pct,
    )
    parent_eligible = (
        hpwl_gate_passed
        and overflow_gate_passed
        and (robust_gate_passed or not config.require_robust_parent_gate)
        and not structural_failure
    )
    negative_memory_only = (
        structural_failure
        or not hpwl_gate_passed
        or not overflow_gate_passed
        or (config.require_robust_parent_gate and not robust_gate_passed)
    )
    native_default_label = _native_default_comparison_label(hpwl, overflow)

    metrics.update(
        {
            "hpwl_delta_pct": hpwl,
            "overflow_delta_pct": overflow,
            "native_default_hpwl_delta_pct": hpwl,
            "native_default_overflow_delta_pct": overflow,
            "native_default_comparison_label": native_default_label,
            "beats_native_default_pareto": (
                native_default_label == "dominates_native_default"
            ),
            "native_default_tradeoff": native_default_label
            in {
                "native_tradeoff_overflow_for_hpwl",
                "native_tradeoff_hpwl_for_overflow",
            },
            "native_default_regression": (
                native_default_label == "regresses_native_default"
            ),
            "combined_score": constrained_score,
            "constrained_score": constrained_score,
            "hpwl_gate_passed": hpwl_gate_passed,
            "overflow_gate_passed": overflow_gate_passed,
            "per_design_hpwl_gate_passed": per_design_hpwl_gate_passed,
            "per_design_overflow_gate_passed": per_design_overflow_gate_passed,
            "design_both_fraction_passed": design_both_fraction_passed,
            "robust_gate_passed": robust_gate_passed,
            "severe_regression_count": severe_regression_count,
            "elite_gate_passed": elite_gate_passed,
            "parent_eligible": parent_eligible,
            "negative_memory_only": negative_memory_only,
            "selection_metric_stage": "final_def_post_detailed_placement",
            "selection_metrics_source": "timing_proxy_fixed_def_evaluator",
            "final_def_selection_metrics_applied": True,
            "final_def_selection_metrics_reason": "complete",
            "active_delta_baseline": "default",
            "per_design_delta_baseline": "default",
            **design_summary,
            "feedback_lesson": _feedback_lesson(
                structural_failure=structural_failure,
                hpwl_gate_passed=hpwl_gate_passed,
                overflow_gate_passed=overflow_gate_passed,
                robust_gate_passed=robust_gate_passed,
                design_summary=design_summary,
                hpwl_delta_pct=hpwl,
                overflow_delta_pct=overflow,
                elite_gate_passed=elite_gate_passed,
            ),
        }
    )


def _native_default_comparison_label(
    hpwl_delta_pct: Any,
    overflow_delta_pct: Any,
) -> str:
    """Classify generated-objective behavior against native DREAMPlace default.

    This label is intentionally diagnostic rather than a hard selection gate.
    CoEvoP&R may eventually accept a routing/PPA tradeoff, but reports must not
    conflate a custom-default or fixed-baseline win with native-default dominance.
    """

    hpwl = _finite_number(hpwl_delta_pct)
    overflow = _finite_number(overflow_delta_pct)
    if hpwl is None or overflow is None:
        return "native_default_unknown"
    hpwl_improved = hpwl <= 0.0
    overflow_improved = overflow <= 0.0
    if hpwl_improved and overflow_improved:
        return "dominates_native_default"
    if not hpwl_improved and overflow_improved:
        return "native_tradeoff_overflow_for_hpwl"
    if hpwl_improved and not overflow_improved:
        return "native_tradeoff_hpwl_for_overflow"
    return "regresses_native_default"


def _aggregate_component_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        payload = row.get("component_summary_json")
        if not payload:
            continue
        try:
            summary = json.loads(str(payload))
        except json.JSONDecodeError:
            continue
        if not isinstance(summary, dict):
            continue
        for name, item in summary.items():
            if isinstance(item, dict):
                grouped.setdefault(str(name), []).append(item)
    aggregated: dict[str, Any] = {}
    for name, items in grouped.items():
        numeric_keys = (
            "start",
            "mid",
            "end",
            "min",
            "max",
            "mean",
            "output_value_mean",
            "output_value_max",
            "value_share_mean",
            "value_share_max",
            "grad_norm_mean",
            "grad_norm_max",
            "grad_ratio_mean",
            "grad_ratio_max",
        )
        entry: dict[str, Any] = {}
        for key in numeric_keys:
            value = _mean_finite(item.get(key) for item in items)
            if value is not None:
                entry[key] = value
        trends = [str(item.get("trend")) for item in items if item.get("trend")]
        if trends:
            entry["trend"] = max(set(trends), key=trends.count)
        trajectory = next(
            (
                list(item.get("trajectory", []))[:10]
                for item in items
                if isinstance(item.get("trajectory"), list)
                and item.get("trajectory")
            ),
            [],
        )
        if trajectory:
            entry["trajectory"] = trajectory
        grad_ratio_trajectory = next(
            (
                list(item.get("grad_ratio_trajectory", []))[:10]
                for item in items
                if isinstance(item.get("grad_ratio_trajectory"), list)
                and item.get("grad_ratio_trajectory")
            ),
            [],
        )
        if grad_ratio_trajectory:
            entry["grad_ratio_trajectory"] = grad_ratio_trajectory
        entry["grad_ratio_count"] = sum(
            int(item.get("grad_ratio_count") or 0) for item in items
        )
        entry["flat_or_saturated"] = any(
            bool(item.get("flat_or_saturated")) for item in items
        )
        entry["cell_count"] = len(items)
        aggregated[name] = entry
    return aggregated


def _component_feedback_lesson(summary: dict[str, Any]) -> list[str]:
    lessons = []
    for name, item in sorted(summary.items()):
        if not isinstance(item, dict):
            continue
        trend = item.get("trend", "unknown")
        flat = bool(item.get("flat_or_saturated"))
        mean = item.get("mean")
        if flat:
            lessons.append(
                f"{name}: flat_or_saturated with mean={mean}; rescale or replace if it was intended to guide placement."
            )
        else:
            lessons.append(f"{name}: trend={trend}, mean={mean}.")
    return lessons[:6]


def _runtime_baseline_objective_ids(
    *,
    tier2_summary: dict[str, Any],
    baseline_paths: list[str],
    baseline_preset_names: list[str],
) -> set[str]:
    """Resolve the baseline portfolio as runtime Tier-2 objective IDs."""

    baseline_ids = {"default", "custom_default"}
    baseline_path_set = {str(Path(path).resolve()) for path in baseline_paths}
    manifest_path = tier2_summary.get("objective_manifest")
    if manifest_path and Path(manifest_path).is_file():
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = []
        manifest_items = manifest if isinstance(manifest, list) else []
        for item in manifest_items:
            objective_path = item.get("objective_path") if isinstance(item, dict) else None
            objective_id = item.get("objective_id") if isinstance(item, dict) else None
            if not objective_path or not objective_id:
                continue
            if str(Path(objective_path).resolve()) in baseline_path_set:
                baseline_ids.add(str(objective_id))

    for preset_name in baseline_preset_names:
        try:
            baseline_ids.add(objective_preset(preset_name).id)
        except KeyError:
            baseline_ids.add(str(preset_name))
    return baseline_ids


def _baseline_portfolio_summary(
    *,
    rows: list[dict[str, Any]],
    rankings: list[Any],
    candidate_objective_id: str,
    baseline_objective_ids: set[str],
    min_effect_pct: float = 0.0,
    min_behavior_distance_pct: float = 0.0,
) -> dict[str, Any]:
    """Compare one generated candidate against the evaluated baseline portfolio.

    This is measured execution feedback only. It does not provide the LLM a
    formula template or evaluator thresholds; it records whether the candidate
    actually out-ranked the fixed baselines run in the same Tier-2 batch.
    """

    ranking_by_id = {str(item.objective_id): item for item in rankings}
    candidate = ranking_by_id.get(candidate_objective_id)
    baseline_summaries = [
        item
        for objective_id, item in ranking_by_id.items()
        if objective_id in baseline_objective_ids and objective_id != candidate_objective_id
    ]
    if candidate is None or not baseline_summaries:
        return {
            "baseline_portfolio_compared": False,
            "baseline_portfolio_size": len(baseline_summaries),
            "baseline_portfolio_objectives": sorted(
                item.objective_id for item in baseline_summaries
            ),
            "baseline_portfolio_gate_passed": False,
            "beats_baseline_portfolio": False,
            "baseline_portfolio_rank_beats_best": False,
            "baseline_portfolio_effect_pct": None,
            "baseline_portfolio_effect_gate_passed": False,
            "baseline_portfolio_behavior_distance_pct": None,
            "baseline_portfolio_behavior_distance_gate_passed": False,
        }

    best_baseline = min(baseline_summaries, key=_baseline_portfolio_rank_key)
    sorted_portfolio = sorted(
        [candidate, *baseline_summaries],
        key=_baseline_portfolio_rank_key,
    )
    candidate_position = next(
        index
        for index, item in enumerate(sorted_portfolio, start=1)
        if item.objective_id == candidate_objective_id
    )
    candidate_beats_best = (
        candidate.structural_failure_count == 0
        and _baseline_portfolio_rank_key(candidate)
        < _baseline_portfolio_rank_key(best_baseline)
    )
    candidate_rows = [
        row for row in rows if str(row.get("objective_id")) == candidate_objective_id
    ]
    best_baseline_rows = [
        row for row in rows if str(row.get("objective_id")) == best_baseline.objective_id
    ]
    candidate_hpwl = _mean_finite(row.get("hpwl_delta_pct") for row in candidate_rows)
    candidate_overflow = _mean_finite(row.get("overflow_delta_pct") for row in candidate_rows)
    best_baseline_hpwl = _mean_finite(row.get("hpwl_delta_pct") for row in best_baseline_rows)
    best_baseline_overflow = _mean_finite(
        row.get("overflow_delta_pct") for row in best_baseline_rows
    )
    hpwl_delta_vs_best = _metric_delta(candidate_hpwl, best_baseline_hpwl)
    overflow_delta_vs_best = _metric_delta(candidate_overflow, best_baseline_overflow)
    pareto_label_vs_best = _baseline_delta_pareto_label(
        hpwl_delta_vs_best,
        overflow_delta_vs_best,
    )
    effect_values = [
        abs(float(left) - float(right))
        for left, right in (
            (candidate_hpwl, best_baseline_hpwl),
            (candidate_overflow, best_baseline_overflow),
        )
        if left is not None
        and right is not None
        and math.isfinite(float(left))
        and math.isfinite(float(right))
    ]
    baseline_effect_pct = max(effect_values) if effect_values else None
    effect_gate_passed = (
        float(min_effect_pct) <= 0.0
        or (
            baseline_effect_pct is not None
            and baseline_effect_pct >= float(min_effect_pct)
        )
    )
    closest_behavior = _closest_baseline_behavior_distance(
        rows=rows,
        candidate_objective_id=candidate_objective_id,
        baseline_objective_ids=baseline_objective_ids,
    )
    behavior_distance_pct = (
        closest_behavior["distance_pct"] if closest_behavior is not None else None
    )
    behavior_gate_passed = (
        float(min_behavior_distance_pct) <= 0.0
        or (
            behavior_distance_pct is not None
            and behavior_distance_pct >= float(min_behavior_distance_pct)
        )
    )
    candidate_gate_passed = candidate_beats_best and effect_gate_passed and behavior_gate_passed
    return {
        "baseline_portfolio_compared": True,
        "baseline_portfolio_size": len(baseline_summaries),
        "baseline_portfolio_objectives": sorted(
            item.objective_id for item in baseline_summaries
        ),
        "baseline_portfolio_candidate_rank": candidate_position,
        "baseline_portfolio_candidate_average_rank": candidate.average_rank,
        "baseline_portfolio_best_objective_id": best_baseline.objective_id,
        "baseline_portfolio_best_average_rank": best_baseline.average_rank,
        "baseline_portfolio_best_hpwl_delta_pct": best_baseline_hpwl,
        "baseline_portfolio_best_overflow_delta_pct": best_baseline_overflow,
        "baseline_portfolio_hpwl_delta_vs_best_pct": hpwl_delta_vs_best,
        "baseline_portfolio_overflow_delta_vs_best_pct": overflow_delta_vs_best,
        "baseline_portfolio_pareto_label_vs_best": pareto_label_vs_best,
        "baseline_portfolio_pareto_dominates_best": (
            pareto_label_vs_best == "dominates_best_baseline"
        ),
        "baseline_portfolio_pareto_tradeoff_vs_best": pareto_label_vs_best
        in {
            "baseline_tradeoff_overflow_for_hpwl",
            "baseline_tradeoff_hpwl_for_overflow",
        },
        "baseline_portfolio_rank_beats_best": candidate_beats_best,
        "baseline_portfolio_effect_pct": baseline_effect_pct,
        "baseline_portfolio_effect_gate_passed": effect_gate_passed,
        "baseline_portfolio_min_effect_pct": float(min_effect_pct),
        "baseline_portfolio_closest_behavior_objective_id": (
            closest_behavior["objective_id"] if closest_behavior is not None else None
        ),
        "baseline_portfolio_behavior_distance_pct": behavior_distance_pct,
        "baseline_portfolio_behavior_distance_gate_passed": behavior_gate_passed,
        "baseline_portfolio_min_behavior_distance_pct": float(
            min_behavior_distance_pct
        ),
        "baseline_portfolio_gate_passed": candidate_gate_passed,
        "beats_baseline_portfolio": candidate_gate_passed,
    }


def _metric_delta(left: Any, right: Any) -> float | None:
    left_value = _finite_number(left)
    right_value = _finite_number(right)
    if left_value is None or right_value is None:
        return None
    return float(left_value) - float(right_value)


def _baseline_delta_pareto_label(
    hpwl_delta_vs_best_pct: Any,
    overflow_delta_vs_best_pct: Any,
) -> str:
    hpwl = _finite_number(hpwl_delta_vs_best_pct)
    overflow = _finite_number(overflow_delta_vs_best_pct)
    if hpwl is None or overflow is None:
        return "baseline_pareto_unknown"
    hpwl_improved = hpwl <= 0.0
    overflow_improved = overflow <= 0.0
    if hpwl == 0.0 and overflow == 0.0:
        return "ties_best_baseline"
    if hpwl_improved and overflow_improved:
        return "dominates_best_baseline"
    if not hpwl_improved and overflow_improved:
        return "baseline_tradeoff_overflow_for_hpwl"
    if hpwl_improved and not overflow_improved:
        return "baseline_tradeoff_hpwl_for_overflow"
    return "regresses_best_baseline"


def _closest_baseline_behavior_distance(
    *,
    rows: list[dict[str, Any]],
    candidate_objective_id: str,
    baseline_objective_ids: set[str],
) -> dict[str, Any] | None:
    candidate = _objective_behavior_vector(rows, candidate_objective_id)
    if candidate is None:
        return None
    closest: dict[str, Any] | None = None
    for objective_id in sorted(baseline_objective_ids):
        if objective_id == candidate_objective_id:
            continue
        baseline = _objective_behavior_vector(rows, objective_id)
        if baseline is None:
            continue
        deltas = [
            abs(float(candidate[key]) - float(baseline[key]))
            for key in sorted(candidate)
            if key in baseline
            and candidate[key] is not None
            and baseline[key] is not None
            and math.isfinite(float(candidate[key]))
            and math.isfinite(float(baseline[key]))
        ]
        if not deltas:
            continue
        distance = max(deltas)
        if closest is None or distance < closest["distance_pct"]:
            closest = {"objective_id": objective_id, "distance_pct": distance}
    return closest


def _objective_behavior_vector(
    rows: list[dict[str, Any]],
    objective_id: str,
) -> dict[str, float] | None:
    values: dict[str, list[float]] = {}
    for row in rows:
        if str(row.get("objective_id")) != str(objective_id):
            continue
        design = str(row.get("design") or "")
        seed = str(row.get("seed") or "")
        for metric in ("hpwl_delta_pct", "overflow_delta_pct"):
            value = _finite_number(row.get(metric))
            if value is None:
                continue
            key = f"{design}/{seed}/{metric}"
            values.setdefault(key, []).append(value)
    if not values:
        return None
    return {
        key: sum(items) / len(items)
        for key, items in values.items()
        if items
    }


def _rank_summary_key(summary: Any) -> tuple[float, int, int, float, str]:
    runtime = summary.mean_runtime_seconds
    return (
        float(summary.average_rank),
        int(summary.structural_failure_count),
        int(summary.severe_regression_count),
        float(runtime) if runtime is not None and math.isfinite(float(runtime)) else math.inf,
        str(summary.objective_id),
    )


def _baseline_portfolio_rank_key(summary: Any) -> tuple[int, int, float, float, str]:
    """Rank fixed-baseline portfolio entries for promotion decisions.

    The global rank table is diagnostic and coefficient-free, so it ranks by
    average metric rank first. Promotion is stricter: a fixed baseline that gets
    a strong average rank by sacrificing one design should not block a generated
    objective that has no structural failures or severe regressions.
    """

    runtime = summary.mean_runtime_seconds
    return (
        int(summary.structural_failure_count),
        int(summary.severe_regression_count),
        float(summary.average_rank),
        float(runtime) if runtime is not None and math.isfinite(float(runtime)) else math.inf,
        str(summary.objective_id),
    )


def _apply_baseline_portfolio_gate(
    metrics: dict[str, Any],
    config: OpenEvolveTier2Config,
) -> None:
    if (
        not config.require_baseline_portfolio_gate
        and not config.require_baseline_portfolio_pareto
    ):
        return
    if (
        config.require_baseline_portfolio_gate
        and bool(metrics.get("baseline_portfolio_gate_passed"))
        and not config.require_baseline_portfolio_pareto
    ):
        metrics.pop("baseline_portfolio_gate_note", None)
        return
    if config.require_baseline_portfolio_gate and not bool(
        metrics.get("baseline_portfolio_gate_passed")
    ):
        _block_baseline_portfolio_parent(
            metrics,
            note=(
                "Generated objective did not outperform the evaluated baseline "
                "portfolio in the same DREAMPlace search batch."
            ),
        )
        best_id = metrics.get("baseline_portfolio_best_objective_id")
        closest_behavior_id = metrics.get("baseline_portfolio_closest_behavior_objective_id")
        effect = _finite_number(metrics.get("baseline_portfolio_effect_pct"))
        min_effect = _finite_number(metrics.get("baseline_portfolio_min_effect_pct"))
        behavior_gate = metrics.get("baseline_portfolio_behavior_distance_gate_passed")
        if best_id:
            if behavior_gate is False and closest_behavior_id:
                metrics["feedback_lesson"] = (
                    "Measured Tier-2 feedback: this objective was not promoted because "
                    f"its measured HPWL/overflow behavior was nearly indistinguishable "
                    f"from fixed baseline {closest_behavior_id}. Change the objective "
                    "mechanism enough to produce a genuinely different placement tradeoff."
                )
            elif (
                min_effect is not None
                and min_effect > 0.0
                and effect is not None
                and effect < min_effect
            ):
                metrics["feedback_lesson"] = (
                    "Measured Tier-2 feedback: this objective was not promoted because "
                    f"its measured effect relative to the strongest fixed baseline ({best_id}) "
                    "was too small to distinguish from a near-default perturbation. "
                    "Change the objective mechanism, not only constants."
                )
            else:
                metrics["feedback_lesson"] = (
                    "Measured Tier-2 feedback: this objective was not promoted because "
                    f"the baseline portfolio still out-ranked it; strongest baseline was {best_id}. "
                    "Change the objective mechanism, not only constants."
                )
        return
    if (
        config.require_baseline_portfolio_pareto
        and not metrics.get("baseline_portfolio_compared")
    ):
        _block_baseline_portfolio_parent(
            metrics,
            note=(
                "Generated objective did not have a valid fixed-baseline portfolio "
                "comparison for Pareto promotion."
            ),
        )
        metrics["feedback_lesson"] = (
            "Measured Tier-2 feedback: this objective was kept as memory but not "
            "promoted because the fixed-baseline portfolio comparison was missing."
        )
        return
    if (
        config.require_baseline_portfolio_pareto
        and metrics.get("baseline_portfolio_compared")
        and not bool(metrics.get("baseline_portfolio_pareto_dominates_best"))
    ):
        label = metrics.get("baseline_portfolio_pareto_label_vs_best")
        best_id = metrics.get("baseline_portfolio_best_objective_id")
        hpwl_vs_best = metrics.get("baseline_portfolio_hpwl_delta_vs_best_pct")
        overflow_vs_best = metrics.get("baseline_portfolio_overflow_delta_vs_best_pct")
        _block_baseline_portfolio_parent(
            metrics,
            note=(
                "Generated objective did not Pareto-improve over the strongest "
                "fixed baseline in the same DREAMPlace search batch."
            ),
        )
        metrics["feedback_lesson"] = (
            "Measured Tier-2 feedback: this objective was kept as memory but not "
            "promoted because it was not a clean Pareto improvement over the strongest "
            f"fixed baseline ({best_id}). Measured label={label}, "
            f"hpwl_vs_best={hpwl_vs_best}, overflow_vs_best={overflow_vs_best}. "
            "Use this as execution feedback to change the placement mechanism rather "
            "than repeating the same fixed-baseline tradeoff."
        )
        metrics["baseline_portfolio_rank_beats_best"] = bool(
            metrics.get("beats_baseline_portfolio")
        )
        metrics["beats_baseline_portfolio"] = False
        return
    metrics.pop("baseline_portfolio_gate_note", None)


def _block_baseline_portfolio_parent(metrics: dict[str, Any], *, note: str) -> None:
    metrics["combined_score"] = 0.0
    metrics["constrained_score"] = 0.0
    metrics["elite_gate_passed"] = False
    metrics["parent_eligible"] = False
    metrics["negative_memory_only"] = True
    metrics["baseline_portfolio_gate_note"] = note


def _apply_native_default_selection_policy(
    metrics: dict[str, Any],
    config: OpenEvolveTier2Config,
    *,
    allow_seed_baseline: bool,
) -> None:
    policy = config.native_default_selection_policy
    if policy not in {"report", "prefer_pareto", "require_pareto"}:
        raise ValueError(
            "native_default_selection_policy must be one of "
            "'report', 'prefer_pareto', or 'require_pareto'"
        )
    label = _native_default_label_from_metrics(metrics)
    metrics["native_default_comparison_label"] = label
    metrics["beats_native_default_pareto"] = label == "dominates_native_default"
    metrics["native_default_tradeoff"] = label in {
        "native_tradeoff_overflow_for_hpwl",
        "native_tradeoff_hpwl_for_overflow",
    }
    metrics["native_default_regression"] = label == "regresses_native_default"
    metrics["native_default_selection_policy"] = policy
    metrics["native_default_pareto_bonus_applied"] = 0.0
    if policy == "report":
        return
    if policy == "prefer_pareto":
        if (
            label == "dominates_native_default"
            and config.native_default_pareto_bonus
            and metrics.get("parent_eligible") is not False
            and not metrics.get("negative_memory_only")
        ):
            bonus = float(config.native_default_pareto_bonus)
            metrics["combined_score"] = float(metrics.get("combined_score") or 0.0) + bonus
            metrics["native_default_pareto_bonus_applied"] = bonus
        return
    if allow_seed_baseline:
        metrics["native_default_gate_note"] = (
            "Seed baseline remains available as an initial parent; generated "
            "children are still evaluated against the native-default policy."
        )
        return
    if label == "dominates_native_default":
        return
    metrics["combined_score"] = 0.0
    metrics["constrained_score"] = 0.0
    metrics["elite_gate_passed"] = False
    metrics["parent_eligible"] = False
    metrics["negative_memory_only"] = True
    metrics["native_default_gate_note"] = (
        "Generated objective did not produce a Pareto improvement relative to "
        "native DREAMPlace default under the evaluator-side native-default "
        "selection policy."
    )
    metrics["feedback_lesson"] = (
        "Measured Tier-2 feedback: this candidate remains useful negative "
        "memory, but it was not promoted because its native-default comparison "
        "was a tradeoff or regression rather than a Pareto improvement. Explore "
        "a different routing-aware mechanism rather than a near-default "
        "perturbation."
    )


def _apply_near_miss_parent_policy(
    metrics: dict[str, Any],
    config: OpenEvolveTier2Config,
    *,
    allow_seed_baseline: bool,
) -> None:
    """Allow useful measured tradeoffs to serve as OpenEvolve stepping stones.

    Final promotion remains governed by the normal elite gates. This policy only
    prevents the memory loop from getting stuck on fixed seed baselines when a
    generated objective is non-structural, behaviorally distinct from the fixed
    portfolio, and shows a bounded measured tradeoff that could be improved by a
    later mutation.
    """

    if not config.allow_near_miss_parents or allow_seed_baseline:
        return
    if metrics.get("parent_eligible") and not metrics.get("negative_memory_only"):
        return
    if metrics.get("structural_failure_count"):
        return
    if metrics.get("severe_regression_count"):
        return
    # In controller mode the searched mechanism is the schedule itself, so a
    # candidate without measured routing-term influence can still be a valid
    # stepping stone.
    if config.objective_mode != "controller" and not metrics.get(
        "routing_aware_tier2_candidate"
    ):
        return
    if metrics.get("baseline_portfolio_behavior_distance_gate_passed") is False:
        return
    # Stepping-stone parenting deliberately does NOT require Pareto domination
    # of the single strongest baseline; that requirement stays on elite
    # promotion. A bounded tradeoff that outranks the whole fixed portfolio is
    # exactly the memory the loop needs to breed from.
    if (
        config.require_baseline_portfolio_gate
        and metrics.get("baseline_portfolio_compared")
        and not bool(metrics.get("beats_baseline_portfolio"))
    ):
        return
    hpwl = _finite_number(metrics.get("hpwl_delta_pct"))
    overflow = _finite_number(metrics.get("overflow_delta_pct"))
    worst_hpwl = _finite_number(metrics.get("worst_design_hpwl_delta_pct"))
    worst_overflow = _finite_number(metrics.get("worst_design_overflow_delta_pct"))
    if hpwl is None or overflow is None:
        return
    if worst_hpwl is None:
        worst_hpwl = hpwl
    if worst_overflow is None:
        worst_overflow = overflow
    if worst_hpwl > config.near_miss_max_worst_hpwl_pct:
        return
    if worst_overflow > config.near_miss_max_worst_overflow_pct:
        return

    has_measured_signal = (
        hpwl <= 0.0
        or overflow <= 0.0
        or _finite_number(metrics.get("custom_default_hpwl_delta_pct")) is not None
        and _finite_number(metrics.get("custom_default_hpwl_delta_pct")) <= 0.0
        or _finite_number(metrics.get("custom_default_overflow_delta_pct")) is not None
        and _finite_number(metrics.get("custom_default_overflow_delta_pct")) <= 0.0
        or _native_default_label_from_metrics(metrics)
        in {
            "dominates_native_default",
            "native_tradeoff_overflow_for_hpwl",
            "native_tradeoff_hpwl_for_overflow",
        }
    )
    if not has_measured_signal:
        return

    improvement_signal = max(0.0, -hpwl) + max(0.0, -overflow)
    bounded_tradeoff_penalty = max(0.0, hpwl) + max(0.0, overflow)
    near_miss_score = max(
        config.near_miss_parent_score_floor,
        config.near_miss_parent_score_floor
        + 0.01 * improvement_signal
        - 0.005 * bounded_tradeoff_penalty,
    )
    metrics["combined_score"] = max(
        float(metrics.get("combined_score") or 0.0),
        float(near_miss_score),
    )
    metrics["constrained_score"] = max(
        float(metrics.get("constrained_score") or 0.0),
        float(near_miss_score),
    )
    metrics["parent_eligible"] = True
    metrics["negative_memory_only"] = False
    metrics["elite_gate_passed"] = False
    metrics["exploration_parent_only"] = True
    metrics["near_miss_parent_score"] = near_miss_score
    metrics["near_miss_parent_reason"] = (
        "Generated objective is a bounded, behaviorally distinct Tier-2 "
        "tradeoff. It can be mutated as OpenEvolve exploration memory, but it "
        "is not a final promoted objective."
    )
    existing_lesson = str(metrics.get("feedback_lesson") or "").strip()
    near_miss_lesson = (
        "Measured Tier-2 near-miss: this objective is distinct enough to mutate "
        "as a stepping stone, but final promotion still requires measured "
        "superiority over fixed baselines and robust multi-design behavior."
    )
    metrics["feedback_lesson"] = (
        f"{existing_lesson} {near_miss_lesson}".strip()
        if existing_lesson
        else near_miss_lesson
    )


def _prepare_pareto_candidate(
    metrics: dict[str, Any],
    config: OpenEvolveTier2Config,
    *,
    allow_seed_baseline: bool,
) -> None:
    """Admit every completed finite placement to coefficient-free selection.

    Pareto ranks are population properties and are assigned after programs are
    inserted into the database. This step deliberately applies no HPWL,
    overflow, or timing preference as an admission gate.
    """

    structural_ok = not bool(metrics.get("structural_failure_count"))
    hpwl = _finite_number(metrics.get("hpwl_delta_pct"))
    overflow = _finite_number(metrics.get("overflow_delta_pct"))
    placement_finite = hpwl is not None and overflow is not None
    budget_ok = bool(metrics.get("iteration_budget_satisfied", True))
    admitted = structural_ok and placement_finite and budget_ok
    metrics.update(
        {
            "selection_policy": config.selection_policy,
            "pareto_multiobjective_selection": True,
            "pareto_admission_passed": admitted,
            "pareto_front": None,
            "pareto_equal_design_rank": None,
            "combined_score": 1.0 if admitted else 0.0,
            "constrained_score": 1.0 if admitted else 0.0,
            # Compatibility fields are admission indicators under this policy,
            # not metric-specific gates.
            "hpwl_gate_passed": admitted,
            "overflow_gate_passed": admitted,
            "robust_gate_passed": admitted,
            "elite_gate_passed": False,
            "parent_eligible": admitted,
            "negative_memory_only": not admitted,
            "exploration_parent_only": admitted and not allow_seed_baseline,
        }
    )
    metrics["feedback_lesson"] = (
        "Completed finite placement admitted to coefficient-free Pareto ranking; "
        "HPWL, overflow, WNS, and TNS are compared without manual metric weights."
        if admitted
        else "Placement was not admitted to Pareto ranking because execution, "
        "iteration-budget, or finite-metric validation failed."
    )


def _apply_timing_controller_admission(
    metrics: dict[str, Any],
    spec: ObjectiveSpec,
    config: OpenEvolveTier2Config,
) -> None:
    """Fail closed when an in-loop timing policy lacks trustworthy STA data."""

    if spec.net_weight_policy is None:
        metrics.setdefault("timing_controller_required", False)
        return

    coverage = _finite_number(metrics.get("in_loop_timing_net_coverage"))
    updates = _finite_number(metrics.get("in_loop_timing_policy_updates"))
    wns_delta = _finite_number(metrics.get("in_loop_wns_delta"))
    tns_delta = _finite_number(metrics.get("in_loop_tns_delta"))
    coverage_ok = (
        coverage is not None
        and coverage >= config.timing_controller_min_net_coverage
    )
    updates_ok = updates is not None and updates >= 1.0
    # The matched native placement is not timing-driven and reports no in-loop
    # WNS/TNS to difference against, so finite in-loop values are sufficient.
    metrics_ok = (wns_delta is not None and tns_delta is not None) or (
        _finite_number(metrics.get("in_loop_wns")) is not None
        and _finite_number(metrics.get("in_loop_tns")) is not None
    )
    budget_ok = bool(metrics.get("iteration_budget_satisfied"))
    admitted = coverage_ok and updates_ok and metrics_ok and budget_ok

    metrics.update(
        {
            "timing_controller_required": True,
            "timing_controller_admission_passed": admitted,
            "timing_controller_net_coverage": coverage,
            "timing_controller_min_net_coverage": (
                config.timing_controller_min_net_coverage
            ),
            "timing_controller_updates": updates,
            "timing_controller_wns_delta": wns_delta,
            "timing_controller_tns_delta": tns_delta,
            "timing_controller_feedback_source": (
                "dreamplace_opentimer_in_loop"
            ),
        }
    )
    # Reuse the existing Pareto timing fields when no separate fixed-DEF
    # timing-proxy evaluation supplied them. The source tag prevents the two
    # measurement paths from being confused in reports.
    if wns_delta is not None:
        metrics.setdefault("timing_proxy_wns_delta", wns_delta)
    if tns_delta is not None:
        metrics.setdefault("timing_proxy_tns_delta", tns_delta)
    metrics.setdefault(
        "timing_proxy_status",
        "success" if admitted else ("partial" if metrics_ok else "failed"),
    )
    metrics.setdefault("timing_proxy_mode", "gate")
    metrics.setdefault("timing_proxy_source_placement_budget_satisfied", budget_ok)

    if admitted:
        return
    metrics["structural_failure_count"] = max(
        int(metrics.get("structural_failure_count") or 0),
        1,
    )
    metrics.update(
        {
            "combined_score": 0.0,
            "constrained_score": 0.0,
            "parent_eligible": False,
            "elite_gate_passed": False,
            "negative_memory_only": True,
            "timing_controller_failure": (
                "insufficient_net_coverage"
                if not coverage_ok
                else "timing_update_not_executed"
                if not updates_ok
                else "nonfinite_or_missing_timing_metrics"
                if not metrics_ok
                else "incomplete_placement_budget"
            ),
            "feedback_lesson": (
                "The in-loop timing policy was retained as negative memory but "
                "excluded from selection because OpenTimer coverage, update "
                "execution, finite WNS/TNS, or the full placement budget was "
                "not satisfied."
            ),
        }
    )


def _refresh_pareto_selection(
    database: ObjectiveProgramDatabase,
    config: OpenEvolveTier2Config,
    timing_admission: dict[str, Any] | None = None,
) -> None:
    if config.selection_policy != "pareto_multiobjective":
        return
    candidates = [
        program
        for program in database.programs.values()
        if program.status == "accepted"
        and bool(program.metrics.get("pareto_admission_passed"))
        and not bool(program.metrics.get("structural_failure_count"))
    ]
    if not candidates:
        database.rebuild_indexes()
        return

    if timing_admission is None:
        timing_admission = default_timing_proxy_admission(
            getattr(config, "timing_proxy_mode", "diagnostic")
        )
    for program in candidates:
        apply_timing_proxy_admission(program.metrics, timing_admission)
    evidence = {program.id: _pareto_evidence(program.metrics) for program in candidates}
    fronts = _nondominated_fronts(candidates, evidence)
    equal_design_ranks = _equal_design_ranks(candidates)
    front_count = len(fronts)
    for front_index, front in enumerate(fronts):
        ordered_front = sorted(
            front,
            key=lambda program: (
                equal_design_ranks.get(program.id, float("inf")),
                _finite_number(program.metrics.get("runtime_seconds"))
                if _finite_number(program.metrics.get("runtime_seconds")) is not None
                else float("inf"),
                program.id,
            ),
        )
        for within_front_rank, program in enumerate(ordered_front, start=1):
            rank = equal_design_ranks.get(program.id, float(within_front_rank))
            # Pure ordinal scalarization for the database API: every member of
            # a better Pareto front outranks every member of the next front;
            # equal-design rank and then runtime order candidates only within
            # that front. No metric coefficient enters this score.
            within_front_quality = (
                len(ordered_front) - within_front_rank + 1
            ) / float(len(ordered_front) + 1)
            score = float(front_count - front_index) + within_front_quality
            metrics = program.metrics
            metrics.update(
                {
                    "pareto_front": front_index,
                    "pareto_equal_design_rank": rank,
                    "pareto_within_front_rank": within_front_rank,
                    "combined_score": score,
                    "constrained_score": score,
                    # Only the nondominated front is allowed to reproduce.
                    # Dominated but structurally valid programs remain in the
                    # database as measured negative memory and MAP-Elites
                    # evidence; no manually weighted metric gate is applied.
                    "parent_eligible": front_index == 0,
                    "negative_memory_only": front_index != 0,
                    "elite_gate_passed": front_index == 0,
                    "exploration_parent_only": False,
                }
            )
    database.rebuild_indexes()


# The four coordinates of the Pareto evidence order, each lower-is-better.
PARETO_COORDINATES = ("wirelength", "overflow", "wns", "tns")
# Evidence tiers from most to least faithful: routed, timing proxy, placement.
PARETO_TIER_PRECEDENCE = ("C", "B", "A")


def _pareto_evidence(metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Measured evidence for each Pareto coordinate, keyed by evidence tier.

    Wirelength and overflow come from the Tier A placement (HPWL, density
    overflow) and, once the candidate is routed, from Tier C (routed
    wirelength, routing overflow). WNS and TNS come from the admitted Tier B
    timing proxy and from Tier C post-route timing. Timing signs are flipped so
    that slack gains align with wirelength and overflow reductions.
    """

    def proxy(metric: str) -> float | None:
        admitted_key = f"timing_evidence_{metric}_delta"
        if admitted_key in metrics:
            return _finite_number(metrics.get(admitted_key))
        return _finite_number(metrics.get(f"timing_proxy_{metric}_delta"))

    def negated(value: float | None) -> float | None:
        return -value if value is not None else None

    tiers = {
        "wirelength": {
            "A": _finite_number(metrics.get("hpwl_delta_pct")),
            "C": _finite_number(metrics.get("routed_wirelength_delta_pct")),
        },
        "overflow": {
            "A": _finite_number(metrics.get("overflow_delta_pct")),
            "C": _finite_number(metrics.get("routed_overflow_delta_pct")),
        },
        "wns": {
            "B": negated(proxy("wns")),
            "C": negated(_finite_number(metrics.get("post_route_wns_gain_ns"))),
        },
        "tns": {
            "B": negated(proxy("tns")),
            "C": negated(_finite_number(metrics.get("post_route_tns_gain_ns"))),
        },
    }
    return {
        coordinate: {tier: value for tier, value in values.items() if value is not None}
        for coordinate, values in tiers.items()
    }


def _dominates(
    left: dict[str, dict[str, float]],
    right: dict[str, dict[str, float]],
) -> bool:
    """Pareto dominance over the coordinates measured for both candidates.

    A coordinate is compared at the most faithful tier both candidates
    received, so a routed measurement is never compared against a
    placement-stage one. Coordinates without a common tier are left out.
    """

    comparable = []
    for coordinate in PARETO_COORDINATES:
        left_tiers = left.get(coordinate, {})
        right_tiers = right.get(coordinate, {})
        for tier in PARETO_TIER_PRECEDENCE:
            if tier in left_tiers and tier in right_tiers:
                comparable.append((left_tiers[tier], right_tiers[tier]))
                break
    if not comparable:
        return False
    return all(left_value <= right_value for left_value, right_value in comparable) and any(
        left_value < right_value for left_value, right_value in comparable
    )


def _nondominated_fronts(
    programs: list[ObjectiveProgram],
    evidence: dict[str, dict[str, dict[str, float]]],
) -> list[list[ObjectiveProgram]]:
    remaining = list(programs)
    fronts: list[list[ObjectiveProgram]] = []
    while remaining:
        front = [
            candidate
            for candidate in remaining
            if not any(
                other.id != candidate.id
                and _dominates(evidence[other.id], evidence[candidate.id])
                for other in remaining
            )
        ]
        if not front:
            front = [remaining[0]]
        fronts.append(front)
        front_ids = {program.id for program in front}
        remaining = [program for program in remaining if program.id not in front_ids]
    return fronts


def _equal_design_ranks(programs: list[ObjectiveProgram]) -> dict[str, float]:
    dimensions: dict[str, dict[str, float]] = {}
    for program in programs:
        per_design = program.metrics.get("per_design_deltas") or []
        for row in per_design:
            if not isinstance(row, dict):
                continue
            design = str(row.get("design") or "unknown")
            for metric in ("hpwl_delta_pct", "overflow_delta_pct"):
                value = _finite_number(row.get(metric))
                if value is not None:
                    dimensions.setdefault(f"{design}:{metric}", {})[program.id] = value
        for metric in ("wns", "tns"):
            # Admitted proxy evidence and tie-level proxy evidence both order
            # candidates inside a front; only admitted evidence can dominate.
            keys = (f"timing_evidence_{metric}_delta", f"timing_tiebreak_{metric}_delta")
            if not any(key in program.metrics for key in keys):
                keys = (f"timing_proxy_{metric}_delta",)
            for key in keys:
                value = _finite_number(program.metrics.get(key))
                if value is not None:
                    dimensions.setdefault(key, {})[program.id] = -value
        for metric in (
            "routed_wirelength_delta_pct",
            "routed_overflow_delta_pct",
        ):
            value = _finite_number(program.metrics.get(metric))
            if value is not None:
                dimensions.setdefault(metric, {})[program.id] = value
        for metric in ("post_route_wns_gain_ns", "post_route_tns_gain_ns"):
            value = _finite_number(program.metrics.get(metric))
            if value is not None:
                dimensions.setdefault(metric, {})[program.id] = -value

    totals = {program.id: 0.0 for program in programs}
    counts = {program.id: 0 for program in programs}
    for values in dimensions.values():
        ordered = sorted(values.items(), key=lambda item: item[1])
        cursor = 0
        while cursor < len(ordered):
            end = cursor + 1
            while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
                end += 1
            average_rank = ((cursor + 1) + end) / 2.0
            for program_id, _value in ordered[cursor:end]:
                totals[program_id] += average_rank
                counts[program_id] += 1
            cursor = end
    return {
        program.id: totals[program.id] / max(counts[program.id], 1)
        for program in programs
    }


def _timing_admission_path(run_root: str | Path) -> Path:
    return Path(run_root) / "timing_proxy_audit" / "admission.json"


def _load_timing_admission(
    config: OpenEvolveTier2Config,
    run_root: str | Path,
) -> dict[str, Any]:
    """Current timing-proxy admission: the latest audit, else the configured mode."""

    path = _timing_admission_path(run_root)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("authority"):
            return payload
    return default_timing_proxy_admission(
        config.timing_proxy_mode if config.timing_proxy_enabled else "diagnostic"
    )


def _timing_audit_source(
    database: ObjectiveProgramDatabase,
) -> tuple[ObjectiveProgram, str, Path] | None:
    """Best measured program whose placements can be perturbed by the audit."""

    best = database.get(database.best_program_id)
    ordered = ([best] if best is not None else []) + sorted(
        database.programs.values(),
        key=lambda program: (program.combined_score, program.id),
        reverse=True,
    )
    for program in ordered:
        if program.status != "accepted" or program.metrics.get("structural_failure_count"):
            continue
        summary = program.artifacts.get("tier2_summary")
        comparison_csv = summary.get("comparison_csv") if isinstance(summary, dict) else None
        objective_id = str(
            program.artifacts.get("objective_id_in_tier2")
            or (program.objective_spec or {}).get("id")
            or ""
        )
        if objective_id and comparison_csv and Path(str(comparison_csv)).is_file():
            return program, objective_id, Path(str(comparison_csv))
    return None


def _timing_downstream_pairs(
    database: ObjectiveProgramDatabase,
) -> dict[str, list[tuple[float, float]]]:
    """Pair each routed candidate's proxy movement with its post-route movement."""

    pairs: dict[str, list[tuple[float, float]]] = {"wns": [], "tns": []}
    for program in database.programs.values():
        metrics = program.metrics
        if metrics.get("migrant"):
            continue
        proxy_rows = metrics.get("timing_proxy_per_design_deltas")
        routed_rows = metrics.get("per_design_routed_evidence")
        if not isinstance(proxy_rows, list) or not isinstance(routed_rows, list):
            continue
        for proxy_row in proxy_rows:
            if not isinstance(proxy_row, dict):
                continue
            design_rows = [
                row
                for row in routed_rows
                if isinstance(row, dict) and row.get("design") == proxy_row.get("design")
            ]
            for metric in ("wns", "tns"):
                proxy_value = _finite_number(proxy_row.get(f"{metric}_delta"))
                routed_value = _mean_finite(row.get(f"{metric}_gain_ns") for row in design_rows)
                if proxy_value is not None and routed_value is not None:
                    pairs[metric].append((proxy_value, routed_value))
    return pairs


def _run_scheduled_timing_proxy_audit(
    *,
    config: OpenEvolveTier2Config,
    database: ObjectiveProgramDatabase,
    iteration: int,
    run_root: Path,
    dreamplace_root: str | Path,
) -> dict[str, Any] | None:
    """Audit the timing proxy and refresh how it enters archive selection.

    Each timing-panel design is audited separately: controlled coordinate
    perturbations of the native and the current best placement measure how far
    proxy WNS/TNS movement merely restates HPWL movement, and routed candidates
    show whether it follows post-route movement. The resulting per-design,
    per-metric outcome decides whether the proxy can dominate (admit), only
    order close candidates (tie), or is left out (reject).
    """

    interval = int(config.timing_proxy_audit_interval)
    if (
        not config.timing_proxy_enabled
        or not config.timing_proxy_panel
        or interval <= 0
        or iteration % interval != 0
    ):
        return None
    audit_root = Path(run_root) / "timing_proxy_audit" / f"iteration_{iteration:04d}"
    source = _timing_audit_source(database)
    timing_panel = load_timing_proxy_panel(config.timing_proxy_panel)
    by_design: dict[str, dict[str, dict[str, Any]]] = {}
    reports = []
    if source is not None:
        program, objective_id, comparison_csv = source
        for design in timing_panel.designs:
            design_root = audit_root / _safe_name(design.name)
            placements_path = placements_from_tier2_comparison(
                comparison_csv,
                design_root / "placements.json",
                objective_ids={objective_id},
                design_names={design.name},
                include_default=True,
                include_custom_default=False,
                minimum_source_iterations=config.minimum_scoring_iterations,
            )
            if _placement_count(placements_path) == 0:
                reports.append({"design": design.name, "status": "no_placements"})
                continue
            try:
                summary = run_timing_proxy_audit(
                    design=design.name,
                    base_config=design.base_config,
                    timing_panel_path=config.timing_proxy_panel,
                    placements_path=placements_path,
                    dreamplace_root=dreamplace_root,
                    run_dir=design_root,
                    perturbations=config.timing_proxy_audit_perturbations,
                    timeout_seconds=timing_panel.timeout_seconds,
                    gpu=timing_panel.gpu,
                    seed=config.seed,
                    gate_threshold=config.timing_proxy_max_hpwl_corr_for_gate,
                    tiebreaker_threshold=config.timing_proxy_max_hpwl_corr_for_tiebreaker,
                    min_pairs=config.timing_proxy_audit_min_pairs,
                )
            except Exception as exc:
                reports.append(
                    {
                        "design": design.name,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            by_design[design.name] = summary["metric_admission"]
            reports.append(
                {
                    "design": design.name,
                    "status": "completed",
                    "correlations": summary.get("correlations"),
                    "audit_summary": str(design_root / "summary.json"),
                }
            )
    if not by_design:
        _write_json(
            {"iteration": iteration, "status": "failed", "designs": reports},
            audit_root / "audit_report.json",
        )
        if iteration == 0 and config.timing_proxy_audit_required:
            raise RuntimeError(
                "timing_proxy_audit_required is true, but the initial timing-proxy "
                "audit produced no per-design result. Check the seed placements and "
                "timing collateral before running objective evolution."
            )
        # A later audit that cannot run keeps the previous admission in force.
        return None

    admission = cap_unstable_timing_proxy_admission(
        summarize_timing_proxy_admission(by_design),
        timing_proxy_downstream_agreement(
            _timing_downstream_pairs(database),
            min_pairs=config.timing_proxy_audit_min_pairs,
        ),
    )
    admission.update(
        {
            "iteration": iteration,
            "source_program_id": source[0].id if source is not None else None,
            "gate_threshold": config.timing_proxy_max_hpwl_corr_for_gate,
            "tiebreaker_threshold": config.timing_proxy_max_hpwl_corr_for_tiebreaker,
            "designs": reports,
        }
    )
    _write_json(admission, audit_root / "audit_report.json")
    _write_json(admission, _timing_admission_path(run_root))
    return admission


def _apply_manuscript_multimetric_selection(
    metrics: dict[str, Any],
    config: OpenEvolveTier2Config,
    *,
    allow_seed_baseline: bool,
) -> None:
    """Apply the manuscript's admitted placement-and-timing selection rule.

    The legacy selector required HPWL and overflow improvements before timing
    could affect a candidate.  That ordering makes an admitted timing proxy a
    weak after-the-fact adjustment.  The manuscript policy instead treats
    converged placement, finite gradients, and bounded placement regressions as
    safeguards, then scores HPWL, overflow, WNS, and TNS together.
    """

    if config.selection_policy == "legacy_hpwl_overflow":
        return
    if config.selection_policy == "pareto_multiobjective":
        _prepare_pareto_candidate(
            metrics,
            config,
            allow_seed_baseline=allow_seed_baseline,
        )
        return
    if config.selection_policy != "manuscript_multiobjective":
        raise ValueError(
            "selection_policy must be 'legacy_hpwl_overflow' or "
            "'manuscript_multiobjective' or 'pareto_multiobjective'"
        )

    metrics.setdefault(
        "legacy_hpwl_overflow_parent_eligible",
        bool(metrics.get("parent_eligible")),
    )
    metrics.setdefault(
        "legacy_hpwl_gate_passed",
        bool(metrics.get("hpwl_gate_passed")),
    )
    metrics.setdefault(
        "legacy_overflow_gate_passed",
        bool(metrics.get("overflow_gate_passed")),
    )
    metrics["selection_policy"] = config.selection_policy
    metrics["manuscript_multimetric_selection"] = True

    hpwl = _finite_number(metrics.get("hpwl_delta_pct"))
    overflow = _finite_number(metrics.get("overflow_delta_pct"))
    worst_hpwl = _finite_number(metrics.get("worst_design_hpwl_delta_pct"))
    worst_overflow = _finite_number(metrics.get("worst_design_overflow_delta_pct"))
    if worst_hpwl is None:
        worst_hpwl = hpwl
    if worst_overflow is None:
        worst_overflow = overflow

    wns_delta = _finite_number(metrics.get("timing_proxy_wns_delta"))
    tns_delta = _finite_number(metrics.get("timing_proxy_tns_delta"))
    tns_delta_pct = _finite_number(metrics.get("timing_proxy_tns_delta_pct"))
    timing_status = str(metrics.get("timing_proxy_status") or "unavailable")
    timing_mode = str(metrics.get("timing_proxy_mode") or config.timing_proxy_mode)
    source_budget_satisfied = bool(
        metrics.get("timing_proxy_source_placement_budget_satisfied")
    )
    timing_metrics_available = wns_delta is not None or tns_delta is not None
    timing_ready = (
        timing_mode in {"gate", "tie_breaker"}
        and timing_status in {"success", "partial"}
        and source_budget_satisfied
        and timing_metrics_available
    )

    wns_regressed = (
        wns_delta is not None
        and wns_delta < -abs(config.timing_proxy_wns_regression_gate_ns)
    )
    tns_regressed = (
        tns_delta is not None
        and tns_delta_pct is not None
        and tns_delta < -abs(config.timing_proxy_tns_regression_min_abs_ns)
        and tns_delta_pct < -abs(config.timing_proxy_tns_regression_pct_gate)
    )
    timing_nonregressed = not (wns_regressed or tns_regressed)
    wns_improved = (
        wns_delta is not None
        and wns_delta >= abs(config.timing_proxy_wns_delta_min_abs_ns)
    )
    tns_improved = (
        tns_delta is not None
        and tns_delta >= abs(config.timing_proxy_tns_regression_min_abs_ns)
        and (tns_delta_pct is None or tns_delta_pct > 0.0)
    )
    timing_improved = wns_improved or tns_improved

    hpwl_bounded = (
        hpwl is not None
        and worst_hpwl is not None
        and hpwl <= config.max_parent_hpwl_regression_pct
        and worst_hpwl <= config.max_parent_hpwl_regression_pct
    )
    overflow_bounded = (
        overflow is not None
        and worst_overflow is not None
        and overflow <= config.max_parent_overflow_regression_pct
        and worst_overflow <= config.max_parent_overflow_regression_pct
    )
    placement_bounded = hpwl_bounded and overflow_bounded
    placement_effect_floor = max(config.min_baseline_portfolio_effect_pct, 0.0)
    placement_improved = bool(
        hpwl is not None
        and hpwl <= -placement_effect_floor
        or overflow is not None
        and overflow <= -placement_effect_floor
    )
    measured_improvement = placement_improved or timing_improved

    structural_ok = not bool(metrics.get("structural_failure_count"))
    routing_ok = allow_seed_baseline or bool(metrics.get("routing_term_gate_passed"))
    portfolio_ok = (
        not config.require_baseline_portfolio_gate
        or bool(metrics.get("beats_baseline_portfolio"))
    )
    custom_default_ok = (
        not config.require_custom_default_gate
        or bool(metrics.get("custom_default_gate_passed"))
    )
    timing_required = config.require_timing_for_generated_parents and not allow_seed_baseline
    timing_admissible = (
        (timing_ready and timing_nonregressed)
        if timing_required
        else (not timing_ready or timing_nonregressed)
    )
    admissible = (
        structural_ok
        and routing_ok
        and portfolio_ok
        and custom_default_ok
        and placement_bounded
        and timing_admissible
    )
    parent_eligible = admissible and (allow_seed_baseline or measured_improvement)

    quality_values: list[tuple[float, float]] = []
    if hpwl is not None and config.selection_hpwl_weight > 0.0:
        quality_values.append(
            (
                config.selection_hpwl_weight,
                math.tanh(
                    -hpwl / max(abs(config.max_parent_hpwl_regression_pct), 1e-12)
                ),
            )
        )
    if overflow is not None and config.selection_overflow_weight > 0.0:
        quality_values.append(
            (
                config.selection_overflow_weight,
                math.tanh(
                    -overflow
                    / max(abs(config.max_parent_overflow_regression_pct), 1e-12)
                ),
            )
        )
    if (
        wns_delta is not None
        and abs(wns_delta) >= abs(config.timing_proxy_wns_delta_min_abs_ns)
        and config.selection_wns_weight > 0.0
    ):
        quality_values.append(
            (
                config.selection_wns_weight,
                math.tanh(
                    wns_delta
                    / max(abs(config.timing_proxy_wns_regression_gate_ns), 1e-12)
                ),
            )
        )
    if (
        tns_delta is not None
        and abs(tns_delta) >= abs(config.timing_proxy_tns_regression_min_abs_ns)
        and tns_delta_pct is not None
        and config.selection_tns_weight > 0.0
    ):
        quality_values.append(
            (
                config.selection_tns_weight,
                math.tanh(
                    tns_delta_pct
                    / max(abs(config.timing_proxy_tns_regression_pct_gate), 1e-12)
                ),
            )
        )
    weight_sum = sum(weight for weight, _value in quality_values)
    normalized_quality = (
        sum(weight * value for weight, value in quality_values) / weight_sum
        if weight_sum > 0.0
        else -1.0
    )
    raw_score = max(0.0, 1.0 + normalized_quality)
    score = raw_score if parent_eligible else 0.0
    elite_gate_passed = bool(
        parent_eligible
        and (
            not config.require_timing_improvement_for_elite
            or timing_improved
        )
    )

    metrics.update(
        {
            "manuscript_multimetric_score_raw": raw_score,
            "manuscript_multimetric_quality": normalized_quality,
            "manuscript_hpwl_safeguard_passed": hpwl_bounded,
            "manuscript_overflow_safeguard_passed": overflow_bounded,
            "manuscript_timing_ready": timing_ready,
            "manuscript_timing_nonregressed": timing_nonregressed,
            "manuscript_wns_improved": wns_improved,
            "manuscript_tns_improved": tns_improved,
            "manuscript_timing_improved": timing_improved,
            "manuscript_measured_improvement": measured_improvement,
            "combined_score": score,
            "constrained_score": score,
            # Under this policy these fields mean bounded-regression safeguards,
            # not mandatory improvement on both placement metrics.
            "hpwl_gate_passed": hpwl_bounded,
            "overflow_gate_passed": overflow_bounded,
            "robust_gate_passed": placement_bounded,
            "elite_gate_passed": elite_gate_passed,
            "parent_eligible": parent_eligible,
            "negative_memory_only": not parent_eligible,
            "exploration_parent_only": bool(parent_eligible and not elite_gate_passed),
        }
    )

    if parent_eligible:
        metrics["feedback_lesson"] = (
            "Measured manuscript-policy feedback: the converged placement is "
            "within HPWL/overflow safeguards and its admitted OpenTimer WNS/TNS "
            "feedback participates directly in the multi-metric score."
        )
    elif not structural_ok:
        metrics["feedback_lesson"] = (
            "Measured manuscript-policy feedback: structural placement failure; "
            "retain only as negative memory."
        )
    elif not routing_ok and metrics.get("measured_routing_influence_required"):
        metrics["feedback_lesson"] = (
            "Measured manuscript-policy feedback: the routing mechanism was "
            "syntactically present but its logged gradient contribution was too "
            "small or unavailable. Avoid tiny coefficients and saturated "
            "transforms; this candidate is negative memory only."
        )
    elif not placement_bounded:
        metrics["feedback_lesson"] = (
            "Measured manuscript-policy feedback: HPWL or overflow exceeded the "
            "catastrophic safeguard. Timing cannot rescue an invalid placement."
        )
    elif timing_required and not timing_ready:
        metrics["feedback_lesson"] = (
            "Measured manuscript-policy feedback: admitted OpenTimer WNS/TNS was "
            "not available from a full-budget placement, so this generated "
            "candidate cannot become a parent."
        )
    elif not timing_nonregressed:
        metrics["feedback_lesson"] = (
            "Measured manuscript-policy feedback: OpenTimer WNS/TNS regressed "
            "beyond the audited noise-aware gate."
        )
    else:
        metrics["feedback_lesson"] = (
            "Measured manuscript-policy feedback: no material improvement was "
            "observed in HPWL, overflow, WNS, or TNS."
        )


def _apply_routing_gate(
    metrics: dict[str, Any],
    spec: ObjectiveSpec,
    config: OpenEvolveTier2Config,
    *,
    allow_nonrouting_parent: bool,
) -> None:
    routing_status = _routing_term_status(spec.ast, config)
    routing_gate_passed = routing_status["routing_term_gate_passed"]
    aggregate_winner = (
        not bool(metrics.get("structural_failure_count"))
        and bool(metrics.get("hpwl_gate_passed"))
        and bool(metrics.get("overflow_gate_passed"))
    )
    all_design_winner = aggregate_winner and bool(metrics.get("robust_gate_passed"))
    metrics.update(
        {
            **routing_status,
            "aggregate_tier2_winner": aggregate_winner,
            "all_design_robust_tier2_winner": all_design_winner,
        }
    )
    if not config.require_nonzero_routing_term:
        return
    if routing_gate_passed:
        if (
            not allow_nonrouting_parent
            and config.require_custom_default_gate
            and not bool(metrics.get("custom_default_gate_passed"))
        ):
            metrics["combined_score"] = 0.0
            metrics["constrained_score"] = 0.0
            metrics["elite_gate_passed"] = False
            metrics["parent_eligible"] = False
            metrics["negative_memory_only"] = True
            metrics["custom_default_gate_note"] = (
                "Generated routing-aware objective did not beat the native-identity "
                "custom_default control under the configured evaluator-side "
                "comparison gates."
            )
            metrics["feedback_lesson"] = (
                "This objective is not promoted because its measured DREAMPlace "
                "behavior is too close to, or worse than, the native-identity "
                "custom_default control. Keep the program nontrivial, but the "
                "routing mechanism must produce a measurable placement effect "
                "without becoming a near-default perturbation."
            )
        return
    if allow_nonrouting_parent:
        metrics["routing_gate_note"] = (
            "Non-routing seed baseline is allowed as a parent, but it is not a "
            "routing-aware Tier-2 candidate."
        )
        return
    metrics["combined_score"] = 0.0
    metrics["constrained_score"] = 0.0
    metrics["elite_gate_passed"] = False
    metrics["parent_eligible"] = False
    metrics["negative_memory_only"] = True
    if config.objective_mode == "native_residual":
        metrics["feedback_lesson"] = (
            "This objective did not contain a nonzero active routing term, so it "
            "cannot be promoted in the native-residual routing-aware run. Keep "
            "native_objective as the base and add a small nonzero soft_rudy_mean, "
            "soft_rudy_pnorm, or pin_density_pnorm correction."
        )
    else:
        metrics["feedback_lesson"] = (
            "This objective did not contain a nonzero active routing term, so it "
            "cannot be promoted in the routing-aware run. Use explicit wirelength "
            "and density terms, and add a material soft_rudy_mean, soft_rudy_pnorm, "
            "route_pressure_long, pin_density_pnorm, or pin_count_weighted_wl mechanism."
        )


def _routing_term_status(spec_ast: dict[str, Any], config: OpenEvolveTier2Config) -> dict[str, Any]:
    coefficients, unsupported = _route_coefficients(spec_ast)
    active_coefficients = [
        {"term": term, "coefficient": coeff}
        for term, coeff in coefficients
        if abs(float(coeff)) > config.min_abs_routing_coeff
    ]
    active_terms = sorted({item["term"] for item in active_coefficients})
    return {
        "routing_term_required": config.require_nonzero_routing_term,
        "routing_term_gate_passed": bool(active_terms) and not unsupported,
        "routing_aware_tier2_candidate": bool(active_terms) and not unsupported,
        "active_routing_terms": active_terms,
        "active_routing_coefficients": active_coefficients,
        "unsupported_routing_terms": sorted(unsupported),
    }


def _apply_measured_routing_influence_gate(
    metrics: dict[str, Any],
    config: OpenEvolveTier2Config,
    *,
    allow_seed_baseline: bool,
) -> None:
    """Require generated routing structure to influence the optimizer gradient.

    Static term presence is insufficient: a tiny coefficient or saturated
    nonlinear transform can make an apparently routing-aware formula numerically
    identical to a wirelength/density baseline. DREAMPlace logs the calibrated
    routing component's gradient norm, which is normalized here by the total
    objective gradient at the same calls.
    """

    metrics["measured_routing_influence_required"] = bool(
        config.require_measured_routing_influence and not allow_seed_baseline
    )
    if allow_seed_baseline or not config.require_measured_routing_influence:
        metrics["measured_routing_influence_gate_passed"] = True
        return

    component_summary = metrics.get("component_summary")
    component = (
        component_summary.get("routing_correction", {})
        if isinstance(component_summary, dict)
        else {}
    )
    sample_count = int(component.get("grad_ratio_count") or 0)
    mean_ratio = _finite_number(component.get("grad_ratio_mean"))
    peak_ratio = _finite_number(component.get("grad_ratio_max"))
    gate_passed = bool(
        sample_count >= config.min_routing_component_samples
        and mean_ratio is not None
        and peak_ratio is not None
        and mean_ratio >= config.min_routing_gradient_ratio_mean
        and peak_ratio >= config.min_routing_gradient_ratio_peak
    )
    metrics.update(
        {
            "measured_routing_influence_gate_passed": gate_passed,
            "routing_influence_component": "routing_correction",
            "routing_influence_sample_count": sample_count,
            "routing_influence_gradient_ratio_mean": mean_ratio,
            "routing_influence_gradient_ratio_peak": peak_ratio,
            "routing_influence_min_samples": config.min_routing_component_samples,
            "routing_influence_min_gradient_ratio_mean": (
                config.min_routing_gradient_ratio_mean
            ),
            "routing_influence_min_gradient_ratio_peak": (
                config.min_routing_gradient_ratio_peak
            ),
        }
    )
    if gate_passed:
        return

    metrics.update(
        {
            "routing_term_gate_passed": False,
            "routing_aware_tier2_candidate": False,
            "combined_score": 0.0,
            "constrained_score": 0.0,
            "elite_gate_passed": False,
            "parent_eligible": False,
            "negative_memory_only": True,
            "feedback_lesson": (
                "The generated routing mechanism was present syntactically but "
                "did not exert a measured nontrivial gradient contribution during "
                "DREAMPlace. Avoid tiny coefficients and saturated transforms; "
                "use the routing-correction component trajectory and gradient "
                "ratios to redesign the mechanism."
            ),
        }
    )


def _run_final_panel(
    *,
    config: OpenEvolveTier2Config,
    database: ObjectiveProgramDatabase,
    run_root: Path,
    dreamplace_root: str | Path,
    resume: bool,
    retry_failed: bool,
) -> dict[str, Any] | None:
    final_candidates, selection_policy = _final_candidate_programs(database, config)
    if not final_candidates:
        return None
    objectives_dir = run_root / "final_candidates"
    objectives_dir.mkdir(parents=True, exist_ok=True)
    objective_paths = []
    for program in final_candidates:
        path = objectives_dir / f"{program.id}.json"
        _write_json(program.objective_spec, path)
        objective_paths.append(str(path))
    baseline_paths = _write_baseline_presets(config.final_baseline_presets, run_root / "final_baselines")
    panel_path = write_dreamplace_panel(
        load_shared_panel(config.final_panel),
        run_root / "final_tier2" / "final_panel.toml",
    )
    summary = run_tier2_dreamplace(
        panel_path=panel_path,
        objective_paths=baseline_paths + objective_paths,
        dreamplace_root=dreamplace_root,
        run_dir=run_root / "final_tier2",
        store_path=run_root / "final_tier2" / "tier2.sqlite",
        resume=resume,
        retry_failed=retry_failed,
        include_default=True,
        include_custom_default=True,
        require_output_artifact=True,
        require_def_output=True,
        disable_legalization=config.final_disable_legalization,
        objective_semantics=(
            "raw" if config.objective_mode == "controller" else None
        ),
        timing_controller_panel=config.timing_controller_panel,
    )
    summary["status"] = "completed"
    summary["selection_policy"] = selection_policy
    summary["selected_programs"] = [
        {
            "program_id": program.id,
            "objective_id": (program.objective_spec or {}).get("id"),
            "search_metrics": program.metrics,
        }
        for program in final_candidates
    ]
    return summary


def _final_candidate_programs(
    database: ObjectiveProgramDatabase,
    config: OpenEvolveTier2Config,
) -> tuple[list[ObjectiveProgram], str]:
    top_k = max(0, config.final_top_k)
    if top_k <= 0:
        return [], "final candidate selection disabled because top_k <= 0"

    candidates = [
        program
        for program in database.programs.values()
        if program.objective_spec
        and not program.metrics.get("seed_baseline")
        and program.metrics.get("routing_aware_tier2_candidate")
        and not program.metrics.get("structural_failure_count")
        and not program.metrics.get("severe_regression_count")
    ]
    candidates.sort(key=_generalization_candidate_key)
    elites = [program for program in candidates if program.is_elite_eligible]
    if elites:
        return elites[:top_k], "final panel receives top search-panel elite candidates"
    aggregate = [
        program
        for program in candidates
        if program.metrics.get("aggregate_tier2_winner")
        or program.metrics.get("hpwl_gate_passed")
        or program.metrics.get("overflow_gate_passed")
    ]
    return (
        aggregate[:top_k],
        "final panel receives top aggregate search-panel candidates because no robust elite was available",
    )


def _run_generalization_panel(
    *,
    config: OpenEvolveTier2Config,
    database: ObjectiveProgramDatabase,
    run_root: Path,
    dreamplace_root: str | Path,
    resume: bool,
    retry_failed: bool,
) -> dict[str, Any] | None:
    candidates, selection_policy = _generalization_candidate_programs(database, config)
    if not candidates:
        summary = {
            "status": "skipped",
            "reason": "no generated candidates are eligible for held-out generalization",
            "selection_policy": selection_policy,
        }
        _write_json(summary, run_root / "generalization_tier2" / "summary.json")
        return summary

    objectives_dir = run_root / "generalization_candidates"
    objectives_dir.mkdir(parents=True, exist_ok=True)
    objective_paths = []
    for program in candidates:
        path = objectives_dir / f"{program.id}.json"
        _write_json(program.objective_spec, path)
        objective_paths.append(str(path))

    baseline_paths = _write_baseline_presets(
        config.generalization_baseline_presets,
        run_root / "generalization_baselines",
    )
    panel_path = write_dreamplace_panel(
        load_shared_panel(config.generalization_panel),
        run_root / "generalization_tier2" / "generalization_panel.toml",
    )
    summary = run_tier2_dreamplace(
        panel_path=panel_path,
        objective_paths=baseline_paths + objective_paths,
        dreamplace_root=dreamplace_root,
        run_dir=run_root / "generalization_tier2",
        store_path=run_root / "generalization_tier2" / "tier2.sqlite",
        resume=resume,
        retry_failed=retry_failed,
        include_default=True,
        include_custom_default=True,
        require_output_artifact=True,
        require_def_output=True,
        disable_legalization=False,
        objective_semantics=(
            "raw" if config.objective_mode == "controller" else None
        ),
        timing_controller_panel=config.timing_controller_panel,
    )
    summary["status"] = "completed"
    summary["selection_policy"] = selection_policy
    summary["held_out_from_evolution"] = True
    summary["selected_programs"] = [
        {
            "program_id": program.id,
            "objective_id": (program.objective_spec or {}).get("id"),
            "search_metrics": program.metrics,
        }
        for program in candidates
    ]
    _write_json(summary, run_root / "generalization_tier2" / "summary.json")
    (run_root / "generalization_tier2" / "generalization_report.md").write_text(
        _build_generalization_report(summary),
        encoding="utf-8",
    )
    return summary


def _generalization_candidate_programs(
    database: ObjectiveProgramDatabase,
    config: OpenEvolveTier2Config,
) -> tuple[list[ObjectiveProgram], str]:
    top_k = max(0, config.generalization_top_k)
    if top_k <= 0:
        return [], "generalization candidate selection disabled because top_k <= 0"

    candidates = [
        program
        for program in database.programs.values()
        if program.objective_spec
        and not program.metrics.get("seed_baseline")
        and program.metrics.get("routing_aware_tier2_candidate")
        and not program.metrics.get("structural_failure_count")
        and not program.metrics.get("severe_regression_count")
    ]
    candidates.sort(key=_generalization_candidate_key)

    elites = [program for program in candidates if program.is_elite_eligible]
    if elites or config.generalization_candidate_policy == "elite_only":
        return (
            elites[:top_k],
            "held-out panel receives top search-panel elite candidates only",
        )
    aggregate = [
        program
        for program in candidates
        if program.metrics.get("aggregate_tier2_winner")
        or program.metrics.get("hpwl_gate_passed")
        or program.metrics.get("overflow_gate_passed")
    ]
    return (
        aggregate[:top_k],
        "held-out panel receives top aggregate search-panel candidates because no robust elite was available",
    )


def _generalization_candidate_key(program: ObjectiveProgram) -> tuple[float, float, float, float, str]:
    def metric(name: str, default: float) -> float:
        value = _finite_number(program.metrics.get(name))
        return value if value is not None else default

    return (
        metric("tier2_average_rank", 999.0),
        metric("worst_design_hpwl_delta_pct", 999.0),
        metric("worst_design_overflow_delta_pct", 999.0),
        -metric("combined_score", 0.0),
        program.id,
    )


def _build_generalization_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Held-Out DREAMPlace Generalization Report",
        "",
        "These results are produced after the evolution loop. They are not used for parent selection or prompt feedback.",
        "",
        f"- Status: {summary.get('status')}",
        f"- Selection policy: {summary.get('selection_policy')}",
        f"- Comparison CSV: {summary.get('comparison_csv')}",
        f"- Metrics CSV: {summary.get('metrics_csv')}",
        "",
        "## Selected Search Candidates",
    ]
    for item in summary.get("selected_programs", []):
        metrics = item.get("search_metrics", {})
        lines.append(
            "- "
            f"{item.get('program_id')} / {item.get('objective_id')}: "
            f"search_hpwl_delta={metrics.get('hpwl_delta_pct')} "
            f"search_overflow_delta={metrics.get('overflow_delta_pct')} "
            f"search_score={metrics.get('combined_score')}"
        )
    return "\n".join(lines) + "\n"


def _build_train_vs_heldout_report(
    summary: dict[str, Any],
    database: ObjectiveProgramDatabase,
) -> str:
    final_summary = summary.get("final_tier2") or {}
    generalization_summary = summary.get("generalization_tier2") or {}
    train_rows = _safe_read_summary_csv(final_summary, "comparison_csv")
    heldout_rows = _safe_read_summary_csv(generalization_summary, "comparison_csv")
    train_designs = sorted({str(row.get("design")) for row in train_rows if row.get("design")})
    heldout_designs = sorted(
        {str(row.get("design")) for row in heldout_rows if row.get("design")}
    )
    selected_program_ids = _selected_generalization_program_ids(generalization_summary)
    selected_objective_ids = _selected_generalization_objective_ids(generalization_summary)

    lines = [
        "# Train-Vs-Heldout Transfer Report",
        "",
        "This report is generated after the evolution loop. Heldout rows are not "
        "used for parent selection, prompt feedback, or candidate scoring.",
        "",
        "## Split",
        "",
        f"- Training designs: {', '.join(train_designs) if train_designs else 'not evaluated'}",
        f"- Heldout designs: {', '.join(heldout_designs) if heldout_designs else 'not evaluated'}",
        f"- Config: {summary.get('config')}",
        f"- Provider metadata: `{json.dumps(summary.get('provider_metadata') or {}, sort_keys=True)}`",
        "",
        "## Selected Generated Objectives",
        "",
    ]
    if not selected_program_ids:
        lines.append("- No generated objective was selected for heldout generalization.")
    else:
        for program_id in selected_program_ids:
            program = database.get(program_id)
            if not program:
                lines.append(f"- {program_id}: program record not found in database.")
                continue
            objective_id = (
                (program.objective_spec or {}).get("id")
                if isinstance(program.objective_spec, dict)
                else None
            )
            metrics = program.metrics or {}
            lines.extend(
                [
                    f"- Program: `{program.id}` objective=`{objective_id}` "
                    f"search_score={metrics.get('combined_score')} "
                    f"search_hpwl_delta={metrics.get('hpwl_delta_pct')} "
                    f"search_overflow_delta={metrics.get('overflow_delta_pct')}",
                    "",
                    "```python",
                    program.code.strip(),
                    "```",
                    "",
                ]
            )

    lines.extend(
        [
            "## Metric Summary",
            "",
            "| Objective | Train Designs | Train HPWL Delta % | Train Overflow Delta % | Heldout Designs | Heldout HPWL Delta % | Heldout Overflow Delta % | Heldout Status |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    objective_ids = _report_objective_order(train_rows, heldout_rows, selected_objective_ids)
    for objective_id in objective_ids:
        train_summary = _summarize_objective_rows(train_rows, objective_id)
        heldout_summary = _summarize_objective_rows(heldout_rows, objective_id)
        lines.append(
            "| "
            f"`{objective_id}` | "
            f"{train_summary['design_count']} | "
            f"{_fmt(train_summary['mean_hpwl_delta_pct'])} | "
            f"{_fmt(train_summary['mean_overflow_delta_pct'])} | "
            f"{heldout_summary['design_count']} | "
            f"{_fmt(heldout_summary['mean_hpwl_delta_pct'])} | "
            f"{_fmt(heldout_summary['mean_overflow_delta_pct'])} | "
            f"{heldout_summary['status']} |"
        )

    lines.extend(
        [
            "",
            "## Transfer Interpretation",
            "",
            _transfer_interpretation(heldout_rows, selected_objective_ids),
            "",
            "## Evidence Files",
            "",
            f"- Final training comparison CSV: {final_summary.get('comparison_csv')}",
            f"- Heldout comparison CSV: {generalization_summary.get('comparison_csv')}",
            f"- Heldout generalization report: {generalization_summary.get('report') or 'see generalization_tier2/generalization_report.md'}",
        ]
    )
    return "\n".join(lines) + "\n"


def _safe_read_summary_csv(summary: dict[str, Any], key: str) -> list[dict[str, Any]]:
    path = summary.get(key) if isinstance(summary, dict) else None
    if not path:
        return []
    candidate = Path(str(path))
    if not candidate.is_file():
        return []
    return _read_csv(candidate)


def _selected_generalization_program_ids(summary: dict[str, Any]) -> list[str]:
    selected: list[str] = []
    for item in (summary or {}).get("selected_programs", []) or []:
        program_id = item.get("program_id")
        if program_id and program_id not in selected:
            selected.append(str(program_id))
    return selected


def _selected_generalization_objective_ids(summary: dict[str, Any]) -> list[str]:
    selected: list[str] = []
    for item in (summary or {}).get("selected_programs", []) or []:
        objective_id = item.get("objective_id")
        if objective_id and objective_id not in selected:
            selected.append(str(objective_id))
    return selected


def _report_objective_order(
    train_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    selected_objective_ids: list[str],
) -> list[str]:
    objective_ids: list[str] = []
    for objective_id in ["default", "custom_default"]:
        if any(row.get("objective_id") == objective_id for row in train_rows + heldout_rows):
            objective_ids.append(objective_id)
    for row in train_rows + heldout_rows:
        objective_id = str(row.get("objective_id") or "")
        if objective_id and objective_id not in objective_ids:
            objective_ids.append(objective_id)
    for objective_id in selected_objective_ids:
        if objective_id not in objective_ids:
            objective_ids.append(objective_id)
    return objective_ids


def _summarize_objective_rows(
    rows: list[dict[str, Any]],
    objective_id: str,
) -> dict[str, Any]:
    selected = [row for row in rows if str(row.get("objective_id")) == objective_id]
    hpwl = [_finite_number(row.get("hpwl_delta_pct")) for row in selected]
    overflow = [_finite_number(row.get("overflow_delta_pct")) for row in selected]
    statuses = [str(row.get("status") or "") for row in selected]
    success_count = sum(1 for status in statuses if status == "success")
    design_count = len({str(row.get("design")) for row in selected if row.get("design")})
    return {
        "design_count": design_count,
        "mean_hpwl_delta_pct": _mean_finite(value for value in hpwl if value is not None),
        "mean_overflow_delta_pct": _mean_finite(
            value for value in overflow if value is not None
        ),
        "status": (
            "not_evaluated"
            if not selected
            else "success"
            if success_count == len(selected)
            else f"partial_success_{success_count}/{len(selected)}"
        ),
    }


def _transfer_interpretation(
    heldout_rows: list[dict[str, Any]],
    selected_program_ids: list[str],
) -> str:
    if not selected_program_ids:
        return (
            "No generated objective reached the heldout panel, so transfer cannot "
            "be claimed for this run."
        )
    messages = []
    for objective_id in selected_program_ids:
        summary = _summarize_objective_rows(heldout_rows, objective_id)
        hpwl = summary.get("mean_hpwl_delta_pct")
        overflow = summary.get("mean_overflow_delta_pct")
        if hpwl is None or overflow is None:
            messages.append(f"`{objective_id}` has incomplete heldout metrics.")
            continue
        if hpwl <= 10.0 and (hpwl < 0.0 or overflow < 0.0):
            messages.append(
                f"`{objective_id}` shows heldout transfer signal: "
                f"hpwl_delta={_fmt(hpwl)}%, overflow_delta={_fmt(overflow)}%."
            )
        else:
            messages.append(
                f"`{objective_id}` does not satisfy the transfer rule: "
                f"hpwl_delta={_fmt(hpwl)}%, overflow_delta={_fmt(overflow)}%."
            )
    return " ".join(messages)


def _fmt(value: Any) -> str:
    number = _finite_number(value)
    return "n/a" if number is None else f"{number:.3f}"


def _run_scheduled_tier3_feedback(
    *,
    config: OpenEvolveTier2Config,
    database: ObjectiveProgramDatabase,
    programs: list[ObjectiveProgram],
    iteration: int,
    run_root: Path,
    chipbench_root: str | Path | None,
    resume: bool,
    retry_failed: bool,
) -> dict[str, Any] | None:
    """Route selected candidates and return their evidence to the archive."""

    if not config.tier3_enabled or config.tier3_interval <= 0:
        return None
    if iteration < max(1, config.tier3_start_iteration):
        return None
    if (iteration - config.tier3_start_iteration) % config.tier3_interval != 0:
        return None
    if chipbench_root is None:
        raise RuntimeError("scheduled routed evaluation requires chipbench_root")
    if not config.tier3_panel:
        raise RuntimeError("scheduled routed evaluation requires a Tier-3 panel")

    candidates = []
    for program in programs:
        objective_id = str((program.objective_spec or {}).get("id") or "")
        summary = program.artifacts.get("tier2_summary") or {}
        comparison_csv = summary.get("comparison_csv")
        if (
            program.status == "accepted"
            and objective_id
            and comparison_csv
            and Path(str(comparison_csv)).is_file()
        ):
            candidates.append((program, objective_id, Path(str(comparison_csv))))
    candidates.sort(
        key=lambda item: (item[0].combined_score, item[0].id),
        reverse=True,
    )
    candidates = candidates[: max(0, config.tier3_top_k)]
    if not candidates:
        return {
            "status": "skipped",
            "iteration": iteration,
            "reason": "no admitted candidate produced a routable DEF",
        }

    baseline_id = config.tier3_baseline_objective_id or "default"
    placement_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    selected_objective_ids = {objective_id for _program, objective_id, _csv in candidates}
    for _program, objective_id, comparison_csv in candidates:
        for row in _read_csv(comparison_csv):
            row_objective_id = str(row.get("objective_id") or "")
            if row_objective_id not in {baseline_id, objective_id}:
                continue
            artifact = str(row.get("output_artifact") or "")
            if row.get("status") != "success" or not artifact.lower().endswith(".def"):
                continue
            key = (str(row["design"]), row_objective_id, int(row["seed"]))
            placement_rows[key] = {
                "design": key[0],
                "objective_id": row_objective_id,
                "seed": key[2],
                "def_path": artifact,
                "source": f"evolution_iteration_{iteration:04d}",
            }

    routed_root = run_root / "scheduled_routed_evaluation" / f"iteration_{iteration:04d}"
    placements_path = _write_json(
        {
            "iteration": iteration,
            "placements": list(placement_rows.values()),
            "selected_objective_ids": sorted(selected_objective_ids),
            "baseline_objective_id": baseline_id,
        },
        routed_root / "placements.json",
    )
    if not placement_rows:
        return {
            "status": "skipped",
            "iteration": iteration,
            "reason": "selected placement rows contain no successful DEF artifacts",
            "placements": str(placements_path),
        }

    summary = run_tier3_openroad(
        panel_path=config.tier3_panel,
        placements_path=placements_path,
        chipbench_root=chipbench_root,
        run_dir=routed_root,
        store_path=routed_root / "tier3.sqlite",
        baseline_objective_id=baseline_id,
        resume=resume,
        retry_failed=retry_failed,
    )
    comparison_rows = _read_csv(Path(summary["comparison_csv"]))
    program_by_objective_id = {
        objective_id: program for program, objective_id, _csv in candidates
    }
    evidence_rows = []
    for objective_id, program in program_by_objective_id.items():
        rows = [
            row for row in comparison_rows
            if str(row.get("objective_id") or "") == objective_id
        ]
        if not rows:
            continue
        evidence = {
            "routed_wirelength_delta_pct": _mean_finite(
                row.get("routed_wirelength_delta_pct") for row in rows
            ),
            "routed_overflow_delta_pct": _mean_finite(
                row.get("grt_overflow_delta_pct") for row in rows
            ),
            "post_route_wns_gain_ns": _mean_finite(
                row.get("wns_gain_ns") for row in rows
            ),
            "post_route_tns_gain_ns": _mean_finite(
                row.get("tns_gain_ns") for row in rows
            ),
            "post_route_success_count": sum(
                1 for row in rows if row.get("status") == "success"
            ),
            "post_route_result_count": len(rows),
            "post_route_evidence_iteration": iteration,
            "post_route_evidence_status": (
                "success"
                if rows and all(row.get("status") == "success" for row in rows)
                else "partial"
            ),
            "per_design_routed_evidence": rows,
        }
        program.metrics.update(evidence)
        program.metrics["tier_c_evaluated"] = True
        program.artifacts["scheduled_routed_evaluation"] = summary
        evidence_rows.append(
            {
                "program_id": program.id,
                "objective_id": objective_id,
                **evidence,
            }
        )

    round_record = {
        "iteration": iteration,
        "selected_objective_ids": sorted(selected_objective_ids),
        "baseline_objective_id": baseline_id,
        "evidence": evidence_rows,
        "comparison_csv": summary["comparison_csv"],
    }
    feedback_file = run_root / "routed_feedback.json"
    previous = _load_json(str(feedback_file)) if feedback_file.is_file() else None
    rounds = [
        item
        for item in (previous or {}).get("rounds", [])
        if isinstance(item, dict) and item.get("iteration") != iteration
    ]
    rounds.append(round_record)
    rounds.sort(key=lambda item: int(item.get("iteration") or 0))
    # The latest round stays at the top level; "rounds" keeps every round so
    # later prompts see all routed evidence gathered during the run.
    feedback = {
        "stage": "scheduled_routed_evaluation",
        **round_record,
        "rounds": rounds,
    }
    feedback_path = _write_json(feedback, feedback_file)
    summary.update(
        {
            "status": "success" if evidence_rows else "partial",
            "iteration": iteration,
            "feedback": str(feedback_path),
            "selected_objective_ids": sorted(selected_objective_ids),
        }
    )
    database.rebuild_indexes()
    return summary


def _run_tier3_for_finalists(
    *,
    config: OpenEvolveTier2Config,
    database: ObjectiveProgramDatabase,
    final_summary: dict[str, Any],
    run_root: Path,
    chipbench_root: str | Path,
    resume: bool,
    retry_failed: bool,
    output_name: str = "tier3_openroad",
) -> dict[str, Any] | None:
    comparison_csv = final_summary.get("comparison_csv")
    if not comparison_csv or not Path(comparison_csv).is_file():
        return {
            "status": "skipped",
            "reason": "final Tier-2 comparison CSV is missing",
        }
    tier3_panel = config.tier3_panel or config.final_panel
    if not tier3_panel:
        return {"status": "skipped", "reason": "Tier-3 panel is not configured"}

    selected_ids: set[str] = {"default"}
    for preset_name in config.tier3_baseline_presets:
        try:
            selected_ids.add(objective_preset(preset_name).id)
        except KeyError:
            continue
    if config.tier3_baseline_objective_id:
        selected_ids.add(config.tier3_baseline_objective_id)

    finalist_programs, selection_policy = _tier3_candidate_programs(database, config)
    final_candidate_ids = _final_tier2_candidate_objective_ids(final_summary)
    selected_ids.update(final_candidate_ids)
    for program in finalist_programs:
        objective_id = str((program.objective_spec or {}).get("id") or "")
        if objective_id:
            selected_ids.add(objective_id)

    rows = _read_csv(Path(comparison_csv))
    placements = []
    for row in rows:
        objective_id = str(row.get("objective_id") or "")
        artifact = str(row.get("output_artifact") or "")
        if objective_id not in selected_ids:
            continue
        if row.get("status") != "success" or not artifact.lower().endswith(".def"):
            continue
        placements.append(
            {
                "design": row["design"],
                "objective_id": objective_id,
                "seed": int(row["seed"]),
                "def_path": artifact,
                "source": "openevolve_final_tier2",
            }
        )

    tier3_root = run_root / output_name
    placements_path = _write_json(
        {
            "placements": placements,
            "selected_objective_ids": sorted(selected_ids),
            "selected_final_tier2_candidate_ids": sorted(final_candidate_ids),
            "selected_finalist_programs": [
                {
                    "program_id": program.id,
                    "objective_id": (program.objective_spec or {}).get("id"),
                    "metrics": program.metrics,
                    "selection_class": (
                        "robust_elite"
                        if program.is_elite_eligible
                        else "aggregate_routing_candidate"
                    ),
                }
                for program in finalist_programs
            ],
            "selection_policy": selection_policy,
        },
        tier3_root / "placements_selected.json",
    )
    if not placements:
        return {
            "status": "skipped",
            "reason": "no successful final Tier-2 DEF placements matched Tier-3 selection",
            "placements": str(placements_path),
        }
    summary = run_tier3_openroad(
        panel_path=tier3_panel,
        placements_path=placements_path,
        chipbench_root=chipbench_root,
        run_dir=tier3_root,
        store_path=tier3_root / "tier3.sqlite",
        baseline_objective_id=config.tier3_baseline_objective_id or "default",
        resume=resume,
        retry_failed=retry_failed,
    )
    summary["placements"] = str(placements_path)
    summary["selected_objective_ids"] = sorted(selected_ids)
    summary["selected_final_tier2_candidate_ids"] = sorted(final_candidate_ids)
    return summary


def _final_tier2_candidate_objective_ids(final_summary: dict[str, Any]) -> set[str]:
    """Return generated final-candidate objective IDs actually run in final Tier-2.

    The program database may continue changing after the final panel candidate
    set is written, especially when calibrated siblings share lineage but have
    different ObjectiveSpec IDs. Tier-3 must route the DEFs from final Tier-2,
    so it should include the objective IDs recorded in the final objective
    manifest, not only the current database-best IDs.
    """

    manifest_path = final_summary.get("objective_manifest")
    if not manifest_path:
        return set()
    path = Path(str(manifest_path))
    if not path.is_file():
        return set()
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    ids: set[str] = set()
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        objective_path = str(entry.get("objective_path") or "")
        source = str(entry.get("source") or "")
        if source == "file" and "final_candidates" in objective_path.replace("\\", "/"):
            objective_id = entry.get("objective_id")
            if objective_id:
                ids.add(str(objective_id))
    return ids


def _tier3_candidate_programs(
    database: ObjectiveProgramDatabase,
    config: OpenEvolveTier2Config,
) -> tuple[list[ObjectiveProgram], str]:
    """Select generated programs for routed validation.

    Robust elite candidates are preferred. If no robust elite exists and the
    policy allows it, top aggregate routing-aware winners are routed as
    validation probes, but they are not labeled as all-design robust winners.
    """

    top_k = max(0, config.tier3_top_k)
    if top_k <= 0:
        return [], "Tier-3 candidate selection disabled because top_k <= 0"

    candidates = [
        program
        for program in database.programs.values()
        if program.objective_spec
        and not program.metrics.get("seed_baseline")
        and program.metrics.get("routing_aware_tier2_candidate")
        and not program.metrics.get("structural_failure_count")
        and not program.metrics.get("severe_regression_count")
    ]

    def metric(program: ObjectiveProgram, name: str, default: float) -> float:
        try:
            value = float(program.metrics.get(name))
        except (TypeError, ValueError):
            return default
        return value if math.isfinite(value) else default

    elites = [program for program in candidates if program.is_elite_eligible]
    elites.sort(
        key=lambda program: (
            metric(program, "tier2_average_rank", 999.0),
            metric(program, "hpwl_delta_pct", 999.0),
            metric(program, "overflow_delta_pct", 999.0),
            program.id,
        )
    )
    if elites or config.tier3_candidate_policy == "elite_only":
        return (
            elites[:top_k],
            "native default, fixed Tier-3 baselines, primary baseline, and top robust routing-aware Tier-2 elite candidates",
        )

    aggregate = [
        program
        for program in candidates
        if program.metrics.get("aggregate_tier2_winner")
        and program.metrics.get("hpwl_gate_passed")
        and program.metrics.get("overflow_gate_passed")
    ]
    aggregate.sort(
        key=lambda program: (
            metric(program, "tier2_average_rank", 999.0),
            metric(program, "worst_design_hpwl_delta_pct", 999.0),
            metric(program, "worst_design_overflow_delta_pct", 999.0),
            metric(program, "hpwl_delta_pct", 999.0),
            metric(program, "overflow_delta_pct", 999.0),
            program.id,
        )
    )
    return (
        aggregate[:top_k],
        "native default, fixed Tier-3 baselines, primary baseline, and top aggregate routing-aware Tier-2 candidates; aggregate candidates are routed for validation only and are not all-design robust winners",
    )


def _seed_initial_programs(
    database: ObjectiveProgramDatabase,
    *,
    config: OpenEvolveTier2Config,
    run_root: Path,
    dreamplace_root: str | Path,
    resume: bool,
    retry_failed: bool,
) -> None:
    if config.evaluate_initial_presets:
        _evaluate_initial_presets(
            database=database,
            config=config,
            run_root=run_root,
            dreamplace_root=dreamplace_root,
            resume=resume,
            retry_failed=retry_failed,
        )
        if any(program.is_parent_eligible for program in database.programs.values()):
            return
        if _promote_bootstrap_seed_parent(database):
            return
        if database.programs:
            return

    for index, name in enumerate(config.initial_presets):
        spec = objective_preset(name)
        program = ObjectiveProgram.from_spec(
            spec,
            program_id=f"seed_{index:02d}_{spec.id}",
            code=objective_code_from_spec(spec),
            generation=0,
            iteration_found=0,
            metrics={
                "combined_score": 0.0,
                "constrained_score": 0.0,
                "seed_baseline": True,
                "manual_safe_baseline": False,
                "parent_eligible": False,
                "hpwl_gate_passed": False,
                "overflow_gate_passed": False,
                "elite_gate_passed": False,
                "structural_failure_count": 1,
                "severe_regression_count": 0,
                "negative_memory_only": True,
                "feedback_lesson": (
                    "Initial preset was not evaluated successfully by DREAMPlace, "
                    "so it cannot be sampled as a parent."
                ),
            },
            artifacts={"preset": name, "baseline_seed_mode": "unevaluated_not_parent"},
            status="failed",
            failure_reason="initial preset has no real Tier-2 metrics",
        )
        database.add(program, target_island=index % len(database.islands))


def _promote_bootstrap_seed_parent(database: ObjectiveProgramDatabase) -> ObjectiveProgram | None:
    candidates = [
        program
        for program in database.programs.values()
        if program.status == "accepted"
        and program.metrics.get("seed_baseline")
        and program.artifacts.get("preset") != "dreamplace_controller_native_identity"
        and not program.metrics.get("structural_failure_count", 0)
        and "hpwl_delta_pct" in program.metrics
        and "overflow_delta_pct" in program.metrics
    ]
    if not candidates:
        return None

    def key(program: ObjectiveProgram) -> tuple[float, float, float, float, float, float]:
        metrics = program.metrics
        routing_seed = bool(ROUTE_CORRECTION_TERMS.intersection(program.term_set))
        severe = float(metrics.get("severe_regression_count") or 0.0)
        rank = _finite_number(metrics.get("tier2_average_rank"))
        hpwl = _finite_number(metrics.get("hpwl_delta_pct"))
        overflow = _finite_number(metrics.get("overflow_delta_pct"))
        worst_hpwl = _finite_number(metrics.get("worst_design_hpwl_delta_pct"))
        return (
            severe,
            rank if rank is not None else 999.0,
            worst_hpwl if worst_hpwl is not None else 1e9,
            hpwl if hpwl is not None else 1e9,
            overflow if overflow is not None else 1e9,
            0.0 if routing_seed else 1.0,
        )

    selected = min(candidates, key=key)
    selected.metrics.update(
        {
            "bootstrap_parent": True,
            "manual_safe_baseline": True,
            "parent_eligible": True,
            "negative_memory_only": False,
            "feedback_lesson": (
                "Strict HPWL/overflow gates produced no eligible seed parent. "
                "This evaluated seed baseline is used only to bootstrap mutation; "
                "it is not considered an elite or final promoted objective."
            ),
        }
    )
    selected.artifacts["bootstrap_parent_reason"] = (
        "best evaluated non-structural seed baseline by severe-regression count, "
        "Tier-2 rank, worst-design HPWL delta, average HPWL delta, overflow "
        "delta, and routing-awareness as a final tie-breaker"
    )
    database.add(selected, target_island=selected.island)
    return selected


def _refresh_database_scores(
    database: ObjectiveProgramDatabase,
    config: OpenEvolveTier2Config,
) -> None:
    for program in database.programs.values():
        metrics = program.metrics
        if "hpwl_delta_pct" not in metrics or "overflow_delta_pct" not in metrics:
            continue
        structural_failure = bool(metrics.get("structural_failure_count", 0))
        hpwl = _active_baseline_delta(metrics, config, metric="hpwl")
        overflow = _active_baseline_delta(metrics, config, metric="overflow")
        metrics["hpwl_delta_pct"] = hpwl
        metrics["overflow_delta_pct"] = overflow
        if (
            metrics.get("selection_metric_stage")
            == "final_def_post_detailed_placement"
            and isinstance(metrics.get("per_design_deltas"), list)
            and metrics.get("per_design_deltas")
        ):
            stored_design_summary = _design_delta_summary(
                metrics["per_design_deltas"]
            )
        else:
            stored_design_summary = _stored_design_summary_for_active_baseline(
                program.artifacts.get("candidate_rows"), config
            )
        if stored_design_summary is not None:
            design_summary = stored_design_summary
            per_design_delta_baseline = config.primary_baseline
            per_design_deltas_stale = False
        else:
            per_design_delta_baseline = metrics.get("per_design_delta_baseline")
            per_design_deltas_stale = per_design_delta_baseline != config.primary_baseline
            worst_design_hpwl = _active_worst_design_delta(metrics, config, metric="hpwl")
            if worst_design_hpwl is None:
                worst_design_hpwl = hpwl
            worst_design_overflow = _active_worst_design_delta(metrics, config, metric="overflow")
            if worst_design_overflow is None:
                worst_design_overflow = overflow
            design_both_fraction = _finite_number(metrics.get("design_both_improvement_fraction"))
            if per_design_deltas_stale or design_both_fraction is None:
                design_both_fraction = (
                    1.0
                    if (hpwl or 0.0) <= 0.0 and (overflow or 0.0) <= 0.0
                    else 0.0
                )
            design_summary = {
                "worst_design_hpwl_delta_pct": worst_design_hpwl,
                "worst_design_overflow_delta_pct": worst_design_overflow,
                "design_hpwl_improvement_fraction": metrics.get(
                    "design_hpwl_improvement_fraction"
                )
                if not per_design_deltas_stale
                else (1.0 if (hpwl or 0.0) <= 0.0 else 0.0),
                "design_overflow_improvement_fraction": metrics.get(
                    "design_overflow_improvement_fraction"
                )
                if not per_design_deltas_stale
                else (1.0 if (overflow or 0.0) <= 0.0 else 0.0),
                "design_both_improvement_fraction": design_both_fraction,
                "cell_both_improvement_fraction": metrics.get(
                    "cell_both_improvement_fraction"
                )
                if not per_design_deltas_stale
                else design_both_fraction,
            }
            if not per_design_deltas_stale:
                design_summary["per_design_deltas"] = metrics.get("per_design_deltas", [])
        worst_design_hpwl = _finite_number(design_summary.get("worst_design_hpwl_delta_pct"))
        if worst_design_hpwl is None:
            worst_design_hpwl = hpwl
        worst_design_overflow = _finite_number(
            design_summary.get("worst_design_overflow_delta_pct")
        )
        if worst_design_overflow is None:
            worst_design_overflow = overflow
        design_both_fraction = _finite_number(
            design_summary.get("design_both_improvement_fraction")
        )
        if design_both_fraction is None:
            design_both_fraction = 0.0
        hpwl_gate_passed = (
            not structural_failure
            and hpwl is not None
            and hpwl <= config.hpwl_gate_search_pct
        )
        overflow_gate_passed = overflow is not None and overflow <= 0.0
        per_design_hpwl_gate_passed = (
            not structural_failure
            and worst_design_hpwl is not None
            and worst_design_hpwl <= config.per_design_hpwl_gate_search_pct
        )
        per_design_overflow_gate_passed = (
            not structural_failure
            and worst_design_overflow is not None
            and worst_design_overflow <= config.per_design_overflow_gate_search_pct
        )
        design_both_fraction_passed = (
            design_both_fraction >= config.min_design_both_improvement_fraction
        )
        robust_gate_passed = (
            per_design_hpwl_gate_passed
            and per_design_overflow_gate_passed
            and design_both_fraction_passed
        )
        severe_regression_count = _active_severe_regression_count(
            hpwl_delta_pct=hpwl,
            overflow_delta_pct=overflow,
            worst_design_hpwl_delta_pct=worst_design_hpwl,
            worst_design_overflow_delta_pct=worst_design_overflow,
            severe_threshold_pct=config.severe_regression_pct,
            overflow_severe_threshold_pct=config.overflow_severe_regression_pct,
            structural_failure=structural_failure,
        )
        elite_gate_passed = (
            hpwl_gate_passed
            and hpwl is not None
            and hpwl <= config.hpwl_gate_elite_pct
            and overflow_gate_passed
            and robust_gate_passed
            and not severe_regression_count
        )
        custom_default_hpwl = _finite_number(metrics.get("custom_default_hpwl_delta_pct"))
        if custom_default_hpwl is None and config.primary_baseline == "custom_default":
            custom_default_hpwl = hpwl
        custom_default_overflow = _finite_number(
            metrics.get("custom_default_overflow_delta_pct")
        )
        if custom_default_overflow is None and config.primary_baseline == "custom_default":
            custom_default_overflow = overflow
        custom_default_hpwl_gate_passed = (
            custom_default_hpwl is not None
            and custom_default_hpwl <= config.custom_default_hpwl_gate_pct
        )
        custom_default_overflow_gate_passed = (
            custom_default_overflow is not None
            and custom_default_overflow <= config.custom_default_overflow_gate_pct
        )
        custom_default_effect_values = [
            abs(value)
            for value in (custom_default_hpwl, custom_default_overflow)
            if value is not None and math.isfinite(value)
        ]
        custom_default_effect_pct = (
            max(custom_default_effect_values) if custom_default_effect_values else None
        )
        custom_default_effect_gate_passed = (
            config.min_custom_default_effect_pct <= 0.0
            or (
                custom_default_effect_pct is not None
                and custom_default_effect_pct >= config.min_custom_default_effect_pct
            )
        )
        custom_default_gate_passed = (
            custom_default_hpwl_gate_passed
            and custom_default_overflow_gate_passed
            and custom_default_effect_gate_passed
        )
        native_default_hpwl = _finite_number(metrics.get("native_default_hpwl_delta_pct"))
        if native_default_hpwl is None and config.primary_baseline == "default":
            native_default_hpwl = hpwl
        native_default_overflow = _finite_number(
            metrics.get("native_default_overflow_delta_pct")
        )
        if native_default_overflow is None and config.primary_baseline == "default":
            native_default_overflow = overflow
        native_default_label = _native_default_comparison_label(
            native_default_hpwl,
            native_default_overflow,
        )
        score = _constrained_score(
            average_rank=float(metrics.get("tier2_average_rank") or 999.0),
            hpwl_delta_pct=hpwl,
            overflow_delta_pct=overflow,
            hpwl_gate_passed=hpwl_gate_passed,
            overflow_gate_passed=overflow_gate_passed,
            robust_gate_passed=robust_gate_passed,
            require_robust_gate=config.require_robust_parent_gate,
            design_both_improvement_fraction=design_both_fraction,
            structural_failure=structural_failure,
            hpwl_gate_search_pct=config.hpwl_gate_search_pct,
        )
        parent_eligible = (
            hpwl_gate_passed
            and overflow_gate_passed
            and (robust_gate_passed or not config.require_robust_parent_gate)
            and not structural_failure
        )
        negative_memory_only = (
            structural_failure
            or not hpwl_gate_passed
            or not overflow_gate_passed
            or (config.require_robust_parent_gate and not robust_gate_passed)
        )
        metrics.update(
            {
                "combined_score": score,
                "constrained_score": score,
                "hpwl_gate_passed": hpwl_gate_passed,
                "overflow_gate_passed": overflow_gate_passed,
                "per_design_hpwl_gate_passed": per_design_hpwl_gate_passed,
                "per_design_overflow_gate_passed": per_design_overflow_gate_passed,
                "design_both_fraction_passed": design_both_fraction_passed,
                "robust_gate_passed": robust_gate_passed,
                "require_robust_parent_gate": config.require_robust_parent_gate,
                "near_miss_parent_due_to_robustness": (
                    parent_eligible and not robust_gate_passed
                ),
                "elite_gate_passed": elite_gate_passed,
                "custom_default_hpwl_gate_passed": custom_default_hpwl_gate_passed,
                "custom_default_overflow_gate_passed": custom_default_overflow_gate_passed,
                "custom_default_effect_pct": custom_default_effect_pct,
                "custom_default_effect_gate_passed": custom_default_effect_gate_passed,
                "custom_default_gate_passed": custom_default_gate_passed,
                "native_default_hpwl_delta_pct": native_default_hpwl,
                "native_default_overflow_delta_pct": native_default_overflow,
                "native_default_comparison_label": native_default_label,
                "beats_native_default_pareto": (
                    native_default_label == "dominates_native_default"
                ),
                "native_default_tradeoff": native_default_label
                in {
                    "native_tradeoff_overflow_for_hpwl",
                    "native_tradeoff_hpwl_for_overflow",
                },
                "native_default_regression": (
                    native_default_label == "regresses_native_default"
                ),
                "severe_regression_count": severe_regression_count,
                "active_delta_baseline": config.primary_baseline,
                "per_design_delta_baseline": per_design_delta_baseline,
                "per_design_deltas_stale": per_design_deltas_stale,
                **design_summary,
                "primary_baseline": config.primary_baseline,
                "parent_eligible": parent_eligible,
                "negative_memory_only": negative_memory_only,
                "feedback_lesson": _feedback_lesson(
                    structural_failure=structural_failure,
                    hpwl_gate_passed=hpwl_gate_passed,
                    overflow_gate_passed=overflow_gate_passed,
                    robust_gate_passed=robust_gate_passed,
                    design_summary=design_summary,
                    hpwl_delta_pct=hpwl,
                    overflow_delta_pct=overflow,
                    elite_gate_passed=elite_gate_passed,
                ),
            }
        )
        if program.objective_spec:
            try:
                spec = parse_objective_spec(
                    program.objective_spec,
                    created_by="openevolve:resume",
                    term_scope="dreamplace_replacement",
                )
                _apply_routing_gate(
                    metrics,
                    spec,
                    config,
                    allow_nonrouting_parent=bool(
                        metrics.get("seed_baseline") or metrics.get("manual_safe_baseline")
                    ),
                )
                _apply_measured_routing_influence_gate(
                    metrics,
                    config,
                    allow_seed_baseline=bool(
                        metrics.get("seed_baseline")
                        or metrics.get("manual_safe_baseline")
                    ),
                )
                if not (metrics.get("seed_baseline") or metrics.get("manual_safe_baseline")):
                    _clear_near_miss_policy_state(metrics)
                    _refresh_baseline_portfolio_metrics(program, metrics, config)
                    _apply_baseline_portfolio_gate(metrics, config)
                _apply_native_default_selection_policy(
                    metrics,
                    config,
                    allow_seed_baseline=bool(
                        metrics.get("seed_baseline") or metrics.get("manual_safe_baseline")
                    ),
                )
                _apply_near_miss_parent_policy(
                    metrics,
                    config,
                    allow_seed_baseline=bool(
                        metrics.get("seed_baseline") or metrics.get("manual_safe_baseline")
                    ),
                )
                _apply_timing_controller_admission(metrics, spec, config)
                apply_timing_proxy_policy(
                    metrics,
                    requested_mode=config.timing_proxy_mode,
                    wns_regression_gate_ns=config.timing_proxy_wns_regression_gate_ns,
                    tns_regression_pct_gate=config.timing_proxy_tns_regression_pct_gate,
                    tns_regression_min_abs_ns=config.timing_proxy_tns_regression_min_abs_ns,
                    wns_delta_min_abs_ns=config.timing_proxy_wns_delta_min_abs_ns,
                    selection_weight=config.timing_proxy_selection_weight,
                )
                _apply_manuscript_multimetric_selection(
                    metrics,
                    config,
                    allow_seed_baseline=bool(
                        metrics.get("seed_baseline")
                        or metrics.get("manual_safe_baseline")
                    ),
                )
            except ObjectiveSpecError:
                metrics.update(
                    {
                        "routing_term_required": config.require_nonzero_routing_term,
                        "routing_term_gate_passed": False,
                        "routing_aware_tier2_candidate": False,
                    }
                )
    if config.selection_policy == "pareto_multiobjective":
        _refresh_pareto_selection(database, config)
        return
    database.archive.clear()
    database.best_program_id = None
    for program in database.programs.values():
        database._update_archive(program)
        database._update_best(program)


def _clear_near_miss_policy_state(metrics: dict[str, Any]) -> None:
    """Remove stale exploration-only flags before replaying current gates."""

    for key in (
        "exploration_parent_only",
        "near_miss_parent_score",
        "near_miss_parent_reason",
    ):
        metrics.pop(key, None)


def _refresh_baseline_portfolio_metrics(
    program: ObjectiveProgram,
    metrics: dict[str, Any],
    config: OpenEvolveTier2Config,
) -> None:
    """Recompute fixed-baseline comparison metrics from persisted Tier-2 rows."""

    if not program.objective_spec:
        return
    objective_id = str(program.objective_spec.get("id") or "")
    if not objective_id:
        return
    tier2_summary = program.artifacts.get("tier2_summary")
    if not isinstance(tier2_summary, dict):
        return
    comparison_csv = tier2_summary.get("comparison_csv")
    if not comparison_csv or not Path(str(comparison_csv)).is_file():
        return
    baseline_ids = {
        str(item)
        for item in program.artifacts.get("baseline_portfolio_ids", [])
        if item is not None
    }
    if not baseline_ids:
        return
    rows = add_outcome_labels(
        _rows_with_primary_baseline(
            _read_csv(Path(str(comparison_csv))),
            primary_baseline=config.primary_baseline,
        ),
        severe_threshold_pct=config.severe_regression_pct,
    )
    rankings = aggregate_rank_scores(
        rows,
        severe_threshold_pct=config.severe_regression_pct,
    )
    metrics.update(
        _baseline_portfolio_summary(
            rows=rows,
            rankings=rankings,
            candidate_objective_id=objective_id,
            baseline_objective_ids=baseline_ids,
            min_effect_pct=config.min_baseline_portfolio_effect_pct,
            min_behavior_distance_pct=config.min_baseline_behavior_distance_pct,
        )
    )


def _evaluate_initial_presets(
    *,
    database: ObjectiveProgramDatabase,
    config: OpenEvolveTier2Config,
    run_root: Path,
    dreamplace_root: str | Path,
    resume: bool,
    retry_failed: bool,
) -> None:
    seed_dir = run_root / "initial_presets"
    preset_dir = seed_dir / "objectives"
    preset_paths: list[str] = []
    preset_by_path: dict[str, tuple[int, str, ObjectiveSpec]] = {}
    blocked_presets: list[dict[str, Any]] = []
    for index, name in enumerate(config.initial_presets):
        spec = objective_preset(name)
        blocked_terms = sorted(
            set(spec.term_set).intersection(config.term_scale_audit_blocked_terms)
        )
        if blocked_terms:
            blocked_presets.append(
                {
                    "index": index,
                    "preset": name,
                    "objective_id": spec.id,
                    "blocked_terms": blocked_terms,
                    "reason": "preset uses terms blocked by the deterministic term-scale audit",
                }
            )
            continue
        path = write_objective_spec(spec, preset_dir / f"{index:02d}_{_safe_name(name)}.json")
        preset_paths.append(str(path))
        preset_by_path[str(path.resolve())] = (index, name, spec)
    _write_json(blocked_presets, seed_dir / "blocked_presets.json")
    for item in blocked_presets:
        spec = objective_preset(str(item["preset"]))
        program = ObjectiveProgram.from_spec(
            spec,
            program_id=f"seed_{int(item['index']):02d}_{spec.id}",
            generation=0,
            iteration_found=0,
            metrics={
                "combined_score": 0.0,
                "constrained_score": 0.0,
                "seed_baseline": True,
                "preset": item["preset"],
                "mechanism_signature": _mechanism_signature(_spec_mechanism_view(spec)),
                "mechanism_terms": sorted(_terms_in_ast(spec.ast)),
                "term_scale_audit_blocked_terms": item["blocked_terms"],
                "parent_eligible": False,
                "negative_memory_only": True,
                "feedback_lesson": (
                    "Initial preset was not launched because its objective uses "
                    "a term blocked by the deterministic term-scale audit."
                ),
            },
            artifacts={"preset": item["preset"], "term_scale_audit": item},
            status="rejected",
            failure_reason="initial_preset_blocked_by_term_scale_audit",
        )
        database.add(program, target_island=int(item["index"]) % len(database.islands))

    panel_path = write_dreamplace_panel(
        load_shared_panel(config.search_panel),
        seed_dir / "search_panel.toml",
    )
    try:
        tier2_summary = run_tier2_dreamplace(
            panel_path=panel_path,
            objective_paths=preset_paths,
            dreamplace_root=dreamplace_root,
            run_dir=seed_dir / "tier2_search",
            store_path=seed_dir / "tier2_search" / "tier2.sqlite",
            resume=resume,
            retry_failed=retry_failed,
            include_default=True,
            include_custom_default=True,
            require_output_artifact=True,
            require_def_output=True,
            disable_legalization=config.search_disable_legalization,
            objective_semantics=(
                "raw" if config.objective_mode == "controller" else None
            ),
            timing_controller_panel=config.timing_controller_panel,
        )
    except Exception as exc:
        for index, name, spec in preset_by_path.values():
            program = ObjectiveProgram.from_spec(
                spec,
                program_id=f"seed_{index:02d}_{spec.id}",
                generation=0,
                iteration_found=0,
                metrics={
                    "combined_score": 0.0,
                    "constrained_score": 0.0,
                    "seed_baseline": True,
                    "mechanism_signature": _mechanism_signature(_spec_mechanism_view(spec)),
                    "mechanism_terms": sorted(_terms_in_ast(spec.ast)),
                    "structural_failure_count": 1,
                    "parent_eligible": False,
                    "negative_memory_only": True,
                    "feedback_lesson": "Initial preset DREAMPlace evaluation failed structurally.",
                },
                artifacts={"preset": name, "tier2_exception": f"{type(exc).__name__}: {exc}"},
                status="failed",
                failure_reason=f"initial_preset_tier2_exception: {type(exc).__name__}: {exc}",
            )
            database.add(program, target_island=index % len(database.islands))
        return

    rows = add_outcome_labels(
        _rows_with_primary_baseline(
            _read_csv(Path(tier2_summary["comparison_csv"])),
            primary_baseline=config.primary_baseline,
        ),
        severe_threshold_pct=config.severe_regression_pct,
    )
    rankings = aggregate_rank_scores(rows, severe_threshold_pct=config.severe_regression_pct)
    ranking_by_id = {ranking.objective_id: ranking for ranking in rankings}
    objective_manifest_path = tier2_summary.get("objective_manifest")
    if objective_manifest_path and Path(objective_manifest_path).is_file():
        objective_manifest = json.loads(Path(objective_manifest_path).read_text(encoding="utf-8"))
        manifest_by_path = {
            str(Path(item["objective_path"]).resolve()): item
            for item in objective_manifest
            if item.get("objective_path")
        }
    else:
        manifest_by_path = {}

    seed_objective_ids = {
        str(manifest_by_path[path_text].get("objective_id") or spec.id)
        if path_text in manifest_by_path
        else spec.id
        for path_text, (_index, _name, spec) in preset_by_path.items()
    }
    timing_metrics, timing_artifacts = _timing_proxy_feedback_for_tier2_summary(
        tier2_summary=tier2_summary,
        config=config,
        iteration_dir=seed_dir,
        dreamplace_root=dreamplace_root,
        objective_ids=seed_objective_ids,
        resume=resume,
        retry_failed=retry_failed,
    )

    for path_text, (index, name, spec) in preset_by_path.items():
        manifest_item = manifest_by_path.get(path_text)
        objective_id = str(manifest_item.get("objective_id")) if manifest_item else spec.id
        candidate_rows = [row for row in rows if row.get("objective_id") == objective_id]
        metrics = _metrics_from_rows(candidate_rows, ranking_by_id.get(objective_id), config)
        metrics.update(timing_metrics.get(objective_id, {}))
        # Seed baselines are the audit's reference placements and always
        # receive the timing proxy when it is enabled.
        _record_tier_b_evidence(metrics, selected=config.timing_proxy_enabled)
        _apply_timing_controller_admission(metrics, spec, config)
        _apply_final_def_selection_metrics(metrics, config)
        _apply_routing_gate(metrics, spec, config, allow_nonrouting_parent=True)
        _apply_measured_routing_influence_gate(
            metrics, config, allow_seed_baseline=True
        )
        _apply_native_default_selection_policy(metrics, config, allow_seed_baseline=True)
        apply_timing_proxy_policy(
            metrics,
            requested_mode=config.timing_proxy_mode,
            wns_regression_gate_ns=config.timing_proxy_wns_regression_gate_ns,
            tns_regression_pct_gate=config.timing_proxy_tns_regression_pct_gate,
            tns_regression_min_abs_ns=config.timing_proxy_tns_regression_min_abs_ns,
            wns_delta_min_abs_ns=config.timing_proxy_wns_delta_min_abs_ns,
            selection_weight=config.timing_proxy_selection_weight,
        )
        _apply_manuscript_multimetric_selection(
            metrics,
            config,
            allow_seed_baseline=True,
        )
        metrics.update(
            {
                "seed_baseline": True,
                "preset": name,
                "manual_safe_baseline": False,
                "mechanism_signature": _mechanism_signature(_spec_mechanism_view(spec)),
                "mechanism_terms": sorted(_terms_in_ast(spec.ast)),
            }
        )
        if (
            config.objective_mode == "native_residual"
            and not set(spec.term_set) <= NATIVE_RESIDUAL_ALLOWED_TERMS
        ):
            metrics["combined_score"] = 0.0
            metrics["constrained_score"] = 0.0
            metrics["parent_eligible"] = False
            metrics["diagnostic_baseline_only"] = True
            metrics["feedback_lesson"] = (
                "This fixed replacement baseline was evaluated for comparison "
                "only. Native-residual runs sample parents only from native "
                "DREAMPlace plus routing-correction objectives."
            )
        if name == "dreamplace_controller_native_identity":
            metrics["combined_score"] = 0.0
            metrics["constrained_score"] = 0.0
            metrics["parent_eligible"] = False
            metrics["diagnostic_baseline_only"] = True
            metrics["feedback_lesson"] = (
                "Native DREAMPlace identity control is retained as a measured "
                "reference. Generated objectives use the public trajectory-aware "
                "policy interface and are proposed from executable search parents."
            )
        if metrics.get("parent_eligible"):
            metrics["feedback_lesson"] = (
                "Evaluated seed baseline completed and is eligible as a parent "
                "under the current selection policy."
            )
        artifacts = {
            "preset": name,
            "tier2_summary": tier2_summary,
            "objective_id_in_tier2": objective_id,
            "candidate_rows": candidate_rows[:10],
            **timing_artifacts,
        }
        status = "accepted" if not metrics.get("structural_failure_count") else "failed"
        program = ObjectiveProgram.from_spec(
            spec,
            program_id=f"seed_{index:02d}_{objective_id}",
            code=objective_code_from_spec(spec),
            generation=0,
            iteration_found=0,
            metrics=metrics,
            artifacts=artifacts,
            status=status,
            failure_reason=None if status == "accepted" else "initial preset structural failure",
        )
        database.add(program, target_island=index % len(database.islands))


def _static_rejection_reason(
    spec: ObjectiveSpec,
    database: ObjectiveProgramDatabase,
    config: OpenEvolveTier2Config,
) -> str | None:
    validation_scope = (
        "dreamplace_native_residual"
        if config.objective_mode == "native_residual"
        else "dreamplace_replacement"
    )
    if config.objective_mode == "controller":
        validation_scope = "dreamplace_controller"
    unsupported = unsupported_terms_for_scope(spec.term_set, validation_scope)
    if unsupported:
        return f"unsupported {validation_scope} terms: {unsupported}"
    blocked_by_audit = sorted(set(spec.term_set).intersection(config.term_scale_audit_blocked_terms))
    if blocked_by_audit:
        return (
            "objective uses terms blocked by the deterministic term-scale audit: "
            f"{blocked_by_audit}"
        )
    if _uses_mutually_exclusive_wirelength_terms(spec.term_set):
        return (
            "objective mixes wirelength_lse with wirelength_wawl or legacy "
            "wirelength; use one smooth wirelength formulation"
        )
    if _uses_mutually_exclusive_density_terms(spec.term_set):
        return (
            "objective mixes density_bell with density_electric or legacy density; "
            "use one density formulation"
        )
    if config.objective_mode == "residual":
        missing_base = sorted(RESIDUAL_BASE_TERMS.difference(spec.term_set))
        if missing_base:
            return f"residual objective is missing required base terms: {missing_base}"
        coeffs, unsupported_routes = _route_coefficients(spec.ast)
        if unsupported_routes:
            return (
                "residual routing corrections must be weighted route-only "
                "observables, compact weighted route-only transforms, or "
                "compact route-anchor interactions; "
                f"unsupported route expression around {sorted(unsupported_routes)}"
            )
        too_large = [
            {"term": term, "coefficient": coeff}
            for term, coeff in coeffs
            if abs(coeff) > config.route_coeff_cap
        ]
        if too_large:
            return (
                f"residual route coefficient exceeds cap {config.route_coeff_cap}: "
                f"{too_large}"
            )
    elif config.objective_mode == "native_residual":
        missing_base = sorted(NATIVE_RESIDUAL_BASE_TERMS.difference(spec.term_set))
        if missing_base:
            return f"native-residual objective is missing required base terms: {missing_base}"
        forbidden = sorted(NATIVE_RESIDUAL_FORBIDDEN_SCORE_TERMS.intersection(spec.term_set))
        if forbidden:
            return (
                "native-residual objectives must not tune fixed wirelength/density "
                f"terms directly; use native_objective plus routing corrections. "
                f"Forbidden terms: {forbidden}"
            )
        disallowed = sorted(set(spec.term_set).difference(NATIVE_RESIDUAL_ALLOWED_TERMS))
        if disallowed:
            return (
                "native-residual objectives may use only native_objective plus "
                "routing-correction terms; "
                f"disallowed terms: {disallowed}"
            )
        coeffs, unsupported_routes = _route_coefficients(spec.ast)
        if unsupported_routes:
            return (
                "native-residual routing corrections must be weighted route "
                "observables, weighted unary transforms of route observables, "
                "or compact route-anchor interactions; "
                f"unsupported route expression around {sorted(unsupported_routes)}"
            )
        too_large = [
            {"term": term, "coefficient": coeff}
            for term, coeff in coeffs
            if abs(coeff) > config.route_coeff_cap
        ]
        if too_large:
            return (
                f"native-residual route coefficient exceeds cap {config.route_coeff_cap}: "
                f"{too_large}"
            )
    elif config.objective_mode == "controller":
        if not spec.is_typed_policy:
            return (
                "official controller runs require typed_policy_v1 with "
                "init_policy/update_policy/objective(features, policy); "
                "legacy state-register programs are readable but not generated"
            )
        privileged_used = sorted(
            set(getattr(spec, "observable_set", []) or [])
            & set(privileged_observable_names())
        )
        if privileged_used:
            return (
                "generated controllers must derive their schedules from "
                "iteration progress, overflow, HPWL trend, and calibration "
                "observables; privileged native-schedule observables are "
                f"reserved for identity controls: {privileged_used}"
            )
        if spec.net_weight_policy is not None:
            uncovered = _timing_controller_uncovered_designs(config)
            if uncovered:
                return (
                    "update_net_weights needs timing analysis during placement, "
                    "and this run has no in-loop timing collateral for "
                    f"{uncovered}; remove update_net_weights and express the "
                    "mechanism through the policy schedules"
                )
        if not (set(spec.term_set) & WIRELENGTH_FAMILY_TERMS):
            return (
                "controller objective must include a canonical wirelength anchor "
                "(wirelength, wirelength_wawl, or wirelength_lse)"
            )
        if not (set(spec.term_set) & DENSITY_FAMILY_TERMS):
            return (
                "controller objective must include a density/utilization anchor "
                "(density, density_electric, or density_bell)"
            )
        registers = spec.state.get("registers", {}) if spec.state else {}
        missing_policy_slots = {"density_weight", "gamma_scale"} - set(registers)
        if missing_policy_slots:
            return f"typed policy is missing required slots: {sorted(missing_policy_slots)}"
        if config.min_state_registers > 0:
            if len(registers) < config.min_state_registers:
                return (
                    "controller objective must define at least "
                    f"{config.min_state_registers} policy slot(s) via "
                    "init_policy; static formulas cannot track the native "
                    "density-weight schedule"
                )
            dynamic = [
                name
                for name, entry in registers.items()
                if _ast_uses_obs(entry.get("update"))
                or _ast_uses_obs(entry.get("init"))
            ]
            if not dynamic:
                return (
                    "controller policy slots must read at least one "
                    "observable (overflow, iter_frac, hpwl_delta_rate, "
                    "gamma_frac, grad_ratio_*) in an "
                    "init or update expression"
                )
        # Route-term coefficient caps are checked only for constant weights;
        # register-weighted route terms are admitted and judged by the
        # autograd gate plus measured behavior.
    elif config.objective_mode == "replacement":
        if not (set(spec.term_set) & WIRELENGTH_FAMILY_TERMS):
            return (
                "replacement objective must include a canonical wirelength anchor "
                "(wirelength, wirelength_wawl, or wirelength_lse); "
                "pin_count_weighted_wl is a routing-aware modifier, not a "
                "standalone replacement for normal wire-span pressure"
            )
        if not (set(spec.term_set) & DENSITY_FAMILY_TERMS):
            return (
                "replacement objective must include a density/utilization anchor "
                "(density, density_electric, or density_bell) so the optimizer "
                "keeps overlap control while adding routing-aware mechanisms"
            )
        if config.min_replacement_nonbaseline_terms > 0:
            nonbaseline_terms = sorted(
                set(spec.term_set).difference(DEFAULT_EQUIVALENT_BASE_TERMS)
            )
            if len(nonbaseline_terms) < config.min_replacement_nonbaseline_terms:
                return (
                    "replacement objective is too close to the explicit "
                    "wirelength+density baseline; use at least "
                    f"{config.min_replacement_nonbaseline_terms} non-baseline "
                    "observable families such as LSE/bell density, route "
                    "pressure, soft RUDY, pin density, or pin-count-weighted "
                    f"wirelength. Found: {nonbaseline_terms}"
                )
        coeffs, unsupported_routes = _route_coefficients(spec.ast)
        if unsupported_routes:
            return (
                "replacement routing terms must be weighted route-only "
                "observables, compact weighted route-only transforms, or "
                "compact route-anchor interactions; "
                f"unsupported route expression around {sorted(unsupported_routes)}"
            )
        too_large = [
            {"term": term, "coefficient": coeff}
            for term, coeff in coeffs
            if abs(coeff) > config.route_coeff_cap
        ]
        if too_large:
            return (
                f"replacement route coefficient exceeds cap {config.route_coeff_cap}: "
                f"{too_large}"
            )
    else:
        return f"unknown objective_mode: {config.objective_mode}"
    if config.require_nonzero_routing_term:
        routing_status = _routing_term_status(spec.ast, config)
        if not routing_status["routing_term_gate_passed"]:
            return (
                "routing-aware Tier-2 evolution requires at least one nonzero "
                "deployable routing mechanism; density/wirelength-only variants "
                "are baseline-like and are evaluated as baselines, not LLM candidates"
            )
    if config.reject_baseline_mechanism_clones:
        mechanism_signature = _mechanism_signature(_spec_mechanism_view(spec))
        closest_baseline = _closest_database_baseline_mechanism(
            _spec_mechanism_view(spec), database
        )
        if (
            closest_baseline
            and config.min_baseline_mechanism_distance > 0.0
            and closest_baseline["baseline_mechanism_distance"]
            < config.min_baseline_mechanism_distance
        ):
            return (
                "candidate is too close to a fixed baseline mechanism from "
                f"{closest_baseline['closest_baseline_objective_id']}; "
                "near-clone objective edits are not treated as discovered objectives"
            )
        for program in database.programs.values():
            if not (program.metrics.get("seed_baseline") or program.metrics.get("manual_safe_baseline")):
                continue
            existing_signature = program.metrics.get("mechanism_signature")
            if not existing_signature and program.objective_spec:
                existing_signature = _mechanism_signature(_payload_mechanism_view(program.objective_spec))
            if existing_signature == mechanism_signature:
                return (
                    "candidate repeats a fixed baseline mechanism structure from "
                    f"{program.id}; coefficient-only edits of evaluated baselines "
                    "are not treated as discovered objectives"
                )
    if config.reject_repeated_negative_mechanisms:
        mechanism_signature = _mechanism_signature(_spec_mechanism_view(spec))
        for program in database.programs.values():
            existing_signature = program.metrics.get("mechanism_signature")
            if not existing_signature and program.objective_spec:
                existing_signature = _mechanism_signature(_payload_mechanism_view(program.objective_spec))
            is_negative_memory = program.metrics.get("negative_memory_only") or program.metrics.get(
                "severe_regression_count", 0
            )
            if not is_negative_memory:
                continue
            if existing_signature == mechanism_signature:
                return (
                    "candidate repeats a mechanism structure that is already "
                    f"negative memory from {program.id}; coefficient changes alone "
                    "are not enough to justify another DREAMPlace run"
                )
            if (
                config.min_negative_mechanism_distance > 0.0
                and program.objective_spec
                and not _is_unmeasured_negative_memory(program)
            ):
                existing_ast = program.objective_spec.get("ast")
                if existing_ast:
                    distance = _mechanism_distance(
                        _spec_mechanism_view(spec),
                        _payload_mechanism_view(program.objective_spec),
                    )
                    if distance < config.min_negative_mechanism_distance:
                        return (
                            "candidate is too close to a measured negative or "
                            f"near-miss mechanism from {program.id} "
                            f"(distance={distance:.3f} < "
                            f"{config.min_negative_mechanism_distance:.3f}); "
                            "make a mechanism-level change before another "
                            "DREAMPlace evaluation"
                        )
    if len(spec.term_set) < 2:
        return "replacement objective must use at least two term families"
    ast_signature = _ast_signature(_spec_mechanism_view(spec))
    for program in database.programs.values():
        existing = program.objective_spec or {}
        if _ast_signature(_payload_mechanism_view(existing)) == ast_signature:
            return f"duplicate objective AST of {program.id}"
    return None


def _database_objective_identity(
    database: ObjectiveProgramDatabase,
) -> tuple[set[str], set[str]]:
    objective_ids = set(database.programs)
    ast_signatures: set[str] = set()
    for program in database.programs.values():
        existing = program.objective_spec or {}
        if existing.get("id"):
            objective_ids.add(str(existing["id"]))
        if existing.get("ast"):
            ast_signatures.add(_ast_signature(existing.get("ast")))
    return objective_ids, ast_signatures


def _ast_signature(ast: Any) -> str:
    return json.dumps(ast, sort_keys=True, separators=(",", ":"))


def _is_unmeasured_negative_memory(program: ObjectiveProgram) -> bool:
    """True for negative-memory records that never ran DREAMPlace.

    Static rejections and provider errors carry no measured optimizer
    behavior. They stay in memory for signature-equality dedup and prompt
    lessons, but distance-based mechanism rejection must only consult
    measured failures; otherwise every rejection grows an exclusion ball
    that eventually swallows the whole controller space.
    """

    metrics = program.metrics or {}
    if metrics.get("static_rejection"):
        return True
    lesson = str(metrics.get("feedback_lesson", ""))
    return lesson.startswith("Static rejection before DREAMPlace") or lesson.startswith(
        "Provider output could not be parsed"
    )


def _ast_uses_obs(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    if node.get("op") == "obs":
        return True
    return any(_ast_uses_obs(child) for child in node.get("args", []))


def _mechanism_view(ast: Any, state: Any = None) -> Any:
    """Combined mechanism structure for DSL v2 specs.

    Controllers legitimately share compact score shapes (wl + state * den);
    their mechanism lives in the register init/update expressions. Folding the
    state block into one synthetic node makes signatures, clone rejection, and
    mechanism distances schedule-aware. Register display names are dropped so
    renaming a register does not change the mechanism.
    """

    registers = None
    if isinstance(state, dict):
        registers = state.get("registers")
    if not isinstance(registers, dict) or not registers:
        return ast
    register_nodes = sorted(
        (
            {"op": "register", "args": [entry.get("init"), entry.get("update")]}
            for entry in registers.values()
            if isinstance(entry, dict)
        ),
        key=lambda node: json.dumps(node, sort_keys=True, separators=(",", ":")),
    )
    return {"op": "controller_program", "args": [ast, *register_nodes]}


def _spec_mechanism_view(spec: ObjectiveSpec) -> Any:
    return _mechanism_view(spec.ast, getattr(spec, "state", None))


def _payload_mechanism_view(payload: dict[str, Any] | None) -> Any:
    payload = payload or {}
    return _mechanism_view(payload.get("ast"), payload.get("state"))


def _mechanism_signature(ast: Any) -> str:
    """Return an AST signature that ignores numeric coefficient values."""
    return json.dumps(_mechanism_shape(ast), sort_keys=True, separators=(",", ":"))


def _mechanism_shape(node: Any) -> Any:
    if not isinstance(node, dict):
        return node
    op = node.get("op")
    if op == "const":
        return {"op": "const"}
    if op == "term":
        return {"op": "term", "name": node.get("name")}
    if op in {"state", "obs", "net"}:
        return {"op": op, "name": node.get("name")}
    shaped = {
        key: _mechanism_shape(value)
        for key, value in node.items()
        if key != "args"
    }
    args = [_mechanism_shape(arg) for arg in node.get("args", [])]
    if op in {"add", "mul"}:
        args = sorted(args, key=lambda value: json.dumps(value, sort_keys=True))
    shaped["args"] = args
    return shaped


def _canonical_mechanism_term(name: str) -> str:
    if name in {"wirelength", "wirelength_wawl"}:
        return "wirelength_wawl"
    if name in {"density", "density_electric"}:
        return "density_electric"
    return name


def _mechanism_token_set(node: Any) -> set[str]:
    tokens: set[str] = set()

    def visit(current: Any, parent_op: str | None = None) -> str:
        if not isinstance(current, dict):
            token = f"literal:{type(current).__name__}"
            tokens.add(token)
            return token
        op = str(current.get("op"))
        if op == "const":
            token = "const"
            tokens.add(token)
            if parent_op:
                tokens.add(f"edge:{parent_op}->{token}")
            return token
        if op == "term":
            name = _canonical_mechanism_term(str(current.get("name")))
            token = f"term:{name}"
            tokens.add(token)
            if parent_op:
                tokens.add(f"edge:{parent_op}->{token}")
            return token
        if op in {"obs", "net"}:
            token = f"{op}:{current.get('name')}"
            tokens.add(token)
            if parent_op:
                tokens.add(f"edge:{parent_op}->{token}")
            return token
        if op == "state":
            # Register display names are not mechanism content.
            token = "state"
            tokens.add(token)
            if parent_op:
                tokens.add(f"edge:{parent_op}->{token}")
            return token
        tokens.add(f"op:{op}")
        if parent_op:
            tokens.add(f"edge:{parent_op}->op:{op}")
        child_tokens = [visit(child, op) for child in current.get("args", [])]
        if child_tokens:
            if op in {"add", "mul"}:
                child_tokens = sorted(child_tokens)
            tokens.add(f"shape:{op}({','.join(child_tokens)})")
        return f"op:{op}"

    visit(_mechanism_shape(node))
    return tokens


def _mechanism_distance(left: Any, right: Any) -> float:
    left_tokens = _mechanism_token_set(left)
    right_tokens = _mechanism_token_set(right)
    if not left_tokens and not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    similarity = len(left_tokens & right_tokens) / len(union)
    return 1.0 - similarity


def _closest_database_baseline_mechanism(
    ast: Any,
    database: ObjectiveProgramDatabase,
) -> dict[str, Any] | None:
    candidates = []
    for program in database.programs.values():
        if not (program.metrics.get("seed_baseline") or program.metrics.get("manual_safe_baseline")):
            continue
        existing = program.objective_spec or {}
        if not existing.get("ast"):
            continue
        candidates.append((program.id, _payload_mechanism_view(existing)))
    return _closest_mechanism_summary(ast, candidates)


def _baseline_mechanism_summary_from_presets(
    ast: Any,
    config: OpenEvolveTier2Config,
) -> dict[str, Any]:
    names = []
    for name in [*config.initial_presets, *config.search_baseline_presets]:
        if name not in names:
            names.append(name)
    candidates = []
    for name in names:
        try:
            preset = objective_preset(name)
        except KeyError:
            continue
        candidates.append((name, _spec_mechanism_view(preset)))
    summary = _closest_mechanism_summary(ast, candidates)
    return summary or {}


def _closest_mechanism_summary(
    ast: Any,
    candidates: list[tuple[str, Any]],
) -> dict[str, Any] | None:
    best: tuple[float, str, Any] | None = None
    for objective_id, candidate_ast in candidates:
        distance = _mechanism_distance(ast, candidate_ast)
        if best is None or distance < best[0]:
            best = (distance, objective_id, candidate_ast)
    if best is None:
        return None
    distance, objective_id, _candidate_ast = best
    return {
        "closest_baseline_objective_id": objective_id,
        "baseline_mechanism_distance": distance,
        "baseline_mechanism_similarity": 1.0 - distance,
    }


def _route_coefficients(ast: dict[str, Any]) -> tuple[list[tuple[str, float]], set[str]]:
    coefficients: list[tuple[str, float]] = []
    unsupported: set[str] = set()

    def visit(node: dict[str, Any], multiplier: float = 1.0) -> None:
        op = node.get("op")
        if op == "term":
            name = str(node.get("name"))
            if name in ROUTE_CORRECTION_TERMS:
                coefficients.append((name, multiplier))
            return
        if op == "const":
            return
        args = [arg for arg in node.get("args", []) if isinstance(arg, dict)]
        if op == "mul":
            const_product = multiplier
            non_constants: list[dict[str, Any]] = []
            for arg in args:
                if arg.get("op") == "const":
                    const_product *= float(arg.get("value", 1.0))
                else:
                    non_constants.append(arg)
            if len(non_constants) == 1:
                visit(non_constants[0], const_product)
                return
            if non_constants and all(_is_route_only_expression(arg) for arg in non_constants):
                for arg in non_constants:
                    visit(arg, const_product)
                return
            if _is_route_interaction_factors(non_constants):
                route_terms = sorted(
                    _terms_in_ast({"op": "mul", "args": non_constants})
                    & ROUTE_CORRECTION_TERMS
                )
                coefficients.append(
                    (
                        "interaction:" + "+".join(route_terms),
                        const_product,
                    )
                )
                return
            if any(_contains_route_term(arg) for arg in non_constants):
                unsupported.update(_terms_in_ast({"op": "mul", "args": non_constants}) & ROUTE_CORRECTION_TERMS)
            return
        if op in {"add", "sub"}:
            if op == "add":
                for arg in args:
                    visit(arg, multiplier)
            elif args:
                visit(args[0], multiplier)
                for arg in args[1:]:
                    visit(arg, -multiplier)
            return
        if op in {"log1p", "sqrt", "square", "softplus", "sigmoid"} and len(args) == 1:
            if _is_route_only_expression(args[0]):
                visit(args[0], multiplier)
            elif _contains_route_term(args[0]):
                unsupported.update(_terms_in_ast(node) & ROUTE_CORRECTION_TERMS)
            return
        if any(_contains_route_term(arg) for arg in args):
            unsupported.update(_terms_in_ast(node) & ROUTE_CORRECTION_TERMS)

    visit(ast)
    return coefficients, unsupported


def _uses_mutually_exclusive_wirelength_terms(term_set: list[str]) -> bool:
    terms = set(term_set)
    return "wirelength_lse" in terms and bool({"wirelength", "wirelength_wawl"} & terms)


def _uses_mutually_exclusive_density_terms(term_set: list[str]) -> bool:
    terms = set(term_set)
    return "density_bell" in terms and bool({"density", "density_electric"} & terms)


def _calibration_combinations(
    *,
    route_coeffs: list[float | None],
    density_coeffs: list[float | None],
    wirelength_coeffs: list[float | None],
    max_count: int,
) -> list[tuple[float | None, float | None, float | None]]:
    pairs = [
        (route, density, wirelength)
        for route in route_coeffs
        for density in density_coeffs
        for wirelength in wirelength_coeffs
    ]
    route_order = _signed_magnitude_order(route_coeffs)
    density_order = sorted(
        density_coeffs,
        key=lambda value: (
            abs(float(value or 1.0) - 1.28),
            abs(float(value or 1.0) - 1.25),
            float(value or 1.0),
        ),
    )
    wirelength_order = sorted(
        wirelength_coeffs,
        key=lambda value: (
            abs(float(value or 1.0) - 1.0),
            float(value or 1.0),
        ),
    )
    ordered_pairs = [
        (route, density, wirelength)
        for route in route_order
        for density in density_order
        for wirelength in wirelength_order
    ]
    if len(ordered_pairs) <= max_count:
        return ordered_pairs
    selected: list[tuple[float | None, float | None, float | None]] = []
    seen: set[tuple[float | None, float | None, float | None]] = set()

    def add(pair: tuple[float | None, float | None, float | None]) -> None:
        if len(selected) >= max_count:
            return
        if pair in pairs and pair not in seen:
            selected.append(pair)
            seen.add(pair)

    # First isolate the LLM-proposed routing mechanism at the baseline
    # wirelength/density scale. Previous runs showed that immediately mixing in
    # density retuning can create overflow wins that are actually density-heavy
    # HPWL regressions, hiding whether the route term itself is useful.
    for route in route_order:
        if len(selected) >= max_count:
            break
        add((route, 1.0 if 1.0 in density_coeffs else density_order[0], 1.0 if 1.0 in wirelength_coeffs else wirelength_order[0]))

    if len(selected) >= max_count:
        return selected

    # Then test a small number of density/wirelength retunes, still ordered by
    # route-coefficient diversity so the batch is not dominated by one sign.
    for route in route_order:
        for density in density_order:
            for wirelength in wirelength_order:
                if len(selected) >= max_count:
                    break
                add((route, density, wirelength))
            if len(selected) >= max_count:
                break
        if len(selected) >= max_count:
            break

    fallback = sorted(
        pairs,
        key=lambda item: (
            abs(float(item[0] or 0.0)),
            abs(float(item[1] or 1.0) - 1.28),
            abs(float(item[2] or 1.0) - 1.0),
            float(item[1] or 1.0),
            float(item[2] or 1.0),
        ),
    )
    for pair in fallback:
        if len(selected) >= max_count:
            break
        if pair not in seen:
            selected.append(pair)
            seen.add(pair)
    return selected


def _signed_magnitude_order(values: list[float | None]) -> list[float | None]:
    values_by_key = {float(value or 0.0): value for value in values}
    ordered_keys: list[float] = []
    if 0.0 in values_by_key:
        ordered_keys.append(0.0)
    magnitudes = sorted({abs(key) for key in values_by_key if key != 0.0})
    for magnitude in magnitudes:
        if magnitude in values_by_key:
            ordered_keys.append(magnitude)
        if -magnitude in values_by_key:
            ordered_keys.append(-magnitude)
    return [values_by_key[key] for key in ordered_keys]


def _calibrated_spec(
    spec: ObjectiveSpec,
    *,
    route_coeff: float | None,
    density_coeff: float | None,
    wirelength_coeff: float | None,
) -> ObjectiveSpec:
    if density_coeff is not None and math.isclose(float(density_coeff), 1.0, rel_tol=0.0, abs_tol=1e-15):
        density_coeff = None
    if wirelength_coeff is not None and math.isclose(float(wirelength_coeff), 1.0, rel_tol=0.0, abs_tol=1e-15):
        wirelength_coeff = None
    calibrated_ast = _calibrate_terms(
        spec.ast,
        route_coeff=route_coeff,
        density_coeff=density_coeff,
        wirelength_coeff=wirelength_coeff,
    )
    calibrated_terms = sorted(_terms_in_ast(calibrated_ast))
    calibrated_components = {
        str(name): copy.deepcopy(component_ast)
        for name, component_ast in list((spec.components or {}).items())[:5]
        if str(name) != "routing_correction"
    }
    routing_component = _routing_correction_component_ast(calibrated_ast)
    if routing_component is not None:
        calibrated_components["routing_correction"] = routing_component
    payload = {
        "id": "",
        "rationale": (
            f"{spec.rationale} Calibrated coefficients for Tier-2 "
            f"search: route={route_coeff}, density={density_coeff}, "
            f"wirelength={wirelength_coeff}."
        ),
        "parent_ids": [spec.id],
        "declared_term_usage": calibrated_terms,
        "ast": calibrated_ast,
        "components": calibrated_components or None,
    }
    term_scope = (
        "dreamplace_native_residual"
        if "native_objective" in calibrated_terms
        else "dreamplace_replacement"
    )
    for component_name, component_ast in calibrated_components.items():
        parse_objective_spec(
            {
                "id": "",
                "rationale": f"validated component {component_name}",
                "parent_ids": [],
                "declared_term_usage": sorted(_terms_in_ast(component_ast)),
                "ast": component_ast,
            },
            created_by=f"{spec.created_by}:calibration-component",
            term_scope=term_scope,
        )
    parsed = parse_objective_spec(
        payload,
        created_by=f"{spec.created_by}:calibration",
        term_scope=term_scope,
    )
    return replace(parsed, components=calibrated_components or None)


def _routing_correction_component_ast(
    ast: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract the scored subtree through which routing observables act.

    The component is diagnostic and is evaluated by DREAMPlace with the same
    normalization and coefficient as the score AST. For additive objectives we
    exclude pure wirelength/density children; for multiplicative or otherwise
    coupled objectives the full route-containing expression is retained.
    """

    if not (_terms_in_ast(ast) & ROUTE_CORRECTION_TERMS):
        return None
    if ast.get("op") != "add":
        return copy.deepcopy(ast)
    route_children = [
        copy.deepcopy(child)
        for child in ast.get("args", [])
        if isinstance(child, dict)
        and (_terms_in_ast(child) & ROUTE_CORRECTION_TERMS)
    ]
    if not route_children:
        return None
    if len(route_children) == 1:
        return route_children[0]
    return {"op": "add", "args": route_children}


def _calibrate_terms(
    node: dict[str, Any],
    *,
    route_coeff: float | None,
    density_coeff: float | None,
    wirelength_coeff: float | None,
) -> dict[str, Any]:
    op = node.get("op")
    args = [arg for arg in node.get("args", []) if isinstance(arg, dict)]
    if route_coeff is not None and op == "mul":
        non_constants = [arg for arg in args if arg.get("op") != "const"]
        if len(non_constants) == 1 and _is_route_interaction_expression(non_constants[0]):
            return {
                "op": "mul",
                "args": [
                    {"op": "const", "value": float(route_coeff)},
                    _strip_route_interaction_coefficients(non_constants[0]),
                ],
            }
    if route_coeff is not None and _is_route_interaction_expression(node):
        return {
            "op": "mul",
            "args": [
                {"op": "const", "value": float(route_coeff)},
                _strip_route_interaction_coefficients(node),
            ],
        }
    if route_coeff is not None and _is_route_expression(node):
        return {
            "op": "mul",
            "args": [
                {"op": "const", "value": float(route_coeff)},
                _strip_route_expression_coefficients(node),
            ],
        }
    if op == "term" and str(node.get("name")) in ROUTE_CORRECTION_TERMS:
        if route_coeff is None:
            return copy.deepcopy(node)
        return {
            "op": "mul",
            "args": [
                {"op": "const", "value": float(route_coeff)},
                copy.deepcopy(node),
            ],
        }
    if op == "term" and str(node.get("name")) in DENSITY_FAMILY_TERMS:
        if density_coeff is None:
            return copy.deepcopy(node)
        return {
            "op": "mul",
            "args": [
                {"op": "const", "value": float(density_coeff)},
                copy.deepcopy(node),
            ],
        }
    if op == "term" and str(node.get("name")) in WIRELENGTH_FAMILY_TERMS:
        if wirelength_coeff is None:
            return copy.deepcopy(node)
        return {
            "op": "mul",
            "args": [
                {"op": "const", "value": float(wirelength_coeff)},
                copy.deepcopy(node),
            ],
        }
    if op in {"term", "const"}:
        return copy.deepcopy(node)
    if op == "mul" and any(
        (terms := _terms_in_ast(arg)) and terms <= DENSITY_FAMILY_TERMS
        for arg in args
    ):
        non_constants = [arg for arg in args if arg.get("op") != "const"]
        if len(non_constants) == 1:
            return _calibrate_terms(
                non_constants[0],
                route_coeff=route_coeff,
                density_coeff=density_coeff,
                wirelength_coeff=wirelength_coeff,
            )
    if op == "mul" and any(
        (terms := _terms_in_ast(arg)) and terms <= WIRELENGTH_FAMILY_TERMS
        for arg in args
    ):
        non_constants = [arg for arg in args if arg.get("op") != "const"]
        if len(non_constants) == 1:
            return _calibrate_terms(
                non_constants[0],
                route_coeff=route_coeff,
                density_coeff=density_coeff,
                wirelength_coeff=wirelength_coeff,
            )
    return {
        "op": op,
        "args": [
            _calibrate_terms(
                arg,
                route_coeff=route_coeff,
                density_coeff=density_coeff,
                wirelength_coeff=wirelength_coeff,
            )
            for arg in args
        ],
    }


def _is_route_expression(node: dict[str, Any]) -> bool:
    op = node.get("op")
    if op == "term":
        return str(node.get("name")) in ROUTE_CORRECTION_TERMS
    if op in {"log1p", "sqrt", "square", "softplus", "sigmoid"}:
        args = [arg for arg in node.get("args", []) if isinstance(arg, dict)]
        return len(args) == 1 and _is_route_only_expression(args[0])
    if op in {"add", "sub"}:
        args = [arg for arg in node.get("args", []) if isinstance(arg, dict)]
        return bool(args) and all(_is_route_only_expression(arg) for arg in args)
    if op == "mul":
        args = [arg for arg in node.get("args", []) if isinstance(arg, dict)]
        non_constants = [arg for arg in args if arg.get("op") != "const"]
        return bool(non_constants) and all(_is_route_only_expression(arg) for arg in non_constants)
    return False


def _is_route_only_expression(node: dict[str, Any]) -> bool:
    terms = _terms_in_ast(node)
    return bool(terms) and terms <= ROUTE_CORRECTION_TERMS and _is_route_expression(node)


def _is_route_interaction_expression(node: dict[str, Any]) -> bool:
    if node.get("op") != "mul":
        return False
    args = [arg for arg in node.get("args", []) if isinstance(arg, dict)]
    non_constants = [arg for arg in args if arg.get("op") != "const"]
    return _is_route_interaction_factors(non_constants)


def _is_route_interaction_factors(factors: list[dict[str, Any]]) -> bool:
    if len(factors) < 2:
        return False
    has_route = False
    has_anchor = False
    for factor in factors:
        if _is_route_only_expression(factor):
            has_route = True
            continue
        if _is_anchor_only_expression(factor):
            has_anchor = True
            continue
        return False
    return has_route and has_anchor


def _is_anchor_only_expression(node: dict[str, Any]) -> bool:
    terms = _terms_in_ast(node)
    anchor_terms = WIRELENGTH_FAMILY_TERMS | DENSITY_FAMILY_TERMS
    if not terms or not terms <= anchor_terms:
        return False
    op = node.get("op")
    if op == "term":
        return str(node.get("name")) in anchor_terms
    if op == "const":
        return True
    args = [arg for arg in node.get("args", []) if isinstance(arg, dict)]
    if op in {"log1p", "sqrt", "square", "softplus", "sigmoid"}:
        return len(args) == 1 and _is_anchor_only_expression(args[0])
    if op in {"add", "sub", "mul"}:
        return bool(args) and all(
            arg.get("op") == "const" or _is_anchor_only_expression(arg)
            for arg in args
        )
    if op == "safe_div" and len(args) == 2:
        return _is_anchor_only_expression(args[0]) and (
            args[1].get("op") == "const" or _is_anchor_only_expression(args[1])
        )
    return False


def _strip_route_interaction_coefficients(node: dict[str, Any]) -> dict[str, Any]:
    """Remove direct scalar wrappers from a route-anchor interaction.

    The replacement coefficient is supplied by calibration. Anchor factors are
    preserved, while route-only factors have their own scalar wrappers removed so
    the calibrated coefficient is applied once to the interaction term.
    """

    if node.get("op") != "mul":
        return copy.deepcopy(node)
    args = [arg for arg in node.get("args", []) if isinstance(arg, dict)]
    stripped_args = []
    for arg in args:
        if arg.get("op") == "const":
            continue
        if _is_route_only_expression(arg):
            stripped_args.append(_strip_route_expression_coefficients(arg))
        else:
            stripped_args.append(copy.deepcopy(arg))
    if len(stripped_args) == 1:
        return stripped_args[0]
    return {"op": "mul", "args": stripped_args}


def _strip_route_expression_coefficients(node: dict[str, Any]) -> dict[str, Any]:
    """Remove scalar wrappers used only to weight a route-expression component."""
    op = node.get("op")
    if op == "mul":
        args = [arg for arg in node.get("args", []) if isinstance(arg, dict)]
        non_constants = [arg for arg in args if arg.get("op") != "const"]
        if len(non_constants) == 1 and _is_route_expression(non_constants[0]):
            return _strip_route_expression_coefficients(non_constants[0])
        if non_constants and all(_is_route_only_expression(arg) for arg in non_constants):
            return {
                "op": "mul",
                "args": [_strip_route_expression_coefficients(arg) for arg in non_constants],
            }
    if op in {"log1p", "sqrt", "square", "softplus", "sigmoid"}:
        args = [arg for arg in node.get("args", []) if isinstance(arg, dict)]
        if len(args) == 1:
            return {
                "op": op,
                "args": [_strip_route_expression_coefficients(args[0])],
            }
    if op in {"add", "sub"}:
        args = [arg for arg in node.get("args", []) if isinstance(arg, dict)]
        return {
            "op": op,
            "args": [_strip_route_expression_coefficients(arg) for arg in args],
        }
    return copy.deepcopy(node)


def _contains_route_term(node: dict[str, Any]) -> bool:
    return bool(_terms_in_ast(node) & ROUTE_CORRECTION_TERMS)


def _terms_in_ast(node: dict[str, Any]) -> set[str]:
    terms: set[str] = set()

    def visit(current: dict[str, Any]) -> None:
        if current.get("op") == "term":
            terms.add(str(current.get("name")))
            return
        for child in current.get("args", []):
            if isinstance(child, dict):
                visit(child)

    visit(node)
    return terms


def _write_baseline_presets(names: list[str], output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        path = write_objective_spec(objective_preset(name), output_dir / f"{_safe_name(name)}.json")
        paths.append(str(path))
    return paths


def _build_report(summary: dict[str, Any], database: ObjectiveProgramDatabase) -> str:
    generated = [
        program
        for program in database.programs.values()
        if not (program.metrics.get("seed_baseline") or program.metrics.get("manual_safe_baseline"))
    ]
    generated_accepted = [program for program in generated if program.status == "accepted"]
    baseline_like_rejections = [
        program
        for program in generated
        if "baseline mechanism" in str(program.failure_reason or program.metrics.get("feedback_lesson") or "")
        or "baseline-like" in str(program.failure_reason or program.metrics.get("feedback_lesson") or "")
    ]
    portfolio_winners = [
        program
        for program in generated_accepted
        if bool(program.metrics.get("beats_baseline_portfolio"))
    ]
    pareto_dominators = [
        program
        for program in generated_accepted
        if bool(program.metrics.get("baseline_portfolio_pareto_dominates_best"))
    ]
    promotion_pareto_dominators = [
        program
        for program in pareto_dominators
        if bool(program.metrics.get("beats_baseline_portfolio"))
        and bool(program.metrics.get("parent_eligible"))
        and not bool(program.metrics.get("negative_memory_only"))
        and _finite_number(program.metrics.get("combined_score")) is not None
        and float(program.metrics.get("combined_score") or 0.0) > 0.0
    ]
    rank_tradeoffs = [
        program
        for program in generated_accepted
        if bool(program.metrics.get("baseline_portfolio_rank_beats_best"))
        and bool(program.metrics.get("baseline_portfolio_pareto_tradeoff_vs_best"))
    ]
    unique_mechanisms = {
        str(program.metrics.get("mechanism_signature") or "")
        for program in generated_accepted
        if program.metrics.get("mechanism_signature")
    }
    lines = [
        "# OpenEvolve-Style Tier-2 DREAMPlace Report",
        "",
        f"- Run directory: {summary['run_dir']}",
        f"- Program count: {summary['program_count']}",
        f"- Archive size: {summary['archive_size']}",
        f"- Best program: {summary.get('best_program_id')}",
        f"- Robust best generated program: {summary.get('robust_best_program_id')}",
        f"- Aggregate best generated program: {summary.get('aggregate_best_generated_program_id')}",
        f"- Final Tier-2 enabled: {summary.get('final_tier2') is not None}",
        f"- Held-out generalization enabled: {summary.get('generalization_tier2') is not None}",
        f"- Final Tier-3 enabled: {summary.get('final_tier3') is not None}",
        f"- Term-scale audit artifact: {summary.get('term_scale_audit') or 'not enabled'}",
        f"- Term calibration artifact: {summary.get('term_calibration') or 'not enabled'}",
        "- Parent selection score: constrained `combined_score`; rank is diagnostic.",
        "- API memory policy: stateless provider calls; local OpenEvolve memory is explicitly inserted into saved prompt messages.",
        "- LLM objective policy: resolved by config. Replacement runs use explicit placement observables; native-residual runs preserve `term(\"native_objective\")` and add routing-aware corrections.",
        "",
        "## Methodology Audit",
        "",
        f"- Generated programs recorded: {len(generated)}",
        f"- Accepted generated programs: {len(generated_accepted)}",
        f"- Unique accepted mechanism signatures: {len(unique_mechanisms)}",
        f"- Generated programs beating the fixed baseline portfolio: {len(portfolio_winners)}",
        f"- Generated programs with raw Pareto improvement over the strongest fixed baseline: {len(pareto_dominators)}",
        f"- Promotion-worthy generated Pareto wins over the strongest fixed baseline: {len(promotion_pareto_dominators)}",
        f"- Generated rank-based baseline wins that are Pareto tradeoffs: {len(rank_tradeoffs)}",
        f"- Baseline-like static rejections: {len(baseline_like_rejections)}",
        "",
        "## Robust Best Status",
        "",
    ]
    robust_metrics = summary.get("robust_best_program_metrics")
    aggregate_metrics = summary.get("aggregate_best_generated_program_metrics")
    pareto_metrics = summary.get("pareto_best_generated_program_metrics")
    if robust_metrics:
        lines.append(
            "- Robust generated objective found: "
            f"{summary.get('robust_best_program_id')} "
            f"score={robust_metrics.get('combined_score')} "
            f"hpwl_delta={robust_metrics.get('hpwl_delta_pct')} "
            f"overflow_delta={robust_metrics.get('overflow_delta_pct')} "
            f"worst_design_hpwl_delta={robust_metrics.get('worst_design_hpwl_delta_pct')} "
            f"worst_design_overflow_delta={robust_metrics.get('worst_design_overflow_delta_pct')} "
            f"beats_fixed_baselines={robust_metrics.get('beats_baseline_portfolio')} "
            f"baseline_pareto_label={robust_metrics.get('baseline_portfolio_pareto_label_vs_best')} "
            f"hpwl_vs_best={robust_metrics.get('baseline_portfolio_hpwl_delta_vs_best_pct')} "
            f"overflow_vs_best={robust_metrics.get('baseline_portfolio_overflow_delta_vs_best_pct')}"
        )
    else:
        lines.append(
            "- No generated objective currently satisfies the cross-design robust "
            "gate. Aggregate winners remain useful memory but are not treated as "
            "robust discoveries."
        )
    if aggregate_metrics:
        lines.append(
            "- Best aggregate generated objective: "
            f"{summary.get('aggregate_best_generated_program_id')} "
            f"score={aggregate_metrics.get('combined_score')} "
            f"hpwl_delta={aggregate_metrics.get('hpwl_delta_pct')} "
            f"overflow_delta={aggregate_metrics.get('overflow_delta_pct')} "
            f"worst_design_hpwl_delta={aggregate_metrics.get('worst_design_hpwl_delta_pct')} "
            f"worst_design_overflow_delta={aggregate_metrics.get('worst_design_overflow_delta_pct')} "
            f"robust_gate_passed={aggregate_metrics.get('robust_gate_passed')}"
        )
    if pareto_metrics:
        lines.append(
            "- Best Pareto generated objective: "
            f"{summary.get('pareto_best_generated_program_id')} "
            f"score={pareto_metrics.get('combined_score')} "
            f"hpwl_delta={pareto_metrics.get('hpwl_delta_pct')} "
            f"overflow_delta={pareto_metrics.get('overflow_delta_pct')} "
            f"best_fixed_baseline={pareto_metrics.get('baseline_portfolio_best_objective_id')} "
            f"hpwl_vs_best={pareto_metrics.get('baseline_portfolio_hpwl_delta_vs_best_pct')} "
            f"overflow_vs_best={pareto_metrics.get('baseline_portfolio_overflow_delta_vs_best_pct')} "
            f"beats_fixed_baselines={pareto_metrics.get('beats_baseline_portfolio')} "
            f"parent_eligible={pareto_metrics.get('parent_eligible')}"
        )
    else:
        lines.append(
            "- No generated objective currently has a promotion-worthy Pareto win "
            "over the strongest fixed baseline. Tiny or blocked raw Pareto "
            "measurements remain listed below as memory, not as discoveries."
        )
    lines.extend([
        "",
        "## MAP-Elites And Island Memory",
        "",
    ])
    map_summary = database.map_elites_summary()
    lines.append(
        "- MAP-Elites enabled: "
        f"{map_summary.get('enabled')} occupied_cells_total={map_summary.get('occupied_cells_total')} "
        f"occupied_cells_by_island={map_summary.get('occupied_cells_by_island')}"
    )
    if map_summary.get("migration_events"):
        lines.append("- Recent migration events:")
        for event in map_summary.get("migration_events", [])[-10:]:
            lines.append(
                "  - "
                f"iteration={event.get('iteration')} "
                f"{event.get('source_island')}->{event.get('target_island')} "
                f"migrated={event.get('migrated')}/{event.get('requested')}"
            )
    else:
        lines.append("- No migration events recorded.")
    lines.extend([
        "",
        "## Top Programs",
    ])
    for program in database.top_programs(10):
        metrics = program.metrics
        native_label = _native_default_label_from_metrics(metrics)
        lines.append(
            "- "
            f"{program.id}: score={metrics.get('combined_score')} "
            f"hpwl_delta={metrics.get('hpwl_delta_pct')} "
            f"overflow_delta={metrics.get('overflow_delta_pct')} "
            f"native_hpwl_delta={metrics.get('native_default_hpwl_delta_pct')} "
            f"native_overflow_delta={metrics.get('native_default_overflow_delta_pct')} "
            f"native_label={native_label} "
            f"rank_vs_fixed_baselines={metrics.get('baseline_portfolio_candidate_rank')} "
            f"rank_win_fixed_baseline={metrics.get('baseline_portfolio_rank_beats_best')} "
            f"promotion_win_fixed_baselines={metrics.get('beats_baseline_portfolio')} "
            f"best_fixed_baseline={metrics.get('baseline_portfolio_best_objective_id')} "
            f"baseline_pareto_label={metrics.get('baseline_portfolio_pareto_label_vs_best')} "
            f"closest_baseline={metrics.get('closest_baseline_objective_id')} "
            f"baseline_distance={metrics.get('baseline_mechanism_distance')} "
            f"terms={'+'.join(program.term_set) or 'none'} "
            f"role={_report_program_role(program)} "
            f"status={program.status}"
        )
    lines.extend(["", "## Best Non-Baseline Candidates"])
    best_generated = sorted(
        generated_accepted,
        key=lambda program: (
            float(program.metrics.get("combined_score") or 0.0),
            -int(program.metrics.get("structural_failure_count", 0) or 0),
            -int(program.metrics.get("severe_regression_count", 0) or 0),
        ),
        reverse=True,
    )[:10]
    if best_generated:
        for program in best_generated:
            metrics = program.metrics
            native_label = _native_default_label_from_metrics(metrics)
            lines.append(
                "- "
                f"{program.id}: score={metrics.get('combined_score')} "
                f"hpwl_delta={metrics.get('hpwl_delta_pct')} "
                f"overflow_delta={metrics.get('overflow_delta_pct')} "
                f"native_hpwl_delta={metrics.get('native_default_hpwl_delta_pct')} "
                f"native_overflow_delta={metrics.get('native_default_overflow_delta_pct')} "
                f"native_label={native_label} "
                f"rank_vs_fixed_baselines={metrics.get('baseline_portfolio_candidate_rank')} "
                f"rank_win_fixed_baseline={metrics.get('baseline_portfolio_rank_beats_best')} "
                f"promotion_win_fixed_baselines={metrics.get('beats_baseline_portfolio')} "
                f"baseline_pareto_label={metrics.get('baseline_portfolio_pareto_label_vs_best')} "
                f"hpwl_vs_best={metrics.get('baseline_portfolio_hpwl_delta_vs_best_pct')} "
                f"overflow_vs_best={metrics.get('baseline_portfolio_overflow_delta_vs_best_pct')} "
                f"closest_baseline={metrics.get('closest_baseline_objective_id')} "
                f"baseline_distance={metrics.get('baseline_mechanism_distance')} "
                f"mechanism_terms={metrics.get('mechanism_terms') or program.term_set} "
                f"parent_eligible={metrics.get('parent_eligible')} "
                f"outcome={metrics.get('outcome_label') or program.status}"
            )
    else:
        lines.append("- None accepted.")
    lines.extend(["", "## Baseline Portfolio Comparison"])
    if generated_accepted:
        for program in best_generated:
            metrics = program.metrics
            best_baseline = metrics.get("baseline_portfolio_best_objective_id")
            if not metrics.get("baseline_portfolio_compared"):
                continue
            lines.append(
                "- "
                f"{program.id}: candidate_rank={metrics.get('baseline_portfolio_candidate_rank')}, "
                f"best_fixed_baseline={best_baseline}, "
                f"best_baseline_hpwl_delta={metrics.get('baseline_portfolio_best_hpwl_delta_pct')}, "
                f"best_baseline_overflow_delta={metrics.get('baseline_portfolio_best_overflow_delta_pct')}, "
                f"hpwl_vs_best={metrics.get('baseline_portfolio_hpwl_delta_vs_best_pct')}, "
                f"overflow_vs_best={metrics.get('baseline_portfolio_overflow_delta_vs_best_pct')}, "
                f"pareto_label={metrics.get('baseline_portfolio_pareto_label_vs_best')}, "
                f"rank_win={metrics.get('baseline_portfolio_rank_beats_best')}, "
                f"promotion_win={metrics.get('beats_baseline_portfolio')}"
            )
    else:
        lines.append("- No accepted generated candidates to compare.")
    lines.extend(
        [
            "",
            "## Native DREAMPlace Default Comparison",
            "",
            "Negative deltas improve over native default. This section is separate from the custom-default and fixed-baseline portfolio comparison.",
        ]
    )
    if best_generated:
        for program in best_generated:
            metrics = program.metrics
            native_label = _native_default_label_from_metrics(metrics)
            lines.append(
                "- "
                f"{program.id}: "
                f"native_label={native_label}, "
                f"native_hpwl_delta={metrics.get('native_default_hpwl_delta_pct')}, "
                f"native_overflow_delta={metrics.get('native_default_overflow_delta_pct')}, "
                f"beats_native_pareto={native_label == 'dominates_native_default'}, "
                f"custom_default_hpwl_delta={metrics.get('custom_default_hpwl_delta_pct')}, "
                f"custom_default_overflow_delta={metrics.get('custom_default_overflow_delta_pct')}"
            )
    else:
        lines.append("- No accepted generated candidates with native-default comparison data.")
    lines.extend(["", "## Recent Failures"])
    failures = database.recent_failures(10)
    if failures:
        for program in failures:
            lines.append(f"- {program.id}: {program.failure_reason or program.metrics}")
    else:
        lines.append("- None recorded.")
    return "\n".join(lines).rstrip() + "\n"


def _report_program_role(program: Any) -> str:
    metrics = program.metrics
    if metrics.get("seed_baseline") or metrics.get("manual_safe_baseline"):
        return "fixed_baseline"
    if metrics.get("negative_memory_only"):
        return "negative_memory"
    if metrics.get("parent_eligible"):
        return "generated_parent_candidate"
    return "generated_evaluated_candidate"


def _native_default_label_from_metrics(metrics: dict[str, Any]) -> str:
    label = metrics.get("native_default_comparison_label")
    if label:
        return str(label)
    return _native_default_comparison_label(
        metrics.get("native_default_hpwl_delta_pct"),
        metrics.get("native_default_overflow_delta_pct"),
    )


def _load_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return {"warning": "feedback file not found", "path": str(candidate)}
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"payload": payload}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _mean_finite(values: Any) -> float | None:
    finite = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            finite.append(numeric)
    return sum(finite) / len(finite) if finite else None


def _design_delta_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_design: dict[str, list[tuple[float | None, float | None]]] = {}
    cell_pairs: list[tuple[float | None, float | None]] = []
    for row in rows:
        hpwl = _finite_number(row.get("hpwl_delta_pct"))
        overflow = _finite_number(row.get("overflow_delta_pct"))
        if hpwl is None and overflow is None:
            continue
        design = str(row.get("design") or "unknown")
        by_design.setdefault(design, []).append((hpwl, overflow))
        cell_pairs.append((hpwl, overflow))

    per_design = []
    for design, pairs in sorted(by_design.items()):
        hpwl_values = [hpwl for hpwl, _overflow in pairs if hpwl is not None]
        overflow_values = [overflow for _hpwl, overflow in pairs if overflow is not None]
        hpwl_mean = sum(hpwl_values) / len(hpwl_values) if hpwl_values else None
        overflow_mean = sum(overflow_values) / len(overflow_values) if overflow_values else None
        per_design.append(
            {
                "design": design,
                "hpwl_delta_pct": hpwl_mean,
                "overflow_delta_pct": overflow_mean,
                "cell_count": len(pairs),
            }
        )

    design_count = len(per_design)
    design_hpwl_wins = sum(
        1
        for item in per_design
        if item["hpwl_delta_pct"] is not None and item["hpwl_delta_pct"] <= 0.0
    )
    design_overflow_wins = sum(
        1
        for item in per_design
        if item["overflow_delta_pct"] is not None and item["overflow_delta_pct"] <= 0.0
    )
    design_both_wins = sum(
        1
        for item in per_design
        if item["hpwl_delta_pct"] is not None
        and item["overflow_delta_pct"] is not None
        and item["hpwl_delta_pct"] <= 0.0
        and item["overflow_delta_pct"] <= 0.0
    )
    cell_both_wins = sum(
        1
        for hpwl, overflow in cell_pairs
        if hpwl is not None and overflow is not None and hpwl <= 0.0 and overflow <= 0.0
    )
    hpwl_design_values = [
        item["hpwl_delta_pct"] for item in per_design if item["hpwl_delta_pct"] is not None
    ]
    overflow_design_values = [
        item["overflow_delta_pct"]
        for item in per_design
        if item["overflow_delta_pct"] is not None
    ]
    return {
        "per_design_deltas": per_design,
        "worst_design_hpwl_delta_pct": max(hpwl_design_values) if hpwl_design_values else None,
        "worst_design_overflow_delta_pct": (
            max(overflow_design_values) if overflow_design_values else None
        ),
        "design_hpwl_improvement_fraction": (
            design_hpwl_wins / design_count if design_count else 0.0
        ),
        "design_overflow_improvement_fraction": (
            design_overflow_wins / design_count if design_count else 0.0
        ),
        "design_both_improvement_fraction": (
            design_both_wins / design_count if design_count else 0.0
        ),
        "cell_both_improvement_fraction": (
            cell_both_wins / len(cell_pairs) if cell_pairs else 0.0
        ),
    }


def _rows_with_primary_baseline(
    rows: list[dict[str, Any]],
    *,
    primary_baseline: str,
) -> list[dict[str, Any]]:
    by_key = {
        (str(row.get("design")), int(row.get("seed") or 0), str(row.get("objective_id"))): row
        for row in rows
    }
    adjusted = []
    for row in rows:
        enriched = dict(row)
        for baseline_name, prefix in (
            ("default", "native_default"),
            ("custom_default", "custom_default"),
        ):
            baseline_row = by_key.get(
                (
                    str(row.get("design")),
                    int(row.get("seed") or 0),
                    baseline_name,
                )
            )
            if baseline_row is None:
                continue
            hpwl_delta = _delta_pct(row.get("hpwl"), baseline_row.get("hpwl"))
            overflow_delta = _delta_pct(row.get("overflow"), baseline_row.get("overflow"))
            if hpwl_delta is not None:
                enriched[f"{prefix}_hpwl_delta_pct"] = hpwl_delta
            if overflow_delta is not None:
                enriched[f"{prefix}_overflow_delta_pct"] = overflow_delta
        if not primary_baseline:
            adjusted.append(enriched)
            continue
        baseline = by_key.get(
            (
                str(row.get("design")),
                int(row.get("seed") or 0),
                primary_baseline,
            )
        )
        if baseline is not None:
            enriched["baseline_objective_id"] = primary_baseline
            hpwl_delta = _delta_pct(row.get("hpwl"), baseline.get("hpwl"))
            overflow_delta = _delta_pct(row.get("overflow"), baseline.get("overflow"))
            if hpwl_delta is not None:
                enriched["hpwl_delta_pct"] = hpwl_delta
            if overflow_delta is not None:
                enriched["overflow_delta_pct"] = overflow_delta
        adjusted.append(enriched)
    return adjusted


def _delta_pct(value: Any, baseline: Any) -> float | None:
    current = _finite_number(value)
    base = _finite_number(baseline)
    if current is None or base is None or base == 0:
        return None
    return 100.0 * (current - base) / abs(base)


def _constrained_score(
    *,
    average_rank: float,
    hpwl_delta_pct: float | None,
    overflow_delta_pct: float | None,
    hpwl_gate_passed: bool,
    overflow_gate_passed: bool,
    robust_gate_passed: bool,
    require_robust_gate: bool,
    design_both_improvement_fraction: float,
    structural_failure: bool,
    hpwl_gate_search_pct: float,
) -> float:
    if structural_failure or not hpwl_gate_passed or not overflow_gate_passed:
        return 0.0
    if require_robust_gate and not robust_gate_passed:
        return 0.0
    rank_component = 1.0 / (1.0 + float(average_rank))
    overflow_reward = max(0.0, -float(overflow_delta_pct or 0.0)) / 100.0
    hpwl_delta = float(hpwl_delta_pct or 0.0)
    hpwl_reward = max(0.0, -hpwl_delta) / 100.0
    if hpwl_gate_search_pct > 0.0:
        hpwl_margin = max(0.0, hpwl_gate_search_pct - max(0.0, hpwl_delta))
        hpwl_reward += 0.001 * hpwl_margin / max(hpwl_gate_search_pct, 1e-9)
    robustness_reward = 0.001 * max(0.0, min(1.0, float(design_both_improvement_fraction)))
    return max(1e-9, hpwl_reward + overflow_reward + robustness_reward + 0.001 * rank_component)


def _feedback_lesson(
    *,
    structural_failure: bool,
    hpwl_gate_passed: bool,
    overflow_gate_passed: bool,
    robust_gate_passed: bool,
    design_summary: dict[str, Any],
    hpwl_delta_pct: float | None,
    overflow_delta_pct: float | None,
    elite_gate_passed: bool,
) -> str:
    if structural_failure:
        return "DREAMPlace could not evaluate this candidate structurally; keep it as failure memory."
    if not hpwl_gate_passed:
        return (
            "Execution produced a large wirelength/HPWL cost relative to the comparison "
            "baseline. Treat this as negative memory and explore a different mechanism "
            "or scale."
        )
    if not overflow_gate_passed:
        return (
            "Execution kept wirelength/HPWL within the configured search range, but "
            "overflow worsened. Treat this as negative memory for the routing mechanism."
        )
    if not robust_gate_passed:
        return (
            "Mean metrics passed, but per-design robustness failed; the candidate's "
            "benefit is not consistent across the panel. "
            f"Worst-design HPWL delta={design_summary.get('worst_design_hpwl_delta_pct')}, "
            f"worst-design overflow delta={design_summary.get('worst_design_overflow_delta_pct')}, "
            f"both-improved design fraction={design_summary.get('design_both_improvement_fraction')}."
        )
    if elite_gate_passed:
        return "Execution metrics satisfy the current promotion policy; eligible for elite feedback."
    if overflow_delta_pct is not None and overflow_delta_pct > 0:
        return "Overflow regressed in execution; the routing-related mechanism needs a different shape or scale."
    if hpwl_delta_pct is not None and hpwl_delta_pct > 0:
        return "Wirelength/HPWL regressed in execution, but not enough to trigger structural rejection."
    return "Candidate produced useful execution feedback under the current evaluator."


def _mean_outcome_label(rows: list[dict[str, Any]]) -> str | None:
    labels = [str(row.get("outcome_label")) for row in rows if row.get("outcome_label")]
    if not labels:
        return None
    counts = {label: labels.count(label) for label in set(labels)}
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _finite_number(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _active_baseline_delta(
    metrics: dict[str, Any],
    config: OpenEvolveTier2Config,
    *,
    metric: str,
) -> float | None:
    if config.primary_baseline == "custom_default":
        value = _finite_number(metrics.get(f"custom_default_{metric}_delta_pct"))
        return value if value is not None else _finite_number(metrics.get(f"{metric}_delta_pct"))
    if config.primary_baseline == "default":
        value = _finite_number(metrics.get(f"native_default_{metric}_delta_pct"))
        return value if value is not None else _finite_number(metrics.get(f"{metric}_delta_pct"))
    return _finite_number(metrics.get(f"{metric}_delta_pct"))


def _active_worst_design_delta(
    metrics: dict[str, Any],
    config: OpenEvolveTier2Config,
    *,
    metric: str,
) -> float | None:
    if metrics.get("per_design_delta_baseline") != config.primary_baseline:
        return _active_baseline_delta(metrics, config, metric=metric)
    if config.primary_baseline != "custom_default":
        return _finite_number(metrics.get(f"worst_design_{metric}_delta_pct"))
    values = []
    for item in metrics.get("per_design_deltas") or []:
        if not isinstance(item, dict):
            continue
        value = _finite_number(item.get(f"{metric}_delta_pct"))
        if value is not None:
            values.append(value)
    return max(values) if values else _finite_number(metrics.get(f"worst_design_{metric}_delta_pct"))


def _stored_design_summary_for_active_baseline(
    rows: Any,
    config: OpenEvolveTier2Config,
) -> dict[str, Any] | None:
    if not isinstance(rows, list) or not rows:
        return None
    adjusted = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        active = dict(row)
        if config.primary_baseline == "custom_default":
            hpwl = _finite_number(row.get("custom_default_hpwl_delta_pct"))
            overflow = _finite_number(row.get("custom_default_overflow_delta_pct"))
        elif config.primary_baseline == "default":
            hpwl = _finite_number(row.get("native_default_hpwl_delta_pct"))
            overflow = _finite_number(row.get("native_default_overflow_delta_pct"))
            if hpwl is None and row.get("baseline_objective_id") == "default":
                hpwl = _finite_number(row.get("hpwl_delta_pct"))
            if overflow is None and row.get("baseline_objective_id") == "default":
                overflow = _finite_number(row.get("overflow_delta_pct"))
        elif row.get("baseline_objective_id") == config.primary_baseline:
            hpwl = _finite_number(row.get("hpwl_delta_pct"))
            overflow = _finite_number(row.get("overflow_delta_pct"))
        else:
            hpwl = None
            overflow = None
        if hpwl is None and overflow is None:
            continue
        if hpwl is not None:
            active["hpwl_delta_pct"] = hpwl
        if overflow is not None:
            active["overflow_delta_pct"] = overflow
        active["baseline_objective_id"] = config.primary_baseline
        adjusted.append(active)
    if not adjusted:
        return None
    return _design_delta_summary(adjusted)


def _active_severe_regression_count(
    *,
    hpwl_delta_pct: float | None,
    overflow_delta_pct: float | None,
    worst_design_hpwl_delta_pct: float | None,
    worst_design_overflow_delta_pct: float | None,
    severe_threshold_pct: float,
    overflow_severe_threshold_pct: float,
    structural_failure: bool,
) -> int:
    if structural_failure:
        return 0
    hpwl_values = [hpwl_delta_pct, worst_design_hpwl_delta_pct]
    overflow_values = [overflow_delta_pct, worst_design_overflow_delta_pct]
    hpwl_severe = any(
        value is not None
        and math.isfinite(float(value))
        and float(value) > severe_threshold_pct
        for value in hpwl_values
    )
    overflow_severe = any(
        value is not None
        and math.isfinite(float(value))
        and float(value) > overflow_severe_threshold_pct
        for value in overflow_values
    )
    return int(hpwl_severe or overflow_severe)


def _write_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
