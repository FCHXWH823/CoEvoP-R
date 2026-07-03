from pathlib import Path

from coevop.eval.openevolve_tier2 import load_openevolve_tier2_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "openevolve_tier2"


def test_current_openevolve_tier2_configs_use_direct_replacement_policy():
    current_configs = [
        path
        for path in CONFIG_DIR.glob("*.toml")
        if "native_residual" not in path.name
    ]
    assert current_configs

    for path in current_configs:
        config = load_openevolve_tier2_config(path)
        assert config.term_scope == "dreamplace_replacement", path.name
        assert config.objective_mode == "replacement", path.name
        assert config.primary_baseline == "default", path.name
        assert not config.primary_baseline.startswith("obj_"), path.name
        assert config.require_nonzero_routing_term is True, path.name
        assert config.min_replacement_nonbaseline_terms >= 2, path.name
        assert config.require_custom_default_gate is False, path.name
        if "timing" in path.name or "design_specific_generalization" in path.name:
            # Timing-driven and design-specific transfer runs use native default
            # as the parent-selection gate and keep the fixed analytical
            # portfolio as reference/final evidence. Forcing every early
            # generated candidate to beat the strongest handcrafted baseline is
            # too strict for dynamic parent-child evolution.
            assert config.require_baseline_portfolio_gate is False, path.name
            assert config.require_baseline_portfolio_pareto is False, path.name
        else:
            assert config.require_baseline_portfolio_gate is True, path.name
            assert config.require_baseline_portfolio_pareto is True, path.name
        assert config.min_baseline_portfolio_effect_pct >= 0.25, path.name
        assert config.min_baseline_behavior_distance_pct >= 0.5, path.name
        assert config.native_default_selection_policy == "prefer_pareto", path.name
        assert config.require_robust_parent_gate is True, path.name
        assert config.allow_near_miss_parents is True, path.name
        assert config.near_miss_max_worst_hpwl_pct <= 25.0, path.name
        assert config.near_miss_max_worst_overflow_pct <= 10.0, path.name
        assert config.near_miss_parent_score_floor > 0.0, path.name
        assert config.reject_baseline_mechanism_clones is True, path.name
        assert config.min_baseline_mechanism_distance >= 0.35, path.name
        assert config.reject_repeated_negative_mechanisms is True, path.name
        assert config.min_negative_mechanism_distance >= 0.55, path.name
        assert config.full_rewrite_after_static_rejections >= 2, path.name
        assert config.max_generation_attempts >= 4, path.name
        assert config.density_coeff_grid == [1.0], path.name
        assert config.wirelength_coeff_grid == [1.0], path.name
        nonzero_route_coeffs = [abs(value) for value in config.route_coeff_grid if value != 0.0]
        assert nonzero_route_coeffs, path.name
        assert min(nonzero_route_coeffs) >= config.min_abs_routing_coeff, path.name
        assert min(nonzero_route_coeffs) <= 1e-8, path.name
        assert max(nonzero_route_coeffs) <= 1e-5, path.name
        assert max(nonzero_route_coeffs) <= config.route_coeff_cap, path.name
        assert config.parent_policy == "hpwl_safe_only", path.name
        assert "dreamplace_explicit_wl_density" in config.search_baseline_presets, path.name
        assert "dreamplace_wawl_electric" in config.search_baseline_presets, path.name
        assert "dreamplace_wawl_density_bell" in config.search_baseline_presets, path.name
        assert "dreamplace_lse_density_bell" in config.search_baseline_presets, path.name


def test_legacy_native_residual_configs_are_explicitly_named_legacy():
    legacy_configs = list(CONFIG_DIR.glob("*native_residual*.toml"))
    assert legacy_configs

    for path in legacy_configs:
        config = load_openevolve_tier2_config(path)
        assert config.objective_mode == "native_residual", path.name
