from pathlib import Path

from coevop.eval.openevolve_tier2 import (
    build_openevolve_prompt_dry_run,
    load_openevolve_tier2_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "openevolve_tier2"


def test_primary_config_uses_complete_controller_loop():
    config = load_openevolve_tier2_config(CONFIG_DIR / "chipbench_controller_tier2.toml")

    assert config.objective_mode == "controller"
    assert config.term_scope == "dreamplace_controller"
    assert config.primary_baseline == "default"
    assert config.max_iterations == 160
    assert config.samples_per_iteration == 3
    assert config.num_islands == 5
    assert config.map_elites_enabled is True
    assert config.selection_policy == "pareto_multiobjective"
    assert config.min_state_registers >= 1
    assert config.timing_proxy_enabled is True
    assert config.timing_proxy_audit_required is True
    assert config.tier3_enabled is True
    assert config.tier3_top_k == 3
    assert config.tier3_interval == 20
    assert config.tier3_start_iteration == 20
    assert Path(config.tier3_panel).name == "chipbench_table1_post_route.toml"


def test_all_supplied_evolution_configs_use_controller_interface():
    configs = sorted(CONFIG_DIR.glob("*.toml"))
    assert configs

    for path in configs:
        config = load_openevolve_tier2_config(path)
        assert config.objective_mode == "controller", path.name
        assert config.term_scope == "dreamplace_controller", path.name
        assert "dreamplace_controller_native_identity" in config.initial_presets, path.name


def test_primary_controller_prompt_dry_run(tmp_path: Path):
    payload = build_openevolve_prompt_dry_run(
        config_path=CONFIG_DIR / "chipbench_controller_tier2.toml",
        output_dir=tmp_path / "prompt",
    )

    assert payload["prompt_audit"]["passed"] is True
    prompt = Path(payload["prompt_messages"]).read_text(encoding="utf-8")
    assert "def init_policy(obs):" in prompt
    assert "def update_policy(policy, obs):" in prompt
    assert "def objective(features, policy):" in prompt
    assert "def objective(features):" not in prompt
    assert "CircuitNet" not in prompt
    assert "Tier-1" not in prompt
    assert "Eureka-style" not in prompt
