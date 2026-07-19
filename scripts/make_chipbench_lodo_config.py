#!/usr/bin/env python3
"""Create a leave-one-design-out CoEvoP&R configuration for ChiPBench."""

from __future__ import annotations

import argparse
from pathlib import Path


DESIGNS = (
    "bp_fe",
    "bp_be",
    "swerv_wrapper",
    "ethernet",
    "dft68",
    "or1200",
    "vga_lcd",
    "mor1kx",
)


def _panel(root: Path, designs: list[str], seeds: tuple[int, ...]) -> str:
    lines = [
        f"seeds = [{', '.join(str(seed) for seed in seeds)}]",
        "iterations = 1000",
        "stop_overflow = 0.0",
        "gpu = 1",
        "timeout_seconds = 7200",
        "log_interval = 10",
        'timing_mode = "none"',
        "",
    ]
    for design in designs:
        config = root / "configs" / "dreamplace_base" / "chipbench_movable" / f"{design}.json"
        lines.extend(
            [
                "[[designs]]",
                f'name = "{design}"',
                f'dreamplace_config = "{config.as_posix()}"',
                'timing_mode = "none"',
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout", required=True, choices=DESIGNS)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    root = args.repo_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    training = [design for design in DESIGNS if design != args.heldout]
    search_panel = output / "search.toml"
    heldout_panel = output / "heldout.toml"
    search_panel.write_text(_panel(root, training, (1000, 1001)), encoding="utf-8")
    heldout_panel.write_text(_panel(root, [args.heldout], (1000, 1001, 1002)), encoding="utf-8")

    timing_panel = root / "configs" / "timing_panels" / "chipbench_proxy_audit.toml"
    route_panel = root / "configs" / "shared_panels" / "chipbench_table1_post_route.toml"
    config = f'''provider = "openai"
term_scope = "dreamplace_controller"
objective_mode = "controller"
mutation_mode = "diff"
minimum_scoring_iterations = 1000
allow_short_search_for_tests = false
max_iterations = 160
samples_per_iteration = 3
population_size = 100
archive_size = 20
num_islands = 5
map_elites_enabled = true
feature_bins = 8
feature_dimensions = ["complexity", "mechanism_family", "hpwl_delta_bucket", "overflow_delta_bucket", "opentimer_wns_delta_bucket"]
migration_interval = 20
migration_rate = 0.10
migration_topology = "ring"
checkpoint_interval = 10
num_top_programs = 3
num_diverse_programs = 2
seed = 42
evaluate_initial_presets = true
primary_baseline = "default"
parent_policy = "pareto_multiobjective"
selection_policy = "pareto_multiobjective"
structured_feedback = true
target_designs = [{', '.join(repr(design) for design in training)}]
min_state_registers = 1
require_nonzero_routing_term = false
max_calibrated_siblings = 0
max_generation_attempts = 4
initial_presets = [
  "dreamplace_controller_native_identity",
  "dreamplace_controller_eplace_smooth",
  "dreamplace_explicit_wl_density",
  "dreamplace_density_135",
]

[timing_proxy]
enabled = true
mode = "gate"
panel = "{timing_panel.as_posix()}"
audit_required = true
max_hpwl_corr_for_gate = 0.70

[search]
panel = "{search_panel.as_posix()}"
disable_legalization = true
baseline_presets = ["dreamplace_controller_native_identity", "dreamplace_controller_eplace_smooth"]

[final]
enabled = false

[generalization]
enabled = true
panel = "{heldout_panel.as_posix()}"
top_k = 5
candidate_policy = "elite_or_aggregate"
blind = true
strict_leakage_guard = true
baseline_presets = ["dreamplace_controller_native_identity", "dreamplace_controller_eplace_smooth"]

[tier3]
enabled = true
panel = "{route_panel.as_posix()}"
top_k = 3
interval = 20
start_iteration = 20
baseline_objective_id = "default"
candidate_policy = "elite_or_aggregate"
'''
    config_path = output / "openevolve.toml"
    config_path.write_text(config, encoding="utf-8")
    print(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
