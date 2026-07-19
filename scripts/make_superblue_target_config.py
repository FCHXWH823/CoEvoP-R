#!/usr/bin/env python3
"""Create a target-design CoEvoP&R configuration for one Superblue circuit."""

from __future__ import annotations

import argparse
from pathlib import Path


DESIGNS = ("1", "3", "4", "5", "7", "10", "16", "18")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", required=True, choices=DESIGNS)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    root = args.repo_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    name = f"superblue{args.design}_ot_notiming"
    base = root / "configs" / "dreamplace_base" / "iccad2015_notiming" / f"{name}.json"
    route_panel = root / "configs" / "shared_panels" / "superblue_post_route.toml"

    def panel(seeds: tuple[int, ...], timeout: int) -> str:
        return (
            f"seeds = [{', '.join(str(seed) for seed in seeds)}]\n"
            "iterations = 1000\nstop_overflow = 0.0\ngpu = 1\n"
            f"timeout_seconds = {timeout}\nlog_interval = 10\ntiming_mode = \"none\"\n\n"
            "[[designs]]\n"
            f"name = \"{name}\"\n"
            f"dreamplace_config = \"{base.as_posix()}\"\n"
            "timing_mode = \"none\"\n"
        )

    search_panel = output / "search.toml"
    final_panel = output / "final.toml"
    search_panel.write_text(panel((1000, 1001), 7200), encoding="utf-8")
    final_panel.write_text(panel((1000, 1001, 1002), 7200), encoding="utf-8")

    config = f'''provider = "openai"
term_scope = "dreamplace_controller"
objective_mode = "controller"
mutation_mode = "diff"
minimum_scoring_iterations = 1000
allow_short_search_for_tests = false
max_iterations = 80
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

[search]
panel = "{search_panel.as_posix()}"
disable_legalization = true
baseline_presets = ["dreamplace_controller_native_identity", "dreamplace_controller_eplace_smooth"]

[final]
enabled = true
panel = "{final_panel.as_posix()}"
disable_legalization = false
top_k = 5
baseline_presets = ["dreamplace_controller_native_identity", "dreamplace_controller_eplace_smooth"]

[tier3]
enabled = true
panel = "{route_panel.as_posix()}"
top_k = 3
interval = 10
start_iteration = 10
baseline_objective_id = "default"
candidate_policy = "elite_or_aggregate"
'''
    config_path = output / "openevolve.toml"
    config_path.write_text(config, encoding="utf-8")
    print(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
