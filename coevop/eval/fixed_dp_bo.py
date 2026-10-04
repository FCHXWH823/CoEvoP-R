"""Fixed-DP schedule BO: the control that keeps DREAMPlace's objective form.

The loss stays wirelength plus a scheduled density weight, exactly the form of
the native DREAMPlace objective, and only the coefficients of its density and
smoothing schedules are searched. A tree-structured Parzen estimator proposes
the coefficients. Every proposal receives the cost-scaled evaluation of the
objective evolution it is compared against, under the same budget: the same
generations, proposals per generation, and scheduled routed evaluations. The
estimator is fit to the archive's Pareto evidence order, so routed evidence
guides the search in the same way it guides objective evolution.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from coevop.eval.openevolve_tier2 import (
    _evaluate_search_panel,
    _is_generated_candidate,
    _load_timing_admission,
    _refresh_pareto_selection,
    _run_scheduled_tier3_feedback,
    _run_scheduled_timing_proxy_audit,
    _seed_initial_programs,
    _timing_admission_path,
    _validate_scoring_iteration_budgets,
    _write_json,
    load_openevolve_tier2_config,
)
from coevop.evolution.openevolve_core import (
    ObjectiveDatabaseConfig,
    ObjectiveProgram,
    ObjectiveProgramDatabase,
)
from coevop.objectives.spec import ObjectiveSpec, parse_objective_spec, write_objective_spec

# Coefficient -> (low, high, log-uniform). DREAMPlace's own schedule sits inside
# every range: density_init 8e-5, ramp [0.95, 1.05], smoothing from 10x down
# to about 0.06x of the base gamma.
SCHEDULE_SPACE: dict[str, tuple[float, float, bool]] = {
    "density_init": (1e-5, 1e-3, True),
    "density_ramp_floor": (0.90, 1.00, False),
    "density_ramp_span": (0.02, 0.20, False),
    "density_ramp_gain": (2e2, 2e4, True),
    "gamma_floor": (0.06, 1.0, True),
    "gamma_ceiling": (1.0, 10.0, True),
    "gamma_slope": (2.0, 20.0, False),
    "gamma_center": (0.1, 0.9, False),
}

# (schedule coefficients, archive score) of every evaluated proposal.
History = list[tuple[dict[str, float], float]]
Proposer = Callable[[History, int, int], list[dict[str, float]]]


def fixed_dp_schedule_spec(params: dict[str, float]) -> ObjectiveSpec:
    """DREAMPlace's objective form with the given schedule coefficients."""

    def const(value: float) -> dict[str, Any]:
        return {"op": "const", "value": float(value)}

    def obs(name: str) -> dict[str, Any]:
        return {"op": "obs", "name": name}

    density_weight = {"op": "state", "name": "density_weight"}
    payload = {
        "id": "",
        "rationale": "Fixed DREAMPlace objective form with tuned schedule coefficients.",
        "parent_ids": [],
        "declared_term_usage": ["density_electric", "wirelength_wawl"],
        "ast": {
            "op": "add",
            "args": [
                {"op": "term", "name": "wirelength_wawl"},
                {"op": "mul", "args": [density_weight, {"op": "term", "name": "density_electric"}]},
            ],
        },
        "components": {
            "wirelength": {"op": "term", "name": "wirelength_wawl"},
            "density": {"op": "term", "name": "density_electric"},
        },
        "state": {
            "interface": "typed_policy_v1",
            "control": "generated",
            "registers": {
                # Gradient-calibrated start, then a multiplicative ramp that
                # grows faster while HPWL is still improving.
                "density_weight": {
                    "init": {
                        "op": "mul",
                        "args": [
                            const(params["density_init"]),
                            obs("grad_ratio_density_electric"),
                        ],
                    },
                    "update": {
                        "op": "mul",
                        "args": [
                            density_weight,
                            {
                                "op": "add",
                                "args": [
                                    const(params["density_ramp_floor"]),
                                    {
                                        "op": "mul",
                                        "args": [
                                            const(params["density_ramp_span"]),
                                            {
                                                "op": "sigmoid",
                                                "args": [
                                                    {
                                                        "op": "mul",
                                                        "args": [
                                                            const(-params["density_ramp_gain"]),
                                                            obs("hpwl_delta_rate"),
                                                        ],
                                                    }
                                                ],
                                            },
                                        ],
                                    },
                                ],
                            },
                        ],
                    },
                },
                # Smoothing anneals from the ceiling to the floor as overflow falls.
                "gamma_scale": {
                    "init": const(params["gamma_ceiling"]),
                    "update": {
                        "op": "add",
                        "args": [
                            const(params["gamma_floor"]),
                            {
                                "op": "mul",
                                "args": [
                                    const(params["gamma_ceiling"] - params["gamma_floor"]),
                                    {
                                        "op": "sigmoid",
                                        "args": [
                                            {
                                                "op": "mul",
                                                "args": [
                                                    const(params["gamma_slope"]),
                                                    {
                                                        "op": "sub",
                                                        "args": [
                                                            obs("overflow"),
                                                            const(params["gamma_center"]),
                                                        ],
                                                    },
                                                ],
                                            }
                                        ],
                                    },
                                ],
                            },
                        ],
                    },
                },
            },
        },
    }
    return parse_objective_spec(payload, created_by="fixed_dp_schedule_bo", term_scope="dreamplace")


def propose_with_tpe(history: History, count: int, seed: int) -> list[dict[str, float]]:
    """Ask a tree-structured Parzen estimator for the next schedule coefficients."""

    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Fixed-DP schedule BO requires Optuna; install it with "
            "pip install 'coevop-platform[baselines]'"
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    distributions = {
        name: optuna.distributions.FloatDistribution(low, high, log=log)
        for name, (low, high, log) in SCHEDULE_SPACE.items()
    }
    # The study is rebuilt each generation because archive scores are ordinal:
    # new evidence, routed evidence in particular, re-ranks earlier proposals.
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, constant_liar=True),
    )
    for params, score in history:
        study.add_trial(
            optuna.trial.create_trial(params=params, distributions=distributions, value=score)
        )
    return [dict(study.ask(distributions).params) for _ in range(count)]


def run_fixed_dp_schedule_bo(
    *,
    config_path: str | Path,
    run_dir: str | Path,
    dreamplace_root: str | Path,
    chipbench_root: str | Path | None = None,
    resume: bool = False,
    retry_failed: bool = False,
    proposer: Proposer = propose_with_tpe,
) -> dict[str, Any]:
    # One population and no LLM: the evolution configuration only supplies the
    # panels, the evidence policy, and the budget to match.
    config = replace(load_openevolve_tier2_config(config_path), num_islands=1)
    scoring_iteration_policy = _validate_scoring_iteration_budgets(config)
    if config.tier3_enabled and config.tier3_interval > 0 and chipbench_root is None:
        raise RuntimeError("scheduled routed evaluation requires chipbench_root")

    run_root = Path(run_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    database = ObjectiveProgramDatabase(
        run_root / "program_db",
        ObjectiveDatabaseConfig(
            population_size=max(
                config.population_size,
                config.max_iterations * config.samples_per_iteration + len(config.initial_presets),
            ),
            archive_size=config.archive_size,
            num_islands=1,
            map_elites_enabled=config.map_elites_enabled,
            feature_bins=config.feature_bins,
            feature_dimensions=tuple(config.feature_dimensions),
            timing_proxy_wns_delta_min_abs_ns=config.timing_proxy_wns_delta_min_abs_ns,
            migration_interval=0,
        ),
    )
    timing_admission = _load_timing_admission(config, run_root)
    if resume:
        database.load()
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
    if not _timing_admission_path(run_root).is_file():
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

    samples = max(1, int(config.samples_per_iteration))
    for generation in range(database.last_iteration + 1, config.max_iterations + 1):
        proposals = proposer(_history(database), samples, config.seed + generation)
        programs = []
        for index, params in enumerate(proposals):
            spec = fixed_dp_schedule_spec(params)
            sample_dir = run_root / f"iteration_{generation:04d}" / f"sample_{index:04d}"
            metrics, artifacts, status, failure_reason = _evaluate_search_panel(
                spec=spec,
                objective_path=write_objective_spec(spec, sample_dir / "objective.json"),
                config=config,
                iteration_dir=sample_dir,
                dreamplace_root=dreamplace_root,
                resume=resume,
                retry_failed=retry_failed,
            )
            program = ObjectiveProgram.from_spec(
                spec,
                program_id=f"iter_{generation:04d}_sample_{index:04d}_{spec.id}",
                generation=generation,
                iteration_found=generation,
                metrics=metrics,
                artifacts={**artifacts, "schedule_parameters": params},
                status=status,
                failure_reason=failure_reason,
            )
            database.add(program, target_island=0)
            programs.append(program)
        _refresh_pareto_selection(database, config, timing_admission)
        scheduled_tier3 = _run_scheduled_tier3_feedback(
            config=config,
            database=database,
            programs=programs,
            iteration=generation,
            run_root=run_root,
            chipbench_root=chipbench_root,
            resume=resume,
            retry_failed=retry_failed,
        )
        scheduled_audit = _run_scheduled_timing_proxy_audit(
            config=config,
            database=database,
            iteration=generation,
            run_root=run_root,
            dreamplace_root=dreamplace_root,
        )
        if scheduled_audit is not None:
            timing_admission = scheduled_audit
        if scheduled_tier3 is not None or scheduled_audit is not None:
            _refresh_pareto_selection(database, config, timing_admission)
        database.save(iteration=generation)

    trials = [program for program in database.programs.values() if _is_schedule_trial(program)]
    best = max(trials, key=lambda program: (program.combined_score, program.id), default=None)
    if best is not None and best.objective_spec:
        _write_json(best.objective_spec, run_root / "best_objective.json")
    summary = {
        "run_dir": str(run_root),
        "config": str(config_path),
        "scoring_iteration_policy": scoring_iteration_policy,
        "generations": config.max_iterations,
        "proposals_per_generation": samples,
        "proposal_count": len(trials),
        "valid_count": sum(
            1
            for program in trials
            if program.status == "accepted" and not program.metrics.get("structural_failure_count")
        ),
        "archive_count": sum(1 for program in trials if program.id in database.archive),
        "routed_count": sum(1 for program in trials if program.metrics.get("tier_c_evaluated")),
        "best_program_id": best.id if best is not None else None,
        "best_schedule_parameters": (
            best.artifacts.get("schedule_parameters") if best is not None else None
        ),
        "best_program_metrics": best.metrics if best is not None else None,
        "program_db": str(database.root),
    }
    _write_json(summary, run_root / "summary.json")
    return summary


def _is_schedule_trial(program: ObjectiveProgram) -> bool:
    return _is_generated_candidate(program) and isinstance(
        program.artifacts.get("schedule_parameters"), dict
    )


def _history(database: ObjectiveProgramDatabase) -> History:
    return [
        (dict(program.artifacts["schedule_parameters"]), program.combined_score)
        for program in sorted(
            database.programs.values(),
            key=lambda program: (program.iteration_found, program.id),
        )
        if _is_schedule_trial(program)
    ]
