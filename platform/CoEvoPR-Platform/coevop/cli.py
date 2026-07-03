"""Command-line entrypoints for the CoEvoP&R platform."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from coevop.backends.dreamplace import (
    apply_patch as apply_dreamplace_patch,
    extract_custom_objective_lines,
    is_patch_applied,
    make_run_config,
    run_dreamplace,
    write_run_summary,
)
from coevop.config import load_config
from coevop.datasets.circuitnet import (
    build_manifest,
    load_manifest,
    manifest_to_dict,
    write_manifest,
)
from coevop.datasets.summaries import compute_summary_rows, write_summary_csv
from coevop.eval.tier1 import (
    evaluate_spec,
    rank_objectives,
    write_rankings_csv,
    write_rankings_json,
)
from coevop.eval.real_tier1 import write_real_tier1_audit
from coevop.eval.end_to_end import run_end_to_end
from coevop.eval.eureka_tier2 import run_eureka_tier2
from coevop.eval.chipbench_generalization import (
    run_chipbench_final_eval_from_existing,
    run_chipbench_generalization,
)
from coevop.eval.openevolve_tier2 import (
    audit_openevolve_prompt_artifacts,
    build_openevolve_prompt_dry_run,
    run_openevolve_tier2,
)
from coevop.eval.shared_panel import check_shared_panel
from coevop.eval.tier2_dreamplace import run_tier2_dreamplace
from coevop.eval.tier3_openroad import recover_tier3_partial_run, run_tier3_openroad
from coevop.eval.timing_proxy import run_timing_proxy_audit, run_timing_proxy_eval
from coevop.evolution.offline import run_offline_evolution, write_top_candidates
from coevop.evolution.memory import parse_operator_weights
from coevop.llm.providers import provider_from_name, provider_status
from coevop.objectives.presets import objective_preset, preset_names
from coevop.objectives.program import parse_objective_program
from coevop.objectives.spec import load_objective_spec, objective_spec_to_dict, write_objective_spec
from coevop.objectives.terms import term_names, unsupported_terms_for_scope
from coevop.store.sqlite import SQLiteStore


def _add_common_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="configs/default.toml",
        help="Path to the platform TOML config.",
    )


def _cmd_env_check(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    payload = {
        "circuitnet_root": str(config.circuitnet_root) if config.circuitnet_root else None,
        "dreamplace_root": str(config.dreamplace_root),
        "chipbench_root": str(config.chipbench_root),
        "run_root": str(config.run_root),
        "llm_provider": config.llm_provider,
        "openroad_backend": config.openroad_backend,
        "exists": {
            "circuitnet_root": config.circuitnet_root.exists() if config.circuitnet_root else False,
            "dreamplace_root": config.dreamplace_root.exists(),
            "chipbench_root": config.chipbench_root.exists(),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_circuitnet_manifest(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    root = Path(args.root) if args.root else config.circuitnet_root
    if root is None:
        raise SystemExit("CIRCUITNET_ROOT is not set and --root was not provided")

    manifest = build_manifest(root, designs=args.design)
    if args.output:
        output = write_manifest(manifest, args.output, pretty=not args.compact)
        print(f"Wrote manifest: {output}")
    else:
        print(json.dumps(manifest_to_dict(manifest), indent=None if args.compact else 2))
    return 0


def _cmd_tier1_offline(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    rows = compute_summary_rows(manifest, max_samples=args.max_samples)
    rankings = rank_objectives(rows)

    summary_output = write_summary_csv(rows, args.summary_output)
    ranking_json_output = write_rankings_json(rankings, len(rows), args.ranking_output)
    ranking_csv_output = write_rankings_csv(rankings, args.ranking_csv_output)

    print(f"Wrote scalar summaries: {summary_output}")
    print(f"Wrote objective rankings: {ranking_json_output}")
    print(f"Wrote objective ranking table: {ranking_csv_output}")
    if rankings:
        best = rankings[0]
        print(f"Best objective: {best.objective_id} score={best.score:.6g}")
    return 0


def _cmd_tier1_score_spec(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    rows = compute_summary_rows(manifest, max_samples=args.max_samples)
    spec = load_objective_spec(args.objective)
    ranking = evaluate_spec(rows, spec)
    ranking_json_output = write_rankings_json([ranking], len(rows), args.output)
    print(f"Wrote generated-objective score: {ranking_json_output}")
    print(f"Objective {ranking.objective_id} score={ranking.score:.6g}")
    return 0


def _load_json_context(path: str | None) -> dict:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise SystemExit("context JSON must be an object")
    return payload


def _cmd_llm_check(args: argparse.Namespace) -> int:
    print(json.dumps(provider_status(), indent=2, sort_keys=True))
    return 0


def _cmd_llm_generate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    provider_name = args.provider or config.llm_provider
    provider = provider_from_name(provider_name)
    context = _load_json_context(args.context)
    spec = provider.generate(context=context, term_scope=args.term_scope)
    output = write_objective_spec(spec, args.output)
    print(f"Wrote objective spec: {output}")
    print(json.dumps(objective_spec_to_dict(spec), indent=2, sort_keys=True))
    return 0


def _cmd_evolve_offline(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    provider_name = args.provider or config.llm_provider
    run_dir = Path(args.run_dir) if args.run_dir else Path(config.run_root) / "evolution" / args.run_name
    summary = run_offline_evolution(
        manifest_path=args.manifest,
        provider_name=provider_name,
        run_dir=run_dir,
        population_size=args.population_size,
        generations=args.generations,
        elite_count=args.elite_count,
        max_attempts_per_candidate=args.max_attempts_per_candidate,
        seed=args.seed,
        db_path=args.db_path,
        max_samples=args.max_samples,
        split_strategy=args.split_strategy,
        selection_split=args.selection_split,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        heldout_fraction=args.heldout_fraction,
        reflection_mode=args.reflection_mode,
        diversity_lambda=args.diversity_lambda,
        operator_weights=parse_operator_weights(args.operator_weights),
        islands=args.islands,
        enable_constant_fit=args.enable_constant_fit,
        validation_design=args.validation_design,
        heldout_design=args.heldout_design,
        term_scope=args.term_scope,
    )
    print(f"Wrote evolution run: {summary.run_dir}")
    print(f"Wrote candidate database: {summary.db_path}")
    print(f"Accepted candidates: {summary.accepted_candidates}")
    print(f"Rejected candidates: {summary.rejected_candidates}")
    if summary.top_candidates:
        best = summary.top_candidates[0]
        print(f"Best candidate: {best['objective_id']} score={best['score']:.6g}")
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


def _cmd_evolution_top(args: argparse.Namespace) -> int:
    store = SQLiteStore(args.db)
    try:
        top_candidates = store.top_candidates(limit=args.limit, tier=args.tier)
    finally:
        store.close()

    if args.output_dir:
        output_dir = Path(args.output_dir)
        json_path, csv_path = write_top_candidates(
            top_candidates,
            output_dir / "top_candidates.json",
            output_dir / "top_candidates.csv",
        )
        print(f"Wrote top candidates: {json_path}")
        print(f"Wrote top candidate table: {csv_path}")
    else:
        print(json.dumps({"top_candidates": top_candidates}, indent=2, sort_keys=True))
    return 0


def _cmd_real_tier1_audit(args: argparse.Namespace) -> int:
    payload = write_real_tier1_audit(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        split_strategy=args.split_strategy,
        seed=args.seed,
        shuffle_trials=args.shuffle_trials,
        random_formula_count=args.random_formula_count,
        evolution_dir=args.evolution_dir,
        validation_design=args.validation_design,
        heldout_design=args.heldout_design,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_tier2_dreamplace(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    objectives = _split_csv_paths(args.objectives)
    payload = run_tier2_dreamplace(
        panel_path=args.panel,
        objective_paths=objectives,
        dreamplace_root=config.dreamplace_root,
        run_dir=args.run_dir,
        store_path=args.store,
        resume=args.resume,
        retry_failed=args.retry_failed,
        include_default=not args.no_default,
        include_custom_default=not args.no_custom_default,
        require_output_artifact=args.require_output_artifact,
        require_def_output=args.require_def_output,
        disable_legalization=not args.keep_legalization,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_shared_panel_check(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    payload = check_shared_panel(
        panel_path=args.panel,
        dreamplace_root=config.dreamplace_root,
        chipbench_root=config.chipbench_root,
        run_dir=args.run_dir,
        skip_runs=args.skip_runs,
    ).to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


def _cmd_tier3_openroad(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    payload = run_tier3_openroad(
        panel_path=args.panel,
        placements_path=args.placements,
        chipbench_root=config.chipbench_root,
        run_dir=args.run_dir,
        store_path=args.store,
        baseline_objective_id=args.baseline_objective_id,
        resume=args.resume,
        retry_failed=args.retry_failed,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_timing_proxy_audit(args: argparse.Namespace) -> int:
    config = load_config(args.platform_config)
    payload = run_timing_proxy_audit(
        design=args.design,
        base_config=args.base_config,
        placements_path=args.placements,
        dreamplace_root=config.dreamplace_root,
        run_dir=args.run_dir,
        perturbations=args.perturbations,
        timeout_seconds=args.timeout_seconds,
        gpu=args.gpu,
        seed=args.seed,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_timing_proxy_eval(args: argparse.Namespace) -> int:
    config = load_config(args.platform_config)
    payload = run_timing_proxy_eval(
        panel_path=args.panel,
        placements_path=args.placements,
        dreamplace_root=config.dreamplace_root,
        run_dir=args.run_dir,
        resume=args.resume,
        retry_failed=args.retry_failed,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_tier3_recover_openroad(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    payload = recover_tier3_partial_run(
        panel_path=args.panel,
        placements_path=args.placements,
        chipbench_root=config.chipbench_root,
        run_dir=args.run_dir,
        store_path=args.store,
        baseline_objective_id=args.baseline_objective_id,
        failure_stage=args.failure_stage,
        message=args.message,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_evolve_end_to_end(args: argparse.Namespace) -> int:
    config = load_config(args.platform_config)
    payload = run_end_to_end(
        config_path=args.config,
        run_dir=args.run_dir,
        dreamplace_root=config.dreamplace_root,
        chipbench_root=config.chipbench_root,
        resume=args.resume,
        retry_failed=args.retry_failed,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_eureka_tier2(args: argparse.Namespace) -> int:
    config = load_config(args.platform_config)
    payload = run_eureka_tier2(
        config_path=args.config,
        run_dir=args.run_dir,
        dreamplace_root=config.dreamplace_root,
        resume=args.resume,
        retry_failed=args.retry_failed,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_openevolve_tier2(args: argparse.Namespace) -> int:
    config = load_config(args.platform_config)
    payload = run_openevolve_tier2(
        config_path=args.config,
        run_dir=args.run_dir,
        dreamplace_root=config.dreamplace_root,
        chipbench_root=config.chipbench_root,
        resume=args.resume,
        retry_failed=args.retry_failed,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_chipbench_generalization(args: argparse.Namespace) -> int:
    config = load_config(args.platform_config)
    payload = run_chipbench_generalization(
        run_dir=args.run_dir,
        dreamplace_root=config.dreamplace_root,
        chipbench_root=config.chipbench_root,
        provider=args.provider,
        model=args.model,
        designs=args.design,
        max_iterations=args.max_iterations,
        samples_per_iteration=args.samples_per_iteration,
        search_seeds=_parse_int_csv(args.search_seeds),
        final_seeds=_parse_int_csv(args.final_seeds),
        search_iterations=args.search_iterations,
        final_iterations=args.final_iterations,
        gpu=args.gpu,
        timeout_seconds=args.timeout_seconds,
        global_route_args=args.global_route_args,
        resume=args.resume,
        retry_failed=args.retry_failed,
        dry_run=args.dry_run,
        skip_evolution=args.skip_evolution,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_chipbench_final_eval(args: argparse.Namespace) -> int:
    config = load_config(args.platform_config)
    payload = run_chipbench_final_eval_from_existing(
        source_run=args.source_run,
        run_dir=args.run_dir,
        dreamplace_root=config.dreamplace_root,
        chipbench_root=config.chipbench_root,
        designs=args.design,
        seeds=_parse_int_csv(args.seeds),
        iterations=args.iterations,
        gpu=args.gpu,
        timeout_seconds=args.timeout_seconds,
        openroad_timeout_seconds=args.openroad_timeout_seconds,
        global_route_args=args.global_route_args,
        resume=args.resume,
        retry_failed=args.retry_failed,
        allow_best_fallback=args.allow_best_fallback,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_openevolve_prompt_audit(args: argparse.Namespace) -> int:
    payload = audit_openevolve_prompt_artifacts(
        args.run_dir,
        write=not args.no_write,
    )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


def _cmd_openevolve_prompt_dry_run(args: argparse.Namespace) -> int:
    payload = build_openevolve_prompt_dry_run(
        config_path=args.config,
        output_dir=args.output_dir,
        from_run_dir=args.from_run_dir,
        parent_preset=args.parent,
        iteration=args.iteration,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["prompt_audit"]["passed"] else 1


def _cmd_objective_preset(args: argparse.Namespace) -> int:
    if args.list:
        print(json.dumps({"presets": preset_names()}, indent=2, sort_keys=True))
        return 0
    if not args.name:
        raise SystemExit("--name is required unless --list is used")
    spec = objective_preset(args.name)
    output = write_objective_spec(spec, args.output)
    print(f"Wrote objective preset: {output}")
    print(json.dumps(objective_spec_to_dict(spec), indent=2, sort_keys=True))
    return 0


def _split_csv_paths(value: str | None) -> list[str]:
    if not value:
        return []
    return [chunk.strip() for chunk in value.split(",") if chunk.strip()]


def _parse_int_csv(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(chunk.strip()) for chunk in value.split(",") if chunk.strip()]


def _cmd_objective_program(args: argparse.Namespace) -> int:
    source = Path(args.source).read_text(encoding="utf-8")
    spec = parse_objective_program(
        source,
        created_by=args.created_by,
        rationale=args.rationale or "",
        parent_ids=args.parent_id or [],
        term_scope=args.term_scope,
    )
    output = write_objective_spec(spec, args.output)
    print(f"Wrote objective spec: {output}")
    print(json.dumps(objective_spec_to_dict(spec), indent=2, sort_keys=True))
    return 0


def _cmd_dreamplace_status(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    payload = {
        "dreamplace_root": str(config.dreamplace_root),
        "placeobj_patch_applied": is_patch_applied(config.dreamplace_root),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_dreamplace_apply_patch(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    applied = apply_dreamplace_patch(config.dreamplace_root, repo_root=Path.cwd())
    print(json.dumps({"applied": applied, "dreamplace_root": str(config.dreamplace_root)}, indent=2))
    return 0


def _cmd_dreamplace_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.objective:
        spec = load_objective_spec(args.objective)
        unsupported = unsupported_terms_for_scope(spec.term_set, "dreamplace")
        if unsupported:
            raise SystemExit(
                "Objective contains terms that the current DREAMPlace patch cannot execute: "
                f"{unsupported}. Use deployable terms only: {term_names('dreamplace')}."
            )
    if args.objective and not is_patch_applied(config.dreamplace_root):
        raise SystemExit(
            "DREAMPlace custom objective patch is not applied. "
            "Run `python3 -m coevop.cli dreamplace-apply-patch` first."
        )

    base_config = (
        Path(args.base_config)
        if args.base_config
        else config.dreamplace_root / "test" / "ispd2005" / "adaptec1.json"
    )
    run_dir = Path(config.run_root) / "dreamplace" / args.run_name
    dreamplace_config = make_run_config(
        base_config=base_config,
        objective_spec=args.objective,
        run_dir=run_dir,
        iterations=args.iterations,
        gpu=args.gpu,
        custom_objective_log_interval=args.log_interval,
        disable_legalization=not args.keep_legalization,
    )
    run = run_dreamplace(
        dreamplace_root=config.dreamplace_root,
        config_path=dreamplace_config,
        run_name=args.run_name,
        run_dir=run_dir,
        timeout_seconds=args.timeout_seconds,
    )
    summary_path = write_run_summary(run, run_dir / "run_summary.json")
    custom_lines = extract_custom_objective_lines(run.log_path)
    print(f"Wrote DREAMPlace config: {dreamplace_config}")
    print(f"Wrote DREAMPlace log: {run.log_path}")
    print(f"Wrote run summary: {summary_path}")
    print(json.dumps({"returncode": run.returncode, "custom_objective_log_lines": custom_lines[-5:]}, indent=2))
    return run.returncode


def _cmd_dreamplace_term_check(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if not is_patch_applied(config.dreamplace_root):
        raise SystemExit(
            "DREAMPlace custom objective patch is not applied. "
            "Run `python3 -m coevop.cli dreamplace-apply-patch` first."
        )
    requested_terms = [term.strip() for term in args.terms.split(",") if term.strip()]
    allowed_terms = set(term_names("dreamplace"))
    unsupported = sorted(term for term in requested_terms if term not in allowed_terms)
    if unsupported:
        raise SystemExit(
            f"Terms are not DREAMPlace-deployable: {unsupported}. "
            f"Allowed terms: {sorted(allowed_terms)}"
        )
    base_config = Path(args.base_config)
    run_root = Path(args.run_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    objective_dir = run_root / "objectives"
    objective_dir.mkdir(parents=True, exist_ok=True)
    results = []
    all_passed = True
    for term in requested_terms:
        spec = load_objective_spec(
            write_objective_spec(
                objective_preset("dreamplace_single_" + term)
                if "dreamplace_single_" + term in preset_names()
                else _single_term_spec(term),
                objective_dir / f"{term}.json",
            )
        )
        term_dir = run_root / term
        dreamplace_config = make_run_config(
            base_config=base_config,
            objective_spec=objective_dir / f"{term}.json",
            run_dir=term_dir,
            iterations=args.iterations,
            gpu=args.gpu,
            custom_objective_log_interval=args.log_interval,
            disable_legalization=True,
        )
        run = run_dreamplace(
            dreamplace_root=config.dreamplace_root,
            config_path=dreamplace_config,
            run_name=f"term_check_{term}",
            run_dir=term_dir,
            timeout_seconds=args.timeout_seconds,
        )
        write_run_summary(run, term_dir / "run_summary.json")
        custom = run.metrics.get("custom_objective", {})
        status = "passed"
        reason = ""
        if run.returncode != 0:
            status = "failed"
            reason = f"DREAMPlace return code {run.returncode}"
        elif int(custom.get("calls") or 0) <= 0:
            status = "failed"
            reason = "custom objective did not execute"
        elif custom.get("last_grad_norm") is None:
            status = "failed"
            reason = "custom objective gradient norm was not logged"
        elif float(custom.get("last_grad_norm") or 0.0) <= args.min_grad_norm:
            status = "failed"
            reason = "custom objective gradient norm is too small"
        if status != "passed":
            all_passed = False
        results.append(
            {
                "term": term,
                "objective_id": spec.id,
                "status": status,
                "reason": reason,
                "returncode": run.returncode,
                "custom_objective": custom,
                "log_path": run.log_path,
                "config_path": run.config_path,
            }
        )
    payload = {"base_config": str(base_config), "results": results}
    output = run_root / "term_check_summary.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all_passed else 1


def _single_term_spec(term: str):
    from coevop.objectives.spec import parse_objective_spec

    return parse_objective_spec(
        {
            "id": "",
            "rationale": f"Single-term DREAMPlace gradient check for {term}.",
            "parent_ids": [],
            "declared_term_usage": [term],
            "ast": {"op": "term", "name": term},
        },
        created_by="term_check",
        term_scope="dreamplace",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coevop")
    subparsers = parser.add_subparsers(dest="command", required=True)

    env_check = subparsers.add_parser("env-check", help="Print resolved platform paths.")
    _add_common_config_arg(env_check)
    env_check.set_defaults(func=_cmd_env_check)

    manifest = subparsers.add_parser(
        "circuitnet-manifest",
        help="Build a CircuitNet N14 routability manifest.",
    )
    _add_common_config_arg(manifest)
    manifest.add_argument("--root", help="CircuitNet root or CircuitNet-N14 directory.")
    manifest.add_argument(
        "--design",
        action="append",
        help="Design directory to include. May be passed multiple times.",
    )
    manifest.add_argument("--output", help="Path to write manifest JSON.")
    manifest.add_argument("--compact", action="store_true", help="Emit compact JSON.")
    manifest.set_defaults(func=_cmd_circuitnet_manifest)

    tier1 = subparsers.add_parser(
        "tier1-offline",
        help="Summarize CircuitNet maps and rank handwritten offline objectives.",
    )
    tier1.add_argument("--manifest", required=True, help="CircuitNet manifest JSON path.")
    tier1.add_argument(
        "--summary-output",
        default="runs/tier1/scalars.csv",
        help="CSV path for per-sample scalar features and labels.",
    )
    tier1.add_argument(
        "--ranking-output",
        default="runs/tier1/rankings.json",
        help="JSON path for objective rankings.",
    )
    tier1.add_argument(
        "--ranking-csv-output",
        default="runs/tier1/rankings.csv",
        help="CSV path for target-level objective correlations.",
    )
    tier1.add_argument(
        "--max-samples",
        type=int,
        help="Optional maximum number of manifest samples to evaluate.",
    )
    tier1.set_defaults(func=_cmd_tier1_offline)

    tier1_spec = subparsers.add_parser(
        "tier1-score-spec",
        help="Score one validated ObjectiveSpec against CircuitNet Tier-1 labels.",
    )
    tier1_spec.add_argument("--manifest", required=True, help="CircuitNet manifest JSON path.")
    tier1_spec.add_argument("--objective", required=True, help="ObjectiveSpec JSON path.")
    tier1_spec.add_argument(
        "--output",
        default="runs/tier1/generated_objective_score.json",
        help="JSON path for the generated objective score.",
    )
    tier1_spec.add_argument(
        "--max-samples",
        type=int,
        help="Optional maximum number of manifest samples to evaluate.",
    )
    tier1_spec.set_defaults(func=_cmd_tier1_score_spec)

    llm_check = subparsers.add_parser(
        "llm-check",
        help="Print LLM provider readiness without exposing API keys.",
    )
    llm_check.set_defaults(func=_cmd_llm_check)

    llm_generate = subparsers.add_parser(
        "llm-generate",
        help="Generate and validate one objective candidate through an LLM provider.",
    )
    _add_common_config_arg(llm_generate)
    llm_generate.add_argument(
        "--provider",
        choices=["mock", "openai", "qwen"],
        help="Provider to use. Defaults to llm_provider from config.",
    )
    llm_generate.add_argument(
        "--context",
        help="Optional JSON context file, such as a Tier-1 ranking summary.",
    )
    llm_generate.add_argument(
        "--output",
        default="runs/llm/generated_objective.json",
        help="Path to write the validated ObjectiveSpec JSON.",
    )
    llm_generate.add_argument(
        "--term-scope",
        choices=["tier1", "deployable", "dreamplace", "dreamplace_replacement"],
        default="tier1",
        help="Term namespace to expose to the provider.",
    )
    llm_generate.set_defaults(func=_cmd_llm_generate)

    evolve_offline = subparsers.add_parser(
        "evolve-offline",
        help="Run the optional CircuitNet proxy-analysis candidate-evolution loop.",
    )
    _add_common_config_arg(evolve_offline)
    evolve_offline.add_argument("--manifest", required=True, help="CircuitNet manifest JSON path.")
    evolve_offline.add_argument(
        "--provider",
        choices=["mock", "openai", "qwen"],
        help="Provider to use. Defaults to llm_provider from config.",
    )
    evolve_offline.add_argument(
        "--run-name",
        default="offline_evolution",
        help="Run directory name under RUN_ROOT/evolution when --run-dir is omitted.",
    )
    evolve_offline.add_argument("--run-dir", help="Explicit output directory for run artifacts.")
    evolve_offline.add_argument("--db-path", help="Explicit SQLite database path.")
    evolve_offline.add_argument("--population-size", type=int, default=20, help="Candidates per generation.")
    evolve_offline.add_argument("--generations", type=int, default=1, help="Number of generations.")
    evolve_offline.add_argument("--elite-count", type=int, default=5, help="Top candidates to retain.")
    evolve_offline.add_argument(
        "--max-attempts-per-candidate",
        type=int,
        default=3,
        help="Provider attempts before moving to the next candidate slot.",
    )
    evolve_offline.add_argument("--seed", type=int, default=0, help="Evolution RNG seed.")
    evolve_offline.add_argument(
        "--max-samples",
        type=int,
        help="Optional maximum number of manifest samples to evaluate.",
    )
    evolve_offline.add_argument(
        "--split-strategy",
        choices=["auto", "family", "design", "sample", "none"],
        default="auto",
        help="How to split CircuitNet rows before Tier-1 scoring.",
    )
    evolve_offline.add_argument(
        "--validation-design",
        help="Force one design into validation and use remaining designs for train/heldout.",
    )
    evolve_offline.add_argument(
        "--heldout-design",
        help="Force one design into heldout and use remaining designs for train/validation.",
    )
    evolve_offline.add_argument(
        "--selection-split",
        choices=["train", "validation", "heldout"],
        default="validation",
        help="Split score used for candidate ranking.",
    )
    evolve_offline.add_argument(
        "--term-scope",
        choices=["tier1", "deployable"],
        default="tier1",
        help=(
            "Term namespace exposed during proxy evolution. Use deployable for "
            "the stricter DREAMPlace-compatible bridge vocabulary."
        ),
    )
    evolve_offline.add_argument(
        "--train-fraction",
        type=float,
        default=0.6,
        help="Target train fraction for grouped splits.",
    )
    evolve_offline.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Target validation fraction for grouped splits.",
    )
    evolve_offline.add_argument(
        "--heldout-fraction",
        type=float,
        default=0.2,
        help="Target held-out fraction for grouped splits.",
    )
    evolve_offline.add_argument(
        "--reflection-mode",
        choices=["none", "accepted"],
        default="accepted",
        help="Whether to ask the provider for concise reflection on accepted candidates.",
    )
    evolve_offline.add_argument(
        "--diversity-lambda",
        type=float,
        default=0.05,
        help="Penalty for selecting candidates with overlapping term sets.",
    )
    evolve_offline.add_argument(
        "--operator-weights",
        help=(
            "Comma-separated operator weights, e.g. "
            "crossover=0.35,mutate=0.30,param_tune=0.20,simplify=0.15."
        ),
    )
    evolve_offline.add_argument(
        "--islands",
        type=int,
        default=5,
        help="Number of in-memory candidate-memory islands.",
    )
    fit_group = evolve_offline.add_mutually_exclusive_group()
    fit_group.add_argument(
        "--enable-constant-fit",
        dest="enable_constant_fit",
        action="store_true",
        default=True,
        help="Store train-split constant-fitted siblings for accepted candidates.",
    )
    fit_group.add_argument(
        "--disable-constant-fit",
        dest="enable_constant_fit",
        action="store_false",
        help="Disable train-split constant-fitted siblings.",
    )
    evolve_offline.set_defaults(func=_cmd_evolve_offline)

    evolution_top = subparsers.add_parser(
        "evolution-top",
        help="Print or write top candidates from an evolution SQLite database.",
    )
    evolution_top.add_argument("--db", required=True, help="Evolution SQLite database path.")
    evolution_top.add_argument("--limit", type=int, default=10, help="Number of candidates to list.")
    evolution_top.add_argument("--tier", default="tier1", help="Evaluation tier to rank by.")
    evolution_top.add_argument("--output-dir", help="Optional directory for top_candidates JSON/CSV.")
    evolution_top.set_defaults(func=_cmd_evolution_top)

    real_tier1_audit = subparsers.add_parser(
        "real-tier1-audit",
        help="Write real-scale CircuitNet Tier-1 audit, controls, and report artifacts.",
    )
    real_tier1_audit.add_argument("--manifest", required=True, help="CircuitNet manifest JSON path.")
    real_tier1_audit.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write manifest/scalars/audit/control/report artifacts.",
    )
    real_tier1_audit.add_argument(
        "--max-samples",
        type=int,
        help="Optional maximum number of manifest samples to audit.",
    )
    real_tier1_audit.add_argument(
        "--split-strategy",
        choices=["auto", "family", "design", "sample", "none"],
        default="auto",
        help="Split strategy used for baseline-by-split diagnostics.",
    )
    real_tier1_audit.add_argument(
        "--validation-design",
        help="Force one design into validation for baseline-by-split diagnostics.",
    )
    real_tier1_audit.add_argument(
        "--heldout-design",
        help="Force one design into heldout for baseline-by-split diagnostics.",
    )
    real_tier1_audit.add_argument("--seed", type=int, default=0, help="Control RNG seed.")
    real_tier1_audit.add_argument(
        "--shuffle-trials",
        type=int,
        default=32,
        help="Number of shuffled-label negative-control trials.",
    )
    real_tier1_audit.add_argument(
        "--random-formula-count",
        type=int,
        default=32,
        help="Number of random legal DSL formulas for negative controls.",
    )
    real_tier1_audit.add_argument(
        "--evolution-dir",
        help="Optional evolve-offline run directory to summarize top candidates.",
    )
    real_tier1_audit.set_defaults(func=_cmd_real_tier1_audit)

    tier2_dreamplace = subparsers.add_parser(
        "tier2-dreamplace",
        help="Run a DREAMPlace Tier-2 design/objective/seed panel.",
    )
    _add_common_config_arg(tier2_dreamplace)
    tier2_dreamplace.add_argument("--panel", required=True, help="DREAMPlace panel TOML path.")
    tier2_dreamplace.add_argument(
        "--objectives",
        default="",
        help=(
            "Comma-separated ObjectiveSpec JSON paths. Native default and custom_default "
            "are added unless disabled."
        ),
    )
    tier2_dreamplace.add_argument("--run-dir", required=True, help="Tier-2 artifact directory.")
    tier2_dreamplace.add_argument("--store", help="SQLite result store path.")
    tier2_dreamplace.add_argument("--resume", action="store_true", help="Reuse completed run summaries.")
    tier2_dreamplace.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed/incomplete cells instead of reusing their summaries.",
    )
    tier2_dreamplace.add_argument(
        "--no-default",
        action="store_true",
        help="Do not include native DREAMPlace default baseline.",
    )
    tier2_dreamplace.add_argument(
        "--no-custom-default",
        action="store_true",
        help="Do not include custom-default normalization baseline.",
    )
    tier2_dreamplace.add_argument(
        "--require-output-artifact",
        action="store_true",
        help="Mark a completed DREAMPlace run failed if no placement artifact is emitted.",
    )
    tier2_dreamplace.add_argument(
        "--require-def-output",
        action="store_true",
        help="Mark a completed DREAMPlace run failed unless the output artifact is a DEF.",
    )
    tier2_dreamplace.add_argument(
        "--keep-legalization",
        action="store_true",
        help="Keep legalization/detailed placement flags from the base config.",
    )
    tier2_dreamplace.set_defaults(func=_cmd_tier2_dreamplace)

    shared_panel_check = subparsers.add_parser(
        "shared-panel-check",
        help="Validate a shared DREAMPlace/ChiPBench panel.",
    )
    _add_common_config_arg(shared_panel_check)
    shared_panel_check.add_argument("--panel", required=True, help="Shared panel TOML path.")
    shared_panel_check.add_argument(
        "--run-dir",
        default="runs/shared_panel_check",
        help="Directory for panel-check logs and summary.",
    )
    shared_panel_check.add_argument(
        "--skip-runs",
        action="store_true",
        help="Only perform static file/tool checks; skip DREAMPlace and ChiPBench smoke runs.",
    )
    shared_panel_check.set_defaults(func=_cmd_shared_panel_check)

    tier3_openroad = subparsers.add_parser(
        "tier3-openroad",
        help="Route/evaluate Tier-2 DEF placements through ChiPBench/OpenROAD.",
    )
    _add_common_config_arg(tier3_openroad)
    tier3_openroad.add_argument("--panel", required=True, help="Shared panel TOML path.")
    tier3_openroad.add_argument(
        "--placements",
        required=True,
        help="Placements JSON produced from Tier-2 comparison rows.",
    )
    tier3_openroad.add_argument("--run-dir", required=True, help="Tier-3 artifact directory.")
    tier3_openroad.add_argument("--store", help="SQLite result store path.")
    tier3_openroad.add_argument(
        "--baseline-objective-id",
        default="default",
        help="Objective id used as the Tier-3 delta baseline.",
    )
    tier3_openroad.add_argument("--resume", action="store_true", help="Reuse successful run summaries.")
    tier3_openroad.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed/incomplete routed evaluations.",
    )
    tier3_openroad.set_defaults(func=_cmd_tier3_openroad)

    timing_proxy_audit = subparsers.add_parser(
        "timing-proxy-audit",
        help="Audit whether DREAMPlace/OpenTimer timing proxy adds signal beyond HPWL.",
    )
    timing_proxy_audit.add_argument(
        "--platform-config",
        default="configs/default.toml",
        help="Platform config for local DREAMPlace root.",
    )
    timing_proxy_audit.add_argument("--design", required=True, help="Design name to audit.")
    timing_proxy_audit.add_argument(
        "--base-config",
        required=True,
        help="Timing-enabled DREAMPlace base config for this design.",
    )
    timing_proxy_audit.add_argument(
        "--placements",
        required=True,
        help="Placements JSON containing default and candidate DEFs.",
    )
    timing_proxy_audit.add_argument("--run-dir", required=True, help="Audit artifact directory.")
    timing_proxy_audit.add_argument(
        "--perturbations",
        type=int,
        default=10,
        help="Number of controlled coordinate perturbations per source placement.",
    )
    timing_proxy_audit.add_argument(
        "--timeout-seconds",
        type=int,
        default=600,
        help="Timeout for each DREAMPlace/OpenTimer timing proxy cell.",
    )
    timing_proxy_audit.add_argument("--gpu", type=int, help="GPU id passed to DREAMPlace.")
    timing_proxy_audit.add_argument("--seed", type=int, default=1000, help="Perturbation RNG seed.")
    timing_proxy_audit.set_defaults(func=_cmd_timing_proxy_audit)

    timing_proxy_eval = subparsers.add_parser(
        "timing-proxy-eval",
        help="Evaluate placement DEFs with DREAMPlace's OpenTimer timing proxy.",
    )
    timing_proxy_eval.add_argument(
        "--platform-config",
        default="configs/default.toml",
        help="Platform config for local DREAMPlace root.",
    )
    timing_proxy_eval.add_argument("--panel", required=True, help="Timing proxy panel TOML path.")
    timing_proxy_eval.add_argument(
        "--placements",
        required=True,
        help="Placements JSON produced from Tier-2 comparison rows.",
    )
    timing_proxy_eval.add_argument("--run-dir", required=True, help="Timing proxy artifact directory.")
    timing_proxy_eval.add_argument("--resume", action="store_true", help="Reuse successful timing cells.")
    timing_proxy_eval.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed timing cells when resuming.",
    )
    timing_proxy_eval.set_defaults(func=_cmd_timing_proxy_eval)

    tier3_recover = subparsers.add_parser(
        "tier3-recover-openroad",
        help="Recover partial Tier-3 metrics after an interrupted ChiPBench/OpenROAD run.",
    )
    _add_common_config_arg(tier3_recover)
    tier3_recover.add_argument("--panel", required=True, help="Shared panel TOML path.")
    tier3_recover.add_argument(
        "--placements",
        required=True,
        help="Placements JSON used by the interrupted Tier-3 run.",
    )
    tier3_recover.add_argument("--run-dir", required=True, help="Tier-3 artifact directory.")
    tier3_recover.add_argument("--store", help="SQLite result store path.")
    tier3_recover.add_argument(
        "--baseline-objective-id",
        default="default",
        help="Objective id used as the Tier-3 delta baseline.",
    )
    tier3_recover.add_argument(
        "--failure-stage",
        default="interrupted",
        help="Failure stage label assigned to recovered partial results.",
    )
    tier3_recover.add_argument(
        "--message",
        default="Tier-3 run was interrupted before final ChiPBench metrics were written.",
        help="Message stored in recovered raw metrics.",
    )
    tier3_recover.set_defaults(func=_cmd_tier3_recover_openroad)

    e2e = subparsers.add_parser(
        "evolve-end-to-end",
        help=(
            "Legacy CircuitNet-to-DREAMPlace orchestrator. Prefer openevolve-tier2 "
            "for the current direct DREAMPlace evolution methodology."
        ),
    )
    e2e.add_argument("--config", required=True, help="End-to-end TOML config path.")
    e2e.add_argument(
        "--platform-config",
        default="configs/default.toml",
        help="Platform config for local DREAMPlace/ChiPBench roots.",
    )
    e2e.add_argument("--run-dir", required=True, help="End-to-end artifact directory.")
    e2e.add_argument("--resume", action="store_true", help="Reuse completed round summaries.")
    e2e.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed Tier-2/Tier-3 cells when resuming.",
    )
    e2e.set_defaults(func=_cmd_evolve_end_to_end)

    eureka_tier2 = subparsers.add_parser(
        "eureka-tier2",
        help="Run Eureka-style multi-sample LLM objective discovery through DREAMPlace Tier-2 only.",
    )
    eureka_tier2.add_argument("--config", required=True, help="Eureka Tier-2 TOML config path.")
    eureka_tier2.add_argument(
        "--platform-config",
        default="configs/default.toml",
        help="Platform config for local DREAMPlace root.",
    )
    eureka_tier2.add_argument("--run-dir", required=True, help="Eureka Tier-2 artifact directory.")
    eureka_tier2.add_argument("--resume", action="store_true", help="Reuse completed DREAMPlace cells.")
    eureka_tier2.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed/incomplete DREAMPlace cells when resuming.",
    )
    eureka_tier2.set_defaults(func=_cmd_eureka_tier2)

    openevolve_tier2 = subparsers.add_parser(
        "openevolve-tier2",
        help="Run OpenEvolve-style memory-based objective discovery through DREAMPlace Tier-2.",
    )
    openevolve_tier2.add_argument(
        "--config",
        required=True,
        help="OpenEvolve-style Tier-2 TOML config path.",
    )
    openevolve_tier2.add_argument(
        "--platform-config",
        default="configs/default.toml",
        help="Platform config for local DREAMPlace root.",
    )
    openevolve_tier2.add_argument(
        "--run-dir",
        required=True,
        help="OpenEvolve-style Tier-2 artifact directory.",
    )
    openevolve_tier2.add_argument(
        "--resume",
        action="store_true",
        help="Resume from program_db metadata and reuse completed DREAMPlace cells.",
    )
    openevolve_tier2.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed/incomplete DREAMPlace cells when resuming.",
    )
    openevolve_tier2.set_defaults(func=_cmd_openevolve_tier2)

    chipbench_generalization = subparsers.add_parser(
        "chipbench-generalization",
        help=(
            "Run or prepare ChipBench-wide design-specific OpenEvolve "
            "DREAMPlace evolution followed by GRT-only OpenROAD evaluation."
        ),
    )
    chipbench_generalization.add_argument(
        "--platform-config",
        default="configs/default.toml",
        help="Platform config for local DREAMPlace/ChiPBench roots.",
    )
    chipbench_generalization.add_argument(
        "--run-dir",
        required=True,
        help="Top-level ChipBench generalization artifact directory.",
    )
    chipbench_generalization.add_argument(
        "--provider",
        default="openai",
        choices=["mock", "openai", "qwen"],
        help="LLM provider used for design-specific evolution.",
    )
    chipbench_generalization.add_argument(
        "--model",
        help="Optional provider model override; for OpenAI this sets OPENAI_MODEL.",
    )
    chipbench_generalization.add_argument(
        "--design",
        action="append",
        help="ChipBench design to include. Repeat to select multiple designs; default is all signature designs.",
    )
    chipbench_generalization.add_argument("--max-iterations", type=int, default=8)
    chipbench_generalization.add_argument("--samples-per-iteration", type=int, default=3)
    chipbench_generalization.add_argument(
        "--search-seeds",
        default="1000",
        help="Comma-separated DREAMPlace seeds for search runs.",
    )
    chipbench_generalization.add_argument(
        "--final-seeds",
        default="1000,1001,1002",
        help="Comma-separated DREAMPlace seeds for final validation and post-GRT.",
    )
    chipbench_generalization.add_argument("--search-iterations", type=int, default=50)
    chipbench_generalization.add_argument("--final-iterations", type=int, default=150)
    chipbench_generalization.add_argument("--gpu", type=int, choices=[0, 1], default=1)
    chipbench_generalization.add_argument("--timeout-seconds", type=int, default=3600)
    chipbench_generalization.add_argument(
        "--global-route-args",
        default="-allow_congestion -verbose -congestion_iterations 5",
        help="OpenROAD global route arguments passed through ChiPBench make.",
    )
    chipbench_generalization.add_argument("--resume", action="store_true")
    chipbench_generalization.add_argument("--retry-failed", action="store_true")
    chipbench_generalization.add_argument(
        "--dry-run",
        action="store_true",
        help="Write configs/audits/reports without running DREAMPlace or OpenROAD.",
    )
    chipbench_generalization.add_argument(
        "--skip-evolution",
        action="store_true",
        help="Prepare configs and aggregate any existing run artifacts only.",
    )
    chipbench_generalization.set_defaults(func=_cmd_chipbench_generalization)

    chipbench_final_eval = subparsers.add_parser(
        "chipbench-final-eval",
        help=(
            "Reuse existing selected ChipBench objectives and rerun only native "
            "default plus finalists with serious DREAMPlace/GRT settings."
        ),
    )
    chipbench_final_eval.add_argument(
        "--platform-config",
        default="configs/default.toml",
        help="Platform config for local DREAMPlace/ChiPBench roots.",
    )
    chipbench_final_eval.add_argument(
        "--source-run",
        required=True,
        help="Existing chipbench-generalization run containing per-design finalists.",
    )
    chipbench_final_eval.add_argument("--run-dir", required=True)
    chipbench_final_eval.add_argument(
        "--design",
        action="append",
        help="Design to include. Repeat to select multiple designs; default is all signature designs.",
    )
    chipbench_final_eval.add_argument(
        "--seeds",
        default="1000,1001,1002",
        help="Comma-separated seeds for serious final validation.",
    )
    chipbench_final_eval.add_argument("--iterations", type=int, default=150)
    chipbench_final_eval.add_argument("--gpu", type=int, choices=[0, 1], default=1)
    chipbench_final_eval.add_argument("--timeout-seconds", type=int, default=7200)
    chipbench_final_eval.add_argument(
        "--openroad-timeout-seconds",
        type=int,
        help="OpenROAD/ChiPBench timeout for post-GRT cells. Use 0 to disable the Python wall-clock timeout.",
    )
    chipbench_final_eval.add_argument(
        "--global-route-args",
        default="-allow_congestion -verbose -congestion_iterations 5",
    )
    chipbench_final_eval.add_argument("--resume", action="store_true")
    chipbench_final_eval.add_argument("--retry-failed", action="store_true")
    chipbench_final_eval.add_argument(
        "--allow-best-fallback",
        action="store_true",
        help="Use best_objective.json when no robust/pareto finalist exists.",
    )
    chipbench_final_eval.set_defaults(func=_cmd_chipbench_final_eval)

    openevolve_prompt_audit = subparsers.add_parser(
        "openevolve-prompt-audit",
        help="Audit saved OpenEvolve Tier-2 prompt_messages.json artifacts for hidden policy leakage.",
    )
    openevolve_prompt_audit.add_argument(
        "--run-dir",
        required=True,
        help="OpenEvolve Tier-2 run directory containing prompt_messages.json artifacts.",
    )
    openevolve_prompt_audit.add_argument(
        "--output",
        help="Optional path to write the aggregate prompt audit summary JSON.",
    )
    openevolve_prompt_audit.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write prompt_audit.json files beside prompt_messages.json.",
    )
    openevolve_prompt_audit.set_defaults(func=_cmd_openevolve_prompt_audit)

    openevolve_prompt_dry_run = subparsers.add_parser(
        "openevolve-prompt-dry-run",
        help=(
            "Build one OpenEvolve Tier-2 prompt and audit it without calling "
            "an LLM provider or DREAMPlace."
        ),
    )
    openevolve_prompt_dry_run.add_argument(
        "--config",
        required=True,
        help="OpenEvolve Tier-2 TOML config.",
    )
    openevolve_prompt_dry_run.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write prompt_context.json, prompt_messages.json, and prompt_audit.json.",
    )
    openevolve_prompt_dry_run.add_argument(
        "--from-run-dir",
        help="Optional existing OpenEvolve run directory whose program_db should seed prompt memory.",
    )
    openevolve_prompt_dry_run.add_argument(
        "--parent",
        help="Optional parent program id or preset name to use for the dry-run prompt.",
    )
    openevolve_prompt_dry_run.add_argument(
        "--iteration",
        type=int,
        default=1,
        help="Iteration index to write into the dry-run context.",
    )
    openevolve_prompt_dry_run.set_defaults(func=_cmd_openevolve_prompt_dry_run)

    objective_preset_parser = subparsers.add_parser(
        "objective-preset",
        help="Write a curated ObjectiveSpec preset.",
    )
    objective_preset_parser.add_argument("--list", action="store_true", help="List preset names.")
    objective_preset_parser.add_argument("--name", choices=preset_names(), help="Preset name to write.")
    objective_preset_parser.add_argument(
        "--output",
        default="runs/objectives/preset_objective.json",
        help="Path to write the preset ObjectiveSpec.",
    )
    objective_preset_parser.set_defaults(func=_cmd_objective_preset)

    objective_program_parser = subparsers.add_parser(
        "objective-program",
        help=(
            "Compile a restricted Python objective program into an ObjectiveSpec "
            "JSON file."
        ),
    )
    objective_program_parser.add_argument(
        "--source",
        required=True,
        help="Path to objective program .py file.",
    )
    objective_program_parser.add_argument(
        "--output",
        default="runs/objectives/program_objective.json",
        help="Path to write the compiled ObjectiveSpec.",
    )
    objective_program_parser.add_argument(
        "--term-scope",
        choices=["tier1", "deployable", "dreamplace", "dreamplace_replacement"],
        default="tier1",
        help="Term namespace allowed in term(\"...\") calls.",
    )
    objective_program_parser.add_argument(
        "--created-by",
        default="program",
        help="Creator label stored in the ObjectiveSpec.",
    )
    objective_program_parser.add_argument(
        "--rationale",
        help="Rationale stored in the ObjectiveSpec.",
    )
    objective_program_parser.add_argument(
        "--parent-id",
        action="append",
        help="Optional parent objective id. May be passed multiple times.",
    )
    objective_program_parser.set_defaults(func=_cmd_objective_program)

    dreamplace_status = subparsers.add_parser(
        "dreamplace-status",
        help="Check DREAMPlace custom objective patch status.",
    )
    _add_common_config_arg(dreamplace_status)
    dreamplace_status.set_defaults(func=_cmd_dreamplace_status)

    dreamplace_patch = subparsers.add_parser(
        "dreamplace-apply-patch",
        help="Apply the tracked external DREAMPlace custom objective patch.",
    )
    _add_common_config_arg(dreamplace_patch)
    dreamplace_patch.set_defaults(func=_cmd_dreamplace_apply_patch)

    dreamplace_run = subparsers.add_parser(
        "dreamplace-run",
        help="Run DREAMPlace, optionally with one ObjectiveSpec as custom objective.",
    )
    _add_common_config_arg(dreamplace_run)
    dreamplace_run.add_argument("--objective", help="Optional ObjectiveSpec JSON path.")
    dreamplace_run.add_argument("--run-name", required=True, help="Run/artifact directory name.")
    dreamplace_run.add_argument(
        "--base-config",
        help="DREAMPlace base JSON config. Defaults to test/ispd2005/adaptec1.json.",
    )
    dreamplace_run.add_argument("--iterations", type=int, default=20, help="Global placement iterations.")
    dreamplace_run.add_argument("--gpu", type=int, choices=[0, 1], help="Override DREAMPlace gpu flag.")
    dreamplace_run.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Custom objective logging interval inside DREAMPlace.",
    )
    dreamplace_run.add_argument(
        "--keep-legalization",
        action="store_true",
        help="Keep legalization/detailed placement flags from the base config.",
    )
    dreamplace_run.add_argument(
        "--timeout-seconds",
        type=int,
        default=900,
        help="DREAMPlace process timeout.",
    )
    dreamplace_run.set_defaults(func=_cmd_dreamplace_run)

    dreamplace_term_check = subparsers.add_parser(
        "dreamplace-term-check",
        help="Run short DREAMPlace finite-value/gradient checks for deployable terms.",
    )
    _add_common_config_arg(dreamplace_term_check)
    dreamplace_term_check.add_argument(
        "--base-config",
        required=True,
        help="DREAMPlace base JSON config to use for the check.",
    )
    dreamplace_term_check.add_argument(
        "--terms",
        required=True,
        help="Comma-separated DREAMPlace term names to check.",
    )
    dreamplace_term_check.add_argument(
        "--run-dir",
        default="runs/dreamplace_term_check",
        help="Directory for generated configs, logs, and term-check summary.",
    )
    dreamplace_term_check.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Short global placement iteration count for each term.",
    )
    dreamplace_term_check.add_argument(
        "--gpu",
        type=int,
        choices=[0, 1],
        help="Override DREAMPlace gpu flag.",
    )
    dreamplace_term_check.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
        help="Timeout per term check.",
    )
    dreamplace_term_check.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Custom objective logging interval for term checks.",
    )
    dreamplace_term_check.add_argument(
        "--min-grad-norm",
        type=float,
        default=1e-12,
        help="Minimum accepted custom objective gradient norm.",
    )
    dreamplace_term_check.set_defaults(func=_cmd_dreamplace_term_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
