"""Command-line entry points for the CoEvoP&R artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coevop.backends.dreamplace import (
    apply_source_patch_series,
    apply_patch as apply_dreamplace_patch,
    extract_custom_objective_lines,
    is_patch_applied,
    make_run_config,
    run_dreamplace,
    write_run_summary,
)
from coevop.config import load_config
from coevop.eval.asap7_transfer import (
    check_asap7_panel,
    generate_asap7_base_configs,
    run_asap7_transfer,
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
from coevop.llm.providers import provider_from_name, provider_status
from coevop.objectives.presets import objective_preset, preset_names
from coevop.objectives.program import parse_objective_program
from coevop.objectives.spec import (
    load_objective_spec,
    objective_spec_to_dict,
    write_objective_spec,
)
from coevop.objectives.terms import term_names, unsupported_terms_for_scope


def _json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _paths(value: str | None) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _ints(value: str | None) -> list[int] | None:
    values = _paths(value)
    return [int(value) for value in values] if values else None


def _config_arg(parser: argparse.ArgumentParser, name: str = "--config") -> None:
    parser.add_argument(name, default="configs/default.toml")


def _cmd_env(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    payload = {
        "dreamplace_root": str(config.dreamplace_root),
        "chipbench_root": str(config.chipbench_root),
        "openroad_flow_root": (
            str(config.openroad_flow_root) if config.openroad_flow_root else None
        ),
        "run_root": str(config.run_root),
        "llm_provider": config.llm_provider,
        "exists": {
            "dreamplace_root": config.dreamplace_root.exists(),
            "chipbench_root": config.chipbench_root.exists(),
            "openroad_flow_root": bool(
                config.openroad_flow_root and config.openroad_flow_root.exists()
            ),
        },
    }
    _json(payload)
    return 0


def _cmd_llm_check(_args: argparse.Namespace) -> int:
    _json(provider_status())
    return 0


def _cmd_llm_generate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    context = {}
    if args.context:
        context = json.loads(Path(args.context).read_text(encoding="utf-8"))
    spec = provider_from_name(args.provider or config.llm_provider).generate(
        context=context,
        term_scope=args.term_scope,
    )
    output = write_objective_spec(spec, args.output)
    _json({"output": str(output), "objective": objective_spec_to_dict(spec)})
    return 0


def _cmd_tier2(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    payload = run_tier2_dreamplace(
        panel_path=args.panel,
        objective_paths=_paths(args.objectives),
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
    _json(payload)
    return 0


def _cmd_panel_check(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    payload = check_shared_panel(
        panel_path=args.panel,
        dreamplace_root=config.dreamplace_root,
        chipbench_root=config.chipbench_root,
        run_dir=args.run_dir,
        skip_runs=args.skip_runs,
    ).to_dict()
    _json(payload)
    return 0 if payload["ok"] else 1


def _cmd_tier3(args: argparse.Namespace) -> int:
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
    _json(payload)
    return 0


def _cmd_tier3_recover(args: argparse.Namespace) -> int:
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
    _json(payload)
    return 0


def _cmd_timing_eval(args: argparse.Namespace) -> int:
    config = load_config(args.platform_config)
    payload = run_timing_proxy_eval(
        panel_path=args.panel,
        placements_path=args.placements,
        dreamplace_root=config.dreamplace_root,
        run_dir=args.run_dir,
        resume=args.resume,
        retry_failed=args.retry_failed,
    )
    _json(payload)
    return 0


def _cmd_timing_audit(args: argparse.Namespace) -> int:
    config = load_config(args.platform_config)
    payload = run_timing_proxy_audit(
        design=args.design,
        base_config=args.base_config,
        timing_panel_path=args.timing_panel,
        placements_path=args.placements,
        dreamplace_root=config.dreamplace_root,
        run_dir=args.run_dir,
        perturbations=args.perturbations,
        timeout_seconds=args.timeout_seconds,
        gpu=args.gpu,
        seed=args.seed,
    )
    _json(payload)
    return 0


def _cmd_evolve(args: argparse.Namespace) -> int:
    config = load_config(args.platform_config)
    payload = run_openevolve_tier2(
        config_path=args.config,
        run_dir=args.run_dir,
        dreamplace_root=config.dreamplace_root,
        chipbench_root=config.chipbench_root,
        resume=args.resume,
        retry_failed=args.retry_failed,
    )
    _json(payload)
    return 0


def _cmd_prompt_audit(args: argparse.Namespace) -> int:
    payload = audit_openevolve_prompt_artifacts(args.run_dir, write=not args.no_write)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _json(payload)
    return 0 if payload["passed"] else 1


def _cmd_prompt_dry_run(args: argparse.Namespace) -> int:
    payload = build_openevolve_prompt_dry_run(
        config_path=args.config,
        output_dir=args.output_dir,
        from_run_dir=args.from_run_dir,
        parent_preset=args.parent,
        iteration=args.iteration,
    )
    _json(payload)
    return 0 if payload["prompt_audit"]["passed"] else 1


def _cmd_asap7_check(args: argparse.Namespace) -> int:
    config = load_config(args.platform_config)
    payload = check_asap7_panel(
        chipbench_root=config.chipbench_root,
        dreamplace_root=config.dreamplace_root,
        dreamplace_config_dir=args.dreamplace_config_dir,
        run_dir=args.run_dir,
        designs=args.design,
        skip_make_dry_run=args.skip_make_dry_run,
    )
    _json(payload)
    return 0 if payload.get("ok") else 1


def _cmd_asap7_configs(args: argparse.Namespace) -> int:
    config = load_config(args.platform_config)
    payload = generate_asap7_base_configs(
        chipbench_root=config.chipbench_root,
        output_dir=args.output_dir,
        designs=args.design,
        flow_variant=args.flow_variant,
        overwrite=args.overwrite,
    )
    _json(payload)
    return 0 if payload.get("ok") else 1


def _cmd_asap7_transfer(args: argparse.Namespace) -> int:
    config = load_config(args.platform_config)
    payload = run_asap7_transfer(
        run_dir=args.run_dir,
        dreamplace_root=config.dreamplace_root,
        chipbench_root=config.chipbench_root,
        dreamplace_config_dir=args.dreamplace_config_dir,
        source_objective=args.source_objective,
        source_root=args.source_root,
        designs=args.design,
        seeds=_ints(args.seeds),
        iterations=args.iterations,
        gpu=args.gpu,
        timeout_seconds=args.timeout_seconds,
        openroad_timeout_seconds=args.openroad_timeout_seconds,
        resume=args.resume,
        retry_failed=args.retry_failed,
        dry_run=args.dry_run,
        adaptation_objectives=_paths(args.adaptation_objectives),
    )
    _json(payload)
    return 0 if payload.get("status") != "blocked" else 1


def _cmd_preset(args: argparse.Namespace) -> int:
    if args.list:
        _json({"presets": preset_names()})
        return 0
    if not args.name:
        raise SystemExit("--name is required unless --list is used")
    spec = objective_preset(args.name)
    output = write_objective_spec(spec, args.output)
    _json({"output": str(output), "objective": objective_spec_to_dict(spec)})
    return 0


def _cmd_program(args: argparse.Namespace) -> int:
    spec = parse_objective_program(
        Path(args.source).read_text(encoding="utf-8"),
        created_by=args.created_by,
        rationale=args.rationale or "",
        parent_ids=args.parent_id or [],
        term_scope=args.term_scope,
    )
    output = write_objective_spec(spec, args.output)
    _json({"output": str(output), "objective": objective_spec_to_dict(spec)})
    return 0


def _cmd_dreamplace_status(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    _json(
        {
            "dreamplace_root": str(config.dreamplace_root),
            "placeobj_patch_applied": is_patch_applied(config.dreamplace_root),
        }
    )
    return 0


def _cmd_dreamplace_patch(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    applied = apply_dreamplace_patch(config.dreamplace_root, repo_root=Path.cwd())
    _json({"applied": applied, "dreamplace_root": str(config.dreamplace_root)})
    return 0


def _cmd_dreamplace_source_patches(args: argparse.Namespace) -> int:
    _json(
        apply_source_patch_series(
            args.source_root,
            repo_root=Path.cwd(),
        )
    )
    return 0


def _cmd_dreamplace_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.objective:
        spec = load_objective_spec(args.objective)
        unsupported = unsupported_terms_for_scope(spec.term_set, "dreamplace")
        if unsupported:
            raise SystemExit(f"unsupported DREAMPlace terms: {unsupported}")
        if not is_patch_applied(config.dreamplace_root):
            raise SystemExit("apply the DREAMPlace integration patch first")
    base_config = Path(args.base_config) if args.base_config else (
        config.dreamplace_root / "test" / "ispd2005" / "adaptec1.json"
    )
    run_dir = Path(config.run_root) / "dreamplace" / args.run_name
    generated = make_run_config(
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
        config_path=generated,
        run_name=args.run_name,
        run_dir=run_dir,
        timeout_seconds=args.timeout_seconds,
    )
    summary = write_run_summary(run, run_dir / "run_summary.json")
    _json(
        {
            "returncode": run.returncode,
            "config": str(generated),
            "log": str(run.log_path),
            "summary": str(summary),
            "custom_objective_log_lines": extract_custom_objective_lines(run.log_path)[-5:],
        }
    )
    return run.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coevop")
    commands = parser.add_subparsers(dest="command", required=True)

    cmd = commands.add_parser("env-check", help="Resolve backend roots.")
    _config_arg(cmd)
    cmd.set_defaults(func=_cmd_env)

    cmd = commands.add_parser("llm-check", help="Check LLM provider readiness.")
    cmd.set_defaults(func=_cmd_llm_check)
    cmd = commands.add_parser("llm-generate", help="Generate and validate one objective.")
    _config_arg(cmd)
    cmd.add_argument("--provider")
    cmd.add_argument("--context")
    cmd.add_argument("--term-scope", default="dreamplace_controller")
    cmd.add_argument("--output", required=True)
    cmd.set_defaults(func=_cmd_llm_generate)

    cmd = commands.add_parser("tier2-dreamplace", help="Run a DREAMPlace panel.")
    _config_arg(cmd)
    cmd.add_argument("--panel", required=True)
    cmd.add_argument("--objectives")
    cmd.add_argument("--run-dir", required=True)
    cmd.add_argument("--store")
    cmd.add_argument("--resume", action="store_true")
    cmd.add_argument("--retry-failed", action="store_true")
    cmd.add_argument("--no-default", action="store_true")
    cmd.add_argument("--no-custom-default", action="store_true")
    cmd.add_argument("--require-output-artifact", action="store_true")
    cmd.add_argument("--require-def-output", action="store_true")
    cmd.add_argument("--keep-legalization", action="store_true")
    cmd.set_defaults(func=_cmd_tier2)

    cmd = commands.add_parser("shared-panel-check", help="Validate a placement and routing panel.")
    _config_arg(cmd)
    cmd.add_argument("--panel", required=True)
    cmd.add_argument("--run-dir", default="runs/shared_panel_check")
    cmd.add_argument("--skip-runs", action="store_true")
    cmd.set_defaults(func=_cmd_panel_check)

    cmd = commands.add_parser("tier3-openroad", help="Run post-route evaluation.")
    _config_arg(cmd)
    cmd.add_argument("--panel", required=True)
    cmd.add_argument("--placements", required=True)
    cmd.add_argument("--run-dir", required=True)
    cmd.add_argument("--store")
    cmd.add_argument("--baseline-objective-id", default="default")
    cmd.add_argument("--resume", action="store_true")
    cmd.add_argument("--retry-failed", action="store_true")
    cmd.set_defaults(func=_cmd_tier3)

    cmd = commands.add_parser("tier3-recover-openroad", help="Recover an interrupted routed run.")
    _config_arg(cmd)
    cmd.add_argument("--panel", required=True)
    cmd.add_argument("--placements", required=True)
    cmd.add_argument("--run-dir", required=True)
    cmd.add_argument("--store")
    cmd.add_argument("--baseline-objective-id", default="default")
    cmd.add_argument("--failure-stage", default="interrupted")
    cmd.add_argument("--message", default="run interrupted")
    cmd.set_defaults(func=_cmd_tier3_recover)

    for name, function, help_text in (
        ("timing-proxy-eval", _cmd_timing_eval, "Evaluate placement-stage timing evidence."),
        ("timing-proxy-audit", _cmd_timing_audit, "Audit timing evidence with controlled perturbations."),
    ):
        cmd = commands.add_parser(name, help=help_text)
        cmd.add_argument("--platform-config", default="configs/default.toml")
        if name.endswith("eval"):
            cmd.add_argument("--panel", required=True)
            cmd.add_argument("--resume", action="store_true")
            cmd.add_argument("--retry-failed", action="store_true")
        else:
            cmd.add_argument("--design", required=True)
            cmd.add_argument("--base-config", required=True)
            cmd.add_argument("--timing-panel")
            cmd.add_argument("--perturbations", type=int, default=4)
            cmd.add_argument("--timeout-seconds", type=int, default=1800)
            cmd.add_argument("--gpu", type=int, default=1)
            cmd.add_argument("--seed", type=int, default=42)
        cmd.add_argument("--placements", required=True)
        cmd.add_argument("--run-dir", required=True)
        cmd.set_defaults(func=function)

    cmd = commands.add_parser("openevolve-tier2", help="Run CoEvo objective evolution.")
    cmd.add_argument("--config", required=True)
    cmd.add_argument("--platform-config", default="configs/default.toml")
    cmd.add_argument("--run-dir", required=True)
    cmd.add_argument("--resume", action="store_true")
    cmd.add_argument("--retry-failed", action="store_true")
    cmd.set_defaults(func=_cmd_evolve)

    cmd = commands.add_parser("openevolve-prompt-audit", help="Audit saved proposal prompts.")
    cmd.add_argument("--run-dir", required=True)
    cmd.add_argument("--output")
    cmd.add_argument("--no-write", action="store_true")
    cmd.set_defaults(func=_cmd_prompt_audit)
    cmd = commands.add_parser("openevolve-prompt-dry-run", help="Build one proposal packet without an API call.")
    cmd.add_argument("--config", required=True)
    cmd.add_argument("--output-dir", required=True)
    cmd.add_argument("--from-run-dir")
    cmd.add_argument("--parent")
    cmd.add_argument("--iteration", type=int, default=1)
    cmd.set_defaults(func=_cmd_prompt_dry_run)

    cmd = commands.add_parser("asap7-panel-check", help="Validate ASAP7 dependencies.")
    cmd.add_argument("--platform-config", default="configs/default.toml")
    cmd.add_argument("--dreamplace-config-dir", default="configs/dreamplace_base/asap7")
    cmd.add_argument("--run-dir", default="runs/asap7_check")
    cmd.add_argument("--design", action="append")
    cmd.add_argument("--skip-make-dry-run", action="store_true")
    cmd.set_defaults(func=_cmd_asap7_check)
    cmd = commands.add_parser("asap7-base-config-gen", help="Generate ASAP7 DREAMPlace inputs.")
    cmd.add_argument("--platform-config", default="configs/default.toml")
    cmd.add_argument("--output-dir", required=True)
    cmd.add_argument("--design", action="append")
    cmd.add_argument("--flow-variant")
    cmd.add_argument("--overwrite", action="store_true")
    cmd.set_defaults(func=_cmd_asap7_configs)
    cmd = commands.add_parser("asap7-transfer", help="Evaluate the frozen Nangate45 champion on ASAP7.")
    cmd.add_argument("--platform-config", default="configs/default.toml")
    cmd.add_argument("--run-dir", required=True)
    cmd.add_argument("--dreamplace-config-dir", default="configs/dreamplace_base/asap7")
    cmd.add_argument("--source-objective")
    cmd.add_argument("--source-root", default="runs")
    cmd.add_argument("--adaptation-objectives")
    cmd.add_argument("--design", action="append")
    cmd.add_argument("--seeds", default="1000,1001,1002")
    cmd.add_argument("--iterations", type=int, default=1000)
    cmd.add_argument("--gpu", type=int, choices=(0, 1), default=1)
    cmd.add_argument("--timeout-seconds", type=int, default=7200)
    cmd.add_argument("--openroad-timeout-seconds", type=int, default=14400)
    cmd.add_argument("--resume", action="store_true")
    cmd.add_argument("--retry-failed", action="store_true")
    cmd.add_argument("--dry-run", action="store_true")
    cmd.set_defaults(func=_cmd_asap7_transfer)

    cmd = commands.add_parser("objective-preset", help="Write a supplied objective control.")
    cmd.add_argument("--list", action="store_true")
    cmd.add_argument("--name")
    cmd.add_argument("--output", default="objective.json")
    cmd.set_defaults(func=_cmd_preset)
    cmd = commands.add_parser("objective-program", help="Parse and validate a symbolic objective program.")
    cmd.add_argument("--source", required=True)
    cmd.add_argument("--output", required=True)
    cmd.add_argument("--created-by", default="manual")
    cmd.add_argument("--rationale")
    cmd.add_argument("--parent-id", action="append")
    cmd.add_argument("--term-scope", default="dreamplace_controller")
    cmd.set_defaults(func=_cmd_program)

    cmd = commands.add_parser("dreamplace-status", help="Check the DREAMPlace integration patch.")
    _config_arg(cmd)
    cmd.set_defaults(func=_cmd_dreamplace_status)
    cmd = commands.add_parser("dreamplace-apply-patch", help="Apply the DREAMPlace integration patch.")
    _config_arg(cmd)
    cmd.set_defaults(func=_cmd_dreamplace_patch)
    cmd = commands.add_parser(
        "dreamplace-apply-source-patches",
        help="Apply the complete DREAMPlace source patch series before building.",
    )
    cmd.add_argument("--source-root", required=True)
    cmd.set_defaults(func=_cmd_dreamplace_source_patches)
    cmd = commands.add_parser("dreamplace-run", help="Run one DREAMPlace objective.")
    _config_arg(cmd)
    cmd.add_argument("--objective")
    cmd.add_argument("--base-config")
    cmd.add_argument("--run-name", required=True)
    cmd.add_argument("--iterations", type=int, default=20)
    cmd.add_argument("--gpu", type=int, choices=(0, 1), default=1)
    cmd.add_argument("--log-interval", type=int, default=1)
    cmd.add_argument("--timeout-seconds", type=int, default=1800)
    cmd.add_argument("--keep-legalization", action="store_true")
    cmd.set_defaults(func=_cmd_dreamplace_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
