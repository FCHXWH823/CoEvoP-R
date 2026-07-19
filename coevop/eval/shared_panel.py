"""Shared DREAMPlace/ChiPBench design panel support."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised by py310 via tomli
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from coevop.backends.dreamplace import is_patch_applied, make_run_config, run_dreamplace
from coevop.eval.chipbench_utils import prepare_chipbench_runtime_dirs


@dataclass(frozen=True)
class SharedPanelDesign:
    name: str
    dreamplace_config: str
    chipbench_config: str | None = None
    mode: str = "global"
    timing_mode: str = "none"
    reference_def: str | None = None
    freeze_placed_macros_for_routing: bool = False


@dataclass(frozen=True)
class SharedPanel:
    seeds: list[int]
    iterations: int
    gpu: int | None
    timeout_seconds: int
    log_interval: int
    timing_mode: str
    stop_overflow: float | None
    designs: list[SharedPanelDesign]


@dataclass(frozen=True)
class SharedPanelCheck:
    panel_path: str
    dreamplace_root: str
    chipbench_root: str
    openroad_path: str | None
    dreamplace_patch_applied: bool
    file_checks: list[dict[str, Any]]
    dreamplace_smoke: dict[str, Any] | None
    chipbench_reference: dict[str, Any] | None
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_shared_panel(path: str | Path) -> SharedPanel:
    panel_path = Path(path)
    payload = tomllib.loads(panel_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in payload.get("seeds", [1000])]
    if not seeds:
        raise ValueError("shared panel must contain at least one seed")
    designs = []
    timing_mode = str(payload.get("timing_mode", "compatibility_only"))
    for item in payload.get("designs", []):
        name = str(item["name"])
        mode = str(item.get("mode", "global"))
        if mode != "global":
            raise ValueError(
                f"shared panel design {name} must use full post-route mode 'global'"
            )
        dreamplace_config = _resolve_panel_path(panel_path, str(item["dreamplace_config"]))
        chipbench_config_value = item.get("chipbench_config")
        chipbench_config = (
            _resolve_panel_path(panel_path, str(chipbench_config_value))
            if chipbench_config_value
            else None
        )
        reference_def = item.get("reference_def")
        designs.append(
            SharedPanelDesign(
                name=name,
                dreamplace_config=str(dreamplace_config),
                chipbench_config=str(chipbench_config) if chipbench_config else None,
                mode=mode,
                timing_mode=str(item.get("timing_mode", timing_mode)),
                reference_def=(
                    str(_resolve_panel_path(panel_path, str(reference_def)))
                    if reference_def
                    else None
                ),
                freeze_placed_macros_for_routing=bool(
                    item.get("freeze_placed_macros_for_routing", False)
                ),
            )
        )
    if not designs:
        raise ValueError("shared panel must contain at least one [[designs]] entry")
    return SharedPanel(
        seeds=seeds,
        iterations=int(payload.get("iterations", 50)),
        gpu=int(payload["gpu"]) if payload.get("gpu") is not None else None,
        timeout_seconds=int(payload.get("timeout_seconds", 900)),
        log_interval=int(payload.get("log_interval", 10)),
        timing_mode=timing_mode,
        stop_overflow=(
            float(payload["stop_overflow"])
            if payload.get("stop_overflow") is not None
            else None
        ),
        designs=designs,
    )


def write_dreamplace_panel(panel: SharedPanel, output: str | Path) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "seeds = [" + ", ".join(str(seed) for seed in panel.seeds) + "]",
        f"iterations = {panel.iterations}",
        f"timeout_seconds = {panel.timeout_seconds}",
        f"log_interval = {panel.log_interval}",
        f'timing_mode = "{panel.timing_mode}"',
    ]
    if panel.gpu is not None:
        lines.append(f"gpu = {panel.gpu}")
    if panel.stop_overflow is not None:
        lines.append(f"stop_overflow = {panel.stop_overflow}")
    lines.append("")
    for design in panel.designs:
        lines.extend(
            [
                "[[designs]]",
                f'name = "{design.name}"',
                f'base_config = "{design.dreamplace_config}"',
                f'timing_mode = "{design.timing_mode}"',
                "",
            ]
        )
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def check_shared_panel(
    *,
    panel_path: str | Path,
    dreamplace_root: str | Path,
    chipbench_root: str | Path,
    run_dir: str | Path,
    skip_runs: bool = False,
) -> SharedPanelCheck:
    panel = load_shared_panel(panel_path)
    dreamplace_root = Path(dreamplace_root)
    chipbench_root = Path(chipbench_root)
    run_root = Path(run_dir)
    run_root.mkdir(parents=True, exist_ok=True)

    file_checks = []
    for design in panel.designs:
        file_checks.append(
            _file_check(design.name, "dreamplace_config", design.dreamplace_config)
        )
        if design.chipbench_config:
            file_checks.append(
                _file_check(design.name, "chipbench_config", design.chipbench_config)
            )
        if design.reference_def:
            file_checks.append(_file_check(design.name, "reference_def", design.reference_def))

    openroad_path = shutil.which("openroad")
    patch_applied = is_patch_applied(dreamplace_root)
    dreamplace_smoke = None
    chipbench_reference = None
    if not skip_runs and panel.designs:
        first = panel.designs[0]
        if Path(first.dreamplace_config).is_file():
            dreamplace_smoke = _run_dreamplace_smoke(
                panel=panel,
                design=first,
                dreamplace_root=dreamplace_root,
                run_dir=run_root / "dreamplace_smoke",
            )
        reference_design = next((design for design in panel.designs if design.reference_def), None)
        if reference_design and Path(reference_design.reference_def or "").is_file():
            chipbench_reference = _run_chipbench_reference(
                design=reference_design,
                chipbench_root=chipbench_root,
                run_dir=run_root / "chipbench_reference",
                timeout_seconds=panel.timeout_seconds,
            )

    ok = (
        all(item["exists"] for item in file_checks)
        and openroad_path is not None
        and patch_applied
        and (skip_runs or dreamplace_smoke is None or dreamplace_smoke.get("status") == "success")
        and (
            skip_runs
            or chipbench_reference is None
            or chipbench_reference.get("status") == "success"
        )
    )
    result = SharedPanelCheck(
        panel_path=str(panel_path),
        dreamplace_root=str(dreamplace_root),
        chipbench_root=str(chipbench_root),
        openroad_path=openroad_path,
        dreamplace_patch_applied=patch_applied,
        file_checks=file_checks,
        dreamplace_smoke=dreamplace_smoke,
        chipbench_reference=chipbench_reference,
        ok=ok,
    )
    _write_json(result.to_dict(), run_root / "shared_panel_check.json")
    return result


def _run_dreamplace_smoke(
    *,
    panel: SharedPanel,
    design: SharedPanelDesign,
    dreamplace_root: Path,
    run_dir: Path,
) -> dict[str, Any]:
    start = time.time()
    try:
        config_path = make_run_config(
            design.dreamplace_config,
            None,
            run_dir,
            iterations=1,
            gpu=panel.gpu,
            seed=panel.seeds[0],
            custom_objective_log_interval=panel.log_interval,
            disable_legalization=True,
            stop_overflow=panel.stop_overflow,
        )
        run = run_dreamplace(
            dreamplace_root=dreamplace_root,
            config_path=config_path,
            run_name=f"shared_panel_check_{design.name}",
            run_dir=run_dir,
            timeout_seconds=min(panel.timeout_seconds, 300),
        )
        return {
            "design": design.name,
            "status": "success" if run.returncode == 0 else "failed",
            "returncode": run.returncode,
            "runtime_seconds": time.time() - start,
            "log_path": run.log_path,
            "metrics": run.metrics,
        }
    except Exception as exc:
        return {
            "design": design.name,
            "status": "failed",
            "returncode": None,
            "runtime_seconds": time.time() - start,
            "error": str(exc),
        }


def _run_chipbench_reference(
    *,
    design: SharedPanelDesign,
    chipbench_root: Path,
    run_dir: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    if not design.chipbench_config:
        return {
            "design": design.name,
            "status": "failed",
            "returncode": None,
            "runtime_seconds": 0.0,
            "error": "chipbench_config is required for ChiPBench reference evaluation",
        }
    run_dir.mkdir(parents=True, exist_ok=True)
    prepare_chipbench_runtime_dirs(chipbench_root)
    evaluate_name = f"coevop_panel_check_{_safe_name(design.name)}"
    log_path = run_dir / "chipbench_reference.log"
    command = [
        "python3",
        "benchmarking/benchmarking.py",
        f"--mode={design.mode}",
        f"--config_setting={design.chipbench_config}",
        f"--def_path={design.reference_def}",
        f"--evaluate_name={evaluate_name}",
    ]
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(
                command,
                cwd=chipbench_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            log.write(f"\nCoEvoP&R ChiPBench timeout after {timeout_seconds} seconds\n")
            returncode = -9
    metrics_path = chipbench_root / "benchmarking_result" / evaluate_name / "metrics.json"
    return {
        "design": design.name,
        "status": "success" if returncode == 0 and metrics_path.is_file() else "failed",
        "returncode": returncode,
        "runtime_seconds": time.time() - start,
        "log_path": str(log_path),
        "metrics_path": str(metrics_path) if metrics_path.is_file() else None,
    }


def _file_check(design: str, kind: str, path: str) -> dict[str, Any]:
    file_path = Path(path)
    return {
        "design": design,
        "kind": kind,
        "path": str(file_path),
        "exists": file_path.is_file(),
        "size_bytes": file_path.stat().st_size if file_path.is_file() else None,
    }


def _resolve_panel_path(panel_path: Path, value: str) -> Path:
    path = Path(os.path.expandvars(value)).expanduser()
    if path.is_absolute():
        return path
    return (panel_path.parent / path).resolve()


def _write_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
