"""DREAMPlace v1 integration helpers."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from coevop.objectives.spec import load_objective_spec
from coevop.objectives.terms import term_names, unsupported_terms_for_scope


PATCH_MARKER = "_coevop_build_extra_ops"
OLD_PATCH_MARKER = "_COEVOP_DEPLOYABLE_TERMS"
PATCH_PATH = Path("patches/dreamplace/custom_objective_placeobj.patch")


@dataclass(frozen=True)
class DreamPlaceRun:
    run_name: str
    run_dir: str
    config_path: str
    log_path: str
    returncode: int
    command: list[str]
    metrics: dict[str, Any]


def placeobj_path(dreamplace_root: str | Path) -> Path:
    return Path(dreamplace_root) / "dreamplace" / "PlaceObj.py"


def is_patch_applied(dreamplace_root: str | Path) -> bool:
    path = placeobj_path(dreamplace_root)
    return path.is_file() and PATCH_MARKER in path.read_text(encoding="utf-8")


def apply_patch(
    dreamplace_root: str | Path,
    patch_path: str | Path = PATCH_PATH,
    repo_root: str | Path = ".",
) -> bool:
    """Apply the tracked DREAMPlace patch. Returns True if a patch was applied."""

    dreamplace_root = Path(dreamplace_root)
    placeobj = placeobj_path(dreamplace_root)
    placeobj_text = placeobj.read_text(encoding="utf-8") if placeobj.is_file() else ""
    if PATCH_MARKER in placeobj_text:
        return False
    if OLD_PATCH_MARKER in placeobj_text:
        backup = _find_placeobj_backup(placeobj)
        if backup is None:
            raise RuntimeError(
                "DREAMPlace has an older CoEvoP&R patch but no PlaceObj.py backup "
                "was found. Restore a clean PlaceObj.py before applying the updated patch."
            )
        shutil.copy2(backup, placeobj)
    elif placeobj.is_file():
        backup = placeobj.with_name("PlaceObj.py.coevop_bak_")
        if not backup.exists():
            shutil.copy2(placeobj, backup)
    full_patch_path = Path(repo_root) / patch_path
    git_root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=dreamplace_root,
            text=True,
        ).strip()
    )
    patch_directory = dreamplace_root.resolve().relative_to(git_root.resolve()).as_posix()
    subprocess.run(
        [
            "git",
            "apply",
            "--recount",
            "--check",
            f"--directory={patch_directory}",
            str(full_patch_path),
        ],
        cwd=git_root,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "apply",
            "--recount",
            f"--directory={patch_directory}",
            str(full_patch_path),
        ],
        cwd=git_root,
        check=True,
    )
    return True


def _find_placeobj_backup(placeobj: Path) -> Path | None:
    candidates = sorted(placeobj.parent.glob("PlaceObj.py.coevop_bak*"))
    if len(placeobj.parents) >= 3:
        source_tree_placeobj = placeobj.parents[2] / "dreamplace" / "PlaceObj.py"
        candidates.append(source_tree_placeobj)
    for candidate in candidates:
        if candidate.is_file() and OLD_PATCH_MARKER not in candidate.read_text(
            encoding="utf-8",
            errors="replace",
        ):
            return candidate
    return None


def make_run_config(
    base_config: str | Path,
    objective_spec: str | Path | None,
    run_dir: str | Path,
    *,
    iterations: int | None = None,
    gpu: int | None = None,
    seed: int | None = None,
    custom_objective_log_interval: int = 25,
    disable_legalization: bool = True,
) -> Path:
    base_config = Path(base_config)
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    result_dir = run_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(base_config.read_text(encoding="utf-8-sig"))
    if objective_spec is not None:
        objective_spec = Path(objective_spec).resolve()
        spec = load_objective_spec(objective_spec)
        unsupported = unsupported_terms_for_scope(spec.term_set, "dreamplace")
        if unsupported:
            raise ValueError(
                "DREAMPlace objective contains unsupported terms "
                f"{unsupported}; current deployable terms are {term_names('dreamplace')}"
            )
        config["custom_objective_spec"] = str(objective_spec)
        config["custom_objective_log_interval"] = custom_objective_log_interval
    config["result_dir"] = str(result_dir)
    config["plot_flag"] = 0
    if gpu is not None:
        config["gpu"] = int(gpu)
    if seed is not None:
        config["random_seed"] = int(seed)
    if iterations is not None:
        for stage in config.get("global_place_stages", []):
            stage["iteration"] = int(iterations)
    if disable_legalization:
        config["legalize_flag"] = 0
        config["detailed_place_flag"] = 0

    output = run_dir / "dreamplace_config.json"
    output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def run_dreamplace(
    dreamplace_root: str | Path,
    config_path: str | Path,
    run_name: str,
    run_dir: str | Path,
    *,
    timeout_seconds: int = 900,
) -> DreamPlaceRun:
    dreamplace_root = Path(dreamplace_root).resolve()
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "dreamplace.log"
    command = ["python3", "dreamplace/Placer.py", str(Path(config_path).resolve())]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(dreamplace_root)
    with log_path.open("w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(
                command,
                cwd=dreamplace_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            log.write(f"\nCoEvoP&R DREAMPlace timeout after {timeout_seconds} seconds\n")
            returncode = -9
    return DreamPlaceRun(
        run_name=run_name,
        run_dir=str(run_dir),
        config_path=str(Path(config_path).resolve()),
        log_path=str(log_path),
        returncode=returncode,
        command=command,
        metrics=parse_dreamplace_log(log_path),
    )


def write_run_summary(run: DreamPlaceRun, output: str | Path) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(run), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def extract_custom_objective_lines(log_path: str | Path) -> list[str]:
    path = Path(log_path)
    if not path.exists():
        return []
    return [
        line.rstrip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if "CoEvoP&R custom objective" in line
    ]


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"
_CUSTOM_TOTAL_RE = re.compile(
    rf"custom objective id=(?P<objective_id>\S+) call=(?P<call>\d+) "
    rf"total=(?P<total>{_FLOAT}).*?wirelength_raw=(?P<wirelength_raw>{_FLOAT}).*?"
    rf"weighted_density_raw=(?P<weighted_density_raw>{_FLOAT})"
)
_CUSTOM_GRAD_RE = re.compile(rf"custom objective grad_norm=(?P<grad_norm>{_FLOAT})")
_CUSTOM_TERM_SCALE_RE = re.compile(
    rf"custom objective term_scale (?P<term>\S+)=(?P<value>{_FLOAT})"
)
_CUSTOM_OUTPUT_SCALE_RE = re.compile(rf"custom objective output_scale=(?P<value>{_FLOAT})")
_CUSTOM_TERMS_RE = re.compile(r"custom objective terms (?P<payload>\{.*\})")
_CUSTOM_COMPONENTS_RE = re.compile(r"custom objective components (?P<payload>\{.*\})")


def parse_dreamplace_log(log_path: str | Path) -> dict[str, Any]:
    path = Path(log_path)
    metrics: dict[str, Any] = {
        "custom_objective": {
            "objective_id": None,
            "calls": 0,
            "last_total": None,
            "last_wirelength_raw": None,
            "last_weighted_density_raw": None,
            "last_grad_norm": None,
            "output_scale": None,
            "term_scales": {},
            "last_terms": {},
            "term_history": [],
            "last_components": {},
            "component_history": [],
            "component_summary": {},
        },
        "last": {},
        "timed_out": False,
        "missing_input_file": None,
        "assertion": None,
    }
    if not path.exists():
        return metrics

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        _update_last_metric(metrics["last"], line, "objective", (r"\bObj(?:ective)?\b",))
        _update_last_metric(metrics["last"], line, "hpwl", (r"\bwHPWL\b", r"\bHPWL\b", r"\bhpwl\b"))
        _update_last_metric(metrics["last"], line, "overflow", (r"\bOverflow\b", r"\boverflow\b"))
        _update_last_metric(metrics["last"], line, "wns", (r"\bWNS\b", r"\bwns\b"))
        _update_last_metric(metrics["last"], line, "tns", (r"\bTNS\b", r"\btns\b"))

        custom_total = _CUSTOM_TOTAL_RE.search(line)
        if custom_total:
            custom = metrics["custom_objective"]
            custom["objective_id"] = custom_total.group("objective_id")
            custom["calls"] = int(custom_total.group("call"))
            custom["last_total"] = float(custom_total.group("total"))
            custom["last_wirelength_raw"] = float(custom_total.group("wirelength_raw"))
            custom["last_weighted_density_raw"] = float(custom_total.group("weighted_density_raw"))
            continue

        custom_grad = _CUSTOM_GRAD_RE.search(line)
        if custom_grad:
            metrics["custom_objective"]["last_grad_norm"] = float(custom_grad.group("grad_norm"))
            continue

        term_scale = _CUSTOM_TERM_SCALE_RE.search(line)
        if term_scale:
            metrics["custom_objective"]["term_scales"][term_scale.group("term")] = float(
                term_scale.group("value")
            )
            continue

        output_scale = _CUSTOM_OUTPUT_SCALE_RE.search(line)
        if output_scale:
            metrics["custom_objective"]["output_scale"] = float(output_scale.group("value"))
            continue

        terms = _CUSTOM_TERMS_RE.search(line)
        if terms:
            try:
                parsed_terms = json.loads(terms.group("payload"))
                metrics["custom_objective"]["last_terms"] = parsed_terms
                metrics["custom_objective"]["term_history"].append(
                    {
                        "call": metrics["custom_objective"].get("calls"),
                        "terms": parsed_terms,
                    }
                )
            except json.JSONDecodeError:
                pass
            continue

        components = _CUSTOM_COMPONENTS_RE.search(line)
        if components:
            try:
                parsed_components = json.loads(components.group("payload"))
                call = parsed_components.get("call", metrics["custom_objective"].get("calls"))
                component_values = parsed_components.get("components", parsed_components)
                metrics["custom_objective"]["last_components"] = component_values
                metrics["custom_objective"]["component_history"].append(
                    {"call": call, "components": component_values}
                )
            except (AttributeError, json.JSONDecodeError):
                pass
            continue

        if "CoEvoP&R DREAMPlace timeout after" in line:
            metrics["timed_out"] = True
            continue

        if "Could not open input file" in line:
            metrics["missing_input_file"] = line.strip()
            continue

        if "[ASSERT" in line:
            metrics["assertion"] = line.strip()

    metrics["custom_objective"]["component_summary"] = summarize_component_history(
        metrics["custom_objective"].get("component_history", [])
    )
    return metrics


def summarize_component_history(history: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Summarize logged objective-component values for prompt feedback.

    DREAMPlace logs components as a sparse time series at the custom objective
    log interval. The summary keeps the signal compact enough for LLM memory
    while preserving whether each component changed, stayed flat, or saturated.
    """

    grouped: dict[str, list[tuple[int | None, float]]] = {}
    for item in history:
        if not isinstance(item, dict):
            continue
        call_value = item.get("call")
        try:
            call = int(call_value) if call_value is not None else None
        except (TypeError, ValueError):
            call = None
        components = item.get("components", {})
        if not isinstance(components, dict):
            continue
        for name, payload in components.items():
            numeric = _component_numeric_value(payload)
            if numeric is None:
                continue
            grouped.setdefault(str(name), []).append((call, numeric))

    summary: dict[str, dict[str, Any]] = {}
    for name, samples in grouped.items():
        samples.sort(key=lambda sample: (-1 if sample[0] is None else sample[0]))
        values = [value for _, value in samples]
        if not values:
            continue
        trajectory = _sample_component_trajectory(values, limit=10)
        start = values[0]
        mid = values[len(values) // 2]
        end = values[-1]
        min_value = min(values)
        max_value = max(values)
        mean_value = sum(values) / len(values)
        span = max_value - min_value
        scale = max(abs(mean_value), abs(start), abs(end), 1.0)
        flat = span <= max(1e-9, 1e-3 * scale)
        delta = end - start
        if flat:
            trend = "flat"
        elif delta > 0:
            trend = "up"
        elif delta < 0:
            trend = "down"
        else:
            trend = "mixed"
        saturated = flat and (abs(mean_value) >= 0.95 or abs(mean_value) <= 1e-9)
        summary[name] = {
            "count": len(values),
            "trajectory": trajectory,
            "start": start,
            "mid": mid,
            "end": end,
            "min": min_value,
            "max": max_value,
            "mean": mean_value,
            "trend": trend,
            "flat_or_saturated": bool(flat or saturated),
        }
    return summary


def _sample_component_trajectory(values: list[float], *, limit: int) -> list[float]:
    if limit <= 0:
        return []
    if len(values) <= limit:
        return [_round_component_value(value) for value in values]
    if limit == 1:
        return [_round_component_value(values[-1])]
    last_index = len(values) - 1
    sampled = []
    previous_index = -1
    for slot in range(limit):
        index = round(slot * last_index / (limit - 1))
        if index == previous_index:
            continue
        sampled.append(_round_component_value(values[index]))
        previous_index = index
    return sampled


def _round_component_value(value: float) -> float:
    return float(f"{value:.6g}")


def _component_numeric_value(payload: Any) -> float | None:
    if isinstance(payload, dict):
        for key in ("normalized", "value", "raw"):
            if key in payload:
                return _finite_component_float(payload.get(key))
        return None
    return _finite_component_float(payload)


def _finite_component_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not (numeric == numeric and abs(numeric) != float("inf")):
        return None
    return numeric


def _update_last_metric(
    target: dict[str, float],
    line: str,
    metric_name: str,
    labels: tuple[str, ...],
) -> None:
    for label in labels:
        match = re.search(rf"{label}\s*[:=, ]+\s*(?P<value>{_FLOAT})", line)
        if match:
            target[metric_name] = float(match.group("value"))
            return
