"""Fail-closed hard-macro audits for DREAMPlace experiment inputs."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from coevop.eval.shared_panel import load_shared_panel


_LEF_MACRO_RE = re.compile(
    r"(?ms)^\s*MACRO\s+(\S+)\s+(.*?)^\s*END\s+\1\s*$"
)
_LEF_BLOCK_RE = re.compile(r"(?m)^\s*CLASS\s+BLOCK\b")
_DEF_COMPONENTS_RE = re.compile(
    r"(?ms)^COMPONENTS\s+\d+\s*;\s*(.*?)^END COMPONENTS"
)
_DEF_COMPONENT_RE = re.compile(r"(?s)^\s*-\s+(\S+)\s+(\S+)(.*)$")
_DEF_STATUS_RE = re.compile(r"\+\s+(FIXED|PLACED|COVER|UNPLACED)\b")
_DEF_LOCATION_RE = re.compile(
    r"\+\s+(?:FIXED|PLACED|COVER)\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+(\S+)"
)


def audit_dreamplace_macro_config(
    config_path: str | Path,
    *,
    dreamplace_root: str | Path,
    require_movable_macros: bool = False,
    reject_fixed_hard_macros: bool = False,
    minimum_movable_macro_count: int = 1,
) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return audit_dreamplace_macro_payload(
        payload,
        config_path=path,
        dreamplace_root=dreamplace_root,
        require_movable_macros=require_movable_macros,
        reject_fixed_hard_macros=reject_fixed_hard_macros,
        minimum_movable_macro_count=minimum_movable_macro_count,
    )


def audit_dreamplace_macro_payload(
    payload: dict[str, Any],
    *,
    config_path: str | Path,
    dreamplace_root: str | Path,
    require_movable_macros: bool = False,
    reject_fixed_hard_macros: bool = False,
    minimum_movable_macro_count: int = 1,
) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    dreamplace_root = Path(dreamplace_root).expanduser().resolve()
    def_path = _resolve_input_path(
        payload.get("def_input"),
        config_path=config_path,
        dreamplace_root=dreamplace_root,
    )
    lef_values = payload.get("lef_input", [])
    if isinstance(lef_values, str):
        lef_values = [lef_values]
    lef_paths = [
        _resolve_input_path(value, config_path=config_path, dreamplace_root=dreamplace_root)
        for value in lef_values
    ]
    missing = [str(path) for path in [def_path, *lef_paths] if not path.is_file()]
    hard_macro_masters = _hard_macro_masters(path for path in lef_paths if path.is_file())
    components = (
        _hard_macro_components(def_path, hard_macro_masters)
        if def_path.is_file()
        else []
    )
    status_counts: dict[str, int] = {}
    for component in components:
        status = str(component["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    movable_count = sum(status_counts.get(status, 0) for status in ("PLACED", "UNPLACED"))
    fixed_count = sum(status_counts.get(status, 0) for status in ("FIXED", "COVER"))
    movable_instances = sorted(
        str(component["instance"])
        for component in components
        if component["status"] in ("PLACED", "UNPLACED")
    )
    selection = payload.get("coevop_superblue_macro_selection") or {}
    expected_movable_count = selection.get("expected_movable_macro_count")
    expected_movable_instances = sorted(
        str(value) for value in selection.get("movable_instances", [])
    )
    failures = []
    if missing:
        failures.append("missing DEF/LEF inputs")
    if require_movable_macros and movable_count < max(1, minimum_movable_macro_count):
        failures.append(
            f"movable hard-macro count {movable_count} is below required "
            f"minimum {max(1, minimum_movable_macro_count)}"
        )
    if reject_fixed_hard_macros and fixed_count:
        failures.append(f"input contains {fixed_count} FIXED/COVER hard macros")
    if expected_movable_count is not None and movable_count != int(expected_movable_count):
        failures.append(
            f"movable hard-macro count {movable_count} does not match expected "
            f"count {int(expected_movable_count)}"
        )
    if expected_movable_instances and movable_instances != expected_movable_instances:
        failures.append("movable hard-macro instance set does not match selection manifest")
    return {
        "config_path": str(config_path),
        "def_path": str(def_path),
        "def_sha256": _sha256(def_path) if def_path.is_file() else None,
        "lef_paths": [str(path) for path in lef_paths],
        "missing_inputs": missing,
        "hard_macro_master_count": len(hard_macro_masters),
        "hard_macro_count": len(components),
        "movable_macro_count": movable_count,
        "fixed_macro_count": fixed_count,
        "movable_instances": movable_instances,
        "expected_movable_macro_count": expected_movable_count,
        "movable_instance_set_matches": (
            movable_instances == expected_movable_instances
            if expected_movable_instances
            else None
        ),
        "status_counts": status_counts,
        "macro_place_flag": payload.get("macro_place_flag"),
        "require_movable_macros": require_movable_macros,
        "reject_fixed_hard_macros": reject_fixed_hard_macros,
        "minimum_movable_macro_count": minimum_movable_macro_count,
        "failures": failures,
        "ok": not failures,
        "components": components,
    }


def audit_shared_panel_macros(
    panel_path: str | Path,
    *,
    dreamplace_root: str | Path,
    require_movable_macros: bool,
    reject_fixed_hard_macros: bool,
    minimum_movable_macro_count: int,
) -> dict[str, Any]:
    panel = load_shared_panel(panel_path)
    designs = []
    for design in panel.designs:
        audit = audit_dreamplace_macro_config(
            design.dreamplace_config,
            dreamplace_root=dreamplace_root,
            require_movable_macros=require_movable_macros,
            reject_fixed_hard_macros=reject_fixed_hard_macros,
            minimum_movable_macro_count=minimum_movable_macro_count,
        )
        audit["design"] = design.name
        designs.append(audit)
    return {
        "panel_path": str(Path(panel_path).resolve()),
        "require_movable_macros": require_movable_macros,
        "reject_fixed_hard_macros": reject_fixed_hard_macros,
        "minimum_movable_macro_count": minimum_movable_macro_count,
        "designs": designs,
        "ok": all(item["ok"] for item in designs),
        "failed_designs": [item["design"] for item in designs if not item["ok"]],
    }


def load_hard_macro_components(
    def_path: str | Path,
    lef_paths: list[str | Path],
) -> list[dict[str, Any]]:
    """Return hard-macro instances and coordinates from one DEF/LEF set."""

    resolved_def = Path(def_path).expanduser().resolve()
    resolved_lefs = [Path(path).expanduser().resolve() for path in lef_paths]
    masters = _hard_macro_masters(path for path in resolved_lefs if path.is_file())
    return _hard_macro_components(resolved_def, masters) if resolved_def.is_file() else []


def audit_output_macro_coordinates(
    *,
    input_components: list[dict[str, Any]],
    output_def: str | Path,
) -> dict[str, Any]:
    """Verify that an emitted DEF contains coordinates for every hard macro."""

    path = Path(output_def).expanduser().resolve()
    masters = {str(item.get("master")) for item in input_components if item.get("master")}
    expected = {str(item["instance"]): item for item in input_components}
    output_components = _hard_macro_components(path, masters) if path.is_file() else []
    all_matching_master_instances = {
        str(item["instance"]): item for item in output_components
    }
    # A macro master may be instantiated by both selected movable macros and
    # fixed obstacles. The audit contract is instance-based: validate only the
    # selected input instances and retain same-master fixed instances as
    # diagnostics instead of treating them as unexpected movable macros.
    observed = {
        name: item
        for name, item in all_matching_master_instances.items()
        if name in expected
    }
    missing = sorted(set(expected) - set(observed))
    nonselected_same_master = sorted(
        set(all_matching_master_instances) - set(expected)
    )
    coordinate_count = sum(
        item.get("x") is not None and item.get("y") is not None for item in observed.values()
    )
    comparable = 0
    changed = 0
    for name, before in expected.items():
        after = observed.get(name)
        if after is None or before.get("x") is None or before.get("y") is None:
            continue
        comparable += 1
        if (before.get("x"), before.get("y")) != (after.get("x"), after.get("y")):
            changed += 1
    coordinate_payload = [
        [name, item.get("x"), item.get("y"), item.get("orientation")]
        for name, item in sorted(observed.items())
    ]
    coordinate_hash = hashlib.sha256(
        json.dumps(coordinate_payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected_count = len(expected)
    ok = (
        path.is_file()
        and expected_count > 0
        and not missing
        and coordinate_count == expected_count
    )
    return {
        "output_def": str(path),
        "output_def_sha256": _sha256(path) if path.is_file() else None,
        "expected_macro_count": expected_count,
        "observed_macro_count": len(observed),
        "coordinate_count": coordinate_count,
        "input_unplaced_count": sum(item.get("x") is None for item in expected.values()),
        "comparable_coordinate_count": comparable,
        "changed_coordinate_count": changed,
        "missing_instances": missing,
        "unexpected_instances": [],
        "same_master_output_instance_count": len(all_matching_master_instances),
        "nonselected_same_master_instance_count": len(nonselected_same_master),
        "nonselected_same_master_instances_sample": nonselected_same_master[:20],
        "coordinate_sha256": coordinate_hash,
        "ok": ok,
    }


def _resolve_input_path(
    value: Any,
    *,
    config_path: Path,
    dreamplace_root: Path,
) -> Path:
    if value is None:
        return config_path.parent / "__missing_input__"
    raw = Path(os.path.expandvars(str(value))).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    candidates = [dreamplace_root / raw, config_path.parent / raw, Path.cwd() / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _hard_macro_masters(lef_paths: Any) -> set[str]:
    masters: set[str] = set()
    for path in lef_paths:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        for match in _LEF_MACRO_RE.finditer(text):
            if _LEF_BLOCK_RE.search(match.group(2)):
                masters.add(match.group(1))
    return masters


def _hard_macro_components(def_path: Path, masters: set[str]) -> list[dict[str, Any]]:
    text = def_path.read_text(encoding="utf-8", errors="replace")
    section = _DEF_COMPONENTS_RE.search(text)
    if section is None:
        return []
    components = []
    for statement in section.group(1).split(";"):
        match = _DEF_COMPONENT_RE.match(statement)
        if match is None or match.group(2) not in masters:
            continue
        status_match = _DEF_STATUS_RE.search(match.group(3))
        location_match = _DEF_LOCATION_RE.search(match.group(3))
        components.append(
            {
                "instance": match.group(1),
                "master": match.group(2),
                "status": status_match.group(1) if status_match else "UNPLACED",
                "x": int(location_match.group(1)) if location_match else None,
                "y": int(location_match.group(2)) if location_match else None,
                "orientation": location_match.group(3) if location_match else None,
            }
        )
    return components


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
