"""Chip-specific profile extraction for OpenEvolve/DREAMPlace prompts."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

from coevop.eval.shared_panel import SharedPanel, SharedPanelDesign, load_shared_panel


DEFAULT_DREAMPLACE_PROFILE_KEYS = (
    "aux_input",
    "lef_input",
    "def_input",
    "verilog_input",
    "bookshelf_variety",
    "unit_horizontal_capacity",
    "unit_vertical_capacity",
    "num_bins_x",
    "num_bins_y",
    "target_density",
    "global_place_stages",
    "gpu",
    "random_seed",
    "timing_opt_flag",
    "routability_opt_flag",
    "enable_fillers",
    "legalize_flag",
)

DEFAULT_CHIPBENCH_PROFILE_KEYS = (
    "DESIGN_NAME",
    "PLATFORM",
    "VERILOG_FILES",
    "SDC_FILE",
    "DIE_AREA",
    "CORE_AREA",
    "PLACE_DENSITY",
    "CORE_UTILIZATION",
    "PLACE_SITE",
    "ADDITIONAL_LEFS",
    "MACRO_PLACEMENT",
    "CLOCK_PORT",
    "CLOCK_PERIOD",
    "GRT_ADJUSTMENT",
)


def build_design_profiles(
    *,
    search_panel_path: str | Path,
    target_designs: list[str] | None = None,
    generalization_panel_path: str | Path | None = None,
    include_file_stats: bool = True,
    max_file_examples: int = 5,
) -> dict[str, Any]:
    """Build prompt-safe static design profiles for the configured panels.

    The profile is intentionally limited to structural/configuration facts that
    are available before evolution starts. It does not read held-out metrics or
    final-panel outcomes.
    """

    search_panel = load_shared_panel(search_panel_path)
    target_names = list(target_designs or [design.name for design in search_panel.designs])
    search_profiles = [
        _profile_design(
            panel=search_panel,
            design=design,
            role=("optimization_target" if design.name in target_names else "search_context"),
            include_file_stats=include_file_stats,
            max_file_examples=max_file_examples,
        )
        for design in search_panel.designs
    ]
    heldout_profiles: list[dict[str, Any]] = []
    if generalization_panel_path:
        generalization_panel = load_shared_panel(generalization_panel_path)
        heldout_profiles = [
            _profile_design(
                panel=generalization_panel,
                design=design,
                role="heldout_generalization",
                include_file_stats=include_file_stats,
                max_file_examples=max_file_examples,
            )
            for design in generalization_panel.designs
        ]

    return {
        "profile_version": 2,
        "source": "static_panel_and_config_files",
        "measurement_policy": (
            "Profiles contain pre-run design/configuration facts only. Held-out "
            "generalization metrics are not included in LLM prompt feedback."
        ),
        "optimization_designs": target_names,
        "search_panel": _panel_summary(search_panel),
        "search_design_profiles": search_profiles,
        "heldout_generalization_profiles": heldout_profiles,
    }


def summarize_baseline_behavior(
    rows: list[dict[str, Any]],
    *,
    objective_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Summarize measured search-panel baseline behavior for prompt memory.

    This summary is safe for later prompts because it is derived from the same
    search/evolution panel used for parent selection. Do not call this on final
    or held-out generalization rows before evolution is complete.
    """

    objective_filter = set(objective_ids or ())
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        objective_id = str(row.get("objective_id") or "")
        if objective_filter and objective_id not in objective_filter:
            continue
        design = str(row.get("design") or "unknown")
        grouped.setdefault((design, objective_id), []).append(row)

    by_design: dict[str, list[dict[str, Any]]] = {}
    for (design, objective_id), items in sorted(grouped.items()):
        hpwl_values = [_finite_number(row.get("hpwl_delta_pct")) for row in items]
        overflow_values = [_finite_number(row.get("overflow_delta_pct")) for row in items]
        runtime_values = [_finite_number(row.get("runtime_seconds")) for row in items]
        success_count = sum(1 for row in items if row.get("status") == "success")
        by_design.setdefault(design, []).append(
            {
                "objective_id": objective_id,
                "cell_count": len(items),
                "success_count": success_count,
                "mean_hpwl_delta_pct": _mean(item for item in hpwl_values if item is not None),
                "mean_overflow_delta_pct": _mean(
                    item for item in overflow_values if item is not None
                ),
                "mean_runtime_seconds": _mean(
                    item for item in runtime_values if item is not None
                ),
                "outcome_counts": _outcome_counts(items),
            }
        )

    notes = []
    for design, summaries in by_design.items():
        best_overflow = _best_by_metric(summaries, "mean_overflow_delta_pct")
        best_hpwl = _best_by_metric(summaries, "mean_hpwl_delta_pct")
        if best_overflow:
            notes.append(
                f"{design}: strongest overflow baseline is {best_overflow['objective_id']} "
                f"({best_overflow.get('mean_overflow_delta_pct')})"
            )
        if best_hpwl:
            notes.append(
                f"{design}: strongest HPWL baseline is {best_hpwl['objective_id']} "
                f"({best_hpwl.get('mean_hpwl_delta_pct')})"
            )

    return {
        "source": "measured_search_panel_rows",
        "objective_filter": sorted(objective_filter),
        "by_design": by_design,
        "notes": notes,
    }


def merge_baseline_behavior(
    design_profiles: dict[str, Any],
    baseline_behavior: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a copy of design profiles with search-panel behavior attached."""

    payload = json.loads(json.dumps(design_profiles))
    if baseline_behavior:
        payload["search_baseline_behavior"] = baseline_behavior
    return payload


def _profile_design(
    *,
    panel: SharedPanel,
    design: SharedPanelDesign,
    role: str,
    include_file_stats: bool,
    max_file_examples: int,
) -> dict[str, Any]:
    dreamplace_config = _read_json_file(design.dreamplace_config)
    chipbench_config = _read_makefile_vars(design.chipbench_config)
    dreamplace_selected = _select_keys(dreamplace_config, DEFAULT_DREAMPLACE_PROFILE_KEYS)
    chipbench_selected = _select_keys(chipbench_config, DEFAULT_CHIPBENCH_PROFILE_KEYS)
    geometry = _geometry_summary(chipbench_selected)
    bookshelf = _bookshelf_summary(dreamplace_config)
    macro = _macro_summary(dreamplace_config)
    file_summary = (
        _file_summary(
            dreamplace_config=dreamplace_config,
            chipbench_config=chipbench_config,
            bookshelf_summary=bookshelf,
            max_examples=max_file_examples,
        )
        if include_file_stats
        else {}
    )
    inferred_pressure = _inferred_pressure_notes(
        dreamplace=dreamplace_selected,
        chipbench=chipbench_selected,
        geometry=geometry,
        bookshelf=bookshelf,
        file_summary=file_summary,
    )
    return {
        "name": design.name,
        "role": role,
        "mode": design.mode,
        "timing_mode": design.timing_mode,
        "panel_iterations": panel.iterations,
        "panel_seed_count": len(panel.seeds),
        "dreamplace_config": design.dreamplace_config,
        "chipbench_config": design.chipbench_config,
        "reference_def_available": bool(design.reference_def),
        "dreamplace_config_summary": dreamplace_selected,
        "chipbench_config_summary": chipbench_selected,
        "geometry_summary": geometry,
        "bookshelf_summary": bookshelf,
        "macro_placement_summary": macro,
        "input_file_summary": file_summary,
        "inferred_placement_pressure": inferred_pressure,
    }


def _macro_summary(dreamplace_config: dict[str, Any]) -> dict[str, Any]:
    """Return compact, pre-run macro facts safe for LLM prompt context."""

    audit = dreamplace_config.get("coevop_macro_preflight") or {}
    generation = dreamplace_config.get("coevop_chipbench_generation_audit") or {}
    movable_count = int(audit.get("movable_macro_count") or 0)
    fixed_count = int(audit.get("fixed_macro_count") or 0)
    macro_place_flag = int(dreamplace_config.get("macro_place_flag") or 0)
    joint_global_placement = movable_count > 0
    return {
        "input_stage": generation.get("input_stage"),
        "macro_place_flag": macro_place_flag,
        "hard_macro_count": int(audit.get("hard_macro_count") or 0),
        "movable_macro_count": movable_count,
        "fixed_macro_count": fixed_count,
        "status_counts": dict(audit.get("status_counts") or {}),
        "macro_halo_x": _finite_number(dreamplace_config.get("macro_halo_x")),
        "macro_halo_y": _finite_number(dreamplace_config.get("macro_halo_y")),
        "input_def_sha256": audit.get("def_sha256"),
        "joint_macro_cell_global_placement": joint_global_placement,
        "specialized_macro_legalizer_enabled": bool(macro_place_flag == 1),
        "fixed_hard_macros_are_obstacles": bool(fixed_count > 0),
        "objective_scope_note": (
            "The evolved objective jointly optimizes the DEF-selected movable macros "
            "and standard cells during global placement. DREAMPlace's specialized "
            "macro legalizer is enabled for this run."
            if joint_global_placement and macro_place_flag == 1
            else (
                "The evolved objective jointly optimizes the DEF-selected movable "
                "macros and standard cells during global placement. The specialized "
                "macro legalizer is disabled and legality is audited separately."
                if joint_global_placement
                else "The current input does not provide verified movable-macro placement."
            )
        ),
    }


def _panel_summary(panel: SharedPanel) -> dict[str, Any]:
    return {
        "design_count": len(panel.designs),
        "seeds": panel.seeds,
        "iterations": panel.iterations,
        "timing_mode": panel.timing_mode,
        "gpu": panel.gpu,
        "timeout_seconds": panel.timeout_seconds,
        "log_interval": panel.log_interval,
    }


def _read_json_file(path: str | Path) -> dict[str, Any]:
    candidate = Path(os.path.expandvars(str(path))).expanduser()
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_makefile_vars(path: str | Path) -> dict[str, str]:
    candidate = Path(os.path.expandvars(str(path))).expanduser()
    try:
        lines = candidate.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return {}
    values: dict[str, str] = {}
    continuation_key: str | None = None
    continuation_value = ""
    for raw_line in lines:
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if continuation_key:
            continued = line.endswith("\\")
            continuation_value += " " + line.rstrip("\\").strip()
            if not continued:
                values[continuation_key] = continuation_value.strip()
                continuation_key = None
                continuation_value = ""
            continue
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?::=|\?=|\+=|=)\s*(.*)$", line)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        if value.endswith("\\"):
            continuation_key = key
            continuation_value = value.rstrip("\\").strip()
        else:
            values[key] = value
    return values


def _select_keys(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    selected = {}
    for key in keys:
        if key in payload:
            selected[key] = _prompt_safe_value(payload[key])
    return selected


def _prompt_safe_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_prompt_safe_value(item) for item in value[:12]]
    if isinstance(value, dict):
        return {
            str(key): _prompt_safe_value(item)
            for key, item in list(value.items())[:12]
        }
    return str(value)


def _geometry_summary(chipbench: dict[str, Any]) -> dict[str, Any]:
    die = _parse_rect(chipbench.get("DIE_AREA"))
    core = _parse_rect(chipbench.get("CORE_AREA"))
    summary: dict[str, Any] = {}
    if die:
        summary["die_area"] = die
    if core:
        summary["core_area"] = core
    if die and core:
        die_area = die["width"] * die["height"]
        core_area = core["width"] * core["height"]
        summary["core_to_die_area_ratio"] = (
            core_area / die_area if die_area > 0 else None
        )
    for key in ("PLACE_DENSITY", "CORE_UTILIZATION", "CLOCK_PERIOD"):
        numeric = _finite_number(chipbench.get(key))
        if numeric is not None:
            summary[key.lower()] = numeric
    return summary


def _parse_rect(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    numbers = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", str(value))]
    if len(numbers) < 4:
        return None
    x0, y0, x1, y1 = numbers[:4]
    width = abs(x1 - x0)
    height = abs(y1 - y0)
    return {
        "llx": x0,
        "lly": y0,
        "urx": x1,
        "ury": y1,
        "width": width,
        "height": height,
        "area": width * height,
        "aspect_ratio": width / height if height > 0 else None,
    }


def _file_summary(
    *,
    dreamplace_config: dict[str, Any],
    chipbench_config: dict[str, str],
    bookshelf_summary: dict[str, Any],
    max_examples: int,
) -> dict[str, Any]:
    paths = []
    for key in (
        "aux_input",
        "lef_input",
        "def_input",
        "verilog_input",
        "placement_file",
    ):
        paths.extend(_extract_paths(dreamplace_config.get(key)))
    for key in (
        "VERILOG_FILES",
        "SDC_FILE",
        "ADDITIONAL_LEFS",
        "MACRO_PLACEMENT",
    ):
        paths.extend(_extract_paths(chipbench_config.get(key)))
    paths.extend(
        str(item.get("path"))
        for item in bookshelf_summary.get("resolved_files", [])
        if isinstance(item, dict) and item.get("path")
    )
    resolved = []
    for text in paths:
        candidate = Path(os.path.expandvars(text)).expanduser()
        item: dict[str, Any] = {"path": text, "exists": candidate.exists()}
        if candidate.exists() and candidate.is_file():
            try:
                item["size_bytes"] = candidate.stat().st_size
            except OSError:
                pass
        resolved.append(item)
    existing = [item for item in resolved if item.get("exists")]
    suffix_counts: dict[str, int] = {}
    for item in existing:
        suffix = Path(str(item["path"])).suffix.lower() or "<no_suffix>"
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    return {
        "declared_file_count": len(resolved),
        "existing_file_count": len(existing),
        "suffix_counts": suffix_counts,
        "examples": resolved[:max_examples],
    }


def _extract_paths(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        paths = []
        for item in value:
            paths.extend(_extract_paths(item))
        return paths
    text = str(value)
    if not text:
        return []
    return [
        token
        for token in re.split(r"\s+", text)
        if token
        and not token.startswith("$(")
        and not token.startswith("-")
        and any(token.lower().endswith(suffix) for suffix in (".aux", ".lef", ".def", ".v", ".vg", ".sdc", ".pl"))
    ]


def _inferred_pressure_notes(
    *,
    dreamplace: dict[str, Any],
    chipbench: dict[str, Any],
    geometry: dict[str, Any],
    bookshelf: dict[str, Any],
    file_summary: dict[str, Any],
) -> list[str]:
    notes = []
    target_density = _finite_number(dreamplace.get("target_density"))
    place_density = _finite_number(chipbench.get("PLACE_DENSITY"))
    core_util = _finite_number(chipbench.get("CORE_UTILIZATION"))
    density_value = target_density or place_density or core_util
    if density_value is not None:
        if density_value >= 0.75:
            notes.append("high-utilization placement; density and routing-pressure terms may interact strongly")
        elif density_value <= 0.55:
            notes.append("lower-utilization placement; excessive spreading may hurt wirelength")
    if geometry.get("core_area", {}).get("aspect_ratio"):
        aspect = geometry["core_area"]["aspect_ratio"]
        if aspect >= 2.0 or aspect <= 0.5:
            notes.append("non-square core aspect ratio; long-net directional pressure may matter")
    suffix_counts = file_summary.get("suffix_counts", {}) if file_summary else {}
    if suffix_counts.get(".def") or suffix_counts.get(".lef"):
        notes.append("LEF/DEF-style physical data is available for DREAMPlace/OpenROAD compatibility")
    if bookshelf.get("format") == "bookshelf":
        notes.append("Bookshelf-style placement data is available; OpenROAD routing collateral is not implied")
    if _finite_number(bookshelf.get("num_terminals")):
        terminal_fraction = _safe_divide(
            _finite_number(bookshelf.get("num_terminals")),
            _finite_number(bookshelf.get("num_nodes")),
        )
        if terminal_fraction is not None and terminal_fraction >= 0.15:
            notes.append("large fixed-terminal fraction with limited spreading capacity")
    if _finite_number(bookshelf.get("num_rows")):
        notes.append("row/site information was parsed from Bookshelf SCL collateral")
    if dreamplace.get("timing_opt_flag") or chipbench.get("CLOCK_PERIOD"):
        notes.append("timing information is configured, but timing objective terms remain experimental unless enabled")
    return notes


def _bookshelf_summary(dreamplace_config: dict[str, Any]) -> dict[str, Any]:
    aux = dreamplace_config.get("aux_input")
    if not aux:
        return {}
    aux_path = Path(os.path.expandvars(str(aux))).expanduser()
    summary: dict[str, Any] = {
        "format": "bookshelf",
        "aux_input": str(aux),
        "aux_exists": aux_path.is_file(),
        "resolved_files": [],
    }
    if not aux_path.is_file():
        return summary
    try:
        text = aux_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return summary
    names = [
        token
        for token in re.split(r"\s+", text.replace(":", " "))
        if token.lower().endswith((".nodes", ".nets", ".scl", ".pl"))
    ]
    for name in names:
        path = (aux_path.parent / name).resolve()
        item = {"kind": path.suffix.lower().lstrip("."), "path": str(path), "exists": path.is_file()}
        summary["resolved_files"].append(item)
        if not path.is_file():
            continue
        if path.suffix.lower() == ".nodes":
            summary.update(_parse_bookshelf_nodes(path))
        elif path.suffix.lower() == ".nets":
            summary.update(_parse_bookshelf_nets(path))
        elif path.suffix.lower() == ".scl":
            summary.update(_parse_bookshelf_scl(path))
        elif path.suffix.lower() == ".pl":
            summary.update(_parse_bookshelf_pl(path))
    return summary


def _parse_bookshelf_nodes(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return payload
    macro_like = 0
    for line in lines:
        if line.startswith("NumNodes"):
            payload["num_nodes"] = _first_number(line)
        elif line.startswith("NumTerminals"):
            payload["num_terminals"] = _first_number(line)
        elif line.strip() and not line.startswith(("UCLA", "#", "Num")):
            numbers = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", line)]
            if len(numbers) >= 2 and numbers[0] * numbers[1] > 10000:
                macro_like += 1
    if macro_like:
        payload["macro_like_large_nodes"] = macro_like
    return payload


def _parse_bookshelf_nets(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return payload
    for line in lines[:2000]:
        if line.startswith("NumNets"):
            payload["num_nets"] = _first_number(line)
        elif line.startswith("NumPins"):
            payload["num_pins"] = _first_number(line)
    return payload


def _parse_bookshelf_scl(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return payload
    widths = []
    for line in lines:
        if line.startswith("NumRows"):
            payload["num_rows"] = _first_number(line)
        elif "Sitewidth" in line:
            value = _first_number(line)
            if value is not None:
                widths.append(value)
    if widths:
        payload["site_width_examples"] = widths[:3]
    return payload


def _parse_bookshelf_pl(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return payload
    fixed = 0
    movable = 0
    xs = []
    ys = []
    for line in lines:
        if not line.strip() or line.startswith(("UCLA", "#")):
            continue
        if "/FIXED" in line or "/FIXED_NI" in line:
            fixed += 1
        else:
            movable += 1
        numbers = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", line)]
        if len(numbers) >= 2:
            xs.append(numbers[0])
            ys.append(numbers[1])
    payload["pl_fixed_count"] = fixed
    payload["pl_movable_count"] = movable
    if xs and ys:
        payload["placement_bbox"] = {
            "llx": min(xs),
            "lly": min(ys),
            "urx": max(xs),
            "ury": max(ys),
            "width": max(xs) - min(xs),
            "height": max(ys) - min(ys),
        }
    return payload


def _first_number(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


def _outcome_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("outcome_label") or row.get("status") or "unknown")
        counts[label] = counts.get(label, 0) + 1
    return counts


def _best_by_metric(items: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    finite = [
        item
        for item in items
        if _finite_number(item.get(key)) is not None
    ]
    if not finite:
        return None
    return min(finite, key=lambda item: float(item[key]))


def _mean(values: Any) -> float | None:
    finite = [float(value) for value in values if _finite_number(value) is not None]
    return sum(finite) / len(finite) if finite else None


def _finite_number(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None
