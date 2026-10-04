"""Evolution configurations for the exposure settings evaluated in the paper.

CoEvoP&R-E uses feedback from the same design on which it is evaluated,
CoEvoP&R-L leaves the target design out of objective evolution and prompt
evidence, and CoEvoP&R-T transfers the best candidate of the ChiPBench
evolution without further search. The E and L configurations are derived from
the primary configuration, so every setting runs the same method with the same
budget and differs only in which designs supply feedback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised by py310 via tomli
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


PRIMARY_CONFIG = Path("configs/openevolve_tier2/chipbench_controller_tier2.toml")


@dataclass(frozen=True)
class BenchmarkFamily:
    name: str
    # Target name accepted on the command line -> design name used in panels.
    designs: dict[str, str]
    dreamplace_config_dir: str
    post_route_panel: str
    # Placement-stage timing panel, when the family ships timing collateral.
    timing_panel: str | None = None
    timing_controller_panel: str | None = None


FAMILIES: dict[str, BenchmarkFamily] = {
    "chipbench": BenchmarkFamily(
        name="chipbench",
        designs={
            name: name
            for name in (
                "bp_fe",
                "bp_be",
                "swerv_wrapper",
                "ethernet",
                "dft68",
                "or1200",
                "vga_lcd",
                "mor1kx",
            )
        },
        dreamplace_config_dir="configs/dreamplace_base/chipbench_movable",
        post_route_panel="configs/shared_panels/chipbench_table1_post_route.toml",
        timing_panel="configs/timing_panels/chipbench_proxy_audit.toml",
        timing_controller_panel="configs/timing_panels/chipbench_timing_controller.toml",
    ),
    "superblue": BenchmarkFamily(
        name="superblue",
        designs={
            number: f"superblue{number}_ot_notiming"
            for number in ("1", "3", "4", "5", "7", "10", "16", "18")
        },
        dreamplace_config_dir="configs/dreamplace_base/iccad2015_notiming",
        post_route_panel="configs/shared_panels/superblue_post_route.toml",
    ),
    "asap7": BenchmarkFamily(
        name="asap7",
        designs={name: name for name in ("gcd", "ibex", "ariane")},
        dreamplace_config_dir="configs/dreamplace_base/asap7",
        post_route_panel="configs/shared_panels/asap7_post_route.toml",
    ),
}


def write_target_evolution_config(
    *,
    family: str,
    design: str,
    output_dir: str | Path,
    repo_root: str | Path,
) -> Path:
    """Write the CoEvoP&R-E configuration for one target design."""

    benchmark = _family(family)
    if design not in benchmark.designs:
        raise ValueError(
            f"unknown {family} design {design!r}; expected one of {sorted(benchmark.designs)}"
        )
    root = Path(repo_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = _primary_config(root)
    panel_name = benchmark.designs[design]

    search_panel = _write_placement_panel(
        output / "search.toml",
        template=_primary_panel(root, config["search"]["panel"]),
        designs=[panel_name],
        benchmark=benchmark,
        root=root,
    )
    final_panel = _write_placement_panel(
        output / "final.toml",
        template=_primary_panel(root, config["final"]["panel"]),
        designs=[panel_name],
        benchmark=benchmark,
        root=root,
    )
    config["target_designs"] = [panel_name]
    config["search"]["panel"] = search_panel.as_posix()
    config["final"]["panel"] = final_panel.as_posix()
    _bind_family(config, benchmark, root, search_designs=[panel_name])
    return _write_config(config, output / "openevolve.toml")


def write_lodo_config(
    *,
    heldout: str,
    output_dir: str | Path,
    repo_root: str | Path,
    family: str = "chipbench",
) -> Path:
    """Write the CoEvoP&R-L configuration that leaves one design out."""

    benchmark = _family(family)
    if heldout not in benchmark.designs:
        raise ValueError(
            f"unknown {family} design {heldout!r}; expected one of {sorted(benchmark.designs)}"
        )
    root = Path(repo_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = _primary_config(root)
    training = [
        panel_name for name, panel_name in benchmark.designs.items() if name != heldout
    ]

    search_panel = _write_placement_panel(
        output / "search.toml",
        template=_primary_panel(root, config["search"]["panel"]),
        designs=training,
        benchmark=benchmark,
        root=root,
    )
    heldout_panel = _write_placement_panel(
        output / "heldout.toml",
        template=_primary_panel(root, config["final"]["panel"]),
        designs=[benchmark.designs[heldout]],
        benchmark=benchmark,
        root=root,
    )
    config["target_designs"] = training
    config["search"]["panel"] = search_panel.as_posix()
    # The held-out design takes the place of the final panel: it is placed and
    # routed only after evolution ends, for the same number of candidates.
    final = config["final"]
    config["final"] = {"enabled": False}
    config["generalization"] = {
        "enabled": True,
        "panel": heldout_panel.as_posix(),
        "top_k": int(final.get("top_k", 8)),
        "candidate_policy": str(config["tier3"].get("candidate_policy", "elite_or_aggregate")),
        "blind": True,
        "strict_leakage_guard": True,
        "baseline_presets": list(final.get("baseline_presets", [])),
    }
    _bind_family(config, benchmark, root, search_designs=training)
    return _write_config(config, output / "openevolve.toml")


def _family(name: str) -> BenchmarkFamily:
    if name not in FAMILIES:
        raise ValueError(f"unknown benchmark family {name!r}; expected one of {sorted(FAMILIES)}")
    return FAMILIES[name]


def _primary_config(root: Path) -> dict[str, Any]:
    return tomllib.loads((root / PRIMARY_CONFIG).read_text(encoding="utf-8"))


def _primary_panel(root: Path, relative: str) -> dict[str, Any]:
    path = ((root / PRIMARY_CONFIG).parent / relative).resolve()
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _bind_family(
    config: dict[str, Any],
    benchmark: BenchmarkFamily,
    root: Path,
    *,
    search_designs: list[str],
) -> None:
    """Point the family-specific panels at absolute paths under the repository."""

    primary_dir = (root / PRIMARY_CONFIG).parent
    if config.get("router_background"):
        config["router_background"] = (
            (primary_dir / str(config["router_background"])).resolve().as_posix()
        )
    config["tier3"]["panel"] = (root / benchmark.post_route_panel).as_posix()

    # Tier B needs timing collateral for at least one search design. Without
    # it the candidates carry placement and routed evidence only.
    timing_designs = _panel_designs(root, benchmark.timing_panel)
    timing_proxy = dict(config.get("timing_proxy") or {})
    if set(search_designs) & timing_designs:
        timing_proxy["panel"] = (root / str(benchmark.timing_panel)).as_posix()
    else:
        timing_proxy = {"enabled": False}
    config["timing_proxy"] = timing_proxy

    # update_net_weights needs timing analysis on every search design.
    timing_controller = dict(config.get("timing_controller") or {})
    timing_controller.pop("panel", None)
    if set(search_designs) <= _panel_designs(root, benchmark.timing_controller_panel):
        timing_controller["panel"] = (root / str(benchmark.timing_controller_panel)).as_posix()
    config["timing_controller"] = timing_controller


def _panel_designs(root: Path, relative: str | None) -> set[str]:
    if not relative or not (root / relative).is_file():
        return set()
    payload = tomllib.loads((root / relative).read_text(encoding="utf-8"))
    return {str(item["name"]) for item in payload.get("designs", [])}


def _write_placement_panel(
    path: Path,
    *,
    template: dict[str, Any],
    designs: list[str],
    benchmark: BenchmarkFamily,
    root: Path,
) -> Path:
    settings = {key: value for key, value in template.items() if key != "designs"}
    lines = [_toml_assignment(key, value) for key, value in settings.items()]
    for design in designs:
        base = root / benchmark.dreamplace_config_dir / f"{design}.json"
        lines.extend(
            [
                "",
                "[[designs]]",
                _toml_assignment("name", design),
                _toml_assignment("dreamplace_config", base.as_posix()),
                _toml_assignment("timing_mode", str(settings.get("timing_mode", "none"))),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_config(config: dict[str, Any], path: Path) -> Path:
    scalars = {key: value for key, value in config.items() if not isinstance(value, dict)}
    tables = {key: value for key, value in config.items() if isinstance(value, dict)}
    lines = [_toml_assignment(key, value) for key, value in scalars.items()]
    for name, table in tables.items():
        lines.extend(["", f"[{name}]"])
        lines.extend(_toml_assignment(key, value) for key, value in table.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _toml_assignment(key: str, value: Any) -> str:
    return f"{key} = {_toml_value(value)}"


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        # JSON string escapes are valid TOML basic-string escapes.
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported configuration value: {value!r}")
