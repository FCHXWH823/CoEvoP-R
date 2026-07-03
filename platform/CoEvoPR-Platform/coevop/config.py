"""Configuration loading for local CoEvoP&R runs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10.
    import tomli as tomllib


@dataclass(frozen=True)
class CoevopConfig:
    circuitnet_root: Path | None
    dreamplace_root: Path
    chipbench_root: Path
    run_root: Path
    llm_provider: str
    openroad_backend: str


def _expand_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    expanded = os.path.expandvars(value)
    expanded = os.path.expanduser(expanded)
    if expanded.startswith("${") and expanded.endswith("}"):
        return ""
    return expanded


def _path_or_none(value: Any) -> Path | None:
    expanded = _expand_value(value)
    if expanded is None or expanded == "":
        return None
    return Path(str(expanded))


def _path(value: Any) -> Path:
    expanded = _expand_value(value)
    if expanded is None or expanded == "":
        raise ValueError("required path config value is empty")
    return Path(str(expanded))


def load_config(path: str | Path = "configs/default.toml") -> CoevopConfig:
    config_path = Path(path)
    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    return CoevopConfig(
        circuitnet_root=_path_or_none(os.environ.get("CIRCUITNET_ROOT", raw.get("circuitnet_root"))),
        dreamplace_root=_path(os.environ.get("DREAMPLACE_ROOT", raw.get("dreamplace_root"))),
        chipbench_root=_path(os.environ.get("CHIPBENCH_ROOT", raw.get("chipbench_root"))),
        run_root=_path(os.environ.get("RUN_ROOT", raw.get("run_root"))),
        llm_provider=str(os.environ.get("LLM_PROVIDER", raw.get("llm_provider", "mock"))),
        openroad_backend=str(os.environ.get("OPENROAD_BACKEND", raw.get("openroad_backend", "chipbench"))),
    )
