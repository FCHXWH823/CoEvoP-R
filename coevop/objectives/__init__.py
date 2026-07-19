"""Validated placement objective programs."""

from coevop.objectives.presets import objective_preset, preset_names
from coevop.objectives.program import parse_objective_program, parse_objective_program_payload
from coevop.objectives.spec import ObjectiveSpec

__all__ = [
    "ObjectiveSpec",
    "objective_preset",
    "preset_names",
    "parse_objective_program",
    "parse_objective_program_payload",
]
