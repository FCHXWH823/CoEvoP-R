"""Objective definitions and scoring helpers."""

from coevop.objectives.baselines import BASELINE_OBJECTIVES, ObjectiveDefinition
from coevop.objectives.presets import objective_preset, preset_names
from coevop.objectives.program import parse_objective_program, parse_objective_program_payload
from coevop.objectives.spec import ObjectiveSpec

__all__ = [
    "BASELINE_OBJECTIVES",
    "ObjectiveDefinition",
    "ObjectiveSpec",
    "objective_preset",
    "preset_names",
    "parse_objective_program",
    "parse_objective_program_payload",
]
