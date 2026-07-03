"""Train-split coefficient fitting for symbolic objectives."""

from __future__ import annotations

import copy
import math
from dataclasses import replace
from typing import Any

import numpy as np

from coevop.eval.tier1 import (
    DEFAULT_TARGETS,
    TARGET_BADNESS_DIRECTIONS,
    evaluate_spec,
)
from coevop.objectives.baselines import _zscore
from coevop.objectives.dsl import score_ast
from coevop.objectives.spec import (
    MAX_CONSTANT,
    MIN_CONSTANT,
    ObjectiveSpec,
    parse_objective_spec,
)

SlotPath = tuple[int, ...]


def fit_objective_constants(
    spec: ObjectiveSpec,
    *,
    train_rows: list[dict[str, object]],
    validation_rows: list[dict[str, object]],
    term_scope: str = "tier1",
) -> tuple[ObjectiveSpec | None, dict[str, Any]]:
    """Fit linear weight constants in ``mul(const, expr)`` nodes on train rows.

    The fitted objective is returned as a sibling of ``spec``. The original
    candidate is never modified. ``None`` means either no eligible constants were
    present or the fitted constants did not change the canonical objective.
    """

    slots = _linear_weight_slots(spec.ast)
    before_weights = [_const_at_path(spec.ast, path) for path in slots]
    report: dict[str, Any] = {
        "parent_objective_id": spec.id,
        "eligible_constant_count": len(slots),
        "constants_changed": [],
        "train_score_before": _score_or_nan(train_rows, spec),
        "validation_score_before": _score_or_nan(validation_rows, spec),
        "train_score_after": math.nan,
        "validation_score_after": math.nan,
        "status": "no_eligible_constants",
    }
    if not slots:
        return None, report

    target = _target_badness_vector(train_rows)
    if not np.any(np.isfinite(target)):
        report["status"] = "no_finite_train_target"
        return None, report

    base_ast = _replace_slot_values(spec.ast, slots, [0.0] * len(slots))
    base = score_ast(train_rows, base_ast)
    columns = []
    for index in range(len(slots)):
        values = [0.0] * len(slots)
        values[index] = 1.0
        basis_ast = _replace_slot_values(spec.ast, slots, values)
        columns.append(score_ast(train_rows, basis_ast) - base)
    design = np.column_stack(columns)
    finite = np.isfinite(target) & np.isfinite(base) & np.all(np.isfinite(design), axis=1)
    if int(np.sum(finite)) < max(2, len(slots)):
        report["status"] = "insufficient_finite_train_rows"
        return None, report

    try:
        fitted, *_ = np.linalg.lstsq(design[finite], target[finite] - base[finite], rcond=None)
    except np.linalg.LinAlgError:
        report["status"] = "lstsq_failed"
        return None, report

    fitted_weights = [_clip_constant(float(value)) for value in fitted]
    changed = [
        {
            "path": ".".join(str(part) for part in path),
            "before": before,
            "after": after,
        }
        for path, before, after in zip(slots, before_weights, fitted_weights)
        if not math.isclose(before, after, rel_tol=1e-6, abs_tol=1e-9)
    ]
    report["constants_changed"] = changed
    if not changed:
        report["status"] = "unchanged"
        return None, report

    fitted_ast = _replace_slot_values(spec.ast, slots, fitted_weights)
    payload = {
        "id": "",
        "rationale": (spec.rationale + " Constant weights fitted on the train split.").strip(),
        "parent_ids": [spec.id],
        "declared_term_usage": spec.term_set,
        "ast": fitted_ast,
    }
    fitted_spec = parse_objective_spec(payload, created_by=spec.created_by, term_scope=term_scope)
    fitted_spec = replace(
        fitted_spec,
        components=spec.components,
        source_program=None,
    )
    if fitted_spec.id == spec.id:
        report["status"] = "canonical_id_unchanged"
        return None, report

    report["train_score_after"] = _score_or_nan(train_rows, fitted_spec)
    report["validation_score_after"] = _score_or_nan(validation_rows, fitted_spec)
    report["status"] = "fitted"
    return fitted_spec, report


def _score_or_nan(rows: list[dict[str, object]], spec: ObjectiveSpec) -> float:
    if not rows:
        return math.nan
    return evaluate_spec(rows, spec).score


def _linear_weight_slots(ast: dict[str, Any]) -> list[SlotPath]:
    slots: list[SlotPath] = []

    def visit(node: dict[str, Any], path: SlotPath, linear_context: bool) -> None:
        op = node.get("op")
        if op in {"term", "const"}:
            return
        args = node.get("args", [])
        if op == "mul" and linear_context:
            const_indices = [
                index
                for index, child in enumerate(args)
                if isinstance(child, dict) and child.get("op") == "const"
            ]
            if len(const_indices) == 1 and len(args) >= 2:
                slots.append(path + (const_indices[0],))
        child_linear = linear_context and op in {"add", "sub"}
        for index, child in enumerate(args):
            visit(child, path + (index,), child_linear)

    visit(ast, (), True)
    return slots


def _const_at_path(ast: dict[str, Any], path: SlotPath) -> float:
    node = _node_at_path(ast, path)
    return float(node["value"])


def _replace_slot_values(
    ast: dict[str, Any],
    slots: list[SlotPath],
    values: list[float],
) -> dict[str, Any]:
    copied = copy.deepcopy(ast)
    for path, value in zip(slots, values):
        _node_at_path(copied, path)["value"] = float(value)
    return copied


def _node_at_path(ast: dict[str, Any], path: SlotPath) -> dict[str, Any]:
    node = ast
    for index in path:
        node = node["args"][index]
    return node


def _target_badness_vector(rows: list[dict[str, object]]) -> np.ndarray:
    columns = []
    for target in DEFAULT_TARGETS:
        values = []
        for row in rows:
            try:
                values.append(float(row.get(target, math.nan)))
            except (TypeError, ValueError):
                values.append(math.nan)
        direction = float(TARGET_BADNESS_DIRECTIONS.get(target, 1))
        columns.append(direction * _zscore(np.asarray(values, dtype=np.float64)))
    if not columns:
        return np.full(len(rows), math.nan, dtype=np.float64)
    stacked = np.column_stack(columns)
    finite = np.isfinite(stacked)
    sums = np.where(finite, stacked, 0.0).sum(axis=1)
    counts = finite.sum(axis=1)
    out = np.full(len(rows), math.nan, dtype=np.float64)
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]
    return out


def _clip_constant(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    magnitude = abs(value)
    if magnitude == 0.0:
        return 0.0
    if magnitude < MIN_CONSTANT:
        return 0.0
    if magnitude > MAX_CONSTANT:
        return math.copysign(MAX_CONSTANT, value)
    return value
