"""Vectorized offline evaluation for safe objective ASTs."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from coevop.objectives.baselines import _column, _zscore
from coevop.objectives.spec import ObjectiveSpec
from coevop.objectives.terms import tier1_column_for


def score_spec(rows: list[dict[str, object]], spec: ObjectiveSpec) -> np.ndarray:
    values = score_ast(rows, spec.ast)
    return values


def score_ast(rows: list[dict[str, object]], ast: dict[str, Any]) -> np.ndarray:
    values = _eval_node(ast, rows)
    values = np.asarray(values, dtype=np.float64)
    values[~np.isfinite(values)] = math.nan
    return values


def score_components(rows: list[dict[str, object]], spec: ObjectiveSpec) -> dict[str, np.ndarray]:
    components = spec.components or {
        term_name: {"op": "term", "name": term_name} for term_name in spec.term_set
    }
    return {name: score_ast(rows, ast) for name, ast in components.items()}


def _eval_node(node: dict[str, Any], rows: list[dict[str, object]]) -> np.ndarray:
    op = node["op"]
    if op == "term":
        column = tier1_column_for(node["name"])
        return _zscore(_column(rows, column))
    if op == "const":
        return np.full(len(rows), float(node["value"]), dtype=np.float64)

    args = [_eval_node(child, rows) for child in node["args"]]
    if op == "add":
        return np.sum(args, axis=0)
    if op == "sub":
        return args[0] - args[1]
    if op == "mul":
        out = np.ones(len(rows), dtype=np.float64)
        for arg in args:
            out *= arg
        return out
    if op == "safe_div":
        denom = np.where(args[1] < 0, np.minimum(args[1], -1e-6), np.maximum(args[1], 1e-6))
        return args[0] / denom
    if op == "log1p":
        return np.log1p(np.maximum(args[0], -0.999999))
    if op == "sqrt":
        return np.sqrt(np.maximum(args[0], 0.0))
    if op == "square":
        return np.square(args[0])
    if op == "softplus":
        return np.logaddexp(0.0, args[0])
    if op == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(args[0], -60.0, 60.0)))
    raise ValueError(f"unsupported objective op: {op}")
