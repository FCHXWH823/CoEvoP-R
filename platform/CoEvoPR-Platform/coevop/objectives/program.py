"""Restricted Python-like objective programs.

This module implements the Eureka-style interface without executing model code.
The LLM may write a small ``objective(features)`` function, but the platform only
parses its Python AST and lowers allowed expressions into the existing JSON DSL.
"""

from __future__ import annotations

import ast as py_ast
import copy
import json
import math
import re
from dataclasses import replace
from typing import Any

from coevop.objectives.spec import (
    ALLOWED_OPERATORS,
    BINARY_OPERATORS,
    MAX_CONSTANT,
    MAX_AST_DEPTH,
    MAX_PRIMITIVE_COUNT,
    MIN_CONSTANT,
    UNARY_OPERATORS,
    VARIADIC_OPERATORS,
    ObjectiveSpec,
    ObjectiveSpecError,
    parse_objective_spec,
)
from coevop.objectives.terms import term_names


ALLOWED_PROGRAM_FUNCTIONS = sorted(ALLOWED_OPERATORS | {"term"})
OBJECTIVE_FUNCTION_NAMES = {"objective", "compute_objective"}
MAX_COMPONENT_COUNT = 6


PROGRAM_SYSTEM_CONTRACT = """Write exactly one restricted Python function:

def objective(features):
    name = term("allowed_term")
    score = ...
    return score, {"component_name": name}

Allowed expression syntax:
- numeric constants
- local variables assigned earlier in the function
- term("name") using an allowed term
- +, -, *, / where / is lowered to safe_div
- log1p(x), sqrt(x), square(x), softplus(x), sigmoid(x), safe_div(x, y)

Size limits:
- keep the final score expression at or below 12 AST primitives total
- keep every returned component expression at or below 12 AST primitives
- keep AST depth at or below 5
- return at most 6 named diagnostic components
- prefer compact formulas over large expressions; nonlinear combinations are allowed
  when they have a clear placement meaning

Forbidden:
- imports, loops, branches, comprehensions, lambdas, attributes, subscripts,
  arbitrary calls, mutation, random numbers, data-dependent control flow, and
  direct access to the raw features object.
"""


def objective_program_schema_for_provider(term_scope: str = "tier1") -> dict[str, Any]:
    """JSON schema for provider output containing restricted objective code."""

    allowed_terms = sorted(term_names(term_scope))
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "id",
            "rationale",
            "parent_ids",
            "declared_term_usage",
            "change_description",
            "objective_program",
        ],
        "properties": {
            "id": {
                "type": "string",
                "description": (
                    "Optional model-proposed identifier. "
                    "The platform recomputes the final id."
                ),
            },
            "rationale": {"type": "string"},
            "parent_ids": {"type": "array", "items": {"type": "string"}},
            "declared_term_usage": {
                "type": "array",
                "items": {"type": "string", "enum": allowed_terms},
            },
            "change_description": {
                "type": "string",
                "description": (
                    "OpenEvolve-style SEARCH/REPLACE summary of the edit. "
                    "The platform records this for memory, but validates the complete "
                    "objective_program field."
                ),
            },
            "objective_program": {
                "type": "string",
                "description": (
                    "Restricted Python code defining objective(features). "
                    "The code is parsed but never executed. Keep the final score "
                    f"expression at or below {MAX_PRIMITIVE_COUNT} AST primitives "
                    f"and depth {MAX_AST_DEPTH}; prefer compact formulas with a "
                    "clear placement meaning."
                ),
            },
        },
    }


def parse_objective_payload(
    payload: dict[str, Any],
    *,
    created_by: str,
    term_scope: str = "tier1",
) -> ObjectiveSpec:
    """Parse either a program-style payload or the legacy raw AST payload."""

    if "objective_program" in payload:
        return parse_objective_program_payload(
            payload,
            created_by=created_by,
            term_scope=term_scope,
        )
    return parse_objective_spec(payload, created_by=created_by, term_scope=term_scope)


def parse_objective_program_payload(
    payload: dict[str, Any],
    *,
    created_by: str,
    term_scope: str = "tier1",
) -> ObjectiveSpec:
    if not isinstance(payload, dict):
        raise ObjectiveSpecError("objective program payload must be a JSON object")
    source = payload.get("objective_program")
    if not isinstance(source, str) or not source.strip():
        raise ObjectiveSpecError("objective_program must be a non-empty string")
    source = _normalize_provider_program_source(source)

    spec = parse_objective_program(
        source,
        created_by=created_by,
        rationale=str(payload.get("rationale", "")).strip(),
        parent_ids=[str(parent) for parent in payload.get("parent_ids", [])],
        term_scope=term_scope,
    )
    declared_terms = sorted(str(term) for term in payload.get("declared_term_usage", []))
    if declared_terms and declared_terms != spec.term_set:
        raise ObjectiveSpecError(
            f"declared_term_usage {declared_terms} does not match program terms {spec.term_set}"
        )
    return spec


def _normalize_provider_program_source(source: str) -> str:
    """Clean common provider formatting artifacts without changing semantics."""

    cleaned = (
        source.replace("\u00a0", " ")
        .replace("\\(n)", "\n")
        .replace("\\r\\n", "\n")
        .replace("\\r", "\n")
        .replace("\\n", "\n")
        .replace("\\$", "")
        .replace("\\[", "")
        .replace("\\]", "")
    )
    cleaned = re.sub(r"(?m)^\\+\$?", "", cleaned)
    return re.sub(r"\\+\s*$", "", cleaned, flags=re.MULTILINE)


def parse_objective_program(
    source: str,
    *,
    created_by: str,
    rationale: str = "",
    parent_ids: list[str] | None = None,
    term_scope: str = "tier1",
) -> ObjectiveSpec:
    """Lower a restricted Python objective function into an ObjectiveSpec."""

    try:
        module = py_ast.parse(source)
    except SyntaxError as exc:
        raise ObjectiveSpecError(f"invalid objective program syntax: {exc}") from exc

    function = _single_objective_function(module)
    lowered = _ProgramLowerer(term_scope=term_scope).lower(function)
    payload = {
        "id": "",
        "rationale": rationale,
        "parent_ids": parent_ids or [],
        "declared_term_usage": lowered.term_set,
        "ast": lowered.score_ast,
    }
    spec = parse_objective_spec(payload, created_by=created_by, term_scope=term_scope)
    return replace(
        spec,
        source_program=source.strip() + "\n",
        components=lowered.components or None,
    )


def program_example(term_scope: str = "tier1") -> str:
    if term_scope in {"dreamplace", "dreamplace_replacement"}:
        return "\n".join(
            [
                "def objective(features):",
                '    wl = term("wirelength_lse")',
                '    den = term("density_bell")',
                '    route = term("route_pressure_long")',
                '    pins = term("pin_density_pnorm")',
                "    access_pressure = log1p(route * pins)",
                "    score = wl + den + 0.01 * access_pressure",
                (
                    '    return score, {"wirelength": wl, "density": den, '
                    '"route_hotspot": route, "pin_access": pins}'
                ),
            ]
        )
    if term_scope == "deployable":
        return "\n".join(
            [
                "def objective(features):",
                '    route = term("soft_rudy_pnorm")',
                '    pins = term("pin_density_pnorm")',
                '    mean_route = term("soft_rudy_mean")',
                "    score = route + 0.35 * pins + 0.2 * mean_route",
                (
                    '    return score, {"route_hotspot": route, '
                    '"pin_access": pins, "route_mean": mean_route}'
                ),
            ]
        )
    return "\n".join(
        [
            "def objective(features):",
            '    rudy = term("rudy_p95")',
            '    density = term("cell_density_p95")',
            "    score = rudy + 0.35 * density",
            '    return score, {"rudy_hotspot": rudy, "density_hotspot": density}',
        ]
    )


class _LoweredProgram:
    def __init__(self, score_ast: dict[str, Any], components: dict[str, dict[str, Any]]) -> None:
        self.score_ast = score_ast
        self.components = components
        self.term_set = sorted(_collect_terms(score_ast, components))


class _ProgramLowerer:
    def __init__(self, *, term_scope: str) -> None:
        self.term_scope = term_scope
        self.allowed_terms = set(term_names(term_scope))
        self.env: dict[str, dict[str, Any]] = {}

    def lower(self, function: py_ast.FunctionDef) -> _LoweredProgram:
        self._validate_signature(function)
        statements = _strip_docstring(function.body)
        if not statements:
            raise ObjectiveSpecError("objective function body is empty")
        if not isinstance(statements[-1], py_ast.Return):
            raise ObjectiveSpecError("objective function must end with a return statement")

        for statement in statements[:-1]:
            if isinstance(statement, py_ast.Assign):
                self._lower_assignment(statement)
                continue
            raise ObjectiveSpecError(
                f"unsupported statement in objective program: {statement.__class__.__name__}"
            )

        score_ast, components = self._lower_return(statements[-1].value)
        return _LoweredProgram(score_ast=score_ast, components=components)

    def _validate_signature(self, function: py_ast.FunctionDef) -> None:
        if function.name not in OBJECTIVE_FUNCTION_NAMES:
            raise ObjectiveSpecError("objective program must define objective(features)")
        args = function.args
        if (
            len(args.args) != 1
            or args.vararg is not None
            or args.kwarg is not None
            or args.kwonlyargs
            or args.defaults
            or args.kw_defaults
        ):
            raise ObjectiveSpecError("objective function must have exactly one argument: features")
        if args.args[0].arg != "features":
            raise ObjectiveSpecError("objective function argument must be named features")

    def _lower_assignment(self, statement: py_ast.Assign) -> None:
        if len(statement.targets) != 1 or not isinstance(statement.targets[0], py_ast.Name):
            raise ObjectiveSpecError("assignments must target one local variable")
        name = statement.targets[0].id
        if name == "features" or name in ALLOWED_PROGRAM_FUNCTIONS:
            raise ObjectiveSpecError(f"invalid assignment target: {name}")
        self.env[name] = self._lower_expr(statement.value)

    def _lower_return(
        self,
        value: py_ast.expr | None,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        if value is None:
            raise ObjectiveSpecError("objective function must return a score expression")
        if isinstance(value, (py_ast.Tuple, py_ast.List)):
            if len(value.elts) != 2:
                raise ObjectiveSpecError("tuple return must be (score, components_dict)")
            score = self._lower_expr(value.elts[0])
            components = self._lower_components(value.elts[1])
            return score, components
        return self._lower_expr(value), {}

    def _lower_components(self, value: py_ast.expr) -> dict[str, dict[str, Any]]:
        if not isinstance(value, py_ast.Dict):
            raise ObjectiveSpecError("objective components must be returned as a dictionary")
        if len(value.keys) > MAX_COMPONENT_COUNT:
            raise ObjectiveSpecError(
                f"objective components exceed limit of {MAX_COMPONENT_COUNT}"
            )
        components: dict[str, dict[str, Any]] = {}
        for key, component_expr in zip(value.keys, value.values):
            if not isinstance(key, py_ast.Constant) or not isinstance(key.value, str):
                raise ObjectiveSpecError("component names must be string constants")
            component_name = str(key.value).strip()
            if not component_name:
                raise ObjectiveSpecError("component names must be non-empty")
            components[component_name] = self._lower_expr(component_expr)
        return components

    def _lower_expr(self, expr: py_ast.expr) -> dict[str, Any]:
        if isinstance(expr, py_ast.Name):
            if expr.id == "features":
                raise ObjectiveSpecError("direct features access is forbidden; use term(\"name\")")
            if expr.id not in self.env:
                raise ObjectiveSpecError(f"unknown local variable: {expr.id}")
            return copy.deepcopy(self.env[expr.id])

        if isinstance(expr, py_ast.Constant):
            if isinstance(expr.value, bool) or not isinstance(expr.value, (int, float)):
                raise ObjectiveSpecError("only numeric constants are allowed in expressions")
            return {"op": "const", "value": float(expr.value)}

        if isinstance(expr, py_ast.UnaryOp):
            if not isinstance(expr.op, py_ast.USub):
                raise ObjectiveSpecError("only unary minus is allowed")
            child = self._lower_expr(expr.operand)
            if child["op"] == "const":
                return {"op": "const", "value": -float(child["value"])}
            return {"op": "mul", "args": [{"op": "const", "value": -1.0}, child]}

        if isinstance(expr, py_ast.BinOp):
            return self._lower_binop(expr)

        if isinstance(expr, py_ast.Call):
            return self._lower_call(expr)

        raise ObjectiveSpecError(f"unsupported expression: {expr.__class__.__name__}")

    def _lower_binop(self, expr: py_ast.BinOp) -> dict[str, Any]:
        left = self._lower_expr(expr.left)
        right = self._lower_expr(expr.right)
        if isinstance(expr.op, py_ast.Add):
            return _flatten_variadic("add", left, right)
        if isinstance(expr.op, py_ast.Sub):
            return {"op": "sub", "args": [left, right]}
        if isinstance(expr.op, py_ast.Mult):
            return _flatten_variadic("mul", left, right)
        if isinstance(expr.op, py_ast.Div):
            return {"op": "safe_div", "args": [left, right]}
        raise ObjectiveSpecError("only +, -, *, and / are allowed")

    def _lower_call(self, expr: py_ast.Call) -> dict[str, Any]:
        if expr.keywords:
            raise ObjectiveSpecError("keyword arguments are not allowed")
        if not isinstance(expr.func, py_ast.Name):
            raise ObjectiveSpecError("only direct calls to allowed objective functions are allowed")
        function_name = expr.func.id

        if function_name == "term":
            if len(expr.args) != 1:
                raise ObjectiveSpecError("term() requires exactly one argument")
            arg = expr.args[0]
            if not isinstance(arg, py_ast.Constant) or not isinstance(arg.value, str):
                raise ObjectiveSpecError("term() argument must be a string constant")
            term_name = str(arg.value)
            if term_name not in self.allowed_terms:
                raise ObjectiveSpecError(
                    f"term {term_name} is not allowed in {self.term_scope} scope"
                )
            return {"op": "term", "name": term_name}

        if function_name not in ALLOWED_OPERATORS:
            raise ObjectiveSpecError(f"unsupported objective helper: {function_name}")
        args = [self._lower_expr(arg) for arg in expr.args]
        if function_name in UNARY_OPERATORS and len(args) != 1:
            raise ObjectiveSpecError(f"{function_name}() requires exactly one argument")
        if function_name in BINARY_OPERATORS and len(args) != 2:
            raise ObjectiveSpecError(f"{function_name}() requires exactly two arguments")
        if function_name in VARIADIC_OPERATORS and not (2 <= len(args) <= 4):
            raise ObjectiveSpecError(f"{function_name}() requires two to four arguments")
        return {"op": function_name, "args": args}


def _single_objective_function(module: py_ast.Module) -> py_ast.FunctionDef:
    statements = _strip_module_docstring(module.body)
    if len(statements) != 1 or not isinstance(statements[0], py_ast.FunctionDef):
        raise ObjectiveSpecError("objective program must contain exactly one function definition")
    function = statements[0]
    if function.decorator_list:
        raise ObjectiveSpecError("decorators are not allowed")
    if function.returns is not None:
        raise ObjectiveSpecError("return annotations are not allowed")
    return function


def _strip_module_docstring(statements: list[py_ast.stmt]) -> list[py_ast.stmt]:
    return _strip_docstring(statements)


def _strip_docstring(statements: list[py_ast.stmt]) -> list[py_ast.stmt]:
    if (
        statements
        and isinstance(statements[0], py_ast.Expr)
        and isinstance(statements[0].value, py_ast.Constant)
        and isinstance(statements[0].value.value, str)
    ):
        return statements[1:]
    return statements


def _flatten_variadic(op: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    args: list[dict[str, Any]] = []
    for child in (left, right):
        if child.get("op") == op and len(child.get("args", [])) < 4:
            args.extend(copy.deepcopy(child["args"]))
        else:
            args.append(child)
    if len(args) > 4:
        return {"op": op, "args": args[:3] + [{"op": op, "args": args[3:]}]}
    return {"op": op, "args": args}


def _collect_terms(score_ast: dict[str, Any], components: dict[str, dict[str, Any]]) -> set[str]:
    terms: set[str] = set()
    score_terms: set[str] = set()

    def visit(node: dict[str, Any], collected_terms: set[str], depth: int = 1) -> None:
        if depth > MAX_AST_DEPTH:
            raise ObjectiveSpecError(f"AST depth exceeds {MAX_AST_DEPTH}")
        op = node.get("op")
        if op == "term":
            collected_terms.add(str(node["name"]))
            return
        if op == "const":
            value = float(node.get("value", math.nan))
            magnitude = abs(value)
            if not math.isfinite(value) or (
                magnitude != 0 and not (MIN_CONSTANT <= magnitude <= MAX_CONSTANT)
            ):
                raise ObjectiveSpecError(
                    f"component constant magnitude must be 0 or within "
                    f"[{MIN_CONSTANT}, {MAX_CONSTANT}]"
                )
            return
        for child in node.get("args", []):
            visit(child, collected_terms, depth + 1)

    visit(score_ast, score_terms)
    terms.update(score_terms)
    for component in components.values():
        component_terms: set[str] = set()
        visit(component, component_terms)
        component_primitives = _count_primitives(component)
        if component_primitives > MAX_PRIMITIVE_COUNT:
            raise ObjectiveSpecError(
                f"component AST primitive count exceeds {MAX_PRIMITIVE_COUNT}"
            )
        extra_terms = component_terms.difference(score_terms)
        if extra_terms:
            raise ObjectiveSpecError(
                f"component terms must appear in score AST: {sorted(extra_terms)}"
            )
        terms.update(component_terms)
    primitive_count = _count_primitives(score_ast)
    if primitive_count > MAX_PRIMITIVE_COUNT:
        raise ObjectiveSpecError(f"AST primitive count exceeds {MAX_PRIMITIVE_COUNT}")
    return terms


def _count_primitives(node: dict[str, Any]) -> int:
    return 1 + sum(_count_primitives(child) for child in node.get("args", []))


def objective_program_to_json(source: str, term_scope: str = "tier1") -> str:
    """Return a compact JSON string useful for prompt examples and tests."""

    spec = parse_objective_program(source, created_by="example", term_scope=term_scope)
    return json.dumps(
        {"ast": spec.ast, "components": spec.components, "term_set": spec.term_set},
        sort_keys=True,
    )
