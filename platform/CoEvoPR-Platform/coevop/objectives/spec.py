"""Safe symbolic objective specifications for LLM-generated candidates."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from coevop.objectives.terms import term_names, tier1_columns


ALLOWED_TERMS = tier1_columns()

ALLOWED_OPERATORS = {
    "add",
    "sub",
    "mul",
    "safe_div",
    "log1p",
    "sqrt",
    "square",
    "softplus",
    "sigmoid",
}

UNARY_OPERATORS = {"log1p", "sqrt", "square", "softplus", "sigmoid"}
BINARY_OPERATORS = {"sub", "safe_div"}
VARIADIC_OPERATORS = {"add", "mul"}
MAX_AST_DEPTH = 5
MAX_PRIMITIVE_COUNT = 12
MIN_CONSTANT = 1e-15
MAX_CONSTANT = 1e6


class ObjectiveSpecError(ValueError):
    """Raised when an LLM candidate violates the objective DSL."""


@dataclass(frozen=True)
class ObjectiveSpec:
    id: str
    ast: dict[str, Any]
    constants: dict[str, float]
    term_set: list[str]
    complexity: int
    created_by: str
    parent_ids: list[str]
    rationale: str
    source_program: str | None = None
    components: dict[str, dict[str, Any]] | None = None


OBJECTIVE_AST_NODE_SCHEMA: dict[str, Any] = {
    "anyOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["op", "name"],
            "properties": {
                "op": {"type": "string", "enum": ["term"]},
                "name": {"type": "string", "enum": sorted(term_names("all"))},
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["op", "value"],
            "properties": {
                "op": {"type": "string", "enum": ["const"]},
                "value": {"type": "number"},
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["op", "args"],
            "properties": {
                "op": {"type": "string", "enum": sorted(UNARY_OPERATORS)},
                "args": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": {"$ref": "#/$defs/node"},
                },
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["op", "args"],
            "properties": {
                "op": {"type": "string", "enum": sorted(BINARY_OPERATORS)},
                "args": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"$ref": "#/$defs/node"},
                },
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["op", "args"],
            "properties": {
                "op": {"type": "string", "enum": sorted(VARIADIC_OPERATORS)},
                "args": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 4,
                    "items": {"$ref": "#/$defs/node"},
                },
            },
        },
    ]
}

OBJECTIVE_SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "rationale", "parent_ids", "declared_term_usage", "ast"],
    "properties": {
        "id": {
            "type": "string",
            "description": (
                "Optional model-proposed identifier. "
                "The platform recomputes the final id."
            ),
        },
        "rationale": {
            "type": "string",
            "description": "Short design rationale for the candidate objective.",
        },
        "parent_ids": {"type": "array", "items": {"type": "string"}},
        "declared_term_usage": {
            "type": "array",
            "items": {"type": "string", "enum": sorted(term_names("all"))},
        },
        "ast": {"$ref": "#/$defs/node"},
    },
    "$defs": {"node": OBJECTIVE_AST_NODE_SCHEMA},
}


def objective_schema_for_provider(term_scope: str = "tier1") -> dict[str, Any]:
    schema = json.loads(json.dumps(OBJECTIVE_SPEC_SCHEMA))
    allowed_terms = sorted(term_names(term_scope))
    schema["properties"]["declared_term_usage"]["items"]["enum"] = allowed_terms
    node_defs = schema["$defs"]["node"]["anyOf"]
    node_defs[0]["properties"]["name"]["enum"] = allowed_terms
    return schema


def parse_objective_spec(
    payload: dict[str, Any],
    created_by: str,
    term_scope: str = "tier1",
) -> ObjectiveSpec:
    if not isinstance(payload, dict):
        raise ObjectiveSpecError("objective payload must be a JSON object")
    if "ast" not in payload:
        raise ObjectiveSpecError("objective payload is missing ast")

    rationale = str(payload.get("rationale", "")).strip()
    parent_ids = [str(parent) for parent in payload.get("parent_ids", [])]
    declared_terms = sorted(str(term) for term in payload.get("declared_term_usage", []))

    terms: set[str] = set()
    constants: list[float] = []
    allowed_terms = set(term_names(term_scope))
    complexity = _validate_node(
        payload["ast"],
        depth=1,
        terms=terms,
        constants=constants,
        allowed_terms=allowed_terms,
    )
    term_set = sorted(terms)

    if declared_terms and declared_terms != term_set:
        raise ObjectiveSpecError(
            f"declared_term_usage {declared_terms} does not match AST terms {term_set}"
        )

    constants_dict = {f"c{i}": value for i, value in enumerate(constants)}
    ast = _canonicalize_node(payload["ast"])
    objective_id = _objective_id(ast)
    return ObjectiveSpec(
        id=objective_id,
        ast=ast,
        constants=constants_dict,
        term_set=term_set,
        complexity=complexity,
        created_by=created_by,
        parent_ids=parent_ids,
        rationale=rationale,
        source_program=None,
        components=None,
    )


def objective_spec_to_dict(spec: ObjectiveSpec) -> dict[str, Any]:
    return asdict(spec)


def write_objective_spec(spec: ObjectiveSpec, output: str | Path) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(objective_spec_to_dict(spec), f, indent=2, sort_keys=True)
        f.write("\n")
    return output_path


def load_objective_spec(path: str | Path) -> ObjectiveSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ObjectiveSpec(
        id=str(payload["id"]),
        ast=dict(payload["ast"]),
        constants={str(k): float(v) for k, v in payload.get("constants", {}).items()},
        term_set=[str(term) for term in payload.get("term_set", [])],
        complexity=int(payload["complexity"]),
        created_by=str(payload["created_by"]),
        parent_ids=[str(parent) for parent in payload.get("parent_ids", [])],
        rationale=str(payload.get("rationale", "")),
        source_program=payload.get("source_program"),
        components=payload.get("components"),
    )


def _validate_node(
    node: Any,
    depth: int,
    terms: set[str],
    constants: list[float],
    allowed_terms: set[str],
) -> int:
    if depth > MAX_AST_DEPTH:
        raise ObjectiveSpecError(f"AST depth exceeds {MAX_AST_DEPTH}")
    if not isinstance(node, dict):
        raise ObjectiveSpecError("AST node must be an object")

    op = node.get("op")
    if op == "term":
        name = node.get("name")
        if name not in allowed_terms:
            raise ObjectiveSpecError(f"unknown objective term: {name}")
        terms.add(str(name))
        return 1
    if op == "const":
        value = _validate_constant(node.get("value"))
        constants.append(value)
        return 1
    if op not in ALLOWED_OPERATORS:
        raise ObjectiveSpecError(f"unknown objective operator: {op}")

    args = node.get("args")
    if not isinstance(args, list):
        raise ObjectiveSpecError(f"operator {op} requires args")
    if op in UNARY_OPERATORS and len(args) != 1:
        raise ObjectiveSpecError(f"operator {op} requires exactly one arg")
    if op in BINARY_OPERATORS and len(args) != 2:
        raise ObjectiveSpecError(f"operator {op} requires exactly two args")
    if op in VARIADIC_OPERATORS and not (2 <= len(args) <= 4):
        raise ObjectiveSpecError(f"operator {op} requires two to four args")

    complexity = 1
    for child in args:
        complexity += _validate_node(child, depth + 1, terms, constants, allowed_terms)
    if complexity > MAX_PRIMITIVE_COUNT:
        raise ObjectiveSpecError(f"AST primitive count exceeds {MAX_PRIMITIVE_COUNT}")
    return complexity


def _validate_constant(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ObjectiveSpecError(f"invalid constant: {value}") from exc
    if not math.isfinite(numeric):
        raise ObjectiveSpecError(f"non-finite constant: {value}")
    magnitude = abs(numeric)
    if magnitude != 0 and not (MIN_CONSTANT <= magnitude <= MAX_CONSTANT):
        raise ObjectiveSpecError(
            f"constant magnitude must be 0 or within [{MIN_CONSTANT}, {MAX_CONSTANT}]"
        )
    return numeric


def _canonicalize_node(node: dict[str, Any]) -> dict[str, Any]:
    op = str(node["op"])
    if op == "term":
        return {"op": "term", "name": str(node["name"])}
    if op == "const":
        return {"op": "const", "value": float(node["value"])}
    return {"op": op, "args": [_canonicalize_node(child) for child in node["args"]]}


def _objective_id(ast: dict[str, Any]) -> str:
    encoded = json.dumps(ast, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return f"obj_{digest}"
