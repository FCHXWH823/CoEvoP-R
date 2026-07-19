"""Safe symbolic objective specifications for LLM-generated candidates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from coevop.objectives.terms import (
    is_component_calibration_observable,
    observable_names,
    term_names,
    tier1_columns,
)


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
MAX_AST_DEPTH = 10
MAX_PRIMITIVE_COUNT = 12
MAX_COMPONENT_COUNT = 6
MIN_CONSTANT = 1e-15
MAX_CONSTANT = 1e6

# Stateful controller extension (DSL v2). See docs/stateful_objective_dsl.md.
MAX_STATE_REGISTERS = 4
MAX_REGISTER_NAME_LENGTH = 24
TYPED_POLICY_INTERFACE = "typed_policy_v1"
TYPED_POLICY_REQUIRED_SLOTS = {"density_weight", "gamma_scale"}
TYPED_POLICY_OPTIONAL_SLOTS = {"route_weight", "pin_weight"}
TYPED_POLICY_SLOTS = TYPED_POLICY_REQUIRED_SLOTS | TYPED_POLICY_OPTIONAL_SLOTS
RESERVED_STATE_REGISTERS = TYPED_POLICY_SLOTS
_REGISTER_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# Optional per-net timing-aware weight-update policy
#   w_{n,t+1} = clip(w_{n,t} * f(net, state, obs), DREAMPlace net-weight limits)
# where f is a restricted elementwise expression over per-net observables.
# The interface is validated and carried by the spec now; runtime application
# activates only when the DREAMPlace run enables the timing controller.
NET_OBSERVABLES = {"criticality", "span", "fanout"}
NET_WEIGHT_POLICY_DEFAULTS = {
    # One policy update per DREAMPlace timing update. The physical cadence is
    # inherited from the timing-driven placement configuration.
    "cadence": 1,
}


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
    # DSL v2: {"registers": {name: {"init": ast, "update": ast}}} or None.
    state: dict[str, Any] | None = None
    observable_set: list[str] = field(default_factory=list)
    # Timing-aware interface: {"update": ast, "cadence": int}. Legacy
    # explicit clip fields remain readable, but official policies inherit the
    # net-weight limit from the DREAMPlace timing configuration.
    net_weight_policy: dict[str, Any] | None = None

    @property
    def is_stateful(self) -> bool:
        return bool(self.state and self.state.get("registers"))

    @property
    def register_names(self) -> list[str]:
        if not self.state:
            return []
        return sorted(self.state.get("registers", {}))

    @property
    def policy_interface(self) -> str | None:
        if not self.state:
            return None
        value = self.state.get("interface")
        return str(value) if value else None

    @property
    def is_typed_policy(self) -> bool:
        return self.policy_interface == TYPED_POLICY_INTERFACE


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
            "required": ["op", "name"],
            "properties": {
                "op": {"type": "string", "enum": ["state"]},
                "name": {"type": "string"},
            },
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["op", "name"],
            "properties": {
                "op": {"type": "string", "enum": ["obs"]},
                "name": {
                    "type": "string",
                    "enum": sorted(observable_names("dreamplace_controller")),
                },
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
        "state": {
            "type": "object",
            "additionalProperties": False,
            "required": ["registers"],
            "properties": {
                "interface": {
                    "type": "string",
                    "enum": [TYPED_POLICY_INTERFACE],
                },
                "control": {
                    "type": "string",
                    "enum": ["generated", "native_identity"],
                },
                "registers": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["init"],
                        "properties": {
                            "init": {"$ref": "#/$defs/node"},
                            "update": {"$ref": "#/$defs/node"},
                        },
                    },
                },
            },
        },
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

    state_payload = payload.get("state")
    (
        registers,
        state_complexity,
        observables,
        state_interface,
        state_control,
    ) = _validate_state_block(state_payload)
    net_weight_policy, policy_complexity, policy_obs = _validate_net_weight_policy(
        payload.get("net_weight_policy"), registers
    )
    observables = observables | policy_obs

    terms: set[str] = set()
    constants: list[float] = []
    allowed_terms = set(term_names(term_scope))
    complexity = _validate_node(
        payload["ast"],
        depth=1,
        terms=terms,
        constants=constants,
        allowed_terms=allowed_terms,
        allowed_state=set(registers),
        allowed_obs=None,
    )
    components = _validate_components(
        payload.get("components"),
        terms=terms,
        constants=constants,
        allowed_terms=allowed_terms,
        allowed_state=set(registers),
    )
    term_set = sorted(terms)

    if declared_terms and declared_terms != term_set:
        raise ObjectiveSpecError(
            f"declared_term_usage {declared_terms} does not match AST terms {term_set}"
        )

    constants_dict = {f"c{i}": value for i, value in enumerate(constants)}
    ast = _canonicalize_node(payload["ast"])
    state = None
    if registers:
        state = {
            "registers": {
                name: {
                    "init": _canonicalize_node(expr["init"]),
                    "update": _canonicalize_node(expr["update"]),
                }
                for name, expr in registers.items()
            }
        }
        if state_interface:
            state["interface"] = state_interface
        if state_control:
            state["control"] = state_control
    objective_id = _objective_id(ast, state, net_weight_policy)
    return ObjectiveSpec(
        id=objective_id,
        ast=ast,
        constants=constants_dict,
        term_set=term_set,
        complexity=complexity + state_complexity + policy_complexity,
        created_by=created_by,
        parent_ids=parent_ids,
        rationale=rationale,
        source_program=None,
        components=components,
        state=state,
        observable_set=sorted(observables),
        net_weight_policy=net_weight_policy,
    )


def _validate_components(
    payload: Any,
    *,
    terms: set[str],
    constants: list[float],
    allowed_terms: set[str],
    allowed_state: set[str],
) -> dict[str, dict[str, Any]] | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ObjectiveSpecError("objective components must be an object")
    if len(payload) > MAX_COMPONENT_COUNT:
        raise ObjectiveSpecError(
            f"objective components exceed limit of {MAX_COMPONENT_COUNT}"
        )
    components: dict[str, dict[str, Any]] = {}
    for raw_name, expression in payload.items():
        name = str(raw_name)
        if not _REGISTER_NAME_RE.fullmatch(name):
            raise ObjectiveSpecError(f"invalid objective component name: {name!r}")
        _validate_node(
            expression,
            depth=1,
            terms=terms,
            constants=constants,
            allowed_terms=allowed_terms,
            allowed_state=allowed_state,
            allowed_obs=None,
        )
        components[name] = _canonicalize_node(expression)
    return components


def _validate_net_weight_policy(
    policy_payload: Any,
    registers: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, int, set[str]]:
    """Validate the per-net timing-aware weight-update policy block."""

    if policy_payload is None:
        return None, 0, set()
    if not isinstance(policy_payload, dict) or "update" not in policy_payload:
        raise ObjectiveSpecError(
            "net_weight_policy must be an object with an update expression"
        )
    observables: set[str] = set()
    complexity = _validate_node(
        policy_payload["update"],
        depth=1,
        terms=set(),
        constants=[],
        allowed_terms=set(),
        allowed_state=set(registers),
        allowed_obs=set(observable_names("all")),
        observables=observables,
        allowed_net=NET_OBSERVABLES,
    )
    cadence = int(policy_payload.get("cadence", NET_WEIGHT_POLICY_DEFAULTS["cadence"]))
    if cadence < 1:
        raise ObjectiveSpecError("net_weight_policy cadence must be >= 1")
    policy: dict[str, Any] = {
        "update": _canonicalize_node(policy_payload["update"]),
        "cadence": cadence,
    }
    control = str(policy_payload.get("control", "generated"))
    if control not in {"generated", "native_identity"}:
        raise ObjectiveSpecError(
            "net_weight_policy control must be 'generated' or 'native_identity'"
        )
    policy["control"] = control
    for clip_key in ("multiplier_clip", "weight_clip"):
        if clip_key not in policy_payload:
            continue
        raw = policy_payload[clip_key]
        if (
            not isinstance(raw, (list, tuple))
            or len(raw) != 2
            or not all(isinstance(v, (int, float)) for v in raw)
        ):
            raise ObjectiveSpecError(f"net_weight_policy {clip_key} must be [lo, hi]")
        lo, hi = float(raw[0]), float(raw[1])
        if not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo <= hi):
            raise ObjectiveSpecError(
                f"net_weight_policy {clip_key} must satisfy 0 <= lo <= hi"
            )
        policy[clip_key] = [lo, hi]
    return policy, complexity, observables


def _validate_state_block(
    state_payload: Any,
) -> tuple[dict[str, dict[str, Any]], int, set[str], str | None, str | None]:
    """Validate a DSL v2 state block. Returns (registers, complexity, observables)."""

    if state_payload is None:
        return {}, 0, set(), None, None
    if not isinstance(state_payload, dict):
        raise ObjectiveSpecError("state block must be a JSON object")
    registers_payload = state_payload.get("registers")
    if not isinstance(registers_payload, dict) or not registers_payload:
        raise ObjectiveSpecError("state block must contain a non-empty registers object")
    if len(registers_payload) > MAX_STATE_REGISTERS:
        raise ObjectiveSpecError(
            f"state registers exceed limit of {MAX_STATE_REGISTERS}"
        )

    interface_payload = state_payload.get("interface")
    state_interface = str(interface_payload) if interface_payload is not None else None
    if state_interface not in {None, TYPED_POLICY_INTERFACE}:
        raise ObjectiveSpecError(f"unsupported state interface: {state_interface}")
    if state_interface == TYPED_POLICY_INTERFACE:
        names = set(registers_payload)
        missing = TYPED_POLICY_REQUIRED_SLOTS - names
        unsupported = names - TYPED_POLICY_SLOTS
        if missing:
            raise ObjectiveSpecError(
                f"typed policy is missing required slots: {sorted(missing)}"
            )
        if unsupported:
            raise ObjectiveSpecError(
                f"typed policy has unsupported slots: {sorted(unsupported)}"
            )
    control_payload = state_payload.get("control")
    state_control = str(control_payload) if control_payload is not None else None
    if state_control not in {None, "generated", "native_identity"}:
        raise ObjectiveSpecError(f"unsupported state control mode: {state_control}")
    if state_control and state_interface != TYPED_POLICY_INTERFACE:
        raise ObjectiveSpecError("state control mode requires typed_policy_v1")

    register_names = set()
    for name in registers_payload:
        if not isinstance(name, str) or not _REGISTER_NAME_RE.match(name):
            raise ObjectiveSpecError(
                f"invalid register name: {name!r} (lowercase identifier required)"
            )
        if len(name) > MAX_REGISTER_NAME_LENGTH:
            raise ObjectiveSpecError(
                f"register name exceeds {MAX_REGISTER_NAME_LENGTH} chars: {name}"
            )
        register_names.add(name)

    # Spec-level validation accepts the privileged set so identity-control
    # presets parse; LLM-visibility is enforced at program parsing and by the
    # evaluator's candidate checks.
    allowed_obs = set(observable_names("all"))
    registers: dict[str, dict[str, Any]] = {}
    total_complexity = 0
    observables: set[str] = set()
    for name, entry in registers_payload.items():
        if not isinstance(entry, dict) or "init" not in entry:
            raise ObjectiveSpecError(f"register {name} must define an init expression")
        init_obs: set[str] = set()
        init_complexity = _validate_node(
            entry["init"],
            depth=1,
            terms=set(),
            constants=[],
            allowed_terms=set(),
            allowed_state=set(),
            allowed_obs=allowed_obs,
            observables=init_obs,
        )
        update_expr = entry.get("update")
        if update_expr is None:
            # Constant register: identity update keeps the init value.
            update_expr = {"op": "state", "name": name}
        update_obs: set[str] = set()
        update_complexity = _validate_node(
            update_expr,
            depth=1,
            terms=set(),
            constants=[],
            allowed_terms=set(),
            allowed_state=register_names,
            allowed_obs=allowed_obs,
            observables=update_obs,
        )
        registers[name] = {"init": entry["init"], "update": update_expr}
        total_complexity += init_complexity + update_complexity
        observables.update(init_obs)
        observables.update(update_obs)
    return registers, total_complexity, observables, state_interface, state_control


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
        state=payload.get("state"),
        observable_set=[str(obs) for obs in payload.get("observable_set", [])],
        net_weight_policy=payload.get("net_weight_policy"),
    )


def _validate_node(
    node: Any,
    depth: int,
    terms: set[str],
    constants: list[float],
    allowed_terms: set[str],
    allowed_state: set[str] | None = None,
    allowed_obs: set[str] | None = None,
    observables: set[str] | None = None,
    allowed_net: set[str] | None = None,
) -> int:
    if depth > MAX_AST_DEPTH:
        raise ObjectiveSpecError(f"AST depth exceeds {MAX_AST_DEPTH}")
    if not isinstance(node, dict):
        raise ObjectiveSpecError("AST node must be an object")

    op = node.get("op")
    if op == "net":
        name = node.get("name")
        if allowed_net is None:
            raise ObjectiveSpecError(
                "net() is only allowed inside the net_weight_policy update expression"
            )
        if name not in allowed_net:
            raise ObjectiveSpecError(f"unknown per-net observable: {name}")
        return 1
    if op == "term":
        name = node.get("name")
        if name not in allowed_terms:
            raise ObjectiveSpecError(f"unknown objective term: {name}")
        terms.add(str(name))
        return 1
    if op == "state":
        name = node.get("name")
        if allowed_state is None or name not in allowed_state:
            raise ObjectiveSpecError(
                f"state register {name!r} is not available in this expression"
            )
        return 1
    if op == "obs":
        name = node.get("name")
        if allowed_obs is None:
            raise ObjectiveSpecError(
                "obs() is only allowed inside policy/state initialization and update expressions"
            )
        if name not in allowed_obs and not is_component_calibration_observable(str(name)):
            raise ObjectiveSpecError(f"unknown observable: {name}")
        if observables is not None:
            observables.add(str(name))
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
        complexity += _validate_node(
            child,
            depth + 1,
            terms,
            constants,
            allowed_terms,
            allowed_state=allowed_state,
            allowed_obs=allowed_obs,
            observables=observables,
            allowed_net=allowed_net,
        )
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
    if op in {"term", "state", "obs", "net"}:
        return {"op": op, "name": str(node["name"])}
    if op == "const":
        return {"op": "const", "value": float(node["value"])}
    return {"op": op, "args": [_canonicalize_node(child) for child in node["args"]]}


def _objective_id(
    ast: dict[str, Any],
    state: dict[str, Any] | None = None,
    net_weight_policy: dict[str, Any] | None = None,
) -> str:
    if state or net_weight_policy:
        payload: Any = {"ast": ast}
        if state:
            payload["state"] = state
        if net_weight_policy:
            payload["net_weight_policy"] = net_weight_policy
    else:
        # Stateless ids stay byte-compatible with DSL v1.
        payload = ast
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return f"obj_{digest}"
