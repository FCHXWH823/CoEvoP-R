"""Parser and validator for restricted trajectory-aware objectives."""

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
    MAX_STATE_REGISTERS,
    MIN_CONSTANT,
    NET_OBSERVABLES,
    UNARY_OPERATORS,
    VARIADIC_OPERATORS,
    ObjectiveSpec,
    ObjectiveSpecError,
    TYPED_POLICY_INTERFACE,
    TYPED_POLICY_OPTIONAL_SLOTS,
    TYPED_POLICY_REQUIRED_SLOTS,
    TYPED_POLICY_SLOTS,
    parse_objective_spec,
)
from coevop.objectives.terms import (
    COMPONENT_GRAD_RATIO_PREFIX,
    COMPONENT_INIT_VALUE_PREFIX,
    is_component_calibration_observable,
    observable_names,
    term_names,
)


ALLOWED_PROGRAM_FUNCTIONS = sorted(
    ALLOWED_OPERATORS | {"term", "state", "policy", "obs", "net"}
)
OBJECTIVE_FUNCTION_NAMES = {"objective", "compute_objective"}
STATE_INIT_FUNCTION_NAME = "init_state"
STATE_UPDATE_FUNCTION_NAME = "update_state"
POLICY_INIT_FUNCTION_NAME = "init_policy"
POLICY_UPDATE_FUNCTION_NAME = "update_policy"
NET_UPDATE_FUNCTION_NAME = "update_net_weights"
MAX_COMPONENT_COUNT = 6

_WIRELENGTH_TERMS = {"wirelength", "wirelength_wawl", "wirelength_lse"}
_DENSITY_TERMS = {"density", "density_electric", "density_bell"}
_ROUTING_TERMS = {"soft_rudy_mean", "soft_rudy_pnorm", "route_pressure_long"}
_PIN_TERMS = {"pin_density_pnorm", "pin_count_weighted_wl"}


PROGRAM_SYSTEM_CONTRACT = """Write one typed trajectory-aware objective using these functions.

def init_policy(obs):
    return {"density_weight": ..., "gamma_scale": ..., ...}

def update_policy(policy, obs):
    return the exact same policy slots using policy("name"), obs("name"), constants

def objective(features, policy):
    score = <wirelength anchor plus policy-weighted physical components>
    return score, {"component_name": ...}

Optional timing-aware net policy (executed at DREAMPlace's configured timing
update cadence after timing criticalities are refreshed):

def update_net_weights(net, policy, obs):
    return <bounded multiplier using net("criticality"), net("span"), net("fanout")>

Required policy slots are density_weight and gamma_scale. Optional slots are
route_weight and pin_weight. update_policy runs once per placement iteration,
at the native density-weight cadence. All updates read the previous policy and
commit together. Adjustable physical-term coefficients must use policy slots;
fixed top-level coefficients are rejected. Constants remain allowed inside
schedule updates and smooth component transformations.
For a named unweighted returned component X, init/update laws may read
obs("init_value_component_X") and obs("grad_ratio_component_X"). These are
measured once from the transformed component itself, not from a proxy term.

Accessor scoping (hard rules):
- obs("name") is allowed only inside init_policy/update_policy
- policy("name") is allowed inside update_policy and objective
- term("name") is allowed only inside objective
- net("name") is allowed only inside update_net_weights

Allowed expression syntax everywhere:
- numeric constants
- local variables assigned earlier in the same function
- +, -, *, / where / is lowered to safe_div
- log1p(x), sqrt(x), square(x), softplus(x), sigmoid(x), safe_div(x, y)

Size limits:
- exactly the required typed slots plus zero or more supported optional slots
- keep every expression (score, each component, each init, each update)
  at or below 12 AST primitives and depth 10
- COUNT PRIMITIVES BEFORE RETURNING: every term()/policy()/obs() call,
  constant, and operator node counts as one primitive, and expanding a local
  variable inlines its whole expression at every use site. Exceeding 12 is
  the single most common rejection. Put complexity into register update
  expressions (each has its own 12-primitive budget) and keep the score a
  short policy combination like wl + policy("density_weight") * den.
- return at most 6 named diagnostic components
- prefer compact formulas with a clear placement meaning

Forbidden:
- imports, loops, branches, comprehensions, lambdas, attributes, subscripts,
  arbitrary calls, mutation, random numbers, data-dependent control flow, and
  direct access to raw features/policy/state/obs objects (use the accessors).
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
                    "Restricted Python code defining the typed objective functions "
                    "init_policy(obs) / "
                    "update_policy(policy, obs) / objective(features, policy). "
                    "update_net_weights(net, policy, obs) is optional. "
                    "The code is parsed but never executed. Keep every expression "
                    f"at or below {MAX_PRIMITIVE_COUNT} AST primitives "
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

    functions = _module_functions(module)
    objective_fn = functions["objective"]
    init_fn = functions.get(STATE_INIT_FUNCTION_NAME)
    update_fn = functions.get(STATE_UPDATE_FUNCTION_NAME)
    policy_init_fn = functions.get(POLICY_INIT_FUNCTION_NAME)
    policy_update_fn = functions.get(POLICY_UPDATE_FUNCTION_NAME)
    typed_policy = policy_init_fn is not None or policy_update_fn is not None
    if typed_policy and (init_fn is not None or update_fn is not None):
        raise ObjectiveSpecError(
            "typed policy functions cannot be mixed with legacy state functions"
        )
    if policy_update_fn is not None and policy_init_fn is None:
        raise ObjectiveSpecError("update_policy requires init_policy to define slots")
    if policy_init_fn is not None:
        init_fn = policy_init_fn
        update_fn = policy_update_fn
    if update_fn is not None and init_fn is None:
        raise ObjectiveSpecError("update_state requires init_state to define registers")

    state_payload: dict[str, Any] | None = None
    register_names: set[str] = set()
    if init_fn is not None:
        init_exprs = _StateFunctionLowerer(
            kind="init",
            register_names=set(),
            typed_policy=typed_policy,
        ).lower(init_fn)
        register_names = set(init_exprs)
        if len(register_names) > MAX_STATE_REGISTERS:
            raise ObjectiveSpecError(
                f"state registers exceed limit of {MAX_STATE_REGISTERS}"
            )
        update_exprs: dict[str, dict[str, Any]] = {}
        if update_fn is not None:
            update_exprs = _StateFunctionLowerer(
                kind="update",
                register_names=register_names,
                typed_policy=typed_policy,
            ).lower(update_fn)
            if set(update_exprs) != register_names:
                noun = "policy slots" if typed_policy else "registers"
                raise ObjectiveSpecError(
                    f"{update_fn.name} must return exactly the {noun} defined by "
                    f"{init_fn.name}: {sorted(register_names)}"
                )
        registers: dict[str, dict[str, Any]] = {}
        for name in sorted(register_names):
            entry: dict[str, Any] = {"init": init_exprs[name]}
            if name in update_exprs:
                entry["update"] = update_exprs[name]
            registers[name] = entry
        state_payload = {"registers": registers}
        if typed_policy:
            state_payload["interface"] = TYPED_POLICY_INTERFACE
            state_payload["control"] = "generated"

    net_update_fn = functions.get(NET_UPDATE_FUNCTION_NAME)
    net_weight_policy: dict[str, Any] | None = None
    if net_update_fn is not None:
        net_weight_policy = {
            "update": _NetUpdateLowerer(
                register_names=register_names,
                typed_policy=typed_policy,
            ).lower(net_update_fn)
        }

    lowered = _ProgramLowerer(
        term_scope=term_scope,
        register_names=register_names,
        register_accessor="policy" if typed_policy else "state",
    ).lower(objective_fn)
    if typed_policy:
        _validate_typed_policy_objective(
            lowered.score_ast,
            lowered.components,
            register_names,
        )
        component_names = set(lowered.components)
        for expression in (
            expression
            for register in (state_payload or {}).get("registers", {}).values()
            for expression in (register.get("init"), register.get("update"))
            if isinstance(expression, dict)
        ):
            for observable in _ast_observable_names(expression):
                for prefix in (COMPONENT_GRAD_RATIO_PREFIX, COMPONENT_INIT_VALUE_PREFIX):
                    if (
                        observable.startswith(prefix)
                        and observable[len(prefix) :] not in component_names
                    ):
                        raise ObjectiveSpecError(
                            f"component calibration observable {observable!r} references "
                            "an objective component that is not returned"
                        )
    payload = {
        "id": "",
        "rationale": rationale,
        "parent_ids": parent_ids or [],
        "declared_term_usage": lowered.term_set,
        "ast": lowered.score_ast,
    }
    if state_payload is not None:
        payload["state"] = state_payload
    if net_weight_policy is not None:
        payload["net_weight_policy"] = net_weight_policy
    spec = parse_objective_spec(payload, created_by=created_by, term_scope=term_scope)
    return replace(
        spec,
        source_program=source.strip() + "\n",
        components=lowered.components or None,
    )


def program_example(term_scope: str = "tier1") -> str:
    if term_scope == "dreamplace_controller":
        return "\n".join(
            [
                "def init_policy(obs):",
                "    return {",
                '        "density_weight": 0.00008 * obs("grad_ratio_density_electric"),',
                '        "gamma_scale": 1.0,',
                '        "route_weight": 0.0,',
                "    }",
                "",
                "def update_policy(policy, obs):",
                '    density_weight = policy("density_weight") * (1.0 + 0.05 * sigmoid(8.0 * (obs("overflow") - 0.15)))',
                '    gamma_scale = sigmoid(8.0 * (obs("overflow") - 0.10))',
                '    route_weight = 0.0005 * sigmoid(6.0 * (0.5 - obs("iter_frac")))',
                '    return {"density_weight": density_weight, "gamma_scale": gamma_scale, "route_weight": route_weight}',
                "",
                "def objective(features, policy):",
                '    wl = term("wirelength_wawl")',
                '    den = term("density_electric")',
                '    route = log1p(term("soft_rudy_pnorm"))',
                '    score = wl + policy("density_weight") * den + policy("route_weight") * route',
                (
                    '    return score, {"wirelength": wl, "density": den, '
                    '"route_hotspot": route}'
                ),
            ]
        )
    if term_scope in {"dreamplace", "dreamplace_replacement", "dreamplace_native_residual"}:
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


def _validate_typed_policy_objective(
    score_ast: dict[str, Any],
    components: dict[str, dict[str, Any]],
    policy_slots: set[str],
) -> None:
    """Enforce the typed policy's physical role bindings.

    The runtime needs one unscaled wirelength anchor and one policy-weighted
    density component so that gamma and the density preconditioner cannot
    silently drift away from the objective being optimized. Optional route and
    pin slots are bound to their corresponding mechanism families.
    """

    summands = list(score_ast.get("args", [])) if score_ast.get("op") == "add" else [score_ast]
    wirelength_anchors = [
        node
        for node in summands
        if node.get("op") == "term" and node.get("name") in _WIRELENGTH_TERMS
    ]
    if len(wirelength_anchors) != 1:
        raise ObjectiveSpecError(
            "typed policy objective requires exactly one unit-coefficient "
            "wirelength anchor"
        )

    bound_slots: dict[str, dict[str, Any]] = {}
    for node in summands:
        if node in wirelength_anchors:
            continue
        slot, component = _typed_weighted_component(node)
        if slot is None or component is None:
            raise ObjectiveSpecError(
                "every non-wirelength top-level component must be multiplied "
                "by a typed policy slot; fixed physical-term coefficients are forbidden"
            )
        if slot in bound_slots:
            raise ObjectiveSpecError(f"typed policy slot {slot!r} is used more than once")
        allowed_family = {
            "density_weight": _DENSITY_TERMS,
            "route_weight": _ROUTING_TERMS,
            "pin_weight": _PIN_TERMS,
        }.get(slot)
        if allowed_family is None:
            raise ObjectiveSpecError(
                f"typed policy slot {slot!r} cannot weight an objective component"
            )
        terms = _ast_term_names(component)
        if not terms or not terms <= allowed_family:
            raise ObjectiveSpecError(
                f"typed policy slot {slot!r} must weight only "
                f"{sorted(allowed_family)}; found {sorted(terms)}"
            )
        bound_slots[slot] = component

    required_weight_slots = {"density_weight"}
    optional_weight_slots = policy_slots & {"route_weight", "pin_weight"}
    missing = (required_weight_slots | optional_weight_slots) - set(bound_slots)
    if missing:
        raise ObjectiveSpecError(
            f"typed policy objective does not use declared weight slots: {sorted(missing)}"
        )
    if "gamma_scale" in _ast_state_names(score_ast):
        raise ObjectiveSpecError(
            "gamma_scale is applied by DREAMPlace and must not appear in the score AST"
        )

    if "wirelength" not in components or "density" not in components:
        raise ObjectiveSpecError(
            "typed policy objective must return named wirelength and density components"
        )
    if not _ast_term_names(components["wirelength"]) <= _WIRELENGTH_TERMS:
        raise ObjectiveSpecError("wirelength component contains a non-wirelength term")
    if not _ast_term_names(components["density"]) <= _DENSITY_TERMS:
        raise ObjectiveSpecError("density component contains a non-density term")
    for component_name, component_ast in components.items():
        if _ast_state_names(component_ast):
            raise ObjectiveSpecError(
                f"typed policy diagnostic component {component_name!r} must be an "
                "unweighted physical expression; policy slots belong only in the score"
            )


def _typed_weighted_component(
    node: dict[str, Any],
) -> tuple[str | None, dict[str, Any] | None]:
    if node.get("op") != "mul":
        return None, None
    args = list(node.get("args", []))
    state_args = [arg for arg in args if arg.get("op") == "state"]
    if len(state_args) != 1:
        return None, None
    slot = str(state_args[0].get("name"))
    component_args = [arg for arg in args if arg is not state_args[0]]
    if len(component_args) != 1:
        return None, None
    return slot, component_args[0]


def _ast_term_names(node: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    if node.get("op") == "term":
        names.add(str(node.get("name")))
    for child in node.get("args", []):
        names.update(_ast_term_names(child))
    return names


def _ast_state_names(node: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    if node.get("op") == "state":
        names.add(str(node.get("name")))
    for child in node.get("args", []):
        names.update(_ast_state_names(child))
    return names


def _ast_observable_names(node: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    if node.get("op") == "obs":
        names.add(str(node.get("name")))
    for child in node.get("args", []):
        names.update(_ast_observable_names(child))
    return names


class _LoweredProgram:
    def __init__(self, score_ast: dict[str, Any], components: dict[str, dict[str, Any]]) -> None:
        self.score_ast = score_ast
        self.components = components
        self.term_set = sorted(_collect_terms(score_ast, components))


class _ProgramLowerer:
    def __init__(
        self,
        *,
        term_scope: str,
        register_names: set[str] | None = None,
        register_accessor: str = "state",
    ) -> None:
        self.term_scope = term_scope
        self.allowed_terms = set(term_names(term_scope))
        self.register_names = register_names or set()
        self.register_accessor = register_accessor
        self.allow_obs = False
        self.allow_net = False
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
            args.vararg is not None
            or args.kwarg is not None
            or args.kwonlyargs
            or args.defaults
            or args.kw_defaults
        ):
            raise ObjectiveSpecError(
                "objective function must not use varargs, kwargs, or defaults"
            )
        arg_names = [arg.arg for arg in args.args]
        if self.register_names:
            if arg_names != ["features", self.register_accessor]:
                raise ObjectiveSpecError(
                    "stateful objective function signature must be "
                    f"objective(features, {self.register_accessor})"
                )
        elif arg_names != ["features"]:
            raise ObjectiveSpecError(
                "objective function must have exactly one argument: features"
            )

    def _lower_assignment(self, statement: py_ast.Assign) -> None:
        if len(statement.targets) != 1 or not isinstance(statement.targets[0], py_ast.Name):
            raise ObjectiveSpecError("assignments must target one local variable")
        name = statement.targets[0].id
        if name in {"features", "policy", "state", "obs", "net"} or name in ALLOWED_PROGRAM_FUNCTIONS:
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
            if expr.id in {"features", "policy", "state", "obs", "net"}:
                raise ObjectiveSpecError(
                    f"direct {expr.id} access is forbidden; use the call accessors "
                    'term("name"), policy("name"), state("name"), obs("name"), net("name")'
                )
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

        if function_name in {"state", "policy"}:
            if function_name != self.register_accessor:
                raise ObjectiveSpecError(
                    f"{function_name}() is not available in this program; "
                    f"use {self.register_accessor}()"
                )
            state_name = _string_call_argument(expr, function_name)
            if state_name not in self.register_names:
                raise ObjectiveSpecError(
                    f"policy/register slot {state_name!r} is not defined by the initializer"
                )
            return {"op": "state", "name": state_name}

        if function_name == "obs":
            if not self.allow_obs:
                raise ObjectiveSpecError(
                    "obs() is only allowed inside policy/state initialization and update functions"
                )
            obs_name = _string_call_argument(expr, "obs")
            if (
                obs_name not in set(observable_names("dreamplace_controller"))
                and not is_component_calibration_observable(obs_name)
            ):
                raise ObjectiveSpecError(f"unknown observable: {obs_name}")
            return {"op": "obs", "name": obs_name}

        if function_name == "net":
            if not getattr(self, "allow_net", False):
                raise ObjectiveSpecError(
                    "net() is only allowed inside update_net_weights"
                )
            net_name = _string_call_argument(expr, "net")
            if net_name not in NET_OBSERVABLES:
                raise ObjectiveSpecError(f"unknown per-net observable: {net_name}")
            return {"op": "net", "name": net_name}

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


class _StateFunctionLowerer(_ProgramLowerer):
    """Lower init_state(obs) / update_state(state, obs) into register expressions."""

    def __init__(
        self,
        *,
        kind: str,
        register_names: set[str],
        typed_policy: bool = False,
    ) -> None:
        if kind not in {"init", "update"}:
            raise ValueError(f"unknown state function kind: {kind}")
        self.kind = kind
        self.typed_policy = typed_policy
        self.term_scope = f"state_{kind}"
        self.allowed_terms = set()
        self.register_names = set() if kind == "init" else set(register_names)
        self.register_accessor = "policy" if typed_policy else "state"
        self.allow_obs = True
        self.allow_net = False
        self.env: dict[str, dict[str, Any]] = {}

    def lower(self, function: py_ast.FunctionDef) -> dict[str, dict[str, Any]]:  # type: ignore[override]
        self._validate_signature(function)
        statements = _strip_docstring(function.body)
        if not statements:
            raise ObjectiveSpecError(f"{function.name} body is empty")
        if not isinstance(statements[-1], py_ast.Return):
            raise ObjectiveSpecError(f"{function.name} must end with a return statement")
        for statement in statements[:-1]:
            if isinstance(statement, py_ast.Assign):
                self._lower_assignment(statement)
                continue
            raise ObjectiveSpecError(
                f"unsupported statement in {function.name}: {statement.__class__.__name__}"
            )
        return self._lower_register_dict(function.name, statements[-1].value)

    def _validate_signature(self, function: py_ast.FunctionDef) -> None:
        args = function.args
        if (
            args.vararg is not None
            or args.kwarg is not None
            or args.kwonlyargs
            or args.defaults
            or args.kw_defaults
        ):
            raise ObjectiveSpecError(
                f"{function.name} must not use varargs, kwargs, or defaults"
            )
        arg_names = [arg.arg for arg in args.args]
        expected = (
            ["obs"]
            if self.kind == "init"
            else [self.register_accessor, "obs"]
        )
        if arg_names != expected:
            raise ObjectiveSpecError(
                f"{function.name} signature must be {function.name}({', '.join(expected)})"
            )

    def _lower_register_dict(
        self, function_name: str, value: py_ast.expr | None
    ) -> dict[str, dict[str, Any]]:
        if not isinstance(value, py_ast.Dict):
            raise ObjectiveSpecError(
                f"{function_name} must return a dictionary of register expressions"
            )
        if len(value.keys) > MAX_STATE_REGISTERS:
            raise ObjectiveSpecError(
                f"state registers exceed limit of {MAX_STATE_REGISTERS}"
            )
        registers: dict[str, dict[str, Any]] = {}
        for key, expr in zip(value.keys, value.values):
            if not isinstance(key, py_ast.Constant) or not isinstance(key.value, str):
                raise ObjectiveSpecError("register names must be string constants")
            name = str(key.value).strip()
            if not name:
                raise ObjectiveSpecError("register names must be non-empty")
            if name in registers:
                raise ObjectiveSpecError(f"duplicate register name: {name}")
            registers[name] = self._lower_expr(expr)
        if not registers:
            raise ObjectiveSpecError(f"{function_name} must define at least one register")
        if self.typed_policy:
            names = set(registers)
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
        return registers


class _NetUpdateLowerer(_ProgramLowerer):
    """Lower update_net_weights(net, state, obs) into a multiplier expression."""

    def __init__(self, *, register_names: set[str], typed_policy: bool = False) -> None:
        self.term_scope = "net_weight_policy"
        self.allowed_terms = set()
        self.register_names = set(register_names)
        self.register_accessor = "policy" if typed_policy else "state"
        self.allow_obs = True
        self.allow_net = True
        self.env: dict[str, dict[str, Any]] = {}

    def lower(self, function: py_ast.FunctionDef) -> dict[str, Any]:  # type: ignore[override]
        self._validate_signature(function)
        statements = _strip_docstring(function.body)
        if not statements:
            raise ObjectiveSpecError(f"{function.name} body is empty")
        if not isinstance(statements[-1], py_ast.Return):
            raise ObjectiveSpecError(f"{function.name} must end with a return statement")
        for statement in statements[:-1]:
            if isinstance(statement, py_ast.Assign):
                self._lower_assignment(statement)
                continue
            raise ObjectiveSpecError(
                f"unsupported statement in {function.name}: {statement.__class__.__name__}"
            )
        value = statements[-1].value
        if value is None or isinstance(value, (py_ast.Tuple, py_ast.List, py_ast.Dict)):
            raise ObjectiveSpecError(
                f"{function.name} must return a single multiplier expression"
            )
        return self._lower_expr(value)

    def _validate_signature(self, function: py_ast.FunctionDef) -> None:
        args = function.args
        if (
            args.vararg is not None
            or args.kwarg is not None
            or args.kwonlyargs
            or args.defaults
            or args.kw_defaults
        ):
            raise ObjectiveSpecError(
                f"{function.name} must not use varargs, kwargs, or defaults"
            )
        arg_names = [arg.arg for arg in args.args]
        if arg_names != ["net", self.register_accessor, "obs"]:
            raise ObjectiveSpecError(
                f"{function.name} signature must be "
                f"{function.name}(net, {self.register_accessor}, obs)"
            )


def _string_call_argument(expr: py_ast.Call, accessor: str) -> str:
    if len(expr.args) != 1:
        raise ObjectiveSpecError(f"{accessor}() requires exactly one argument")
    arg = expr.args[0]
    if not isinstance(arg, py_ast.Constant) or not isinstance(arg.value, str):
        raise ObjectiveSpecError(f"{accessor}() argument must be a string constant")
    return str(arg.value)


def _module_functions(module: py_ast.Module) -> dict[str, py_ast.FunctionDef]:
    statements = _strip_module_docstring(module.body)
    allowed_names = OBJECTIVE_FUNCTION_NAMES | {
        STATE_INIT_FUNCTION_NAME,
        STATE_UPDATE_FUNCTION_NAME,
        POLICY_INIT_FUNCTION_NAME,
        POLICY_UPDATE_FUNCTION_NAME,
        NET_UPDATE_FUNCTION_NAME,
    }
    functions: dict[str, py_ast.FunctionDef] = {}
    for statement in statements:
        if not isinstance(statement, py_ast.FunctionDef):
            raise ObjectiveSpecError(
                "objective program may contain only function definitions"
            )
        if statement.decorator_list:
            raise ObjectiveSpecError("decorators are not allowed")
        if statement.returns is not None:
            raise ObjectiveSpecError("return annotations are not allowed")
        if statement.name not in allowed_names:
            raise ObjectiveSpecError(
                f"unsupported function in objective program: {statement.name}"
            )
        key = "objective" if statement.name in OBJECTIVE_FUNCTION_NAMES else statement.name
        if key in functions:
            raise ObjectiveSpecError(f"duplicate function definition: {statement.name}")
        functions[key] = statement
    if "objective" not in functions:
        raise ObjectiveSpecError("objective program must define objective(features)")
    return functions


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
