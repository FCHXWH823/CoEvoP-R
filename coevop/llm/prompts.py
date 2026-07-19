"""Prompt construction for LLM objective generation."""

from __future__ import annotations

import json
from typing import Any

from coevop.objectives.spec import (
    ALLOWED_OPERATORS,
    MAX_AST_DEPTH,
    MAX_PRIMITIVE_COUNT,
)
from coevop.objectives.program import PROGRAM_SYSTEM_CONTRACT, program_example
from coevop.objectives.terms import (
    observable_descriptions,
    observable_names,
    term_descriptions,
    term_names,
)


SYSTEM_PROMPT = f"""You are generating symbolic placement objective candidates for CoEvoP&R.
Return only JSON matching the supplied schema.
Write the objective as restricted Python code in the objective_program string.
The platform parses the code but never executes arbitrary model output.
Use only allowed terms and operators. Prefer simple, differentiable, stable formulas.
The objective score should increase when the current placement state is more costly
or routing-risky. External labels and tool reports are feedback/evaluation targets,
not objective inputs.

{PROGRAM_SYSTEM_CONTRACT}"""


def _environment_description(term_scope: str) -> dict[str, Any]:
    if term_scope == "dreamplace_controller":
        return {
            "view": "DREAMPlace stateful-controller objective environment",
            "observable_semantics": (
                "Terms are RAW differentiable placement observables (no hidden "
                "normalization); their magnitudes differ per design. Scalar "
                "optimizer observables (iteration progress, overflow, HPWL "
                "trend, gamma anneal position) and logged calibration constants "
                "(grad_ratio_*, init_value_*) are readable ONLY inside "
                "init_policy/update_policy. Construct policy weights explicitly "
                "from calibration observables instead of guessing absolute "
                "constants."
            ),
            "controller_semantics": (
                "Typed policy slots update once per placement iteration, at the "
                "same cadence DREAMPlace uses for its own density-weight "
                "schedule. All update expressions read the previous register "
                "values and commit together. The objective is re-evaluated "
                "with the committed policy every iteration. density_weight drives "
                "both the density component and gradient preconditioning, while "
                "gamma_scale drives wirelength smoothing. Direct fixed physical-term "
                "coefficients are not part of the typed interface."
            ),
            "label_policy": (
                "External labels and tool reports are feedback/evaluation "
                "targets, not objective inputs. Use only term(), policy(), and "
                "obs() accessors."
            ),
            "deployability": (
                "Candidates are accepted only if every policy slot stays finite "
                "at every iteration and the score is finite with usable "
                "gradients during DREAMPlace runs."
            ),
        }
    if term_scope == "dreamplace_native_residual":
        return {
            "view": "DREAMPlace native-residual objective environment",
            "observable_semantics": (
                "The objective preserves DREAMPlace's native adaptive objective "
                "through term(\"native_objective\") and may add differentiable "
                "routing or pin-access correction observables."
            ),
            "label_policy": (
                "External labels and tool reports are feedback/evaluation targets, "
                "not objective inputs. Use only term(\"...\") observables."
            ),
            "deployability": (
                "All exposed terms are executable by the DREAMPlace custom "
                "objective patch. Candidates are accepted only if they are finite, "
                "differentiable, and non-degenerate during DREAMPlace runs."
            ),
        }
    if term_scope == "dreamplace_replacement":
        return {
            "view": "DREAMPlace replacement-objective environment",
            "observable_semantics": (
                "Each term is a differentiable placement-state observable computed "
                "inside DREAMPlace during optimization. The generated formula may "
                "replace the default DREAMPlace objective. Default DREAMPlace is "
                "represented by explicit wirelength and density terms rather than a "
                "privileged integrated input."
            ),
            "label_policy": (
                "External labels and tool reports are feedback/evaluation targets, "
                "not objective inputs. Use only term(\"...\") observables."
            ),
            "deployability": (
                "All exposed terms are supported by the DREAMPlace custom objective "
                "patch. Candidates are accepted only if they are finite, "
                "differentiable, and non-degenerate during DREAMPlace runs."
            ),
        }
    if term_scope == "dreamplace":
        return {
            "view": "DREAMPlace optimizer environment",
            "observable_semantics": (
                "Each term is a differentiable placement-state observable computed "
                "from the current DREAMPlace placement during optimization."
            ),
            "label_policy": (
                "Do not use external labels or tool reports as objective inputs. They "
                "are used only after evaluation to score whether the objective helped."
            ),
            "deployability": "All exposed terms must be executable by the DREAMPlace custom objective patch.",
        }
    if term_scope == "deployable":
        return {
            "view": "Tier-1/Tier-2 bridge environment",
            "observable_semantics": (
                "Each term has an optional proxy-analysis scalar and a differentiable "
                "DREAMPlace implementation for placement evaluation."
            ),
            "label_policy": (
                "Labels are never objective inputs. Use the observables to express "
                "routing-aware placement cost; the evaluator measures correlation or "
                "placement impact afterward."
            ),
            "deployability": (
                "Formulas generated in this scope should be candidates for later "
                "DREAMPlace deployment, subject to term-gradient checks."
            ),
        }
    return {
        "view": "CircuitNet proxy-feedback environment",
        "observable_semantics": (
            "Each term is a scalar summary of a placement-state feature map from "
            "CircuitNet. These proxies are used for cheap search and filtering."
        ),
        "label_policy": (
            "Labels are evaluation feedback only. Do not write formulas that depend "
            "on target or label fields."
        ),
        "deployability": (
            "Some proxy-analysis terms are not yet optimizer terms. Prefer deployable "
            "terms when the context asks for DREAMPlace-ready candidates."
        ),
    }


def _mutation_mode_guidance(mutation_mode: str, context: dict[str, Any]) -> dict[str, Any]:
    objective_mode = str(context.get("objective_mode", "replacement"))
    policy = dict(context.get("search_policy", {}))
    guidance = {
        "mode": mutation_mode,
        "api_memory": "The API is stateless. Use only the explicit context in this request.",
        "return_complete_program": True,
    }
    if mutation_mode == "diff":
        guidance["instruction"] = (
            "Make a small SEARCH/REPLACE-style conceptual edit to the parent program. "
            "Prefer changing the objective mechanism, observable family, or nonlinear "
            "interaction; coefficient-only edits are allowed only when they support a "
            "clear mechanism change. Return the complete edited objective_program "
            "string because the platform validates complete programs."
        )
    else:
        guidance["instruction"] = "You may generate a fresh complete objective_program."
    if objective_mode == "residual":
        guidance["residual_contract"] = {
            "required_base": 'base = term("wirelength") + term("density")',
            "score_shape": "score = base + route_correction",
            "must_include_terms": ["wirelength", "density"],
            "calibration_policy": {
                "model_role": (
                    "Propose a physically plausible residual structure. The "
                    "platform will evaluate calibrated siblings, so focus on the "
                    "objective mechanism rather than guessing the exact final scale."
                ),
            },
            "evaluation_policy": (
                "DREAMPlace/OpenROAD-style metrics are measured after execution. "
                "The evaluator decides whether wirelength, congestion, timing, "
                "runtime, and stability tradeoffs improve fitness."
            ),
            "robustness_policy": {
                "instruction": (
                    "The evaluator records per-design behavior after execution. "
                    "Use feedback from prior placements to avoid mechanisms that "
                    "only win by sacrificing other designs."
                ),
            },
        }
    elif objective_mode == "controller":
        guidance["controller_contract"] = {
            "status": "stateful_controller_mode",
            "program_shape": (
                "init_policy(obs) -> density_weight, gamma_scale, and optional "
                "route_weight/pin_weight; optional update_policy(policy, obs) "
                "returns the same slots; objective(features, policy) returns "
                "(score, components)"
            ),
            "why": (
                "DREAMPlace's default objective is a schedule, not a fixed "
                "formula: density pressure ramps multiplicatively while HPWL "
                "improves. Static weighted sums cannot represent such "
                "schedules. Typed policy slots let the candidate express its own "
                "schedule from measured progress."
            ),
            "schedule_inputs": [
                "iter_frac",
                "overflow",
                "hpwl_delta_rate",
                "gamma_frac",
            ],
            "calibration_inputs": (
                "grad_ratio_<term> and init_value_<term> are constants "
                "measured at the first evaluation on the current design and "
                "logged. Use them to build design-portable weights, e.g. "
                'lam0 = 0.00008 * obs("grad_ratio_density_electric").'
            ),
            "recommended_patterns": [
                "multiplicative density ramp gated on hpwl_delta_rate or overflow",
                "late-stage routing pressure ramp gated on iter_frac or overflow",
                "policy-weighted route term with grad_ratio calibration",
            ],
            "feedback_interpretation": (
                "Policy trajectories and per-term component trajectories are "
                "reported after each run. A policy slot that stays constant is "
                "not controlling anything; a component whose value is orders "
                "of magnitude off the wirelength scale needs recalibration via "
                "grad_ratio_*/init_value_* rather than bare constants."
            ),
            "evaluation_policy": (
                "The evaluator measures HPWL, overflow, runtime, gradient "
                "health, and per-design robustness after full placement runs; "
                "routed and timing evidence arrives for finalists. Propose a "
                "mechanism with a clear physical meaning."
            ),
        }
    elif objective_mode == "native_residual":
        guidance["native_residual_contract"] = {
            "status": "native_default_preserving_mode",
            "required_base": 'base = term("native_objective")',
            "score_shape": "score = base + routing_correction",
            "must_include_terms": ["native_objective"],
            "why": (
                "This mode compares directly against default DREAMPlace on large "
                "or out-of-family designs. Keep DREAMPlace's adaptive native "
                "wirelength-density schedule intact, and discover only a routing-"
                "aware residual correction."
            ),
            "routing_terms": [
                "soft_rudy_mean",
                "soft_rudy_pnorm",
                "route_pressure_long",
                "pin_density_pnorm",
                "pin_count_weighted_wl",
            ],
            "recommended_patterns": [
                "base + c * soft_rudy_mean",
                "base + c * log1p(soft_rudy_pnorm)",
                "base + c * route_pressure_long",
                "base + c * pin_density_pnorm",
            ],
            "evaluation_policy": (
                "The evaluator measures wirelength, congestion/overflow, timing "
                "proxy when available, runtime, and stability after execution. "
                "It may create calibrated siblings, so propose a mechanism rather "
                "than trying to guess the exact final coefficient."
            ),
        }
    return guidance


def generation_messages(
    context: dict[str, Any] | None = None,
    term_scope: str = "tier1",
    mutation_mode: str = "full",
) -> list[dict[str, str]]:
    context = context or {}
    if context.get("prompt_messages"):
        return [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in context["prompt_messages"]
        ]
    objective_mode = str(context.get("objective_mode", "replacement"))
    example_program = program_example(term_scope)
    if objective_mode == "native_residual":
        example_program = "\n".join(
            [
                "def objective(features):",
                '    base = term("native_objective")',
                '    route = term("soft_rudy_pnorm")',
                "    score = base + 0.001 * log1p(route)",
                '    return score, {"native": base, "route_hotspot": route}',
            ]
        )
    payload = {
        "task": "Propose one routing-aware symbolic objective candidate.",
        "term_scope": term_scope,
        "placement_environment": _environment_description(term_scope),
        "allowed_terms": term_names(term_scope),
        "term_descriptions": term_descriptions(term_scope),
        "allowed_operators": sorted(ALLOWED_OPERATORS),
        "objective_api": {
            "entrypoint": "def objective(features): ...",
            "term_access": 'term("allowed_term_name")',
            "return_format": "return score, {component_name: component_expression}",
            "example_program": example_program,
        },
        "mutation_guidance": _mutation_mode_guidance(mutation_mode, context),
        "constraints": {
            "max_ast_depth": MAX_AST_DEPTH,
            "max_primitive_count": MAX_PRIMITIVE_COUNT,
            "no_arbitrary_code": True,
            "no_data_dependent_branches": True,
            "output_must_be_finite_after_vectorized_evaluation": True,
        },
        "context": context,
    }
    _add_controller_api(payload, term_scope)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, indent=2, sort_keys=True)},
    ]


def _add_controller_api(payload: dict[str, Any], term_scope: str) -> None:
    if term_scope != "dreamplace_controller":
        return
    payload["allowed_observables"] = observable_names("dreamplace_controller")
    payload["observable_descriptions"] = observable_descriptions("dreamplace_controller")
    api = payload.setdefault("objective_api", {})
    api["entrypoint"] = (
        "def objective(features, policy): ... with init_policy(obs) and optional "
        "update_policy(policy, obs)"
    )
    api["policy_access"] = 'policy("density_weight"|"gamma_scale"|"route_weight"|"pin_weight")'
    api["obs_access"] = 'obs("observable_name"), init_policy/update_policy only'
    api["required_policy_slots"] = ["density_weight", "gamma_scale"]
    api["optional_policy_slots"] = ["route_weight", "pin_weight"]


def mutation_messages(
    parents: list[dict[str, Any]],
    feedback: dict[str, Any] | None = None,
    term_scope: str = "tier1",
) -> list[dict[str, str]]:
    feedback = feedback or {}
    payload = {
        "task": "Mutate or recombine the parent objective candidates into one new candidate.",
        "term_scope": term_scope,
        "placement_environment": _environment_description(term_scope),
        "allowed_terms": term_names(term_scope),
        "term_descriptions": term_descriptions(term_scope),
        "allowed_operators": sorted(ALLOWED_OPERATORS),
        "objective_api": {
            "entrypoint": "def objective(features): ...",
            "term_access": 'term("allowed_term_name")',
            "return_format": "return score, {component_name: component_expression}",
            "example_program": program_example(term_scope),
        },
        "parents": parents,
        "feedback": feedback,
        "constraints": {
            "max_ast_depth": MAX_AST_DEPTH,
            "max_primitive_count": MAX_PRIMITIVE_COUNT,
            "preserve_interpretability": True,
            "avoid_duplicate_formula": True,
        },
    }
    _add_controller_api(payload, term_scope)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, indent=2, sort_keys=True)},
    ]


def reflection_messages(candidate: dict[str, Any], metrics: dict[str, Any]) -> list[dict[str, str]]:
    payload = {
        "task": "Reflect on why this placement objective did or did not work.",
        "candidate": candidate,
        "metrics": metrics,
        "instructions": (
            "Return concise text only, no new formula. Analyze component usefulness, "
            "flat or near-constant components, scale imbalance, train-validation gaps, "
            "target-specific correlations, complexity, and DREAMPlace deployability. "
            "Do not infer from heldout metrics; heldout is not provided during evolution."
        ),
    }
    return [
        {"role": "system", "content": "You analyze placement-objective experiment results."},
        {"role": "user", "content": json.dumps(payload, indent=2, sort_keys=True)},
    ]
