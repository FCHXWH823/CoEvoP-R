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
from coevop.objectives.terms import term_descriptions, term_names


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
    elif objective_mode == "native_residual":
        guidance["native_residual_contract"] = {
            "status": "legacy_only_not_llm_facing",
            "required_base": 'base = term("wirelength_wawl") + term("density_electric")',
            "score_shape": "score = base + routing_correction",
            "must_include_terms": ["wirelength_wawl", "density_electric"],
            "why": (
                "Current CoEvoP&R prompts do not expose a magic integrated default "
                "term. Use explicit DREAMPlace terms so the model can reason about "
                "each objective component."
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
                "base + c1 * route_pressure_long + c2 * pin_density_pnorm",
                "base + c * sqrt(pin_count_weighted_wl)",
            ],
            "evaluation_policy": (
                "Native DREAMPlace behavior is one reference point, but the prompt "
                "should focus on meaningful routed-PPA mechanisms. The evaluator "
                "scores the resulting wirelength, congestion, timing, runtime, and "
                "stability tradeoffs after execution."
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
            "example_program": program_example(term_scope),
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
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, indent=2, sort_keys=True)},
    ]


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
