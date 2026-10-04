"""Archive-conditioned prompt sampler for CoEvoP&R objective programs."""

from __future__ import annotations

import json
import math
from typing import Any

from coevop.objectives.program import PROGRAM_SYSTEM_CONTRACT, program_example
from coevop.objectives.spec import ALLOWED_OPERATORS, MAX_AST_DEPTH, MAX_PRIMITIVE_COUNT
from coevop.objectives.terms import (
    observable_descriptions,
    observable_names,
    term_descriptions,
    term_names,
)
from coevop.prompt.templates import TemplateManager

# Tier C measurements attached to a routed candidate. Negative deltas and
# positive gains are improvements over the matched DREAMPlace placement.
ROUTED_EVIDENCE_KEYS = (
    "routed_wirelength_delta_pct",
    "routed_overflow_delta_pct",
    "post_route_wns_gain_ns",
    "post_route_tns_gain_ns",
)


class CoEvoPromptSampler:
    """Build archive-conditioned system and user prompts.

    The prompt presents the parent objective, archive evidence, placement
    interface, and required output schema described in the manuscript.
    """

    def __init__(self, template_dir: str | None = None) -> None:
        self.template_manager = TemplateManager(template_dir)

    def build_prompt(
        self,
        *,
        current_program: str,
        program_metrics: dict[str, Any],
        previous_programs: list[dict[str, Any]],
        top_programs: list[dict[str, Any]],
        inspirations: list[dict[str, Any]],
        term_scope: str,
        objective_mode: str,
        mutation_mode: str,
        artifacts: dict[str, Any] | None = None,
        feature_dimensions: list[str] | None = None,
        extra_context: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        if objective_mode == "controller":
            template_name = (
                "controller_diff_user"
                if mutation_mode == "diff"
                else "controller_full_rewrite_user"
            )
            system_template = "controller_system_message"
        else:
            template_name = "diff_user" if mutation_mode == "diff" else "full_rewrite_user"
            system_template = "system_message"
        user_template = self.template_manager.get_template(template_name)
        system_message = self.template_manager.get_template(system_template)
        feature_dimensions = feature_dimensions or []
        fitness = _fitness_score(program_metrics, feature_dimensions)
        user_message = user_template.format(
            fitness_score=f"{fitness:.6g}",
            feature_coords=_feature_coordinates(program_metrics, feature_dimensions),
            feature_dimensions=", ".join(feature_dimensions) if feature_dimensions else "None",
            improvement_areas=self._improvement_areas(program_metrics, previous_programs),
            task_environment=self._task_environment(
                term_scope=term_scope,
                objective_mode=objective_mode,
                extra_context=extra_context or {},
            ),
            artifacts=self._artifacts_section(artifacts or {}),
            evolution_history=self._evolution_history(
                previous_programs=previous_programs,
                top_programs=top_programs,
                inspirations=inspirations,
            ),
            current_program=current_program,
            language="python",
        )
        return {
            "system": system_message,
            "user": user_message,
        }

    def _task_environment(
        self,
        *,
        term_scope: str,
        objective_mode: str,
        extra_context: dict[str, Any],
    ) -> str:
        terms = term_names(term_scope)
        descriptions = term_descriptions(term_scope)
        lines = [
            "The Python objective environment is:",
            "",
            "```python",
            PROGRAM_SYSTEM_CONTRACT,
            "```",
            "",
            "Task: write a differentiable placement objective for DREAMPlace that improves downstream routed PPA after evaluation.",
            "The evaluator, not the prompt, decides whether tradeoffs among wirelength, congestion, timing, runtime, and stability are good.",
            (
                "Use term(\"...\") for differentiable placement terms and the "
                "typed policy and observable accessors defined below."
                if term_scope == "dreamplace_controller"
                else "Use only placement-state observables available through term(\"...\")."
            ),
            "Do not use external labels or tool reports as direct objective inputs.",
            "Explore objective mechanisms that are meaningfully different from previously evaluated baselines and failures.",
            "The DSL is not limited to weighted sums: smooth transforms and compact multiplicative interactions are valid when they remain differentiable and physically interpretable.",
            "",
            f"Objective mode: {objective_mode}",
            f"Allowed AST operators: {', '.join(sorted(ALLOWED_OPERATORS))}",
            f"AST size limits: depth <= {MAX_AST_DEPTH}, primitives <= {MAX_PRIMITIVE_COUNT}",
            "",
            "Available differentiable placement terms:",
        ]
        for term in terms:
            lines.append(f"- {term}: {descriptions.get(term, '')}")
        if term_scope == "dreamplace_controller":
            obs_descriptions = observable_descriptions("dreamplace_controller")
            lines.extend(
                [
                    "",
                    "Scalar optimizer observables, readable ONLY inside "
                    "init_policy(obs) and update_policy(policy, obs):",
                ]
            )
            for obs_name in observable_names("dreamplace_controller"):
                lines.append(f"- {obs_name}: {obs_descriptions.get(obs_name, '')}")
            lines.extend(
                [
                    "",
                    "Controller semantics: update_policy runs once per placement "
                    "iteration at the native density-weight cadence; all update "
                    "expressions read previous policy values and commit "
                    "together. Terms are RAW (no hidden normalization); build "
                    "weights from grad_ratio_*/init_value_* calibration "
                    "observables. Required density_weight drives both the density "
                    "component and preconditioning; required gamma_scale drives "
                    "wirelength smoothing. Optional route_weight and pin_weight "
                    "bind only to their physical families. Policy slots must stay "
                    "finite and nonnegative at every iteration.",
                ]
            )
        lines.extend(
            [
                "",
                "Syntax-only objective program example, not a recommended formula:",
                "```python",
                _objective_example(term_scope, objective_mode),
                "```",
            ]
        )
        policy = extra_context.get("search_policy") or {}
        if policy:
            evaluator_policy = {
                key: value
                for key, value in {
                    "active_routing_terms": policy.get("active_routing_terms"),
                    "routing_mechanism_policy": policy.get("routing_mechanism_policy"),
                    "baseline_similarity": policy.get("baseline_similarity"),
                    "coefficient_calibration": policy.get("coefficient_calibration"),
                    "selection_feedback": policy.get("selection_feedback"),
                    "evolution_memory": policy.get("evolution_memory"),
                    "composition_mechanisms": policy.get("composition_mechanisms"),
                    "net_weight_policy": policy.get("net_weight_policy"),
                    "timing_proxy_feedback": policy.get("timing_proxy_feedback"),
                    "target_design_context": policy.get("target_design_context"),
                    "chip_design_profiles": policy.get("chip_design_profiles"),
                }.items()
                if value not in (None, [], {})
            }
            lines.extend(
                [
                    "",
                    "Evaluator policy available for context:",
                    "```json",
                    json.dumps(evaluator_policy, indent=2, sort_keys=True),
                    "```",
                ]
            )
        return "\n".join(lines)

    def _evolution_history(
        self,
        *,
        previous_programs: list[dict[str, Any]],
        top_programs: list[dict[str, Any]],
        inspirations: list[dict[str, Any]],
    ) -> str:
        previous_template = self.template_manager.get_template("previous_attempt")
        top_template = self.template_manager.get_template("top_program")
        history_template = self.template_manager.get_template("evolution_history")

        previous_blocks = []
        for index, program in enumerate(previous_programs[-3:], start=1):
            previous_blocks.append(
                previous_template.format(
                    attempt_number=index,
                    changes=program.get("failure_reason")
                    or program.get("metrics", {}).get("feedback_lesson")
                    or "candidate evaluated",
                    performance=_format_metrics(program.get("metrics", {})),
                    outcome=program.get("status", "unknown"),
                )
            )

        top_blocks = []
        for index, program in enumerate(top_programs, start=1):
            top_blocks.append(
                top_template.format(
                    program_number=index,
                    score=f"{_fitness_score(program.get('metrics', {}), []):.6g}",
                    language="python",
                    program_snippet=program.get("code", ""),
                    key_features=_key_features(program),
                )
            )

        inspirations_section = ""
        if inspirations:
            inspiration_template = self.template_manager.get_template("inspiration_program")
            inspiration_blocks = []
            for index, program in enumerate(inspirations, start=1):
                inspiration_blocks.append(
                    inspiration_template.format(
                        program_number=index,
                        score=f"{_fitness_score(program.get('metrics', {}), []):.6g}",
                        program_type=_program_type(program),
                        language="python",
                        program_snippet=program.get("code", ""),
                        unique_features=_key_features(program),
                    )
                )
            inspirations_section = self.template_manager.get_template(
                "inspirations_section"
            ).format(inspiration_programs="\n\n".join(inspiration_blocks))

        return history_template.format(
            previous_attempts="\n\n".join(previous_blocks) or "No previous attempts yet.",
            top_programs="\n\n".join(top_blocks) or "No top programs yet.",
            inspirations_section=inspirations_section,
        )

    def _artifacts_section(self, artifacts: dict[str, Any]) -> str:
        if not artifacts:
            return ""
        sections = []
        for key, value in artifacts.items():
            text = value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True)
            if len(text) > 8000:
                text = text[:8000] + "\n... (truncated)"
            sections.append(f"### {key}\n```\n{text}\n```")
        return "# Evaluation Artifacts\n\n" + "\n\n".join(sections)

    def _improvement_areas(
        self,
        metrics: dict[str, Any],
        previous_programs: list[dict[str, Any]],
    ) -> str:
        areas = []
        if previous_programs:
            previous = previous_programs[-1].get("metrics", {})
            current_fitness = _fitness_score(metrics, [])
            previous_fitness = _fitness_score(previous, [])
            if current_fitness > previous_fitness:
                areas.append(
                    f"Fitness improved from {previous_fitness:.6g} to {current_fitness:.6g}."
                )
            elif current_fitness < previous_fitness:
                areas.append(
                    f"Fitness declined from {previous_fitness:.6g} to {current_fitness:.6g}."
                )
            else:
                areas.append(f"Fitness is unchanged at {current_fitness:.6g}.")
        if metrics.get("negative_memory_only"):
            areas.append("This program is negative memory; use its metrics as a warning, not as a parent pattern.")
        if metrics.get("routing_term_gate_passed") is False:
            areas.append("The program lacks a material active routing mechanism.")
        if metrics.get("structural_failure_count"):
            areas.append("The program failed structurally during evaluation.")
        if not areas:
            areas.append("Explore a meaningful objective mechanism that may improve evaluator fitness.")
        return "\n".join(f"- {area}" for area in areas)


def _format_metrics(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "none"
    parts = []
    priority_keys = [
        "hpwl_delta_pct",
        "overflow_delta_pct",
        "mechanism_family",
        "hpwl_delta_bucket",
        "overflow_delta_bucket",
        "opentimer_wns_delta_bucket",
        "timing_proxy_wns_delta",
        "timing_proxy_tns_delta",
        "timing_proxy_tns_delta_pct",
        "timing_proxy_mode",
        "timing_proxy_parent_signal",
        *ROUTED_EVIDENCE_KEYS,
        "outcome_label",
        "feedback_lesson",
    ]
    ordered_keys = [key for key in priority_keys if key in metrics]
    ordered_keys.extend(key for key in sorted(metrics) if key not in set(ordered_keys))
    for key in ordered_keys:
        value = metrics[key]
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            parts.append(f"{key}: {float(value):.6g}")
        elif isinstance(value, (str, bool)) or value is None:
            parts.append(f"{key}: {value}")
    return ", ".join(parts[:20])


def _fitness_score(metrics: dict[str, Any], feature_dimensions: list[str]) -> float:
    for key in ("combined_score", "fitness", "score"):
        if key in metrics and key not in feature_dimensions:
            value = _finite_float(metrics.get(key))
            if value is not None:
                return value
    return 0.0


def _feature_coordinates(metrics: dict[str, Any], feature_dimensions: list[str]) -> str:
    if not feature_dimensions:
        return "None"
    parts = []
    for name in feature_dimensions:
        value = metrics.get(name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            parts.append(f"{name}={float(value):.6g}")
        else:
            parts.append(f"{name}={value}")
    return ", ".join(parts)


def _key_features(program: dict[str, Any]) -> str:
    metrics = program.get("metrics", {})
    terms = program.get("term_set") or []
    pieces = []
    if terms:
        pieces.append("terms=" + "+".join(str(term) for term in terms))
    for key in (
        "hpwl_delta_pct",
        "overflow_delta_pct",
        "mechanism_family",
        "hpwl_delta_bucket",
        "overflow_delta_bucket",
        "opentimer_wns_delta_bucket",
        "timing_proxy_wns_delta",
        "timing_proxy_tns_delta",
        "timing_proxy_tns_delta_pct",
        "timing_proxy_parent_signal",
        *ROUTED_EVIDENCE_KEYS,
        "custom_grad_norm",
        "outcome_label",
    ):
        if key in metrics:
            pieces.append(f"{key}={metrics[key]}")
    if not pieces:
        pieces.append("no summarized metrics")
    return ", ".join(pieces)


def _program_type(program: dict[str, Any]) -> str:
    metrics = program.get("metrics", {})
    if metrics.get("negative_memory_only"):
        return "negative-memory"
    if metrics.get("routing_aware_tier2_candidate"):
        return "routing-aware"
    if metrics.get("seed_baseline"):
        return "baseline"
    return "diverse"


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _objective_example(term_scope: str, objective_mode: str) -> str:
    if term_scope == "dreamplace_controller" or objective_mode == "controller":
        return program_example("dreamplace_controller")
    if term_scope in {"dreamplace", "dreamplace_replacement", "dreamplace_native_residual"}:
        if objective_mode == "native_residual":
            return "\n".join(
                [
                    "def objective(features):",
                    '    native = term("native_objective")',
                    '    route = term("soft_rudy_pnorm")',
                    "    correction = 0.001 * log1p(route)",
                    "    score = native + correction",
                    (
                        '    return score, {"native": native, '
                        '"route_hotspot": route}'
                    ),
                ]
            )
        if objective_mode == "residual":
            return "\n".join(
                [
                    "def objective(features):",
                    '    wirelength = term("wirelength_wawl")',
                    '    density = term("density_electric")',
                    '    route = term("soft_rudy_mean")',
                    "    correction = 0.01 * log1p(route)",
                    "    score = wirelength + density + correction",
                    '    return score, {"wirelength": wirelength, "density": density, "route_mean": route}',
                ]
            )
        return "\n".join(
            [
                "def objective(features):",
                '    wirelength = term("wirelength_lse")',
                '    density = term("density_bell")',
                '    route = term("soft_rudy_mean")',
                "    route_pressure = log1p(route)",
                "    score = wirelength + density + 0.01 * route_pressure",
                '    return score, {"wirelength": wirelength, "density": density, "route_pressure": route}',
            ]
        )
    return program_example(term_scope)
