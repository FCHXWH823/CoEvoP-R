"""Curated objective specs used for smoke tests and baselines."""

from __future__ import annotations

from typing import Any

from coevop.objectives.spec import ObjectiveSpec, parse_objective_spec


def _wirelength_density_payload(rationale: str) -> dict[str, Any]:
    return {
        "id": "",
        "rationale": rationale,
        "parent_ids": [],
        "declared_term_usage": ["density", "wirelength"],
        "ast": {
            "op": "add",
            "args": [
                {"op": "term", "name": "wirelength"},
                {"op": "term", "name": "density"},
            ],
        },
    }


def _explicit_wawl_electric_payload(rationale: str) -> dict[str, Any]:
    return {
        "id": "",
        "rationale": rationale,
        "parent_ids": [],
        "declared_term_usage": ["density_electric", "wirelength_wawl"],
        "ast": {
            "op": "add",
            "args": [
                {"op": "term", "name": "wirelength_wawl"},
                {"op": "term", "name": "density_electric"},
            ],
        },
    }


def _single_term_payload(term: str, rationale: str) -> dict[str, Any]:
    return {
        "id": "",
        "rationale": rationale,
        "parent_ids": [],
        "declared_term_usage": [term],
        "ast": {"op": "term", "name": term},
    }


PRESET_PAYLOADS: dict[str, dict[str, Any]] = {
    "dreamplace_custom_default": _wirelength_density_payload(
        (
            "Legacy explicit custom-path baseline: smooth wirelength plus "
            "density. This is not DREAMPlace's native adaptive default; use "
            "dreamplace_native_default for native identity checks."
        )
    ),
    "dreamplace_explicit_wl_density": _wirelength_density_payload(
        (
            "Explicit custom-path baseline: DREAMPlace smooth wirelength plus "
            "explicit density observable. This is a fixed symbolic baseline for "
            "comparison with generated replacement objectives."
        )
    ),
    "dreamplace_wl_density": _wirelength_density_payload(
        "DREAMPlace-compatible baseline: smooth wirelength plus weighted density."
    ),
    "dreamplace_wawl_electric": _explicit_wawl_electric_payload(
        "Explicit DREAMPlace default-style baseline: weighted-average wirelength plus electric density."
    ),
    "dreamplace_wawl_density_bell": {
        "id": "",
        "rationale": (
            "Simple explicit alternative-density baseline: weighted-average "
            "wirelength plus NTUPlace3 bell-shaped density potential. This "
            "guards against mistaking a generated tiny routing perturbation of "
            "WAWL+bell density for a genuinely routing-aware objective."
        ),
        "parent_ids": [],
        "declared_term_usage": ["density_bell", "wirelength_wawl"],
        "ast": {
            "op": "add",
            "args": [
                {"op": "term", "name": "wirelength_wawl"},
                {"op": "term", "name": "density_bell"},
            ],
        },
    },
    "dreamplace_lse_density_bell": {
        "id": "",
        "rationale": (
            "Alternative smooth objective using log-sum-exp wirelength and "
            "NTUPlace3 bell-shaped density potential."
        ),
        "parent_ids": [],
        "declared_term_usage": ["density_bell", "wirelength_lse"],
        "ast": {
            "op": "add",
            "args": [
                {"op": "term", "name": "wirelength_lse"},
                {"op": "term", "name": "density_bell"},
            ],
        },
    },
    "dreamplace_density_heavy": {
        "id": "",
        "rationale": "DREAMPlace-compatible density-heavy variant for smoke testing.",
        "parent_ids": [],
        "declared_term_usage": ["density", "wirelength"],
        "ast": {
            "op": "add",
            "args": [
                {"op": "term", "name": "wirelength"},
                {
                    "op": "mul",
                    "args": [
                        {"op": "const", "value": 1.25},
                        {"op": "term", "name": "density"},
                    ],
                },
            ],
        },
    },
    "dreamplace_density_135": {
        "id": "",
        "rationale": (
            "DREAMPlace-compatible tuned-density baseline discovered by the "
            "OpenEvolve Tier-2 run: wirelength plus 1.35 times density. This "
            "is the main non-routing baseline for later routing-aware search."
        ),
        "parent_ids": [],
        "declared_term_usage": ["density", "wirelength"],
        "ast": {
            "op": "add",
            "args": [
                {"op": "term", "name": "wirelength"},
                {
                    "op": "mul",
                    "args": [
                        {"op": "const", "value": 1.35},
                        {"op": "term", "name": "density"},
                    ],
                },
            ],
        },
    },
    "dreamplace_native_default": {
        "id": "",
        "rationale": (
            "Legacy DREAMPlace-compatible native-default control: use DREAMPlace's "
            "adaptive wirelength plus density-weight objective as a custom term. "
            "This remains for old-run regression checks only and is not exposed to "
            "new LLM-generated objectives."
        ),
        "parent_ids": [],
        "declared_term_usage": ["native_objective"],
        "ast": {"op": "term", "name": "native_objective"},
    },
    "dreamplace_native_soft_rudy_mean": {
        "id": "",
        "rationale": (
            "Native-DREAMPlace residual with a small mean soft-RUDY correction. "
            "This tests whether a routing-aware correction can improve overflow "
            "without replacing DREAMPlace's adaptive density schedule."
        ),
        "parent_ids": [],
        "declared_term_usage": ["native_objective", "soft_rudy_mean"],
        "ast": {
            "op": "add",
            "args": [
                {"op": "term", "name": "native_objective"},
                {
                    "op": "mul",
                    "args": [
                        {"op": "const", "value": 0.003},
                        {"op": "term", "name": "soft_rudy_mean"},
                    ],
                },
            ],
        },
    },
    "dreamplace_native_soft_rudy_pnorm": {
        "id": "",
        "rationale": (
            "Native-DREAMPlace residual with a small hotspot soft-RUDY p-norm "
            "correction. This keeps the adaptive native objective intact while "
            "nudging hotspot route pressure."
        ),
        "parent_ids": [],
        "declared_term_usage": ["native_objective", "soft_rudy_pnorm"],
        "ast": {
            "op": "add",
            "args": [
                {"op": "term", "name": "native_objective"},
                {
                    "op": "mul",
                    "args": [
                        {"op": "const", "value": 0.001},
                        {"op": "term", "name": "soft_rudy_pnorm"},
                    ],
                },
            ],
        },
    },
    "dreamplace_soft_rudy_mean": _single_term_payload(
        "soft_rudy_mean",
        "Single-term DREAMPlace soft-RUDY mean objective for promotion checks.",
    ),
    "dreamplace_soft_rudy_pnorm": _single_term_payload(
        "soft_rudy_pnorm",
        "Single-term DREAMPlace soft-RUDY p-norm objective for promotion checks.",
    ),
    "dreamplace_pin_density_pnorm": _single_term_payload(
        "pin_density_pnorm",
        "Single-term DREAMPlace pin-density p-norm objective for promotion checks.",
    ),
    "dreamplace_timing_weighted_wirelength": _single_term_payload(
        "timing_weighted_wirelength",
        "Legacy single-term DREAMPlace timing-weighted wirelength compatibility check.",
    ),
    "dreamplace_wirelength_wawl": _single_term_payload(
        "wirelength_wawl",
        "Single-term DREAMPlace weighted-average wirelength objective for promotion checks.",
    ),
    "dreamplace_wirelength_lse": _single_term_payload(
        "wirelength_lse",
        "Single-term DREAMPlace log-sum-exp wirelength objective for promotion checks.",
    ),
    "dreamplace_density_electric": _single_term_payload(
        "density_electric",
        "Single-term DREAMPlace electric-density objective for promotion checks.",
    ),
    "dreamplace_density_bell": _single_term_payload(
        "density_bell",
        "Single-term DREAMPlace bell-shaped density objective for promotion checks.",
    ),
    "dreamplace_route_pressure_long": _single_term_payload(
        "route_pressure_long",
        "Single-term DREAMPlace long-net route-pressure objective for promotion checks.",
    ),
    "dreamplace_pin_count_weighted_wl": _single_term_payload(
        "pin_count_weighted_wl",
        "Single-term DREAMPlace pin-count-weighted wirelength objective for promotion checks.",
    ),
    "dreamplace_routing_aware_smoke": {
        "id": "",
        "rationale": (
            "DREAMPlace-compatible routing-aware smoke objective over wirelength, "
            "density, route-demand pressure, and pin pressure."
        ),
        "parent_ids": [],
        "declared_term_usage": [
            "density_electric",
            "route_pressure_long",
            "soft_rudy_pnorm",
            "wirelength_wawl",
        ],
        "ast": {
            "op": "add",
            "args": [
                {"op": "term", "name": "wirelength_wawl"},
                {"op": "term", "name": "density_electric"},
                {
                    "op": "mul",
                    "args": [
                        {"op": "const", "value": 0.25},
                        {"op": "term", "name": "soft_rudy_pnorm"},
                    ],
                },
                {
                    "op": "mul",
                    "args": [
                        {"op": "const", "value": 0.05},
                        {"op": "term", "name": "route_pressure_long"},
                    ],
                },
            ],
        },
    },
}


def preset_names() -> list[str]:
    return sorted(PRESET_PAYLOADS)


def objective_preset(name: str, created_by: str = "preset") -> ObjectiveSpec:
    if name not in PRESET_PAYLOADS:
        raise KeyError(f"unknown objective preset: {name}")
    return parse_objective_spec(PRESET_PAYLOADS[name], created_by=created_by, term_scope="dreamplace")
