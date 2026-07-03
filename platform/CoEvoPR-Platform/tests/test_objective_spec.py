import numpy as np
import pytest

from coevop.objectives.dsl import score_spec
from coevop.objectives.spec import ObjectiveSpecError, objective_schema_for_provider, parse_objective_spec
from coevop.objectives.terms import term_names


VALID_PAYLOAD = {
    "id": "",
    "rationale": "RUDY hotspot with a density hotspot correction.",
    "parent_ids": [],
    "declared_term_usage": ["cell_density_p95", "rudy_p95"],
    "ast": {
        "op": "add",
        "args": [
            {"op": "term", "name": "rudy_p95"},
            {
                "op": "mul",
                "args": [
                    {"op": "const", "value": 0.35},
                    {"op": "term", "name": "cell_density_p95"},
                ],
            },
        ],
    },
}


def test_parse_objective_spec_computes_stable_id_and_terms() -> None:
    spec = parse_objective_spec(VALID_PAYLOAD, created_by="test")
    same_spec = parse_objective_spec(VALID_PAYLOAD, created_by="test")

    assert spec.id == same_spec.id
    assert spec.id.startswith("obj_")
    assert spec.term_set == ["cell_density_p95", "rudy_p95"]
    assert spec.complexity == 5
    assert spec.constants == {"c0": 0.35}


def test_parse_objective_spec_rejects_unknown_term() -> None:
    payload = dict(VALID_PAYLOAD)
    payload["declared_term_usage"] = ["made_up_term"]
    payload["ast"] = {"op": "term", "name": "made_up_term"}

    with pytest.raises(ObjectiveSpecError):
        parse_objective_spec(payload, created_by="test")


def test_tier1_schema_does_not_expose_misleading_hpwl_aliases() -> None:
    tier1_terms = term_names("tier1")
    schema = objective_schema_for_provider("tier1")

    assert "smooth_hpwl_proxy" not in tier1_terms
    assert "fanout_weighted_hpwl_proxy" not in tier1_terms
    assert "wirelength" not in tier1_terms
    assert "rudy_p95" in schema["properties"]["declared_term_usage"]["items"]["enum"]


def test_dreamplace_scope_rejects_tier1_only_terms() -> None:
    payload = {
        "id": "",
        "rationale": "RUDY is not currently deployable inside DREAMPlace.",
        "parent_ids": [],
        "declared_term_usage": ["rudy_p95"],
        "ast": {"op": "term", "name": "rudy_p95"},
    }

    with pytest.raises(ObjectiveSpecError):
        parse_objective_spec(payload, created_by="test", term_scope="dreamplace")


def test_dreamplace_scope_accepts_only_deployable_terms() -> None:
    payload = {
        "id": "",
        "rationale": "Deployable DREAMPlace objective.",
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

    spec = parse_objective_spec(payload, created_by="test", term_scope="dreamplace")

    assert spec.term_set == ["density", "wirelength"]


def test_dreamplace_replacement_scope_excludes_legacy_or_deferred_terms() -> None:
    replacement_terms = term_names("dreamplace_replacement")

    assert "native_objective" not in replacement_terms
    assert "density_pnorm" not in replacement_terms
    assert "timing_weighted_wirelength" not in replacement_terms
    assert "wirelength_wawl" in replacement_terms
    assert "wirelength_lse" in replacement_terms
    assert "density_electric" in replacement_terms
    assert "density_bell" in replacement_terms
    assert "route_pressure_long" in replacement_terms
    assert "pin_count_weighted_wl" in replacement_terms


def test_internal_dreamplace_scope_still_loads_legacy_native_objective() -> None:
    payload = {
        "id": "",
        "rationale": "Legacy native DREAMPlace objective residual base.",
        "parent_ids": [],
        "declared_term_usage": ["native_objective", "soft_rudy_pnorm"],
        "ast": {
            "op": "add",
            "args": [
                {"op": "term", "name": "native_objective"},
                {
                    "op": "mul",
                    "args": [
                        {"op": "const", "value": 0.003},
                        {"op": "term", "name": "soft_rudy_pnorm"},
                    ],
                },
            ],
        },
    }

    spec = parse_objective_spec(payload, created_by="test", term_scope="dreamplace")

    assert spec.term_set == ["native_objective", "soft_rudy_pnorm"]


def test_deployable_scope_is_gradient_promoted_bridge_only() -> None:
    assert term_names("deployable") == [
        "pin_density_pnorm",
        "soft_rudy_mean",
        "soft_rudy_pnorm",
    ]
    assert "density_pnorm" in term_names("tier1")
    assert "density_pnorm" not in term_names("dreamplace")


def test_parse_objective_spec_rejects_declared_term_mismatch() -> None:
    payload = dict(VALID_PAYLOAD)
    payload["declared_term_usage"] = ["rudy_p95"]

    with pytest.raises(ObjectiveSpecError):
        parse_objective_spec(payload, created_by="test")


def test_score_spec_is_vectorized_and_finite() -> None:
    spec = parse_objective_spec(VALID_PAYLOAD, created_by="test")
    rows = [
        {"feature.rudy.p95": 1.0, "feature.cell_density.p95": 4.0},
        {"feature.rudy.p95": 2.0, "feature.cell_density.p95": 3.0},
        {"feature.rudy.p95": 3.0, "feature.cell_density.p95": 2.0},
        {"feature.rudy.p95": 4.0, "feature.cell_density.p95": 1.0},
    ]

    scores = score_spec(rows, spec)

    assert scores.shape == (4,)
    assert np.all(np.isfinite(scores))
    assert scores[-1] > scores[0]
