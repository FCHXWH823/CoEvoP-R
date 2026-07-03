import pytest

from coevop.objectives.program import (
    objective_program_schema_for_provider,
    parse_objective_program,
    parse_objective_program_payload,
)
from coevop.objectives.spec import ObjectiveSpecError


PROGRAM = """
def objective(features):
    rudy = term("rudy_p95")
    density = term("cell_density_p95")
    score = rudy + 0.35 * density
    return score, {"rudy_hotspot": rudy, "density_hotspot": density}
"""


def test_parse_objective_program_lowers_to_safe_ast() -> None:
    spec = parse_objective_program(
        PROGRAM,
        created_by="test",
        rationale="Eureka-style restricted program.",
        term_scope="tier1",
    )

    assert spec.id.startswith("obj_")
    assert spec.term_set == ["cell_density_p95", "rudy_p95"]
    assert spec.source_program is not None
    assert set(spec.components or {}) == {"density_hotspot", "rudy_hotspot"}
    assert spec.ast == {
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
    }


def test_parse_objective_program_payload_checks_declared_terms() -> None:
    payload = {
        "id": "",
        "rationale": "Mismatch should be rejected.",
        "parent_ids": [],
        "declared_term_usage": ["rudy_p95"],
        "objective_program": PROGRAM,
    }

    with pytest.raises(ObjectiveSpecError, match="declared_term_usage"):
        parse_objective_program_payload(payload, created_by="test")


def test_objective_program_payload_normalizes_provider_newline_artifacts() -> None:
    spec = parse_objective_program_payload(
        {
            "id": "",
            "rationale": "Provider formatting artifact.",
            "parent_ids": [],
            "declared_term_usage": ["soft_rudy_mean", "soft_rudy_pnorm"],
            "objective_program": (
                'def objective(features):\u00a0 \\(n)'
                '    hot = term("soft_rudy_pnorm")\u00a0 \\(n)'
                '    mean = term("soft_rudy_mean")\u00a0 \\(n)'
                "    score = hot + 0.2 * mean\u00a0 \\(n)"
                '    return score, {"hot": hot, "mean": mean}'
            ),
        },
        created_by="test",
        term_scope="deployable",
    )

    assert spec.term_set == ["soft_rudy_mean", "soft_rudy_pnorm"]


def test_objective_program_payload_normalizes_literal_carriage_returns() -> None:
    spec = parse_objective_program_payload(
        {
            "id": "",
            "rationale": "Provider literal carriage returns.",
            "parent_ids": [],
            "declared_term_usage": ["pin_density_pnorm", "soft_rudy_pnorm"],
            "objective_program": (
                'def objective(features):\\r'
                '    route = term("soft_rudy_pnorm")\\r'
                '    pins = term("pin_density_pnorm")\\r'
                "    score = route + 0.25 * pins\\r"
                '    return score, {"route": route, "pins": pins}'
            ),
        },
        created_by="test",
        term_scope="deployable",
    )

    assert spec.term_set == ["pin_density_pnorm", "soft_rudy_pnorm"]


def test_objective_program_payload_normalizes_provider_line_markers() -> None:
    spec = parse_objective_program_payload(
        {
            "id": "",
            "rationale": "Provider line markers.",
            "parent_ids": [],
            "declared_term_usage": [
                "pin_density_pnorm",
                "soft_rudy_mean",
                "soft_rudy_pnorm",
            ],
            "objective_program": (
                'def objective(features):\\\n'
                '    route = term("soft_rudy_pnorm")\\\n'
                '\\    mean0 = term("soft_rudy_mean")\\\n'
                '\\$    mean = term("soft_rudy_mean")\\\n'
                '\\$    pins = term("pin_density_pnorm")\\\n'
                "    score = route + 0.1 * mean0 + 0.2 * mean + 0.25 * pins \\[\\]\n"
                '    return score, {"route": route, "mean": mean, "pins": pins}'
            ),
        },
        created_by="test",
        term_scope="deployable",
    )

    assert spec.term_set == [
        "pin_density_pnorm",
        "soft_rudy_mean",
        "soft_rudy_pnorm",
    ]


def test_objective_program_rejects_arbitrary_python() -> None:
    source = """
def objective(features):
    import os
    return term("rudy_p95")
"""

    with pytest.raises(ObjectiveSpecError, match="unsupported statement"):
        parse_objective_program(source, created_by="test")


def test_objective_program_rejects_code_after_return() -> None:
    source = """
def objective(features):
    rudy = term("rudy_p95")
    return rudy
    density = term("cell_density_p95")
"""

    with pytest.raises(ObjectiveSpecError, match="return"):
        parse_objective_program(source, created_by="test")


def test_objective_program_rejects_too_many_components() -> None:
    source = """
def objective(features):
    rudy = term("rudy_p95")
    return rudy, {
        "c0": rudy,
        "c1": rudy,
        "c2": rudy,
        "c3": rudy,
        "c4": rudy,
        "c5": rudy,
        "c6": rudy,
    }
"""

    with pytest.raises(ObjectiveSpecError, match="components exceed limit"):
        parse_objective_program(source, created_by="test")


def test_objective_program_rejects_oversized_component_ast() -> None:
    oversized_expr = " + ".join(["rudy"] * 13)
    source = f"""
def objective(features):
    rudy = term("rudy_p95")
    too_big = {oversized_expr}
    return rudy, {{"too_big": too_big}}
"""

    with pytest.raises(ObjectiveSpecError, match="component AST primitive count"):
        parse_objective_program(source, created_by="test")


def test_objective_program_rejects_invalid_component_constant() -> None:
    source = """
def objective(features):
    rudy = term("rudy_p95")
    return rudy, {"bad_scale": 1e999 * rudy}
"""

    with pytest.raises(ObjectiveSpecError, match="component constant magnitude"):
        parse_objective_program(source, created_by="test")


def test_objective_program_rejects_direct_feature_access() -> None:
    source = """
def objective(features):
    score = features["rudy_p95"]
    return score
"""

    with pytest.raises(ObjectiveSpecError):
        parse_objective_program(source, created_by="test")


def test_objective_program_rejects_unknown_variable() -> None:
    source = """
def objective(features):
    score = missing + term("rudy_p95")
    return score
"""

    with pytest.raises(ObjectiveSpecError, match="unknown local variable"):
        parse_objective_program(source, created_by="test")


def test_objective_program_rejects_invalid_call() -> None:
    source = """
def objective(features):
    score = abs(term("rudy_p95"))
    return score
"""

    with pytest.raises(ObjectiveSpecError, match="unsupported objective helper"):
        parse_objective_program(source, created_by="test")


def test_program_schema_exposes_only_scoped_terms() -> None:
    dreamplace_schema = objective_program_schema_for_provider("dreamplace")
    dreamplace_terms = dreamplace_schema["properties"]["declared_term_usage"]["items"]["enum"]
    replacement_schema = objective_program_schema_for_provider("dreamplace_replacement")
    replacement_terms = replacement_schema["properties"]["declared_term_usage"]["items"]["enum"]
    deployable_schema = objective_program_schema_for_provider("deployable")
    deployable_terms = deployable_schema["properties"]["declared_term_usage"]["items"]["enum"]

    assert "wirelength" in dreamplace_terms
    assert "density" in dreamplace_terms
    assert "soft_rudy_pnorm" in dreamplace_terms
    assert "pin_density_pnorm" in dreamplace_terms
    assert "native_objective" in dreamplace_terms
    assert "timing_weighted_wirelength" in dreamplace_terms
    assert "native_objective" not in replacement_terms
    assert "timing_weighted_wirelength" not in replacement_terms
    assert "wirelength_lse" in replacement_terms
    assert "density_bell" in replacement_terms
    assert "route_pressure_long" in replacement_terms
    assert "pin_count_weighted_wl" in replacement_terms
    assert "wirelength" not in deployable_terms
    assert "density_pnorm" not in dreamplace_terms
    assert "density_pnorm" not in replacement_terms
    assert deployable_terms == [
        "pin_density_pnorm",
        "soft_rudy_mean",
        "soft_rudy_pnorm",
    ]
