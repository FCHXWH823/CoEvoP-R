import pytest

from coevop.objectives.spec import MAX_AST_DEPTH, ObjectiveSpecError, parse_objective_spec
from coevop.objectives.terms import term_names


VALID_PAYLOAD = {
    "id": "",
    "rationale": "Smooth wirelength with density pressure.",
    "parent_ids": [],
    "declared_term_usage": ["density_electric", "wirelength_wawl"],
    "ast": {
        "op": "add",
        "args": [
            {"op": "term", "name": "wirelength_wawl"},
            {
                "op": "mul",
                "args": [
                    {"op": "const", "value": 0.35},
                    {"op": "term", "name": "density_electric"},
                ],
            },
        ],
    },
}


def _parse(payload: dict):
    return parse_objective_spec(
        payload,
        created_by="test",
        term_scope="dreamplace_controller",
    )


def test_parse_objective_spec_computes_stable_id_and_terms() -> None:
    spec = _parse(VALID_PAYLOAD)
    same_spec = _parse(VALID_PAYLOAD)

    assert spec.id == same_spec.id
    assert spec.id.startswith("obj_")
    assert spec.term_set == ["density_electric", "wirelength_wawl"]
    assert spec.complexity == 5
    assert spec.constants == {"c0": 0.35}


def test_parse_objective_spec_rejects_unknown_term() -> None:
    payload = dict(VALID_PAYLOAD)
    payload["declared_term_usage"] = ["made_up_term"]
    payload["ast"] = {"op": "term", "name": "made_up_term"}

    with pytest.raises(ObjectiveSpecError):
        _parse(payload)


def test_parse_objective_spec_accepts_depth_ten_and_rejects_depth_eleven() -> None:
    assert MAX_AST_DEPTH == 10

    def nested_ast(depth: int) -> dict:
        node = {"op": "term", "name": "soft_rudy_pnorm"}
        for _ in range(depth - 1):
            node = {"op": "log1p", "args": [node]}
        return node

    payload = {
        "id": "",
        "rationale": "Exercise the AST depth boundary.",
        "parent_ids": [],
        "declared_term_usage": ["soft_rudy_pnorm"],
        "ast": nested_ast(10),
    }
    _parse(payload)

    payload["ast"] = nested_ast(11)
    with pytest.raises(ObjectiveSpecError, match="AST depth exceeds 10"):
        _parse(payload)


def test_controller_scope_contains_physical_placement_terms() -> None:
    terms = set(term_names("dreamplace_controller"))

    assert "wirelength_wawl" in terms
    assert "density_electric" in terms
    assert "soft_rudy_pnorm" in terms
    assert "pin_density_pnorm" in terms


def test_parse_objective_spec_rejects_declared_term_mismatch() -> None:
    payload = dict(VALID_PAYLOAD)
    payload["declared_term_usage"] = ["wirelength_wawl"]

    with pytest.raises(ObjectiveSpecError):
        _parse(payload)
