import math

from coevop.eval.tier1 import DEFAULT_TARGETS, evaluate_spec
from coevop.objectives.fitting import fit_objective_constants
from coevop.objectives.program import parse_objective_program


def _rows(start: int, stop: int) -> list[dict[str, object]]:
    rows = []
    for value in range(start, stop):
        density = float(value)
        rudy = float(stop - value)
        row = {
            "feature.rudy.p95": rudy,
            "feature.cell_density.p95": density,
        }
        for target in DEFAULT_TARGETS:
            row[target] = density
        rows.append(row)
    return rows


def test_fit_objective_constants_creates_validation_preserving_sibling() -> None:
    spec = parse_objective_program(
        """
def objective(features):
    rudy = term("rudy_p95")
    density = term("cell_density_p95")
    score = 0.1 * rudy + 0.1 * density
    return score, {"rudy": rudy, "density": density}
""",
        created_by="test",
    )
    train_rows = _rows(1, 9)
    validation_rows = _rows(9, 13)

    fitted, report = fit_objective_constants(
        spec,
        train_rows=train_rows,
        validation_rows=validation_rows,
    )

    assert fitted is not None
    assert fitted.id != spec.id
    assert fitted.parent_ids == [spec.id]
    assert report["status"] == "fitted"
    assert report["constants_changed"]
    before = evaluate_spec(validation_rows, spec).score
    after = evaluate_spec(validation_rows, fitted).score
    assert math.isnan(before) or after >= before
    assert after > 0.99
