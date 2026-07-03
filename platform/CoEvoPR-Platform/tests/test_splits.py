import pytest

from coevop.datasets.splits import selected_rows, split_rows, split_rows_by_design_holdout


def _rows(families: list[str]) -> list[dict[str, object]]:
    return [
        {
            "sample_id": f"sample_{index}",
            "design": f"{family}-small",
            "family": family,
        }
        for index, family in enumerate(families)
    ]


def test_auto_split_prefers_family_when_enough_families() -> None:
    rows = _rows(["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"])

    split_set = split_rows(rows, strategy="auto", seed=1)

    assert split_set.strategy == "family"
    assert split_set.group_field == "family"
    assert split_set.row_counts["train"] > 0
    assert split_set.row_counts["validation"] > 0
    assert split_set.row_counts["heldout"] > 0


def test_auto_split_falls_back_to_sample_for_single_design() -> None:
    rows = _rows(["vortex"] * 12)

    split_set = split_rows(rows, strategy="auto", seed=0)

    assert split_set.strategy == "sample"
    assert split_set.row_counts == {"train": 8, "validation": 2, "heldout": 2}
    assert selected_rows(split_set, "validation") == split_set.splits["validation"].rows


def test_none_split_uses_all_rows_for_train() -> None:
    rows = _rows(["vortex"] * 4)

    split_set = split_rows(rows, strategy="none")

    assert split_set.row_counts == {"train": 4, "validation": 0, "heldout": 0}
    assert selected_rows(split_set, "validation") == rows


def test_split_rows_by_design_holdout_for_leave_one_design_out() -> None:
    rows = [
        {"sample_id": "a0", "design": "A"},
        {"sample_id": "a1", "design": "A"},
        {"sample_id": "b0", "design": "B"},
        {"sample_id": "c0", "design": "C"},
    ]

    split_set = split_rows_by_design_holdout(rows, validation_design="C")

    assert split_set.strategy == "fixed_design"
    assert split_set.row_counts == {"train": 3, "validation": 1, "heldout": 0}
    assert split_set.splits["train"].group_keys == ["A", "B"]
    assert split_set.splits["validation"].group_keys == ["C"]


def test_split_rows_by_design_holdout_rejects_missing_design() -> None:
    rows = [{"sample_id": "a0", "design": "A"}]

    with pytest.raises(ValueError, match="validation design not found"):
        split_rows_by_design_holdout(rows, validation_design="missing")
