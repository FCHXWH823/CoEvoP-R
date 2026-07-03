"""Deterministic train/validation/held-out splits for CircuitNet summaries."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable


SPLIT_NAMES = ("train", "validation", "heldout")


@dataclass(frozen=True)
class RowSplit:
    name: str
    rows: list[dict[str, object]]
    group_keys: list[str]


@dataclass(frozen=True)
class SplitSet:
    strategy: str
    group_field: str
    seed: int
    splits: dict[str, RowSplit]

    @property
    def row_counts(self) -> dict[str, int]:
        return {name: len(split.rows) for name, split in self.splits.items()}

    @property
    def group_counts(self) -> dict[str, int]:
        return {name: len(split.group_keys) for name, split in self.splits.items()}


def split_rows(
    rows: list[dict[str, object]],
    *,
    strategy: str = "auto",
    seed: int = 0,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    heldout_fraction: float = 0.2,
) -> SplitSet:
    if strategy not in {"auto", "family", "design", "sample", "none"}:
        raise ValueError(f"unknown split strategy: {strategy}")
    if not rows:
        return _empty_split(strategy=strategy, group_field="none", seed=seed)

    resolved_strategy = _resolve_strategy(rows, strategy)
    if resolved_strategy == "none":
        return SplitSet(
            strategy="none",
            group_field="none",
            seed=seed,
            splits={
                "train": RowSplit("train", list(rows), ["all"]),
                "validation": RowSplit("validation", [], []),
                "heldout": RowSplit("heldout", [], []),
            },
        )

    group_field = {
        "family": "family",
        "design": "design",
        "sample": "sample_id",
    }[resolved_strategy]
    grouped = _group_rows(rows, group_field)

    if resolved_strategy != "sample" and len(grouped) < 3:
        return split_rows(
            rows,
            strategy="sample",
            seed=seed,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            heldout_fraction=heldout_fraction,
        )

    rng = random.Random(seed)
    group_keys = sorted(grouped)
    rng.shuffle(group_keys)

    counts = _split_counts(
        len(group_keys),
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        heldout_fraction=heldout_fraction,
    )
    train_keys = group_keys[: counts["train"]]
    validation_keys = group_keys[counts["train"] : counts["train"] + counts["validation"]]
    heldout_keys = group_keys[counts["train"] + counts["validation"] :]
    key_by_split = {
        "train": train_keys,
        "validation": validation_keys,
        "heldout": heldout_keys,
    }
    splits = {
        name: RowSplit(name, _rows_for_keys(grouped, keys), list(keys))
        for name, keys in key_by_split.items()
    }
    return SplitSet(
        strategy=resolved_strategy,
        group_field=group_field,
        seed=seed,
        splits=splits,
    )


def selected_rows(split_set: SplitSet, split_name: str) -> list[dict[str, object]]:
    if split_name not in split_set.splits:
        raise ValueError(f"unknown split: {split_name}")
    rows = split_set.splits[split_name].rows
    if rows:
        return rows
    return split_set.splits["train"].rows


def split_rows_by_design_holdout(
    rows: list[dict[str, object]],
    *,
    validation_design: str | None = None,
    heldout_design: str | None = None,
    seed: int = 0,
) -> SplitSet:
    validation_design = validation_design.strip() if validation_design else None
    heldout_design = heldout_design.strip() if heldout_design else None
    if validation_design is None and heldout_design is None:
        raise ValueError("validation_design or heldout_design is required")
    if validation_design and heldout_design and validation_design == heldout_design:
        raise ValueError("validation_design and heldout_design must differ")

    train_rows = []
    validation_rows = []
    heldout_rows = []
    for row in rows:
        design = str(row.get("design", "")).strip()
        if validation_design and design == validation_design:
            validation_rows.append(row)
        elif heldout_design and design == heldout_design:
            heldout_rows.append(row)
        else:
            train_rows.append(row)

    if validation_design and not validation_rows:
        raise ValueError(f"validation design not found in rows: {validation_design}")
    if heldout_design and not heldout_rows:
        raise ValueError(f"heldout design not found in rows: {heldout_design}")
    if not train_rows:
        raise ValueError("fixed design split leaves no train rows")

    return SplitSet(
        strategy="fixed_design",
        group_field="design",
        seed=seed,
        splits={
            "train": RowSplit("train", train_rows, sorted(_unique_values(train_rows, "design"))),
            "validation": RowSplit(
                "validation",
                validation_rows,
                sorted(_unique_values(validation_rows, "design")),
            ),
            "heldout": RowSplit(
                "heldout",
                heldout_rows,
                sorted(_unique_values(heldout_rows, "design")),
            ),
        },
    )


def _empty_split(strategy: str, group_field: str, seed: int) -> SplitSet:
    return SplitSet(
        strategy=strategy,
        group_field=group_field,
        seed=seed,
        splits={name: RowSplit(name, [], []) for name in SPLIT_NAMES},
    )


def _resolve_strategy(rows: list[dict[str, object]], strategy: str) -> str:
    if strategy != "auto":
        return strategy
    if len(_unique_values(rows, "family")) >= 3:
        return "family"
    if len(_unique_values(rows, "design")) >= 3:
        return "design"
    if len(rows) >= 3:
        return "sample"
    return "none"


def _unique_values(rows: list[dict[str, object]], field: str) -> set[str]:
    return {str(row.get(field, "")) for row in rows if str(row.get(field, "")).strip()}


def _group_rows(rows: Iterable[dict[str, object]], field: str) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for index, row in enumerate(rows):
        key = str(row.get(field, "")).strip() or f"row_{index}"
        grouped.setdefault(key, []).append(row)
    return grouped


def _split_counts(
    group_count: int,
    *,
    train_fraction: float,
    validation_fraction: float,
    heldout_fraction: float,
) -> dict[str, int]:
    if group_count <= 0:
        return {"train": 0, "validation": 0, "heldout": 0}
    if group_count == 1:
        return {"train": 1, "validation": 0, "heldout": 0}
    if group_count == 2:
        return {"train": 1, "validation": 1, "heldout": 0}

    fractions = _normalized_fractions(train_fraction, validation_fraction, heldout_fraction)
    validation = max(1, round(group_count * fractions["validation"]))
    heldout = max(1, round(group_count * fractions["heldout"]))
    train = group_count - validation - heldout
    if train < 1:
        train = 1
        overflow = train + validation + heldout - group_count
        while overflow > 0 and (validation > 1 or heldout > 1):
            if validation >= heldout and validation > 1:
                validation -= 1
            elif heldout > 1:
                heldout -= 1
            overflow -= 1
    return {"train": train, "validation": validation, "heldout": group_count - train - validation}


def _normalized_fractions(
    train_fraction: float,
    validation_fraction: float,
    heldout_fraction: float,
) -> dict[str, float]:
    values = {
        "train": max(0.0, float(train_fraction)),
        "validation": max(0.0, float(validation_fraction)),
        "heldout": max(0.0, float(heldout_fraction)),
    }
    total = sum(values.values())
    if total <= 0.0:
        return {"train": 0.6, "validation": 0.2, "heldout": 0.2}
    return {name: value / total for name, value in values.items()}


def _rows_for_keys(
    grouped: dict[str, list[dict[str, object]]],
    keys: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in keys:
        rows.extend(grouped[key])
    return rows
