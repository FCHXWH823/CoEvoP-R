"""Scalar summary extraction from CircuitNet feature and label maps."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from coevop.datasets.circuitnet import CircuitNetManifest


STAT_NAMES = ("mean", "p95", "p99", "max", "sum", "nonzero_frac")


def load_npz_array(path: str | Path) -> np.ndarray:
    with np.load(Path(path)) as data:
        key = "data" if "data" in data.files else data.files[0]
        return np.asarray(data[key], dtype=np.float64)


def summarize_array(array: np.ndarray) -> dict[str, float]:
    flat = np.ravel(array).astype(np.float64, copy=False)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {name: math.nan for name in STAT_NAMES}

    return {
        "mean": float(np.mean(finite)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
        "sum": float(np.sum(finite)),
        "nonzero_frac": float(np.count_nonzero(finite) / finite.size),
    }


def _mean_present(row: dict[str, object], columns: Iterable[str]) -> float:
    values = []
    for column in columns:
        value = row.get(column)
        if isinstance(value, int | float) and math.isfinite(float(value)):
            values.append(float(value))
    return float(np.mean(values)) if values else math.nan


def _add_derived_targets(row: dict[str, object]) -> None:
    row["target.congestion_egr_overflow_mean"] = _mean_present(
        row,
        (
            "label.congestion_egr_horizontal_overflow.mean",
            "label.congestion_egr_vertical_overflow.mean",
        ),
    )
    row["target.congestion_egr_overflow_p95"] = _mean_present(
        row,
        (
            "label.congestion_egr_horizontal_overflow.p95",
            "label.congestion_egr_vertical_overflow.p95",
        ),
    )
    row["target.congestion_gr_overflow_mean"] = _mean_present(
        row,
        (
            "label.congestion_gr_horizontal_overflow.mean",
            "label.congestion_gr_vertical_overflow.mean",
        ),
    )
    row["target.congestion_gr_overflow_p95"] = _mean_present(
        row,
        (
            "label.congestion_gr_horizontal_overflow.p95",
            "label.congestion_gr_vertical_overflow.p95",
        ),
    )
    row["target.congestion_gr_util_p95"] = _mean_present(
        row,
        (
            "label.congestion_gr_horizontal_util.p95",
            "label.congestion_gr_vertical_util.p95",
        ),
    )
    row["target.drc_hotspot_mean"] = row.get("label.drc_all.mean", math.nan)
    row["target.drc_hotspot_sum"] = row.get("label.drc_all.sum", math.nan)


def compute_summary_rows(
    manifest: CircuitNetManifest,
    max_samples: int | None = None,
) -> list[dict[str, object]]:
    n14_root = Path(manifest.n14_root)
    rows: list[dict[str, object]] = []
    seen = 0

    for design in manifest.designs:
        for sample in design.samples:
            if max_samples is not None and seen >= max_samples:
                return rows
            row: dict[str, object] = {
                "sample_id": sample.sample_id,
                "design": sample.design,
                "family": design.family,
            }

            for source_name, relative_path in sample.features.items():
                stats = summarize_array(load_npz_array(n14_root / relative_path))
                for stat_name, value in stats.items():
                    row[f"feature.{source_name}.{stat_name}"] = value

            for source_name, relative_path in sample.labels.items():
                stats = summarize_array(load_npz_array(n14_root / relative_path))
                for stat_name, value in stats.items():
                    row[f"label.{source_name}.{stat_name}"] = value

            _add_derived_targets(row)
            rows.append(row)
            seen += 1

    return rows


def write_summary_csv(rows: list[dict[str, object]], output: str | Path) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _fieldnames(rows: list[dict[str, object]]) -> list[str]:
    metadata = ["sample_id", "design", "family"]
    keys = set()
    for row in rows:
        keys.update(row)
    return metadata + sorted(key for key in keys if key not in metadata)
