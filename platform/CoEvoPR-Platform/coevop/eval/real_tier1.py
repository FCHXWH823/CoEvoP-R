"""Real-scale Tier-1 audit and negative controls for CircuitNet runs."""

from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from coevop.datasets.circuitnet import (
    ROUTABILITY_FEATURE_DIRS,
    ROUTABILITY_LABEL_DIRS,
    CircuitNetManifest,
    load_manifest,
)
from coevop.datasets.splits import SplitSet, split_rows, split_rows_by_design_holdout
from coevop.datasets.summaries import compute_summary_rows, write_summary_csv
from coevop.eval.tier1 import DEFAULT_TARGETS, evaluate_spec, rank_objectives
from coevop.objectives.baselines import BASELINE_OBJECTIVES
from coevop.objectives.spec import ObjectiveSpec, parse_objective_spec
from coevop.objectives.terms import term_names


def write_real_tier1_audit(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    max_samples: int | None = None,
    split_strategy: str = "auto",
    seed: int = 0,
    shuffle_trials: int = 32,
    random_formula_count: int = 32,
    evolution_dir: str | Path | None = None,
    validation_design: str | None = None,
    heldout_design: str | None = None,
) -> dict[str, Any]:
    """Write audit/control artifacts for one real Tier-1 run directory."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    rows = compute_summary_rows(manifest, max_samples=max_samples)
    if validation_design or heldout_design:
        split_set = split_rows_by_design_holdout(
            rows,
            validation_design=validation_design,
            heldout_design=heldout_design,
            seed=seed,
        )
    else:
        split_set = split_rows(rows, strategy=split_strategy, seed=seed)

    manifest_copy = output / "manifest.json"
    shutil.copyfile(manifest_path, manifest_copy)
    scalars_path = write_summary_csv(rows, output / "scalars.csv")

    label_audit = build_label_audit(manifest, rows)
    label_audit_path = _write_json(label_audit, output / "label_audit.json")

    baseline_payload = baseline_rankings_by_split(split_set)
    baseline_path = _write_json(baseline_payload, output / "baseline_rankings_by_split.json")

    controls = negative_controls(
        rows,
        seed=seed,
        shuffle_trials=shuffle_trials,
        random_formula_count=random_formula_count,
    )
    controls_path = _write_json(controls, output / "negative_controls.json")

    report = build_real_tier1_report(
        label_audit=label_audit,
        baseline_payload=baseline_payload,
        controls=controls,
        evolution_dir=Path(evolution_dir) if evolution_dir else output,
    )
    report_path = output / "real_tier1_report.md"
    report_path.write_text(report, encoding="utf-8")

    return {
        "output_dir": str(output),
        "manifest": str(manifest_copy),
        "scalars": str(scalars_path),
        "label_audit": str(label_audit_path),
        "baseline_rankings_by_split": str(baseline_path),
        "negative_controls": str(controls_path),
        "real_tier1_report": str(report_path),
        "row_count": len(rows),
        "design_count": len(manifest.designs),
    }


def build_label_audit(
    manifest: CircuitNetManifest,
    rows: list[dict[str, object]],
) -> dict[str, Any]:
    by_design_rows: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_design_rows.setdefault(str(row.get("design", "")), []).append(row)

    design_payloads = []
    for design in manifest.designs:
        design_rows = by_design_rows.get(design.name, [])
        feature_missing = {
            name: design.sample_count - sum(1 for sample in design.samples if name in sample.features)
            for name in ROUTABILITY_FEATURE_DIRS
        }
        label_missing = {
            name: design.sample_count - sum(1 for sample in design.samples if name in sample.labels)
            for name in ROUTABILITY_LABEL_DIRS
        }
        design_payloads.append(
            {
                "design": design.name,
                "family": design.family,
                "manifest_sample_count": design.sample_count,
                "summary_row_count": len(design_rows),
                "missing_features": feature_missing,
                "missing_labels": label_missing,
                "target_distributions": {
                    target: _distribution(design_rows, target) for target in DEFAULT_TARGETS
                },
            }
        )

    warnings = []
    for design in design_payloads:
        for source, missing_count in design["missing_labels"].items():
            if missing_count:
                warnings.append(
                    f"{design['design']} is missing {missing_count} label maps for {source}"
                )
        for target, stats in design["target_distributions"].items():
            for flag in stats["flags"]:
                warnings.append(f"{design['design']} {target}: {flag}")

    overall_targets = {target: _distribution(rows, target) for target in DEFAULT_TARGETS}
    for target, stats in overall_targets.items():
        for flag in stats["flags"]:
            warnings.append(f"overall {target}: {flag}")

    return {
        "row_count": len(rows),
        "design_count": len(manifest.designs),
        "total_manifest_samples": sum(design.sample_count for design in manifest.designs),
        "designs": design_payloads,
        "overall_target_distributions": overall_targets,
        "warnings": warnings,
    }


def baseline_rankings_by_split(split_set: SplitSet) -> dict[str, Any]:
    return {
        "split_strategy": split_set.strategy,
        "group_field": split_set.group_field,
        "row_counts": split_set.row_counts,
        "group_counts": split_set.group_counts,
        "rankings_by_split": {
            name: [asdict(ranking) for ranking in rank_objectives(split.rows)]
            if split.rows
            else []
            for name, split in split_set.splits.items()
        },
        "within_design_rankings": _within_design_rankings(
            [row for split in split_set.splits.values() for row in split.rows]
        ),
    }


def negative_controls(
    rows: list[dict[str, object]],
    *,
    seed: int = 0,
    shuffle_trials: int = 32,
    random_formula_count: int = 32,
) -> dict[str, Any]:
    rng = random.Random(seed)
    shuffled = _shuffled_label_controls(rows, rng=rng, trials=shuffle_trials)
    random_formulas = _random_formula_controls(rows, rng=rng, count=random_formula_count)
    return {
        "seed": seed,
        "shuffle_trials": shuffle_trials,
        "random_formula_count": random_formula_count,
        "shuffled_label": shuffled,
        "random_formula": random_formulas,
    }


def build_real_tier1_report(
    *,
    label_audit: dict[str, Any],
    baseline_payload: dict[str, Any],
    controls: dict[str, Any],
    evolution_dir: Path,
) -> str:
    rankings = baseline_payload.get("rankings_by_split", {})
    validation_rankings = rankings.get("validation") or rankings.get("train") or []
    best_baseline = validation_rankings[0] if validation_rankings else None
    top_candidates = _load_top_candidates(evolution_dir)
    best_candidate = top_candidates[0] if top_candidates else None
    warning_lines = label_audit.get("warnings", [])[:12]

    lines = [
        "# Real Tier-1 CircuitNet Report",
        "",
        "## Dataset",
        f"- Designs: {label_audit.get('design_count')}",
        f"- Rows: {label_audit.get('row_count')}",
        f"- Manifest samples: {label_audit.get('total_manifest_samples')}",
        "",
        "## Label Audit",
    ]
    if warning_lines:
        lines.extend(f"- WARNING: {warning}" for warning in warning_lines)
        if len(label_audit.get("warnings", [])) > len(warning_lines):
            lines.append(f"- Additional warnings: {len(label_audit['warnings']) - len(warning_lines)}")
    else:
        lines.append("- No missing-label or target-distribution warnings.")

    lines.extend(
        [
            "",
            "## Baseline Sanity",
            (
                f"- Best validation baseline: {best_baseline['objective_id']} "
                f"score={best_baseline['score']:.6g}"
                if best_baseline
                else "- No finite validation baseline."
            ),
            f"- Split row counts: {baseline_payload.get('row_counts')}",
            "",
            "## Negative Controls",
            (
                "- Shuffled-label best-score max: "
                f"{controls['shuffled_label']['best_score_summary']['max']:.6g}"
            ),
            (
                "- Random-formula best score: "
                f"{controls['random_formula']['best_score']:.6g}"
            ),
            "",
            "## Evolution Summary",
        ]
    )
    if best_candidate:
        metrics = best_candidate.get("metrics", {})
        lines.extend(
            [
                f"- Best generated candidate: {best_candidate['objective_id']}",
                f"- Candidate score: {best_candidate.get('score')}",
                f"- Candidate terms: {','.join(best_candidate.get('term_set', []))}",
                f"- DREAMPlace deployable: {metrics.get('dreamplace_deployable')}",
                f"- Operator: {metrics.get('operator')}",
            ]
        )
    else:
        lines.append("- No generated top-candidate artifact found yet.")

    lines.extend(
        [
            "",
            "## Tier-2 Gate",
            "- Treat this report as proxy evidence only.",
            "- Move to DREAMPlace/OpenROAD only after baselines, controls, and heldout behavior are sane.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _distribution(rows: list[dict[str, object]], column: str) -> dict[str, Any]:
    values = []
    for row in rows:
        try:
            values.append(float(row.get(column, math.nan)))
        except (TypeError, ValueError):
            values.append(math.nan)
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    flags = []
    if finite.size == 0:
        flags.append("all_nan")
        return {
            "finite_count": 0,
            "nan_count": int(array.size),
            "mean": None,
            "std": None,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
            "nonzero_frac": None,
            "flags": flags,
        }
    std = float(np.std(finite))
    nonzero_frac = float(np.count_nonzero(finite) / finite.size)
    if std < 1e-12:
        flags.append("near_constant")
    if nonzero_frac < 0.01:
        flags.append("sparse_nonzero")
    return {
        "finite_count": int(finite.size),
        "nan_count": int(array.size - finite.size),
        "mean": float(np.mean(finite)),
        "std": std,
        "min": float(np.min(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
        "nonzero_frac": nonzero_frac,
        "flags": flags,
    }


def _within_design_rankings(rows: list[dict[str, object]]) -> dict[str, Any]:
    by_design: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_design.setdefault(str(row.get("design", "")), []).append(row)
    return {
        design: [asdict(ranking) for ranking in rank_objectives(design_rows)]
        for design, design_rows in sorted(by_design.items())
        if design_rows
    }


def _shuffled_label_controls(
    rows: list[dict[str, object]],
    *,
    rng: random.Random,
    trials: int,
) -> dict[str, Any]:
    trial_rows = []
    best_scores = []
    for trial in range(trials):
        copied = [dict(row) for row in rows]
        for target in DEFAULT_TARGETS:
            values = [row.get(target, math.nan) for row in copied]
            rng.shuffle(values)
            for row, value in zip(copied, values):
                row[target] = value
        rankings = rank_objectives(copied)
        best = rankings[0] if rankings else None
        best_score = best.score if best is not None else math.nan
        if math.isfinite(best_score):
            best_scores.append(float(best_score))
        trial_rows.append(
            {
                "trial": trial,
                "best_objective_id": best.objective_id if best else None,
                "best_score": best_score,
            }
        )
    return {
        "trials": trial_rows,
        "best_score_summary": _numeric_summary(best_scores),
    }


def _random_formula_controls(
    rows: list[dict[str, object]],
    *,
    rng: random.Random,
    count: int,
) -> dict[str, Any]:
    candidates = []
    for index in range(count):
        spec = _random_formula(rng, index)
        ranking = evaluate_spec(rows, spec)
        candidates.append(
            {
                "objective_id": spec.id,
                "score": ranking.score,
                "term_set": spec.term_set,
                "ast": spec.ast,
            }
        )
    candidates.sort(
        key=lambda item: item["score"] if math.isfinite(item["score"]) else -math.inf,
        reverse=True,
    )
    best_score = candidates[0]["score"] if candidates else math.nan
    return {
        "best_score": best_score,
        "top_candidates": candidates[:10],
    }


def _random_formula(rng: random.Random, index: int) -> ObjectiveSpec:
    terms = term_names("tier1")
    term_count = rng.choice([1, 2, 3])
    selected = rng.sample(terms, term_count)
    args: list[dict[str, Any]] = []
    for term in selected:
        weight = rng.choice([-1.0, -0.5, 0.25, 0.5, 1.0, 1.5])
        term_ast = {"op": "term", "name": term}
        if weight == 1.0:
            args.append(term_ast)
        else:
            args.append(
                {
                    "op": "mul",
                    "args": [{"op": "const", "value": weight}, term_ast],
                }
            )
    ast = args[0] if len(args) == 1 else {"op": "add", "args": args}
    return parse_objective_spec(
        {
            "id": "",
            "rationale": f"Random legal negative-control formula {index}.",
            "parent_ids": [],
            "declared_term_usage": selected,
            "ast": ast,
        },
        created_by="negative_control",
        term_scope="tier1",
    )


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "mean": None, "p95": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def _load_top_candidates(evolution_dir: Path) -> list[dict[str, Any]]:
    path = evolution_dir / "top_candidates.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("top_candidates", [])
    return candidates if isinstance(candidates, list) else []


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
