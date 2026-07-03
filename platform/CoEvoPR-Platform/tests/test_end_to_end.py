import csv
import json
from pathlib import Path

from coevop.eval import end_to_end
from coevop.eval.end_to_end import promote_candidates, run_end_to_end
from coevop.evolution.offline import EvolutionSummary
from coevop.objectives.presets import objective_preset
from coevop.objectives.spec import parse_objective_spec, write_objective_spec


def _shared_panel(tmp_path: Path) -> Path:
    dreamplace_config = tmp_path / "bp_fe.json"
    chipbench_config = tmp_path / "config.mk"
    dreamplace_config.write_text("{}", encoding="utf-8")
    chipbench_config.write_text("DESIGN_NAME=bp_fe_top\n", encoding="utf-8")
    panel = tmp_path / "shared_panel.toml"
    panel.write_text(
        "\n".join(
            [
                "seeds = [1000]",
                "iterations = 1",
                "timeout_seconds = 9",
                "",
                "[[designs]]",
                'name = "bp_fe"',
                f'dreamplace_config = "{dreamplace_config.as_posix()}"',
                f'chipbench_config = "{chipbench_config.as_posix()}"',
                'mode = "global"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return panel


def test_promote_candidates_filters_tier1_only_terms(tmp_path: Path) -> None:
    deployable = write_objective_spec(
        objective_preset("dreamplace_routing_aware_smoke"),
        tmp_path / "deployable.json",
    )
    tier1_only = write_objective_spec(
        parse_objective_spec(
            {
                "id": "",
                "rationale": "tier1-only",
                "parent_ids": [],
                "declared_term_usage": ["rudy_p95"],
                "ast": {"op": "term", "name": "rudy_p95"},
            },
            created_by="test",
            term_scope="tier1",
        ),
        tmp_path / "tier1_only.json",
    )

    promoted = promote_candidates(
        [
            {"artifact_path": str(tier1_only), "score": 3.0},
            {"artifact_path": str(deployable), "score": 2.0},
        ],
        output_dir=tmp_path / "promoted",
        top_k=5,
    )

    assert len(promoted) == 1
    assert promoted[0]["dreamplace_deployable"] is True


def test_promote_candidates_respects_min_terms(tmp_path: Path) -> None:
    single_term = write_objective_spec(
        objective_preset("dreamplace_soft_rudy_mean"),
        tmp_path / "single_term.json",
    )
    multi_term = write_objective_spec(
        objective_preset("dreamplace_routing_aware_smoke"),
        tmp_path / "multi_term.json",
    )

    promoted = promote_candidates(
        [
            {"artifact_path": str(single_term), "score": 3.0},
            {"artifact_path": str(multi_term), "score": 2.0},
        ],
        output_dir=tmp_path / "promoted_min_terms",
        top_k=5,
        min_terms=2,
    )

    assert len(promoted) == 1
    assert promoted[0]["objective_id"] == objective_preset("dreamplace_routing_aware_smoke").id


def test_evolve_end_to_end_mock_round(monkeypatch, tmp_path: Path) -> None:
    objective_path = write_objective_spec(
        objective_preset("dreamplace_routing_aware_smoke"),
        tmp_path / "candidate.json",
    )
    panel = _shared_panel(tmp_path)
    prior_feedback = tmp_path / "prior_feedback.json"
    prior_feedback.write_text(
        json.dumps(
            {
                "severe_regressions": [
                    {"objective_id": "bad_pin_only", "hpwl_delta_pct": 299.5}
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "e2e.toml"
    config.write_text(
        "\n".join(
            [
                f'manifest = "{(tmp_path / "manifest.json").as_posix()}"',
                'provider = "mock"',
                "rounds = 1",
                "",
                "[tier1]",
                "population_size = 1",
                "generations = 1",
                "elite_count = 1",
                "",
                "[promotion]",
                "top_k_tier2 = 1",
                "min_terms_tier2 = 2",
                "",
                "[tier2]",
                f'panel = "{panel.as_posix()}"',
                'baseline_presets = ["dreamplace_density_heavy"]',
                "",
                "[tier3]",
                "enabled = false",
                "",
                "[feedback]",
                f'prior_feedback = "{prior_feedback.as_posix()}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_offline(**kwargs):
        assert kwargs["external_feedback"]["severe_regressions"][0]["objective_id"] == "bad_pin_only"
        return EvolutionSummary(
            run_dir=str(kwargs["run_dir"]),
            db_path=str(Path(kwargs["run_dir"]) / "evolution.sqlite"),
            provider="mock",
            dataset="synthetic",
            row_count=3,
            split_strategy="none",
            split_row_counts={"train": 1, "validation": 1, "heldout": 1},
            selection_split="validation",
            term_scope="deployable",
            reflection_mode="accepted",
            diversity_lambda=0.05,
            operator_weights={},
            islands=5,
            constant_fit_enabled=True,
            requested_candidates=1,
            generations=1,
            accepted_candidates=1,
            rejected_candidates=0,
            top_candidates=[
                {
                    "objective_id": "candidate",
                    "score": 1.0,
                    "artifact_path": str(objective_path),
                    "term_set": ["soft_rudy_mean"],
                    "complexity": 1,
                    "metrics": {
                        "split_scores": {
                            "train": {"score": 0.5},
                            "validation": {"score": 1.0},
                            "heldout": {"score": -0.2},
                        }
                    },
                }
            ],
        )

    def fake_tier2(**kwargs):
        run_dir = Path(kwargs["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        def_path = tmp_path / "candidate.def"
        def_path.write_text("VERSION 5.8 ;\n", encoding="utf-8")
        comparison_csv = run_dir / "comparison_table.csv"
        with comparison_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "design",
                    "objective_id",
                    "seed",
                    "status",
                    "failure_stage",
                    "hpwl_delta_pct",
                    "overflow_delta_pct",
                    "runtime_seconds",
                    "output_artifact",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "design": "bp_fe",
                    "objective_id": "default",
                    "seed": 1000,
                    "status": "success",
                    "failure_stage": "",
                    "hpwl_delta_pct": 0.0,
                    "overflow_delta_pct": 0.0,
                    "runtime_seconds": 10,
                    "output_artifact": str(def_path),
                }
            )
            writer.writerow(
                {
                    "design": "bp_fe",
                    "objective_id": "candidate",
                    "seed": 1000,
                    "status": "success",
                    "failure_stage": "",
                    "hpwl_delta_pct": -1.0,
                    "overflow_delta_pct": -2.0,
                    "runtime_seconds": 11,
                    "output_artifact": str(def_path),
                }
            )
        return {
            "comparison_csv": str(comparison_csv),
            "success_count": 2,
            "failure_count": 0,
            "result_count": 2,
        }

    monkeypatch.setattr(end_to_end, "run_offline_evolution", fake_offline)
    monkeypatch.setattr(end_to_end, "run_tier2_dreamplace", fake_tier2)

    summary = run_end_to_end(
        config_path=config,
        run_dir=tmp_path / "e2e",
        dreamplace_root=tmp_path,
        chipbench_root=tmp_path,
    )

    assert summary["round_count"] == 1
    round_dir = tmp_path / "e2e" / "round_000"
    assert (round_dir / "promoted_objectives" / "manifest.json").exists()
    assert (round_dir / "tier2_dreamplace" / "placements.json").exists()
    feedback = json.loads((round_dir / "feedback.json").read_text(encoding="utf-8"))
    assert feedback["best_candidates"][0]["objective_id"] == "candidate"
    assert "heldout_score" not in json.dumps(feedback)
