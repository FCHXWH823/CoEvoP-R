import csv
import json
from pathlib import Path

from coevop.eval import eureka_tier2
from coevop.eval.eureka_tier2 import (
    build_prompt_context,
    load_eureka_tier2_config,
    run_eureka_tier2,
    static_rejection_reason,
)
from coevop.objectives.program import parse_objective_program
from coevop.objectives.spec import load_objective_spec, parse_objective_spec


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


def _config(tmp_path: Path, panel: Path) -> Path:
    prior_feedback = tmp_path / "prior_feedback.json"
    prior_feedback.write_text(
        json.dumps(
            {
                "severe_regressions": [
                    {"objective_id": "pin_only", "hpwl_delta_pct": 299.5}
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "eureka.toml"
    config.write_text(
        "\n".join(
            [
                'provider = "mock"',
                'term_scope = "dreamplace_replacement"',
                "eureka_iterations = 2",
                "samples_per_iteration = 2",
                "candidates_per_iteration = 1",
                "temperature = 1.0",
                "",
                "[search]",
                f'panel = "{panel.as_posix()}"',
                'baseline_presets = ["dreamplace_routing_aware_smoke"]',
                "",
                "[final]",
                "enabled = false",
                "",
                "[feedback]",
                f'prior_feedback = "{prior_feedback.as_posix()}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config


def test_eureka_prompt_context_contains_environment_and_no_keys(tmp_path: Path) -> None:
    config = load_eureka_tier2_config(_config(tmp_path, _shared_panel(tmp_path)))

    context = build_prompt_context(
        config=config,
        previous_feedback={"note": "previous"},
        iteration_index=0,
        iteration_feedback=None,
    )
    encoded = json.dumps(context)

    assert "DREAMPlace" in encoded
    assert "custom objective" in encoded
    assert "pin_density_pnorm" in encoded
    assert "OPENAI_API_KEY" not in encoded
    assert "severe_regression_pct" not in encoded
    assert "Good candidates must" not in encoded
    assert "HPWL must not regress" not in encoded
    assert "exact thresholds are evaluator-side" in encoded


def test_static_rejection_rejects_single_term_replacement() -> None:
    spec = parse_objective_spec(
        {
            "id": "",
            "rationale": "too simple",
            "parent_ids": [],
            "declared_term_usage": ["pin_density_pnorm"],
            "ast": {"op": "term", "name": "pin_density_pnorm"},
        },
        created_by="test",
        term_scope="dreamplace_replacement",
    )

    assert static_rejection_reason(spec) == "replacement objective must use at least two term families"


def test_static_rejection_rejects_internal_native_objective() -> None:
    spec = parse_objective_program(
        "\n".join(
            [
                "def objective(features):",
                '    native = term("native_objective")',
                '    route = term("route_pressure_long")',
                "    score = native + 1e-8 * route",
                '    return score, {"native": native, "route": route}',
            ]
        ),
        created_by="test",
        term_scope="dreamplace",
    )

    reason = static_rejection_reason(spec)

    assert reason is not None
    assert "unsupported dreamplace_replacement terms" in reason
    assert "native_objective" in reason


def test_eureka_tier2_mock_loop_creates_artifacts(monkeypatch, tmp_path: Path) -> None:
    panel = _shared_panel(tmp_path)
    config = _config(tmp_path, panel)

    def fake_tier2(**kwargs):
        run_dir = Path(kwargs["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        comparison_csv = run_dir / "comparison_table.csv"
        rows = [
            {
                "design": "bp_fe",
                "objective_id": "default",
                "seed": 1000,
                "status": "success",
                "failure_stage": "",
                "hpwl_delta_pct": 0.0,
                "overflow_delta_pct": 0.0,
                "runtime_seconds": 10,
                "output_artifact": str(tmp_path / "default.def"),
            },
            {
                "design": "bp_fe",
                "objective_id": "custom_default",
                "seed": 1000,
                "status": "success",
                "failure_stage": "",
                "hpwl_delta_pct": 0.0,
                "overflow_delta_pct": 0.0,
                "runtime_seconds": 10,
                "output_artifact": str(tmp_path / "custom.def"),
            },
        ]
        for objective_path in kwargs["objective_paths"]:
            spec = load_objective_spec(objective_path)
            rows.append(
                {
                    "design": "bp_fe",
                    "objective_id": spec.id,
                    "seed": 1000,
                    "status": "success",
                    "failure_stage": "",
                    "hpwl_delta_pct": -1.0,
                    "overflow_delta_pct": -2.0,
                    "runtime_seconds": 11,
                    "output_artifact": str(tmp_path / f"{spec.id}.def"),
                }
            )
        with comparison_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return {
            "comparison_csv": str(comparison_csv),
            "success_count": len(rows),
            "failure_count": 0,
            "result_count": len(rows),
        }

    monkeypatch.setattr(eureka_tier2, "run_tier2_dreamplace", fake_tier2)

    summary = run_eureka_tier2(
        config_path=config,
        run_dir=tmp_path / "run",
        dreamplace_root=tmp_path,
    )

    assert summary["candidate_attempt_count"] == 4
    assert summary["valid_candidate_count"] == 4
    assert summary["final_tier2"] is None
    assert not (tmp_path / "run" / "tier3_openroad").exists()
    sample_dir = tmp_path / "run" / "iteration_000" / "samples" / "sample_0000"
    assert (sample_dir / "prompt_messages.json").exists()
    assert (sample_dir / "prompt_context.md").exists()
    assert (sample_dir / "raw_response.json").exists()
    feedback = json.loads(
        (tmp_path / "run" / "iteration_001" / "feedback.json").read_text(encoding="utf-8")
    )
    assert feedback["best_generated"] is not None
