import json
from pathlib import Path

from coevop.eval import chipbench_generalization
from coevop.eval.chipbench_generalization import (
    run_chipbench_final_eval_from_existing,
    run_chipbench_generalization,
)
from coevop.eval.shared_panel import load_shared_panel


def test_chipbench_generalization_dry_run_writes_absolute_configs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    dreamplace_root = tmp_path / "DREAMPlace" / "install"
    chipbench_root = tmp_path / "ChiPBench"
    (dreamplace_root / "benchmarks" / "chipbench" / "bp_fe").mkdir(parents=True)
    (chipbench_root / "flow" / "designs" / "nangate45" / "bp_fe_top").mkdir(parents=True)
    (dreamplace_root / "benchmarks" / "chipbench" / "bp_fe" / "bp_fe.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (chipbench_root / "flow" / "designs" / "nangate45" / "bp_fe_top" / "config.mk").write_text(
        "DESIGN_NAME=bp_fe_top\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "run"
    summary = run_chipbench_generalization(
        run_dir=run_dir,
        dreamplace_root=dreamplace_root,
        chipbench_root=chipbench_root,
        provider="mock",
        designs=["bp_fe"],
        max_iterations=1,
        samples_per_iteration=1,
        search_seeds=[1000],
        final_seeds=[1000],
        search_iterations=1,
        final_iterations=1,
        dry_run=True,
    )

    assert summary["generalization_report"].endswith("generalization_report.md")
    generated_config = run_dir / "configs" / "bp_fe" / "openevolve_tier2.toml"
    text = generated_config.read_text(encoding="utf-8")
    assert f'panel = "{(run_dir / "configs" / "bp_fe" / "search_panel.toml").resolve().as_posix()}"' in text
    assert 'target_designs = ["bp_fe"]' in text

    post_grt_panel = load_shared_panel(run_dir / "configs" / "bp_fe" / "post_grt_panel.toml")
    assert post_grt_panel.designs[0].mode == "grt_only"
    assert "-allow_congestion" in (post_grt_panel.designs[0].global_route_args or "")

    audit = json.loads((run_dir / "environment_audit.json").read_text(encoding="utf-8"))
    assert audit["provider"] == "mock"
    assert audit["dreamplace4_baseline"]["installed_default_is_authoritative"] is True


def test_chipbench_final_eval_reuses_existing_finalist(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    dreamplace_root = tmp_path / "DREAMPlace" / "install"
    chipbench_root = tmp_path / "ChiPBench"
    (dreamplace_root / "benchmarks" / "chipbench" / "bp_fe").mkdir(parents=True)
    (chipbench_root / "flow" / "designs" / "nangate45" / "bp_fe_top").mkdir(parents=True)
    (dreamplace_root / "benchmarks" / "chipbench" / "bp_fe" / "bp_fe.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (chipbench_root / "flow" / "designs" / "nangate45" / "bp_fe_top" / "config.mk").write_text(
        "DESIGN_NAME=bp_fe_top\n",
        encoding="utf-8",
    )
    source_run = tmp_path / "source"
    finalist = source_run / "bp_fe" / "openevolve_tier2" / "robust_best_objective.json"
    finalist.parent.mkdir(parents=True)
    finalist.write_text(
        json.dumps({"id": "obj_candidate", "term_set": ["wirelength"], "ast": {"op": "term", "name": "wirelength"}, "constants": {}, "complexity": 1, "created_by": "test", "parent_ids": [], "rationale": ""}),
        encoding="utf-8",
    )

    def fake_tier2(**kwargs):
        comparison = Path(kwargs["run_dir"]) / "comparison_table.csv"
        comparison.parent.mkdir(parents=True)
        def_path = comparison.parent / "placement.def"
        def_path.write_text("VERSION 5.8 ;\n", encoding="utf-8")
        comparison.write_text(
            "design,objective_id,seed,status,output_artifact\n"
            f"bp_fe,default,1000,success,{def_path}\n"
            f"bp_fe,obj_candidate,1000,success,{def_path}\n",
            encoding="utf-8",
        )
        return {"comparison_csv": str(comparison)}

    def fake_tier3(**kwargs):
        panel = load_shared_panel(kwargs["panel_path"])
        assert panel.timeout_seconds == 0
        comparison = Path(kwargs["run_dir"]) / "comparison_table.csv"
        comparison.parent.mkdir(parents=True, exist_ok=True)
        comparison.write_text(
            "design,objective_id,seed,status,grt_overflow_delta_pct\n"
            "bp_fe,default,1000,success,0\n"
            "bp_fe,obj_candidate,1000,success,-1\n",
            encoding="utf-8",
        )
        return {"comparison_csv": str(comparison)}

    monkeypatch.setattr(chipbench_generalization, "run_tier2_dreamplace", fake_tier2)
    monkeypatch.setattr(chipbench_generalization, "run_tier3_openroad", fake_tier3)

    summary = run_chipbench_final_eval_from_existing(
        source_run=source_run,
        run_dir=tmp_path / "serious",
        dreamplace_root=dreamplace_root,
        chipbench_root=chipbench_root,
        designs=["bp_fe"],
        seeds=[1000],
        iterations=150,
        openroad_timeout_seconds=0,
    )

    assert summary["designs"][0]["status"] == "completed"
    assert Path(summary["post_grt_metrics_csv"]).is_file()
