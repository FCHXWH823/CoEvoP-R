import csv
import json
from pathlib import Path

import pytest

from coevop.eval.timing_proxy import (
    TIMING_PROXY_RC_MODEL,
    _prepare_timing_sdc,
    apply_timing_proxy_policy,
    audit_correlations,
    load_timing_proxy_panel,
    normalize_timing_proxy_status,
    perturb_def,
    placements_from_tier2_comparison,
    timing_proxy_mode_from_correlations,
)


def test_timing_panel_parser_resolves_paths(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    lib = tmp_path / "a.lib"
    sdc = tmp_path / "a.sdc"
    verilog = tmp_path / "a.v"
    ot = tmp_path / "ot-shell"
    for path in (base, lib, sdc, verilog, ot):
        path.write_text("", encoding="utf-8")
    panel_path = tmp_path / "panel.toml"
    panel_path.write_text(
        "\n".join(
            [
                'ot_shell = "ot-shell"',
                "timeout_seconds = 123",
                "iterations = 2",
                "threads = 4",
                "",
                "[[designs]]",
                'name = "toy"',
                'base_config = "base.json"',
                'lib = "a.lib"',
                'sdc = "a.sdc"',
                'verilog = "a.v"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    panel = load_timing_proxy_panel(panel_path)

    assert panel.timeout_seconds == 123
    assert panel.iterations == 2
    assert panel.threads == 4
    assert panel.ot_shell == str(ot)
    assert panel.designs[0].base_config == str(base)
    assert panel.designs[0].rc_estimate == TIMING_PROXY_RC_MODEL


def test_prepare_timing_sdc_repairs_unambiguous_scalar_port(tmp_path: Path) -> None:
    sdc = tmp_path / "input.sdc"
    verilog = tmp_path / "input.v"
    sdc.write_text(
        "set_input_delay -clock CLK 0.1 [get_ports {icache_id_i_0}]\n"
        "set_input_delay -clock CLK 0.1 [get_ports {other_0}]\n",
        encoding="utf-8",
    )
    verilog.write_text(
        "module top(clk_i, icache_id_i, other_0);\n"
        "input clk_i;\n"
        "input icache_id_i;\n"
        "input other_0;\n"
        "endmodule\n",
        encoding="utf-8",
    )

    output = _prepare_timing_sdc(sdc=sdc, verilog=verilog, run_dir=tmp_path / "run")
    normalized = output.read_text(encoding="utf-8")
    meta = json.loads((output.parent / "sdc_normalization.json").read_text(encoding="utf-8"))

    assert "icache_id_i_0" not in normalized
    assert "get_ports {icache_id_i}" in normalized
    assert "get_ports {other_0}" in normalized
    assert meta["replacements"] == {"icache_id_i_0": "icache_id_i"}
    assert meta["added_input_transition_count"] == 3
    assert meta["added_output_load_count"] == 0
    assert "set_input_transition 0.05 [get_ports icache_id_i]" in normalized


def test_perturb_def_modifies_only_placed_components(tmp_path: Path) -> None:
    source = tmp_path / "input.def"
    source.write_text(
        "\n".join(
            [
                "- u1 + PLACED ( 100 200 ) N ;",
                "- u2 + FIXED ( 300 400 ) N ;",
                "- u3 + PLACED ( 500 600 ) N ;",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = perturb_def(source, tmp_path / "out.def", magnitude_dbu=10, seed=7)
    text = output.read_text(encoding="utf-8")

    assert "+ FIXED ( 300 400 )" in text
    assert "+ PLACED ( 100 200 )" not in text
    meta = json.loads((tmp_path / "out.def.perturbation.json").read_text(encoding="utf-8"))
    assert meta["changed_component_count"] == 2


def test_correlation_mode_thresholds() -> None:
    high = timing_proxy_mode_from_correlations(
        {"hpwl_wns_pearson": 0.99},
        gate_threshold=0.70,
        tiebreaker_threshold=0.95,
    )
    mid = timing_proxy_mode_from_correlations(
        {"hpwl_wns_pearson": 0.8},
        gate_threshold=0.70,
        tiebreaker_threshold=0.95,
    )
    low = timing_proxy_mode_from_correlations(
        {"hpwl_wns_pearson": 0.4},
        gate_threshold=0.70,
        tiebreaker_threshold=0.95,
    )

    assert high["mode"] == "diagnostic"
    assert mid["mode"] == "tie_breaker"
    assert low["mode"] == "gate"


def test_audit_correlations_uses_finite_delta_pairs() -> None:
    rows = [
        {"hpwl_delta": 0.0, "wns_delta": 0.0, "tns_delta": 0.0},
        {"hpwl_delta": 1.0, "wns_delta": -2.0, "tns_delta": -3.0},
        {"hpwl_delta": 2.0, "wns_delta": -4.0, "tns_delta": -6.0},
        {"hpwl_delta": "nan", "wns_delta": 1.0, "tns_delta": 1.0},
    ]

    corr = audit_correlations(rows)

    assert corr["pair_count_wns"] == 3
    assert corr["pair_count_tns"] == 3
    assert corr["hpwl_wns_pearson"] == pytest.approx(-1.0)
    assert corr["hpwl_tns_spearman"] == pytest.approx(-1.0)


def test_timing_proxy_policy_modes() -> None:
    diagnostic = {
        "combined_score": 1.0,
        "parent_eligible": True,
        "timing_proxy_status": "success",
        "timing_proxy_wns_delta": -100.0,
        "timing_proxy_tns_delta_pct": -100.0,
    }
    apply_timing_proxy_policy(
        diagnostic,
        requested_mode="diagnostic",
        wns_regression_gate_ns=0.2,
        tns_regression_pct_gate=10.0,
    )
    assert diagnostic["parent_eligible"] is True
    assert diagnostic["timing_proxy_parent_signal"] == "not_used"

    gated = dict(diagnostic)
    apply_timing_proxy_policy(
        gated,
        requested_mode="gate",
        wns_regression_gate_ns=0.2,
        tns_regression_pct_gate=10.0,
        tns_regression_min_abs_ns=0.05,
    )
    assert gated["parent_eligible"] is False
    assert gated["negative_memory_only"] is True

    tied = {
        "combined_score": 1.0,
        "parent_eligible": True,
        "timing_proxy_status": "success",
        "timing_proxy_wns_delta": 1.0,
        "timing_proxy_tns_delta": 0.01,
        "timing_proxy_tns_delta_pct": 5.0,
    }
    apply_timing_proxy_policy(
        tied,
        requested_mode="tie_breaker",
        wns_regression_gate_ns=0.2,
        tns_regression_pct_gate=10.0,
        tns_regression_min_abs_ns=0.05,
    )
    first_score = tied["combined_score"]
    apply_timing_proxy_policy(
        tied,
        requested_mode="tie_breaker",
        wns_regression_gate_ns=0.2,
        tns_regression_pct_gate=10.0,
        tns_regression_min_abs_ns=0.05,
    )
    assert tied["timing_proxy_parent_signal"] == "tie_breaker"
    assert tied["combined_score"] == first_score

    noisy_tns = {
        "combined_score": 1.0,
        "parent_eligible": True,
        "timing_proxy_status": "success",
        "timing_proxy_wns_delta": 0.0,
        "timing_proxy_tns_delta": -0.01,
        "timing_proxy_tns_delta_pct": -100.0,
    }
    apply_timing_proxy_policy(
        noisy_tns,
        requested_mode="gate",
        wns_regression_gate_ns=0.2,
        tns_regression_pct_gate=10.0,
        tns_regression_min_abs_ns=0.05,
    )
    assert noisy_tns["parent_eligible"] is True


def test_timing_proxy_status_normalization() -> None:
    assert normalize_timing_proxy_status("success") == "success"
    assert normalize_timing_proxy_status("missing_timing_collateral") == "missing_timing_collateral"
    assert normalize_timing_proxy_status("unexpected") == "unavailable"


def test_placements_from_tier2_comparison_filters_successful_defs(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.csv"
    with comparison.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["design", "objective_id", "seed", "status", "output_artifact"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "design": "bp_fe",
                "objective_id": "default",
                "seed": "1000",
                "status": "success",
                "output_artifact": "/tmp/default.def",
            }
        )
        writer.writerow(
            {
                "design": "bp_fe",
                "objective_id": "cand",
                "seed": "1000",
                "status": "success",
                "output_artifact": "/tmp/cand.def",
            }
        )
        writer.writerow(
            {
                "design": "bp_fe",
                "objective_id": "failed",
                "seed": "1000",
                "status": "failed",
                "output_artifact": "/tmp/failed.def",
            }
        )
        writer.writerow(
            {
                "design": "bp_fe",
                "objective_id": "not_def",
                "seed": "1000",
                "status": "success",
                "output_artifact": "/tmp/not_def.txt",
            }
        )
        writer.writerow(
            {
                "design": "ethernet",
                "objective_id": "cand",
                "seed": "1000",
                "status": "success",
                "output_artifact": "/tmp/ethernet.def",
            }
        )

    output = placements_from_tier2_comparison(
        comparison,
        tmp_path / "placements.json",
        objective_ids={"cand"},
        design_names={"bp_fe"},
        include_default=True,
        include_custom_default=False,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert [item["objective_id"] for item in payload["placements"]] == ["default", "cand"]
