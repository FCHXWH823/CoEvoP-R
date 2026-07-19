import csv
import json
from pathlib import Path

import pytest

from coevop.eval.timing_proxy import (
    TIMING_PROXY_RC_MODEL,
    TimingProxyDesign,
    TimingProxyPanel,
    TimingProxyResult,
    _make_timing_proxy_config,
    _plain_opentimer_sdc,
    _prepare_timing_def,
    _prepare_timing_verilog,
    _prepare_timing_lib,
    _prepare_timing_sdc,
    _result_row,
    _write_audit_panel,
    apply_timing_proxy_policy,
    audit_correlations,
    load_timing_proxy_panel,
    normalize_timing_proxy_status,
    perturb_def,
    placements_from_tier2_comparison,
    timing_proxy_metrics_by_objective,
    timing_proxy_mode_from_correlations,
)


def test_timing_result_row_preserves_source_iteration_budget() -> None:
    result = TimingProxyResult(
        design="toy",
        objective_id="obj_test",
        seed=1000,
        variant="original",
        status="success",
        failure_stage=None,
        timing_proxy_rc_model=TIMING_PROXY_RC_MODEL,
        timing_proxy_hpwl=100.0,
        timing_proxy_wns=-0.1,
        timing_proxy_tns=-1.0,
        timing_proxy_num_violating_paths=None,
        timing_proxy_runtime_seconds=1.0,
        def_path="placed.def",
        run_dir="run",
        config_path="config.json",
        log_path="dreamplace.log",
        returncode=0,
        artifacts={},
        source_placement_iterations=1000,
        source_placement_completed_iterations=2001,
        source_placement_budget_satisfied=True,
    )

    row = _result_row(result)

    assert row["source_placement_iterations"] == 1000
    assert row["source_placement_completed_iterations"] == 2001
    assert row["source_placement_budget_satisfied"] is True


def test_timing_metrics_expose_final_def_placement_deltas(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.csv"
    fieldnames = [
        "design",
        "objective_id",
        "timing_proxy_status",
        "timing_proxy_hpwl",
        "placement_overflow",
        "timing_proxy_hpwl_delta_pct",
        "placement_overflow_delta_pct",
        "timing_proxy_wns",
        "timing_proxy_tns",
        "timing_proxy_net_coverage",
        "timing_proxy_total_nets",
        "timing_proxy_skipped_nets",
        "timing_proxy_pin_incomplete_nets",
        "timing_proxy_wns_delta",
        "timing_proxy_tns_delta",
        "timing_proxy_tns_delta_pct",
        "source_placement_iterations",
        "source_placement_completed_iterations",
        "source_placement_budget_satisfied",
    ]
    with comparison.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "design": "mor1kx",
                "objective_id": "default",
                "timing_proxy_status": "success",
                "timing_proxy_hpwl": 100.0,
                "placement_overflow": 0.10,
                "source_placement_iterations": 1000,
                "source_placement_completed_iterations": 2001,
                "source_placement_budget_satisfied": "true",
            }
        )
        writer.writerow(
            {
                "design": "mor1kx",
                "objective_id": "candidate",
                "timing_proxy_status": "success",
                "timing_proxy_hpwl": 92.0,
                "placement_overflow": 0.104,
                "timing_proxy_hpwl_delta_pct": -8.0,
                "placement_overflow_delta_pct": 4.0,
                "timing_proxy_wns": -0.001,
                "timing_proxy_tns": -0.02,
                "timing_proxy_net_coverage": 0.983,
                "timing_proxy_total_nets": 1000,
                "timing_proxy_skipped_nets": 10,
                "timing_proxy_pin_incomplete_nets": 7,
                "timing_proxy_wns_delta": -0.00001,
                "timing_proxy_tns_delta": -0.001,
                "timing_proxy_tns_delta_pct": -5.0,
                "source_placement_iterations": 1000,
                "source_placement_completed_iterations": 2001,
                "source_placement_budget_satisfied": "true",
            }
        )

    metrics = timing_proxy_metrics_by_objective(
        {"comparison_csv": str(comparison)},
        baseline_objective_id="default",
    )["candidate"]

    assert metrics["final_def_hpwl_delta_pct"] == pytest.approx(-8.0)
    assert metrics["final_def_overflow_delta_pct"] == pytest.approx(4.0)
    assert metrics["final_def_per_design_deltas"] == [
        {
            "design": "mor1kx",
            "hpwl_delta_pct": pytest.approx(-8.0),
            "overflow_delta_pct": pytest.approx(4.0),
            "cell_count": 1,
        }
    ]
    assert metrics["timing_proxy_source_placement_budget_satisfied"] is True
    assert metrics["timing_proxy_net_coverage"] == pytest.approx(0.983)
    assert metrics["timing_proxy_total_nets"] == 1000


def test_audit_panel_preserves_verified_timing_collateral(tmp_path: Path) -> None:
    dreamplace_root = tmp_path / "dreamplace"
    (dreamplace_root / "bin").mkdir(parents=True)
    (dreamplace_root / "bin" / "ot-shell").write_text("", encoding="utf-8")
    base = tmp_path / "base.json"
    base.write_text("{}", encoding="utf-8")
    lib = tmp_path / "a.lib"
    sdc = tmp_path / "a.sdc"
    verilog = tmp_path / "a.v"
    for path in (lib, sdc, verilog):
        path.write_text("", encoding="utf-8")
    source = tmp_path / "source.toml"
    source.write_text(
        "\n".join(
            [
                f'ot_shell = "{(dreamplace_root / "bin" / "ot-shell").as_posix()}"',
                "threads = 2",
                "legalize = false",
                "fixed_position = true",
                "min_net_coverage = 0.9",
                "[[designs]]",
                'name = "mor1kx"',
                f'base_config = "{base.as_posix()}"',
                f'lib = "{lib.as_posix()}"',
                f'sdc = "{sdc.as_posix()}"',
                f'verilog = "{verilog.as_posix()}"',
                'top_module = "mor1kx"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output = _write_audit_panel(
        tmp_path / "audit.toml",
        base_config=base,
        source_panel=source,
        design="mor1kx",
        dreamplace_root=dreamplace_root,
        timeout_seconds=60,
        gpu=0,
    )
    panel = load_timing_proxy_panel(output)

    assert panel.fixed_position is True
    assert panel.legalize is False
    assert panel.threads == 2
    assert panel.designs[0].lib == str(lib)
    assert panel.designs[0].sdc == str(sdc)
    assert panel.designs[0].verilog == str(verilog)
    assert panel.designs[0].top_module == "mor1kx"


def test_timing_proxy_config_preserves_supplied_placement(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    base.write_text(
        json.dumps(
            {
                "random_center_init_flag": 1,
                "gp_noise_ratio": 0.025,
                "global_place_stages": [{"iteration": 1000}],
            }
        ),
        encoding="utf-8",
    )
    placement = tmp_path / "placed.def"
    placement.write_text(
        "VERSION 5.8 ;\nCOMPONENTS 0 ;\nEND COMPONENTS\nEND DESIGN\n",
        encoding="utf-8",
    )
    design = TimingProxyDesign(name="toy", base_config=str(base))
    panel = TimingProxyPanel(
        ot_shell=str(tmp_path / "ot-shell"),
        timeout_seconds=60,
        iterations=1,
        gpu=None,
        threads=1,
        designs=[design],
    )

    output = _make_timing_proxy_config(
        design=design,
        panel=panel,
        placement_def=placement,
        run_dir=tmp_path / "run",
    )
    config = json.loads(output.read_text(encoding="utf-8"))

    assert config["random_center_init_flag"] == 0
    assert config["gp_noise_ratio"] == 0.0


def test_timing_proxy_config_supports_late_global_static_fallback(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    base.write_text(
        json.dumps(
            {
                "global_place_stages": [
                    {"iteration": 1000, "learning_rate": 0.01, "optimizer": "nesterov"}
                ],
                "legalize_flag": 1,
                "detailed_place_flag": 1,
            }
        ),
        encoding="utf-8",
    )
    placement = tmp_path / "placed.def"
    placement.write_text(
        "VERSION 5.8 ;\nCOMPONENTS 0 ;\nEND COMPONENTS\nEND DESIGN\n",
        encoding="utf-8",
    )
    panel = TimingProxyPanel(
        ot_shell=str(tmp_path / "ot-shell"),
        timeout_seconds=60,
        iterations=511,
        gpu=None,
        threads=1,
        designs=[TimingProxyDesign(name="toy", base_config=str(base))],
        legalize=False,
        learning_rate=1e-12,
    )

    output = _make_timing_proxy_config(
        design=panel.designs[0],
        panel=panel,
        placement_def=placement,
        run_dir=tmp_path / "run",
    )
    config = json.loads(output.read_text(encoding="utf-8"))

    assert config["legalize_flag"] == 0
    assert config["detailed_place_flag"] == 0
    assert config["global_place_stages"][0]["iteration"] == 511
    assert config["global_place_stages"][0]["learning_rate"] == pytest.approx(1e-12)


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


def test_timing_panel_parser_accepts_fixed_position_driver(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    ot = tmp_path / "ot-shell"
    base.write_text("{}", encoding="utf-8")
    ot.write_text("", encoding="utf-8")
    panel_path = tmp_path / "panel.toml"
    panel_path.write_text(
        'ot_shell = "ot-shell"\nfixed_position = true\nlegalize = false\n'
        'position_jitter = 0.00001\n'
        'min_net_coverage = 0.9\n'
        'derive_verilog_from_def = true\n'
        '[[designs]]\nname = "toy"\nbase_config = "base.json"\n',
        encoding="utf-8",
    )

    panel = load_timing_proxy_panel(panel_path)

    assert panel.fixed_position is True
    assert panel.legalize is False
    assert panel.position_jitter == pytest.approx(1e-5)
    assert panel.min_net_coverage == pytest.approx(0.9)
    assert panel.derive_verilog_from_def is True


def test_timing_config_reconstructs_named_port_netlist_from_evaluated_def(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.json"
    base.write_text(
        json.dumps(
            {
                "global_place_stages": [
                    {"iteration": 10, "learning_rate": 0.01, "optimizer": "nesterov"}
                ]
            }
        ),
        encoding="utf-8",
    )
    sdc = tmp_path / "floorplan.sdc"
    sdc.write_text(
        "current_design toy_top\n"
        "create_clock -name CLK -period 4 [get_ports {clk}]\n",
        encoding="utf-8",
    )
    placement = tmp_path / "placed.def"
    placement.write_text(
        "\n".join(
            [
                "VERSION 5.8 ;",
                "DESIGN toy_top ;",
                "COMPONENTS 2 ;",
                "  - u/path INV_X1 + PLACED ( 0 0 ) N ;",
                "  - _1_ NAND2_X1",
                "    + PLACED ( 10 20 ) N ;",
                "END COMPONENTS",
                "PINS 3 ;",
                "  - clk + NET clk + DIRECTION INPUT + USE SIGNAL ;",
                "  - data[0] + NET data[0] + DIRECTION INPUT + USE SIGNAL ;",
                "  - out + NET result + DIRECTION OUTPUT + USE SIGNAL ;",
                "END PINS",
                "NETS 4 ;",
                "  - clk ( PIN clk ) ( u/path A ) + USE CLOCK ;",
                "  - data[0] ( PIN data[0] ) ( _1_ A1 ) + USE SIGNAL ;",
                "  - mid ( u/path ZN )",
                "    ( _1_ A2 ) + USE SIGNAL ;",
                "  - result ( PIN out ) ( _1_ ZN ) + USE SIGNAL ;",
                "END NETS",
                "END DESIGN",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    panel = TimingProxyPanel(
        ot_shell=str(tmp_path / "ot-shell"),
        timeout_seconds=60,
        iterations=1,
        gpu=None,
        threads=1,
        designs=[
            TimingProxyDesign(
                name="toy",
                base_config=str(base),
                sdc=str(sdc),
                top_module="toy_top",
            )
        ],
        legalize=False,
        scalarize_vector_nets=True,
        derive_verilog_from_def=True,
    )

    output = _make_timing_proxy_config(
        design=panel.designs[0],
        panel=panel,
        placement_def=placement,
        run_dir=tmp_path / "run",
    )

    config = json.loads(output.read_text(encoding="utf-8"))
    normalized_def = Path(config["def_input"]).read_text(encoding="utf-8")
    normalized_verilog = Path(config["verilog_input"]).read_text(encoding="utf-8")
    normalized_sdc = Path(config["sdc_input"]).read_text(encoding="utf-8")
    manifest = json.loads(
        (tmp_path / "run" / "def_timing_netlist" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert "module toy_top" in normalized_verilog
    assert "u_path" in normalized_def
    assert "u_path" in normalized_verilog
    assert "data[0]" not in normalized_def
    assert "data[0]" not in normalized_verilog
    assert "v_data_0" in normalized_def
    assert "v_data_0" in normalized_verilog
    assert "assign out = result;" in normalized_verilog
    assert "create_clock [get_ports {clk}] -name CLK -period 4" in normalized_sdc
    assert manifest["component_count"] == 2
    assert manifest["port_count"] == 3
    assert manifest["net_count"] == 4
    assert manifest["connected_instance_pin_count"] == 5


def test_timing_panel_parser_accepts_lib_list(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    lib_a = tmp_path / "a.lib"
    lib_b = tmp_path / "b.lib"
    sdc = tmp_path / "a.sdc"
    verilog = tmp_path / "a.v"
    ot = tmp_path / "ot-shell"
    for path in (base, lib_a, lib_b, sdc, verilog, ot):
        path.write_text("", encoding="utf-8")
    panel_path = tmp_path / "panel.toml"
    panel_path.write_text(
        "\n".join(
            [
                'ot_shell = "ot-shell"',
                "",
                "[[designs]]",
                'name = "toy"',
                'base_config = "base.json"',
                'lib = ["a.lib", "b.lib"]',
                'sdc = "a.sdc"',
                'verilog = "a.v"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    panel = load_timing_proxy_panel(panel_path)

    assert panel.designs[0].lib == [str(lib_a), str(lib_b)]


def test_timing_panel_parser_accepts_top_module(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    verilog = tmp_path / "a.v"
    ot = tmp_path / "ot-shell"
    for path in (base, verilog, ot):
        path.write_text("", encoding="utf-8")
    panel_path = tmp_path / "panel.toml"
    panel_path.write_text(
        "\n".join(
            [
                'ot_shell = "ot-shell"',
                "",
                "[[designs]]",
                'name = "ibex"',
                'base_config = "base.json"',
                'verilog = "a.v"',
                'top_module = "ibex_core"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    panel = load_timing_proxy_panel(panel_path)

    assert panel.designs[0].top_module == "ibex_core"


def test_prepare_timing_lib_merges_lib_list(tmp_path: Path) -> None:
    lib_a = tmp_path / "a.lib"
    lib_b = tmp_path / "b.lib"
    lib_a.write_text("library(a) {\n  cell (A) {}\n}\n", encoding="utf-8")
    lib_b.write_text("library(b) {\n  cell (B) {}\n}\n", encoding="utf-8")

    merged = _prepare_timing_lib(lib=[str(lib_a), str(lib_b)], run_dir=tmp_path / "run")

    text = merged.read_text(encoding="utf-8")
    assert text.count("library(") == 1
    assert "library(a)" in text
    assert "cell (A)" in text
    assert "cell (B)" in text
    manifest = json.loads((merged.parent / "lib_merge_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sources"] == [str(lib_a), str(lib_b)]


def test_prepare_timing_verilog_strips_yosys_attributes(tmp_path: Path) -> None:
    verilog = tmp_path / "input.v"
    verilog.write_text(
        '(* top = 1 *)\nmodule gcd(clk);\n  (* src = "x" *) input clk;\nendmodule\n',
        encoding="utf-8",
    )

    output = _prepare_timing_verilog(verilog=verilog, run_dir=tmp_path / "run")

    text = output.read_text(encoding="utf-8")
    assert "(*" not in text
    assert "module gcd" in text


def test_prepare_timing_verilog_sanitizes_all_escaped_bus_identifiers(tmp_path: Path) -> None:
    verilog = tmp_path / "input.v"
    verilog.write_text(
        r"""module top(\data[0] , \data[1] , \out[0] );
input \data[0] ;
input \data[1] ;
output \out[0] ;
assign \out[0] = \data[0] ;
endmodule
""",
        encoding="utf-8",
    )

    output = _prepare_timing_verilog(verilog=verilog, run_dir=tmp_path / "run")

    text = output.read_text(encoding="utf-8")
    meta = json.loads((output.parent / "verilog_normalization.json").read_text(encoding="utf-8"))
    assert "\\data" not in text
    assert "v_data_0" in text
    assert "v_data_1" in text
    assert "v_out_0" in text
    assert meta["renamed_escaped_identifier_count"] == 3


def test_prepare_timing_verilog_moves_requested_top_module_first(tmp_path: Path) -> None:
    verilog = tmp_path / "input.v"
    verilog.write_text(
        "\n".join(
            [
                "module helper(a);",
                "input a;",
                "endmodule",
                "module ibex_core(clk_i, done_o);",
                "input clk_i;",
                "output done_o;",
                "endmodule",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output = _prepare_timing_verilog(
        verilog=verilog,
        run_dir=tmp_path / "run",
        top_module="ibex_core",
    )

    text = output.read_text(encoding="utf-8")
    assert text.find("module ibex_core") < text.find("module helper")
    meta = json.loads((output.parent / "verilog_normalization.json").read_text(encoding="utf-8"))
    assert meta["top_module_requested"] == "ibex_core"
    assert meta["top_module_found"] == "ibex_core"
    assert meta["top_module_moved_first"] is True


def test_prepare_timing_sdc_uses_selected_top_module_ports_only(tmp_path: Path) -> None:
    sdc = tmp_path / "input.sdc"
    verilog = tmp_path / "input.v"
    sdc.write_text(
        "\n".join(
            [
                "current_design ibex_core",
                "set clk_name core_clock",
                "set clk_port_name clk_i",
                "set clk_period 10",
                "set clk_io_pct 0.2",
                "set non_clock_inputs [lsearch -inline -all -not -exact [all_inputs] $clk_port]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    verilog.write_text(
        "\n".join(
            [
                "module helper(a);",
                "input a;",
                "endmodule",
                "module ibex_core(clk_i, rst_ni, done_o);",
                "input clk_i;",
                "input rst_ni;",
                "output done_o;",
                "endmodule",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prepared_verilog = _prepare_timing_verilog(
        verilog=verilog,
        run_dir=tmp_path / "run",
        top_module="ibex_core",
    )

    output = _prepare_timing_sdc(sdc=sdc, verilog=prepared_verilog, run_dir=tmp_path / "run")
    normalized = output.read_text(encoding="utf-8")

    assert "get_ports {clk_i}" in normalized
    assert "get_ports {rst_ni}" in normalized
    assert "get_ports {done_o}" in normalized
    assert "get_ports {a}" not in normalized


def test_prepare_timing_def_and_verilog_share_safe_instance_names(tmp_path: Path) -> None:
    source_def = tmp_path / "input.def"
    source_def.write_text(
        "\n".join(
            [
                "VERSION 5.8 ;",
                "COMPONENTS 3 ;",
                "  - _296_ INVx1_ASAP7_75t_R + PLACED ( 0 0 ) N ;",
                "  - dpath.a_reg.out\\[0\\]$_DFFE_PP_ DFFHQNx1_ASAP7_75t_R + PLACED ( 1 2 ) N ;",
                "  - dpath.a_reg.out\\[1\\]$_DFFE_PP_ DFFHQNx1_ASAP7_75t_R + PLACED ( 3 4 ) N ;",
                "END COMPONENTS",
                "NETS 1 ;",
                "  - n1 ( _296_ Y ) ( dpath.a_reg.out\\[0\\]$_DFFE_PP_ QN ) ( dpath.a_reg.out\\[1\\]$_DFFE_PP_ D ) ;",
                "END NETS",
                "END DESIGN",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    verilog = tmp_path / "input.v"
    verilog.write_text(
        "\n".join(
            [
                "module gcd(clk);",
                "input clk;",
                "INVx1_ASAP7_75t_R _296_ (.A(clk));",
                "DFFHQNx1_ASAP7_75t_R \\dpath.a_reg.out[0]$_DFFE_PP_  (.CLK(clk));",
                "DFFHQNx1_ASAP7_75t_R \\dpath.a_reg.out[1]$_DFFE_PP_  (.CLK(clk));",
                "endmodule",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    normalized_def, name_map = _prepare_timing_def(def_file=source_def, run_dir=tmp_path / "run")
    normalized_verilog = _prepare_timing_verilog(
        verilog=verilog,
        run_dir=tmp_path / "run",
        instance_name_map=name_map,
    )

    def_text = normalized_def.read_text(encoding="utf-8")
    verilog_text = normalized_verilog.read_text(encoding="utf-8")
    assert "dpath.a_reg.out" not in def_text
    assert "\\dpath.a_reg.out" not in verilog_text
    assert "_296_" in def_text
    assert "_296_" in verilog_text
    assert "dpath_a_reg_out_0_DFFE_PP" in def_text
    assert "dpath_a_reg_out_0_DFFE_PP" in verilog_text
    assert "dpath_a_reg_out_1_DFFE_PP" in def_text
    assert "dpath_a_reg_out_1_DFFE_PP" in verilog_text


def test_prepare_timing_def_preserves_plain_and_bus_nets_and_maps_escaped_nets(
    tmp_path: Path,
) -> None:
    source_def = tmp_path / "input.def"
    source_def.write_text(
        "\n".join(
            [
                "VERSION 5.8 ;",
                "COMPONENTS 0 ;",
                "END COMPONENTS",
                "NETS 3 ;",
                "  - plain_net ( PIN A ) ;",
                "  - data[0] ( PIN C ) ;",
                r"  - \\data[0] ( PIN B ) ;",
                "END NETS",
                "END DESIGN",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    verilog = tmp_path / "input.v"
    verilog.write_text(
        "module top(plain_net, data, \\data[0] );\n"
        "input plain_net;\n"
        "input [0:0] data;\n"
        "input \\data[0] ;\n"
        "endmodule\n",
        encoding="utf-8",
    )

    normalized_def, name_map = _prepare_timing_def(
        def_file=source_def,
        run_dir=tmp_path / "run",
    )
    normalized_verilog = _prepare_timing_verilog(
        verilog=verilog,
        run_dir=tmp_path / "run",
        instance_name_map=name_map,
    )

    def_text = normalized_def.read_text(encoding="utf-8")
    verilog_text = normalized_verilog.read_text(encoding="utf-8")
    assert "plain_net" in def_text
    assert "plain_net" in verilog_text
    assert "data[0]" in def_text
    assert "data[0]" not in name_map
    assert "v_data_0" in def_text
    assert "v_data_0" in verilog_text
    assert "plain_net" not in name_map

    scalarized_def, scalarized_map = _prepare_timing_def(
        def_file=source_def,
        run_dir=tmp_path / "scalarized_run",
        scalarize_vector_nets=True,
    )
    scalarized_text = scalarized_def.read_text(encoding="utf-8")
    assert "data[0]" not in scalarized_text
    assert scalarized_map["data[0]"].startswith("v_data_0")


def test_prepare_timing_sdc_expands_wildcard_ports_with_explicit_clock(tmp_path: Path) -> None:
    sdc = tmp_path / "input.sdc"
    verilog = tmp_path / "input.v"
    sdc.write_text(
        "create_clock [get_ports {clk}] -name main_clock -period 8\n"
        "set_input_delay -clock main_clock 1 [get_ports {data[*]}]\n",
        encoding="utf-8",
    )
    verilog.write_text(
        "module top(clk, v_data_0, v_data_1, done);\n"
        "input clk;\n"
        "input v_data_0;\n"
        "input v_data_1;\n"
        "output done;\n"
        "endmodule\n",
        encoding="utf-8",
    )

    output = _prepare_timing_sdc(sdc=sdc, verilog=verilog, run_dir=tmp_path / "run")
    normalized = output.read_text(encoding="utf-8")
    meta = json.loads((output.parent / "sdc_normalization.json").read_text(encoding="utf-8"))

    assert "data[*]" not in normalized
    assert "create_clock [get_ports {clk}] -name main_clock -period 8" in normalized
    assert "set_input_delay -clock main_clock 1.6 [get_ports {v_data_0}]" in normalized
    assert "set_input_delay -clock main_clock 1.6 [get_ports {v_data_1}]" in normalized
    assert meta["simplified_sdc"] is True


def test_plain_opentimer_sdc_expands_orfs_helper_script() -> None:
    text = "\n".join(
        [
            "current_design gcd",
            "set clk_name core_clock",
            "set clk_port_name clk",
            "set clk_period 390",
            "set clk_io_pct 0.2",
            "set non_clock_inputs [lsearch -inline -all -not -exact [all_inputs] $clk_port]",
        ]
    )
    ports = {"clk": "input", "req_msg": "input", "reset": "input", "resp_msg": "output"}

    output = _plain_opentimer_sdc(text, ports)

    assert "current_design" not in output
    assert "lsearch" not in output
    assert "create_clock [get_ports {clk}] -name core_clock -period 390" in output
    assert "set_input_delay -clock core_clock 0 [get_ports {clk}]" in output
    assert "set_input_delay -clock core_clock 78 [get_ports {req_msg}]" in output
    assert "set_input_delay -clock core_clock 78 [get_ports {reset}]" in output
    assert "set_output_delay -clock core_clock 78 [get_ports {resp_msg}]" in output


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
        "timing_proxy_source_placement_budget_satisfied": True,
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
        "timing_proxy_source_placement_budget_satisfied": True,
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

    timing_improved = {
        "combined_score": 1.0,
        "parent_eligible": True,
        "timing_proxy_status": "success",
        "timing_proxy_wns_delta": 0.05,
        "timing_proxy_tns_delta": 0.5,
        "timing_proxy_tns_delta_pct": 10.0,
        "timing_proxy_source_placement_budget_satisfied": True,
    }
    apply_timing_proxy_policy(
        timing_improved,
        requested_mode="gate",
        wns_regression_gate_ns=0.2,
        tns_regression_pct_gate=10.0,
        tns_regression_min_abs_ns=0.05,
        wns_delta_min_abs_ns=0.05,
        selection_weight=0.1,
    )
    assert timing_improved["combined_score"] > 1.0
    assert timing_improved["timing_proxy_quality"] > 0.0
    assert timing_improved["timing_proxy_parent_signal"] == "soft_gate"

    timing_regressed = {
        "combined_score": 1.0,
        "parent_eligible": True,
        "timing_proxy_status": "success",
        "timing_proxy_wns_delta": -0.05,
        "timing_proxy_tns_delta": -0.01,
        "timing_proxy_tns_delta_pct": -5.0,
        "timing_proxy_source_placement_budget_satisfied": True,
    }
    apply_timing_proxy_policy(
        timing_regressed,
        requested_mode="gate",
        wns_regression_gate_ns=0.2,
        tns_regression_pct_gate=10.0,
        tns_regression_min_abs_ns=0.05,
        wns_delta_min_abs_ns=0.05,
        selection_weight=0.1,
    )
    assert timing_regressed["parent_eligible"] is True
    assert timing_regressed["combined_score"] < 1.0
    assert timing_regressed["timing_proxy_quality"] < 0.0

    noisy_tns = {
        "combined_score": 1.0,
        "parent_eligible": True,
        "timing_proxy_status": "success",
        "timing_proxy_wns_delta": 0.0,
        "timing_proxy_tns_delta": -0.01,
        "timing_proxy_tns_delta_pct": -100.0,
        "timing_proxy_source_placement_budget_satisfied": True,
    }
    apply_timing_proxy_policy(
        noisy_tns,
        requested_mode="gate",
        wns_regression_gate_ns=0.2,
        tns_regression_pct_gate=10.0,
        tns_regression_min_abs_ns=0.05,
    )
    assert noisy_tns["parent_eligible"] is True

    incomplete_source = {
        "combined_score": 1.0,
        "parent_eligible": True,
        "timing_proxy_status": "success",
        "timing_proxy_wns_delta": 1.0,
        "timing_proxy_tns_delta": 1.0,
        "timing_proxy_tns_delta_pct": 1.0,
        "timing_proxy_source_placement_budget_satisfied": False,
    }
    apply_timing_proxy_policy(
        incomplete_source,
        requested_mode="gate",
        wns_regression_gate_ns=0.2,
        tns_regression_pct_gate=10.0,
        tns_regression_min_abs_ns=0.05,
    )
    assert incomplete_source["parent_eligible"] is False
    assert incomplete_source["negative_memory_only"] is True
    assert incomplete_source["timing_proxy_parent_signal"] == "source_placement_incomplete"


def test_timing_proxy_status_normalization() -> None:
    assert normalize_timing_proxy_status("success") == "success"
    assert normalize_timing_proxy_status("missing_timing_collateral") == "missing_timing_collateral"
    assert normalize_timing_proxy_status("unexpected") == "unavailable"


def test_placements_from_tier2_comparison_filters_successful_defs(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.csv"
    with comparison.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "design",
                "objective_id",
                "seed",
                "status",
                "requested_iterations",
                "completed_iterations",
                "iteration_budget_satisfied",
                "output_artifact",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "design": "bp_fe",
                "objective_id": "default",
                "seed": "1000",
                "status": "success",
                "requested_iterations": "1000",
                "completed_iterations": "1000",
                "iteration_budget_satisfied": "true",
                "output_artifact": "/tmp/default.def",
            }
        )
        writer.writerow(
            {
                "design": "bp_fe",
                "objective_id": "cand",
                "seed": "1000",
                "status": "success",
                "requested_iterations": "1000",
                "completed_iterations": "1000",
                "iteration_budget_satisfied": "true",
                "output_artifact": "/tmp/cand.def",
            }
        )
        writer.writerow(
            {
                "design": "bp_fe",
                "objective_id": "failed",
                "seed": "1000",
                "status": "failed",
                "requested_iterations": "1000",
                "completed_iterations": "1000",
                "iteration_budget_satisfied": "true",
                "output_artifact": "/tmp/failed.def",
            }
        )
        writer.writerow(
            {
                "design": "bp_fe",
                "objective_id": "not_def",
                "seed": "1000",
                "status": "success",
                "requested_iterations": "1000",
                "completed_iterations": "1000",
                "iteration_budget_satisfied": "true",
                "output_artifact": "/tmp/not_def.txt",
            }
        )
        writer.writerow(
            {
                "design": "ethernet",
                "objective_id": "cand",
                "seed": "1000",
                "status": "success",
                "requested_iterations": "1000",
                "completed_iterations": "1000",
                "iteration_budget_satisfied": "true",
                "output_artifact": "/tmp/ethernet.def",
            }
        )
        writer.writerow(
            {
                "design": "bp_fe",
                "objective_id": "short_run",
                "seed": "1000",
                "status": "success",
                "requested_iterations": "50",
                "completed_iterations": "50",
                "iteration_budget_satisfied": "true",
                "output_artifact": "/tmp/short.def",
            }
        )

    output = placements_from_tier2_comparison(
        comparison,
        tmp_path / "placements.json",
        objective_ids={"cand", "short_run"},
        design_names={"bp_fe"},
        include_default=True,
        include_custom_default=False,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert [item["objective_id"] for item in payload["placements"]] == ["default", "cand"]
    assert all(item["requested_iterations"] == 1000 for item in payload["placements"])
