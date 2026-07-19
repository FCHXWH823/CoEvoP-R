#!/usr/bin/env python3
"""Evaluate one DEF with DREAMPlace's OpenTimer RC path without moving cells."""

from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

# DREAMPlace still uses the NumPy 1.x string alias while current environments
# may provide NumPy 2.x. Restore the equivalent bytes scalar before importing
# DREAMPlace modules so fixed-position timing evaluation remains portable.
if not hasattr(np, "string_"):
    np.string_ = np.bytes_  # type: ignore[attr-defined]

# The driver runs with cwd set to the DREAMPlace install root. Unlike
# dreamplace/Placer.py, this external script is not located inside that module
# directory, so add it explicitly before importing DREAMPlace's flat modules.
dreamplace_module_dir = Path.cwd() / "dreamplace"
if str(dreamplace_module_dir) not in sys.path:
    sys.path.insert(0, str(dreamplace_module_dir))

import NonLinearPlace
import Params
import PlaceDB
import PlaceObj
import Timer


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: dreamplace_opentimer_fixed_def.py CONFIG.json")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)-7s] %(name)s - %(message)s")
    params = Params.Params()
    params.load(sys.argv[1])
    if not getattr(params, "timing_opt_flag", 0):
        raise RuntimeError("fixed-position OpenTimer driver requires timing_opt_flag=1")
    if getattr(params, "timer_engine", "").lower() != "opentimer":
        raise RuntimeError("fixed-position timing driver currently supports OpenTimer only")

    torch.set_num_threads(max(1, int(getattr(params, "num_threads", 1))))
    np.random.seed(params.random_seed)
    started = time.time()
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    print("CoEvoP&R fixed-position stage=placedb_ready", flush=True)
    timer = Timer.Timer(timer_engine="opentimer")
    timer(params, placedb)
    print("CoEvoP&R fixed-position stage=timer_parsed", flush=True)
    timer.update_timing()
    print("CoEvoP&R fixed-position stage=initial_timing_updated", flush=True)
    placer = NonLinearPlace.NonLinearPlace(params, placedb, timer)
    print("CoEvoP&R fixed-position stage=placement_ops_ready", flush=True)
    pos_device = placer.pos[0].data.detach().clone()
    pos = pos_device.cpu()
    jitter = float(getattr(params, "timing_proxy_position_jitter", 0.0))
    if jitter > 0:
        movable = int(placedb.num_movable_nodes)
        node_count = int(placedb.num_nodes)
        indices = torch.arange(movable, dtype=pos.dtype)
        # A deterministic sub-site perturbation avoids exact coordinate ties in
        # the RC-tree copy. Reported HPWL and placement coordinates are unchanged.
        x_jitter = (torch.remainder(indices * 0.61803398875, 1.0) - 0.5) * jitter
        y_jitter = (torch.remainder(indices * 0.41421356237, 1.0) - 0.5) * jitter
        pos[:movable] += x_jitter
        pos[node_count : node_count + movable] += y_jitter
        print(f"CoEvoP&R fixed-position position_jitter={jitter:.9g}", flush=True)
    timing_op = placer.op_collections.timing_op
    timing_op(pos)
    print("CoEvoP&R fixed-position stage=rc_tree_ready", flush=True)
    timer.update_timing()
    print("CoEvoP&R fixed-position stage=placement_timing_updated", flush=True)

    time_unit = timer.time_unit()
    print("CoEvoP&R fixed-position stage=time_unit_ready", flush=True)
    tns = timer.report_tns_elw(split=1) / (time_unit * 1e17)
    print("CoEvoP&R fixed-position stage=tns_ready", flush=True)
    wns = timer.report_wns(split=1) / (time_unit * 1e15)
    print("CoEvoP&R fixed-position stage=wns_ready", flush=True)
    # Placement operators live on the configured device. Keep their input on
    # that device while the RC-tree construction uses the CPU copy above.
    hpwl = float(placer.op_collections.hpwl_op(pos_device).detach().cpu().item())
    if placer.op_collections.density_overflow_op is None:
        stages = getattr(params, "global_place_stages", None) or []
        if not stages:
            raise RuntimeError("fixed-position overflow evaluation needs a global placement stage")
        density_model = PlaceObj.PlaceObj(
            0.0,
            params,
            placedb,
            placer.data_collections,
            placer.op_collections,
            stages[0],
        ).to(pos_device.device)
        if density_model.op_collections.density_overflow_op is None:
            raise RuntimeError("failed to initialize DREAMPlace density-overflow operator")
    overflow_raw, max_density_raw = placer.op_collections.density_overflow_op(pos_device)
    overflow = float(
        (overflow_raw / float(placedb.total_movable_node_area)).detach().cpu().item()
    )
    max_density = float(max_density_raw.detach().cpu().item())
    if not all(math.isfinite(value) for value in (hpwl, overflow, max_density, wns, tns)):
        raise RuntimeError(
            "non-finite fixed-position result: "
            f"HPWL={hpwl}, Overflow={overflow}, MaxDensity={max_density}, "
            f"WNS={wns}, TNS={tns}"
        )
    logging.info(
        "CoEvoP&R fixed-position OpenTimer HPWL %.9g, Overflow %.9g, "
        "MaxDensity %.9g, TNS %.9g, WNS %.9g, runtime %.3fs",
        hpwl,
        overflow,
        max_density,
        tns,
        wns,
        time.time() - started,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
