# Backend Setup

CoEvoP&R orchestrates four external execution paths. DREAMPlace performs analytical placement and executes validated objectives. The OpenTimer executable supplies placement-stage timing evidence. ChiPBench invokes OpenROAD for post-route evaluation. OpenROAD-flow-scripts supplies the ASAP7 platform and design collateral.

## Required roots

```bash
export COEVOP_ROOT=/path/to/CoEvoP-R
export DREAMPLACE_ROOT=/path/to/DREAMPlace/install
export CHIPBENCH_ROOT=/path/to/ChiPBench
export OPENROAD_FLOW_ROOT=/path/to/OpenROAD-flow-scripts
export PYTHONPATH="$COEVOP_ROOT"
```

Required files

```text
$DREAMPLACE_ROOT/dreamplace/Placer.py
$DREAMPLACE_ROOT/dreamplace/PlaceObj.py
$DREAMPLACE_ROOT/bin/ot-shell
$CHIPBENCH_ROOT/benchmarking/benchmarking.py
$CHIPBENCH_ROOT/flow/designs/nangate45/
$OPENROAD_FLOW_ROOT/flow/platforms/asap7/
```

## DREAMPlace

Apply the complete patch series to a DREAMPlace 4.0 source checkout, then build the installed runtime.

```bash
python3 -m coevop.cli dreamplace-apply-source-patches \
  --source-root /path/to/DREAMPlace
cd /path/to/DREAMPlace
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install \
  -DPython_EXECUTABLE="$(command -v python3)"
make -j"$(nproc)"
make install
export DREAMPLACE_ROOT=/path/to/DREAMPlace/install
cd "$COEVOP_ROOT"
python3 -m coevop.cli dreamplace-status
```

The patch binds the `typed_policy_v1` program fields to DREAMPlace terms, policy state, density preconditioning, smoothing, routing pressure, pin pressure, and optional timing-net weights. It also records scalar values, gradients, named components, and state trajectories.

## Placement-stage timing

`$DREAMPLACE_ROOT/bin/ot-shell` must execute successfully. The timing panel
references Liberty and SDC constraints from the matched ChiPBench flow. The
evaluator reconstructs the flat timing netlist from each placement DEF and
records a run-local manifest before invoking DREAMPlace's OpenTimer path.
Export the four-design SDC collateral with
`scripts/export_table1_timing_collateral.py` before running
`timing-proxy-eval` or `timing-proxy-audit`.

## OpenROAD and ChiPBench

Verify the executable and panel inputs.

```bash
command -v openroad
openroad -version
python3 -m coevop.cli shared-panel-check \
  --panel configs/shared_panels/chipbench_table1_post_route.toml \
  --run-dir runs/backend_check \
  --skip-runs
```

ChiPBench receives the DEF emitted by DREAMPlace and runs the configured OpenROAD flow. CoEvoP&R parses routed wirelength, routing overflow, WNS, TNS, DRC, power, and area from the produced metrics.

## ASAP7

The paper uses gcd, ibex, and ariane implemented with the ASAP7 standard-cell library. The expected design configuration paths are

```text
$CHIPBENCH_ROOT/flow/designs/asap7/gcd/config.mk
$CHIPBENCH_ROOT/flow/designs/asap7/ibex/config.mk
$CHIPBENCH_ROOT/flow/designs/asap7/ariane/config.mk
```

Run the dependency check before evaluation.

```bash
python3 -m coevop.cli asap7-panel-check \
  --platform-config configs/default.toml \
  --dreamplace-config-dir configs/dreamplace_base/asap7 \
  --run-dir runs/asap7_check
```

## Acceptance criteria

A backend installation is ready when the unit tests pass, the DREAMPlace patch is detected, the controller identity run records finite gradients and state updates, the timing panel reaches its configured net-coverage threshold, and the OpenROAD smoke run produces `comparison_table.csv` with finite routed metrics.
