# Backend Setup Guide: DREAMPlace, OpenROAD, ChiPBench, and Optional CircuitNet

This repository is an orchestration layer. It does not vendor DREAMPlace,
OpenROAD, ChiPBench, or CircuitNet. A new user should be able to clone
`CoEvoPR-Platform`, install the external backends in a Linux-compatible
environment, set environment variables, and run the same checks described here.

WSL2 Ubuntu is acceptable. From the platform's point of view, WSL is simply the
Linux runtime.

## 1. Expected Directory Model

Recommended layout:

```bash
$HOME/CoEvoPR-Platform
$HOME/DREAMPlace/install
$HOME/ChiPBench
$HOME/datasets/CircuitNet/CircuitNet-N14   # optional proxy analysis only
```

The exact locations can differ, but the user must set:

```bash
export COEVOP_ROOT=$HOME/CoEvoPR-Platform
export DREAMPLACE_ROOT=$HOME/DREAMPlace/install
export CHIPBENCH_ROOT=$HOME/ChiPBench
export PYTHONPATH=$COEVOP_ROOT
```

Set `CIRCUITNET_ROOT` only if running optional CircuitNet proxy analysis:

```bash
export CIRCUITNET_ROOT=$HOME/datasets/CircuitNet/CircuitNet-N14
```

Committed configs use `${HOME}` paths where possible. For nonstandard paths,
either export the variables above or copy the TOML config and edit it locally.
Do not commit local absolute-path configs if they contain machine-specific paths.

## 2. Python Environment

Use Python 3.10 or newer. The verified WSL environment used Python 3.10.

```bash
cd "$COEVOP_ROOT"
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
python3 -m pytest
```

Expected result:

```text
All tests pass. The exact test count changes as the platform evolves.
```

If the user does not want a virtual environment, system Python works as long as
the package dependencies from `pyproject.toml` are installed.

## 3. Optional CircuitNet Proxy-Analysis Setup

CircuitNet is optional. The main CoEvoP&R evolution loop is direct
`openevolve-tier2` DREAMPlace feedback on ChiPBench/Nangate45 designs.
CircuitNet is used only for proxy analysis and term justification. It is not
used as an objective input and it is not stored in this repo.

Expected root:

```bash
echo "$CIRCUITNET_ROOT"
```

The local N14 structure should contain the routability/timing files needed by
the manifest builder. Verify:

```bash
python3 -m coevop.cli env-check
python3 -m coevop.cli circuitnet-manifest \
  --root "$CIRCUITNET_ROOT" \
  --output data_manifests/circuitnet_n14_real.local.json
```

The generated manifest is local data and is git-ignored by default when named
`*.local.json`.

Run a cheap proxy-analysis baseline check:

```bash
python3 -m coevop.cli tier1-offline \
  --manifest data_manifests/circuitnet_n14_real.local.json \
  --summary-output runs/proxy_analysis/circuitnet/scalars.csv \
  --ranking-output runs/proxy_analysis/circuitnet/rankings.json \
  --ranking-csv-output runs/proxy_analysis/circuitnet/rankings.csv
```

## 4. DREAMPlace Setup

### 4.1 Required DREAMPlace Structure

`DREAMPLACE_ROOT` must point to the install/runtime root containing:

```text
$DREAMPLACE_ROOT/dreamplace/Placer.py
$DREAMPLACE_ROOT/dreamplace/PlaceObj.py
$DREAMPLACE_ROOT/test/ispd2005/adaptec1.json
```

For the shared ChiPBench panel, it should also contain:

```text
$DREAMPLACE_ROOT/benchmarks/chipbench/bp_fe/bp_fe.json
```

Verify:

```bash
test -f "$DREAMPLACE_ROOT/dreamplace/Placer.py"
test -f "$DREAMPLACE_ROOT/dreamplace/PlaceObj.py"
python3 -m coevop.cli dreamplace-status
```

### 4.2 Apply The CoEvoP&R DREAMPlace Patch

The platform injects custom objectives through an external DREAMPlace patch.
The patch is tracked in this repo under:

```text
patches/dreamplace/custom_objective_placeobj.patch
```

Apply/check it:

```bash
cd "$COEVOP_ROOT"
python3 -m coevop.cli dreamplace-status
python3 -m coevop.cli dreamplace-apply-patch
python3 -m coevop.cli dreamplace-status
```

Expected status after patching:

```json
{
  "placeobj_patch_applied": true
}
```

The patch command expects `DREAMPLACE_ROOT` to be inside a git checkout, because
it uses `git apply`. If a user has only copied installed files and no git
metadata exists, they should either reinstall DREAMPlace from source or apply
the patch manually to `$DREAMPLACE_ROOT/dreamplace/PlaceObj.py`.

### 4.3 DREAMPlace Smoke Test

Native default placement:

```bash
python3 -m coevop.cli dreamplace-run \
  --run-name adaptec1_default_smoke \
  --base-config "$DREAMPLACE_ROOT/test/ispd2005/adaptec1.json" \
  --iterations 20 \
  --gpu 1
```

If no CUDA-capable GPU is available, use:

```bash
--gpu 0
```

Custom objective smoke:

```bash
python3 -m coevop.cli objective-preset \
  --name dreamplace_routing_aware_smoke \
  --output runs/objectives/dreamplace_routing_aware_smoke.json

python3 -m coevop.cli dreamplace-run \
  --objective runs/objectives/dreamplace_routing_aware_smoke.json \
  --run-name adaptec1_routing_aware_smoke \
  --base-config "$DREAMPLACE_ROOT/test/ispd2005/adaptec1.json" \
  --iterations 5 \
  --gpu 1 \
  --log-interval 1
```

The log should contain `CoEvoP&R custom objective` lines and a finite gradient
norm.

### 4.4 Deployable Term Gradient Check

Before using a term in claims, run:

```bash
python3 -m coevop.cli dreamplace-term-check \
  --base-config "$DREAMPLACE_ROOT/test/ispd2005/adaptec1.json" \
  --terms wirelength_wawl,wirelength_lse,density_electric,density_bell,soft_rudy_mean,soft_rudy_pnorm,route_pressure_long,pin_density_pnorm,pin_count_weighted_wl \
  --gpu 1 \
  --iterations 3 \
  --log-interval 1 \
  --run-dir runs/dreamplace_term_check/adaptec1_promoted
```

A promoted term must have:

- finite custom objective value,
- nonzero custom objective calls,
- finite gradient norm,
- no DREAMPlace crash.

## 5. OpenROAD And ChiPBench Setup

### 5.1 Required ChiPBench Structure

`CHIPBENCH_ROOT` must point to the ChiPBench checkout/root containing:

```text
$CHIPBENCH_ROOT/benchmarking/benchmarking.py
$CHIPBENCH_ROOT/flow/
$CHIPBENCH_ROOT/flow/designs/nangate45/bp_fe_top/config.mk
```

Verify:

```bash
test -f "$CHIPBENCH_ROOT/benchmarking/benchmarking.py"
test -f "$CHIPBENCH_ROOT/flow/designs/nangate45/bp_fe_top/config.mk"
```

OpenROAD must be available either through ChiPBench's own install or through
`PATH`:

```bash
command -v openroad
openroad -version || true
```

### 5.2 Writable Flow Runtime Directories

ChiPBench writes under:

```text
$CHIPBENCH_ROOT/flow/def_tmp
$CHIPBENCH_ROOT/flow/logs
$CHIPBENCH_ROOT/flow/objects
$CHIPBENCH_ROOT/flow/results
$CHIPBENCH_ROOT/flow/reports
```

These directories must be writable by the current Linux user. The CoEvoP&R
ChiPBench wrapper attempts to preserve and move aside non-writable runtime
directories by renaming them to `.coevop_readonly_backup_*` and recreating
writable directories. If this fails, fix permissions manually outside the repo.

Check:

```bash
for d in def_tmp logs objects results reports; do
  mkdir -p "$CHIPBENCH_ROOT/flow/$d"
  test -w "$CHIPBENCH_ROOT/flow/$d" || echo "not writable: $d"
done
```

### 5.3 Shared `bp_fe` Panel Data

The shared smoke panel is:

```text
configs/shared_panels/chipbench_smoke.toml
```

It expects:

```text
$DREAMPLACE_ROOT/benchmarks/chipbench/bp_fe/bp_fe.json
$CHIPBENCH_ROOT/flow/designs/nangate45/bp_fe_top/config.mk
$CHIPBENCH_ROOT/bp_fe_placed.def
```

Some DREAMPlace installs have `bp_fe.json` pointing to:

```text
benchmarks/chipbench/bp_fe/def/macro_placed.def
```

If that DEF is missing but ChiPBench has it, create a symlink:

```bash
mkdir -p "$DREAMPLACE_ROOT/benchmarks/chipbench/bp_fe/def"
ln -s "$CHIPBENCH_ROOT/dataset/data/bp_fe/def/macro_placed.def" \
  "$DREAMPLACE_ROOT/benchmarks/chipbench/bp_fe/def/macro_placed.def"
```

Only do this if the source file exists:

```bash
test -f "$CHIPBENCH_ROOT/dataset/data/bp_fe/def/macro_placed.def"
```

## 6. Backend Verification Commands

### 6.1 Static Environment Check

```bash
cd "$COEVOP_ROOT"
python3 -m coevop.cli env-check
python3 -m coevop.cli shared-panel-check \
  --panel configs/shared_panels/chipbench_smoke.toml \
  --run-dir runs/shared_panel_check/chipbench_smoke_static \
  --skip-runs
```

This checks files, OpenROAD visibility, and DREAMPlace patch status without
launching long placement/routing jobs.

### 6.2 DREAMPlace Runtime Check

```bash
python3 -m coevop.cli shared-panel-check \
  --panel configs/shared_panels/chipbench_smoke.toml \
  --run-dir runs/shared_panel_check/chipbench_smoke_runtime
```

This launches:

- one short DREAMPlace run on the first shared-panel design,
- one ChiPBench reference evaluation if `reference_def` is configured.

The ChiPBench reference route can be long. The `bp_fe` panel timeout is set to
2400 seconds by default.

## 7. Direct DREAMPlace Evolution Smoke Flow

After DREAMPlace and ChiPBench are configured:

```bash
python3 -m coevop.cli openevolve-tier2 \
  --config configs/openevolve_tier2/chipbench_direct_tier2.toml \
  --run-dir runs/openevolve_tier2/chipbench_direct_tier2_v1 \
  --resume
```

Default behavior:

- evaluates seed baselines through real DREAMPlace first,
- prompts the LLM using OpenEvolve-style local memory,
- validates generated objective programs against the restricted DSL,
- runs DREAMPlace on the shared ChiPBench/Nangate45 search panel,
- feeds HPWL/overflow/runtime/gradient/failure feedback into later prompts,
- keeps Tier-3 disabled unless selected-finalist routing is requested.

To run Tier-3, set:

```toml
[tier3]
enabled = true
```

Then rerun `openevolve-tier2`, or call `tier3-openroad` directly on the
selected-finalist placements:

```bash
python3 -m coevop.cli tier3-openroad \
  --panel configs/shared_panels/chipbench_smoke.toml \
  --placements runs/openevolve_tier2/<run>/tier3_selected_placements.json \
  --run-dir runs/openevolve_tier2/<run>/tier3_openroad \
  --resume
```

For selected-finalist routing feedback without full detailed routing, use the
global-route-only panel:

```bash
python3 -m coevop.cli tier3-openroad \
  --panel configs/shared_panels/chipbench_tier3_bpfe_grt_only.toml \
  --placements runs/openevolve_tier2/<run>/tier3_selected_placements.json \
  --run-dir runs/openevolve_tier2/<run>/tier3_openroad_grt_only \
  --baseline-objective-id obj_90b92426fda2542f \
  --resume
```

This path runs ChiPBench/OpenROAD through `do-grt` and records global-route
feedback. It is not a substitute for final routed PPA because detailed routing,
DRC, and final routed wirelength are not produced.

The `chipbench_tier3_bpfe_grt_only.toml` panel sets:

```toml
global_route_args = "-allow_congestion -verbose -congestion_iterations 5"
```

This overrides the `bp_fe_top` default of 50 GRT congestion iterations for
selected-finalist feedback. Use the full-route panels when final routed PPA is
needed.

## 8. LLM Provider Setup

Mock provider requires no key:

```bash
export LLM_PROVIDER=mock
```

OpenAI provider:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4.1-mini
export LLM_PROVIDER=openai
```

Qwen/DashScope provider:

```bash
export DASHSCOPE_API_KEY=...
export QWEN_MODEL=qwen-max
export LLM_PROVIDER=qwen
```

Do not store API keys in the repo. Local key files are for private use only and
must stay outside Git.

## 9. What A New User Can Run

Immediately after clone and Python setup:

```bash
python3 -m pytest
python3 -m coevop.cli llm-check
```

After DREAMPlace setup and patch:

```bash
python3 -m coevop.cli dreamplace-status
python3 -m coevop.cli dreamplace-term-check ...
python3 -m coevop.cli openevolve-tier2 \
  --config configs/openevolve_tier2/chipbench_direct_tier2.toml \
  --run-dir runs/openevolve_tier2/chipbench_direct_tier2_v1 \
  --resume
```

After ChiPBench/OpenROAD setup:

```bash
python3 -m coevop.cli shared-panel-check ...
python3 -m coevop.cli tier3-openroad ...
```

After optional CircuitNet setup:

```bash
python3 -m coevop.cli circuitnet-manifest ...
python3 -m coevop.cli tier1-offline ...
```

## 10. Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `PlaceObj patch applied: false` | DREAMPlace patch not applied | Run `python3 -m coevop.cli dreamplace-apply-patch` |
| `git rev-parse` error during patch | `DREAMPLACE_ROOT` is not inside a git checkout | Reinstall DREAMPlace from source or apply the patch manually |
| `Could not open input file ... macro_placed.def` | DREAMPlace benchmark JSON points to missing DEF | Create the symlink described in Section 5.3 |
| OpenROAD not found | `openroad` is not on `PATH` | Add OpenROAD binary directory to `PATH` or use ChiPBench's bundled OpenROAD |
| `Permission denied` under `flow/logs`, `flow/objects`, etc. | ChiPBench runtime dirs are root-owned | Let the wrapper move them aside, or fix ownership/permissions manually |
| `ChiPBench timeout` | The routing/evaluation flow is long | Increase `timeout_seconds` in the panel config |
| `Tier-2 accepts deployable terms only` | Candidate uses CircuitNet-only proxy terms | Rerun direct evolution with `term_scope = "dreamplace_replacement"` |
| `custom objective did not execute` | DREAMPlace config did not include `custom_objective_spec` or patch is missing | Check generated `dreamplace_config.json` and patch status |
| No DEF in Tier-2 output | DREAMPlace placement completed without routable DEF artifact | Use `--keep-legalization` and a design/config that emits DEF |

## 11. Reproducibility Checklist

Before sharing results, record:

- `git rev-parse HEAD` for `CoEvoPR-Platform`,
- DREAMPlace commit/hash or source version,
- OpenROAD version,
- ChiPBench commit/hash or source version,
- Docker/WSL/Linux version,
- GPU model and CUDA version if GPU is used,
- optional `CIRCUITNET_ROOT`, manifest hash, and design list when proxy analysis is used,
- panel TOML,
- ObjectiveSpec JSON files,
- seeds,
- `summary.json`, `metrics.csv`, `comparison_table.csv`, and SQLite stores.

The platform writes most run artifacts under `runs/`; this directory is
git-ignored and should be archived separately for experiment records.
