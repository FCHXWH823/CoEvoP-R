# CoEvoPR Platform

CoEvoPR Platform is the implementation repository for CoEvoP&R: LLM-driven
discovery of symbolic, differentiable, routing-aware placement objective
functions.

The platform is an orchestrator. It does not vendor DREAMPlace, OpenROAD,
ChiPBench, CircuitNet, or API keys. It provides the objective language, LLM
provider adapters, OpenEvolve-style memory loop, DREAMPlace/OpenROAD runners,
safety checks, result storage, and reports needed to connect those external
backends into one reproducible research workflow.

Target research direction: publishable EDA methodology for routing-aware
placement objective discovery.

## Repository Hygiene

This repo is intended to be publishable source code, not a run-artifact bundle.
Keep machine-specific paths in ignored `*.local.toml` files and set external
roots such as `DREAMPLACE_ROOT`, `CHIPBENCH_ROOT`, `CIRCUITNET_ROOT`, and LLM
API keys through the environment. Generated `runs/`, datasets, databases,
logs, and external tool checkouts are intentionally excluded from Git.

## Current Methodology

The main CoEvoP&R optimization loop is now direct DREAMPlace feedback on a
shared Nangate45/ChiPBench panel:

```text
LLM/OpenEvolve memory
-> generate restricted objective_program / ObjectiveSpec
-> validate symbolic objective
-> batch DREAMPlace placement over design x objective x seed
-> score actual HPWL, overflow, runtime, gradients, and failures
-> store execution feedback in local memory
-> mutate from top/diverse/failed programs
-> route only selected finalists with OpenROAD/ChiPBench
```

CircuitNet is not the main fitness loop. It is retained only as optional proxy
analysis for term justification and cheap sanity checks. A CircuitNet
correlation result is not treated as proof that an objective improves placement
or routed PPA.

The primary evolution command is:

```bash
python3 -m coevop.cli openevolve-tier2 \
  --config configs/openevolve_tier2/chipbench_direct_tier2.toml \
  --run-dir runs/openevolve_tier2/chipbench_direct_tier2_v1 \
  --resume
```

## Why CircuitNet Is Optional

CircuitNet labels are passive observations of existing placements. They can show
that terms such as RUDY, pin pressure, or density hotspots correlate with
congestion labels, but they do not prove that optimizing a formula inside
DREAMPlace improves placement.

Two implications drive the current design:

- A high CircuitNet correlation is not sufficient. A formula can rank congested
  samples correctly but create bad DREAMPlace gradients and destroy HPWL.
- A good DREAMPlace objective is not guaranteed to have high CircuitNet
  correlation. It may work because of trajectory-level optimizer behavior that
  static CircuitNet rows do not measure.

Therefore, CircuitNet tools remain in the repo as `proxy_analysis` utilities,
but parent selection and elite promotion for the main research loop should come
from actual DREAMPlace execution on the shared ChiPBench panel.

## Implemented Capabilities

- Restricted Eureka-style objective programs lowered into safe ObjectiveSpec
  JSON.
- Provider adapters for `mock`, OpenAI/GPT-family models, and Qwen/DashScope.
- OpenEvolve-style Tier-2 memory loop with parent-child lineage, islands,
  top/diverse/failed program memory, checkpoints, prompt logs, raw responses,
  and DREAMPlace artifacts.
- DREAMPlace custom objective patch and `design x objective x seed` evaluator.
- Shared ChiPBench panel support for same-design DREAMPlace-to-OpenROAD
  validation.
- Tier-3 ChiPBench/OpenROAD evaluator with final metrics parsing and partial
  metric recovery when routing times out.
- Optional CircuitNet manifest/scalar/proxy-analysis tooling.

## Objective API

The LLM does not place circuits directly and does not receive labels/reports as
objective inputs. It writes a restricted symbolic objective over placement-state
observables:

```python
def objective(features):
    wl = term("wirelength_wawl")
    den = term("density_electric")
    route = term("soft_rudy_pnorm")
    pins = term("pin_density_pnorm")
    route_cost = log1p(route)
    pin_cost = sqrt(pins)
    score = wl + den + 0.01 * route_cost + 0.005 * pin_cost
    return score, {
        "wirelength": wl,
        "density": den,
        "route_hotspot": route,
        "pin_access": pins,
    }
```

The platform parses this program with Python's AST module, validates it against
a restricted grammar, and lowers it into ObjectiveSpec JSON. Arbitrary
model-generated Python is never executed. Imports, loops, branches,
comprehensions, attributes, direct `features[...]` access, unknown calls,
randomness, and statements after `return` are rejected.

Allowed expression operators:

```text
+  -  *  safe_div  log1p  sqrt  square  softplus  sigmoid
```

Current limits:

```text
max AST depth = 5
max primitive count = 12
max diagnostic components = 6
```

## Main Evolution Path

### 1. Backend Check

```bash
python3 -m coevop.cli env-check
python3 -m coevop.cli llm-check
python3 -m coevop.cli dreamplace-status
python3 -m coevop.cli shared-panel-check \
  --panel configs/shared_panels/chipbench_search_tier2_multidesign.toml \
  --run-dir runs/shared_panel_check/chipbench_search_static \
  --skip-runs
```

### 2. Direct Tier-2 Evolution

```bash
python3 -m coevop.cli openevolve-tier2 \
  --config configs/openevolve_tier2/chipbench_direct_tier2.toml \
  --run-dir runs/openevolve_tier2/chipbench_direct_tier2_v1 \
  --resume
```

This run:

- evaluates initial baselines by real DREAMPlace placement first;
- prompts the LLM with OpenEvolve-style memory and Eureka-style environment
  code;
- requires each generated candidate to be a DREAMPlace-deployable objective;
- rejects default-like generated candidates before wasting DREAMPlace runtime;
- evaluates candidates on the search panel using DREAMPlace metrics;
- compares generated candidates against the evaluated fixed-baseline portfolio,
  not only against one handpicked baseline;
- stores successful candidates, metric regressions, and failures in local
  memory;
- selects parents from candidates that satisfy the configured policy;
- optionally runs a final multi-design/multi-seed Tier-2 panel.

### 2b. Design-Specific Evolution With Held-Out Generalization

Use this path when the research question is: "Can the LLM evolve an objective
specialized to one chip design, and does that objective transfer to other
circuits?"

```bash
python3 -m coevop.cli openevolve-tier2 \
  --config configs/openevolve_tier2/chipbench_design_specific_generalization_gpt41mini.toml \
  --run-dir runs/openevolve_tier2/bpfe_design_specific_generalization_v1 \
  --resume
```

The configured split is:

- search/evolution target: `bp_fe`;
- held-out generalization panel: `ethernet`, `isa_npu`;
- feedback boundary: held-out generalization metrics are produced after
  evolution and are not inserted into LLM prompts or parent selection.

The LLM prompt now includes a chip-specific design profile, saved as
`design_profiles.json` in the run directory and embedded in every
`prompt_context.json`. This profile is derived from the shared panel,
DREAMPlace config, and ChiPBench config. It includes the optimization design,
held-out design names, mode/timing settings, selected physical config fields
such as target density, bin counts, core/die geometry when available, declared
input-file summaries, and search-panel baseline behavior after the initial
DREAMPlace baseline run. Held-out generalization metrics are deliberately not
included in this prompt profile.

Outputs are written under:

- `iteration_*/tier2_search/` for design-specific search feedback;
- `generalization_tier2/` for held-out DREAMPlace transfer results;
- `generalization_tier2/generalization_report.md` for a concise transfer
  summary.

### 3. Selected-Finalist Routing

Run OpenROAD only for native default, fixed baselines, and top Tier-2 finalists:

```bash
python3 -m coevop.cli tier3-openroad \
  --panel configs/shared_panels/chipbench_tier3_bpfe_grt_only.toml \
  --placements runs/openevolve_tier2/<run>/tier3_selected_placements.json \
  --run-dir runs/openevolve_tier2/<run>/tier3_openroad_grt_only \
  --baseline-objective-id default \
  --resume
```

Global-route-only results are feedback about global routing congestion. Final
routed PPA claims require full OpenROAD detailed routing to complete.

## Objective Terms

The LLM can only use terms exposed by the selected scope. For the main direct
Tier-2 loop, `term_scope = "dreamplace_replacement"` exposes these
DREAMPlace-side differentiable terms:

- `wirelength`, alias for `wirelength_wawl`
- `wirelength_wawl`, DREAMPlace weighted-average smooth wirelength
- `wirelength_lse`, fixed-gamma log-sum-exp smooth wirelength
- `density`, alias for `density_electric`
- `density_electric`, raw electric-density penalty
- `density_bell`, bell-shaped density-potential penalty
- `soft_rudy_mean`, differentiable route-demand mean
- `soft_rudy_pnorm`, differentiable route-demand hotspot proxy
- `route_pressure_long`, long-net route-pressure hotspot proxy
- `pin_density_pnorm`, differentiable pin-access hotspot proxy
- `pin_count_weighted_wl`, WAWL with static sqrt(pin-count) net weighting

`native_objective`, `density_pnorm`, and `timing_weighted_wirelength` are not
LLM-facing in `dreamplace_replacement`. Native DREAMPlace remains the evaluator
baseline `default`; legacy/internal scopes can still read older artifacts that
used `native_objective`.

Static validation rejects candidates that mix WAWL and LSE wirelength terms, and
also rejects candidates that mix electric and bell density terms. CircuitNet-only
proxy terms remain available for optional proxy analysis, but they are not
allowed in direct Tier-2 objectives unless a matching differentiable DREAMPlace
implementation exists.

For current OpenEvolve Tier-2 configs, generated candidates are also rejected if
they are structurally indistinguishable from fixed baselines with only coefficient
changes. The prompt does not expose exact gate thresholds; those checks live in
the evaluator and appear to the LLM only as measured feedback from prior runs.
Calibration siblings currently sweep routing-mechanism coefficients only; base
wirelength and density coefficient grids are fixed at identity in official
direct configs to avoid rediscovering handcrafted density tuning.

## Repository Layout

```text
coevop/
  backends/       DREAMPlace backend wrappers
  datasets/       optional CircuitNet manifest and feature/label loading
  eval/           DREAMPlace, OpenROAD, ranking, proxy-analysis, and legacy runners
  evolution/      OpenEvolve-style memory plus optional offline proxy evolution
  llm/            provider adapters and prompt construction
  objectives/     objective specs, DSL, restricted programs, terms, presets
  prompt/         OpenEvolve-style template loader and sampler
  prompts/        package prompt templates
  store/          SQLite-backed metadata helpers
configs/
  openevolve_tier2/   direct DREAMPlace evolution configs
  shared_panels/      shared DREAMPlace/OpenROAD panels
  dreamplace_panels/  DREAMPlace smoke panels
  end_to_end/         legacy CircuitNet-to-DREAMPlace configs
docs/
  direct_tier2_methodology.md
  backend_setup.md
  circuitnet_proxy_analysis.md
patches/
  dreamplace/
prompts/
  router_objective_background.md
scripts/
tests/
```

## Backend Requirements

Recommended Linux/WSL layout:

```bash
$HOME/CoEvoPR-Platform
$HOME/DREAMPlace/install
$HOME/ChiPBench
$HOME/datasets/CircuitNet/CircuitNet-N14   # optional proxy analysis only
```

Environment variables:

```bash
export COEVOP_ROOT=$HOME/CoEvoPR-Platform
export DREAMPLACE_ROOT=$HOME/DREAMPlace/install
export CHIPBENCH_ROOT=$HOME/ChiPBench
export PYTHONPATH=$COEVOP_ROOT
```

CircuitNet is optional for the main flow:

```bash
export CIRCUITNET_ROOT=$HOME/datasets/CircuitNet/CircuitNet-N14
```

Read [docs/backend_setup.md](docs/backend_setup.md) before DREAMPlace or
OpenROAD runs.

## Installation

```bash
cd "$COEVOP_ROOT"
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .[dev]
python3 -m pytest
```

Current verified local status:

```text
Windows: 220 passed
WSL:     run before paper-scale DREAMPlace/OpenROAD experiments
```

## LLM Providers

Mock provider:

```bash
export LLM_PROVIDER=mock
```

OpenAI/GPT-family provider:

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

Use a model your API account can access. ChatGPT Plus membership and API billing
are separate. Never commit API keys or local key files.

## Optional CircuitNet Proxy Analysis

CircuitNet tools are still available for cheap term analysis:

```bash
python3 -m coevop.cli circuitnet-manifest \
  --root "$CIRCUITNET_ROOT" \
  --output data_manifests/circuitnet_n14_real.local.json

python3 -m coevop.cli tier1-offline \
  --manifest data_manifests/circuitnet_n14_real.local.json \
  --summary-output runs/proxy_analysis/circuitnet/scalars.csv \
  --ranking-output runs/proxy_analysis/circuitnet/rankings.json \
  --ranking-csv-output runs/proxy_analysis/circuitnet/rankings.csv
```

Use this only as proxy evidence. Do not use it as the main fitness claim.

## Reproducibility Checklist

For every serious experiment, archive:

- CoEvoPR Platform commit hash.
- DREAMPlace commit/hash or install version.
- OpenROAD version.
- ChiPBench commit/hash or install version.
- Linux/WSL/Docker environment.
- GPU model, CUDA version, and GPU setting.
- shared panel TOML files.
- ObjectiveSpec JSON files.
- seeds.
- prompt messages, raw responses, `summary.json`, `metrics.csv`,
  `comparison_table.csv`, reports, logs, DEF artifacts, and SQLite stores.

Run artifacts under `runs/` are intentionally git-ignored.

## Documentation

- [Direct Tier-2 methodology](docs/direct_tier2_methodology.md)
- [Backend setup](docs/backend_setup.md)
- [CircuitNet proxy analysis](docs/circuitnet_proxy_analysis.md)
- [DREAMPlace patch notes](patches/dreamplace/README.md)

## Development Notes

Run tests before sharing changes:

```bash
python3 -m pytest
```

Do not commit:

- API keys.
- `runs/`.
- CircuitNet data.
- DREAMPlace/OpenROAD/ChiPBench checkouts.
- local manifests ending in `.local.json`.
- SQLite databases and large logs.
