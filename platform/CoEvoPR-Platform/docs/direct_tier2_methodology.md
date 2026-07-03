# Direct Tier-2 Methodology

This document describes the current main CoEvoP&R methodology.

## Decision

The core evolution loop no longer uses CircuitNet as Tier-1 fitness.

The main loop is direct DREAMPlace placement feedback on a shared
Nangate45/ChiPBench panel:

```text
LLM/OpenEvolve memory
-> sample one parent from an island/MAP-Elites cell
-> generate K independent restricted objective_program samples
-> parse and validate ObjectiveSpec
-> run DREAMPlace over design x objective x seed
-> score HPWL, overflow, runtime, gradient stability, failures, and component traces
-> store metrics and artifacts in local memory
-> mutate from top, diverse, failed, and component-feedback programs
-> periodically migrate eligible programs across islands
-> route selected finalists with OpenROAD/ChiPBench
```

The primary command is:

```bash
python3 -m coevop.cli openevolve-tier2 \
  --config configs/openevolve_tier2/chipbench_direct_tier2.toml \
  --run-dir runs/openevolve_tier2/chipbench_direct_tier2_v1 \
  --resume
```

## Rationale

CircuitNet is a passive labeled dataset. It can be useful for showing that a
placement-state observable is related to congestion, but it does not measure
what happens when DREAMPlace optimizes a generated objective.

The key failure modes are:

- High CircuitNet correlation is not sufficient. A congestion proxy can rank
  samples correctly while producing bad gradients and large HPWL regression
  inside DREAMPlace.
- High CircuitNet correlation is not necessary. A good objective may work
  through optimizer trajectory effects that are invisible in static CircuitNet
  rows.
- The serious validation panel is ChiPBench/Nangate45. Using CircuitNet design
  families for selection and ChiPBench designs for validation creates a domain
  discontinuity.

Therefore CircuitNet is retained as optional proxy analysis, not the main
optimization target.

## What The LLM Sees

The prompt follows the OpenEvolve outer structure and the Eureka inner
environment-code style.

The LLM receives:

- current parent objective program;
- current fitness and feature coordinates;
- top programs;
- diverse inspiration programs;
- recent failures and severe regressions;
- DREAMPlace execution metrics;
- router/objective background;
- exact restricted Python objective API;
- allowed DREAMPlace-deployable terms.

The API is stateless. The platform explicitly inserts memory into every prompt
and saves:

- `prompt_context.json`
- `prompt_messages.json`
- `raw_response.json`
- objective JSON
- DREAMPlace metrics and logs
- evolution trace JSONL
- program database checkpoints

## OpenEvolve/Eureka Mechanisms Implemented

The current `openevolve-tier2` path implements these mechanisms explicitly:

- **Eureka-style component feedback.** The restricted objective returns named
  components. The DREAMPlace patch logs component time series, and the parser
  summarizes each component with start/mid/end/min/max/mean/trend and
  flat-or-saturated status. Later prompts receive compact component feedback so
  the LLM can rescale, remove, or redesign components that were ineffective.
- **K independent samples per iteration.** One parent/context can generate
  multiple independent children. Provider retries are separate from samples:
  retries repair invalid output, while samples are distinct objective proposals.
- **Cell-based MAP-Elites memory.** The archive is derived from island-local
  feature cells over complexity, mechanism signature, HPWL delta, overflow
  delta, and gradient stability. A cell keeps only the best program by measured
  evaluator fitness.
- **Island migration.** Islands are behavioral subpopulations. At configured
  intervals, eligible programs migrate through a ring topology and are inserted
  through the MAP-Elites cell discipline.
- **CoEvoPR-specific archetype memory.** In addition to OpenEvolve-style memory,
  the platform summarizes domain-specific mechanism families such as route
  pressure, pin access, and alternative density/wirelength smoothing.

## Objective Contract

The model must produce one JSON object containing:

- `id`
- `rationale`
- `parent_ids`
- `declared_term_usage`
- `change_description`
- `objective_program`

`change_description` uses OpenEvolve-style SEARCH/REPLACE format. The
`objective_program` field contains the complete restricted Python function after
the edit.

Example:

```python
def objective(features):
    wl = term("wirelength_wawl")
    den = term("density_electric")
    rudy = term("soft_rudy_pnorm")
    pins = term("pin_density_pnorm")
    route = log1p(rudy)
    pin = sqrt(pins)
    score = wl + den + 0.01 * route + 0.005 * pin
    return score, {
        "wirelength": wl,
        "density": den,
        "route_hotspot": route,
        "pin_access": pin,
    }
```

The platform parses the AST and never executes arbitrary generated code.

## Main DREAMPlace Terms

For `term_scope = "dreamplace_replacement"`, the current allowed terms are:

- `wirelength`, alias for `wirelength_wawl`
- `wirelength_wawl`
- `wirelength_lse`
- `density`, alias for `density_electric`
- `density_electric`
- `density_bell`
- `soft_rudy_mean`
- `soft_rudy_pnorm`
- `route_pressure_long`
- `pin_density_pnorm`
- `pin_count_weighted_wl`

`native_objective`, `density_pnorm`, and `timing_weighted_wirelength` are not
allowed in this LLM-facing mode. Native DREAMPlace is kept as the external
baseline `default`, not as a term the model can simply wrap.

Static validation rejects mixed wirelength families
(`wirelength_lse` together with `wirelength_wawl` or `wirelength`) and mixed
density families (`density_bell` together with `density_electric` or
`density`). CircuitNet-only terms are not allowed in this mode. Current direct
configs also reject generated candidates that repeat a fixed baseline mechanism
with only coefficient changes.

## Fitness Policy

The evaluator, not the prompt, decides whether a tradeoff is good.

The direct Tier-2 config uses native DREAMPlace `default` as the primary
reported baseline:

```toml
primary_baseline = "default"
```

The `custom_default` identity path is still reported as a normalization control.
Generated candidates are also ranked against the full fixed-baseline portfolio
evaluated in the same search batch. Parent selection can require that a
generated candidate beat that portfolio, which prevents a candidate from looking
good only because it beat a weak single baseline.

The fixed-baseline portfolio must include simple explicit term-family baselines,
not only the native default. In particular, it includes:

- `dreamplace_wawl_electric`: WAWL wirelength plus electric density;
- `dreamplace_wawl_density_bell`: WAWL wirelength plus bell-shaped density;
- `dreamplace_lse_density_bell`: LSE wirelength plus bell-shaped density;
- density-heavy and routing-aware smoke baselines.

The `dreamplace_wawl_density_bell` guard is important. A June 26, 2026
three-design guard run showed that the then-current best generated objective

```python
score = wirelength_wawl + density_bell - 1e-7 * sigmoid(pin_density_pnorm * soft_rudy_pnorm)
```

still beat native default, but did not beat the simpler
`wirelength_wawl + density_bell` baseline under the same legalized DREAMPlace
settings. Therefore future generated objectives should not be considered
methodologically successful unless they beat this closer simple baseline as well
as native default. This is an evaluator-side requirement; it should not be
rewritten into the LLM prompt as a leading instruction.

Reports distinguish two levels of success:

- `beats_baseline_portfolio`: the generated candidate wins the evaluator's
  rank-based fixed-baseline gate.
- `baseline_portfolio_pareto_label_vs_best`: the stricter interpretation of
  whether the candidate Pareto-dominates, trades off against, ties, or regresses
  relative to the strongest fixed baseline in that same batch.

For meeting and paper claims, rank-based wins should be described as tradeoffs
unless `baseline_portfolio_pareto_label_vs_best` is
`dominates_best_baseline`.

The current direct configs do not use density or wirelength coefficient grids as
the main search mechanism. Those grids are fixed to identity. Calibration
siblings are used to test routing-mechanism scale, while fixed density-heavy
objectives are evaluated as baselines.

Metrics include:

- HPWL delta;
- overflow delta;
- runtime;
- objective calls;
- custom gradient norm;
- failure stage;
- structural failure count;
- severe regression count.

Completed regressions are retained as negative memory. Structural failures are
not eligible as parents.

Exact HPWL gates, coefficient grids, and portfolio-gate booleans are evaluator
configuration, not prompt instructions. The LLM sees the placement environment,
allowed terms, previous objective programs, and measured feedback. This keeps the
workflow closer to OpenEvolve/Eureka and avoids prompting the model to copy a
threshold or clone the native objective.

## OpenROAD Usage

OpenROAD is too expensive for every inner-loop candidate. The intended flow is:

```text
many candidates: DREAMPlace Tier-2 only
top finalists: OpenROAD global-route or full-route evaluation
paper evidence: completed same-design DREAMPlace-to-OpenROAD runs
```

Global-route-only feedback can guide search but does not replace final routed
PPA evidence.

## CircuitNet's Remaining Role

CircuitNet can still be used for:

- validating that observables such as RUDY and pin pressure relate to congestion
  labels;
- checking negative controls;
- producing cheap background evidence for term justification;
- experimenting with proxy scoring outside the main loop.

It should not be used as the primary parent-selection or final claim metric.
