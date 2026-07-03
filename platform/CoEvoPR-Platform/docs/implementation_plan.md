# CoEvoP&R Platform Implementation Plan

This file records the current implementation direction. The original v0 plan
used CircuitNet as a cheap Tier-1 scoring loop. That plan is now historical.

## Current Main Path: Direct Tier-2 Evolution

The primary platform path is direct DREAMPlace feedback on a shared
ChiPBench/Nangate45 design panel:

```text
LLM/OpenEvolve memory
-> restricted objective_program
-> ObjectiveSpec validation
-> batched DREAMPlace placement
-> HPWL/overflow/runtime/gradient scoring
-> local memory feedback
-> mutation/crossover/refinement
-> selected-finalist OpenROAD routing
```

CircuitNet is optional proxy analysis only.

## v1: Direct DREAMPlace Objective Evolution

Deliverables:

- OpenEvolve-style objective-program database.
- File-backed OpenEvolve prompt templates.
- Eureka-style objective environment in prompts.
- Restricted Python objective parser.
- DREAMPlace-deployable term scope.
- DREAMPlace search-panel evaluator over `design x objective x seed`.
- Baseline seeding from native default and fixed symbolic baselines.
- Parent selection from actual DREAMPlace metrics.
- Negative memory for structural failures and metric regressions.
- Evaluator-side rejection of baseline-clone objectives and fixed-baseline
  portfolio gating, so LLM candidates must beat more than a single easy
  baseline without being prompted with exact thresholds.

Primary command:

```bash
python3 -m coevop.cli openevolve-tier2 \
  --config configs/openevolve_tier2/chipbench_direct_tier2.toml \
  --run-dir runs/openevolve_tier2/chipbench_direct_tier2_v1 \
  --resume
```

## v2: Shared-Panel OpenROAD Validation

Run OpenROAD/ChiPBench only for selected finalists, not every inner-loop
candidate.

Inputs:

- DEF emitted by DREAMPlace candidate run.
- Shared ChiPBench design config.
- Objective ID and seed metadata from the Tier-2 run.

Outputs:

- global-route overflow;
- routed wirelength when detailed route completes;
- DRC/DRV count when available;
- WNS/TNS when timing is configured;
- power/area/runtime;
- failure stage and partial metrics on timeout.

## Optional: CircuitNet Proxy Analysis

CircuitNet can be used to justify term families and test cheap correlations,
but it is not the main selection loop.

Allowed uses:

- show that RUDY/pin/density observables relate to congestion labels;
- run shuffled-label and random-formula controls;
- inspect label sparsity and missing-label issues;
- provide background evidence in reports.

Disallowed as main evidence:

- claiming that high CircuitNet correlation proves placement improvement;
- selecting final objectives solely from CircuitNet correlation;
- mixing CircuitNet design-family selection with ChiPBench final claims without
  explicit domain-shift caveats.

## Repository Rule

Do not vendor CircuitNet, DREAMPlace, ChiPBench, or OpenROAD into this
repository. Keep external paths configurable and keep generated artifacts under
`runs/`.
