# Router-Aware Objective Background

CoEvoP&R searches for differentiable placement objectives that improve downstream placement and routing behavior using observables available during placement. The LLM should not use routed labels as direct objective inputs. It receives only placement-state observables that can be computed during placement and then receives DREAMPlace/OpenROAD feedback after evaluation.

## Routing Model

OpenROAD routing decomposes the problem into global routing and detailed routing. Global routing estimates demand on coarse routing resources; detailed routing assigns exact tracks and vias. A placement is easier to route when routing demand is spatially balanced, pin access is not too concentrated, and large fixed blocks or macros do not force many nets through narrow channels.

For a routing edge or bin, overflow is:

```text
overflow = max(0, demand - capacity)
```

Total overflow or overflow percentage is therefore a congestion signal. Lower overflow usually means fewer routing detours and fewer detailed-routing failures, but excessive spreading can increase wirelength and hurt timing. The evaluator measures these tradeoffs after placement; the objective should express a physically meaningful placement cost using only current placement observables.

## Why HPWL Is Not Enough

HPWL measures each net by its bounding-box span. It is cheap and strongly correlated with wirelength, but it does not directly model local routing capacity, pin-access pressure, blockages, or the fact that several nets may compete for the same narrow routing region. A pure HPWL objective can make compact placements that are good for wirelength but locally congested. Conversely, a routing-only objective can spread cells too much and cause catastrophic HPWL/timing regression.

Useful search patterns include alternative smooth wirelength models, alternative
density penalties, route-demand pressure, pin-access pressure, and compact
interactions among route-only observables:

```text
score = f(wirelength_variant, density_variant, route_demand, pin_access)
score = wirelength_variant + density_variant + c * g(route_demand, pin_access)
```

Default DREAMPlace is not exposed as a privileged integrated term. Instead,
the objective is composed from explicit placement observables such as smooth
wirelength and density penalties. The evaluator, not the prompt, decides whether
the resulting PPA tradeoff is good.

## Placement Observables Available To Objectives

- `wirelength`: backward-compatible alias for `wirelength_wawl`.
- `wirelength_wawl`: DREAMPlace weighted-average smooth wirelength. This is the default-style differentiable bridge to HPWL.
- `wirelength_lse`: DREAMPlace log-sum-exp smooth wirelength with fixed sharpening gamma. Use as an alternative to WAWL, not together with it.
- `density`: backward-compatible alias for `density_electric`.
- `density_electric`: DREAMPlace electrostatic density penalty. It discourages placement overlap and controls legalization pressure.
- `density_bell`: DREAMPlace bell-shaped density potential. Use as an alternative to electric density, not together with it.
- `soft_rudy_mean`: differentiable soft route-demand proxy. It estimates average routing demand from net bounding boxes.
- `soft_rudy_pnorm`: hotspot-sensitive soft RUDY aggregation. It approximates high-demand regions without using a non-differentiable percentile.
- `route_pressure_long`: differentiable long-net route-pressure proxy using a design-relative smooth gate.
- `pin_density_pnorm`: differentiable pin-access pressure proxy. It penalizes local pin concentration that can hurt detailed routing.
- `pin_count_weighted_wl`: weighted-average smooth wirelength with static sqrt(pin-count) net weighting to emphasize high-fanout nets.

## Objective Design Rules

1. Use only differentiable placement observables available through the objective API.
2. Do not use routed labels, proxy-dataset targets, or OpenROAD reports as direct objective inputs.
3. Avoid copying a baseline objective without a material new mechanism.
4. Avoid mechanisms that are indistinguishable from an existing objective after DREAMPlace evaluation.
5. Use component outputs to expose the mechanism being tested.
6. Treat negative feedback as information: some route-pressure mechanisms may spread cells too much, while some mechanisms may be too weak to change placement behavior.

## Tier-3 Confirmation

Tier-2 DREAMPlace metrics are placement feedback only. A routed PPA claim requires Tier-3 OpenROAD/ChiPBench evaluation of selected DEF placements. Tier-3 should be run only for native default, fixed baselines, and the top few routing-aware candidates, because routing every candidate is too expensive for the inner evolution loop.
