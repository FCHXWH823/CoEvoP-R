# Physical Context for Objective Proposal

CoEvoP&R evolves differentiable objectives that operate on information available during analytical placement. Routed and timing reports enter the archive as measured evidence for later proposals.

## Routing behavior

Routing overflow arises where local demand exceeds routing capacity.

```text
overflow = max(0, demand - capacity)
```

Wirelength, density, routing demand, pin access, and macro blockages jointly shape this behavior. Compact placements may shorten nets while concentrating demand. Strong spreading pressure may reduce congestion while lengthening critical connections. Candidate objectives therefore combine a smooth wirelength anchor with density and optional routing or pin pressure.

## Trajectory-aware objective

The typed objective defines schedules for density weight and wirelength smoothing. Optional schedules control routing and pin pressure. The update laws may use placement progress, overflow, HPWL trend, smoothing position, initial component values, and gradient ratios.

The scalar loss is assembled from the following differentiable terms.

- `wirelength_wawl` and `wirelength_lse`
- `density_electric` and `density_bell`
- `soft_rudy_mean` and `soft_rudy_pnorm`
- `route_pressure_long`
- `pin_density_pnorm`
- `pin_count_weighted_wl`

Each proposal returns named components so that its physical mechanism can be connected to measured placement trajectories.

## Evidence flow

DREAMPlace supplies placement metrics and objective traces for every validated candidate. The placement-stage timing evaluator supplies admitted WNS and TNS evidence. Scheduled routed evaluation supplies routed wirelength, routing overflow, and post-route WNS and TNS. The archive retains these measurements together with successful objectives and failure memory.
