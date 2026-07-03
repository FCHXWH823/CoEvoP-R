# CoEvoP&R Weekly Update - July 2, 2026

## Summary

This week we moved CoEvoP&R from prompt-only objective generation to execution-driven objective discovery. The platform now lets an LLM generate a symbolic DREAMPlace objective, runs the objective on a real ChipBench circuit, evaluates the resulting placement with DREAMPlace and OpenROAD global routing, and feeds the measured results back into the next round of objective generation.

The latest completed serious run is on the `ethernet` circuit. Across three final seeds, the best LLM-generated objective improves native installed DREAMPlace default on placement, routing, timing, and runtime metrics.

## What Changed

1. **OpenEvolve-style memory.** The platform now stores previous objectives, parent-child lineage, execution metrics, failures, and useful negative examples. The OpenAI API remains stateless; all memory is explicitly managed by our local platform and inserted into later prompts.

2. **Eureka-style execution feedback.** The LLM is given the objective API, available DREAMPlace terms, validation rules, and previous measured outcomes. It writes a restricted objective program, which is validated before DREAMPlace ever sees it.

3. **Direct DREAMPlace and OpenROAD evaluation.** We no longer rely on CircuitNet proxy scoring for the main loop. The current serious tests use real DREAMPlace placement and OpenROAD global routing on ChipBench circuits.

4. **Circuit-specific objectives.** Each circuit can have its own prompt context, local memory, and selected objective. This lets the platform search for objectives adapted to each chip's routing and density behavior.

## Latest LLM-Generated Objective

The best Ethernet objective found in the latest run is:

```python
def objective(features):
    density_electric = term("density_electric")
    pin_density_pnorm = term("pin_density_pnorm")
    route_pressure_long = term("route_pressure_long")
    wirelength_wawl = term("wirelength_wawl")
    score = (
        wirelength_wawl
        + density_electric
        + (-3e-08 * softplus(pin_density_pnorm * route_pressure_long))
    )
    return score, {
        "density_electric": density_electric,
        "pin_density_pnorm": pin_density_pnorm,
        "route_pressure_long": route_pressure_long,
        "wirelength_wawl": wirelength_wawl,
    }
```

This objective keeps the normal wirelength and density structure, then adds a small routing-aware correction using pin-density pressure and long-route pressure.

## Ethernet Result

Run setting:

| Item | Setting |
|---|---|
| Circuit | `ethernet` |
| LLM | `gpt-4.1-mini` |
| Baseline | native installed DREAMPlace default |
| Final seeds | 1000, 1001, 1002 |
| Placement | DREAMPlace, 150 iterations |
| Routing check | OpenROAD global routing |

Three-seed average:

| Metric | DREAMPlace default | LLM objective | Change |
|---|---:|---:|---:|
| Placement HPWL | 8,264,768 | 4,358,323 | -47.12% |
| Placement overflow | 0.7974 | 0.6320 | -20.73% |
| OpenROAD estimated wirelength | 4,159,001 | 2,650,659 | -36.08% |
| OpenROAD global-route overflow | 482,109 | 79,725 | -83.03% |
| WNS | -0.7445 ns | -0.5544 ns | +0.1901 ns |
| TNS | -78.51 ns | -62.71 ns | +15.80 ns |
| DRC count | 0 | 0 | no regression |
| OpenROAD runtime | 303.3 s | 167.1 s | faster |

Negative values mean improvement for HPWL, overflow, and wirelength. Positive WNS/TNS changes mean timing improved.

## Current Interpretation

The Ethernet run shows that the current platform can discover a compact, interpretable objective that improves both placement and OpenROAD global-routing behavior against the native DREAMPlace default. The objective is not just a density-weight retuning: it includes a routing-aware correction based on pin-density and long-route pressure.

The result is still circuit-specific. We should next repeat the same protocol on `swerv_wrapper` and `isa_npu`, then test whether the best objective from one circuit transfers to the others.

## Current Best Evidence

| Circuit | Best generated objective | Seeds | Main result |
|---|---|---:|---|
| `bp_fe` | `obj_34cad18eb2fc5f98` | 3 | OpenROAD global-route overflow improved by 83.47%; WNS improved by 0.793 ns |
| `bp_be` | `obj_1c623761f3ce9552` | 3 | OpenROAD global-route overflow improved by 91.39%; WNS improved by 0.941 ns |
| `ethernet` | `obj_705dbe679dcdb0f7` | 3 | OpenROAD global-route overflow improved by 83.03%; WNS improved by 0.190 ns |

## Next Step

Run the same serious protocol on the remaining signature circuits, especially `swerv_wrapper`, then build a cross-circuit transfer table. That table will show whether CoEvoP&R is discovering chip-specific fixes only, or objectives that generalize across designs.
