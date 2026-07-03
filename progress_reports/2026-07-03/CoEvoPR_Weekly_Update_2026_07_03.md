# CoEvoP&R Weekly Update - July 3, 2026

## What Changed

This week we changed CoEvoP&R from a prompt-only prototype into an execution-driven objective discovery platform. The LLM now proposes a symbolic placement objective, the platform runs it through DREAMPlace, evaluates the placement with OpenROAD routing-stage metrics, and feeds the measured results back into later objective-generation rounds.

The main implementation changes are:

- **OpenEvolve-style memory:** the platform stores previous objectives, parent-child lineage, execution results, failures, and useful negative examples. The OpenAI API has no hidden memory; all memory is local and explicit.
- **Eureka-style feedback:** prompts include the objective API, allowed DREAMPlace terms, validation rules, and previous measured outcomes.
- **Direct physical evaluation:** the main loop now uses real DREAMPlace placement and OpenROAD routing-stage evaluation on ChipBench circuits, instead of relying on CircuitNet proxy scoring.
- **Circuit-specific search:** each circuit can receive its own prompt context and memory, so the platform can discover objectives adapted to that chip.

## Latest LLM-Generated Objective

The best Ethernet objective found by `gpt-4.1-mini` was:

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

This objective keeps the standard wirelength-density structure but adds a routing-aware correction based on pin-density pressure and long-route pressure.

## Latest Results

The latest completed serious run uses the `ethernet` ChipBench circuit, three seeds, DREAMPlace placement, and OpenROAD routing-stage evaluation. The comparison baseline is the native installed DREAMPlace default objective.

| Metric | DREAMPlace default | LLM objective | Improvement |
|---|---:|---:|---:|
| Placement HPWL | 8,264,768 | 4,358,323 | 47.12% lower |
| Placement overflow | 0.7974 | 0.6320 | 20.73% lower |
| OpenROAD estimated wirelength | 4,159,001 | 2,650,659 | 36.08% lower |
| OpenROAD routing overflow | 482,109 | 79,725 | 83.03% lower |
| WNS | -0.7445 ns | -0.5544 ns | +0.1901 ns |
| TNS | -78.51 ns | -62.71 ns | +15.80 ns |
| DRC count | 0 | 0 | no regression |
| OpenROAD runtime | 303.3 s | 167.1 s | faster |

We now have strong positive results on three ChipBench circuits:

| Circuit | Best objective | Main result versus DREAMPlace default |
|---|---|---|
| `bp_fe` | `obj_34cad18eb2fc5f98` | OpenROAD routing overflow improved by 83.47%; WNS improved by 0.793 ns |
| `bp_be` | `obj_1c623761f3ce9552` | OpenROAD routing overflow improved by 91.39%; WNS improved by 0.941 ns |
| `ethernet` | `obj_705dbe679dcdb0f7` | OpenROAD routing overflow improved by 83.03%; WNS improved by 0.190 ns |

The key takeaway is that CoEvoP&R can discover compact, interpretable objectives that improve the native DREAMPlace default on placement quality and routing-stage behavior for multiple ChipBench circuits.
