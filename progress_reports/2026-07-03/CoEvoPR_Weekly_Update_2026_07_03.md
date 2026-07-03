# CoEvoP&R Weekly Update - July 3, 2026

## Summary

This week we expanded CoEvoP&R to four circuits in ChipBench. The completed panel covers `bp_fe`, `bp_be`, `ethernet`, and `swerv_wrapper` with three final seeds per circuit.

Across the completed circuits, the selected objectives improve native installed DREAMPlace default on placement HPWL, placement overflow, OpenROAD estimated wirelength, routing congestion, WNS, and TNS. All four routed validations preserve zero DRC count at the global-routing stage.

## What Changed

1. **Four-circuit serious validation.** We completed or consolidated three-seed post-GRT validations for `bp_fe`, `bp_be`, `ethernet`, and `swerv_wrapper`.

2. **Circuit-specific objective selection.** Each circuit uses its own selected objective, matching the current CoEvoP&R search mode where feedback and memory specialize to each circuit.

3. **Consistent metric extraction.** Placement HPWL and placement overflow come from DREAMPlace post-placement rows. Estimated wirelength, routing congestion, WNS, TNS, DRC, and runtime come from OpenROAD global-route-stage rows.

4. **Mean-value reporting.** All percentage changes below use the displayed three-seed mean metric values. Lower is better for HPWL, placement overflow, post-GRT wirelength, routing congestion, and runtime. Higher is better for WNS and TNS changes.

## Selected Objective Summary

| Circuit | Selected objective | Formula summary |
|---|---|---|
| `bp_fe` | `obj_34cad18eb2fc5f98` | `wirelength + density + 3e-08*sigmoid(pin_density_pnorm*soft_rudy_pnorm)` |
| `bp_be` | `obj_1c623761f3ce9552` | `wirelength_wawl + density_electric - 1e-07*log1p(route_pressure_long) - 1e-07*sqrt(pin_count_weighted_wl)` |
| `ethernet` | `obj_705dbe679dcdb0f7` | `wirelength_wawl + density_electric - 3e-08*softplus(pin_density_pnorm*route_pressure_long)` |
| `swerv_wrapper` | `obj_4fd265b50589cb27` | `wirelength_wawl + density_electric + 3e-08*softplus(soft_rudy_pnorm + route_pressure_long)` |

## Run Setting

| Item | Setting |
|---|---|
| Circuits | `bp_fe`, `bp_be`, `ethernet`, `swerv_wrapper` |
| LLM | `gpt-4.1-mini` |
| Baseline | native installed DREAMPlace default |
| Final seeds | 1000, 1001, 1002 |
| Placement | DREAMPlace, 150 final iterations |
| Routing check | OpenROAD global routing |
| Reported values | three-seed means |

## Four-Circuit Result

| Circuit | Placement HPWL | Placement overflow | Post-GRT wirelength | Routing congestion | WNS | TNS | DRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bp_fe` | 13.56% lower | 72.81% lower | 45.88% lower | 83.47% lower | +0.793 ns | +187.80 ns | 0 -> 0 |
| `bp_be` | 53.86% lower | 63.48% lower | 68.34% lower | 91.39% lower | +0.941 ns | +1,549.32 ns | 0 -> 0 |
| `ethernet` | 47.27% lower | 20.73% lower | 36.27% lower | 83.46% lower | +0.190 ns | +15.80 ns | 0 -> 0 |
| `swerv_wrapper` | 24.59% lower | 70.09% lower | 30.35% lower | 48.33% lower | +0.178 ns | +2,116.62 ns | 0 -> 0 |

CoEvoP&R improves all four completed routed validations over native installed DREAMPlace default. The largest gains appear in routing congestion, where the selected objectives reduce OpenROAD global-route overflow by 48.33% to 91.39%.

## Per-Circuit Details

### `bp_fe`

| Metric | DREAMPlace default | CoEvoP&R objective | Change |
|---|---:|---:|---:|
| Placement HPWL | 13,860,393 | 11,980,846 | 13.56% lower |
| Placement overflow | 0.8284 | 0.2252 | 72.81% lower |
| OpenROAD estimated wirelength | 6,815,809 | 3,688,642 | 45.88% lower |
| OpenROAD routing congestion | 1,189,494 | 196,633 | 83.47% lower |
| WNS | -1.2002 ns | -0.4075 ns | +0.7927 ns |
| TNS | -208.94 ns | -21.14 ns | +187.80 ns |
| DRC count | 0 | 0 | matched |
| OpenROAD runtime | 436.9 s | 203.2 s | 53.49% lower |

### `bp_be`

| Metric | DREAMPlace default | CoEvoP&R objective | Change |
|---|---:|---:|---:|
| Placement HPWL | 37,004,280 | 17,072,617 | 53.86% lower |
| Placement overflow | 0.8028 | 0.2932 | 63.48% lower |
| OpenROAD estimated wirelength | 19,466,631 | 6,162,597 | 68.34% lower |
| OpenROAD routing congestion | 5,747,707 | 494,619 | 91.39% lower |
| WNS | -1.9515 ns | -1.0101 ns | +0.9414 ns |
| TNS | -1,674.16 ns | -124.84 ns | +1,549.32 ns |
| DRC count | 0 | 0 | matched |
| OpenROAD runtime | 1,752.0 s | 579.6 s | 66.92% lower |

### `ethernet`

| Metric | DREAMPlace default | CoEvoP&R objective | Change |
|---|---:|---:|---:|
| Placement HPWL | 8,264,768 | 4,358,323 | 47.27% lower |
| Placement overflow | 0.7974 | 0.6320 | 20.73% lower |
| OpenROAD estimated wirelength | 4,159,001 | 2,650,659 | 36.27% lower |
| OpenROAD routing congestion | 482,109 | 79,725 | 83.46% lower |
| WNS | -0.7445 ns | -0.5544 ns | +0.1901 ns |
| TNS | -78.51 ns | -62.71 ns | +15.80 ns |
| DRC count | 0 | 0 | matched |
| OpenROAD runtime | 303.3 s | 167.1 s | 44.90% lower |

### `swerv_wrapper`

| Metric | DREAMPlace default | CoEvoP&R objective | Change |
|---|---:|---:|---:|
| Placement HPWL | 71,220,113 | 53,706,933 | 24.59% lower |
| Placement overflow | 0.8704 | 0.2604 | 70.09% lower |
| OpenROAD estimated wirelength | 38,701,880 | 26,954,876 | 30.35% lower |
| OpenROAD routing congestion | 13,658,018 | 7,057,211 | 48.33% lower |
| WNS | -3.5919 ns | -3.4137 ns | +0.1782 ns |
| TNS | -9,382.35 ns | -7,265.72 ns | +2,116.62 ns |
| DRC count | 0 | 0 | matched |
| OpenROAD runtime | 6,204.5 s | 2,202.1 s | 64.51% lower |

## Current Interpretation

The serious-scale panel shows that CoEvoP&R discovers compact, differentiable objectives that improve both placement-stage metrics and post-GRT routing behavior. The selected objectives form a family of formulas: they use different combinations of smooth wirelength, density, soft RUDY, pin pressure, long-route pressure, and pin-count-weighted wirelength.

Routing-stage behavior now anchors the evidence. All four completed circuits reduce OpenROAD estimated wirelength and routing congestion while improving WNS and TNS.

## Next Step

Generalize to another benchmark ICCAD2015. Finish the manuscript draft.
