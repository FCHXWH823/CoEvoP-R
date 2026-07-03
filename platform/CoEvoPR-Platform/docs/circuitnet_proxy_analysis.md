# CircuitNet Proxy Analysis

CircuitNet is optional in the current CoEvoP&R methodology.

It is not the main optimization loop, and it is not required to run
`openevolve-tier2`.

## What CircuitNet Can Show

CircuitNet can provide cheap evidence that placement-state observables are
related to routing labels:

- RUDY-like route demand;
- pin-aware routing pressure;
- cell-density hotspots;
- macro-region occupancy;
- congestion/DRV labels.

This is useful for term justification and sanity checking.

## What CircuitNet Cannot Prove

CircuitNet correlation does not prove that an objective improves placement.

A formula can correlate with congestion labels but still be a bad DREAMPlace
objective because:

- it may create unstable or overly large gradients;
- it may spread cells and regress HPWL/timing;
- it may only rank static rows rather than improve optimizer trajectories;
- it may not generalize from CircuitNet design families to ChiPBench/Nangate45
  designs.

Similarly, a good DREAMPlace objective does not necessarily need high CircuitNet
correlation. It may be effective because of how the optimizer responds during
placement.

## Recommended Use

Use CircuitNet only as proxy analysis:

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

Do not use these rankings as final evidence that an objective is good.

## Historical Names

Some code and command names still contain `tier1` for compatibility:

- `tier1-offline`
- `tier1-score-spec`
- `evolve-offline`
- `real-tier1-audit`
- `coevop.eval.tier1`
- `coevop.evolution.offline`

These should now be interpreted as CircuitNet proxy-analysis tools, not the main
research loop.
