# Historical Real Tier-1 CircuitNet Test

This document is kept for compatibility with earlier scripts and reports.

The current main CoEvoP&R methodology no longer uses CircuitNet as the primary
evolution loop. Use:

[circuitnet_proxy_analysis.md](circuitnet_proxy_analysis.md)

for the current interpretation.

Historical commands still work:

```bash
export CIRCUITNET_ROOT=$HOME/datasets/CircuitNet/CircuitNet-N14
bash scripts/real_tier1_wsl.sh
```

The script writes artifacts under `runs/real_tier1/`, including manifests,
scalar summaries, baseline rankings, negative controls, and
`real_tier1_report.md`.

Interpretation:

- useful for proxy analysis and term justification;
- not proof of placement improvement;
- not the main parent-selection source for direct Tier-2 evolution;
- not final evidence for DATE-level claims.
