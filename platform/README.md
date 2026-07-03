# CoEvoP&R Platform Source Package

This directory contains the source-only CoEvoP&R platform package used for
LLM-guided placement-objective evolution.

The platform is stored under:

```text
platform/CoEvoPR-Platform/
```

This subtree intentionally does not include full generated run directories,
datasets, API keys, DREAMPlace, OpenROAD, ChiPBench, or local
machine-specific config files. Those remain external dependencies. See:

```text
platform/CoEvoPR-Platform/README.md
platform/CoEvoPR-Platform/docs/backend_setup.md
```

The full local evidence for the 2026-07-03 report is kept outside Git under
the platform repo's ignored `runs/` directory. A compact placement artifact
package for the reported four-circuit results is included at:

```text
platform/CoEvoPR-Platform/artifacts/2026_07_03_report_placements/
```

It contains compressed final DEFs, exact per-seed DREAMPlace configs/summaries,
and the metric CSVs needed to trace the reported numbers.
