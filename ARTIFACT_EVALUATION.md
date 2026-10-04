# CoEvoP&R Artifact Evaluation

This artifact provides the objective interface, archive-conditioned evolution loop, DREAMPlace integration, placement-stage timing evaluation, scheduled OpenROAD evaluation, paper configurations, and unit tests. External EDA tools and benchmark data are not redistributed.

## Software check

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[dev]'
python3 -m pytest -q
python3 -m coevop.cli --help
```

## Backend check

Set `DREAMPLACE_ROOT`, `CHIPBENCH_ROOT`, `OPENROAD_FLOW_ROOT`, and `PYTHONPATH` as described in `README.md`. Then run

```bash
python3 -m coevop.cli env-check
python3 -m coevop.cli dreamplace-status
python3 -m coevop.cli shared-panel-check \
  --panel configs/shared_panels/chipbench_table1_post_route.toml \
  --run-dir runs/artifact_check \
  --skip-runs
```

The static check verifies the DREAMPlace configuration, ChiPBench design configuration, OpenROAD executable, and integration patch. The smoke commands in `README.md` exercise each backend with a real run.

## Method path

```text
archive-conditioned objective proposal
-> restricted parsing and validation
-> stateful DREAMPlace objective execution
-> placement and timing evidence
-> MAP-Elites archive update and island migration
-> scheduled post-route evaluation
-> routed evidence returned to the archive and next proposal
```

The primary command is

```bash
python3 -m coevop.cli openevolve-tier2 \
  --config configs/openevolve_tier2/chipbench_controller_tier2.toml \
  --platform-config configs/default.toml \
  --run-dir runs/chipbench_controller \
  --resume
```

The paper configuration uses 160 generations, three proposals per generation, five islands, MAP-Elites memory, a four-design timing-evidence panel, and a 20-generation cadence for scheduled post-route evaluation, the timing-proxy audit, and island migration.

## Result files

Each run retains the resolved configuration, validated objective programs, lineage, prompt records, component and state traces, placement metrics, output DEFs, timing evidence, routed metrics, archive state, and failure records. Post-route paper values are aggregated from `comparison_table.csv` with `scripts/aggregate_post_route_results.py`.

## External requirements

Reproducing objective evolution requires an OpenAI API credential and `OPENAI_MODEL=gpt-5.4`. The model-family rows use the `anthropic` and `qwen` providers with their own credentials. DREAMPlace, OpenTimer, OpenROAD, ChiPBench, OpenROAD-flow-scripts, ChiPBench Nangate45, ICCAD 2015 Superblue, and ASAP7 collateral must be installed separately. Runtime and API cost depend on the selected panel and hardware.
