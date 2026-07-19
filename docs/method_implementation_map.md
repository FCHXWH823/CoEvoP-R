# Method to Code Map

| Manuscript mechanism | Implementation |
|---|---|
| Restricted symbolic proposal parser | `coevop/objectives/program.py` |
| Candidate schema and physical-role validation | `coevop/objectives/spec.py` |
| Placement terms and progress observables | `coevop/objectives/terms.py` |
| Archive-conditioned prompt construction | `coevop/evolution/openevolve_core.py`, `coevop/prompt/` |
| LLM provider adapter | `coevop/llm/providers.py` |
| MAP-Elites archive, islands, migration, and lineage | `coevop/evolution/openevolve_core.py` |
| CoEvo objective-evolution loop | `coevop/eval/openevolve_tier2.py` |
| DREAMPlace process and trace parser | `coevop/backends/dreamplace.py` |
| Objective embedding in DREAMPlace | `patches/dreamplace/custom_objective_placeobj.patch` |
| Design, objective, and seed placement panel | `coevop/eval/tier2_dreamplace.py` |
| Placement-stage timing evidence and audit | `coevop/eval/timing_proxy.py` |
| Shared DREAMPlace and OpenROAD panels | `coevop/eval/shared_panel.py` |
| Post-route OpenROAD evaluation | `coevop/eval/tier3_openroad.py` |
| Post-route table aggregation | `scripts/aggregate_post_route_results.py` |

## Proposal and embedding

The proposal model emits a restricted Python-like program with `init_policy`, `update_policy`, and `objective`. The parser converts the source to a typed data structure. Schema, grammar, physical-role, differentiability, bound, finite-value, gradient, and runtime checks determine admission. An admitted candidate is bound to DREAMPlace wirelength, density, routing, pin, smoothing, and optional net-weight controls.

## Evolution loop

`run_openevolve_tier2` builds prompt context from a parent program, nearby archive elites, measured metric changes, component and state traces, and failure memory. Accepted children enter an island-local MAP-Elites archive. Nondominated evidence determines reproduction and elite status. Ring migration periodically exchanges elites among islands.

## Cost-scaled evidence

Every candidate receives DREAMPlace placement metrics. Admitted timing cells receive placement-stage timing evidence from the configured timing panel. At the configured interval, selected candidates enter the ChiPBench and OpenROAD post-route panel. Routed wirelength, routing overflow, WNS, and TNS are attached to the program records. The archive is refreshed before `routed_feedback.json` is included in the next proposal packet.

## Primary configuration

`configs/openevolve_tier2/chipbench_controller_tier2.toml` specifies the complete paper loop. It uses five islands, MAP-Elites memory, three proposals per generation, 160 generations, four-design timing evidence, and routed evaluation of three candidates every 20 generations.
