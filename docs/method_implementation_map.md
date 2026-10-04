# Method to Code Map

| Manuscript mechanism | Implementation |
|---|---|
| Restricted symbolic proposal parser | `coevop/objectives/program.py` |
| Candidate schema and physical-role validation | `coevop/objectives/spec.py` |
| Placement terms and progress observables | `coevop/objectives/terms.py` |
| Archive-conditioned prompt construction | `coevop/evolution/openevolve_core.py`, `coevop/prompt/` |
| LLM provider adapters (OpenAI, Anthropic, Qwen) | `coevop/llm/providers.py` |
| MAP-Elites archive, islands, migration, and lineage | `coevop/evolution/openevolve_core.py` |
| CoEvo objective-evolution loop | `coevop/eval/openevolve_tier2.py` |
| DREAMPlace process and trace parser | `coevop/backends/dreamplace.py` |
| Objective embedding in DREAMPlace | `patches/dreamplace/custom_objective_placeobj.patch` |
| Design, objective, and seed placement panel | `coevop/eval/tier2_dreamplace.py` |
| Placement-stage timing evidence, audit, and admission | `coevop/eval/timing_proxy.py` |
| CoEvoP&R-E and CoEvoP&R-L configurations | `coevop/eval/exposure.py` |
| Fixed-DP schedule BO control | `coevop/eval/fixed_dp_bo.py` |
| Shared DREAMPlace and OpenROAD panels | `coevop/eval/shared_panel.py` |
| Post-route OpenROAD evaluation | `coevop/eval/tier3_openroad.py` |
| Post-route table aggregation | `scripts/aggregate_post_route_results.py` |

## Proposal and embedding

The proposal model emits a restricted Python-like program with `init_policy`, `update_policy`, and `objective`. The parser converts the source to a typed data structure. Schema, grammar, physical-role, differentiability, bound, finite-value, gradient, and runtime checks determine admission. An admitted candidate is bound to DREAMPlace wirelength, density, routing, pin, smoothing, and optional net-weight controls.

## Evolution loop

`run_openevolve_tier2` builds prompt context from a parent program, nearby archive elites, measured metric changes, component and state traces, routed evidence from every scheduled round, and failure memory. Accepted children enter an island-local MAP-Elites archive. Nondominated evidence determines reproduction and elite status. Every `migration_interval` generations all islands exchange elites along the ring, and the archive is Pareto-refreshed after the exchange.

## Cost-scaled evidence

Every validated candidate receives DREAMPlace placement metrics (Tier A). A candidate whose placement completed with finite HPWL and overflow receives placement-stage timing evidence from the configured timing panel (Tier B). At the configured interval, selected candidates enter the ChiPBench and OpenROAD post-route panel (Tier C). Routed wirelength, routing overflow, WNS, and TNS are attached to the program records. The archive is refreshed before `routed_feedback.json` is included in the next proposal packet.

## Pareto evidence order

`_pareto_evidence` and `_dominates` implement the order of the manuscript over four lower-is-better coordinates: wirelength, overflow, negated WNS gain, and negated TNS gain. A candidate lists each coordinate at every tier it received. Two candidates are compared on the coordinates measured for both, each at the most faithful common tier: routed measurements when both were routed, otherwise the placement-stage ones. Candidates on the same front are ordered by equal-design rank and then by runtime.

## Timing-proxy audit

`_run_scheduled_timing_proxy_audit` audits the proxy on the seed placements and every `audit_interval` generations. `timing_proxy_metric_admission` turns each design's HPWL-redundancy correlations into an Admit, Tie, or Reject outcome for WNS and for TNS, and `timing_proxy_downstream_agreement` limits a metric to Tie when its movement disagrees with post-route movement. `apply_timing_proxy_admission` then splits a candidate's proxy deltas: admitted designs supply the selection coordinate, tied designs only order a front, and rejected designs contribute to neither. The unsplit deltas remain archive features.

## Primary configuration

`configs/openevolve_tier2/chipbench_controller_tier2.toml` specifies the complete paper loop. It uses five islands, MAP-Elites memory, three proposals per generation, 160 generations, four-design timing evidence, and a 20-generation cadence for routed evaluation of three candidates, the timing-proxy audit, and island migration.

## Exposure settings

`coevop/eval/exposure.py` derives the other evolution configurations from the primary one, so they share its method and budget. `write_target_evolution_config` restricts feedback to the design being evaluated (CoEvoP&R-E) for ChiPBench, Superblue, and ASAP7. `write_lodo_config` removes one design from feedback and evaluates it only after evolution (CoEvoP&R-L). CoEvoP&R-T applies the frozen champion of the primary ChiPBench evolution without further search.
