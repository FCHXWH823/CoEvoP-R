# CoEvoP&R — Co-Evolution of Placement & Routing

## Project Overview

Modern VLSI placement tools rely on **HPWL (Half-Perimeter Wirelength)** as a proxy for routing quality. This is a well-known simplification: HPWL ignores congestion, layer constraints, detour paths, and the complex feedback between placement decisions and actual routing outcomes.

**CoEvoP&R** investigates whether a **large language model (LLM) agent** can:

1. Understand existing routing algorithms (global routing, detailed routing, congestion modeling, etc.)
2. **Evolve better placement objective functions** that more accurately reflect true routing cost
3. Iteratively co-refine the objective as both placement and routing knowledge improve

The long-term goal is an agent-driven flow where the placement cost function is no longer hand-crafted but **automatically evolved** through reasoning over routing literature and empirical feedback.

---

## Motivation

| Current Practice | Problem |
|---|---|
| HPWL as routing proxy | Ignores congestion, layer stack, and detour |
| Fixed objective function | Cannot adapt to different design styles or technologies |
| Placement & routing decoupled | Suboptimal globally; costly ECO iterations |

An evolved objective could incorporate:
- Congestion-aware wirelength weighting
- Net criticality / timing awareness
- Routing-topology-aware cost terms (e.g., Steiner tree estimates)
- Technology-specific penalties (e.g., preferred routing directions)

---

## Agent Flow (Proposed)

```
┌─────────────────────────────────────────────────┐
│  LLM Agent                                      │
│                                                 │
│  1. Read & summarize routing algorithm papers   │
│  2. Extract cost-relevant features              │
│  3. Propose candidate objective function        │
│  4. Evaluate via placement + routing run        │
│  5. Reflect on result → refine objective        │
│  6. Repeat (co-evolution loop)                  │
└─────────────────────────────────────────────────┘
```

---

## Progress & Paper Reading Log

> **Instructions for collaborators:**  
> When you read a relevant paper, add a row to the table below. Include the key takeaway and how it might inform the objective function design.

| Date | Contributor | Paper | Venue | Key Takeaway | Relevance to Objective |
|------|-------------|-------|-------|--------------|------------------------|
| 2026.5.24 | Weihua Xiao | Evolution of Optimization Algorithms for Global Placement via Large Language Models | — | Uses LLM to evolve three components of placement: initialization, preconditioner, and optimizer. Does **not** evolve the objective function — HPWL is still used as-is. | Directly motivates our goal: the algorithmic pipeline can be LLM-evolved, but the objective (HPWL) remains a fixed proxy for routing. Our work targets exactly this gap — evolving the objective function itself to better reflect true routing cost. |

### Paper Categories of Interest

- **Placement:** DREAMPlace, RePlAce, ePlace, NTUplace, AutoDMP, Mixed-size placement
- **Global Routing:** FLUTE, FastRoute, CUGR, NCTUgr
- **Detailed Routing:** TritonRoute, Dr. CU
- **Routing-aware Placement:** Routability-driven placement, pin accessibility
- **LLM for EDA:** ChipNeMo, EDA Corpus, LLM-assisted optimization
- **Evolutionary / AutoML objective search:** FunSearch, EvoPrompting, symbolic regression

---

## Ideas

See the [`ideas/`](./ideas/) folder for each collaborator's written proposals and brainstorming notes.

---

## Repository Structure

```
CoEvoP&R/
├── README.md          ← this file (progress log + paper notes)
├── ideas/             ← collaborator idea files (ideally one per person)
│   └── example.md
└── ...                ← code, experiments, benchmarks (TBD)
```

---

## Getting Involved

1. Read the overview above and the papers in your area.
2. Log your paper notes in the table above.
3. Write your ideas or proposals in `ideas/<your_name>.md`.
4. Open a discussion / PR to propose changes to the agent flow design.
