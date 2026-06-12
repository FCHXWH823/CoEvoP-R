# Idea: LLM-Driven Symbolic Discovery of Routing-Aware Placement Objectives

**Author:** Ruogu
**Date:** 2026-06-10

---

## Problem Statement

Modern analytical placers like RePlAce, ePlace, and DREAMPlace all optimize a smooth wirelength plus density surrogate, with HPWL acting as the evaluation metric and a differentiable approximation of it serving as the loss the placer actually minimizes. Both share a blind spot. They ignore congestion, routing topology, layer stack effects, timing slack, and the entire post-placement optimization flow.

The literature from 2024 and 2025 has made this gap unambiguous.

| Paper | Finding | Implication |
|---|---|---|
| ChiPBench (Wang et al., 2024) | HPWL correlates with post-route wirelength but only weakly with WNS, TNS, congestion, and power. MacroHPWL barely correlates at all. | Stop tuning intermediate metrics and target real PPA. |
| LaMPlace (Geng et al., ICLR 2025) | Learning a mask to optimize cross-stage timing gives +43.0% WNS and +30.4% TNS over HPWL. | HPWL is leaving major headroom on the table. |
| GOALPlace (Agnesina et al., 2024) | Aligning placement with post-route optimized cell density avoids ECO churn. | Cross-stage information has to enter at placement, not later. |
| EvoLLM (Yao et al., 2025) | An LLM-evolved optimizer with HPWL as fitness reaches 5% case-by-case and around 1% generalized. | The bottleneck is the objective function, not the optimizer. |

The consensus across the field is that the objective function, not the optimizer, is where cross-stage information has to enter. Two existing answers each have a load-bearing weakness.

| Approach | Strength | Weakness |
|---|---|---|
| Learned neural surrogates (LaMPlace, GOALPlace, RoutePlacer) | Captures non-linear cross-stage coupling | Black-box, not portable, hard to audit, unlikely to ship into commercial tools |
| Hand-crafted symbolic terms (HPWL, RUDY, density penalties) | Drop-in, fast, auditable, differentiable | Fixed, cannot adapt, and adding a new term costs significant engineering effort |

What is missing is a symbolic, interpretable, drop-in objective that nonetheless captures cross-stage signal, discovered automatically rather than hand-engineered.

Hand-engineering has been the historical default. An EDA expert reads routing papers, develops intuition about what makes a placement route well, proposes a feature like RUDY or a weighted combination of HPWL and density, implements it, and iterates. The result is a small library of fixed terms accumulated over decades. Classical symbolic regression and genetic programming could in principle automate this loop, but without domain knowledge they are forced to rediscover concepts like RUDY from scratch, which makes them sample-inefficient when the right answer involves features from a specific literature. Large language models bring that literature in as prior. They have read the placement papers, they know what RUDY is, and they can propose plausible combinations the way an expert would. Evidence from adjacent domains has accumulated quickly.

| Precedent | Domain | What is evolved | Result |
|---|---|---|---|
| Eureka (Ma et al., ICLR 2024) | Robotic control (RL) | Python reward function code | Beats human-engineered rewards on 83% of 29 tasks, 52% average improvement |
| LLM-Guided Objective Evolution (Zhang et al., NeurIPS 2025) | Mobility-on-demand dispatch | High-level allocation objective refined by prompt-based harmony search | Training-free, beats RL and hand-engineered baselines on dynamic ride-hailing |
| AlphaEvolve (Novikov et al., DeepMind 2025) | General algorithmic discovery | Entire codebases via evolutionary search | One reported result is a hardware-accelerator circuit simplification, not at the placement-objective layer |

The pattern across robotics, operations research, and general algorithm discovery is consistent. An LLM-driven evolutionary loop produces objectives or programs that outperform hand-engineered baselines while staying interpretable. The work proposed here transports the same template to EDA placement, where the artifact is a closed-form, differentiable, drop-in objective function for analytical placers rather than a Python reward program for reinforcement learning or an operations-research dispatch heuristic.

---

## Core Idea

The proposal follows the LLM-driven evolutionary discovery template established by FunSearch (Romera-Paredes et al., 2023), EoH (Liu et al., 2024), and LLM-SR (Shojaee et al., 2025), and applies it to closed-form placement objectives. The LLM proposes candidate symbolic objectives over standard placement-stage features. Each candidate is scored by its rank correlation against post-route routability and timing labels drawn from public cross-stage corpora such as CircuitNet 2.0 (Jiang et al., ICLR 2024) and end-to-end OpenROAD runs following the ChiPBench protocol (Wang et al., 2024). High-scoring candidates are kept and fed back to the LLM for mutation. The intended output is a single closed-form, differentiable function that drops into an analytical placer in place of the HPWL-plus-density surrogate.

The pipeline at the level of commitments made at this stage:

```
Labeled placement set with post-route labels
       │  (CircuitNet 2.0, ISPD / ICCAD via OpenROAD)
       ▼
LLM proposes candidate symbolic objective
       │
       ▼
Score candidate by rank correlation against post-route labels
       │
       ▼
Select promising candidates, feed them back to the LLM for mutation
       │
       └── loop ──┐
                  ▼
Outer validation on top candidates:
real DREAMPlace + OpenROAD → real post-route PPA
```

The features the LLM draws from are the standard placement-stage quantities already established by the routability- and timing-prediction literature: HPWL, RUDY (Spindler and Johannes, DATE 2007) and its variants, cell and pin density, net-level statistics such as fanout and bounding-box area, and the routing-demand features used in CircuitNet (Chai et al., 2022; Jiang et al., 2024), LHNN (Wang et al., DAC 2022), and RoutePlacer (Hou et al., KDD 2024). The exact feature shortlist, the symbolic operator set, the selection strategy, the prompt structure, and the complexity budget are pilot-stage decisions and not committed here.

What makes the loop affordable at academic scale is that fitness evaluation does not run a real placer. EvoLLM (Yao et al., 2025) had to invoke DREAMPlace for every candidate, which is what forced their 210-GPU cluster. Here, scoring a candidate is a forward pass of a symbolic function over precomputed features on already-labeled placements, which is orders of magnitude cheaper. A secondary benefit, consistent with LLM-SR's out-of-distribution results, is that a bounded-complexity symbolic candidate is less able to overfit to the idiosyncrasies of any single labeled design.

The cleanest contrast with EvoLLM:

| Dimension | EvoLLM | This proposal |
|---|---|---|
| Layer evolved | DREAMPlace optimizer code | Placement objective function |
| Fitness signal | HPWL after a real DREAMPlace run | Rank correlation against post-route labels |
| End artifact | Algorithm pinned to one placer | Math expression portable to any analytical placer |
| Interpretability | Readable optimizer code | Closed-form math, drop-in |

---

## How It Connects to Routing

The labels driving the fitness signal are post-route PPA metrics extracted from public cross-stage corpora, so any discovered symbolic objective already inherits routing-stage awareness through the labels themselves. From this base, three increasingly ambitious interpretations of "co-evolution" between placement and routing are possible.

| Level | Description | Status |
|---|---|---|
| 1. Routing-aware objective | The labels encode routing-stage outcomes such as congestion maps, design-rule violation hotspots, IR-drop, and net delay. The discovered symbolic objective inherits this signal. | Baseline contribution, defensible as a standalone paper. |
| 2. Loop closure with real routing | Outer validation runs OpenROAD on top candidates, computes real post-route PPA, and uses the gap as a correction signal to re-rank or trigger another round of label collection. | What the first paper should target. |
| 3. Joint placement and routing evolution | Simultaneously evolve a placement objective and a routing cost function so the resulting placer and router beat the baseline pair on end-to-end PPA. | Harder, since co-evolving coupled populations is unstable and modifying the router source is invasive. Deferred to a follow-up paper. |

The first paper targets level one with a level-two validation protocol, and level three is the natural sequel.

---

## Proposed Experiment / Validation

The data setup uses public cross-stage corpora that prior work on routability and PPA prediction has already used. CircuitNet 2.0 (Jiang et al., ICLR 2024) provides about ten thousand placement samples across CPU, GPU, and AI-chip designs at 14 nm FinFET, generated through commercial flows (Synopsys Design Compiler plus Cadence Innovus). The labels it ships are post-route congestion maps, design-rule-violation hotspots, IR-drop maps, and per-net delay graphs. Scalar metrics like WNS, TNS, and total wirelength are not provided directly and have to be aggregated from the per-net delay graphs or extracted from separately generated STA and routing reports. ISPD 2005 and ICCAD 2015 benchmarks run through OpenROAD using the ChiPBench protocol (Wang et al., 2024) provide end-to-end PPA evaluation that prior placement work reports against, including the missing scalar metrics. Held-out subsets of CircuitNet at different design styles serve as out-of-distribution probes.

| Dataset | Use |
|---|---|
| CircuitNet 2.0 (14 nm FinFET, ~10K samples) | Inner-loop fitness against post-route congestion, DRV, IR-drop, and net delay labels |
| ISPD 2005 / ICCAD 2015 via OpenROAD | End-to-end post-route validation following the ChiPBench protocol, source of scalar WNS / TNS / wirelength |
| Held-out CircuitNet subset | Out-of-distribution generalization probe |

The baselines are the obvious comparisons across prior work.

| Baseline | What it represents |
|---|---|
| HPWL via default DREAMPlace | Status quo |
| HPWL plus hand-tuned RUDY and density terms | Hand-engineering baseline |
| AutoDMP-tuned HPWL (Agnesina et al., 2023) | Parameter tuning without structural change |
| Neural surrogate trained on the same labels | Black-box accuracy ceiling for cross-stage prediction |
| EvoLLM-evolved DREAMPlace (Yao et al., 2025) | Optimizer-evolution baseline |

The substantive comparisons are not against HPWL. They are against the neural ceiling, to see how much accuracy is lost by being symbolic, and against EvoLLM, to see whether evolving the objective beats evolving the optimizer.

The metrics follow ChiPBench at the end-to-end level (WNS, TNS, total wirelength, DRC violations, congestion overflow, power, area) and use Spearman or Kendall rank correlation against the labels at the inner-loop level. Generalization is reported as the gap between per-design and cross-design held-out improvement, mirroring how EvoLLM presents its results so the numbers are directly comparable.

The experimental questions worth answering, stated qualitatively, are whether a symbolic candidate can match the rank correlation of a neural surrogate trained on the same labels, whether routing-stage features carry signal beyond HPWL, whether cross-design generalization is meaningfully better than EvoLLM's, whether top candidates ranked highly on labels also win on real OpenROAD post-route metrics, and whether a domain expert can read and interpret the discovered formulas. Specific quantitative thresholds and a compute budget belong in a follow-up planning document once the pilot has run.

---

## Open Questions

- Does ranking on CircuitNet labels predict real post-route PPA? If no, the two-loop architecture collapses and we are forced back to expensive inner-loop placer runs.
- How do we feed per-feature feedback to the LLM beyond a scalar fitness? Eureka's reflection is worth 28.6% of its gain, and our analogue is still undesigned.
- Can a bounded closed-form formula even capture cross-stage PPA? If no, the contribution becomes "best closed-form approximation" rather than "solved."
- One global f_sym, or a family conditioned on a design fingerprint? EvoLLM's 10× per-design vs cross-design gap is the warning.
- Does DREAMPlace's HPWL-tuned wrapper (density weight schedule, learning rate, line search) work with an arbitrary f_sym, or does it need joint retuning?

---

## References

- Yao et al., *Evolution of Optimization Algorithms for Global Placement via Large Language Models*, arXiv [2504.17801](https://arxiv.org/abs/2504.17801), 2025.
- Geng et al., *LaMPlace: Learning to Optimize Cross-Stage Metrics in Macro Placement*, [ICLR 2025](https://openreview.net/forum?id=YLIsIzC74j).
- Agnesina et al., *GOALPlace: Begin with the End in Mind*, arXiv [2407.04579](https://arxiv.org/abs/2407.04579), 2024.
- Agnesina et al., *AutoDMP: Automated DREAMPlace-based Macro Placement*, [ISPD 2023](https://d1qx31qr3h6wln.cloudfront.net/publications/AutoDMP.pdf).
- Wang et al., *Benchmarking End-To-End Performance of AI-Based Chip Placement Algorithms* (ChiPBench), arXiv [2407.15026](https://arxiv.org/abs/2407.15026), 2024.
- Jiang et al., *CircuitNet 2.0: An Advanced Dataset for Promoting Machine Learning Innovations in Realistic Chip Design Environment*, [ICLR 2024](https://openreview.net/forum?id=nMFSUjxMIl).
- Chai et al., *CircuitNet: An Open-Source Dataset for ML in EDA*, arXiv [2208.01040](https://arxiv.org/abs/2208.01040), 2022.
- Spindler and Johannes, *Fast and Accurate Routing Demand Estimation* (RUDY), [DATE 2007](https://past.date-conference.com/proceedings-archive/2007/DATE07/PDFFILES/08.7_1.PDF).
- Hou et al., *RoutePlacer: An End-to-End Routability-Aware Placer with Graph Neural Network*, [KDD 2024](https://arxiv.org/abs/2406.02651).
- Wang et al., *LHNN: Lattice Hypergraph Neural Network for VLSI Congestion Prediction*, [DAC 2022](https://arxiv.org/abs/2203.12831).
- Cheng et al., *RePlAce: Advancing Solution Quality and Routability Validation in Global Placement*, IEEE TCAD 2019.
- Lin et al., *DREAMPlace: Deep Learning Toolkit-Enabled GPU Acceleration for Modern VLSI Placement*, DAC 2019.
- Romera-Paredes et al., *Mathematical discoveries from program search with large language models* (FunSearch), Nature 2023.
- Liu et al., *Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model* (EoH), [ICML 2024](https://proceedings.mlr.press/v235/liu24bs.html), arXiv [2401.02051](https://arxiv.org/abs/2401.02051).
- Shojaee et al., *LLM-SR: Scientific Equation Discovery via Programming with Large Language Models*, [ICLR 2025 Oral](https://openreview.net/forum?id=m2nmp8P5in), arXiv [2404.18400](https://arxiv.org/abs/2404.18400). Code at [deep-symbolic-mathematics/LLM-SR](https://github.com/deep-symbolic-mathematics/LLM-SR).
- Ma et al., *Eureka: Human-Level Reward Design via Coding Large Language Models*, [ICLR 2024](https://openreview.net/forum?id=IEduRUO55F), arXiv [2310.12931](https://arxiv.org/abs/2310.12931).
- Novikov et al., *AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery*, arXiv [2506.13131](https://arxiv.org/abs/2506.13131), DeepMind 2025.
- Zhang et al., *Hierarchical Optimization via LLM-Guided Objective Evolution for Mobility-on-Demand Systems*, [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/117702), arXiv [2510.10644](https://arxiv.org/abs/2510.10644).
- Hemadri et al., *VeriLoC: Line-of-Code Level Prediction of Hardware Design Quality from Verilog Code*, arXiv [2506.07239](https://arxiv.org/abs/2506.07239), 2025.
- Liu et al., *ChipNeMo: Domain-Adapted LLMs for Chip Design*, arXiv [2311.00176](https://arxiv.org/abs/2311.00176), 2023.
- Liu et al., *RTLCoder: Outperforming GPT-3.5 in Design RTL Generation with Our Open-Source Dataset and Lightweight Solution*, arXiv [2312.08617](https://arxiv.org/abs/2312.08617), LAD 2024.
- Fang et al., *MasterRTL: A Pre-Synthesis PPA Estimation Framework for Any RTL Design*, arXiv [2311.08441](https://arxiv.org/abs/2311.08441), 2023.
- Pan et al., *A Survey of Research in Large Language Models for Electronic Design Automation*, arXiv [2501.09655](https://arxiv.org/abs/2501.09655), 2025.
- Zang et al., *The Dawn of Agentic EDA: A Survey of Autonomous Digital Chip Design*, arXiv [2512.23189](https://arxiv.org/abs/2512.23189), 2025.
- Zhang et al., *A Systematic Survey on Large Language Models for Evolutionary Optimization: From Modeling to Solving*, arXiv [2509.08269](https://arxiv.org/abs/2509.08269), 2025.
