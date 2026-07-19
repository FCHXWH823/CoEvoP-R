# OpenEvolve Attribution

CoEvoP&R's OpenEvolve-style memory loop adapts architectural ideas from
OpenEvolve:

- Repository: https://github.com/algorithmicsuperintelligence/openevolve
- License: Apache-2.0

The CoEvoP&R implementation keeps a separate safety boundary: the LLM evolves
restricted placement-objective programs that are parsed into `ObjectiveSpec`
JSON and are not executed as arbitrary Python code.

Adapted mechanism classes include:

- persistent program database with parent-child lineage;
- prompt memory containing top, diverse, failed, and recent programs;
- K-sample evolutionary iterations;
- island-local populations with ring migration;
- MAP-Elites-style feature-cell archive discipline.

CoEvoP&R-specific additions include:

- DREAMPlace objective injection and gradient safety checks;
- placement-objective component trajectory feedback;
- routing-aware placement term library;
- archetype memory for placement/routing mechanism families.

The OpenEvolve repository is not vendored as a runtime dependency.
