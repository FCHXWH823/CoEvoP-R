# Prompt Templates

Prompt templates will live here.

The v0 prompt format must ask the model for structured JSON only:

- candidate rationale,
- parent objective IDs,
- objective AST,
- declared term usage,
- expected routing/timing intuition.

No arbitrary executable Python from the model should be accepted by the evaluator.

The v0 provider adapters construct prompts in `coevop.llm.prompts` and validate
every response through `coevop.objectives.spec` before writing an `ObjectiveSpec`.
