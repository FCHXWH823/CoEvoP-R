# Proposal Resources

The default prompt templates in `coevop/prompts/defaults` assemble the archive-conditioned proposal packet. `router_objective_background.md` provides the physical placement and routing context inserted into the primary evolution configuration.

Every model response follows the structured schema returned by `objective_program_schema_for_provider`. The restricted parser accepts the typed objective functions, validates their AST, and emits a complete objective candidate for DREAMPlace execution.
