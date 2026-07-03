"""Objective term registry.

The registry makes the train/deploy boundary explicit:

- Tier-1 proxy terms are CircuitNet scalar summaries used for offline feedback.
- DREAMPlace-deployable terms are differentiable tensors available in the
  external DREAMPlace custom objective patch.
- Diagnostic-only terms may be logged or analyzed, but must not enter the
  optimizer.

Do not add a term to both Tier-1 and DREAMPlace scopes unless the unit
convention and semantic bridge are documented deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TermSpec:
    name: str
    description: str
    tier1_column: str | None = None
    dreamplace_term: str | None = None
    diagnostic_term: str | None = None
    badness_direction: int = 1
    llm_visible: bool = True

    @property
    def is_tier1(self) -> bool:
        return self.tier1_column is not None

    @property
    def is_dreamplace(self) -> bool:
        return self.dreamplace_term is not None

    @property
    def is_diagnostic(self) -> bool:
        return self.diagnostic_term is not None

    @property
    def is_both_scope(self) -> bool:
        return self.is_tier1 and self.is_dreamplace


TERM_REGISTRY: dict[str, TermSpec] = {
    "rudy_mean": TermSpec(
        name="rudy_mean",
        tier1_column="feature.rudy.mean",
        description="Mean CircuitNet RUDY routing demand. Higher means more routing pressure.",
    ),
    "rudy_p95": TermSpec(
        name="rudy_p95",
        tier1_column="feature.rudy.p95",
        description="95th percentile CircuitNet RUDY hotspot pressure.",
    ),
    "rudy_pin_p95": TermSpec(
        name="rudy_pin_p95",
        tier1_column="feature.rudy_pin.p95",
        description="95th percentile pin-aware CircuitNet RUDY pressure.",
    ),
    "rudy_short_p95": TermSpec(
        name="rudy_short_p95",
        tier1_column="feature.rudy_short.p95",
        description="95th percentile short-net CircuitNet RUDY pressure.",
    ),
    "rudy_long_p95": TermSpec(
        name="rudy_long_p95",
        tier1_column="feature.rudy_long.p95",
        description="95th percentile long-net CircuitNet RUDY pressure.",
    ),
    "rudy_pin_long_p95": TermSpec(
        name="rudy_pin_long_p95",
        tier1_column="feature.rudy_pin_long.p95",
        description="95th percentile pin-aware long-net CircuitNet RUDY pressure.",
    ),
    "cell_density_mean": TermSpec(
        name="cell_density_mean",
        tier1_column="feature.cell_density.mean",
        description="Mean CircuitNet cell-density map value.",
    ),
    "cell_density_p95": TermSpec(
        name="cell_density_p95",
        tier1_column="feature.cell_density.p95",
        description="95th percentile CircuitNet cell-density hotspot value.",
    ),
    "macro_region_mean": TermSpec(
        name="macro_region_mean",
        tier1_column="feature.macro_region.mean",
        description="Mean macro-region occupancy feature.",
    ),
    "wirelength": TermSpec(
        name="wirelength",
        dreamplace_term="wirelength",
        description=(
            "Backward-compatible alias for wirelength_wawl: DREAMPlace weighted-average "
            "smooth wirelength."
        ),
    ),
    "wirelength_wawl": TermSpec(
        name="wirelength_wawl",
        dreamplace_term="wirelength_wawl",
        description=(
            "DREAMPlace weighted-average smooth wirelength. This is the explicit "
            "wirelength component corresponding to the default global-placement "
            "wirelength model."
        ),
    ),
    "wirelength_lse": TermSpec(
        name="wirelength_lse",
        dreamplace_term="wirelength_lse",
        description=(
            "DREAMPlace log-sum-exp smooth wirelength with fixed low-overflow gamma "
            "0.1 * base_gamma. Use as a sharper alternative to wirelength_wawl, "
            "not together with it."
        ),
    ),
    "native_objective": TermSpec(
        name="native_objective",
        dreamplace_term="native_objective",
        description=(
            "DREAMPlace's native adaptive objective at the current iteration: "
            "smooth wirelength plus the current adaptive density-weighted electric "
            "density penalty. Use this as the base term when comparing directly "
            "against default DREAMPlace instead of replacing the density schedule "
            "with a fixed coefficient."
        ),
        llm_visible=False,
    ),
    "density": TermSpec(
        name="density",
        dreamplace_term="density",
        description="Backward-compatible alias for density_electric.",
    ),
    "density_electric": TermSpec(
        name="density_electric",
        dreamplace_term="density_electric",
        description=(
            "DREAMPlace raw electrostatic density penalty. This is the explicit "
            "electric-density observable behind the default density mechanism."
        ),
    ),
    "density_bell": TermSpec(
        name="density_bell",
        dreamplace_term="density_bell",
        description=(
            "DREAMPlace NTUPlace3 bell-shaped density-potential penalty. Use as an "
            "alternative to density_electric, especially when testing macro-heavy designs."
        ),
    ),
    "density_pnorm": TermSpec(
        name="density_pnorm",
        tier1_column="feature.cell_density.p95",
        diagnostic_term="density_pnorm",
        description=(
            "Hotspot-sensitive density pressure: CircuitNet cell-density p95 in "
            "Tier-1. DREAMPlace density p-norm is diagnostic-only until it passes "
            "single-term gradient promotion checks."
        ),
    ),
    "soft_rudy_mean": TermSpec(
        name="soft_rudy_mean",
        tier1_column="feature.rudy.mean",
        dreamplace_term="soft_rudy_mean",
        description=(
            "Mean route-demand pressure: CircuitNet RUDY mean in Tier-1 and a "
            "differentiable DREAMPlace soft-RUDY net-bounding-box proxy in Tier-2."
        ),
    ),
    "soft_rudy_pnorm": TermSpec(
        name="soft_rudy_pnorm",
        tier1_column="feature.rudy.p95",
        dreamplace_term="soft_rudy_pnorm",
        description=(
            "Hotspot route-demand pressure: CircuitNet RUDY p95 in Tier-1 and a "
            "smooth p-norm aggregation of the DREAMPlace soft-RUDY proxy in Tier-2."
        ),
    ),
    "pin_density_pnorm": TermSpec(
        name="pin_density_pnorm",
        tier1_column="feature.rudy_pin.p95",
        dreamplace_term="pin_density_pnorm",
        description=(
            "Local pin-access pressure: CircuitNet pin-aware RUDY p95 in Tier-1 and "
            "a differentiable pin-weighted soft-RUDY p-norm proxy in Tier-2."
        ),
    ),
    "route_pressure_long": TermSpec(
        name="route_pressure_long",
        dreamplace_term="route_pressure_long",
        description=(
            "Differentiable long-net route-pressure p-norm. Long nets are selected "
            "by a smooth design-relative gate with threshold 2.0 * mean valid net span."
        ),
    ),
    "pin_count_weighted_wl": TermSpec(
        name="pin_count_weighted_wl",
        dreamplace_term="pin_count_weighted_wl",
        description=(
            "Weighted-average smooth wirelength with static per-net weights scaled by "
            "sqrt(pin_count) / mean(sqrt(pin_count)). This emphasizes high-fanout nets."
        ),
    ),
    "timing_weighted_wirelength": TermSpec(
        name="timing_weighted_wirelength",
        dreamplace_term="timing_weighted_wirelength",
        description=(
            "DREAMPlace smooth wirelength using the current timing-updated net "
            "weights. DREAMPlace-only until timing labels are promoted into the "
            "Tier-1 bridge."
        ),
        llm_visible=False,
    ),
}


def term_names(scope: str = "tier1") -> list[str]:
    if scope == "tier1":
        return sorted(name for name, spec in TERM_REGISTRY.items() if spec.is_tier1)
    if scope in {"dreamplace", "dreamplace_deployable"}:
        return sorted(name for name, spec in TERM_REGISTRY.items() if spec.is_dreamplace)
    if scope == "dreamplace_replacement":
        return sorted(
            name
            for name, spec in TERM_REGISTRY.items()
            if spec.is_dreamplace and spec.llm_visible
        )
    if scope in {"deployable", "both"}:
        return sorted(name for name, spec in TERM_REGISTRY.items() if spec.is_both_scope)
    if scope in {"diagnostic", "diagnostic_only"}:
        return sorted(name for name, spec in TERM_REGISTRY.items() if spec.is_diagnostic)
    if scope == "all":
        return sorted(TERM_REGISTRY)
    raise ValueError(f"unknown objective term scope: {scope}")


def term_descriptions(scope: str = "tier1") -> dict[str, str]:
    return {name: TERM_REGISTRY[name].description for name in term_names(scope)}


def tier1_columns() -> dict[str, str]:
    return {
        name: spec.tier1_column
        for name, spec in TERM_REGISTRY.items()
        if spec.tier1_column is not None
    }


def tier1_column_for(term_name: str) -> str:
    spec = TERM_REGISTRY.get(term_name)
    if spec is None:
        raise KeyError(f"unknown objective term: {term_name}")
    if spec.tier1_column is None:
        raise KeyError(f"objective term {term_name} is not available in Tier-1 CircuitNet scoring")
    return spec.tier1_column


def dreamplace_term_for(term_name: str) -> str:
    spec = TERM_REGISTRY.get(term_name)
    if spec is None:
        raise KeyError(f"unknown objective term: {term_name}")
    if spec.dreamplace_term is None:
        raise KeyError(f"objective term {term_name} is not deployable in DREAMPlace")
    return spec.dreamplace_term


def validate_term_registry() -> None:
    for name, spec in TERM_REGISTRY.items():
        if spec.name != name:
            raise ValueError(f"term registry key/name mismatch: {name} != {spec.name}")
        if not (spec.is_tier1 or spec.is_dreamplace or spec.is_diagnostic):
            raise ValueError(f"term {name} does not declare any scope")
        if spec.badness_direction not in {-1, 1}:
            raise ValueError(f"term {name} has invalid badness direction")


def unsupported_terms_for_scope(terms: list[str], scope: str) -> list[str]:
    allowed = set(term_names(scope))
    return sorted(term for term in terms if term not in allowed)


validate_term_registry()
