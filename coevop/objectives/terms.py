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
            "Mean differentiable route-demand pressure derived from smooth "
            "net bounding boxes in the current DREAMPlace placement."
        ),
    ),
    "soft_rudy_pnorm": TermSpec(
        name="soft_rudy_pnorm",
        tier1_column="feature.rudy.p95",
        dreamplace_term="soft_rudy_pnorm",
        description=(
            "Hotspot-sensitive differentiable route-demand pressure formed by a "
            "smooth p-norm aggregation over net bounding boxes."
        ),
    ),
    "pin_density_pnorm": TermSpec(
        name="pin_density_pnorm",
        tier1_column="feature.rudy_pin.p95",
        dreamplace_term="pin_density_pnorm",
        description=(
            "Differentiable local pin-access pressure formed by a pin-weighted "
            "smooth route-demand p-norm."
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


OBSERVABLE_REGISTRY: dict[str, str] = {
    "iter_frac": (
        "Completed global-placement iterations divided by the configured stage "
        "iteration budget, in [0, 1]."
    ),
    "overflow": (
        "Most recently evaluated density overflow. Starts at 1.0 before the "
        "first evaluation and decreases toward the stop overflow as spreading "
        "progresses."
    ),
    "hpwl_delta_rate": (
        "Relative HPWL change between the last two evaluated iterations: "
        "(hpwl_t - hpwl_prev) / max(hpwl_prev, 1). Zero at the first iteration. "
        "Negative values mean HPWL is still improving."
    ),
    "gamma_frac": (
        "Current wirelength smoothing gamma divided by the base gamma. "
        "Anneals downward as overflow decreases."
    ),
}

# Privileged observables: implemented in the patch and used by identity-control
# presets and diagnostics only. Exposing the native schedule to generated
# controllers would let candidates modify it instead of discovering their own.
PRIVILEGED_OBSERVABLE_REGISTRY: dict[str, str] = {
    "native_density_weight": (
        "The effective native density coefficient (density_factor * "
        "density_weight) as maintained by DREAMPlace's own update rule during "
        "this run. Privileged: identity-control presets and diagnostics only, "
        "never exposed to generated controllers."
    ),
}

_GRAD_RATIO_PREFIX = "grad_ratio_"
_INIT_VALUE_PREFIX = "init_value_"
COMPONENT_GRAD_RATIO_PREFIX = "grad_ratio_component_"
COMPONENT_INIT_VALUE_PREFIX = "init_value_component_"


def is_component_calibration_observable(name: str) -> bool:
    return any(
        name.startswith(prefix) and bool(name[len(prefix) :])
        for prefix in (COMPONENT_GRAD_RATIO_PREFIX, COMPONENT_INIT_VALUE_PREFIX)
    )


def observable_names(scope: str = "dreamplace_controller") -> list[str]:
    """Names of read-only scalar optimizer observables for controller mode.

    Scope "dreamplace_controller" is the LLM-visible set. Scope "privileged"
    (or "all") additionally includes identity-control observables that
    generated candidates must not use.
    """

    if scope not in {"dreamplace_controller", "privileged", "all"}:
        raise ValueError(f"unknown observable scope: {scope}")
    calibration = [
        prefix + name
        for prefix in (_GRAD_RATIO_PREFIX, _INIT_VALUE_PREFIX)
        for name in term_names("dreamplace_replacement")
    ]
    names = list(OBSERVABLE_REGISTRY) + calibration
    if scope in {"privileged", "all"}:
        names += list(PRIVILEGED_OBSERVABLE_REGISTRY)
    return sorted(names)


def privileged_observable_names() -> list[str]:
    return sorted(PRIVILEGED_OBSERVABLE_REGISTRY)


def observable_descriptions(scope: str = "dreamplace_controller") -> dict[str, str]:
    descriptions: dict[str, str] = {}
    for name in observable_names(scope):
        if name.startswith(_GRAD_RATIO_PREFIX):
            term = name[len(_GRAD_RATIO_PREFIX):]
            descriptions[name] = (
                f"||grad wirelength_wawl|| / ||grad {term}|| measured once at the "
                "calibration call, constant afterwards, and logged. Multiplying a "
                "raw term by its grad ratio gives it wirelength-comparable "
                "gradient scale."
            )
        elif name.startswith(_INIT_VALUE_PREFIX):
            term = name[len(_INIT_VALUE_PREFIX):]
            descriptions[name] = (
                f"Raw value of {term} at the calibration call, constant "
                "afterwards, and logged. Divide by it to express a term "
                "relative to its initial magnitude on this design."
            )
        elif name in PRIVILEGED_OBSERVABLE_REGISTRY:
            descriptions[name] = PRIVILEGED_OBSERVABLE_REGISTRY[name]
        else:
            descriptions[name] = OBSERVABLE_REGISTRY[name]
    return descriptions


def term_names(scope: str = "tier1") -> list[str]:
    if scope == "tier1":
        return sorted(name for name, spec in TERM_REGISTRY.items() if spec.is_tier1)
    if scope in {"dreamplace", "dreamplace_deployable"}:
        return sorted(name for name, spec in TERM_REGISTRY.items() if spec.is_dreamplace)
    if scope == "dreamplace_native_residual":
        return sorted(
            [
                "native_objective",
                "soft_rudy_mean",
                "soft_rudy_pnorm",
                "route_pressure_long",
                "pin_density_pnorm",
                "pin_count_weighted_wl",
            ]
        )
    if scope in {"dreamplace_replacement", "dreamplace_controller"}:
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
