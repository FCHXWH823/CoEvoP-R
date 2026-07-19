"""Provider adapters for objective-candidate generation."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

from coevop.llm.http import post_json
from coevop.llm.prompts import generation_messages, mutation_messages, reflection_messages
from coevop.objectives.program import (
    objective_program_schema_for_provider,
    parse_objective_payload,
)
from coevop.objectives.spec import (
    ObjectiveSpec,
    objective_spec_to_dict,
)


@dataclass(frozen=True)
class ProviderTrace:
    spec: ObjectiveSpec
    messages: list[dict[str, str]]
    raw_response: dict[str, Any] | None
    usage: dict[str, Any] | None
    metadata: dict[str, Any]


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def generate(
        self,
        context: dict[str, Any] | None = None,
        term_scope: str = "tier1",
    ) -> ObjectiveSpec:
        raise NotImplementedError

    @abstractmethod
    def mutate(
        self,
        parents: list[ObjectiveSpec],
        feedback: dict[str, Any] | None = None,
        term_scope: str = "tier1",
    ) -> ObjectiveSpec:
        raise NotImplementedError

    @abstractmethod
    def reflect(self, candidate: ObjectiveSpec, metrics: dict[str, Any]) -> str:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "resolved_model": getattr(self, "model", None),
            "base_url": getattr(self, "base_url", None),
        }

    def generate_traced(
        self,
        context: dict[str, Any] | None = None,
        term_scope: str = "tier1",
        mutation_mode: str = "full",
    ) -> ProviderTrace:
        messages = generation_messages(context, term_scope, mutation_mode=mutation_mode)
        spec = self.generate(context=context, term_scope=term_scope)
        return ProviderTrace(
            spec=spec,
            messages=messages,
            raw_response=None,
            usage=None,
            metadata=self.metadata(),
        )

    def generate_many_traced(
        self,
        *,
        contexts: list[dict[str, Any] | None],
        term_scope: str = "tier1",
        mutation_mode: str = "full",
    ) -> list[ProviderTrace]:
        """Generate independent traced samples.

        The default path intentionally loops sequentially. Provider subclasses
        can override this later if their API supports safe batched independent
        samples with separate raw responses and token accounting.
        """

        return [
            self.generate_traced(
                context=context,
                term_scope=term_scope,
                mutation_mode=mutation_mode,
            )
            for context in contexts
        ]


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(self) -> None:
        self._counter = 0

    def _mock_route_coeff(self, context: dict[str, Any] | None) -> float:
        policy = dict((context or {}).get("search_policy", {}))
        cap = abs(float(policy.get("route_coeff_cap", 0.005)))
        grid = [float(value) for value in policy.get("route_coeff_grid", [])]
        safe_nonzero = sorted(
            value for value in grid if value > 0.0 and abs(value) <= cap
        )
        if safe_nonzero:
            return safe_nonzero[0]
        return min(1e-10, cap) if cap > 0.0 else 0.0

    def _mock_density_coeff(self, context: dict[str, Any] | None, default: float) -> float:
        policy = dict((context or {}).get("search_policy", {}))
        grid = [float(value) for value in policy.get("density_coeff_grid", [])]
        if not grid:
            return default
        return min(grid, key=lambda value: abs(value - default))

    def generate(
        self,
        context: dict[str, Any] | None = None,
        term_scope: str = "tier1",
    ) -> ObjectiveSpec:
        if (
            term_scope == "dreamplace_controller"
            or str((context or {}).get("objective_mode", "")) == "controller"
        ):
            index = self._counter
            self._counter += 1
            growth_center = [0.15, 0.2, 0.25][index % 3]
            route_gain = [0.0, 0.0005, 0.001][index % 3]
            payload = {
                "id": "",
                "rationale": (
                    "Mock stateful controller: gradient-ratio calibrated density "
                    "ramp gated on overflow, with an optional late routing term."
                ),
                "parent_ids": [],
                "declared_term_usage": sorted(
                    ["wirelength_wawl", "density_electric", "soft_rudy_pnorm"]
                    if route_gain > 0
                    else ["wirelength_wawl", "density_electric"]
                ),
                "change_description": (
                    "Mock controller edit: adjust the overflow gate center and "
                    "routing ramp gain."
                ),
                "objective_program": "\n".join(
                    [
                        "def init_policy(obs):",
                        "    return {",
                        '        "density_weight": 0.00008 * obs("grad_ratio_density_electric"),',
                        '        "gamma_scale": 1.0,',
                    ]
                    + ([f'        "route_weight": {route_gain:.12g},'] if route_gain > 0 else [])
                    + [
                        "    }",
                        "",
                        "def update_policy(policy, obs):",
                        '    density_weight = policy("density_weight") * (1.0 + 0.05 * sigmoid(8.0 * '
                        f'(obs("overflow") - {growth_center:.12g})))',
                        '    gamma_scale = 1.0 + 9.0 * sigmoid(8.0 * (obs("overflow") - 0.5))',
                    ]
                    + (
                        [
                            '    route_weight = policy("route_weight") * (1.0 + 0.5 * obs("iter_frac"))',
                            '    return {"density_weight": density_weight, "gamma_scale": gamma_scale, "route_weight": route_weight}',
                        ]
                        if route_gain > 0
                        else [
                            '    return {"density_weight": density_weight, "gamma_scale": gamma_scale}',
                        ]
                    )
                    + [
                        "",
                        "def objective(features, policy):",
                        '    wl = term("wirelength_wawl")',
                        '    den = term("density_electric")',
                    ]
                    + (
                        [
                            '    route = log1p(term("soft_rudy_pnorm"))',
                            '    score = wl + policy("density_weight") * den + policy("route_weight") * route',
                            (
                                '    return score, {"wirelength": wl, "density": den, '
                                '"route_hotspot": route}'
                            ),
                        ]
                        if route_gain > 0
                        else [
                            '    score = wl + policy("density_weight") * den',
                            '    return score, {"wirelength": wl, "density": den}',
                        ]
                    )
                ),
            }
            return parse_objective_payload(
                payload,
                created_by=self.name,
                term_scope="dreamplace_controller",
            )
        if term_scope in {"dreamplace", "dreamplace_replacement", "dreamplace_native_residual"}:
            if str((context or {}).get("objective_mode", "")) == "native_residual":
                route_coeff = self._mock_route_coeff(context)
                payloads = [
                    {
                        "id": "",
                        "rationale": (
                            "Native-residual mock objective preserving DREAMPlace "
                            "default and adding calibrated soft-RUDY pressure."
                        ),
                        "parent_ids": [],
                        "declared_term_usage": ["native_objective", "soft_rudy_mean"],
                        "change_description": (
                            "Mock native-residual edit: keep native_objective as "
                            "the base and add a small route-demand correction."
                        ),
                        "objective_program": "\n".join(
                            [
                                "def objective(features):",
                                '    native = term("native_objective")',
                                '    route = term("soft_rudy_mean")',
                                f"    score = native + {route_coeff:.12g} * log1p(route)",
                                '    return score, {"native": native, "route_mean": route}',
                            ]
                        ),
                    },
                    {
                        "id": "",
                        "rationale": (
                            "Native-residual mock objective preserving DREAMPlace "
                            "default and adding long-net route pressure."
                        ),
                        "parent_ids": [],
                        "declared_term_usage": ["native_objective", "route_pressure_long"],
                        "change_description": (
                            "Mock native-residual edit: keep native_objective and "
                            "add a calibrated long-route correction."
                        ),
                        "objective_program": "\n".join(
                            [
                                "def objective(features):",
                                '    native = term("native_objective")',
                                '    long_route = term("route_pressure_long")',
                                f"    score = native + {route_coeff:.12g} * sqrt(long_route)",
                                '    return score, {"native": native, "long_route": long_route}',
                            ]
                        ),
                    },
                ]
                payload = payloads[self._counter % len(payloads)]
                self._counter += 1
                return parse_objective_payload(
                    payload,
                    created_by=self.name,
                    term_scope=term_scope,
                )
            route_coeff = self._mock_route_coeff(context)
            density_coeff = self._mock_density_coeff(context, 1.26)
            payloads = [
                {
                    "id": "",
                    "rationale": (
                        "DREAMPlace replacement mock objective using alternate "
                        "wirelength/density models and a route-pin interaction."
                    ),
                    "parent_ids": [],
                    "declared_term_usage": [
                        "density_bell",
                        "pin_density_pnorm",
                        "route_pressure_long",
                        "wirelength_lse",
                    ],
                    "change_description": (
                        "Mock full rewrite: replace default-style WAWL/electric "
                        "density with LSE/bell terms and add log1p(route*pins)."
                    ),
                    "objective_program": "\n".join(
                        [
                            "def objective(features):",
                            '    wl = term("wirelength_lse")',
                            '    den = term("density_bell")',
                            '    route = term("route_pressure_long")',
                            '    pins = term("pin_density_pnorm")',
                            "    access_pressure = log1p(route * pins)",
                            (
                                f"    score = wl + {density_coeff:.12g} * den + "
                                f"{route_coeff:.12g} * access_pressure"
                            ),
                            (
                                '    return score, {"wirelength": wl, "density": den, '
                                '"route_hotspot": route, "pin_access": pins}'
                            ),
                        ]
                    ),
                },
                {
                    "id": "",
                    "rationale": (
                        "DREAMPlace replacement mock objective with pin-access pressure "
                        "coupled to hotspot route demand."
                    ),
                    "parent_ids": [],
                    "declared_term_usage": [
                        "density_bell",
                        "pin_density_pnorm",
                        "soft_rudy_pnorm",
                        "wirelength_lse",
                    ],
                    "change_description": (
                        "Mock full rewrite: combine soft RUDY hotspot and pin "
                        "access through a compact sqrt(route*pins) interaction."
                    ),
                    "objective_program": "\n".join(
                        [
                            "def objective(features):",
                            '    wl = term("wirelength_lse")',
                            '    den = term("density_bell")',
                            '    route = term("soft_rudy_pnorm")',
                            '    pins = term("pin_density_pnorm")',
                            "    access_pressure = sqrt(route * pins)",
                            (
                                f"    score = wl + {density_coeff:.12g} * den + "
                                f"{route_coeff:.12g} * access_pressure"
                            ),
                            (
                                '    return score, {"wirelength": wl, "density": den, '
                                '"route_hotspot": route, "pin_access": pins}'
                            ),
                        ]
                    ),
                },
                {
                    "id": "",
                    "rationale": (
                        "DREAMPlace replacement mock objective balancing mean and "
                        "hotspot route pressure."
                    ),
                    "parent_ids": [],
                    "declared_term_usage": [
                        "density_electric",
                        "soft_rudy_mean",
                        "soft_rudy_pnorm",
                        "wirelength_wawl",
                    ],
                    "change_description": (
                        "Mock full rewrite: use a route-only mean-hotspot "
                        "interaction rather than a single added route term."
                    ),
                    "objective_program": "\n".join(
                        [
                            "def objective(features):",
                            '    wl = term("wirelength_wawl")',
                            '    den = term("density_electric")',
                            '    route = term("soft_rudy_pnorm")',
                            '    route_mean = term("soft_rudy_mean")',
                            "    route_balance = log1p(route * route_mean)",
                            (
                                f"    score = wl + {density_coeff:.12g} * den + "
                                f"{route_coeff:.12g} * route_balance"
                            ),
                            (
                                '    return score, {"wirelength": wl, "density": den, '
                                '"route_hotspot": route, "route_mean": route_mean}'
                            ),
                        ]
                    ),
                },
                {
                    "id": "",
                    "rationale": (
                        "DREAMPlace replacement mock objective emphasizing high-fanout "
                        "wirelength and long-net route pressure."
                    ),
                    "parent_ids": [],
                    "declared_term_usage": [
                        "density_electric",
                        "pin_count_weighted_wl",
                        "route_pressure_long",
                        "wirelength_wawl",
                    ],
                    "change_description": (
                        "Mock full rewrite: add a compact long-route and high-fanout "
                        "wirelength interaction as a distinct mechanism."
                    ),
                    "objective_program": "\n".join(
                        [
                            "def objective(features):",
                            '    wl = term("wirelength_wawl")',
                            '    den = term("density_electric")',
                            '    long_route = term("route_pressure_long")',
                            '    fanout_wl = term("pin_count_weighted_wl")',
                            "    pressure = sqrt(long_route * fanout_wl)",
                            (
                                f"    score = wl + {density_coeff:.12g} * den + "
                                f"{route_coeff:.12g} * pressure"
                            ),
                            (
                                '    return score, {"wirelength": wl, "density": den, '
                                '"long_route": long_route, "fanout_wl": fanout_wl}'
                            ),
                        ]
                    ),
                },
            ]
            payload = payloads[self._counter % len(payloads)]
            self._counter += 1
            return parse_objective_payload(
                payload,
                created_by=self.name,
                term_scope=term_scope,
            )
        if term_scope == "deployable":
            payload = {
                "id": "",
                "rationale": (
                    "Deployable bridge objective over soft route demand and "
                    "pin-access pressure."
                ),
                "parent_ids": [],
                "declared_term_usage": [
                    "pin_density_pnorm",
                    "soft_rudy_mean",
                    "soft_rudy_pnorm",
                ],
                "change_description": (
                    "Mock full rewrite: combine route hotspot, pin-access, and "
                    "mean route demand into a compact bridge objective."
                ),
                "objective_program": "\n".join(
                    [
                        "def objective(features):",
                        '    route = term("soft_rudy_pnorm")',
                        '    pins = term("pin_density_pnorm")',
                        '    mean_route = term("soft_rudy_mean")',
                        "    score = route + 0.35 * pins + 0.2 * mean_route",
                        (
                            '    return score, {"route_hotspot": route, '
                            '"pin_access": pins, "route_mean": mean_route}'
                        ),
                    ]
                ),
            }
            return parse_objective_payload(
                payload,
                created_by=self.name,
                term_scope=term_scope,
            )
        payloads = [
            {
                "id": "",
                "rationale": (
                    "Combine RUDY hotspot pressure with cell-density hotspot pressure."
                ),
                "parent_ids": [],
                "declared_term_usage": ["cell_density_p95", "rudy_p95"],
                "change_description": "Mock full rewrite over Tier-1 proxy features.",
                "objective_program": "\n".join(
                    [
                        "def objective(features):",
                        '    rudy = term("rudy_p95")',
                        '    density = term("cell_density_p95")',
                        "    score = rudy + 0.35 * density",
                        (
                            '    return score, {"rudy_hotspot": rudy, '
                            '"density_hotspot": density}'
                        ),
                    ]
                ),
            },
            {
                "id": "",
                "rationale": (
                    "Blend long-net and pin-aware routing pressure for congestion filtering."
                ),
                "parent_ids": [],
                "declared_term_usage": ["rudy_long_p95", "rudy_pin_long_p95"],
                "change_description": "Mock full rewrite over long-net proxy features.",
                "objective_program": "\n".join(
                    [
                        "def objective(features):",
                        '    long_net = term("rudy_long_p95")',
                        '    pin_long = term("rudy_pin_long_p95")',
                        "    score = long_net + 0.75 * pin_long",
                        (
                            '    return score, {"long_net_rudy": long_net, '
                            '"pin_long_rudy": pin_long}'
                        ),
                    ]
                ),
            },
        ]
        payload = payloads[self._counter % len(payloads)]
        self._counter += 1
        return parse_objective_payload(
            payload,
            created_by=self.name,
            term_scope=term_scope,
        )

    def mutate(
        self,
        parents: list[ObjectiveSpec],
        feedback: dict[str, Any] | None = None,
        term_scope: str = "tier1",
    ) -> ObjectiveSpec:
        parent_ids = [parent.id for parent in parents]
        if term_scope in {"dreamplace", "dreamplace_replacement", "dreamplace_native_residual"}:
            route_coeff = self._mock_route_coeff(feedback)
            density_coeff = self._mock_density_coeff(feedback, 1.26)
            if term_scope == "dreamplace_native_residual":
                payload = {
                    "id": "",
                    "rationale": (
                        "Mock native-residual mutation keeps DREAMPlace's native "
                        "objective and revises the routing correction mechanism."
                    ),
                    "parent_ids": parent_ids,
                    "declared_term_usage": ["native_objective", "soft_rudy_pnorm"],
                    "change_description": "\n".join(
                        [
                            "<<<<<<< SEARCH",
                            "score = native + route_coeff * route",
                            "=======",
                            "score = native + route_coeff * log1p(route)",
                            ">>>>>>> REPLACE",
                        ]
                    ),
                    "objective_program": "\n".join(
                        [
                            "def objective(features):",
                            '    native = term("native_objective")',
                            '    route = term("soft_rudy_pnorm")',
                            f"    score = native + {route_coeff:.12g} * log1p(route)",
                            '    return score, {"native": native, "route_hotspot": route}',
                        ]
                    ),
                }
                return parse_objective_payload(
                    payload,
                    created_by=self.name,
                    term_scope=term_scope,
                )
            payload = {
                "id": "",
                "rationale": (
                    "Mock DREAMPlace replacement mutation introduces a route-pin "
                    "interaction while keeping explicit wirelength and density support."
                ),
                "parent_ids": parent_ids,
                "declared_term_usage": [
                    "density_bell",
                    "pin_density_pnorm",
                    "route_pressure_long",
                    "wirelength_lse",
                ],
                "change_description": "\n".join(
                    [
                        "<<<<<<< SEARCH",
                        "score = wl + density * den + route_coeff * route",
                        "=======",
                        "access_pressure = log1p(route * pins)",
                        "score = wl + density * den + route_coeff * access_pressure",
                        ">>>>>>> REPLACE",
                    ]
                ),
                "objective_program": "\n".join(
                    [
                        "def objective(features):",
                        '    wl = term("wirelength_lse")',
                        '    den = term("density_bell")',
                        '    route = term("route_pressure_long")',
                        '    pins = term("pin_density_pnorm")',
                        "    access_pressure = log1p(route * pins)",
                        (
                            f"    score = wl + {density_coeff:.12g} * den + "
                            f"{route_coeff:.12g} * access_pressure"
                        ),
                        (
                            '    return score, {"wirelength": wl, "density": den, '
                            '"route_mean": route, "pin_access": pins}'
                        ),
                    ]
                ),
            }
            return parse_objective_payload(
                payload,
                created_by=self.name,
                term_scope=term_scope,
            )
        if term_scope == "deployable":
            payload = {
                "id": "",
                "rationale": (
                    "Mock deployable mutation emphasizes route-demand hotspots "
                    "while retaining density pressure."
                ),
                "parent_ids": parent_ids,
                "declared_term_usage": [
                    "pin_density_pnorm",
                    "soft_rudy_mean",
                    "soft_rudy_pnorm",
                ],
                "change_description": "Mock deployable mutation over route-demand terms.",
                "objective_program": "\n".join(
                    [
                        "def objective(features):",
                        '    route_hot = term("soft_rudy_pnorm")',
                        '    route_mean = term("soft_rudy_mean")',
                        '    pins = term("pin_density_pnorm")',
                        "    score = route_hot + 0.2 * route_mean + 0.25 * pins",
                        (
                            '    return score, {"route_hotspot": route_hot, '
                            '"route_mean": route_mean, "pin_access": pins}'
                        ),
                    ]
                ),
            }
            return parse_objective_payload(
                payload,
                created_by=self.name,
                term_scope=term_scope,
            )
        payload = {
            "id": "",
            "rationale": "Mock mutation adds density pressure to a RUDY hotspot parent.",
            "parent_ids": parent_ids,
            "declared_term_usage": ["cell_density_p95", "rudy_pin_p95"],
            "change_description": "Mock Tier-1 mutation over RUDY pin pressure and density.",
            "objective_program": "\n".join(
                [
                    "def objective(features):",
                    '    pin_rudy = term("rudy_pin_p95")',
                    '    density = term("cell_density_p95")',
                    "    score = pin_rudy + 0.25 * density",
                    (
                        '    return score, {"pin_rudy": pin_rudy, '
                        '"density_hotspot": density}'
                    ),
                ]
            ),
        }
        return parse_objective_payload(
            payload,
            created_by=self.name,
            term_scope=term_scope,
        )

    def reflect(self, candidate: ObjectiveSpec, metrics: dict[str, Any]) -> str:
        return (
            f"Mock reflection for {candidate.id}: compare component correlations "
            "and preserve terms "
            "that improve validation congestion without overfitting train behavior."
        )

    def generate_traced(
        self,
        context: dict[str, Any] | None = None,
        term_scope: str = "tier1",
        mutation_mode: str = "full",
    ) -> ProviderTrace:
        messages = generation_messages(context, term_scope, mutation_mode=mutation_mode)
        spec = self.generate(context=context, term_scope=term_scope)
        return ProviderTrace(
            spec=spec,
            messages=messages,
            raw_response={"mock_objective": objective_spec_to_dict(spec)},
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            metadata=self.metadata(),
        )


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-5.4")
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip(
            "/",
        )
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAIProvider")

    def generate(
        self,
        context: dict[str, Any] | None = None,
        term_scope: str = "tier1",
    ) -> ObjectiveSpec:
        return self._objective_request(
            generation_messages(context, term_scope),
            term_scope=term_scope,
        )

    def generate_traced(
        self,
        context: dict[str, Any] | None = None,
        term_scope: str = "tier1",
        mutation_mode: str = "full",
    ) -> ProviderTrace:
        messages = generation_messages(context, term_scope, mutation_mode=mutation_mode)
        response = self._responses_request(messages, structured=True, term_scope=term_scope)
        payload = _parse_json_text(_extract_responses_text(response))
        spec = parse_objective_payload(payload, created_by=self.name, term_scope=term_scope)
        return ProviderTrace(
            spec=spec,
            messages=messages,
            raw_response=response,
            usage=_usage_from_response(response),
            metadata=self.metadata(),
        )

    def mutate(
        self,
        parents: list[ObjectiveSpec],
        feedback: dict[str, Any] | None = None,
        term_scope: str = "tier1",
    ) -> ObjectiveSpec:
        parent_payloads = [objective_spec_to_dict(parent) for parent in parents]
        return self._objective_request(
            mutation_messages(parent_payloads, feedback, term_scope),
            term_scope=term_scope,
        )

    def reflect(self, candidate: ObjectiveSpec, metrics: dict[str, Any]) -> str:
        response = self._responses_request(
            reflection_messages(asdict(candidate), metrics),
            structured=False,
        )
        return _extract_responses_text(response).strip()

    def _objective_request(self, messages: list[dict[str, str]], term_scope: str) -> ObjectiveSpec:
        response = self._responses_request(messages, structured=True, term_scope=term_scope)
        payload = _parse_json_text(_extract_responses_text(response))
        return parse_objective_payload(payload, created_by=self.name, term_scope=term_scope)

    def _responses_request(
        self,
        messages: list[dict[str, str]],
        structured: bool,
        term_scope: str = "tier1",
    ) -> dict[str, Any]:
        input_text = "\n\n".join(
            f"{message['role'].upper()}:\n{message['content']}" for message in messages
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_text,
            "store": False,
        }
        if structured:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "coevop_objective_program",
                    "strict": True,
                    "schema": objective_program_schema_for_provider(term_scope),
                }
            }
        response = post_json(
            f"{self.base_url}/responses",
            payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        return response


class QwenProvider(LLMProvider):
    name = "qwen"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = (
            api_key or os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        )
        self.model = model or os.environ.get("QWEN_MODEL", "qwen-plus")
        self.base_url = (
            base_url
            or os.environ.get("QWEN_BASE_URL")
            or os.environ.get("DASHSCOPE_BASE_URL")
            or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        ).rstrip("/")
        if not self.api_key:
            raise RuntimeError("QWEN_API_KEY or DASHSCOPE_API_KEY is required for QwenProvider")

    def generate(
        self,
        context: dict[str, Any] | None = None,
        term_scope: str = "tier1",
    ) -> ObjectiveSpec:
        return self._objective_request(
            generation_messages(context, term_scope),
            term_scope=term_scope,
        )

    def generate_traced(
        self,
        context: dict[str, Any] | None = None,
        term_scope: str = "tier1",
        mutation_mode: str = "full",
    ) -> ProviderTrace:
        messages = generation_messages(context, term_scope, mutation_mode=mutation_mode)
        schema_message = {
            "role": "system",
            "content": (
                "Return a single JSON object matching this JSON Schema. "
                "Schema: "
                f"{json.dumps(objective_program_schema_for_provider(term_scope), sort_keys=True)}"
            ),
        }
        request_messages = [schema_message, *messages]
        response = self._chat_request(request_messages, json_mode=True)
        payload = _parse_json_text(_extract_chat_text(response))
        spec = parse_objective_payload(payload, created_by=self.name, term_scope=term_scope)
        return ProviderTrace(
            spec=spec,
            messages=request_messages,
            raw_response=response,
            usage=_usage_from_response(response),
            metadata=self.metadata(),
        )

    def mutate(
        self,
        parents: list[ObjectiveSpec],
        feedback: dict[str, Any] | None = None,
        term_scope: str = "tier1",
    ) -> ObjectiveSpec:
        parent_payloads = [objective_spec_to_dict(parent) for parent in parents]
        return self._objective_request(
            mutation_messages(parent_payloads, feedback, term_scope),
            term_scope=term_scope,
        )

    def reflect(self, candidate: ObjectiveSpec, metrics: dict[str, Any]) -> str:
        response = self._chat_request(
            reflection_messages(asdict(candidate), metrics),
            json_mode=False,
        )
        return _extract_chat_text(response).strip()

    def _objective_request(self, messages: list[dict[str, str]], term_scope: str) -> ObjectiveSpec:
        schema_message = {
            "role": "system",
            "content": (
                "Return a single JSON object matching this JSON Schema. "
                "Schema: "
                f"{json.dumps(objective_program_schema_for_provider(term_scope), sort_keys=True)}"
            ),
        }
        response = self._chat_request([schema_message, *messages], json_mode=True)
        payload = _parse_json_text(_extract_chat_text(response))
        return parse_objective_payload(payload, created_by=self.name, term_scope=term_scope)

    def _chat_request(self, messages: list[dict[str, str]], json_mode: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        # gpt-5-family and o-series reasoning models only accept the default
        # sampling temperature; sending an explicit value is a 400.
        if not self.model.startswith(("gpt-5", "o1", "o3", "o4")):
            payload["temperature"] = 0.7
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return post_json(
            f"{self.base_url}/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )


def provider_from_name(name: str) -> LLMProvider:
    normalized = name.lower().strip()
    if normalized == "mock":
        return MockProvider()
    if normalized == "openai":
        return OpenAIProvider()
    if normalized == "qwen":
        return QwenProvider()
    raise ValueError(f"unknown LLM provider: {name}")


def provider_status() -> dict[str, dict[str, Any]]:
    return {
        "mock": {"available": True, "required_env": []},
        "openai": {
            "available": bool(os.environ.get("OPENAI_API_KEY")),
            "required_env": ["OPENAI_API_KEY"],
            "model_env": "OPENAI_MODEL",
        },
        "qwen": {
            "available": bool(
                os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
            ),
            "required_env": ["QWEN_API_KEY or DASHSCOPE_API_KEY"],
            "model_env": "QWEN_MODEL",
            "base_url_env": "QWEN_BASE_URL or DASHSCOPE_BASE_URL",
        },
    }


def _extract_responses_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return str(response["output_text"])
    chunks: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if isinstance(content.get("text"), str):
                chunks.append(content["text"])
    if chunks:
        return "".join(chunks)
    raise RuntimeError(
        f"could not extract text from Responses API payload: {response}"
    )


def _extract_chat_text(response: dict[str, Any]) -> str:
    try:
        return str(response["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"could not extract text from chat completion payload: {response}"
        ) from exc


def _parse_json_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"provider returned invalid JSON: {text[:500]}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("provider JSON response must be an object")
    return payload


def _usage_from_response(response: dict[str, Any]) -> dict[str, Any] | None:
    usage = response.get("usage")
    return dict(usage) if isinstance(usage, dict) else None
