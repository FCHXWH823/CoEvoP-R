import json
import sys
from types import SimpleNamespace

import pytest

from coevop.llm.providers import (
    AnthropicProvider,
    MockProvider,
    OpenAIProvider,
    provider_from_name,
    provider_status,
)
from coevop.objectives.program import program_example
from coevop.objectives.spec import ObjectiveSpec


def test_mock_provider_generates_valid_objective_spec() -> None:
    provider = MockProvider()

    spec = provider.generate(context={"note": "test"})

    assert isinstance(spec, ObjectiveSpec)
    assert spec.created_by == "mock"
    assert spec.id.startswith("obj_")
    assert spec.term_set
    assert spec.source_program is not None
    assert spec.components


def test_mock_provider_mutates_with_parent_lineage() -> None:
    provider = MockProvider()
    parent = provider.generate()

    child = provider.mutate([parent], feedback={"score": 0.4})

    assert child.parent_ids == [parent.id]
    assert child.created_by == "mock"


def test_mock_provider_can_generate_dreamplace_scoped_objective() -> None:
    provider = MockProvider()

    spec = provider.generate(term_scope="dreamplace")

    assert spec.term_set == [
        "density_bell",
        "pin_density_pnorm",
        "route_pressure_long",
        "wirelength_lse",
    ]
    assert "route * pins" in spec.source_program


def test_mock_provider_can_generate_traced_replacement_objective() -> None:
    provider = MockProvider()

    trace = provider.generate_traced(term_scope="dreamplace_replacement")

    assert trace.spec.term_set == [
        "density_bell",
        "pin_density_pnorm",
        "route_pressure_long",
        "wirelength_lse",
    ]
    assert trace.metadata["provider"] == "mock"
    assert trace.raw_response is not None
    assert trace.messages


def test_provider_registry_and_status() -> None:
    assert isinstance(provider_from_name("mock"), MockProvider)

    status = provider_status()
    assert status["mock"]["available"] is True
    assert "openai" in status
    assert "qwen" in status
    assert status["anthropic"]["model_env"] == "ANTHROPIC_MODEL"


def test_openai_provider_defaults_to_paper_model(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    provider = OpenAIProvider(api_key="test-key")

    assert provider.model == "gpt-5.4"


class _FakeAnthropicSDK:
    """Stand-in for the Anthropic SDK module that records one streamed request."""

    def __init__(self, text: str, stop_reason: str = "end_turn") -> None:
        self.requests: list[dict] = []
        self.client_kwargs: list[dict] = []
        sdk = self

        class _Stream:
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def get_final_message(self):
                payload = {
                    "content": [
                        {"type": "thinking", "thinking": ""},
                        {"type": "text", "text": text},
                    ],
                    "stop_reason": stop_reason,
                    "usage": {"input_tokens": 11, "output_tokens": 7},
                }
                return SimpleNamespace(stop_reason=stop_reason, to_dict=lambda: payload)

        class Anthropic:
            def __init__(self, **kwargs):
                sdk.client_kwargs.append(kwargs)
                self.messages = SimpleNamespace(stream=self._stream)

            def _stream(self, **request):
                sdk.requests.append(request)
                return _Stream()

        self.Anthropic = Anthropic


def _controller_payload() -> str:
    return json.dumps(
        {
            "id": "",
            "rationale": "test",
            "parent_ids": [],
            "declared_term_usage": ["density_electric", "soft_rudy_pnorm", "wirelength_wawl"],
            "change_description": "test",
            "objective_program": program_example("dreamplace_controller"),
        }
    )


def test_anthropic_provider_requests_a_schema_constrained_program(monkeypatch) -> None:
    sdk = _FakeAnthropicSDK(_controller_payload())
    monkeypatch.setitem(sys.modules, "anthropic", sdk)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)

    provider = provider_from_name("anthropic")
    trace = provider.generate_traced(
        context={
            "prompt_messages": [
                {"role": "system", "content": "interface"},
                {"role": "user", "content": "propose"},
            ]
        },
        term_scope="dreamplace_controller",
    )

    assert isinstance(provider, AnthropicProvider)
    # The SDK resolves credentials itself when no key is injected.
    assert sdk.client_kwargs == [{}]
    request = sdk.requests[0]
    assert request["model"] == "claude-opus-4-8"
    assert request["system"] == "interface"
    assert request["messages"] == [{"role": "user", "content": "propose"}]
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert "objective_program" in request["output_config"]["format"]["schema"]["required"]
    assert "temperature" not in request
    assert trace.spec.is_typed_policy
    assert trace.usage == {"input_tokens": 11, "output_tokens": 7}
    assert trace.metadata["resolved_model"] == "claude-opus-4-8"


def test_anthropic_provider_surfaces_declined_and_truncated_turns(monkeypatch) -> None:
    for stop_reason in ("refusal", "max_tokens"):
        monkeypatch.setitem(
            sys.modules, "anthropic", _FakeAnthropicSDK(_controller_payload(), stop_reason)
        )
        with pytest.raises(RuntimeError, match=stop_reason):
            AnthropicProvider(model="claude-opus-4-8").generate_traced(
                context={"prompt_messages": [{"role": "user", "content": "propose"}]},
                term_scope="dreamplace_controller",
            )
