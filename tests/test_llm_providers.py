from coevop.llm.providers import MockProvider, OpenAIProvider, provider_from_name, provider_status
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


def test_openai_provider_defaults_to_paper_model(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    provider = OpenAIProvider(api_key="test-key")

    assert provider.model == "gpt-5.4"
