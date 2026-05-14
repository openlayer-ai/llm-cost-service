"""Unit tests for costs.providers."""

from __future__ import annotations

from unittest.mock import patch

from service.costs.entities import LlmCostEntity
from service.costs.providers import LiteLLMCostProvider


def _entry(**kwargs) -> dict:
    defaults = {
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 2e-6,
        "litellm_provider": "openai",
    }
    defaults.update(kwargs)
    return defaults


class TestLiteLLMCostProviderFetchCosts:
    def test_returns_entities_for_models_with_both_costs(self):
        fake = {
            "gpt-4o": _entry(litellm_provider="openai"),
            "claude-3-opus-20240229": _entry(litellm_provider="anthropic", input_cost_per_token=15e-6, output_cost_per_token=75e-6),
        }
        with patch("litellm.model_cost", fake):
            result = LiteLLMCostProvider().fetch_costs()
        assert len(result) == 2
        assert all(isinstance(e, LlmCostEntity) for e in result)

    def test_skips_models_missing_input_cost(self):
        fake = {"gpt-4o": _entry(input_cost_per_token=None)}
        with patch("litellm.model_cost", fake):
            result = LiteLLMCostProvider().fetch_costs()
        assert result == []

    def test_skips_models_missing_output_cost(self):
        fake = {"gpt-4o": _entry(output_cost_per_token=None)}
        with patch("litellm.model_cost", fake):
            result = LiteLLMCostProvider().fetch_costs()
        assert result == []

    def test_skips_models_missing_provider(self):
        fake = {"gpt-4o": _entry(litellm_provider=None)}
        with patch("litellm.model_cost", fake):
            result = LiteLLMCostProvider().fetch_costs()
        assert result == []

    def test_includes_zero_cost_models(self):
        fake = {"free-model": _entry(input_cost_per_token=0, output_cost_per_token=0, litellm_provider="groq")}
        with patch("litellm.model_cost", fake):
            result = LiteLLMCostProvider().fetch_costs()
        assert len(result) == 1
        assert result[0].prompt_cost_per_token == 0

    def test_entity_fields_map_correctly(self):
        fake = {"gpt-4o": {"input_cost_per_token": 2.5e-6, "output_cost_per_token": 10e-6, "litellm_provider": "openai"}}
        with patch("litellm.model_cost", fake):
            result = LiteLLMCostProvider().fetch_costs()
        e = result[0]
        assert e.provider == "openai"
        assert e.model == "gpt-4o"
        assert e.prompt_cost_per_token == 2.5e-6
        assert e.completion_cost_per_token == 10e-6
        assert e.source == "litellm"
