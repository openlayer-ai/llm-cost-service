"""Unit tests for costs.providers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from service.costs.entities import LlmCostEntity
from service.costs.providers import LiteLLMCostProvider, OpenRouterCostProvider


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


def _or_entry(model_id: str, prompt: str | None = "1e-6", completion: str | None = "2e-6", **extra) -> dict:
    pricing: dict = {}
    if prompt is not None:
        pricing["prompt"] = prompt
    if completion is not None:
        pricing["completion"] = completion
    pricing.update(extra)
    return {"id": model_id, "pricing": pricing}


def _stub_httpx(payload: dict):
    """Return a (httpx_module_mock, captured_url_list) tuple for patching."""
    captured: list[str] = []

    class _Resp:
        def json(self):
            return payload

        def raise_for_status(self):
            return None

    class _Client:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def get(self, url: str):
            captured.append(url)
            return _Resp()

    httpx_mod = MagicMock()
    httpx_mod.Client = _Client
    return httpx_mod, captured


class TestOpenRouterCostProviderFetchCosts:
    def test_parses_vanilla_entry(self):
        payload = {"data": [_or_entry("anthropic/claude-opus-4.7", "0.000005", "0.000025")]}
        httpx_mod, _ = _stub_httpx(payload)
        with patch.dict("sys.modules", {"httpx": httpx_mod}):
            result = OpenRouterCostProvider().fetch_costs()
        assert len(result) == 1
        e = result[0]
        assert e.provider == "anthropic"
        assert e.model == "claude-opus-4.7"
        assert e.prompt_cost_per_token == 5e-6
        assert e.completion_cost_per_token == 25e-6
        assert e.source == "openrouter"

    def test_strips_tilde_prefix_from_vendor(self):
        payload = {"data": [_or_entry("~anthropic/claude-opus-latest", "0.000005", "0.000025")]}
        httpx_mod, _ = _stub_httpx(payload)
        with patch.dict("sys.modules", {"httpx": httpx_mod}):
            result = OpenRouterCostProvider().fetch_costs()
        assert result[0].provider == "anthropic"
        assert result[0].model == "claude-opus-latest"

    def test_preserves_variant_suffixes(self):
        payload = {"data": [_or_entry("openai/gpt-oss-120b:free", "0", "0")]}
        httpx_mod, _ = _stub_httpx(payload)
        with patch.dict("sys.modules", {"httpx": httpx_mod}):
            result = OpenRouterCostProvider().fetch_costs()
        assert result[0].model == "gpt-oss-120b:free"

    def test_keeps_zero_priced_entries(self):
        payload = {"data": [_or_entry("baidu/cobuddy:free", "0", "0")]}
        httpx_mod, _ = _stub_httpx(payload)
        with patch.dict("sys.modules", {"httpx": httpx_mod}):
            result = OpenRouterCostProvider().fetch_costs()
        assert len(result) == 1
        assert result[0].prompt_cost_per_token == 0.0
        assert result[0].completion_cost_per_token == 0.0

    def test_skips_entries_with_missing_pricing(self):
        payload = {"data": [_or_entry("anthropic/claude-opus-4.7", prompt=None)]}
        httpx_mod, _ = _stub_httpx(payload)
        with patch.dict("sys.modules", {"httpx": httpx_mod}):
            result = OpenRouterCostProvider().fetch_costs()
        assert result == []

    def test_skips_malformed_id_without_slash(self):
        payload = {"data": [{"id": "weird-id", "pricing": {"prompt": "1e-6", "completion": "2e-6"}}]}
        httpx_mod, _ = _stub_httpx(payload)
        with patch.dict("sys.modules", {"httpx": httpx_mod}):
            result = OpenRouterCostProvider().fetch_costs()
        assert result == []

    def test_lowercases_provider(self):
        payload = {"data": [_or_entry("Anthropic/claude-opus-4.7", "0.000005", "0.000025")]}
        httpx_mod, _ = _stub_httpx(payload)
        with patch.dict("sys.modules", {"httpx": httpx_mod}):
            result = OpenRouterCostProvider().fetch_costs()
        assert result[0].provider == "anthropic"

    def test_handles_slashes_inside_slug(self):
        payload = {"data": [_or_entry("openai/some/nested-model", "1e-6", "2e-6")]}
        httpx_mod, _ = _stub_httpx(payload)
        with patch.dict("sys.modules", {"httpx": httpx_mod}):
            result = OpenRouterCostProvider().fetch_costs()
        assert result[0].provider == "openai"
        assert result[0].model == "some/nested-model"

    def test_uses_configured_base_url(self):
        payload = {"data": []}
        httpx_mod, captured = _stub_httpx(payload)
        with patch.dict("sys.modules", {"httpx": httpx_mod}):
            OpenRouterCostProvider(base_url="https://example.com/api/v1").fetch_costs()
        assert captured == ["https://example.com/api/v1/models"]
