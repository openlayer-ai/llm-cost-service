"""Unit tests for costs.providers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from service.costs.entities import LlmCostEntity
from service.costs.providers import (
    _LITELLM_PRICE_FIELDS,
    _OPENROUTER_PRICE_FIELDS,
    LiteLLMCostProvider,
    OpenRouterCostProvider,
    _build_price_details,
    _coerce_price,
    _litellm_is_chat_capable,
    _openrouter_is_chat_capable,
)


class TestCoercePrice:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (1e-6, 1e-6),  # float (litellm)
            (0, 0.0),  # zero is a valid price
            ("2.5e-6", 2.5e-6),  # string (openrouter)
            ("0", 0.0),
        ],
    )
    def test_returns_float_for_usable_numbers(self, value, expected):
        assert _coerce_price(value) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "value",
        [None, "", "abc", "n/a", [], {}, -1e-6, "-0.5", True, False],
    )
    def test_returns_none_for_unusable_values(self, value):
        # bool is an int subclass but must be rejected; negatives rejected too.
        assert _coerce_price(value) is None


class TestBuildPriceDetails:
    def test_maps_present_fields_to_canonical_keys(self):
        source = {
            "cache_read_input_token_cost": 1.25e-6,
            "cache_creation_input_token_cost": 3e-6,
            "input_cost_per_audio_token": 4e-5,
        }
        result = _build_price_details(source, _LITELLM_PRICE_FIELDS)
        assert result == {
            "cached_tokens": 1.25e-6,
            "cache_creation_tokens": 3e-6,
            "audio_input_tokens": 4e-5,
        }

    def test_omits_absent_and_unusable_fields(self):
        source = {
            "cache_read_input_token_cost": 1.25e-6,
            "cache_creation_input_token_cost": None,  # absent value
            "input_cost_per_audio_token": "oops",  # not a number
        }
        result = _build_price_details(source, _LITELLM_PRICE_FIELDS)
        assert result == {"cached_tokens": 1.25e-6}

    def test_empty_when_no_granular_fields(self):
        assert _build_price_details({"input_cost_per_token": 1e-6}, _LITELLM_PRICE_FIELDS) == {}

    def test_does_not_include_base_input_output(self):
        # input/output live at the root scalars, not in price_details.
        assert "input_tokens" not in _LITELLM_PRICE_FIELDS
        assert "output_tokens" not in _LITELLM_PRICE_FIELDS
        assert "input_tokens" not in _OPENROUTER_PRICE_FIELDS
        assert "output_tokens" not in _OPENROUTER_PRICE_FIELDS

    def test_openrouter_string_prices_coerced(self):
        result = _build_price_details(
            {"input_cache_read": "1.5e-7", "input_cache_write": "5e-7"},
            _OPENROUTER_PRICE_FIELDS,
        )
        assert result == {"cached_tokens": 1.5e-7, "cache_creation_tokens": 5e-7}


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

    def test_strips_redundant_provider_prefix(self):
        # azure keys carry a redundant "azure/" prefix that duplicates the
        # provider column; it must be stripped from the model name.
        fake = {"azure/codex-mini": _entry(litellm_provider="azure")}
        with patch("litellm.model_cost", fake):
            e = LiteLLMCostProvider().fetch_costs()[0]
        assert e.provider == "azure"
        assert e.model == "codex-mini"

    def test_preserves_region_sub_namespace(self):
        # Only the leading "<provider>/" segment is stripped, so region
        # sub-namespaces (which carry distinct pricing) survive.
        fake = {"azure/eu/gpt-4o-2024-08-06": _entry(litellm_provider="azure")}
        with patch("litellm.model_cost", fake):
            e = LiteLLMCostProvider().fetch_costs()[0]
        assert e.model == "eu/gpt-4o-2024-08-06"

    def test_leaves_bare_keys_untouched(self):
        # First-party vendors (and the odd unprefixed azure key) are already
        # bare and must pass through unchanged.
        fake = {
            "gpt-4o": _entry(litellm_provider="openai"),
            "computer-use-preview": _entry(litellm_provider="azure"),
        }
        with patch("litellm.model_cost", fake):
            models = {(e.provider, e.model) for e in LiteLLMCostProvider().fetch_costs()}
        assert models == {("openai", "gpt-4o"), ("azure", "computer-use-preview")}

    def test_only_strips_matching_provider_prefix(self):
        # A "/" in the key that isn't the provider prefix stays put.
        fake = {"foo/bar": _entry(litellm_provider="openai")}
        with patch("litellm.model_cost", fake):
            e = LiteLLMCostProvider().fetch_costs()[0]
        assert e.model == "foo/bar"

    @pytest.mark.parametrize("bare_first", [True, False])
    def test_dedupes_collision_keeps_priced_over_zero(self, bare_first):
        # Mirrors the real "gemini-exp-1206" quirk: the bare key holds the real
        # price and the prefixed key is a zero-priced alias. One row survives,
        # and the non-zero price wins regardless of dict order.
        bare = ("gemini-exp-1206", _entry(
            litellm_provider="gemini", input_cost_per_token=3e-7, output_cost_per_token=2.5e-6
        ))
        prefixed = ("gemini/gemini-exp-1206", _entry(
            litellm_provider="gemini", input_cost_per_token=0, output_cost_per_token=0
        ))
        items = [bare, prefixed] if bare_first else [prefixed, bare]
        with patch("litellm.model_cost", dict(items)):
            result = LiteLLMCostProvider().fetch_costs()
        assert len(result) == 1
        e = result[0]
        assert (e.provider, e.model) == ("gemini", "gemini-exp-1206")
        assert e.prompt_cost_per_token == 3e-7
        assert e.completion_cost_per_token == 2.5e-6

    @pytest.mark.parametrize("bare_first", [True, False])
    def test_dedupes_collision_prefixed_breaks_price_tie(self, bare_first):
        # When price doesn't decide (both priced), the prefixed/canonical entry
        # wins the tie — deterministically, not by dict order. Distinct prices
        # here only to observe which entry survived.
        bare = ("computer-use-preview", _entry(
            litellm_provider="azure", input_cost_per_token=1e-6, output_cost_per_token=1e-6
        ))
        prefixed = ("azure/computer-use-preview", _entry(
            litellm_provider="azure", input_cost_per_token=3e-6, output_cost_per_token=12e-6
        ))
        items = [bare, prefixed] if bare_first else [prefixed, bare]
        with patch("litellm.model_cost", dict(items)):
            result = LiteLLMCostProvider().fetch_costs()
        assert len(result) == 1
        e = result[0]
        assert (e.provider, e.model) == ("azure", "computer-use-preview")
        assert e.prompt_cost_per_token == 3e-6  # the prefixed entry's price

    def test_price_details_carries_only_granular_extras(self):
        fake = {
            "gpt-4o": _entry(
                input_cost_per_token=2.5e-6,
                output_cost_per_token=10e-6,
                cache_read_input_token_cost=1.25e-6,
                input_cost_per_audio_token=4e-5,
            )
        }
        with patch("litellm.model_cost", fake):
            e = LiteLLMCostProvider().fetch_costs()[0]
        # base input/output stay on the root scalars, not duplicated here.
        assert e.price_details == {"cached_tokens": 1.25e-6, "audio_input_tokens": 4e-5}

    def test_price_details_empty_for_basic_model(self):
        fake = {"gpt-4o-mini": _entry()}
        with patch("litellm.model_cost", fake):
            e = LiteLLMCostProvider().fetch_costs()[0]
        assert e.price_details == {}

    @pytest.mark.parametrize("mode", ["chat", "responses", "Chat", "  responses  "])
    def test_is_chat_capable_true_for_chat_modes(self, mode):
        # "responses" matters: Responses-API-only models (o1-pro, gpt-5-pro,
        # gpt-5-codex) are tagged mode="responses" but are still chat targets.
        fake = {"m": _entry(mode=mode)}
        with patch("litellm.model_cost", fake):
            assert LiteLLMCostProvider().fetch_costs()[0].is_chat_capable is True

    @pytest.mark.parametrize(
        "mode",
        [
            "embedding",
            "image_generation",
            "audio_speech",
            "audio_transcription",
            "moderation",
            "rerank",
            "completion",  # legacy text-completion endpoint, not a chat target
        ],
    )
    def test_is_chat_capable_false_for_non_chat_modes(self, mode):
        fake = {"m": _entry(mode=mode)}
        with patch("litellm.model_cost", fake):
            assert LiteLLMCostProvider().fetch_costs()[0].is_chat_capable is False

    def test_is_chat_capable_none_when_mode_absent(self):
        # No mode in the entry -> unknown -> NULL (consumer falls back).
        fake = {"m": _entry()}
        with patch("litellm.model_cost", fake):
            assert LiteLLMCostProvider().fetch_costs()[0].is_chat_capable is None


class TestLiteLLMIsChatCapableHelper:
    @pytest.mark.parametrize(
        "entry,expected",
        [
            ({"mode": "chat"}, True),
            ({"mode": "responses"}, True),
            ({"mode": "embedding"}, False),
            ({"mode": "image_generation"}, False),
            ({"mode": ""}, None),
            ({"mode": None}, None),
            ({}, None),  # no mode key at all
        ],
    )
    def test_classifies_from_mode(self, entry, expected):
        assert _litellm_is_chat_capable(entry) is expected


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

    def test_price_details_carries_only_granular_extras(self):
        payload = {
            "data": [
                _or_entry(
                    "anthropic/claude-x", "5e-6", "25e-6", input_cache_read="5e-7"
                )
            ]
        }
        httpx_mod, _ = _stub_httpx(payload)
        with patch.dict("sys.modules", {"httpx": httpx_mod}):
            e = OpenRouterCostProvider().fetch_costs()[0]
        assert e.price_details == {"cached_tokens": 5e-7}

    def test_uses_configured_base_url(self):
        payload = {"data": []}
        httpx_mod, captured = _stub_httpx(payload)
        with patch.dict("sys.modules", {"httpx": httpx_mod}):
            OpenRouterCostProvider(base_url="https://example.com/api/v1").fetch_costs()
        assert captured == ["https://example.com/api/v1/models"]

    def test_is_chat_capable_true_for_text_output(self):
        entry = _or_entry("openai/gpt-x", "1e-6", "2e-6")
        entry["architecture"] = {"output_modalities": ["text"]}
        httpx_mod, _ = _stub_httpx({"data": [entry]})
        with patch.dict("sys.modules", {"httpx": httpx_mod}):
            assert OpenRouterCostProvider().fetch_costs()[0].is_chat_capable is True

    def test_is_chat_capable_true_when_text_among_multiple_outputs(self):
        # Multimodal output that still includes text is a usable chat target.
        entry = _or_entry("google/gemini-x", "1e-6", "2e-6")
        entry["architecture"] = {"output_modalities": ["image", "text"]}
        httpx_mod, _ = _stub_httpx({"data": [entry]})
        with patch.dict("sys.modules", {"httpx": httpx_mod}):
            assert OpenRouterCostProvider().fetch_costs()[0].is_chat_capable is True

    def test_is_chat_capable_false_for_non_text_output(self):
        entry = _or_entry("vendor/image-only", "1e-6", "2e-6")
        entry["architecture"] = {"output_modalities": ["image"]}
        httpx_mod, _ = _stub_httpx({"data": [entry]})
        with patch.dict("sys.modules", {"httpx": httpx_mod}):
            assert OpenRouterCostProvider().fetch_costs()[0].is_chat_capable is False

    def test_is_chat_capable_none_when_architecture_absent(self):
        # _or_entry produces no architecture key -> unknown -> NULL.
        payload = {"data": [_or_entry("openai/gpt-x", "1e-6", "2e-6")]}
        httpx_mod, _ = _stub_httpx(payload)
        with patch.dict("sys.modules", {"httpx": httpx_mod}):
            assert OpenRouterCostProvider().fetch_costs()[0].is_chat_capable is None


class TestOpenRouterIsChatCapableHelper:
    @pytest.mark.parametrize(
        "entry,expected",
        [
            ({"architecture": {"output_modalities": ["text"]}}, True),
            ({"architecture": {"output_modalities": ["text", "image"]}}, True),
            ({"architecture": {"output_modalities": ["image"]}}, False),
            ({"architecture": {"output_modalities": []}}, None),
            ({"architecture": {}}, None),  # no modalities key
            ({}, None),  # no architecture key
        ],
    )
    def test_classifies_from_output_modalities(self, entry, expected):
        assert _openrouter_is_chat_capable(entry) is expected
