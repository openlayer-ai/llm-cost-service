"""Unit tests for costs.entities."""

from __future__ import annotations

import dataclasses

import pytest

from service.costs.entities import LlmCostEntity


class TestLlmCostEntity:
    def test_stores_all_fields_correctly(self):
        entity = LlmCostEntity(
            provider="openai",
            model="gpt-4o",
            prompt_cost_per_token=2.5e-6,
            completion_cost_per_token=10e-6,
            source="litellm",
        )
        assert entity.provider == "openai"
        assert entity.model == "gpt-4o"
        assert entity.prompt_cost_per_token == 2.5e-6
        assert entity.completion_cost_per_token == 10e-6
        assert entity.source == "litellm"

    def test_updated_at_defaults_to_none(self):
        entity = LlmCostEntity(
            provider="openai",
            model="gpt-4o",
            prompt_cost_per_token=2.5e-6,
            completion_cost_per_token=10e-6,
            source="litellm",
        )
        assert entity.updated_at is None

    def test_is_immutable(self):
        entity = LlmCostEntity(
            provider="openai",
            model="gpt-4o",
            prompt_cost_per_token=2.5e-6,
            completion_cost_per_token=10e-6,
            source="litellm",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            entity.provider = "anthropic"  # type: ignore[misc]
