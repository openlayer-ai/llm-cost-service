"""Unit tests for costs.repositories."""

from __future__ import annotations

import pytest_asyncio

from service.costs.entities import LlmCostEntity
from service.costs.repositories import LlmCostRepository


@pytest_asyncio.fixture
async def repository(session) -> LlmCostRepository:
    return LlmCostRepository(session)


class TestGet:
    async def test_returns_entity_when_found(self, repository, openai_entity):
        await repository.upsert_all([openai_entity])
        result = await repository.get(provider="openai", model="gpt-4o")
        assert result is not None
        assert result.provider == "openai"
        assert result.model == "gpt-4o"

    async def test_returns_none_when_model_not_found(self, repository, openai_entity):
        await repository.upsert_all([openai_entity])
        result = await repository.get(provider="openai", model="nonexistent")
        assert result is None

    async def test_returns_none_when_provider_not_found(self, repository, openai_entity):
        await repository.upsert_all([openai_entity])
        result = await repository.get(provider="unknown", model="gpt-4o")
        assert result is None

    async def test_is_case_sensitive(self, repository, openai_entity):
        await repository.upsert_all([openai_entity])
        result = await repository.get(provider="OpenAI", model="gpt-4o")
        assert result is None


class TestListByProvider:
    async def test_returns_matching_rows(self, repository, openai_entity, anthropic_entity):
        await repository.upsert_all([openai_entity, anthropic_entity])
        result = await repository.list_by_provider(provider="openai")
        assert len(result) == 1
        assert result[0].provider == "openai"

    async def test_returns_empty_list_for_unknown_provider(self, repository):
        result = await repository.list_by_provider(provider="unknown")
        assert result == []

    async def test_excludes_other_providers(self, repository, openai_entity, anthropic_entity):
        await repository.upsert_all([openai_entity, anthropic_entity])
        result = await repository.list_by_provider(provider="openai")
        models = {e.model for e in result}
        assert "claude-3-opus-20240229" not in models


class TestListAll:
    async def test_returns_all_rows(self, repository, openai_entity, anthropic_entity):
        await repository.upsert_all([openai_entity, anthropic_entity])
        result = await repository.list_all()
        providers = {e.provider for e in result}
        assert "openai" in providers
        assert "anthropic" in providers

    async def test_returns_empty_list_when_table_empty(self, repository):
        result = await repository.list_all()
        assert result == []


class TestGetStatus:
    async def test_returns_zero_counts_when_table_empty(self, repository):
        status = await repository.get_status()
        assert status["total_models"] == 0
        assert status["total_providers"] == 0
        assert status["last_refreshed_at"] is None

    async def test_counts_models_and_providers(self, repository, openai_entity, anthropic_entity):
        await repository.upsert_all([openai_entity, anthropic_entity])
        status = await repository.get_status()
        assert status["total_models"] == 2
        assert status["total_providers"] == 2

    async def test_counts_multiple_models_per_provider_correctly(self, repository, openai_entity):
        second = LlmCostEntity("openai", "gpt-4o-mini", 1.5e-7, 6e-7, "litellm")
        await repository.upsert_all([openai_entity, second])
        status = await repository.get_status()
        assert status["total_models"] == 2
        assert status["total_providers"] == 1

    async def test_last_refreshed_at_is_set_after_upsert(self, repository, openai_entity):
        await repository.upsert_all([openai_entity])
        status = await repository.get_status()
        assert status["last_refreshed_at"] is not None


class TestUpsertAll:
    async def test_inserts_new_rows(self, repository, openai_entity):
        await repository.upsert_all([openai_entity])
        result = await repository.get(provider="openai", model="gpt-4o")
        assert result is not None

    async def test_updates_existing_rows(self, repository, openai_entity):
        await repository.upsert_all([openai_entity])
        updated = LlmCostEntity(
            provider="openai",
            model="gpt-4o",
            prompt_cost_per_token=99e-6,
            completion_cost_per_token=99e-6,
            source="litellm",
        )
        await repository.upsert_all([updated])
        result = await repository.get(provider="openai", model="gpt-4o")
        assert result.prompt_cost_per_token == 99e-6

    async def test_returns_row_count(self, repository, openai_entity, anthropic_entity):
        count = await repository.upsert_all([openai_entity, anthropic_entity])
        assert count == 2

    async def test_empty_list_returns_zero(self, repository):
        count = await repository.upsert_all([])
        assert count == 0
