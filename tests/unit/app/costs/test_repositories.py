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


class TestListModelNamesByProvider:
    async def test_returns_distinct_model_names(self, repository, openai_entity):
        second = LlmCostEntity("openai", "gpt-4o-mini", 1.5e-7, 6e-7, "litellm")
        await repository.upsert_all([openai_entity, second])
        result = await repository.list_model_names_by_provider(provider="openai")
        assert result == ["gpt-4o", "gpt-4o-mini"]

    async def test_returns_sorted_results(self, repository):
        entities = [
            LlmCostEntity("openai", "gpt-4o-mini", 1e-7, 2e-7, "litellm"),
            LlmCostEntity("openai", "gpt-3.5-turbo", 5e-7, 1.5e-6, "litellm"),
            LlmCostEntity("openai", "gpt-4o", 2.5e-6, 10e-6, "litellm"),
        ]
        await repository.upsert_all(entities)
        result = await repository.list_model_names_by_provider(provider="openai")
        assert result == sorted(result)

    async def test_returns_empty_list_for_unknown_provider(self, repository):
        result = await repository.list_model_names_by_provider(provider="unknown")
        assert result == []

    async def test_excludes_other_providers(self, repository, openai_entity, anthropic_entity):
        await repository.upsert_all([openai_entity, anthropic_entity])
        result = await repository.list_model_names_by_provider(provider="openai")
        assert "claude-3-opus-20240229" not in result
        assert "gpt-4o" in result


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

    async def test_multiple_sources_for_same_model_coexist(self, repository):
        litellm_row = LlmCostEntity("openai", "gpt-4o", 2.5e-6, 10e-6, "litellm")
        openrouter_row = LlmCostEntity("openai", "gpt-4o", 2.5e-6, 10e-6, "openrouter")
        await repository.upsert_all([litellm_row, openrouter_row])
        all_rows = await repository.list_all(resolved=False)
        assert len(all_rows) == 2
        assert {r.source for r in all_rows} == {"litellm", "openrouter"}


class TestResolvedReads:
    async def test_get_without_source_returns_precedence_winner(self, repository):
        litellm_row = LlmCostEntity("openai", "gpt-4o", 1e-6, 2e-6, "litellm")
        openrouter_row = LlmCostEntity("openai", "gpt-4o", 9e-6, 9e-6, "openrouter")
        await repository.upsert_all([litellm_row, openrouter_row])
        result = await repository.get(provider="openai", model="gpt-4o")
        # litellm precedes openrouter in _SOURCE_PRECEDENCE
        assert result.source == "litellm"
        assert result.prompt_cost_per_token == 1e-6

    async def test_get_with_source_pins_to_source(self, repository):
        litellm_row = LlmCostEntity("openai", "gpt-4o", 1e-6, 2e-6, "litellm")
        openrouter_row = LlmCostEntity("openai", "gpt-4o", 9e-6, 9e-6, "openrouter")
        await repository.upsert_all([litellm_row, openrouter_row])
        result = await repository.get(provider="openai", model="gpt-4o", source="openrouter")
        assert result.source == "openrouter"
        assert result.prompt_cost_per_token == 9e-6

    async def test_get_with_unknown_source_returns_none(self, repository, openai_entity):
        await repository.upsert_all([openai_entity])
        result = await repository.get(provider="openai", model="gpt-4o", source="nonexistent")
        assert result is None

    async def test_list_all_resolved_collapses_duplicates(self, repository):
        await repository.upsert_all([
            LlmCostEntity("openai", "gpt-4o", 1e-6, 2e-6, "litellm"),
            LlmCostEntity("openai", "gpt-4o", 9e-6, 9e-6, "openrouter"),
            LlmCostEntity("openai", "gpt-4o-mini", 1e-7, 2e-7, "openrouter"),
        ])
        rows = await repository.list_all(resolved=True)
        assert len(rows) == 2
        gpt4o = next(r for r in rows if r.model == "gpt-4o")
        assert gpt4o.source == "litellm"

    async def test_list_all_unresolved_returns_all_source_rows(self, repository):
        await repository.upsert_all([
            LlmCostEntity("openai", "gpt-4o", 1e-6, 2e-6, "litellm"),
            LlmCostEntity("openai", "gpt-4o", 9e-6, 9e-6, "openrouter"),
        ])
        rows = await repository.list_all(resolved=False)
        assert len(rows) == 2

    async def test_list_by_provider_resolved_collapses_duplicates(self, repository):
        await repository.upsert_all([
            LlmCostEntity("openai", "gpt-4o", 1e-6, 2e-6, "litellm"),
            LlmCostEntity("openai", "gpt-4o", 9e-6, 9e-6, "openrouter"),
        ])
        rows = await repository.list_by_provider(provider="openai", resolved=True)
        assert len(rows) == 1
        assert rows[0].source == "litellm"

    async def test_list_all_filtered_by_source(self, repository):
        await repository.upsert_all([
            LlmCostEntity("openai", "gpt-4o", 1e-6, 2e-6, "litellm"),
            LlmCostEntity("openai", "gpt-4o-mini", 1e-7, 2e-7, "openrouter"),
        ])
        rows = await repository.list_all(source="openrouter")
        assert len(rows) == 1
        assert rows[0].source == "openrouter"

    async def test_get_status_total_models_counts_distinct_pairs(self, repository):
        await repository.upsert_all([
            LlmCostEntity("openai", "gpt-4o", 1e-6, 2e-6, "litellm"),
            LlmCostEntity("openai", "gpt-4o", 9e-6, 9e-6, "openrouter"),
            LlmCostEntity("openai", "gpt-4o-mini", 1e-7, 2e-7, "litellm"),
        ])
        status = await repository.get_status()
        assert status["total_models"] == 2
        assert status["total_providers"] == 1
