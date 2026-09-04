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

    async def test_dedupes_duplicate_pk_within_batch(self, repository):
        # Two entities sharing (provider, model, source) must not crash the
        # single ON CONFLICT insert; last writer wins.
        first = LlmCostEntity("azure", "computer-use-preview", 1e-6, 1e-6, "litellm")
        second = LlmCostEntity("azure", "computer-use-preview", 3e-6, 12e-6, "litellm")
        count = await repository.upsert_all([first, second])
        assert count == 1
        result = await repository.get(provider="azure", model="computer-use-preview")
        assert result.prompt_cost_per_token == 3e-6


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


class TestDeleteMissing:
    async def test_deletes_rows_not_in_keep_set(self, repository, openai_entity, anthropic_entity):
        await repository.upsert_all([openai_entity, anthropic_entity])
        deleted = await repository.delete_missing("litellm", {("openai", "gpt-4o")})
        assert deleted == 1
        remaining = {(e.provider, e.model) for e in await repository.list_all(resolved=False)}
        assert remaining == {("openai", "gpt-4o")}

    async def test_only_touches_given_source(self, repository, openai_entity):
        other = LlmCostEntity("openai", "gpt-4o", 2e-6, 8e-6, "openrouter")
        await repository.upsert_all([openai_entity, other])
        # Keep nothing from litellm: its row goes, the openrouter twin stays.
        deleted = await repository.delete_missing("litellm", set())
        assert deleted == 1
        remaining = await repository.list_all(resolved=False)
        assert [(e.provider, e.model, e.source) for e in remaining] == [("openai", "gpt-4o", "openrouter")]

    async def test_returns_zero_when_nothing_is_stale(self, repository, openai_entity):
        await repository.upsert_all([openai_entity])
        assert await repository.delete_missing("litellm", {("openai", "gpt-4o")}) == 0

    async def test_returns_zero_on_empty_table(self, repository):
        assert await repository.delete_missing("litellm", {("openai", "gpt-4o")}) == 0

    async def test_clears_legacy_prefixed_duplicates(self, repository):
        # The situation from the audit: a stale "wandb/<model>" row alongside
        # the current stripped name. A refresh that emits only the stripped
        # name prunes the prefixed leftover.
        stale = LlmCostEntity("wandb", "wandb/deepseek-ai/DeepSeek-R1-0528", 0.135, 0.54, "litellm")
        fresh = LlmCostEntity("wandb", "deepseek-ai/DeepSeek-R1-0528", 1.35e-6, 5.4e-6, "litellm")
        await repository.upsert_all([stale, fresh])
        deleted = await repository.delete_missing("litellm", {("wandb", "deepseek-ai/DeepSeek-R1-0528")})
        assert deleted == 1
        rows = await repository.list_by_provider("wandb", resolved=False)
        assert [e.model for e in rows] == ["deepseek-ai/DeepSeek-R1-0528"]

    async def test_chunks_large_deletes(self, repository, monkeypatch):
        monkeypatch.setattr(LlmCostRepository, "_DELETE_CHUNK", 3)
        rows = [LlmCostEntity("p", f"m{i}", 1e-6, 1e-6, "litellm") for i in range(10)]
        await repository.upsert_all(rows)
        deleted = await repository.delete_missing("litellm", {("p", "m0")})
        assert deleted == 9
        assert [e.model for e in await repository.list_all(resolved=False)] == ["m0"]
