"""Unit tests for costs.services."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import cast

from service.costs.entities import LlmCostEntity
from service.costs.repositories import LlmCostRepository
from service.costs.services import GetLlmCostService, ListLlmCostsService, RefreshLlmCostsService

_NOW = datetime.now(timezone.utc)


def _entity(**kwargs) -> LlmCostEntity:
    defaults = dict(
        provider="openai",
        model="gpt-4o",
        prompt_cost_per_token=2.5e-6,
        completion_cost_per_token=10e-6,
        source="litellm",
        updated_at=_NOW,
    )
    defaults.update(kwargs)
    return LlmCostEntity(**defaults)


class _RepositoryStub:
    def __init__(self, *, entity=None, all_costs=None, provider_costs=None):
        self._entity = entity
        self._all = all_costs or []
        self._by_provider = provider_costs or []
        self.get_calls: list[dict] = []
        self.list_all_calls = 0
        self.list_by_provider_calls: list[str] = []
        self.upserted: list[LlmCostEntity] = []
        self.pruned: list[tuple[str, set]] = []
        # Pretend this many stale rows exist per source, for delete_missing.
        self.stale_per_source: dict[str, int] = {}

    async def get(self, *, provider, model, source=None):
        self.get_calls.append({"provider": provider, "model": model, "source": source})
        return self._entity

    async def list_all(self, *, source=None, resolved=True):
        self.list_all_calls += 1
        return self._all

    async def list_by_provider(self, *, provider, source=None, resolved=True):
        self.list_by_provider_calls.append(provider)
        return self._by_provider

    async def upsert_all(self, entities):
        self.upserted = entities
        return len(entities)

    async def delete_missing(self, source, keep):
        self.pruned.append((source, set(keep)))
        return self.stale_per_source.get(source, 0)


class _ProviderStub:
    def __init__(self, costs: list[LlmCostEntity]):
        self._costs = costs

    def fetch_costs(self) -> list[LlmCostEntity]:
        return self._costs


class _FailingProvider:
    def fetch_costs(self) -> list[LlmCostEntity]:
        raise RuntimeError("boom")


class TestRefreshLlmCostsService:
    async def test_upserts_costs_fetched_from_provider(self):
        costs = [_entity(), _entity(provider="anthropic", model="claude-3-opus-20240229")]
        repo = _RepositoryStub()
        await RefreshLlmCostsService([_ProviderStub(costs)], cast(LlmCostRepository, repo)).execute()
        assert len(repo.upserted) == 2

    async def test_returns_correct_row_count(self):
        costs = [_entity(), _entity(provider="anthropic", model="claude-3-opus-20240229")]
        repo = _RepositoryStub()
        response = await RefreshLlmCostsService([_ProviderStub(costs)], cast(LlmCostRepository, repo)).execute()
        assert response.rows_affected == 2

    async def test_returns_non_negative_duration_ms(self):
        repo = _RepositoryStub()
        response = await RefreshLlmCostsService([_ProviderStub([])], cast(LlmCostRepository, repo)).execute()
        assert response.duration_ms >= 0

    async def test_runs_multiple_providers(self):
        litellm_costs = [_entity(source="litellm")]
        openrouter_costs = [_entity(provider="anthropic", model="claude-opus-4.7", source="openrouter")]
        repo = _RepositoryStub()
        response = await RefreshLlmCostsService(
            [_ProviderStub(litellm_costs), _ProviderStub(openrouter_costs)],
            cast(LlmCostRepository, repo),
        ).execute()
        assert response.rows_affected == 2
        assert response.per_source == {"litellm": 1, "openrouter": 1}
        # Both sources' entities reach the repo in a single upsert batch.
        assert len(repo.upserted) == 2
        assert {e.source for e in repo.upserted} == {"litellm", "openrouter"}

    async def test_one_provider_failing_does_not_abort_others(self):
        costs = [_entity(source="litellm")]
        repo = _RepositoryStub()
        response = await RefreshLlmCostsService(
            [_ProviderStub(costs), _FailingProvider()],
            cast(LlmCostRepository, repo),
        ).execute()
        assert response.rows_affected == 1
        assert response.per_source == {"litellm": 1}
        assert response.failed_sources == ["_FailingProvider"]

    async def test_rejects_empty_provider_list(self):
        import pytest
        repo = _RepositoryStub()
        with pytest.raises(ValueError):
            RefreshLlmCostsService([], cast(LlmCostRepository, repo))

    async def test_prunes_stale_rows_per_successful_source(self):
        litellm_costs = [_entity(source="litellm"), _entity(model="gpt-4o-mini", source="litellm")]
        openrouter_costs = [_entity(provider="anthropic", model="claude-opus-4.7", source="openrouter")]
        repo = _RepositoryStub()
        repo.stale_per_source = {"litellm": 3, "openrouter": 1}
        response = await RefreshLlmCostsService(
            [_ProviderStub(litellm_costs), _ProviderStub(openrouter_costs)],
            cast(LlmCostRepository, repo),
        ).execute()
        assert response.rows_deleted == 4
        assert dict(repo.pruned) == {
            "litellm": {("openai", "gpt-4o"), ("openai", "gpt-4o-mini")},
            "openrouter": {("anthropic", "claude-opus-4.7")},
        }

    async def test_does_not_prune_failed_source(self):
        # An outage on one source must not wipe its existing rows.
        repo = _RepositoryStub()
        repo.stale_per_source = {"litellm": 2, "openrouter": 99}
        response = await RefreshLlmCostsService(
            [_ProviderStub([_entity(source="litellm")]), _FailingProvider()],
            cast(LlmCostRepository, repo),
        ).execute()
        assert [s for s, _ in repo.pruned] == ["litellm"]
        assert response.rows_deleted == 2

    async def test_does_not_prune_source_that_returned_nothing(self):
        repo = _RepositoryStub()
        repo.stale_per_source = {"litellm": 99}
        response = await RefreshLlmCostsService(
            [_ProviderStub([])], cast(LlmCostRepository, repo)
        ).execute()
        assert repo.pruned == []
        assert response.rows_deleted == 0


class TestGetLlmCostService:
    async def test_returns_entity_when_found(self):
        entity = _entity()
        repo = _RepositoryStub(entity=entity)
        response = await GetLlmCostService(cast(LlmCostRepository, repo)).execute(provider="openai", model="gpt-4o")
        assert response.cost == entity

    async def test_returns_response_with_none_when_not_found(self):
        repo = _RepositoryStub(entity=None)
        response = await GetLlmCostService(cast(LlmCostRepository, repo)).execute(provider="openai", model="missing")
        assert response.cost is None

    async def test_normalizes_provider_to_lowercase(self):
        repo = _RepositoryStub(entity=_entity())
        await GetLlmCostService(cast(LlmCostRepository, repo)).execute(provider="OpenAI", model="gpt-4o")
        assert repo.get_calls[0]["provider"] == "openai"

    async def test_strips_whitespace_from_provider(self):
        repo = _RepositoryStub(entity=_entity())
        await GetLlmCostService(cast(LlmCostRepository, repo)).execute(provider="  openai  ", model="gpt-4o")
        assert repo.get_calls[0]["provider"] == "openai"


class TestListLlmCostsService:
    async def test_returns_all_costs_when_no_provider_given(self):
        costs = [_entity(), _entity(provider="anthropic", model="claude-3-opus-20240229")]
        repo = _RepositoryStub(all_costs=costs)
        response = await ListLlmCostsService(cast(LlmCostRepository, repo)).execute()
        assert len(response.costs) == 2
        assert repo.list_all_calls == 1

    async def test_returns_filtered_costs_when_provider_given(self):
        entity = _entity()
        repo = _RepositoryStub(provider_costs=[entity])
        response = await ListLlmCostsService(cast(LlmCostRepository, repo)).execute(provider="openai")
        assert len(response.costs) == 1
        assert repo.list_by_provider_calls == ["openai"]

    async def test_normalizes_provider_to_lowercase(self):
        repo = _RepositoryStub(provider_costs=[])
        await ListLlmCostsService(cast(LlmCostRepository, repo)).execute(provider="OpenAI")
        assert repo.list_by_provider_calls[0] == "openai"

    async def test_returns_empty_list_when_no_costs_exist(self):
        repo = _RepositoryStub()
        response = await ListLlmCostsService(cast(LlmCostRepository, repo)).execute()
        assert response.costs == []
