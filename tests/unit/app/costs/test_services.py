"""Unit tests for costs.services."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import cast

import pytest
from fastapi import HTTPException

from app.costs.entities import LlmCostEntity
from app.costs.repositories import LlmCostRepository
from app.costs.services import GetLlmCostService, ListLlmCostsService, RefreshLlmCostsService

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

    async def get(self, *, provider, model):
        self.get_calls.append({"provider": provider, "model": model})
        return self._entity

    async def list_all(self):
        self.list_all_calls += 1
        return self._all

    async def list_by_provider(self, *, provider):
        self.list_by_provider_calls.append(provider)
        return self._by_provider

    async def upsert_all(self, entities):
        self.upserted = entities
        return len(entities)


class _ProviderStub:
    def __init__(self, costs: list[LlmCostEntity]):
        self._costs = costs

    def fetch_costs(self) -> list[LlmCostEntity]:
        return self._costs


class TestRefreshLlmCostsService:
    async def test_upserts_costs_fetched_from_provider(self):
        costs = [_entity(), _entity(provider="anthropic", model="claude-3-opus-20240229")]
        repo = _RepositoryStub()
        await RefreshLlmCostsService(_ProviderStub(costs), cast(LlmCostRepository, repo)).execute()
        assert len(repo.upserted) == 2

    async def test_returns_correct_row_count(self):
        costs = [_entity(), _entity(provider="anthropic", model="claude-3-opus-20240229")]
        repo = _RepositoryStub()
        response = await RefreshLlmCostsService(_ProviderStub(costs), cast(LlmCostRepository, repo)).execute()
        assert response.rows_affected == 2

    async def test_returns_non_negative_duration_ms(self):
        repo = _RepositoryStub()
        response = await RefreshLlmCostsService(_ProviderStub([]), cast(LlmCostRepository, repo)).execute()
        assert response.duration_ms >= 0


class TestGetLlmCostService:
    async def test_returns_entity_when_found(self):
        entity = _entity()
        repo = _RepositoryStub(entity=entity)
        response = await GetLlmCostService(cast(LlmCostRepository, repo)).execute(provider="openai", model="gpt-4o")
        assert response.cost == entity

    async def test_raises_404_when_not_found(self):
        repo = _RepositoryStub(entity=None)
        with pytest.raises(HTTPException) as exc_info:
            await GetLlmCostService(cast(LlmCostRepository, repo)).execute(provider="openai", model="missing")
        assert exc_info.value.status_code == 404

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
