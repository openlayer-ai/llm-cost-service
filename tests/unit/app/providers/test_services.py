"""Unit tests for providers.services."""

from __future__ import annotations

from typing import cast

from service.costs.repositories import LlmCostRepository
from service.providers.services import ListProviderModelsService


class _RepositoryStub:
    def __init__(self, *, model_names: list[str] | None = None):
        self._model_names = model_names or []
        self.list_model_names_calls: list[str] = []

    async def list_model_names_by_provider(self, *, provider):
        self.list_model_names_calls.append(provider)
        return self._model_names


class TestListProviderModelsService:
    async def test_returns_model_names_for_known_provider(self):
        repo = _RepositoryStub(model_names=["gpt-4o", "gpt-4o-mini"])
        response = await ListProviderModelsService(cast(LlmCostRepository, repo)).execute(
            provider="openai"
        )
        assert response.provider == "openai"
        assert response.model_names == ["gpt-4o", "gpt-4o-mini"]

    async def test_returns_empty_response_when_no_models_found(self):
        repo = _RepositoryStub(model_names=[])
        response = await ListProviderModelsService(cast(LlmCostRepository, repo)).execute(
            provider="unknown"
        )
        assert response.provider == "unknown"
        assert response.model_names == []

    async def test_normalizes_provider_to_lowercase(self):
        repo = _RepositoryStub(model_names=["gpt-4o"])
        await ListProviderModelsService(cast(LlmCostRepository, repo)).execute(
            provider="OpenAI"
        )
        assert repo.list_model_names_calls[0] == "openai"

    async def test_strips_whitespace_from_provider(self):
        repo = _RepositoryStub(model_names=["gpt-4o"])
        await ListProviderModelsService(cast(LlmCostRepository, repo)).execute(
            provider="  openai  "
        )
        assert repo.list_model_names_calls[0] == "openai"
