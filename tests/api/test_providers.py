"""API tests for /v1/providers endpoints."""

from __future__ import annotations

from service.costs.entities import LlmCostEntity
from service.costs.repositories import LlmCostRepository


class TestListProviderModels:
    async def test_returns_sorted_distinct_model_names(self, client, session, openai_entity):
        second = LlmCostEntity("openai", "gpt-4o-mini", 1.5e-7, 6e-7, "litellm")
        await LlmCostRepository(session).upsert_all([openai_entity, second])
        resp = await client.get("/v1/providers/openai/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["provider"] == "openai"
        assert [m["name"] for m in body["models"]] == ["gpt-4o", "gpt-4o-mini"]

    async def test_provider_lookup_is_case_insensitive(self, client, session, openai_entity):
        await LlmCostRepository(session).upsert_all([openai_entity])
        resp = await client.get("/v1/providers/OpenAI/models")
        assert resp.status_code == 200
        assert resp.json()["provider"] == "openai"

    async def test_returns_404_for_unknown_provider(self, client):
        resp = await client.get("/v1/providers/bogus/models")
        assert resp.status_code == 404

    async def test_excludes_other_providers(self, client, session, openai_entity, anthropic_entity):
        await LlmCostRepository(session).upsert_all([openai_entity, anthropic_entity])
        resp = await client.get("/v1/providers/openai/models")
        names = [m["name"] for m in resp.json()["models"]]
        assert "claude-3-opus-20240229" not in names

    async def test_sets_cache_control_header(self, client, session, openai_entity):
        await LlmCostRepository(session).upsert_all([openai_entity])
        resp = await client.get("/v1/providers/openai/models")
        assert resp.headers.get("cache-control") == "public, max-age=3600"
