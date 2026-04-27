"""API tests for /v1/costs endpoints."""

from __future__ import annotations

from unittest.mock import patch

from app.costs.entities import LlmCostEntity
from app.costs.repositories import LlmCostRepository


class TestListCosts:
    async def test_returns_all_costs(self, client, session, openai_entity, anthropic_entity):
        await LlmCostRepository(session).upsert_all([openai_entity, anthropic_entity])
        resp = await client.get("/v1/costs")
        assert resp.status_code == 200
        assert len(resp.json()["costs"]) == 2

    async def test_filters_by_provider(self, client, session, openai_entity, anthropic_entity):
        await LlmCostRepository(session).upsert_all([openai_entity, anthropic_entity])
        resp = await client.get("/v1/costs?provider=openai")
        assert resp.status_code == 200
        costs = resp.json()["costs"]
        assert all(c["provider"] == "openai" for c in costs)

    async def test_returns_empty_list_when_no_costs(self, client):
        resp = await client.get("/v1/costs")
        assert resp.status_code == 200
        assert resp.json()["costs"] == []

    async def test_response_includes_expected_fields(self, client, session, openai_entity):
        await LlmCostRepository(session).upsert_all([openai_entity])
        resp = await client.get("/v1/costs")
        cost = resp.json()["costs"][0]
        assert "provider" in cost
        assert "model" in cost
        assert "prompt_cost_per_token" in cost
        assert "completion_cost_per_token" in cost
        assert "source" in cost
        assert "updated_at" in cost


class TestGetCost:
    async def test_returns_cost_for_known_model(self, client, session, openai_entity):
        await LlmCostRepository(session).upsert_all([openai_entity])
        resp = await client.get("/v1/costs/openai/gpt-4o")
        assert resp.status_code == 200
        assert resp.json()["provider"] == "openai"
        assert resp.json()["model"] == "gpt-4o"

    async def test_returns_404_for_unknown_model(self, client):
        resp = await client.get("/v1/costs/openai/nonexistent-model")
        assert resp.status_code == 404

    async def test_handles_slash_in_model_name(self, client, session):
        entity = LlmCostEntity(
            provider="gemini",
            model="gemini/gemini-1.5-pro",
            prompt_cost_per_token=1e-6,
            completion_cost_per_token=2e-6,
            source="litellm",
        )
        await LlmCostRepository(session).upsert_all([entity])
        resp = await client.get("/v1/costs/gemini/gemini/gemini-1.5-pro")
        assert resp.status_code == 200
        assert resp.json()["model"] == "gemini/gemini-1.5-pro"

    async def test_provider_lookup_is_case_insensitive(self, client, session, openai_entity):
        await LlmCostRepository(session).upsert_all([openai_entity])
        resp = await client.get("/v1/costs/OpenAI/gpt-4o")
        assert resp.status_code == 200


class TestRefreshCosts:
    async def test_returns_rows_affected(self, client):
        fake_costs = [
            LlmCostEntity("openai", "gpt-4o", 2.5e-6, 10e-6, "litellm"),
            LlmCostEntity("anthropic", "claude-3-opus-20240229", 15e-6, 75e-6, "litellm"),
        ]
        with patch("app.api.costs.LiteLLMCostProvider") as MockProvider:
            MockProvider.return_value.fetch_costs.return_value = fake_costs
            resp = await client.post("/v1/costs/refresh")
        assert resp.status_code == 200
        assert resp.json()["rows_affected"] == 2

    async def test_returns_duration_ms(self, client):
        with patch("app.api.costs.LiteLLMCostProvider") as MockProvider:
            MockProvider.return_value.fetch_costs.return_value = []
            resp = await client.post("/v1/costs/refresh")
        assert resp.json()["duration_ms"] >= 0

    async def test_costs_are_queryable_after_refresh(self, client):
        fake_costs = [LlmCostEntity("openai", "gpt-4o", 2.5e-6, 10e-6, "litellm")]
        with patch("app.api.costs.LiteLLMCostProvider") as MockProvider:
            MockProvider.return_value.fetch_costs.return_value = fake_costs
            await client.post("/v1/costs/refresh")
        resp = await client.get("/v1/costs/openai/gpt-4o")
        assert resp.status_code == 200

    async def test_returns_401_when_cron_secret_set_and_header_missing(self, client, monkeypatch):
        monkeypatch.setenv("CRON_SECRET", "test-secret")
        from app import config as config_mod
        config_mod.get_settings.cache_clear()
        resp = await client.post("/v1/costs/refresh")
        assert resp.status_code == 401
        config_mod.get_settings.cache_clear()

    async def test_returns_401_when_cron_secret_set_and_header_wrong(self, client, monkeypatch):
        monkeypatch.setenv("CRON_SECRET", "test-secret")
        from app import config as config_mod
        config_mod.get_settings.cache_clear()
        resp = await client.post("/v1/costs/refresh", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401
        config_mod.get_settings.cache_clear()

    async def test_returns_200_when_cron_secret_set_and_header_correct(self, client, monkeypatch):
        monkeypatch.setenv("CRON_SECRET", "test-secret")
        from app import config as config_mod
        config_mod.get_settings.cache_clear()
        with patch("app.api.costs.LiteLLMCostProvider") as MockProvider:
            MockProvider.return_value.fetch_costs.return_value = []
            resp = await client.post("/v1/costs/refresh", headers={"Authorization": "Bearer test-secret"})
        assert resp.status_code == 200
        config_mod.get_settings.cache_clear()


class TestGetStatus:
    async def test_returns_zero_counts_when_table_empty(self, client):
        resp = await client.get("/v1/costs/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_models"] == 0
        assert data["total_providers"] == 0
        assert data["last_refreshed_at"] is None

    async def test_returns_counts_after_upsert(self, client, session, openai_entity, anthropic_entity):
        from app.costs.repositories import LlmCostRepository
        await LlmCostRepository(session).upsert_all([openai_entity, anthropic_entity])
        resp = await client.get("/v1/costs/status")
        data = resp.json()
        assert data["total_models"] == 2
        assert data["total_providers"] == 2
        assert data["last_refreshed_at"] is not None


class TestHealth:
    async def test_returns_ok(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
