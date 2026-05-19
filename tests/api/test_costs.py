"""API tests for /v1/costs endpoints."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

from service.costs.entities import LlmCostEntity
from service.costs.repositories import LlmCostRepository


@contextmanager
def _patched_providers(litellm_costs=None, openrouter_costs=None):
    """Patch both refresh providers so tests don't hit the network."""
    litellm_costs = litellm_costs or []
    openrouter_costs = openrouter_costs or []
    with patch("service.api.costs.LiteLLMCostProvider") as LiteMock, \
         patch("service.api.costs.OpenRouterCostProvider") as ORMock:
        LiteMock.return_value.fetch_costs.return_value = litellm_costs
        ORMock.return_value.fetch_costs.return_value = openrouter_costs
        yield


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
        with _patched_providers(litellm_costs=fake_costs):
            resp = await client.get("/v1/costs/refresh")
        assert resp.status_code == 200
        assert resp.json()["rows_affected"] == 2

    async def test_returns_duration_ms(self, client):
        with _patched_providers():
            resp = await client.get("/v1/costs/refresh")
        assert resp.json()["duration_ms"] >= 0

    async def test_returns_per_source_counts(self, client):
        litellm = [LlmCostEntity("openai", "gpt-4o", 2.5e-6, 10e-6, "litellm")]
        openrouter = [
            LlmCostEntity("anthropic", "claude-opus-4.7", 5e-6, 25e-6, "openrouter"),
            LlmCostEntity("anthropic", "claude-opus-4.7-fast", 30e-6, 150e-6, "openrouter"),
        ]
        with _patched_providers(litellm_costs=litellm, openrouter_costs=openrouter):
            resp = await client.get("/v1/costs/refresh")
        body = resp.json()
        assert body["per_source"] == {"litellm": 1, "openrouter": 2}
        assert body["failed_sources"] == []

    async def test_costs_are_queryable_after_refresh(self, client):
        fake_costs = [LlmCostEntity("openai", "gpt-4o", 2.5e-6, 10e-6, "litellm")]
        with _patched_providers(litellm_costs=fake_costs):
            await client.get("/v1/costs/refresh")
        resp = await client.get("/v1/costs/openai/gpt-4o")
        assert resp.status_code == 200

    async def test_returns_401_when_cron_secret_set_and_header_missing(self, client, monkeypatch):
        monkeypatch.setenv("CRON_SECRET", "test-secret")
        from service import config as config_mod
        config_mod.get_settings.cache_clear()
        resp = await client.get("/v1/costs/refresh")
        assert resp.status_code == 401
        config_mod.get_settings.cache_clear()

    async def test_returns_401_when_cron_secret_set_and_header_wrong(self, client, monkeypatch):
        monkeypatch.setenv("CRON_SECRET", "test-secret")
        from service import config as config_mod
        config_mod.get_settings.cache_clear()
        resp = await client.get("/v1/costs/refresh", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401
        config_mod.get_settings.cache_clear()

    async def test_returns_200_when_cron_secret_set_and_header_correct(self, client, monkeypatch):
        monkeypatch.setenv("CRON_SECRET", "test-secret")
        from service import config as config_mod
        config_mod.get_settings.cache_clear()
        with _patched_providers():
            resp = await client.get("/v1/costs/refresh", headers={"Authorization": "Bearer test-secret"})
        assert resp.status_code == 200
        config_mod.get_settings.cache_clear()


class TestMultiSourceQueries:
    async def _seed_dual_source(self, session):
        await LlmCostRepository(session).upsert_all([
            LlmCostEntity("openai", "gpt-4o", 1e-6, 2e-6, "litellm"),
            LlmCostEntity("openai", "gpt-4o", 9e-6, 9e-6, "openrouter"),
            LlmCostEntity("anthropic", "claude-opus-4.7", 5e-6, 25e-6, "openrouter"),
        ])

    async def test_list_resolved_collapses_duplicate_models(self, client, session):
        await self._seed_dual_source(session)
        resp = await client.get("/v1/costs?provider=openai")
        costs = resp.json()["costs"]
        assert len(costs) == 1
        assert costs[0]["source"] == "litellm"

    async def test_list_unresolved_returns_all_source_rows(self, client, session):
        await self._seed_dual_source(session)
        resp = await client.get("/v1/costs?provider=openai&resolved=false")
        costs = resp.json()["costs"]
        assert len(costs) == 2
        assert {c["source"] for c in costs} == {"litellm", "openrouter"}

    async def test_list_filtered_by_source(self, client, session):
        await self._seed_dual_source(session)
        resp = await client.get("/v1/costs?source=openrouter")
        costs = resp.json()["costs"]
        assert all(c["source"] == "openrouter" for c in costs)
        models = {c["model"] for c in costs}
        assert "gpt-4o" in models and "claude-opus-4.7" in models

    async def test_get_without_source_returns_precedence_winner(self, client, session):
        await self._seed_dual_source(session)
        resp = await client.get("/v1/costs/openai/gpt-4o")
        assert resp.status_code == 200
        assert resp.json()["source"] == "litellm"

    async def test_get_with_source_pins_to_source(self, client, session):
        await self._seed_dual_source(session)
        resp = await client.get("/v1/costs/openai/gpt-4o?source=openrouter")
        assert resp.status_code == 200
        assert resp.json()["source"] == "openrouter"

    async def test_get_with_unknown_source_returns_404(self, client, session):
        await self._seed_dual_source(session)
        resp = await client.get("/v1/costs/openai/gpt-4o?source=nope")
        assert resp.status_code == 404


class TestGetStatus:
    async def test_returns_zero_counts_when_table_empty(self, client):
        resp = await client.get("/v1/costs/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_models"] == 0
        assert data["total_providers"] == 0
        assert data["last_refreshed_at"] is None

    async def test_returns_counts_after_upsert(self, client, session, openai_entity, anthropic_entity):
        from service.costs.repositories import LlmCostRepository
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
