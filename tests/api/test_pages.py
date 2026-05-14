"""API tests for the dashboard page."""

from __future__ import annotations

from service.costs.repositories import LlmCostRepository


class TestIndexPage:
    async def test_returns_html(self, client):
        resp = await client.get("/")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")

    async def test_sets_cache_control(self, client):
        resp = await client.get("/")
        assert "max-age" in resp.headers.get("cache-control", "")

    async def test_renders_rows_for_each_cost(
        self, client, session, openai_entity, anthropic_entity
    ):
        await LlmCostRepository(session).upsert_all([openai_entity, anthropic_entity])
        resp = await client.get("/")
        body = resp.text
        assert "gpt-4o" in body
        assert "claude-3-opus-20240229" in body
        assert "openai" in body
        assert "anthropic" in body

    async def test_renders_status_counts(
        self, client, session, openai_entity, anthropic_entity
    ):
        await LlmCostRepository(session).upsert_all([openai_entity, anthropic_entity])
        resp = await client.get("/")
        body = resp.text
        # 2 models — appears as the value in the Total models card.
        assert ">2<" in body

    async def test_renders_provider_filter_options(
        self, client, session, openai_entity, anthropic_entity
    ):
        await LlmCostRepository(session).upsert_all([openai_entity, anthropic_entity])
        resp = await client.get("/")
        body = resp.text
        assert '<option value="openai">openai</option>' in body
        assert '<option value="anthropic">anthropic</option>' in body

    async def test_handles_empty_database(self, client):
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "Openlayer" in resp.text
