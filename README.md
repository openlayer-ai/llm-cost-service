# llm-cost-service

Standalone service that fetches LLM cost data from LiteLLM daily and exposes it over a simple public REST API.

Consumers (Openlayer platform, gateway) fetch the full cost table on startup and keep it in memory, refreshing periodically. This centralizes cost updates so any number of consumers can stay in sync without each owning their own cron job or cost table.

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/costs` | List all costs. Optional `?provider=` filter. |
| `GET` | `/v1/costs/{provider}/{model}` | Get cost for a specific model. `model` may contain slashes (e.g. `gemini/gemini-1.5-pro`). Returns 404 if not found. |
| `POST` | `/v1/costs/refresh` | Fetch latest costs from LiteLLM and upsert into the DB. |
| `GET` | `/health` | Liveness check. |

Example response for `GET /v1/costs`:

```json
{
  "costs": [
    {
      "provider": "openai",
      "model": "gpt-4o",
      "prompt_cost_per_token": 2.5e-6,
      "completion_cost_per_token": 1e-5,
      "source": "litellm",
      "updated_at": "2026-04-27T00:00:00Z"
    }
  ]
}
```

## Running locally

```bash
cp .env.example .env
# edit .env and set DATABASE_URL

docker compose up
```

The app will run Alembic migrations on startup, then serve on `http://localhost:8080`.

To trigger a manual refresh:

```bash
curl -X POST http://localhost:8080/v1/costs/refresh
```

## Running tests

```bash
uv sync
uv run pytest
```

Tests use an in-memory SQLite database — no Postgres required.

## Deployment

### Docker

Set `DATABASE_URL` to a Postgres connection string (e.g. `postgresql+asyncpg://...`). The scheduler runs in-process and calls `/v1/costs/refresh` every 24 hours.

### Vercel + Neon

1. Set `DATABASE_URL` to your Neon connection string.
2. Set `ENABLE_SCHEDULER=false` — Vercel Cron handles the daily refresh via `vercel.json`.
3. Deploy: `vercel deploy`.

## Architecture

```
APScheduler / Vercel Cron (daily)
        ↓
LiteLLMCostProvider          ← plug in more providers here later
        ↓
LlmCostRepository (Postgres)
        ↓
FastAPI REST API
```

The `CostDataProvider` protocol makes it straightforward to add new cost sources (e.g. scraping provider pricing pages) without touching the rest of the stack.
