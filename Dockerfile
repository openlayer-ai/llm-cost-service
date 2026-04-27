FROM python:3.11-slim AS builder

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml .
RUN uv sync --no-dev --no-editable

FROM python:3.11-slim

RUN useradd --create-home appuser
WORKDIR /app

COPY --from=builder /app/.venv .venv
COPY alembic.ini .
COPY alembic/ alembic/
COPY app/ app/

ENV PATH="/app/.venv/bin:$PATH"
USER appuser

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
