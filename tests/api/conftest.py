"""API test fixtures: wired FastAPI app with in-memory DB."""

from __future__ import annotations

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.deps import get_db_session
from app.main import app


@pytest_asyncio.fixture
async def client(engine):
    """AsyncClient wired to the real app with an in-memory DB.

    Uses dependency_overrides to inject the in-memory session without
    touching the lifespan (no Alembic, no scheduler).
    """
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _override_db():
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = _override_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()
