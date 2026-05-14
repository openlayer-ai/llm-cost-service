"""Shared fixtures: in-memory SQLite engine and session for all test layers."""

from __future__ import annotations

import os
from datetime import datetime, timezone

# Must be set before service.config is imported anywhere in this process.
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from service.costs.entities import LlmCostEntity
from service.costs.models import Base

_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def engine():
    engine = create_async_engine(_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def session(engine):
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as s:
        yield s


@pytest.fixture
def now() -> datetime:
    return datetime.now(timezone.utc)


@pytest.fixture
def openai_entity(now: datetime) -> LlmCostEntity:
    return LlmCostEntity(
        provider="openai",
        model="gpt-4o",
        prompt_cost_per_token=2.5e-6,
        completion_cost_per_token=10e-6,
        source="litellm",
        updated_at=now,
    )


@pytest.fixture
def anthropic_entity(now: datetime) -> LlmCostEntity:
    return LlmCostEntity(
        provider="anthropic",
        model="claude-3-opus-20240229",
        prompt_cost_per_token=15e-6,
        completion_cost_per_token=75e-6,
        source="litellm",
        updated_at=now,
    )
