from collections.abc import AsyncGenerator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession


async def get_db_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    # Use session factory set by the lifespan when available (Docker).
    # On Vercel the lifespan may not run, so we initialize lazily on first request.
    factory = getattr(request.app.state, "db_session_factory", None)
    if factory is None:
        from service.config import get_settings
        from service.database import create_engine, create_session_factory

        engine = create_engine(get_settings().database_url)
        factory = create_session_factory(engine)
        request.app.state.db_session_factory = factory

    async with factory() as session:
        yield session
