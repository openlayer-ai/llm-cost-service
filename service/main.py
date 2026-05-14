import asyncio
import logging
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI

from service.api.costs import router as costs_router
from service.api.health import router as health_router
from service.api.pages import router as pages_router
from service.database import create_engine, create_session_factory

logger = logging.getLogger(__name__)


async def _run_refresh(session_factory) -> None:
    from service.costs.providers import LiteLLMCostProvider
    from service.costs.repositories import LlmCostRepository
    from service.costs.services import RefreshLlmCostsService

    async with session_factory() as session:
        repository = LlmCostRepository(session)
        response = await RefreshLlmCostsService(LiteLLMCostProvider(), repository).execute()
    logger.info(
        "Scheduled refresh complete: %d rows affected in %.0fms",
        response.rows_affected,
        response.duration_ms,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    from service.config import get_settings

    settings = get_settings()

    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    app.state.db_session_factory = session_factory

    # Migrations and scheduler only run in Docker (ENABLE_SCHEDULER=true).
    # On Vercel (ENABLE_SCHEDULER=false), run migrations manually:
    #   DATABASE_URL=<neon-url> uv run alembic upgrade head
    scheduler = None
    if settings.enable_scheduler:
        from alembic import command
        from alembic.config import Config as AlembicConfig

        alembic_cfg = AlembicConfig("alembic.ini")
        await asyncio.to_thread(command.upgrade, alembic_cfg, "head")

        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            _run_refresh,
            "interval",
            hours=settings.scheduler_interval_hours,
            args=[session_factory],
        )
        scheduler.start()
        logger.info("Scheduler started: refresh every %dh", settings.scheduler_interval_hours)

    yield

    if scheduler is not None:
        scheduler.shutdown()
    await engine.dispose()


app = FastAPI(title="LLM Cost Service", lifespan=lifespan)
app.include_router(pages_router)
app.include_router(costs_router)
app.include_router(health_router)
