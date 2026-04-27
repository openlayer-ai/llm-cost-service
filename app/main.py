import asyncio
import logging
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI

from alembic import command
from alembic.config import Config as AlembicConfig
from app.api.costs import router as costs_router
from app.api.health import router as health_router
from app.database import create_engine, create_session_factory

logger = logging.getLogger(__name__)


async def _run_refresh(session_factory) -> None:
    from app.costs.providers import LiteLLMCostProvider
    from app.costs.repositories import LlmCostRepository
    from app.costs.services import RefreshLlmCostsService

    async with session_factory() as session:
        repository = LlmCostRepository(session)
        response = await RefreshLlmCostsService(
            LiteLLMCostProvider(), repository
        ).execute()
    logger.info(
        "Scheduled refresh complete: %d rows affected in %.0fms",
        response.rows_affected,
        response.duration_ms,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.config import get_settings

    settings = get_settings()

    # Run DB migrations in a thread — command.upgrade calls asyncio.run() internally,
    # which conflicts with the already-running uvicorn event loop if called directly.
    alembic_cfg = AlembicConfig("alembic.ini")
    await asyncio.to_thread(command.upgrade, alembic_cfg, "head")

    # Init DB
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    app.state.db_session_factory = session_factory

    # Start scheduler
    scheduler = None
    if settings.enable_scheduler:
        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            _run_refresh,
            "interval",
            hours=settings.scheduler_interval_hours,
            args=[session_factory],
        )
        scheduler.start()
        logger.info(
            "Scheduler started: refresh every %dh", settings.scheduler_interval_hours
        )

    yield

    if scheduler is not None:
        scheduler.shutdown()
    await engine.dispose()


app = FastAPI(title="LLM Cost Service", lifespan=lifespan)
app.include_router(costs_router)
app.include_router(health_router)
