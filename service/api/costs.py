from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession

from service.config import get_settings
from service.costs.entities import LlmCostEntity
from service.costs.providers import LiteLLMCostProvider, OpenRouterCostProvider
from service.costs.repositories import LlmCostRepository
from service.costs.schemas import (
    ListCostsResponse,
    LlmCostSchema,
    RefreshResponse,
    StatusResponse,
)
from service.costs.services import (
    GetLlmCostService,
    ListLlmCostsService,
    RefreshLlmCostsService,
)
from service.deps import get_db_session

router = APIRouter(prefix="/v1/costs", tags=["costs"])

_READ_CACHE = "public, max-age=3600"


def _build_providers() -> list:
    return [LiteLLMCostProvider(), OpenRouterCostProvider()]


def _to_schema(entity: LlmCostEntity) -> LlmCostSchema:
    return LlmCostSchema(
        provider=entity.provider,
        model=entity.model,
        prompt_cost_per_token=entity.prompt_cost_per_token,
        completion_cost_per_token=entity.completion_cost_per_token,
        price_details=entity.price_details or {},
        source=entity.source,
        updated_at=entity.updated_at,
    )


@router.get("/status", response_model=StatusResponse)
async def get_status(
    session: AsyncSession = Depends(get_db_session),
) -> StatusResponse:
    data = await LlmCostRepository(session).get_status()
    return StatusResponse(**data)


@router.get("", response_model=ListCostsResponse)
async def list_costs(
    response: Response,
    provider: str | None = Query(None, description="Filter by provider name"),
    source: str | None = Query(
        None, description="Filter to a specific cost data source (e.g. 'litellm', 'openrouter')"
    ),
    resolved: bool = Query(
        True,
        description="Collapse multi-source rows to one row per (provider, model) using precedence",
    ),
    session: AsyncSession = Depends(get_db_session),
) -> ListCostsResponse:
    response.headers["Cache-Control"] = _READ_CACHE
    repository = LlmCostRepository(session)
    result = await ListLlmCostsService(repository).execute(
        provider=provider, source=source, resolved=resolved
    )
    return ListCostsResponse(costs=[_to_schema(c) for c in result.costs])


@router.get("/{provider}/{model:path}", response_model=LlmCostSchema)
async def get_cost(
    provider: str,
    model: str,
    response: Response,
    source: str | None = Query(
        None, description="Pin to a specific cost data source (e.g. 'litellm', 'openrouter')"
    ),
    session: AsyncSession = Depends(get_db_session),
) -> LlmCostSchema:
    response.headers["Cache-Control"] = _READ_CACHE
    repository = LlmCostRepository(session)
    result = await GetLlmCostService(repository).execute(
        provider=provider, model=model, source=source
    )
    if result.cost is None:
        raise HTTPException(
            status_code=404,
            detail=f"No cost data found for provider={provider!r}, model={model!r}.",
        )
    return _to_schema(result.cost)


@router.get("/refresh", response_model=RefreshResponse)
async def refresh_costs(
    authorization: str | None = Header(None),
    session: AsyncSession = Depends(get_db_session),
) -> RefreshResponse:
    secret = get_settings().cron_secret
    if secret and authorization != f"Bearer {secret}":
        raise HTTPException(status_code=401, detail="Invalid or missing authorization.")
    repository = LlmCostRepository(session)
    result = await RefreshLlmCostsService(_build_providers(), repository).execute()
    return RefreshResponse(
        rows_affected=result.rows_affected,
        duration_ms=result.duration_ms,
        per_source=result.per_source,
        failed_sources=result.failed_sources,
    )
