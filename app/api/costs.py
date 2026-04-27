from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.costs.entities import LlmCostEntity
from app.costs.providers import LiteLLMCostProvider
from app.costs.repositories import LlmCostRepository
from app.costs.schemas import (
    ListCostsResponse,
    LlmCostSchema,
    RefreshResponse,
    StatusResponse,
)
from app.costs.services import (
    GetLlmCostService,
    ListLlmCostsService,
    RefreshLlmCostsService,
)
from app.deps import get_db_session

router = APIRouter(prefix="/v1/costs", tags=["costs"])

_READ_CACHE = "public, max-age=3600"


def _to_schema(entity: LlmCostEntity) -> LlmCostSchema:
    return LlmCostSchema(
        provider=entity.provider,
        model=entity.model,
        prompt_cost_per_token=entity.prompt_cost_per_token,
        completion_cost_per_token=entity.completion_cost_per_token,
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
    session: AsyncSession = Depends(get_db_session),
) -> ListCostsResponse:
    response.headers["Cache-Control"] = _READ_CACHE
    repository = LlmCostRepository(session)
    result = await ListLlmCostsService(repository).execute(provider=provider)
    return ListCostsResponse(costs=[_to_schema(c) for c in result.costs])


@router.get("/{provider}/{model:path}", response_model=LlmCostSchema)
async def get_cost(
    provider: str,
    model: str,
    response: Response,
    session: AsyncSession = Depends(get_db_session),
) -> LlmCostSchema:
    response.headers["Cache-Control"] = _READ_CACHE
    repository = LlmCostRepository(session)
    result = await GetLlmCostService(repository).execute(provider=provider, model=model)
    return _to_schema(result.cost)


@router.post("/refresh", response_model=RefreshResponse)
async def refresh_costs(
    authorization: str | None = Header(None),
    session: AsyncSession = Depends(get_db_session),
) -> RefreshResponse:
    secret = get_settings().cron_secret
    if secret and authorization != f"Bearer {secret}":
        raise HTTPException(status_code=401, detail="Invalid or missing authorization.")
    repository = LlmCostRepository(session)
    result = await RefreshLlmCostsService(LiteLLMCostProvider(), repository).execute()
    return RefreshResponse(
        rows_affected=result.rows_affected, duration_ms=result.duration_ms
    )
