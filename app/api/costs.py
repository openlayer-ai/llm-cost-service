from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.costs.entities import LlmCostEntity
from app.costs.providers import LiteLLMCostProvider
from app.costs.repositories import LlmCostRepository
from app.costs.schemas import ListCostsResponse, LlmCostSchema, RefreshResponse
from app.costs.services import (
    GetLlmCostService,
    ListLlmCostsService,
    RefreshLlmCostsService,
)
from app.deps import get_db_session

router = APIRouter(prefix="/v1/costs", tags=["costs"])


def _to_schema(entity: LlmCostEntity) -> LlmCostSchema:
    return LlmCostSchema(
        provider=entity.provider,
        model=entity.model,
        prompt_cost_per_token=entity.prompt_cost_per_token,
        completion_cost_per_token=entity.completion_cost_per_token,
        source=entity.source,
        updated_at=entity.updated_at,
    )


@router.get("", response_model=ListCostsResponse)
async def list_costs(
    provider: str | None = Query(None, description="Filter by provider name"),
    session: AsyncSession = Depends(get_db_session),
) -> ListCostsResponse:
    repository = LlmCostRepository(session)
    response = await ListLlmCostsService(repository).execute(provider=provider)
    return ListCostsResponse(costs=[_to_schema(c) for c in response.costs])


@router.get("/{provider}/{model:path}", response_model=LlmCostSchema)
async def get_cost(
    provider: str,
    model: str,
    session: AsyncSession = Depends(get_db_session),
) -> LlmCostSchema:
    repository = LlmCostRepository(session)
    response = await GetLlmCostService(repository).execute(
        provider=provider, model=model
    )
    return _to_schema(response.cost)


@router.post("/refresh", response_model=RefreshResponse)
async def refresh_costs(
    session: AsyncSession = Depends(get_db_session),
) -> RefreshResponse:
    repository = LlmCostRepository(session)
    response = await RefreshLlmCostsService(LiteLLMCostProvider(), repository).execute()
    return RefreshResponse(
        rows_affected=response.rows_affected,
        duration_ms=response.duration_ms,
    )
