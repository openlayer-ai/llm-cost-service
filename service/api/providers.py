from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession

from service.costs.repositories import LlmCostRepository
from service.deps import get_db_session
from service.providers.schemas import ListProviderModelsResponse, ModelInfoSchema
from service.providers.services import ListProviderModelsService

router = APIRouter(prefix="/v1/providers", tags=["providers"])

_READ_CACHE = "public, max-age=3600"


@router.get("/{provider}/models", response_model=ListProviderModelsResponse)
async def list_provider_models(
    provider: str,
    response: Response,
    session: AsyncSession = Depends(get_db_session),
) -> ListProviderModelsResponse:
    response.headers["Cache-Control"] = _READ_CACHE
    repository = LlmCostRepository(session)
    result = await ListProviderModelsService(repository).execute(provider=provider)
    if not result.model_names:
        raise HTTPException(
            status_code=404,
            detail=f"No models found for provider={provider!r}.",
        )
    return ListProviderModelsResponse(
        provider=result.provider,
        models=[ModelInfoSchema(name=n) for n in result.model_names],
    )
