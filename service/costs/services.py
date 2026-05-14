import time
from dataclasses import dataclass

from fastapi import HTTPException

from service.costs.entities import LlmCostEntity
from service.costs.providers import CostDataProvider
from service.costs.repositories import LlmCostRepository


class RefreshLlmCostsService:
    @dataclass(frozen=True)
    class Response:
        rows_affected: int
        duration_ms: float

    def __init__(
        self, provider: CostDataProvider, repository: LlmCostRepository
    ) -> None:
        self.provider = provider
        self.repository = repository

    async def execute(self) -> Response:
        start = time.monotonic()
        entities = self.provider.fetch_costs()
        rows_affected = await self.repository.upsert_all(entities)
        duration_ms = (time.monotonic() - start) * 1000
        return self.Response(rows_affected=rows_affected, duration_ms=duration_ms)


class GetLlmCostService:
    @dataclass(frozen=True)
    class Response:
        cost: LlmCostEntity

    def __init__(self, repository: LlmCostRepository) -> None:
        self.repository = repository

    async def execute(self, provider: str, model: str) -> Response:
        normalized = provider.lower().strip()
        cost = await self.repository.get(provider=normalized, model=model)
        if cost is None:
            raise HTTPException(
                status_code=404,
                detail=f"No cost data found for provider={provider!r}, model={model!r}.",
            )
        return self.Response(cost=cost)


class ListLlmCostsService:
    @dataclass(frozen=True)
    class Response:
        costs: list[LlmCostEntity]

    def __init__(self, repository: LlmCostRepository) -> None:
        self.repository = repository

    async def execute(self, provider: str | None = None) -> Response:
        if provider is not None:
            normalized = provider.lower().strip()
            costs = await self.repository.list_by_provider(provider=normalized)
        else:
            costs = await self.repository.list_all()
        return self.Response(costs=costs)
