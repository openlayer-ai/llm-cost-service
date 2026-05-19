import logging
import time
from dataclasses import dataclass, field

from service.costs.entities import LlmCostEntity
from service.costs.providers import CostDataProvider
from service.costs.repositories import LlmCostRepository

logger = logging.getLogger(__name__)


class RefreshLlmCostsService:
    @dataclass(frozen=True)
    class Response:
        rows_affected: int
        duration_ms: float
        per_source: dict[str, int] = field(default_factory=dict)
        failed_sources: list[str] = field(default_factory=list)

    def __init__(
        self,
        providers: list[CostDataProvider],
        repository: LlmCostRepository,
    ) -> None:
        if not providers:
            raise ValueError("RefreshLlmCostsService requires at least one provider")
        self.providers = providers
        self.repository = repository

    async def execute(self) -> Response:
        start = time.monotonic()

        all_entities: list[LlmCostEntity] = []
        per_source: dict[str, int] = {}
        failed_sources: list[str] = []

        for provider in self.providers:
            label = type(provider).__name__
            try:
                entities = provider.fetch_costs()
            except Exception:
                # A failing source must not wipe out the others.
                logger.exception("Cost provider %s failed; skipping", label)
                failed_sources.append(label)
                continue

            source_name = entities[0].source if entities else label
            per_source[source_name] = per_source.get(source_name, 0) + len(entities)
            all_entities.extend(entities)

        rows_affected = await self.repository.upsert_all(all_entities)
        duration_ms = (time.monotonic() - start) * 1000
        return self.Response(
            rows_affected=rows_affected,
            duration_ms=duration_ms,
            per_source=per_source,
            failed_sources=failed_sources,
        )


class GetLlmCostService:
    @dataclass(frozen=True)
    class Response:
        cost: LlmCostEntity | None

    def __init__(self, repository: LlmCostRepository) -> None:
        self.repository = repository

    async def execute(
        self,
        provider: str,
        model: str,
        source: str | None = None,
    ) -> Response:
        normalized = provider.lower().strip()
        normalized_source = source.lower().strip() if source else None
        cost = await self.repository.get(
            provider=normalized, model=model, source=normalized_source
        )
        return self.Response(cost=cost)


class ListLlmCostsService:
    @dataclass(frozen=True)
    class Response:
        costs: list[LlmCostEntity]

    def __init__(self, repository: LlmCostRepository) -> None:
        self.repository = repository

    async def execute(
        self,
        provider: str | None = None,
        source: str | None = None,
        resolved: bool = True,
    ) -> Response:
        normalized_source = source.lower().strip() if source else None
        if provider is not None:
            normalized = provider.lower().strip()
            costs = await self.repository.list_by_provider(
                provider=normalized, source=normalized_source, resolved=resolved
            )
        else:
            costs = await self.repository.list_all(
                source=normalized_source, resolved=resolved
            )
        return self.Response(costs=costs)
