from datetime import datetime, timezone

from sqlalchemy import distinct, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from service.costs.entities import LlmCostEntity
from service.costs.models import LlmCost


class LlmCostRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, provider: str, model: str) -> LlmCostEntity | None:
        result = await self.session.execute(
            select(LlmCost).where(
                LlmCost.provider == provider,
                LlmCost.model == model,
            )
        )
        row = result.scalar_one_or_none()
        return self._to_entity(row) if row is not None else None

    async def list_by_provider(self, provider: str) -> list[LlmCostEntity]:
        result = await self.session.execute(
            select(LlmCost).where(LlmCost.provider == provider)
        )
        return [self._to_entity(row) for row in result.scalars().all()]

    async def list_model_names_by_provider(self, provider: str) -> list[str]:
        result = await self.session.execute(
            select(distinct(LlmCost.model))
            .where(LlmCost.provider == provider)
            .order_by(LlmCost.model)
        )
        return list(result.scalars().all())

    async def list_all(self) -> list[LlmCostEntity]:
        result = await self.session.execute(select(LlmCost))
        return [self._to_entity(row) for row in result.scalars().all()]

    async def upsert_all(self, entities: list[LlmCostEntity]) -> int:
        if not entities:
            return 0

        now = datetime.now(timezone.utc)
        rows = [
            {
                "provider": e.provider,
                "model": e.model,
                "prompt_cost_per_token": e.prompt_cost_per_token,
                "completion_cost_per_token": e.completion_cost_per_token,
                "source": e.source,
                "updated_at": now,
            }
            for e in entities
        ]

        stmt = pg_insert(LlmCost).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["provider", "model"],
            set_={
                "prompt_cost_per_token": stmt.excluded.prompt_cost_per_token,
                "completion_cost_per_token": stmt.excluded.completion_cost_per_token,
                "source": stmt.excluded.source,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount

    async def get_status(self) -> dict:
        result = await self.session.execute(
            select(
                func.count().label("total_models"),
                func.count(distinct(LlmCost.provider)).label("total_providers"),
                func.max(LlmCost.updated_at).label("last_refreshed_at"),
            )
        )
        row = result.one()
        return {
            "total_models": row.total_models,
            "total_providers": row.total_providers,
            "last_refreshed_at": row.last_refreshed_at,
        }

    @staticmethod
    def _to_entity(row: LlmCost) -> LlmCostEntity:
        return LlmCostEntity(
            provider=row.provider,
            model=row.model,
            prompt_cost_per_token=row.prompt_cost_per_token,
            completion_cost_per_token=row.completion_cost_per_token,
            source=row.source,
            updated_at=row.updated_at,
        )
