from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.costs.entities import LlmCostEntity
from app.costs.models import LlmCost


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
