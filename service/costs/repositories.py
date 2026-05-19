from datetime import datetime, timezone

from sqlalchemy import distinct, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from service.costs.entities import LlmCostEntity
from service.costs.models import LlmCost

# Source priority used by resolved reads when the same (provider, model) row
# exists from multiple sources. Lower index wins. Add new sources here as we
# integrate them — order is the only knob.
_SOURCE_PRECEDENCE: tuple[str, ...] = ("litellm", "openrouter")
# Any source not in the list ranks behind every named source.
_UNKNOWN_SOURCE_RANK = len(_SOURCE_PRECEDENCE)


def _source_rank(source: str) -> int:
    try:
        return _SOURCE_PRECEDENCE.index(source)
    except ValueError:
        return _UNKNOWN_SOURCE_RANK


def _resolve(entities: list[LlmCostEntity]) -> list[LlmCostEntity]:
    """Collapse rows from multiple sources to one row per (provider, model)
    using the precedence list. Stable on input order for equal-rank ties."""
    winners: dict[tuple[str, str], LlmCostEntity] = {}
    for entity in entities:
        key = (entity.provider, entity.model)
        current = winners.get(key)
        if current is None or _source_rank(entity.source) < _source_rank(current.source):
            winners[key] = entity
    return list(winners.values())


class LlmCostRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(
        self,
        provider: str,
        model: str,
        source: str | None = None,
    ) -> LlmCostEntity | None:
        stmt = select(LlmCost).where(
            LlmCost.provider == provider,
            LlmCost.model == model,
        )
        if source is not None:
            stmt = stmt.where(LlmCost.source == source)
        result = await self.session.execute(stmt)
        rows = [self._to_entity(r) for r in result.scalars().all()]
        if not rows:
            return None
        if source is not None:
            # Source pin: at most one row by PK.
            return rows[0]
        return _resolve(rows)[0]

    async def list_by_provider(
        self,
        provider: str,
        source: str | None = None,
        resolved: bool = True,
    ) -> list[LlmCostEntity]:
        stmt = select(LlmCost).where(LlmCost.provider == provider)
        if source is not None:
            stmt = stmt.where(LlmCost.source == source)
        result = await self.session.execute(stmt)
        rows = [self._to_entity(r) for r in result.scalars().all()]
        return _resolve(rows) if resolved and source is None else rows

    async def list_model_names_by_provider(self, provider: str) -> list[str]:
        result = await self.session.execute(
            select(distinct(LlmCost.model))
            .where(LlmCost.provider == provider)
            .order_by(LlmCost.model)
        )
        return list(result.scalars().all())

    async def list_all(
        self,
        source: str | None = None,
        resolved: bool = True,
    ) -> list[LlmCostEntity]:
        stmt = select(LlmCost)
        if source is not None:
            stmt = stmt.where(LlmCost.source == source)
        result = await self.session.execute(stmt)
        rows = [self._to_entity(r) for r in result.scalars().all()]
        return _resolve(rows) if resolved and source is None else rows

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
            index_elements=["provider", "model", "source"],
            set_={
                "prompt_cost_per_token": stmt.excluded.prompt_cost_per_token,
                "completion_cost_per_token": stmt.excluded.completion_cost_per_token,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount

    async def get_status(self) -> dict:
        # total_models counts distinct (provider, model) so the count reflects
        # the resolved table, not the per-source row count. Use a subquery for
        # the tuple-distinct since SQLite (test env) doesn't support
        # COUNT(DISTINCT (col1, col2)).
        distinct_models = (
            select(LlmCost.provider, LlmCost.model).distinct().subquery()
        )
        total_models = (
            await self.session.execute(select(func.count()).select_from(distinct_models))
        ).scalar_one()

        result = await self.session.execute(
            select(
                func.count(distinct(LlmCost.provider)).label("total_providers"),
                func.max(LlmCost.updated_at).label("last_refreshed_at"),
            )
        )
        row = result.one()
        return {
            "total_models": total_models,
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
