from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class LlmCost(Base):
    __tablename__ = "llm_costs"

    provider: Mapped[str] = mapped_column(String(64), primary_key=True)
    model: Mapped[str] = mapped_column(String(256), primary_key=True)
    source: Mapped[str] = mapped_column(String(64), primary_key=True)
    prompt_cost_per_token: Mapped[float] = mapped_column(Float, nullable=False)
    completion_cost_per_token: Mapped[float] = mapped_column(Float, nullable=False)
    # Open-ended {token_category: cost_per_token} map. Nullable for back-compat
    # with rows written before this column existed (treated as {} on read).
    # Generic JSON maps to JSONB on Postgres and JSON on the SQLite test env.
    price_details: Mapped[dict[str, float] | None] = mapped_column(JSON, nullable=True)
    # Whether the model is a usable chat target. Nullable for back-compat with
    # rows written before this column existed (NULL = unknown; consumers fall
    # back to their own heuristic).
    is_chat_capable: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
