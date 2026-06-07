"""add is_chat_capable to llm_costs

Adds a normalized chat-capability flag so consumers can decide which models to
offer in a chat-model selector without re-deriving it from model names. Computed
per source from each provider's own capability metadata (LiteLLM ``mode``,
OpenRouter output modalities) and normalized to a single boolean. Nullable for
back-compat with existing rows (NULL = unknown; consumers fall back to their own
heuristic); purely additive so consumers that ignore it keep working.

Revision ID: 005
Revises: 004
Create Date: 2026-06-07

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "llm_costs",
        sa.Column("is_chat_capable", sa.Boolean(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("llm_costs", "is_chat_capable")
