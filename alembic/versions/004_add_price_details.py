"""add price_details to llm_costs

Adds an open per-category price map ({token_category: cost_per_token}) so
granular costs (cached, cache_creation, audio, ...) are served alongside the
scalar prompt/completion costs. Sourced from LiteLLM/OpenRouter per-category
fields. Nullable for back-compat with existing rows (read as {} when null);
purely additive so consumers that ignore it keep working.

Revision ID: 004
Revises: 003
Create Date: 2026-06-05

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "llm_costs",
        sa.Column("price_details", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("llm_costs", "price_details")
