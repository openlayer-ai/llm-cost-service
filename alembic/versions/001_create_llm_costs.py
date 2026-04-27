"""create llm_costs table

Revision ID: 001
Revises:
Create Date: 2026-04-27

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "llm_costs",
        sa.Column("provider", sa.String(64), nullable=False),
        sa.Column("model", sa.String(256), nullable=False),
        sa.Column("prompt_cost_per_token", sa.Float(), nullable=False),
        sa.Column("completion_cost_per_token", sa.Float(), nullable=False),
        sa.Column("source", sa.String(64), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("provider", "model"),
    )


def downgrade() -> None:
    op.drop_table("llm_costs")
