"""remove bogus sample_spec placeholder row

LiteLLM's model_cost JSON ships a `sample_spec` entry that documents the
schema rather than a real model. Its provider is the docs URL
("one of https://docs.litellm.ai/docs/providers"). We filter it out in
`LiteLLMCostProvider.fetch_costs`, but `upsert_all` doesn't delete rows
missing from the source — so existing deployments still carry the row
until removed by hand. This migration deletes it idempotently.

Revision ID: 002
Revises: 001
Create Date: 2026-05-15

"""

from typing import Sequence, Union

from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DELETE FROM llm_costs WHERE model = 'sample_spec'")


def downgrade() -> None:
    # Not restoring intentionally — the source row is bogus.
    pass
