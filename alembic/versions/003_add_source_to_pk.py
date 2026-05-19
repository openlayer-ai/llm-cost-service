"""add source to llm_costs primary key

Previously the PK was (provider, model), which let any new cost source
(OpenRouter, Helicone, future direct-scrapers) silently overwrite the
existing row for the same model when the refresh ran. The new PK keeps
one row per (provider, model, source) so multiple sources can coexist
and a precedence list is applied at read time.

Revision ID: 003
Revises: 002
Create Date: 2026-05-19

"""

from typing import Sequence, Union

from alembic import op

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE llm_costs DROP CONSTRAINT llm_costs_pkey")
    op.execute(
        "ALTER TABLE llm_costs ADD CONSTRAINT llm_costs_pkey "
        "PRIMARY KEY (provider, model, source)"
    )


def downgrade() -> None:
    # Collapse to one row per (provider, model) by keeping the most recently
    # updated row before restoring the old PK. Without this dedupe, the
    # constraint would fail on any table where multiple sources coexist.
    op.execute(
        """
        DELETE FROM llm_costs a
        USING llm_costs b
        WHERE a.provider = b.provider
          AND a.model = b.model
          AND (
              a.updated_at < b.updated_at
              OR (a.updated_at = b.updated_at AND a.source > b.source)
          )
        """
    )
    op.execute("ALTER TABLE llm_costs DROP CONSTRAINT llm_costs_pkey")
    op.execute(
        "ALTER TABLE llm_costs ADD CONSTRAINT llm_costs_pkey "
        "PRIMARY KEY (provider, model)"
    )
