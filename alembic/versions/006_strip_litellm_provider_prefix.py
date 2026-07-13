"""strip redundant <provider>/ prefix from litellm model names

LiteLLM keys many models with a redundant ``<litellm_provider>/`` prefix
(``azure/codex-mini``, ``gemini/gemini-1.5-pro``, ``azure/eu/gpt-4o-...``) that
duplicates the ``provider`` column and makes model names harder for consumers
to match. ``LiteLLMCostProvider.fetch_costs`` now strips that leading segment
at ingest, so fresh rows are written under the bare name (``codex-mini``,
``eu/gpt-4o-...``).

``upsert_all`` never deletes rows missing from the source, so existing
deployments still carry the old prefixed rows until removed by hand. This
migration deletes the now-orphaned prefixed litellm rows idempotently; the next
refresh repopulates them under the stripped names. Only ``source='litellm'``
rows are touched — OpenRouter rows were never prefixed.

Revision ID: 006
Revises: 005
Create Date: 2026-07-13

"""

from typing import Sequence, Union

from alembic import op

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # A row is prefixed when its model starts with "<provider>/". Expressed
    # portably for Postgres (prod) and SQLite (tests): match the provider name
    # followed by "/" at position 1 of the model string.
    bind = op.get_bind()
    if bind.dialect.name == "sqlite":
        op.execute(
            "DELETE FROM llm_costs "
            "WHERE source = 'litellm' "
            "AND instr(model, provider || '/') = 1"
        )
    else:
        op.execute(
            "DELETE FROM llm_costs "
            "WHERE source = 'litellm' "
            "AND position(provider || '/' in model) = 1"
        )


def downgrade() -> None:
    # Not restoring intentionally — the prefixed names are the artifact we
    # removed; the next refresh writes the stripped names.
    pass
