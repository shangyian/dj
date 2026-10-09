"""add metric source table

Revision ID: ms0001src
Revises: tbl0001idx
Create Date: 2026-10-09 00:00:00.000000+00:00

"""

import sqlalchemy as sa
from alembic import op

revision = "ms0001src"
down_revision = "tbl0001idx"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "metricsource",
        sa.Column(
            "id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            nullable=False,
        ),
        sa.Column(
            "node_revision_id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("query", sa.String(), nullable=False),
        sa.Column(
            "parent_node_id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(["node_revision_id"], ["noderevision.id"]),
        sa.ForeignKeyConstraint(["parent_node_id"], ["node.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("node_revision_id", "ordinal"),
    )


def downgrade():
    op.drop_table("metricsource")
