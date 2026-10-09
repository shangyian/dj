"""add decomposition to metric source

Revision ID: ms0002dec
Revises: ms0001src
Create Date: 2026-10-09 01:51:14.339762+00:00

"""

import sqlalchemy as sa
from alembic import op

revision = "ms0002dec"
down_revision = "ms0001src"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("metricsource", schema=None) as batch_op:
        batch_op.add_column(sa.Column("components", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("derived_expression", sa.Text(), nullable=True))


def downgrade():
    with op.batch_alter_table("metricsource", schema=None) as batch_op:
        batch_op.drop_column("derived_expression")
        batch_op.drop_column("components")
