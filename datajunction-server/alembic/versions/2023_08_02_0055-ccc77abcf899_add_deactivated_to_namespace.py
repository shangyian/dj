"""Add deactivated to namespace

Revision ID: ccc77abcf899
Revises: 4147da2ac841
Create Date: 2023-08-02 00:55:12.995328+00:00

"""
# pylint: disable=no-member, invalid-name, missing-function-docstring, unused-import, no-name-in-module

import sqlalchemy as sa
import sqlmodel

from alembic import op

# revision identifiers, used by Alembic.
revision = "ccc77abcf899"
down_revision = "4147da2ac841"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column(
        "nodenamespace",
        sa.Column("deactivated_at", sa.DateTime(timezone=True), nullable=True),
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("nodenamespace", "deactivated_at")
    # ### end Alembic commands ###
