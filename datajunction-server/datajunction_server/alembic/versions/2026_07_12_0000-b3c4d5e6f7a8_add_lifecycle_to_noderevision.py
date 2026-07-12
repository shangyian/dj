"""
Add lifecycle column to noderevision

Revision ID: b3c4d5e6f7a8
Revises: 3c7a9f1b2e4d
Create Date: 2026-07-12 00:00:00.000000+00:00

Adds `lifecycle` (dev/experimental/stable/deprecated/retired) and backfills it
from the existing `mode`: draft -> dev, published -> stable. `mode` is retained
and derived from `lifecycle` on write.
"""
# pylint: disable=no-member, invalid-name, missing-function-docstring, unused-import, no-name-in-module

import sqlalchemy as sa
from alembic import op

revision = "b3c4d5e6f7a8"
down_revision = "3c7a9f1b2e4d"
branch_labels = None
depends_on = None

lifecycle_enum = sa.Enum(
    "DEV",
    "EXPERIMENTAL",
    "STABLE",
    "DEPRECATED",
    "RETIRED",
    name="lifecyclestate",
)


def upgrade():
    lifecycle_enum.create(op.get_bind(), checkfirst=True)
    op.add_column(
        "noderevision",
        sa.Column("lifecycle", lifecycle_enum, nullable=True),
    )
    op.execute(
        "UPDATE noderevision SET lifecycle = "
        "(CASE mode WHEN 'DRAFT' THEN 'DEV' ELSE 'STABLE' END)::lifecyclestate",
    )
    op.alter_column("noderevision", "lifecycle", nullable=False)


def downgrade():
    op.drop_column("noderevision", "lifecycle")
    lifecycle_enum.drop(op.get_bind(), checkfirst=True)
