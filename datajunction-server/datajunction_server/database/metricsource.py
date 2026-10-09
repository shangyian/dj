"""An additional query source for a multi-source metric."""

from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from datajunction_server.database.base import Base

if TYPE_CHECKING:
    from datajunction_server.database.node import Node, NodeRevision


class MetricSource(Base):
    """A metric query beyond `NodeRevision.query`, which is ordinal 0."""

    __tablename__ = "metricsource"
    __table_args__ = (sa.UniqueConstraint("node_revision_id", "ordinal"),)

    id: Mapped[int] = mapped_column(
        sa.BigInteger().with_variant(sa.Integer, "sqlite"),
        primary_key=True,
    )
    node_revision_id: Mapped[int] = mapped_column(
        sa.ForeignKey("noderevision.id"),
        nullable=False,
    )
    ordinal: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    query: Mapped[str] = mapped_column(sa.String, nullable=False)
    parent_node_id: Mapped[int | None] = mapped_column(
        sa.ForeignKey("node.id"),
        nullable=True,
    )

    node_revision: Mapped["NodeRevision"] = relationship(
        "NodeRevision",
        back_populates="additional_sources",
        foreign_keys=[node_revision_id],
    )
    parent_node: Mapped["Node | None"] = relationship(
        "Node",
        foreign_keys=[parent_node_id],
    )
