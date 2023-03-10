"""
Models for tables.
"""

from typing import TYPE_CHECKING, List, Optional, Tuple, TypedDict

from sqlmodel import Field, Relationship

from dj.models.base import BaseSQLModel

if TYPE_CHECKING:
    from dj.models.catalog import Catalog
    from dj.models.column import Column
    from dj.models.database import Database
    from dj.models.node import NodeRevision


class TableYAML(TypedDict, total=False):
    """
    Schema of a table in the YAML file.
    """

    catalog: Optional[str]
    schema: Optional[str]
    table: str
    cost: float


class TableColumns(BaseSQLModel, table=True):  # type: ignore
    """
    Join table for table columns.
    """

    table_id: Optional[int] = Field(
        default=None,
        foreign_key="table.id",
        primary_key=True,
    )
    column_id: Optional[int] = Field(
        default=None,
        foreign_key="column.id",
        primary_key=True,
    )


class TableBase(BaseSQLModel):
    """
    A base table.
    """

    schema_: Optional[str] = Field(default=None, alias="schema")
    table: str
    cost: float = 1.0


# class TableNodeRevision(BaseSQLModel, table=True):  # type: ignore
#     """
#     Link between a table and a node revision.
#     """

#     table_id: Optional[int] = Field(
#         default=None,
#         foreign_key="table.id",
#         primary_key=True,
#     )
#     node_revision_id: Optional[int] = Field(
#         default=None,
#         foreign_key="noderevision.id",
#         primary_key=True,
#     )


class Table(TableBase, table=True):  # type: ignore
    """
    A table with data.

    Nodes can have data in multiple tables, in different databases.
    """

    id: Optional[int] = Field(default=None, primary_key=True)

    # node: "NodeRevision" = Relationship(
    #     back_populates="tables",
    #     link_model=TableNodeRevision,
    #     sa_relationship_kwargs={
    #         "primaryjoin": "Table.id==TableNodeRevision.table_id",
    #         "secondaryjoin": "NodeRevision.id==TableNodeRevision.node_revision_id",
    #     },
    # )

    # catalog_id: Optional[int] = Field(foreign_key="catalog.id")
    # catalog: Optional["Catalog"] = Relationship(back_populates="tables")

    database_id: int = Field(foreign_key="database.id")
    database: "Database" = Relationship(back_populates="tables")

    columns: List["Column"] = Relationship(
        link_model=TableColumns,
        sa_relationship_kwargs={
            "primaryjoin": "Table.id==TableColumns.table_id",
            "secondaryjoin": "Column.id==TableColumns.column_id",
            "cascade": "all, delete",
        },
    )

    def to_yaml(self) -> TableYAML:
        """
        Serialize the table for YAML.
        """
        return {
            "catalog": self.catalog,
            "schema": self.schema_,
            "table": self.table,
            "cost": self.cost,
        }

    def identifier(
        self,
    ) -> Tuple[Optional[str], Optional[str], str]:  # pragma: no cover
        """
        Unique identifier for this table.
        """
        # Catalogs will soon be required and this return can be simplified
        return (
            self.catalog.name if self.catalog else None,  # pylint: disable=no-member
            self.schema_,
            self.table,
        )

    def __hash__(self):
        return hash(self.id)


class CreateColumn(BaseSQLModel):
    """
    A column creation request
    """

    name: str
    type: str


class CreateTable(TableBase):
    """
    Create table input
    """

    database_name: str
    catalog_name: str
    columns: List[CreateColumn]
