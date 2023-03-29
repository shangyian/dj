"""
Models for columns.
"""
from typing import TYPE_CHECKING, List, Optional, Tuple, TypedDict

from sqlalchemy import TypeDecorator, String
from sqlalchemy.sql.schema import Column as SqlaColumn
from sqlmodel import Field, Relationship

from dj.models.base import BaseSQLModel
from dj.sql.parsing.types import ColumnType
from sqlalchemy.types import Text

if TYPE_CHECKING:
    from dj.models.attribute import ColumnAttribute
    from dj.models.node import Node


class ColumnYAML(TypedDict, total=False):
    """
    Schema of a column in the YAML file.
    """

    type: str
    dimension: str


class ColumnTypeDecorator(TypeDecorator):
    impl = Text

    def process_bind_param(self, value: ColumnType, dialect):
        return value.__str__()

    def process_result_value(self, value, dialect):
        return ColumnType(value, value)


class Column(BaseSQLModel, table=True):  # type: ignore
    """
    A column.

    Columns can be physical (associated with ``Table`` objects) or abstract (associated
    with ``Node`` objects).
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    type: ColumnType = Field(sa_column=SqlaColumn(ColumnTypeDecorator, nullable=False))

    dimension_id: Optional[int] = Field(default=None, foreign_key="node.id")
    dimension: "Node" = Relationship()
    dimension_column: Optional[str] = None

    attributes: List["ColumnAttribute"] = Relationship(
        back_populates="column",
        sa_relationship_kwargs={
            "lazy": "joined",
        },
    )

    def identifier(self) -> Tuple[str, ColumnType]:
        """
        Unique identifier for this column.
        """
        return self.name, self.type

    def __hash__(self) -> int:
        return hash(self.id)

    class Config:
        arbitrary_types_allowed = True


class ColumnAttributeInput(BaseSQLModel):
    """
    A column attribute input
    """

    attribute_type_namespace: Optional[str] = "system"
    attribute_type_name: str
    column_name: str
