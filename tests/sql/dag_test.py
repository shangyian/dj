"""
Tests for ``datajunction.sql.dag``.
"""

import pytest

from datajunction.models.database import Column, Database, Table
from datajunction.models.node import Node
from datajunction.sql.dag import get_computable_databases, get_referenced_columns
from datajunction.typing import ColumnType


def test_get_computable_databases() -> None:
    """
    Test ``get_computable_databases``.
    """
    database_1 = Database(id=1, name="shared", URI="sqlite://", cost=1.0)
    database_2 = Database(id=2, name="not shared", URI="sqlite://", cost=2.0)
    database_3 = Database(id=3, name="fast", URI="sqlite://", cost=0.1)

    parent_a = Node(
        name="A",
        tables=[
            Table(database=database_1, table="A"),
            Table(database=database_2, table="A"),
        ],
    )

    parent_b = Node(
        name="B",
        tables=[Table(database=database_1, table="B")],
    )

    child = Node(
        name="C",
        tables=[Table(database=database_3, table="C")],
        parents=[parent_a, parent_b],
    )

    assert {database.name for database in get_computable_databases(child)} == {
        "fast",
        "shared",
    }
    assert {database.name for database in get_computable_databases(parent_a)} == {
        "shared",
        "not shared",
    }
    assert {database.name for database in get_computable_databases(parent_b)} == {
        "shared",
    }


def test_get_computable_databases_heterogeneous_columns() -> None:
    """
    Test ``get_computable_databases`` when columns are heterogeneous.
    """
    database_1 = Database(id=1, name="one", URI="sqlite://", cost=1.0)
    database_2 = Database(id=2, name="two", URI="sqlite://", cost=2.0)

    parent = Node(
        name="core.A",
        tables=[
            Table(
                database=database_1,
                table="A",
                columns=[
                    Column(name="ds", type=ColumnType.STR),
                    Column(name="user_id", type=ColumnType.INT),
                ],
            ),
            Table(
                database=database_2,
                table="A",
                columns=[
                    Column(name="ds", type=ColumnType.STR),
                ],
            ),
        ],
    )

    child_1 = Node(
        name="core.B",
        expression="SELECT COUNT(core.A.user_id) FROM core.A",
        parents=[parent],
    )

    assert {database.name for database in get_computable_databases(child_1)} == {
        "one",
    }

    child_2 = Node(
        name="core.C",
        expression="SELECT COUNT(user_id) FROM core.A",
        parents=[parent],
    )

    assert {database.name for database in get_computable_databases(child_2)} == {
        "one",
    }


def test_get_referenced_columns() -> None:
    """
    Test ``get_referenced_columns``.
    """
    database = Database(id=1, name="one", URI="sqlite://", cost=1.0)

    parent_1 = Node(
        name="core.A",
        tables=[
            Table(
                database=database,
                table="A",
                columns=[
                    Column(name="ds", type=ColumnType.STR),
                    Column(name="user_id", type=ColumnType.INT),
                ],
            ),
        ],
    )
    parent_2 = Node(
        name="core.B",
        tables=[
            Table(
                database=database,
                table="B",
                columns=[
                    Column(name="ds", type=ColumnType.STR),
                    Column(name="event_id", type=ColumnType.INT),
                ],
            ),
        ],
    )

    assert get_referenced_columns("SELECT core.A.ds FROM core.A", [parent_1]) == {
        "core.A": {"ds"},
    }
    assert get_referenced_columns("SELECT ds FROM core.A", [parent_1]) == {
        "core.A": {"ds"},
    }
    assert (
        get_referenced_columns(
            "SELECT ds FROM core.A WHERE user_id > 0",
            [parent_1],
        )
        == {"core.A": {"ds", "user_id"}}
    )
    assert (
        get_referenced_columns(
            (
                "SELECT core.A.ds, core.A.user_id, core.B.event_id "
                "FROM core.A JOIN core.B ON core.A.ds = core.B.ds"
            ),
            [parent_1, parent_2],
        )
        == {"core.A": {"ds", "user_id"}, "core.B": {"ds", "event_id"}}
    )
    assert (
        get_referenced_columns(
            (
                "SELECT user_id, event_id "
                "FROM core.A JOIN core.B ON core.A.ds = core.B.ds"
            ),
            [parent_1, parent_2],
        )
        == {"core.A": {"ds", "user_id"}, "core.B": {"ds", "event_id"}}
    )
    with pytest.raises(Exception) as excinfo:
        get_referenced_columns(
            (
                "SELECT ds, user_id, event_id "
                "FROM core.A JOIN core.B ON core.A.ds = core.B.ds"
            ),
            [parent_1, parent_2],
        )
    assert str(excinfo.value) == "Column ds is ambiguous"
    with pytest.raises(Exception) as excinfo:
        get_referenced_columns("SELECT invalid FROM core.A", [parent_1])
    assert str(excinfo.value) == "Column invalid not found in any parent"