"""
Tests for ``datajunction_server.sql.dag``.
"""
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from datajunction_server.database.column import Column
from datajunction_server.database.database import Database
from datajunction_server.database.node import Node, NodeRevision
from datajunction_server.models.node import DimensionAttributeOutput, NodeType
from datajunction_server.sql.dag import get_dimensions, topological_sort
from datajunction_server.sql.parsing.types import IntegerType, StringType


@pytest.mark.asyncio
async def test_get_dimensions(session: AsyncSession) -> None:
    """
    Test ``get_dimensions``.
    """
    database = Database(id=1, name="one", URI="sqlite://")
    session.add(database)

    dimension_ref = Node(name="B", type=NodeType.DIMENSION, current_version="1")
    dimension = NodeRevision(
        node=dimension_ref,
        name=dimension_ref.name,
        type=dimension_ref.type,
        display_name="B",
        version="1",
        columns=[
            Column(name="id", type=IntegerType(), order=0),
            Column(name="attribute", type=StringType(), order=1),
        ],
    )
    dimension_ref.current = dimension
    session.add(dimension)
    session.add(dimension_ref)

    parent_ref = Node(name="A", current_version="1", type=NodeType.SOURCE)
    parent = NodeRevision(
        node=parent_ref,
        name=parent_ref.name,
        type=parent_ref.type,
        display_name="A",
        version="1",
        columns=[
            Column(name="ds", type=StringType(), order=0),
            Column(name="b_id", type=IntegerType(), dimension=dimension_ref, order=1),
        ],
    )
    parent_ref.current = parent
    session.add(parent)
    session.add(parent_ref)

    child_ref = Node(name="C", current_version="1", type=NodeType.METRIC)
    child = NodeRevision(
        node=child_ref,
        name=child_ref.name,
        display_name="C",
        version="1",
        query="SELECT COUNT(*) FROM A",
        parents=[parent_ref],
        type=NodeType.METRIC,
    )
    child_ref.current = child
    session.add(child)
    session.add(child_ref)
    await session.commit()

    assert await get_dimensions(session, child_ref) == [
        DimensionAttributeOutput(
            name="B.attribute",
            node_name="B",
            node_display_name="B",
            is_primary_key=False,
            type="string",
            path=["A.b_id"],
        ),
        DimensionAttributeOutput(
            name="B.id",
            node_name="B",
            node_display_name="B",
            is_primary_key=False,
            type="int",
            path=["A.b_id"],
        ),
    ]


@pytest.mark.asyncio
async def test_topological_sort(session: AsyncSession) -> None:
    """
    Test ``topological_sort``.
    """
    node_A = Node(name="test.A", type=NodeType.TRANSFORM)
    node_rev_A = NodeRevision(
        node=node_A,
        name=node_A.name,
        parents=[],
    )
    node_A.current = node_rev_A
    session.add(node_A)
    session.add(node_rev_A)

    node_B = Node(name="test.B", type=NodeType.TRANSFORM)
    node_rev_B = NodeRevision(
        node=node_B,
        name=node_B.name,
        parents=[node_A],
    )
    node_B.current = node_rev_B
    session.add(node_B)
    session.add(node_rev_B)

    node_C = Node(name="test.C", type=NodeType.TRANSFORM)
    node_rev_C = NodeRevision(
        node=node_C,
        name=node_C.name,
        parents=[node_A],
    )
    node_C.current = node_rev_C
    session.add(node_C)
    session.add(node_rev_C)

    node_D = Node(name="test.D", type=NodeType.TRANSFORM)
    node_rev_D = NodeRevision(
        node=node_D,
        name=node_D.name,
        parents=[node_B, node_C],
    )
    node_D.current = node_rev_D
    session.add(node_D)
    session.add(node_rev_D)

    node_E = Node(name="test.E", type=NodeType.TRANSFORM)
    node_rev_E = NodeRevision(
        node=node_E,
        name=node_E.name,
        parents=[node_D],
    )
    node_E.current = node_rev_E
    session.add(node_E)
    session.add(node_rev_E)

    node_F = Node(name="test.F", type=NodeType.TRANSFORM)
    node_rev_D.parents.append(node_F)
    node_rev_F = NodeRevision(
        node=node_F,
        name=node_F.name,
        parents=[node_E],
    )
    node_F.current = node_rev_F
    session.add(node_F)
    session.add(node_rev_F)

    ordering = topological_sort([node_A, node_B, node_C, node_D, node_E])
    assert [node.name for node in ordering] == [node_A.name, node_C.name, node_B.name, node_D.name, node_E.name]
