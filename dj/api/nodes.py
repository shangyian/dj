"""
Node related APIs.
"""

import logging
from datetime import datetime
from http import HTTPStatus
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, SQLModel, select

from dj.models.column import ColumnType
from dj.models.node import Node, NodeType, NodeHistory
from dj.sql.dag import get_referenced_columns_from_sql
from dj.utils import get_session

_logger = logging.getLogger(__name__)
router = APIRouter()


class SimpleColumn(SQLModel):
    """
    A simplified column schema, without ID or dimensions.
    """

    name: str
    type: ColumnType


class NodeMetadata(SQLModel):
    """
    A node with information about columns and if it is a metric.
    """

    id: int
    name: str
    description: str = ""

    created_at: datetime
    updated_at: datetime

    type: NodeType
    query: Optional[str] = None

    columns: List[SimpleColumn]


@router.get("/nodes/", response_model=List[NodeMetadata])
def read_nodes(*, session: Session = Depends(get_session)) -> List[NodeMetadata]:
    """
    List the available nodes.
    """
    return session.exec(select(Node)).all()


@router.get("/nodes/{node_id}/history", response_model=List[NodeHistory])
def get_node_history(node_id: int, *, session: Session = Depends(get_session)) -> List[NodeMetadata]:
    """
    List the available nodes.
    """
    node = session.get(Node, node_id)
    return node.revisions


@router.post("/nodes/{node_id}/", response_model=List[NodeMetadata])
async def druid_spec(
    node_id: int,
    database_id: Optional[int] = None,
    *,
    session: Session = Depends(get_session),
) -> List[NodeMetadata]:
    """
    Return SQL for a metric.

    A database can be optionally specified. If no database is specified the optimal one
    will be used.
    """
    node = session.get(Node, node_id)

    if node.type != NodeType.TRANSFORM:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Druid ingestion spec is only available for transform nodes."
        )

    downstream_metrics = node.downstream_metrics()

    referenced_columns = get_referenced_columns_from_sql(node.query, node.parents)

    if downstream_metrics:
        for metric in downstream_metrics:
            print(metric)
    print(referenced_columns)
    return downstream_metrics
