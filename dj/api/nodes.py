"""
Node related APIs.
"""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, SQLModel, select

from dj.models.column import ColumnType
from dj.models.node import Node, NodeType, NodeHistory
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
    nodes = session.exec(select(Node)).all()
    print(nodes)
    return nodes


@router.get("/nodes/{node_id}/history", response_model=List[NodeHistory])
def get_node_history(node_id: int, *, session: Session = Depends(get_session)) -> List[NodeMetadata]:
    """
    List the available nodes.
    """
    node = session.get(Node, node_id)
    return node.revisions
