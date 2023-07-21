"""
Cube related APIs.
"""
import logging

from fastapi import APIRouter, Depends
from sqlmodel import Session

from datajunction_server.api.helpers import get_node_by_name
from datajunction_server.models.cube import CubeRevisionMetadata
from datajunction_server.models.node import NodeType
from datajunction_server.utils import get_session

_logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/cubes/{name}/", response_model=CubeRevisionMetadata)
def get_a_cube(
    name: str, *, session: Session = Depends(get_session)
) -> CubeRevisionMetadata:
    """
    Get information on a cube
    """
    node = get_node_by_name(session=session, name=name, node_type=NodeType.CUBE)
    return node.current
