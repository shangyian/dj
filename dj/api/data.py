"""
Data related APIs.
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlmodel import Session

from dj.api.helpers import get_node_by_name, get_query
from dj.errors import DJException, DJInvalidInputException
from dj.models.metric import TranslatedSQL
from dj.models.node import AvailabilityState, AvailabilityStateBase, NodeType
from dj.models.query import QueryWithResults
from dj.service_clients import QueryServiceClient
from dj.utils import get_query_service_client, get_session

_logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/data/{node_name}/availability/")
def add_availability(
    node_name: str,
    data: AvailabilityStateBase,
    *,
    session: Session = Depends(get_session),
) -> JSONResponse:
    """
    Add an availability state to a node
    """
    node = get_node_by_name(session, node_name)

    # Source nodes require that any availability states set are for one of the defined tables
    node_revision = node.current
    if data.catalog != node_revision.catalog.name:
        raise DJException(
            "Cannot set availability state in different catalog: "
            f"{data.catalog}, {node_revision.catalog}",
        )
    if node.current.type == NodeType.SOURCE:
        if node_revision.schema_ != data.schema_ or node_revision.table != data.table:
            raise DJException(
                message=(
                    "Cannot set availability state, "
                    "source nodes require availability "
                    "states to match the set table: "
                    f"{data.catalog}."
                    f"{data.schema_}."
                    f"{data.table} "
                    "does not match "
                    f"{node_revision.catalog.name}."
                    f"{node_revision.schema_}."
                    f"{node_revision.table} "
                ),
            )

    # Merge the new availability state with the current availability state if one exists
    if (
        node_revision.availability
        and node_revision.availability.catalog == node.current.catalog.name
        and node_revision.availability.schema_ == data.schema_
        and node_revision.availability.table == data.table
    ):
        # Currently, we do not consider type information. We should eventually check the type of
        # the partition values in order to cast them before sorting.
        data.max_partition = max(
            (
                node_revision.availability.max_partition,
                data.max_partition,
            ),
        )
        data.min_partition = min(
            (
                node_revision.availability.min_partition,
                data.min_partition,
            ),
        )

    db_new_availability = AvailabilityState.from_orm(data)
    node_revision.availability = db_new_availability
    session.add(node_revision)
    session.commit()
    return JSONResponse(
        status_code=200,
        content={"message": "Availability state successfully posted"},
    )


@router.get("/data/{node_name}/")
def data_for_node(
    node_name: str,
    dimensions: List[str] = Query([]),
    filters: List[str] = Query([]),
    *,
    session: Session = Depends(get_session),
    query_service_client: QueryServiceClient = Depends(get_query_service_client),
) -> QueryWithResults:
    """
    Gets data for a node
    """
    node = get_node_by_name(session, node_name)
    if node.type not in (NodeType.METRIC, NodeType.CUBE, NodeType.DIMENSION):
        raise DJException(message=f"Can't get data for node type {node.type}!")

    if node.type == NodeType.DIMENSION:
        if dimensions or filters:
            raise DJInvalidInputException(
                message=f"Cannot set filters or dimensions for node type {node.type}!",
            )

    query_ast = get_query(
        session=session,
        metric=node_name,
        dimensions=dimensions,
        filters=filters,
    )
    query = TranslatedSQL(sql=str(query_ast))
    available_engines = node.current.catalog.engines

    result = query_service_client.submit_query(
        engine_name=available_engines[0].name,
        engine_version=available_engines[0].version,
        catalog_name=node.current.catalog.name,
        query=query.sql,
        async_=False,
    )
    return result
