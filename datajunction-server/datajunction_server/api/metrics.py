"""
Metric related APIs.
"""

from http import HTTPStatus

from fastapi import BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.orm.attributes import set_committed_value
from sqlalchemy.sql.operators import is_

from datajunction_server.api.nodes import list_nodes
from datajunction_server.construction.build_v3.decomposition import (
    is_derived_metric,
)
from datajunction_server.construction.build_v3.dimensions import resolve_dimensions
from datajunction_server.construction.build_v3.loaders import load_nodes
from datajunction_server.construction.build_v3.source_matcher import (
    find_matching_source,
)
from datajunction_server.construction.build_v3.types import BuildContext
from datajunction_server.database.node import Node, NodeRevision
from datajunction_server.errors import DJError, DJInvalidInputException, ErrorCode
from datajunction_server.internal.access.authentication.http import SecureAPIRouter
from datajunction_server.internal.access.authorization import (
    AccessChecker,
    get_access_checker,
)
from datajunction_server.internal.caching.cachelib_cache import get_cache
from datajunction_server.internal.caching.interface import Cache
from datajunction_server.models.metric import (
    CompatibleDimensions,
    CompatibleDimensionsInput,
    Metric,
)
from datajunction_server.models.node import (
    DimensionAttributeOutput,
    MetricDirection,
    MetricMetadataOptions,
    MetricUnit,
)
from datajunction_server.models.node_type import NodeType
from datajunction_server.sql.dag import get_dimensions, get_shared_dimensions
from datajunction_server.utils import (
    get_session,
    get_settings,
)

settings = get_settings()
router = SecureAPIRouter(tags=["metrics"])


async def get_metric(session: AsyncSession, name: str) -> Node:
    """
    Return a metric node given a node name.
    """
    node = await Node.get_by_name(
        session,
        name,
        options=[
            selectinload(Node.current).options(
                *NodeRevision.default_load_options(),
                selectinload(NodeRevision.required_dimensions),
            ),
        ],
        raise_if_not_exists=True,
    )
    if node.type != NodeType.METRIC:  # type: ignore
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Not a metric node: `{name}`",
        )
    return node  # type: ignore


@router.get("/metrics/", response_model=list[str])
async def list_metrics(
    prefix: str | None = None,
    *,
    session: AsyncSession = Depends(get_session),
    access_checker: AccessChecker = Depends(get_access_checker),
    cache: Cache = Depends(get_cache),
    background_tasks: BackgroundTasks,
) -> list[str]:
    """
    List all available metrics.
    """
    metrics = cache.get("metrics")
    if metrics is None:
        metrics = await list_nodes(
            node_type=NodeType.METRIC,
            prefix=prefix,
            session=session,
            access_checker=access_checker,
        )
        background_tasks.add_task(cache.set, "metrics", metrics)
    return metrics


@router.get("/metrics/metadata")
async def list_metric_metadata() -> MetricMetadataOptions:
    """
    Return available metric metadata attributes
    """
    return_obj = MetricMetadataOptions(
        directions=[MetricDirection(e) for e in MetricDirection],
        units=[MetricUnit(e).value for e in MetricUnit],
    )
    return return_obj


@router.get("/metrics/{name}/", response_model=Metric)
async def get_a_metric(
    name: str,
    *,
    session: AsyncSession = Depends(get_session),
) -> Metric:
    """
    Return a metric by name.
    """
    node = await get_metric(session, name)
    # get_dimensions handles derived metrics by finding ultimate non-metric parents
    dims = await get_dimensions(session, node)
    metric = await Metric.parse_node(node, dims, session)
    return metric


@router.post(
    "/metrics/{name}/compatible-dimensions/",
    response_model=CompatibleDimensions,
)
async def check_compatible_dimensions(
    name: str,
    data: CompatibleDimensionsInput,
    *,
    session: AsyncSession = Depends(get_session),
) -> CompatibleDimensions:
    """Check whether one metric source serves the dimensions."""
    await get_metric(session, name)
    ctx = BuildContext(session=session, metrics=[name], dimensions=data.dimensions)
    await load_nodes(ctx)
    metric = ctx.nodes[name]
    if is_derived_metric(ctx, metric):
        raise DJInvalidInputException(f"Cannot check derived metric {name}.")
    if name not in ctx.source_metrics:
        set_committed_value(metric.current, "additional_sources", [])
    try:
        match = await find_matching_source(ctx, metric, ctx.dimensions, set())
        resolve_dimensions(ctx, match.parent_node)
    except DJInvalidInputException as exc:
        return CompatibleDimensions(compatible=False, reason=exc.message)
    return CompatibleDimensions(compatible=True, source=match.ordinal)


@router.get(
    "/metrics/common/dimensions/",
    response_model=list[DimensionAttributeOutput],
)
async def get_common_dimensions(
    metric: list[str] = Query(
        title="List of metrics to find common dimensions for",
        default=[],
    ),
    session: AsyncSession = Depends(get_session),
) -> list[DimensionAttributeOutput]:
    """
    Return common dimensions for a set of metrics.
    """
    input_errors = []
    statement = (
        select(Node)
        .where(Node.name.in_(metric))  # type: ignore
        .where(is_(Node.deactivated_at, None))
    )
    metric_nodes = (await session.execute(statement)).scalars().all()
    for node in metric_nodes:
        if node.type != NodeType.METRIC:
            input_errors.append(
                DJError(
                    message=f"Not a metric node: {node.name}",
                    code=ErrorCode.NODE_TYPE_ERROR,
                ),
            )
    if not metric_nodes:
        input_errors.append(
            DJError(
                message=f"Metric nodes not found: {','.join(metric)}",
                code=ErrorCode.UNKNOWN_NODE,
            ),
        )

    if input_errors:
        raise DJInvalidInputException(errors=input_errors)
    return (await get_shared_dimensions(session, metric_nodes)).shared
