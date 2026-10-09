"""Resolve the additional query sources of a metric."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from datajunction_server.database.metricsource import MetricSource
from datajunction_server.database.node import NodeRevision
from datajunction_server.errors import DJInvalidInputException
from datajunction_server.internal.validation import validate_node_data
from datajunction_server.models.node_type import NodeType


async def resolve_metric_sources(
    session: AsyncSession,
    metric_name: str,
    queries: list[str],
) -> list[MetricSource]:
    """Build one MetricSource per query, each bound to its parent node."""
    sources = []
    for ordinal, query in enumerate(queries, start=1):
        probe = NodeRevision(name=metric_name, type=NodeType.METRIC, query=query)
        validator = await validate_node_data(probe, session)
        if validator.errors:
            raise DJInvalidInputException(
                message=f"Additional query {ordinal} is invalid.",
                errors=validator.errors,
            )
        parents = list(validator.dependencies_map)
        if len(parents) != 1 or parents[0].type == NodeType.METRIC:
            raise DJInvalidInputException(
                message=(
                    f"Additional query {ordinal} must reference "
                    "exactly one non-metric node."
                ),
            )
        sources.append(
            MetricSource(
                ordinal=ordinal,
                query=query,
                parent_node_id=parents[0].node_id,
            ),
        )
    return sources


async def copy_metric_sources(
    session: AsyncSession,
    revision: NodeRevision,
) -> list[MetricSource]:
    """Copy a revision's sources as new rows."""
    with session.no_autoflush:
        rows = (
            await session.execute(
                select(MetricSource)
                .where(MetricSource.node_revision_id == revision.id)
                .order_by(MetricSource.ordinal),
            )
        ).scalars()
        return [
            MetricSource(
                ordinal=row.ordinal,
                query=row.query,
                parent_node_id=row.parent_node_id,
            )
            for row in rows
        ]
