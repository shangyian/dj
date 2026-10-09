"""Resolve the additional query sources of a metric."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from datajunction_server.database.metricsource import MetricSource
from datajunction_server.database.node import Node, NodeRevision
from datajunction_server.errors import DJInvalidInputException
from datajunction_server.internal.validation import validate_node_data
from datajunction_server.models.decompose import MetricComponent
from datajunction_server.models.node_type import NodeType
from datajunction_server.sql.decompose import MetricComponentExtractor


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
        components, derived_expression = await decompose_source(session, probe)
        sources.append(
            MetricSource(
                ordinal=ordinal,
                query=query,
                parent_node_id=parents[0].node_id,
                components=components,
                derived_expression=derived_expression,
            ),
        )
    return sources


async def decompose_source(
    session: AsyncSession,
    probe: NodeRevision,
) -> tuple[list[MetricComponent], str]:
    """Decompose a source query as a base metric."""
    metric = Node(name=probe.name, type=NodeType.METRIC)
    metric.current = probe
    components, derived_sql = await MetricComponentExtractor(0).extract(
        session,
        nodes_cache={probe.name: metric},
        # Parent already checked non-metric
        parent_map={probe.name: []},
        metric_node=metric,
        validate_fixed_grain_windows=False,
    )
    return components, str(derived_sql)


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
                components=row.components,
                derived_expression=row.derived_expression,
            )
            for row in rows
        ]
