"""Pick which of a metric's query sources serves a request."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from datajunction_server.construction.build_v3.decomposition import decompose_metric
from datajunction_server.construction.build_v3.dimensions import (
    _local_reference_dimension_column,
    find_join_path,
    parse_dimension_ref,
)
from datajunction_server.database.preaggregation import (
    compute_expression_hash,
    measure_identity_token,
)
from datajunction_server.errors import DJInvalidInputException
from datajunction_server.models.decompose import MetricComponent

if TYPE_CHECKING:
    from datajunction_server.construction.build_v3.types import BuildContext
    from datajunction_server.database.node import Node


@dataclass(frozen=True)
class SourceMatch:
    """The source chosen for a metric, with its decomposition."""

    parent_node: Node
    ordinal: int
    components: list[MetricComponent]
    derived_expression: str | None


def _source_gaps(
    ctx: BuildContext,
    parent: Node,
    components: list[MetricComponent] | None,
    dimensions: list[str],
    required_measures: set[str],
) -> tuple[int, list[str]]:
    """Join count and what the source lacks."""
    joins = 0
    gaps: list[str] = []
    columns = {col.name for col in parent.current.columns} if parent.current else set()
    for ref in dimensions:
        dim_ref = parse_dimension_ref(ref)
        if dim_ref.node_name == parent.name and not dim_ref.role:
            if dim_ref.column_name not in columns:
                gaps.append(f"cannot reach {ref}")
            continue
        join_path = find_join_path(ctx, parent, dim_ref.node_name, dim_ref.role)
        if join_path:
            joins += len(join_path.links)
        elif _local_reference_dimension_column(ctx, parent, dim_ref) is None:
            gaps.append(f"cannot reach {ref}")
    if components is None:
        return joins, [*gaps, "is not decomposed"]
    measures = {
        measure_identity_token(
            compute_expression_hash(component.expression),
            component.aggregation,
            component.params,
        )
        for component in components
    }
    gaps.extend(
        f"lacks measure {measure}" for measure in sorted(required_measures - measures)
    )
    return joins, gaps


async def find_matching_source(
    ctx: BuildContext,
    metric_node: Node,
    dimensions: list[str],
    required_measures: set[str],
) -> SourceMatch:
    """Fewest-join source covering the request."""
    decomposed = await decompose_metric(
        ctx.session,
        metric_node,
        nodes_cache=ctx.nodes,
        parent_map=ctx.parent_map,
        dialect=ctx.combiner_dialect or ctx.dialect,
    )
    sources: list[tuple[int, Node | None, list[MetricComponent] | None, str | None]] = [
        (
            0,
            ctx.get_parent_node(metric_node),
            decomposed.components,
            str(decomposed.derived_ast),
        ),
    ]
    nodes_by_id = {node.id: node for node in ctx.nodes.values()}
    for source in metric_node.current.additional_sources:  # type: ignore[union-attr]
        parent_id = source.parent_node_id
        sources.append(
            (
                source.ordinal,
                nodes_by_id.get(parent_id) if parent_id else None,
                source.components,
                source.derived_expression,
            ),
        )

    candidates: list[tuple[tuple[int, int], SourceMatch]] = []
    eliminated: list[str] = []
    for ordinal, parent, components, derived_expression in sources:
        if parent is None:
            eliminated.append(f"Source {ordinal}: has no parent node.")
            continue
        joins, gaps = _source_gaps(
            ctx,
            parent,
            components,
            dimensions,
            required_measures,
        )
        if gaps or components is None:
            eliminated.append(f"Source {ordinal}: {', '.join(gaps)}.")
            continue
        match = SourceMatch(parent, ordinal, components, derived_expression)
        candidates.append(((joins, ordinal), match))

    if not candidates:
        raise DJInvalidInputException(
            " ".join(
                [f"No source of {metric_node.name} covers the request.", *eliminated],
            ),
        )
    return min(candidates, key=lambda c: c[0])[1]
