"""Tests for source_matcher.py - picking a metric's query source."""

import pytest

from datajunction_server.construction.build_v3.source_matcher import (
    SourceMatch,
    find_matching_source,
)
from datajunction_server.construction.build_v3.types import BuildContext
from datajunction_server.database.column import Column
from datajunction_server.database.dimensionlink import DimensionLink
from datajunction_server.database.metricsource import MetricSource
from datajunction_server.database.node import Node, NodeRevision
from datajunction_server.database.preaggregation import (
    compute_expression_hash,
    measure_identity_token,
)
from datajunction_server.errors import DJInvalidInputException
from datajunction_server.models.decompose import (
    Aggregability,
    AggregationRule,
    MetricComponent,
)
from datajunction_server.models.node_type import NodeType
from datajunction_server.sql.parsing.types import IntegerType

METRIC = "test.sources.revenue"
DAILY = "test.sources.orders_daily"
REGIONAL = "test.sources.orders_by_region_daily"
CUSTOMER = "test.sources.customer"
REGION = "test.sources.region"
PRODUCT = "test.sources.product"


def make_node(
    node_id: int,
    name: str,
    node_type: NodeType,
    columns: list[str],
    query: str | None = None,
) -> Node:
    """A node with a current revision."""
    node = Node(id=node_id, name=name, type=node_type, current_version="v1.0")
    node.current = NodeRevision(
        id=node_id,
        name=name,
        type=node_type,
        version="v1.0",
        query=query,
        columns=[
            Column(name=col, type=IntegerType(), order=order)
            for order, col in enumerate(columns)
        ],
    )
    return node


def sum_of(expression: str) -> MetricComponent:
    """A SUM component over an expression."""
    return MetricComponent(
        name=f"{expression}_sum",
        expression=expression,
        aggregation="SUM",
        merge="SUM",
        rule=AggregationRule(type=Aggregability.FULL),
    )


def token(expression: str) -> str:
    """Identity token of a SUM measure."""
    return measure_identity_token(compute_expression_hash(expression), "SUM")


def link_to(ctx: BuildContext, parent: Node, *dims: Node, role: str = "") -> None:
    """Preload a join path from parent through dims."""
    links = [DimensionLink(join_sql="a = b", dimension=dim) for dim in dims]
    ctx.join_paths[(parent.current.id, dims[-1].name, role)] = links
    for dim in dims:
        ctx.nodes[dim.name] = dim


def build_ctx(
    extra_parent: str,
    components: list[MetricComponent] | None,
) -> tuple[BuildContext, Node]:
    """A metric over orders_daily plus one extra source."""
    ctx = BuildContext(session=None, metrics=[METRIC], dimensions=[])
    daily = make_node(1, DAILY, NodeType.TRANSFORM, ["customer_id", "amount"])
    regional = make_node(
        2,
        REGIONAL,
        NodeType.TRANSFORM,
        ["customer_id", "region_id", "amount"],
    )
    metric = make_node(
        3,
        METRIC,
        NodeType.METRIC,
        ["revenue"],
        query=f"SELECT SUM(amount) FROM {DAILY}",
    )
    parent = {DAILY: daily, REGIONAL: regional}[extra_parent]
    metric.current.additional_sources = [
        MetricSource(
            ordinal=1,
            query=f"SELECT SUM(amount) FROM {extra_parent}",
            parent_node_id=parent.id,
            components=components,
            derived_expression="SELECT amount_sum",
        ),
    ]
    for node in (daily, regional, metric):
        ctx.nodes[node.name] = node
    ctx.parent_map[METRIC] = [DAILY]
    customer = make_node(10, CUSTOMER, NodeType.DIMENSION, ["customer_id", "name"])
    region = make_node(11, REGION, NodeType.DIMENSION, ["region_id", "name"])
    ctx.nodes[PRODUCT] = make_node(12, PRODUCT, NodeType.DIMENSION, ["name"])
    link_to(ctx, daily, customer)
    link_to(ctx, daily, customer, region)
    link_to(ctx, regional, customer)
    link_to(ctx, regional, region)
    return ctx, metric


@pytest.mark.asyncio
async def test_picks_source_reaching_dimension():
    """Source 1 reaches region in fewer joins than source 0."""
    ctx, metric = build_ctx(REGIONAL, [sum_of("amount")])
    match = await find_matching_source(ctx, metric, [f"{REGION}.name"], set())
    assert match == SourceMatch(
        parent_node=ctx.nodes[REGIONAL],
        ordinal=1,
        components=[sum_of("amount")],
        derived_expression="SELECT amount_sum",
    )


@pytest.mark.asyncio
async def test_ties_break_by_declaration_order():
    """Equal joins pick source 0, including its decomposition."""
    ctx, metric = build_ctx(REGIONAL, [sum_of("amount")])
    match = await find_matching_source(
        ctx,
        metric,
        [f"{CUSTOMER}.name"],
        {token("amount")},
    )
    assert match == SourceMatch(
        parent_node=ctx.nodes[DAILY],
        ordinal=0,
        components=[
            MetricComponent(
                name="amount_sum_58a7a4e7",
                expression="amount",
                aggregation="SUM",
                merge="SUM",
                rule=AggregationRule(type=Aggregability.FULL),
            ),
        ],
        derived_expression=f"SELECT  SUM(amount_sum_58a7a4e7) \n FROM {DAILY}\n\n",
    )


@pytest.mark.asyncio
async def test_shared_parent_is_separate_source():
    """A source sharing source 0's parent can still win on measures."""
    ctx, metric = build_ctx(DAILY, [sum_of("amount * 2")])
    match = await find_matching_source(
        ctx,
        metric,
        [f"{CUSTOMER}.name"],
        {token("amount * 2")},
    )
    assert (match.parent_node, match.ordinal) == (ctx.nodes[DAILY], 1)


@pytest.mark.asyncio
async def test_raises_when_no_source_covers():
    """Each eliminated source names what it lacks."""
    ctx, metric = build_ctx(REGIONAL, [sum_of("amount")])
    with pytest.raises(DJInvalidInputException) as exc_info:
        await find_matching_source(
            ctx,
            metric,
            [f"{PRODUCT}.name", f"{REGIONAL}.region_id"],
            {token("amount"), token("price")},
        )
    assert exc_info.value.message == (
        f"No source of {METRIC} covers the request. "
        f"Source 0: cannot reach {PRODUCT}.name, "
        f"cannot reach {REGIONAL}.region_id, "
        f"lacks measure {token('price')}. "
        f"Source 1: cannot reach {PRODUCT}.name, "
        f"lacks measure {token('price')}."
    )


@pytest.mark.asyncio
async def test_undecomposed_or_orphaned_source_is_skipped():
    """NULL components or a missing parent rule a source out."""
    ctx, metric = build_ctx(REGIONAL, None)
    metric.current.additional_sources.append(
        MetricSource(ordinal=2, query="SELECT 1", parent_node_id=None),
    )
    with pytest.raises(DJInvalidInputException) as exc_info:
        await find_matching_source(ctx, metric, [f"{REGIONAL}.region_id"], set())
    assert exc_info.value.message == (
        f"No source of {METRIC} covers the request. "
        f"Source 0: cannot reach {REGIONAL}.region_id. "
        f"Source 1: is not decomposed. "
        f"Source 2: has no parent node."
    )


@pytest.mark.asyncio
async def test_reference_dimension_needs_no_join():
    """A column tagged with a dimension reaches it."""
    ctx, metric = build_ctx(REGIONAL, [sum_of("amount")])
    ctx.nodes[DAILY].current.columns.append(
        Column(
            name="product_name",
            type=IntegerType(),
            order=2,
            dimension_id=12,
            dimension_column="name",
        ),
    )
    ctx.reference_dimension_names[12] = PRODUCT
    match = await find_matching_source(ctx, metric, [f"{PRODUCT}.name"], set())
    assert (match.parent_node, match.ordinal) == (ctx.nodes[DAILY], 0)


@pytest.mark.asyncio
async def test_missing_local_column_is_unreachable():
    """A parent's own column must exist."""
    ctx, metric = build_ctx(REGIONAL, [sum_of("amount")])
    with pytest.raises(DJInvalidInputException) as exc_info:
        await find_matching_source(ctx, metric, [f"{REGIONAL}.nope"], set())
    assert exc_info.value.message == (
        f"No source of {METRIC} covers the request. "
        f"Source 0: cannot reach {REGIONAL}.nope. "
        f"Source 1: cannot reach {REGIONAL}.nope."
    )


def swap_region_roles(ctx: BuildContext) -> None:
    """Source 0 links region as home, source 1 as shipping."""
    daily, regional = ctx.nodes[DAILY], ctx.nodes[REGIONAL]
    del ctx.join_paths[(daily.current.id, REGION, "")]
    del ctx.join_paths[(regional.current.id, REGION, "")]
    customer, region = ctx.nodes[CUSTOMER], ctx.nodes[REGION]
    link_to(ctx, daily, customer, region, role="home")
    link_to(ctx, regional, region, role="shipping")


@pytest.mark.asyncio
async def test_role_routes_to_source_with_role():
    """The source linking the role wins despite more joins."""
    ctx, metric = build_ctx(REGIONAL, [sum_of("amount")])
    swap_region_roles(ctx)
    match = await find_matching_source(ctx, metric, [f"{REGION}.name[home]"], set())
    assert (match.parent_node, match.ordinal) == (ctx.nodes[DAILY], 0)


@pytest.mark.asyncio
async def test_unlinked_role_is_unreachable():
    """A role no source links names every source."""
    ctx, metric = build_ctx(REGIONAL, [sum_of("amount")])
    swap_region_roles(ctx)
    with pytest.raises(DJInvalidInputException) as exc_info:
        await find_matching_source(ctx, metric, [f"{REGION}.name[billing]"], set())
    assert exc_info.value.message == (
        f"No source of {METRIC} covers the request. "
        f"Source 0: cannot reach {REGION}.name[billing]. "
        f"Source 1: cannot reach {REGION}.name[billing]."
    )


@pytest.mark.asyncio
async def test_role_free_request_takes_any_role():
    """A bare reference still picks the fewest joins."""
    ctx, metric = build_ctx(REGIONAL, [sum_of("amount")])
    swap_region_roles(ctx)
    match = await find_matching_source(ctx, metric, [f"{REGION}.name"], set())
    assert (match.parent_node, match.ordinal) == (ctx.nodes[REGIONAL], 1)
