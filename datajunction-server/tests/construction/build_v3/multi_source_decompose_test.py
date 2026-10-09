"""Tests for decomposing additional metric sources."""

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from datajunction_server.database.metricsource import MetricSource
from datajunction_server.database.node import Node, NodeRevision
from datajunction_server.models.decompose import (
    Aggregability,
    AggregationRule,
    MetricComponent,
)
from datajunction_server.sql.decompose import MetricComponentExtractor

METRIC_NAME = "default.multi_source_revenue"
BASE_QUERY = "SELECT SUM(price) FROM default.repair_order_details"
AVG_QUERY = "SELECT AVG(price) FROM default.repair_order_details"
COUNT_QUERY = "SELECT COUNT(repair_order_id) FROM default.repair_orders_fact"

AVG_COMPONENTS = [
    MetricComponent(
        name="price_count_252381cf",
        expression="price",
        aggregation="COUNT",
        merge="SUM",
        rule=AggregationRule(type=Aggregability.FULL),
    ),
    MetricComponent(
        name="price_sum_252381cf",
        expression="price",
        aggregation="SUM",
        merge="SUM",
        rule=AggregationRule(type=Aggregability.FULL),
    ),
]
AVG_DERIVED = (
    "SELECT  SUM(price_sum_252381cf) / NULLIF(SUM(price_count_252381cf), 0) \n"
    " FROM default.repair_order_details\n\n"
)
COUNT_COMPONENTS = [
    MetricComponent(
        name="repair_order_id_count_bd241964",
        expression="repair_order_id",
        aggregation="COUNT",
        merge="SUM",
        rule=AggregationRule(type=Aggregability.FULL),
    ),
]
COUNT_DERIVED = (
    "SELECT  SUM(repair_order_id_count_bd241964) \n FROM default.repair_orders_fact\n\n"
)


async def create_metric(client: AsyncClient, name: str, **extra) -> None:
    response = await client.post(
        "/nodes/metric/",
        json={"name": name, "query": BASE_QUERY, **extra},
    )
    assert response.status_code == 201


async def current_sources(session: AsyncSession) -> list[MetricSource]:
    session.expire_all()
    return list(
        (
            await session.execute(
                select(MetricSource)
                .options(joinedload(MetricSource.node_revision))
                .join(NodeRevision, MetricSource.node_revision_id == NodeRevision.id)
                .join(Node, Node.id == NodeRevision.node_id)
                .where(
                    Node.name == METRIC_NAME,
                    Node.current_version == NodeRevision.version,
                )
                .order_by(MetricSource.ordinal),
            )
        ).scalars(),
    )


def decomposition(sources: list[MetricSource]) -> list[tuple]:
    return [(s.ordinal, s.components, s.derived_expression) for s in sources]


@pytest.mark.asyncio
async def test_source_decomposes_like_plain_metric(
    client_with_roads: AsyncClient,
    session: AsyncSession,
) -> None:
    await create_metric(client_with_roads, METRIC_NAME, additional_queries=[AVG_QUERY])
    await create_metric(client_with_roads, "default.plain_avg_price")
    response = await client_with_roads.patch(
        "/nodes/default.plain_avg_price/",
        json={"query": AVG_QUERY},
    )
    assert response.status_code == 200
    plain = await Node.get_by_name(session, "default.plain_avg_price")
    plain_components, plain_derived = await MetricComponentExtractor(
        plain.current.id,  # type: ignore
    ).extract(session, validate_fixed_grain_windows=False)

    sources = await current_sources(session)
    assert decomposition(sources) == [(1, AVG_COMPONENTS, AVG_DERIVED)]
    assert decomposition(sources) == [(1, plain_components, str(plain_derived))]


@pytest.mark.asyncio
async def test_patch_refreshes_source_decomposition(
    client_with_roads: AsyncClient,
    session: AsyncSession,
) -> None:
    await create_metric(client_with_roads, METRIC_NAME, additional_queries=[AVG_QUERY])
    response = await client_with_roads.patch(
        f"/nodes/{METRIC_NAME}/",
        json={"additional_queries": [COUNT_QUERY, AVG_QUERY]},
    )
    assert response.status_code == 200
    assert decomposition(await current_sources(session)) == [
        (1, COUNT_COMPONENTS, COUNT_DERIVED),
        (2, AVG_COMPONENTS, AVG_DERIVED),
    ]


@pytest.mark.asyncio
async def test_unrelated_patch_keeps_source_decomposition(
    client_with_roads: AsyncClient,
    session: AsyncSession,
) -> None:
    await create_metric(client_with_roads, METRIC_NAME, additional_queries=[AVG_QUERY])
    response = await client_with_roads.patch(
        f"/nodes/{METRIC_NAME}/",
        json={"query": "SELECT MAX(price) FROM default.repair_order_details"},
    )
    assert response.status_code == 200
    sources = await current_sources(session)
    assert [s.node_revision.version for s in sources] == ["v2.0"]
    assert decomposition(sources) == [(1, AVG_COMPONENTS, AVG_DERIVED)]


FACT_SUM_QUERY = "SELECT SUM(price) FROM default.repair_orders_fact"
REAGGREGATE = {
    "rules": [
        {
            "dimension": "default.repair_orders_fact.repair_order_id",
            "fn": "last_value",
        },
    ],
}
METRIC_SETTINGS = [{"reaggregate": REAGGREGATE}, {"fixed_grain": []}]


async def plain_decomposition(
    client: AsyncClient,
    session: AsyncSession,
    settings: dict,
) -> list[tuple]:
    response = await client.post(
        "/nodes/metric/",
        json={"name": "default.plain_fact_price", "query": FACT_SUM_QUERY, **settings},
    )
    assert response.status_code == 201
    plain = await Node.get_by_name(session, "default.plain_fact_price")
    components, derived = await MetricComponentExtractor(
        plain.current.id,  # type: ignore
    ).extract(session, validate_fixed_grain_windows=False)
    return [(1, components, str(derived))]


def source_rules(sources: list[MetricSource]) -> list[list[AggregationRule]]:
    return [[c.rule for c in s.components or []] for s in sources]


EXPECTED_RULES = {
    "reaggregate": [
        [
            AggregationRule(
                type=Aggregability.FULL,
                reaggregate=REAGGREGATE["rules"][0],
            ),
        ],
    ],
    "fixed_grain": [[AggregationRule(type=Aggregability.FULL, fixed_grain=[])]],
}


@pytest.mark.asyncio
@pytest.mark.parametrize("settings", METRIC_SETTINGS)
async def test_source_uses_metric_settings(
    client_with_roads: AsyncClient,
    session: AsyncSession,
    settings: dict,
) -> None:
    response = await client_with_roads.post(
        "/nodes/metric/",
        json={
            "name": METRIC_NAME,
            "query": COUNT_QUERY,
            "additional_queries": [FACT_SUM_QUERY],
            **settings,
        },
    )
    assert response.status_code == 201
    sources = await current_sources(session)
    assert source_rules(sources) == EXPECTED_RULES[next(iter(settings))]
    assert decomposition(sources) == await plain_decomposition(
        client_with_roads,
        session,
        settings,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("settings", METRIC_SETTINGS)
async def test_settings_patch_refreshes_sources(
    client_with_roads: AsyncClient,
    session: AsyncSession,
    settings: dict,
) -> None:
    response = await client_with_roads.post(
        "/nodes/metric/",
        json={
            "name": METRIC_NAME,
            "query": COUNT_QUERY,
            "additional_queries": [FACT_SUM_QUERY],
        },
    )
    assert response.status_code == 201
    response = await client_with_roads.patch(f"/nodes/{METRIC_NAME}/", json=settings)
    assert response.status_code == 200
    sources = await current_sources(session)
    assert [s.node_revision.version for s in sources] == ["v2.0"]
    assert source_rules(sources) == EXPECTED_RULES[next(iter(settings))]
    assert decomposition(sources) == await plain_decomposition(
        client_with_roads,
        session,
        settings,
    )
