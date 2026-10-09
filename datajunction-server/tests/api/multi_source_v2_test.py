"""Tests that v2 SQL builds reject multi-source metrics."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from datajunction_server.errors import DJInvalidInputException
from datajunction_server.internal.sql import (
    build_sql_for_multiple_metrics,
    get_measures_query,
)
from tests.construction.build_v3.multi_source_routing_integration_test import (
    create_multi_source_metric,
)

DIMENSIONS = ["v3.product.category"]
MESSAGE = "Multi-source metrics require the v3 SQL build."
V2_PATHS = ["/sql/", "/sql/measures/v2/"]
MULTI_SOURCE = [
    "v3.multi_source_quantity",
    "v3.multi_source_doubled",
    "v3.multi_source_quadrupled",
]


async def create_derived_chain(client: AsyncClient) -> None:
    """Derived metrics over a multi-source base."""
    await create_multi_source_metric(client)
    for name, query in [
        ("v3.multi_source_doubled", "SELECT v3.multi_source_quantity * 2"),
        ("v3.multi_source_quadrupled", "SELECT v3.multi_source_doubled * 2"),
        ("v3.revenue_doubled", "SELECT v3.total_revenue * 2"),
    ]:
        response = await client.post(
            "/nodes/metric/",
            json={"name": name, "query": query, "mode": "published"},
        )
        assert response.status_code == 201, response.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", V2_PATHS)
@pytest.mark.parametrize("metric", MULTI_SOURCE)
async def test_v2_endpoint_rejects_multi_source(client_with_build_v3, path, metric):
    await create_derived_chain(client_with_build_v3)
    response = await client_with_build_v3.get(
        path,
        params={"metrics": [metric], "dimensions": DIMENSIONS},
    )
    assert (response.status_code, response.json()["message"]) == (422, MESSAGE)


@pytest.mark.asyncio
@pytest.mark.parametrize("metric", MULTI_SOURCE)
async def test_v2_builders_reject_multi_source(
    client_with_build_v3,
    session: AsyncSession,
    metric,
):
    """Materialization builds call these directly."""
    await create_derived_chain(client_with_build_v3)
    with pytest.raises(DJInvalidInputException) as metrics_error:
        await build_sql_for_multiple_metrics(session, [metric], DIMENSIONS)
    with pytest.raises(DJInvalidInputException) as measures_error:
        await get_measures_query(session, [metric], list(DIMENSIONS), [])
    assert (metrics_error.value.message, measures_error.value.message) == (
        MESSAGE,
        MESSAGE,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("path", V2_PATHS)
@pytest.mark.parametrize("metric", ["v3.total_revenue", "v3.revenue_doubled"])
async def test_v2_keeps_single_source(client_with_build_v3, path, metric):
    await create_derived_chain(client_with_build_v3)
    response = await client_with_build_v3.get(
        path,
        params={"metrics": [metric], "dimensions": DIMENSIONS},
    )
    assert response.status_code == 200, response.json()
