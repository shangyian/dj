"""Tests that v2 SQL builds reject multi-source metrics."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from datajunction_server.errors import DJInvalidInputException
from datajunction_server.internal.sql import (
    build_sql_for_multiple_metrics,
    get_measures_query,
)
from tests.construction.build_v3.multi_source_routing_integration_test import (
    create_multi_source_metric,
)

METRICS = ["v3.multi_source_quantity"]
DIMENSIONS = ["v3.product.category"]
MESSAGE = "Multi-source metrics require the v3 SQL build."


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/sql/", "/sql/measures/v2/"])
async def test_v2_endpoint_rejects_multi_source(client_with_build_v3, path):
    await create_multi_source_metric(client_with_build_v3)
    response = await client_with_build_v3.get(
        path,
        params={"metrics": METRICS, "dimensions": DIMENSIONS},
    )
    assert (response.status_code, response.json()["message"]) == (422, MESSAGE)


@pytest.mark.asyncio
async def test_v2_builders_reject_multi_source(
    client_with_build_v3,
    session: AsyncSession,
):
    """Materialization builds call these directly."""
    await create_multi_source_metric(client_with_build_v3)
    with pytest.raises(DJInvalidInputException) as metrics_error:
        await build_sql_for_multiple_metrics(session, METRICS, DIMENSIONS)
    with pytest.raises(DJInvalidInputException) as measures_error:
        await get_measures_query(session, METRICS, list(DIMENSIONS), [])
    assert (metrics_error.value.message, measures_error.value.message) == (
        MESSAGE,
        MESSAGE,
    )
