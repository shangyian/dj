"""Tests for checking a metric's dimension compatibility."""

import pytest
from httpx import AsyncClient

from tests.sql.dag_multi_source_dimensions_test import (  # noqa: F401
    multi_source_client,
)


async def check(client: AsyncClient, metric: str, dimensions: list[str]):
    response = await client.post(
        f"/metrics/{metric}/compatible-dimensions/",
        json={"dimensions": dimensions},
    )
    return response.status_code, response.json()


@pytest.mark.asyncio
async def test_tie_picks_first_source(multi_source_client):
    assert await check(
        multi_source_client,
        "ms.total_amount",
        ["ms.customer.customer_size"],
    ) == (200, {"compatible": True, "source": 0, "reason": None})


@pytest.mark.asyncio
async def test_picks_source_reaching_dimension(multi_source_client):
    assert await check(
        multi_source_client,
        "ms.total_amount",
        ["ms.customer.customer_size", "ms.region.region_size"],
    ) == (200, {"compatible": True, "source": 1, "reason": None})


@pytest.mark.asyncio
async def test_no_source_reaches_dimension(multi_source_client):
    assert await check(
        multi_source_client,
        "ms.total_amount",
        ["ms.region.region_size[home]"],
    ) == (
        200,
        {
            "compatible": False,
            "source": None,
            "reason": (
                "No source of ms.total_amount covers the request. "
                "Source 0: cannot reach ms.region.region_size[home]. "
                "Source 1: cannot reach ms.region.region_size[home]."
            ),
        },
    )


@pytest.mark.asyncio
async def test_unknown_column_first_source(multi_source_client):
    assert await check(
        multi_source_client, "ms.total_amount", ["ms.customer.nope"],
    ) == (
        200,
        {
            "compatible": False,
            "source": None,
            "reason": (
                "Column `nope` does not exist on node `ms.customer` "
                "(referenced as `ms.customer.nope`)."
            ),
        },
    )


@pytest.mark.asyncio
async def test_unknown_column_second_source(multi_source_client):
    assert await check(
        multi_source_client,
        "ms.total_amount",
        ["ms.customer.customer_size", "ms.region.nope"],
    ) == (
        200,
        {
            "compatible": False,
            "source": None,
            "reason": (
                "Column `nope` does not exist on node `ms.region` "
                "(referenced as `ms.region.nope`)."
            ),
        },
    )


@pytest.mark.asyncio
async def test_unknown_dimension_node(multi_source_client):
    assert await check(multi_source_client, "ms.total_amount", ["ms.nope.x"]) == (
        200,
        {
            "compatible": False,
            "source": None,
            "reason": (
                "No source of ms.total_amount covers the request. "
                "Source 0: cannot reach ms.nope.x. "
                "Source 1: cannot reach ms.nope.x."
            ),
        },
    )


@pytest.mark.asyncio
async def test_single_source_metric(multi_source_client):
    assert [
        await check(multi_source_client, "ms.daily_amount", dimensions)
        for dimensions in (
            [],
            ["ms.customer.customer_size"],
            ["ms.orders_daily.amount"],
            ["ms.region.region_size"],
        )
    ] == [
        (200, {"compatible": True, "source": 0, "reason": None}),
        (200, {"compatible": True, "source": 0, "reason": None}),
        (200, {"compatible": True, "source": 0, "reason": None}),
        (
            200,
            {
                "compatible": False,
                "source": None,
                "reason": (
                    "No source of ms.daily_amount covers the request. "
                    "Source 0: cannot reach ms.region.region_size."
                ),
            },
        ),
    ]


@pytest.mark.asyncio
async def test_malformed_dimension(multi_source_client):
    message = (
        "Reference `region_size` is not fully qualified. Use the "
        "`node.column` form (e.g., `v3.order_details.status`) so DJ "
        "can route the reference to the correct node."
    )
    assert await check(multi_source_client, "ms.total_amount", ["region_size"]) == (
        422,
        {"message": message, "errors": [], "warnings": []},
    )


@pytest.mark.asyncio
async def test_not_a_metric(multi_source_client):
    assert await check(multi_source_client, "ms.customer", []) == (
        400,
        {"detail": "Not a metric node: `ms.customer`"},
    )


@pytest.mark.asyncio
async def test_missing_metric(multi_source_client):
    assert await check(multi_source_client, "ms.nope", []) == (
        404,
        {
            "message": "A node with name `ms.nope` does not exist.",
            "errors": [],
            "warnings": [],
        },
    )


@pytest.mark.asyncio
async def test_derived_metric_rejected(multi_source_client):
    response = await multi_source_client.post(
        "/nodes/metric/",
        json={
            "name": "ms.double_amount",
            "query": "SELECT ms.total_amount * 2",
            "mode": "published",
        },
    )
    assert response.status_code == 201, response.json()
    assert await check(multi_source_client, "ms.double_amount", []) == (
        422,
        {
            "message": "Cannot check derived metric ms.double_amount.",
            "errors": [],
            "warnings": [],
        },
    )


@pytest.mark.asyncio
async def test_agrees_with_v3_sql(multi_source_client):
    requests = [
        ("ms.total_amount", ["ms.customer.customer_size"]),
        ("ms.total_amount", ["ms.region.region_size"]),
        ("ms.total_amount", ["ms.region.region_size[home]"]),
        ("ms.daily_amount", ["ms.customer.customer_size"]),
        ("ms.daily_amount", ["ms.region.region_size"]),
        ("ms.total_amount", ["ms.customer.nope"]),
        ("ms.total_amount", ["ms.customer.customer_size", "ms.region.nope"]),
        ("ms.daily_amount", ["ms.customer.nope"]),
    ]
    answers = []
    for metric, dimensions in requests:
        _, body = await check(multi_source_client, metric, dimensions)
        response = await multi_source_client.get(
            "/sql/metrics/v3/",
            params={"metrics": [metric], "dimensions": dimensions},
        )
        answers.append((body["compatible"], response.status_code))
    assert answers == [
        (True, 200),
        (True, 200),
        (False, 422),
        (True, 200),
        (False, 422),
        (False, 422),
        (False, 422),
        (False, 422),
    ]
