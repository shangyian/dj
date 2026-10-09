"""Tests for dimension availability per metric source."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from datajunction_server.sql.dag import get_dimension_attributes


async def post(client: AsyncClient, path: str, body: dict) -> None:
    response = await client.post(path, json=body)
    assert response.status_code in (200, 201), response.json()


async def create_source(client: AsyncClient, name: str, columns: list[str]) -> None:
    await post(
        client,
        "/nodes/source/",
        {
            "name": f"ms.{name}",
            "catalog": "default",
            "schema_": "ms",
            "table": name,
            "columns": [{"name": col, "type": "int"} for col in columns],
            "mode": "published",
        },
    )


async def create_dimension(client: AsyncClient, name: str, key: str) -> None:
    await create_source(client, f"{name}_table", [key, f"{name}_size"])
    await post(
        client,
        "/nodes/dimension/",
        {
            "name": f"ms.{name}",
            "query": f"SELECT {key}, {name}_size FROM ms.{name}_table",
            "primary_key": [key],
            "mode": "published",
        },
    )


async def link(client: AsyncClient, node: str, dimension: str, key: str) -> None:
    await post(
        client,
        f"/nodes/ms.{node}/link",
        {
            "dimension_node": f"ms.{dimension}",
            "join_type": "left",
            "join_on": f"ms.{node}.{key} = ms.{dimension}.{key}",
        },
    )


async def create_metric(client: AsyncClient, name: str, query: str, extra: list):
    await post(
        client,
        "/nodes/metric/",
        {
            "name": f"ms.{name}",
            "query": query,
            "additional_queries": extra,
            "mode": "published",
        },
    )


@pytest.fixture
async def multi_source_client(client: AsyncClient) -> AsyncClient:
    """Daily orders link customer; regional orders add region."""
    await post(client, "/namespaces/ms/", {})
    await create_dimension(client, "customer", "customer_id")
    await create_dimension(client, "region", "region_id")
    await create_source(client, "orders_daily", ["customer_id", "amount"])
    await create_source(
        client,
        "orders_by_region",
        ["customer_id", "region_id", "amount"],
    )
    await link(client, "orders_daily", "customer", "customer_id")
    await link(client, "orders_by_region", "customer", "customer_id")
    await link(client, "orders_by_region", "region", "region_id")
    await create_metric(
        client,
        "total_amount",
        "SELECT SUM(amount) FROM ms.orders_daily",
        ["SELECT SUM(amount) FROM ms.orders_by_region"],
    )
    await create_metric(
        client,
        "daily_amount",
        "SELECT SUM(amount) FROM ms.orders_daily",
        [],
    )
    return client


def attribute(
    name: str,
    path: list[str],
    source: int | None,
    filter_only: bool = False,
) -> dict:
    node_name, column = name.rsplit(".", 1)
    data = {
        "name": name,
        "node_name": node_name,
        "node_display_name": node_name.split(".")[-1].title(),
        "column_display_name": column.replace("_", " ").title(),
        "properties": ["primary_key"] if column.endswith("_id") else [],
        "type": "int",
        "path": path,
        "filter_only": filter_only,
    }
    if source is not None:
        data["source"] = source
    return data


CUSTOMER_PATH = ["ms.total_amount", "ms.customer"]
REGION_PATH = ["ms.total_amount", "ms.region"]


@pytest.mark.asyncio
async def test_each_source_tags_dimensions_it_reaches(
    multi_source_client: AsyncClient,
    session: AsyncSession,
) -> None:
    attrs = await get_dimension_attributes(session, "ms.total_amount")
    assert [(a.name, a.source) for a in attrs] == [
        ("ms.customer.customer_id", 0),
        ("ms.customer.customer_size", 0),
        ("ms.customer.customer_id", 1),
        ("ms.customer.customer_size", 1),
        ("ms.region.region_id", 1),
        ("ms.region.region_size", 1),
    ]


@pytest.mark.asyncio
async def test_endpoint_lists_dimensions_per_source(
    multi_source_client: AsyncClient,
) -> None:
    response = await multi_source_client.get("/nodes/ms.total_amount/dimensions/")
    assert response.status_code == 200
    assert response.json() == [
        attribute("ms.customer.customer_id", CUSTOMER_PATH, 0),
        attribute("ms.customer.customer_size", CUSTOMER_PATH, 0),
        attribute("ms.customer.customer_id", CUSTOMER_PATH, 1),
        attribute("ms.customer.customer_size", CUSTOMER_PATH, 1),
        attribute("ms.region.region_id", REGION_PATH, 1),
        attribute("ms.region.region_size", REGION_PATH, 1),
        attribute("ms.customer.customer_id", ["ms.orders_daily"], 0, True),
        attribute("ms.customer.customer_id", ["ms.orders_by_region"], 1, True),
        attribute("ms.region.region_id", ["ms.orders_by_region"], 1, True),
    ]


@pytest.mark.asyncio
async def test_single_source_metric_has_no_source_tag(
    multi_source_client: AsyncClient,
) -> None:
    response = await multi_source_client.get("/nodes/ms.daily_amount/dimensions/")
    assert response.status_code == 200
    path = ["ms.daily_amount", "ms.customer"]
    assert response.json() == [
        attribute("ms.customer.customer_id", path, None),
        attribute("ms.customer.customer_size", path, None),
        attribute("ms.customer.customer_id", ["ms.orders_daily"], None, True),
    ]


@pytest.mark.asyncio
async def test_shared_parent_is_tagged_per_source(
    multi_source_client: AsyncClient,
    session: AsyncSession,
) -> None:
    await create_metric(
        multi_source_client,
        "repeated_amount",
        "SELECT SUM(amount) FROM ms.orders_daily",
        [
            "SELECT MAX(amount) FROM ms.orders_by_region",
            "SELECT SUM(amount) FROM ms.orders_daily",
        ],
    )
    attrs = await get_dimension_attributes(session, "ms.repeated_amount")
    assert [(a.name, a.source) for a in attrs] == [
        ("ms.customer.customer_id", 0),
        ("ms.customer.customer_size", 0),
        ("ms.customer.customer_id", 1),
        ("ms.customer.customer_size", 1),
        ("ms.region.region_id", 1),
        ("ms.region.region_size", 1),
        ("ms.customer.customer_id", 2),
        ("ms.customer.customer_size", 2),
    ]


@pytest.mark.asyncio
async def test_derived_metric_lists_first_source_dimensions(
    multi_source_client: AsyncClient,
    session: AsyncSession,
) -> None:
    await create_metric(
        multi_source_client,
        "doubled_amount",
        "SELECT ms.total_amount * 2",
        [],
    )
    attrs = await get_dimension_attributes(session, "ms.doubled_amount")
    assert [(a.name, a.path, a.source) for a in attrs] == [
        (
            "ms.customer.customer_id",
            ["ms.doubled_amount", "ms.orders_daily"],
            None,
        ),
        (
            "ms.customer.customer_size",
            ["ms.doubled_amount", "ms.orders_daily"],
            None,
        ),
    ]


@pytest.mark.asyncio
async def test_transform_source_lists_upstream_filters(
    multi_source_client: AsyncClient,
) -> None:
    await post(
        multi_source_client,
        "/nodes/transform/",
        {
            "name": "ms.regional_orders",
            "query": "SELECT customer_id, amount FROM ms.orders_by_region",
            "mode": "published",
        },
    )
    await create_metric(
        multi_source_client,
        "unlinked_amount",
        "SELECT SUM(amount) FROM ms.orders_daily",
        ["SELECT SUM(amount) FROM ms.regional_orders"],
    )
    response = await multi_source_client.get(
        "/nodes/ms.unlinked_amount/dimensions/",
    )
    assert response.status_code == 200
    path = ["ms.unlinked_amount", "ms.customer"]
    assert response.json() == [
        attribute("ms.customer.customer_id", path, 0),
        attribute("ms.customer.customer_size", path, 0),
        attribute("ms.customer.customer_id", ["ms.orders_daily"], 0, True),
        attribute("ms.customer.customer_id", ["ms.orders_by_region"], 1, True),
        attribute("ms.region.region_id", ["ms.orders_by_region"], 1, True),
    ]
