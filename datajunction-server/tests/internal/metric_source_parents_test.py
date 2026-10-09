"""Tests for resolving parents of additional metric sources."""

import pytest
from httpx import AsyncClient

BASE_QUERY = "SELECT COUNT(repair_order_id) FROM default.repair_orders"


async def post_metric(client: AsyncClient, additional_queries: list[str]):
    return await client.post(
        "/nodes/metric/",
        json={
            "name": "default.multi_source_order_count",
            "query": BASE_QUERY,
            "additional_queries": additional_queries,
        },
    )


@pytest.mark.asyncio
async def test_additional_source_resolves_its_own_parent_node(
    client_with_roads: AsyncClient,
) -> None:
    details_query = "SELECT COUNT(repair_order_id) FROM default.repair_order_details"
    fact_query = "SELECT COUNT(repair_order_id) FROM default.repair_orders_fact"
    response = await post_metric(client_with_roads, [details_query, fact_query])
    assert response.status_code == 201

    node_response = await client_with_roads.get(
        "/nodes/default.multi_source_order_count/",
    )
    body = node_response.json()
    assert body["query"] == BASE_QUERY
    assert body["parents"] == [{"name": "default.repair_orders"}]
    assert body["additional_sources"] == [
        {
            "ordinal": 1,
            "query": details_query,
            "parent_node": {"name": "default.repair_order_details"},
        },
        {
            "ordinal": 2,
            "query": fact_query,
            "parent_node": {"name": "default.repair_orders_fact"},
        },
    ]


@pytest.mark.asyncio
async def test_single_source_metric_has_no_additional_sources(
    client_with_roads: AsyncClient,
) -> None:
    response = await client_with_roads.get("/nodes/default.num_repair_orders/")
    assert response.json()["additional_sources"] == []


@pytest.mark.asyncio
async def test_additional_source_with_unresolvable_parent_is_rejected(
    client_with_roads: AsyncClient,
) -> None:
    response = await post_metric(
        client_with_roads,
        ["SELECT SUM(x) FROM default.nonexistent_table"],
    )
    assert response.status_code == 422
    body = response.json()
    assert body["message"] == "Additional query 1 is invalid."
    assert [error["code"] for error in body["errors"]] == [
        "MISSING_PARENT",
        "TYPE_INFERENCE",
    ]
    assert body["errors"][0] == {
        "code": "MISSING_PARENT",
        "message": (
            "Node definition contains references to nodes that do not exist: "
            "default.nonexistent_table"
        ),
        "debug": {"missing_parents": ["default.nonexistent_table"]},
        "context": "",
    }


@pytest.mark.asyncio
async def test_additional_source_with_missing_column_is_rejected(
    client_with_roads: AsyncClient,
) -> None:
    response = await post_metric(
        client_with_roads,
        ["SELECT SUM(nope) FROM default.repair_order_details"],
    )
    assert response.status_code == 422
    assert response.json() == {
        "message": "Additional query 1 is invalid.",
        "errors": [
            {
                "code": "MISSING_COLUMNS",
                "message": "Metric definition references missing columns: nope",
                "debug": None,
                "context": "",
            },
            {
                "code": "TYPE_INFERENCE",
                "message": "Cannot resolve type of column `nope` in SUM(nope)",
                "debug": {
                    "columns": ["default_DOT_multi_source_order_count"],
                    "errors": [
                        {
                            "code": "INVALID_COLUMN",
                            "message": "Column `nope` does not exist on any valid table.",
                            "debug": {"namespace": ""},
                            "context": "",
                        },
                    ],
                },
                "context": "",
            },
        ],
        "warnings": [],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "SELECT COUNT(o.repair_order_id) FROM default.repair_orders o "
        "JOIN default.repair_order_details d "
        "ON o.repair_order_id = d.repair_order_id",
        "SELECT default.num_repair_orders * 2",
    ],
)
async def test_additional_source_needs_one_non_metric_parent(
    client_with_roads: AsyncClient,
    query: str,
) -> None:
    response = await post_metric(client_with_roads, [query])
    assert response.status_code == 422
    assert response.json() == {
        "message": "Additional query 1 must reference exactly one non-metric node.",
        "errors": [],
        "warnings": [],
    }


@pytest.mark.asyncio
async def test_non_metric_rejects_additional_queries(
    client_with_roads: AsyncClient,
) -> None:
    response = await client_with_roads.post(
        "/nodes/transform/",
        json={
            "name": "default.multi_source_transform",
            "query": "SELECT repair_order_id FROM default.repair_orders",
            "additional_queries": ["SELECT 1"],
        },
    )
    assert response.status_code == 422
    assert response.json() == {
        "message": "Only metric nodes can have additional queries.",
        "errors": [],
        "warnings": [],
    }


DETAILS_QUERY = "SELECT COUNT(repair_order_id) FROM default.repair_order_details"
FACT_QUERY = "SELECT COUNT(repair_order_id) FROM default.repair_orders_fact"
DETAILS_SOURCE = {
    "ordinal": 1,
    "query": DETAILS_QUERY,
    "parent_node": {"name": "default.repair_order_details"},
}
METRIC_NAME = "default.multi_source_order_count"


async def get_sources(client: AsyncClient, name: str = METRIC_NAME):
    body = (await client.get(f"/nodes/{name}/")).json()
    return body["version"], body["additional_sources"]


@pytest.mark.asyncio
async def test_unrelated_patch_keeps_sources(
    client_with_roads: AsyncClient,
) -> None:
    await post_metric(client_with_roads, [DETAILS_QUERY])
    response = await client_with_roads.patch(
        f"/nodes/{METRIC_NAME}/",
        json={"description": "New description"},
    )
    assert response.status_code == 200
    assert await get_sources(client_with_roads) == ("v1.1", [DETAILS_SOURCE])


@pytest.mark.asyncio
async def test_patch_replaces_sources_and_bumps_version(
    client_with_roads: AsyncClient,
) -> None:
    await post_metric(client_with_roads, [DETAILS_QUERY])
    response = await client_with_roads.patch(
        f"/nodes/{METRIC_NAME}/",
        json={"additional_queries": [FACT_QUERY]},
    )
    assert response.status_code == 200
    assert await get_sources(client_with_roads) == (
        "v2.0",
        [
            {
                "ordinal": 1,
                "query": FACT_QUERY,
                "parent_node": {"name": "default.repair_orders_fact"},
            },
        ],
    )


@pytest.mark.asyncio
async def test_patch_with_empty_list_clears_sources(
    client_with_roads: AsyncClient,
) -> None:
    await post_metric(client_with_roads, [DETAILS_QUERY])
    response = await client_with_roads.patch(
        f"/nodes/{METRIC_NAME}/",
        json={"additional_queries": []},
    )
    assert response.status_code == 200
    assert await get_sources(client_with_roads) == ("v2.0", [])


@pytest.mark.asyncio
async def test_patch_with_same_sources_keeps_version(
    client_with_roads: AsyncClient,
) -> None:
    await post_metric(client_with_roads, [DETAILS_QUERY])
    response = await client_with_roads.patch(
        f"/nodes/{METRIC_NAME}/",
        json={"additional_queries": [DETAILS_QUERY]},
    )
    assert response.status_code == 200
    assert await get_sources(client_with_roads) == ("v1.0", [DETAILS_SOURCE])


@pytest.mark.asyncio
async def test_patch_rejects_invalid_source(
    client_with_roads: AsyncClient,
) -> None:
    await post_metric(client_with_roads, [DETAILS_QUERY])
    response = await client_with_roads.patch(
        f"/nodes/{METRIC_NAME}/",
        json={"additional_queries": ["SELECT default.num_repair_orders * 2"]},
    )
    assert response.status_code == 422
    assert response.json() == {
        "message": "Additional query 1 must reference exactly one non-metric node.",
        "errors": [],
        "warnings": [],
    }
    assert await get_sources(client_with_roads) == ("v1.0", [DETAILS_SOURCE])


@pytest.mark.asyncio
async def test_patch_non_metric_rejects_additional_queries(
    client_with_roads: AsyncClient,
) -> None:
    response = await client_with_roads.patch(
        "/nodes/default.repair_orders_fact/",
        json={"additional_queries": [DETAILS_QUERY]},
    )
    assert response.status_code == 422
    assert response.json() == {
        "message": "Only metric nodes can have additional queries.",
        "errors": [],
        "warnings": [],
    }


@pytest.mark.asyncio
async def test_copy_keeps_sources(
    client_with_roads: AsyncClient,
) -> None:
    await post_metric(client_with_roads, [DETAILS_QUERY])
    response = await client_with_roads.post(
        f"/nodes/{METRIC_NAME}/copy",
        params={"new_name": "default.copied_order_count"},
    )
    assert response.status_code == 200
    assert await get_sources(client_with_roads, "default.copied_order_count") == (
        "v1.0",
        [DETAILS_SOURCE],
    )


def amounts_source(amount_type: str) -> dict:
    return {
        "name": "default.multi_source_amounts",
        "catalog": "default",
        "schema_": "roads",
        "table": "multi_source_amounts",
        "mode": "published",
        "columns": [{"name": "amount", "type": amount_type}],
    }


@pytest.mark.asyncio
async def test_revalidate_after_column_change_keeps_sources(
    client_with_roads: AsyncClient,
) -> None:
    response = await client_with_roads.post(
        "/nodes/source/",
        json=amounts_source("int"),
    )
    assert response.status_code == 200
    response = await client_with_roads.post(
        "/nodes/metric/",
        json={
            "name": METRIC_NAME,
            "query": "SELECT SUM(amount) FROM default.multi_source_amounts",
            "additional_queries": [DETAILS_QUERY],
        },
    )
    assert response.status_code == 201
    assert await get_sources(client_with_roads) == ("v1.0", [DETAILS_SOURCE])

    response = await client_with_roads.patch(
        "/nodes/default.multi_source_amounts/",
        json={"columns": amounts_source("double")["columns"]},
    )
    assert response.status_code == 200
    response = await client_with_roads.post(f"/nodes/{METRIC_NAME}/validate/")
    assert response.status_code == 200
    body = (await client_with_roads.get(f"/nodes/{METRIC_NAME}/")).json()
    assert body["columns"][0]["type"] == "double"
    assert (body["version"], body["additional_sources"]) == (
        "v2.0",
        [DETAILS_SOURCE],
    )


@pytest.mark.asyncio
async def test_recreate_inactive_metric_uses_new_sources(
    client_with_roads: AsyncClient,
) -> None:
    await post_metric(client_with_roads, [DETAILS_QUERY])
    response = await client_with_roads.delete(f"/nodes/{METRIC_NAME}/")
    assert response.status_code == 200
    response = await post_metric(client_with_roads, [FACT_QUERY])
    assert response.status_code == 201
    assert (await get_sources(client_with_roads))[1] == [
        {
            "ordinal": 1,
            "query": FACT_QUERY,
            "parent_node": {"name": "default.repair_orders_fact"},
        },
    ]
