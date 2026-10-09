"""Tests for routing a metric request to one of its sources."""

import pytest
from httpx import AsyncClient

from . import assert_sql_equal

ORDER_LINES_CTE = """
v3_order_lines AS (
    SELECT oi.product_id, oi.quantity
    FROM default.v3.order_items oi
)
"""


async def create_multi_source_metric(client: AsyncClient) -> None:
    """Order lines link only product; order details link more."""
    response = await client.post(
        "/nodes/transform/",
        json={
            "name": "v3.order_lines",
            "description": "Order lines without order headers",
            "query": (
                "SELECT oi.order_id, oi.line_number, oi.product_id, oi.quantity "
                "FROM v3.src_order_items oi"
            ),
            "mode": "published",
            "primary_key": ["order_id", "line_number"],
        },
    )
    assert response.status_code == 201, response.json()
    response = await client.post(
        "/nodes/v3.order_lines/link",
        json={
            "dimension_node": "v3.product",
            "join_type": "left",
            "join_on": "v3.order_lines.product_id = v3.product.product_id",
        },
    )
    assert response.status_code == 201, response.json()
    response = await client.post(
        "/nodes/metric/",
        json={
            "name": "v3.multi_source_quantity",
            "query": "SELECT SUM(quantity) FROM v3.order_lines",
            "additional_queries": ["SELECT SUM(quantity) FROM v3.order_details"],
            "mode": "published",
        },
    )
    assert response.status_code == 201, response.json()


async def metrics_sql(client: AsyncClient, **params) -> dict:
    response = await client.get("/sql/metrics/v3/", params=params)
    assert response.status_code == 200, response.json()
    return response.json()


@pytest.mark.asyncio
async def test_routes_to_source_reaching_dimension(client_with_build_v3):
    await create_multi_source_metric(client_with_build_v3)
    result = await metrics_sql(
        client_with_build_v3,
        metrics=["v3.multi_source_quantity"],
        dimensions=["v3.customer.name[customer]"],
    )
    assert_sql_equal(
        result["sql"],
        """
        WITH
        v3_customer AS (
            SELECT customer_id, name
            FROM default.v3.customers
        ),
        v3_order_details AS (
            SELECT o.customer_id, oi.quantity
            FROM default.v3.orders o
            JOIN default.v3.order_items oi ON o.order_id = oi.order_id
        ),
        order_details_0 AS (
            SELECT t2.name name_customer, SUM(t1.quantity) quantity_sum_06b64d2e
            FROM v3_order_details t1
            LEFT OUTER JOIN v3_customer t2 ON t1.customer_id = t2.customer_id
            GROUP BY t2.name
        )
        SELECT order_details_0.name_customer AS name_customer,
               SUM(order_details_0.quantity_sum_06b64d2e) AS multi_source_quantity
        FROM order_details_0
        GROUP BY order_details_0.name_customer
        """,
    )


@pytest.mark.asyncio
async def test_tie_keeps_first_source(client_with_build_v3):
    await create_multi_source_metric(client_with_build_v3)
    result = await metrics_sql(
        client_with_build_v3,
        metrics=["v3.multi_source_quantity"],
        dimensions=["v3.product.category"],
    )
    assert_sql_equal(
        result["sql"],
        f"""
        WITH
        {ORDER_LINES_CTE},
        v3_product AS (
            SELECT product_id, category
            FROM default.v3.products
        ),
        order_lines_0 AS (
            SELECT t2.category, SUM(t1.quantity) quantity_sum_58eaf43b
            FROM v3_order_lines t1
            LEFT OUTER JOIN v3_product t2 ON t1.product_id = t2.product_id
            GROUP BY t2.category
        )
        SELECT order_lines_0.category AS category,
               SUM(order_lines_0.quantity_sum_58eaf43b) AS multi_source_quantity
        FROM order_lines_0
        GROUP BY order_lines_0.category
        """,
    )


@pytest.mark.asyncio
async def test_mixed_parents_get_own_groups(client_with_build_v3):
    await create_multi_source_metric(client_with_build_v3)
    result = await metrics_sql(
        client_with_build_v3,
        metrics=["v3.multi_source_quantity", "v3.total_revenue"],
        dimensions=["v3.product.category"],
    )
    assert_sql_equal(
        result["sql"],
        f"""
        WITH
        v3_order_details AS (
            SELECT oi.product_id, oi.quantity * oi.unit_price AS line_total
            FROM default.v3.orders o
            JOIN default.v3.order_items oi ON o.order_id = oi.order_id
        ),
        v3_product AS (
            SELECT product_id, category
            FROM default.v3.products
        ),
        {ORDER_LINES_CTE},
        order_details_0 AS (
            SELECT t2.category, SUM(t1.line_total) line_total_sum_e1f61696
            FROM v3_order_details t1
            LEFT OUTER JOIN v3_product t2 ON t1.product_id = t2.product_id
            GROUP BY t2.category
        ),
        order_lines_0 AS (
            SELECT t2.category, SUM(t1.quantity) quantity_sum_58eaf43b
            FROM v3_order_lines t1
            LEFT OUTER JOIN v3_product t2 ON t1.product_id = t2.product_id
            GROUP BY t2.category
        )
        SELECT COALESCE(order_details_0.category, order_lines_0.category) AS category,
               SUM(order_lines_0.quantity_sum_58eaf43b) AS multi_source_quantity,
               SUM(order_details_0.line_total_sum_e1f61696) AS total_revenue
        FROM order_details_0
        FULL OUTER JOIN order_lines_0
            ON order_details_0.category = order_lines_0.category
        GROUP BY 1
        """,
    )


@pytest.mark.asyncio
async def test_mixed_request_routes_each_metric(client_with_build_v3):
    await create_multi_source_metric(client_with_build_v3)
    result = await metrics_sql(
        client_with_build_v3,
        metrics=["v3.multi_source_quantity", "v3.total_revenue"],
        dimensions=["v3.customer.name[customer]"],
    )
    assert_sql_equal(
        result["sql"],
        """
        WITH
        v3_customer AS (
            SELECT customer_id, name
            FROM default.v3.customers
        ),
        v3_order_details AS (
            SELECT o.customer_id, oi.quantity, oi.quantity * oi.unit_price AS line_total
            FROM default.v3.orders o
            JOIN default.v3.order_items oi ON o.order_id = oi.order_id
        ),
        order_details_0 AS (
            SELECT t2.name name_customer,
                   SUM(t1.quantity) quantity_sum_06b64d2e,
                   SUM(t1.line_total) line_total_sum_e1f61696
            FROM v3_order_details t1
            LEFT OUTER JOIN v3_customer t2 ON t1.customer_id = t2.customer_id
            GROUP BY t2.name
        )
        SELECT order_details_0.name_customer AS name_customer,
               SUM(order_details_0.quantity_sum_06b64d2e) AS multi_source_quantity,
               SUM(order_details_0.line_total_sum_e1f61696) AS total_revenue
        FROM order_details_0
        GROUP BY order_details_0.name_customer
        """,
    )


@pytest.mark.asyncio
async def test_derived_metric_routes_its_base(client_with_build_v3):
    await create_multi_source_metric(client_with_build_v3)
    response = await client_with_build_v3.post(
        "/nodes/metric/",
        json={
            "name": "v3.multi_source_quantity_doubled",
            "query": "SELECT v3.multi_source_quantity * 2",
            "mode": "published",
        },
    )
    assert response.status_code == 201, response.json()
    result = await metrics_sql(
        client_with_build_v3,
        metrics=["v3.multi_source_quantity_doubled"],
        dimensions=["v3.customer.name[customer]"],
    )
    assert_sql_equal(
        result["sql"],
        """
        WITH
        v3_customer AS (
            SELECT customer_id, name
            FROM default.v3.customers
        ),
        v3_order_details AS (
            SELECT o.customer_id, oi.quantity
            FROM default.v3.orders o
            JOIN default.v3.order_items oi ON o.order_id = oi.order_id
        ),
        order_details_0 AS (
            SELECT t2.name name_customer, SUM(t1.quantity) quantity_sum_06b64d2e
            FROM v3_order_details t1
            LEFT OUTER JOIN v3_customer t2 ON t1.customer_id = t2.customer_id
            GROUP BY t2.name
        )
        SELECT order_details_0.name_customer AS name_customer,
               SUM(order_details_0.quantity_sum_06b64d2e) * 2
                   AS multi_source_quantity_doubled
        FROM order_details_0
        GROUP BY order_details_0.name_customer
        """,
    )


@pytest.mark.asyncio
async def test_no_source_covers_request(client_with_build_v3):
    await create_multi_source_metric(client_with_build_v3)
    response = await client_with_build_v3.get(
        "/sql/metrics/v3/",
        params={
            "metrics": ["v3.multi_source_quantity"],
            "dimensions": ["v3.date.week[page]"],
        },
    )
    assert (response.status_code, response.json()["message"]) == (
        422,
        "No source of v3.multi_source_quantity covers the request. "
        "Source 0: cannot reach v3.date.week[page]. "
        "Source 1: cannot reach v3.date.week[page].",
    )


@pytest.mark.asyncio
async def test_single_source_metric_sql_is_unchanged(client_with_build_v3):
    """Same request and expected SQL as test_simple_single_metric."""
    result = await metrics_sql(
        client_with_build_v3,
        metrics=["v3.total_revenue"],
        dimensions=["v3.order_details.status"],
    )
    assert_sql_equal(
        result["sql"],
        """
        WITH
        v3_order_details AS (
            SELECT o.status, oi.quantity * oi.unit_price AS line_total
            FROM default.v3.orders o
            JOIN default.v3.order_items oi ON o.order_id = oi.order_id
        ),
        order_details_0 AS (
            SELECT t1.status, SUM(t1.line_total) line_total_sum_e1f61696
            FROM v3_order_details t1
            GROUP BY t1.status
        )
        SELECT order_details_0.status AS status,
               SUM(order_details_0.line_total_sum_e1f61696) AS total_revenue
        FROM order_details_0
        GROUP BY order_details_0.status
        """,
    )
