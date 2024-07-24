"""Tests for all /sql endpoints that use node SQL build v2"""
# pylint: disable=line-too-long,too-many-lines
import duckdb
import pytest
from httpx import AsyncClient

from datajunction_server.sql.parsing.backends.antlr4 import parse


@pytest.mark.parametrize(
    "metrics, dimensions, filters, sql, columns, rows",
    [
        # One metric with two measures + one local dimension. Both referenced measures should
        # show up in the generated measures SQL
        (
            ["default.total_repair_order_discounts"],
            ["default.dispatcher.dispatcher_id"],
            [],
            """
            WITH default_DOT_repair_orders_fact AS (
              SELECT
                default_DOT_repair_orders.repair_order_id,
                default_DOT_repair_orders.municipality_id,
                default_DOT_repair_orders.hard_hat_id,
                default_DOT_repair_orders.dispatcher_id,
                default_DOT_repair_orders.order_date,
                default_DOT_repair_orders.dispatched_date,
                default_DOT_repair_orders.required_date,
                default_DOT_repair_order_details.discount,
                default_DOT_repair_order_details.price,
                default_DOT_repair_order_details.quantity,
                default_DOT_repair_order_details.repair_type_id,
                default_DOT_repair_order_details.price *
                  default_DOT_repair_order_details.quantity AS total_repair_cost,
                default_DOT_repair_orders.dispatched_date -
                  default_DOT_repair_orders.order_date AS time_to_dispatch,
                default_DOT_repair_orders.dispatched_date -
                  default_DOT_repair_orders.required_date AS dispatch_delay
              FROM roads.repair_orders AS default_DOT_repair_orders
              JOIN roads.repair_order_details AS default_DOT_repair_order_details
                ON default_DOT_repair_orders.repair_order_id =
                  default_DOT_repair_order_details.repair_order_id
            )
            SELECT
              default_DOT_repair_orders_fact.dispatcher_id
                default_DOT_dispatcher_DOT_dispatcher_id,
              default_DOT_repair_orders_fact.discount
                default_DOT_repair_orders_fact_DOT_discount,
              default_DOT_repair_orders_fact.price
                default_DOT_repair_orders_fact_DOT_price
            FROM default_DOT_repair_orders_fact
            """,
            [
                {
                    "column": "dispatcher_id",
                    "name": "default_DOT_dispatcher_DOT_dispatcher_id",
                    "node": "default.dispatcher",
                    "semantic_entity": "default.dispatcher.dispatcher_id",
                    "semantic_type": "dimension",
                    "type": "int",
                },
                {
                    "column": "discount",
                    "name": "default_DOT_repair_orders_fact_DOT_discount",
                    "node": "default.repair_orders_fact",
                    "semantic_entity": "default.repair_orders_fact.discount",
                    "semantic_type": "measure",
                    "type": "float",
                },
                {
                    "column": "price",
                    "name": "default_DOT_repair_orders_fact_DOT_price",
                    "node": "default.repair_orders_fact",
                    "semantic_entity": "default.repair_orders_fact.price",
                    "semantic_type": "measure",
                    "type": "float",
                },
            ],
            [
                (3, 0.05000000074505806, 63708.0),
                (1, 0.05000000074505806, 67253.0),
                (2, 0.05000000074505806, 66808.0),
                (1, 0.05000000074505806, 18497.0),
                (2, 0.05000000074505806, 76463.0),
                (2, 0.05000000074505806, 87858.0),
                (2, 0.05000000074505806, 63918.0),
                (3, 0.05000000074505806, 21083.0),
                (2, 0.05000000074505806, 74555.0),
                (3, 0.05000000074505806, 27222.0),
                (1, 0.05000000074505806, 73600.0),
                (3, 0.009999999776482582, 54901.0),
                (1, 0.009999999776482582, 51594.0),
                (2, 0.009999999776482582, 65114.0),
                (3, 0.009999999776482582, 48919.0),
                (3, 0.009999999776482582, 70418.0),
                (3, 0.009999999776482582, 29684.0),
                (1, 0.009999999776482582, 62928.0),
                (3, 0.009999999776482582, 97916.0),
                (1, 0.009999999776482582, 44120.0),
                (3, 0.009999999776482582, 53374.0),
                (1, 0.009999999776482582, 87289.0),
                (1, 0.009999999776482582, 92366.0),
                (2, 0.009999999776482582, 47857.0),
                (2, 0.009999999776482582, 68745.0),
            ],
        ),
        # # Two metrics with overlapping measures + one joinable dimension
        (
            [
                "default.total_repair_order_discounts",
                "default.avg_repair_order_discounts",
            ],
            ["default.dispatcher.dispatcher_id"],
            [],
            """
            WITH default_DOT_repair_orders_fact AS (
              SELECT
                default_DOT_repair_orders.repair_order_id,
                default_DOT_repair_orders.municipality_id,
                default_DOT_repair_orders.hard_hat_id,
                default_DOT_repair_orders.dispatcher_id,
                default_DOT_repair_orders.order_date,
                default_DOT_repair_orders.dispatched_date,
                default_DOT_repair_orders.required_date,
                default_DOT_repair_order_details.discount,
                default_DOT_repair_order_details.price,
                default_DOT_repair_order_details.quantity,
                default_DOT_repair_order_details.repair_type_id,
                default_DOT_repair_order_details.price * default_DOT_repair_order_details.quantity
                  AS total_repair_cost,
                default_DOT_repair_orders.dispatched_date - default_DOT_repair_orders.order_date
                  AS time_to_dispatch,
                default_DOT_repair_orders.dispatched_date - default_DOT_repair_orders.required_date
                  AS dispatch_delay
              FROM roads.repair_orders AS default_DOT_repair_orders
              JOIN roads.repair_order_details AS default_DOT_repair_order_details
                ON default_DOT_repair_orders.repair_order_id =
                  default_DOT_repair_order_details.repair_order_id
            )
            SELECT
              default_DOT_repair_orders_fact.dispatcher_id
                default_DOT_dispatcher_DOT_dispatcher_id,
              default_DOT_repair_orders_fact.discount
                default_DOT_repair_orders_fact_DOT_discount,
              default_DOT_repair_orders_fact.price
                default_DOT_repair_orders_fact_DOT_price
            FROM default_DOT_repair_orders_fact
            """,
            [
                {
                    "column": "dispatcher_id",
                    "name": "default_DOT_dispatcher_DOT_dispatcher_id",
                    "node": "default.dispatcher",
                    "semantic_entity": "default.dispatcher.dispatcher_id",
                    "semantic_type": "dimension",
                    "type": "int",
                },
                {
                    "column": "discount",
                    "name": "default_DOT_repair_orders_fact_DOT_discount",
                    "node": "default.repair_orders_fact",
                    "semantic_entity": "default.repair_orders_fact.discount",
                    "semantic_type": "measure",
                    "type": "float",
                },
                {
                    "column": "price",
                    "name": "default_DOT_repair_orders_fact_DOT_price",
                    "node": "default.repair_orders_fact",
                    "semantic_entity": "default.repair_orders_fact.price",
                    "semantic_type": "measure",
                    "type": "float",
                },
            ],
            [
                (3, 0.05000000074505806, 63708.0),
                (1, 0.05000000074505806, 67253.0),
                (2, 0.05000000074505806, 66808.0),
                (1, 0.05000000074505806, 18497.0),
                (2, 0.05000000074505806, 76463.0),
                (2, 0.05000000074505806, 87858.0),
                (2, 0.05000000074505806, 63918.0),
                (3, 0.05000000074505806, 21083.0),
                (2, 0.05000000074505806, 74555.0),
                (3, 0.05000000074505806, 27222.0),
                (1, 0.05000000074505806, 73600.0),
                (3, 0.009999999776482582, 54901.0),
                (1, 0.009999999776482582, 51594.0),
                (2, 0.009999999776482582, 65114.0),
                (3, 0.009999999776482582, 48919.0),
                (3, 0.009999999776482582, 70418.0),
                (3, 0.009999999776482582, 29684.0),
                (1, 0.009999999776482582, 62928.0),
                (3, 0.009999999776482582, 97916.0),
                (1, 0.009999999776482582, 44120.0),
                (3, 0.009999999776482582, 53374.0),
                (1, 0.009999999776482582, 87289.0),
                (1, 0.009999999776482582, 92366.0),
                (2, 0.009999999776482582, 47857.0),
                (2, 0.009999999776482582, 68745.0),
            ],
        ),
        # Two metrics with different measures + two dimensions from different sources
        (
            ["default.avg_time_to_dispatch", "default.total_repair_cost"],
            [
                "default.us_state.state_name",
                "default.dispatcher.company_name",
                "default.hard_hat.last_name",
            ],
            [
                "default.us_state.state_name = 'New Jersey'",
                "default.hard_hat.last_name IN ('Brian')",
            ],
            """
            WITH default_DOT_repair_orders_fact AS (
              SELECT
                default_DOT_repair_orders.repair_order_id,
                default_DOT_repair_orders.municipality_id,
                default_DOT_repair_orders.hard_hat_id,
                default_DOT_repair_orders.dispatcher_id,
                default_DOT_repair_orders.order_date,
                default_DOT_repair_orders.dispatched_date,
                default_DOT_repair_orders.required_date,
                default_DOT_repair_order_details.discount,
                default_DOT_repair_order_details.price,
                default_DOT_repair_order_details.quantity,
                default_DOT_repair_order_details.repair_type_id,
                default_DOT_repair_order_details.price * default_DOT_repair_order_details.quantity
                  AS total_repair_cost,
                default_DOT_repair_orders.dispatched_date - default_DOT_repair_orders.order_date
                  AS time_to_dispatch,
                default_DOT_repair_orders.dispatched_date - default_DOT_repair_orders.required_date
                  AS dispatch_delay
              FROM roads.repair_orders AS default_DOT_repair_orders
              JOIN roads.repair_order_details AS default_DOT_repair_order_details
              ON default_DOT_repair_orders.repair_order_id =
                default_DOT_repair_order_details.repair_order_id
            ),
            default_DOT_hard_hat AS (
              SELECT
                default_DOT_hard_hats.hard_hat_id,
                default_DOT_hard_hats.last_name,
                default_DOT_hard_hats.first_name,
                default_DOT_hard_hats.title,
                default_DOT_hard_hats.birth_date,
                default_DOT_hard_hats.hire_date,
                default_DOT_hard_hats.address,
                default_DOT_hard_hats.city,
                default_DOT_hard_hats.state,
                default_DOT_hard_hats.postal_code,
                default_DOT_hard_hats.country,
                default_DOT_hard_hats.manager,
                default_DOT_hard_hats.contractor_id
              FROM roads.hard_hats AS default_DOT_hard_hats
              WHERE default_DOT_hard_hats.last_name IN ('Brian')
            ),
            default_DOT_us_state AS (
              SELECT
                default_DOT_us_states.state_id,
                default_DOT_us_states.state_name,
                default_DOT_us_states.state_abbr AS state_short,
                default_DOT_us_states.state_region
              FROM roads.us_states AS default_DOT_us_states
              WHERE  default_DOT_us_states.state_name = 'New Jersey'
            ),
            default_DOT_dispatcher AS (
              SELECT
                default_DOT_dispatchers.dispatcher_id,
                default_DOT_dispatchers.company_name,
                default_DOT_dispatchers.phone
              FROM roads.dispatchers AS default_DOT_dispatchers
            )
            SELECT
              default_DOT_repair_orders_fact.total_repair_cost
                default_DOT_repair_orders_fact_DOT_total_repair_cost,
              default_DOT_repair_orders_fact.time_to_dispatch
                default_DOT_repair_orders_fact_DOT_time_to_dispatch,
              default_DOT_us_state.state_name
                default_DOT_us_state_DOT_state_name,
              default_DOT_dispatcher.company_name
                default_DOT_dispatcher_DOT_company_name,
              default_DOT_hard_hat.last_name
                default_DOT_hard_hat_DOT_last_name
            FROM default_DOT_repair_orders_fact
            INNER JOIN default_DOT_hard_hat
              ON default_DOT_repair_orders_fact.hard_hat_id = default_DOT_hard_hat.hard_hat_id
            INNER JOIN default_DOT_us_state
              ON default_DOT_hard_hat.state = default_DOT_us_state.state_short
            LEFT JOIN default_DOT_dispatcher
              ON default_DOT_repair_orders_fact.dispatcher_id =default_DOT_dispatcher.dispatcher_id
            """,
            [
                {
                    "column": "total_repair_cost",
                    "name": "default_DOT_repair_orders_fact_DOT_total_repair_cost",
                    "node": "default.repair_orders_fact",
                    "semantic_entity": "default.repair_orders_fact.total_repair_cost",
                    "semantic_type": "measure",
                    "type": "float",
                },
                {
                    "column": "time_to_dispatch",
                    "name": "default_DOT_repair_orders_fact_DOT_time_to_dispatch",
                    "node": "default.repair_orders_fact",
                    "semantic_entity": "default.repair_orders_fact.time_to_dispatch",
                    "semantic_type": "measure",
                    "type": "timestamp",
                },
                {
                    "column": "state_name",
                    "name": "default_DOT_us_state_DOT_state_name",
                    "node": "default.us_state",
                    "semantic_entity": "default.us_state.state_name",
                    "semantic_type": "dimension",
                    "type": "string",
                },
                {
                    "column": "company_name",
                    "name": "default_DOT_dispatcher_DOT_company_name",
                    "node": "default.dispatcher",
                    "semantic_entity": "default.dispatcher.company_name",
                    "semantic_type": "dimension",
                    "type": "string",
                },
                {
                    "column": "last_name",
                    "name": "default_DOT_hard_hat_DOT_last_name",
                    "node": "default.hard_hat",
                    "semantic_entity": "default.hard_hat.last_name",
                    "semantic_type": "dimension",
                    "type": "string",
                },
            ],
            [
                (92366.0, 204, "New Jersey", "Pothole Pete", "Brian"),
                (44120.0, 196, "New Jersey", "Pothole Pete", "Brian"),
                (18497.0, 146, "New Jersey", "Pothole Pete", "Brian"),
                (63708.0, 150, "New Jersey", "Federal Roads Group", "Brian"),
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_measures_sql_with_filters__v2(  # pylint: disable=too-many-arguments
    metrics,
    dimensions,
    filters,
    sql,
    columns,
    rows,
    module__client_with_roads: AsyncClient,
    duckdb_conn: duckdb.DuckDBPyConnection,  # pylint: disable=c-extension-no-member
):
    """
    Test ``GET /sql/measures`` with various metrics, filters, and dimensions.
    """
    await module__client_with_roads.post(
        "/nodes/default.repair_orders_fact/link",
        json={
            "dimension_node": "default.hard_hat",
            "join_type": "inner",
            "join_on": (
                "default.repair_orders_fact.hard_hat_id = default.hard_hat.hard_hat_id"
            ),
        },
    )
    await module__client_with_roads.post(
        "/nodes/default.repair_orders_fact/link",
        json={
            "dimension_node": "default.dispatcher",
            "join_type": "left",
            "join_on": (
                "default.repair_orders_fact.dispatcher_id = default.dispatcher.dispatcher_id"
            ),
        },
    )
    await module__client_with_roads.post(
        "/nodes/default.hard_hat/link",
        json={
            "dimension_node": "default.us_state",
            "join_type": "inner",
            "join_on": ("default.hard_hat.state = default.us_state.state_short"),
        },
    )
    sql_params = {
        "metrics": metrics,
        "dimensions": dimensions,
        "filters": filters,
    }
    response = await module__client_with_roads.get(
        "/sql/measures/v2",
        params=sql_params,
    )
    data = response.json()
    com = list(data.values())[0]
    assert str(parse(str(sql))) == str(parse(str(com["sql"])))
    result = duckdb_conn.sql(com["sql"])
    assert set(result.fetchall()) == set(rows)
    assert com["columns"] == columns
