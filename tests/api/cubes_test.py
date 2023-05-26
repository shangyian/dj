"""
Tests for the cubes API.
"""
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from dj.service_clients import QueryServiceClient
from tests.sql.utils import compare_query_strings


def test_read_cube(client_with_examples: TestClient) -> None:
    """
    Test ``GET /cubes/{name}``.
    """
    # Create a cube
    response = client_with_examples.post(
        "/nodes/cube/",
        json={
            "metrics": ["default.number_of_account_types"],
            "dimensions": ["default.account_type.account_type_name"],
            "filters": [],
            "description": "A cube of number of accounts grouped by account type",
            "mode": "published",
            "name": "default.number_of_accounts_by_account_type",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["version"] == "v1.0"
    assert data["type"] == "cube"
    assert data["name"] == "default.number_of_accounts_by_account_type"
    assert data["display_name"] == "Default: Number Of Accounts By Account Type"

    # Read the cube
    response = client_with_examples.get(
        "/cubes/default.number_of_accounts_by_account_type",
    )
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "cube"
    assert data["name"] == "default.number_of_accounts_by_account_type"
    assert data["display_name"] == "Default: Number Of Accounts By Account Type"
    assert data["version"] == "v1.0"
    assert data["description"] == "A cube of number of accounts grouped by account type"
    assert compare_query_strings(
        data["query"],
        """
        WITH m0_default_DOT_number_of_account_types AS (
          SELECT
            default_DOT_account_type.account_type_name,
            count(default_DOT_account_type.id) default_DOT_number_of_account_types
          FROM (
            SELECT
              default_DOT_account_type_table.account_type_classification,
              default_DOT_account_type_table.account_type_name,
              default_DOT_account_type_table.id
            FROM accounting.account_type_table AS default_DOT_account_type_table
          ) AS default_DOT_account_type
          GROUP BY  default_DOT_account_type.account_type_name
        )
        SELECT
          m0_default_DOT_number_of_account_types.default_DOT_number_of_account_types,
          m0_default_DOT_number_of_account_types.account_type_name
        FROM m0_default_DOT_number_of_account_types
        """,
    )


def test_create_invalid_cube(client_with_examples: TestClient):
    """
    Check that creating a cube with a query fails appropriately
    """
    response = client_with_examples.post(
        "/nodes/cube/",
        json={
            "description": "A cube of number of accounts grouped by account type",
            "mode": "published",
            "query": "SELECT 1",
            "cube_elements": [
                "default.number_of_account_types",
                "default.account_type",
            ],
            "name": "default.cubes_shouldnt_have_queries",
        },
    )
    assert response.status_code == 422
    data = response.json()
    assert data["detail"] == [
        {
            "loc": ["body", "metrics"],
            "msg": "field required",
            "type": "value_error.missing",
        },
        {
            "loc": ["body", "dimensions"],
            "msg": "field required",
            "type": "value_error.missing",
        },
    ]

    # Check that creating a cube with no cube elements fails appropriately
    response = client_with_examples.post(
        "/nodes/cube/",
        json={
            "metrics": ["default.account_type"],
            "dimensions": ["default.account_type.account_type_name"],
            "description": "A cube of number of accounts grouped by account type",
            "mode": "published",
            "name": "default.cubes_must_have_elements",
        },
    )
    assert response.status_code == 422
    data = response.json()
    assert data == {
        "message": "Node default.account_type of type dimension cannot be added to a cube. "
        "Did you mean to add a dimension attribute?",
        "errors": [],
        "warnings": [],
    }

    # Check that creating a cube with incompatible nodes fails appropriately
    response = client_with_examples.post(
        "/nodes/cube/",
        json={
            "metrics": ["default.number_of_account_types"],
            "dimensions": ["default.payment_type.payment_type_name"],
            "description": "",
            "mode": "published",
            "name": "default.cubes_cant_use_source_nodes",
        },
    )
    assert response.status_code == 422
    data = response.json()
    assert data == {
        "message": "The dimension attribute `default.payment_type.payment_type_name` "
        "is not available on every metric and thus cannot be included.",
        "errors": [],
        "warnings": [],
    }

    # Check that creating a cube with no metric nodes fails appropriately
    response = client_with_examples.post(
        "/nodes/cube/",
        json={
            "metrics": [],
            "dimensions": ["default.account_type.account_type_name"],
            "description": "",
            "mode": "published",
            "name": "default.cubes_must_have_metrics",
        },
    )
    assert response.status_code == 422
    data = response.json()
    assert data == {
        "message": "At least one metric is required",
        "errors": [],
        "warnings": [],
    }

    # Check that creating a cube with no dimension nodes fails appropriately
    response = client_with_examples.post(
        "/nodes/cube/",
        json={
            "metrics": ["default.number_of_account_types"],
            "dimensions": [],
            "description": "A cube of number of accounts grouped by account type",
            "mode": "published",
            "name": "default.cubes_must_have_dimensions",
        },
    )
    assert response.status_code == 422
    data = response.json()
    assert data == {
        "message": "At least one dimension is required",
        "errors": [],
        "warnings": [],
    }


def test_raise_on_cube_with_multiple_catalogs(
    client_with_examples: TestClient,
) -> None:
    """
    Test raising when creating a cube with multiple catalogs
    """
    # Create a cube
    response = client_with_examples.post(
        "/nodes/cube/",
        json={
            "metrics": ["default.number_of_account_types", "basic.num_comments"],
            "dimensions": ["default.account_type.account_type_name"],
            "description": "multicatalog cube's raise an error",
            "mode": "published",
            "name": "default.multicatalog",
        },
    )
    assert not response.ok
    data = response.json()
    assert "Metrics and dimensions cannot be from multiple catalogs" in data["message"]


def test_create_cube_with_materialization(client_with_query_service: TestClient):
    """
    Testing creating cube with materialization configs
    """
    metrics_list = [
        "default.discounted_orders_rate",
        "default.num_repair_orders",
        "default.avg_repair_price",
        "default.total_repair_cost",
        "default.total_repair_order_discounts",
    ]

    response = client_with_query_service.post(
        "/nodes/cube/",
        json={
            "metrics": metrics_list,
            "dimensions": [
                "default.hard_hat.country",
                "default.hard_hat.postal_code",
                "default.hard_hat.city",
                "default.hard_hat.state",
                "default.dispatcher.company_name",
                "default.municipality_dim.local_region",
            ],
            "filters": ["default.hard_hat.state='AZ'"],
            "description": "Cube of various metrics related to repairs",
            "mode": "published",
            "name": "default.repairs_cube_with_materialization",
            "materialization_configs": [
                {
                    "engine": {
                        "name": "druid",
                        "version": "",
                    },
                    "config": {
                        "druid": {
                            "granularity": "DAY",
                            "intervals": [],
                            "timestamp_column": "something",
                            "parse_spec_format": "parquet",
                        },
                        "spark": {},
                    },
                    "schedule": "0 * * * *",
                },
            ],
        },
    )
    default_materialization = response.json()["materialization_configs"][0]
    assert default_materialization["job"] == "DruidCubeMaterializationJob"
    assert default_materialization["schedule"] == "0 * * * *"


@pytest.fixture
def client_with_repairs_cube(client_with_query_service: TestClient):
    """
    Adds a repairs cube with a new double total repair cost metric to the test client
    """
    metrics_list = [
        "default.discounted_orders_rate",
        "default.num_repair_orders",
        "default.avg_repair_price",
        "default.total_repair_cost",
        "default.total_repair_order_discounts",
    ]

    # Metric that doubles the total repair cost to test the sum(x) + sum(y) scenario
    client_with_query_service.post(
        "/nodes/metric/",
        json={
            "description": "Double total repair cost",
            "query": (
                "SELECT sum(price) + sum(price) as default_DOT_double_total_repair_cost "
                "FROM default.repair_order_details"
            ),
            "mode": "published",
            "name": "default.double_total_repair_cost",
        },
    )
    # Should succeed
    response = client_with_query_service.post(
        "/nodes/cube/",
        json={
            "metrics": metrics_list + ["default.double_total_repair_cost"],
            "dimensions": [
                "default.hard_hat.country",
                "default.hard_hat.postal_code",
                "default.hard_hat.city",
                "default.hard_hat.state",
                "default.dispatcher.company_name",
                "default.municipality_dim.local_region",
            ],
            "filters": ["default.hard_hat.state='AZ'"],
            "description": "Cube of various metrics related to repairs",
            "mode": "published",
            "name": "default.repairs_cube",
        },
    )
    assert response.status_code == 201
    return client_with_query_service


def test_invalid_cube(client_with_examples: TestClient):
    """
    Test that creating a cube without valid dimensions fails
    """
    metrics_list = [
        "default.discounted_orders_rate",
        "default.num_repair_orders",
        "default.avg_repair_price",
        "default.total_repair_cost",
        "default.total_repair_order_discounts",
    ]
    # Should fail because dimension attribute isn't available
    response = client_with_examples.post(
        "/nodes/cube/",
        json={
            "metrics": metrics_list,
            "dimensions": [
                "default.contractor.company_name",
            ],
            "description": "Cube of various metrics related to repairs",
            "mode": "published",
            "name": "default.repairs_cube",
        },
    )
    assert response.json()["message"] == (
        "The dimension attribute `default.contractor.company_name` "
        "is not available on every metric and thus cannot be included."
    )


def test_create_cube(  # pylint: disable=redefined-outer-name
    client_with_repairs_cube: TestClient,
):
    """
    Tests cube creation and the generated cube SQL
    """
    response = client_with_repairs_cube.get("/nodes/default.repairs_cube/")
    results = response.json()

    assert results["name"] == "default.repairs_cube"
    assert results["display_name"] == "Default: Repairs Cube"
    assert results["description"] == "Cube of various metrics related to repairs"
    expected_query = """
WITH
m0_default_DOT_discounted_orders_rate AS (SELECT  default_DOT_hard_hat.city,
    default_DOT_dispatcher.company_name,
    default_DOT_hard_hat.country,
    CAST(sum(if(default_DOT_repair_order_details.discount > 0.0, 1, 0)) AS DOUBLE) / count(*) AS default_DOT_discounted_orders_rate,
    default_DOT_municipality_dim.local_region,
    default_DOT_hard_hat.postal_code,
    default_DOT_hard_hat.state
 FROM roads.repair_order_details AS default_DOT_repair_order_details LEFT OUTER JOIN (SELECT  default_DOT_repair_orders.dispatcher_id,
    default_DOT_repair_orders.hard_hat_id,
    default_DOT_repair_orders.municipality_id,
    default_DOT_repair_orders.repair_order_id
 FROM roads.repair_orders AS default_DOT_repair_orders) AS default_DOT_repair_order ON default_DOT_repair_order_details.repair_order_id = default_DOT_repair_order.repair_order_id
LEFT OUTER JOIN (SELECT  default_DOT_dispatchers.company_name,
    default_DOT_dispatchers.dispatcher_id
 FROM roads.dispatchers AS default_DOT_dispatchers) AS default_DOT_dispatcher ON default_DOT_repair_order.dispatcher_id = default_DOT_dispatcher.dispatcher_id
LEFT OUTER JOIN (SELECT  default_DOT_hard_hats.city,
    default_DOT_hard_hats.country,
    default_DOT_hard_hats.hard_hat_id,
    default_DOT_hard_hats.postal_code,
    default_DOT_hard_hats.state
 FROM roads.hard_hats AS default_DOT_hard_hats) AS default_DOT_hard_hat ON default_DOT_repair_order.hard_hat_id = default_DOT_hard_hat.hard_hat_id
LEFT OUTER JOIN (SELECT  default_DOT_municipality.local_region,
    default_DOT_municipality.municipality_id
 FROM roads.municipality AS default_DOT_municipality LEFT  JOIN roads.municipality_municipality_type AS default_DOT_municipality_municipality_type ON default_DOT_municipality.municipality_id = default_DOT_municipality_municipality_type.municipality_id
LEFT  JOIN roads.municipality_type AS default_DOT_municipality_type ON default_DOT_municipality_municipality_type.municipality_type_id = default_DOT_municipality_type.municipality_type_desc) AS default_DOT_municipality_dim ON default_DOT_repair_order.municipality_id = default_DOT_municipality_dim.municipality_id
 WHERE  default_DOT_hard_hat.state = 'AZ'
 GROUP BY  default_DOT_hard_hat.country, default_DOT_hard_hat.postal_code, default_DOT_hard_hat.city, default_DOT_hard_hat.state, default_DOT_dispatcher.company_name, default_DOT_municipality_dim.local_region
),
m1_default_DOT_num_repair_orders AS (SELECT  default_DOT_hard_hat.city,
    default_DOT_dispatcher.company_name,
    default_DOT_hard_hat.country,
    count(default_DOT_repair_orders.repair_order_id) default_DOT_num_repair_orders,
    default_DOT_municipality_dim.local_region,
    default_DOT_hard_hat.postal_code,
    default_DOT_hard_hat.state
 FROM roads.repair_orders AS default_DOT_repair_orders LEFT OUTER JOIN (SELECT  default_DOT_dispatchers.company_name,
    default_DOT_dispatchers.dispatcher_id
 FROM roads.dispatchers AS default_DOT_dispatchers) AS default_DOT_dispatcher ON default_DOT_repair_orders.dispatcher_id = default_DOT_dispatcher.dispatcher_id
LEFT OUTER JOIN (SELECT  default_DOT_hard_hats.city,
    default_DOT_hard_hats.country,
    default_DOT_hard_hats.hard_hat_id,
    default_DOT_hard_hats.postal_code,
    default_DOT_hard_hats.state
 FROM roads.hard_hats AS default_DOT_hard_hats) AS default_DOT_hard_hat ON default_DOT_repair_orders.hard_hat_id = default_DOT_hard_hat.hard_hat_id
LEFT OUTER JOIN (SELECT  default_DOT_municipality.local_region,
    default_DOT_municipality.municipality_id
 FROM roads.municipality AS default_DOT_municipality LEFT  JOIN roads.municipality_municipality_type AS default_DOT_municipality_municipality_type ON default_DOT_municipality.municipality_id = default_DOT_municipality_municipality_type.municipality_id
LEFT  JOIN roads.municipality_type AS default_DOT_municipality_type ON default_DOT_municipality_municipality_type.municipality_type_id = default_DOT_municipality_type.municipality_type_desc) AS default_DOT_municipality_dim ON default_DOT_repair_orders.municipality_id = default_DOT_municipality_dim.municipality_id
 WHERE  default_DOT_hard_hat.state = 'AZ'
 GROUP BY  default_DOT_hard_hat.country, default_DOT_hard_hat.postal_code, default_DOT_hard_hat.city, default_DOT_hard_hat.state, default_DOT_dispatcher.company_name, default_DOT_municipality_dim.local_region
),
m2_default_DOT_avg_repair_price AS (SELECT  default_DOT_hard_hat.city,
    default_DOT_dispatcher.company_name,
    default_DOT_hard_hat.country,
    avg(default_DOT_repair_order_details.price) AS default_DOT_avg_repair_price,
    default_DOT_municipality_dim.local_region,
    default_DOT_hard_hat.postal_code,
    default_DOT_hard_hat.state
 FROM roads.repair_order_details AS default_DOT_repair_order_details LEFT OUTER JOIN (SELECT  default_DOT_repair_orders.dispatcher_id,
    default_DOT_repair_orders.hard_hat_id,
    default_DOT_repair_orders.municipality_id,
    default_DOT_repair_orders.repair_order_id
 FROM roads.repair_orders AS default_DOT_repair_orders) AS default_DOT_repair_order ON default_DOT_repair_order_details.repair_order_id = default_DOT_repair_order.repair_order_id
LEFT OUTER JOIN (SELECT  default_DOT_dispatchers.company_name,
    default_DOT_dispatchers.dispatcher_id
 FROM roads.dispatchers AS default_DOT_dispatchers) AS default_DOT_dispatcher ON default_DOT_repair_order.dispatcher_id = default_DOT_dispatcher.dispatcher_id
LEFT OUTER JOIN (SELECT  default_DOT_hard_hats.city,
    default_DOT_hard_hats.country,
    default_DOT_hard_hats.hard_hat_id,
    default_DOT_hard_hats.postal_code,
    default_DOT_hard_hats.state
 FROM roads.hard_hats AS default_DOT_hard_hats) AS default_DOT_hard_hat ON default_DOT_repair_order.hard_hat_id = default_DOT_hard_hat.hard_hat_id
LEFT OUTER JOIN (SELECT  default_DOT_municipality.local_region,
    default_DOT_municipality.municipality_id
 FROM roads.municipality AS default_DOT_municipality LEFT  JOIN roads.municipality_municipality_type AS default_DOT_municipality_municipality_type ON default_DOT_municipality.municipality_id = default_DOT_municipality_municipality_type.municipality_id
LEFT  JOIN roads.municipality_type AS default_DOT_municipality_type ON default_DOT_municipality_municipality_type.municipality_type_id = default_DOT_municipality_type.municipality_type_desc) AS default_DOT_municipality_dim ON default_DOT_repair_order.municipality_id = default_DOT_municipality_dim.municipality_id
 WHERE  default_DOT_hard_hat.state = 'AZ'
 GROUP BY  default_DOT_hard_hat.country, default_DOT_hard_hat.postal_code, default_DOT_hard_hat.city, default_DOT_hard_hat.state, default_DOT_dispatcher.company_name, default_DOT_municipality_dim.local_region
),
m3_default_DOT_total_repair_cost AS (SELECT  default_DOT_hard_hat.city,
    default_DOT_dispatcher.company_name,
    default_DOT_hard_hat.country,
    sum(default_DOT_repair_order_details.price) default_DOT_total_repair_cost,
    default_DOT_municipality_dim.local_region,
    default_DOT_hard_hat.postal_code,
    default_DOT_hard_hat.state
 FROM roads.repair_order_details AS default_DOT_repair_order_details LEFT OUTER JOIN (SELECT  default_DOT_repair_orders.dispatcher_id,
    default_DOT_repair_orders.hard_hat_id,
    default_DOT_repair_orders.municipality_id,
    default_DOT_repair_orders.repair_order_id
 FROM roads.repair_orders AS default_DOT_repair_orders) AS default_DOT_repair_order ON default_DOT_repair_order_details.repair_order_id = default_DOT_repair_order.repair_order_id
LEFT OUTER JOIN (SELECT  default_DOT_dispatchers.company_name,
    default_DOT_dispatchers.dispatcher_id
 FROM roads.dispatchers AS default_DOT_dispatchers) AS default_DOT_dispatcher ON default_DOT_repair_order.dispatcher_id = default_DOT_dispatcher.dispatcher_id
LEFT OUTER JOIN (SELECT  default_DOT_hard_hats.city,
    default_DOT_hard_hats.country,
    default_DOT_hard_hats.hard_hat_id,
    default_DOT_hard_hats.postal_code,
    default_DOT_hard_hats.state
 FROM roads.hard_hats AS default_DOT_hard_hats) AS default_DOT_hard_hat ON default_DOT_repair_order.hard_hat_id = default_DOT_hard_hat.hard_hat_id
LEFT OUTER JOIN (SELECT  default_DOT_municipality.local_region,
    default_DOT_municipality.municipality_id
 FROM roads.municipality AS default_DOT_municipality LEFT  JOIN roads.municipality_municipality_type AS default_DOT_municipality_municipality_type ON default_DOT_municipality.municipality_id = default_DOT_municipality_municipality_type.municipality_id
LEFT  JOIN roads.municipality_type AS default_DOT_municipality_type ON default_DOT_municipality_municipality_type.municipality_type_id = default_DOT_municipality_type.municipality_type_desc) AS default_DOT_municipality_dim ON default_DOT_repair_order.municipality_id = default_DOT_municipality_dim.municipality_id
 WHERE  default_DOT_hard_hat.state = 'AZ'
 GROUP BY  default_DOT_hard_hat.country, default_DOT_hard_hat.postal_code, default_DOT_hard_hat.city, default_DOT_hard_hat.state, default_DOT_dispatcher.company_name, default_DOT_municipality_dim.local_region
),
m4_default_DOT_total_repair_order_discounts AS (SELECT  default_DOT_hard_hat.city,
    default_DOT_dispatcher.company_name,
    default_DOT_hard_hat.country,
    sum(default_DOT_repair_order_details.price * default_DOT_repair_order_details.discount) default_DOT_total_repair_order_discounts,
    default_DOT_municipality_dim.local_region,
    default_DOT_hard_hat.postal_code,
    default_DOT_hard_hat.state
 FROM roads.repair_order_details AS default_DOT_repair_order_details LEFT OUTER JOIN (SELECT  default_DOT_repair_orders.dispatcher_id,
    default_DOT_repair_orders.hard_hat_id,
    default_DOT_repair_orders.municipality_id,
    default_DOT_repair_orders.repair_order_id
 FROM roads.repair_orders AS default_DOT_repair_orders) AS default_DOT_repair_order ON default_DOT_repair_order_details.repair_order_id = default_DOT_repair_order.repair_order_id
LEFT OUTER JOIN (SELECT  default_DOT_dispatchers.company_name,
    default_DOT_dispatchers.dispatcher_id
 FROM roads.dispatchers AS default_DOT_dispatchers) AS default_DOT_dispatcher ON default_DOT_repair_order.dispatcher_id = default_DOT_dispatcher.dispatcher_id
LEFT OUTER JOIN (SELECT  default_DOT_hard_hats.city,
    default_DOT_hard_hats.country,
    default_DOT_hard_hats.hard_hat_id,
    default_DOT_hard_hats.postal_code,
    default_DOT_hard_hats.state
 FROM roads.hard_hats AS default_DOT_hard_hats) AS default_DOT_hard_hat ON default_DOT_repair_order.hard_hat_id = default_DOT_hard_hat.hard_hat_id
LEFT OUTER JOIN (SELECT  default_DOT_municipality.local_region,
    default_DOT_municipality.municipality_id
 FROM roads.municipality AS default_DOT_municipality LEFT  JOIN roads.municipality_municipality_type AS default_DOT_municipality_municipality_type ON default_DOT_municipality.municipality_id = default_DOT_municipality_municipality_type.municipality_id
LEFT  JOIN roads.municipality_type AS default_DOT_municipality_type ON default_DOT_municipality_municipality_type.municipality_type_id = default_DOT_municipality_type.municipality_type_desc) AS default_DOT_municipality_dim ON default_DOT_repair_order.municipality_id = default_DOT_municipality_dim.municipality_id
 WHERE  default_DOT_hard_hat.state = 'AZ'
 GROUP BY  default_DOT_hard_hat.country, default_DOT_hard_hat.postal_code, default_DOT_hard_hat.city, default_DOT_hard_hat.state, default_DOT_dispatcher.company_name, default_DOT_municipality_dim.local_region
),
m5_default_DOT_double_total_repair_cost AS (SELECT  default_DOT_hard_hat.city,
    default_DOT_dispatcher.company_name,
    default_DOT_hard_hat.country,
    sum(default_DOT_repair_order_details.price) + sum(default_DOT_repair_order_details.price) AS default_DOT_double_total_repair_cost,
    default_DOT_municipality_dim.local_region,
    default_DOT_hard_hat.postal_code,
    default_DOT_hard_hat.state
 FROM roads.repair_order_details AS default_DOT_repair_order_details LEFT OUTER JOIN (SELECT  default_DOT_repair_orders.dispatcher_id,
    default_DOT_repair_orders.hard_hat_id,
    default_DOT_repair_orders.municipality_id,
    default_DOT_repair_orders.repair_order_id
 FROM roads.repair_orders AS default_DOT_repair_orders) AS default_DOT_repair_order ON default_DOT_repair_order_details.repair_order_id = default_DOT_repair_order.repair_order_id
LEFT OUTER JOIN (SELECT  default_DOT_dispatchers.company_name,
    default_DOT_dispatchers.dispatcher_id
 FROM roads.dispatchers AS default_DOT_dispatchers) AS default_DOT_dispatcher ON default_DOT_repair_order.dispatcher_id = default_DOT_dispatcher.dispatcher_id
LEFT OUTER JOIN (SELECT  default_DOT_hard_hats.city,
    default_DOT_hard_hats.country,
    default_DOT_hard_hats.hard_hat_id,
    default_DOT_hard_hats.postal_code,
    default_DOT_hard_hats.state
 FROM roads.hard_hats AS default_DOT_hard_hats) AS default_DOT_hard_hat ON default_DOT_repair_order.hard_hat_id = default_DOT_hard_hat.hard_hat_id
LEFT OUTER JOIN (SELECT  default_DOT_municipality.local_region,
    default_DOT_municipality.municipality_id
 FROM roads.municipality AS default_DOT_municipality LEFT  JOIN roads.municipality_municipality_type AS default_DOT_municipality_municipality_type ON default_DOT_municipality.municipality_id = default_DOT_municipality_municipality_type.municipality_id
LEFT  JOIN roads.municipality_type AS default_DOT_municipality_type ON default_DOT_municipality_municipality_type.municipality_type_id = default_DOT_municipality_type.municipality_type_desc) AS default_DOT_municipality_dim ON default_DOT_repair_order.municipality_id = default_DOT_municipality_dim.municipality_id
 WHERE  default_DOT_hard_hat.state = 'AZ'
 GROUP BY  default_DOT_hard_hat.country, default_DOT_hard_hat.postal_code, default_DOT_hard_hat.city, default_DOT_hard_hat.state, default_DOT_dispatcher.company_name, default_DOT_municipality_dim.local_region
)SELECT  m0_default_DOT_discounted_orders_rate.default_DOT_discounted_orders_rate,
    m1_default_DOT_num_repair_orders.default_DOT_num_repair_orders,
    m2_default_DOT_avg_repair_price.default_DOT_avg_repair_price,
    m3_default_DOT_total_repair_cost.default_DOT_total_repair_cost,
    m4_default_DOT_total_repair_order_discounts.default_DOT_total_repair_order_discounts,
    m5_default_DOT_double_total_repair_cost.default_DOT_double_total_repair_cost,
    COALESCE(m0_default_DOT_discounted_orders_rate.city, m1_default_DOT_num_repair_orders.city, m2_default_DOT_avg_repair_price.city, m3_default_DOT_total_repair_cost.city, m4_default_DOT_total_repair_order_discounts.city, m5_default_DOT_double_total_repair_cost.city) city,
    COALESCE(m0_default_DOT_discounted_orders_rate.company_name, m1_default_DOT_num_repair_orders.company_name, m2_default_DOT_avg_repair_price.company_name, m3_default_DOT_total_repair_cost.company_name, m4_default_DOT_total_repair_order_discounts.company_name, m5_default_DOT_double_total_repair_cost.company_name) company_name,
    COALESCE(m0_default_DOT_discounted_orders_rate.country, m1_default_DOT_num_repair_orders.country, m2_default_DOT_avg_repair_price.country, m3_default_DOT_total_repair_cost.country, m4_default_DOT_total_repair_order_discounts.country, m5_default_DOT_double_total_repair_cost.country) country,
    COALESCE(m0_default_DOT_discounted_orders_rate.local_region, m1_default_DOT_num_repair_orders.local_region, m2_default_DOT_avg_repair_price.local_region, m3_default_DOT_total_repair_cost.local_region, m4_default_DOT_total_repair_order_discounts.local_region, m5_default_DOT_double_total_repair_cost.local_region) local_region,
    COALESCE(m0_default_DOT_discounted_orders_rate.postal_code, m1_default_DOT_num_repair_orders.postal_code, m2_default_DOT_avg_repair_price.postal_code, m3_default_DOT_total_repair_cost.postal_code, m4_default_DOT_total_repair_order_discounts.postal_code, m5_default_DOT_double_total_repair_cost.postal_code) postal_code,
    COALESCE(m0_default_DOT_discounted_orders_rate.state, m1_default_DOT_num_repair_orders.state, m2_default_DOT_avg_repair_price.state, m3_default_DOT_total_repair_cost.state, m4_default_DOT_total_repair_order_discounts.state, m5_default_DOT_double_total_repair_cost.state) state
 FROM m0_default_DOT_discounted_orders_rate FULL OUTER JOIN m1_default_DOT_num_repair_orders ON m0_default_DOT_discounted_orders_rate.city = m1_default_DOT_num_repair_orders.city AND m0_default_DOT_discounted_orders_rate.company_name = m1_default_DOT_num_repair_orders.company_name AND m0_default_DOT_discounted_orders_rate.country = m1_default_DOT_num_repair_orders.country AND m0_default_DOT_discounted_orders_rate.local_region = m1_default_DOT_num_repair_orders.local_region AND m0_default_DOT_discounted_orders_rate.postal_code = m1_default_DOT_num_repair_orders.postal_code AND m0_default_DOT_discounted_orders_rate.state = m1_default_DOT_num_repair_orders.state
FULL OUTER JOIN m2_default_DOT_avg_repair_price ON m0_default_DOT_discounted_orders_rate.city = m2_default_DOT_avg_repair_price.city AND m0_default_DOT_discounted_orders_rate.company_name = m2_default_DOT_avg_repair_price.company_name AND m0_default_DOT_discounted_orders_rate.country = m2_default_DOT_avg_repair_price.country AND m0_default_DOT_discounted_orders_rate.local_region = m2_default_DOT_avg_repair_price.local_region AND m0_default_DOT_discounted_orders_rate.postal_code = m2_default_DOT_avg_repair_price.postal_code AND m0_default_DOT_discounted_orders_rate.state = m2_default_DOT_avg_repair_price.state
FULL OUTER JOIN m3_default_DOT_total_repair_cost ON m0_default_DOT_discounted_orders_rate.city = m3_default_DOT_total_repair_cost.city AND m0_default_DOT_discounted_orders_rate.company_name = m3_default_DOT_total_repair_cost.company_name AND m0_default_DOT_discounted_orders_rate.country = m3_default_DOT_total_repair_cost.country AND m0_default_DOT_discounted_orders_rate.local_region = m3_default_DOT_total_repair_cost.local_region AND m0_default_DOT_discounted_orders_rate.postal_code = m3_default_DOT_total_repair_cost.postal_code AND m0_default_DOT_discounted_orders_rate.state = m3_default_DOT_total_repair_cost.state
FULL OUTER JOIN m4_default_DOT_total_repair_order_discounts ON m0_default_DOT_discounted_orders_rate.city = m4_default_DOT_total_repair_order_discounts.city AND m0_default_DOT_discounted_orders_rate.company_name = m4_default_DOT_total_repair_order_discounts.company_name AND m0_default_DOT_discounted_orders_rate.country = m4_default_DOT_total_repair_order_discounts.country AND m0_default_DOT_discounted_orders_rate.local_region = m4_default_DOT_total_repair_order_discounts.local_region AND m0_default_DOT_discounted_orders_rate.postal_code = m4_default_DOT_total_repair_order_discounts.postal_code AND m0_default_DOT_discounted_orders_rate.state = m4_default_DOT_total_repair_order_discounts.state
FULL OUTER JOIN m5_default_DOT_double_total_repair_cost ON m0_default_DOT_discounted_orders_rate.city = m5_default_DOT_double_total_repair_cost.city AND m0_default_DOT_discounted_orders_rate.company_name = m5_default_DOT_double_total_repair_cost.company_name AND m0_default_DOT_discounted_orders_rate.country = m5_default_DOT_double_total_repair_cost.country AND m0_default_DOT_discounted_orders_rate.local_region = m5_default_DOT_double_total_repair_cost.local_region AND m0_default_DOT_discounted_orders_rate.postal_code = m5_default_DOT_double_total_repair_cost.postal_code AND m0_default_DOT_discounted_orders_rate.state = m5_default_DOT_double_total_repair_cost.state

    """
    print("RES", results["query"])
    assert compare_query_strings(results["query"], expected_query)


def test_cube_materialization_sql_and_measures(
    client_with_repairs_cube: TestClient,  # pylint: disable=redefined-outer-name
):
    """
    Verifies a cube's materialization SQL + measures
    """
    response = client_with_repairs_cube.get("/cubes/default.repairs_cube/")
    data = response.json()
    assert data["cube_elements"] == [
        {
            "name": "default_DOT_discounted_orders_rate",
            "node_name": "default.discounted_orders_rate",
            "type": "metric",
        },
        {
            "name": "default_DOT_num_repair_orders",
            "node_name": "default.num_repair_orders",
            "type": "metric",
        },
        {
            "name": "default_DOT_avg_repair_price",
            "node_name": "default.avg_repair_price",
            "type": "metric",
        },
        {
            "name": "default_DOT_total_repair_cost",
            "node_name": "default.total_repair_cost",
            "type": "metric",
        },
        {
            "name": "default_DOT_total_repair_order_discounts",
            "node_name": "default.total_repair_order_discounts",
            "type": "metric",
        },
        {
            "name": "default_DOT_double_total_repair_cost",
            "node_name": "default.double_total_repair_cost",
            "type": "metric",
        },
        {"name": "country", "node_name": "default.hard_hat", "type": "dimension"},
        {"name": "postal_code", "node_name": "default.hard_hat", "type": "dimension"},
        {"name": "city", "node_name": "default.hard_hat", "type": "dimension"},
        {"name": "state", "node_name": "default.hard_hat", "type": "dimension"},
        {
            "name": "company_name",
            "node_name": "default.dispatcher",
            "type": "dimension",
        },
        {
            "name": "local_region",
            "node_name": "default.municipality_dim",
            "type": "dimension",
        },
    ]
    expected_materialization_query = """
WITH
m0_default_DOT_discounted_orders_rate AS (SELECT  default_DOT_dispatcher.company_name,
    count(*) placeholder_count,
    default_DOT_hard_hat.country,
    sum(if(default_DOT_repair_order_details.discount > 0.0, 1, 0)) discount_sum,
    default_DOT_hard_hat.city,
    default_DOT_hard_hat.postal_code,
    default_DOT_municipality_dim.local_region,
    default_DOT_hard_hat.state
 FROM roads.repair_order_details AS default_DOT_repair_order_details LEFT OUTER JOIN (SELECT  default_DOT_repair_orders.dispatcher_id,
    default_DOT_repair_orders.hard_hat_id,
    default_DOT_repair_orders.municipality_id,
    default_DOT_repair_orders.repair_order_id
 FROM roads.repair_orders AS default_DOT_repair_orders) AS default_DOT_repair_order ON default_DOT_repair_order_details.repair_order_id = default_DOT_repair_order.repair_order_id
LEFT OUTER JOIN (SELECT  default_DOT_dispatchers.company_name,
    default_DOT_dispatchers.dispatcher_id
 FROM roads.dispatchers AS default_DOT_dispatchers) AS default_DOT_dispatcher ON default_DOT_repair_order.dispatcher_id = default_DOT_dispatcher.dispatcher_id
LEFT OUTER JOIN (SELECT  default_DOT_hard_hats.city,
    default_DOT_hard_hats.country,
    default_DOT_hard_hats.hard_hat_id,
    default_DOT_hard_hats.postal_code,
    default_DOT_hard_hats.state
 FROM roads.hard_hats AS default_DOT_hard_hats) AS default_DOT_hard_hat ON default_DOT_repair_order.hard_hat_id = default_DOT_hard_hat.hard_hat_id
LEFT OUTER JOIN (SELECT  default_DOT_municipality.local_region,
    default_DOT_municipality.municipality_id
 FROM roads.municipality AS default_DOT_municipality LEFT  JOIN roads.municipality_municipality_type AS default_DOT_municipality_municipality_type ON default_DOT_municipality.municipality_id = default_DOT_municipality_municipality_type.municipality_id
LEFT  JOIN roads.municipality_type AS default_DOT_municipality_type ON default_DOT_municipality_municipality_type.municipality_type_id = default_DOT_municipality_type.municipality_type_desc) AS default_DOT_municipality_dim ON default_DOT_repair_order.municipality_id = default_DOT_municipality_dim.municipality_id
 WHERE  default_DOT_hard_hat.state = 'AZ'
 GROUP BY  default_DOT_hard_hat.country, default_DOT_hard_hat.postal_code, default_DOT_hard_hat.city, default_DOT_hard_hat.state, default_DOT_dispatcher.company_name, default_DOT_municipality_dim.local_region
),
m1_default_DOT_num_repair_orders AS (SELECT  default_DOT_dispatcher.company_name,
    count(default_DOT_repair_orders.repair_order_id) repair_order_id_count,
    default_DOT_hard_hat.country,
    default_DOT_hard_hat.city,
    default_DOT_hard_hat.postal_code,
    default_DOT_municipality_dim.local_region,
    default_DOT_hard_hat.state
 FROM roads.repair_orders AS default_DOT_repair_orders LEFT OUTER JOIN (SELECT  default_DOT_dispatchers.company_name,
    default_DOT_dispatchers.dispatcher_id
 FROM roads.dispatchers AS default_DOT_dispatchers) AS default_DOT_dispatcher ON default_DOT_repair_orders.dispatcher_id = default_DOT_dispatcher.dispatcher_id
LEFT OUTER JOIN (SELECT  default_DOT_hard_hats.city,
    default_DOT_hard_hats.country,
    default_DOT_hard_hats.hard_hat_id,
    default_DOT_hard_hats.postal_code,
    default_DOT_hard_hats.state
 FROM roads.hard_hats AS default_DOT_hard_hats) AS default_DOT_hard_hat ON default_DOT_repair_orders.hard_hat_id = default_DOT_hard_hat.hard_hat_id
LEFT OUTER JOIN (SELECT  default_DOT_municipality.local_region,
    default_DOT_municipality.municipality_id
 FROM roads.municipality AS default_DOT_municipality LEFT  JOIN roads.municipality_municipality_type AS default_DOT_municipality_municipality_type ON default_DOT_municipality.municipality_id = default_DOT_municipality_municipality_type.municipality_id
LEFT  JOIN roads.municipality_type AS default_DOT_municipality_type ON default_DOT_municipality_municipality_type.municipality_type_id = default_DOT_municipality_type.municipality_type_desc) AS default_DOT_municipality_dim ON default_DOT_repair_orders.municipality_id = default_DOT_municipality_dim.municipality_id
 WHERE  default_DOT_hard_hat.state = 'AZ'
 GROUP BY  default_DOT_hard_hat.country, default_DOT_hard_hat.postal_code, default_DOT_hard_hat.city, default_DOT_hard_hat.state, default_DOT_dispatcher.company_name, default_DOT_municipality_dim.local_region
),
m2_default_DOT_avg_repair_price AS (SELECT  count(default_DOT_repair_order_details.price) price_count,
    default_DOT_dispatcher.company_name,
    sum(default_DOT_repair_order_details.price) price_sum,
    default_DOT_hard_hat.country,
    default_DOT_hard_hat.city,
    default_DOT_hard_hat.postal_code,
    default_DOT_municipality_dim.local_region,
    default_DOT_hard_hat.state
 FROM roads.repair_order_details AS default_DOT_repair_order_details LEFT OUTER JOIN (SELECT  default_DOT_repair_orders.dispatcher_id,
    default_DOT_repair_orders.hard_hat_id,
    default_DOT_repair_orders.municipality_id,
    default_DOT_repair_orders.repair_order_id
 FROM roads.repair_orders AS default_DOT_repair_orders) AS default_DOT_repair_order ON default_DOT_repair_order_details.repair_order_id = default_DOT_repair_order.repair_order_id
LEFT OUTER JOIN (SELECT  default_DOT_dispatchers.company_name,
    default_DOT_dispatchers.dispatcher_id
 FROM roads.dispatchers AS default_DOT_dispatchers) AS default_DOT_dispatcher ON default_DOT_repair_order.dispatcher_id = default_DOT_dispatcher.dispatcher_id
LEFT OUTER JOIN (SELECT  default_DOT_hard_hats.city,
    default_DOT_hard_hats.country,
    default_DOT_hard_hats.hard_hat_id,
    default_DOT_hard_hats.postal_code,
    default_DOT_hard_hats.state
 FROM roads.hard_hats AS default_DOT_hard_hats) AS default_DOT_hard_hat ON default_DOT_repair_order.hard_hat_id = default_DOT_hard_hat.hard_hat_id
LEFT OUTER JOIN (SELECT  default_DOT_municipality.local_region,
    default_DOT_municipality.municipality_id
 FROM roads.municipality AS default_DOT_municipality LEFT  JOIN roads.municipality_municipality_type AS default_DOT_municipality_municipality_type ON default_DOT_municipality.municipality_id = default_DOT_municipality_municipality_type.municipality_id
LEFT  JOIN roads.municipality_type AS default_DOT_municipality_type ON default_DOT_municipality_municipality_type.municipality_type_id = default_DOT_municipality_type.municipality_type_desc) AS default_DOT_municipality_dim ON default_DOT_repair_order.municipality_id = default_DOT_municipality_dim.municipality_id
 WHERE  default_DOT_hard_hat.state = 'AZ'
 GROUP BY  default_DOT_hard_hat.country, default_DOT_hard_hat.postal_code, default_DOT_hard_hat.city, default_DOT_hard_hat.state, default_DOT_dispatcher.company_name, default_DOT_municipality_dim.local_region
),
m3_default_DOT_total_repair_cost AS (SELECT  default_DOT_dispatcher.company_name,
    sum(default_DOT_repair_order_details.price) price_sum,
    default_DOT_hard_hat.country,
    default_DOT_hard_hat.city,
    default_DOT_hard_hat.postal_code,
    default_DOT_municipality_dim.local_region,
    default_DOT_hard_hat.state
 FROM roads.repair_order_details AS default_DOT_repair_order_details LEFT OUTER JOIN (SELECT  default_DOT_repair_orders.dispatcher_id,
    default_DOT_repair_orders.hard_hat_id,
    default_DOT_repair_orders.municipality_id,
    default_DOT_repair_orders.repair_order_id
 FROM roads.repair_orders AS default_DOT_repair_orders) AS default_DOT_repair_order ON default_DOT_repair_order_details.repair_order_id = default_DOT_repair_order.repair_order_id
LEFT OUTER JOIN (SELECT  default_DOT_dispatchers.company_name,
    default_DOT_dispatchers.dispatcher_id
 FROM roads.dispatchers AS default_DOT_dispatchers) AS default_DOT_dispatcher ON default_DOT_repair_order.dispatcher_id = default_DOT_dispatcher.dispatcher_id
LEFT OUTER JOIN (SELECT  default_DOT_hard_hats.city,
    default_DOT_hard_hats.country,
    default_DOT_hard_hats.hard_hat_id,
    default_DOT_hard_hats.postal_code,
    default_DOT_hard_hats.state
 FROM roads.hard_hats AS default_DOT_hard_hats) AS default_DOT_hard_hat ON default_DOT_repair_order.hard_hat_id = default_DOT_hard_hat.hard_hat_id
LEFT OUTER JOIN (SELECT  default_DOT_municipality.local_region,
    default_DOT_municipality.municipality_id
 FROM roads.municipality AS default_DOT_municipality LEFT  JOIN roads.municipality_municipality_type AS default_DOT_municipality_municipality_type ON default_DOT_municipality.municipality_id = default_DOT_municipality_municipality_type.municipality_id
LEFT  JOIN roads.municipality_type AS default_DOT_municipality_type ON default_DOT_municipality_municipality_type.municipality_type_id = default_DOT_municipality_type.municipality_type_desc) AS default_DOT_municipality_dim ON default_DOT_repair_order.municipality_id = default_DOT_municipality_dim.municipality_id
 WHERE  default_DOT_hard_hat.state = 'AZ'
 GROUP BY  default_DOT_hard_hat.country, default_DOT_hard_hat.postal_code, default_DOT_hard_hat.city, default_DOT_hard_hat.state, default_DOT_dispatcher.company_name, default_DOT_municipality_dim.local_region
),
m4_default_DOT_total_repair_order_discounts AS (SELECT  default_DOT_dispatcher.company_name,
    default_DOT_hard_hat.country,
    default_DOT_hard_hat.city,
    default_DOT_hard_hat.postal_code,
    sum(default_DOT_repair_order_details.price * default_DOT_repair_order_details.discount) price_discount_sum,
    default_DOT_municipality_dim.local_region,
    default_DOT_hard_hat.state
 FROM roads.repair_order_details AS default_DOT_repair_order_details LEFT OUTER JOIN (SELECT  default_DOT_repair_orders.dispatcher_id,
    default_DOT_repair_orders.hard_hat_id,
    default_DOT_repair_orders.municipality_id,
    default_DOT_repair_orders.repair_order_id
 FROM roads.repair_orders AS default_DOT_repair_orders) AS default_DOT_repair_order ON default_DOT_repair_order_details.repair_order_id = default_DOT_repair_order.repair_order_id
LEFT OUTER JOIN (SELECT  default_DOT_dispatchers.company_name,
    default_DOT_dispatchers.dispatcher_id
 FROM roads.dispatchers AS default_DOT_dispatchers) AS default_DOT_dispatcher ON default_DOT_repair_order.dispatcher_id = default_DOT_dispatcher.dispatcher_id
LEFT OUTER JOIN (SELECT  default_DOT_hard_hats.city,
    default_DOT_hard_hats.country,
    default_DOT_hard_hats.hard_hat_id,
    default_DOT_hard_hats.postal_code,
    default_DOT_hard_hats.state
 FROM roads.hard_hats AS default_DOT_hard_hats) AS default_DOT_hard_hat ON default_DOT_repair_order.hard_hat_id = default_DOT_hard_hat.hard_hat_id
LEFT OUTER JOIN (SELECT  default_DOT_municipality.local_region,
    default_DOT_municipality.municipality_id
 FROM roads.municipality AS default_DOT_municipality LEFT  JOIN roads.municipality_municipality_type AS default_DOT_municipality_municipality_type ON default_DOT_municipality.municipality_id = default_DOT_municipality_municipality_type.municipality_id
LEFT  JOIN roads.municipality_type AS default_DOT_municipality_type ON default_DOT_municipality_municipality_type.municipality_type_id = default_DOT_municipality_type.municipality_type_desc) AS default_DOT_municipality_dim ON default_DOT_repair_order.municipality_id = default_DOT_municipality_dim.municipality_id
 WHERE  default_DOT_hard_hat.state = 'AZ'
 GROUP BY  default_DOT_hard_hat.country, default_DOT_hard_hat.postal_code, default_DOT_hard_hat.city, default_DOT_hard_hat.state, default_DOT_dispatcher.company_name, default_DOT_municipality_dim.local_region
),
m5_default_DOT_double_total_repair_cost AS (SELECT  default_DOT_dispatcher.company_name,
    sum(default_DOT_repair_order_details.price) price_sum,
    default_DOT_hard_hat.country,
    default_DOT_hard_hat.city,
    default_DOT_hard_hat.postal_code,
    default_DOT_municipality_dim.local_region,
    default_DOT_hard_hat.state
 FROM roads.repair_order_details AS default_DOT_repair_order_details LEFT OUTER JOIN (SELECT  default_DOT_repair_orders.dispatcher_id,
    default_DOT_repair_orders.hard_hat_id,
    default_DOT_repair_orders.municipality_id,
    default_DOT_repair_orders.repair_order_id
 FROM roads.repair_orders AS default_DOT_repair_orders) AS default_DOT_repair_order ON default_DOT_repair_order_details.repair_order_id = default_DOT_repair_order.repair_order_id
LEFT OUTER JOIN (SELECT  default_DOT_dispatchers.company_name,
    default_DOT_dispatchers.dispatcher_id
 FROM roads.dispatchers AS default_DOT_dispatchers) AS default_DOT_dispatcher ON default_DOT_repair_order.dispatcher_id = default_DOT_dispatcher.dispatcher_id
LEFT OUTER JOIN (SELECT  default_DOT_hard_hats.city,
    default_DOT_hard_hats.country,
    default_DOT_hard_hats.hard_hat_id,
    default_DOT_hard_hats.postal_code,
    default_DOT_hard_hats.state
 FROM roads.hard_hats AS default_DOT_hard_hats) AS default_DOT_hard_hat ON default_DOT_repair_order.hard_hat_id = default_DOT_hard_hat.hard_hat_id
LEFT OUTER JOIN (SELECT  default_DOT_municipality.local_region,
    default_DOT_municipality.municipality_id
 FROM roads.municipality AS default_DOT_municipality LEFT  JOIN roads.municipality_municipality_type AS default_DOT_municipality_municipality_type ON default_DOT_municipality.municipality_id = default_DOT_municipality_municipality_type.municipality_id
LEFT  JOIN roads.municipality_type AS default_DOT_municipality_type ON default_DOT_municipality_municipality_type.municipality_type_id = default_DOT_municipality_type.municipality_type_desc) AS default_DOT_municipality_dim ON default_DOT_repair_order.municipality_id = default_DOT_municipality_dim.municipality_id
 WHERE  default_DOT_hard_hat.state = 'AZ'
 GROUP BY  default_DOT_hard_hat.country, default_DOT_hard_hat.postal_code, default_DOT_hard_hat.city, default_DOT_hard_hat.state, default_DOT_dispatcher.company_name, default_DOT_municipality_dim.local_region
)SELECT  m0_default_DOT_discounted_orders_rate.placeholder_count,
    m0_default_DOT_discounted_orders_rate.discount_sum,
    m1_default_DOT_num_repair_orders.repair_order_id_count,
    m2_default_DOT_avg_repair_price.price_count,
    m2_default_DOT_avg_repair_price.price_sum,
    m3_default_DOT_total_repair_cost.price_sum,
    m4_default_DOT_total_repair_order_discounts.price_discount_sum,
    m5_default_DOT_double_total_repair_cost.price_sum,
    COALESCE(m0_default_DOT_discounted_orders_rate.company_name, m1_default_DOT_num_repair_orders.company_name, m2_default_DOT_avg_repair_price.company_name, m3_default_DOT_total_repair_cost.company_name, m4_default_DOT_total_repair_order_discounts.company_name, m5_default_DOT_double_total_repair_cost.company_name) company_name,
    COALESCE(m0_default_DOT_discounted_orders_rate.country, m1_default_DOT_num_repair_orders.country, m2_default_DOT_avg_repair_price.country, m3_default_DOT_total_repair_cost.country, m4_default_DOT_total_repair_order_discounts.country, m5_default_DOT_double_total_repair_cost.country) country,
    COALESCE(m0_default_DOT_discounted_orders_rate.city, m1_default_DOT_num_repair_orders.city, m2_default_DOT_avg_repair_price.city, m3_default_DOT_total_repair_cost.city, m4_default_DOT_total_repair_order_discounts.city, m5_default_DOT_double_total_repair_cost.city) city,
    COALESCE(m0_default_DOT_discounted_orders_rate.postal_code, m1_default_DOT_num_repair_orders.postal_code, m2_default_DOT_avg_repair_price.postal_code, m3_default_DOT_total_repair_cost.postal_code, m4_default_DOT_total_repair_order_discounts.postal_code, m5_default_DOT_double_total_repair_cost.postal_code) postal_code,
    COALESCE(m0_default_DOT_discounted_orders_rate.local_region, m1_default_DOT_num_repair_orders.local_region, m2_default_DOT_avg_repair_price.local_region, m3_default_DOT_total_repair_cost.local_region, m4_default_DOT_total_repair_order_discounts.local_region, m5_default_DOT_double_total_repair_cost.local_region) local_region,
    COALESCE(m0_default_DOT_discounted_orders_rate.state, m1_default_DOT_num_repair_orders.state, m2_default_DOT_avg_repair_price.state, m3_default_DOT_total_repair_cost.state, m4_default_DOT_total_repair_order_discounts.state, m5_default_DOT_double_total_repair_cost.state) state
 FROM m0_default_DOT_discounted_orders_rate FULL OUTER JOIN m1_default_DOT_num_repair_orders ON m0_default_DOT_discounted_orders_rate.city = m1_default_DOT_num_repair_orders.city AND m0_default_DOT_discounted_orders_rate.company_name = m1_default_DOT_num_repair_orders.company_name AND m0_default_DOT_discounted_orders_rate.country = m1_default_DOT_num_repair_orders.country AND m0_default_DOT_discounted_orders_rate.local_region = m1_default_DOT_num_repair_orders.local_region AND m0_default_DOT_discounted_orders_rate.postal_code = m1_default_DOT_num_repair_orders.postal_code AND m0_default_DOT_discounted_orders_rate.state = m1_default_DOT_num_repair_orders.state
FULL OUTER JOIN m2_default_DOT_avg_repair_price ON m0_default_DOT_discounted_orders_rate.city = m2_default_DOT_avg_repair_price.city AND m0_default_DOT_discounted_orders_rate.company_name = m2_default_DOT_avg_repair_price.company_name AND m0_default_DOT_discounted_orders_rate.country = m2_default_DOT_avg_repair_price.country AND m0_default_DOT_discounted_orders_rate.local_region = m2_default_DOT_avg_repair_price.local_region AND m0_default_DOT_discounted_orders_rate.postal_code = m2_default_DOT_avg_repair_price.postal_code AND m0_default_DOT_discounted_orders_rate.state = m2_default_DOT_avg_repair_price.state
FULL OUTER JOIN m3_default_DOT_total_repair_cost ON m0_default_DOT_discounted_orders_rate.city = m3_default_DOT_total_repair_cost.city AND m0_default_DOT_discounted_orders_rate.company_name = m3_default_DOT_total_repair_cost.company_name AND m0_default_DOT_discounted_orders_rate.country = m3_default_DOT_total_repair_cost.country AND m0_default_DOT_discounted_orders_rate.local_region = m3_default_DOT_total_repair_cost.local_region AND m0_default_DOT_discounted_orders_rate.postal_code = m3_default_DOT_total_repair_cost.postal_code AND m0_default_DOT_discounted_orders_rate.state = m3_default_DOT_total_repair_cost.state
FULL OUTER JOIN m4_default_DOT_total_repair_order_discounts ON m0_default_DOT_discounted_orders_rate.city = m4_default_DOT_total_repair_order_discounts.city AND m0_default_DOT_discounted_orders_rate.company_name = m4_default_DOT_total_repair_order_discounts.company_name AND m0_default_DOT_discounted_orders_rate.country = m4_default_DOT_total_repair_order_discounts.country AND m0_default_DOT_discounted_orders_rate.local_region = m4_default_DOT_total_repair_order_discounts.local_region AND m0_default_DOT_discounted_orders_rate.postal_code = m4_default_DOT_total_repair_order_discounts.postal_code AND m0_default_DOT_discounted_orders_rate.state = m4_default_DOT_total_repair_order_discounts.state
FULL OUTER JOIN m5_default_DOT_double_total_repair_cost ON m0_default_DOT_discounted_orders_rate.city = m5_default_DOT_double_total_repair_cost.city AND m0_default_DOT_discounted_orders_rate.company_name = m5_default_DOT_double_total_repair_cost.company_name AND m0_default_DOT_discounted_orders_rate.country = m5_default_DOT_double_total_repair_cost.country AND m0_default_DOT_discounted_orders_rate.local_region = m5_default_DOT_double_total_repair_cost.local_region AND m0_default_DOT_discounted_orders_rate.postal_code = m5_default_DOT_double_total_repair_cost.postal_code AND m0_default_DOT_discounted_orders_rate.state = m5_default_DOT_double_total_repair_cost.state

    """
    assert compare_query_strings(
        data["materialization_configs"][0]["config"]["query"],
        expected_materialization_query,
    )
    assert data["materialization_configs"][0]["job"] == "DefaultCubeMaterialization"
    assert data["materialization_configs"][0]["config"]["measures"] == {
        "default_DOT_avg_repair_price": [
            {
                "name": "price_count",
                "agg": "count",
                "type": "bigint",
            },
            {
                "name": "price_sum",
                "agg": "sum",
                "type": "double",
            },
        ],
        "default_DOT_double_total_repair_cost": [
            {
                "agg": "sum",
                "type": "double",
                "name": "price_sum",
            },
        ],
        "default_DOT_discounted_orders_rate": [
            {
                "agg": "sum",
                "type": "bigint",
                "name": "discount_sum",
            },
            {"agg": "count", "type": "bigint", "name": "placeholder_count"},
        ],
        "default_DOT_num_repair_orders": [
            {
                "name": "repair_order_id_count",
                "agg": "count",
                "type": "bigint",
            },
        ],
        "default_DOT_total_repair_order_discounts": [
            {
                "agg": "sum",
                "type": "double",
                "name": "price_discount_sum",
            },
        ],
        "default_DOT_total_repair_cost": [
            {
                "name": "price_sum",
                "agg": "sum",
                "type": "double",
            },
        ],
    }


def test_add_materialization_cube_failures(
    client_with_repairs_cube: TestClient,  # pylint: disable=redefined-outer-name
):
    """
    Verifies failure modes when adding materialization config to cube nodes
    """
    response = client_with_repairs_cube.post(
        "/nodes/default.repairs_cube/materialization/",
        json={
            "engine": {"name": "druid", "version": ""},
            "config": {},
            "schedule": "",
        },
    )
    assert response.json() == {
        "message": "No change has been made to the materialization config for node "
        "`default.repairs_cube` and engine `druid` as the config does not have valid "
        "configuration for engine `druid`. \nExpecting 'druid' key in `config`.",
    }

    response = client_with_repairs_cube.post(
        "/nodes/default.repairs_cube/materialization/",
        json={
            "engine": {"name": "druid", "version": ""},
            "config": {
                "druid": {"a": "b"},
                "spark": {},
            },
            "schedule": "",
        },
    )
    assert response.json() == {
        "message": "No change has been made to the materialization config for node "
        "`default.repairs_cube` and engine `druid` as the config does not have "
        "valid configuration for engine `druid`. "
        "\n* field required: granularity"
        "\n* field required: timestamp_column",
    }


def test_add_materialization_config_to_cube(
    client_with_repairs_cube: TestClient,  # pylint: disable=redefined-outer-name
    query_service_client: Iterator[QueryServiceClient],
):
    """
    Verifies adding materialization config to a cube
    """
    response = client_with_repairs_cube.post(
        "/nodes/default.repairs_cube/materialization/",
        json={
            "engine": {"name": "druid", "version": ""},
            "config": {
                "druid": {
                    "granularity": "DAY",
                    "timestamp_column": "something",
                },
                "spark": {},
            },
            "schedule": "",
        },
    )
    assert response.json() == {
        "message": "Successfully updated materialization config for node "
        "`default.repairs_cube` and engine `druid`.",
    }
    called_kwargs = [
        call_[1]
        for call_ in query_service_client.materialize_cube.call_args_list  # type: ignore
    ][0]
    assert called_kwargs["node_name"] == "default.repairs_cube"
    assert called_kwargs["node_type"] == "cube"
    assert called_kwargs["schedule"] == "@daily"
    assert called_kwargs["spark_conf"] == {}
    dimensions_sorted = sorted(
        called_kwargs["druid_spec"]["dataSchema"]["parser"]["parseSpec"][
            "dimensionsSpec"
        ]["dimensions"],
    )
    called_kwargs["druid_spec"]["dataSchema"]["parser"]["parseSpec"]["dimensionsSpec"][
        "dimensions"
    ] = dimensions_sorted
    called_kwargs["druid_spec"]["dataSchema"]["metricsSpec"] = sorted(
        called_kwargs["druid_spec"]["dataSchema"]["metricsSpec"],
        key=lambda x: x["fieldName"],
    )
    assert called_kwargs["druid_spec"] == {
        "dataSchema": {
            "dataSource": "default_DOT_repairs_cube",
            "parser": {
                "parseSpec": {
                    "format": "parquet",
                    "dimensionsSpec": {
                        "dimensions": [
                            "city",
                            "company_name",
                            "country",
                            "local_region",
                            "postal_code",
                            "state",
                        ],
                    },
                    "timestampSpec": {"column": "something", "format": "yyyyMMdd"},
                },
            },
            "metricsSpec": [
                {
                    "fieldName": "discount_sum",
                    "name": "discount_sum",
                    "type": "longSum",
                },
                {
                    "fieldName": "placeholder_count",
                    "name": "placeholder_count",
                    "type": "longSum",
                },
                {"fieldName": "price_count", "name": "price_count", "type": "longSum"},
                {
                    "fieldName": "price_discount_sum",
                    "name": "price_discount_sum",
                    "type": "doubleSum",
                },
                {"fieldName": "price_sum", "name": "price_sum", "type": "doubleSum"},
                {
                    "fieldName": "repair_order_id_count",
                    "name": "repair_order_id_count",
                    "type": "longSum",
                },
            ],
            "granularitySpec": {
                "type": "uniform",
                "segmentGranularity": "DAY",
                "intervals": None,
            },
        },
    }
    response = client_with_repairs_cube.get("/nodes/default.repairs_cube/")
    materialization_configs = response.json()["materialization_configs"]
    assert len(materialization_configs) == 2
    druid_materialization = [
        materialization
        for materialization in materialization_configs
        if materialization["engine"]["name"] == "druid"
    ][0]
    assert druid_materialization["engine"] == {
        "name": "druid",
        "version": "",
        "uri": None,
        "dialect": "druid",
    }
    assert set(druid_materialization["config"]["dimensions"]) == {
        "postal_code",
        "city",
        "local_region",
        "country",
        "state",
        "company_name",
    }
    assert druid_materialization["config"]["partitions"] is None
    assert druid_materialization["schedule"] == "@daily"
