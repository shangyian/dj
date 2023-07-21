"""
Integration tests to be run against the latest full demo datajunction environment
"""
# pylint: disable=too-many-lines
import namesgenerator
import pandas
import pytest

from datajunction import DJReader
from datajunction.exceptions import DJClientException
from datajunction.models import AvailabilityState, ColumnAttribute, NodeMode, NodeStatus


@pytest.mark.skipif("not config.getoption('integration')")
def test_integration():  # pylint: disable=too-many-statements
    """
    Integration test
    """
    dj = DJReader()  # pylint: disable=invalid-name

    # Create a namespace
    namespace = f"integration.python.{namesgenerator.get_random_name()}"
    dj.new_namespace(namespace)

    # List namespaces
    matching_namespace = None
    for existing_namespace in dj.list_namespaces():
        if existing_namespace == namespace:
            matching_namespace = existing_namespace
    assert matching_namespace

    # Create a source
    dj.new_source(
        name=f"{namespace}.repair_orders",
        description="Repair orders",
        catalog="warehouse",
        schema_="roads",
        table="repair_orders",
        columns=[
            {"name": "repair_order_id", "type": "int"},
            {"name": "municipality_id", "type": "string"},
            {"name": "hard_hat_id", "type": "int"},
            {"name": "order_date", "type": "timestamp"},
            {"name": "required_date", "type": "timestamp"},
            {"name": "dispatched_date", "type": "timestamp"},
            {"name": "dispatcher_id", "type": "int"},
        ],
    ).save()

    # Get source
    dj.source(f"{namespace}.repair_orders")

    # Create a transform
    dj.new_transform(
        name=f"{namespace}.repair_orders_w_dispatchers",
        description="Repair orders that have a dispatcher",
        query=f"""
            SELECT
            repair_order_id,
            municipality_id,
            hard_hat_id,
            dispatcher_id
            FROM {namespace}.repair_orders
            WHERE dispatcher_id IS NOT NULL
        """,
    ).save()

    # Get transform
    dj.transform(f"{namespace}.repair_orders_w_dispatchers")

    # Create a source and dimension node
    dj.new_source(
        name=f"{namespace}.dispatchers",
        description="Different third party dispatcher companies that coordinate repairs",
        catalog="warehouse",
        schema_="roads",
        table="dispatchers",
        columns=[
            {"name": "dispatcher_id", "type": "int"},
            {"name": "company_name", "type": "string"},
            {"name": "phone", "type": "string"},
        ],
    ).save()
    dj.new_dimension(
        name=f"{namespace}.all_dispatchers",
        description="All dispatchers",
        primary_key=["dispatcher_id"],
        query=f"""
            SELECT
            dispatcher_id,
            company_name,
            phone
            FROM {namespace}.dispatchers
        """,
    ).save()

    # Get dimension
    dj.dimension(f"{namespace}.all_dispatchers")

    # Create metrics
    metric = dj.new_metric(
        name=f"{namespace}.num_repair_orders",
        description="Number of repair orders",
        query=f"SELECT count(repair_order_id) FROM {namespace}.repair_orders",
    )
    metric.save(NodeMode.PUBLISHED)

    # List metrics
    assert f"{namespace}.num_repair_orders" in dj.list_metrics()

    # Create a dimension link
    source = dj.source(f"{namespace}.repair_orders")
    source.link_dimension(
        column="dispatcher_id",
        dimension=f"{namespace}.all_dispatchers",
        dimension_column="dispatcher_id",
    )
    source.unlink_dimension(
        column="dispatcher_id",
        dimension=f"{namespace}.all_dispatchers",
        dimension_column="dispatcher_id",
    )
    source.link_dimension(
        column="dispatcher_id",
        dimension=f"{namespace}.all_dispatchers",
        dimension_column="dispatcher_id",
    )

    # List dimensions for a metric
    dj.metric(f"{namespace}.num_repair_orders").dimensions()

    # List common dimensions for multiple metrics
    common_dimensions = dj.common_dimensions(
        metrics=[
            "default.num_repair_orders",
            "default.avg_repair_price",
            "default.total_repair_cost",
        ],
    )
    assert common_dimensions == [
        {
            "name": "default.dispatcher.company_name",
            "type": "string",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.dispatcher_id",
            ],
        },
        {
            "name": "default.dispatcher.dispatcher_id",
            "type": "int",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.dispatcher_id",
            ],
        },
        {
            "name": "default.dispatcher.phone",
            "type": "string",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.dispatcher_id",
            ],
        },
        {
            "name": "default.hard_hat.address",
            "type": "string",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.hard_hat_id",
            ],
        },
        {
            "name": "default.hard_hat.birth_date",
            "type": "date",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.hard_hat_id",
            ],
        },
        {
            "name": "default.hard_hat.city",
            "type": "string",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.hard_hat_id",
            ],
        },
        {
            "name": "default.hard_hat.contractor_id",
            "type": "int",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.hard_hat_id",
            ],
        },
        {
            "name": "default.hard_hat.country",
            "type": "string",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.hard_hat_id",
            ],
        },
        {
            "name": "default.hard_hat.first_name",
            "type": "string",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.hard_hat_id",
            ],
        },
        {
            "name": "default.hard_hat.hard_hat_id",
            "type": "int",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.hard_hat_id",
            ],
        },
        {
            "name": "default.hard_hat.hire_date",
            "type": "date",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.hard_hat_id",
            ],
        },
        {
            "name": "default.hard_hat.last_name",
            "type": "string",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.hard_hat_id",
            ],
        },
        {
            "name": "default.hard_hat.manager",
            "type": "int",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.hard_hat_id",
            ],
        },
        {
            "name": "default.hard_hat.postal_code",
            "type": "string",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.hard_hat_id",
            ],
        },
        {
            "name": "default.hard_hat.state",
            "type": "string",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.hard_hat_id",
            ],
        },
        {
            "name": "default.hard_hat.title",
            "type": "string",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.hard_hat_id",
            ],
        },
        {
            "name": "default.municipality_dim.contact_name",
            "type": "string",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.municipality_id",
            ],
        },
        {
            "name": "default.municipality_dim.contact_title",
            "type": "string",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.municipality_id",
            ],
        },
        {
            "name": "default.municipality_dim.local_region",
            "type": "string",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.municipality_id",
            ],
        },
        {
            "name": "default.municipality_dim.municipality_id",
            "type": "string",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.municipality_id",
            ],
        },
        {
            "name": "default.municipality_dim.municipality_type_desc",
            "type": "string",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.municipality_id",
            ],
        },
        {
            "name": "default.municipality_dim.municipality_type_id",
            "type": "string",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.municipality_id",
            ],
        },
        {
            "name": "default.municipality_dim.state_id",
            "type": "int",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.municipality_id",
            ],
        },
        {"name": "default.repair_orders.repair_order_id", "type": "int", "path": []},
        {
            "name": "default.us_state.state_abbr",
            "type": "string",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.hard_hat_id",
                "default.hard_hat.state",
            ],
        },
        {
            "name": "default.us_state.state_id",
            "type": "int",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.hard_hat_id",
                "default.hard_hat.state",
            ],
        },
        {
            "name": "default.us_state.state_name",
            "type": "string",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.hard_hat_id",
                "default.hard_hat.state",
            ],
        },
        {
            "name": "default.us_state.state_region",
            "type": "int",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.hard_hat_id",
                "default.hard_hat.state",
            ],
        },
        {
            "name": "default.us_state.state_region_description",
            "type": "string",
            "path": [
                "default.repair_orders.repair_order_id",
                "default.repair_order.hard_hat_id",
                "default.hard_hat.state",
            ],
        },
    ]

    # Get SQL for a set of metrics and dimensions
    query = dj.sql(
        metrics=[
            "default.num_repair_orders",
            "default.avg_repair_price",
            "default.total_repair_cost",
        ],
        dimensions=[
            "default.us_state.state_abbr",
            "default.us_state.state_id",
            "default.us_state.state_name",
        ],
        filters=[],
    )
    expected_query = """
    WITH
    m0_default_DOT_num_repair_orders AS (SELECT  default_DOT_us_state.state_abbr,
            default_DOT_us_state.state_id,
            default_DOT_us_state.state_name,
            count(default_DOT_repair_orders.repair_order_id) default_DOT_num_repair_orders
    FROM roads.repair_orders AS default_DOT_repair_orders LEFT OUTER JOIN (SELECT  default_DOT_repair_orders.dispatcher_id,
            default_DOT_repair_orders.hard_hat_id,
            default_DOT_repair_orders.municipality_id,
            default_DOT_repair_orders.repair_order_id
    FROM roads.repair_orders AS default_DOT_repair_orders)
    AS default_DOT_repair_order ON default_DOT_repair_orders.repair_order_id = default_DOT_repair_order.repair_order_id
    LEFT OUTER JOIN (SELECT  default_DOT_hard_hats.birth_date,
            default_DOT_hard_hats.hard_hat_id,
            default_DOT_hard_hats.hire_date,
            default_DOT_hard_hats.state
    FROM roads.hard_hats AS default_DOT_hard_hats)
    AS default_DOT_hard_hat ON default_DOT_repair_order.hard_hat_id = default_DOT_hard_hat.hard_hat_id
    LEFT OUTER JOIN (SELECT  default_DOT_us_states.state_abbr,
            default_DOT_us_states.state_id,
            default_DOT_us_states.state_name
    FROM roads.us_states AS default_DOT_us_states LEFT  JOIN roads.us_region AS default_DOT_us_region ON default_DOT_us_states.state_region = default_DOT_us_region.us_region_description)
    AS default_DOT_us_state ON default_DOT_hard_hat.state = default_DOT_us_state.state_abbr
    GROUP BY  default_DOT_us_state.state_abbr, default_DOT_us_state.state_id, default_DOT_us_state.state_name
    ),
    m1_default_DOT_avg_repair_price AS (SELECT  default_DOT_us_state.state_abbr,
            default_DOT_us_state.state_id,
            default_DOT_us_state.state_name,
            avg(default_DOT_repair_order_details.price) default_DOT_avg_repair_price
    FROM roads.repair_order_details AS default_DOT_repair_order_details LEFT OUTER JOIN (SELECT  default_DOT_repair_orders.dispatcher_id,
            default_DOT_repair_orders.hard_hat_id,
            default_DOT_repair_orders.municipality_id,
            default_DOT_repair_orders.repair_order_id
    FROM roads.repair_orders AS default_DOT_repair_orders)
    AS default_DOT_repair_order ON default_DOT_repair_order_details.repair_order_id = default_DOT_repair_order.repair_order_id
    LEFT OUTER JOIN (SELECT  default_DOT_hard_hats.birth_date,
            default_DOT_hard_hats.hard_hat_id,
            default_DOT_hard_hats.hire_date,
            default_DOT_hard_hats.state
    FROM roads.hard_hats AS default_DOT_hard_hats)
    AS default_DOT_hard_hat ON default_DOT_repair_order.hard_hat_id = default_DOT_hard_hat.hard_hat_id
    LEFT OUTER JOIN (SELECT  default_DOT_us_states.state_abbr,
            default_DOT_us_states.state_id,
            default_DOT_us_states.state_name
    FROM roads.us_states AS default_DOT_us_states LEFT  JOIN roads.us_region AS default_DOT_us_region ON default_DOT_us_states.state_region = default_DOT_us_region.us_region_description)
    AS default_DOT_us_state ON default_DOT_hard_hat.state = default_DOT_us_state.state_abbr
    GROUP BY  default_DOT_us_state.state_abbr, default_DOT_us_state.state_id, default_DOT_us_state.state_name
    ),
    m2_default_DOT_total_repair_cost AS (SELECT  default_DOT_us_state.state_abbr,
            default_DOT_us_state.state_id,
            default_DOT_us_state.state_name,
            sum(default_DOT_repair_order_details.price) default_DOT_total_repair_cost
    FROM roads.repair_order_details AS default_DOT_repair_order_details LEFT OUTER JOIN (SELECT  default_DOT_repair_orders.dispatcher_id,
            default_DOT_repair_orders.hard_hat_id,
            default_DOT_repair_orders.municipality_id,
            default_DOT_repair_orders.repair_order_id
    FROM roads.repair_orders AS default_DOT_repair_orders)
    AS default_DOT_repair_order ON default_DOT_repair_order_details.repair_order_id = default_DOT_repair_order.repair_order_id
    LEFT OUTER JOIN (SELECT  default_DOT_hard_hats.birth_date,
            default_DOT_hard_hats.hard_hat_id,
            default_DOT_hard_hats.hire_date,
            default_DOT_hard_hats.state
    FROM roads.hard_hats AS default_DOT_hard_hats)
    AS default_DOT_hard_hat ON default_DOT_repair_order.hard_hat_id = default_DOT_hard_hat.hard_hat_id
    LEFT OUTER JOIN (SELECT  default_DOT_us_states.state_abbr,
            default_DOT_us_states.state_id,
            default_DOT_us_states.state_name
    FROM roads.us_states AS default_DOT_us_states LEFT  JOIN roads.us_region AS default_DOT_us_region ON default_DOT_us_states.state_region = default_DOT_us_region.us_region_description)
    AS default_DOT_us_state ON default_DOT_hard_hat.state = default_DOT_us_state.state_abbr
    GROUP BY  default_DOT_us_state.state_abbr, default_DOT_us_state.state_id, default_DOT_us_state.state_name
    )SELECT  m0_default_DOT_num_repair_orders.default_DOT_num_repair_orders,
            m1_default_DOT_avg_repair_price.default_DOT_avg_repair_price,
            m2_default_DOT_total_repair_cost.default_DOT_total_repair_cost,
            COALESCE(m0_default_DOT_num_repair_orders.state_abbr, m1_default_DOT_avg_repair_price.state_abbr, m2_default_DOT_total_repair_cost.state_abbr) state_abbr,
            COALESCE(m0_default_DOT_num_repair_orders.state_id, m1_default_DOT_avg_repair_price.state_id, m2_default_DOT_total_repair_cost.state_id) state_id,
            COALESCE(m0_default_DOT_num_repair_orders.state_name, m1_default_DOT_avg_repair_price.state_name, m2_default_DOT_total_repair_cost.state_name) state_name
    FROM m0_default_DOT_num_repair_orders FULL OUTER JOIN m1_default_DOT_avg_repair_price ON m0_default_DOT_num_repair_orders.state_abbr = m1_default_DOT_avg_repair_price.state_abbr AND m0_default_DOT_num_repair_orders.state_id = m1_default_DOT_avg_repair_price.state_id AND m0_default_DOT_num_repair_orders.state_name = m1_default_DOT_avg_repair_price.state_name
    FULL OUTER JOIN m2_default_DOT_total_repair_cost ON m0_default_DOT_num_repair_orders.state_abbr = m2_default_DOT_total_repair_cost.state_abbr AND m0_default_DOT_num_repair_orders.state_id = m2_default_DOT_total_repair_cost.state_id AND m0_default_DOT_num_repair_orders.state_name = m2_default_DOT_total_repair_cost.state_name
    """

    assert "".join(query.split()) == "".join(expected_query.split())

    # Get data for a set of metrics and dimensions
    result = dj.data(
        metrics=[
            "default.num_repair_orders",
            "default.avg_repair_price",
            "default.total_repair_cost",
        ],
        dimensions=[
            "default.us_state.state_abbr",
            "default.us_state.state_id",
            "default.us_state.state_name",
        ],
        filters=[],
        async_=False,
    )
    expected_df = pandas.DataFrame.from_dict(
        {
            "default_DOT_num_repair_orders": {
                0: 486,
                1: 486,
                2: 729,
                3: 729,
                4: 1215,
                5: 972,
                6: 243,
                7: 243,
                8: 972,
            },
            "default_DOT_avg_repair_price": {
                0: 65682.0,
                1: 39301.5,
                2: 65595.66666666667,
                3: 76555.33333333333,
                4: 64190.6,
                5: 54672.75,
                6: 53374.0,
                7: 70418.0,
                8: 54083.5,
            },
            "default_DOT_total_repair_cost": {
                0: 31921452.0,
                1: 19100529.0,
                2: 47819241.0,
                3: 55808838.0,
                4: 77991579.0,
                5: 53141913.0,
                6: 12969882.0,
                7: 17111574.0,
                8: 52569162.0,
            },
            "state_abbr": {
                0: "AZ",
                1: "CT",
                2: "GA",
                3: "MA",
                4: "MI",
                5: "NJ",
                6: "NY",
                7: "OK",
                8: "PA",
            },
            "state_id": {0: 3, 1: 7, 2: 11, 3: 22, 4: 23, 5: 31, 6: 33, 7: 37, 8: 39},
            "state_name": {
                0: "Arizona",
                1: "Connecticut",
                2: "Georgia",
                3: "Massachusetts",
                4: "Michigan",
                5: "New Jersey",
                6: "New York",
                7: "Oklahoma",
                8: "Pennsylvania",
            },
        },
    )
    pandas.testing.assert_frame_equal(result, expected_df)

    # Create a transform 2 downstream from a transform 1
    dj.new_transform(
        name=f"{namespace}.repair_orders_w_hard_hats",
        description="Repair orders that have a hard hat assigned",
        query=f"""
            SELECT
            repair_order_id,
            municipality_id,
            hard_hat_id,
            dispatcher_id
            FROM {namespace}.repair_orders_w_dispatchers
            WHERE hard_hat_id IS NOT NULL
        """,
    ).save()

    # Get transform 2 that's downstream from transform 1 and make sure it's valid
    transform_2 = dj.transform(f"{namespace}.repair_orders_w_hard_hats")
    assert transform_2.status == NodeStatus.VALID

    # Add an availability state to the transform
    response = transform_2.add_availability(
        AvailabilityState(
            catalog="default",
            schema_="materialized",
            table="contractor",
            valid_through_ts=1688660209,
        ),
    )
    assert response == {"message": "Availability state successfully posted"}

    # Add an availability state to the transform
    response = transform_2.set_column_attributes(
        [
            ColumnAttribute(
                attribute_type_name="dimension",
                column_name="hard_hat_id",
            ),
        ],
    )
    assert response == [
        {
            "attributes": [
                {"attribute_type": {"name": "dimension", "namespace": "system"}},
            ],
            "dimension": None,
            "name": "hard_hat_id",
            "type": "int",
        },
    ]

    # Create a draft transform 4 that's downstream from a not yet created transform 3
    dj.new_transform(
        name=f"{namespace}.repair_orders_w_repair_order_id",
        description="Repair orders without a null ID",
        query=f"""
            SELECT
            repair_order_id,
            municipality_id,
            hard_hat_id,
            dispatcher_id
            FROM {namespace}.repair_orders_w_municipalities
            WHERE repair_order_id IS NOT NULL
        """,
    ).save(mode=NodeMode.DRAFT)
    transform_4 = dj.transform(f"{namespace}.repair_orders_w_repair_order_id")
    transform_4.sync()
    assert transform_4.mode == NodeMode.DRAFT
    assert transform_4.status == NodeStatus.INVALID
    # Check that transform 4 is invalid because transform 3 does not exist
    with pytest.raises(DJClientException):
        transform_4.check()

    # Create a draft transform 3 that's downstream from transform 2
    dj.new_transform(
        name=f"{namespace}.repair_orders_w_municipalities",
        description="Repair orders that have a municipality listed",
        query=f"""
            SELECT
            repair_order_id,
            municipality_id,
            hard_hat_id,
            dispatcher_id
            FROM {namespace}.repair_orders_w_hard_hats
            WHERE municipality_id IS NOT NULL
        """,
    ).save(NodeMode.DRAFT)
    transform_3 = dj.transform(f"{namespace}.repair_orders_w_municipalities")
    # Check that transform 3 is valid
    assert transform_3.check() == NodeStatus.VALID

    # Check that transform 4 is now valid after transform 3 was created
    transform_4.sync()
    assert transform_4.check() == NodeStatus.VALID

    # Check that publishing transform 3 works
    transform_3.publish()
