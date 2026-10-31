"""Deploy warnings for role-less dimension refs reachable only through roles."""

import pytest

from datajunction_server.models.deployment import (
    ColumnSpec,
    DeploymentSpec,
    DimensionJoinLinkSpec,
    DimensionReferenceLinkSpec,
    DimensionSpec,
    MetricSpec,
    SourceSpec,
)

from tests.api.deployments_test import deploy_and_wait, impact_and_poll

# v3.date reaches order_details only as `order` and page_views_enriched only as
# `page`; v3.product is linked to both without a role.
REVENUE = "SELECT SUM(line_total) FROM v3.order_details"


def _warnings(data) -> list[str]:
    return [w["message"] for w in data["warnings"]]


@pytest.mark.asyncio
async def test_warns_for_each_position(client_with_build_v3):
    spec = DeploymentSpec(
        namespace="rl_positions",
        nodes=[
            # Reaggregate rules are for base metrics only.
            MetricSpec(
                name="base",
                query=REVENUE,
                owners=["dj"],
                fixed_grain=["v3.date.month"],
                reaggregate={
                    "rules": [{"dimension": "v3.date.year", "fn": "last_value"}],
                },
                required_filters=[
                    {"dimension": "v3.date.week", "default": "v3.date.week >= 1"},
                ],
                recommended_filters=[
                    {
                        "dimension": "v3.date.day_of_week",
                        "default": "v3.date.day_of_week >= 1",
                    },
                ],
            ),
            MetricSpec(
                name="wow",
                query=(
                    "SELECT v3.total_revenue - LAG(v3.total_revenue, 1) "
                    "OVER (ORDER BY v3.date.week)"
                ),
                owners=["dj"],
                required_dimensions=["v3.date.week"],
            ),
        ],
    )
    data = await deploy_and_wait(client_with_build_v3, spec)
    assert data["status"] == "success", data["results"]
    node = (await client_with_build_v3.get("/nodes/rl_positions.wow/")).json()
    assert node["status"] == "valid", data["results"]

    head = (
        "uses dimension refs with no role. A ref with no role matches only a "
        "role-free dimension link, so SQL generation will reject these: "
    )
    only = "v3.date is reachable from v3.order_details only through roles: customer->registration, order. "
    assert _warnings(data) == [
        f"Metric rl_positions.base {head}"
        f"fixed_grain `v3.date.month`: {only}Specify one, e.g. v3.date.month[order]; "
        f"reaggregate `v3.date.year`: {only}Specify one, e.g. v3.date.year[order]; "
        f"recommended_filters default `v3.date.day_of_week`: {only}"
        "Specify one, e.g. v3.date.day_of_week[order]; "
        f"required_filters default `v3.date.week`: {only}"
        "Specify one, e.g. v3.date.week[order]",
        f"Metric rl_positions.wow {head}"
        f"required_dimensions `v3.date.week`: {only}"
        "Specify one, e.g. v3.date.week[order]; "
        f"window ORDER BY `v3.date.week`: {only}"
        "Specify one, e.g. v3.date.week[order]",
    ]
    assert {w["code"] for w in data["warnings"]} == {"INVALID_DIMENSION"}

    # Unchanged metrics are not re-checked.
    data = await deploy_and_wait(client_with_build_v3, spec)
    assert data["status"] == "success", data["results"]
    assert data["warnings"] == []


@pytest.mark.asyncio
async def test_no_warning_for_resolvable_refs(client_with_build_v3):
    spec = DeploymentSpec(
        namespace="rl_ok",
        nodes=[
            MetricSpec(
                name="roled",
                query=(
                    "SELECT v3.total_revenue - LAG(v3.total_revenue, 1) "
                    "OVER (ORDER BY v3.date.week[order])"
                ),
                owners=["dj"],
                required_dimensions=["v3.date.week[order]"],
            ),
            # A role-free link, a column of the fact itself, and a wildcard
            # requirement attribute.
            MetricSpec(
                name="plain",
                query=REVENUE,
                owners=["dj"],
                fixed_grain=["v3.product.category"],
                required_dimensions=["v3.product.category", "v3.order_details.status"],
                required_filters=[
                    {"dimension": "v3.date.week"},
                    {
                        "dimension": "v3.date.day_of_week",
                        "default": "v3.date.day_of_week[order] >= 1",
                    },
                ],
                recommended_filters=[
                    {"dimension": ["v3.date.day_of_week", "v3.customer.name"]}
                ],
            ),
            # Aggregate-expression attributes keep any-role resolution.
            MetricSpec(
                name="distinct_weeks",
                query="SELECT COUNT(DISTINCT v3.date.week) FROM v3.order_details",
                owners=["dj"],
            ),
        ],
    )
    data = await deploy_and_wait(client_with_build_v3, spec)
    assert data["status"] == "success", data["results"]
    assert data["warnings"] == []


@pytest.mark.asyncio
async def test_no_warning_for_reference_dimension(client):
    spec = DeploymentSpec(
        namespace="rl_ref",
        nodes=[
            SourceSpec(
                name="events",
                catalog="default",
                schema="rl",
                table="events",
                columns=[
                    ColumnSpec(name="event_id", type="int"),
                    ColumnSpec(name="country_code", type="string"),
                    ColumnSpec(name="amount", type="int"),
                ],
                dimension_links=[
                    DimensionReferenceLinkSpec(
                        node_column="country_code",
                        dimension="${prefix}country.code",
                    ),
                ],
            ),
            SourceSpec(
                name="countries",
                catalog="default",
                schema="rl",
                table="countries",
                columns=[ColumnSpec(name="code", type="string")],
            ),
            DimensionSpec(
                name="country",
                query="SELECT code FROM ${prefix}countries",
                primary_key=["code"],
                owners=["dj"],
            ),
            MetricSpec(
                name="total",
                query="SELECT SUM(amount) FROM ${prefix}events",
                owners=["dj"],
                required_dimensions=["${prefix}country.code"],
            ),
        ],
    )
    data = await deploy_and_wait(client, spec)
    assert data["status"] == "success", data["results"]
    assert data["warnings"] == []


@pytest.mark.asyncio
async def test_no_warning_when_roles_come_from_another_link(client):
    """A role-free join link next to a role link keeps a bare ref valid."""
    spec = DeploymentSpec(
        namespace="rl_mixed",
        nodes=[
            SourceSpec(
                name="orders",
                catalog="default",
                schema="rl",
                table="orders",
                columns=[
                    ColumnSpec(name="order_id", type="int"),
                    ColumnSpec(name="created", type="int"),
                    ColumnSpec(name="shipped", type="int"),
                    ColumnSpec(name="amount", type="int"),
                ],
                dimension_links=[
                    DimensionJoinLinkSpec(
                        dimension_node="${prefix}day",
                        join_on="${prefix}orders.created = ${prefix}day.day_id",
                    ),
                    DimensionJoinLinkSpec(
                        dimension_node="${prefix}day",
                        join_on="${prefix}orders.shipped = ${prefix}day.day_id",
                        role="shipped",
                    ),
                ],
            ),
            SourceSpec(
                name="days",
                catalog="default",
                schema="rl",
                table="days",
                columns=[ColumnSpec(name="day_id", type="int")],
            ),
            DimensionSpec(
                name="day",
                query="SELECT day_id FROM ${prefix}days",
                primary_key=["day_id"],
                owners=["dj"],
            ),
            MetricSpec(
                name="total",
                query="SELECT SUM(amount) FROM ${prefix}orders",
                owners=["dj"],
                required_dimensions=["${prefix}day.day_id"],
            ),
        ],
    )
    data = await deploy_and_wait(client, spec)
    assert data["status"] == "success", data["results"]
    assert data["warnings"] == []


@pytest.mark.asyncio
async def test_derived_metric_names_every_offending_fact(client_with_build_v3):
    spec = DeploymentSpec(
        namespace="rl_derived",
        nodes=[
            MetricSpec(
                name="revenue_per_view",
                query="SELECT v3.total_revenue / v3.page_view_count",
                owners=["dj"],
                required_dimensions=["v3.date.week"],
            ),
        ],
    )
    data = await deploy_and_wait(client_with_build_v3, spec)
    assert data["status"] == "success", data["results"]
    assert _warnings(data) == [
        "Metric rl_derived.revenue_per_view uses dimension refs with no role. A "
        "ref with no role matches only a role-free dimension link, so SQL "
        "generation will reject these: required_dimensions `v3.date.week`: "
        "v3.date is reachable from v3.order_details, v3.page_views_enriched "
        "only through roles: customer->registration, order, page. Specify one, e.g. v3.date.week[order]",
    ]


@pytest.mark.asyncio
async def test_impact_dry_run_reports_the_warning(client_with_build_v3):
    spec = DeploymentSpec(
        namespace="rl_impact",
        nodes=[
            MetricSpec(
                name="revenue",
                query=REVENUE,
                owners=["dj"],
                required_dimensions=["v3.date.week"],
            ),
        ],
    )
    data = await impact_and_poll(client_with_build_v3, spec)
    assert data["status"] == "success", data
    assert len(data["warnings"]) == 1
    assert "rl_impact.revenue" in data["warnings"][0]["message"]
    response = await client_with_build_v3.get("/nodes/rl_impact.revenue/")
    assert response.status_code == 404
