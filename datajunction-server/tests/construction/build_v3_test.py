"""
Tests for Build V3 SQL generation.

These tests cover:
- Chunk 1: Minimal Measures SQL (no joins)
- Chunk 2: Dimension Joins
- Chunk 3: Multiple Metrics
"""

import re
import pytest

from datajunction_server.construction.build_v3 import (
    AliasRegistry,
    ScopedAliasRegistry,
)
from datajunction_server.construction.build_v3.builder import parse_dimension_ref
from datajunction_server.sql.parsing.backends.antlr4 import parse as parse_sql


def assert_sql_equal(
    actual_sql: str,
    expected_sql: str,
    normalize_aliases: bool = False,
):
    """
    Assert that two SQL strings are semantically equal.

    Uses the DJ SQL parser to normalize both strings before comparison.
    This handles whitespace differences, keyword casing, etc.

    Args:
        actual_sql: The actual SQL generated
        expected_sql: The expected SQL
        normalize_aliases: If True, normalizes component hash suffixes (e.g., sum_x_abc123 -> sum_x_*)
    """
    if normalize_aliases:
        # Normalize hash-based component names: sum_foo_abc123 -> sum_foo_HASH
        hash_pattern = r"(_[a-f0-9]{8})(?=[\s,)]|$)"
        actual_sql = re.sub(hash_pattern, "_HASH", actual_sql)
        expected_sql = re.sub(hash_pattern, "_HASH", expected_sql)

    actual_parsed = str(parse_sql(actual_sql))
    expected_parsed = str(parse_sql(expected_sql))

    assert actual_parsed == expected_parsed, (
        f"\n\nActual SQL:\n{actual_parsed}\n\nExpected SQL:\n{expected_parsed}"
    )


# =============================================================================
# Alias Registry Tests (Pure Unit Tests)
# =============================================================================


class TestAliasRegistry:
    """Tests for the AliasRegistry class."""

    def test_basic_registration(self):
        """Test basic alias registration."""
        registry = AliasRegistry()

        alias = registry.register("orders.customer.country")
        assert alias == "country"

        # Same semantic name returns same alias
        alias2 = registry.register("orders.customer.country")
        assert alias2 == "country"

    def test_conflict_resolution(self):
        """Test that conflicts are resolved by adding qualifiers."""
        registry = AliasRegistry()

        # First registration gets short name
        alias1 = registry.register("orders.country")
        assert alias1 == "country"

        # Second registration with same ending gets qualified
        alias2 = registry.register("customers.country")
        assert alias2 == "customers_country"

    def test_deep_conflict_resolution(self):
        """Test that deep conflicts get progressively longer names."""
        registry = AliasRegistry()

        alias1 = registry.register("a.b.country")
        assert alias1 == "country"

        alias2 = registry.register("c.b.country")
        assert alias2 == "b_country"

        # Third registration needs even more qualification
        alias3 = registry.register("d.b.country")
        assert alias3 == "d_b_country"

    def test_numeric_fallback(self):
        """Test numeric suffix fallback when all names are taken."""
        registry = AliasRegistry()

        # Register all possible combinations
        registry.register("country")
        registry._used_aliases.add("_country")  # Block all possibilities

        # Force numeric fallback
        alias = registry.register("x.country")
        assert alias == "x_country" or alias.startswith("country_")

    def test_get_alias(self):
        """Test looking up an alias."""
        registry = AliasRegistry()

        registry.register("orders.total")

        assert registry.get_alias("orders.total") == "total"
        assert registry.get_alias("nonexistent") is None

    def test_get_semantic(self):
        """Test reverse lookup from alias to semantic name."""
        registry = AliasRegistry()

        registry.register("orders.total")

        assert registry.get_semantic("total") == "orders.total"
        assert registry.get_semantic("nonexistent") is None

    def test_clean_part(self):
        """Test that invalid characters are cleaned."""
        registry = AliasRegistry()

        alias = registry.register("orders.customer-name")
        assert alias == "customer_name"

        alias = registry.register("orders.some@email")
        assert alias == "some_email"

    def test_role_in_semantic_name(self):
        """Test that roles in semantic names ALWAYS produce role-suffixed aliases."""
        registry = AliasRegistry()

        # Without role gets short name
        alias1 = registry.register("v3.location.country")
        assert alias1 == "country"

        # With role ALWAYS gets role-suffixed name
        alias2 = registry.register("v3.location.country[from]")
        assert alias2 == "country_from"

        alias3 = registry.register("v3.location.country[to]")
        assert alias3 == "country_to"

    def test_role_always_included(self):
        """Test that role is always included in alias, even if first."""
        registry = AliasRegistry()

        # Even the first role-based registration includes the role
        alias1 = registry.register("v3.location.country[from]")
        assert alias1 == "country_from"  # Role always included

        alias2 = registry.register("v3.location.country[to]")
        assert alias2 == "country_to"

    def test_multi_hop_role_path(self):
        """Test that multi-hop role paths use the last role part."""
        registry = AliasRegistry()

        # Role-based registration always includes role suffix
        alias1 = registry.register("v3.date.year[order]")
        assert alias1 == "year_order"

        # Multi-hop path uses last part of role (registration)
        alias2 = registry.register("v3.date.year[customer->registration]")
        assert alias2 == "year_registration"

        # Another multi-hop with different path
        alias3 = registry.register("v3.location.country[customer->home]")
        assert alias3 == "country_home"

        alias4 = registry.register("v3.location.country[from]")
        assert alias4 == "country_from"

    def test_all_roles_same_column(self):
        """Test multiple roles all pointing to the same column."""
        registry = AliasRegistry()

        alias1 = registry.register("v3.location.city[from]")
        assert alias1 == "city_from"

        alias2 = registry.register("v3.location.city[to]")
        assert alias2 == "city_to"

        alias3 = registry.register("v3.location.city[customer->home]")
        assert alias3 == "city_home"


class TestScopedAliasRegistry:
    """Tests for the ScopedAliasRegistry class."""

    def test_push_pop_scope(self):
        """Test scope push and pop."""
        registry = ScopedAliasRegistry()

        # Register in root scope
        alias1 = registry.register("orders.total")
        assert alias1 == "total"

        # Push new scope
        registry.push_scope()
        assert registry.scope_depth == 1

        # Can register same name in new scope (it returns existing)
        alias2 = registry.register("orders.total")
        assert alias2 == "total"

        # Pop scope
        registry.pop_scope()
        assert registry.scope_depth == 0
        assert registry.get_alias("orders.total") == "total"

    def test_pop_empty_scope_raises(self):
        """Test that popping empty scope raises error."""
        registry = ScopedAliasRegistry()

        with pytest.raises(ValueError):
            registry.pop_scope()


# =============================================================================
# BUILD_V3 Example Model Tests
# =============================================================================


class TestBuildV3Example:
    """Tests that verify the BUILD_V3 example model loads correctly."""

    @pytest.mark.asyncio
    async def test_example_nodes_exist(self, client_with_build_v3):
        """Test that all BUILD_V3 nodes are created successfully."""
        # Check source nodes
        response = await client_with_build_v3.get("/nodes/v3.src_orders")
        assert response.status_code == 200
        assert response.json()["name"] == "v3.src_orders"

        response = await client_with_build_v3.get("/nodes/v3.src_order_items")
        assert response.status_code == 200

        response = await client_with_build_v3.get("/nodes/v3.src_page_views")
        assert response.status_code == 200

        # Check transform nodes
        response = await client_with_build_v3.get("/nodes/v3.order_details")
        assert response.status_code == 200
        assert response.json()["type"] == "transform"

        response = await client_with_build_v3.get("/nodes/v3.page_views_enriched")
        assert response.status_code == 200
        assert response.json()["type"] == "transform"

        # Check dimension nodes
        response = await client_with_build_v3.get("/nodes/v3.customer")
        assert response.status_code == 200
        assert response.json()["type"] == "dimension"

        response = await client_with_build_v3.get("/nodes/v3.date")
        assert response.status_code == 200

        response = await client_with_build_v3.get("/nodes/v3.location")
        assert response.status_code == 200

        response = await client_with_build_v3.get("/nodes/v3.product")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_base_metrics_exist(self, client_with_build_v3):
        """Test that base metrics are created."""
        metrics = [
            "v3.total_revenue",
            "v3.total_quantity",
            "v3.order_count",
            "v3.customer_count",
            "v3.page_view_count",
            "v3.visitor_count",
        ]

        for metric_name in metrics:
            response = await client_with_build_v3.get(f"/nodes/{metric_name}")
            assert response.status_code == 200, f"Metric {metric_name} not found"
            assert response.json()["type"] == "metric"

    @pytest.mark.asyncio
    async def test_derived_metrics_exist(self, client_with_build_v3):
        """Test that derived metrics are created."""
        derived_metrics = [
            "v3.avg_order_value",
            "v3.conversion_rate",
            "v3.revenue_per_visitor",
            "v3.wow_revenue_change",
        ]

        for metric_name in derived_metrics:
            response = await client_with_build_v3.get(f"/nodes/{metric_name}")
            assert response.status_code == 200, (
                f"Derived metric {metric_name} not found"
            )

    @pytest.mark.asyncio
    async def test_dimension_links_with_roles(self, client_with_build_v3):
        """Test that dimension links with roles are created correctly."""
        # Get order_details node and check its dimension links
        response = await client_with_build_v3.get("/nodes/v3.order_details")
        assert response.status_code == 200

        node_data = response.json()
        dimension_links = node_data.get("dimension_links", [])

        # Should have links to: customer, date (order), location (from), location (to), product
        link_targets = [
            (link["dimension"]["name"], link.get("role")) for link in dimension_links
        ]

        # Check that we have multiple roles for location
        location_links = [t for t in link_targets if t[0] == "v3.location"]
        assert len(location_links) >= 2, (
            "Should have at least 2 location links (from, to)"
        )

        # Check for date link with order role
        date_links = [t for t in link_targets if t[0] == "v3.date"]
        assert len(date_links) >= 1, "Should have date link"

    @pytest.mark.asyncio
    async def test_customer_dimension_has_secondary_links(self, client_with_build_v3):
        """Test that customer dimension has links to date and location for multi-hop traversal."""
        response = await client_with_build_v3.get("/nodes/v3.customer")
        assert response.status_code == 200

        node_data = response.json()
        dimension_links = node_data.get("dimension_links", [])

        link_targets = [link["dimension"]["name"] for link in dimension_links]

        # Customer should link to date (registration) and location (home)
        assert "v3.date" in link_targets, "Customer should link to date dimension"
        assert "v3.location" in link_targets, (
            "Customer should link to location dimension"
        )


# =============================================================================
# V3 SQL Generation API Tests
# =============================================================================


class TestMeasuresSQLEndpoint:
    """Tests for the /sql/measures/v3/ endpoint."""

    @pytest.mark.asyncio
    async def test_single_metric_single_dimension(self, client_with_build_v3):
        """
        Test the simplest case: one metric, one dimension.
        """
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": ["v3.total_revenue"],
                "dimensions": ["v3.order_details.status"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Parse and compare SQL structure
        # For single-component metrics, we use the metric name (not hash)
        assert_sql_equal(
            data["sql"],
            """
            WITH v3_order_details AS (
                SELECT o.status, oi.quantity * oi.unit_price AS line_total
                FROM v3.src_orders o
                JOIN v3.src_order_items oi ON o.order_id = oi.order_id
            )
            SELECT t1.status, SUM(t1.line_total) total_revenue
            FROM v3_order_details t1
            GROUP BY t1.status
            """,
        )

        assert "_DOT_" not in data["sql"]
        assert data["columns"] == [
            {
                "name": "status",
                "type": "string",
                "column": None,
                "node": None,
                "semantic_entity": "v3.order_details.status",
                "semantic_type": "dimension",
            },
            {
                "name": "total_revenue",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.total_revenue",
                "semantic_type": "metric",
            },
        ]

    @pytest.mark.asyncio
    async def test_no_metrics_raises_error(self, client_with_build_v3):
        """Test that empty metrics raises an error."""
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": [],
                "dimensions": ["v3.order_details.status"],
            },
        )

        # Should return an error (4xx status)
        assert response.status_code >= 400

    @pytest.mark.asyncio
    async def test_nonexistent_metric_raises_error(self, client_with_build_v3):
        """Test that nonexistent metric raises an error."""
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": ["nonexistent.metric"],
                "dimensions": ["v3.order_details.status"],
            },
        )

        # Should return an error
        assert response.status_code >= 400
        assert "not found" in response.text.lower()


class TestMetricsSQLEndpoint:
    """Tests for the /sql/metrics/v3/ endpoint (placeholder for Chunk 5)."""

    @pytest.mark.asyncio
    async def test_not_implemented(self, client_with_build_v3):
        """Test that metrics SQL endpoint returns not implemented error."""
        response = await client_with_build_v3.get(
            "/sql/metrics/v3/",
            params={
                "metrics": ["v3.total_revenue"],
                "dimensions": ["v3.order_details.status"],
            },
        )

        # Should return 501 Not Implemented or similar error
        assert response.status_code >= 400


# =============================================================================
# Chunk 2: Dimension Reference Parsing Tests
# =============================================================================


class TestDimensionRefParsing:
    """Tests for dimension reference parsing."""

    def test_simple_dimension_ref(self):
        """Test parsing a simple dimension reference."""
        ref = parse_dimension_ref("v3.customer.name")
        assert ref.node_name == "v3.customer"
        assert ref.column_name == "name"
        assert ref.role is None

    def test_dimension_ref_with_role(self):
        """Test parsing a dimension reference with role."""
        ref = parse_dimension_ref("v3.date.month[order]")
        assert ref.node_name == "v3.date"
        assert ref.column_name == "month"
        assert ref.role == "order"

    def test_dimension_ref_with_multi_hop_role(self):
        """Test parsing a dimension reference with multi-hop role."""
        ref = parse_dimension_ref("v3.date.month[customer->registration]")
        assert ref.node_name == "v3.date"
        assert ref.column_name == "month"
        assert ref.role == "customer->registration"

    def test_dimension_ref_with_deep_role_path(self):
        """Test parsing dimension reference with deep role path."""
        ref = parse_dimension_ref("v3.location.country[customer->home]")
        assert ref.node_name == "v3.location"
        assert ref.column_name == "country"
        assert ref.role == "customer->home"


# =============================================================================
# Chunk 2: Dimension Join Tests
# =============================================================================


class TestDimensionJoins:
    """Tests for dimension join functionality (Chunk 2)."""

    @pytest.mark.asyncio
    async def test_direct_dimension_join(self, client_with_build_v3):
        """
        Test joining to a dimension via direct link.

        Query: revenue by customer name (requires join to customer dimension)
        """
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": ["v3.total_revenue"],
                "dimensions": ["v3.customer.name"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Parse and compare SQL structure
        # Single-component metrics use metric name (not hash)
        assert_sql_equal(
            data["sql"],
            """
            WITH v3_order_details AS (
                SELECT customer_id, oi.quantity * oi.unit_price AS line_total
                FROM v3.src_orders o
                JOIN v3.src_order_items oi ON o.order_id = oi.order_id
            ),
            v3_customer AS (
                SELECT customer_id, name
                FROM v3.src_customers
            )
            SELECT t2.name, SUM(t1.line_total) total_revenue
            FROM v3_order_details t1
            LEFT OUTER JOIN v3_customer t2 ON t1.customer_id = t2.customer_id
            GROUP BY t2.name
            """,
        )

        assert "_DOT_" not in data["sql"]
        assert data["columns"] == [
            {
                "name": "name",
                "type": "string",
                "column": None,
                "node": None,
                "semantic_entity": "v3.customer.name",
                "semantic_type": "dimension",
            },
            {
                "name": "total_revenue",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.total_revenue",
                "semantic_type": "metric",
            },
        ]

    @pytest.mark.asyncio
    async def test_dimension_join_with_role(self, client_with_build_v3):
        """
        Test joining to a dimension with a specific role.

        Query: revenue by order date month (uses role "order" for date)
        """
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": ["v3.total_revenue"],
                "dimensions": ["v3.date.month[order]"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert_sql_equal(
            data["sql"],
            """
            WITH v3_order_details AS (
                SELECT order_date, oi.quantity * oi.unit_price AS line_total
                FROM v3.src_orders o
                JOIN v3.src_order_items oi ON o.order_id = oi.order_id
            ),
            v3_date AS (
                SELECT date_id, month
                FROM v3.src_dates
            )
            SELECT t2.month AS month_order, SUM(t1.line_total) total_revenue
            FROM v3_order_details t1
            LEFT OUTER JOIN v3_date t2 ON t1.order_date = t2.date_id
            GROUP BY t2.month
            """,
        )

        assert "_DOT_" not in data["sql"]
        assert data["columns"] == [
            {
                "name": "month_order",
                "type": "string",
                "column": None,
                "node": None,
                "semantic_entity": "v3.date.month[order]",
                "semantic_type": "dimension",
            },
            {
                "name": "total_revenue",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.total_revenue",
                "semantic_type": "metric",
            },
        ]

    @pytest.mark.asyncio
    async def test_multiple_dimension_joins_same_dimension(self, client_with_build_v3):
        """
        Test joining to the same dimension twice with different roles.

        Query: revenue by from_location and to_location
        """
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": ["v3.total_revenue"],
                "dimensions": [
                    "v3.location.country[from]",
                    "v3.location.country[to]",
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert_sql_equal(
            data["sql"],
            """
            WITH v3_order_details AS (
                SELECT from_location_id, to_location_id, oi.quantity * oi.unit_price AS line_total
                FROM v3.src_orders o
                JOIN v3.src_order_items oi ON o.order_id = oi.order_id
            ),
            v3_location AS (
                SELECT location_id, country
                FROM v3.src_locations
            )
            SELECT t2.country AS country_from, t3.country AS country_to, SUM(t1.line_total) total_revenue
            FROM v3_order_details t1
            LEFT OUTER JOIN v3_location t2 ON t1.from_location_id = t2.location_id
            LEFT OUTER JOIN v3_location t3 ON t1.to_location_id = t3.location_id
            GROUP BY t2.country, t3.country
            """,
        )

        assert "_DOT_" not in data["sql"]
        assert data["columns"] == [
            {
                "name": "country_from",
                "type": "string",
                "column": None,
                "node": None,
                "semantic_entity": "v3.location.country[from]",
                "semantic_type": "dimension",
            },
            {
                "name": "country_to",
                "type": "string",
                "column": None,
                "node": None,
                "semantic_entity": "v3.location.country[to]",
                "semantic_type": "dimension",
            },
            {
                "name": "total_revenue",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.total_revenue",
                "semantic_type": "metric",
            },
        ]

    @pytest.mark.asyncio
    async def test_mixed_local_and_joined_dimensions(self, client_with_build_v3):
        """
        Test query with both local dimensions and joined dimensions.

        Query: revenue by status (local) and customer name (joined)
        """
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": ["v3.total_revenue"],
                "dimensions": [
                    "v3.order_details.status",  # Local
                    "v3.customer.name",  # Requires join
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert_sql_equal(
            data["sql"],
            """
            WITH v3_order_details AS (
                SELECT o.status, customer_id, oi.quantity * oi.unit_price AS line_total
                FROM v3.src_orders o
                JOIN v3.src_order_items oi ON o.order_id = oi.order_id
            ),
            v3_customer AS (
                SELECT customer_id, name
                FROM v3.src_customers
            )
            SELECT t1.status, t2.name, SUM(t1.line_total) total_revenue
            FROM v3_order_details t1
            LEFT OUTER JOIN v3_customer t2 ON t1.customer_id = t2.customer_id
            GROUP BY t1.status, t2.name
            """,
        )

        assert "_DOT_" not in data["sql"]
        assert data["columns"] == [
            {
                "name": "status",
                "type": "string",
                "column": None,
                "node": None,
                "semantic_entity": "v3.order_details.status",
                "semantic_type": "dimension",
            },
            {
                "name": "name",
                "type": "string",
                "column": None,
                "node": None,
                "semantic_entity": "v3.customer.name",
                "semantic_type": "dimension",
            },
            {
                "name": "total_revenue",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.total_revenue",
                "semantic_type": "metric",
            },
        ]

    @pytest.mark.asyncio
    async def test_product_dimension_join(self, client_with_build_v3):
        """
        Test joining to product dimension.

        Query: revenue by product category
        """
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": ["v3.total_revenue"],
                "dimensions": ["v3.product.category"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert_sql_equal(
            data["sql"],
            """
            WITH v3_order_details AS (
                SELECT product_id, oi.quantity * oi.unit_price AS line_total
                FROM v3.src_orders o
                JOIN v3.src_order_items oi ON o.order_id = oi.order_id
            ),
            v3_product AS (
                SELECT product_id, category
                FROM v3.src_products
            )
            SELECT t2.category, SUM(t1.line_total) total_revenue
            FROM v3_order_details t1
            LEFT OUTER JOIN v3_product t2 ON t1.product_id = t2.product_id
            GROUP BY t2.category
            """,
        )

        assert "_DOT_" not in data["sql"]
        assert data["columns"] == [
            {
                "name": "category",
                "type": "string",
                "column": None,
                "node": None,
                "semantic_entity": "v3.product.category",
                "semantic_type": "dimension",
            },
            {
                "name": "total_revenue",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.total_revenue",
                "semantic_type": "metric",
            },
        ]


# =============================================================================
# Chunk 3: Multiple Metrics Tests
# =============================================================================


class TestMultipleMetrics:
    """Tests for multiple metrics support (Chunk 3)."""

    @pytest.mark.asyncio
    async def test_two_metrics_same_parent(self, client_with_build_v3):
        """
        Test requesting two metrics from the same parent node.

        Query: total_revenue and total_quantity by status
        """
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": ["v3.total_revenue", "v3.total_quantity"],
                "dimensions": ["v3.order_details.status"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Parse and compare SQL structure
        # Both metrics are single-component, so they use metric names
        assert_sql_equal(
            data["sql"],
            """
            WITH v3_order_details AS (
                SELECT o.status, oi.quantity, oi.quantity * oi.unit_price AS line_total
                FROM v3.src_orders o
                JOIN v3.src_order_items oi ON o.order_id = oi.order_id
            )
            SELECT t1.status,
                   SUM(t1.line_total) total_revenue,
                   SUM(t1.quantity) total_quantity
            FROM v3_order_details t1
            GROUP BY t1.status
            """,
        )

        assert "_DOT_" not in data["sql"]
        assert data["columns"] == [
            {
                "name": "status",
                "type": "string",
                "column": None,
                "node": None,
                "semantic_entity": "v3.order_details.status",
                "semantic_type": "dimension",
            },
            {
                "name": "total_revenue",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.total_revenue",
                "semantic_type": "metric",
            },
            {
                "name": "total_quantity",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.total_quantity",
                "semantic_type": "metric",
            },
        ]

    @pytest.mark.asyncio
    async def test_three_metrics_same_parent(self, client_with_build_v3):
        """
        Test requesting three metrics from the same parent node.

        Query: revenue, quantity, and order_count by status
        """
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": ["v3.total_revenue", "v3.total_quantity", "v3.order_count"],
                "dimensions": ["v3.order_details.status"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert_sql_equal(
            data["sql"],
            """
            WITH v3_order_details AS (
                SELECT o.status, oi.quantity, oi.quantity * oi.unit_price AS line_total
                FROM v3.src_orders o
                JOIN v3.src_order_items oi ON o.order_id = oi.order_id
            )
            SELECT t1.status,
                   SUM(t1.line_total) total_revenue,
                   SUM(t1.quantity) total_quantity,
                   COUNT(*) order_count
            FROM v3_order_details t1
            GROUP BY t1.status
            """,
        )

        assert "_DOT_" not in data["sql"]
        assert data["columns"] == [
            {
                "name": "status",
                "type": "string",
                "column": None,
                "node": None,
                "semantic_entity": "v3.order_details.status",
                "semantic_type": "dimension",
            },
            {
                "name": "total_revenue",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.total_revenue",
                "semantic_type": "metric",
            },
            {
                "name": "total_quantity",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.total_quantity",
                "semantic_type": "metric",
            },
            {
                "name": "order_count",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.order_count",
                "semantic_type": "metric",
            },
        ]

    @pytest.mark.asyncio
    async def test_multiple_metrics_with_dimension_join(self, client_with_build_v3):
        """
        Test multiple metrics with a dimension join.

        Query: revenue and quantity by customer name
        """
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": ["v3.total_revenue", "v3.total_quantity"],
                "dimensions": ["v3.customer.name"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert_sql_equal(
            data["sql"],
            """
            WITH v3_order_details AS (
                SELECT customer_id, oi.quantity, oi.quantity * oi.unit_price AS line_total
                FROM v3.src_orders o
                JOIN v3.src_order_items oi ON o.order_id = oi.order_id
            ),
            v3_customer AS (
                SELECT customer_id, name
                FROM v3.src_customers
            )
            SELECT t2.name, SUM(t1.line_total) total_revenue, SUM(t1.quantity) total_quantity
            FROM v3_order_details t1
            LEFT OUTER JOIN v3_customer t2 ON t1.customer_id = t2.customer_id
            GROUP BY t2.name
            """,
        )

        assert "_DOT_" not in data["sql"]
        assert data["columns"] == [
            {
                "name": "name",
                "type": "string",
                "column": None,
                "node": None,
                "semantic_entity": "v3.customer.name",
                "semantic_type": "dimension",
            },
            {
                "name": "total_revenue",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.total_revenue",
                "semantic_type": "metric",
            },
            {
                "name": "total_quantity",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.total_quantity",
                "semantic_type": "metric",
            },
        ]

    @pytest.mark.asyncio
    async def test_cross_fact_metrics_not_supported_yet(self, client_with_build_v3):
        """
        Test that metrics from different parent nodes raise an error.

        Query: total_revenue (from order_details) and page_view_count (from page_views)
        """
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": ["v3.total_revenue", "v3.page_view_count"],
                "dimensions": ["v3.order_details.status"],
            },
        )

        # Should return an error - cross-fact not yet supported
        assert response.status_code >= 400
        assert (
            "same parent" in response.text.lower()
            or "cross-fact" in response.text.lower()
        )

    @pytest.mark.asyncio
    async def test_multiple_metrics_multiple_dimensions(self, client_with_build_v3):
        """
        Test multiple metrics with multiple dimensions.

        Query: revenue and quantity by status and customer name
        """
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": ["v3.total_revenue", "v3.total_quantity"],
                "dimensions": [
                    "v3.order_details.status",
                    "v3.customer.name",
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert_sql_equal(
            data["sql"],
            """
            WITH v3_order_details AS (
                SELECT o.status, customer_id, oi.quantity, oi.quantity * oi.unit_price AS line_total
                FROM v3.src_orders o
                JOIN v3.src_order_items oi ON o.order_id = oi.order_id
            ),
            v3_customer AS (
                SELECT customer_id, name
                FROM v3.src_customers
            )
            SELECT t1.status, t2.name, SUM(t1.line_total) total_revenue, SUM(t1.quantity) total_quantity
            FROM v3_order_details t1
            LEFT OUTER JOIN v3_customer t2 ON t1.customer_id = t2.customer_id
            GROUP BY t1.status, t2.name
            """,
        )

        assert "_DOT_" not in data["sql"]
        assert data["columns"] == [
            {
                "name": "status",
                "type": "string",
                "column": None,
                "node": None,
                "semantic_entity": "v3.order_details.status",
                "semantic_type": "dimension",
            },
            {
                "name": "name",
                "type": "string",
                "column": None,
                "node": None,
                "semantic_entity": "v3.customer.name",
                "semantic_type": "dimension",
            },
            {
                "name": "total_revenue",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.total_revenue",
                "semantic_type": "metric",
            },
            {
                "name": "total_quantity",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.total_quantity",
                "semantic_type": "metric",
            },
        ]

    @pytest.mark.asyncio
    async def test_multi_component_metric(self, client_with_build_v3):
        """
        Test a metric that decomposes into multiple components.

        AVG(unit_price) decomposes into:
        - COUNT(unit_price)
        - SUM(unit_price)

        The measures SQL should output both components with hash-suffixed names,
        and semantic_type should be "metric_component" (not "metric").
        """
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": ["v3.avg_unit_price"],
                "dimensions": ["v3.order_details.status"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Print actual SQL and columns for debugging
        print("SQL:", data["sql"])
        print("Columns:", data["columns"])

        assert "_DOT_" not in data["sql"]

        # Should have 1 dimension + 2 metric components = 3 columns
        assert len(data["columns"]) == 3
        assert data["columns"][0] == {
            "name": "status",
            "type": "string",
            "column": None,
            "node": None,
            "semantic_entity": "v3.order_details.status",
            "semantic_type": "dimension",
        }
        # Components have hash suffixes and semantic_type "metric_component"
        assert data["columns"][1]["semantic_type"] == "metric_component"
        assert data["columns"][2]["semantic_type"] == "metric_component"
        # Names should NOT be just "avg_unit_price" (that's for single-component metrics)
        assert data["columns"][1]["name"] != "avg_unit_price"
        assert data["columns"][2]["name"] != "avg_unit_price"

        # Verify SQL structure (component names have hashes, so we check structure)
        sql_upper = data["sql"].upper()
        assert "COUNT" in sql_upper
        assert "SUM" in sql_upper
        assert "UNIT_PRICE" in sql_upper
        assert "GROUP BY" in sql_upper

    @pytest.mark.asyncio
    async def test_mixed_single_and_multi_component_metrics(self, client_with_build_v3):
        """
        Test mixing single-component metrics with multi-component metrics.

        - total_revenue: single component (SUM) → semantic_type: "metric"
        - avg_unit_price: multi-component (COUNT + SUM) → semantic_type: "metric_component"
        """
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": ["v3.total_revenue", "v3.avg_unit_price"],
                "dimensions": ["v3.order_details.status"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Print actual SQL and columns for debugging
        print("SQL:", data["sql"])
        print("Columns:", data["columns"])

        assert "_DOT_" not in data["sql"]

        # Should have 1 dimension + 1 single-component metric + 2 multi-component metrics = 4 columns
        assert len(data["columns"]) == 4
        assert data["columns"][0] == {
            "name": "status",
            "type": "string",
            "column": None,
            "node": None,
            "semantic_entity": "v3.order_details.status",
            "semantic_type": "dimension",
        }
        # Single-component metric has clean name and type "metric"
        assert data["columns"][1] == {
            "name": "total_revenue",
            "type": "number",
            "column": None,
            "node": None,
            "semantic_entity": "v3.total_revenue",
            "semantic_type": "metric",
        }
        # Multi-component metrics have type "metric_component"
        assert data["columns"][2]["semantic_type"] == "metric_component"
        assert data["columns"][3]["semantic_type"] == "metric_component"

    @pytest.mark.asyncio
    async def test_multiple_metrics_with_same_component(self, client_with_build_v3):
        """
        Test metrics that could theoretically share components.

        - avg_unit_price: decomposes into COUNT(unit_price) + SUM(unit_price)
        - total_unit_price: is just SUM(unit_price) (single component)

        Currently: NO component sharing - SUM(unit_price) appears twice.
        total_unit_price gets clean name (single-component metric).
        avg_unit_price components get hash suffixes (multi-component metric).

        TODO: Consider implementing component deduplication for efficiency.
        """
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": ["v3.avg_unit_price", "v3.total_unit_price"],
                "dimensions": ["v3.order_details.status"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Print actual SQL and columns for debugging
        print("SQL:", data["sql"])
        print("Columns:", data["columns"])

        assert "_DOT_" not in data["sql"]

        # Currently NO component sharing:
        # - avg_unit_price: 2 components (COUNT + SUM) with hash suffixes
        # - total_unit_price: 1 component (SUM) with clean metric name
        # Total: 1 dimension + 3 metric columns = 4 columns
        assert len(data["columns"]) == 4
        assert data["columns"][0] == {
            "name": "status",
            "type": "string",
            "column": None,
            "node": None,
            "semantic_entity": "v3.order_details.status",
            "semantic_type": "dimension",
        }

        # avg_unit_price components have type "metric_component" (multi-component)
        avg_cols = [
            c for c in data["columns"] if c["semantic_type"] == "metric_component"
        ]
        assert len(avg_cols) == 2

        # total_unit_price has type "metric" and clean name (single-component)
        total_col = next(c for c in data["columns"] if c["name"] == "total_unit_price")
        assert total_col["semantic_type"] == "metric"

        # Verify SUM appears TWICE (no sharing currently)
        sql_upper = data["sql"].upper()
        assert sql_upper.count("SUM") == 2  # No sharing - appears twice

    @pytest.mark.asyncio
    async def test_limited_aggregability_count_distinct(self, client_with_build_v3):
        """
        Test LIMITED aggregability metric: COUNT(DISTINCT customer_id).

        For measures SQL to be re-aggregatable, it must include the level column
        (customer_id) in the GROUP BY. This allows downstream to do
        COUNT(DISTINCT customer_id) over the measures output.

        Expected behavior:
        - customer_id is included in SELECT and GROUP BY as a dimension
        - No metric column is output (the metric is just the dimension at this grain)
        - Metrics SQL layer will apply COUNT(DISTINCT customer_id) over this output
        """
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": ["v3.customer_count"],
                "dimensions": ["v3.order_details.status"],
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Print actual SQL and columns for debugging
        print("SQL:", data["sql"])
        print("Columns:", data["columns"])

        assert "_DOT_" not in data["sql"]

        # Verify SQL structure
        assert_sql_equal(
            data["sql"],
            """
            WITH
            v3_order_details AS (
                SELECT o.customer_id, o.status
                FROM v3.src_orders o
                JOIN v3.src_order_items oi ON o.order_id = oi.order_id
            )
            SELECT t1.status, t1.customer_id
            FROM v3_order_details t1
            GROUP BY t1.status, t1.customer_id
            """,
        )

        # Should have: status (explicit dimension) + customer_id (implicit dimension for metric)
        assert data["columns"] == [
            {
                "name": "status",
                "type": "string",
                "column": None,
                "node": None,
                "semantic_entity": "v3.order_details.status",
                "semantic_type": "dimension",
            },
            {
                "name": "customer_id",
                "type": "int",  # customer_id is int in v3.src_orders
                "column": None,
                "node": None,
                "semantic_entity": "v3.order_details.customer_id",
                "semantic_type": "dimension",
            },
        ]


# =============================================================================
# Comprehensive Metric Tests - All Defined Metrics
# =============================================================================


class TestAllMetrics:
    """
    Test all metrics defined in BUILD_V3 example.

    Metrics are organized by type:
    - Base metrics on order_details: total_revenue, total_quantity, order_count, customer_count
    - Base metrics on page_views_enriched: page_view_count, product_view_count, session_count, visitor_count
    - Derived same-fact ratios: avg_order_value, avg_items_per_order, revenue_per_customer, pages_per_session
    - Derived cross-fact ratios: conversion_rate, revenue_per_visitor, revenue_per_page_view
    - Derived period-over-period: wow_revenue_change, wow_order_growth, mom_revenue_change
    """

    # All base metrics from order_details
    ORDER_DETAILS_BASE_METRICS = [
        "v3.total_revenue",
        "v3.total_quantity",
        "v3.order_count",
        "v3.customer_count",
    ]

    # All base metrics from page_views_enriched
    PAGE_VIEWS_BASE_METRICS = [
        "v3.page_view_count",
        "v3.product_view_count",
        "v3.session_count",
        "v3.visitor_count",
    ]

    # Derived metrics - same fact ratios (order_details)
    SAME_FACT_DERIVED_ORDER = [
        "v3.avg_order_value",  # revenue / orders
        "v3.avg_items_per_order",  # quantity / orders
        "v3.revenue_per_customer",  # revenue / customers
    ]

    # Derived metrics - same fact ratios (page_views)
    SAME_FACT_DERIVED_PAGE = [
        "v3.pages_per_session",  # page_views / sessions
    ]

    # Derived metrics - cross-fact ratios
    CROSS_FACT_DERIVED = [
        "v3.conversion_rate",  # orders / visitors (order_details + page_views)
        "v3.revenue_per_visitor",  # revenue / visitors (order_details + page_views)
        "v3.revenue_per_page_view",  # revenue / page_views (order_details + page_views)
    ]

    # Derived metrics - period-over-period (window functions, aggregability: NONE)
    PERIOD_OVER_PERIOD = [
        "v3.wow_revenue_change",
        "v3.wow_order_growth",
        "v3.mom_revenue_change",
    ]

    @pytest.mark.asyncio
    async def test_all_order_details_base_metrics(self, client_with_build_v3):
        """
        Test all base metrics from order_details with multiple dimensions.

        Dimensions: status (local), customer name (joined)
        """
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": self.ORDER_DETAILS_BASE_METRICS,
                "dimensions": [
                    "v3.order_details.status",
                    "v3.customer.name",
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert_sql_equal(
            data["sql"],
            """
            WITH v3_order_details AS (
                SELECT o.status, customer_id, oi.quantity, oi.quantity * oi.unit_price AS line_total
                FROM v3.src_orders o
                JOIN v3.src_order_items oi ON o.order_id = oi.order_id
            ),
            v3_customer AS (
                SELECT customer_id, name
                FROM v3.src_customers
            )
            SELECT t1.status, t2.name,
                   SUM(t1.line_total) total_revenue,
                   SUM(t1.quantity) total_quantity,
                   COUNT(*) order_count,
                   COUNT(DISTINCT t1.customer_id) customer_count
            FROM v3_order_details t1
            LEFT OUTER JOIN v3_customer t2 ON t1.customer_id = t2.customer_id
            GROUP BY t1.status, t2.name
            """,
        )

        assert "_DOT_" not in data["sql"]
        assert data["columns"] == [
            {
                "name": "status",
                "type": "string",
                "column": None,
                "node": None,
                "semantic_entity": "v3.order_details.status",
                "semantic_type": "dimension",
            },
            {
                "name": "name",
                "type": "string",
                "column": None,
                "node": None,
                "semantic_entity": "v3.customer.name",
                "semantic_type": "dimension",
            },
            {
                "name": "total_revenue",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.total_revenue",
                "semantic_type": "metric",
            },
            {
                "name": "total_quantity",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.total_quantity",
                "semantic_type": "metric",
            },
            {
                "name": "order_count",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.order_count",
                "semantic_type": "metric",
            },
            {
                "name": "customer_count",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.customer_count",
                "semantic_type": "metric",
            },
        ]

    @pytest.mark.asyncio
    async def test_all_page_views_base_metrics(self, client_with_build_v3):
        """
        Test all base metrics from page_views_enriched with multiple dimensions.

        Dimensions: device_type (local), is_mobile (local)
        """
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": self.PAGE_VIEWS_BASE_METRICS,
                "dimensions": [
                    "v3.page_views_enriched.device_type",
                    "v3.page_views_enriched.is_mobile",
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert_sql_equal(
            data["sql"],
            """
            WITH v3_page_views_enriched AS (
                SELECT device_type,
                       CASE WHEN device_type IN ('phone', 'tablet') THEN true ELSE false END AS is_mobile,
                       page_id, session_id, visitor_id
                FROM v3.src_page_views
            )
            SELECT t1.device_type, t1.is_mobile,
                   COUNT(*) page_view_count,
                   COUNT(DISTINCT t1.page_id) product_view_count,
                   COUNT(DISTINCT t1.session_id) session_count,
                   COUNT(DISTINCT t1.visitor_id) visitor_count
            FROM v3_page_views_enriched t1
            GROUP BY t1.device_type, t1.is_mobile
            """,
        )

        assert "_DOT_" not in data["sql"]
        assert data["columns"] == [
            {
                "name": "device_type",
                "type": "string",
                "column": None,
                "node": None,
                "semantic_entity": "v3.page_views_enriched.device_type",
                "semantic_type": "dimension",
            },
            {
                "name": "is_mobile",
                "type": "string",
                "column": None,
                "node": None,
                "semantic_entity": "v3.page_views_enriched.is_mobile",
                "semantic_type": "dimension",
            },
            {
                "name": "page_view_count",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.page_view_count",
                "semantic_type": "metric",
            },
            {
                "name": "product_view_count",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.product_view_count",
                "semantic_type": "metric",
            },
            {
                "name": "session_count",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.session_count",
                "semantic_type": "metric",
            },
            {
                "name": "visitor_count",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.visitor_count",
                "semantic_type": "metric",
            },
        ]

    @pytest.mark.asyncio
    async def test_order_details_metrics_with_three_dimensions(
        self,
        client_with_build_v3,
    ):
        """
        Test order_details base metrics with three dimensions:
        - status (local)
        - customer name (joined via customer)
        - product category (joined via product)
        """
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": self.ORDER_DETAILS_BASE_METRICS,
                "dimensions": [
                    "v3.order_details.status",
                    "v3.customer.name",
                    "v3.product.category",
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert_sql_equal(
            data["sql"],
            """
            WITH v3_order_details AS (
                SELECT o.order_id, o.customer_id, o.status, oi.product_id, oi.quantity, oi.quantity * oi.unit_price AS line_total
                FROM v3.src_orders o
                JOIN v3.src_order_items oi ON o.order_id = oi.order_id
            ),
            v3_customer AS (
                SELECT customer_id, name
                FROM v3.src_customers
            ),
            v3_product AS (
                SELECT product_id, category
                FROM v3.src_products
            )
            SELECT t1.status, t2.name, t3.category,
                    t1.order_id, t1.customer_id,
                   SUM(t1.line_total) total_revenue,
                   SUM(t1.quantity) total_quantity
            FROM v3_order_details t1
            LEFT OUTER JOIN v3_customer t2 ON t1.customer_id = t2.customer_id
            LEFT OUTER JOIN v3_product t3 ON t1.product_id = t3.product_id
            GROUP BY t1.status, t2.name, t3.category, t1.order_id, t1.customer_id
            """,
        )
        # http://localhost:8000/sql/measures/v3?metrics=growth.xp.funnel.subscription_net_realized_revenue&metrics=growth.xp.funnel.nonmember_average_service_days&metrics=growth.xp.funnel.nonmember_average_paid_days&metrics=growth.xp.funnel.cumulative_retention_rate&metrics=growth.xp.funnel.current_member_count&metrics=growth.xp.funnel.paid_signup_rate&metrics=growth.xp.funnel.registration_rate&metrics=growth.xp.funnel.voluntary_cancel_rate&metrics=growth.xp.funnel.completed_signup_rate&metrics=growth.xp.funnel.involuntary_cancel_rate&metrics=growth.xp.funnel.provided_mop_rate&metrics=growth.xp.funnel.paid_p2_rate&metrics=growth.xp.funnel.nonmember_average_price_per_signup&metrics=growth.xp.funnel.allocation_count&metrics=growth.xp.funnel.adjusted_allocation_count&dimensions=users.lizzyg.applaunch.partner_name.device_type_id&dimensions=growth.xp.platform.platform&dimensions=growth.xp.device_category.device_category&dimensions=common.dimensions.xp.allocation_region_date.dateint&dimensions=growth.xp.tenure.tenure&dimensions=common.dimensions.xp.is_bot.is_bot&dimensions=common.dimensions.xp.ab_test_cell.cell_id&dimensions=common.dimensions.xp.is_fraud.is_fraud&dimensions=growth.xp.funnel.is_registered.is_registered&dimensions=growth.xp.funnel.market_segmentation.market_segment_name&dimensions=growth.xp.funnel.is_signed_up.is_signed_up&dimensions=growth.xp.funnel.signup_plan_name.signup_plan_name&dimensions=growth.xp.funnel.is_ads_market.is_ads_market&dimensions=users.lizzyg.applaunch.partner_name.partner_name&dimensions=growth.xp.mop_category.mop_category&dimensions=growth.xp.funnel.alloc_membership_status.alloc_membership_status&dimensions=growth.xp.membership_status.membership_status&dimensions=common.dimensions.geo_country.forecast_subregion_desc&dimensions=common.dimensions.xp.ab_test_plan.group
        assert "_DOT_" not in data["sql"]
        assert len(data["columns"]) == 7  # 3 dimensions + 4 metrics

        # Verify dimension columns
        dim_entities = [
            c["semantic_entity"]
            for c in data["columns"]
            if c["semantic_type"] == "dimension"
        ]
        assert "v3.order_details.status" in dim_entities
        assert "v3.customer.name" in dim_entities
        assert "v3.product.category" in dim_entities

        # Verify metric columns
        metric_names = [
            c["name"] for c in data["columns"] if c["semantic_type"] == "metric"
        ]
        assert set(metric_names) == {
            "total_revenue",
            "total_quantity",
            "order_count",
            "customer_count",
        }

    @pytest.mark.asyncio
    async def test_derived_same_fact_metrics_not_yet_supported(
        self,
        client_with_build_v3,
    ):
        """
        Test derived metrics (same-fact ratios) - expected to fail until derived metrics are implemented.

        These metrics (avg_order_value, avg_items_per_order, etc.) reference other metrics
        in their definitions and require the metrics SQL endpoint, not measures.
        """
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": self.SAME_FACT_DERIVED_ORDER,
                "dimensions": ["v3.order_details.status"],
            },
        )

        # Derived metrics should fail on measures endpoint
        # (they need build_metrics_sql which computes final expressions)
        assert response.status_code >= 400

    @pytest.mark.asyncio
    async def test_cross_fact_derived_metrics_not_yet_supported(
        self,
        client_with_build_v3,
    ):
        """
        Test cross-fact derived metrics - expected to fail as cross-fact is not yet supported.

        These metrics (conversion_rate, revenue_per_visitor, etc.) combine metrics
        from different parent nodes (order_details and page_views_enriched).
        """
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": self.CROSS_FACT_DERIVED,
                "dimensions": ["v3.order_details.status"],
            },
        )

        # Cross-fact metrics should fail
        assert response.status_code >= 400

    @pytest.mark.asyncio
    async def test_period_over_period_metrics_not_yet_supported(
        self,
        client_with_build_v3,
    ):
        """
        Test period-over-period metrics with window functions - expected to fail.

        These metrics (wow_revenue_change, etc.) use LAG() window functions and have
        aggregability: NONE, meaning they cannot be pre-aggregated.
        """
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": self.PERIOD_OVER_PERIOD,
                "dimensions": ["v3.date.week[order]"],
            },
        )

        # Period-over-period metrics with window functions should fail on measures endpoint
        assert response.status_code >= 400

    @pytest.mark.asyncio
    async def test_mixing_base_metrics_from_different_facts(self, client_with_build_v3):
        """
        Test mixing base metrics from different parent nodes - expected to fail.

        Cannot combine order_details metrics with page_views metrics in same query (yet).
        """
        mixed_metrics = [
            "v3.total_revenue",  # from order_details
            "v3.page_view_count",  # from page_views_enriched
        ]

        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": mixed_metrics,
                "dimensions": ["v3.order_details.status"],
            },
        )

        # Should fail - cross-fact not supported
        assert response.status_code >= 400
        assert (
            "same parent" in response.text.lower()
            or "cross-fact" in response.text.lower()
        )

    @pytest.mark.asyncio
    async def test_dimensions_with_multiple_roles_same_dimension(
        self,
        client_with_build_v3,
    ):
        """
        Test querying with multiple roles to the same dimension type.

        Uses both from_location and to_location (both link to v3.location with different roles).
        Also includes the order date.
        """
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": ["v3.total_revenue", "v3.order_count"],
                "dimensions": [
                    "v3.date.month[order]",  # Order date month
                    "v3.location.country[from]",  # From location country
                    "v3.location.country[to]",  # To location country
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert_sql_equal(
            data["sql"],
            """
            WITH v3_order_details AS (
                SELECT order_date, from_location_id, to_location_id, oi.quantity * oi.unit_price AS line_total
                FROM v3.src_orders o
                JOIN v3.src_order_items oi ON o.order_id = oi.order_id
            ),
            v3_date AS (
                SELECT date_id, month
                FROM v3.src_dates
            ),
            v3_location AS (
                SELECT location_id, country
                FROM v3.src_locations
            )
            SELECT t2.month AS month_order, t3.country AS country_from, t4.country AS country_to,
                   SUM(t1.line_total) total_revenue,
                   COUNT(*) order_count
            FROM v3_order_details t1
            LEFT OUTER JOIN v3_date t2 ON t1.order_date = t2.date_id
            LEFT OUTER JOIN v3_location t3 ON t1.from_location_id = t3.location_id
            LEFT OUTER JOIN v3_location t4 ON t1.to_location_id = t4.location_id
            GROUP BY t2.month, t3.country, t4.country
            """,
        )

        assert "_DOT_" not in data["sql"]
        assert data["columns"] == [
            {
                "name": "month_order",
                "type": "string",
                "column": None,
                "node": None,
                "semantic_entity": "v3.date.month[order]",
                "semantic_type": "dimension",
            },
            {
                "name": "country_from",
                "type": "string",
                "column": None,
                "node": None,
                "semantic_entity": "v3.location.country[from]",
                "semantic_type": "dimension",
            },
            {
                "name": "country_to",
                "type": "string",
                "column": None,
                "node": None,
                "semantic_entity": "v3.location.country[to]",
                "semantic_type": "dimension",
            },
            {
                "name": "total_revenue",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.total_revenue",
                "semantic_type": "metric",
            },
            {
                "name": "order_count",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.order_count",
                "semantic_type": "metric",
            },
        ]

    @pytest.mark.asyncio
    async def test_dimensions_with_different_date_roles(self, client_with_build_v3):
        """
        Test querying order date vs customer registration date (different roles to same dimension).

        Dimension links:
        - v3.order_details -> v3.date with role "order" (direct)
        - v3.order_details -> v3.customer -> v3.date with role "registration" (multi-hop)
        """
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": ["v3.total_revenue"],
                "dimensions": [
                    "v3.date.year[order]",  # Order year (direct)
                    "v3.date.year[customer->registration]",  # Customer registration year (multi-hop)
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Should have two separate joins to v3_date (for different roles)
        assert_sql_equal(
            data["sql"],
            """
            WITH v3_order_details AS (
                SELECT order_date, customer_id, oi.quantity * oi.unit_price AS line_total
                FROM v3.src_orders o
                JOIN v3.src_order_items oi ON o.order_id = oi.order_id
            ),
            v3_date AS (
                SELECT date_id, year
                FROM v3.src_dates
            ),
            v3_customer AS (
                SELECT customer_id, registration_date
                FROM v3.src_customers
            )
            SELECT t2.year AS year_order, t4.year AS year_registration,
                   SUM(t1.line_total) total_revenue
            FROM v3_order_details t1
            LEFT OUTER JOIN v3_date t2 ON t1.order_date = t2.date_id
            LEFT OUTER JOIN v3_customer t3 ON t1.customer_id = t3.customer_id
            LEFT OUTER JOIN v3_date t4 ON t3.registration_date = t4.date_id
            GROUP BY t2.year, t4.year
            """,
        )

        assert "_DOT_" not in data["sql"]
        assert data["columns"] == [
            {
                "name": "year_order",
                "type": "string",
                "column": None,
                "node": None,
                "semantic_entity": "v3.date.year[order]",
                "semantic_type": "dimension",
            },
            {
                "name": "year_registration",
                "type": "string",
                "column": None,
                "node": None,
                "semantic_entity": "v3.date.year[customer->registration]",
                "semantic_type": "dimension",
            },
            {
                "name": "total_revenue",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.total_revenue",
                "semantic_type": "metric",
            },
        ]

    @pytest.mark.asyncio
    async def test_multi_hop_location_dimension(self, client_with_build_v3):
        """
        Test multi-hop dimension path: order_details -> customer -> location (customer's home).

        Compare with direct location roles (from/to) vs the multi-hop customer home location.
        """
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": ["v3.total_revenue"],
                "dimensions": [
                    "v3.location.country[from]",  # From location (direct)
                    "v3.location.country[customer->home]",  # Customer's home location (multi-hop)
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert_sql_equal(
            data["sql"],
            """
            WITH v3_order_details AS (
                SELECT from_location_id, customer_id, oi.quantity * oi.unit_price AS line_total
                FROM v3.src_orders o
                JOIN v3.src_order_items oi ON o.order_id = oi.order_id
            ),
            v3_location AS (
                SELECT location_id, country
                FROM v3.src_locations
            ),
            v3_customer AS (
                SELECT customer_id, location_id
                FROM v3.src_customers
            )
            SELECT t2.country AS country_from, t4.country AS country_home,
                   SUM(t1.line_total) total_revenue
            FROM v3_order_details t1
            LEFT OUTER JOIN v3_location t2 ON t1.from_location_id = t2.location_id
            LEFT OUTER JOIN v3_customer t3 ON t1.customer_id = t3.customer_id
            LEFT OUTER JOIN v3_location t4 ON t3.location_id = t4.location_id
            GROUP BY t2.country, t4.country
            """,
        )

        assert "_DOT_" not in data["sql"]
        assert data["columns"] == [
            {
                "name": "country_from",
                "type": "string",
                "column": None,
                "node": None,
                "semantic_entity": "v3.location.country[from]",
                "semantic_type": "dimension",
            },
            {
                "name": "country_home",
                "type": "string",
                "column": None,
                "node": None,
                "semantic_entity": "v3.location.country[customer->home]",
                "semantic_type": "dimension",
            },
            {
                "name": "total_revenue",
                "type": "number",
                "column": None,
                "node": None,
                "semantic_entity": "v3.total_revenue",
                "semantic_type": "metric",
            },
        ]

    @pytest.mark.asyncio
    async def test_all_location_roles_in_single_query(self, client_with_build_v3):
        """
        Test querying all three location roles in a single query:
        - from location (direct, role="from")
        - to location (direct, role="to")
        - customer home location (multi-hop, role="customer->home")

        This tests that we can have 3 joins to the same dimension table with different paths.
        """
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": ["v3.total_revenue"],
                "dimensions": [
                    "v3.location.city[from]",
                    "v3.location.city[to]",
                    "v3.location.city[customer->home]",
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert_sql_equal(
            data["sql"],
            """
            WITH v3_order_details AS (
                SELECT from_location_id, to_location_id, customer_id, oi.quantity * oi.unit_price AS line_total
                FROM v3.src_orders o
                JOIN v3.src_order_items oi ON o.order_id = oi.order_id
            ),
            v3_location AS (
                SELECT location_id, city
                FROM v3.src_locations
            ),
            v3_customer AS (
                SELECT customer_id, location_id
                FROM v3.src_customers
            )
            SELECT t2.city AS city_from, t3.city AS city_to, t5.city AS city_home,
                   SUM(t1.line_total) total_revenue
            FROM v3_order_details t1
            LEFT OUTER JOIN v3_location t2 ON t1.from_location_id = t2.location_id
            LEFT OUTER JOIN v3_location t3 ON t1.to_location_id = t3.location_id
            LEFT OUTER JOIN v3_customer t4 ON t1.customer_id = t4.customer_id
            LEFT OUTER JOIN v3_location t5 ON t4.location_id = t5.location_id
            GROUP BY t2.city, t3.city, t5.city
            """,
        )

        assert "_DOT_" not in data["sql"]
        # Should have 3 dimension columns (city with different roles) + 1 metric
        assert len(data["columns"]) == 4
        assert data["columns"][0]["semantic_entity"] == "v3.location.city[from]"
        assert data["columns"][0]["name"] == "city_from"
        assert data["columns"][1]["semantic_entity"] == "v3.location.city[to]"
        assert data["columns"][1]["name"] == "city_to"
        assert (
            data["columns"][2]["semantic_entity"] == "v3.location.city[customer->home]"
        )
        assert data["columns"][2]["name"] == "city_home"

    @pytest.mark.asyncio
    async def test_complex_multi_dimension_multi_role_query(self, client_with_build_v3):
        """
        Test a complex query with multiple dimensions across different roles:
        - Local dimension: status
        - Customer name (via customer role)
        - Order date month (via order role)
        - Customer registration year (via customer->registration multi-hop)
        - Customer home country (via customer->home multi-hop)
        """
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": self.ORDER_DETAILS_BASE_METRICS,
                "dimensions": [
                    "v3.order_details.status",
                    "v3.customer.name",
                    "v3.date.month[order]",
                    "v3.date.year[customer->registration]",
                    "v3.location.country[customer->home]",
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify structure
        assert "_DOT_" not in data["sql"]

        # Should have 5 dimensions + 4 metrics = 9 columns
        assert len(data["columns"]) == 9

        # Check all dimension semantic entities
        dim_entities = [
            c["semantic_entity"]
            for c in data["columns"]
            if c["semantic_type"] == "dimension"
        ]
        assert "v3.order_details.status" in dim_entities
        assert "v3.customer.name" in dim_entities
        assert "v3.date.month[order]" in dim_entities
        assert "v3.date.year[customer->registration]" in dim_entities
        assert "v3.location.country[customer->home]" in dim_entities

        # Check all metrics present
        metric_names = [
            c["name"] for c in data["columns"] if c["semantic_type"] == "metric"
        ]
        assert set(metric_names) == {
            "total_revenue",
            "total_quantity",
            "order_count",
            "customer_count",
        }


class TestInnerCTEFlattening:
    """
    Tests for flattening inner CTEs within transforms.

    When a transform has its own WITH clause (inner CTEs), those CTEs need to be
    extracted and prefixed to avoid name collisions in the final query.
    """

    @pytest.mark.asyncio
    async def test_transform_with_inner_cte(self, client_with_build_v3):
        """
        Test that transforms with inner CTEs have those CTEs flattened and prefixed.

        Creates:
        - A transform with an inner CTE: WITH order_totals AS (...) SELECT ...
        - A metric on that transform

        The generated SQL should have:
        - v3_transform_with_cte__order_totals AS (...)  -- prefixed inner CTE
        - v3_transform_with_cte AS (SELECT ... FROM v3_transform_with_cte__order_totals)
        """
        # Create a transform that has an inner CTE
        transform_response = await client_with_build_v3.post(
            "/nodes/transform",
            json={
                "name": "v3.transform_with_cte",
                "type": "transform",
                "description": "Transform with inner CTE for testing CTE flattening",
                "query": """
                    WITH order_totals AS (
                        SELECT
                            o.order_id,
                            o.customer_id,
                            SUM(oi.quantity * oi.unit_price) AS total_amount
                        FROM v3.src_orders o
                        JOIN v3.src_order_items oi ON o.order_id = oi.order_id
                        GROUP BY o.order_id, o.customer_id
                    )
                    SELECT
                        customer_id,
                        COUNT(*) AS order_count,
                        SUM(total_amount) AS total_spent
                    FROM order_totals
                    GROUP BY customer_id
                """,
                "mode": "published",
            },
        )
        assert transform_response.status_code == 201, transform_response.json()

        # Create a metric on the transform
        metric_response = await client_with_build_v3.post(
            "/nodes/metric",
            json={
                "name": "v3.total_customer_spend",
                "type": "metric",
                "description": "Total spend across all customers",
                "query": "SELECT SUM(total_spent) FROM v3.transform_with_cte",
                "mode": "published",
            },
        )
        assert metric_response.status_code == 201, metric_response.json()

        # Request the measures SQL
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": ["v3.total_customer_spend"],
                "dimensions": [],
            },
        )

        assert response.status_code == 200, response.json()
        data = response.json()
        sql = data["sql"]
        assert_sql_equal(
            sql,
            """
            WITH v3_transform_with_cte__order_totals AS (
              SELECT
                o.order_id,
                o.customer_id,
                SUM(oi.quantity * oi.unit_price) AS total_amount
              FROM default.v3.orders o
              JOIN default.v3.order_items oi ON o.order_id = oi.order_id
              GROUP BY  o.order_id, o.customer_id
            ),
            v3_transform_with_cte AS (
              SELECT
                SUM(total_amount) AS total_spent
              FROM v3_transform_with_cte__order_totals order_totals
              GROUP BY  customer_id
            )
            SELECT  SUM(t1.total_spent) total_customer_spend
            FROM v3_transform_with_cte t1
            """,
        )

    @pytest.mark.asyncio
    async def test_transform_with_multiple_inner_ctes(self, client_with_build_v3):
        """
        Test that transforms with multiple inner CTEs have all of them flattened.
        """
        # Create a transform with multiple inner CTEs
        transform_response = await client_with_build_v3.post(
            "/nodes/v3.transform_multi_cte/",
            json={
                "name": "v3.transform_multi_cte",
                "type": "transform",
                "description": "Transform with multiple inner CTEs",
                "query": """
                    WITH
                        order_counts AS (
                            SELECT customer_id, COUNT(*) AS num_orders
                            FROM v3.src_orders
                            GROUP BY customer_id
                        ),
                        item_totals AS (
                            SELECT order_id, SUM(quantity) AS total_items
                            FROM v3.src_order_items
                            GROUP BY order_id
                        )
                    SELECT
                        oc.customer_id,
                        oc.num_orders,
                        SUM(it.total_items) AS total_items_purchased
                    FROM order_counts oc
                    JOIN v3.src_orders o ON oc.customer_id = o.customer_id
                    JOIN item_totals it ON o.order_id = it.order_id
                    GROUP BY oc.customer_id, oc.num_orders
                """,
                "mode": "published",
            },
        )
        assert transform_response.status_code == 201, transform_response.json()

        # Create a metric
        metric_response = await client_with_build_v3.post(
            "/nodes/v3.avg_items_per_customer/",
            json={
                "name": "v3.avg_items_per_customer",
                "type": "metric",
                "description": "Average items purchased per customer",
                "query": "SELECT AVG(total_items_purchased) FROM v3.transform_multi_cte",
                "mode": "published",
            },
        )
        assert metric_response.status_code == 201, metric_response.json()

        # Request the measures SQL
        response = await client_with_build_v3.get(
            "/sql/measures/v3/",
            params={
                "metrics": ["v3.avg_items_per_customer"],
                "dimensions": [],
            },
        )

        assert response.status_code == 200, response.json()
        data = response.json()
        sql = data["sql"]

        # Verify both inner CTEs are flattened with prefix
        assert "v3_transform_multi_cte__order_counts" in sql, (
            f"First inner CTE should be prefixed. Got:\n{sql}"
        )
        assert "v3_transform_multi_cte__item_totals" in sql, (
            f"Second inner CTE should be prefixed. Got:\n{sql}"
        )

        # Verify only one WITH clause
        with_count = sql.upper().count("WITH")
        assert with_count == 1, (
            f"Should have only one WITH clause, found {with_count}. Got:\n{sql}"
        )
