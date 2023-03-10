"""
Tests for tags.
"""

import pytest
from fastapi.testclient import TestClient


class TestTags:
    """
    Test tags API endpoints.
    """

    @pytest.fixture
    def client(self, client: TestClient, load_examples) -> TestClient:
        """
        Add nodes to facilitate testing of tags endpoints.
        """
        load_examples(client)
        # Add source node
        response = client.post(
            "/nodes/",
            json={
                "columns": {
                    "id": {"type": "INT"},
                    "item_name": {"type": "STR"},
                    "sold_count": {"type": "INT"},
                    "price_per_unit": {"type": "FLOAT"},
                    "psp": {"type": "STR"},
                },
                "description": "A source table for sales",
                "mode": "published",
                "name": "sales",
                "catalog": "default",
                "schema_": "revenue",
                "table": "sales",
                "type": "source",
            },
        )
        assert response.status_code == 201

        # Add dimension node
        response = client.post(
            "/nodes/",
            json={
                "description": "Item dimension",
                "query": (
                    "SELECT item_name " "account_type_classification FROM " "sales"
                ),
                "mode": "published",
                "name": "items",
                "type": "dimension",
            },
        )
        assert response.status_code == 201

        # Add metrics
        response = client.post(
            "/nodes/",
            json={
                "description": "Total units sold",
                "query": "SELECT SUM(sold_count) as num_sold FROM sales",
                "mode": "published",
                "name": "items_sold_count",
                "type": "metric",
            },
        )
        assert response.status_code == 201

        response = client.post(
            "/nodes/",
            json={
                "description": "Total profit",
                "query": "SELECT SUM(sold_count * price_per_unit) as num_sold FROM sales",
                "mode": "published",
                "name": "total_profit",
                "type": "metric",
            },
        )
        assert response.status_code == 201
        return client

    def create_tag(self, client: TestClient):
        """
        Creates a tag.
        """
        response = client.post(
            "/tags/",
            json={
                "name": "sales_report",
                "display_name": "Sales Report",
                "description": "All metrics for sales",
                "tag_type": "group",
                "tag_metadata": {},
            },
        )
        return response

    def test_create_and_read_tag(self, client: TestClient) -> None:
        """
        Test ``POST /tags`` and ``GET /tags/{name}``
        """
        response = self.create_tag(client)
        expected_tag_output = {
            "tag_metadata": {},
            "display_name": "Sales Report",
            "id": 1,
            "description": "All metrics for sales",
            "name": "sales_report",
            "tag_type": "group",
        }
        assert response.status_code == 201
        assert response.json() == expected_tag_output

        response = client.get("/tags/sales_report/")
        assert response.status_code == 200
        assert response.json() == expected_tag_output

        # Creating it again should raise an exception
        response = self.create_tag(client)
        response_data = response.json()
        assert (
            response_data["message"] == "A tag with name `sales_report` already exists!"
        )

    def test_update_tag(self, client: TestClient) -> None:
        """
        Tests updating a tag.
        """
        response = self.create_tag(client)
        assert response.status_code == 201

        # Trying updating the tag
        response = client.patch(
            "/tags/sales_report/",
            json={
                "description": "Helpful sales metrics",
                "tag_metadata": {"order": 1},
            },
        )
        assert response.status_code == 200
        response_data = response.json()
        assert response_data == {
            "tag_metadata": {"order": 1},
            "display_name": "Sales Report",
            "id": 1,
            "description": "Helpful sales metrics",
            "name": "sales_report",
            "tag_type": "group",
        }

        # Trying updating the tag
        response = client.patch(
            "/tags/sales_report/",
            json={},
        )
        assert response.json() == {
            "tag_metadata": {"order": 1},
            "display_name": "Sales Report",
            "id": 1,
            "description": "Helpful sales metrics",
            "name": "sales_report",
            "tag_type": "group",
        }

    def test_list_tags(self, client: TestClient) -> None:
        """
        Test ``GET /tags``
        """
        response = self.create_tag(client)
        assert response.status_code == 201

        response = client.get("/tags/")
        assert response.status_code == 200
        response_data = response.json()

        assert response_data == [
            {
                "description": "All metrics for sales",
                "display_name": "Sales Report",
                "name": "sales_report",
                "tag_metadata": {},
                "tag_type": "group",
            },
        ]

        client.post(
            "/tags/",
            json={
                "name": "impressions_report",
                "display_name": "Impressions Report",
                "description": "Metrics for various types of impressions",
                "tag_type": "group",
                "tag_metadata": {},
            },
        )

        client.post(
            "/tags/",
            json={
                "name": "rotors",
                "display_name": "Rotors",
                "description": "Department of brakes",
                "tag_type": "business_area",
                "tag_metadata": {},
            },
        )

        response = client.get("/tags/?tag_type=group")
        assert response.status_code == 200
        response_data = response.json()
        assert response_data == [
            {
                "description": "All metrics for sales",
                "tag_metadata": {},
                "name": "sales_report",
                "display_name": "Sales Report",
                "tag_type": "group",
            },
            {
                "description": "Metrics for various types of impressions",
                "tag_metadata": {},
                "name": "impressions_report",
                "display_name": "Impressions Report",
                "tag_type": "group",
            },
        ]

        response = client.get("/tags/?tag_type=business_area")
        assert response.status_code == 200
        response_data = response.json()
        assert response_data == [
            {
                "name": "rotors",
                "display_name": "Rotors",
                "description": "Department of brakes",
                "tag_type": "business_area",
                "tag_metadata": {},
            },
        ]

    def test_add_tag_to_node(self, client: TestClient) -> None:
        """
        Test ``POST /tags`` and ``GET /tags/{name}``
        """
        response = self.create_tag(client)
        assert response.status_code == 201

        # Trying tag a node with a nonexistent tag should fail
        response = client.post(
            "/nodes/items_sold_count/tag/?tag_name=random_tag",
        )
        assert response.status_code == 404
        response_data = response.json()
        assert (
            response_data["message"] == "A tag with name `random_tag` does not exist."
        )

        # Trying tag a node with an existing tag should succeed
        response = client.post(
            "/nodes/items_sold_count/tag/?tag_name=sales_report",
        )
        assert response.status_code == 201
        response_data = response.json()
        assert (
            response_data["message"]
            == "Node `items_sold_count` has been successfully tagged with tag `sales_report`"
        )

        # Test finding all nodes for that tag
        response = client.get(
            "/tags/sales_report/nodes/",
        )
        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data) == 1
        assert response_data == [
            "items_sold_count",
        ]

        # Tag a second node
        response = client.post(
            "/nodes/total_profit/tag/?tag_name=sales_report",
        )
        assert response.status_code == 201
        response_data = response.json()
        assert (
            response_data["message"]
            == "Node `total_profit` has been successfully tagged with tag `sales_report`"
        )

        # Check finding nodes for tag
        response = client.get(
            "/tags/sales_report/nodes/",
        )
        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data) == 2
        assert response_data == [
            "items_sold_count",
            "total_profit",
        ]

        # Check finding nodes for tag
        response = client.get(
            "/tags/random_tag/nodes/",
        )
        assert response.status_code == 404
        response_data = response.json()
        assert (
            response_data["message"] == "A tag with name `random_tag` does not exist."
        )

        # Check finding nodes for tag
        response = client.get(
            "/tags/sales_report/nodes/?node_type=transform",
        )
        assert response.status_code == 200
        response_data = response.json()
        assert response_data == []
