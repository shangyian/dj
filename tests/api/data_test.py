"""
Tests for the data API.
"""

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from dj.models import Catalog, Table
from dj.models.column import Column, ColumnType
from dj.models.node import Node, NodeRevision, NodeType


class TestAvailabilityState:  # pylint: disable=too-many-public-methods
    """
    Test ``POST /data/availability/{node_name}/``.
    """

    @pytest.fixture
    def client(self, client: TestClient, load_examples) -> TestClient:
        """
        load examples
        """
        load_examples(client)
        return client

    def test_setting_availability_state(
        self,
        session: Session,
        client: TestClient,
    ) -> None:
        """
        Test adding an availability state
        """
        response = client.post(
            "/data/availability/large_revenue_payments_and_business_only/",
            json={
                "catalog": "default",
                "schema_": "accounting",
                "table": "pmts",
                "valid_through_ts": 20230125,
                "max_partition": ["2023", "01", "25"],
                "min_partition": ["2022", "01", "01"],
            },
        )
        data = response.json()

        assert response.status_code == 200
        assert data == {"message": "Availability state successfully posted"}

        statement = select(Node).where(
            Node.name == "large_revenue_payments_and_business_only",
        )
        large_revenue_payments_and_business_only = session.exec(statement).one()
        node_dict = large_revenue_payments_and_business_only.current.availability.dict()
        node_dict.pop("updated_at")
        assert node_dict == {
            "valid_through_ts": 20230125,
            "catalog": "default",
            "min_partition": ["2022", "01", "01"],
            "table": "pmts",
            "max_partition": ["2023", "01", "25"],
            "schema_": "accounting",
            "id": 1,
        }

    def test_setting_availability_state_multiple_times(
        self,
        session: Session,
        client: TestClient,
    ) -> None:
        """
        Test adding multiple availability states
        """
        response = client.post(
            "/data/availability/large_revenue_payments_and_business_only/",
            json={
                "catalog": "default",
                "schema_": "accounting",
                "table": "pmts",
                "valid_through_ts": 20230125,
                "max_partition": ["2023", "01", "25"],
                "min_partition": ["2022", "01", "01"],
            },
        )
        data = response.json()

        assert response.status_code == 200
        assert data == {"message": "Availability state successfully posted"}

        response = client.post(
            "/data/availability/large_revenue_payments_and_business_only/",
            json={
                "catalog": "default",
                "schema_": "accounting",
                "table": "pmts",
                "valid_through_ts": 20230125,
                "max_partition": ["2023", "01", "25"],
                "min_partition": ["2022", "01", "01"],
            },
        )
        data = response.json()

        assert response.status_code == 200
        assert data == {"message": "Availability state successfully posted"}

        response = client.post(
            "/data/availability/large_revenue_payments_and_business_only/",
            json={
                "catalog": "default",
                "schema_": "new_accounting",
                "table": "new_payments_table",
                "valid_through_ts": 20230125,
                "max_partition": ["2023", "01", "25"],
                "min_partition": ["2022", "01", "01"],
            },
        )
        data = response.json()

        assert response.status_code == 200
        assert data == {"message": "Availability state successfully posted"}

        statement = select(Node).where(
            Node.name == "large_revenue_payments_and_business_only",
        )
        large_revenue_payments_and_business_only = session.exec(statement).one()
        node_dict = large_revenue_payments_and_business_only.current.availability.dict()
        node_dict.pop("updated_at")
        assert node_dict == {
            "valid_through_ts": 20230125,
            "catalog": "default",
            "min_partition": ["2022", "01", "01"],
            "table": "new_payments_table",
            "max_partition": ["2023", "01", "25"],
            "schema_": "new_accounting",
            "id": 3,
        }

    def test_that_update_at_timestamp_is_being_updated(
        self,
        session: Session,
        client: TestClient,
    ) -> None:
        """
        Test that the `updated_at` attribute is being updated
        """
        response = client.post(
            "/data/availability/large_revenue_payments_and_business_only/",
            json={
                "catalog": "default",
                "schema_": "accounting",
                "table": "pmts",
                "valid_through_ts": 20230125,
                "max_partition": ["2023", "01", "25"],
                "min_partition": ["2022", "01", "01"],
            },
        )
        assert response.status_code == 200
        statement = select(Node).where(
            Node.name == "large_revenue_payments_and_business_only",
        )
        large_revenue_payments_and_business_only = session.exec(statement).one()
        updated_at_1 = (
            large_revenue_payments_and_business_only.current.availability.dict()[
                "updated_at"
            ]
        )

        response = client.post(
            "/data/availability/large_revenue_payments_and_business_only/",
            json={
                "catalog": "default",
                "schema_": "accounting",
                "table": "pmts",
                "valid_through_ts": 20230125,
                "max_partition": ["2023", "01", "25"],
                "min_partition": ["2022", "01", "01"],
            },
        )
        assert response.status_code == 200

        session.refresh(large_revenue_payments_and_business_only)
        updated_at_2 = (
            large_revenue_payments_and_business_only.current.availability.dict()[
                "updated_at"
            ]
        )

        assert updated_at_2 > updated_at_1

    def test_raising_when_node_does_not_exist(
        self,
        client: TestClient,
    ) -> None:
        """
        Test raising when setting availability state on non-existent node
        """
        response = client.post(
            "/data/availability/nonexistentnode/",
            json={
                "catalog": "default",
                "schema_": "accounting",
                "table": "pmts",
                "valid_through_ts": 20230125,
                "max_partition": ["2023", "01", "25"],
                "min_partition": ["2022", "01", "01"],
            },
        )
        data = response.json()

        assert response.status_code == 404
        assert data == {
            "message": "A node with name `nonexistentnode` does not exist.",
            "errors": [],
            "warnings": [],
        }

    def test_merging_in_a_higher_max_partition(
        self,
        session: Session,
        client: TestClient,
    ) -> None:
        """
        Test that the higher max_partition value is used when merging in an availability state
        """
        client.post(
            "/data/availability/large_revenue_payments_only/",
            json={
                "catalog": "default",
                "schema_": "accounting",
                "table": "large_pmts",
                "valid_through_ts": 20230101,
                "max_partition": ["2023", "01", "01"],
                "min_partition": ["2022", "01", "01"],
            },
        )
        response = client.post(
            "/data/availability/large_revenue_payments_only/",
            json={
                "catalog": "default",
                "schema_": "accounting",
                "table": "large_pmts",
                "valid_through_ts": 20230102,
                "max_partition": [
                    "2023",
                    "01",
                    "02",
                ],  # should be used since it's a higher max_partition
                "min_partition": [
                    "2023",
                    "01",
                    "02",
                ],  # should be ignored since it's a higher min_partition
            },
        )
        data = response.json()

        assert response.status_code == 200
        assert data == {"message": "Availability state successfully posted"}

        statement = select(Node).where(
            Node.name == "large_revenue_payments_only",
        )
        large_revenue_payments_only = session.exec(statement).one()
        node_dict = large_revenue_payments_only.current.availability.dict()
        node_dict.pop("updated_at")
        assert node_dict == {
            "valid_through_ts": 20230102,
            "catalog": "default",
            "min_partition": ["2022", "01", "01"],
            "table": "large_pmts",
            "max_partition": ["2023", "01", "02"],
            "schema_": "accounting",
            "id": 2,
        }

    def test_merging_in_a_lower_min_partition(
        self,
        session: Session,
        client: TestClient,
    ) -> None:
        """
        Test that the lower min_partition value is used when merging in an availability state
        """
        client.post(
            "/data/availability/large_revenue_payments_only/",
            json={
                "catalog": "default",
                "schema_": "accounting",
                "table": "large_pmts",
                "valid_through_ts": 20230101,
                "max_partition": ["2023", "01", "01"],
                "min_partition": ["2022", "01", "01"],
            },
        )
        response = client.post(
            "/data/availability/large_revenue_payments_only/",
            json={
                "catalog": "default",
                "schema_": "accounting",
                "table": "large_pmts",
                "valid_through_ts": 20230101,
                "max_partition": [
                    "2021",
                    "12",
                    "31",
                ],  # should be ignored since it's a lower max_partition
                "min_partition": [
                    "2021",
                    "12",
                    "31",
                ],  # should be used since it's a lower min_partition
            },
        )
        data = response.json()

        assert response.status_code == 200
        assert data == {"message": "Availability state successfully posted"}

        statement = select(Node).where(
            Node.name == "large_revenue_payments_only",
        )
        large_revenue_payments_only = session.exec(statement).one()
        node_dict = large_revenue_payments_only.current.availability.dict()
        node_dict.pop("updated_at")
        assert node_dict == {
            "valid_through_ts": 20230101,
            "catalog": "default",
            "min_partition": ["2021", "12", "31"],
            "table": "large_pmts",
            "max_partition": ["2023", "01", "01"],
            "schema_": "accounting",
            "id": 2,
        }

    def test_moving_back_valid_through_ts(
        self,
        session: Session,
        client: TestClient,
    ) -> None:
        """
        Test that the valid through timestamp can be moved backwards
        """
        client.post(
            "/data/availability/large_revenue_payments_only/",
            json={
                "catalog": "default",
                "schema_": "accounting",
                "table": "large_pmts",
                "valid_through_ts": 20230101,
                "max_partition": ["2023", "01", "01"],
                "min_partition": ["2022", "01", "01"],
            },
        )
        response = client.post(
            "/data/availability/large_revenue_payments_only/",
            json={
                "catalog": "default",
                "schema_": "accounting",
                "table": "large_pmts",
                "valid_through_ts": 20221231,
                "max_partition": [
                    "2023",
                    "01",
                    "01",
                ],  # should be ignored since it's a lower max_partition
                "min_partition": [
                    "2022",
                    "01",
                    "01",
                ],  # should be used since it's a lower min_partition
            },
        )
        data = response.json()

        assert response.status_code == 200
        assert data == {"message": "Availability state successfully posted"}

        statement = select(Node).where(
            Node.name == "large_revenue_payments_only",
        )
        large_revenue_payments_only = session.exec(statement).one()
        node_dict = large_revenue_payments_only.current.availability.dict()
        node_dict.pop("updated_at")
        assert node_dict == {
            "valid_through_ts": 20221231,
            "catalog": "default",
            "min_partition": ["2022", "01", "01"],
            "table": "large_pmts",
            "max_partition": ["2023", "01", "01"],
            "schema_": "accounting",
            "id": 2,
        }

    def test_setting_availablity_state_on_a_source_node(
        self,
        session: Session,
        client: TestClient,
    ) -> None:
        """
        Test setting the availability state on a source node
        """
        response = client.post(
            "/data/availability/revenue/",
            json={
                "catalog": "default",
                "schema_": "accounting",
                "table": "revenue",
                "valid_through_ts": 20230101,
                "max_partition": ["2023", "01", "01"],
                "min_partition": ["2022", "01", "01"],
            },
        )
        data = response.json()

        assert response.status_code == 200
        assert data == {"message": "Availability state successfully posted"}

        statement = select(Node).where(
            Node.name == "revenue",
        )
        revenue = session.exec(statement).one()
        node_dict = revenue.current.availability.dict()
        node_dict.pop("updated_at")
        assert node_dict == {
            "valid_through_ts": 20230101,
            "catalog": "default",
            "min_partition": ["2022", "01", "01"],
            "table": "revenue",
            "max_partition": ["2023", "01", "01"],
            "schema_": "accounting",
            "id": 1,
        }

    def test_raise_on_setting_invalid_availability_state_on_a_source_node(
        self,
        client: TestClient,
    ) -> None:
        """
        Test raising availability state doesn't match existing source node table
        """
        response = client.post(
            "/data/availability/revenue/",
            json={
                "catalog": "default",
                "schema_": "accounting",
                "table": "large_pmts",
                "valid_through_ts": 20230101,
                "max_partition": ["2023", "01", "01"],
                "min_partition": ["2022", "01", "01"],
            },
        )
        data = response.json()

        assert response.status_code == 500
        assert data == {
            'message': (
                "Cannot set availability state, source nodes require availability states "
                "match the set table: default.accounting.large_pmts does not match "
                "default.accounting.revenue "
            ),
            'errors': [],
            'warnings': []
        }
