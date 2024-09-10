"""
Fixtures for testing.
"""
# pylint: disable=redefined-outer-name, invalid-name

from typing import Iterator
from unittest.mock import patch

import duckdb
import pytest
from cachelib.simple import SimpleCache
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from djqs.api.main import app
from djqs.config import Settings
from djqs.utils import get_settings


@pytest.fixture(scope="session", autouse=True)
def mock_get_settings():
    with patch("djqs.utils.get_settings") as mocked_get_settings:
        mocked_get_settings.return_value = Settings(
            index="sqlite://",
            results_backend=SimpleCache(default_timeout=0),
            configuration_file="./config.djqs.yml",
        )
        yield mocked_get_settings


pytest_plugins = ["djqs.api.main"]


@pytest.fixture
def settings(mocker: MockerFixture) -> Iterator[Settings]:
    """
    Custom settings for unit tests.
    """
    settings = Settings(
        index="sqlite://",
        results_backend=SimpleCache(default_timeout=0),
        configuration_file="./tests/config.djqs.yml",
    )

    mocker.patch(
        "djqs.utils.get_settings",
        return_value=settings,
    )

    yield settings


@pytest.fixture
def settings_no_config_file(mocker: MockerFixture) -> Iterator[Settings]:
    """
    Custom settings for unit tests.
    """
    settings = Settings(
        index="sqlite://",
        results_backend=SimpleCache(default_timeout=0),
    )

    mocker.patch(
        "djqs.utils.get_settings",
        return_value=settings,
    )

    yield settings


@pytest.fixture()
def client(settings: Settings) -> Iterator[TestClient]:
    """
    Create a client for testing APIs.
    """

    def get_settings_override() -> Settings:
        return settings

    app.dependency_overrides[get_settings] = get_settings_override

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture()
def client_no_config_file(
    settings_no_config_file: Settings,
) -> Iterator[TestClient]:
    """
    Create a client for testing APIs.
    """

    def get_settings_override() -> Settings:
        return settings_no_config_file

    app.dependency_overrides[get_settings] = get_settings_override

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def duckdb_conn():
    """
    A duckdb connection to a roads database
    """
    return duckdb.connect(
        database="docker/default.duckdb",
    )
