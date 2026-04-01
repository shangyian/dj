"""
Tests for the healthcheck API.
"""

import asyncio

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from datajunction_server import __version__


@pytest.mark.asyncio
async def test_successful_health(module__client: AsyncClient) -> None:
    """
    Test ``GET /health/``.
    """
    response = await module__client.get("/health/")
    data = response.json()
    assert data == [{"name": "database", "status": "ok"}]


@pytest.mark.asyncio
async def test_failed_health(
    session: AsyncSession,
    client: AsyncClient,
    mocker,
) -> None:
    """
    Test failed healthcheck.
    """
    future: asyncio.Future = asyncio.Future()
    future.set_result(mocker.MagicMock())
    session.execute = mocker.MagicMock(return_value=future)
    response = await client.get("/health/")
    data = response.json()
    assert data == [{"name": "database", "status": "failed"}]


@pytest.mark.asyncio
async def test_server_info(module__client: AsyncClient) -> None:
    """
    Test ``GET /info/`` returns server version and min_client_version.
    """
    response = await module__client.get("/info/")
    assert response.status_code == 200
    data = response.json()
    assert data["server_version"] == __version__
    assert "min_client_version" in data


@pytest.mark.asyncio
async def test_server_info_min_client_version_equals_server_version(
    module__client: AsyncClient,
) -> None:
    """
    Test that min_client_version equals server_version (released together).
    """
    response = await module__client.get("/info/")
    data = response.json()
    assert data["min_client_version"] == data["server_version"]
