"""
Application healthchecks.
"""

from typing import List

from fastapi import APIRouter, Depends
from pydantic.main import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from datajunction_server import __version__
from datajunction_server.enum import StrEnum
from datajunction_server.utils import get_session

router = APIRouter(tags=["health"])


class HealthcheckStatus(StrEnum):
    """
    Possible health statuses.
    """

    OK = "ok"
    FAILED = "failed"


class HealthCheck(BaseModel):
    """
    A healthcheck response.
    """

    name: str
    status: HealthcheckStatus


async def database_health(session: AsyncSession) -> HealthcheckStatus:
    """
    The status of the database.
    """
    try:
        result = (await session.execute(select(1))).one()
        health_status = (
            HealthcheckStatus.OK if result == (1,) else HealthcheckStatus.FAILED
        )
        return health_status
    except Exception:  # pragma: no cover
        return HealthcheckStatus.FAILED  # pragma: no cover


class ServerInfo(BaseModel):
    """Server version and compatibility information."""

    server_version: str
    min_client_version: str


@router.get("/info/", response_model=ServerInfo)
async def server_info() -> ServerInfo:
    """Return server version and minimum compatible client version."""
    return ServerInfo(
        server_version=__version__,
        min_client_version=__version__,
    )


@router.get("/health/", response_model=List[HealthCheck])
async def health_check(
    session: AsyncSession = Depends(get_session),
) -> List[HealthCheck]:
    """
    Healthcheck for services.
    """
    return [
        HealthCheck(
            name="database",
            status=await database_health(session),
        ),
    ]
