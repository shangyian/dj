"""Tests for the MetricSource table and NodeRevision.queries."""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from datajunction_server.database.metricsource import MetricSource
from datajunction_server.database.node import Node, NodeRevision, NodeType
from datajunction_server.database.user import OAuthProvider, User


@pytest_asyncio.fixture
async def user(session: AsyncSession) -> User:
    """A user fixture."""
    user = User(username="testuser", oauth_provider=OAuthProvider.BASIC)
    session.add(user)
    await session.commit()
    return user


async def make_revision(
    session: AsyncSession,
    user: User,
    name: str,
    query: str,
) -> NodeRevision:
    """Create a metric node and its first revision."""
    node = Node(
        name=name,
        type=NodeType.METRIC,
        current_version="1",
        created_by_id=user.id,
    )
    revision = NodeRevision(
        name=name,
        type=NodeType.METRIC,
        version="1",
        query=query,
        node=node,
        created_by_id=user.id,
    )
    session.add_all([node, revision])
    await session.flush()
    return revision


@pytest.mark.asyncio
async def test_queries_defaults_to_single_query(session: AsyncSession, user: User):
    revision = await make_revision(
        session,
        user,
        "default.single_source_metric",
        "SELECT SUM(a) FROM default.orders",
    )
    await session.commit()
    await session.refresh(revision, ["additional_sources"])
    assert revision.queries == ["SELECT SUM(a) FROM default.orders"]


@pytest.mark.asyncio
async def test_queries_includes_additional_sources_in_order(
    session: AsyncSession,
    user: User,
):
    revision = await make_revision(
        session,
        user,
        "default.multi_source_metric",
        "SELECT SUM(is_completed) FROM default.orders_daily",
    )
    session.add_all(
        [
            MetricSource(
                node_revision_id=revision.id,
                ordinal=2,
                query="SELECT SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) "
                "FROM default.orders_by_region_daily",
            ),
            MetricSource(
                node_revision_id=revision.id,
                ordinal=1,
                query="SELECT SUM(completed_flag) FROM default.orders_enriched_daily",
            ),
        ],
    )
    await session.commit()
    await session.refresh(revision, ["additional_sources"])
    assert revision.queries == [
        "SELECT SUM(is_completed) FROM default.orders_daily",
        "SELECT SUM(completed_flag) FROM default.orders_enriched_daily",
        "SELECT SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) "
        "FROM default.orders_by_region_daily",
    ]


@pytest.mark.asyncio
async def test_queries_empty_without_query(session: AsyncSession, user: User):
    revision = await make_revision(session, user, "default.no_query", "")
    revision.query = None
    await session.commit()
    await session.refresh(revision, ["additional_sources"])
    assert revision.queries == []
