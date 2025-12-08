"""
Collection GraphQL queries.
"""

from typing import Annotated

import strawberry
from sqlalchemy import select
from sqlalchemy.sql.operators import is_
from strawberry.types import Info

from datajunction_server.api.graphql.scalars.collection import Collection
from datajunction_server.database.collection import Collection as DBCollection
from datajunction_server.database.user import User


async def list_collections(
    created_by: Annotated[
        str | None,
        strawberry.argument(
            description="Filter to collections created by this user",
        ),
    ] = None,
    limit: Annotated[
        int | None,
        strawberry.argument(description="Limit collections"),
    ] = 100,
    *,
    info: Info,
) -> list[Collection]:
    """
    List collections, optionally filtered by creator.
    """
    session = info.context["session"]

    statement = select(DBCollection).where(is_(DBCollection.deactivated_at, None))

    if created_by:
        statement = statement.join(
            User,
            DBCollection.created_by_id == User.id,
        ).where(User.username == created_by)

    statement = statement.order_by(DBCollection.created_at.desc())

    if limit and limit > 0:
        statement = statement.limit(limit)

    result = await session.execute(statement)
    collections = result.unique().scalars().all()

    return [Collection.from_db_collection(c) for c in collections]

