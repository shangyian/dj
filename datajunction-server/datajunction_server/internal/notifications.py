"""
Module related to all things notifications
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from datajunction_server.database.notification_preference import NotificationPreference
from datajunction_server.database.user import PrincipalKind, User
from datajunction_server.internal.history import EntityType


async def get_entity_notification_preferences(
    session: AsyncSession,
    entity_name: str,
    entity_type: EntityType,
) -> list[NotificationPreference]:
    """
    Get all user preferences for a specific notification preference
    """
    result = await session.execute(
        select(NotificationPreference)
        .join(User, NotificationPreference.user_id == User.id)
        .options(selectinload(NotificationPreference.user))
        .where(NotificationPreference.entity_name == entity_name)
        .where(NotificationPreference.entity_type == entity_type)
        # Service accounts and groups have nowhere to notify
        .where(User.kind == PrincipalKind.USER),
    )
    return result.scalars().all()
