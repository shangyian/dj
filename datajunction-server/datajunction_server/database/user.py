"""User database schema."""
from typing import Optional

from sqlalchemy import BigInteger, Integer
from sqlalchemy.orm import Mapped, mapped_column

from datajunction_server.database.connection import Base
from datajunction_server.enum import StrEnum
from datajunction_server.models.base import sqlalchemy_enum_with_name


class OAuthProvider(StrEnum):
    """
    Support oauth providers
    """

    BASIC = "basic"
    GITHUB = "github"
    GOOGLE = "google"


class User(Base):  # pylint: disable=too-few-public-methods
    """Class for a user."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
    )
    username: Mapped[str]
    password: Mapped[Optional[str]]
    email: Mapped[Optional[str]]
    name: Mapped[Optional[str]]
    oauth_provider: Mapped[OAuthProvider] = mapped_column(
        sqlalchemy_enum_with_name(OAuthProvider),
    )
    is_admin: Mapped[bool] = mapped_column(default=False)