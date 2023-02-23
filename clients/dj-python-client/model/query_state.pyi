# coding: utf-8

"""
    DJ server

    A DataJunction metrics repository  # noqa: E501

    The version of the OpenAPI document: 0.0.post1.dev1+g2c5d4fa
    Generated by: https://openapi-generator.tech
"""

from datetime import date, datetime  # noqa: F401
import decimal  # noqa: F401
import functools  # noqa: F401
import io  # noqa: F401
import re  # noqa: F401
import typing  # noqa: F401
import typing_extensions  # noqa: F401
import uuid  # noqa: F401

import frozendict  # noqa: F401

from dj-python-client import schemas  # noqa: F401


class QueryState(
    schemas.EnumBase,
    schemas.StrSchema
):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.

    
    Different states of a query.
    
    """
    
    @schemas.classproperty
    def UNKNOWN(cls):
        return cls("UNKNOWN")
    
    @schemas.classproperty
    def ACCEPTED(cls):
        return cls("ACCEPTED")
    
    @schemas.classproperty
    def SCHEDULED(cls):
        return cls("SCHEDULED")
    
    @schemas.classproperty
    def RUNNING(cls):
        return cls("RUNNING")
    
    @schemas.classproperty
    def FINISHED(cls):
        return cls("FINISHED")
    
    @schemas.classproperty
    def CANCELED(cls):
        return cls("CANCELED")
    
    @schemas.classproperty
    def FAILED(cls):
        return cls("FAILED")
