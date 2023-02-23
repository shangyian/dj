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


class ColumnType(
    schemas.EnumBase,
    schemas.StrSchema
):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.

    
    Types for columns.

    These represent the values from the ``python_type`` attribute in SQLAlchemy columns.
    
    """


    class MetaOapg:
        enum_value_to_name = {
            "BYTES": "BYTES",
            "STR": "STR",
            "FLOAT": "FLOAT",
            "INT": "INT",
            "DECIMAL": "DECIMAL",
            "BOOL": "BOOL",
            "TIMESTAMP": "TIMESTAMP",
            "DATE": "DATE",
            "TIME": "TIME",
            "TIMEDELTA": "TIMEDELTA",
            "LIST": "LIST",
            "DICT": "DICT",
        }
    
    @schemas.classproperty
    def BYTES(cls):
        return cls("BYTES")
    
    @schemas.classproperty
    def STR(cls):
        return cls("STR")
    
    @schemas.classproperty
    def FLOAT(cls):
        return cls("FLOAT")
    
    @schemas.classproperty
    def INT(cls):
        return cls("INT")
    
    @schemas.classproperty
    def DECIMAL(cls):
        return cls("DECIMAL")
    
    @schemas.classproperty
    def BOOL(cls):
        return cls("BOOL")
    
    @schemas.classproperty
    def TIMESTAMP(cls):
        return cls("TIMESTAMP")
    
    @schemas.classproperty
    def DATE(cls):
        return cls("DATE")
    
    @schemas.classproperty
    def TIME(cls):
        return cls("TIME")
    
    @schemas.classproperty
    def TIMEDELTA(cls):
        return cls("TIMEDELTA")
    
    @schemas.classproperty
    def LIST(cls):
        return cls("LIST")
    
    @schemas.classproperty
    def DICT(cls):
        return cls("DICT")
