# coding: utf-8

"""
    DJ server

    A DataJunction metrics layer  # noqa: E501

    The version of the OpenAPI document: 0.0.post1.dev1+gd5a7903
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

from djopenapi import schemas  # noqa: F401


class ColumnAttributeInput(
    schemas.DictSchema
):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.

    A column attribute input
    """


    class MetaOapg:
        required = {
            "attribute_type_name",
            "column_name",
        }
        
        class properties:
            attribute_type_name = schemas.StrSchema
            column_name = schemas.StrSchema
            attribute_type_namespace = schemas.StrSchema
            __annotations__ = {
                "attribute_type_name": attribute_type_name,
                "column_name": column_name,
                "attribute_type_namespace": attribute_type_namespace,
            }
    
    attribute_type_name: MetaOapg.properties.attribute_type_name
    column_name: MetaOapg.properties.column_name
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["attribute_type_name"]) -> MetaOapg.properties.attribute_type_name: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["column_name"]) -> MetaOapg.properties.column_name: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["attribute_type_namespace"]) -> MetaOapg.properties.attribute_type_namespace: ...
    
    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema: ...
    
    def __getitem__(self, name: typing.Union[typing_extensions.Literal["attribute_type_name", "column_name", "attribute_type_namespace", ], str]):
        # dict_instance[name] accessor
        return super().__getitem__(name)
    
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["attribute_type_name"]) -> MetaOapg.properties.attribute_type_name: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["column_name"]) -> MetaOapg.properties.column_name: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["attribute_type_namespace"]) -> typing.Union[MetaOapg.properties.attribute_type_namespace, schemas.Unset]: ...
    
    @typing.overload
    def get_item_oapg(self, name: str) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]: ...
    
    def get_item_oapg(self, name: typing.Union[typing_extensions.Literal["attribute_type_name", "column_name", "attribute_type_namespace", ], str]):
        return super().get_item_oapg(name)
    

    def __new__(
        cls,
        *_args: typing.Union[dict, frozendict.frozendict, ],
        attribute_type_name: typing.Union[MetaOapg.properties.attribute_type_name, str, ],
        column_name: typing.Union[MetaOapg.properties.column_name, str, ],
        attribute_type_namespace: typing.Union[MetaOapg.properties.attribute_type_namespace, str, schemas.Unset] = schemas.unset,
        _configuration: typing.Optional[schemas.Configuration] = None,
        **kwargs: typing.Union[schemas.AnyTypeSchema, dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, None, list, tuple, bytes],
    ) -> 'ColumnAttributeInput':
        return super().__new__(
            cls,
            *_args,
            attribute_type_name=attribute_type_name,
            column_name=column_name,
            attribute_type_namespace=attribute_type_namespace,
            _configuration=_configuration,
            **kwargs,
        )
