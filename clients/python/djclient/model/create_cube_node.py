# coding: utf-8

"""
    DJ server

    A DataJunction metrics layer  # noqa: E501

    The version of the OpenAPI document: 0.0.post1.dev1+g5d0aa56
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

from djclient import schemas  # noqa: F401


class CreateCubeNode(
    schemas.DictSchema
):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.

    A create object for cube nodes
    """


    class MetaOapg:
        required = {
            "mode",
            "name",
            "cube_elements",
            "description",
            "type",
        }
        
        class properties:
            
            
            class cube_elements(
                schemas.ListSchema
            ):
            
            
                class MetaOapg:
                    items = schemas.StrSchema
            
                def __new__(
                    cls,
                    _arg: typing.Union[typing.Tuple[typing.Union[MetaOapg.items, str, ]], typing.List[typing.Union[MetaOapg.items, str, ]]],
                    _configuration: typing.Optional[schemas.Configuration] = None,
                ) -> 'cube_elements':
                    return super().__new__(
                        cls,
                        _arg,
                        _configuration=_configuration,
                    )
            
                def __getitem__(self, i: int) -> MetaOapg.items:
                    return super().__getitem__(i)
            description = schemas.StrSchema
        
            @staticmethod
            def mode() -> typing.Type['NodeMode']:
                return NodeMode
            name = schemas.StrSchema
        
            @staticmethod
            def type() -> typing.Type['NodeType']:
                return NodeType
            display_name = schemas.StrSchema
            __annotations__ = {
                "cube_elements": cube_elements,
                "description": description,
                "mode": mode,
                "name": name,
                "type": type,
                "display_name": display_name,
            }
    
    mode: 'NodeMode'
    name: MetaOapg.properties.name
    cube_elements: MetaOapg.properties.cube_elements
    description: MetaOapg.properties.description
    type: 'NodeType'
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["cube_elements"]) -> MetaOapg.properties.cube_elements: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["description"]) -> MetaOapg.properties.description: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["mode"]) -> 'NodeMode': ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["name"]) -> MetaOapg.properties.name: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["type"]) -> 'NodeType': ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["display_name"]) -> MetaOapg.properties.display_name: ...
    
    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema: ...
    
    def __getitem__(self, name: typing.Union[typing_extensions.Literal["cube_elements", "description", "mode", "name", "type", "display_name", ], str]):
        # dict_instance[name] accessor
        return super().__getitem__(name)
    
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["cube_elements"]) -> MetaOapg.properties.cube_elements: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["description"]) -> MetaOapg.properties.description: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["mode"]) -> 'NodeMode': ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["name"]) -> MetaOapg.properties.name: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["type"]) -> 'NodeType': ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["display_name"]) -> typing.Union[MetaOapg.properties.display_name, schemas.Unset]: ...
    
    @typing.overload
    def get_item_oapg(self, name: str) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]: ...
    
    def get_item_oapg(self, name: typing.Union[typing_extensions.Literal["cube_elements", "description", "mode", "name", "type", "display_name", ], str]):
        return super().get_item_oapg(name)
    

    def __new__(
        cls,
        *_args: typing.Union[dict, frozendict.frozendict, ],
        mode: 'NodeMode',
        name: typing.Union[MetaOapg.properties.name, str, ],
        cube_elements: typing.Union[MetaOapg.properties.cube_elements, list, tuple, ],
        description: typing.Union[MetaOapg.properties.description, str, ],
        type: 'NodeType',
        display_name: typing.Union[MetaOapg.properties.display_name, str, schemas.Unset] = schemas.unset,
        _configuration: typing.Optional[schemas.Configuration] = None,
        **kwargs: typing.Union[schemas.AnyTypeSchema, dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, None, list, tuple, bytes],
    ) -> 'CreateCubeNode':
        return super().__new__(
            cls,
            *_args,
            mode=mode,
            name=name,
            cube_elements=cube_elements,
            description=description,
            type=type,
            display_name=display_name,
            _configuration=_configuration,
            **kwargs,
        )

from djclient.model.node_mode import NodeMode
from djclient.model.node_type import NodeType