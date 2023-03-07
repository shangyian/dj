# coding: utf-8

"""
    DJ server

    A DataJunction metrics layer  # noqa: E501

    The version of the OpenAPI document: 0.0.post1.dev1+g0f458e3
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


class CubeRevisionMetadata(
    schemas.DictSchema
):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.

    Metadata for a cube node
    """


    class MetaOapg:
        required = {
            "node_revision_id",
            "updated_at",
            "name",
            "cube_elements",
            "display_name",
            "type",
            "version",
            "node_id",
        }
        
        class properties:
            node_revision_id = schemas.IntSchema
            node_id = schemas.IntSchema
        
            @staticmethod
            def type() -> typing.Type['NodeType']:
                return NodeType
            name = schemas.StrSchema
            display_name = schemas.StrSchema
            version = schemas.StrSchema
            
            
            class cube_elements(
                schemas.ListSchema
            ):
            
            
                class MetaOapg:
                    
                    @staticmethod
                    def items() -> typing.Type['CubeElementMetadata']:
                        return CubeElementMetadata
            
                def __new__(
                    cls,
                    _arg: typing.Union[typing.Tuple['CubeElementMetadata'], typing.List['CubeElementMetadata']],
                    _configuration: typing.Optional[schemas.Configuration] = None,
                ) -> 'cube_elements':
                    return super().__new__(
                        cls,
                        _arg,
                        _configuration=_configuration,
                    )
            
                def __getitem__(self, i: int) -> 'CubeElementMetadata':
                    return super().__getitem__(i)
            updated_at = schemas.DateTimeSchema
            description = schemas.StrSchema
        
            @staticmethod
            def availability() -> typing.Type['AvailabilityState']:
                return AvailabilityState
            __annotations__ = {
                "node_revision_id": node_revision_id,
                "node_id": node_id,
                "type": type,
                "name": name,
                "display_name": display_name,
                "version": version,
                "cube_elements": cube_elements,
                "updated_at": updated_at,
                "description": description,
                "availability": availability,
            }
    
    node_revision_id: MetaOapg.properties.node_revision_id
    updated_at: MetaOapg.properties.updated_at
    name: MetaOapg.properties.name
    cube_elements: MetaOapg.properties.cube_elements
    display_name: MetaOapg.properties.display_name
    type: 'NodeType'
    version: MetaOapg.properties.version
    node_id: MetaOapg.properties.node_id
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["node_revision_id"]) -> MetaOapg.properties.node_revision_id: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["node_id"]) -> MetaOapg.properties.node_id: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["type"]) -> 'NodeType': ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["name"]) -> MetaOapg.properties.name: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["display_name"]) -> MetaOapg.properties.display_name: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["version"]) -> MetaOapg.properties.version: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["cube_elements"]) -> MetaOapg.properties.cube_elements: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["updated_at"]) -> MetaOapg.properties.updated_at: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["description"]) -> MetaOapg.properties.description: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["availability"]) -> 'AvailabilityState': ...
    
    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema: ...
    
    def __getitem__(self, name: typing.Union[typing_extensions.Literal["node_revision_id", "node_id", "type", "name", "display_name", "version", "cube_elements", "updated_at", "description", "availability", ], str]):
        # dict_instance[name] accessor
        return super().__getitem__(name)
    
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["node_revision_id"]) -> MetaOapg.properties.node_revision_id: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["node_id"]) -> MetaOapg.properties.node_id: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["type"]) -> 'NodeType': ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["name"]) -> MetaOapg.properties.name: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["display_name"]) -> MetaOapg.properties.display_name: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["version"]) -> MetaOapg.properties.version: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["cube_elements"]) -> MetaOapg.properties.cube_elements: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["updated_at"]) -> MetaOapg.properties.updated_at: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["description"]) -> typing.Union[MetaOapg.properties.description, schemas.Unset]: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["availability"]) -> typing.Union['AvailabilityState', schemas.Unset]: ...
    
    @typing.overload
    def get_item_oapg(self, name: str) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]: ...
    
    def get_item_oapg(self, name: typing.Union[typing_extensions.Literal["node_revision_id", "node_id", "type", "name", "display_name", "version", "cube_elements", "updated_at", "description", "availability", ], str]):
        return super().get_item_oapg(name)
    

    def __new__(
        cls,
        *_args: typing.Union[dict, frozendict.frozendict, ],
        node_revision_id: typing.Union[MetaOapg.properties.node_revision_id, decimal.Decimal, int, ],
        updated_at: typing.Union[MetaOapg.properties.updated_at, str, datetime, ],
        name: typing.Union[MetaOapg.properties.name, str, ],
        cube_elements: typing.Union[MetaOapg.properties.cube_elements, list, tuple, ],
        display_name: typing.Union[MetaOapg.properties.display_name, str, ],
        type: 'NodeType',
        version: typing.Union[MetaOapg.properties.version, str, ],
        node_id: typing.Union[MetaOapg.properties.node_id, decimal.Decimal, int, ],
        description: typing.Union[MetaOapg.properties.description, str, schemas.Unset] = schemas.unset,
        availability: typing.Union['AvailabilityState', schemas.Unset] = schemas.unset,
        _configuration: typing.Optional[schemas.Configuration] = None,
        **kwargs: typing.Union[schemas.AnyTypeSchema, dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, None, list, tuple, bytes],
    ) -> 'CubeRevisionMetadata':
        return super().__new__(
            cls,
            *_args,
            node_revision_id=node_revision_id,
            updated_at=updated_at,
            name=name,
            cube_elements=cube_elements,
            display_name=display_name,
            type=type,
            version=version,
            node_id=node_id,
            description=description,
            availability=availability,
            _configuration=_configuration,
            **kwargs,
        )

from djclient.model.availability_state import AvailabilityState
from djclient.model.cube_element_metadata import CubeElementMetadata
from djclient.model.node_type import NodeType
