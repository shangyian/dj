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

from djclient import schemas  # noqa: F401


class QueryWithResults(
    schemas.DictSchema
):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.

    Model for query with results.
    """


    class MetaOapg:
        required = {
            "database_id",
            "submitted_query",
            "id",
            "results",
            "errors",
        }
        
        class properties:
            database_id = schemas.IntSchema
            id = schemas.UUIDSchema
            submitted_query = schemas.StrSchema
        
            @staticmethod
            def results() -> typing.Type['QueryResults']:
                return QueryResults
            
            
            class errors(
                schemas.ListSchema
            ):
            
            
                class MetaOapg:
                    items = schemas.StrSchema
            
                def __new__(
                    cls,
                    _arg: typing.Union[typing.Tuple[typing.Union[MetaOapg.items, str, ]], typing.List[typing.Union[MetaOapg.items, str, ]]],
                    _configuration: typing.Optional[schemas.Configuration] = None,
                ) -> 'errors':
                    return super().__new__(
                        cls,
                        _arg,
                        _configuration=_configuration,
                    )
            
                def __getitem__(self, i: int) -> MetaOapg.items:
                    return super().__getitem__(i)
            catalog = schemas.StrSchema
            schema = schemas.StrSchema
            executed_query = schemas.StrSchema
            scheduled = schemas.DateTimeSchema
            started = schemas.DateTimeSchema
            finished = schemas.DateTimeSchema
            
            
            class state(
                schemas.ComposedSchema,
            ):
            
            
                class MetaOapg:
                    
                    @classmethod
                    @functools.lru_cache()
                    def all_of(cls):
                        # we need this here to make our import statements work
                        # we must store _composed_schemas in here so the code is only run
                        # when we invoke this method. If we kept this at the class
                        # level we would get an error because the class level
                        # code would be run when this module is imported, and these composed
                        # classes don't exist yet because their module has not finished
                        # loading
                        return [
                            QueryState,
                        ]
            
            
                def __new__(
                    cls,
                    *_args: typing.Union[dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader, ],
                    _configuration: typing.Optional[schemas.Configuration] = None,
                    **kwargs: typing.Union[schemas.AnyTypeSchema, dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, None, list, tuple, bytes],
                ) -> 'state':
                    return super().__new__(
                        cls,
                        *_args,
                        _configuration=_configuration,
                        **kwargs,
                    )
            progress = schemas.NumberSchema
            
            
            class next(
                schemas.StrSchema
            ):
            
            
                class MetaOapg:
                    format = 'uri'
                    max_length = 65536
                    min_length = 1
            
            
            class previous(
                schemas.StrSchema
            ):
            
            
                class MetaOapg:
                    format = 'uri'
                    max_length = 65536
                    min_length = 1
            __annotations__ = {
                "database_id": database_id,
                "id": id,
                "submitted_query": submitted_query,
                "results": results,
                "errors": errors,
                "catalog": catalog,
                "schema": schema,
                "executed_query": executed_query,
                "scheduled": scheduled,
                "started": started,
                "finished": finished,
                "state": state,
                "progress": progress,
                "next": next,
                "previous": previous,
            }
    
    database_id: MetaOapg.properties.database_id
    submitted_query: MetaOapg.properties.submitted_query
    id: MetaOapg.properties.id
    results: 'QueryResults'
    errors: MetaOapg.properties.errors
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["database_id"]) -> MetaOapg.properties.database_id: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["id"]) -> MetaOapg.properties.id: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["submitted_query"]) -> MetaOapg.properties.submitted_query: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["results"]) -> 'QueryResults': ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["errors"]) -> MetaOapg.properties.errors: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["catalog"]) -> MetaOapg.properties.catalog: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["schema"]) -> MetaOapg.properties.schema: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["executed_query"]) -> MetaOapg.properties.executed_query: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["scheduled"]) -> MetaOapg.properties.scheduled: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["started"]) -> MetaOapg.properties.started: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["finished"]) -> MetaOapg.properties.finished: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["state"]) -> MetaOapg.properties.state: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["progress"]) -> MetaOapg.properties.progress: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["next"]) -> MetaOapg.properties.next: ...
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["previous"]) -> MetaOapg.properties.previous: ...
    
    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema: ...
    
    def __getitem__(self, name: typing.Union[typing_extensions.Literal["database_id", "id", "submitted_query", "results", "errors", "catalog", "schema", "executed_query", "scheduled", "started", "finished", "state", "progress", "next", "previous", ], str]):
        # dict_instance[name] accessor
        return super().__getitem__(name)
    
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["database_id"]) -> MetaOapg.properties.database_id: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["id"]) -> MetaOapg.properties.id: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["submitted_query"]) -> MetaOapg.properties.submitted_query: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["results"]) -> 'QueryResults': ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["errors"]) -> MetaOapg.properties.errors: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["catalog"]) -> typing.Union[MetaOapg.properties.catalog, schemas.Unset]: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["schema"]) -> typing.Union[MetaOapg.properties.schema, schemas.Unset]: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["executed_query"]) -> typing.Union[MetaOapg.properties.executed_query, schemas.Unset]: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["scheduled"]) -> typing.Union[MetaOapg.properties.scheduled, schemas.Unset]: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["started"]) -> typing.Union[MetaOapg.properties.started, schemas.Unset]: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["finished"]) -> typing.Union[MetaOapg.properties.finished, schemas.Unset]: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["state"]) -> typing.Union[MetaOapg.properties.state, schemas.Unset]: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["progress"]) -> typing.Union[MetaOapg.properties.progress, schemas.Unset]: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["next"]) -> typing.Union[MetaOapg.properties.next, schemas.Unset]: ...
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["previous"]) -> typing.Union[MetaOapg.properties.previous, schemas.Unset]: ...
    
    @typing.overload
    def get_item_oapg(self, name: str) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]: ...
    
    def get_item_oapg(self, name: typing.Union[typing_extensions.Literal["database_id", "id", "submitted_query", "results", "errors", "catalog", "schema", "executed_query", "scheduled", "started", "finished", "state", "progress", "next", "previous", ], str]):
        return super().get_item_oapg(name)
    

    def __new__(
        cls,
        *_args: typing.Union[dict, frozendict.frozendict, ],
        database_id: typing.Union[MetaOapg.properties.database_id, decimal.Decimal, int, ],
        submitted_query: typing.Union[MetaOapg.properties.submitted_query, str, ],
        id: typing.Union[MetaOapg.properties.id, str, uuid.UUID, ],
        results: 'QueryResults',
        errors: typing.Union[MetaOapg.properties.errors, list, tuple, ],
        catalog: typing.Union[MetaOapg.properties.catalog, str, schemas.Unset] = schemas.unset,
        schema: typing.Union[MetaOapg.properties.schema, str, schemas.Unset] = schemas.unset,
        executed_query: typing.Union[MetaOapg.properties.executed_query, str, schemas.Unset] = schemas.unset,
        scheduled: typing.Union[MetaOapg.properties.scheduled, str, datetime, schemas.Unset] = schemas.unset,
        started: typing.Union[MetaOapg.properties.started, str, datetime, schemas.Unset] = schemas.unset,
        finished: typing.Union[MetaOapg.properties.finished, str, datetime, schemas.Unset] = schemas.unset,
        state: typing.Union[MetaOapg.properties.state, dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader, schemas.Unset] = schemas.unset,
        progress: typing.Union[MetaOapg.properties.progress, decimal.Decimal, int, float, schemas.Unset] = schemas.unset,
        next: typing.Union[MetaOapg.properties.next, str, schemas.Unset] = schemas.unset,
        previous: typing.Union[MetaOapg.properties.previous, str, schemas.Unset] = schemas.unset,
        _configuration: typing.Optional[schemas.Configuration] = None,
        **kwargs: typing.Union[schemas.AnyTypeSchema, dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, None, list, tuple, bytes],
    ) -> 'QueryWithResults':
        return super().__new__(
            cls,
            *_args,
            database_id=database_id,
            submitted_query=submitted_query,
            id=id,
            results=results,
            errors=errors,
            catalog=catalog,
            schema=schema,
            executed_query=executed_query,
            scheduled=scheduled,
            started=started,
            finished=finished,
            state=state,
            progress=progress,
            next=next,
            previous=previous,
            _configuration=_configuration,
            **kwargs,
        )

from djclient.model.query_results import QueryResults
from djclient.model.query_state import QueryState