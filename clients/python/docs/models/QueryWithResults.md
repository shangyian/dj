# dj_python_client.model.query_with_results.QueryWithResults

Model for query with results.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  | Model for query with results. | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**database_id** | decimal.Decimal, int,  | decimal.Decimal,  |  | 
**submitted_query** | str,  | str,  |  | 
**id** | str, uuid.UUID,  | str,  |  | value must be a uuid
**results** | [**QueryResults**](QueryResults.md) | [**QueryResults**](QueryResults.md) |  | 
**[errors](#errors)** | list, tuple,  | tuple,  |  | 
**catalog** | str,  | str,  |  | [optional] 
**schema** | str,  | str,  |  | [optional] 
**executed_query** | str,  | str,  |  | [optional] 
**scheduled** | str, datetime,  | str,  |  | [optional] value must conform to RFC-3339 date-time
**started** | str, datetime,  | str,  |  | [optional] value must conform to RFC-3339 date-time
**finished** | str, datetime,  | str,  |  | [optional] value must conform to RFC-3339 date-time
**[state](#state)** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | [optional] if omitted the server will use the default value of UNKNOWN
**progress** | decimal.Decimal, int, float,  | decimal.Decimal,  |  | [optional] if omitted the server will use the default value of 0.0
**next** | str,  | str,  |  | [optional] 
**previous** | str,  | str,  |  | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# errors

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple,  | tuple,  |  | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
items | str,  | str,  |  | 

# state

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | if omitted the server will use the default value of UNKNOWN

### Composed Schemas (allOf/anyOf/oneOf/not)
#### allOf
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[QueryState](QueryState.md) | [**QueryState**](QueryState.md) | [**QueryState**](QueryState.md) |  | 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

