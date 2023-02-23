# djclient.model.node_metadata.NodeMetadata

A node with information about columns and if it is a metric.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  | A node with information about columns and if it is a metric. | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**updated_at** | str, datetime,  | str,  |  | value must conform to RFC-3339 date-time
**[columns](#columns)** | list, tuple,  | tuple,  |  | 
**name** | str,  | str,  |  | 
**created_at** | str, datetime,  | str,  |  | value must conform to RFC-3339 date-time
**id** | decimal.Decimal, int,  | decimal.Decimal,  |  | 
**type** | [**NodeType**](NodeType.md) | [**NodeType**](NodeType.md) |  | 
**description** | str,  | str,  |  | [optional] if omitted the server will use the default value of ""
**query** | str,  | str,  |  | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# columns

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple,  | tuple,  |  | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[**SimpleColumn**](SimpleColumn.md) | [**SimpleColumn**](SimpleColumn.md) | [**SimpleColumn**](SimpleColumn.md) |  | 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

