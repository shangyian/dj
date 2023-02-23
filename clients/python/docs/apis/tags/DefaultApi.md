<a name="__pageTop"></a>
# djclient.apis.tags.default_api.DefaultApi

All URIs are relative to *http://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**handle_http_get_graphql_get**](#handle_http_get_graphql_get) | **get** /graphql | Handle Http Get
[**handle_http_post_graphql_post**](#handle_http_post_graphql_post) | **post** /graphql | Handle Http Post
[**read_databases_databases_get**](#read_databases_databases_get) | **get** /databases/ | Read Databases
[**read_metric_metrics_node_id_get**](#read_metric_metrics_node_id_get) | **get** /metrics/{node_id}/ | Read Metric
[**read_metrics_data_metrics_node_id_data_get**](#read_metrics_data_metrics_node_id_data_get) | **get** /metrics/{node_id}/data/ | Read Metrics Data
[**read_metrics_metrics_get**](#read_metrics_metrics_get) | **get** /metrics/ | Read Metrics
[**read_metrics_sql_metrics_node_id_sql_get**](#read_metrics_sql_metrics_node_id_sql_get) | **get** /metrics/{node_id}/sql/ | Read Metrics Sql
[**read_nodes_nodes_get**](#read_nodes_nodes_get) | **get** /nodes/ | Read Nodes
[**read_query_queries_query_id_get**](#read_query_queries_query_id_get) | **get** /queries/{query_id}/ | Read Query
[**submit_query_queries_post**](#submit_query_queries_post) | **post** /queries/ | Submit Query

# **handle_http_get_graphql_get**
<a name="handle_http_get_graphql_get"></a>
> bool, date, datetime, dict, float, int, list, str, none_type handle_http_get_graphql_get()

Handle Http Get

### Example

```python
import djclient
from djclient.apis.tags import default_api
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = djclient.Configuration(
    host = "http://localhost"
)

# Enter a context with an instance of the API client
with djclient.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example, this endpoint has no required or optional parameters
    try:
        # Handle Http Get
        api_response = api_instance.handle_http_get_graphql_get()
        pprint(api_response)
    except djclient.ApiException as e:
        print("Exception when calling DefaultApi->handle_http_get_graphql_get: %s\n" % e)
```
### Parameters
This endpoint does not need any parameter.

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#handle_http_get_graphql_get.ApiResponseFor200) | The GraphiQL integrated development environment.
404 | [ApiResponseFor404](#handle_http_get_graphql_get.ApiResponseFor404) | Not found if GraphiQL is not enabled.

#### handle_http_get_graphql_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | 

#### handle_http_get_graphql_get.ApiResponseFor404
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | Unset | body was not defined |
headers | Unset | headers were not defined |

### Authorization

No authorization required

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **handle_http_post_graphql_post**
<a name="handle_http_post_graphql_post"></a>
> bool, date, datetime, dict, float, int, list, str, none_type handle_http_post_graphql_post()

Handle Http Post

### Example

```python
import djclient
from djclient.apis.tags import default_api
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = djclient.Configuration(
    host = "http://localhost"
)

# Enter a context with an instance of the API client
with djclient.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example, this endpoint has no required or optional parameters
    try:
        # Handle Http Post
        api_response = api_instance.handle_http_post_graphql_post()
        pprint(api_response)
    except djclient.ApiException as e:
        print("Exception when calling DefaultApi->handle_http_post_graphql_post: %s\n" % e)
```
### Parameters
This endpoint does not need any parameter.

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#handle_http_post_graphql_post.ApiResponseFor200) | Successful Response

#### handle_http_post_graphql_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | 

### Authorization

No authorization required

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **read_databases_databases_get**
<a name="read_databases_databases_get"></a>
> [Database] read_databases_databases_get()

Read Databases

List the available databases.

### Example

```python
import djclient
from djclient.apis.tags import default_api
from djclient.model.database import Database
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = djclient.Configuration(
    host = "http://localhost"
)

# Enter a context with an instance of the API client
with djclient.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example, this endpoint has no required or optional parameters
    try:
        # Read Databases
        api_response = api_instance.read_databases_databases_get()
        pprint(api_response)
    except djclient.ApiException as e:
        print("Exception when calling DefaultApi->read_databases_databases_get: %s\n" % e)
```
### Parameters
This endpoint does not need any parameter.

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#read_databases_databases_get.ApiResponseFor200) | Successful Response

#### read_databases_databases_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple,  | tuple,  |  | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[**Database**]({{complexTypePrefix}}Database.md) | [**Database**]({{complexTypePrefix}}Database.md) | [**Database**]({{complexTypePrefix}}Database.md) |  | 

### Authorization

No authorization required

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **read_metric_metrics_node_id_get**
<a name="read_metric_metrics_node_id_get"></a>
> Metric read_metric_metrics_node_id_get(node_id)

Read Metric

Return a metric by ID.

### Example

```python
import djclient
from djclient.apis.tags import default_api
from djclient.model.http_validation_error import HTTPValidationError
from djclient.model.metric import Metric
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = djclient.Configuration(
    host = "http://localhost"
)

# Enter a context with an instance of the API client
with djclient.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'node_id': 1,
    }
    try:
        # Read Metric
        api_response = api_instance.read_metric_metrics_node_id_get(
            path_params=path_params,
        )
        pprint(api_response)
    except djclient.ApiException as e:
        print("Exception when calling DefaultApi->read_metric_metrics_node_id_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
node_id | NodeIdSchema | | 

# NodeIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
decimal.Decimal, int,  | decimal.Decimal,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#read_metric_metrics_node_id_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#read_metric_metrics_node_id_get.ApiResponseFor422) | Validation Error

#### read_metric_metrics_node_id_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**Metric**](../../models/Metric.md) |  | 


#### read_metric_metrics_node_id_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

No authorization required

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **read_metrics_data_metrics_node_id_data_get**
<a name="read_metrics_data_metrics_node_id_data_get"></a>
> QueryWithResults read_metrics_data_metrics_node_id_data_get(node_id)

Read Metrics Data

Return data for a metric.

### Example

```python
import djclient
from djclient.apis.tags import default_api
from djclient.model.http_validation_error import HTTPValidationError
from djclient.model.query_with_results import QueryWithResults
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = djclient.Configuration(
    host = "http://localhost"
)

# Enter a context with an instance of the API client
with djclient.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'node_id': 1,
    }
    query_params = {
    }
    try:
        # Read Metrics Data
        api_response = api_instance.read_metrics_data_metrics_node_id_data_get(
            path_params=path_params,
            query_params=query_params,
        )
        pprint(api_response)
    except djclient.ApiException as e:
        print("Exception when calling DefaultApi->read_metrics_data_metrics_node_id_data_get: %s\n" % e)

    # example passing only optional values
    path_params = {
        'node_id': 1,
    }
    query_params = {
        'database_id': 1,
        'd': [],
        'f': [],
    }
    try:
        # Read Metrics Data
        api_response = api_instance.read_metrics_data_metrics_node_id_data_get(
            path_params=path_params,
            query_params=query_params,
        )
        pprint(api_response)
    except djclient.ApiException as e:
        print("Exception when calling DefaultApi->read_metrics_data_metrics_node_id_data_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
query_params | RequestQueryParams | |
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
database_id | DatabaseIdSchema | | optional
d | DSchema | | optional
f | FSchema | | optional


# DatabaseIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
decimal.Decimal, int,  | decimal.Decimal,  |  | 

# DSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple,  | tuple,  |  | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
items | str,  | str,  |  | 

# FSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple,  | tuple,  |  | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
items | str,  | str,  |  | 

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
node_id | NodeIdSchema | | 

# NodeIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
decimal.Decimal, int,  | decimal.Decimal,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#read_metrics_data_metrics_node_id_data_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#read_metrics_data_metrics_node_id_data_get.ApiResponseFor422) | Validation Error

#### read_metrics_data_metrics_node_id_data_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**QueryWithResults**](../../models/QueryWithResults.md) |  | 


#### read_metrics_data_metrics_node_id_data_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

No authorization required

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **read_metrics_metrics_get**
<a name="read_metrics_metrics_get"></a>
> [Metric] read_metrics_metrics_get()

Read Metrics

List all available metrics.

### Example

```python
import djclient
from djclient.apis.tags import default_api
from djclient.model.metric import Metric
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = djclient.Configuration(
    host = "http://localhost"
)

# Enter a context with an instance of the API client
with djclient.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example, this endpoint has no required or optional parameters
    try:
        # Read Metrics
        api_response = api_instance.read_metrics_metrics_get()
        pprint(api_response)
    except djclient.ApiException as e:
        print("Exception when calling DefaultApi->read_metrics_metrics_get: %s\n" % e)
```
### Parameters
This endpoint does not need any parameter.

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#read_metrics_metrics_get.ApiResponseFor200) | Successful Response

#### read_metrics_metrics_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple,  | tuple,  |  | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[**Metric**]({{complexTypePrefix}}Metric.md) | [**Metric**]({{complexTypePrefix}}Metric.md) | [**Metric**]({{complexTypePrefix}}Metric.md) |  | 

### Authorization

No authorization required

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **read_metrics_sql_metrics_node_id_sql_get**
<a name="read_metrics_sql_metrics_node_id_sql_get"></a>
> TranslatedSQL read_metrics_sql_metrics_node_id_sql_get(node_id)

Read Metrics Sql

Return SQL for a metric.  A database can be optionally specified. If no database is specified the optimal one will be used.

### Example

```python
import djclient
from djclient.apis.tags import default_api
from djclient.model.http_validation_error import HTTPValidationError
from djclient.model.translated_sql import TranslatedSQL
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = djclient.Configuration(
    host = "http://localhost"
)

# Enter a context with an instance of the API client
with djclient.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'node_id': 1,
    }
    query_params = {
    }
    try:
        # Read Metrics Sql
        api_response = api_instance.read_metrics_sql_metrics_node_id_sql_get(
            path_params=path_params,
            query_params=query_params,
        )
        pprint(api_response)
    except djclient.ApiException as e:
        print("Exception when calling DefaultApi->read_metrics_sql_metrics_node_id_sql_get: %s\n" % e)

    # example passing only optional values
    path_params = {
        'node_id': 1,
    }
    query_params = {
        'database_id': 1,
        'd': [],
        'f': [],
    }
    try:
        # Read Metrics Sql
        api_response = api_instance.read_metrics_sql_metrics_node_id_sql_get(
            path_params=path_params,
            query_params=query_params,
        )
        pprint(api_response)
    except djclient.ApiException as e:
        print("Exception when calling DefaultApi->read_metrics_sql_metrics_node_id_sql_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
query_params | RequestQueryParams | |
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
database_id | DatabaseIdSchema | | optional
d | DSchema | | optional
f | FSchema | | optional


# DatabaseIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
decimal.Decimal, int,  | decimal.Decimal,  |  | 

# DSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple,  | tuple,  |  | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
items | str,  | str,  |  | 

# FSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple,  | tuple,  |  | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
items | str,  | str,  |  | 

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
node_id | NodeIdSchema | | 

# NodeIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
decimal.Decimal, int,  | decimal.Decimal,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#read_metrics_sql_metrics_node_id_sql_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#read_metrics_sql_metrics_node_id_sql_get.ApiResponseFor422) | Validation Error

#### read_metrics_sql_metrics_node_id_sql_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**TranslatedSQL**](../../models/TranslatedSQL.md) |  | 


#### read_metrics_sql_metrics_node_id_sql_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

No authorization required

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **read_nodes_nodes_get**
<a name="read_nodes_nodes_get"></a>
> [NodeMetadata] read_nodes_nodes_get()

Read Nodes

List the available nodes.

### Example

```python
import djclient
from djclient.apis.tags import default_api
from djclient.model.node_metadata import NodeMetadata
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = djclient.Configuration(
    host = "http://localhost"
)

# Enter a context with an instance of the API client
with djclient.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example, this endpoint has no required or optional parameters
    try:
        # Read Nodes
        api_response = api_instance.read_nodes_nodes_get()
        pprint(api_response)
    except djclient.ApiException as e:
        print("Exception when calling DefaultApi->read_nodes_nodes_get: %s\n" % e)
```
### Parameters
This endpoint does not need any parameter.

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#read_nodes_nodes_get.ApiResponseFor200) | Successful Response

#### read_nodes_nodes_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple,  | tuple,  |  | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[**NodeMetadata**]({{complexTypePrefix}}NodeMetadata.md) | [**NodeMetadata**]({{complexTypePrefix}}NodeMetadata.md) | [**NodeMetadata**]({{complexTypePrefix}}NodeMetadata.md) |  | 

### Authorization

No authorization required

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **read_query_queries_query_id_get**
<a name="read_query_queries_query_id_get"></a>
> QueryWithResults read_query_queries_query_id_get(query_id)

Read Query

Fetch information about a query.  For paginated queries we move the data from the results backend to the cache for a short period, anticipating additional requests.

### Example

```python
import djclient
from djclient.apis.tags import default_api
from djclient.model.http_validation_error import HTTPValidationError
from djclient.model.query_with_results import QueryWithResults
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = djclient.Configuration(
    host = "http://localhost"
)

# Enter a context with an instance of the API client
with djclient.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'query_id': "query_id_example",
    }
    query_params = {
    }
    try:
        # Read Query
        api_response = api_instance.read_query_queries_query_id_get(
            path_params=path_params,
            query_params=query_params,
        )
        pprint(api_response)
    except djclient.ApiException as e:
        print("Exception when calling DefaultApi->read_query_queries_query_id_get: %s\n" % e)

    # example passing only optional values
    path_params = {
        'query_id': "query_id_example",
    }
    query_params = {
        'limit': 0,
        'offset': 0,
    }
    try:
        # Read Query
        api_response = api_instance.read_query_queries_query_id_get(
            path_params=path_params,
            query_params=query_params,
        )
        pprint(api_response)
    except djclient.ApiException as e:
        print("Exception when calling DefaultApi->read_query_queries_query_id_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
query_params | RequestQueryParams | |
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
limit | LimitSchema | | optional
offset | OffsetSchema | | optional


# LimitSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
decimal.Decimal, int,  | decimal.Decimal,  |  | if omitted the server will use the default value of 0

# OffsetSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
decimal.Decimal, int,  | decimal.Decimal,  |  | if omitted the server will use the default value of 0

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
query_id | QueryIdSchema | | 

# QueryIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str, uuid.UUID,  | str,  |  | value must be a uuid

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#read_query_queries_query_id_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#read_query_queries_query_id_get.ApiResponseFor422) | Validation Error

#### read_query_queries_query_id_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**QueryWithResults**](../../models/QueryWithResults.md) |  | 


#### read_query_queries_query_id_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

No authorization required

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **submit_query_queries_post**
<a name="submit_query_queries_post"></a>
> QueryWithResults submit_query_queries_post(any_type)

Submit Query

Run or schedule a query.  This endpoint is different from others in that it accepts both JSON and msgpack, and can also return JSON or msgpack, depending on HTTP headers.

### Example

```python
import djclient
from djclient.apis.tags import default_api
from djclient.model.http_validation_error import HTTPValidationError
from djclient.model.query_with_results import QueryWithResults
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = djclient.Configuration(
    host = "http://localhost"
)

# Enter a context with an instance of the API client
with djclient.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    header_params = {
    }
    body = dict(
        database_id=1,
        catalog="catalog_example",
        schema="schema_example",
        submitted_query="submitted_query_example",
    )
    try:
        # Submit Query
        api_response = api_instance.submit_query_queries_post(
            header_params=header_params,
            body=body,
        )
        pprint(api_response)
    except djclient.ApiException as e:
        print("Exception when calling DefaultApi->submit_query_queries_post: %s\n" % e)

    # example passing only optional values
    header_params = {
        'accept': "accept_example",
    }
    body = dict(
        database_id=1,
        catalog="catalog_example",
        schema="schema_example",
        submitted_query="submitted_query_example",
    )
    try:
        # Submit Query
        api_response = api_instance.submit_query_queries_post(
            header_params=header_params,
            body=body,
        )
        pprint(api_response)
    except djclient.ApiException as e:
        print("Exception when calling DefaultApi->submit_query_queries_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson, SchemaForRequestBodyApplicationMsgpack] | required |
header_params | RequestHeaderParams | |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', 'application/msgpack', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson

Model for submitted queries.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  | Model for submitted queries. | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**database_id** | decimal.Decimal, int,  | decimal.Decimal,  |  | 
**submitted_query** | str,  | str,  |  | 
**catalog** | str,  | str,  |  | [optional] 
**schema** | str,  | str,  |  | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# SchemaForRequestBodyApplicationMsgpack

Model for submitted queries.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  | Model for submitted queries. | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**database_id** | decimal.Decimal, int,  | decimal.Decimal,  |  | 
**submitted_query** | str,  | str,  |  | 
**catalog** | str,  | str,  |  | [optional] 
**schema** | str,  | str,  |  | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

### header_params
#### RequestHeaderParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
accept | AcceptSchema | | optional

# AcceptSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#submit_query_queries_post.ApiResponseFor200) | Return results as JSON or msgpack
422 | [ApiResponseFor422](#submit_query_queries_post.ApiResponseFor422) | Validation Error

#### submit_query_queries_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, Unset, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**QueryWithResults**](../../models/QueryWithResults.md) |  | 


#### submit_query_queries_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

No authorization required

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

