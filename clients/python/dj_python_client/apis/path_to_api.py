import typing_extensions

from dj_python_client.paths import PathValues
from dj_python_client.apis.paths.databases_ import Databases
from dj_python_client.apis.paths.queries_ import Queries
from dj_python_client.apis.paths.queries_query_id_ import QueriesQueryId
from dj_python_client.apis.paths.metrics_ import Metrics
from dj_python_client.apis.paths.metrics_node_id_ import MetricsNodeId
from dj_python_client.apis.paths.metrics_node_id_data_ import MetricsNodeIdData
from dj_python_client.apis.paths.metrics_node_id_sql_ import MetricsNodeIdSql
from dj_python_client.apis.paths.nodes_ import Nodes
from dj_python_client.apis.paths.graphql import Graphql

PathToApi = typing_extensions.TypedDict(
    'PathToApi',
    {
        PathValues.DATABASES_: Databases,
        PathValues.QUERIES_: Queries,
        PathValues.QUERIES_QUERY_ID_: QueriesQueryId,
        PathValues.METRICS_: Metrics,
        PathValues.METRICS_NODE_ID_: MetricsNodeId,
        PathValues.METRICS_NODE_ID_DATA_: MetricsNodeIdData,
        PathValues.METRICS_NODE_ID_SQL_: MetricsNodeIdSql,
        PathValues.NODES_: Nodes,
        PathValues.GRAPHQL: Graphql,
    }
)

path_to_api = PathToApi(
    {
        PathValues.DATABASES_: Databases,
        PathValues.QUERIES_: Queries,
        PathValues.QUERIES_QUERY_ID_: QueriesQueryId,
        PathValues.METRICS_: Metrics,
        PathValues.METRICS_NODE_ID_: MetricsNodeId,
        PathValues.METRICS_NODE_ID_DATA_: MetricsNodeIdData,
        PathValues.METRICS_NODE_ID_SQL_: MetricsNodeIdSql,
        PathValues.NODES_: Nodes,
        PathValues.GRAPHQL: Graphql,
    }
)
