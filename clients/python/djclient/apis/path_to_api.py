import typing_extensions

from djclient.paths import PathValues
from djclient.apis.paths.databases_ import Databases
from djclient.apis.paths.queries_ import Queries
from djclient.apis.paths.queries_query_id_ import QueriesQueryId
from djclient.apis.paths.metrics_ import Metrics
from djclient.apis.paths.metrics_node_id_ import MetricsNodeId
from djclient.apis.paths.metrics_node_id_data_ import MetricsNodeIdData
from djclient.apis.paths.metrics_node_id_sql_ import MetricsNodeIdSql
from djclient.apis.paths.nodes_ import Nodes
from djclient.apis.paths.graphql import Graphql

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
