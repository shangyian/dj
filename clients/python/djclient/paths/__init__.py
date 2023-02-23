# do not import all endpoints into this module because that uses a lot of memory and stack frames
# if you need the ability to import all endpoints from this module, import them with
# from djclient.apis.path_to_api import path_to_api

import enum


class PathValues(str, enum.Enum):
    DATABASES_ = "/databases/"
    QUERIES_ = "/queries/"
    QUERIES_QUERY_ID_ = "/queries/{query_id}/"
    METRICS_ = "/metrics/"
    METRICS_NODE_ID_ = "/metrics/{node_id}/"
    METRICS_NODE_ID_DATA_ = "/metrics/{node_id}/data/"
    METRICS_NODE_ID_SQL_ = "/metrics/{node_id}/sql/"
    NODES_ = "/nodes/"
    GRAPHQL = "/graphql"
