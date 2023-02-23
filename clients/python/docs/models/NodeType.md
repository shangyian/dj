# dj_python_client.model.node_type.NodeType

     Node type.      A node can have 4 types, currently:      1. SOURCE nodes are root nodes in the DAG, and point to tables or views in a DB.     2. TRANSFORM nodes are SQL transformations, reading from SOURCE/TRANSFORM nodes.     3. METRIC nodes are leaves in the DAG, and have a single aggregation query.     4. DIMENSION nodes are special SOURCE nodes that can be auto-joined with METRICS.     

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |      Node type.      A node can have 4 types, currently:      1. SOURCE nodes are root nodes in the DAG, and point to tables or views in a DB.     2. TRANSFORM nodes are SQL transformations, reading from SOURCE/TRANSFORM nodes.     3. METRIC nodes are leaves in the DAG, and have a single aggregation query.     4. DIMENSION nodes are special SOURCE nodes that can be auto-joined with METRICS.      | must be one of ["source", "transform", "metric", "dimension", ] 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

