"""Functions to add to an ast DJ node queries"""
import collections
# pylint: disable=too-many-arguments,too-many-locals,too-many-nested-blocks,too-many-branches,R0401
from typing import Dict, List, Optional, Union, cast, DefaultDict, Set, Tuple

from sqlmodel import Session

from dj.construction.utils import amenable_name
from dj.errors import DJException
from dj.models.column import Column
from dj.models.node import BuildCriteria, NodeRevision, NodeType
from dj.sql.parsing.ast2 import CompileContext
from dj.sql.parsing.backends.antlr4 import ast, parse


def _get_tables_from_select(select: ast.SelectExpression) -> DefaultDict[NodeRevision, List[ast.Table]]:
    """
    Extract all tables (source, transform, dimensions)
    """
    tables: DefaultDict[NodeRevision, List[ast.Table]] = collections.defaultdict(list)

    for table in select.find_all(ast.Table):
        if node := table.dj_node:  # pragma: no cover
            tables[node] = tables.get(node, [])
            tables[node].append(table)
    return tables


def _get_dim_cols_from_select(
    select: ast.SelectExpression,
) -> Dict[NodeRevision, List[ast.Column]]:
    """
    Extract all dimension nodes referenced as columns
    """
    dimension_columns: Dict[NodeRevision, List[ast.Column]] = {}

    for col in select.find_all(ast.Column):
        if isinstance(col.table, ast.Table):
            if node := col.table.dj_node:  # pragma: no cover
                if node.type == NodeType.DIMENSION:
                    dimension_columns[node] = dimension_columns.get(node, [])
                    dimension_columns[node].append(col)
    return dimension_columns


def join_path(
    dimension_node: NodeRevision,
    initial_nodes: Set[NodeRevision]
) -> Dict[Tuple[NodeRevision, NodeRevision], List[Column]]:
    """
    For a dimension node, we want to find a possible join path between it
    and any of the nodes that are directly referenced in the original query. If
    no join path exists, returns an empty dict.
    """
    processed = set()

    to_process = collections.deque([])
    # mapping of node to join -> a list of join columns
    join_info: Dict[Tuple[NodeRevision], List[Column]] = {}
    to_process.extend([(node, join_info.copy()) for node in initial_nodes])

    while to_process:
        current_node, path = to_process.popleft()
        processed.add(current_node)
        dimensions_to_columns = collections.defaultdict(list)

        # From the columns on the current node, find the next layer of
        # dimension nodes that can be joined in
        for col in current_node.columns:
            if col.dimension:
                dimensions_to_columns[col.dimension.current].append(col)

        # Go through all potential dimensions and their join columns
        for joinable_dim, join_cols in dimensions_to_columns.items():
            next_join_path = {**path, **{(current_node, joinable_dim): join_cols}}
            full_join_path = (joinable_dim, next_join_path)
            if joinable_dim == dimension_node:
                for col in join_cols:
                    if col.dimension_column is None and not any(
                        dim_col.name == "id" for dim_col in dimension_node.columns
                    ):
                        raise DJException(
                            f"Node {current_node.name} specifying dimension "
                            f"{joinable_dim.name} on column {col.name} does not"
                            f" specify a dimension column, but {dimension_node.name} "
                            f"does not have the default key `id`.",
                        )
                return full_join_path
            if joinable_dim not in processed:
                to_process.append(full_join_path)
                for parent in joinable_dim.parents:
                    to_process.append((parent.current, next_join_path))
    return None


def join_tables_for_dimensions(
    session: Session,
    dimension_columns: Dict[NodeRevision, List[ast.Column]],
    tables: DefaultDict[NodeRevision, List[ast.Table]],
    build_criteria: Optional[BuildCriteria] = None,
):
    """
    Joins the tables necessary for a set of filter and group by dimensions
    onto the select expression.
    In some cases, the necessary tables will already be on the select and
    no additional joins will be needed. However, if the tables are not in
    the select, it will traverse through available linked tables (via dimension
    nodes) and join them in.
    """

    for dim_node, dim_cols in dimension_columns.items():
        selects_map = {}
        for dim_col in dim_cols:
            select = cast(ast.Select, dim_col.get_nearest_parent_of_type(ast.Select))
            selects_map[id(select)] = select
        for select in selects_map.values():
            initial_nodes = {table.dj_node for table in select.find_all(ast.Table) if table.dj_node}
            if dim_node not in initial_nodes:  # need to join dimension
                node, paths = join_path(dim_node, initial_nodes)
                join_info: List[Tuple[NodeRevision], List[Column]] = sorted(
                    paths.items(), key=lambda x: x[0][1].name
                )
                for (start_node, table_node), cols in join_info:
                    join_on = []
                    start_node_alias = amenable_name(start_node.name)
                    namespace = ast.Name(start_node.name.split(".")[:-1])
                    tables[start_node].append(
                        ast.Table(
                            ast.Name(
                                start_node.name.split(".")[-1],
                                namespace=namespace or None,
                            ),
                            _dj_node=start_node,
                        )
                    )
                    table_node_alias = amenable_name(table_node.name)
                    tables[table_node].append(
                        ast.Table(
                            ast.Name(
                                table_node.name.split(".")[-1],
                                namespace=ast.Name(table_node.name.split(".")[-1]),
                            ),
                            _dj_node=table_node
                        )
                    )

                    for col in cols:
                        join_on.append(
                            ast.BinaryOp.Eq(
                                ast.Column(
                                    ast.Name(col.name),
                                    _table=tables[start_node][0],
                                ),
                                ast.Column(
                                    ast.Name(col.dimension_column or "id"),
                                    _table=tables[table_node][0],
                                ),
                            ),
                        )
                        join_table = cast(
                            Optional[ast.Table],
                            _get_node_table(table_node, build_criteria),
                        )  # got a materialization
                        if (
                                join_table is None
                        ):  # no materialization - recurse to build dimension first  # pragma: no cover
                            join_query = parse(cast(str, table_node.query))
                            join_table = build_ast(session, join_query)
                            join_table.parenthesized = True
                        join_ast = ast.Alias(  # type: ignore
                            ast.Name(table_node_alias),
                            child=join_table,
                            as_=True,
                        )
                    if join_on:  # table had stuff to join
                        select.from_.relations[-1].extensions.append(  # pragma: no cover
                            ast.Join(
                                'LEFT OUTER',
                                join_ast,  # type: ignore
                                ast.JoinCriteria(on=ast.BinaryOp.And(*join_on)),  # pylint: disable=E1120
                            ),
                        )


def _build_tables_on_select(
    session: Session,
    select: ast.SelectExpression,
    tables: Dict[NodeRevision, List[ast.Table]],
    build_criteria: Optional[BuildCriteria] = None,
):
    """
    Add all nodes not agg or filter dimensions to the select
    """
    for node, tbls in tables.items():

        node_table = cast(
            Optional[ast.Table],
            _get_node_table(node, build_criteria),
        )  # got a materialization
        if node_table is None:  # no materialization - recurse to node first
            node_query = parse(cast(str, node.query))
            node_table = build_ast(  # type: ignore
                session,
                node_query,
                build_criteria,
            ).select  # pylint: disable=W0212

        alias = amenable_name(node.node.name)
        node_ast = ast.Alias(ast.Name(alias), child=node_table)  # type: ignore
        for tbl in tbls:
            select.replace(tbl, node_ast)


def dimension_columns_mapping(select: ast.SelectExpression) -> Dict[NodeRevision, List[ast.Column]]:
    """
    Extract all dimension nodes referenced by columns
    """
    dimension_nodes_to_columns: Dict[NodeRevision, List[ast.Column]] = {}

    for col in select.find_all(ast.Column):
        if isinstance(col.table, ast.Table):
            if node := col.table.dj_node:  # pragma: no cover
                if node.type == NodeType.DIMENSION:
                    dimension_nodes_to_columns[node] = dimension_nodes_to_columns.get(node, [])
                    dimension_nodes_to_columns[node].append(col)
    return dimension_nodes_to_columns


# flake8: noqa: C901
def _build_select_ast(
    session: Session,
    select: ast.SelectExpression,
    build_criteria: Optional[BuildCriteria] = None,
):
    """
    Transforms a select ast by replacing dj node references with their asts
    # get all columns from filters + group bys
    # some of them can be sourced directly from tables on the select, others cannot
    # for the ones that cannot be sourced from the select, attempt to join them via dimension links
    """
    # Get all tables directly on the select that have an attached DJ node
    tables = _get_tables_from_select(select)
    dimension_columns = dimension_columns_mapping(select)
    join_tables_for_dimensions(session, dimension_columns, tables, build_criteria)
    _build_tables_on_select(session, select, tables, build_criteria)


def add_filters_and_dimensions_to_query_ast(
    query: ast.Query,
    dialect: Optional[str] = None,
    filters: Optional[List[str]] = None,
    dimensions: Optional[List[str]] = None,
):
    """
    Add filters and dimensions to a query ast
    """
    projection_addition = []
    if filters:
        filter_asts = (  # pylint: disable=consider-using-ternary
            query.select.where and [query.select.where] or []
        )

        for filter_ in filters:
            temp_select = parse(f"select * where {filter_}").select
            filter_asts.append(
                # use parse to get the asts from the strings we got
                temp_select.where,  # type:ignore
            )
        query.select.where = ast.BinaryOp.And(*filter_asts)

    if dimensions:
        for agg in dimensions:
            temp_select = parse(
                f"select * group by {agg}",
            ).select
            query.select.group_by += temp_select.group_by  # type:ignore
            projection_addition += list(temp_select.find_all(ast.Column))
    query.select.projection += [
        col.set_api_column(True).copy() for col in set(projection_addition)
    ]

    # Cannot select for columns that aren't in GROUP BY and aren't aggregations
    if query.select.group_by:
        query.select.projection = [
            col
            for col in query.select.projection
            if col.is_aggregation()
            or col.name.name in {gc.name.name for gc in query.select.group_by}  # type: ignore
        ]


def _get_node_table(
    node: NodeRevision,
    build_criteria: Optional[BuildCriteria] = None,
    as_select: bool = False,
) -> Optional[Union[ast.Select, ast.Table]]:
    """
    If a node is available, return a table for the available state
    """
    table = None
    if node.type == NodeType.SOURCE:
        namespace = None
        if node.table:
            name = ast.Name(
                node.table,
                namespace=ast.Name(node.schema_)
                if node.schema_ else None
            )
        else:
            name = ast.Name(node.name, '"')
        table = ast.Table(name, namespace, _dj_node=node)
    elif node.availability and node.availability.is_available(
        criteria=build_criteria,
    ):  # pragma: no cover
        namespace = (
            ast.Name(node.availability.schema_)
            if node.availability.schema_ else None
        )
        table = ast.Table(
            ast.Name(
                node.availability.table,
                namespace=namespace
            ),
            namespace,
            _dj_node=node,
        )
    if table and as_select:  # pragma: no cover
        return ast.Select(
            projection=[ast.Wildcard()],
            from_=ast.From(relations=[ast.Relation(table)])
        )
    return table


def build_node(  # pylint: disable=too-many-arguments
    session: Session,
    node: NodeRevision,
    dialect: Optional[str] = None,
    filters: Optional[List[str]] = None,
    dimensions: Optional[List[str]] = None,
    build_criteria: Optional[BuildCriteria] = None,
) -> ast.Query:
    """
    Determines the optimal way to build the Node and does so
    """
    # if no dimensions need to be added then we can see if the node is directly materialized
    if not (filters or dimensions):
        if select := cast(
            ast.Select,
            _get_node_table(node, build_criteria, as_select=True),
        ):
            return ast.Query(select=select)  # pragma: no cover

    if node.query:
        query = parse(node.query)
    else:
        query = build_source_node_query(node)

    add_filters_and_dimensions_to_query_ast(
        query,
        dialect,
        filters,
        dimensions,
    )

    return build_ast(session, query, build_criteria)


def build_source_node_query(node: NodeRevision):
    """
    Returns a query that selects each column explicitly in the source node.
    """
    name = ast.Name(node.name, '"')
    table = ast.Table(name, None, _dj_node=node)
    select = ast.Select(
        projection=[
            ast.Column(ast.Name(tbl_col.name), _table=table)
            for tbl_col in node.columns
        ],
        from_=ast.From(relations=[ast.Relation(table)]),
    )
    return ast.Query(select=select)


def build_ast(  # pylint: disable=too-many-arguments
    session: Session,
    query: ast.Query,
    build_criteria: Optional[BuildCriteria] = None,
) -> ast.Query:
    """
    Determines the optimal way to build the query AST and does so
    """
    context = CompileContext(session=session, exception=DJException(), query=query)
    query.compile(context)
    query.build(session, build_criteria)
    return query
