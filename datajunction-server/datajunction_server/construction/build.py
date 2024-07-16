# pylint: disable=too-many-arguments,too-many-locals,too-many-nested-blocks,too-many-branches,R0401,too-many-lines,protected-access,line-too-long
"""Functions for building DJ node queries"""
import collections
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import DefaultDict, Deque, Dict, List, Optional, Set, Tuple, Union, cast

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from this import d

from datajunction_server.construction.utils import to_namespaced_name
from datajunction_server.database import Engine
from datajunction_server.database.column import Column
from datajunction_server.database.node import Node, NodeRevision
from datajunction_server.database.user import User
from datajunction_server.errors import (
    DJError,
    DJException,
    DJInvalidInputException,
    ErrorCode,
)
from datajunction_server.internal.engines import get_engine
from datajunction_server.models import access
from datajunction_server.models.column import SemanticType
from datajunction_server.models.engine import Dialect
from datajunction_server.models.materialization import GenericCubeConfig
from datajunction_server.models.metric import TranslatedSQL
from datajunction_server.models.node import BuildCriteria
from datajunction_server.models.node_type import NodeType
from datajunction_server.naming import LOOKUP_CHARS, amenable_name, from_amenable_name
from datajunction_server.sql.dag import get_dimensions, get_shared_dimensions
from datajunction_server.sql.parsing.ast import CompileContext
from datajunction_server.sql.parsing.backends.antlr4 import ast, parse
from datajunction_server.sql.parsing.types import (
    ColumnType,
    DoubleType,
    LongType,
    TimestampType,
)
from datajunction_server.utils import SEPARATOR

_logger = logging.getLogger(__name__)


def _get_tables_from_select(
    select: ast.SelectExpression,
) -> DefaultDict[NodeRevision, List[ast.Table]]:
    """
    Extract all tables (source, transform, dimensions)
    directly on the select that have an attached DJ node
    """
    tables: DefaultDict[NodeRevision, List[ast.Table]] = collections.defaultdict(list)

    for table in select.find_all(ast.Table):
        if node := table.dj_node:  # pragma: no cover
            tables[node].append(table)
    return tables


async def _join_path(
    session: AsyncSession,
    dimension_node: NodeRevision,
    initial_nodes: Set[NodeRevision],
) -> Tuple[NodeRevision, Dict[Tuple[NodeRevision, NodeRevision], List[Column]]]:
    """
    For a dimension node, we want to find a possible join path between it
    and any of the nodes that are directly referenced in the original query. If
    no join path exists, returns an empty dict.
    """
    processed = set()

    to_process: Deque[
        Tuple[NodeRevision, Dict[Tuple[NodeRevision], List[Column]]]
    ] = collections.deque([])
    join_info: Dict[Tuple[NodeRevision], List[Column]] = {}
    to_process.extend([(node, join_info.copy()) for node in initial_nodes])
    possible_join_paths = []

    while to_process:
        current_node, path = to_process.popleft()
        processed.add(current_node)
        dimensions_to_columns = collections.defaultdict(list)

        # From the columns on the current node, find the next layer of
        # dimension nodes that can be joined in
        for col in current_node.columns:
            if col.dimension and col.dimension.type == NodeType.DIMENSION:
                dimensions_to_columns[col.dimension.current].append(col)

        # Go through all potential dimensions and their join columns
        for joinable_dim, join_cols in dimensions_to_columns.items():
            next_join_path = {**path, **{(current_node, joinable_dim): join_cols}}
            full_join_path = (joinable_dim, next_join_path)
            if joinable_dim == dimension_node:
                for col in join_cols:
                    dim_pk = dimension_node.primary_key()
                    if not col.dimension_column:  # pragma: no cover
                        if len(dim_pk) != 1:
                            raise DJException(  # pragma: no cover
                                f"Node {current_node.name} specifying dimension "
                                f"{joinable_dim.name} on column {col.name} does not"
                                f" specify a dimension column, and {dimension_node.name} "
                                f"has a compound primary key.",
                            )
                        col.dimension_column = dim_pk[0].name

                possible_join_paths.append(full_join_path)  # type: ignore
            if joinable_dim not in processed:  # pragma: no cover
                to_process.append(full_join_path)
                await session.refresh(joinable_dim, ["parents"])
                for parent in joinable_dim.parents:
                    to_process.append((parent.current, next_join_path))
    return min(possible_join_paths, key=len)  # type: ignore


async def _get_or_build_join_table(
    session: AsyncSession,
    table_node: NodeRevision,
    build_criteria: Optional[BuildCriteria],
):
    """
    Build the join table from a materialization if one is available, or recurse
    to build it from the dimension node's query if not
    """
    table_node_alias = amenable_name(table_node.name)
    await session.refresh(table_node, ["availability"])
    await session.refresh(table_node, ["columns"])
    join_table = cast(
        Optional[ast.TableExpression],
        _get_node_table(table_node, build_criteria),
    )
    if not join_table:  # pragma: no cover
        join_query = parse(cast(str, table_node.query))
        join_table = await build_ast(session, join_query)  # type: ignore
        join_table.parenthesized = True  # type: ignore

    join_table = cast(ast.TableExpression, join_table)  # type: ignore
    right_alias = ast.Name(table_node_alias)
    join_right = ast.Alias(  # type: ignore
        right_alias,
        child=join_table,
        as_=True,
    )
    await join_table.compile(CompileContext(session, DJException()))
    join_table.set_alias(right_alias)  # type: ignore
    return join_right


async def _build_joins_for_dimension_link(
    session: AsyncSession,
    initial_nodes: Set[NodeRevision],
    tables: DefaultDict[NodeRevision, List[ast.Table]],
    build_criteria: Optional[BuildCriteria],
    required_dimension_columns: List[ast.Column],
    join_path: List,
    filters: Optional[List[str]] = None,
) -> List[ast.Join]:
    """
    Returns the join ASTs needed to bring in the dimension node from
    the set of initial nodes.
    """
    join_asts = []
    for link in join_path:
        join_query = await build_ast(session, link.join_sql_ast())
        join = join_query.select.from_.relations[-1].extensions[0]  # type: ignore
        # Assemble table on left of join
        if not tables[link.node_revision]:
            continue  # pragma: no cover
        left_table = (
            tables[link.node_revision][0].child  # type: ignore
            if isinstance(tables[link.node_revision][0], ast.Alias)
            else tables[link.node_revision][0]
        )
        for dim_col in required_dimension_columns:
            await left_table.add_ref_column(dim_col)

        # Assemble table on right of join
        join_right = await _get_or_build_join_table(
            session,
            link.dimension.current,  # if isinstance(dim_node, NodeRevision) else dim_node.current,
            build_criteria,
        )
        initial_nodes.add(link.dimension.current)
        tables[link.dimension.current].append(join_right)  # type: ignore

        # Optimize query by filtering down to only the necessary columns
        selected_columns = {col.name.name for col in required_dimension_columns}
        join_columns = {
            join_col.name.name
            for join_col in join.criteria.on.find_all(ast.Column)  # type: ignore
        }
        joinable_dim_columns = {
            col.name.name
            for dim_link in link.dimension.current.dimension_links
            for col in dim_link.joins()[0].criteria.on.find_all(ast.Column)
        }
        columns_in_filter_clauses = set()
        if filters:
            filter_asts = [
                parse(f"select * where {filter_}").select.where  # type: ignore
                for filter_ in filters
            ]
            for clause in filter_asts:
                for col in clause.find_all(ast.Column):  # type: ignore
                    node_name = SEPARATOR.join(col.identifier().split(SEPARATOR)[:-1])
                    column_name = col.identifier().split(SEPARATOR)[-1]
                    if node_name == link.dimension.name:
                        columns_in_filter_clauses.add(column_name)

        necessary_columns = (
            selected_columns.union(join_columns)
            .union(
                joinable_dim_columns,
            )
            .union(columns_in_filter_clauses)
        )
        if isinstance(join_right.child, ast.Query):
            join_right.child.select.projection = [
                col
                for col in join_right.child.select.projection
                if col.alias_or_name.name in necessary_columns  # type: ignore
            ]

        # Replace the join right query
        join.right = join_right.child  # type: ignore
        for dim_col in required_dimension_columns:
            await join_right.child.add_ref_column(dim_col)
        join_asts.append(
            ast.Join(
                str(link.join_type).upper(),
                join_right,  # type: ignore
                join.criteria,
            ),
        )
    return join_asts


async def _build_joins_for_dimension(
    session: AsyncSession,
    dim_node: NodeRevision,
    initial_nodes: Set[NodeRevision],
    tables: DefaultDict[NodeRevision, List[ast.Table]],
    build_criteria: Optional[BuildCriteria],
    required_dimension_columns: List[ast.Column],
) -> List[ast.Join]:
    """
    Returns the join ASTs needed to bring in the dimension node from
    the set of initial nodes.
    """
    _, paths = await _join_path(session, dim_node, initial_nodes)
    asts = []
    for connecting_nodes, join_columns in paths.items():
        start_node, table_node = connecting_nodes  # type: ignore
        join_on = []

        # Assemble table on left of join
        left_table = (
            tables[start_node][0].child  # type: ignore
            if isinstance(tables[start_node][0], ast.Alias)
            else tables[start_node][0]
        )
        join_left_columns = {
            col.alias_or_name.name: col for col in left_table.columns  # type: ignore
        }

        # Assemble table on right of join
        join_right = await _get_or_build_join_table(
            session,
            table_node,
            build_criteria,
        )

        # Optimize query by filtering down to only the necessary columns
        selected_columns = {col.name.name for col in required_dimension_columns}
        available_join_columns = {
            col.dimension_column for col in join_columns if col.dimension_column
        }
        primary_key_columns = {col.name for col in table_node.primary_key()}
        joinable_dim_columns = {
            col.name for col in table_node.columns if col.dimension_id
        }
        required_mapping = (
            selected_columns.union(available_join_columns)
            .union(primary_key_columns)
            .union(joinable_dim_columns)
        )
        join_right.child.select.projection = [
            col
            for col in join_right.child.select.projection
            if col.alias_or_name.name in required_mapping
        ]

        initial_nodes.add(table_node)
        tables[table_node].append(join_right)  # type: ignore
        join_right_columns = {
            col.alias_or_name.name: col  # type: ignore
            for col in join_right.child.columns
        }

        # Assemble join ON clause
        for join_col in join_columns:
            join_table_pk = table_node.primary_key()
            if join_col.name in join_left_columns and (
                join_col.dimension_column in join_right_columns
                or join_table_pk[0].name in join_right_columns
            ):
                await left_table.add_ref_column(
                    cast(ast.Column, join_left_columns[join_col.name]),
                )
                join_on.append(
                    ast.BinaryOp.Eq(
                        join_left_columns[join_col.name],
                        join_right_columns[
                            join_col.dimension_column or join_table_pk[0].name
                        ],
                        use_alias_as_name=True,
                    ),
                )
            else:
                raise DJInvalidInputException(  # pragma: no cover
                    f"The specified join column {join_col.dimension_column} "
                    f"does not exist on {table_node.name}",
                )
            for dim_col in required_dimension_columns:
                await join_right.child.add_ref_column(dim_col)

        if join_on:  # pragma: no cover
            asts.append(
                ast.Join(
                    "LEFT",
                    join_right,  # type: ignore
                    ast.JoinCriteria(
                        on=ast.BinaryOp.And(*join_on),  # pylint: disable=E1120
                    ),
                ),
            )
    return asts


async def join_tables_for_dimensions(
    session: AsyncSession,
    dimension_nodes_to_columns: Dict[NodeRevision, List[ast.Column]],
    tables: DefaultDict[NodeRevision, List[ast.Table]],
    build_criteria: Optional[BuildCriteria] = None,
    filters: Optional[List[str]] = None,
):
    """
    Joins the tables necessary for a set of filter and group by dimensions
    onto the select expression.

    In some cases, the necessary tables will already be on the select and
    no additional joins will be needed. However, if the tables are not in
    the select, it will traverse through available linked tables (via dimension
    nodes) and join them in.
    """
    initial_nodes = set(tables)
    for dim_node, required_dimension_columns in sorted(
        dimension_nodes_to_columns.items(),
        key=lambda x: x[0].name,
    ):
        # Find all the selects that contain the different dimension columns
        selects_map = {
            cast(ast.Select, dim_col.get_nearest_parent_of_type(ast.Select))
            for dim_col in required_dimension_columns
        }

        # If there was a link configured to this dimension node, find the saved join path
        # If the dimension has a role (encoded in its subscript), remove the role
        join_path = None
        for col in required_dimension_columns:
            if isinstance(col.parent, ast.Subscript):
                col.parent.swap(col)  # pragma: no cover
            if col.table.identifier() == dim_node.name and col.table.path:  # type: ignore
                join_path = col.table.path  # type: ignore

        # Join the source tables (if necessary) for these dimension columns
        # onto each select clause
        for select in selects_map:
            join_asts = []
            if dim_node not in initial_nodes:  # need to join dimension
                if join_path:
                    join_asts = await _build_joins_for_dimension_link(
                        session,
                        initial_nodes,
                        tables,
                        build_criteria,
                        required_dimension_columns,
                        join_path,
                        filters,
                    )
                else:
                    join_asts = await _build_joins_for_dimension(
                        session,
                        dim_node,
                        initial_nodes,
                        tables,
                        build_criteria,
                        required_dimension_columns,
                    )
                if join_asts and select.from_:  # pragma: no cover
                    existing_joins = set(select.from_.relations[-1].extensions)
                    select.from_.relations[-1].extensions.extend(  # pragma: no cover
                        [x for x in join_asts if x not in existing_joins],
                    )


def build_filters(
    node: NodeRevision,
    node_table: Optional[ast.TableExpression],
    filters: Optional[List[str]],
) -> List[ast.Expression]:
    """
    Returns a list of built filter expressions based on the provided node
    and its table expression.
    """
    filter_asts: List[ast.Expression] = []
    if not filters:
        return filter_asts
    dimensions_to_columns_map = node.dimensions_to_columns_map()
    for filter_ in filters:
        temp_select = parse(f"select * where {filter_}").select
        referenced_dimensions = list(temp_select.find_all(ast.Column))
        # We can only push down the filter if all dimensions referenced by the filter
        # are available as foreign key columns on the node
        all_referenced_dimensions_available_on_node = all(
            dim.identifier() in dimensions_to_columns_map
            for dim in referenced_dimensions
        )
        if all_referenced_dimensions_available_on_node:
            # Renames the columns from dimension attributes to columns that match those
            # dimension attributes on the node
            for dim in referenced_dimensions:
                col_name = dimensions_to_columns_map[dim.identifier()].alias_or_name
                dim.name = ast.Name(name=col_name)
                if node_table:  # pragma: no cover
                    dim.add_table(node_table)
            filter_asts.append(
                temp_select.where,  # type: ignore
            )
    return filter_asts


async def _build_tables_on_select(
    session: AsyncSession,
    select: ast.SelectExpression,
    tables: Dict[NodeRevision, List[ast.Table]],
    memoized_queries: Dict[int, ast.Query],
    build_criteria: Optional[BuildCriteria] = None,
    filters: Optional[List[str]] = None,
    dimensions: Optional[List[str]] = None,
    access_control=None,
):
    """
    Add all nodes not agg or filter dimensions to the select
    """
    context = CompileContext(session=session, exception=DJException())

    # `tables` is a mapping between DJ nodes and table expressions on the query AST
    # If there is more than one table expression on the AST, this means the same DJ
    # node is referenced multiple times in the query, likely through being joined
    # more than once.
    for node, tbls in tables.items():
        await session.refresh(node, ["dimension_links"])

        # Try to find a physical table attached to this node, if one exists.
        physical_table = cast(
            Optional[ast.Table],
            _get_node_table(node, build_criteria),
        )

        for tbl in tbls:
            # If no attached physical table was found, recursively build the node
            if physical_table is None:
                node_query = parse(cast(str, node.query))
                if hash(node_query) in memoized_queries:  # pragma: no cover
                    query_ast = memoized_queries[hash(node_query)]  # type: ignore
                else:
                    query_ast = await build_ast(  # type: ignore
                        session,
                        node_query,
                        memoized_queries,
                        build_criteria,
                        filters,
                        dimensions,
                        access_control,
                    )
                    memoized_queries[hash(node_query)] = query_ast

                alias = amenable_name(node.name)
                node_ast = ast.Alias(ast.Name(alias), child=query_ast, as_=True)  # type: ignore
                query_ast.parenthesized = True  # type: ignore

                filter_asts = build_filters(node, node_ast, filters)  # type: ignore
            else:
                alias = amenable_name(node.name)
                node_ast = ast.Alias(ast.Name(alias), child=physical_table, as_=True)  # type: ignore
                filter_asts = build_filters(node, physical_table, filters)

            # Remove columns that are not part of the table expression reference
            if (
                isinstance(node_ast, ast.Alias)
                and isinstance(node_ast.child, ast.Query)  # type: ignore
                and isinstance(tbl, ast.Alias)
            ):
                node_ast.child.select.projection = [
                    col
                    for col in node_ast.child.select.projection
                    if col in set(tbl.child.select.projection)
                ]

            # Rename the built table expression to use the user-provided alias (derived
            # from the node query)
            if isinstance(tbl, ast.Aliasable) and hasattr(tbl, "alias") and tbl.alias:
                if isinstance(node_ast, ast.Alias) and isinstance(
                    node_ast.child,
                    ast.Table,
                ):
                    node_ast.name = tbl.alias
                    node_ast.alias.name = tbl.alias
                elif len(tbls) > 1:
                    node_ast.alias.name = tbl.alias  # type: ignore

            await node_ast.compile(context)

            # This will push down the filter expression to CTEs, subqueries, and top-level queries.
            # Each `tbl` is a TableExpression that references the node in question, where the
            # provided filters might apply. We find the table reference's nearest parent SELECT
            # clause and populate its WHERE clause with the filters.
            if filter_asts:
                if nearest_select := tbl.get_nearest_parent_of_type(
                    ast.Select,
                ):  # pragma: no cover
                    if nearest_select.where:
                        filter_asts.append(nearest_select.where)
                    nearest_select.where = (
                        ast.BinaryOp.And(  # pylint: disable=no-value-for-parameter
                            *filter_asts
                        )
                    )

            select.replace(
                tbl,
                node_ast,
                copy=False,
            )

            await select.compile(context)
            for col in select.find_all(ast.Column):
                if (
                    col._table
                    and isinstance(col._table, ast.Table)
                    and col._table.dj_node
                    and col._table.dj_node.name == node.name
                ):
                    col._table = node_ast


def dimension_columns_mapping(
    select: ast.SelectExpression,
) -> Dict[NodeRevision, List[ast.Column]]:
    """
    Extract all dimension nodes referenced by columns
    """
    dimension_nodes_to_columns: DefaultDict[
        NodeRevision,
        List[ast.Column],
    ] = collections.defaultdict(list)

    for col in select.find_all(ast.Column):
        if isinstance(col.table, ast.Table):
            if node := col.table.dj_node:  # pragma: no cover
                if node.type == NodeType.DIMENSION:
                    dimension_nodes_to_columns[node].append(col)
    return dimension_nodes_to_columns


# flake8: noqa: C901
async def _build_select_ast(
    session: AsyncSession,
    select: ast.SelectExpression,
    memoized_queries: Dict[int, ast.Query],
    build_criteria: Optional[BuildCriteria] = None,
    filters: Optional[List[str]] = None,
    dimensions: Optional[List[str]] = None,
    access_control=None,
):
    """
    Transforms a select ast by replacing dj node references with their asts
    Starts by extracting all dimensions-backed columns from filters + group bys.
    Some of them can be sourced directly from tables on the select, others cannot
    For the ones that cannot be sourced directly, attempt to join them via dimension links.
    """
    tables = _get_tables_from_select(select)
    # dimension_columns = dimension_columns_mapping(select)
    # print("tables", len(tables))
    # only one: users.anhl.playback, 1
    # await join_tables_for_dimensions(
    #     session,
    #     dimension_columns,
    #     tables,
    #     build_criteria,
    #     filters,
    # )
    await _build_tables_on_select(
        session,
        select,
        tables,
        memoized_queries,
        build_criteria,
        filters,
        dimensions,
        access_control,
    )


def rename_dimension_primary_keys_to_foreign_keys(
    dimension_node: Node,
    current_node: NodeRevision,
    col: ast.Column,
) -> ast.Column:
    """
    Optimize the query build by renaming any requested dimension node primary key columns to
    foreign keys on the current processing node. This results in one less join if the user is
    only requesting the dimension node's primary key, since that column is already present via
    current node's foreign key column.
    """
    if (
        dimension_node.name != current_node.name
        and dimension_node.type == NodeType.DIMENSION
    ):
        dimension_pk = ",".join(
            [pk_col.name for pk_col in dimension_node.current.primary_key()],
        )
        if dimension_pk == col.alias_or_name.name:
            foreign_key = [
                col
                for col in current_node.columns
                if col.dimension and col.dimension.name == dimension_node.name
            ]
            if foreign_key:
                col.name = ast.Name(  # pragma: no cover
                    foreign_key[0].name,
                    namespace=to_namespaced_name(current_node.name),
                )
            col.set_alias(
                ast.Name(amenable_name(dimension_node.name + SEPARATOR + dimension_pk)),
            )
    links = [
        link
        for link in current_node.dimension_links
        if link.dimension.name == dimension_node.name
    ]
    if links:
        link = links[0]
        foreign_keys_map = {
            k.identifier(): v.identifier()
            for k, v in link.foreign_key_mapping().items()
        }
        original_col_identifier = col.identifier()
        if col.identifier() in foreign_keys_map:
            foreign_key = foreign_keys_map[col.identifier()]
            col.name = ast.Name(
                foreign_key.split(SEPARATOR)[-1],  # type: ignore
                namespace=to_namespaced_name(current_node.name),
            )
            col.set_alias(
                ast.Name(amenable_name(original_col_identifier)),
            )
    return col


# pylint: disable=R0915
async def add_filters_dimensions_orderby_limit_to_query_ast(
    session: AsyncSession,
    node: NodeRevision,
    query: ast.Query,
    dialect: Optional[str] = None,  # pylint: disable=unused-argument
    filters: Optional[List[str]] = None,
    dimensions: Optional[List[str]] = None,
    orderby: Optional[List[str]] = None,
    limit: Optional[int] = None,
    include_dimensions_in_groupby: bool = True,
    access_control: Optional[access.AccessControlStore] = None,
):
    """
    Add filters and dimensions to a query ast
    """
    projection_addition = {}
    node_ = await Node.get_by_name(
        session,
        node.name,
        options=[
            selectinload(Node.current).options(
                selectinload(NodeRevision.parents).options(selectinload(Node.current)),
            ),
        ],
    )
    all_dimensions = [dim.name for dim in await get_dimensions(session, node_)]  # type: ignore

    if dimensions:
        for agg in dimensions:
            temp_select = parse(
                f"select * group by {agg}",
            ).select

            for dim in temp_select.group_by:
                # if the dimension has a role (encoded in its subscript), attach the
                # dimension role to the column metadata
                if isinstance(dim, ast.Subscript):
                    if isinstance(dim.index, ast.Lambda):
                        dim.expr.role = str(dim.index)
                    else:
                        dim.expr.role = dim.index.identifier()  # type: ignore
                    dim.swap(dim.expr)

            if include_dimensions_in_groupby:
                query.select.group_by += temp_select.group_by  # type:ignore
            for col in temp_select.find_all(ast.Column):
                projection_addition[col.identifier(False)] = col

                if access_control:
                    dj_node = await access_control.add_request_by_node_name(
                        session,
                        col,
                    )
                    if dj_node:
                        await session.refresh(node, ["columns"])
                        await session.refresh(node, ["dimension_links"])
                        for link in node.dimension_links:
                            await session.refresh(link, ["dimension"])
                        rename_dimension_primary_keys_to_foreign_keys(
                            dj_node,
                            node,
                            col,
                        )

    if filters:
        filter_asts = (  # pylint: disable=consider-using-ternary
            query.select.where and [query.select.where] or []
        )

        for filter_ in filters:
            # use parse to get the asts from the strings we got
            temp_select = parse(f"select * where {filter_}").select
            for col in temp_select.find_all(ast.Column):
                # if the dimension has a role (encoded in its subscript), attach the
                # dimension role to the column metadata
                if isinstance(col.parent, ast.Subscript):
                    if isinstance(col.parent.index, ast.Lambda):
                        col.role = str(col.parent.index)
                    else:
                        col.role = col.parent.index.identifier()  # type: ignore
                    col.parent.swap(col)
                if not dimensions and col.identifier(False) in all_dimensions:
                    projection_addition[col.identifier(False)] = col
                if access_control:
                    dj_node = await access_control.add_request_by_node_name(
                        session,
                        col,
                    )
                    if dj_node:  # pragma: no cover
                        await session.refresh(node, ["dimension_links"])
                        rename_dimension_primary_keys_to_foreign_keys(
                            dj_node,
                            node,
                            col,
                        )
            filter_asts.append(
                temp_select.where,  # type:ignore
            )
        query.select.where = ast.BinaryOp.And(*filter_asts)

    if not query.select.organization:
        query.select.organization = ast.Organization([])

    if orderby:
        for order in orderby:
            temp_query = parse(
                f"select * order by {order}",
            )
            query.select.organization.order += (  # type:ignore
                temp_query.select.organization.order  # type:ignore
            )
            for col in temp_query.find_all(ast.Column):
                if access_control:  # pragma: no cover
                    await access_control.add_request_by_node_name(
                        session,
                        col,
                    )

    # add all used dimension columns to the projection without duplicates
    projection_update = []
    for exp in query.select.projection:
        if not isinstance(exp, ast.Column):
            projection_update.append(exp)
        else:
            ident = exp.identifier(False)
            added = None
            for exist_idents, exist_cols in projection_addition.items():
                if exist_idents.endswith(ident) or ident.endswith(exist_idents):
                    projection_update.append(exist_cols)
                    added = exist_idents
                    break

            if added is None:
                projection_update.append(exp)
            else:
                del projection_addition[added]

    projection_update += list(projection_addition.values())

    query.select.projection = projection_update

    if limit is not None:
        query.select.limit = ast.Number(limit)


def _get_node_table(
    node: NodeRevision,
    build_criteria: Optional[BuildCriteria] = None,
    as_select: bool = False,
) -> Optional[Union[ast.Select, ast.Table]]:
    """
    If a node has a materialization available, return the materialized table
    """
    table = None
    can_use_materialization = (
        build_criteria and node.name != build_criteria.target_node_name
    )
    if node.type == NodeType.SOURCE:
        if node.table:
            name = ast.Name(
                node.table,
                namespace=ast.Name(
                    node.schema_,
                    namespace=ast.Name(node.catalog.name)
                    if node.schema_ == "iceberg"
                    else None,
                )
                if node.schema_
                else None,
            )
        else:
            name = to_namespaced_name(node.name)
        table = ast.Table(
            name,
            _columns=[
                ast.Column(name=ast.Name(col.name), _type=col.type)
                for col in node.columns
            ],
            _dj_node=node,
        )
    elif (
        can_use_materialization
        and node.availability
        and node.availability.is_available(
            criteria=build_criteria,
        )
    ):  # pragma: no cover
        table = ast.Table(
            ast.Name(
                node.availability.table,
                namespace=(
                    ast.Name(node.availability.schema_)
                    if node.availability.schema_
                    else None
                ),
            ),
            _columns=[
                ast.Column(name=ast.Name(col.name), _type=col.type)
                for col in node.columns
            ],
            _dj_node=node,
        )
    if table and as_select:  # pragma: no cover
        return ast.Select(
            projection=table.columns,  # type: ignore
            from_=ast.From(relations=[ast.Relation(table)]),
        )
    return table


def get_default_criteria(
    node: NodeRevision,
    engine: Optional[Engine] = None,
) -> BuildCriteria:
    """
    Get the default build criteria for a node.
    """
    # Set the dialect by using the provided engine, if any. If no engine is specified,
    # set the dialect by finding available engines for this node, or default to Spark
    dialect = (
        engine.dialect
        if engine
        else (
            node.catalog.engines[0].dialect
            if node.catalog and node.catalog.engines and node.catalog.engines[0].dialect
            else Dialect.SPARK
        )
    )
    return BuildCriteria(
        dialect=dialect,
        target_node_name=node.name,
    )


def add_filters(query_ast: ast.Query, filters: List[str]):
    pass


def add_dimensions(query_ast: ast.Query, dimensions: List[str], groupby: bool = False):
    pass


async def dimension_join_path(
    session: AsyncSession,
    node: NodeRevision,
    dimension: str,
) -> Optional[List["DimensionLink"]]:
    """
    Find the join path between this node and the dimension attribute.
    If none exists, return None

    For each dimension, it can be:
    (a) a local dimension on this node
    (b) in one of the linked dimensions
    (c) require further traversal up the linked dims graph
    """
    # Check if it is a local dimension
    if dimension.startswith(node.name):
        for col in node.columns:
            if f"{node.name}.{col.name}" == dimension and col.is_dimensional():
                return []

    dimension_node_name = SEPARATOR.join(dimension.split(SEPARATOR)[:-1])

    # If it's not a local dimension, traverse the node's dimensions graph
    # This queue tracks the dimension link being processed and the path to that link
    await session.refresh(node, ["dimension_links"])
    processing_queue = collections.deque(
        [
            (link, [link]) for link in node.dimension_links
        ],  # Start with first layer of linked dims
    )
    while processing_queue:
        current_link, join_path = processing_queue.pop()
        await session.refresh(current_link, ["dimension"])
        if current_link.dimension.name == dimension_node_name:
            return join_path
        await session.refresh(current_link.dimension, ["current"])
        await session.refresh(current_link.dimension.current, ["dimension_links"])
        processing_queue.extend(
            [
                (link, join_path + [link])
                for link in current_link.dimension.current.dimension_links
            ],
        )
    return None


class RequestType(str, Enum):
    DIMENSION = "dimension"
    FILTER = "filter"


@dataclass
class DimensionJoin:
    # request_type: RequestType
    # dimension: str
    # filter_ast: str

    join_path: List[ast.Join]
    requested_dimensions: List[str]
    node_query: Optional[ast.Query] = None


async def compile_node_ast(session, node) -> ast.Query:
    """
    Parses the node's query into an AST and compiles it.
    """
    node_ast = parse(node.query)
    ctx = CompileContext(session, DJException())
    await node_ast.compile(ctx)
    return node_ast


async def build_and_alias_ast(
    session, node, node_ast, build_criteria, filters, dimensions,
):
    node_ast = await build_ast(
        session,
        node_ast,
        build_criteria=build_criteria,
        filters=filters,
        dimensions=dimensions,
    )
    node_alias = ast.Name(amenable_name(node.name))
    node_ast.set_alias(node_alias)
    node_ast.parenthesized = True
    node_ast.as_ = True
    return node_ast


def update_filter_column_with_foreign_key(
    dimension_attr: str,  # this is the canonical dimension
    filter_dim: ast.Column,  # this is the re
    node_ast: ast.Query,
    link: "DimensionLink",
):
    """
    Modifies the referenced dimension in the filter expression with the foreign key column from the node.
    """
    foreign_key = link.foreign_keys_reversed[dimension_attr]
    for node_col in node_ast.select.projection:
        if node_col.alias_or_name.identifier() == foreign_key.split(SEPARATOR)[-1]:
            filter_dim.parent.replace(
                filter_dim,
                node_col.child if isinstance(node_col, ast.Alias) else node_col,
            )
            filter_dim.dimension_ref = dimension_attr


async def needs_dimension_join(
    session: AsyncSession,
    dimension_attribute: str,
    join_path: List["DimensionLink"],
) -> bool:
    if len(join_path) == 1:
        link = join_path[0]
        await session.refresh(link.dimension, ["current"])
        await session.refresh(link.dimension.current, ["columns"])
        if dimension_attribute in link.foreign_keys_reversed:
            return False
    return True


# def apply_filters_to_ast():
#     if not any(join_required[req] for req in join_required.values()):
#         if not node_ast.select.where:
#             node_ast.select.where = filter_ast.where
#         else:
#             node_ast.select.where = ast.BinaryOp(
#                 ast.BinaryOpKind.And,
#                 node_ast.select.where,
#                 filter_ast.where,
#             )


def combine_filter_conditions(existing_condition, new_condition):
    """Combines the existing where clause with"""
    if not existing_condition:
        return new_condition
    return ast.BinaryOp.And(existing_condition, new_condition)


def build_join_for_link(
    link: "DimensionLink",
    join_left: ast.Query,
    join_right: ast.Query,
):
    join_ast = link.joins()[0]
    join_ast.right = join_right.alias
    dimension_node_columns = join_right.select.column_mapping
    node_columns = join_left.select.column_mapping
    for col in join_ast.criteria.find_all(ast.Column):
        chunks = col.identifier().split(SEPARATOR)
        column_node = SEPARATOR.join(chunks[:-1])
        column_name = chunks[-1]
        if column_node == link.dimension.name:
            col.parent.replace(
                col,
                ast.Column(
                    name=ast.Name(column_name),
                    _table=join_right,
                    _type=dimension_node_columns.get(column_name),
                ),
            )
        elif column_node == link.node_revision.name:
            col.parent.replace(
                col,
                ast.Column(
                    name=ast.Name(column_name),
                    _table=join_left,
                    _type=node_columns.get(column_name),
                ),
            )
    return join_ast


async def build_node(  # pylint: disable=too-many-arguments
    session: AsyncSession,
    node: NodeRevision,
    filters: Optional[List[str]] = None,
    dimensions: Optional[List[str]] = None,
    orderby: Optional[List[str]] = None,
    limit: Optional[int] = None,
    build_criteria: Optional[BuildCriteria] = None,
    access_control: Optional[access.AccessControlStore] = None,
    include_dimensions_in_groupby: bool = None,
) -> ast.Query:
    """
    Determines the optimal way to build the Node and does so
    """
    print("1. Building node", node.name, "with dimensions", dimensions)
    if access_control:
        access_control.add_request_by_node(node)

    if include_dimensions_in_groupby is None:
        include_dimensions_in_groupby = node.type == NodeType.METRIC

    # Set the dialect by finding available engines for this node, or default to Spark
    build_criteria = build_criteria or get_default_criteria(node)

    node_ast = await compile_node_ast(session, node)

    # Keep track of all the dimension nodes that need to be joined based on the
    # requested dimensions and filters
    dimension_node_joins = {}

    await session.refresh(node, ["required_dimensions"])
    requested_dimensions = list(
        set(dimensions or [] + [required.name for required in node.required_dimensions]),
    )
    for dim in requested_dimensions:
        dimension_node = SEPARATOR.join(dim.split(SEPARATOR)[:-1])
        if dimension_node not in dimension_node_joins:
            join_path = await dimension_join_path(session, node, dim)
            if await needs_dimension_join(session, dim, join_path):
                dimension_node_joins[dimension_node] = DimensionJoin(
                    join_path=join_path, requested_dimensions=[dim],
                )
        else:
            if dim not in dimension_node_joins[dimension_node].requested_dimensions:
                dimension_node_joins[dimension_node].requested_dimensions.append(dim)

    filter_asts = [
        parse(f"select * where {filter_}").select for filter_ in filters or []
    ]
    remaining_filters = []
    for filter_ast in filter_asts:
        # This mapping keeps track of each referenced dimension in the filter expression
        # and whether it needs a join to pull it in or if it can reuse a column on the node
        join_required = {}

        for filter_dim in filter_ast.where.find_all(ast.Column):
            dimension_attr = filter_dim.identifier()
            join_required[dimension_attr] = True
            dimension_node = SEPARATOR.join(dimension_attr.split(SEPARATOR)[:-1])
            if dimension_node not in dimension_node_joins:
                join_path = await dimension_join_path(session, node, dimension_attr)
                join_required[dimension_attr] = await needs_dimension_join(
                    session, dimension_attr, join_path,
                )
                # Only add it to the dimension node joins tracker if a join is needed
                if join_required[dimension_attr]:
                    dimension_node_joins[dimension_node] = DimensionJoin(
                        join_path=await dimension_join_path(
                            session, node, dimension_attr,
                        ),
                        requested_dimensions=[dimension_attr],
                    )
            else:
                if (
                    dimension_attr
                    not in dimension_node_joins[dimension_node].requested_dimensions
                ):
                    dimension_node_joins[dimension_node].requested_dimensions.append(
                        dimension_attr,
                    )

            # If it matches FK then update
            if not join_required[dimension_attr] and join_path:
                update_filter_column_with_foreign_key(
                    dimension_attr, filter_dim, node_ast, join_path[0],
                )

        # If the entire filter expression does not need any joins, add it to the node
        if not any(join_required[req] for req in join_required):
            node_ast.select.where = combine_filter_conditions(
                node_ast.select.where, filter_ast.where,
            )
        else:
            remaining_filters.append(filter_ast)

    print("ddddd", dimension_node_joins)
    print("2. node_ast after adding filters and dims that dont need join", node_ast)

    # Collect all necessary CTEs: the node query and the dimension node joins
    node_alias = ast.Name(amenable_name(node.name))
    node_ast = await build_ast(
        session,
        node_ast,
        build_criteria=build_criteria,
        filters=filters,
        dimensions=dimensions,
    )
    node_ast = node_ast.to_cte(node_alias)

    print("3. node_ast after calling build_ast", node_ast)

    final_ast = ast.Query(
        select=ast.Select(
            projection=[
                ast.Column(
                    ast.Name(col.alias_or_name.name),
                    _table=node_ast,
                    _type=col.type,
                )
                for col in node_ast.select.projection
            ],
            from_=ast.From(relations=[ast.Relation(node_alias)]),
        ),
        ctes=[node_ast],
    )
    print("4. final_ast with node CTE", final_ast)

    # Start building the joins using the CTEs
    cte_mapping = {}  # dimension node to CTE
    for dimension_node, dimension_join in dimension_node_joins.items():
        join_path = dimension_join.join_path
        requested_dimensions = dimension_join.requested_dimensions

        join_left = node_ast
        for link in join_path:
            for dim in requested_dimensions:
                if dim in link.foreign_keys_reversed:
                    # Mark the dimension ref on the column if it's a FK without a join
                    foreign_key_column_name = link.foreign_keys_reversed[dim].split(
                        SEPARATOR,
                    )[-1]
                    for col in join_left.select.projection:
                        if (
                            col.alias_or_name.identifier()
                            == foreign_key_column_name
                        ):
                            col.dimension_ref = dim
                            final_ast.select.projection.append(
                                ast.Column(
                                    name=ast.Name(col.alias_or_name.name),
                                    alias=ast.Name(amenable_name(dim)),
                                    _table=join_left,
                                    _type=col.type,
                                ),
                            )
            if all(dim in link.foreign_keys_reversed for dim in requested_dimensions):
                continue

            if link.dimension.name in cte_mapping:
                dimension_join.node_query = cte_mapping[link.dimension.name]
                continue

            await session.refresh(link.dimension, ["current"])
            await session.refresh(link.dimension.current, ["columns"])
            # 1. Build the dimension node
            dimension_node_query = await build_ast(
                session,
                parse(link.dimension.current.query),
                build_criteria=build_criteria,
                # filters=filters,
            )
            dimension_node_query = dimension_node_query.to_cte(
                ast.Name(amenable_name(link.dimension.name)),
                final_ast,
            )
            dimension_join.node_query = dimension_node_query
            cte_mapping[link.dimension.name] = dimension_node_query

            # 2. Add it to the list of CTEs
            final_ast.ctes.append(dimension_node_query)
            # 3. Build the join statement
            join_ast = build_join_for_link(
                link,
                join_left,
                dimension_node_query,
            )
            final_ast.select.from_.relations[-1].extensions.append(join_ast)
            print("5b. Building CTE for dimension join", final_ast)
            join_left = dimension_node_query

        # Add the requested dimensions to the final SELECT
        for dim in set(requested_dimensions):
            dim_node = SEPARATOR.join(dim.split(SEPARATOR)[:-1])
            print("processing", dim, "dim_node", dim_node)
            if dim_node in dimension_node_joins:
                for col in dimension_node_joins[dim_node].node_query.select.projection:
                    print("node_query", dimension_node_joins[dim_node].node_query)
                    # print(dim, "col.alias_or_name.name", col.dimension_ref if hasattr(col, "dimension_ref") else None)
                    if col.alias_or_name.name == dim.split(SEPARATOR)[-1] or (
                        hasattr(col, "dimension_ref") and col.dimension_ref == dim
                    ):
                        final_ast.select.projection.append(
                            ast.Column(
                                name=ast.Name(col.alias_or_name.name),
                                alias=ast.Name(amenable_name(dim)),
                                _table=dimension_node_joins[dim_node].node_query,
                                _type=col.type,
                            ),
                        )

    for filter_ast in remaining_filters:
        for filter_dim in filter_ast.where.find_all(ast.Column):
            dim_node = SEPARATOR.join(filter_dim.identifier().split(SEPARATOR)[:-1])
            for col in dimension_node_joins[dim_node].node_query.select.projection:
                if (
                    col.alias_or_name.name
                    == filter_dim.identifier().split(SEPARATOR)[-1]
                ):
                    filter_dim.parent.replace(
                        filter_dim,
                        ast.Column(
                            name=ast.Name(col.alias_or_name.name),
                            alias=ast.Name(amenable_name(filter_dim.identifier())),
                            _table=dimension_node_joins[dim_node].node_query,
                            _type=col.type,
                        ),
                    )
        final_ast.select.where = combine_filter_conditions(
            final_ast.select.where, filter_ast.where,
        )
    print("5. final_ast after joins", final_ast)
    return final_ast


def rename_columns(built_ast: ast.Query, node: NodeRevision):
    """
    Rename columns in the built ast to fully qualified column names.
    """
    projection = []
    node_columns = {col.name for col in node.columns}
    for expression in built_ast.select.projection:
        if (
            not isinstance(expression, ast.Alias)
            and not isinstance(
                expression,
                ast.Wildcard,
            )
            and not (hasattr(expression, "alias") and expression.alias)  # type: ignore
        ):
            alias_name = expression.alias_or_name.identifier(False)  # type: ignore
            if expression.alias_or_name.name in node_columns:  # type: ignore
                alias_name = node.name + SEPARATOR + expression.alias_or_name.name  # type: ignore
            expression = expression.copy()
            expression.set_semantic_entity(alias_name)  # type: ignore
            expression.set_alias(ast.Name(amenable_name(alias_name)))
            projection.append(expression)
        else:
            expression = expression.copy()
            if isinstance(
                expression,
                ast.Aliasable,
            ) and not isinstance(  # pragma: no cover
                expression,
                ast.Wildcard,
            ):
                column_ref = expression.alias_or_name.identifier()
                if column_ref in node_columns:  # type: ignore
                    alias_name = node.name + SEPARATOR + column_ref  # type: ignore
                    expression.set_semantic_entity(alias_name)
                else:
                    expression.set_semantic_entity(from_amenable_name(column_ref))
            projection.append(expression)  # type: ignore
    built_ast.select.projection = projection

    if built_ast.select.where:
        for col in built_ast.select.where.find_all(ast.Column):
            if hasattr(col, "alias"):  # pragma: no cover
                col.alias = None

    if built_ast.select.group_by:
        for i in range(  # pylint: disable=consider-using-enumerate
            len(built_ast.select.group_by),
        ):
            if hasattr(built_ast.select.group_by[i], "alias"):  # pragma: no cover
                built_ast.select.group_by[i] = ast.Column(
                    name=built_ast.select.group_by[i].name,  # type: ignore
                    # pylint:disable=protected-access
                    _type=built_ast.select.group_by[i].type,  # type: ignore
                    # pylint:disable=protected-access
                    _table=built_ast.select.group_by[i]._table,  # type: ignore
                )
                built_ast.select.group_by[i].alias = None
    return built_ast


def group_metrics_by_parent(
    metric_nodes: List[Node],
) -> DefaultDict[Node, List[NodeRevision]]:
    """
    Group metrics by their parent node
    """
    common_parents = collections.defaultdict(list)
    for metric_node in metric_nodes:
        immediate_parent = metric_node.current.parents[0]
        common_parents[immediate_parent].append(metric_node.current)
    return common_parents


async def validate_shared_dimensions(
    session: AsyncSession,
    metric_nodes: List[Node],
    dimensions: List[str],
    filters: List[str],
):
    """
    Determine if dimensions are shared.
    """
    shared_dimensions = [
        dim.name for dim in await get_shared_dimensions(session, metric_nodes)
    ]
    for dimension_attribute in dimensions:
        if dimension_attribute not in shared_dimensions:
            message = (
                f"The dimension attribute `{dimension_attribute}` is not "
                "available on every metric and thus cannot be included."
            )
            raise DJInvalidInputException(
                message,
                errors=[DJError(code=ErrorCode.INVALID_DIMENSION, message=message)],
            )

    for filter_ in filters:
        temp_select = parse(f"select * where {filter_}").select
        columns_in_filter = temp_select.where.find_all(ast.Column)  # type: ignore
        dims_without_prefix = {dim.split(".")[-1]: dim for dim in shared_dimensions}
        for col in columns_in_filter:
            if (
                str(col) not in shared_dimensions
                and str(col.parent) not in shared_dimensions
            ):
                potential_dimension_match = (
                    f" Did you mean `{dims_without_prefix[str(col)]}`?"
                    if str(col) in dims_without_prefix
                    else ""
                )
                message = (
                    f"The filter `{filter_}` references the dimension attribute "
                    f"`{col}`, which is not available on every"
                    f" metric and thus cannot be included.{potential_dimension_match}"
                )
                raise DJInvalidInputException(
                    message,
                    errors=[DJError(code=ErrorCode.INVALID_DIMENSION, message=message)],
                )


async def build_metric_nodes(
    session: AsyncSession,
    metric_nodes: List[Node],
    filters: List[str],
    dimensions: List[str],
    orderby: List[str],
    limit: Optional[int] = None,
    build_criteria: Optional[BuildCriteria] = None,
    access_control: Optional[access.AccessControlStore] = None,
):
    """
    Build a single query for all metrics in the list, including the specified
    group bys (dimensions) and filters. The metric nodes should share the same set
    of dimensional attributes. Then we can:
    (a) Group metrics by their parent nodes.
    (b) Build a query for each parent node with the dimensional attributes referenced joined in
    (c) For all metrics that reference the parent node, insert the metric expression
        into the parent node's select and build the parent query
    (d) Set the rest of the parent query's attributes (filters, orderby, limit etc)
    (e) Join together the transforms on the shared dimensions
    (f) Select all the requested metrics and dimensions in the final SELECT
    """
    if any(metric_node.type != NodeType.METRIC for metric_node in metric_nodes):
        raise DJInvalidInputException(  # pragma: no cover
            "Cannot build a query for multiple nodes if one or more "
            "of them aren't metric nodes.",
        )

    await validate_shared_dimensions(session, metric_nodes, dimensions, filters)

    combined_ast: ast.Query = ast.Query(
        select=ast.Select(from_=ast.From(relations=[])),
        ctes=[],
    )
    final_mapping = {}
    initial_dimension_columns = []
    all_dimension_columns = []

    orderby_sort_items: List[ast.SortItem] = []
    orderby = orderby or []
    orderby_mapping = {}
    for order in orderby:
        orderby_metric = None
        for metric_node in metric_nodes:
            if metric_node.name.lower() in order.lower():
                orderby_metric = metric_node.name
                break
        orderby_mapping[order] = orderby_metric

    context = CompileContext(session=session, exception=DJException())
    common_parents = group_metrics_by_parent(metric_nodes)

    for parent_node, metrics in common_parents.items():
        # parent_node_ = await Node.get_by_name(session, parent_node.name)
        parent_ast = await build_node(
            session=session,
            node=parent_node.current,
            dimensions=dimensions,
            filters=filters,
            build_criteria=build_criteria,
            include_dimensions_in_groupby=True,
            access_control=access_control,
        )

        # Select only columns that were one of the chosen dimensions
        parent_ast.select.projection = [
            expr
            for expr in parent_ast.select.projection
            if (
                expr.alias_or_name.identifier(False).replace(  # type: ignore
                    f"_{LOOKUP_CHARS.get('.')}_",
                    SEPARATOR,
                )
                + (f"[{expr.role}]" if expr.role else "")  # type: ignore
            )
            in dimensions
        ]
        # Re-alias the dimensions with better names, but keep the group by alias-free
        if parent_ast.select.group_by:  # pragma: no cover
            for i in range(  # pylint: disable=consider-using-enumerate
                len(parent_ast.select.group_by),
            ):
                if hasattr(parent_ast.select.group_by[i], "alias"):
                    parent_ast.select.group_by[i] = ast.Column(
                        name=parent_ast.select.group_by[i].name,  # type: ignore
                        _type=parent_ast.select.group_by[i].type,  # type: ignore
                        # pylint: disable=protected-access
                        _table=parent_ast.select.group_by[i]._table,  # type: ignore
                    )

            if parent_ast.select.where:
                for col in parent_ast.select.where.find_all(ast.Column):
                    if hasattr(col, "alias"):  # pragma: no cover
                        col.alias = None

        for expr in parent_ast.select.projection:
            expr.set_alias(
                ast.Name(amenable_name(expr.alias_or_name.identifier(False))),  # type: ignore
            )
        await parent_ast.compile(context)

        # Add the metric expression into the parent node query
        for metric_node in metrics:
            metric_query = parse(
                NodeRevision.format_metric_alias(
                    metric_node.query,  # type: ignore
                    metric_node.name,
                ),
            )
            await metric_query.compile(context)
            await metric_query.build(session, {})
            parent_ast.select.projection.extend(metric_query.select.projection)

        # Add the WITH statements to the combined query
        parent_ast_alias = ast.Name(amenable_name(parent_node.name))
        parent_ast.alias = parent_ast_alias
        parent_ast.parenthesized = True
        parent_ast.as_ = True
        combined_ast.ctes += [parent_ast]

        # Add the metric and dimensions to the final query layer's SELECT
        current_cte_as_table = ast.Table(parent_ast_alias)

        # If the parent node contains an orderby, parse and add it to the order items
        # bind the table for this built metric to all columns in the
        organization = cast(ast.Organization, parent_ast.select.organization)
        parent_ast.select.organization = None
        if organization:  # pragma: no cover
            for col in organization.find_all(ast.Column):
                col.add_table(current_cte_as_table)
            orderby_sort_items += organization.order  # type: ignore

        final_select_columns = [
            ast.Column(
                name=col.alias_or_name,  # type: ignore
                _table=current_cte_as_table,
                _type=col.type,  # type: ignore
            )
            for col in parent_ast.select.projection
        ]

        metric_columns = []
        dimension_columns = []
        metric_column_identifiers = {amenable_name(metric.name) for metric in metrics}
        for col in final_select_columns:
            if col.name.name in metric_column_identifiers:
                for metric_node in metrics:
                    if amenable_name(metric_node.name) in col.alias_or_name.name:
                        final_mapping[metric_node.name] = col
                        col.set_semantic_entity(
                            metric_node.name + SEPARATOR + metric_node.columns[0].name,
                        )
                        col.set_semantic_type(SemanticType.METRIC)
                metric_columns.append(col)
            else:
                col.set_semantic_entity(from_amenable_name(col.alias_or_name.name))
                col.set_semantic_type(SemanticType.DIMENSION)
                dimension_columns.append(col)

        all_dimension_columns += dimension_columns
        combined_ast.select.projection.extend(metric_columns)

        # Each time we build another parent node CTE clause, add it
        # to the join clause with the current CTEs on the selected dimensions
        if not combined_ast.select.from_.relations:  # type: ignore
            initial_dimension_columns = dimension_columns
            combined_ast.select.from_.relations.append(  # type: ignore
                ast.Relation(current_cte_as_table),
            )
        else:
            join_parents_on = [
                ast.BinaryOp.Eq(initial_dim_col, current_dim_col)
                for initial_dim_col, current_dim_col in zip(
                    initial_dimension_columns,
                    dimension_columns,
                )
            ]
            combined_ast.select.from_.relations[0].extensions.append(  # type: ignore
                ast.Join(
                    "FULL OUTER",
                    ast.Table(parent_ast_alias),
                    ast.JoinCriteria(
                        on=ast.BinaryOp.And(*join_parents_on),
                    ),
                ),
            )

    # Include dimensions in the final select: COALESCE the dimensions across
    # all parent nodes, which will be used as the joins
    dimension_grouping: Dict[str, List] = {}
    for col in all_dimension_columns:
        dimension_grouping.setdefault(col.identifier(), []).append(col)
    dimension_columns = [
        ast.Function(name=ast.Name("COALESCE"), args=list(columns))
        .set_alias(ast.Name(col_name))
        .set_semantic_entity(columns[0].semantic_entity)
        .set_semantic_type(columns[0].semantic_type)
        if len(columns) > 1
        else columns[0]
        for col_name, columns in dimension_grouping.items()
    ]
    for dim_col in dimension_columns:
        dimension_name = dim_col.alias_or_name.name.replace(
            f"_{LOOKUP_CHARS.get('.')}_",
            SEPARATOR,
        )
        final_mapping[dimension_name] = dim_col

    combined_ast.select.projection.extend(dimension_columns)

    # go through the orderby items and make sure we put them in the order the user requested them in
    for idx, sort_item in enumerate(orderby_mapping):
        if isinstance(sort_item, ast.SortItem):
            orderby_sort_items.insert(idx, sort_item)  # pragma: no cover
        else:
            sort_expr_list = sort_item.split(" ")
            if sort_item in final_mapping:  # pragma: no cover
                orderby_sort_items.insert(
                    idx,
                    ast.SortItem(
                        expr=final_mapping[sort_expr_list[0]].alias_or_name,  # type: ignore
                        asc=sort_expr_list[1] if len(sort_expr_list) >= 2 else "",
                        nulls=sort_expr_list[2] if len(sort_expr_list) == 3 else "",
                    ),
                )

    combined_ast.select.organization = ast.Organization(orderby_sort_items)

    if limit is not None:
        combined_ast.select.limit = ast.Number(limit)
    return combined_ast


def build_temp_select(temp_query: str):
    """
    Builds an intermediate select ast used to build cube queries
    """
    temp_select = parse(temp_query).select

    for col in temp_select.find_all(ast.Column):
        dimension_column = col.identifier(False).replace(
            SEPARATOR,
            f"_{LOOKUP_CHARS.get(SEPARATOR)}_",
        )
        col.name = ast.Name(dimension_column)
    return temp_select


def build_materialized_cube_node(
    selected_metrics: List[Column],
    selected_dimensions: List[Column],
    cube: NodeRevision,
    filters: List[str] = None,
    orderby: List[str] = None,
    limit: Optional[int] = None,
) -> ast.Query:
    """
    Build query for a materialized cube node
    """
    combined_ast: ast.Query = ast.Query(
        select=ast.Select(from_=ast.From(relations=[])),
        ctes=[],
    )
    materialization_config = cube.materializations[0]
    cube_config = GenericCubeConfig.parse_obj(materialization_config.config)

    if materialization_config.name == "default":
        # TODO: remove after we migrate old Druid materializations  # pylint: disable=fixme
        selected_metric_keys = [
            col.name for col in selected_metrics
        ]  # pragma: no cover
    else:
        selected_metric_keys = [
            col.node_revision().name for col in selected_metrics  # type: ignore
        ]

    # Assemble query for materialized cube based on the previously saved measures
    # combiner expression for each metric
    for metric_key in selected_metric_keys:
        if metric_key in cube_config.measures:  # pragma: no cover
            metric_measures = cube_config.measures[metric_key]
            measures_combiner_ast = parse(f"SELECT {metric_measures.combiner}")
            measures_type_lookup = {
                (
                    measure.name
                    if materialization_config.name == "default"
                    else measure.field_name
                ): measure.type
                for measure in metric_measures.measures
            }
            for col in measures_combiner_ast.find_all(ast.Column):
                col.add_type(
                    ColumnType(
                        measures_type_lookup[col.alias_or_name.name],  # type: ignore
                    ),
                )
            combined_ast.select.projection.extend(
                [
                    proj.set_alias(ast.Name(amenable_name(metric_key)))
                    for proj in measures_combiner_ast.select.projection
                ],
            )

    # Add in selected dimension attributes to the query
    for selected_dim in selected_dimensions:
        dimension_column = ast.Column(
            name=ast.Name(
                (
                    selected_dim.node_revision().name  # type: ignore
                    + SEPARATOR
                    + selected_dim.name
                ).replace(SEPARATOR, f"_{LOOKUP_CHARS.get(SEPARATOR)}_"),
            ),
        )
        combined_ast.select.projection.append(dimension_column)
        combined_ast.select.group_by.append(dimension_column)

    # Add in filters to the query
    filter_asts = []
    for filter_ in filters:  # type: ignore
        temp_select = build_temp_select(
            f"select * where {filter_}",
        )
        filter_asts.append(temp_select.where)

    if filter_asts:  # pragma: no cover
        # pylint: disable=no-value-for-parameter
        combined_ast.select.where = ast.BinaryOp.And(*filter_asts)

    # Add orderby
    if orderby:  # pragma: no cover
        temp_select = build_temp_select(
            f"select * order by {','.join(orderby)}",
        )
        combined_ast.select.organization = temp_select.organization

    # Add limit
    if limit:  # pragma: no cover
        combined_ast.select.limit = ast.Number(value=limit)

    # Set up FROM clause
    combined_ast.select.from_.relations.append(  # type: ignore
        ast.Relation(primary=ast.Table(ast.Name(cube.availability.table))),  # type: ignore
    )
    return combined_ast


def build_source_node_query(node: NodeRevision):
    """
    Returns a query that selects each column explicitly in the source node.
    """
    table = ast.Table(to_namespaced_name(node.name), None, _dj_node=node)
    select = ast.Select(
        projection=[
            ast.Column(ast.Name(tbl_col.name), _table=table) for tbl_col in node.columns
        ],
        from_=ast.From(relations=[ast.Relation(table)]),
    )
    return ast.Query(select=select)


async def build_ast(  # pylint: disable=too-many-arguments
    session: AsyncSession,
    query: ast.Query,
    memoized_queries: Dict[int, ast.Query] = None,
    build_criteria: Optional[BuildCriteria] = None,
    filters: Optional[List[str]] = None,
    dimensions: Optional[List[str]] = None,
    access_control=None,
) -> ast.Query:
    """
    Determines the optimal way to build the query AST and does so
    """
    memoized_queries = memoized_queries or {}

    start = time.time()
    context = CompileContext(session=session, exception=DJException())
    hashed_query = hash(query)
    if hashed_query in memoized_queries:
        query = memoized_queries[hashed_query]  # pragma: no cover
    else:
        await query.compile(context)
        memoized_queries[hashed_query] = query
    end = time.time()
    _logger.info("Finished compiling query %s in %s", str(query)[-100:], end - start)

    start = time.time()
    await query.build(
        session,
        memoized_queries,
        build_criteria,
        filters,
        dimensions,
        access_control,
    )
    end = time.time()
    _logger.info("Finished building query in %s", end - start)
    return query


async def metrics_to_measures(
    session: AsyncSession,
    metric_nodes: List[Node],
) -> Tuple[DefaultDict[str, Set[str]], DefaultDict[str, Set[str]]]:
    """
    For the given metric nodes, returns a mapping between the metrics' referenced parent nodes
    and the list of necessary measures to extract from the parent node.
    The structure is:
    {
        "parent_node_name1": ["measure_columnA", "measure_columnB"],
        "parent_node_name2": ["measure_columnX"],
    }
    """
    ctx = CompileContext(session, DJException())
    metric_to_measures = collections.defaultdict(set)
    parents_to_measures = collections.defaultdict(set)
    for metric_node in metric_nodes:
        metric_ast = parse(metric_node.current.query)
        await metric_ast.compile(ctx)
        for col in metric_ast.find_all(ast.Column):
            if col.table:  # pragma: no cover
                parents_to_measures[col.table.dj_node.name].add(  # type: ignore
                    col.alias_or_name.name,
                )
                metric_to_measures[metric_node.name].add(
                    col.alias_or_name.name,
                )
    return parents_to_measures, metric_to_measures


async def get_measures_query(
    session: AsyncSession,
    metrics: List[str],
    dimensions: List[str],
    filters: List[str],
    engine_name: Optional[str] = None,
    engine_version: Optional[str] = None,
    current_user: Optional[User] = None,
    validate_access: access.ValidateAccessFn = None,
    cast_timestamp_to_ms: bool = False,
    include_all_columns: bool = False,
) -> TranslatedSQL:
    """
    Builds the measures SQL for a set of metrics with dimensions and filters.
    This SQL can be used to produce an intermediate table with all the measures
    and dimensions needed for an analytics database (e.g., Druid).
    """
    from datajunction_server.api.helpers import (  # pylint: disable=import-outside-toplevel
        assemble_column_metadata,
        validate_cube,
    )

    engine = (
        await get_engine(session, engine_name, engine_version)
        if engine_name and engine_version
        else None
    )
    build_criteria = BuildCriteria(
        dialect=engine.dialect if engine and engine.dialect else Dialect.SPARK,
    )
    access_control = access.AccessControlStore(
        validate_access=validate_access,
        user=current_user,
        base_verb=access.ResourceRequestVerb.READ,
    )

    combined_ast: ast.Query = ast.Query(
        select=ast.Select(from_=ast.From(relations=[])),
        ctes=[],
    )
    initial_dimension_columns = []
    all_dimension_columns = []

    if not filters:
        filters = []

    (_, metric_nodes, _, _, _) = await validate_cube(
        session,
        metrics,
        dimensions,
    )
    context = CompileContext(session=session, exception=DJException())
    common_parents = group_metrics_by_parent(metric_nodes)

    # Mapping between each metric node and its measures
    parents_to_measures, _ = await metrics_to_measures(
        session,
        metric_nodes,
    )

    column_name_regex = r"([A-Za-z0-9_\.]+)(\[[A-Za-z0-9_]+\])?"
    matcher = re.compile(column_name_regex)
    dimensions_without_roles = [matcher.findall(dim)[0][0] for dim in dimensions]

    for parent_node, _ in common_parents.items():  # type: ignore
        measure_columns, dimensional_columns = [], []
        parent_ast = await build_node(
            session=session,
            node=parent_node.current,
            dimensions=dimensions,
            filters=filters,
            build_criteria=build_criteria,
            include_dimensions_in_groupby=False,
            access_control=access_control,
        )

        print("parent_ast", parent_ast)

        # Select only columns that were one of the necessary measures
        if not include_all_columns:
            parent_ast.select.projection = [
                expr
                for expr in parent_ast.select.projection
                if from_amenable_name(expr.alias_or_name.identifier(False)).split(  # type: ignore
                    SEPARATOR,
                )[
                    -1
                ]
                in parents_to_measures[parent_node.name]
                or from_amenable_name(expr.alias_or_name.identifier(False))  # type: ignore
                in dimensions_without_roles
            ]
        await session.refresh(parent_node.current, ["columns"])
        parent_ast = rename_columns(parent_ast, parent_node.current)
        print("parent_ast after select only cols", parent_ast)

        # Sort the selected columns into dimension vs measure columns and
        # generate identifiers for them
        for expr in parent_ast.select.projection:
            column_identifier = expr.alias_or_name.identifier(False)  # type: ignore
            if from_amenable_name(column_identifier) in dimensions_without_roles:
                dimensional_columns.append(expr)
                expr.set_semantic_type(SemanticType.DIMENSION)  # type: ignore
            else:
                measure_columns.append(expr)
                expr.set_semantic_type(SemanticType.MEASURE)  # type: ignore
        await parent_ast.compile(context)

        # Add the WITH statements to the combined query
        parent_ast_alias = parent_ast.ctes[-1].alias
        # parent_ast.alias = parent_ast_alias
        # parent_ast.parenthesized = True
        # parent_ast.as_ = True
        combined_ast.ctes.extend(parent_ast.ctes)

        # Add the measures and dimensions to the final query layer's SELECT
        current_cte_as_table = ast.Table(parent_ast_alias)

        # Add dimensions to the final query layer's SELECT
        outer_dimensional_columns = []
        for col in dimensional_columns:
            col_without_alias = ast.Column(
                name=ast.Name(col.alias_or_name.name),  # type: ignore
                _table=parent_ast,
                _type=col.type,  # type: ignore
                semantic_entity=col.semantic_entity,  # type: ignore
                semantic_type=col.semantic_type,  # type: ignore
            )
            outer_dimensional_columns.append(col_without_alias)
        all_dimension_columns += outer_dimensional_columns

        # Add measures to the final query layer's SELECT
        for col in measure_columns:
            combined_ast.select.projection.append(
                ast.Column(
                    name=ast.Name(col.alias_or_name.name),  # type: ignore
                    _table=parent_ast,
                    _type=col.type,  # type: ignore
                    semantic_entity=col.semantic_entity,  # type: ignore
                    semantic_type=col.semantic_type,  # type: ignore
                ),
            )

        # Each time we build another parent node CTE clause, add it
        # to the join clause with the current CTEs on the selected dimensions
        if not combined_ast.select.from_.relations:  # type: ignore
            initial_dimension_columns = outer_dimensional_columns
            combined_ast.select.from_.relations.append(  # type: ignore
                ast.Relation(current_cte_as_table),
            )
        else:
            join_parents_on = [
                ast.BinaryOp.Eq(initial_dim_col, current_dim_col)
                for initial_dim_col, current_dim_col in zip(
                    initial_dimension_columns,
                    outer_dimensional_columns,
                )
            ]
            if join_parents_on:  # pragma: no cover
                combined_ast.select.from_.relations[0].extensions.append(  # type: ignore
                    ast.Join(
                        "FULL OUTER",
                        ast.Table(parent_ast_alias),
                        ast.JoinCriteria(
                            on=ast.BinaryOp.And(*join_parents_on),
                        ),
                    ),
                )

    # Include dimensions in the final select: COALESCE the dimensions across
    # all parent nodes, which will be used as the joins
    dimension_grouping: Dict[str, List] = {}
    for col in all_dimension_columns:
        dimension_grouping.setdefault(
            col.alias_or_name.identifier(False),  # type: ignore
            [],
        ).append(
            col,
        )

    dimension_columns = [
        ast.Function(name=ast.Name("COALESCE"), args=list(columns))
        .set_alias(
            ast.Name(col_name),
        )
        .set_semantic_entity(columns[0].semantic_entity)
        .set_semantic_type(columns[0].semantic_type)
        if len(columns) > 1
        else columns[0]
        for col_name, columns in dimension_grouping.items()
    ]
    combined_ast.select.projection.extend(dimension_columns)
    for col in dimension_columns:
        if col.type == TimestampType() and cast_timestamp_to_ms:  # type: ignore
            col = (
                ast.Cast(
                    expression=ast.BinaryOp(
                        op=ast.BinaryOpKind.Multiply,
                        left=ast.Cast(
                            data_type=DoubleType(),
                            expression=col.copy().child
                            if isinstance(col, ast.Alias)
                            else col,
                        ),  # type: ignore
                        right=ast.Number(1000),
                    ),
                    data_type=LongType(),
                )
                .set_alias(ast.Name("timestamp_column"))  # type: ignore
                .set_semantic_entity(col.semantic_entity)  # type: ignore
                .set_semantic_type(SemanticType.TIMESTAMP)
            )
            combined_ast.select.projection.append(col)

    # Assemble column metadata
    columns_metadata = []
    for col in combined_ast.select.projection:
        metadata = assemble_column_metadata(  # pragma: no cover
            cast(ast.Column, col),
        )
        columns_metadata.append(metadata)
    dependencies, _ = await combined_ast.extract_dependencies(
        CompileContext(session, DJException()),
    )
    return TranslatedSQL(
        sql=str(combined_ast),
        columns=columns_metadata,
        dialect=build_criteria.dialect,
        upstream_tables=[
            f"{dep.catalog.name}.{dep.schema_}.{dep.table}"
            for dep in dependencies
            if dep.type == NodeType.SOURCE
        ],
    )
