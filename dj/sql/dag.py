"""
DAG related functions.
"""

import asyncio
import operator
from collections import defaultdict
from io import StringIO
from typing import Any, DefaultDict, Dict, List, Optional, Set

import asciidag.graph
import asciidag.node
from sqlmodel import Session, select
from sqloxide import parse_sql

from dj.constants import DJ_DATABASE_ID
from dj.errors import DJException
from dj.models.database import Database
from dj.models.node import Node, NodeRevision
from dj.sql.parse import find_nodes_by_key
from dj.typing import ParseTree
from dj.utils import get_settings

settings = get_settings()


def render_dag(dependencies: Dict[str, Set[str]], **kwargs: Any) -> str:
    """
    Render the DAG of dependencies.
    """
    out = StringIO()
    graph = asciidag.graph.Graph(out, **kwargs)

    asciidag_nodes: Dict[str, asciidag.node.Node] = {}
    tips = sorted(
        [build_asciidag(name, dependencies, asciidag_nodes) for name in dependencies],
        key=lambda n: n.item,
    )

    graph.show_nodes(tips)
    out.seek(0)
    return out.getvalue()


def build_asciidag(
    name: str,
    dependencies: Dict[str, Set[str]],
    asciidag_nodes: Dict[str, asciidag.node.Node],
) -> asciidag.node.Node:
    """
    Build the nodes for ``asciidag``.
    """
    if name in asciidag_nodes:
        asciidag_node = asciidag_nodes[name]
    else:
        asciidag_node = asciidag.node.Node(name)
        asciidag_nodes[name] = asciidag_node

    asciidag_node.parents = sorted(
        [
            build_asciidag(child, dependencies, asciidag_nodes)
            for child in dependencies[name]
        ],
        key=lambda n: n.item,
    )

    return asciidag_node


def get_computable_databases(
    node: NodeRevision,
    columns: Optional[Set[str]] = None,
) -> Set[Database]:
    """
    Return all the databases where a given node can be computed.

    This takes into consideration the node query, since some of the columns might
    not be present in all databases.
    """
    if columns is None:
        columns = {column.name for column in node.columns}

    # add all the databases where the node is explicitly materialized
    tables = [
        table
        for table in node.tables
        if columns <= {column.name for column in table.columns}
    ]
    databases = {table.database for table in tables}

    # add all the databases that are common between the parents and match all the columns
    parent_columns = get_referenced_columns_from_sql(node.query, node.parents)
    if node.parents:
        parent_databases = [
            get_computable_databases(parent.current, parent_columns[parent.name])
            for parent in node.parents
        ]
        databases |= set.intersection(*parent_databases)

    return databases


async def get_database_for_nodes(
    session: Session,
    nodes: List[Node],
    node_columns: Dict[str, Set[str]],
    database_name: Optional[str] = None,
    check_database_online: bool = True,
) -> Database:
    """
    Given a list of nodes, return the best database to compute metric.

    When no nodes are passed, the database with the lowest cost is returned.
    """
    if nodes:
        databases = set.intersection(
            *[
                get_computable_databases(node.current, node_columns[node.name])
                for node in nodes
            ]
        )
    else:
        databases = session.exec(
            select(Database).where(Database.id != DJ_DATABASE_ID),
        ).all()

    if not databases:
        raise Exception("No valid database was found")

    # if a specific database was requested, return it if it's online
    if database_name is not None:
        for database in databases:
            if database.name == database_name:
                if check_database_online and await database.do_ping():
                    return database
                return database
        raise Exception(f"Unknown database `{database_name}`")

    return await get_cheapest_online_database(
        databases,
        check_database_online=check_database_online,
    )


async def get_cheapest_online_database(
    databases: Set[Database],
    check_database_online: bool = True,
) -> Database:
    """
    Return the cheapest online database.

    The function will try to wait until the fastest database is pinged successfully. If
    it's offline, it will try to wait until the second fastest, and so on. After waiting
    for ``settings.do_ping_timeout`` it will return the fastest database that is online.
    """
    return {0}


def get_referenced_columns_from_sql(
    query: Optional[str],
    parents: List[Node],
) -> DefaultDict[str, Set[str]]:
    """
    Given a SQL query, return the referenced columns.

    Referenced columns are a dictionary mapping parent name to column name(s).
    """
    if not query:
        return defaultdict(set)

    tree = parse_sql(query, dialect="ansi")

    return get_referenced_columns_from_tree(tree, parents)


def get_referenced_columns_from_tree(
    tree: ParseTree,
    parents: List[Node],
) -> DefaultDict[str, Set[str]]:
    """
    Return the columns referenced in parents given a parse tree.
    """
    referenced_columns: DefaultDict[str, Set[str]] = defaultdict(set)

    parent_columns = {
        parent.name: {column.name for column in parent.current.columns}
        for parent in parents
    }

    # compound identifiers are fully qualified
    for compound_identifier in find_nodes_by_key(tree, "CompoundIdentifier"):
        parent = ".".join(part["value"] for part in compound_identifier[:-1])
        column = compound_identifier[-1]["value"]
        referenced_columns[parent].add(column)

    # for regular identifiers we need to figure out which parent the columns belongs to
    for identifier in find_nodes_by_key(tree, "Identifier"):
        column = identifier["value"]
        candidates = [
            parent for parent, columns in parent_columns.items() if column in columns
        ]
        if not candidates:
            raise Exception(f"Column {column} not found in any parent")
        if len(candidates) > 1:
            raise Exception(f"Column {column} is ambiguous")
        parent = candidates[0]
        referenced_columns[parent].add(column)

    return referenced_columns


def get_dimensions(node: Node) -> List[str]:
    """
    Return the available dimensions in a given node.
    """
    dimensions = []
    for parent in node.current.parents:
        for column in parent.current.columns:
            dimensions.append(f"{parent.name}.{column.name}")

            if column.dimension:
                for dimension_column in column.dimension.current.columns:
                    dimensions.append(
                        f"{column.dimension.name}.{dimension_column.name}",
                    )

    return sorted(dimensions)
