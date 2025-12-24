"""
Build V3: SQL Generation

It generates both measures SQL (pre-aggregated to dimensional grain) and
metrics SQL (with final metric expressions applied).
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from functools import reduce
from typing import Optional, cast, Any

from sqlalchemy import select, text, bindparam
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload

from datajunction_server.construction.build_v3.alias_registry import AliasRegistry
from datajunction_server.database.dimensionlink import DimensionLink
from datajunction_server.database.node import Node, NodeRevision
from datajunction_server.errors import DJInvalidInputException
from datajunction_server.models.decompose import MetricComponent, Aggregability
from datajunction_server.models.dialect import Dialect
from datajunction_server.models.node_type import NodeType
from datajunction_server.sql.decompose import MetricComponentExtractor
from datajunction_server.sql.parsing import ast
from datajunction_server.sql.parsing.backends.antlr4 import parse
from datajunction_server.utils import SEPARATOR
from datajunction_server.construction.build_v3.types import (
    BuildContext,
    ColumnMetadata,
    GrainGroupSQL,
    GeneratedMeasuresSQL,
    GeneratedSQL,
    JoinPath,
    ResolvedDimension,
    DecomposedMetricInfo,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def get_column_type(node: Node, column_name: str) -> str:
    """
    Look up the column type from a node's columns.

    Returns the string representation of the column type, or "string" as fallback.
    """
    if node.current and node.current.columns:
        for col in node.current.columns:
            if col.name == column_name:
                return str(col.type) if col.type else "string"
    return "string"


# =============================================================================
# Node Loading
# =============================================================================


async def find_upstream_node_names(
    session: AsyncSession,
    starting_node_names: list[str],
) -> tuple[set[str], dict[str, list[str]]]:
    """
    Find all upstream node names using a lightweight recursive CTE.

    This returns just node names (not full nodes) to minimize query overhead.
    Uses NodeRelationship to traverse the parent-child relationships.

    Returns:
        Tuple of:
        - Set of all upstream node names (including the starting nodes)
        - Dict mapping child_name -> list of parent_names (for metrics to find their parents)
    """
    if not starting_node_names:
        return set(), {}

    # Lightweight recursive CTE - returns node names AND parent-child relationships
    recursive_query = text("""
        WITH RECURSIVE upstream AS (
            -- Base case: get the starting nodes' current revision IDs
            SELECT
                nr.id as revision_id,
                n.name as node_name,
                CAST(NULL AS TEXT) as child_name
            FROM node n
            JOIN noderevision nr ON n.id = nr.node_id AND n.current_version = nr.version
            WHERE n.name IN :starting_names
            AND n.deactivated_at IS NULL

            UNION

            -- Recursive case: find parents of current nodes
            SELECT
                parent_nr.id as revision_id,
                parent_n.name as node_name,
                u.node_name as child_name
            FROM upstream u
            JOIN noderelationship nrel ON u.revision_id = nrel.child_id
            JOIN node parent_n ON nrel.parent_id = parent_n.id
            JOIN noderevision parent_nr ON parent_n.id = parent_nr.node_id
                AND parent_n.current_version = parent_nr.version
            WHERE parent_n.deactivated_at IS NULL
        )
        SELECT DISTINCT node_name, child_name FROM upstream
    """).bindparams(bindparam("starting_names", expanding=True))

    result = await session.execute(
        recursive_query,
        {"starting_names": list(starting_node_names)},
    )
    rows = result.fetchall()

    # Collect all node names and parent relationships
    all_names: set[str] = set()
    # child_name -> list of parent_names
    parent_map: dict[str, list[str]] = {}

    for node_name, child_name in rows:
        all_names.add(node_name)
        if child_name:
            # node_name is the parent of child_name
            if child_name not in parent_map:
                parent_map[child_name] = []
            parent_map[child_name].append(node_name)

    return all_names, parent_map


async def find_join_paths_batch(
    session: AsyncSession,
    source_revision_ids: set[int],
    target_dimension_names: set[str],
    max_depth: int = 5,
) -> dict[tuple[int, str, str], list[int]]:
    """
    Find join paths from multiple source nodes to all target dimension nodes
    using a single recursive CTE query.

    Args:
        source_revision_ids: Set of source node revision IDs to find paths from
        target_dimension_names: Set of dimension node names to find paths to
        max_depth: Maximum path depth to search

    Returns a dict mapping (source_revision_id, dimension_node_name, role_path)
    to the list of DimensionLink IDs forming the path.

    The role_path is a "->" separated string of roles at each step.
    Empty roles are represented as empty strings.

    This is O(1) database calls instead of O(sources * nodes * depth) individual queries.
    """
    if not target_dimension_names or not source_revision_ids:
        return {}

    # Single recursive CTE to find all paths from all sources at once
    recursive_query = text("""
        WITH RECURSIVE paths AS (
            -- Base case: first level dimension links from any source node
            SELECT
                dl.node_revision_id as source_rev_id,
                dl.id as link_id,
                n.name as dim_name,
                CAST(dl.id AS TEXT) as path,
                COALESCE(dl.role, '') as role_path,
                1 as depth
            FROM dimensionlink dl
            JOIN node n ON dl.dimension_id = n.id
            WHERE dl.node_revision_id IN :source_revision_ids

            UNION ALL

            -- Recursive case: follow dimension_links from each dimension node
            SELECT
                paths.source_rev_id as source_rev_id,
                dl2.id as link_id,
                n2.name as dim_name,
                paths.path || ',' || CAST(dl2.id AS TEXT) as path,
                paths.role_path || '->' || COALESCE(dl2.role, '') as role_path,
                paths.depth + 1 as depth
            FROM paths
            JOIN node prev_node ON paths.dim_name = prev_node.name
            JOIN noderevision nr ON prev_node.current_version = nr.version AND nr.node_id = prev_node.id
            JOIN dimensionlink dl2 ON dl2.node_revision_id = nr.id
            JOIN node n2 ON dl2.dimension_id = n2.id
            WHERE paths.depth < :max_depth
        )
        SELECT source_rev_id, dim_name, path, role_path, depth
        FROM paths
        WHERE dim_name IN :target_names
        ORDER BY depth ASC
    """).bindparams(
        bindparam("source_revision_ids", expanding=True),
        bindparam("target_names", expanding=True),
    )

    result = await session.execute(
        recursive_query,
        {
            "source_revision_ids": list(source_revision_ids),
            "max_depth": max_depth,
            "target_names": list(target_dimension_names),
        },
    )
    rows = result.fetchall()

    # Build paths dict keyed by (source_rev_id, dim_name, role_path)
    paths: dict[tuple[int, str, str], list[int]] = {}
    for source_rev_id, dim_name, path_str, role_path, depth in rows:
        key = (source_rev_id, dim_name, role_path or "")
        if key not in paths:  # Keep first (shortest) path
            paths[key] = [int(x) for x in path_str.split(",")]

    return paths


async def load_dimension_links_batch(
    session: AsyncSession,
    link_ids: set[int],
) -> dict[int, DimensionLink]:
    """
    Batch load DimensionLinks and their associated dimension Nodes.
    Returns a dict mapping link_id to DimensionLink object.
    """
    if not link_ids:
        return {}

    stmt = (
        select(DimensionLink)
        .where(DimensionLink.id.in_(link_ids))
        .options(
            joinedload(DimensionLink.dimension).options(
                joinedload(Node.current).options(
                    selectinload(NodeRevision.columns),
                    joinedload(NodeRevision.catalog),
                    selectinload(NodeRevision.dimension_links).options(
                        joinedload(DimensionLink.dimension),
                    ),
                ),
            ),
            joinedload(DimensionLink.node_revision),
        )
    )
    result = await session.execute(stmt)
    links = result.scalars().unique().all()

    return {link.id: link for link in links}


async def preload_join_paths(
    ctx: BuildContext,
    source_revision_ids: set[int],
    target_dimension_names: set[str],
) -> None:
    """
    Preload all join paths from multiple source nodes to target dimensions.

    Uses a single recursive CTE query to find paths from ALL sources at once,
    then a single batch load for DimensionLink objects. Results are stored
    in ctx.join_paths.

    This is O(2) queries regardless of how many source nodes we have.
    """
    if not target_dimension_names or not source_revision_ids:
        return

    # Find all paths from all sources using recursive CTE (single query)
    path_ids = await find_join_paths_batch(
        ctx.session,
        source_revision_ids,
        target_dimension_names,
    )

    # Collect all link IDs we need to load
    all_link_ids: set[int] = set()
    for link_id_list in path_ids.values():
        all_link_ids.update(link_id_list)

    # Batch load all DimensionLinks (single query)
    link_dict = await load_dimension_links_batch(ctx.session, all_link_ids)

    # Store in context, keyed by (source_revision_id, dim_name, role_path)
    for (source_rev_id, dim_name, role_path), link_id_list in path_ids.items():
        links = [link_dict[lid] for lid in link_id_list if lid in link_dict]
        ctx.join_paths[(source_rev_id, dim_name, role_path)] = links
        # Also cache dimension nodes
        for link in links:
            if link.dimension and link.dimension.name not in ctx.nodes:
                ctx.nodes[link.dimension.name] = link.dimension

    logger.debug(
        f"[BuildV3] Preloaded {len(path_ids)} join paths for "
        f"{len(source_revision_ids)} sources in 2 queries",
    )


async def load_nodes(ctx: BuildContext) -> None:
    """
    Load all nodes needed for SQL generation with minimal database queries.

    Query 1: Find all upstream node names using lightweight recursive CTE
    Query 2: Batch load all those nodes with eager loading
    Query 3-4: Find join paths and batch load dimension links

    Total: ~4 queries regardless of graph size.
    """
    # Collect initial node names (metrics + explicit dimension nodes)
    initial_node_names = set(ctx.metrics)

    # Parse dimension references to get target dimension node names
    target_dim_names: set[str] = set()
    for dim in ctx.dimensions:
        dim_ref = parse_dimension_ref(dim)
        if dim_ref.node_name:
            initial_node_names.add(dim_ref.node_name)
            target_dim_names.add(dim_ref.node_name)

    # Query 1: Find ALL upstream node names AND parent relationships using lightweight recursive CTE
    # This includes all transitive dependencies (sources, transforms, dimensions)
    all_node_names, parent_map = await find_upstream_node_names(
        ctx.session,
        list(initial_node_names),
    )

    # Store parent map in context for later use (e.g., get_parent_node)
    ctx.parent_map = parent_map

    # Also include the initial nodes themselves
    all_node_names.update(initial_node_names)

    logger.debug(f"[BuildV3] Found {len(all_node_names)} nodes to load")

    # Query 2: Batch load all nodes with appropriate eager loading
    # NOTE: parents and dimension_links are NOT loaded here - parent relationships
    # come from Query 1's parent_map, dimension_links from Queries 3-4
    stmt = (
        select(Node)
        .where(Node.name.in_(all_node_names))
        .where(Node.deactivated_at.is_(None))
        .options(
            selectinload(Node.current).options(
                selectinload(NodeRevision.columns),
                selectinload(NodeRevision.catalog),
                selectinload(NodeRevision.required_dimensions),
            ),
        )
    )

    result = await ctx.session.execute(stmt)
    nodes = result.scalars().unique().all()

    # Cache all loaded nodes
    for node in nodes:
        ctx.nodes[node.name] = node

    # Collect parent revision IDs for join path lookup (using parent_map from Query 1)
    # For derived metrics, we need to recursively find fact parents through the metric chain
    parent_revision_ids: set[int] = set()

    def collect_fact_parents(metric_name: str, visited: set[str]) -> None:
        """Recursively collect fact/transform parent revision IDs from metrics."""
        if metric_name in visited:
            return
        visited.add(metric_name)

        metric_node = ctx.nodes.get(metric_name)
        if not metric_node:
            return

        parent_names = ctx.parent_map.get(metric_name, [])
        for parent_name in parent_names:
            parent_node = ctx.nodes.get(parent_name)
            if not parent_node:
                continue

            if parent_node.type == NodeType.METRIC:
                # Parent is another metric - recurse to find its fact parents
                collect_fact_parents(parent_name, visited)
            elif parent_node.current:
                # Parent is a fact/transform - collect its revision ID
                parent_revision_ids.add(parent_node.current.id)

    for metric_name in ctx.metrics:
        collect_fact_parents(metric_name, set())

    logger.debug(f"[BuildV3] Loaded {len(ctx.nodes)} nodes")

    # Queries 3-4: Preload join paths for ALL parent nodes in a single batch
    await preload_join_paths(ctx, parent_revision_ids, target_dim_names)


# =============================================================================
# Topological Sort & Table Reference Rewriting
# =============================================================================


def get_cte_name(node_name: str) -> str:
    """
    Generate a CTE-safe name from a node name.

    Replaces dots with underscores to create valid SQL identifiers.
    Uses the same logic as amenable_name for consistency.
    """
    return node_name.replace(SEPARATOR, "_").replace("-", "_")


def get_table_references_from_ast(query_ast: ast.Query) -> set[str]:
    """
    Extract all table references from a query AST.

    Returns set of table names (as dotted strings like 'v3.src_orders').
    """
    table_names: set[str] = set()
    for table in query_ast.find_all(ast.Table):
        # Get the full table name including namespace
        table_name = str(table.name)
        if table_name:
            table_names.add(table_name)
    return table_names


def topological_sort_nodes(ctx: BuildContext, node_names: set[str]) -> list[Node]:
    """
    Sort nodes in topological order (dependencies first).

    Uses the query AST to find table references and determine dependencies.
    Source nodes have no dependencies and come first.
    Transform/dimension nodes depend on what they reference in their queries.

    Returns:
        List of nodes sorted so dependencies come before dependents.
    """
    # Build dependency graph
    dependencies: dict[str, set[str]] = {}
    node_map: dict[str, Node] = {}

    for name in node_names:
        node = ctx.nodes.get(name)
        if not node:
            continue
        node_map[name] = node

        if node.type == NodeType.SOURCE:
            # Sources have no dependencies
            dependencies[name] = set()
        elif node.type == NodeType.METRIC:
            # Metrics depend on their parent node (handled separately, skip)
            continue
        elif node.current and node.current.query:
            # Transform/dimension - parse query to find references (using cache)
            try:
                query_ast = ctx.get_parsed_query(node)
                refs = get_table_references_from_ast(query_ast)
                # Only keep references that are in our node set
                dependencies[name] = {r for r in refs if r in node_names}
            except Exception:
                # If we can't parse, assume no dependencies
                dependencies[name] = set()
        else:
            dependencies[name] = set()

    # Kahn's algorithm for topological sort
    in_degree = {name: 0 for name in dependencies}
    for deps in dependencies.values():
        for dep in deps:
            if dep in in_degree:
                in_degree[dep] = in_degree.get(dep, 0) + 1

    # Wait, that's backwards. Let me fix the in-degree calculation.
    # in_degree[X] = number of nodes that X depends on
    in_degree = {name: len(deps) for name, deps in dependencies.items()}

    # Build reverse mapping: which nodes depend on this node?
    dependents: dict[str, list[str]] = {name: [] for name in dependencies}
    for name, deps in dependencies.items():
        for dep in deps:
            if dep in dependents:
                dependents[dep].append(name)

    # Start with nodes that have no dependencies (in_degree == 0)
    # Sort to ensure deterministic output order
    queue = sorted([name for name, degree in in_degree.items() if degree == 0])
    sorted_names: list[str] = []

    while queue:
        current = queue.pop(0)
        sorted_names.append(current)
        # Reduce in-degree for all dependents
        # Collect new zero-degree nodes and sort for determinism
        new_ready = []
        for dependent in dependents.get(current, []):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                new_ready.append(dependent)
        queue.extend(sorted(new_ready))

    # Return sorted nodes (excluding any we couldn't sort due to cycles)
    return [node_map[name] for name in sorted_names if name in node_map]


def rewrite_table_references(
    query_ast: ast.Query,
    ctx: BuildContext,
    cte_names: dict[str, str],
    inner_cte_renames: Optional[dict[str, str]] = None,
) -> ast.Query:
    """
    Rewrite table references in a query AST.

    - Source nodes -> physical table names (catalog.schema.table)
    - Transform/dimension nodes -> CTE names
    - Inner CTE names -> prefixed CTE names with alias to original name
      e.g., `FROM base` -> `FROM prefix_base base` (keeps column refs like base.col working)

    Args:
        query_ast: The query AST to rewrite (modified in place)
        ctx: Build context with loaded nodes
        cte_names: Mapping of node names to their CTE names
        inner_cte_renames: Optional mapping of inner CTE old names to prefixed names

    Returns:
        The modified query AST
    """
    inner_cte_renames = inner_cte_renames or {}

    for table in query_ast.find_all(ast.Table):
        table_name = str(table.name)

        # First check if it's an inner CTE reference that needs renaming
        if table_name in inner_cte_renames:
            # Use the prefixed name but alias it to the original name
            # So `FROM base` becomes `FROM prefix_base base`
            # This keeps column references like `base.col` working
            new_name = inner_cte_renames[table_name]
            table.name = ast.Name(new_name)
            # Only set alias if not already set (preserve existing aliases)
            if not table.alias:
                table.alias = ast.Name(table_name)
            continue

        # Then check if it's a node reference
        ref_node = ctx.nodes.get(table_name)
        if ref_node:
            if ref_node.type == NodeType.SOURCE:
                # Replace with physical table name
                physical_name = get_physical_table_name(ref_node)
                if physical_name:
                    table.name = ast.Name(physical_name)
            elif table_name in cte_names:
                # Replace with CTE name
                table.name = ast.Name(cte_names[table_name])

    return query_ast


def filter_cte_projection(
    query_ast: ast.Query,
    columns_to_select: set[str],
) -> ast.Query:
    """
    Filter a query's projection to only include specified columns.

    This modifies the SELECT clause to only project columns that are
    actually needed downstream.

    Args:
        query_ast: The query AST to modify
        columns_to_select: Set of column names to keep

    Returns:
        Modified query AST with filtered projection
    """
    if not query_ast.select.projection:
        return query_ast

    new_projection = []
    for expr in query_ast.select.projection:
        # Get the name this column will be known by
        if isinstance(expr, ast.Alias):
            col_name = str(expr.alias.name) if expr.alias else None
            if not col_name and isinstance(expr.child, ast.Column):
                col_name = str(expr.child.name.name)
        elif isinstance(expr, ast.Column):
            col_name = str(expr.alias.name) if expr.alias else str(expr.name.name)
        else:
            # Keep expressions we can't analyze
            new_projection.append(expr)
            continue

        # Keep if it's in our needed set
        if col_name and col_name in columns_to_select:
            new_projection.append(expr)

    # If we filtered everything, keep original (shouldn't happen)
    if new_projection:
        query_ast.select.projection = new_projection

    return query_ast


def get_metric_node(ctx: BuildContext, metric_name: str) -> Node:
    """Get a metric node by name, raising if not found or not a metric."""
    node = ctx.nodes.get(metric_name)
    if not node:
        raise DJInvalidInputException(f"Metric not found: {metric_name}")
    if node.type != NodeType.METRIC:
        raise DJInvalidInputException(f"Not a metric node: {metric_name}")
    return node


def get_parent_node(ctx: BuildContext, metric_node: Node) -> Node:
    """Get the parent node of a metric (the node it's defined on)."""
    # Use parent_map from Query 1 instead of the parents relationship
    parent_names = ctx.parent_map.get(metric_node.name, [])
    if not parent_names:
        raise DJInvalidInputException(f"Metric {metric_node.name} has no parent node")

    # Metrics typically have one parent (the node they SELECT FROM)
    parent_name = parent_names[0]
    parent = ctx.nodes.get(parent_name)
    if not parent:
        raise DJInvalidInputException(f"Parent node not found: {parent_name}")
    return parent


def get_metric_expression_ast(ctx: BuildContext, metric_node: Node) -> ast.Expression:
    """
    Extract the metric expression AST from the metric's query.

    A metric query looks like: SELECT <expression> FROM <parent>
    We want to extract just the expression as an AST node.

    Returns:
        The expression AST node (with alias removed if present)
    """
    if not metric_node.current or not metric_node.current.query:
        raise DJInvalidInputException(f"Metric {metric_node.name} has no query")

    # Use cached parsed query
    query_ast = ctx.get_parsed_query(metric_node)
    if not query_ast.select.projection:
        raise DJInvalidInputException(f"Metric {metric_node.name} has no projection")

    # Get the first projection expression (the metric expression)
    expr = query_ast.select.projection[0]

    # Remove alias if present - we want the raw expression
    if isinstance(expr, ast.Alias):
        expr = expr.child

    # Copy the expression before modifying to protect the cache
    expr = expr.copy()
    if isinstance(expr, ast.Aliasable) and expr.alias:
        expr.alias = None

    # Clear parent reference so we can attach to new query
    expr.clear_parent()

    return cast(ast.Expression, expr)


async def decompose_metric(
    session: AsyncSession,
    metric_node: Node,
) -> DecomposedMetricInfo:
    """
    Decompose a metric into its constituent components.

    Uses MetricComponentExtractor to break down aggregations like:
    - SUM(x) -> [sum_x component]
    - AVG(x) -> [sum_x component, count_x component]
    - COUNT(DISTINCT x) -> [distinct_x component with LIMITED aggregability]

    Returns:
        DecomposedMetricInfo with components, combiner expression, and aggregability
    """
    if not metric_node.current:
        raise DJInvalidInputException(
            f"Metric {metric_node.name} has no current revision",
        )

    # Use the existing MetricComponentExtractor
    extractor = MetricComponentExtractor(metric_node.current.id)
    components, derived_ast = await extractor.extract(session)

    # Extract combiner expression from the derived query AST
    # The first projection element is the metric expression with component references
    combiner = (
        str(derived_ast.select.projection[0]) if derived_ast.select.projection else ""
    )

    # Determine overall aggregability (worst case among components)
    if not components:
        # No decomposable aggregations found - treat as NONE
        aggregability = Aggregability.NONE
    elif any(c.rule.type == Aggregability.NONE for c in components):
        aggregability = Aggregability.NONE
    elif any(c.rule.type == Aggregability.LIMITED for c in components):
        aggregability = Aggregability.LIMITED
    else:
        aggregability = Aggregability.FULL

    return DecomposedMetricInfo(
        metric_node=metric_node,
        components=components,
        aggregability=aggregability,
        combiner=combiner,
        derived_ast=derived_ast,
    )


def build_component_expression(component: MetricComponent) -> ast.Expression:
    """
    Build the accumulate expression AST for a metric component.

    For simple aggregations like SUM, this is: SUM(expression)
    For templates like "SUM(POWER({}, 2))", expands to: SUM(POWER(expression, 2))
    """
    if not component.aggregation:
        # No aggregation - just return the expression as a column
        return ast.Column(name=ast.Name(component.expression))

    # Check if it's a template with {}
    if "{" in component.aggregation:
        # Template like "SUM(POWER({}, 2))" - expand it
        expanded = component.aggregation.replace("{}", component.expression)
        # Parse as expression
        expr_ast = parse(f"SELECT {expanded}").select.projection[0]
        if isinstance(expr_ast, ast.Alias):
            expr_ast = expr_ast.child
        expr_ast.clear_parent()
        return cast(ast.Expression, expr_ast)
    else:
        # Simple function name like "SUM" - build SUM(expression)
        arg_expr = parse(f"SELECT {component.expression}").select.projection[0]
        func = ast.Function(
            name=ast.Name(component.aggregation),
            args=[cast(ast.Expression, arg_expr)],
        )
        return func


# =============================================================================
# Dimension Resolution
# =============================================================================


@dataclass
class DimensionRef:
    """Parsed dimension reference."""

    node_name: str
    column_name: str
    role: Optional[str] = None


def parse_dimension_ref(dim_ref: str) -> DimensionRef:
    """
    Parse a dimension reference string.

    Formats:
    - "v3.customer.name" -> node=v3.customer, col=name, role=None
    - "v3.customer.name[order]" -> node=v3.customer, col=name, role=order
    - "v3.date.month[customer->registration]" -> node=v3.date, col=month, role=customer->registration
    """
    # Extract role if present
    role = None
    if "[" in dim_ref:
        dim_part, role_part = dim_ref.rsplit("[", 1)
        role = role_part.rstrip("]")
    else:
        dim_part = dim_ref

    # Split into node and column
    parts = dim_part.rsplit(SEPARATOR, 1)
    if len(parts) == 2:
        node_name, column_name = parts
    else:
        # Assume single part is column name on current node
        node_name = ""
        column_name = parts[0]

    return DimensionRef(node_name=node_name, column_name=column_name, role=role)


def find_join_path(
    ctx: BuildContext,
    from_node: Node,
    target_dim_name: str,
    role: Optional[str] = None,
) -> Optional[JoinPath]:
    """
    Find the join path from a node to a target dimension.

    Uses preloaded join paths from ctx.join_paths (populated by load_nodes).
    This is a pure in-memory lookup - no database queries.

    For single-hop joins:
        fact -> dimension (direct link)

    For multi-hop joins (role like "customer->home"):
        fact -> customer -> location

    If no role is specified, will find ANY path to the dimension (first match).
    This handles cases where the dimension link has a role but the user
    doesn't specify one.

    Returns None if no path found.
    """
    if not from_node.current:
        return None

    source_revision_id = from_node.current.id
    role_path = role or ""

    # Look up preloaded path with exact role match
    key = (source_revision_id, target_dim_name, role_path)
    links = ctx.join_paths.get(key)

    if links:
        # Path found in preloaded cache
        return JoinPath(
            links=links,
            target_dimension=links[-1].dimension,
            role=role,
        )

    # Fallback: if no role specified, find ANY path to this dimension
    # This handles cases where the dimension link has a role but user didn't specify one
    if not role:
        for (src_id, dim_name, stored_role), path_links in ctx.join_paths.items():
            if src_id == source_revision_id and dim_name == target_dim_name:
                logger.debug(
                    f"[BuildV3] Using path with role '{stored_role}' for "
                    f"dimension {target_dim_name} (no role specified)",
                )
                return JoinPath(
                    links=path_links,
                    target_dimension=path_links[-1].dimension,
                    role=stored_role or None,
                )

    return None


def can_skip_join_for_dimension(
    dim_ref: DimensionRef,
    join_path: Optional[JoinPath],
    parent_node: Node,
) -> tuple[bool, Optional[str]]:
    """
    Check if we can skip joining to the dimension and use a local column instead.

    This optimization applies when the requested dimension column is the join key
    itself. For example, if requesting v3.customer.customer_id and the join is:
        v3.order_details.customer_id = v3.customer.customer_id
    We can use v3.order_details.customer_id directly without joining.

    Args:
        dim_ref: The parsed dimension reference
        join_path: The join path to the dimension (if any)
        parent_node: The parent/fact node

    Returns:
        Tuple of (can_skip: bool, local_column_name: str | None)
    """
    if not join_path or not join_path.links:
        return False, None

    # Only optimize single-hop joins for now
    if len(join_path.links) > 1:
        return False, None

    link = join_path.links[0]

    # Get the dimension column being requested (fully qualified)
    dim_col_fqn = f"{dim_ref.node_name}{SEPARATOR}{dim_ref.column_name}"

    # Check if this dimension column is in the foreign keys mapping
    if parent_col := link.foreign_keys_reversed.get(dim_col_fqn):
        return True, parent_col.split(SEPARATOR)[-1]
    return False, None


def resolve_dimensions(
    ctx: BuildContext,
    parent_node: Node,
) -> list[ResolvedDimension]:
    """
    Resolve all requested dimensions to their join paths.

    Includes optimization: if the requested dimension is the join key itself,
    we skip the join and use the local column instead.

    Returns a list of ResolvedDimension objects with join path information.
    """
    resolved = []

    for dim in ctx.dimensions:
        dim_ref = parse_dimension_ref(dim)

        # Check if it's a local dimension (column on the parent node itself)
        is_local = False
        if dim_ref.node_name == parent_node.name:
            is_local = True
        elif not dim_ref.node_name:
            # No node specified, assume it's local
            is_local = True
            dim_ref.node_name = parent_node.name

        if is_local:
            resolved.append(
                ResolvedDimension(
                    original_ref=dim,
                    node_name=dim_ref.node_name,
                    column_name=dim_ref.column_name,
                    role=dim_ref.role,
                    join_path=None,
                    is_local=True,
                ),
            )
        else:
            # Need to find join path
            join_path = find_join_path(
                ctx,
                parent_node,
                dim_ref.node_name,
                dim_ref.role,
            )

            if not join_path and dim_ref.role:
                # Try finding via role path
                # For "v3.date.month[customer->registration]", the target is v3.date
                # but the role path is through customer first
                role_parts = dim_ref.role.split("->")
                if len(role_parts) > 1:
                    # Multi-hop: find path through intermediate dimensions
                    join_path = find_join_path(
                        ctx,
                        parent_node,
                        dim_ref.node_name,
                        dim_ref.role,
                    )

            # Optimization: if requesting the join key column, skip the join
            can_skip, local_col = can_skip_join_for_dimension(
                dim_ref,
                join_path,
                parent_node,
            )
            if can_skip and local_col:
                logger.info(
                    f"[BuildV3] Skipping join for {dim} - using local column {local_col}",
                )
                resolved.append(
                    ResolvedDimension(
                        original_ref=dim,
                        node_name=parent_node.name,  # Use parent node
                        column_name=local_col,  # Use local column name
                        role=dim_ref.role,
                        join_path=None,  # No join needed!
                        is_local=True,
                    ),
                )
            else:
                resolved.append(
                    ResolvedDimension(
                        original_ref=dim,
                        node_name=dim_ref.node_name,
                        column_name=dim_ref.column_name,
                        role=dim_ref.role,
                        join_path=join_path,
                        is_local=False,
                    ),
                )

    return resolved


# =============================================================================
# AST Construction
# =============================================================================


def make_name(dotted_name: str) -> ast.Name:
    """
    Create an AST Name from a dotted string like 'catalog.schema.table'.

    The Name class uses nested namespace attributes:
    'a.b.c' becomes Name('c', namespace=Name('b', namespace=Name('a')))
    """
    parts = dotted_name.split(SEPARATOR)
    if not parts:
        return ast.Name("")

    # Build from left to right, each becoming the namespace of the next
    result = ast.Name(parts[0])
    for part in parts[1:]:
        result = ast.Name(part, namespace=result)

    return result


def amenable_name(name: str) -> str:
    """Convert a node name to a SQL-safe identifier (for CTEs)."""
    return name.replace(SEPARATOR, "_").replace("-", "_")


def get_physical_table_name(node: Node) -> Optional[str]:
    """
    Get the physical table name for a source node.

    For source nodes: Returns catalog.schema.table
    For other nodes: Returns None (they need CTEs)
    """
    rev = node.current
    if not rev:
        return None

    if node.type == NodeType.SOURCE:
        parts = []
        if rev.catalog:
            parts.append(rev.catalog.name)
        if rev.schema_:
            parts.append(rev.schema_)
        if rev.table:
            parts.append(rev.table)
        else:
            parts.append(node.name)
        return SEPARATOR.join(parts)

    return None


def extract_columns_from_expression(expr: ast.Expression) -> set[str]:
    """
    Extract all column names referenced in an expression.
    """
    columns: set[str] = set()
    for col in expr.find_all(ast.Column):
        # Get the column name (last part of the identifier)
        if col.name:
            columns.add(col.name.name)
    return columns


def get_table_reference(ctx: BuildContext, node: Node) -> tuple[str, bool]:
    """
    Get the table reference for a node.

    Returns:
        (reference_name, is_cte): The name to use and whether it's a CTE

    For source nodes: Returns (catalog.schema.table, False)
    For transform/dimension: Returns (cte_name, True) - must add CTE
    """
    # Check for physical table first
    physical = get_physical_table_name(node)
    if physical:
        return (physical, False)

    # For transforms/dimensions, use CTE name
    cte_name = amenable_name(node.name)
    return (cte_name, True)


def flatten_inner_ctes(
    query_ast: ast.Query,
    outer_cte_name: str,
) -> tuple[list[tuple[str, ast.Query]], dict[str, str]]:
    """
    Extract inner CTEs from a query and rename them to avoid collisions.

    If a transform has:
        WITH temp AS (SELECT ...) SELECT * FROM temp

    We extract 'temp' as 'v3_transform__temp' and return the rename mapping.
    The caller is responsible for rewriting references using the returned mapping.

    Args:
        query_ast: The parsed query that may contain inner CTEs
        outer_cte_name: The name of the outer CTE (e.g., 'v3_order_details')

    Returns:
        Tuple of:
        - List of (prefixed_cte_name, cte_query) tuples for the extracted CTEs
        - Dict mapping old CTE names to new prefixed names (for reference rewriting)
    """
    if not query_ast.ctes:
        return [], {}

    extracted_ctes: list[tuple[str, ast.Query]] = []

    # Build mapping of old CTE name -> new prefixed name
    inner_cte_renames: dict[str, str] = {}
    for inner_cte in query_ast.ctes:
        if inner_cte.alias:
            old_name = (
                inner_cte.alias.name
                if hasattr(inner_cte.alias, "name")
                else str(inner_cte.alias)
            )
            new_name = f"{outer_cte_name}__{old_name}"
            inner_cte_renames[old_name] = new_name

    # Extract each inner CTE with renamed name
    for inner_cte in query_ast.ctes:
        if inner_cte.alias:
            old_name = (
                inner_cte.alias.name
                if hasattr(inner_cte.alias, "name")
                else str(inner_cte.alias)
            )
            new_name = inner_cte_renames[old_name]

            # Create a new Query for the CTE content
            cte_query = ast.Query(select=inner_cte.select)
            if inner_cte.ctes:
                # Recursively flatten if this CTE also has CTEs
                nested_ctes, nested_renames = flatten_inner_ctes(cte_query, new_name)
                extracted_ctes.extend(nested_ctes)
                inner_cte_renames.update(nested_renames)

            extracted_ctes.append((new_name, cte_query))

    # Clear inner CTEs from the original query
    query_ast.ctes = []

    return extracted_ctes, inner_cte_renames


def collect_node_ctes(
    ctx: BuildContext,
    nodes_to_include: list[Node],
    needed_columns_by_node: Optional[dict[str, set[str]]] = None,
) -> list[tuple[str, ast.Query]]:
    """
    Collect CTEs for all non-source nodes, recursively expanding table references.

    This handles the full dependency chain:
    - Source nodes -> replaced with physical table names (catalog.schema.table)
    - Transform/dimension nodes -> recursive CTEs with dependencies resolved
    - Inner CTEs within transforms -> flattened and prefixed to avoid collisions

    Args:
        ctx: Build context
        nodes_to_include: List of nodes to create CTEs for
        needed_columns_by_node: Optional dict of node_name -> set of column names
            If provided, CTEs will only select the needed columns.

    Returns list of (cte_name, query_ast) tuples in dependency order.
    """
    # Collect all node names that need CTEs (including transitive dependencies)
    all_node_names: set[str] = set()

    def collect_refs(node: Node, visited: set[str]) -> None:
        if node.name in visited:
            return
        visited.add(node.name)

        if node.type == NodeType.SOURCE:
            return  # Sources don't become CTEs

        all_node_names.add(node.name)

        if node.current and node.current.query:
            try:
                # Use cached parsed query for reference extraction
                query_ast = ctx.get_parsed_query(node)
                refs = get_table_references_from_ast(query_ast)
                for ref in refs:
                    ref_node = ctx.nodes.get(ref)
                    if ref_node:
                        collect_refs(ref_node, visited)
            except Exception:
                pass

    # Collect from all starting nodes
    for node in nodes_to_include:
        collect_refs(node, set())

    # Topologically sort all collected nodes
    sorted_nodes = topological_sort_nodes(ctx, all_node_names)

    # Build CTE name mapping
    cte_names: dict[str, str] = {}
    for node in sorted_nodes:
        cte_names[node.name] = get_cte_name(node.name)

    # Build CTEs in dependency order
    ctes: list[tuple[str, ast.Query]] = []
    for node in sorted_nodes:
        if node.type == NodeType.SOURCE:
            continue

        if not node.current or not node.current.query:
            continue

        # Parse the node's query
        query_ast = parse(node.current.query)

        cte_name = cte_names[node.name]

        # Flatten any inner CTEs to avoid nested WITH clauses
        # Returns extracted CTEs and mapping of old names -> prefixed names
        inner_ctes, inner_cte_renames = flatten_inner_ctes(query_ast, cte_name)

        # Rewrite table references in extracted inner CTEs
        # (they may reference sources that need -> physical table names)
        for inner_cte_name, inner_cte_query in inner_ctes:
            rewrite_table_references(inner_cte_query, ctx, cte_names, inner_cte_renames)

        ctes.extend(inner_ctes)

        # Rewrite table references in main query
        # (sources -> physical tables, transforms -> CTE names, inner CTEs -> prefixed names)
        rewrite_table_references(query_ast, ctx, cte_names, inner_cte_renames)

        # Apply column filtering if specified
        needed_cols = None
        if needed_columns_by_node:
            needed_cols = needed_columns_by_node.get(node.name)

        if needed_cols:
            query_ast = filter_cte_projection(query_ast, needed_cols)

        ctes.append((cte_name, query_ast))

    return ctes


def get_table_reference_parts(node: Node) -> list[str]:
    """
    Get the parts of the fully qualified table reference for a source/transform/dimension node.

    For source nodes: [catalog, schema, table]
    For other nodes: [amenable_name] (CTE reference)

    Returns:
        List of name parts
    """
    rev = node.current
    if not rev:
        raise DJInvalidInputException(f"Node {node.name} has no current revision")

    # For source nodes, build catalog.schema.table
    if node.type == NodeType.SOURCE:
        parts = []
        if rev.catalog:
            parts.append(rev.catalog.name)
        if rev.schema_:
            parts.append(rev.schema_)
        if rev.table:
            parts.append(rev.table)
        else:
            # Fall back to node name
            parts.append(node.name)
        return parts

    # For transform/dimension nodes, use amenable name (CTE reference)
    return [amenable_name(node.name)]


def build_join_clause(
    ctx: BuildContext,
    link: DimensionLink,
    left_alias: str,
    right_alias: str,
) -> ast.Join:
    """
    Build a JOIN clause AST from a dimension link.

    Args:
        ctx: Build context
        link: The dimension link defining the join
        left_alias: Alias for the left (source) table
        right_alias: Alias for the right (dimension) table

    Returns:
        AST Join node
    """
    # Parse the join SQL to get the ON clause
    # link.join_sql looks like: "v3.order_details.customer_id = v3.customer.customer_id"
    join_sql = link.join_sql

    # Replace the original node names with aliases in the join condition
    left_node_name = link.node_revision.name
    right_node_name = link.dimension.name

    # Build a simple ON clause by parsing the join SQL
    # We'll create a binary comparison
    on_clause = parse(f"SELECT 1 WHERE {join_sql}").select.where

    # Now we need to rewrite column references to use our aliases
    def rewrite_column_refs(expr):
        """Recursively rewrite column references to use table aliases."""
        if isinstance(expr, ast.Column):
            if expr.name and expr.name.namespace:
                full_name = expr.identifier()
                if full_name.startswith(left_node_name + SEPARATOR):
                    col_name = full_name[len(left_node_name) + 1 :]
                    expr.name = ast.Name(col_name, namespace=ast.Name(left_alias))
                elif full_name.startswith(right_node_name + SEPARATOR):
                    col_name = full_name[len(right_node_name) + 1 :]
                    expr.name = ast.Name(col_name, namespace=ast.Name(right_alias))

        # Recurse into children
        for child in expr.children if hasattr(expr, "children") else []:
            if child:
                rewrite_column_refs(child)

    if on_clause:
        rewrite_column_refs(on_clause)

    # Determine join type (as string for ast.Join)
    from datajunction_server.models.dimensionlink import JoinType

    join_type_str = "LEFT OUTER"  # Default
    if link.join_type == JoinType.INNER:
        join_type_str = "INNER"
    elif link.join_type == JoinType.LEFT:
        join_type_str = "LEFT OUTER"
    elif link.join_type == JoinType.RIGHT:
        join_type_str = "RIGHT OUTER"
    elif link.join_type == JoinType.FULL:
        join_type_str = "FULL OUTER"

    # Build the right table reference
    right_table_parts = get_table_reference_parts(link.dimension)
    right_table_name = make_name(SEPARATOR.join(right_table_parts))

    # Create the join
    right_expr: ast.Expression = cast(
        ast.Expression,
        ast.Alias(
            child=ast.Table(name=right_table_name),
            alias=ast.Name(right_alias),
        ),
    )
    join = ast.Join(
        join_type=join_type_str,
        right=right_expr,
        criteria=ast.JoinCriteria(on=on_clause) if on_clause else None,
    )

    return join


def build_select_ast(
    ctx: BuildContext,
    metric_expressions: list[tuple[str, ast.Expression]],
    resolved_dimensions: list[ResolvedDimension],
    parent_node: Node,
    grain_columns: list[str] | None = None,
) -> ast.Query:
    """
    Build a SELECT AST for measures SQL with JOIN support.

    Args:
        ctx: Build context
        metric_expressions: List of (alias, expression AST) tuples
        resolved_dimensions: List of resolved dimension objects
        parent_node: The parent node (fact/transform)
        grain_columns: Optional list of columns required in GROUP BY for LIMITED
                       aggregability (e.g., ["customer_id"] for COUNT DISTINCT).
                       These are added to the output grain to enable re-aggregation.

    Returns:
        AST Query node
    """
    # Build projection (SELECT clause)
    # Use Any type to satisfy ast.Select.projection which accepts Union[Aliasable, Expression, Column]
    projection: list[Any] = []
    grain_columns = grain_columns or []

    # Generate alias for the main table
    main_alias = ctx.next_table_alias(parent_node.name)

    # Track which dimension nodes need joins and their aliases
    # Key by (node_name, role) to support multiple joins to same dimension with different roles
    dim_aliases: dict[tuple[str, Optional[str]], str] = {}  # (node_name, role) -> alias
    joins: list[ast.Join] = []

    # Process dimensions to build joins
    for resolved_dim in resolved_dimensions:
        if not resolved_dim.is_local and resolved_dim.join_path:
            # Need to add join(s) for this dimension
            current_left_alias = main_alias

            for link in resolved_dim.join_path.links:
                dim_node_name = link.dimension.name
                dim_key = (dim_node_name, resolved_dim.role)

                # Generate alias for dimension table if not already created
                # Key includes role to allow multiple joins to same dimension with different roles
                if dim_key not in dim_aliases:
                    # Use role as part of alias if present to distinguish multiple joins to same dim
                    if resolved_dim.role:
                        alias_base = resolved_dim.role.replace("->", "_")
                    else:
                        alias_base = dim_node_name.split(SEPARATOR)[-1]
                    dim_alias = ctx.next_table_alias(alias_base)
                    dim_aliases[dim_key] = dim_alias

                    # Build join clause
                    join = build_join_clause(ctx, link, current_left_alias, dim_alias)
                    joins.append(join)

                # For multi-hop, the next join's left is this dimension
                current_left_alias = dim_aliases[dim_key]

    # Add dimension columns to projection
    for resolved_dim in resolved_dimensions:
        # Determine table alias for this dimension's column
        if resolved_dim.is_local:
            table_alias = main_alias
        elif resolved_dim.join_path:
            # Use alias of the final dimension in the path, keyed by (name, role)
            final_dim_name = resolved_dim.join_path.target_node_name
            dim_key = (final_dim_name, resolved_dim.role)
            table_alias = dim_aliases.get(dim_key, main_alias)
        else:
            table_alias = main_alias

        # Build column reference with table alias
        col_ref = ast.Column(
            name=ast.Name(
                resolved_dim.column_name,
                namespace=ast.Name(table_alias),
            ),
        )

        # Register and apply clean alias
        clean_alias = ctx.alias_registry.register(resolved_dim.original_ref)
        if clean_alias != resolved_dim.column_name:
            col_ref.alias = ast.Name(clean_alias)

        projection.append(col_ref)

    # Add grain columns for LIMITED aggregability (e.g., customer_id for COUNT DISTINCT)
    # These are added to the output so the result can be re-aggregated
    grain_col_refs: list[ast.Column] = []
    for grain_col in grain_columns:
        col_ref = ast.Column(
            name=ast.Name(
                grain_col,
                namespace=ast.Name(main_alias),
            ),
        )
        grain_col_refs.append(col_ref)
        projection.append(col_ref)

    # Add metric expressions
    for alias_name, expr in metric_expressions:
        clean_alias = ctx.alias_registry.register(alias_name)

        # Rewrite column references in expression to use main table alias
        def add_table_prefix(e):
            if isinstance(e, ast.Column):
                if e.name and not (e.name.namespace and e.name.namespace.name):
                    # Add table alias to unqualified columns
                    e.name = ast.Name(e.name.name, namespace=ast.Name(main_alias))
            for child in e.children if hasattr(e, "children") else []:
                if child:
                    add_table_prefix(child)

        add_table_prefix(expr)

        # Clone expression and add alias
        aliased_expr = ast.Alias(
            alias=ast.Name(clean_alias),
            child=expr,
        )
        projection.append(aliased_expr)

    # Build GROUP BY (use same column references as projection, without aliases)
    group_by: list[ast.Expression] = []
    for resolved_dim in resolved_dimensions:
        if resolved_dim.is_local:
            table_alias = main_alias
        elif resolved_dim.join_path:
            final_dim_name = resolved_dim.join_path.target_node_name
            dim_key = (final_dim_name, resolved_dim.role)
            table_alias = dim_aliases.get(dim_key, main_alias)
        else:
            table_alias = main_alias

        group_by.append(
            ast.Column(
                name=ast.Name(
                    resolved_dim.column_name,
                    namespace=ast.Name(table_alias),
                ),
            ),
        )

    # Add grain columns to GROUP BY for LIMITED aggregability
    for grain_col in grain_columns:
        group_by.append(
            ast.Column(
                name=ast.Name(
                    grain_col,
                    namespace=ast.Name(main_alias),
                ),
            ),
        )

    # Collect all nodes that need CTEs and their needed columns
    nodes_for_ctes: list[Node] = []
    needed_columns_by_node: dict[str, set[str]] = {}

    # Collect columns needed from parent node
    parent_needed_cols: set[str] = set()

    # Add local dimension columns
    for resolved_dim in resolved_dimensions:
        if resolved_dim.is_local:
            parent_needed_cols.add(resolved_dim.column_name)

    # Add grain columns for LIMITED aggregability
    parent_needed_cols.update(grain_columns)

    # Add columns from metric expressions
    for _, expr in metric_expressions:
        parent_needed_cols.update(extract_columns_from_expression(expr))

    # Add join key columns (from the left side of joins)
    for resolved_dim in resolved_dimensions:
        if resolved_dim.join_path:
            for link in resolved_dim.join_path.links:
                # Extract the column name from the join_sql for the parent side
                # join_sql is like "v3.order_details.customer_id = v3.customer.customer_id"
                if link.join_sql:
                    join_cols = parse(f"SELECT 1 WHERE {link.join_sql}").select.where
                    if join_cols:
                        for col in join_cols.find_all(ast.Column):
                            col_id = col.identifier()
                            # Check if column is from parent node
                            if col_id.startswith(parent_node.name + SEPARATOR):
                                col_name = col_id.split(SEPARATOR)[-1]
                                parent_needed_cols.add(col_name)

    # Parent node needs CTE if it's not a source
    if parent_node.type != NodeType.SOURCE:
        nodes_for_ctes.append(parent_node)
        needed_columns_by_node[parent_node.name] = parent_needed_cols

    # Dimension nodes from joins need CTEs
    for resolved_dim in resolved_dimensions:
        if resolved_dim.join_path:
            for link in resolved_dim.join_path.links:
                dim_node = link.dimension
                if dim_node and dim_node.type != NodeType.SOURCE:
                    if dim_node not in nodes_for_ctes:
                        nodes_for_ctes.append(dim_node)

                    # Collect needed columns for this dimension
                    dim_cols: set[str] = set()

                    # Add the dimension column being selected
                    if resolved_dim.join_path.target_node_name == dim_node.name:
                        dim_cols.add(resolved_dim.column_name)

                    # Add join key columns from this dimension
                    if link.join_sql:
                        join_cols = parse(
                            f"SELECT 1 WHERE {link.join_sql}",
                        ).select.where
                        if join_cols:
                            for col in join_cols.find_all(ast.Column):
                                col_id = col.identifier()
                                if col_id.startswith(dim_node.name + SEPARATOR):
                                    col_name = col_id.split(SEPARATOR)[-1]
                                    dim_cols.add(col_name)

                    # Merge with existing if any
                    if dim_node.name in needed_columns_by_node:
                        needed_columns_by_node[dim_node.name].update(dim_cols)
                    else:
                        needed_columns_by_node[dim_node.name] = dim_cols

    # Build CTEs for all non-source nodes with column filtering
    ctes = collect_node_ctes(ctx, nodes_for_ctes, needed_columns_by_node)

    # Build FROM clause with main table
    table_parts = get_table_reference_parts(parent_node)
    table_name = make_name(SEPARATOR.join(table_parts))

    # Create relation with joins
    primary_expr: ast.Expression = cast(
        ast.Expression,
        ast.Alias(
            child=ast.Table(name=table_name),
            alias=ast.Name(main_alias),
        ),
    )
    relation = ast.Relation(
        primary=primary_expr,
        extensions=joins,
    )

    from_clause = ast.From(relations=[relation])

    # Build SELECT
    select = ast.Select(
        projection=projection,
        from_=from_clause,
        group_by=group_by if group_by else [],
    )

    # Build Query with CTEs
    query = ast.Query(select=select)

    # Add CTEs to the query
    if ctes:
        cte_list = []
        for cte_name, cte_query in ctes:
            # Convert the query to a CTE using to_cte method
            cte_query.to_cte(ast.Name(cte_name), query)
            cte_list.append(cte_query)
        query.ctes = cte_list

    return query


# =============================================================================
# Metric Grouping (Chunk 3)
# =============================================================================


@dataclass
class MetricGroup:
    """
    A group of metrics that share the same parent node.

    All metrics in a group can be computed in the same SELECT statement.
    Contains decomposed metric info with components and aggregability.
    """

    parent_node: Node
    decomposed_metrics: list[DecomposedMetricInfo]  # Decomposed metrics with components

    @property
    def overall_aggregability(self) -> Aggregability:
        """
        Get the worst-case aggregability across all metrics in this group.
        """
        if not self.decomposed_metrics:
            return Aggregability.NONE

        if any(m.aggregability == Aggregability.NONE for m in self.decomposed_metrics):
            return Aggregability.NONE
        if any(
            m.aggregability == Aggregability.LIMITED for m in self.decomposed_metrics
        ):
            return Aggregability.LIMITED
        return Aggregability.FULL

    def get_all_components(self) -> list[tuple[Node, MetricComponent]]:
        """Get all components with their source metric node."""
        result = []
        for decomposed in self.decomposed_metrics:
            for component in decomposed.components:
                result.append((decomposed.metric_node, component))
        return result


@dataclass
class GrainGroup:
    """
    A group of metric components that share the same effective grain.

    Components in the same grain group can be computed in a single SELECT
    with the same GROUP BY clause.

    Grain groups are determined by aggregability:
    - FULL: requested dimensions only
    - LIMITED: requested dimensions + level columns (e.g., customer_id for COUNT DISTINCT)
    - NONE: native grain (primary key of parent node)
    """

    parent_node: Node
    aggregability: Aggregability
    grain_columns: list[str]  # Columns to GROUP BY (beyond requested dimensions)
    components: list[tuple[Node, MetricComponent]]  # (metric_node, component) pairs

    @property
    def grain_key(self) -> tuple[str, Aggregability, tuple[str, ...]]:
        """
        Key for grouping: (parent_name, aggregability, sorted grain columns).
        """
        return (
            self.parent_node.name,
            self.aggregability,
            tuple(sorted(self.grain_columns)),
        )


def get_native_grain(node: Node) -> list[str]:
    """
    Get the native grain (primary key columns) of a node.

    For transforms/dimensions, this is their primary key columns.
    If no PK is defined, returns empty list (meaning row-level grain).
    """
    if not node.current:
        return []

    pk_columns = []
    for col in node.current.columns:
        # Check if this column is part of the primary key
        if col.has_primary_key_attribute:
            pk_columns.append(col.name)

    return pk_columns


def analyze_grain_groups(
    metric_group: MetricGroup,
    requested_dimensions: list[str],
) -> list[GrainGroup]:
    """
    Analyze a MetricGroup and split it into GrainGroups based on aggregability.

    Each GrainGroup contains components that can be computed at the same grain.

    Rules:
    - FULL aggregability: grain = requested dimensions
    - LIMITED aggregability: grain = requested dimensions + level columns
    - NONE aggregability: grain = native grain (PK of parent)

    Args:
        metric_group: MetricGroup with decomposed metrics
        requested_dimensions: Dimensions requested by user (column names only)

    Returns:
        List of GrainGroups, one per unique grain
    """
    parent_node = metric_group.parent_node

    # Group components by their effective grain
    # Key: (aggregability, tuple of additional grain columns)
    grain_buckets: dict[
        tuple[Aggregability, tuple[str, ...]],
        list[tuple[Node, MetricComponent]],
    ] = {}

    for metric_node, component in metric_group.get_all_components():
        agg_type = component.rule.type

        # Explicitly type the key to satisfy mypy
        key: tuple[Aggregability, tuple[str, ...]]
        if agg_type == Aggregability.FULL:
            # FULL: no additional grain columns needed
            key = (Aggregability.FULL, ())
        elif agg_type == Aggregability.LIMITED:
            # LIMITED: add level columns to grain
            level_cols = tuple(sorted(component.rule.level or []))
            key = (Aggregability.LIMITED, level_cols)
        else:  # NONE
            # NONE: use native grain (PK columns)
            native_grain = get_native_grain(parent_node)
            key = (Aggregability.NONE, tuple(sorted(native_grain)))

        if key not in grain_buckets:
            grain_buckets[key] = []
        grain_buckets[key].append((metric_node, component))

    # Convert buckets to GrainGroup objects
    grain_groups = []
    for (agg_type, grain_cols), components in grain_buckets.items():
        grain_groups.append(
            GrainGroup(
                parent_node=parent_node,
                aggregability=agg_type,
                grain_columns=list(grain_cols),
                components=components,
            ),
        )

    # Sort groups: FULL first, then LIMITED, then NONE (for consistent output)
    agg_order = {Aggregability.FULL: 0, Aggregability.LIMITED: 1, Aggregability.NONE: 2}
    grain_groups.sort(
        key=lambda g: (agg_order.get(g.aggregability, 3), g.grain_columns),
    )

    return grain_groups


def get_fact_parent(ctx: BuildContext, metric_node: Node) -> Node:
    """
    Get the fact/transform parent of a metric, traversing through derived metrics.

    For base metrics: returns the direct parent (a fact/transform)
    For derived metrics: returns None (caller should get base metrics first)
    """
    parent_names = ctx.parent_map.get(metric_node.name, [])
    if not parent_names:
        raise DJInvalidInputException(f"Metric {metric_node.name} has no parent node")

    # Check if first parent is a metric (derived) or fact/transform (base)
    first_parent_name = parent_names[0]
    first_parent = ctx.nodes.get(first_parent_name)

    if not first_parent:
        raise DJInvalidInputException(f"Parent node not found: {first_parent_name}")

    return first_parent


def get_base_metrics_for_derived(ctx: BuildContext, metric_node: Node) -> list[Node]:
    """
    For a derived metric, get all the base metrics it depends on.

    Returns list of base metric nodes (metrics that SELECT FROM a fact/transform, not other metrics).
    """
    base_metrics = []
    visited = set()

    def collect_bases(node: Node):
        if node.name in visited:
            return
        visited.add(node.name)

        parent_names = ctx.parent_map.get(node.name, [])
        for parent_name in parent_names:
            parent = ctx.nodes.get(parent_name)
            if not parent:
                continue

            if parent.type == NodeType.METRIC:
                # Parent is also a metric - recurse
                collect_bases(parent)
            else:
                # Parent is a fact/transform - this is a base metric
                base_metrics.append(node)
                break  # Found the base, don't check other parents

    collect_bases(metric_node)
    return base_metrics


def is_derived_metric(ctx: BuildContext, metric_node: Node) -> bool:
    """Check if a metric is derived (references other metrics) vs base (references fact/transform)."""
    parent_names = ctx.parent_map.get(metric_node.name, [])
    if not parent_names:
        return False

    first_parent = ctx.nodes.get(parent_names[0])
    return first_parent is not None and first_parent.type == NodeType.METRIC


async def decompose_and_group_metrics(
    ctx: BuildContext,
) -> list[MetricGroup]:
    """
    Decompose metrics and group them by parent node (fact/transform).

    For base metrics: groups by direct parent
    For derived metrics: decomposes into base metrics and groups by their parents

    This enables cross-fact derived metrics by producing separate grain groups
    for each underlying fact.

    Returns:
        List of MetricGroup, one per unique parent node, with decomposed metrics.
    """
    # Map parent node name -> list of DecomposedMetricInfo
    parent_groups: dict[str, list[DecomposedMetricInfo]] = {}
    parent_nodes: dict[str, Node] = {}

    for metric_name in ctx.metrics:
        metric_node = get_metric_node(ctx, metric_name)

        if is_derived_metric(ctx, metric_node):
            # Derived metric - get base metrics and decompose each
            base_metric_nodes = get_base_metrics_for_derived(ctx, metric_node)

            for base_metric in base_metric_nodes:
                # Get the fact/transform parent of the base metric
                parent_node = get_parent_node(ctx, base_metric)

                # Decompose the BASE metric (not the derived one)
                decomposed = await decompose_metric(ctx.session, base_metric)

                parent_name = parent_node.name
                if parent_name not in parent_groups:
                    parent_groups[parent_name] = []
                    parent_nodes[parent_name] = parent_node

                parent_groups[parent_name].append(decomposed)
        else:
            # Base metric - use direct parent
            parent_node = get_parent_node(ctx, metric_node)
            decomposed = await decompose_metric(ctx.session, metric_node)

            parent_name = parent_node.name
            if parent_name not in parent_groups:
                parent_groups[parent_name] = []
                parent_nodes[parent_name] = parent_node

            parent_groups[parent_name].append(decomposed)

    # Build MetricGroup objects
    return [
        MetricGroup(parent_node=parent_nodes[name], decomposed_metrics=metrics)
        for name, metrics in parent_groups.items()
    ]


# =============================================================================
# Main Entry Points
# =============================================================================


def build_grain_group_sql(
    ctx: BuildContext,
    grain_group: GrainGroup,
    resolved_dimensions: list[ResolvedDimension],
    components_per_metric: dict[str, int],
) -> GrainGroupSQL:
    """
    Build SQL for a single grain group.

    Args:
        ctx: Build context
        grain_group: The grain group to generate SQL for
        resolved_dimensions: Pre-resolved dimensions with join paths
        components_per_metric: Metric name -> component count mapping

    Returns:
        GrainGroupSQL with SQL and metadata for this grain group
    """
    parent_node = grain_group.parent_node

    # Build list of component expressions with their aliases
    component_expressions: list[tuple[str, ast.Expression]] = []
    component_metadata: list[tuple[str, MetricComponent, Node, bool]] = []

    # Track which metrics are covered by this grain group
    metrics_covered: set[str] = set()

    # Track which components we've already added (deduplicate by component name)
    seen_components: set[str] = set()

    for metric_node, component in grain_group.components:
        metrics_covered.add(metric_node.name)

        # Deduplicate components - same component may appear for multiple derived metrics
        if component.name in seen_components:
            continue
        seen_components.add(component.name)

        # For NONE aggregability, we output raw columns, not aggregations
        if grain_group.aggregability == Aggregability.NONE:
            # Just output the column reference, no aggregation
            # The actual aggregation (MEDIAN, etc.) happens in metrics SQL
            if component.column:
                col_ast = ast.Column(
                    name=ast.Name(component.column),
                    _table=None,
                )
                component_alias = component.column
                component_expressions.append((component_alias, col_ast))
                component_metadata.append(
                    (component_alias, component, metric_node, False),
                )
            continue

        # Skip LIMITED aggregability components with no aggregation
        # These are represented by grain columns instead
        if component.rule.type == Aggregability.LIMITED and not component.aggregation:
            continue

        num_components = components_per_metric.get(metric_node.name, 1)
        is_simple = num_components == 1

        if is_simple:
            component_alias = metric_node.name.split(SEPARATOR)[-1]
        else:
            component_alias = component.name

        expr_ast = build_component_expression(component)
        component_expressions.append((component_alias, expr_ast))
        component_metadata.append((component_alias, component, metric_node, is_simple))

    # Determine grain columns for this group
    if grain_group.aggregability == Aggregability.NONE:
        # NONE: use native grain (PK columns)
        effective_grain_columns = grain_group.grain_columns
    elif grain_group.aggregability == Aggregability.LIMITED:
        # LIMITED: use level columns from components
        effective_grain_columns = grain_group.grain_columns
    else:
        # FULL: no additional grain columns
        effective_grain_columns = []

    # Build AST
    query_ast = build_select_ast(
        ctx,
        metric_expressions=component_expressions,
        resolved_dimensions=resolved_dimensions,
        parent_node=parent_node,
        grain_columns=effective_grain_columns,
    )

    # Build column metadata
    columns_metadata = []

    # Add dimension columns
    for resolved_dim in resolved_dimensions:
        alias = (
            ctx.alias_registry.get_alias(resolved_dim.original_ref)
            or resolved_dim.column_name
        )
        if resolved_dim.is_local:
            col_type = get_column_type(parent_node, resolved_dim.column_name)
        else:
            dim_node = ctx.nodes.get(resolved_dim.node_name)
            col_type = (
                get_column_type(dim_node, resolved_dim.column_name)
                if dim_node
                else "string"
            )
        columns_metadata.append(
            ColumnMetadata(
                name=alias,
                semantic_name=resolved_dim.original_ref,
                type=col_type,
                semantic_type="dimension",
            ),
        )

    # Add grain columns (for LIMITED and NONE)
    for grain_col in effective_grain_columns:
        col_type = get_column_type(parent_node, grain_col)
        columns_metadata.append(
            ColumnMetadata(
                name=grain_col,
                semantic_name=f"{parent_node.name}{SEPARATOR}{grain_col}",
                type=col_type,
                semantic_type="dimension",  # Added for aggregability (e.g., customer_id for COUNT DISTINCT)
            ),
        )

    # Add metric component columns
    for comp_alias, component, metric_node, is_simple in component_metadata:
        if grain_group.aggregability == Aggregability.NONE:
            # NONE: raw column, will be aggregated in metrics SQL
            columns_metadata.append(
                ColumnMetadata(
                    name=comp_alias,
                    semantic_name=f"{metric_node.name}:{component.column}",
                    type="number",
                    semantic_type="metric_input",  # Raw input for non-aggregatable metric
                ),
            )
        elif is_simple:
            columns_metadata.append(
                ColumnMetadata(
                    name=ctx.alias_registry.get_alias(comp_alias) or comp_alias,
                    semantic_name=metric_node.name,
                    type="number",
                    semantic_type="metric",
                ),
            )
        else:
            columns_metadata.append(
                ColumnMetadata(
                    name=ctx.alias_registry.get_alias(comp_alias) or comp_alias,
                    semantic_name=f"{metric_node.name}:{component.name}",
                    type="number",
                    semantic_type="metric_component",
                ),
            )

    # Build the full grain list (GROUP BY columns)
    # Start with dimension column aliases
    full_grain = []
    for resolved_dim in resolved_dimensions:
        alias = (
            ctx.alias_registry.get_alias(resolved_dim.original_ref)
            or resolved_dim.column_name
        )
        full_grain.append(alias)

    # Add any additional grain columns (from LIMITED/NONE aggregability)
    for grain_col in effective_grain_columns:
        if grain_col not in full_grain:
            full_grain.append(grain_col)

    # Sort for deterministic output
    full_grain.sort()

    return GrainGroupSQL(
        query=query_ast,
        columns=columns_metadata,
        grain=full_grain,
        aggregability=grain_group.aggregability,
        metrics=list(metrics_covered),
    )


async def build_measures_sql(
    session: AsyncSession,
    metrics: list[str],
    dimensions: list[str],
    filters: list[str] | None = None,
    dialect: Dialect = Dialect.SPARK,
) -> GeneratedMeasuresSQL:
    """
    Build measures SQL for a set of metrics and dimensions.

    This is the main entry point for V3 measures SQL generation.

    Measures SQL aggregates metric components to the requested dimensional
    grain, producing one SQL query per grain group. Different aggregability
    levels (FULL, LIMITED, NONE) result in different grain groups.

    Use cases:
    - Materialization: Each grain group can be materialized separately
    - Live queries: Pass to build_metrics_sql() to get a single combined query

    Args:
        session: Database session
        metrics: List of metric node names
        dimensions: List of dimension names (format: "node.column" or "node.column[role]")
        filters: Optional list of filter expressions
        dialect: SQL dialect for output

    Returns:
        GeneratedMeasuresSQL with one GrainGroupSQL per aggregation level
    """
    # Create context
    ctx = BuildContext(
        session=session,
        metrics=metrics,
        dimensions=dimensions,
        filters=filters or [],
        dialect=dialect,
    )

    # Load all required nodes
    await load_nodes(ctx)

    # Validate we have at least one metric
    if not ctx.metrics:
        raise DJInvalidInputException("At least one metric is required")

    # Decompose metrics and group by parent node
    metric_groups = await decompose_and_group_metrics(ctx)

    # Process each metric group into grain group SQLs
    # Cross-fact metrics produce separate grain groups (one per parent node)
    # Currently we only support single metric group; cross-fact will iterate over all
    all_grain_group_sqls: list[GrainGroupSQL] = []
    for metric_group in metric_groups:
        grain_group_sqls = process_metric_group(ctx, metric_group)
        all_grain_group_sqls.extend(grain_group_sqls)

    return GeneratedMeasuresSQL(
        grain_groups=all_grain_group_sqls,
        dialect=dialect,
        requested_dimensions=dimensions,
    )


def process_metric_group(
    ctx: BuildContext,
    metric_group: MetricGroup,
) -> list[GrainGroupSQL]:
    """
    Process a single MetricGroup into one or more GrainGroupSQLs.

    This handles:
    1. Counting components per metric for naming strategy
    2. Analyzing grain groups by aggregability
    3. Resolving dimension join paths
    4. Building SQL for each grain group

    Args:
        ctx: Build context
        metric_group: The metric group to process

    Returns:
        List of GrainGroupSQL, one per aggregability level
    """
    parent_node = metric_group.parent_node

    # Count components per metric to determine naming strategy
    components_per_metric: dict[str, int] = {}
    for decomposed in metric_group.decomposed_metrics:
        components_per_metric[decomposed.metric_node.name] = len(decomposed.components)

    # Analyze grain groups - split by aggregability
    # Extract just the column names from dimensions for grain analysis
    dim_column_names = [parse_dimension_ref(d).column_name for d in ctx.dimensions]
    grain_groups = analyze_grain_groups(metric_group, dim_column_names)

    # Resolve dimensions (find join paths) - shared across grain groups
    resolved_dimensions = resolve_dimensions(ctx, parent_node)

    # Build SQL for each grain group
    grain_group_sqls: list[GrainGroupSQL] = []
    for grain_group in grain_groups:
        # Reset alias registry for each grain group to avoid conflicts
        ctx.alias_registry = AliasRegistry()
        ctx._table_alias_counter = 0

        grain_group_sql = build_grain_group_sql(
            ctx,
            grain_group,
            resolved_dimensions,
            components_per_metric,
        )
        grain_group_sqls.append(grain_group_sql)

    return grain_group_sqls


# Placeholder for metrics SQL (Chunk 5)
async def build_metrics_sql(
    session: AsyncSession,
    metrics: list[str],
    dimensions: list[str],
    filters: list[str] | None = None,
    dialect: Dialect = Dialect.SPARK,
) -> GeneratedSQL:
    """
    Build metrics SQL for a set of metrics and dimensions.

    This is the main entry point for V3 metrics SQL generation.

    Metrics SQL applies final metric expressions on top of measures,
    including handling derived metrics. It produces a single executable
    query that:
    1. Uses measures SQL output as CTEs (one per grain group)
    2. JOINs grain groups if metrics come from different facts/aggregabilities
    3. Applies combiner expressions for multi-component metrics
    4. Computes derived metrics that reference other metrics

    Architecture:
    - Layer 0 (Measures): Grain group CTEs from build_measures_sql()
    - Layer 1 (Base Metrics): Combiner expressions applied
    - Layer 2+ (Derived Metrics): Metrics referencing other metrics
    """
    # Step 1: Get measures SQL with grain groups
    measures_result = await build_measures_sql(
        session=session,
        metrics=metrics,
        dimensions=dimensions,
        filters=filters,
        dialect=dialect,
    )

    if not measures_result.grain_groups:
        raise DJInvalidInputException("No grain groups produced from measures SQL")

    # Step 2: Build context for metrics processing
    ctx = BuildContext(
        session=session,
        metrics=metrics,
        dimensions=dimensions,
        filters=filters or [],
        dialect=dialect,
    )

    # Load nodes for combiner expression extraction
    await load_nodes(ctx)

    # Step 3: Decompose all metrics to get combiner expressions
    # This includes both requested metrics AND base metrics from grain groups
    decomposed_metrics: dict[str, DecomposedMetricInfo] = {}

    # First decompose requested metrics
    for metric_name in metrics:
        metric_node = ctx.nodes.get(metric_name)
        if metric_node:
            decomposed = await decompose_metric(session, metric_node)
            decomposed_metrics[metric_name] = decomposed

    # Also decompose base metrics from grain groups (needed for component mapping)
    all_grain_group_metrics = set()
    for gg in measures_result.grain_groups:
        all_grain_group_metrics.update(gg.metrics)

    for metric_name in all_grain_group_metrics:
        if metric_name not in decomposed_metrics:
            metric_node = ctx.nodes.get(metric_name)
            if metric_node:
                decomposed = await decompose_metric(session, metric_node)
                decomposed_metrics[metric_name] = decomposed

    # Step 4: Build metric dependency DAG and compute layers
    metric_layers = compute_metric_layers(ctx, decomposed_metrics)

    # Step 5: Generate the combined SQL (returns GeneratedSQL with AST query)
    return generate_metrics_sql(
        ctx,
        measures_result,
        decomposed_metrics,
        metric_layers,
    )


def compute_metric_layers(
    ctx: BuildContext,
    decomposed_metrics: dict[str, DecomposedMetricInfo],
) -> list[list[str]]:
    """
    Compute the order in which metrics should be evaluated.

    Returns a list of layers, where each layer contains metric names
    that can be computed in parallel (no dependencies on each other).

    Layer 0: Base metrics (no metric dependencies)
    Layer 1+: Derived metrics (depend on metrics in previous layers)
    """
    # Build dependency graph
    # A metric depends on another if its parent node is that metric
    dependencies: dict[str, set[str]] = {name: set() for name in decomposed_metrics}

    for metric_name in decomposed_metrics:
        # Use parent_map from context instead of accessing lazy-loaded relationships
        parent_names = ctx.parent_map.get(metric_name, [])
        for parent_name in parent_names:
            parent_node = ctx.nodes.get(parent_name)
            if (
                parent_node
                and parent_node.type == NodeType.METRIC
                and parent_name in decomposed_metrics
            ):
                dependencies[metric_name].add(parent_name)

    # Topological sort into layers
    layers: list[list[str]] = []
    computed: set[str] = set()

    while len(computed) < len(decomposed_metrics):
        # Find metrics whose dependencies are all computed
        layer = [
            name
            for name, deps in dependencies.items()
            if name not in computed and deps <= computed
        ]

        if not layer:
            # Circular dependency - shouldn't happen
            remaining = set(decomposed_metrics.keys()) - computed
            raise DJInvalidInputException(
                f"Circular dependency detected in metrics: {remaining}",
            )

        layers.append(sorted(layer))  # Sort for deterministic output
        computed.update(layer)

    return layers


def generate_metrics_sql(
    ctx: BuildContext,
    measures_result: GeneratedMeasuresSQL,
    decomposed_metrics: dict[str, DecomposedMetricInfo],
    metric_layers: list[list[str]],
) -> GeneratedSQL:
    """
    Generate the final metrics SQL query.

    This combines grain groups from measures SQL and applies
    combiner expressions for each metric. Works for both single
    and multiple grain groups (FULL OUTER JOINs them together).

    Works entirely with AST objects - no string parsing needed.
    Returns a GeneratedSQL with the query as an AST.
    """
    grain_groups = measures_result.grain_groups
    dimensions = measures_result.requested_dimensions

    # For cross-fact or cross-aggregability queries, we need to:
    # 1. Extract CTEs from each grain group and flatten them (with prefixes)
    # 2. Create a final CTE for each grain group's main SELECT
    # 3. FULL OUTER JOIN them on the common dimensions
    # 4. Apply combiner expressions in the final SELECT

    all_cte_asts: list[ast.Query] = []
    cte_aliases: list[str] = []

    for i, gg in enumerate(grain_groups):
        alias = f"gg{i}"
        cte_aliases.append(alias)

        # gg.query is already an AST - no need to parse!
        gg_query = gg.query

        if gg_query.ctes:
            # Extract existing CTEs with prefixed names to avoid collisions
            for inner_cte in gg_query.ctes:
                cte_name = str(inner_cte.alias) if inner_cte.alias else "unnamed_cte"
                prefixed_name = f"{alias}_{cte_name}"

                # Clone the CTE and update its alias to the prefixed name
                prefixed_cte = deepcopy(inner_cte)
                prefixed_cte.alias = ast.Name(prefixed_name)
                all_cte_asts.append(prefixed_cte)

            # Build the grain group CTE (main SELECT with updated table refs)
            # Clone the query and clear its CTEs
            gg_main = deepcopy(gg_query)
            gg_main.ctes = []

            # Update table references in the main SELECT to use prefixed CTE names
            for inner_cte in gg_query.ctes:
                original_name = (
                    str(inner_cte.alias) if inner_cte.alias else "unnamed_cte"
                )
                prefixed_name = f"{alias}_{original_name}"
                # Find and update table references
                for table in gg_main.find_all(ast.Table):
                    if hasattr(table, "name") and str(table.name) == original_name:
                        table.name = ast.Name(prefixed_name)

            # Convert to CTE with the grain group alias
            gg_main.to_cte(ast.Name(alias), None)
            all_cte_asts.append(gg_main)
        else:
            # No inner CTEs - just convert the query to a CTE directly
            gg_cte = deepcopy(gg_query)
            gg_cte.to_cte(ast.Name(alias), None)
            all_cte_asts.append(gg_cte)

    # Build projection as AST expressions
    # Use Any type since projection accepts Union[Aliasable, Expression, Column]
    projection: list[Any] = []
    columns_metadata: list[ColumnMetadata] = []

    # Parse dimensions to get column names and preserve mapping to original refs
    dim_info: list[tuple[str, str]] = []  # (original_dim_ref, col_alias)
    for dim in dimensions:
        # Generate a consistent alias for this dimension
        # Using register() ensures we get a proper alias (with role suffix if applicable)
        col_name = ctx.alias_registry.register(dim)
        dim_info.append((dim, col_name))

    # Add dimension columns using COALESCE across all grain groups
    for original_dim_ref, dim_col in dim_info:
        # Build COALESCE(gg0.col, gg1.col, ...) AS col
        coalesce_args: list[ast.Expression] = [
            ast.Column(name=ast.Name(dim_col), _table=ast.Table(ast.Name(alias)))
            for alias in cte_aliases
        ]
        coalesce_func = ast.Function(ast.Name("COALESCE"), args=coalesce_args)
        aliased_coalesce = coalesce_func.set_alias(ast.Name(dim_col))
        aliased_coalesce.set_as(True)  # Include "AS" in output
        projection.append(aliased_coalesce)

        columns_metadata.append(
            ColumnMetadata(
                name=dim_col,
                semantic_name=original_dim_ref,  # Preserve original dimension reference
                type="string",
                semantic_type="dimension",
            ),
        )

    # Build maps for resolving references in derived metric expressions:
    # 1. base_metric_columns: metric_name -> (gg_alias, output_col_name)
    # 2. component_columns: component_name -> (gg_alias, actual_col_name)
    # 3. base_metric_exprs: metric_name -> SQL expression string (for use in derived metrics)
    base_metric_columns: dict[str, tuple[str, str]] = {}
    component_columns: dict[str, tuple[str, str]] = {}
    base_metric_exprs: dict[str, str] = {}  # For building derived metric expressions

    # Determine which metrics were explicitly requested by the user
    requested_metrics_set = set(ctx.metrics)

    # Process base metric columns from each grain group
    for i, gg in enumerate(grain_groups):
        alias = cte_aliases[i]
        for metric_name in gg.metrics:
            decomposed = decomposed_metrics.get(metric_name)
            short_name = metric_name.split(SEPARATOR)[-1]

            # Determine if this base metric was explicitly requested
            # (not just needed for a derived metric)
            is_explicitly_requested = metric_name in requested_metrics_set

            if decomposed and len(decomposed.components) > 1:
                # Multi-component - apply combiner with table alias prefix
                combiner_expr = decomposed.combiner
                # Prefix component references with table alias
                for comp in decomposed.components:
                    combiner_expr = combiner_expr.replace(
                        comp.name,
                        f"{alias}.{comp.name}",
                    )
                    # Track component for derived metric resolution
                    component_columns[comp.name] = (alias, comp.name)

                # Store expression for derived metrics
                base_metric_exprs[metric_name] = combiner_expr

                # Only add to projection if explicitly requested
                if is_explicitly_requested:
                    # Parse the expression and add alias
                    expr_ast = parse(f"SELECT {combiner_expr}").select.projection[0]
                    aliased_expr = expr_ast.set_alias(ast.Name(short_name))
                    aliased_expr.set_as(True)
                    projection.append(aliased_expr)
                    columns_metadata.append(
                        ColumnMetadata(
                            name=short_name,
                            semantic_name=metric_name,
                            type="number",
                            semantic_type="metric",
                        ),
                    )
            else:
                # Single component - find the actual column name in the grain group
                # For LIMITED aggregability, the column is the grain column (e.g., order_id)
                if decomposed and decomposed.components:
                    comp = decomposed.components[0]
                    # For LIMITED aggregability, use the component's expression (grain column)
                    if decomposed.aggregability == Aggregability.LIMITED:
                        actual_col = comp.expression  # e.g., "order_id"
                        # The expression for derived metrics includes the aggregation
                        agg_expr = decomposed.combiner.replace(
                            comp.name,
                            f"{alias}.{actual_col}",
                        )
                        base_metric_exprs[metric_name] = agg_expr
                    else:
                        # For FULL, use the column from metadata
                        actual_col = next(
                            (
                                c.name
                                for c in gg.columns
                                if c.semantic_name == metric_name
                            ),
                            short_name,
                        )
                        base_metric_exprs[metric_name] = f"{alias}.{actual_col}"

                    # Track component for derived metric resolution
                    component_columns[comp.name] = (alias, actual_col)

                    # Only add to projection if explicitly requested
                    if is_explicitly_requested:
                        if decomposed.aggregability == Aggregability.LIMITED:
                            expr_ast = parse(f"SELECT {agg_expr}").select.projection[0]
                        else:
                            expr_ast = ast.Column(
                                name=ast.Name(actual_col),
                                _table=ast.Table(ast.Name(alias)),
                            )
                        aliased_expr = expr_ast.set_alias(ast.Name(short_name))
                        aliased_expr.set_as(True)
                        projection.append(aliased_expr)
                        columns_metadata.append(
                            ColumnMetadata(
                                name=short_name,
                                semantic_name=metric_name,
                                type="number",
                                semantic_type="metric",
                            ),
                        )
                else:
                    col_name = next(
                        (c.name for c in gg.columns if c.semantic_name == metric_name),
                        short_name,
                    )
                    base_metric_exprs[metric_name] = f"{alias}.{col_name}"

                    # Only add to projection if explicitly requested
                    if is_explicitly_requested:
                        expr_ast = ast.Column(
                            name=ast.Name(col_name),
                            _table=ast.Table(ast.Name(alias)),
                        )
                        aliased_expr = expr_ast.set_alias(ast.Name(short_name))
                        aliased_expr.set_as(True)
                        projection.append(aliased_expr)
                        columns_metadata.append(
                            ColumnMetadata(
                                name=short_name,
                                semantic_name=metric_name,
                                type="number",
                                semantic_type="metric",
                            ),
                        )

            # Track for derived metric resolution
            base_metric_columns[metric_name] = (alias, short_name)

    # Now handle derived metrics that reference the base metrics
    # These are in ctx.metrics but not in any grain group's metrics
    all_grain_group_metrics = set()
    for gg in grain_groups:
        all_grain_group_metrics.update(gg.metrics)

    for metric_name in ctx.metrics:
        if metric_name in all_grain_group_metrics:
            # Already handled as a base metric
            continue

        decomposed = decomposed_metrics.get(metric_name)
        if not decomposed:
            continue

        # This is a derived metric - get its combiner expression
        short_name = metric_name.split(SEPARATOR)[-1]
        combiner_expr = decomposed.combiner

        # Replace component name references with qualified column references
        # The combiner uses component names (e.g., order_id_distinct_f93d50ab)
        for comp_name, (gg_alias, col_name) in component_columns.items():
            combiner_expr = combiner_expr.replace(comp_name, f"{gg_alias}.{col_name}")

        # Parse the expression and add alias
        expr_ast = parse(f"SELECT {combiner_expr}").select.projection[0]
        aliased_expr = expr_ast.set_alias(ast.Name(short_name))
        aliased_expr.set_as(True)
        projection.append(aliased_expr)
        columns_metadata.append(
            ColumnMetadata(
                name=short_name,
                semantic_name=metric_name,
                type="number",
                semantic_type="metric",
            ),
        )

    # Build FROM clause with JOINs as AST
    dim_col_aliases = [col_alias for _, col_alias in dim_info]

    # Build JOIN extensions for the Relation
    join_extensions: list[ast.Join] = []
    for i in range(1, len(cte_aliases)):
        # Build join condition: gg0.dim1 = ggN.dim1 AND gg0.dim2 = ggN.dim2 ...
        conditions: list[ast.Expression] = []
        for dim_col in dim_col_aliases:
            left_col = ast.Column(
                name=ast.Name(dim_col),
                _table=ast.Table(ast.Name(cte_aliases[0])),
            )
            right_col = ast.Column(
                name=ast.Name(dim_col),
                _table=ast.Table(ast.Name(cte_aliases[i])),
            )
            conditions.append(ast.BinaryOp.Eq(left_col, right_col))

        # Combine conditions with AND
        if len(conditions) == 1:
            on_clause = conditions[0]
        else:
            on_clause = reduce(lambda a, b: ast.BinaryOp.And(a, b), conditions)

        # Build the JOIN with criteria
        join_extensions.append(
            ast.Join(
                right=ast.Table(ast.Name(cte_aliases[i])),
                criteria=ast.JoinCriteria(on=on_clause),
                join_type="FULL OUTER",
            ),
        )

    # Build the FROM clause as a Relation with primary table and join extensions
    from_clause = ast.From(
        relations=[
            ast.Relation(
                primary=ast.Table(ast.Name(cte_aliases[0])),
                extensions=join_extensions,
            ),
        ],
    )

    # Build the final SELECT
    select_ast = ast.Select(projection=projection, from_=from_clause)

    # Build the final Query with all CTEs
    final_query = ast.Query(select=select_ast, ctes=all_cte_asts)

    return GeneratedSQL(
        query=final_query,
        columns=columns_metadata,
        dialect=measures_result.dialect,
    )
