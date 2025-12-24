"""
Build V3: SQL Generation

This module implements the V3 SQL generation system for DataJunction.
It generates both measures SQL (pre-aggregated to dimensional grain) and
metrics SQL (with final metric expressions applied).

See ARCHITECTURE.md for design documentation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

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

logger = logging.getLogger(__name__)


# =============================================================================
# Context
# =============================================================================


@dataclass
class BuildContext:
    """
    Immutable context passed through the SQL generation pipeline.

    Contains all the information needed to build SQL for a set of metrics
    and dimensions.
    """

    session: AsyncSession
    metrics: list[str]
    dimensions: list[str]
    filters: list[str] = field(default_factory=list)
    dialect: Dialect = Dialect.SPARK
    alias_registry: AliasRegistry = field(default_factory=AliasRegistry)

    # Loaded data (populated by load_nodes)
    nodes: dict[str, Node] = field(default_factory=dict)

    # Preloaded join paths: (source_revision_id, dim_name, role) -> list[DimensionLink]
    # Populated by load_nodes() using a single recursive CTE query
    join_paths: dict[tuple[int, str, str], list[DimensionLink]] = field(
        default_factory=dict,
    )

    # Table alias counter for generating unique aliases
    _table_alias_counter: int = field(default=0)

    def next_table_alias(self, base_name: str) -> str:
        """Generate a unique table alias."""
        self._table_alias_counter += 1
        # Use short alias like t1, t2, etc.
        return f"t{self._table_alias_counter}"


@dataclass
class GeneratedSQL:
    """Output of the SQL generation pipeline."""

    sql: str
    columns: list[ColumnMetadata]
    dialect: Dialect


@dataclass
class ColumnMetadata:
    """
    Metadata about a column in the generated SQL.

    This is V3's simplified column metadata focused on what's actually useful:
    - Identifying the output column name
    - Linking back to the semantic entity (node.column for dims, node for metrics)
    - Distinguishing column types via semantic_type
    """

    name: str  # SQL alias in output (clean name)
    semantic_name: (
        str  # Full semantic path (e.g., 'v3.customer.name' or 'v3.total_revenue')
    )
    type: str  # SQL type (string, number, etc.)
    semantic_type: str  # "dimension", "metric", or future: "metric_component"


@dataclass
class JoinPath:
    """
    Represents a path from a fact/transform to a dimension via dimension links.
    """

    links: list[DimensionLink]  # Ordered list of links to traverse
    target_dimension: Node  # The final dimension node
    role: Optional[str] = (
        None  # Role qualifier if specified (e.g., "from", "to", "customer->home")
    )

    @property
    def target_node_name(self) -> str:
        return self.target_dimension.name


@dataclass
class ResolvedDimension:
    """
    A dimension that has been resolved to its join path.
    """

    original_ref: str  # Original reference (e.g., "v3.customer.name[order]")
    node_name: str  # Dimension node name (e.g., "v3.customer")
    column_name: str  # Column name (e.g., "name")
    role: Optional[str]  # Role if specified (e.g., "order")
    join_path: Optional[
        JoinPath
    ]  # Join path from fact to this dimension (None if local)
    is_local: bool  # True if dimension is on the fact table itself


@dataclass
class DecomposedMetricInfo:
    """
    Information about a decomposed metric.

    Contains the metric's components (for measures SQL) and aggregability info.
    """

    metric_node: Node
    components: list[MetricComponent]  # The decomposed components
    aggregability: Aggregability  # Overall aggregability (FULL, LIMITED, NONE)

    @property
    def is_fully_decomposable(self) -> bool:
        """True if all components have FULL aggregability."""
        return all(c.rule.type == Aggregability.FULL for c in self.components)


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
) -> set[str]:
    """
    Find all upstream node names using a lightweight recursive CTE.

    This returns just node names (not full nodes) to minimize query overhead.
    Uses NodeRelationship to traverse the parent-child relationships.

    Returns:
        Set of all upstream node names (including the starting nodes).
    """
    if not starting_node_names:
        return set()

    # Lightweight recursive CTE - only returns node names
    recursive_query = text("""
        WITH RECURSIVE upstream AS (
            -- Base case: get the starting nodes' current revision IDs
            SELECT
                nr.id as revision_id,
                n.name as node_name
            FROM node n
            JOIN noderevision nr ON n.id = nr.node_id AND n.current_version = nr.version
            WHERE n.name IN :starting_names
            AND n.deactivated_at IS NULL

            UNION

            -- Recursive case: find parents of current nodes
            SELECT
                parent_nr.id as revision_id,
                parent_n.name as node_name
            FROM upstream u
            JOIN noderelationship nrel ON u.revision_id = nrel.child_id
            JOIN node parent_n ON nrel.parent_id = parent_n.id
            JOIN noderevision parent_nr ON parent_n.id = parent_nr.node_id
                AND parent_n.current_version = parent_nr.version
            WHERE parent_n.deactivated_at IS NULL
        )
        SELECT DISTINCT node_name FROM upstream
    """).bindparams(bindparam("starting_names", expanding=True))

    result = await session.execute(
        recursive_query,
        {"starting_names": list(starting_node_names)},
    )
    return {row[0] for row in result.fetchall()}


def _dimension_link_eager_load():
    """Common eager loading options for dimension links and their dimensions."""
    return selectinload(NodeRevision.dimension_links).options(
        joinedload(DimensionLink.dimension).options(
            selectinload(Node.current).options(
                selectinload(NodeRevision.columns),
                selectinload(NodeRevision.catalog),
                selectinload(NodeRevision.dimension_links),
            ),
        ),
    )


async def find_join_paths_batch(
    session: AsyncSession,
    source_revision_id: int,
    target_dimension_names: set[str],
    max_depth: int = 5,
) -> dict[tuple[str, str], list[int]]:
    """
    Find join paths from a source node to all target dimension nodes using a
    single recursive CTE query.

    Returns a dict mapping (dimension_node_name, role_path) to the list of
    DimensionLink IDs forming the path.

    The role_path is a "->" separated string of roles at each step.
    Empty roles are represented as empty strings.

    This is O(1) database calls instead of O(nodes * depth) individual queries.
    """
    if not target_dimension_names:
        return {}

    # Single recursive CTE to find all paths at once
    recursive_query = text("""
        WITH RECURSIVE paths AS (
            -- Base case: first level dimension links from the source node
            SELECT
                dl.id as link_id,
                n.name as dim_name,
                CAST(dl.id AS TEXT) as path,
                COALESCE(dl.role, '') as role_path,
                1 as depth
            FROM dimensionlink dl
            JOIN node n ON dl.dimension_id = n.id
            WHERE dl.node_revision_id = :source_revision_id

            UNION ALL

            -- Recursive case: follow dimension_links from each dimension node
            SELECT
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
        SELECT dim_name, path, role_path, depth
        FROM paths
        WHERE dim_name IN :target_names
        ORDER BY depth ASC
    """).bindparams(bindparam("target_names", expanding=True))

    result = await session.execute(
        recursive_query,
        {
            "source_revision_id": source_revision_id,
            "max_depth": max_depth,
            "target_names": list(target_dimension_names),
        },
    )
    rows = result.fetchall()

    # Build paths dict keyed by (dim_name, role_path)
    paths: dict[tuple[str, str], list[int]] = {}
    for dim_name, path_str, role_path, depth in rows:
        key = (dim_name, role_path or "")
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
    source_revision_id: int,
    target_dimension_names: set[str],
) -> None:
    """
    Preload all join paths from a source node to target dimensions.

    Uses a single recursive CTE query to find paths, then a single batch
    load for DimensionLink objects. Results are stored in ctx.join_paths.
    """
    if not target_dimension_names:
        return

    # Find all paths using recursive CTE (single query)
    path_ids = await find_join_paths_batch(
        ctx.session,
        source_revision_id,
        target_dimension_names,
    )

    # Collect all link IDs we need to load
    all_link_ids: set[int] = set()
    for link_id_list in path_ids.values():
        all_link_ids.update(link_id_list)

    # Batch load all DimensionLinks (single query)
    link_dict = await load_dimension_links_batch(ctx.session, all_link_ids)

    # Store in context, keyed by (source_revision_id, dim_name, role_path)
    for (dim_name, role_path), link_id_list in path_ids.items():
        links = [link_dict[lid] for lid in link_id_list if lid in link_dict]
        ctx.join_paths[(source_revision_id, dim_name, role_path)] = links
        # Also cache dimension nodes
        for link in links:
            if link.dimension and link.dimension.name not in ctx.nodes:
                ctx.nodes[link.dimension.name] = link.dimension

    logger.debug(f"[BuildV3] Preloaded {len(path_ids)} join paths in 2 queries")


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

    # Query 1: Find ALL upstream node names using lightweight recursive CTE
    # This includes all transitive dependencies (sources, transforms, dimensions)
    all_node_names = await find_upstream_node_names(
        ctx.session,
        list(initial_node_names),
    )

    # Also include the initial nodes themselves
    all_node_names.update(initial_node_names)

    logger.debug(f"[BuildV3] Found {len(all_node_names)} nodes to load")

    # Query 2: Batch load all nodes with appropriate eager loading
    stmt = (
        select(Node)
        .where(Node.name.in_(all_node_names))
        .where(Node.deactivated_at.is_(None))
        .options(
            selectinload(Node.current).options(
                selectinload(NodeRevision.columns),
                selectinload(NodeRevision.catalog),
                selectinload(NodeRevision.required_dimensions),
                _dimension_link_eager_load(),
                # Still need parents for metrics to find their parent node
                selectinload(NodeRevision.parents),
            ),
        )
    )

    result = await ctx.session.execute(stmt)
    nodes = result.scalars().unique().all()

    # Cache all loaded nodes and collect parent revision IDs (for join path lookup)
    parent_revision_ids: set[int] = set()
    for node in nodes:
        ctx.nodes[node.name] = node
        # Collect parent revision IDs for join path lookup
        if node.type == NodeType.METRIC and node.current and node.current.parents:
            for parent in node.current.parents:
                if parent.current:
                    parent_revision_ids.add(parent.current.id)

    logger.debug(f"[BuildV3] Loaded {len(ctx.nodes)} nodes")

    # Queries 3-4: Preload join paths for each parent node to target dimensions
    for parent_revision_id in parent_revision_ids:
        await preload_join_paths(ctx, parent_revision_id, target_dim_names)


# =============================================================================
# Topological Sort & Table Reference Rewriting
# =============================================================================


def get_physical_table_name(node: Node) -> str:
    """
    Get the fully qualified physical table name for a source node.

    Returns: catalog.schema.table format
    """
    if not node.current:
        raise DJInvalidInputException(f"Node {node.name} has no current revision")
    rev = node.current
    if not rev.catalog:
        raise DJInvalidInputException(f"Source node {node.name} has no catalog")
    return f"{rev.catalog.name}.{rev.schema_}.{rev.table}"


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
            # Transform/dimension - parse query to find references
            try:
                query_ast = parse(node.current.query)
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
    queue = [name for name, degree in in_degree.items() if degree == 0]
    sorted_names: list[str] = []

    while queue:
        current = queue.pop(0)
        sorted_names.append(current)
        # Reduce in-degree for all dependents
        for dependent in dependents.get(current, []):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    # Return sorted nodes (excluding any we couldn't sort due to cycles)
    return [node_map[name] for name in sorted_names if name in node_map]


def rewrite_table_references(
    query_ast: ast.Query,
    ctx: BuildContext,
    cte_names: dict[str, str],
) -> ast.Query:
    """
    Rewrite table references in a query AST.

    - Source nodes → physical table names (catalog.schema.table)
    - Transform/dimension nodes → CTE names

    Args:
        query_ast: The query AST to rewrite (modified in place)
        ctx: Build context with loaded nodes
        cte_names: Mapping of node names to their CTE names

    Returns:
        The modified query AST
    """
    for table in query_ast.find_all(ast.Table):
        table_name = str(table.name)
        ref_node = ctx.nodes.get(table_name)

        if ref_node:
            if ref_node.type == NodeType.SOURCE:
                # Replace with physical table name
                physical_name = get_physical_table_name(ref_node)
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
    if not metric_node.current or not metric_node.current.parents:
        raise DJInvalidInputException(f"Metric {metric_node.name} has no parent node")

    parent_ref = metric_node.current.parents[0]
    parent = ctx.nodes.get(parent_ref.name)
    if not parent:
        raise DJInvalidInputException(f"Parent node not found: {parent_ref.name}")
    return parent


def get_metric_expression_ast(metric_node: Node) -> ast.Expression:
    """
    Extract the metric expression AST from the metric's query.

    A metric query looks like: SELECT <expression> FROM <parent>
    We want to extract just the expression as an AST node.

    Returns:
        The expression AST node (with alias removed if present)
    """
    if not metric_node.current or not metric_node.current.query:
        raise DJInvalidInputException(f"Metric {metric_node.name} has no query")

    query_ast = parse(metric_node.current.query)
    if not query_ast.select.projection:
        raise DJInvalidInputException(f"Metric {metric_node.name} has no projection")

    # Get the first projection expression (the metric expression)
    expr = query_ast.select.projection[0]

    # Remove alias if present - we want the raw expression
    if isinstance(expr, ast.Alias):
        expr = expr.child
    elif hasattr(expr, "alias") and expr.alias:
        # Clear the alias so we can add our own
        expr = expr.copy()
        expr.alias = None

    # Clear parent reference so we can attach to new query
    expr.clear_parent()

    return expr


async def decompose_metric(
    session: AsyncSession,
    metric_node: Node,
) -> DecomposedMetricInfo:
    """
    Decompose a metric into its constituent components.

    Uses MetricComponentExtractor to break down aggregations like:
    - SUM(x) → [sum_x component]
    - AVG(x) → [sum_x component, count_x component]
    - COUNT(DISTINCT x) → [distinct_x component with LIMITED aggregability]

    Returns:
        DecomposedMetricInfo with components and aggregability
    """
    if not metric_node.current:
        raise DJInvalidInputException(
            f"Metric {metric_node.name} has no current revision",
        )

    # Use the existing MetricComponentExtractor
    extractor = MetricComponentExtractor(metric_node.current.id)
    components, _ = await extractor.extract(session)

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
        return expr_ast
    else:
        # Simple function name like "SUM" - build SUM(expression)
        func = ast.Function(
            name=ast.Name(component.aggregation),
            args=[parse(f"SELECT {component.expression}").select.projection[0]],
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

    Returns None if no path found.
    """
    if not from_node.current:
        return None

    source_revision_id = from_node.current.id
    role_path = role or ""

    # Look up preloaded path
    key = (source_revision_id, target_dim_name, role_path)
    links = ctx.join_paths.get(key)

    if links:
        # Path found in preloaded cache
        return JoinPath(
            links=links,
            target_dimension=links[-1].dimension,
            role=role,
        )

    # Fallback: check for direct links without role (may have been eagerly loaded)
    if not role and from_node.current.dimension_links:
        for link in from_node.current.dimension_links:
            if link.dimension and link.dimension.name == target_dim_name:
                return JoinPath(
                    links=[link],
                    target_dimension=link.dimension,
                    role=role,
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


def build_node_cte(
    node: Node,
    needed_columns: Optional[set[str]] = None,
) -> Optional[ast.Query]:
    """
    Build a CTE query for a transform or dimension node.

    If needed_columns is provided, modifies the query's projection to only
    include the columns that are needed, reducing the data transferred.

    Returns the parsed query AST, or None for source nodes.
    """
    if node.type == NodeType.SOURCE:
        return None

    rev = node.current
    if not rev or not rev.query:
        return None

    # Parse the node's query
    query_ast = parse(rev.query)

    # If no column filter, return the full query
    if not needed_columns:
        return query_ast

    # Filter the projection to only include needed columns
    # Build a map of output column name -> projection expression
    original_projection = query_ast.select.projection

    # Build new projection with only needed columns
    new_projection: list[ast.Expression] = []

    for expr in original_projection:
        # Determine the output name of this expression
        if isinstance(expr, ast.Alias):
            output_name = expr.alias.name if expr.alias else None
        elif isinstance(expr, ast.Column):
            output_name = (
                expr.alias.name
                if expr.alias
                else (expr.name.name if expr.name else None)
            )
        else:
            # For other expressions, check if they have an alias
            output_name = (
                expr.alias.name if hasattr(expr, "alias") and expr.alias else None
            )

        # Include if this column is needed
        if output_name and output_name in needed_columns:
            new_projection.append(expr)

    # If we filtered some columns, update the projection
    if new_projection:
        query_ast.select.projection = new_projection

    return query_ast


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


def collect_node_ctes(
    ctx: BuildContext,
    nodes_to_include: list[Node],
    needed_columns_by_node: Optional[dict[str, set[str]]] = None,
) -> list[tuple[str, ast.Query]]:
    """
    Collect CTEs for all non-source nodes, recursively expanding table references.

    This handles the full dependency chain:
    - Source nodes → replaced with physical table names (catalog.schema.table)
    - Transform/dimension nodes → recursive CTEs with dependencies resolved

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
                query_ast = parse(node.current.query)
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

        # Rewrite table references (sources → physical tables, transforms → CTE names)
        rewrite_table_references(query_ast, ctx, cte_names)

        # Apply column filtering if specified
        needed_cols = None
        if needed_columns_by_node:
            needed_cols = needed_columns_by_node.get(node.name)

        if needed_cols:
            query_ast = filter_cte_projection(query_ast, needed_cols)

        cte_name = cte_names[node.name]
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
    join = ast.Join(
        join_type=join_type_str,
        right=ast.Alias(
            child=ast.Table(name=right_table_name),
            alias=ast.Name(right_alias),
        ),
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
    projection: list[ast.Expression] = []
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
    relation = ast.Relation(
        primary=ast.Alias(
            child=ast.Table(name=table_name),
            alias=ast.Name(main_alias),
        ),
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


async def decompose_and_group_metrics(
    ctx: BuildContext,
) -> list[MetricGroup]:
    """
    Decompose metrics and group them by parent node.

    This replaces the simple group_metrics_by_parent with decomposition support.

    Returns:
        List of MetricGroup, one per unique parent node, with decomposed metrics.
    """
    # Map parent node name -> list of DecomposedMetricInfo
    parent_groups: dict[str, list[DecomposedMetricInfo]] = {}
    parent_nodes: dict[str, Node] = {}

    for metric_name in ctx.metrics:
        # Get metric and parent nodes (in-memory)
        metric_node = get_metric_node(ctx, metric_name)
        parent_node = get_parent_node(ctx, metric_node)

        # Decompose the metric (requires DB for MetricComponentExtractor)
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


async def build_measures_sql(
    session: AsyncSession,
    metrics: list[str],
    dimensions: list[str],
    filters: list[str] | None = None,
    dialect: Dialect = Dialect.SPARK,
) -> GeneratedSQL:
    """
    Build measures SQL for a set of metrics and dimensions.

    This is the main entry point for V3 measures SQL generation.

    Measures SQL aggregates metric components to the requested dimensional
    grain, but does NOT apply final metric expressions.

    Args:
        session: Database session
        metrics: List of metric node names
        dimensions: List of dimension names (format: "node.column" or "node.column[role]")
        filters: Optional list of filter expressions
        dialect: SQL dialect for output

    Returns:
        GeneratedSQL with the SQL string and column metadata
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

    # For now (Chunk 3), we only support metrics from a single parent
    # Multi-parent support (cross-fact) will be added in a later chunk
    if len(metric_groups) > 1:
        parent_names = [g.parent_node.name for g in metric_groups]
        raise DJInvalidInputException(
            f"All metrics must come from the same parent node. "
            f"Found metrics from: {', '.join(parent_names)}. "
            f"Cross-fact metrics will be supported in a future version.",
        )

    # Get the single metric group
    metric_group = metric_groups[0]
    parent_node = metric_group.parent_node

    # Count components per metric to determine naming strategy
    # Single component metrics use the metric name; multi-component use hash names
    components_per_metric: dict[str, int] = {}
    for decomposed in metric_group.decomposed_metrics:
        components_per_metric[decomposed.metric_node.name] = len(decomposed.components)

    # Collect grain columns from LIMITED aggregability components
    # These must be included in GROUP BY for the output to be re-aggregatable
    grain_columns: list[str] = []
    for metric_node, component in metric_group.get_all_components():
        if component.rule.type == Aggregability.LIMITED and component.rule.level:
            for level_col in component.rule.level:
                if level_col not in grain_columns:
                    grain_columns.append(level_col)

    # Build list of component expressions with their aliases
    # For measures SQL, we output components (not the full metric expression)
    component_expressions: list[tuple[str, ast.Expression]] = []
    component_metadata: list[
        tuple[str, MetricComponent, Node, bool]
    ] = []  # (alias, component, metric_node, is_simple)

    for metric_node, component in metric_group.get_all_components():
        # Skip LIMITED aggregability components with no aggregation
        # These are represented by grain columns instead (e.g., customer_id for COUNT DISTINCT)
        if component.rule.type == Aggregability.LIMITED and not component.aggregation:
            continue

        num_components = components_per_metric.get(metric_node.name, 1)
        is_simple = num_components == 1

        if is_simple:
            # Single component: use metric name as alias
            component_alias = metric_node.name.split(SEPARATOR)[-1]
        else:
            # Multiple components: use component name (includes hash for uniqueness)
            component_alias = component.name

        expr_ast = build_component_expression(component)
        component_expressions.append((component_alias, expr_ast))
        component_metadata.append((component_alias, component, metric_node, is_simple))

    # Resolve dimensions (find join paths)
    resolved_dimensions = resolve_dimensions(ctx, parent_node)

    # Build AST with JOIN support and component expressions
    # Pass grain_columns for LIMITED aggregability support
    query_ast = build_select_ast(
        ctx,
        metric_expressions=component_expressions,
        resolved_dimensions=resolved_dimensions,
        parent_node=parent_node,
        grain_columns=grain_columns,
    )

    # Generate SQL string from AST
    sql_str = str(query_ast)

    # Build column metadata
    columns_metadata = []

    # Add dimension columns
    for resolved_dim in resolved_dimensions:
        alias = (
            ctx.alias_registry.get_alias(resolved_dim.original_ref)
            or resolved_dim.column_name
        )
        # Get column type from appropriate node
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

    # Add grain columns for LIMITED aggregability (e.g., customer_id for COUNT DISTINCT)
    for grain_col in grain_columns:
        col_type = get_column_type(parent_node, grain_col)
        columns_metadata.append(
            ColumnMetadata(
                name=grain_col,
                semantic_name=f"{parent_node.name}{SEPARATOR}{grain_col}",
                type=col_type,
                semantic_type="dimension",  # Dimension required by metric's aggregability
            ),
        )

    # Add metric component columns
    for comp_alias, component, metric_node, is_simple in component_metadata:
        if is_simple:
            # Simple metric (single component): use metric name, type is "metric"
            columns_metadata.append(
                ColumnMetadata(
                    name=ctx.alias_registry.get_alias(comp_alias) or comp_alias,
                    semantic_name=metric_node.name,
                    type="number",  # TODO: Get actual type from component
                    semantic_type="metric",
                ),
            )
        else:
            # Complex metric (multiple components): use component name, type is "metric_component"
            columns_metadata.append(
                ColumnMetadata(
                    name=ctx.alias_registry.get_alias(comp_alias) or comp_alias,
                    semantic_name=f"{metric_node.name}:{component.name}",  # metric:component format
                    type="number",  # TODO: Get actual type from component
                    semantic_type="metric_component",
                ),
            )

    return GeneratedSQL(
        sql=sql_str,
        columns=columns_metadata,
        dialect=dialect,
    )


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
    including handling derived metrics.

    This will be implemented in Chunk 5.
    """
    raise NotImplementedError("Metrics SQL not yet implemented (Chunk 5)")
