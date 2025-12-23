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
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from datajunction_server.construction.build_v3.alias_registry import AliasRegistry
from datajunction_server.database.node import Node, NodeRevision
from datajunction_server.errors import DJInvalidInputException
from datajunction_server.models.dialect import Dialect
from datajunction_server.models.node_type import NodeType
from datajunction_server.sql.parsing import ast
from datajunction_server.sql.parsing.backends.antlr4 import parse

if TYPE_CHECKING:
    from datajunction_server.database.column import Column as DBColumn

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


@dataclass
class GeneratedSQL:
    """Output of the SQL generation pipeline."""
    sql: str
    columns: list[ColumnMetadata]
    dialect: Dialect


@dataclass
class ColumnMetadata:
    """Metadata about a column in the generated SQL."""
    name: str           # SQL alias (clean name)
    semantic_name: str  # Full semantic name (e.g., 'orders.country')
    type: str           # Column type


# =============================================================================
# Node Loading
# =============================================================================


async def load_nodes(ctx: BuildContext) -> None:
    """
    Load all nodes needed for SQL generation.
    
    This includes:
    - Metric nodes
    - Their parent nodes (the nodes metrics are defined on)
    - Dimension nodes (if different from parent)
    """
    # Collect all node names we need
    node_names_to_load = set(ctx.metrics)
    
    # Also need dimension node names (from "node.column" format)
    for dim in ctx.dimensions:
        if '.' in dim:
            # Extract node name from "node.column" or "node.column[role]"
            dim_ref = dim.split('[')[0]  # Remove [role] if present
            parts = dim_ref.rsplit('.', 1)
            if len(parts) >= 1:
                node_names_to_load.add(parts[0])
    
    # Load metric nodes with their parents
    stmt = (
        select(Node)
        .where(Node.name.in_(node_names_to_load))
        .where(Node.deactivated_at.is_(None))
        .options(
            selectinload(Node.current).options(
                selectinload(NodeRevision.columns),
                selectinload(NodeRevision.parents),
                selectinload(NodeRevision.required_dimensions),
            ),
        )
    )
    
    result = await ctx.session.execute(stmt)
    nodes = result.scalars().unique().all()
    
    for node in nodes:
        ctx.nodes[node.name] = node
    
    # Load parent nodes of metrics
    parent_names_to_load = set()
    for name in ctx.metrics:
        if name in ctx.nodes:
            node = ctx.nodes[name]
            if node.current and node.current.parents:
                for parent in node.current.parents:
                    if parent.name not in ctx.nodes:
                        parent_names_to_load.add(parent.name)
    
    if parent_names_to_load:
        stmt = (
            select(Node)
            .where(Node.name.in_(parent_names_to_load))
            .where(Node.deactivated_at.is_(None))
            .options(
                selectinload(Node.current).options(
                    selectinload(NodeRevision.columns),
                    selectinload(NodeRevision.catalog),
                ),
            )
        )
        result = await ctx.session.execute(stmt)
        parent_nodes = result.scalars().unique().all()
        for node in parent_nodes:
            ctx.nodes[node.name] = node
    
    logger.debug(f"[BuildV3] Loaded {len(ctx.nodes)} nodes")


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
        raise DJInvalidInputException(
            f"Metric {metric_node.name} has no parent node"
        )
    
    parent_ref = metric_node.current.parents[0]
    parent = ctx.nodes.get(parent_ref.name)
    if not parent:
        raise DJInvalidInputException(
            f"Parent node not found: {parent_ref.name}"
        )
    return parent


def get_metric_expression(metric_node: Node) -> str:
    """
    Extract the metric expression from the metric's query.
    
    A metric query looks like: SELECT <expression> FROM <parent>
    We want to extract just the expression.
    """
    if not metric_node.current or not metric_node.current.query:
        raise DJInvalidInputException(
            f"Metric {metric_node.name} has no query"
        )
    
    query_ast = parse(metric_node.current.query)
    if not query_ast.select.projection:
        raise DJInvalidInputException(
            f"Metric {metric_node.name} has no projection"
        )
    
    # Return the first projection expression (the metric expression)
    expr = query_ast.select.projection[0]
    
    # Remove alias if present (we'll add our own)
    if hasattr(expr, 'alias') and expr.alias:
        # Return just the expression without alias
        if hasattr(expr, 'child'):
            return str(expr.child)
    
    return str(expr)


# =============================================================================
# AST Construction
# =============================================================================


def get_table_reference(parent_node: Node) -> str:
    """
    Get the fully qualified table reference for a source/transform node.
    
    For source nodes: catalog.schema.table
    For transform nodes: build recursively (not implemented in Chunk 1)
    """
    rev = parent_node.current
    if not rev:
        raise DJInvalidInputException(
            f"Node {parent_node.name} has no current revision"
        )
    
    # For source nodes, build catalog.schema.table
    if parent_node.type == NodeType.SOURCE:
        parts = []
        if rev.catalog:
            parts.append(rev.catalog.name)
        if rev.schema_:
            parts.append(rev.schema_)
        if rev.table:
            parts.append(rev.table)
        else:
            # Fall back to node name
            parts.append(parent_node.name)
        return '.'.join(parts)
    
    # For other nodes, use node name as table reference for now
    # (will be replaced with CTE in later chunks)
    return parent_node.name


def build_select_ast(
    ctx: BuildContext,
    metric_expressions: list[tuple[str, str]],  # (alias, expression)
    dimension_columns: list[str],
    table_ref: str,
) -> ast.Query:
    """
    Build a SELECT AST for measures SQL.
    
    Args:
        ctx: Build context
        metric_expressions: List of (alias, SQL expression) tuples
        dimension_columns: List of dimension column names  
        table_ref: Table reference string
        
    Returns:
        AST Query node
    """
    # Build projection (SELECT clause)
    projection_parts = []
    
    # Add dimensions first
    for dim_col in dimension_columns:
        # Register alias
        alias = ctx.alias_registry.register(dim_col)
        if alias != dim_col:
            projection_parts.append(f"{dim_col} AS {alias}")
        else:
            projection_parts.append(dim_col)
    
    # Add metric expressions
    for alias, expr in metric_expressions:
        clean_alias = ctx.alias_registry.register(alias)
        projection_parts.append(f"{expr} AS {clean_alias}")
    
    # Build GROUP BY (same as dimensions)
    group_by_parts = dimension_columns.copy()
    
    # Construct SQL
    select_clause = ", ".join(projection_parts)
    group_by_clause = ", ".join(group_by_parts) if group_by_parts else ""
    
    sql = f"SELECT {select_clause} FROM {table_ref}"
    if group_by_clause:
        sql += f" GROUP BY {group_by_clause}"
    
    return parse(sql)


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
        dimensions: List of dimension names (format: "node.column")
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
    
    # For Chunk 1: Single metric, single dimension, same table (no joins)
    # More complex cases will be handled in later chunks
    
    # Get metric and its parent
    metric_node = get_metric_node(ctx, ctx.metrics[0])
    parent_node = get_parent_node(ctx, metric_node)
    
    # Get metric expression
    metric_expr = get_metric_expression(metric_node)
    metric_alias = metric_node.name.split('.')[-1]  # Use last part of name
    
    # Get dimension columns (just the column name part)
    dimension_columns = []
    for dim in ctx.dimensions:
        if '.' in dim:
            # Format: "node.column" - extract just column
            col_name = dim.rsplit('.', 1)[-1]
            # Remove [role] if present
            col_name = col_name.split('[')[0]
            dimension_columns.append(col_name)
        else:
            dimension_columns.append(dim)
    
    # Get table reference
    table_ref = get_table_reference(parent_node)
    
    # Build AST
    query_ast = build_select_ast(
        ctx,
        metric_expressions=[(metric_alias, metric_expr)],
        dimension_columns=dimension_columns,
        table_ref=table_ref,
    )
    
    # Generate SQL string
    sql_str = str(query_ast)
    
    # Build column metadata
    columns_metadata = []
    for dim in dimension_columns:
        alias = ctx.alias_registry.get_alias(dim) or dim
        columns_metadata.append(ColumnMetadata(
            name=alias,
            semantic_name=dim,
            type="string",  # TODO: Get actual type
        ))
    
    columns_metadata.append(ColumnMetadata(
        name=ctx.alias_registry.get_alias(metric_alias) or metric_alias,
        semantic_name=metric_node.name,
        type="number",  # TODO: Get actual type
    ))
    
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

