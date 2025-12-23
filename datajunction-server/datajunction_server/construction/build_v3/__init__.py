"""
Build V3: Clean SQL Generation for DataJunction

This module provides a clean-slate reimplementation of SQL generation that:
- Separates measures (pre-aggregated) from metrics (fully computed)
- Properly handles metric decomposition and derived metrics
- Supports explicit hierarchies and inferred dimension link chains
- Respects aggregability rules and required dimensions
- Produces clean SQL without _DOT_ pollution

See ARCHITECTURE.md for full documentation.

Usage:
    from datajunction_server.construction.build_v3 import build_measures_sql, build_metrics_sql
    
    # Generate measures SQL (pre-aggregated to dimensional grain)
    result = await build_measures_sql(
        session,
        metrics=['total_revenue', 'avg_order_value'],
        dimensions=['country', 'quarter'],
    )
    print(result.sql)
    
    # Generate metrics SQL (with final metric expressions)
    result = await build_metrics_sql(
        session,
        metrics=['avg_order_value'],
        dimensions=['country'],
    )
    print(result.sql)
"""

from datajunction_server.construction.build_v3.builder import (
    build_measures_sql,
    build_metrics_sql,
    BuildContext,
    GeneratedSQL,
    ColumnMetadata,
)
from datajunction_server.construction.build_v3.alias_registry import (
    AliasRegistry,
    ScopedAliasRegistry,
)

__all__ = [
    # Main entry points
    'build_measures_sql',
    'build_metrics_sql',
    # Context and types
    'BuildContext',
    'GeneratedSQL',
    'ColumnMetadata',
    # Alias registry
    'AliasRegistry',
    'ScopedAliasRegistry',
]

