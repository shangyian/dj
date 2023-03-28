"""
Functions for extracting DJ information from an AST
"""

from typing import Dict, List, Optional, Tuple

from sqlmodel import Session

from dj.construction.compile import CompoundBuildException, make_name
from dj.errors import DJException
from dj.models.node import NodeRevision, NodeType
from dj.sql.parsing import ast2 as ast, parse

def extract_dependencies_from_compiled_query_ast(
    query: ast.Query,
) -> Tuple[Dict[NodeRevision, List[ast.Table]], Dict[str, List[ast.Table]]]:
    """Find all dependencies in a compiled query"""
    deps: Dict[NodeRevision, List[ast.Table]] = {}
    danglers: Dict[str, List[ast.Table]] = {}
    for table in query.find_all(ast.Table):
        if node := table.dj_node:
            deps[node] = deps.get(node, [])
            deps[node].append(table)
        else:
            name = make_name(table.namespace, table.name.name)
            danglers[name] = danglers.get(name, [])
            danglers[name].append(table)

    return deps, danglers