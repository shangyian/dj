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
from dj.models.node import Node, NodeRevision
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
