"""
DAG related functions.
"""
import itertools
from typing import Dict, List, Optional, Set, Union

from sqlalchemy import and_, func, join, literal, or_, select
from sqlalchemy.orm import Session, aliased, joinedload
from sqlalchemy.sql.operators import is_

from datajunction_server.database.attributetype import AttributeType, ColumnAttribute
from datajunction_server.database.column import Column
from datajunction_server.database.node import (
    Node,
    NodeColumns,
    NodeRelationship,
    NodeRevision,
)
from datajunction_server.models.node import DimensionAttributeOutput
from datajunction_server.models.node_type import NodeType
from datajunction_server.utils import SEPARATOR, get_settings

settings = get_settings()


def _node_output_options():
    """
    Statement options to retrieve all NodeOutput objects in one query
    """
    return [
        joinedload(Node.current).options(
            joinedload(NodeRevision.columns).options(
                joinedload(Column.attributes).joinedload(
                    ColumnAttribute.attribute_type,
                ),
                joinedload(Column.dimension),
                joinedload(Column.partition),
            ),
            joinedload(NodeRevision.catalog),
            joinedload(NodeRevision.parents),
        ),
        joinedload(Node.tags),
    ]


def get_downstream_nodes(
    session: Session,
    node_name: str,
    node_type: NodeType = None,
    include_deactivated: bool = True,
    include_cubes: bool = True,
) -> List[Node]:
    """
    Gets all downstream children of the given node, filterable by node type.
    Uses a recursive CTE query to build out all descendants from the node.
    """
    node = session.execute(select(Node).where(Node.name == node_name)).scalar()

    initial_dag = (
        select(
            NodeRelationship.parent_id,
            NodeRevision.node_id,
        )
        .where(NodeRelationship.parent_id == node.id)
        .join(NodeRevision, NodeRelationship.child_id == NodeRevision.id)
        .join(
            Node,
            (Node.id == NodeRevision.node_id)
            & (Node.current_version == NodeRevision.version),
        )
    )
    if not include_cubes:
        initial_dag = initial_dag.where((NodeRevision.type != NodeType.CUBE))
    dag = initial_dag.cte("downstreams", recursive=True)

    next_layer = (
        select(
            dag.c.parent_id,
            NodeRevision.node_id,
        )
        .join(NodeRelationship, dag.c.node_id == NodeRelationship.parent_id)
        .join(NodeRevision, NodeRelationship.child_id == NodeRevision.id)
        .join(Node, Node.id == NodeRevision.node_id)
    )
    if not include_cubes:
        next_layer = next_layer.where(NodeRevision.type != NodeType.CUBE)

    paths = dag.union_all(next_layer)

    final_select = select(Node)
    if not include_deactivated:
        final_select = final_select.where(is_(Node.deactivated_at, None))
    statement = final_select.join(paths, paths.c.node_id == Node.id).options(
        *_node_output_options()
    )
    results = session.execute(statement).unique().scalars().all()

    return [
        downstream
        for downstream in results
        if downstream.type == node_type or node_type is None
    ]


def get_upstream_nodes(
    session: Session,
    node_name: str,
    node_type: NodeType = None,
    include_deactivated: bool = True,
) -> List[Node]:
    """
    Gets all upstreams of the given node, filterable by node type.
    Uses a recursive CTE query to build out all parents of the node.
    """
    node = (
        session.execute(
            select(Node).where(
                (Node.name == node_name) & (is_(Node.deactivated_at, None)),
            ),
        )
        .unique()
        .scalar()
    )

    dag = (
        select(
            NodeRelationship.child_id,
            NodeRevision.id,
            NodeRevision.node_id,
        )
        .where(NodeRelationship.child_id == node.current.id)
        .join(Node, NodeRelationship.parent_id == Node.id)
        .join(
            NodeRevision,
            (Node.id == NodeRevision.node_id)
            & (Node.current_version == NodeRevision.version),
        )
    ).cte("upstreams", recursive=True)

    paths = dag.union_all(
        select(
            dag.c.child_id,
            NodeRevision.id,
            NodeRevision.node_id,
        )
        .join(NodeRelationship, dag.c.id == NodeRelationship.child_id)
        .join(Node, NodeRelationship.parent_id == Node.id)
        .join(
            NodeRevision,
            (Node.id == NodeRevision.node_id)
            & (Node.current_version == NodeRevision.version),
        ),
    )

    node_selector = select(Node)
    if not include_deactivated:
        node_selector = node_selector.where(is_(Node.deactivated_at, None))
    statement = (
        node_selector.join(paths, paths.c.node_id == Node.id)
        .join(
            NodeRevision,
            (Node.current_version == NodeRevision.version)
            & (Node.id == NodeRevision.node_id),
        )
        .options(*_node_output_options())
    )

    results = session.execute(statement).unique().scalars().all()
    return [
        upstream
        for upstream in results
        if upstream.type == node_type or node_type is None
    ]


def get_dimensions_dag(
    session: Session,
    node_revision: NodeRevision,
    with_attributes: bool = True,
) -> List[Union[DimensionAttributeOutput, Node]]:
    """
    Gets the dimensions graph of the given node revision with a single recursive CTE query.
    This graph is split out into dimension attributes or dimension nodes depending on the
    `with_attributes` flag.
    """

    initial_node = aliased(NodeRevision, name="initial_node")
    dimension_node = aliased(Node, name="dimension_node")
    dimension_rev = aliased(NodeRevision, name="dimension_rev")
    current_node = aliased(Node, name="current_node")
    current_rev = aliased(NodeRevision, name="current_rev")
    next_node = aliased(Node, name="next_node")
    next_rev = aliased(NodeRevision, name="next_rev")
    column = aliased(Column, name="c")

    # Recursive CTE
    dimensions_graph = (
        select(
            initial_node.id.label("path_start"),
            column.name.label("col_name"),
            dimension_node.id.label("path_end"),
            (initial_node.name + "." + column.name + "," + dimension_node.name).label(
                "join_path",
            ),
            dimension_node.name.label("node_name"),
            dimension_rev.id.label("node_revision_id"),
            dimension_rev.display_name.label("node_display_name"),
        )
        .select_from(initial_node)
        .join(NodeColumns, initial_node.id == NodeColumns.node_id)
        .join(column, NodeColumns.column_id == column.id)
        .join(
            dimension_node,
            (column.dimension_id == dimension_node.id)
            & (is_(dimension_node.deactivated_at, None)),
        )
        .join(
            dimension_rev,
            and_(
                dimension_rev.version == dimension_node.current_version,
                dimension_rev.node_id == dimension_node.id,
            ),
        )
        .where(initial_node.id == node_revision.id)
    ).cte("dimensions_graph", recursive=True)

    paths = dimensions_graph.union_all(
        select(
            dimensions_graph.c.path_start,
            column.name.label("col_name"),
            next_node.id.label("path_end"),
            (
                dimensions_graph.c.join_path + "." + column.name + "," + next_node.name
            ).label(
                "join_path",
            ),
            next_node.name.label("node_name"),
            next_rev.id.label("node_revision_id"),
            next_rev.display_name.label("node_display_name"),
        ).select_from(
            dimensions_graph.join(
                current_node,
                dimensions_graph.c.path_end == current_node.id,
            )
            .join(
                current_rev,
                and_(
                    current_rev.version == current_node.current_version,
                    current_rev.node_id == current_node.id,
                ),
            )
            .join(NodeColumns, current_rev.id == NodeColumns.node_id)
            .join(column, NodeColumns.column_id == column.id)
            .join(
                next_node,
                (column.dimension_id == next_node.id)
                & (is_(next_node.deactivated_at, None)),
            )
            .join(
                next_rev,
                and_(
                    next_rev.version == next_node.current_version,
                    next_rev.node_id == next_node.id,
                ),
            ),
        ),
    )

    # Final SELECT statements
    # ----
    # If attributes was set to False, we only need to return the dimension nodes
    if not with_attributes:
        return (
            session.execute(
                select(Node)
                .select_from(paths)
                .join(Node, paths.c.node_name == Node.name)
                .options(*_node_output_options()),
            )
            .unique()
            .scalars()
            .all()
        )

    # Otherwise return the dimension attributes, which include both the dimension
    # attributes on the dimension nodes in the DAG as well as the local dimension
    # attributes on the initial node
    group_concat = (
        func.group_concat
        if session.bind.dialect.name in ("sqlite",)
        else func.string_agg
    )
    final_query = (
        select(
            paths.c.node_name,
            paths.c.node_display_name,
            column.name,
            column.type,
            group_concat(AttributeType.name, ",").label(
                "column_attribute_type_name",
            ),
            paths.c.join_path,
        )
        .select_from(paths)
        .join(NodeColumns, NodeColumns.node_id == paths.c.node_revision_id)
        .join(column, NodeColumns.column_id == column.id)
        .join(ColumnAttribute, column.id == ColumnAttribute.column_id, isouter=True)
        .join(
            AttributeType,
            ColumnAttribute.attribute_type_id == AttributeType.id,
            isouter=True,
        )
        .group_by(
            paths.c.node_name,
            paths.c.node_display_name,
            column.name,
            column.type,
            paths.c.join_path,
        )
        .union_all(
            select(
                NodeRevision.name,
                NodeRevision.display_name,
                Column.name,
                Column.type,
                group_concat(AttributeType.name, ",").label(
                    "column_attribute_type_name",
                ),
                literal("").label("join_path"),
            )
            .select_from(NodeRevision)
            .join(NodeColumns, NodeColumns.node_id == NodeRevision.id)
            .join(Column, NodeColumns.column_id == Column.id)
            .join(
                ColumnAttribute,
                Column.id == ColumnAttribute.column_id,
                isouter=True,
            )
            .join(
                AttributeType,
                ColumnAttribute.attribute_type_id == AttributeType.id,
                isouter=True,
            )
            .group_by(
                NodeRevision.name,
                NodeRevision.display_name,
                Column.name,
                Column.type,
                "join_path",
            )
            .where(NodeRevision.id == node_revision.id),
        )
    )

    # Only include a given column it's an attribute on a dimension node or
    # if the column is tagged with the attribute type 'dimension'
    return sorted(
        [
            DimensionAttributeOutput(
                name=f"{node_name}.{column_name}",
                node_name=node_name,
                node_display_name=node_display_name,
                is_primary_key=(
                    attribute_types is not None and "primary_key" in attribute_types
                ),
                type=str(column_type),
                path=join_path.split(",")[:-1] if join_path else [],
            )
            for (
                node_name,
                node_display_name,
                column_name,
                column_type,
                attribute_types,
                join_path,
            ) in session.execute(
                final_query,
            ).all()
            if (  # column has dimension attribute
                join_path == ""
                and attribute_types is not None
                and ("dimension" in attribute_types or "primary_key" in attribute_types)
            )
            or (  # column is on dimension node
                join_path != ""
                or (
                    node_name == node_revision.name
                    and node_revision.type == NodeType.DIMENSION
                )
            )
        ],
        key=lambda x: x.name,
    )


def get_dimensions(
    session: Session,
    node: Node,
    with_attributes: bool = True,
) -> List[Union[DimensionAttributeOutput, Node]]:
    """
    Return all available dimensions for a given node.
    * Setting `attributes` to True will return a list of dimension attributes,
    * Setting `attributes` to False will return a list of dimension nodes
    """
    if node.type == NodeType.METRIC:
        return get_dimensions_dag(
            session,
            node.current.parents[0].current,
            with_attributes,
        )
    return get_dimensions_dag(session, node.current, with_attributes)


def check_convergence(path1: List[str], path2: List[str]) -> bool:
    """
    Determines whether two join paths converge before we reach the
    final element, the dimension attribute.
    """
    if path1 == path2:
        return True  # pragma: no cover
    len1 = len(path1)
    len2 = len(path2)
    min_len = min(len1, len2)

    for i in range(min_len):
        partial1 = path1[len1 - i - 1 :]
        partial2 = path2[len2 - i - 1 :]
        if partial1 == partial2:
            return True

    # TODO: Once we introduce dimension roles, we can remove this.  # pylint: disable=fixme
    # To workaround this for now, we're using column names as the effective role of the dimension
    if (
        path1
        and path2
        and path1[-1].split(SEPARATOR)[-1] == path2[-1].split(SEPARATOR)[-1]
    ):
        return True
    return False  # pragma: no cover


def group_dimensions_by_name(
    session: Session,
    node: Node,
) -> Dict[str, List[DimensionAttributeOutput]]:
    """
    Group the dimensions for the node by the dimension attribute name
    """
    return {
        k: list(v)
        for k, v in itertools.groupby(
            get_dimensions(session, node),
            key=lambda dim: dim.name,
        )
    }


def get_shared_dimensions(
    session: Session,
    metric_nodes: List[Node],
) -> List[DimensionAttributeOutput]:
    """
    Return a list of dimensions that are common between the nodes.
    """
    find_latest_node_revisions = [
        and_(
            NodeRevision.name == metric_node.name,
            NodeRevision.version == metric_node.current_version,
        )
        for metric_node in metric_nodes
    ]
    statement = (
        select(Node)
        .where(or_(*find_latest_node_revisions))
        .select_from(
            join(
                join(
                    NodeRevision,
                    NodeRelationship,
                ),
                Node,
                NodeRelationship.parent_id == Node.id,
            ),
        )
    )
    parents = list(set(session.execute(statement).scalars().all()))
    common = group_dimensions_by_name(session, parents[0])
    for node in parents[1:]:
        node_dimensions = group_dimensions_by_name(session, node)

        # Merge each set of dimensions based on the name and path
        to_delete = set(common.keys() - node_dimensions.keys())
        common_dim_keys = common.keys() & list(node_dimensions.keys())
        if not common_dim_keys:
            return []
        for common_dim in common_dim_keys:
            for existing_attr in common[common_dim]:
                for new_attr in node_dimensions[common_dim]:
                    converged = check_convergence(existing_attr.path, new_attr.path)
                    if not converged:
                        to_delete.add(common_dim)  # pragma: no cover

        for dim_key in to_delete:
            del common[dim_key]  # pragma: no cover

    return sorted(
        [y for x in common.values() for y in x],
        key=lambda x: (x.name, x.path),
    )


def get_nodes_with_dimension(
    session: Session,
    dimension_node: Node,
    node_types: Optional[List[NodeType]] = None,
) -> List[NodeRevision]:
    """
    Find all nodes that can be joined to a given dimension
    """
    to_process = [dimension_node]
    processed: Set[str] = set()
    final_set: Set[NodeRevision] = set()
    while to_process:
        current_node = to_process.pop()
        processed.add(current_node.name)

        # Dimension nodes are used to expand the searchable graph by finding
        # the next layer of nodes that are linked to this dimension
        if current_node.type == NodeType.DIMENSION:
            statement = (
                select(NodeRevision)
                .join(
                    Node,
                    onclause=(
                        (NodeRevision.node_id == Node.id)
                        & (Node.current_version == NodeRevision.version)
                    ),  # pylint: disable=superfluous-parens
                )
                .join(
                    NodeColumns,
                    onclause=(
                        NodeRevision.id == NodeColumns.node_id
                    ),  # pylint: disable=superfluous-parens
                )
                .join(
                    Column,
                    onclause=(
                        NodeColumns.column_id == Column.id
                    ),  # pylint: disable=superfluous-parens
                )
                .where(
                    Column.dimension_id.in_(  # type: ignore  # pylint: disable=no-member
                        [current_node.id],
                    ),
                )
            )
            node_revisions = session.execute(statement).unique().scalars().all()
            for node_rev in node_revisions:
                to_process.append(node_rev.node)
        else:
            # All other nodes are added to the result set
            final_set.add(current_node.current)
            for child in current_node.children:
                if child.name not in processed:
                    to_process.append(child.node)
    if node_types:
        return [node for node in final_set if node.type in node_types]
    return list(final_set)


def get_nodes_with_common_dimensions(
    session: Session,
    common_dimensions: List[Node],
    node_types: Optional[List[NodeType]] = None,
) -> List[NodeRevision]:
    """
    Find all nodes that share a list of common dimensions
    """
    nodes_that_share_dimensions = set()
    first = True
    for dimension in common_dimensions:
        new_nodes = get_nodes_with_dimension(session, dimension, node_types)
        if first:
            nodes_that_share_dimensions = set(new_nodes)
            first = False
        else:
            nodes_that_share_dimensions = nodes_that_share_dimensions.intersection(
                set(new_nodes),
            )
            if not nodes_that_share_dimensions:
                break
    return list(nodes_that_share_dimensions)
