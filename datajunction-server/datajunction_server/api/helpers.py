# pylint: disable=too-many-lines
"""
Helpers for API endpoints
"""
import asyncio
import http.client
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from typing import Dict, List, Optional, Set, Tuple, Union

from sqlalchemy import select
from sqlalchemy.exc import MissingGreenlet
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload
from sqlalchemy.sql.operators import and_, is_

from datajunction_server.construction.build import (
    build_materialized_cube_node,
    build_metric_nodes,
    build_node,
    rename_columns,
    validate_shared_dimensions,
)
from datajunction_server.construction.dj_query import build_dj_query
from datajunction_server.database.attributetype import AttributeType
from datajunction_server.database.catalog import Catalog
from datajunction_server.database.column import Column
from datajunction_server.database.engine import Engine
from datajunction_server.database.history import ActivityType, EntityType, History
from datajunction_server.database.namespace import NodeNamespace
from datajunction_server.database.node import (
    MissingParent,
    Node,
    NodeMissingParents,
    NodeRevision,
)
from datajunction_server.database.user import User
from datajunction_server.errors import (
    DJAlreadyExistsException,
    DJDoesNotExistException,
    DJError,
    DJException,
    DJInvalidInputException,
    DJNodeNotFound,
    ErrorCode,
)
from datajunction_server.internal.engines import get_engine
from datajunction_server.models import access
from datajunction_server.models.attribute import RESERVED_ATTRIBUTE_NAMESPACE
from datajunction_server.models.base import labelize
from datajunction_server.models.engine import Dialect
from datajunction_server.models.history import status_change_history
from datajunction_server.models.metric import TranslatedSQL
from datajunction_server.models.node import BuildCriteria, NodeRevisionBase, NodeStatus
from datajunction_server.models.node_type import NodeType
from datajunction_server.models.query import ColumnMetadata, QueryWithResults
from datajunction_server.naming import LOOKUP_CHARS
from datajunction_server.service_clients import QueryServiceClient
from datajunction_server.sql.dag import get_downstream_nodes, get_nodes_with_dimension
from datajunction_server.sql.parsing import ast
from datajunction_server.sql.parsing.backends.antlr4 import SqlSyntaxError, parse
from datajunction_server.sql.parsing.backends.exceptions import DJParseException
from datajunction_server.typing import END_JOB_STATES, UTCDatetime
from datajunction_server.utils import SEPARATOR

_logger = logging.getLogger(__name__)

COLUMN_NAME_REGEX = r"([A-Za-z0-9_\.]+)(\[[A-Za-z0-9_]+\])?"


async def get_node_namespace(  # pylint: disable=too-many-arguments
    session: AsyncSession,
    namespace: str,
    raise_if_not_exists: bool = True,
) -> NodeNamespace:
    """
    Get a node namespace
    """
    statement = select(NodeNamespace).where(NodeNamespace.namespace == namespace)
    node_namespace = (await session.execute(statement)).scalar_one_or_none()
    if raise_if_not_exists:  # pragma: no cover
        if not node_namespace:
            raise DJDoesNotExistException(
                message=(f"node namespace `{namespace}` does not exist."),
                http_status_code=404,
            )
    return node_namespace


async def get_node_by_name(  # pylint: disable=too-many-arguments
    session: AsyncSession,
    name: Optional[str],
    node_type: Optional[NodeType] = None,
    with_current: bool = False,
    raise_if_not_exists: bool = True,
    include_inactive: bool = False,
) -> Node:
    """
    Get a node by name
    """
    statement = select(Node).where(Node.name == name)
    if not include_inactive:
        statement = statement.where(is_(Node.deactivated_at, None))
    if node_type:
        statement = statement.where(Node.type == node_type)
    if with_current:
        statement = statement.options(joinedload(Node.current)).options(
            joinedload(Node.tags),
        )
        result = await session.execute(statement)
        node = result.unique().scalar_one_or_none()
    else:
        result = await session.execute(statement)
        node = result.unique().scalar_one_or_none()
    if raise_if_not_exists:
        if not node:
            raise DJNodeNotFound(
                message=(
                    f"A {'' if not node_type else node_type + ' '}"
                    f"node with name `{name}` does not exist."
                ),
                http_status_code=404,
            )
    return node


async def raise_if_node_exists(session: AsyncSession, name: str) -> None:
    """
    Raise an error if the node with the given name already exists.
    """
    node = await Node.get_by_name(session, name, raise_if_not_exists=False)
    if node:
        raise DJAlreadyExistsException(
            message=f"A node with name `{name}` already exists.",
            http_status_code=HTTPStatus.CONFLICT,
        )


async def get_column(
    session: AsyncSession,
    node: NodeRevision,
    column_name: str,
) -> Column:
    """
    Get a column from a node revision
    """
    requested_column = None
    await session.refresh(node, ["columns"])
    for node_column in node.columns:
        if node_column.name == column_name:
            requested_column = node_column
            break

    if not requested_column:
        raise DJDoesNotExistException(
            message=f"Column {column_name} does not exist on node {node.name}",
            http_status_code=404,
        )
    return requested_column


async def get_attribute_type(
    session: AsyncSession,
    name: str,
    namespace: Optional[str] = RESERVED_ATTRIBUTE_NAMESPACE,
) -> Optional[AttributeType]:
    """
    Gets an attribute type by name.
    """
    statement = (
        select(AttributeType)
        .where(AttributeType.name == name)
        .where(AttributeType.namespace == namespace)
    )
    return (await session.execute(statement)).scalar_one_or_none()


async def get_catalog_by_name(session: AsyncSession, name: str) -> Catalog:
    """
    Get a catalog by name
    """
    statement = (
        select(Catalog).where(Catalog.name == name).options(joinedload(Catalog.engines))
    )
    catalog = (await session.execute(statement)).scalar()
    if not catalog:
        raise DJDoesNotExistException(
            message=f"Catalog with name `{name}` does not exist.",
            http_status_code=404,
        )
    return catalog


async def get_query(  # pylint: disable=too-many-arguments
    session: AsyncSession,
    node_name: str,
    dimensions: List[str],
    filters: List[str],
    orderby: List[str],
    limit: Optional[int] = None,
    engine: Optional[Engine] = None,
    access_control: Optional[access.AccessControlStore] = None,
) -> ast.Query:
    """
    Get a query for a metric, dimensions, and filters
    """
    node = await Node.get_by_name(session, node_name)

    # Builds the node for the engine's dialect if one is set or defaults to Spark
    if (
        not engine
        and node.current  # type: ignore
        and node.current.catalog  # type: ignore
        and node.current.catalog.engines  # type: ignore
    ):
        engine = node.current.catalog.engines[0]  # type: ignore
    build_criteria = BuildCriteria(
        dialect=(engine.dialect if engine and engine.dialect else Dialect.SPARK),
    )

    query_ast = await build_node(
        session=session,
        node=node.current,  # type: ignore
        filters=filters,
        dimensions=dimensions,
        orderby=orderby,
        limit=limit,
        build_criteria=build_criteria,
        access_control=access_control,
    )
    query_ast = rename_columns(query_ast, node.current)  # type: ignore
    return query_ast


def find_bound_dimensions(
    validated_node: NodeRevision,
    dependencies_map: Dict[NodeRevision, List[ast.Table]],
) -> Tuple[Set[str], List[Column]]:
    """
    Finds the matched required dimensions
    """
    invalid_required_dimensions = set()
    matched_bound_columns = []
    required_dimensions_mapping = {}
    for col in validated_node.required_dimensions:
        column_name = col.name if isinstance(col, Column) else col
        for parent in dependencies_map.keys():
            parent_columns = {
                parent_col.name: parent_col for parent_col in parent.columns
            }
            required_dimensions_mapping[column_name] = parent_columns.get(column_name)
    for column_name, required_column in required_dimensions_mapping.items():
        if required_column is not None:
            matched_bound_columns.append(required_column)
        else:
            invalid_required_dimensions.add(column_name)
    return invalid_required_dimensions, matched_bound_columns  # type: ignore


@dataclass
class NodeValidator:
    """
    Node validation
    """

    status: NodeStatus = NodeStatus.VALID
    columns: List[Column] = field(default_factory=list)
    required_dimensions: List[Column] = field(default_factory=list)
    dependencies_map: Dict[NodeRevision, List[ast.Table]] = field(default_factory=dict)
    missing_parents_map: Dict[str, List[ast.Table]] = field(default_factory=dict)
    type_inference_failures: List[str] = field(default_factory=list)
    errors: List[DJError] = field(default_factory=list)

    def differs_from(self, node_revision: NodeRevision):
        """
        Compared to the provided node revision, returns whether the validation
        indicates that the nodes differ.
        """
        if node_revision.status != self.status:
            return True
        existing_columns_map = {col.name: col for col in self.columns}
        for col in node_revision.columns:
            if col.name not in existing_columns_map:
                return True  # pragma: no cover
            if existing_columns_map[col.name].type != col.type:
                return True  # pragma: no cover
        return False

    def modified_columns(self, node_revision: NodeRevision):
        """
        Compared to the provided node revision, returns the modified columns
        """
        initial_node_columns = {col.name: col for col in node_revision.columns}
        updated_columns = set(initial_node_columns.keys()).difference(
            {n.name for n in self.columns},
        )
        for column in self.columns:
            if column.name in initial_node_columns:
                if initial_node_columns[column.name].type != column.type:
                    updated_columns.add(column.name)  # pragma: no cover
            else:  # pragma: no cover
                updated_columns.add(column.name)  # pragma: no cover
        return updated_columns


async def validate_node_data(  # pylint: disable=too-many-locals,too-many-statements
    data: Union[NodeRevisionBase, NodeRevision],
    session: AsyncSession,
) -> NodeValidator:
    """
    Validate a node. This function should never raise any errors.
    It will build the lists of issues (including errors) and return them all
    for the caller to decide what to do.
    """
    node_validator = NodeValidator()

    if isinstance(data, NodeRevision):
        validated_node = data
    else:
        node = Node(name=data.name, type=data.type)
        validated_node = NodeRevision(**data.dict())
        validated_node.node = node

    ctx = ast.CompileContext(session=session, exception=DJException())

    # Try to parse the node's query, extract dependencies and missing parents
    try:
        formatted_query = (
            NodeRevision.format_metric_alias(
                validated_node.query,  # type: ignore
                validated_node.name,
            )
            if validated_node.type == NodeType.METRIC
            else validated_node.query
        )
        query_ast = parse(formatted_query)  # type: ignore
        (
            dependencies_map,
            missing_parents_map,
        ) = await query_ast.bake_ctes().extract_dependencies(ctx)
        node_validator.dependencies_map = dependencies_map
        node_validator.missing_parents_map = missing_parents_map
    except (DJParseException, ValueError, SqlSyntaxError) as raised_exceptions:
        node_validator.status = NodeStatus.INVALID
        node_validator.errors.append(
            DJError(code=ErrorCode.INVALID_SQL_QUERY, message=str(raised_exceptions)),
        )
        return node_validator

    # Add aliases for any unnamed columns and confirm that all column types can be inferred
    query_ast.select.add_aliases_to_unnamed_columns()

    # Invalid parents will invalidate this node
    # Note: we include source nodes here because they sometimes appear to be invalid, but
    # this is a bug that needs to be fixed
    invalid_parents = {
        parent.name
        for parent in node_validator.dependencies_map
        if parent.type != NodeType.SOURCE and parent.status == NodeStatus.INVALID
    }
    if invalid_parents:
        node_validator.status = NodeStatus.INVALID

    try:
        column_mapping = {col.name: col for col in validated_node.columns}
    except MissingGreenlet:
        column_mapping = {}
    node_validator.columns = []
    type_inference_failures = {}
    for idx, col in enumerate(query_ast.select.projection):
        column = None
        column_name = col.alias_or_name.name  # type: ignore
        existing_column = column_mapping.get(column_name)
        try:
            column_type = str(col.type)  # type: ignore
            column = Column(
                name=column_name,
                display_name=labelize(column_name),
                type=column_type,
                attributes=existing_column.attributes if existing_column else [],
                dimension=existing_column.dimension if existing_column else None,
                order=idx,
            )
        except DJParseException as parse_exc:
            type_inference_failures[column_name] = parse_exc.message
            node_validator.status = NodeStatus.INVALID
        except TypeError:  # pragma: no cover
            type_inference_failures[
                column_name
            ] = f"Unknown TypeError on column {column_name}."
            node_validator.status = NodeStatus.INVALID
        if column:
            node_validator.columns.append(column)

    # check that bound dimensions are from parent nodes
    try:
        invalid_required_dimensions, matched_bound_columns = find_bound_dimensions(
            validated_node,
            dependencies_map,
        )
        node_validator.required_dimensions = matched_bound_columns
    except MissingGreenlet:
        invalid_required_dimensions = set()
        node_validator.required_dimensions = []

    if missing_parents_map or type_inference_failures or invalid_required_dimensions:
        # update status
        node_validator.status = NodeStatus.INVALID
        # build errors
        missing_parents_error = (
            [
                DJError(
                    code=ErrorCode.MISSING_PARENT,
                    message=f"Node definition contains references to nodes that do not "
                    f"exist: {','.join(missing_parents_map.keys())}",
                    debug={"missing_parents": list(missing_parents_map.keys())},
                ),
            ]
            if missing_parents_map
            else []
        )
        type_inference_error = (
            [
                DJError(
                    code=ErrorCode.TYPE_INFERENCE,
                    message=(
                        f"Unable to infer type for some columns on node `{data.name}`.\n"
                        + ("\n\t* " if type_inference_failures else "")
                        + "\n\t* ".join(
                            [val[:103] for val in type_inference_failures.values()],
                        )
                    ),
                    debug={
                        "columns": type_inference_failures,
                        "errors": ctx.exception.errors,
                    },
                ),
            ]
            if type_inference_failures
            else []
        )
        invalid_required_dimensions_error = (
            [
                DJError(
                    code=ErrorCode.INVALID_COLUMN,
                    message=(
                        "Node definition contains references to columns as "
                        "required dimensions that are not on parent nodes."
                    ),
                    debug={
                        "invalid_required_dimensions": list(
                            invalid_required_dimensions,
                        ),
                    },
                ),
            ]
            if invalid_required_dimensions
            else []
        )
        errors = (
            missing_parents_error
            + type_inference_error
            + invalid_required_dimensions_error
        )
        node_validator.errors.extend(errors)

    return node_validator


async def resolve_downstream_references(
    session: AsyncSession,
    node_revision: NodeRevision,
    current_user: Optional[User] = None,
) -> List[NodeRevision]:
    """
    Find all node revisions with missing parent references to `node` and resolve them
    """
    missing_parents = (
        (
            await session.execute(
                select(MissingParent).where(MissingParent.name == node_revision.name),
            )
        )
        .scalars()
        .all()
    )
    newly_valid_nodes = []
    for missing_parent in missing_parents:
        missing_parent_links = (
            (
                await session.execute(
                    select(NodeMissingParents).where(
                        NodeMissingParents.missing_parent_id == missing_parent.id,
                    ),
                )
            )
            .scalars()
            .all()
        )
        for (
            link
        ) in missing_parent_links:  # Remove from missing parents and add to parents
            downstream_node_id = link.referencing_node_id
            downstream_node_revision = (
                (
                    await session.execute(
                        select(NodeRevision)
                        .where(NodeRevision.id == downstream_node_id)
                        .options(
                            joinedload(NodeRevision.missing_parents),
                            joinedload(NodeRevision.parents),
                        ),
                    )
                )
                .unique()
                .scalar_one()
            )
            await session.refresh(node_revision, ["node"])
            await session.refresh(
                downstream_node_revision,
                ["parents", "missing_parents"],
            )
            downstream_node_revision.parents.append(node_revision.node)
            downstream_node_revision.missing_parents.remove(missing_parent)
            node_validator = await validate_node_data(
                data=downstream_node_revision,
                session=session,
            )
            event = None
            if downstream_node_revision.status != node_validator.status:
                event = status_change_history(
                    downstream_node_revision,
                    downstream_node_revision.status,
                    node_validator.status,
                    parent_node=node_revision.name,
                    current_user=current_user,
                )

            downstream_node_revision.status = node_validator.status
            downstream_node_revision.columns = node_validator.columns
            if node_validator.status == NodeStatus.VALID:
                newly_valid_nodes.append(downstream_node_revision)
            session.add(downstream_node_revision)
            if event:
                session.add(event)
            await session.commit()
            await session.refresh(downstream_node_revision)

        await session.delete(missing_parent)  # Remove missing parent reference to node
    return newly_valid_nodes


async def propagate_valid_status(
    session: AsyncSession,
    valid_nodes: List[NodeRevision],
    catalog_id: int,
    current_user: Optional[User] = None,
) -> None:
    """
    Propagate a valid status by revalidating all downstream nodes
    """
    while valid_nodes:
        resolved_nodes = []
        for node_revision in valid_nodes:
            if node_revision.status != NodeStatus.VALID:
                raise DJException(
                    f"Cannot propagate valid status: Node `{node_revision.name}` is not valid",
                )
            downstream_nodes = await get_downstream_nodes(
                session=session,
                node_name=node_revision.name,
            )
            newly_valid_nodes = []
            for node in downstream_nodes:
                node_validator = await validate_node_data(
                    data=node.current,
                    session=session,
                )
                node.current.status = node_validator.status
                if node_validator.status == NodeStatus.VALID:
                    node.current.columns = node_validator.columns or []
                    node.current.status = NodeStatus.VALID
                    node.current.catalog_id = catalog_id
                    session.add(
                        status_change_history(
                            node.current,
                            NodeStatus.INVALID,
                            NodeStatus.VALID,
                            current_user=current_user,
                        ),
                    )
                    newly_valid_nodes.append(node.current)
                session.add(node.current)
                await session.commit()
                await session.refresh(node.current)
            resolved_nodes.extend(newly_valid_nodes)
        valid_nodes = resolved_nodes


def map_dimensions_to_roles(dimensions: List[str]) -> Dict[str, str]:
    """
    Returns a mapping between dimension attributes and their roles.
    For example, ["default.users.user_id[user]"] would turn into
    {"default.users.user_id": "[user]"}
    """
    dimension_roles = [re.findall(COLUMN_NAME_REGEX, dim)[0] for dim in dimensions]
    return {dim_rols[0]: dim_rols[1] for dim_rols in dimension_roles}


async def validate_cube(  # pylint: disable=too-many-locals
    session: AsyncSession,
    metric_names: List[str],
    dimension_names: List[str],
    require_dimensions: bool = True,
) -> Tuple[List[Column], List[Node], List[Node], List[Column], Optional[Catalog]]:
    """
    Validate that a set of metrics and dimensions can be built together.
    """
    metrics_sorting_order = {val: idx for idx, val in enumerate(metric_names)}
    metric_nodes: List[Node] = sorted(
        await Node.get_by_names(
            session,
            metric_names,
            options=[
                joinedload(Node.current).options(
                    selectinload(NodeRevision.columns),
                    joinedload(NodeRevision.catalog),
                    selectinload(NodeRevision.parents),
                ),
            ],
        ),
        key=lambda x: metrics_sorting_order.get(x.name, 0),
    )

    # Verify that all metrics exist
    if len(metric_nodes) != len(metric_names):
        not_found = set(metric_names) - {metric.name for metric in metric_nodes}
        raise DJNodeNotFound(
            f"The following metric nodes were not found: {', '.join(not_found)}",
        )

    # Verify that all metrics are in valid status
    invalid_metrics = [
        metric.name
        for metric in metric_nodes
        if metric.current.status == NodeStatus.INVALID
    ]
    if invalid_metrics:
        raise DJInvalidInputException(
            f"The following metric nodes are invalid: {', '.join(invalid_metrics)}",
        )

    metrics: List[Column] = [metric.current.columns[0] for metric in metric_nodes]
    catalogs = [metric.current.catalog for metric in metric_nodes]
    catalog = catalogs[0] if catalogs else None

    # Verify that the provided metrics are metric nodes
    if not metrics:
        raise DJException(
            message=("At least one metric is required"),
            http_status_code=http.client.UNPROCESSABLE_ENTITY,
        )
    non_metrics = [metric for metric in metric_nodes if metric.type != NodeType.METRIC]
    if non_metrics:
        raise DJException(
            message=(
                f"Node {non_metrics[0].name} of type {non_metrics[0].type} "  # type: ignore
                f"cannot be added to a cube."
                + " Did you mean to add a dimension attribute?"
                if non_metrics[0].type == NodeType.DIMENSION  # type: ignore
                else ""
            ),
            http_status_code=http.client.UNPROCESSABLE_ENTITY,
        )

    # Verify that the provided dimension attributes exist
    dimension_attributes: List[List[str]] = [
        dimension_attribute.rsplit(".", 1) for dimension_attribute in dimension_names
    ]
    dimension_node_names = [node_name for node_name, _ in dimension_attributes]
    dimension_nodes: Dict[str, Node] = {
        node.name: node
        for node in await Node.get_by_names(
            session,
            dimension_node_names,
            options=[
                joinedload(Node.current).options(
                    selectinload(NodeRevision.columns).options(
                        joinedload(Column.node_revisions),
                    ),
                ),
            ],
        )
    }
    missing_dimensions = set(dimension_node_names) - set(dimension_nodes)
    if missing_dimensions:  # pragma: no cover
        missing_dimension_attributes = ", ".join(  # pragma: no cover
            [
                attr
                for node_name, attr in dimension_attributes
                if node_name in missing_dimensions
            ],
        )
        raise DJException(  # pragma: no cover
            f"Please make sure that `{missing_dimension_attributes}` "
            "is a dimensional attribute.",
        )

    dimension_mapping: Dict[str, Node] = {
        attr: dimension_nodes[node_name] for node_name, attr in dimension_attributes
    }
    dimensions: List[Column] = []
    for node_name, column_name in dimension_attributes:
        dimension_node = dimension_mapping[column_name]
        columns = {col.name: col for col in dimension_node.current.columns}  # type: ignore

        column_name_without_role = column_name
        match = re.fullmatch(COLUMN_NAME_REGEX, column_name)
        if match:
            column_name_without_role = match.groups()[0]

        if column_name_without_role in columns:  # pragma: no cover
            dimensions.append(columns[column_name_without_role])

    if require_dimensions and not dimensions:
        raise DJInvalidInputException(message="At least one dimension is required")

    if len(set(catalogs)) > 1:
        raise DJInvalidInputException(
            message=(
                f"Metrics and dimensions cannot be from multiple catalogs: {catalogs}"
            ),
        )

    if len(set(catalogs)) < 1:  # pragma: no cover
        raise DJInvalidInputException(
            message=("Metrics and dimensions must be part of a common catalog"),
        )

    await validate_shared_dimensions(
        session,
        metric_nodes,
        dimension_names,
        [],
    )
    return metrics, metric_nodes, list(dimension_nodes.values()), dimensions, catalog


async def get_history(
    session: AsyncSession,
    entity_type: EntityType,
    entity_name: str,
    offset: int,
    limit: int,
):
    """
    Get the history for a given entity type and name
    """
    return (
        (
            await session.execute(
                select(History)
                .where(History.entity_type == entity_type)
                .where(History.entity_name == entity_name)
                .offset(offset)
                .limit(limit)
                .order_by(History.created_at.desc()),
            )
        )
        .scalars()
        .all()
    )


def validate_orderby(
    orderby: List[str],
    metrics: List[str],
    dimension_attributes: List[str],
):
    """
    Validate that all elements in an order by match a metric or dimension attribute
    """
    invalid_orderbys = []
    for orderby_element in orderby:
        if orderby_element.split(" ")[0] not in metrics + dimension_attributes:
            invalid_orderbys.append(orderby_element)
    if invalid_orderbys:
        raise DJException(
            message=(
                f"Columns {invalid_orderbys} in order by clause must also be "
                "specified in the metrics or dimensions"
            ),
        )


async def find_existing_cube(
    session: AsyncSession,
    metric_columns: List[Column],
    dimension_columns: List[Column],
    materialized: bool = True,
) -> Optional[NodeRevision]:
    """
    Find an existing cube with these metrics and dimensions, if any.
    If `materialized` is set, it will only look for materialized cubes.
    """
    element_names = [col.name for col in (metric_columns + dimension_columns)]
    statement = select(Node).join(
        NodeRevision,
        onclause=(
            and_(
                (Node.id == NodeRevision.node_id),
                (Node.current_version == NodeRevision.version),
            )
        ),
    )
    for name in element_names:
        statement = statement.filter(
            NodeRevision.cube_elements.any(Column.name == name),  # type: ignore  # pylint: disable=no-member
        ).options(
            joinedload(Node.current).options(
                joinedload(NodeRevision.materializations),
                joinedload(NodeRevision.availability),
            ),
        )

    existing_cubes = (await session.execute(statement)).unique().scalars().all()
    for cube in existing_cubes:
        if not materialized or (  # pragma: no cover
            materialized and cube.current.materializations and cube.current.availability
        ):
            return cube.current
    return None


async def build_sql_for_multiple_metrics(  # pylint: disable=too-many-arguments,too-many-locals
    session: AsyncSession,
    metrics: List[str],
    dimensions: List[str],
    filters: List[str] = None,
    orderby: List[str] = None,
    limit: Optional[int] = None,
    engine_name: Optional[str] = None,
    engine_version: Optional[str] = None,
    access_control: Optional[access.AccessControlStore] = None,
    use_materialized: bool = True,
) -> Tuple[TranslatedSQL, Engine, Catalog]:
    """
    Build SQL for multiple metrics. Used by both /sql and /data endpoints
    """
    if not filters:
        filters = []
    if not orderby:
        orderby = []

    metric_columns, metric_nodes, _, dimension_columns, _ = await validate_cube(
        session,
        metrics,
        dimensions,
        require_dimensions=False,
    )
    leading_metric_node = await Node.get_by_name(
        session,
        metrics[0],
        options=[
            joinedload(Node.current).options(
                joinedload(NodeRevision.catalog).options(joinedload(Catalog.engines)),
            ),
        ],
    )
    available_engines = leading_metric_node.current.catalog.engines  # type: ignore

    # Try to find a built cube that already has the given metrics and dimensions
    # The cube needs to have a materialization configured and an availability state
    # posted in order for us to use the materialized datasource
    cube = await find_existing_cube(
        session,
        metric_columns,
        dimension_columns,
        materialized=True,
    )
    if cube:
        catalog = await get_catalog_by_name(session, cube.availability.catalog)  # type: ignore
        available_engines = catalog.engines + available_engines

    # Check if selected engine is available
    engine = (
        await get_engine(session, engine_name, engine_version)  # type: ignore
        if engine_name
        else available_engines[0]
    )
    if engine not in available_engines:
        raise DJInvalidInputException(  # pragma: no cover
            f"The selected engine is not available for the node {metrics[0]}. "
            f"Available engines include: {', '.join(engine.name for engine in available_engines)}",
        )

    validate_orderby(orderby, metrics, dimensions)

    if cube and cube.materializations and cube.availability and use_materialized:
        if access_control:  # pragma: no cover
            access_control.add_request_by_node(cube)
            access_control.state = access.AccessControlState.INDIRECT
            access_control.raise_if_invalid_requests()
        materialized_cube_catalog = await get_catalog_by_name(
            session,
            cube.availability.catalog,
        )
        query_ast = build_materialized_cube_node(  # pylint: disable=E1121
            metric_columns,
            dimension_columns,
            cube,
            filters,
            orderby,
            limit,
        )
        query_metric_columns = [
            ColumnMetadata(
                name=col.name,
                type=str(col.type),
                column=col.name,
                node=col.node_revision().name,  # type: ignore
            )
            for col in metric_columns
        ]
        query_dimension_columns = [
            ColumnMetadata(
                name=(col.node_revision().name + SEPARATOR + col.name).replace(  # type: ignore
                    SEPARATOR,
                    f"_{LOOKUP_CHARS.get(SEPARATOR)}_",
                ),
                type=str(col.type),
                node=col.node_revision().name,  # type: ignore
                column=col.name,  # type: ignore
            )
            for col in dimension_columns
        ]
        return (
            TranslatedSQL(
                sql=str(query_ast),
                columns=query_metric_columns + query_dimension_columns,
                dialect=materialized_cube_catalog.engines[0].dialect,
            ),
            engine,
            cube.catalog,
        )

    query_ast = await build_metric_nodes(
        session,
        metric_nodes,
        filters=filters or [],
        dimensions=dimensions or [],
        orderby=orderby or [],
        limit=limit,
        access_control=access_control,
    )
    columns = [
        assemble_column_metadata(col)  # type: ignore
        for col in query_ast.select.projection
    ]
    return (
        TranslatedSQL(
            sql=str(query_ast),
            columns=columns,
            dialect=engine.dialect if engine else None,
            upstream_tables=[
                f"{leading_metric_node.current.catalog.name}.{tbl.identifier()}"  # type: ignore
                for tbl in query_ast.find_all(ast.Table)
                if tbl.dj_node and tbl.dj_node.type == NodeType.SOURCE
            ],
        ),
        engine,
        leading_metric_node.current.catalog,  # type: ignore
    )


async def query_event_stream(  # pylint: disable=too-many-arguments
    query: QueryWithResults,
    query_service_client: QueryServiceClient,
    columns: List[Column],
    request,
    timeout: float = 0.0,
    stream_delay: float = 0.5,
    retry_timeout: int = 5000,
):
    """
    A generator of events from a query submitted to the query service
    """
    starting_time = time.time()
    # Start with query and query_next as the initial state of the query
    query_prev = query_next = query
    query_id = query_prev.id
    _logger.info("sending initial event to the client for query %s", query_id)
    yield {
        "event": "message",
        "id": uuid.uuid4(),
        "retry": retry_timeout,
        "data": json.dumps(query.json()),
    }
    # Continuously check the query until it's complete
    while not timeout or (time.time() - starting_time < timeout):
        # Check if the client closed the connection
        if await request.is_disconnected():  # pragma: no cover
            _logger.error("connection closed by the client")
            break

        # Check the current state of the query
        query_next = query_service_client.get_query(  # type: ignore # pragma: no cover
            query_id=query_id,
        )
        if query_next.state in END_JOB_STATES:  # pragma: no cover
            _logger.info(  # pragma: no cover
                "query end state detected (%s), sending final event to the client",
                query_next.state,
            )
            if query_next.results.__root__:  # pragma: no cover
                query_next.results.__root__[0].columns = columns or []
            yield {
                "event": "message",
                "id": uuid.uuid4(),
                "retry": retry_timeout,
                "data": json.dumps(query_next.json()),
            }
            _logger.info("connection closed by the server")
            break
        if query_prev != query_next:  # pragma: no cover
            _logger.info(
                "query information has changed, sending an event to the client",
            )
            yield {
                "event": "message",
                "id": uuid.uuid4(),
                "retry": retry_timeout,
                "data": json.dumps(query_next.json()),
            }

            query = query_next
        await asyncio.sleep(stream_delay)  # pragma: no cover


async def build_sql_for_dj_query(  # pylint: disable=too-many-arguments,too-many-locals
    session: AsyncSession,
    query: str,
    access_control: access.AccessControl,
    engine_name: Optional[str] = None,
    engine_version: Optional[str] = None,
) -> Tuple[TranslatedSQL, Engine, Catalog]:
    """
    Build SQL for multiple metrics. Used by /djsql endpoints
    """

    query_ast, dj_nodes = await build_dj_query(session, query)

    for node in dj_nodes:
        access_control.add_request_by_node(
            node.current,
        )

    access_control.validate_and_raise()

    leading_metric_node = dj_nodes[0]
    available_engines = leading_metric_node.current.catalog.engines

    # Check if selected engine is available
    engine = (
        await get_engine(session, engine_name, engine_version)  # type: ignore
        if engine_name
        else available_engines[0]
    )

    if engine not in available_engines:
        raise DJInvalidInputException(  # pragma: no cover
            f"The selected engine is not available for the node {leading_metric_node.name}. "
            f"Available engines include: {', '.join(engine.name for engine in available_engines)}",
        )

    columns = [
        ColumnMetadata(name=col.alias_or_name.name, type=str(col.type))  # type: ignore
        for col in query_ast.select.projection
    ]

    return (
        TranslatedSQL(
            sql=str(query_ast),
            columns=columns,
            dialect=engine.dialect if engine else None,
        ),
        engine,
        leading_metric_node.current.catalog,
    )


async def deactivate_node(
    session: AsyncSession,
    name: str,
    message: str = None,
    current_user: Optional[User] = None,
):
    """
    Deactivates a node and propagates to all downstreams.
    """
    node = await get_node_by_name(session, name, with_current=True)

    # Find all downstream nodes and mark them as invalid
    downstreams = await get_downstream_nodes(session, node.name)
    for downstream in downstreams:
        if downstream.current.status != NodeStatus.INVALID:
            downstream.current.status = NodeStatus.INVALID
            session.add(
                status_change_history(
                    downstream.current,
                    NodeStatus.VALID,
                    NodeStatus.INVALID,
                    parent_node=node.name,
                    current_user=current_user,
                ),
            )
            session.add(downstream)

    now = datetime.utcnow()
    node.deactivated_at = UTCDatetime(
        year=now.year,
        month=now.month,
        day=now.day,
        hour=now.hour,
        minute=now.minute,
        second=now.second,
    )
    session.add(node)
    session.add(
        History(
            entity_type=EntityType.NODE,
            entity_name=node.name,
            node=node.name,
            activity_type=ActivityType.DELETE,
            details={"message": message} if message else {},
            user=current_user.username if current_user else None,
        ),
    )
    await session.commit()
    await session.refresh(node, ["current"])


async def activate_node(
    session: AsyncSession,
    name: str,
    message: str = None,
    current_user: Optional[User] = None,
):
    """Restores node and revalidate all downstreams."""
    node = await get_node_by_name(
        session,
        name,
        with_current=True,
        include_inactive=True,
    )
    if not node.deactivated_at:
        raise DJException(
            http_status_code=HTTPStatus.BAD_REQUEST,
            message=f"Cannot restore `{name}`, node already active.",
        )
    node.deactivated_at = None  # type: ignore

    # Find all downstream nodes and revalidate them
    downstreams = await get_downstream_nodes(session, node.name)
    for downstream in downstreams:
        old_status = downstream.current.status
        if downstream.type == NodeType.CUBE:
            downstream.current.status = NodeStatus.VALID
            for element in downstream.current.cube_elements:
                await session.refresh(element, ["node_revisions"])
                if (
                    element.node_revisions
                    and element.node_revisions[-1].status == NodeStatus.INVALID
                ):  # pragma: no cover
                    downstream.current.status = NodeStatus.INVALID
        else:
            # We should not fail node restoration just because of some nodes
            # that have been invalid already and stay that way.
            node_validator = await validate_node_data(downstream.current, session)
            downstream.current.status = node_validator.status
            if node_validator.errors:
                downstream.current.status = NodeStatus.INVALID
        session.add(downstream)
        if old_status != downstream.current.status:
            session.add(
                status_change_history(
                    downstream.current,
                    old_status,
                    downstream.current.status,
                    parent_node=node.name,
                    current_user=current_user,
                ),
            )

    session.add(node)
    session.add(
        History(
            entity_type=EntityType.NODE,
            entity_name=node.name,
            node=node.name,
            activity_type=ActivityType.RESTORE,
            details={"message": message} if message else {},
            user=current_user.username if current_user else None,
        ),
    )
    await session.commit()


async def revalidate_node(
    name: str,
    session: AsyncSession,
    parent_node: str = None,
    current_user: Optional[User] = None,
):
    """
    Revalidate a single existing node and update its status appropriately
    """
    node = await Node.get_by_name(
        session,
        name,
        options=[
            joinedload(Node.current).options(*NodeRevision.default_load_options()),
            joinedload(Node.tags),
        ],
        raise_if_not_exists=True,
    )
    current_node_revision = node.current  # type: ignore
    if current_node_revision.type == NodeType.SOURCE:
        if current_node_revision.status != NodeStatus.VALID:  # pragma: no cover
            current_node_revision.status = NodeStatus.VALID
            session.add(
                status_change_history(
                    current_node_revision,
                    NodeStatus.INVALID,
                    NodeStatus.VALID,
                    current_user=current_user,
                ),
            )
            session.add(current_node_revision)
            await session.commit()
            await session.refresh(current_node_revision)
        return NodeStatus.VALID

    if current_node_revision.type == NodeType.CUBE:
        cube_node = await Node.get_cube_by_name(session, name)
        current_node_revision = cube_node.current  # type: ignore
        cube_metrics = [metric.name for metric in current_node_revision.cube_metrics()]
        cube_dimensions = current_node_revision.cube_dimensions()
        try:
            await validate_cube(
                session,
                metric_names=cube_metrics,
                dimension_names=cube_dimensions,
                require_dimensions=True,
            )
            current_node_revision.status = NodeStatus.VALID
        except DJException:  # pragma: no cover
            current_node_revision.status = NodeStatus.INVALID
        session.add(current_node_revision)
        await session.commit()
        return current_node_revision.status
    previous_status = current_node_revision.status
    node_validator = await validate_node_data(current_node_revision, session)

    node = await Node.get_by_name(
        session,
        name,
        options=[
            joinedload(Node.current).options(*NodeRevision.default_load_options()),
            joinedload(Node.tags),
        ],
        raise_if_not_exists=True,
    )
    node.current.status = node_validator.status  # type: ignore
    if previous_status != node.current.status:  # type: ignore  # pragma: no cover
        session.add(node)
        session.add(
            status_change_history(
                node.current,  # type: ignore
                previous_status,
                node.current.status,  # type: ignore
                parent_node=parent_node,
                current_user=current_user,
            ),
        )
        await session.commit()
    await session.refresh(node.current)  # type: ignore
    await session.refresh(node, ["current"])
    return node.current.status  # type: ignore


async def hard_delete_node(
    name: str,
    session: AsyncSession,
    current_user: Optional[User] = None,
):
    """
    Hard delete a node, destroying all links and invalidating all downstream nodes.
    This should be used with caution, deactivating a node is preferred.
    """
    node = await Node.get_by_name(
        session,
        name,
        options=[joinedload(Node.current), joinedload(Node.revisions)],
        include_inactive=True,
        raise_if_not_exists=False,
    )
    downstream_nodes = await get_downstream_nodes(session=session, node_name=name)

    linked_nodes = []
    if node.type == NodeType.DIMENSION:  # type: ignore
        linked_nodes = await get_nodes_with_dimension(
            session=session,
            dimension_node=node,  # type: ignore
        )

    await session.delete(node)
    await session.commit()
    impact = []  # Aggregate all impact of this deletion to include in response

    # Revalidate all downstream nodes
    for node in downstream_nodes:
        session.add(  # Capture this in the downstream node's history
            History(
                entity_type=EntityType.DEPENDENCY,
                entity_name=name,
                node=node.name,
                activity_type=ActivityType.DELETE,
                user=current_user.username if current_user else None,
            ),
        )
        status = await revalidate_node(
            name=node.name,
            session=session,
            parent_node=name,
            current_user=current_user,
        )
        impact.append(
            {
                "name": node.name,
                "status": status,
                "effect": "downstream node is now invalid",
            },
        )

    # Revalidate all linked nodes
    for node in linked_nodes:
        session.add(  # Capture this in the downstream node's history
            History(
                entity_type=EntityType.LINK,
                entity_name=name,
                node=node.name,
                activity_type=ActivityType.DELETE,
                user=current_user.username if current_user else None,
            ),
        )
        status = await revalidate_node(
            name=node.name,
            session=session,
            current_user=current_user,
        )
        impact.append(
            {
                "name": node.name,
                "status": status,
                "effect": "broken link",
            },
        )
    session.add(  # Capture this in the downstream node's history
        History(
            entity_type=EntityType.NODE,
            entity_name=name,
            node=name,
            activity_type=ActivityType.DELETE,
            details={
                "impact": impact,
            },
            user=current_user.username if current_user else None,
        ),
    )
    await session.commit()  # Commit the history events
    return impact


def assemble_column_metadata(
    column: ast.Column,
    # node_name: Union[List[str], str],
) -> ColumnMetadata:
    """
    Extract column metadata from AST
    """
    metadata = ColumnMetadata(
        name=column.alias_or_name.name,
        type=str(column.type),
        column=(
            column.semantic_entity.split(SEPARATOR)[-1]
            if hasattr(column, "semantic_entity") and column.semantic_entity
            else None
        ),
        node=(
            SEPARATOR.join(column.semantic_entity.split(SEPARATOR)[:-1])
            if hasattr(column, "semantic_entity") and column.semantic_entity
            else None
        ),
        semantic_entity=column.semantic_entity
        if hasattr(column, "semantic_entity")
        else None,
        semantic_type=column.semantic_type
        if hasattr(column, "semantic_type")
        else None,
    )
    return metadata
