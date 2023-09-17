"""Nodes endpoint helper functions"""
import logging
from collections import defaultdict, deque
from http import HTTPStatus
from typing import Any, Dict, List, Optional

from fastapi import Depends
from sqlmodel import Session, select

from datajunction_server.api.helpers import (
    activate_node,
    get_attribute_type,
    get_engine,
    get_node_by_name,
    propagate_valid_status,
    resolve_downstream_references,
    validate_cube,
    validate_node_data,
)
from datajunction_server.construction.build import build_metric_nodes
from datajunction_server.errors import DJDoesNotExistException, DJError, DJException
from datajunction_server.internal.materializations import (
    build_cube_config,
    create_new_materialization,
    schedule_materialization_jobs,
)
from datajunction_server.materialization.jobs import (
    DefaultCubeMaterialization,
    DruidCubeMaterializationJob,
)
from datajunction_server.models import (
    AttributeType,
    Column,
    ColumnAttribute,
    History,
    Node,
    NodeRevision,
)
from datajunction_server.models.attribute import (
    AttributeTypeIdentifier,
    UniquenessScope,
)
from datajunction_server.models.base import labelize
from datajunction_server.models.history import (
    ActivityType,
    EntityType,
    status_change_history,
)
from datajunction_server.models.materialization import (
    DruidCubeConfig,
    Materialization,
    UpsertMaterialization,
)
from datajunction_server.models.node import (
    DEFAULT_DRAFT_VERSION,
    DEFAULT_PUBLISHED_VERSION,
    CreateCubeNode,
    CreateNode,
    CreateSourceNode,
    LineageColumn,
    MissingParent,
    NodeMode,
    NodeStatus,
    NodeType,
    UpdateNode,
)
from datajunction_server.service_clients import QueryServiceClient
from datajunction_server.sql.parsing import ast
from datajunction_server.sql.parsing.ast import CompileContext
from datajunction_server.sql.parsing.backends.antlr4 import parse
from datajunction_server.sql.parsing.backends.exceptions import DJParseException
from datajunction_server.utils import (
    Version,
    VersionUpgrade,
    get_query_service_client,
    get_session,
)

_logger = logging.getLogger(__name__)


def get_node_column(node: Node, column_name: str) -> Column:
    """
    Gets the specified column on a node
    """
    column_map = {column.name: column for column in node.current.columns}
    if column_name not in column_map:
        raise DJDoesNotExistException(
            message=f"Column `{column_name}` does not exist on node `{node.name}`!",
        )
    column = column_map[column_name]
    return column


def validate_column_attributes(
    session: Session,
    column: Column,
    attribute: AttributeTypeIdentifier,
    node: Node,
) -> ColumnAttribute:
    """
    Run some validation and build column attribute.
    """
    existing_attributes = {attr.attribute_type.name: attr for attr in column.attributes}
    if attribute.name in existing_attributes:
        return existing_attributes[attribute.name]

    # Verify attribute type exists
    attribute_type = get_attribute_type(
        session,
        attribute.name,
        attribute.namespace,
    )
    if not attribute_type:
        raise DJDoesNotExistException(
            message=f"Attribute type `{attribute.namespace}"
            f".{attribute.name}` "
            f"does not exist!",
        )

    # Verify that the attribute type is allowed for this node
    if node.type not in attribute_type.allowed_node_types:
        raise DJException(
            message=f"Attribute type `{attribute.namespace}.{attribute_type.name}` "
            f"not allowed on node type `{node.type}`!",
        )

    return ColumnAttribute(
        attribute_type=attribute_type,
        column=column,
    )


def set_node_column_attributes(
    session: Session,
    node: Node,
    column_name: str,
    attributes: List[AttributeTypeIdentifier],
) -> List[Column]:
    """
    Sets the column attributes on the node if allowed.
    """
    column = get_node_column(node, column_name)
    all_columns_map = {column.name: column for column in node.current.columns}

    old_column_attributes = column.attributes
    column.attributes = []
    for attribute in attributes:
        column.attributes.append(
            validate_column_attributes(session, column, attribute, node),
        )

    # Validate column attributes by building mapping between
    # attribute scope and columns
    attributes_columns_map = defaultdict(set)
    all_columns = all_columns_map.values()

    for _col in all_columns:
        for attribute in _col.attributes:
            scopes_map = {
                UniquenessScope.NODE: attribute.attribute_type,
                UniquenessScope.COLUMN_TYPE: _col.type,
            }
            attributes_columns_map[
                (  # type: ignore
                    attribute.attribute_type,
                    tuple(
                        scopes_map[item]
                        for item in attribute.attribute_type.uniqueness_scope
                    ),
                )
            ].add(_col.name)

    for (attribute, _), columns in attributes_columns_map.items():
        if len(columns) > 1 and attribute.uniqueness_scope:
            column.attributes = old_column_attributes
            raise DJException(
                message=f"The column attribute `{attribute.name}` is scoped to be "
                f"unique to the `{attribute.uniqueness_scope}` level, but there "
                "is more than one column tagged with it: "
                f"`{', '.join(sorted(list(columns)))}`",
            )

    session.add(column)
    session.add(
        History(
            entity_type=EntityType.COLUMN_ATTRIBUTE,
            node=node.name,
            activity_type=ActivityType.SET_ATTRIBUTE,
            details={
                "column": column.name,
                "attributes": [attr.dict() for attr in attributes],
            },
        ),
    )
    session.commit()
    session.refresh(column)

    session.refresh(node)
    session.refresh(node.current)
    return [column]


def create_node_revision(
    data: CreateNode,
    node_type: NodeType,
    session: Session,
) -> NodeRevision:
    """
    Create a non-source node revision.
    """
    node_revision = NodeRevision(
        name=data.name,
        namespace=data.namespace,
        display_name=data.display_name if data.display_name else labelize(data.name),
        description=data.description,
        type=node_type,
        status=NodeStatus.VALID,
        query=data.query,
        mode=data.mode,
        required_dimensions=data.required_dimensions or [],
    )
    node_validator = validate_node_data(node_revision, session)
    print("node_validator", node_revision.name, node_validator.status)
    if node_validator.status == NodeStatus.INVALID:
        if node_revision.mode == NodeMode.DRAFT:
            node_revision.status = NodeStatus.INVALID
        else:
            raise DJException(
                http_status_code=HTTPStatus.BAD_REQUEST,
                errors=node_validator.errors,
            )
    else:
        node_revision.status = NodeStatus.VALID
    node_revision.missing_parents = [
        MissingParent(name=missing_parent)
        for missing_parent in node_validator.missing_parents_map
    ]
    node_revision.required_dimensions = node_validator.required_dimensions
    new_parents = [node.name for node in node_validator.dependencies_map]
    catalog_ids = [node.catalog_id for node in node_validator.dependencies_map]
    if node_revision.mode == NodeMode.PUBLISHED and not len(set(catalog_ids)) <= 1:
        raise DJException(
            f"Cannot create nodes with multi-catalog dependencies: {set(catalog_ids)}",
        )
    catalog_id = next(iter(catalog_ids), 0)
    parent_refs = session.exec(
        select(Node).where(
            # pylint: disable=no-member
            Node.name.in_(  # type: ignore
                new_parents,
            ),
        ),
    ).all()
    node_revision.parents = parent_refs

    _logger.info(
        "Parent nodes for %s (%s): %s",
        data.name,
        node_revision.version,
        [p.name for p in node_revision.parents],
    )
    node_revision.columns = node_validator.columns or []
    node_revision.catalog_id = catalog_id
    return node_revision


def create_cube_node_revision(  # pylint: disable=too-many-locals
    session: Session,
    data: CreateCubeNode,
) -> NodeRevision:
    """
    Create a cube node revision.
    """
    (
        metric_columns,
        metric_nodes,
        dimension_nodes,
        dimension_columns,
        catalog,
    ) = validate_cube(
        session,
        data.metrics,
        data.dimensions,
    )

    combined_ast = build_metric_nodes(
        session,
        metric_nodes,
        filters=data.filters or [],
        dimensions=data.dimensions or [],
        orderby=data.orderby or [],
        limit=data.limit or None,
    )
    dimension_attribute = session.exec(
        select(AttributeType).where(AttributeType.name == "dimension"),
    ).one()
    dimensions_set = {dim.rsplit(".", 1)[1] for dim in data.dimensions}

    node_columns = []
    status = NodeStatus.VALID
    type_inference_failed_columns = []
    for col in combined_ast.select.projection:
        try:
            column_type = col.type  # type: ignore
            column_attributes = (
                [ColumnAttribute(attribute_type=dimension_attribute)]
                if col.alias_or_name.name in dimensions_set
                else []
            )
            node_columns.append(
                Column(
                    name=col.alias_or_name.name,
                    type=column_type,
                    attributes=column_attributes,
                ),
            )
        except DJParseException:  # pragma: no cover
            type_inference_failed_columns.append(col.alias_or_name.name)  # type: ignore
            status = NodeStatus.INVALID

    node_revision = NodeRevision(
        name=data.name,
        namespace=data.namespace,
        description=data.description,
        type=NodeType.CUBE,
        query=str(combined_ast),
        columns=node_columns,
        cube_elements=metric_columns + dimension_columns,
        parents=list(set(dimension_nodes + metric_nodes)),
        status=status,
        catalog=catalog,
    )

    # Set up a default materialization for the cube. Note that this does not get used
    # for any actual materialization, but is for storing info needed for materialization
    node_revision.materializations = []
    default_materialization = UpsertMaterialization(
        name="placeholder",
        engine=node_revision.catalog.engines[0],  # pylint: disable=no-member
        schedule="@daily",
        config={},
        job="CubeMaterializationJob",
    )
    engine = get_engine(
        session,
        name=default_materialization.engine.name,
        version=default_materialization.engine.version,
    )
    cube_custom_config = build_cube_config(
        node_revision,
        combined_ast,
    )
    new_materialization = Materialization(
        name=cube_custom_config.identifier(),
        node_revision=node_revision,
        engine=engine,
        config=cube_custom_config,
        schedule=default_materialization.schedule,
        job=(
            DefaultCubeMaterialization.__name__
            if not isinstance(cube_custom_config, DruidCubeConfig)
            else DruidCubeMaterializationJob.__name__
        ),
    )
    node_revision.materializations.append(new_materialization)
    return node_revision


def save_node(
    session: Session,
    node_revision: NodeRevision,
    node: Node,
    node_mode: NodeMode,
):
    """
    Links the node and node revision together and saves them
    """
    node_revision.node = node
    node_revision.version = (
        str(DEFAULT_DRAFT_VERSION)
        if node_mode == NodeMode.DRAFT
        else str(DEFAULT_PUBLISHED_VERSION)
    )
    node.current_version = node_revision.version
    node_revision.extra_validation()

    node_revision = add_lineage_to_node(session, node_revision)

    session.add(node)
    session.add(
        History(
            node=node.name,
            entity_type=EntityType.NODE,
            entity_name=node.name,
            activity_type=ActivityType.CREATE,
        ),
    )
    session.commit()

    newly_valid_nodes = resolve_downstream_references(
        session=session,
        node_revision=node_revision,
    )
    propagate_valid_status(
        session=session,
        valid_nodes=newly_valid_nodes,
        catalog_id=node.current.catalog_id,  # pylint: disable=no-member
    )
    session.refresh(node.current)


def add_lineage_to_node(session: Session, node_revision: NodeRevision):
    if node_revision.status == NodeStatus.VALID and node_revision.type not in (
        NodeType.SOURCE,
        NodeType.CUBE,
    ):
        node_revision.lineage = [
            lineage.dict()
            for lineage in get_column_level_lineage(session, node_revision)
        ]
    else:
        node_revision.lineage = []
    return node_revision


def _update_node(
    name: str,
    data: UpdateNode,
    session: Session,
    query_service_client: QueryServiceClient = Depends(get_query_service_client),
):
    """
    Update the named node with the changes defined in the UpdateNode object.
    Propagate these changes to all of the node's downstream children.
    """
    node = get_node_by_name(session, name, for_update=True, include_inactive=True)
    old_revision = node.current
    new_revision = create_new_revision_from_existing(
        session,
        query_service_client,
        old_revision,
        node,
        data,
    )

    if not new_revision:
        return node  # type: ignore

    node.current_version = new_revision.version

    new_revision.extra_validation()
    new_revision = add_lineage_to_node(session, new_revision)
    session.add(new_revision)
    session.add(node)

    session.add(
        History(
            entity_type=EntityType.NODE,
            entity_name=node.name,
            node=node.name,
            activity_type=ActivityType.UPDATE,
            details={
                "version": new_revision.version,
            },
        ),
    )

    if new_revision.status != old_revision.status:
        session.add(
            status_change_history(
                new_revision,
                old_revision.status,
                new_revision.status,
            ),
        )
    session.commit()
    session.refresh(new_revision)
    session.refresh(node)

    history_events = {}
    old_columns_map = {col.name: col.type for col in old_revision.columns}
    history_events[node.name] = {
        "name": node.name,
        "current_version": node.current_version,
        "previous_version": old_revision.version,
        "updated_columns": [
            col.name
            for col in new_revision.columns
            if col.name not in old_columns_map or old_columns_map[col.name] != col.type
        ],
    }
    propagate_update_downstream(session, node, history_events)
    session.refresh(node.current)
    return node


def propagate_update_downstream(
    session: Session,
    node: Node,
    history_events: Dict[str, Any],
):
    """
    Propagate the updated node's changes to all of its downstream children.

    Some potential changes to the upstream node and their effects on downstream nodes:
    - altered column names: may invalidate downstream nodes
    - altered column types: may invalidate downstream nodes
    - new columns: won't affect downstream nodes
    """
    processed = set()

    # Each entry being processed is a list that represents the changelog of affected nodes
    # The last entry in the list is the current node that's being processed
    to_process = deque([[node.current, child] for child in node.children])

    while to_process:
        changelog = to_process.popleft()
        child = changelog[-1]

        # Only process if it hasn't already been processed before
        if child.name not in processed:
            processed.add(child.name)

            node_validator = validate_node_data(
                data=child,
                session=session,
            )

            # Update the child only if its columns or status have changed
            if node_validator.differs_from(child):
                # Create a new node revision
                new_revision = copy_existing_node_revision(child)
                new_revision.version = str(
                    Version.parse(child.version).next_major_version(),
                )

                initial_node_columns = {col.name: col for col in new_revision.columns}
                new_revision.status = node_validator.status
                new_revision = add_lineage_to_node(session, new_revision)

                # Update any columns that are new or have modified types
                updated_columns = []
                for column in node_validator.columns:
                    if column.name in initial_node_columns:
                        if initial_node_columns[column.name].type != column.type:
                            updated_columns.append(column.name)
                            initial_node_columns[column.name].type = column.type
                    else:
                        updated_columns.append(column.name)
                        new_revision.columns.append(
                            Column(name=column.name, type=column.type),
                        )

                # Save the new revision of the child
                new_revision.node = child.node
                new_revision.node.current_version = new_revision.version
                session.add(new_revision)
                session.add(new_revision.node)

                # Add grandchildren for processing
                for grandchild in child.node.children:
                    new_changelog = changelog + [grandchild]
                    to_process.append(new_changelog)

                # Record history event
                history_events[child.name] = {
                    "name": child.name,
                    "current_version": new_revision.version,
                    "previous_version": child.version,
                    "updated_columns": updated_columns,
                }
                event = History(
                    entity_type=EntityType.NODE,
                    entity_name=child.name,
                    node=child.name,
                    activity_type=ActivityType.STATUS_CHANGE,
                    details={
                        "upstreams": [
                            history_events[entry.name]
                            for entry in changelog
                            if entry.name in history_events
                        ],
                        "reason": f"Caused by update of `{node.name}` to {node.current_version}",
                    },
                    pre={"status": child.status},
                    post={"status": node_validator.status},
                )
                session.add(event)
                session.commit()
                session.refresh(new_revision)
                session.refresh(new_revision.node)
                session.refresh(new_revision.node.current)


def _create_node_from_inactive(
    new_node_type: NodeType,
    data: CreateSourceNode,
    session: Session = Depends(get_session),
) -> Optional[Node]:
    """
    If the node existed and is inactive the re-creation takes different steps than
    creating it from scratch.
    """
    previous_inactive_node = get_node_by_name(
        session,
        name=data.name,
        raise_if_not_exists=False,
        include_inactive=True,
    )
    if previous_inactive_node and previous_inactive_node.deactivated_at:
        if previous_inactive_node.type != new_node_type:
            raise DJException(  # pragma: no cover
                message=f"A node with name `{data.name}` of a `{previous_inactive_node.type.value}` "  # pylint: disable=line-too-long
                "type existed before. If you want to re-created with a different type now, "
                "you need to remove all the traces of the previous node with a <TODO> command.",
                http_status_code=HTTPStatus.CONFLICT,
            )
        _update_node(
            name=data.name,
            data=UpdateNode(
                # MutableNodeFields
                display_name=data.display_name,
                description=data.description,
                mode=data.mode,
                # SourceNodeFields
                catalog=data.catalog,
                schema_=data.schema_,
                table=data.table,
                columns=data.columns,
            ),
            session=session,
        )
        try:
            activate_node(name=data.name, session=session)
            return get_node_by_name(session, data.name, with_current=True)
        except Exception as exc:  # pragma: no cover
            raise DJException(
                f"Restoring node `{data.name}` failed: {exc}",
            ) from exc

    return None


def copy_existing_node_revision(old_revision: NodeRevision):
    """
    Create an exact copy of the node revision
    """
    node = old_revision.node
    return NodeRevision(
        name=old_revision.name,
        node_id=node.id,
        version=old_revision.version,
        display_name=old_revision.display_name,
        description=old_revision.description,
        query=old_revision.query,
        type=old_revision.type,
        columns=old_revision.columns,
        catalog=old_revision.catalog,
        schema_=old_revision.schema_,
        table=old_revision.table,
        parents=old_revision.parents,
        mode=old_revision.mode,
        materializations=old_revision.materializations,
        status=old_revision.status,
    )


def create_new_revision_from_existing(  # pylint: disable=too-many-locals,too-many-arguments,too-many-branches
    session: Session,
    query_service_client: QueryServiceClient,
    old_revision: NodeRevision,
    node: Node,
    data: UpdateNode = None,
    version_upgrade: VersionUpgrade = None,
) -> Optional[NodeRevision]:
    """
    Creates a new revision from an existing node revision.
    """
    minor_changes = (
        (data and data.description and old_revision.description != data.description)
        or (data and data.mode and old_revision.mode != data.mode)
        or (
            data
            and data.display_name
            and old_revision.display_name != data.display_name
        )
    )

    if node.type == NodeType.METRIC:
        data.query = NodeRevision.format_metric_alias(data.query, node.name)  # type: ignore

    query_changes = (
        old_revision.type != NodeType.SOURCE
        and data
        and data.query
        and old_revision.query != data.query
    )
    column_changes = (
        old_revision.type == NodeType.SOURCE
        and data is not None
        and data.columns is not None
        and ({col.identifier() for col in old_revision.columns} != data.columns)
    )
    pk_changes = (
        data is not None
        and data.primary_key
        and {col.name for col in old_revision.primary_key()} != set(data.primary_key)
    )
    major_changes = query_changes or column_changes or pk_changes

    # If nothing has changed, do not create the new node revision
    if not minor_changes and not major_changes and not version_upgrade:
        return None

    old_version = Version.parse(node.current_version)
    new_mode = data.mode if data and data.mode else old_revision.mode
    new_revision = NodeRevision(
        name=old_revision.name,
        node_id=node.id,
        version=str(
            old_version.next_major_version()
            if major_changes or version_upgrade == VersionUpgrade.MAJOR
            else old_version.next_minor_version(),
        ),
        display_name=(
            data.display_name
            if data and data.display_name
            else old_revision.display_name
        ),
        description=(
            data.description if data and data.description else old_revision.description
        ),
        query=(data.query if data and data.query else old_revision.query),
        type=old_revision.type,
        columns=[
            Column(
                name=column_data.name,
                type=column_data.type,
                dimension_column=column_data.dimension,
                attributes=column_data.attributes or [],
            )
            for column_data in data.columns
        ]
        if data and data.columns
        else old_revision.columns,
        catalog=old_revision.catalog,
        schema_=old_revision.schema_,
        table=old_revision.table,
        parents=[],
        mode=new_mode,
        materializations=[],
        status=old_revision.status,
    )

    # Link the new revision to its parents if a new revision was created and update its status
    if new_revision.type != NodeType.SOURCE:
        node_validator = validate_node_data(new_revision, session)
        new_revision.status = node_validator.status

        if node_validator.errors:
            if new_mode == NodeMode.DRAFT:
                new_revision.status = NodeStatus.INVALID
            else:
                raise DJException(
                    http_status_code=HTTPStatus.BAD_REQUEST,
                    errors=node_validator.errors,
                )

        # Keep the dimension links and attributes on the columns from the node's
        # last revision if any existed
        # old_columns_mapping = {col.name: col for col in old_revision.columns}
        # for col in node_validator.columns:
        #     if col.name in old_columns_mapping:
        #         col.dimension_id = old_columns_mapping[col.name].dimension_id
        #         col.attributes = old_columns_mapping[col.name].attributes or []

        new_parents = [n.name for n in node_validator.dependencies_map]
        parent_refs = session.exec(
            select(Node).where(
                # pylint: disable=no-member
                Node.name.in_(  # type: ignore
                    new_parents,
                ),
            ),
        ).all()
        new_revision.parents = list(parent_refs)
        new_revision.columns = node_validator.columns or []

        # Update the primary key if one was set in the input
        if data is not None and data.primary_key:
            pk_attribute = session.exec(
                select(AttributeType).where(AttributeType.name == "primary_key"),
            ).one()
            for col in new_revision.columns:
                # Remove the primary key attribute if it's not in the updated PK
                if col.has_primary_key_attribute() and col.name not in data.primary_key:
                    col.attributes = [
                        attr
                        for attr in col.attributes
                        if attr.attribute_type.name != "primary_key"
                    ]
                # Add (or keep) the primary key attribute if it is in the updated PK
                if col.name in data.primary_key and not col.has_primary_key_attribute():
                    col.attributes.append(
                        ColumnAttribute(column=col, attribute_type=pk_attribute),
                    )

        # Set the node's validity status
        invalid_primary_key = (
            new_revision.type == NodeType.DIMENSION and not new_revision.primary_key()
        )
        if invalid_primary_key:
            new_revision.status = NodeStatus.INVALID

        new_revision.missing_parents = [
            MissingParent(name=missing_parent)
            for missing_parent in node_validator.missing_parents_map
        ]
        _logger.info(
            "Parent nodes for %s (v%s): %s",
            new_revision.name,
            new_revision.version,
            [p.name for p in new_revision.parents],
        )
        new_revision.required_dimensions = node_validator.required_dimensions

    # Handle materializations
    active_materializations = [
        mat for mat in old_revision.materializations if not mat.deactivated_at
    ]
    if active_materializations and query_changes:
        for old in active_materializations:
            new_revision.materializations.append(
                create_new_materialization(
                    session,
                    new_revision,
                    UpsertMaterialization(
                        **old.dict(), **{"engine": old.engine.dict()}
                    ),
                ),
            )
        schedule_materialization_jobs(
            new_revision.materializations,
            query_service_client,
        )
    return new_revision


def get_column_level_lineage(
    session: Session,
    node_revision: NodeRevision,
) -> List[LineageColumn]:
    """
    Gets the column-level lineage for the node
    """
    return [
        column_lineage(
            session,
            node_revision,
            col.name,
        )
        for col in node_revision.columns
    ]


def column_lineage(
    session: Session,
    node_rev: NodeRevision,
    column_name: str,
) -> LineageColumn:
    """
    Helper function to determine the lineage for a column on a node.
    """
    if node_rev.type == NodeType.SOURCE:
        return LineageColumn(
            node_name=node_rev.name,
            node_type=node_rev.type,
            display_name=node_rev.display_name,
            column_name=column_name,
            lineage=[],
        )

    ctx = CompileContext(session, DJException())
    query_ast = parse(node_rev.query)
    query_ast.compile(ctx)
    query_ast.select.add_aliases_to_unnamed_columns()

    lineage_column = LineageColumn(
        column_name=column_name,
        node_name=node_rev.name,
        node_type=node_rev.type,
        display_name=node_rev.display_name,
        lineage=[],
    )

    # Find the expression AST for the column on the node
    column = [
        col
        for col in query_ast.select.projection
        if (  # pragma: no cover
            col != ast.Null() and col.alias_or_name.name == column_name  # type: ignore
        )
    ][0]
    column_or_child = column.child if isinstance(column, ast.Alias) else column  # type: ignore
    column_expr = (
        column_or_child.expression  # type: ignore
        if hasattr(column_or_child, "expression") and column_or_child.expression
        else column_or_child
    )

    # At every layer, expand the lineage search tree with all columns referenced
    # by the current column's expression. If we reach an actual table with a DJ
    # node attached, save this to the lineage record. Otherwise, continue the search
    processed = list(column_expr.find_all(ast.Column))
    seen = set()
    while processed:
        current = processed.pop()
        if current in seen:
            continue
        if (
            hasattr(current, "table")
            and isinstance(current.table, ast.Table)
            and current.table.dj_node
        ):
            lineage_column.lineage.append(  # type: ignore
                column_lineage(
                    session,
                    current.table.dj_node,
                    current.name.name
                    if not current.is_struct_ref
                    else current.struct_column_name,
                ),
            )
        else:
            expr_column_deps = (
                list(
                    current.expression.find_all(ast.Column),
                )
                if current.expression
                else []
            )
            for col_dep in expr_column_deps:
                processed.append(col_dep)
        seen.update({current})
    return lineage_column
