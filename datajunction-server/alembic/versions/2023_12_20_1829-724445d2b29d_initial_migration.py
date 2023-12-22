"""Initial migration

Revision ID: 724445d2b29d
Revises:
Create Date: 2023-12-20 18:29:11.681799+00:00

"""
# pylint: disable=no-member, invalid-name, missing-function-docstring, unused-import, no-name-in-module

import sqlalchemy as sa
import sqlalchemy_utils

from alembic import op

# revision identifiers, used by Alembic.
revision = "724445d2b29d"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "attributetype",
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("allowed_node_types", sa.JSON(), nullable=True),
        sa.Column("uniqueness_scope", sa.JSON(), nullable=True),
        sa.Column("namespace", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_attributetype")),
        sa.UniqueConstraint(
            "namespace",
            "name",
            name=op.f("uq_attributetype_namespace"),
        ),
    )
    op.create_table(
        "availabilitystate",
        sa.Column("categorical_partitions", sa.JSON(), nullable=True),
        sa.Column("temporal_partitions", sa.JSON(), nullable=True),
        sa.Column("min_temporal_partition", sa.JSON(), nullable=True),
        sa.Column("max_temporal_partition", sa.JSON(), nullable=True),
        sa.Column("partitions", sa.JSON(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("catalog", sa.String(), nullable=False),
        sa.Column("schema_", sa.String(), nullable=True),
        sa.Column("table", sa.String(), nullable=False),
        sa.Column("valid_through_ts", sa.Integer(), nullable=False),
        sa.Column("url", sa.String(), nullable=True),
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_availabilitystate")),
    )
    op.create_table(
        "catalog",
        sa.Column("uuid", sqlalchemy_utils.types.uuid.UUIDType(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("extra_params", sa.JSON(), nullable=True),
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_catalog")),
    )
    op.create_table(
        "node",
        sa.Column("name", sa.String(), nullable=True),
        sa.Column(
            "type",
            sa.Enum(
                "SOURCE",
                "TRANSFORM",
                "METRIC",
                "DIMENSION",
                "CUBE",
                name="nodetype",
            ),
            nullable=True,
        ),
        sa.Column("display_name", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deactivated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("namespace", sa.String(), nullable=True),
        sa.Column("current_version", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_node")),
        sa.UniqueConstraint("name", "namespace", name="unique_node_namespace_name"),
        sa.UniqueConstraint("name", name=op.f("uq_node_name")),
    )
    op.create_table(
        "measures",
        sa.Column("display_name", sa.String(), nullable=True),
        sa.Column(
            "additive",
            sa.Enum(
                "ADDITIVE",
                "NON_ADDITIVE",
                "SEMI_ADDITIVE",
                name="aggregationrule",
            ),
            nullable=True,
        ),
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_measures")),
        sa.UniqueConstraint("name", name=op.f("uq_measures_name")),
    )
    op.create_table(
        "partition",
        sa.Column("type_", sa.String(), nullable=False),
        sa.Column("granularity", sa.String(), nullable=True),
        sa.Column("format", sa.String(), nullable=True),
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("column_id", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_partition")),
    )
    op.create_table(
        "column",
        sa.Column("display_name", sa.String(), nullable=True),
        sa.Column("type", sa.String(), nullable=True),
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("dimension_id", sa.BigInteger(), nullable=True),
        sa.Column("dimension_column", sa.String(), nullable=True),
        sa.Column("measure_id", sa.BigInteger(), nullable=True),
        sa.Column("partition_id", sa.BigInteger(), nullable=True),
        sa.ForeignKeyConstraint(
            ["dimension_id"],
            ["node.id"],
            name=op.f("fk_column_dimension_id_node"),
        ),
        sa.ForeignKeyConstraint(
            ["measure_id"],
            ["measures.id"],
            name=op.f("fk_column_measure_id_measures"),
        ),
        sa.ForeignKeyConstraint(
            ["partition_id"],
            ["partition.id"],
            name=op.f("fk_column_partition_id_partition"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_column")),
    )
    with op.batch_alter_table("partition", schema=None) as batch_op:
        batch_op.create_foreign_key(
            "fk_partition_column_id_column",
            "column",
            ["column_id"],
            ["id"],
        )
    op.create_table(
        "database",
        sa.Column("uuid", sqlalchemy_utils.types.uuid.UUIDType(), nullable=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("extra_params", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("URI", sa.String(), nullable=False),
        sa.Column("read_only", sa.Boolean(), nullable=False),
        sa.Column("async", sa.Boolean(), nullable=False),
        sa.Column("cost", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_database")),
        sa.UniqueConstraint("name", name=op.f("uq_database_name")),
    )
    op.create_table(
        "engine",
        sa.Column(
            "dialect",
            sa.Enum("SPARK", "TRINO", "DRUID", name="dialect"),
            nullable=True,
        ),
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("version", sa.String(), nullable=False),
        sa.Column("uri", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_engine")),
    )
    op.create_table(
        "history",
        sa.Column("pre", sa.JSON(), nullable=True),
        sa.Column("post", sa.JSON(), nullable=True),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("entity_type", sa.String(), nullable=True),
        sa.Column("entity_name", sa.String(), nullable=True),
        sa.Column("node", sa.String(), nullable=True),
        sa.Column("activity_type", sa.String(), nullable=True),
        sa.Column("user", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_history")),
    )
    op.create_table(
        "metricmetadata",
        sa.Column(
            "direction",
            sa.Enum(
                "HIGHER_IS_BETTER",
                "LOWER_IS_BETTER",
                "NEUTRAL",
                name="metricdirection",
            ),
            nullable=True,
        ),
        sa.Column(
            "unit",
            sa.Enum(
                "UNKNOWN",
                "UNITLESS",
                "PERCENTAGE",
                "PROPORTION",
                "DOLLAR",
                "SECOND",
                "MINUTE",
                "HOUR",
                "DAY",
                "WEEK",
                "MONTH",
                "YEAR",
                name="metricunit",
            ),
            nullable=True,
        ),
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_metricmetadata")),
    )
    op.create_table(
        "missingparent",
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_missingparent")),
    )
    op.create_table(
        "nodenamespace",
        sa.Column("deactivated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("namespace", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("namespace", name=op.f("pk_nodenamespace")),
        sa.UniqueConstraint("namespace", name=op.f("uq_nodenamespace_namespace")),
    )
    op.create_table(
        "tag",
        sa.Column("display_name", sa.String(), nullable=True),
        sa.Column("tag_metadata", sa.JSON(), nullable=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("tag_type", sa.String(), nullable=False),
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_tag")),
        sa.UniqueConstraint("name", name=op.f("uq_tag_name")),
    )
    op.create_table(
        "users",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("username", sa.String(), nullable=False),
        sa.Column("password", sa.String(), nullable=True),
        sa.Column("email", sa.String(), nullable=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column(
            "oauth_provider",
            sa.Enum("BASIC", "GITHUB", "GOOGLE", name="oauthprovider"),
            nullable=False,
        ),
        sa.Column("is_admin", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
    )
    op.create_table(
        "catalogengines",
        sa.Column("catalog_id", sa.BigInteger(), nullable=False),
        sa.Column("engine_id", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["catalog_id"],
            ["catalog.id"],
            name=op.f("fk_catalogengines_catalog_id_catalog"),
        ),
        sa.ForeignKeyConstraint(
            ["engine_id"],
            ["engine.id"],
            name=op.f("fk_catalogengines_engine_id_engine"),
        ),
        sa.PrimaryKeyConstraint(
            "catalog_id",
            "engine_id",
            name=op.f("pk_catalogengines"),
        ),
    )
    op.create_table(
        "columnattribute",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("attribute_type_id", sa.BigInteger(), nullable=True),
        sa.Column("column_id", sa.BigInteger(), nullable=True),
        sa.ForeignKeyConstraint(
            ["attribute_type_id"],
            ["attributetype.id"],
            name=op.f("fk_columnattribute_attribute_type_id_attributetype"),
        ),
        sa.ForeignKeyConstraint(
            ["column_id"],
            ["column.id"],
            name=op.f("fk_columnattribute_column_id_column"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_columnattribute")),
        sa.UniqueConstraint(
            "attribute_type_id",
            "column_id",
            name=op.f("uq_columnattribute_attribute_type_id"),
        ),
    )
    op.create_table(
        "noderevision",
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("display_name", sa.String(), nullable=True),
        sa.Column(
            "type",
            sa.Enum(
                "SOURCE",
                "TRANSFORM",
                "METRIC",
                "DIMENSION",
                "CUBE",
                name="nodetype",
            ),
            nullable=True,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lineage", sa.JSON(), nullable=True),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("query", sa.String(), nullable=True),
        sa.Column("mode", sa.String(), nullable=False),
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("version", sa.String(), nullable=True),
        sa.Column("node_id", sa.BigInteger(), nullable=True),
        sa.Column("catalog_id", sa.BigInteger(), nullable=True),
        sa.Column("schema_", sa.String(), nullable=True),
        sa.Column("table", sa.String(), nullable=True),
        sa.Column("metric_metadata_id", sa.BigInteger(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(
            ["catalog_id"],
            ["catalog.id"],
            name=op.f("fk_noderevision_catalog_id_catalog"),
        ),
        sa.ForeignKeyConstraint(
            ["metric_metadata_id"],
            ["metricmetadata.id"],
            name=op.f("fk_noderevision_metric_metadata_id_metricmetadata"),
        ),
        sa.ForeignKeyConstraint(
            ["node_id"],
            ["node.id"],
            name=op.f("fk_noderevision_node_id_node"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_noderevision")),
        sa.UniqueConstraint("version", "node_id", name=op.f("uq_noderevision_version")),
    )
    op.create_table(
        "table",
        sa.Column("schema_", sa.String(), nullable=True),
        sa.Column("table", sa.String(), nullable=False),
        sa.Column("cost", sa.Float(), nullable=False),
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("database_id", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["database_id"],
            ["database.id"],
            name=op.f("fk_table_database_id_database"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_table")),
    )
    op.create_table(
        "tagnoderelationship",
        sa.Column("tag_id", sa.BigInteger(), nullable=False),
        sa.Column("node_id", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["node_id"],
            ["node.id"],
            name=op.f("fk_tagnoderelationship_node_id_node"),
        ),
        sa.ForeignKeyConstraint(
            ["tag_id"],
            ["tag.id"],
            name=op.f("fk_tagnoderelationship_tag_id_tag"),
        ),
        sa.PrimaryKeyConstraint(
            "tag_id",
            "node_id",
            name=op.f("pk_tagnoderelationship"),
        ),
    )
    op.create_table(
        "cube",
        sa.Column("cube_id", sa.BigInteger(), nullable=False),
        sa.Column("cube_element_id", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["cube_element_id"],
            ["column.id"],
            name=op.f("fk_cube_cube_element_id_column"),
        ),
        sa.ForeignKeyConstraint(
            ["cube_id"],
            ["noderevision.id"],
            name=op.f("fk_cube_cube_id_noderevision"),
        ),
        sa.PrimaryKeyConstraint("cube_id", "cube_element_id", name=op.f("pk_cube")),
    )
    op.create_table(
        "materialization",
        sa.Column(
            "strategy",
            sa.Enum(
                "FULL",
                "SNAPSHOT",
                "SNAPSHOT_PARTITION",
                "INCREMENTAL_TIME",
                "VIEW",
                name="materializationstrategy",
            ),
            nullable=True,
        ),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("job", sa.String(), nullable=True),
        sa.Column("deactivated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("node_revision_id", sa.BigInteger(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("schedule", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(
            ["node_revision_id"],
            ["noderevision.id"],
            name=op.f("fk_materialization_node_revision_id_noderevision"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_materialization")),
        sa.UniqueConstraint("name", "node_revision_id", name="name_node_revision_uniq"),
    )
    op.create_table(
        "metric_required_dimensions",
        sa.Column("metric_id", sa.BigInteger(), nullable=False),
        sa.Column("bound_dimension_id", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["bound_dimension_id"],
            ["column.id"],
            name=op.f("fk_metric_required_dimensions_bound_dimension_id_column"),
        ),
        sa.ForeignKeyConstraint(
            ["metric_id"],
            ["noderevision.id"],
            name=op.f("fk_metric_required_dimensions_metric_id_noderevision"),
        ),
        sa.PrimaryKeyConstraint(
            "metric_id",
            "bound_dimension_id",
            name=op.f("pk_metric_required_dimensions"),
        ),
    )
    op.create_table(
        "nodeavailabilitystate",
        sa.Column("availability_id", sa.BigInteger(), nullable=False),
        sa.Column("node_id", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["availability_id"],
            ["availabilitystate.id"],
            name=op.f("fk_nodeavailabilitystate_availability_id_availabilitystate"),
        ),
        sa.ForeignKeyConstraint(
            ["node_id"],
            ["noderevision.id"],
            name=op.f("fk_nodeavailabilitystate_node_id_noderevision"),
        ),
        sa.PrimaryKeyConstraint(
            "availability_id",
            "node_id",
            name=op.f("pk_nodeavailabilitystate"),
        ),
    )
    op.create_table(
        "nodecolumns",
        sa.Column("node_id", sa.BigInteger(), nullable=False),
        sa.Column("column_id", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["column_id"],
            ["column.id"],
            name=op.f("fk_nodecolumns_column_id_column"),
        ),
        sa.ForeignKeyConstraint(
            ["node_id"],
            ["noderevision.id"],
            name=op.f("fk_nodecolumns_node_id_noderevision"),
        ),
        sa.PrimaryKeyConstraint("node_id", "column_id", name=op.f("pk_nodecolumns")),
    )
    op.create_table(
        "nodemissingparents",
        sa.Column("missing_parent_id", sa.BigInteger(), nullable=False),
        sa.Column("referencing_node_id", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["missing_parent_id"],
            ["missingparent.id"],
            name=op.f("fk_nodemissingparents_missing_parent_id_missingparent"),
        ),
        sa.ForeignKeyConstraint(
            ["referencing_node_id"],
            ["noderevision.id"],
            name=op.f("fk_nodemissingparents_referencing_node_id_noderevision"),
        ),
        sa.PrimaryKeyConstraint(
            "missing_parent_id",
            "referencing_node_id",
            name=op.f("pk_nodemissingparents"),
        ),
    )
    op.create_table(
        "noderelationship",
        sa.Column("parent_id", sa.BigInteger(), nullable=False),
        sa.Column("parent_version", sa.String(), nullable=True),
        sa.Column("child_id", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["child_id"],
            ["noderevision.id"],
            name=op.f("fk_noderelationship_child_id_noderevision"),
        ),
        sa.ForeignKeyConstraint(
            ["parent_id"],
            ["node.id"],
            name=op.f("fk_noderelationship_parent_id_node"),
        ),
        sa.PrimaryKeyConstraint(
            "parent_id",
            "child_id",
            name=op.f("pk_noderelationship"),
        ),
    )
    op.create_table(
        "tablecolumns",
        sa.Column("table_id", sa.BigInteger(), nullable=False),
        sa.Column("column_id", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["column_id"],
            ["column.id"],
            name=op.f("fk_tablecolumns_column_id_column"),
        ),
        sa.ForeignKeyConstraint(
            ["table_id"],
            ["table.id"],
            name=op.f("fk_tablecolumns_table_id_table"),
        ),
        sa.PrimaryKeyConstraint("table_id", "column_id", name=op.f("pk_tablecolumns")),
    )
    op.create_table(
        "backfill",
        sa.Column("spec", sa.JSON(), nullable=True),
        sa.Column("urls", sa.JSON(), nullable=True),
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("materialization_id", sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(
            ["materialization_id"],
            ["materialization.id"],
            name=op.f("fk_backfill_materialization_id_materialization"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_backfill")),
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table("backfill")
    op.drop_table("tablecolumns")
    op.drop_table("noderelationship")
    op.drop_table("nodemissingparents")
    op.drop_table("nodecolumns")
    op.drop_table("nodeavailabilitystate")
    op.drop_table("metric_required_dimensions")
    op.drop_table("materialization")
    op.drop_table("cube")
    op.drop_table("tagnoderelationship")
    op.drop_table("table")
    op.drop_table("noderevision")
    op.drop_table("columnattribute")
    op.drop_table("catalogengines")
    op.drop_table("users")
    op.drop_table("tag")
    op.drop_table("partition")
    op.drop_table("nodenamespace")
    op.drop_table("node")
    op.drop_table("missingparent")
    op.drop_table("metricmetadata")
    op.drop_table("measures")
    op.drop_table("history")
    op.drop_table("engine")
    op.drop_table("database")
    op.drop_table("column")
    op.drop_table("catalog")
    op.drop_table("availabilitystate")
    op.drop_table("attributetype")
    # ### end Alembic commands ###
