"""
Tests for internal namespace functions
"""

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from datajunction_server.database.namespace import NodeNamespace
from datajunction_server.database.user import User
from datajunction_server.internal.namespaces import (
    _merge_list_with_key,
    _merge_yaml_preserving_comments,
    create_or_reactivate_namespace,
    is_default_branch_namespace,
    node_spec_to_yaml,
)
from datajunction_server.models.deployment import MetricSpec, TransformSpec
from datajunction_server.models.namespace import NamespaceWriteStatus
from datajunction_server.models.node_type import NodeType


async def test_create_or_reactivate_namespace_reports_already_exists(
    session,
    current_user: User,
):
    """
    An existing namespace is reported back, not raised.

    The register_table / register_view callers discard the result and only need
    the namespace to exist, so raising here would make registering into any
    existing namespace fail.
    """

    async def save_history(event, session):
        """No-op history recorder."""

    async def create():
        return await create_or_reactivate_namespace(
            "already_exists_ns",
            include_parents=False,
            session=session,
            current_user=current_user,
            save_history=save_history,
        )

    assert (await create()).status == NamespaceWriteStatus.CREATED
    assert (await create()).status == NamespaceWriteStatus.ALREADY_EXISTS


class TestIsDefaultBranchNamespace:
    """
    Telling the default-branch view of a repo apart from a branch namespace.

    Production work -- a scheduled materialization workflow, a Druid datasource --
    is only started from the former, so this has to be wrong in neither direction:
    a branch read as the default branch schedules work that outlives the branch,
    and the default branch read as a branch silently stops production
    materializing.
    """

    @staticmethod
    async def _namespaces(session, *rows: NodeNamespace) -> None:
        """Persist namespace rows in order, so a parent exists before its child."""
        for row in rows:
            session.add(row)
            await session.commit()

    async def test_no_git_configuration_is_the_default_branch(self, session):
        """
        A namespace with nothing git-related on it, the common case: there is no
        branch here to infer, and it must keep materializing as it always has.
        """
        await self._namespaces(session, NodeNamespace(namespace="idb_plain"))
        assert await is_default_branch_namespace(session, "idb_plain") is True

    async def test_repo_without_branching_is_the_default_branch(self, session):
        """
        A namespace that owns a repo and is pinned to a branch, with no
        `default_branch` and no parent: a flat repo-backed namespace. No branch
        namespace was ever made from it -- that needs a `default_branch` to branch
        from -- so its branch, whatever it is called, is the only view there is.
        """
        await self._namespaces(
            session,
            NodeNamespace(
                namespace="idb_flat",
                github_repo_path="corp/flat",
                git_branch="trunk",
            ),
        )
        assert await is_default_branch_namespace(session, "idb_flat") is True

    async def test_git_root_itself_is_the_default_branch(self, session):
        """The namespace that configures the repo is never a branch of itself."""
        await self._namespaces(
            session,
            NodeNamespace(
                namespace="idb_root",
                github_repo_path="corp/root",
                default_branch="main",
            ),
        )
        assert await is_default_branch_namespace(session, "idb_root") is True

    async def test_branch_namespaces_under_a_git_root(self, session):
        """
        The shape the branch API and `dj push` produce: a git root naming the default
        branch, with a namespace per branch carrying its own `git_branch`. A
        subnamespace of a branch resolves through it.
        """
        await self._namespaces(
            session,
            NodeNamespace(
                namespace="idb_repo",
                github_repo_path="corp/repo",
                default_branch="main",
            ),
            NodeNamespace(
                namespace="idb_repo.main",
                git_branch="main",
                parent_namespace="idb_repo",
            ),
            NodeNamespace(
                namespace="idb_repo.feature_x",
                git_branch="feature-x",
                parent_namespace="idb_repo",
            ),
            NodeNamespace(namespace="idb_repo.feature_x.cubes"),
        )
        assert await is_default_branch_namespace(session, "idb_repo.main") is True
        assert await is_default_branch_namespace(session, "idb_repo.feature_x") is False
        assert (
            await is_default_branch_namespace(session, "idb_repo.feature_x.cubes")
        ) is False

    async def test_unpopulated_git_branch_falls_back_to_the_segment(self, session):
        """
        A namespace under a git root that carries no `git_branch` of its own -- an
        ordinary deploy creates one this way, never touching the branch columns.

        Its final segment is then the only thing left to go on: `main` under a root
        whose default branch is `main` is the default branch, and anything else is
        treated as a branch rather than assumed to be production.
        """
        await self._namespaces(
            session,
            NodeNamespace(
                namespace="idb_bare",
                github_repo_path="corp/bare",
                default_branch="main",
            ),
            NodeNamespace(namespace="idb_bare.main"),
            NodeNamespace(namespace="idb_bare.scratch"),
        )
        assert await is_default_branch_namespace(session, "idb_bare.main") is True
        assert await is_default_branch_namespace(session, "idb_bare.scratch") is False

    async def test_no_default_branch_to_compare_against_is_a_branch(self, session):
        """
        A namespace pointing at a parent, with no `default_branch` recorded anywhere.

        The default-branch namespace carries a `parent_namespace` too, so there is
        nothing here to tell the two apart -- and of the two ways to be wrong, not
        scheduling is the one that gets reported and can be overridden per deploy.
        """
        await self._namespaces(
            session,
            NodeNamespace(namespace="idb_nodefault", github_repo_path="corp/nodefault"),
            NodeNamespace(
                namespace="idb_nodefault.main",
                git_branch="main",
                parent_namespace="idb_nodefault",
            ),
        )
        assert (
            await is_default_branch_namespace(session, "idb_nodefault.main")
        ) is False

    async def test_default_branch_resolves_through_a_sibling_parent(self, session):
        """
        A branch namespace may point at a sibling rather than at a string ancestor
        (`idb_sib.feature` -> `idb_sib.main`), which is then where the repo and the
        default branch are configured. One FK hop keeps that visible; without it the
        branch would look like a namespace with no default branch to compare to.
        """
        await self._namespaces(
            session,
            NodeNamespace(namespace="idb_sib"),
            NodeNamespace(
                namespace="idb_sib.main",
                github_repo_path="corp/sib",
                git_branch="main",
                default_branch="main",
            ),
            NodeNamespace(
                namespace="idb_sib.feature",
                git_branch="feature",
                parent_namespace="idb_sib.main",
            ),
        )
        assert await is_default_branch_namespace(session, "idb_sib.main") is True
        assert await is_default_branch_namespace(session, "idb_sib.feature") is False


class TestMergeListWithKey:
    """Tests for _merge_list_with_key function"""

    def test_merge_yaml_list_preserves_attribute_order_when_unchanged(self):
        """Test that attribute order is preserved when the attribute set hasn't changed"""
        from ruamel.yaml import YAML

        yaml = YAML()

        # Create existing list by parsing YAML to get proper CommentedMap structure
        existing_yaml = """
- name: col1
  type: int
  attributes:
    - primary_key
    - dimension
"""
        existing = yaml.load(existing_yaml)

        # Create new list with same attributes but different order
        new_yaml = """
- name: col1
  type: int
  attributes:
    - dimension
    - primary_key
"""
        new_list = yaml.load(new_yaml)

        result = _merge_list_with_key(existing, new_list, "name")

        # Should preserve the original attribute order since the set is unchanged
        assert result[0]["attributes"] == ["primary_key", "dimension"]

    def test_merge_yaml_list_updates_attributes_when_changed(self):
        """Test that attributes are updated when the set changes"""
        # Create existing list with specific attribute order
        existing = CommentedSeq(
            [
                CommentedMap(
                    {
                        "name": "col1",
                        "type": "int",
                        "attributes": ["primary_key", "dimension"],
                    },
                ),
            ],
        )

        # Create new list with different attributes
        new_list = CommentedSeq(
            [
                CommentedMap(
                    {
                        "name": "col1",
                        "type": "int",
                        "attributes": ["primary_key"],  # Removed "dimension"
                    },
                ),
            ],
        )

        result = _merge_list_with_key(existing, new_list, "name")

        # Should update to new attributes since the set changed
        assert result[0]["attributes"] == ["primary_key"]

    def test_merge_yaml_list_handles_non_list_attributes(self):
        """Test that non-list attributes in existing item don't cause issues"""
        # Create existing list with non-list attributes value
        existing = CommentedSeq(
            [
                CommentedMap(
                    {
                        "name": "col1",
                        "type": "int",
                        "attributes": "not_a_list",  # Not a list
                    },
                ),
            ],
        )

        # Create new list with proper list attributes
        new_list = CommentedSeq(
            [
                CommentedMap(
                    {
                        "name": "col1",
                        "type": "int",
                        "attributes": ["primary_key"],
                    },
                ),
            ],
        )

        result = _merge_list_with_key(existing, new_list, "name")

        # Should update since existing wasn't a list
        assert result[0]["attributes"] == ["primary_key"]

    def test_merge_yaml_list_adds_attributes_when_missing_in_existing(self):
        """Test that attributes are added when they don't exist in existing item"""
        # Create existing list without attributes
        existing = CommentedSeq(
            [
                CommentedMap(
                    {
                        "name": "col1",
                        "type": "int",
                    },
                ),
            ],
        )

        # Create new list with attributes
        new_list = CommentedSeq(
            [
                CommentedMap(
                    {
                        "name": "col1",
                        "type": "int",
                        "attributes": ["primary_key"],
                    },
                ),
            ],
        )

        result = _merge_list_with_key(existing, new_list, "name")

        # Should add the new attributes
        assert result[0]["attributes"] == ["primary_key"]

    def test_merge_via_yaml_preserving_comments_with_unchanged_attributes(self):
        """Test attribute order preservation through the full YAML merge flow"""
        yaml = YAML()

        # Create existing YAML with columns that have specific attribute order
        existing_yaml = """
name: test_node
type: transform
columns:
  - name: col1
    type: int
    attributes:
      - primary_key
      - dimension
  - name: col2
    type: string
"""
        existing = yaml.load(existing_yaml)

        # Create new YAML with same attributes but different order
        new_yaml = """
name: test_node
type: transform
columns:
  - name: col1
    type: int
    attributes:
      - dimension
      - primary_key
  - name: col2
    type: string
"""
        new_data = yaml.load(new_yaml)

        # Merge the YAML structures
        result = _merge_yaml_preserving_comments(existing, new_data, yaml)

        # Should preserve the original attribute order since the set is unchanged
        assert result["columns"][0]["attributes"] == ["primary_key", "dimension"]

    def test_merge_yaml_preserves_cube_metrics_dimensions_order(self):
        """Test that cube metrics and dimensions order is preserved from existing YAML"""
        yaml = YAML()

        # Create existing cube YAML with specific order
        existing_yaml = """
node_type: cube
name: my_cube
description: A cube
metrics:
  - metric_z
  - metric_a
  - metric_m
dimensions:
  - dim_y
  - dim_b
  - dim_x
"""
        existing = yaml.load(existing_yaml)

        # Create new YAML with same items but different order (simulating DB order)
        new_yaml = """
node_type: cube
name: my_cube
description: A cube updated
metrics:
  - metric_a
  - metric_m
  - metric_z
dimensions:
  - dim_b
  - dim_x
  - dim_y
"""
        new_data = yaml.load(new_yaml)

        # Merge the YAML structures
        result = _merge_yaml_preserving_comments(existing, new_data, yaml)

        # Should preserve the original order from existing YAML
        assert result["metrics"] == ["metric_z", "metric_a", "metric_m"]
        assert result["dimensions"] == ["dim_y", "dim_b", "dim_x"]
        # Description should be updated though
        assert result["description"] == "A cube updated"

    def test_merge_yaml_cube_adds_new_metrics_at_end(self):
        """Test that new metrics/dimensions are added at the end when preserving order"""
        yaml = YAML()

        # Create existing cube YAML
        existing_yaml = """
node_type: cube
name: my_cube
metrics:
  - metric_a
  - metric_b
dimensions:
  - dim_x
"""
        existing = yaml.load(existing_yaml)

        # Create new YAML with additional items
        new_yaml = """
node_type: cube
name: my_cube
metrics:
  - metric_a
  - metric_b
  - metric_c
dimensions:
  - dim_x
  - dim_y
"""
        new_data = yaml.load(new_yaml)

        # Merge the YAML structures
        result = _merge_yaml_preserving_comments(existing, new_data, yaml)

        # Should preserve existing order and add new items at end
        assert result["metrics"] == ["metric_a", "metric_b", "metric_c"]
        assert result["dimensions"] == ["dim_x", "dim_y"]

    def test_merge_yaml_cube_removes_deleted_metrics(self):
        """Test that removed metrics/dimensions are not included in result"""
        yaml = YAML()

        # Create existing cube YAML
        existing_yaml = """
node_type: cube
name: my_cube
metrics:
  - metric_a
  - metric_b
  - metric_c
dimensions:
  - dim_x
  - dim_y
"""
        existing = yaml.load(existing_yaml)

        # Create new YAML with some items removed
        new_yaml = """
node_type: cube
name: my_cube
metrics:
  - metric_a
  - metric_c
dimensions:
  - dim_x
"""
        new_data = yaml.load(new_yaml)

        # Merge the YAML structures
        result = _merge_yaml_preserving_comments(existing, new_data, yaml)

        # Should only include items that are in new data, in original order
        assert result["metrics"] == ["metric_a", "metric_c"]
        assert result["dimensions"] == ["dim_x"]


class TestNodeSpecToYaml:
    """Tests for node_spec_to_yaml formatting and determinism"""

    def test_owners_are_sorted_on_fresh_dump(self):
        """owners are sorted alphabetically when there is no existing YAML to merge"""
        spec = MetricSpec(
            name="ns.metrics.revenue",
            node_type=NodeType.METRIC,
            owners=["zara@netflix.com", "alice@netflix.com", "bob@netflix.com"],
            query="SELECT SUM(rev) FROM ns.transforms.t",
        )
        assert node_spec_to_yaml(spec).splitlines() == [
            "name: ns.metrics.revenue",
            "node_type: metric",
            "owners:",
            "  - alice@netflix.com",
            "  - bob@netflix.com",
            "  - zara@netflix.com",
            "mode: published",
            "query: SELECT SUM(rev) FROM ns.transforms.t",
        ]

    def test_owners_preserve_order_on_merge(self):
        """owners preserve existing file order when merging with existing YAML"""
        existing_yaml = (
            "name: ns.metrics.revenue\n"
            "node_type: metric\n"
            "owners:\n"
            "  - zara@netflix.com\n"
            "  - alice@netflix.com\n"
            "mode: published\n"
            "query: SELECT SUM(rev) FROM ns.transforms.t\n"
        )
        spec = MetricSpec(
            name="ns.metrics.revenue",
            node_type=NodeType.METRIC,
            owners=["alice@netflix.com", "zara@netflix.com"],
            query="SELECT SUM(rev) FROM ns.transforms.t",
        )
        result = node_spec_to_yaml(spec, existing_yaml=existing_yaml)
        owners_lines = [line for line in result.splitlines() if "@netflix.com" in line]
        assert owners_lines == ["  - zara@netflix.com", "  - alice@netflix.com"]

    def test_tags_are_sorted_on_fresh_dump(self):
        """tags are sorted alphabetically when there is no existing YAML to merge"""
        spec = MetricSpec(
            name="ns.metrics.revenue",
            node_type=NodeType.METRIC,
            tags=["ratio_metric", "core", "finance"],
            query="SELECT SUM(rev) FROM ns.transforms.t",
        )
        assert node_spec_to_yaml(spec).splitlines() == [
            "name: ns.metrics.revenue",
            "node_type: metric",
            "tags:",
            "  - core",
            "  - finance",
            "  - ratio_metric",
            "mode: published",
            "query: SELECT SUM(rev) FROM ns.transforms.t",
        ]

    def test_tags_preserve_order_on_merge(self):
        """tags preserve existing file order when merging with existing YAML"""
        existing_yaml = (
            "name: ns.metrics.revenue\n"
            "node_type: metric\n"
            "tags:\n"
            "  - ratio_metric\n"
            "  - core\n"
            "mode: published\n"
            "query: SELECT SUM(rev) FROM ns.transforms.t\n"
        )
        spec = MetricSpec(
            name="ns.metrics.revenue",
            node_type=NodeType.METRIC,
            tags=["core", "ratio_metric"],
            query="SELECT SUM(rev) FROM ns.transforms.t",
        )
        result = node_spec_to_yaml(spec, existing_yaml=existing_yaml)
        tag_lines = [line for line in result.splitlines() if "  - " in line]
        assert tag_lines == ["  - ratio_metric", "  - core"]

    def test_metric_with_legacy_string_unit_round_trips(self):
        """Metric authored with legacy `unit: dollar` emits `unit: dollar` on export."""
        spec = MetricSpec(
            name="ns.metrics.revenue",
            node_type=NodeType.METRIC,
            unit="dollar",
            query="SELECT SUM(amount) FROM ns.transforms.t",
        )
        output = node_spec_to_yaml(spec)
        assert "unit: dollar" in output
        # Not a structured dict in the legacy case.
        assert "kind:" not in output

    def test_metric_with_structured_unit_emits_at_spec_level(self):
        """Structured `unit:` at the metric spec level emits as a nested dict."""
        spec = MetricSpec(
            name="ns.metrics.revenue_eur",
            node_type=NodeType.METRIC,
            unit={"kind": "currency", "code": "EUR"},
            query="SELECT SUM(amount_eur) FROM ns.transforms.t",
        )
        output = node_spec_to_yaml(spec)
        # Emitted at spec level, not on a columns block.
        assert "unit:" in output
        assert "kind: currency" in output
        assert "code: EUR" in output

    def test_metric_with_compound_unit_emits_nested(self):
        spec = MetricSpec(
            name="ns.metrics.qps",
            node_type=NodeType.METRIC,
            unit={
                "numerator": {"kind": "count"},
                "denominator": {"kind": "time", "code": "s"},
            },
            query="SELECT 1",
        )
        output = node_spec_to_yaml(spec)
        assert "numerator:" in output
        assert "denominator:" in output
        assert "kind: time" in output
        assert "code: s" in output

    def test_metric_export_suppresses_per_column_unit(self):
        """A metric's columns[].unit must NOT appear in YAML export — the metric
        emits its unit at the spec top level instead."""
        spec = MetricSpec(
            name="ns.metrics.revenue",
            node_type=NodeType.METRIC,
            unit={"kind": "currency", "code": "USD"},
            query="SELECT SUM(amount) FROM ns.transforms.t",
            columns=[
                {
                    "name": "revenue",
                    "unit": {"kind": "currency", "code": "USD"},
                },
            ],
        )
        output = node_spec_to_yaml(spec)
        # Spec-level unit is present...
        assert "kind: currency" in output
        # ...but columns block (if present) does not contain a duplicate unit
        if "columns:" in output:
            # Find columns block and inspect its body
            columns_idx = output.find("columns:")
            columns_block = output[columns_idx:]
            # The metric-level `unit:` line lives before `columns:` — so any
            # unit-related text after `columns:` would be a per-column emit.
            assert "unit:" not in columns_block

    def test_column_with_unit_is_exported(self):
        """A column whose only customization is a unit must still be exported.

        `_has_column_customizations` recognizes `unit` alongside display_name,
        description, attributes, and partition; without that, a unit-only
        column would be treated as "unmodified" and silently dropped.
        """
        spec = TransformSpec(
            name="ns.transforms.t",
            node_type=NodeType.TRANSFORM,
            query="SELECT revenue FROM ns.source.s",
            columns=[
                {
                    "name": "revenue",
                    "unit": {"kind": "currency", "code": "USD"},
                },
            ],
        )
        lines = node_spec_to_yaml(spec).splitlines()
        assert "columns:" in lines
        assert "  - name: revenue" in lines
        # Unit appears as a nested mapping; check key + values are present
        unit_idx = next(i for i, line in enumerate(lines) if line.strip() == "unit:")
        # The following lines should be the kind/code mapping
        assert "kind: currency" in lines[unit_idx + 1]
        assert "code: USD" in lines[unit_idx + 2]

    def test_column_without_unit_is_not_exported_with_noise(self):
        """A column with no customizations (including no unit) is omitted from YAML.

        The new `unit` field must not introduce `unit: null` lines on every
        column when no unit is set.
        """
        spec = TransformSpec(
            name="ns.transforms.t",
            node_type=NodeType.TRANSFORM,
            query="SELECT id FROM ns.source.s",
            columns=[{"name": "id"}],  # no unit, no customizations
        )
        output = node_spec_to_yaml(spec)
        assert "unit:" not in output
        assert "columns:" not in output  # column itself filtered out

    def test_compound_unit_round_trips(self):
        """Compound units serialize cleanly and the structure is preserved."""
        spec = TransformSpec(
            name="ns.transforms.t",
            node_type=NodeType.TRANSFORM,
            query="SELECT qps FROM ns.source.s",
            columns=[
                {
                    "name": "qps",
                    "unit": {
                        "numerator": {"kind": "count"},
                        "denominator": {"kind": "time", "code": "s"},
                    },
                },
            ],
        )
        output = node_spec_to_yaml(spec)
        assert "numerator:" in output
        assert "denominator:" in output
        assert "kind: time" in output
        assert "code: s" in output

    def test_column_attributes_are_sorted(self):
        """column attributes are sorted alphabetically regardless of input order"""
        spec = TransformSpec(
            name="ns.transforms.t",
            node_type=NodeType.TRANSFORM,
            query="SELECT id FROM ns.source.s",
            columns=[
                {
                    "name": "id",
                    "display_name": "ID",
                    "attributes": ["dimension", "primary_key"],
                },
            ],
        )
        assert node_spec_to_yaml(spec).splitlines() == [
            "name: ns.transforms.t",
            "node_type: transform",
            "mode: published",
            "columns:",
            "  - name: id",
            "    display_name: ID",
            "    attributes:",
            "      - dimension",
            "      - primary_key",
            "query: SELECT id FROM ns.source.s",
        ]

    def test_output_is_deterministic(self):
        """calling node_spec_to_yaml twice with the same spec gives identical output"""
        spec = MetricSpec(
            name="ns.metrics.revenue",
            node_type=NodeType.METRIC,
            owners=["zara@netflix.com", "alice@netflix.com"],
            tags=["ratio_metric", "core"],
            query="SELECT SUM(rev) FROM ns.transforms.t",
        )
        assert node_spec_to_yaml(spec) == node_spec_to_yaml(spec)

    def test_no_yaml_document_start_marker(self):
        """output does not start with --- (explicit_start=False)"""
        spec = MetricSpec(
            name="ns.metrics.revenue",
            node_type=NodeType.METRIC,
            query="SELECT SUM(rev) FROM ns.transforms.t",
        )
        assert node_spec_to_yaml(spec).splitlines() == [
            "name: ns.metrics.revenue",
            "node_type: metric",
            "mode: published",
            "query: SELECT SUM(rev) FROM ns.transforms.t",
        ]

    def test_multiline_query_uses_literal_block_style(self):
        """multiline queries are serialized with |- literal block style"""
        spec = MetricSpec(
            name="ns.metrics.revenue",
            node_type=NodeType.METRIC,
            query="SELECT SUM(rev)\nFROM ns.transforms.t",
        )
        assert node_spec_to_yaml(spec).splitlines() == [
            "name: ns.metrics.revenue",
            "node_type: metric",
            "mode: published",
            "query: |-",
            "  SELECT SUM(rev)",
            "  FROM ns.transforms.t",
        ]

    def test_short_lists_use_inline_style(self):
        """short lists (owners, tags) use inline [a, b] style after yamlfix"""
        spec = MetricSpec(
            name="ns.metrics.revenue",
            node_type=NodeType.METRIC,
            owners=["alice@netflix.com"],
            tags=["core"],
            query="SELECT SUM(rev) FROM ns.transforms.t",
        )
        assert node_spec_to_yaml(spec).splitlines() == [
            "name: ns.metrics.revenue",
            "node_type: metric",
            "owners:",
            "  - alice@netflix.com",
            "tags:",
            "  - core",
            "mode: published",
            "query: SELECT SUM(rev) FROM ns.transforms.t",
        ]
