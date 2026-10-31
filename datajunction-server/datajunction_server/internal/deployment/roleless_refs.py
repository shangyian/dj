"""
Warn about stored dimension refs that carry no role although the dimension is
reachable only through role-bearing links.

A role-less ref means the "null role": it matches only a role-free dimension
link. Creation and deploy validation accept such refs, so they would deploy
valid and then fail when SQL is generated.
"""

from __future__ import annotations

from datajunction_server.construction.build_v3.filters import (
    dimension_ref_of_expression,
)
from datajunction_server.construction.build_v3.utils import (
    extract_filter_dimension_refs,
)
from datajunction_server.database.node import NodeRevision
from datajunction_server.errors import DJError, ErrorCode
from datajunction_server.internal.deployment.dimension_reachability import (
    DimensionReachability,
)
from datajunction_server.models.filter_requirements import parse_filter_requirements
from datajunction_server.models.reaggregate import parse_reaggregate_spec
from datajunction_server.sql.parsing import ast
from datajunction_server.sql.parsing.backends.antlr4 import parse
from datajunction_server.utils import SEPARATOR


def roleless_refs(revision: NodeRevision) -> set[tuple[str, str]]:
    """
    (position, ref) pairs for the role-less `node.column` refs a metric stores in
    required_dimensions, fixed_grain, reaggregate rules, the default text of
    required/recommended filters, and window ORDER BY.

    The `dimension` attribute of a filter requirement is left out (a role-less
    attribute is a declared wildcard), as is anything inside the aggregate
    expression (those keep any-role resolution).
    """
    found: list[tuple[str, str]] = [
        ("required_dimensions", ref) for ref in revision.required_dimensions_refs
    ]
    found += [("fixed_grain", ref) for ref in revision.fixed_grain or []]
    if reaggregate := parse_reaggregate_spec(revision.reaggregate):
        found += [("reaggregate", rule.dimension) for rule in reaggregate.rules]
    for position, entries in (
        ("required_filters default", revision.required_filters),
        ("recommended_filters default", revision.recommended_filters),
    ):
        for entry in parse_filter_requirements(entries) or []:
            if entry.default:
                found += [
                    (position, ref)
                    for ref in extract_filter_dimension_refs([entry.default])
                ]
    for function in parse(revision.query).find_all(ast.Function):
        if function.over:
            for sort_item in function.over.order_by:
                ref = dimension_ref_of_expression(sort_item.expr) or ""
                found.append(("window ORDER BY", ref))
    return {
        (position, ref)
        for position, ref in found
        if SEPARATOR in ref and "[" not in ref
    }


def roleless_ref_warnings(
    metric_refs: dict[str, set[tuple[str, str]]],
    metric_facts: dict[str, dict[str, int]],
    reachability: DimensionReachability,
) -> list[DJError]:
    """
    One warning per metric, listing refs that some fact reaches only through roles.

    `metric_facts` maps a metric to {fact name: revision id}; derived metrics are
    already resolved to their base facts. A multi-fact metric is flagged when any
    fact that reaches the dimension does so only through roles, since the other
    facts' role-free paths cannot make the query succeed for that one.
    """
    warnings: list[DJError] = []
    for metric_name in sorted(metric_refs):
        problems: list[str] = []
        for position, ref in sorted(metric_refs[metric_name]):
            dimension = ref.rsplit(SEPARATOR, 1)[0]
            facts = {
                fact: rev_id
                for fact, rev_id in metric_facts.get(metric_name, {}).items()
                if reachability.is_reachable(rev_id, dimension)
                and not reachability.is_reachable_under_role(rev_id, dimension, None)
            }
            if not facts:
                continue
            roles = sorted(
                {
                    role
                    for rev_id in facts.values()
                    for role in reachability.roles_reaching(rev_id, dimension)
                },
            )
            # Prefer a single-hop role as the example.
            example = min(roles, key=lambda role: ("->" in role, role))
            problems.append(
                f"{position} `{ref}`: {dimension} is reachable from "
                f"{', '.join(sorted(facts))} only through roles: "
                f"{', '.join(roles)}. Specify one, e.g. {ref}[{example}]",
            )
        if problems:
            warnings.append(
                DJError(
                    code=ErrorCode.INVALID_DIMENSION,
                    message=(
                        f"Metric {metric_name} uses dimension refs with no role. "
                        "A ref with no role matches only a role-free dimension "
                        "link, so SQL generation will reject these: "
                        + "; ".join(problems)
                    ),
                ),
            )
    return warnings
