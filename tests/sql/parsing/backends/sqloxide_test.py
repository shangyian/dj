"""
tests for the backend that takes sqloxide output and transforms it into an DJ ast
"""
# pylint: disable=line-too-long

import pytest

from dj.sql.parsing.ast import Between, Boolean, Column
from dj.sql.parsing.ast import MapSubscript as ASTMapSubscript
from dj.sql.parsing.ast import Name, Number, UnaryOp, UnaryOpKind
from dj.sql.parsing.ast import Wildcard as ASTWildcard
from dj.sql.parsing.backends.exceptions import DJParseException
from dj.sql.parsing.backends.sqloxide import (
    parse,
    parse_expression,
    parse_op,
    parse_value,
)
from tests.sql.utils import TPCDS_QUERY_SET, read_query


def test_cte_suquery_sql_parse_error():
    """
    test parsing a subquery with ctes fail
    """
    query = """
    select * from
    (WITH
  eid AS
  (
    SELECT EmployeeID
    FROM Employees
  )
SELECT * from eid)
    """
    with pytest.raises(DJParseException):
        parse(query)


def test_case_when_null_sql_parse(case_when_null):
    """
    test parsing a case_when_null query
    """
    assert case_when_null.compare(parse(read_query("case_when_null.sql")))


def test_trivial_sql_parse(trivial_query):
    """
    test parsing a trivial query
    """
    assert trivial_query.compare(parse(read_query("trivial_query.sql")))


def test_derived_subquery_parse(derived_subquery):
    """
    test parsing a query with a from (select...)
    """
    assert derived_subquery.compare(parse(read_query("derived_subquery.sql")))


def test_derived_subquery_parse_lateral_fail():
    """
    test parsing a query with a from (select...)
    """
    with pytest.raises(DJParseException):
        parse(
            """SELECT * FROM   tbl t
            LEFT JOIN LATERAL
                (SELECT * FROM b WHERE b.t_id = t.t_id) t
            ON TRUE;""",
        )


def test_parse_cast():
    """
    test parsing a query with a Cast
    """
    assert (
        str(
            parse(
                """SELECT CAST('2022-01-01T12:34:56Z' AS TIMESTAMP)""",
            ).select.projection[0],
        )
        == "CAST('2022-01-01T12:34:56Z' AS TIMESTAMP)"
    )

    assert (
        str(
            parse(
                """SELECT CAST('2022-01-01T12:34:56Z' AS DATETIME)""",
            ).select.projection[0],
        )
        == "CAST('2022-01-01T12:34:56Z' AS TIMESTAMP)"
    )

    assert (
        str(
            parse(
                """SELECT CASE WHEN
                     CAST(COUNT(*) AS INTEGER) > 0
                   THEN
                     CAST(CAST(SUM(COALESCE(something, 0)) AS INTEGER) AS DOUBLE)
                     / CAST(CAST(COUNT(*) AS INTEGER) AS DOUBLE)
                   ELSE
                     NULL END""",
            ).select.projection[0],
        )
        == """(CASE
        WHEN CAST(COUNT(*) AS INT) > 0 THEN CAST(CAST(SUM(COALESCE(something, 0)) AS INT) AS DOUBLE) / CAST(CAST(COUNT(*) AS INT) AS DOUBLE)
        ELSE None
    END)"""
    )


@pytest.mark.skipif("not config.getoption('tpcds')")
@pytest.mark.parametrize("query_name", TPCDS_QUERY_SET)
def test_parse_tpcds(request, query_name):
    """
    test tpcds queries parse properly
    """
    expected_ast = request.getfixturevalue(query_name)
    query = read_query(f"{query_name}.sql")
    parsed = parse(query)
    assert expected_ast.compare(parsed)


def test_parse_boolean():
    """
    test parsing a sqloxide boolean
    """
    assert Boolean(True) == parse_value({"Value": {"Boolean": True}})


def test_parse_negated_between():
    """
    test parsing a not between
    """
    assert parse_op(
        {
            "Between": {
                "expr": {"Identifier": {"value": "x", "quote_style": None}},
                "negated": True,
                "low": {"Value": {"Number": ("0", False)}},
                "high": {"Value": {"Number": ("1", False)}},
            },
        },
    ) == UnaryOp(
        op=UnaryOpKind.Not,
        expr=Between(
            expr=Column(Name(name="x", quote_style="")),
            low=Number(value=0),
            high=Number(value=1),
        ),
    )


def test_multi_statement_exception():
    """
    tests we only will parse a single sql statement at a time
    """
    with pytest.raises(DJParseException):
        parse("select x from a; select x from b;")


def test_union_exception():
    """
    tests that unions are currently unsupported
    """
    with pytest.raises(DJParseException):
        parse("select x from a union select x from b")


def test_join_must_specify_on():
    """
    tests to make sure a join must specify an on clause
    """
    with pytest.raises(DJParseException):
        parse("select * from a inner join b")


def test_wildcard_parsetree():
    """
    tests parsing the very particular case of a wilcard as a key
    """
    parse_expression({"Wildcard": {}}).compare(ASTWildcard())


def test_map_subscript():
    """
    tests parsing the very particular case of a wilcard as a key
    """
    parse_expression(
        {
            "MapAccess": {
                "column": {"Identifier": {"value": "names_map", "quote_style": None}},
                "keys": [{"Value": {"SingleQuotedString": "first"}}],
            },
        },
    ).compare(
        ASTMapSubscript(Column(Name(name="names_map", quote_style="")), keys=["first"]),
    )
