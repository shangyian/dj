"""Test type inference."""

# pylint: disable=W0621,C0325
import pytest
from sqlalchemy import select
from sqlmodel import Session

from dj.construction.inference import get_type_of_expression
from dj.errors import DJException
from dj.models.node import Node
from dj.sql.parsing import ast2
from dj.sql.parsing.ast2 import CompileContext
from dj.sql.parsing.backends.exceptions import DJParseException
from dj.sql.parsing.backends.antlr4 import parse
from dj.typing import ColumnType


def test_infer_column_with_table(construction_session: Session):
    """
    Test getting the type of a column that has a table
    """
    node = next(
        construction_session.exec(
            select(Node).filter(
                Node.name == "dbt.source.jaffle_shop.orders",
            ),
        ),
    )[0]
    table = ast2.Table(ast2.Name("orders"), _dj_node=node.current)
    assert (
        get_type_of_expression(ast2.Column(ast2.Name("id"), _table=table))
        == ColumnType.INT
    )
    assert (
        get_type_of_expression(ast2.Column(ast2.Name("user_id"), _table=table))
        == ColumnType.INT
    )
    assert (
        get_type_of_expression(ast2.Column(ast2.Name("order_date"), _table=table))
        == ColumnType.DATE
    )
    assert (
        get_type_of_expression(ast2.Column(ast2.Name("status"), _table=table))
        == ColumnType.STR
    )


def test_infer_values():
    """
    Test inferring types from values directly
    """
    assert get_type_of_expression(ast2.String(value="foo")) == ColumnType.STR
    assert get_type_of_expression(ast2.Number(value=10)) == ColumnType.INT
    assert get_type_of_expression(ast2.Number(value=-10)) == ColumnType.INT
    assert get_type_of_expression(ast2.Number(value=922337203685477)) == ColumnType.LONG
    assert get_type_of_expression(ast2.Number(value=-922337203685477)) == ColumnType.LONG
    assert get_type_of_expression(ast2.Number(value=3.4e39)) == ColumnType.DOUBLE
    assert get_type_of_expression(ast2.Number(value=-3.4e39)) == ColumnType.DOUBLE
    assert get_type_of_expression(ast2.Number(value=3.4e38)) == ColumnType.FLOAT
    assert get_type_of_expression(ast2.Number(value=-3.4e38)) == ColumnType.FLOAT


def test_raise_on_invalid_infer_binary_op():
    """
    Test raising when trying to infer types from an invalid binary op
    """
    with pytest.raises(DJParseException) as exc_info:
        get_type_of_expression(
            ast2.BinaryOp(
                op=ast2.BinaryOpKind.Modulo,
                left=ast2.String(value="foo"),
                right=ast2.String(value="bar"),
            ),
        )

    assert (
        "Incompatible types in binary operation 'foo' % 'bar'. "
        "Got left STR, right STR."
    ) in str(exc_info.value)


def test_infer_column_with_an_aliased_table(construction_session: Session):
    """
    Test getting the type of a column that has an aliased table
    """
    node = next(
        construction_session.exec(
            select(Node).filter(
                Node.name == "dbt.source.jaffle_shop.orders",
            ),
        ),
    )[0]
    table = ast2.Table(ast2.Name("orders"), _dj_node=node.current)
    alias = ast2.Alias(
        alias=ast2.Name(
            name="foo",
            namespace=ast2.Name(
                name="a",
                namespace=ast2.Name(
                    name="b",
                    namespace=ast2.Name("c"),
                ),
            ),
        ),
        child=table,
    )
    assert get_type_of_expression(ast2.Column(ast2.Name("id"), _table=alias)) == ColumnType.INT
    assert get_type_of_expression(ast2.Column(ast2.Name("user_id"), _table=alias)) == ColumnType.INT
    assert get_type_of_expression(ast2.Column(ast2.Name("order_date"), _table=alias)) == ColumnType.DATE
    assert get_type_of_expression(ast2.Column(ast2.Name("status"), _table=alias)) == ColumnType.STR
    assert get_type_of_expression(ast2.Column(ast2.Name("_etl_loaded_at"), _table=alias)) == ColumnType.TIMESTAMP


def test_raising_when_table_has_no_dj_node():
    """
    Test raising when getting the type of a column that has a table with no DJ node
    """
    table = ast2.Table(ast2.Name("orders"))
    col = ast2.Column(ast2.Name("status"), _table=table)

    with pytest.raises(DJParseException) as exc_info:
        get_type_of_expression(col)

    assert (
        "Cannot resolve type of column orders.status. "
        "column's table does not have a DJ Node."
    ) in str(exc_info.value)


def test_raising_when_expression_parent_not_a_table():
    """
    Test raising when getting the type of a column thats parent is not a table
    """
    query = parse("select 1")
    col = ast2.Column(
        ast2.Name("status"),
        _table=query.select,
    )  # intentionally adding a non-table AST node

    with pytest.raises(DJParseException) as exc_info:
        get_type_of_expression(col)

    assert (
        "DJ does not currently traverse subqueries for type information. Consider extraction first."
    ) in str(exc_info.value)


def test_raising_when_select_has_multiple_expressions_in_projection():
    """
    Test raising when a select has more than one in projection
    """
    select = parse("select 1, 2").select

    with pytest.raises(DJParseException) as exc_info:
        get_type_of_expression(select)

    assert ("single expression in its projection") in str(exc_info.value)


def test_raising_when_between_different_types():
    """
    Test raising when a between has multiple types
    """
    select = parse("select 1 between 'hello' and TRUE").select

    with pytest.raises(DJParseException) as exc_info:
        get_type_of_expression(select)

    assert ("BETWEEN expects all elements to have the same type") in str(exc_info.value)


def test_raising_when_unop_bad_type():
    """
    Test raising when a unop gets a bad type
    """
    select = parse("select not 'hello'").select

    with pytest.raises(DJParseException) as exc_info:
        get_type_of_expression(select)

    assert ("Incompatible type in unary operation") in str(exc_info.value)


def test_raising_when_expression_has_no_parent():
    """
    Test raising when getting the type of a column that has no parent
    """
    col = ast2.Column(ast2.Name("status"), _table=None)

    with pytest.raises(DJParseException) as exc_info:
        get_type_of_expression(col)

    assert "Cannot resolve type of column status." in str(exc_info.value)


def test_infer_map_subscripts(construction_session: Session):
    """
    Test inferring map subscript types
    """
    query = parse(
        """
        SELECT
          names_map["first"] as first_name,
          names_map["last"] as last_name,
          user_metadata["propensity_score"] as propensity_score,
          user_metadata["propensity_score"]["weighted"] as weighted_propensity_score,
          user_metadata["propensity_score"]["weighted"]["year"] as weighted_propensity_score_year
        FROM basic.source.users
    """,
    )
    exc = DJException()
    ctx = CompileContext(session=construction_session, query=query, exception=exc)
    query.compile(ctx)
    types = [
        ColumnType.STR,
        ColumnType.STR,
        ColumnType.MAP["str", ColumnType.MAP["str", "float"]],
        ColumnType.MAP["str", "float"],
        ColumnType.FLOAT,
    ]
    assert types == [exp.type for exp in query.select.projection]


def test_infer_types_complicated(construction_session: Session):
    """
    Test inferring complicated types
    """
    query = parse(
        """
      SELECT id+1-2/3*5%6&10|8^5,
      CAST('2022-01-01T12:34:56Z' AS TIMESTAMP),
      -- Raw('average({id})', 'INT', True),
      -- Raw('aggregate(array(1, 2, {id}), 0, (acc, x) -> acc + x, acc -> acc * 10)', 'INT'),
      -- Raw('NOW()', 'datetime'),
      DATE_TRUNC('day', '2014-03-10'),
      NOW(),
      Coalesce(NULL, 5),
      Coalesce(NULL),
      NULL,
      MAX(id) OVER
        (PARTITION BY first_name ORDER BY last_name)
        AS running_total,
      MAX(id) OVER
        (PARTITION BY first_name ORDER BY last_name)
        AS running_total,
      MIN(id) OVER
        (PARTITION BY first_name ORDER BY last_name)
        AS running_total,
      AVG(id) OVER
        (PARTITION BY first_name ORDER BY last_name)
        AS running_total,
      COUNT(id) OVER
        (PARTITION BY first_name ORDER BY last_name)
        AS running_total,
      SUM(id) OVER
        (PARTITION BY first_name ORDER BY last_name)
        AS running_total,
      NOT TRUE,
      10,
      id>5,
      id<5,
      id>=5,
      id<=5,
      id BETWEEN 4 AND 5,
      id IN (5, 5),
      id NOT IN (3, 4),
      id NOT IN (SELECT -5),
      first_name LIKE 'Ca%',
      id is null,
      (id=5)=TRUE,
      'hello world',
      first_name as fn,
      last_name<>'yoyo' and last_name='yoyo' or last_name='yoyo',
      last_name,
      bizarre,
      (select 5.0),
      CASE WHEN first_name = last_name THEN COUNT(DISTINCT first_name) ELSE
      COUNT(DISTINCT last_name) END
      FROM (
      SELECT id,
         first_name,
         last_name<>'yoyo' and last_name='yoyo' or last_name='yoyo' as bizarre,
         last_name
      FROM dbt.source.jaffle_shop.customers
        )
    """,
    )
    exc = DJException()
    ctx = CompileContext(session=construction_session, query=query, exception=exc)
    query.compile(ctx)
    types = [
        ColumnType.INT,
        ColumnType.TIMESTAMP,
        # ColumnType.INT,
        # ColumnType.INT,
        # ColumnType.TIMESTAMP,
        ColumnType.TIMESTAMP,
        ColumnType.TIMESTAMP,
        ColumnType.INT,
        ColumnType.NULL,
        ColumnType.NULL,
        ColumnType.INT,
        ColumnType.INT,
        ColumnType.INT,
        ColumnType.INT,
        ColumnType.INT,
        ColumnType.INT,
        ColumnType.BOOL,
        ColumnType.INT,
        ColumnType.BOOL,
        ColumnType.BOOL,
        ColumnType.BOOL,
        ColumnType.BOOL,
        ColumnType.BOOL,
        ColumnType.BOOL,
        ColumnType.BOOL,
        ColumnType.BOOL,
        ColumnType.BOOL,
        ColumnType.BOOL,
        ColumnType.BOOL,
        ColumnType.STR,
        ColumnType.STR,
        ColumnType.BOOL,
        ColumnType.STR,
        ColumnType.BOOL,
        ColumnType.FLOAT,
        ColumnType.INT,
    ]
    assert types == [exp.type for exp in query.select.projection]


def test_infer_bad_case_types(construction_session: Session):
    """
    Test inferring mismatched case types.
    """
    with pytest.raises(Exception) as excinfo:
        query = parse(
            """
            SELECT
            CASE WHEN first_name = last_name THEN COUNT(DISTINCT first_name) ELSE last_name END
            FROM dbt.source.jaffle_shop.customers
            """,
        )
        exc = DJException()
        ctx = CompileContext(session=construction_session, query=query, exception=exc)
        query.compile(ctx)
        [  # pylint: disable=pointless-statement
            exp.type for exp in query.select.projection
        ]

    assert str(excinfo.value) == "Not all the same type in CASE! Found: INT, STR"
