"""
testing ast Nodes and their methods
"""

import pytest

from sqlmodel import Session

from dj.errors import DJException
from dj.sql.parsing.backends.antlr4 import parse
from dj.sql.parsing import ast2 as ast
from dj.construction.utils import make_name

def test_ast_compile_table(session, client_with_examples):
    """
    Test compiling the primary table from a query
    
    Includes client_with_examples fixture so that examples are loaded into session
    """
    query = parse("SELECT hard_hat_id, last_name, first_name FROM hard_hats")
    exc = DJException()
    ctx = ast.CompileContext(session=session, query=query, exception=exc)
    query.select.from_.relations[0].primary.compile(ctx)
    assert not exc.errors


def test_ast_compile_table_missing_node(session):
    """
    Test compiling a table when the node is missing
    """
    query = parse("SELECT a FROM foo")
    exc = DJException()
    ctx = ast.CompileContext(session=session, query=query, exception=exc)
    query.select.from_.relations[0].primary.compile(ctx)
    assert f"No node `foo` exists of kind" in exc.errors[0].message
    
    query = parse("SELECT a FROM foo, bar, baz")
    exc = DJException()
    ctx = ast.CompileContext(session=session, query=query, exception=exc)
    query.select.from_.relations[0].primary.compile(ctx)
    assert f"No node `foo` exists of kind" in exc.errors[0].message
    query.select.from_.relations[1].primary.compile(ctx)
    assert f"No node `bar` exists of kind" in exc.errors[1].message
    query.select.from_.relations[2].primary.compile(ctx)
    assert f"No node `baz` exists of kind" in exc.errors[2].message

    query = parse("SELECT a FROM foo LEFT JOIN bar")
    exc = DJException()
    ctx = ast.CompileContext(session=session, query=query, exception=exc)
    query.select.from_.relations[0].primary.compile(ctx)
    assert f"No node `foo` exists of kind" in exc.errors[0].message
    query.select.from_.relations[0].extensions[0].right.compile(ctx)
    assert f"No node `bar` exists of kind" in exc.errors[1].message

    query = parse("SELECT a FROM foo LEFT JOIN (SELECT b FROM bar) b")
    exc = DJException()
    ctx = ast.CompileContext(session=session, query=query, exception=exc)
    query.select.from_.relations[0].primary.compile(ctx)
    assert f"No node `foo` exists of kind" in exc.errors[0].message
    query.select.from_.relations[0].extensions[0].right.select.from_.relations[0].primary.compile(ctx)
    assert f"No node `bar` exists of kind" in exc.errors[1].message


def test_ast_compile_query(session, client_with_examples):
    """
    Test compiling an entire query
    """
    query = parse("SELECT hard_hat_id, last_name, first_name FROM hard_hats")
    exc = DJException()
    ctx = ast.CompileContext(session=session, query=query, exception=exc)
    query.compile(ctx)
    assert not exc.errors


def test_ast_compile_query_missing_columns(session, client_with_examples):
    """
    Test compiling a query with missing columns
    """
    query = parse("SELECT hard_hat_id, column_foo, column_bar FROM hard_hats")
    exc = DJException()
    ctx = ast.CompileContext(session=session, query=query, exception=exc)
    query.compile(ctx)
    assert "Column`column_foo` does not exist on any valid table." in exc.errors[0].message
    assert "Column`column_bar` does not exist on any valid table." in exc.errors[1].message


@pytest.mark.parametrize(
    "name,expected_make_name",
    [
        (ast.Name("d", namespace=ast.Name(name="c", namespace=ast.Name("b", namespace=ast.Name("a")))), "a.b.c.d"),
        (ast.Name("node-name", namespace=ast.Name("b", namespace=ast.Name("a"))), "a.b.node-name"),
        (ast.Name("node-[name]"), "node-[name]"),
        (ast.Name("c", namespace=ast.Name("b", namespace=ast.Name("a"))), "a.b.c"),
        (
            ast.Name("node&(name)", namespace=ast.Name("c", namespace=ast.Name("b", namespace=ast.Name("a")))),
            "a.b.c.node&(name)",
        ),
        (
            ast.Name("+d", namespace=ast.Name("c", namespace=ast.Name("b", namespace=ast.Name("a")))),
            "a.b.c.+d",
        ),
        (
            ast.Name("-d", namespace=ast.Name("c", namespace=ast.Name("b", namespace=ast.Name("a")))),
            "a.b.c.-d",
        ),
        (
            ast.Name("~~d", namespace=ast.Name("c", namespace=ast.Name("b", namespace=ast.Name("a")))),
            "a.b.c.~~d",
        ),
    ],
)
def test_make_name(
    name: ast.Name,
    expected_make_name: str,
):
    """
    Test making names from a namespace and a name
    """
    assert make_name(name) == expected_make_name


def test_ast_compile_missing_references(session: Session):
    """
    Test getting dependencies from a query that has dangling references when set not to raise
    """
    query = parse("select a, b, c from does_not_exist")
    exc = DJException()
    ctx = ast.CompileContext(session=session, query=query, exception=exc)
    query.compile(ctx)
    _, danglers = query.extract_dependencies()
    assert "does_not_exist" in danglers


def test_ast_compile_raise_on_ambiguous_column(session: Session, client_with_examples):
    """
    Test raising on ambiguous column
    """
    query = parse(
        "SELECT country FROM basic.transform.country_agg a "
        "LEFT JOIN basic.dimension.countries b on a.country = b.country",
    )
    exc = DJException()
    ctx = ast.CompileContext(session=session, query=query, exception=exc)
    query.compile(ctx)
    assert "Column `country` found in multiple tables. Consider namespacing." in exc.errors[0].message


def test_ast_compile_having(session: Session, client_with_examples):
    """
    Test using having
    """
    query = parse(
        "SELECT order_date, status FROM dbt.source.jaffle_shop.orders "
        "GROUP BY dbt.dimension.customers.id "
        "HAVING dbt.dimension.customers.id=1"
    )
    exc = DJException()
    ctx = ast.CompileContext(session=session, query=query, exception=exc)
    query.compile(ctx)
    assert not exc.errors