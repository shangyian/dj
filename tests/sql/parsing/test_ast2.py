"""
testing ast Nodes and their methods
"""

import pytest

from fastapi.testclient import TestClient
from sqlmodel import Session

from dj.errors import DJException
from dj.sql.parsing.backends.antlr4 import parse
from dj.sql.parsing import ast2 as ast
from dj.sql.parsing import types
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
    
    node = query.select.from_.relations[0].primary._dj_node
    assert node
    assert node.name == "hard_hats"


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
    
    node = query.select.from_.relations[0].primary._dj_node
    assert node
    assert node.name == "hard_hats"


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
    
    node = query.select.from_.relations[0].primary._dj_node
    assert node
    assert node.name == "hard_hats"


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
    
    node = query.select.from_.relations[0].primary._dj_node
    assert node
    assert node.name == "dbt.source.jaffle_shop.orders"


def test_ast_compile_lateral_view_explode1(session: Session):
    """
    Test lateral view explode
    """

    query = parse("""SELECT a, b, c, c_age.col, d_age.col
    FROM (SELECT 1 as a, 2 as b, 3 as c, ARRAY(30,60) as d, ARRAY(40,80) as e) AS foo
    LATERAL VIEW EXPLODE(d) c_age
    LATERAL VIEW EXPLODE(e) d_age;
    """)
    exc = DJException()
    ctx = ast.CompileContext(session=session, query=query, exception=exc)
    query.compile(ctx)

    assert not exc.errors
    
    assert query.columns[0].is_compiled()
    assert query.columns[1].is_compiled()
    assert query.columns[2].is_compiled()
    assert query.columns[3].is_compiled()
    assert query.columns[4].is_compiled()
    assert query.columns[0].name == ast.Name(name="a", quote_style="", namespace=None)
    assert query.columns[1].name == ast.Name(name="b", quote_style="", namespace=None)
    assert query.columns[2].name == ast.Name(name="c", quote_style="", namespace=None)
    assert query.columns[3].name == ast.Name(name="c_age", quote_style="", namespace=None)
    assert query.columns[4].name == ast.Name(name="d_age", quote_style="", namespace=None)
    assert isinstance(query.columns[0].type, types.IntegerType)
    assert isinstance(query.columns[1].type, types.IntegerType)
    assert isinstance(query.columns[2].type, types.IntegerType)
    assert isinstance(query.columns[3].type, types.IntegerType)
    assert isinstance(query.columns[4].type, types.IntegerType)
    assert query.columns[0].table.alias_or_name == ast.Name(name="foo", quote_style="", namespace=None)
    assert query.columns[1].table.alias_or_name == ast.Name(name="foo", quote_style="", namespace=None)
    assert query.columns[2].table.alias_or_name == ast.Name(name="foo", quote_style="", namespace=None)
    assert query.columns[3].table.alias_or_name == ast.Name(name="foo", quote_style="", namespace=None)
    assert query.columns[4].table.alias_or_name == ast.Name(name="foo", quote_style="", namespace=None)


def test_ast_compile_lateral_view_explode2(session: Session):
    """
    Test lateral view explode
    """
    
    query = parse("""SELECT a, b, c, c_age, d_age
    FROM (SELECT 1 as a, 2 as b, 3 as c, ARRAY(30,60) as d, ARRAY(40,80) as e) AS foo
    LATERAL VIEW EXPLODE(d) AS c_age
    LATERAL VIEW EXPLODE(e) AS d_age;""")
    exc = DJException()
    ctx = ast.CompileContext(session=session, query=query, exception=exc)
    query.compile(ctx)

    assert query.columns[0].is_compiled()
    assert query.columns[1].is_compiled()
    assert query.columns[2].is_compiled()
    assert query.columns[3].is_compiled()
    assert query.columns[4].is_compiled()
    assert query.columns[0].name == ast.Name(name="a", quote_style="", namespace=None)
    assert query.columns[1].name == ast.Name(name="b", quote_style="", namespace=None)
    assert query.columns[2].name == ast.Name(name="c", quote_style="", namespace=None)
    assert query.columns[3].name == ast.Name(name="c_age", quote_style="", namespace=None)
    assert query.columns[4].name == ast.Name(name="d_age", quote_style="", namespace=None)
    assert isinstance(query.columns[0].type, types.IntegerType)
    assert isinstance(query.columns[1].type, types.IntegerType)
    assert isinstance(query.columns[2].type, types.IntegerType)
    assert isinstance(query.columns[3].type, types.IntegerType)
    assert isinstance(query.columns[4].type, types.IntegerType)
    assert query.columns[0].table.alias_or_name == ast.Name(name="foo", quote_style="", namespace=None)
    assert query.columns[1].table.alias_or_name == ast.Name(name="foo", quote_style="", namespace=None)
    assert query.columns[2].table.alias_or_name == ast.Name(name="foo", quote_style="", namespace=None)
    assert query.columns[3].table.alias_or_name == ast.Name(name="foo", quote_style="", namespace=None)
    assert query.columns[4].table.alias_or_name == ast.Name(name="foo", quote_style="", namespace=None)


def test_ast_compile_lateral_view_explode3(session: Session):
    """
    Test lateral view explode
    """

    query = parse("""SELECT a, b, c, d, e
    FROM (SELECT 1 as a, 2 as b, 3 as c) AS foo
    LATERAL VIEW EXPLODE(ARRAY(30, 60)) AS d
    LATERAL VIEW EXPLODE(ARRAY(40, 80)) AS e;""")
    exc = DJException()
    ctx = ast.CompileContext(session=session, query=query, exception=exc)
    query.compile(ctx)

    assert query.columns[0].is_compiled()
    assert query.columns[1].is_compiled()
    assert query.columns[2].is_compiled()
    assert query.columns[3].is_compiled()
    assert query.columns[4].is_compiled()
    assert query.columns[0].name == ast.Name(name="a", quote_style="", namespace=None)
    assert query.columns[1].name == ast.Name(name="b", quote_style="", namespace=None)
    assert query.columns[2].name == ast.Name(name="c", quote_style="", namespace=None)
    assert query.columns[3].name == ast.Name(name="d", quote_style="", namespace=None)
    assert query.columns[4].name == ast.Name(name="e", quote_style="", namespace=None)
    assert isinstance(query.columns[0].type, types.IntegerType)
    assert isinstance(query.columns[1].type, types.IntegerType)
    assert isinstance(query.columns[2].type, types.IntegerType)
    assert isinstance(query.columns[3].type, types.IntegerType)
    assert isinstance(query.columns[4].type, types.IntegerType)
    assert query.columns[0].table.alias_or_name == ast.Name(name="foo", quote_style="", namespace=None)
    assert query.columns[1].table.alias_or_name == ast.Name(name="foo", quote_style="", namespace=None)
    assert query.columns[2].table.alias_or_name == ast.Name(name="foo", quote_style="", namespace=None)
    assert query.columns[3].table.alias_or_name == ast.Name(name="foo", quote_style="", namespace=None)
    assert query.columns[4].table.alias_or_name == ast.Name(name="foo", quote_style="", namespace=None)


def test_ast_compile_lateral_view_explode4(session: Session, client: TestClient):
    """
    Test lateral view explode
    """
    response = client.post("/catalogs/", json={"name": "default"})
    assert response.ok
    response = client.post("/nodes/source/", json={
            "columns": {
                "a": {"type": "int"},
            },
            "description": "Placeholder source node",
            "mode": "published",
            "name": "a",
            "catalog": "default",
            "schema_": "a",
            "table": "a",
        }
    )
    assert response.ok
    response = client.post("/nodes/transform/", json={
            "description": "A projection with an array",
            "query": "SELECT ARRAY(30, 60) as foo_array FROM a",
            "mode": "published",
            "name": "foo_array_example",
        }
    )
    assert response.ok

    query = parse("""
        SELECT foo_array, a, b
        FROM foo_array_example
        LATERAL VIEW EXPLODE(foo_array) AS a
        LATERAL VIEW EXPLODE(foo_array) AS b;
    """)
    exc = DJException()
    ctx = ast.CompileContext(session=session, query=query, exception=exc)
    query.compile(ctx)

    assert query.columns[0].is_compiled()
    assert query.columns[1].is_compiled()
    assert query.columns[2].is_compiled()
    assert query.columns[0].name == ast.Name(name="foo_array", quote_style="", namespace=None)
    assert query.columns[1].name == ast.Name(name="a", quote_style="", namespace=None)
    assert query.columns[2].name == ast.Name(name="b", quote_style="", namespace=None)
    assert isinstance(query.columns[0].type, types.IntegerType)
    assert isinstance(query.columns[1].type, types.IntegerType)
    assert isinstance(query.columns[2].type, types.IntegerType)
    assert query.columns[0].table.alias_or_name == ast.Name(name="foo_array_example", quote_style="", namespace=None)
    assert query.columns[1].table.alias_or_name == ast.Name(name="foo_array_example", quote_style="", namespace=None)
    assert query.columns[2].table.alias_or_name == ast.Name(name="foo_array_example", quote_style="", namespace=None)
