"""
testing ast Nodes and their methods
"""

import pytest

from dj.errors import DJException
from dj.sql.parsing.backends.antlr4 import parse
from dj.sql.parsing.ast2 import CompileContext


def test_ast_compile_table(session, client_with_examples):
    """
    Test compiling the primary table from a query
    
    Includes client_with_examples fixture so that examples are loaded into session
    """
    query_ast = parse("SELECT hard_hat_id, last_name, first_name FROM hard_hats")
    exc = DJException()
    ctx = CompileContext(session=session, query=query_ast, exception=exc)
    query_ast.select.from_.relations[0].primary.compile(ctx)
    assert not exc.errors



def test_ast_compile_table_missing_node(session):
    query_ast = parse("SELECT a FROM foo")
    exc = DJException()
    ctx = CompileContext(session=session, query=query_ast, exception=exc)
    query_ast.select.from_.relations[0].primary.compile(ctx)
    assert f"No node `foo` exists of kind" in exc.errors[0].message
    
    query_ast = parse("SELECT a FROM foo, bar, baz")
    exc = DJException()
    ctx = CompileContext(session=session, query=query_ast, exception=exc)
    query_ast.select.from_.relations[0].primary.compile(ctx)
    assert f"No node `foo` exists of kind" in exc.errors[0].message
    query_ast.select.from_.relations[1].primary.compile(ctx)
    assert f"No node `bar` exists of kind" in exc.errors[1].message
    query_ast.select.from_.relations[2].primary.compile(ctx)
    assert f"No node `baz` exists of kind" in exc.errors[2].message

    query_ast = parse("SELECT a FROM foo LEFT JOIN bar")
    exc = DJException()
    ctx = CompileContext(session=session, query=query_ast, exception=exc)
    query_ast.select.from_.relations[0].primary.compile(ctx)
    assert f"No node `foo` exists of kind" in exc.errors[0].message
    query_ast.select.from_.relations[0].extensions[0].right.compile(ctx)
    assert f"No node `bar` exists of kind" in exc.errors[1].message

    query_ast = parse("SELECT a FROM foo LEFT JOIN (SELECT b FROM bar) b")
    exc = DJException()
    ctx = CompileContext(session=session, query=query_ast, exception=exc)
    query_ast.select.from_.relations[0].primary.compile(ctx)
    assert f"No node `foo` exists of kind" in exc.errors[0].message
    query_ast.select.from_.relations[0].extensions[0].right.select.from_.relations[0].primary.compile(ctx)
    assert f"No node `bar` exists of kind" in exc.errors[1].message