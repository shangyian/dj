"""
testing ast Nodes and their methods
"""

import pytest

from dj.errors import DJException
from dj.sql.parsing.backends.antlr4 import parse
from dj.sql.parsing.ast2 import CompileContext

@pytest.mark.parametrize("query", [
    ("SELECT a from foo", "foo"),
])
def test_ast_compile_query(session, query, table):
    query_ast = parse(query)
    exc = DJException()
    ctx = CompileContext(session=session, query=query_ast, exception=exc)
    query_ast.select.from_.relations[0].primary.compile(ctx)
    assert f"No node `{table}` exists of kind" in exc.errors[0].message
