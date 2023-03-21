"""
test parsing tpcds queries into DJ ASTs
"""

from difflib import SequenceMatcher

import pytest

from dj.sql.parsing.backends.antlr4 import parse as parse_antlr4_to_ast, parse_statement
from dj.sql.parsing.backends.sqloxide import parse
import re

ansi_tpcds_files = [
        ("./ansi/query1.sql"),
        ("./ansi/query2.sql"),
        ("./ansi/query3.sql"),
        ("./ansi/query4.sql"),
        ("./ansi/query5.sql"),
        ("./ansi/query6.sql"),
        ("./ansi/query7.sql"),
        ("./ansi/query8.sql"),
        ("./ansi/query9.sql"),
        ("./ansi/query10.sql"),
        ("./ansi/query11.sql"),
        ("./ansi/query12.sql"),
        ("./ansi/query13.sql"),
        ("./ansi/query14.sql"),
        ("./ansi/query15.sql"),
        ("./ansi/query16.sql"),
        ("./ansi/query17.sql"),
        ("./ansi/query18.sql"),
        ("./ansi/query19.sql"),
        ("./ansi/query20.sql"),
        ("./ansi/query21.sql"),
        ("./ansi/query22.sql"),
        ("./ansi/query23.sql"),
        ("./ansi/query24.sql"),
        ("./ansi/query25.sql"),
        ("./ansi/query26.sql"),
        ("./ansi/query27.sql"),
        ("./ansi/query28.sql"),
        ("./ansi/query29.sql"),
        ("./ansi/query30.sql"),
        ("./ansi/query31.sql"),
        ("./ansi/query32.sql"),
        ("./ansi/query33.sql"),
        ("./ansi/query34.sql"),
        ("./ansi/query35.sql"),
        ("./ansi/query36.sql"),
        ("./ansi/query37.sql"),
        ("./ansi/query38.sql"),
        ("./ansi/query39.sql"),
        ("./ansi/query40.sql"),
        ("./ansi/query41.sql"),
        ("./ansi/query42.sql"),
        ("./ansi/query43.sql"),
        ("./ansi/query44.sql"),
        ("./ansi/query45.sql"),
        ("./ansi/query46.sql"),
        ("./ansi/query47.sql"),
        ("./ansi/query48.sql"),
        ("./ansi/query49.sql"),
        ("./ansi/query50.sql"),
        ("./ansi/query51.sql"),
        ("./ansi/query52.sql"),
        ("./ansi/query53.sql"),
        ("./ansi/query54.sql"),
        ("./ansi/query55.sql"),
        ("./ansi/query56.sql"),
        ("./ansi/query57.sql"),
        ("./ansi/query58.sql"),
        ("./ansi/query59.sql"),
        ("./ansi/query60.sql"),
        ("./ansi/query61.sql"),
        ("./ansi/query62.sql"),
        ("./ansi/query63.sql"),
        ("./ansi/query64.sql"),
        ("./ansi/query65.sql"),
        ("./ansi/query66.sql"),
        ("./ansi/query67.sql"),
        ("./ansi/query68.sql"),
        ("./ansi/query69.sql"),
        ("./ansi/query70.sql"),
        ("./ansi/query71.sql"),
        ("./ansi/query72.sql"),
        ("./ansi/query73.sql"),
        ("./ansi/query74.sql"),
        ("./ansi/query75.sql"),
        ("./ansi/query76.sql"),
        ("./ansi/query77.sql"),
        ("./ansi/query78.sql"),
        ("./ansi/query79.sql"),
        ("./ansi/query80.sql"),
        ("./ansi/query81.sql"),
        ("./ansi/query82.sql"),
        ("./ansi/query83.sql"),
        ("./ansi/query84.sql"),
        ("./ansi/query85.sql"),
        ("./ansi/query86.sql"),
        ("./ansi/query87.sql"),
        ("./ansi/query88.sql"),
        ("./ansi/query89.sql"),
        ("./ansi/query90.sql"),
        ("./ansi/query91.sql"),
        ("./ansi/query92.sql"),
        ("./ansi/query93.sql"),
        ("./ansi/query94.sql"),
        ("./ansi/query95.sql"),
        ("./ansi/query96.sql"),
        ("./ansi/query97.sql"),
        ("./ansi/query98.sql"),
        ("./ansi/query99.sql"),
    ]

spark_tpcds_files = [
        ("./sparksql/query1.sql"),
        ("./sparksql/query2.sql"),
        ("./sparksql/query3.sql"),
        ("./sparksql/query4.sql"),
        ("./sparksql/query5.sql"),
        ("./sparksql/query6.sql"),
        ("./sparksql/query7.sql"),
        ("./sparksql/query8.sql"),
        ("./sparksql/query9.sql"),
        ("./sparksql/query10.sql"),
        ("./sparksql/query11.sql"),
        ("./sparksql/query12.sql"),
        ("./sparksql/query13.sql"),
        ("./sparksql/query14.sql"),
        ("./sparksql/query15.sql"),
        ("./sparksql/query16.sql"),
        ("./sparksql/query17.sql"),
        ("./sparksql/query18.sql"),
        ("./sparksql/query19.sql"),
        ("./sparksql/query20.sql"),
        ("./sparksql/query21.sql"),
        ("./sparksql/query22.sql"),
        ("./sparksql/query23.sql"),
        ("./sparksql/query24.sql"),
        ("./sparksql/query25.sql"),
        ("./sparksql/query26.sql"),
        ("./sparksql/query27.sql"),
        ("./sparksql/query28.sql"),
        ("./sparksql/query29.sql"),
        ("./sparksql/query30.sql"),
        ("./sparksql/query31.sql"),
        ("./sparksql/query32.sql"),
        ("./sparksql/query33.sql"),
        ("./sparksql/query34.sql"),
        ("./sparksql/query35.sql"),
        ("./sparksql/query36.sql"),
        ("./sparksql/query37.sql"),
        ("./sparksql/query38.sql"),
        ("./sparksql/query39.sql"),
        ("./sparksql/query40.sql"),
        ("./sparksql/query41.sql"),
        ("./sparksql/query42.sql"),
        ("./sparksql/query43.sql"),
        ("./sparksql/query44.sql"),
        ("./sparksql/query45.sql"),
        ("./sparksql/query46.sql"),
        ("./sparksql/query47.sql"),
        ("./sparksql/query48.sql"),
        ("./sparksql/query49.sql"),
        ("./sparksql/query50.sql"),
        ("./sparksql/query51.sql"),
        ("./sparksql/query52.sql"),
        ("./sparksql/query53.sql"),
        ("./sparksql/query54.sql"),
        ("./sparksql/query55.sql"),
        ("./sparksql/query56.sql"),
        ("./sparksql/query57.sql"),
        ("./sparksql/query58.sql"),
        ("./sparksql/query59.sql"),
        ("./sparksql/query60.sql"),
        ("./sparksql/query61.sql"),
        ("./sparksql/query62.sql"),
        ("./sparksql/query63.sql"),
        ("./sparksql/query64.sql"),
        ("./sparksql/query65.sql"),
        ("./sparksql/query66.sql"),
        ("./sparksql/query67.sql"),
        ("./sparksql/query68.sql"),
        ("./sparksql/query69.sql"),
        ("./sparksql/query70.sql"),
        ("./sparksql/query71.sql"),
        ("./sparksql/query72.sql"),
        ("./sparksql/query73.sql"),
        ("./sparksql/query74.sql"),
        ("./sparksql/query75.sql"),
        ("./sparksql/query76.sql"),
        ("./sparksql/query77.sql"),
        ("./sparksql/query78.sql"),
        ("./sparksql/query79.sql"),
        ("./sparksql/query80.sql"),
        ("./sparksql/query81.sql"),
        ("./sparksql/query82.sql"),
        ("./sparksql/query83.sql"),
        ("./sparksql/query84.sql"),
        ("./sparksql/query85.sql"),
        ("./sparksql/query86.sql"),
        ("./sparksql/query87.sql"),
        ("./sparksql/query88.sql"),
        ("./sparksql/query89.sql"),
        ("./sparksql/query90.sql"),
        ("./sparksql/query91.sql"),
        ("./sparksql/query92.sql"),
        ("./sparksql/query93.sql"),
        ("./sparksql/query94.sql"),
        ("./sparksql/query95.sql"),
        ("./sparksql/query96.sql"),
        ("./sparksql/query97.sql"),
        ("./sparksql/query98.sql"),
        ("./sparksql/query99.sql"),
    ]

trino_tpcds_files = [
        ("./trino/query1.sql"),
        ("./trino/query2.sql"),
        ("./trino/query3.sql"),
        ("./trino/query4.sql"),
        ("./trino/query5.sql"),
        ("./trino/query6.sql"),
        ("./trino/query7.sql"),
        ("./trino/query8.sql"),
        ("./trino/query9.sql"),
        ("./trino/query10.sql"),
        ("./trino/query11.sql"),
        ("./trino/query12.sql"),
        ("./trino/query13.sql"),
        ("./trino/query14_1.sql"),
        ("./trino/query14_2.sql"),
        ("./trino/query15.sql"),
        ("./trino/query16.sql"),
        ("./trino/query17.sql"),
        ("./trino/query18.sql"),
        ("./trino/query19.sql"),
        ("./trino/query20.sql"),
        ("./trino/query21.sql"),
        ("./trino/query22.sql"),
        ("./trino/query23_1.sql"),
        ("./trino/query23_2.sql"),
        ("./trino/query24_1.sql"),
        ("./trino/query24_2.sql"),
        ("./trino/query25.sql"),
        ("./trino/query26.sql"),
        ("./trino/query27.sql"),
        ("./trino/query28.sql"),
        ("./trino/query29.sql"),
        ("./trino/query30.sql"),
        ("./trino/query31.sql"),
        ("./trino/query32.sql"),
        ("./trino/query33.sql"),
        ("./trino/query34.sql"),
        ("./trino/query35.sql"),
        ("./trino/query36.sql"),
        ("./trino/query37.sql"),
        ("./trino/query38.sql"),
        ("./trino/query39_1.sql"),
        ("./trino/query39_2.sql"),
        ("./trino/query40.sql"),
        ("./trino/query41.sql"),
        ("./trino/query42.sql"),
        ("./trino/query43.sql"),
        ("./trino/query44.sql"),
        ("./trino/query45.sql"),
        ("./trino/query46.sql"),
        ("./trino/query47.sql"),
        ("./trino/query48.sql"),
        ("./trino/query49.sql"),
        ("./trino/query50.sql"),
        ("./trino/query51.sql"),
        ("./trino/query52.sql"),
        ("./trino/query53.sql"),
        ("./trino/query54.sql"),
        ("./trino/query55.sql"),
        ("./trino/query56.sql"),
        ("./trino/query57.sql"),
        ("./trino/query58.sql"),
        ("./trino/query59.sql"),
        ("./trino/query60.sql"),
        ("./trino/query61.sql"),
        ("./trino/query62.sql"),
        ("./trino/query63.sql"),
        ("./trino/query64.sql"),
        ("./trino/query65.sql"),
        ("./trino/query66.sql"),
        ("./trino/query67.sql"),
        ("./trino/query68.sql"),
        ("./trino/query69.sql"),
        ("./trino/query70.sql"),
        ("./trino/query71.sql"),
        ("./trino/query72.sql"),
        ("./trino/query73.sql"),
        ("./trino/query74.sql"),
        ("./trino/query75.sql"),
        ("./trino/query76.sql"),
        ("./trino/query77.sql"),
        ("./trino/query78.sql"),
        ("./trino/query79.sql"),
        ("./trino/query80.sql"),
        ("./trino/query81.sql"),
        ("./trino/query82.sql"),
        ("./trino/query83.sql"),
        ("./trino/query84.sql"),
        ("./trino/query85.sql"),
        ("./trino/query86.sql"),
        ("./trino/query87.sql"),
        ("./trino/query88.sql"),
        ("./trino/query89.sql"),
        ("./trino/query90.sql"),
        ("./trino/query91.sql"),
        ("./trino/query92.sql"),
        ("./trino/query93.sql"),
        ("./trino/query94.sql"),
        ("./trino/query95.sql"),
        ("./trino/query96.sql"),
        ("./trino/query97.sql"),
        ("./trino/query98.sql"),
        ("./trino/query99.sql"),
    ]

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def compare_parse_trees(tree1, tree2):
    """
    Recursively compare two ANTLR parse trees for equality.
    """
    # Check if the node types are the same
    if type(tree1) != type(tree2):
        return False
    
    # Check if the node texts are the same
    if tree1.getText() != tree2.getText():
        return False
    
    # Check if the number of child nodes is the same
    if tree1.getChildCount() != tree2.getChildCount():
        return False
    
    # Recursively compare child nodes
    for i in range(tree1.getChildCount()):
        child1 = tree1.getChild(i)
        child2 = tree2.getChild(i)
        if not compare_parse_trees(child1, child2):
            return False
    
    # If all checks passed, the trees are equal
    return True

AS_RE = re.compile(r'\sAS\s|\sas\s')

def compare_query_strings(str1, str2):
    """
    Recursively compare two ANTLR parse trees for equality, ignoring certain elements.
    """

    str1 = AS_RE.sub(" ", str1)
    str2 = AS_RE.sub(" ", str2)
    tree1 = parse_statement(str1)
    tree2 = parse_statement(str2)
    
    return compare_parse_trees(tree1, tree2)

@pytest.mark.skipif("not config.getoption('tpcds')")
@pytest.mark.parametrize(
    "query_file",
    ansi_tpcds_files + spark_tpcds_files + trino_tpcds_files,
)
def test_tpcds_parse(query_file, request, monkeypatch):
    """
    Test that TPCDS queries parse with no errors
    """
    monkeypatch.chdir(request.fspath.dirname)
    with open(query_file, encoding="UTF-8") as file:
        content = file.read()
        for query in content.split(";"):
            if not query.isspace():
                parse_statement(query)


@pytest.mark.skipif("not config.getoption('tpcds')")
@pytest.mark.parametrize(
    "query_file",
    ansi_tpcds_files + spark_tpcds_files + trino_tpcds_files,
)
def test_tpcds_to_ast(query_file, request, monkeypatch):
    """
    Test that TPCDS queries are converted into DJ ASTs with no errors
    """
    monkeypatch.chdir(request.fspath.dirname)
    with open(query_file, encoding="UTF-8") as file:
        content = file.read()
        for query in content.split(";"):
            if not query.isspace():
                parse_antlr4_to_ast(query)


@pytest.mark.skipif("not config.getoption('tpcds')")
@pytest.mark.parametrize(
    "query_file",
    spark_tpcds_files,
)
def test_tpcds_circular_parse(query_file, request, monkeypatch):
    """
    Test that the string representation of TPCDS DJ ASTs can be re-parsed
    """
    monkeypatch.chdir(request.fspath.dirname)
    with open(query_file, encoding="UTF-8") as file:
        content = file.read()
        for query in content.split(";"):
            if not query.isspace():
                query_ast = parse_antlr4_to_ast(query)
                parse_statement(str(query_ast))


@pytest.mark.skipif("not config.getoption('tpcds')")
@pytest.mark.parametrize(
    "query_file",
    spark_tpcds_files,
)
def test_tpcds_antlr_tree_comparisons(
    query_file,
    request,
    monkeypatch,
):
    """
    Test str -> parse(1) -> DJ AST -> str -> parse(2) and comparing (1) and (2)
    """
    monkeypatch.chdir(request.fspath.dirname)
    with open(query_file, encoding="UTF-8") as file:
        content = file.read()
        for query in content.split(";"):
            if not query.isspace():
                q = parse_antlr4_to_ast(query)
                assert compare_query_strings(query, str(q))


@pytest.mark.skipif("not config.getoption('tpcds')")
@pytest.mark.parametrize(
    "query_file",
    ansi_tpcds_files + spark_tpcds_files + trino_tpcds_files,
)
def test_tpcds_str_similarities(
    query_file,
    request,
    monkeypatch,
):
    """
    Test str -> parse(1) -> DJ AST -> str and checking similarity of (1) and (2)
    """
    monkeypatch.chdir(request.fspath.dirname)
    with open(query_file, encoding="UTF-8") as file:
        content = file.read()
        for query in content.split(";"):
            if not query.isspace():
                query_ast = parse_antlr4_to_ast(query)
                assert similar(query, str(query_ast)) > 0.9
