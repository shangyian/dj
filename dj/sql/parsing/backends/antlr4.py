import logging

import antlr4
from antlr4 import InputStream
from antlr4 import RecognitionException
from antlr4.error.ErrorListener import ErrorListener
from antlr4.error.ErrorStrategy import BailErrorStrategy
from antlr4.error.Errors import ParseCancellationException


from dj.sql.parsing.backends.grammar.generated.SqlBaseLexer import SqlBaseLexer
from dj.sql.parsing.backends.grammar.generated.SqlBaseParser import SqlBaseParser
from dj.sql.parsing.ast2 import BinaryOp, Query, Organization, SortItem, Select, Name, Number, Function, Wildcard, Table, Join, From

logger = logging.getLogger(__name__)

class RemoveIdentifierBackticks(antlr4.ParseTreeListener):
    @staticmethod
    def exitQuotedIdentifier(ctx):  # pylint: disable=invalid-name,unused-argument
        def identity(token):
            return token

        return identity

    @staticmethod
    def enterNonReserved(ctx):  # pylint: disable=invalid-name,unused-argument
        def add_backtick(token):
            return "`{0}`".format(token)

        return add_backtick


class ParseErrorListener(ErrorListener):
    def syntaxError(
            self, recognizer, offendingSymbol, line, column, msg, e
    ):  # pylint: disable=invalid-name,no-self-use,too-many-arguments
        raise SqlSyntaxError(f"Parse error {line}:{column}:"
                             f""
                             , msg)


class UpperCaseCharStream:
    """
    Make SQL token detection case insensitive and allow identifier without
    backticks to be seen as e.g. column names
    """

    def __init__(self, wrapped):
        self.wrapped = wrapped

    def getText(self, interval, *args):  # pylint: disable=invalid-name
        if args or (self.size() > 0 and (interval.b - interval.a >= 0)):
            return self.wrapped.getText(interval, *args)
        return ""

    def LA(self, i: int):  # pylint: disable=invalid-name
        token = self.wrapped.LA(i)
        if token in (0, -1):
            return token
        return ord(chr(token).upper())

    def __getattr__(self, item):
        return getattr(self.wrapped, item)


class ExplicitBailErrorStrategy(BailErrorStrategy):
    """
    Bail Error Strategy throws a ParseCancellationException,
    This strategy simply throw a more explicit exception
    """
    def recover(self, recognizer, e: RecognitionException):
        try:
            super(ExplicitBailErrorStrategy, self).recover(recognizer, e)
        except ParseCancellationException:
            raise SqlParsingError from e


class EarlyBailSqlLexer(SqlBaseLexer):
    def recover(self, re: RecognitionException):
        raise SqlLexicalError from re


def build_parser(stream, strict_mode=False, early_bail=True):
    if not strict_mode:
        stream = UpperCaseCharStream(stream)
    if early_bail:
        lexer = EarlyBailSqlLexer(stream)
    else:
        lexer = SqlBaseLexer(stream)
    lexer.removeErrorListeners()
    lexer.addErrorListener(ParseErrorListener())
    token_stream = antlr4.CommonTokenStream(lexer)
    parser = SqlBaseParser(token_stream)
    parser.addParseListener(RemoveIdentifierBackticks())
    parser.removeErrorListeners()
    parser.addErrorListener(ParseErrorListener())
    if early_bail:
        parser._errHandler = ExplicitBailErrorStrategy()
    return parser


class SqlParsingError(Exception):
    pass


class SqlLexicalError(SqlParsingError):
    pass


class SqlSyntaxError(SqlParsingError):
    pass


def string_to_ast(string, rule, *, strict_mode=False, debug=False, early_bail=False):
    parser = build_string_parser(string, strict_mode, early_bail)
    tree = getattr(parser, rule)()
    if debug:
        print_tree(tree, printer=logger.warning)
    return tree


def build_string_parser(string, strict_mode=False, early_bail=True):
    string_as_stream = InputStream(string)
    parser = build_parser(string_as_stream, strict_mode, early_bail)
    return parser


def parse_sql(string, rule, converter, debug=False):
    tree = string_to_ast(string, rule, debug=debug)
    return converter(tree) if converter else tree


def parse_statement(string, converter=None, debug=False):
    return parse_sql(string, 'singleStatement', converter, debug)


def print_tree(tree, printer=print):
    for line in tree_to_strings(tree, indent=0):
        printer(line)


def tree_to_strings(tree, indent=0):
    symbol = ("[" + tree.symbol.text + "]") if hasattr(tree, "symbol") else ""
    node_as_string = type(tree).__name__ + symbol
    result = ["|" + "-" * indent + node_as_string]
    if hasattr(tree, 'children') and tree.children:
        for child in tree.children:
            result += tree_to_strings(child, indent + 1)
    return result


from dj.sql.parsing.backends.antlr4 import parse_statement, print_tree, SqlBaseParser as sbp

from dj.sql.parsing import ast
from dj.sql.parsing.backends.exceptions import DJParseException
import antlr4
from functools import singledispatch
from typing import Tuple

@singledispatch
def visit(ctx):
    import pdb; pdb.set_trace()
    
def visit_children(ctx, nones=False):
    return list(filter(lambda child: child is not None if nones==False else True, map(visit, ctx.children)))

@visit.register
def _(ctx: antlr4.tree.Tree.TerminalNodeImpl):
    return None

@visit.register
def _(ctx: sbp.SingleStatementContext):
    return visit(ctx.statement())


@visit.register
def _(ctx: sbp.StatementDefaultContext):
    return visit(ctx.query())


@visit.register
def _(ctx: sbp.QueryContext):
    ctes = []
    if ctes_ctx := ctx.ctes():
        ctes = visit(ctes_ctx)

    limit, order = visit(ctx.queryOrganization())

    select = visit(ctx.queryTerm())

    return Query(
        ctes=ctes,
        select=select,
    )


@visit.register
def _(ctx: sbp.QueryOrganizationContext):
    
    order=list(map(visit, ctx.order))
    sort=list(map(visit, ctx.sort))
    org = Organization(order, sort)
    return visit(ctx.limit), org

@visit.register
def _(ctx: sbp.SortItemContext):
    expr = visit(ctx.expression())
    order=""
    if ordering:=ctx.ordering:
        order=ordering.text.upper()
    null_order=""
    if null_order:=ctx.nullOrder:
        nulls="NULLS "+null_order.text
    return SortItem(expr, order, nulls)

@visit.register
def _(ctx: sbp.ExpressionContext):
    return visit(ctx.booleanExpression())

@visit.register
def _(ctx: sbp.PredicatedContext):
    if value_expr := ctx.valueExpression():
        return visit(value_expr)
    import pdb; pdb.set_trace()

@visit.register
def _(ctx: sbp.ValueExpressionContext):
    if primary:=ctx.primaryExpression():     
        return visit(primary)

@visit.register
def _(ctx: sbp.ArithmeticBinaryContext):
    return BinaryOp(ctx.operator.text, visit(ctx.left), visit(ctx.right))

    

@visit.register
def _(ctx: sbp.ColumnReferenceContext):
    return visit(ctx.identifier())

@visit.register
def _(ctx: sbp.QueryTermContext):
    #TODO: other branches
    return visit(ctx.queryPrimary())

@visit.register
def _(ctx: sbp.QueryPrimaryContext):
    return visit(ctx.querySpecification())

@visit.register
def _(ctx: sbp.QuerySpecificationContext):
    quantifier, projection = visit(ctx.selectClause())
    from_ = visit(ctx.fromClause())

    return Select(
        projection=projection,
        from_=from_,
    )

@visit.register
def _(ctx: sbp.SelectClauseContext):
    quantifier =""
    if quant:=ctx.setQuantifier():
        quantifier = visit(quant)
    projection = visit(ctx.namedExpressionSeq())
    return quantifier, projection

@visit.register
def _(ctx: sbp.SetQuantifierContext):
    if ctx.DISTINCT():
        return "DISTINCT"
    if ctx.ALL():
        return "ALL"
    return ""

@visit.register
def _(ctx: sbp.NamedExpressionSeqContext):
    return list(map(visit, ctx.children))
    
@visit.register
def _(ctx: sbp.NamedExpressionContext):
    
    expr = visit(ctx.expression())
    if alias:= ctx.name:
        expr.alias=visit(alias)
    return expr

@visit.register
def _(ctx: sbp.ErrorCapturingIdentifierContext):
    name = visit(ctx.identifier())
    if extra:=visit(ctx.errorCapturingIdentifierExtra()):
        name.name+=extra
        name.quote_style='"'
    return name
   
@visit.register
def _(ctx: sbp.ErrorIdentContext):
    names = list(map(visit, ctx.identifier()))
    return "-".join(name.name for name in names)

@visit.register
def _(ctx: sbp.RealIdentContext):
    return ""
    
@visit.register
def _(ctx: sbp.IdentifierContext):
    return visit(ctx.strictIdentifier())
    
@visit.register
def _(ctx: sbp.UnquotedIdentifierContext):
    return Name(ctx.getText())

@visit.register
def _(ctx: sbp.ConstantDefaultContext):
    return visit(ctx.constant())

@visit.register
def _(ctx: sbp.NumericLiteralContext):
    return Number(ctx.number().getText())

@visit.register
def _(ctx: sbp.DereferenceContext):
    base = visit(ctx.base)
    field = visit(ctx.fieldName)
    if isinstance(base, Name):
        field.namespace=base
        return field
    import pdb; pdb.set_trace()
    
@visit.register
def _(ctx: sbp.ColumnReferenceContext):
    return visit(ctx.identifier())


@visit.register
def _(ctx: sbp.FunctionCallContext):
    name = visit(ctx.functionName())
    quantifier = ''
    if quant_ctx:=ctx.setQuantifier():
        quantifier=visit(quant_ctx)
    args = list(map(visit, ctx.argument))
    
    return Function(args, quantifier=quantifier)

@visit.register
def _(ctx: sbp.FunctionNameContext):
    if qual_name:=ctx.qualifiedName():
        return visit(qual_name)
    return Name(ctx.getText())

@visit.register
def _(ctx: sbp.QualifiedNameContext):
    names = visit_children(ctx)
    for i in range(len(names)-1, 0, -1):
        names[i].namespace=names[i-1]
    return names[-1]

@visit.register
def _(ctx: sbp.StarContext):
    namespace = None
    if qual_name:=ctx.qualifiedName():
        namespace= visit(qual_name)
    star = Wildcard()
    star.name.namespace=namespace
    return star

@visit.register
def _(ctx: sbp.FromClauseContext):
    relations = list(map(visit, ctx.relation()))
    laterals = list(map(visit, ctx.lateralView()))
    tables=[rel for rel in relations if isinstance(rel, Table)]
    joins=[rel for rel in relations if isinstance(rel, Join)]
    return From(tables, joins, laterals)

@visit.register
def _(ctx: sbp.RelationContext):
    return visit(ctx.relationPrimary())

@visit.register
def _(ctx: sbp.TableNameContext):
    if ctx.temporalClause():
        import pdb; pdb.set_trace()
        
    name = visit(ctx.multipartIdentifier())
    alias = visit(ctx.tableAlias())
    table = Table(name)
    table.alias = alias
    return table

@visit.register
def _(ctx: sbp.MultipartIdentifierContext):
    names = visit_children(ctx)
    for i in range(len(names)-1, 0, -1):
        names[i].namespace=names[i-1]
    return names[-1]

@visit.register
def _(ctx: sbp.TableAliasContext):
    name = None
    if ident:=ctx.strictIdentifier():
        name = visit(ident)
    if ctx.identifierList():
        import pdb;pdb.set_trace()
    return name

@visit.register
def _(ctx: sbp.QuotedIdentifierAlternativeContext):
    return visit(ctx.quotedIdentifier())

@visit.register
def _(ctx: sbp.QuotedIdentifierContext):
    if ident:=ctx.BACKQUOTED_IDENTIFIER():
        return Name(ident.getText()[1:-1], quote_style="`")
    return Name(ctx.DOUBLEQUOTED_STRING().getText()[1:-1], quote_style='"')

def parse(
    sql: str
) -> ast.Query:
    """
    Parse a string into a DJ ast using the ANTLR4 backend.
    """
    tree = parse_statement(sql)
    query = visit(tree)
    return query
