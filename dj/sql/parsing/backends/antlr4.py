import inspect
import logging

import antlr4
from antlr4 import InputStream, RecognitionException
from antlr4.error.ErrorListener import ErrorListener
from antlr4.error.Errors import ParseCancellationException
from antlr4.error.ErrorStrategy import BailErrorStrategy

from dj.sql.parsing import ast2 as ast
from dj.sql.parsing.backends.exceptions import DJParseException
from dj.sql.parsing.backends.grammar.generated.SqlBaseLexer import SqlBaseLexer
from dj.sql.parsing.backends.grammar.generated.SqlBaseParser import SqlBaseParser
from dj.sql.parsing.backends.grammar.generated.SqlBaseParser import SqlBaseParser as sbp

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
        self, recognizer, offendingSymbol, line, column, msg, e,
    ):  # pylint: disable=invalid-name,no-self-use,too-many-arguments
        raise SqlSyntaxError(f"Parse error {line}:{column}:" f"", msg)


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
    return parse_sql(string, "singleStatement", converter, debug)


def print_tree(tree, printer=print):
    for line in tree_to_strings(tree, indent=0):
        printer(line)


def tree_to_strings(tree, indent=0):
    symbol = ("[" + tree.symbol.text + "]") if hasattr(tree, "symbol") else ""
    node_as_string = type(tree).__name__ + symbol
    result = ["|" + "-" * indent + node_as_string]
    if hasattr(tree, "children") and tree.children:
        for child in tree.children:
            result += tree_to_strings(child, indent + 1)
    return result

def context_parenthesis(ctx):
    if hasattr(ctx, "LEFT_PAREN") and hasattr(ctx, "RIGHT_PAREN") and ctx.LEFT_PAREN() and ctx.RIGHT_PAREN():
        return True
    return False

class Visitor:
    def __init__(self):
        self.registry = {}

    def register(self, func):
        params = inspect.signature(func).parameters
        type_ = params[list(params.keys())[0]].annotation
        if type_ == inspect.Parameter.empty:
            raise ValueError(
                "No type annotation found for the first parameter of the visitor.",
            )
        if type_ in self.registry:
            raise ValueError(
                f"A visitor is already registered for type {type_.__name__}.",
            )
        self.registry[type_] = func
        return func

    def __call__(self, ctx):
        if type(ctx) == antlr4.tree.Tree.TerminalNodeImpl:
            return None
        func = self.registry.get(type(ctx), None)
        if func is None:
            line, col = ctx.start.line, ctx.start.column
            raise TypeError(f"{line}:{col} No visitor registered for type {type(ctx).__name__}")
        result = func(ctx)
        if result is None:
            line, col = ctx.start.line, ctx.start.column
            raise DJParseException(f"{line}:{col} Could not parse {ctx.getText()}")
        if isinstance(result, ast.Expression) and context_parenthesis(ctx):
            result.set_parenthesized(True)
        return result


visit = Visitor()


@visit.register
def _(ctx: list, nones=False):
    return list(
        filter(
            lambda child: child is not None if nones == False else True, map(visit, ctx),
        ),
    )


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
    limit, organization = visit(ctx.queryOrganization())

    select = visit(ctx.queryTerm())

    return ast.Query(ctes=ctes, select=select, limit=limit, organization=organization)


@visit.register
def _(ctx: sbp.QueryOrganizationContext):

    order = visit((ctx.order))
    sort = visit((ctx.sort))
    org = ast.Organization(order, sort)
    limit = None
    if ctx.limit:
        limit = visit(ctx.limit)
    return limit, org


@visit.register
def _(ctx: sbp.SortItemContext):
    expr = visit(ctx.expression())
    order = ""
    if ordering := ctx.ordering:
        order = ordering.text.upper()
    nulls = ""
    if null_order := ctx.nullOrder:
        nulls = "NULLS " + null_order.text
    return ast.SortItem(expr, order, nulls)


@visit.register
def _(ctx: sbp.ExpressionContext):
    return visit(ctx.booleanExpression())


@visit.register
def _(ctx: sbp.PredicatedContext):
    if value_expr := ctx.valueExpression():
        return visit(value_expr)


@visit.register
def _(ctx: sbp.ValueExpressionContext):
    if primary := ctx.primaryExpression():
        return visit(primary)


@visit.register
def _(ctx: sbp.ValueExpressionDefaultContext):
    return visit(ctx.primaryExpression())


@visit.register
def _(ctx: sbp.ArithmeticBinaryContext):
    return ast.BinaryOp(ctx.operator.text, visit(ctx.left), visit(ctx.right))


@visit.register
def _(ctx: sbp.ColumnReferenceContext):
    return ast.Column(visit(ctx.identifier()))


@visit.register
def _(ctx: sbp.QueryTermDefaultContext):
    return visit(ctx.queryPrimary())


@visit.register
def _(ctx: sbp.QueryPrimaryDefaultContext):
    return visit(ctx.querySpecification())


@visit.register
def _(ctx: sbp.QueryTermContext):
    if primary_query := ctx.queryPrimary():
        return visit(primary_query)


@visit.register
def _(ctx: sbp.QueryPrimaryContext):
    return visit(ctx.querySpecification())


@visit.register
def _(ctx: sbp.RegularQuerySpecificationContext):
    quantifier, projection = visit(ctx.selectClause())
    from_ = visit(ctx.fromClause()) if ctx.fromClause() else None
    where = None
    if where_clause:=ctx.whereClause():
        where = visit(where_clause)
    return ast.Select(
        quantifier=quantifier,
        projection=projection,
        from_=from_,
        where=where
    )


@visit.register
def _(ctx: sbp.SelectClauseContext):
    quantifier = ""
    if quant := ctx.setQuantifier():
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
    return visit(ctx.children)


@visit.register
def _(ctx: sbp.NamedExpressionContext):
    expr = visit(ctx.expression())
    if alias := ctx.name:
        expr.set_alias(visit(alias))
    return expr


@visit.register
def _(ctx: sbp.ErrorCapturingIdentifierContext):
    name = visit(ctx.identifier())
    if extra := visit(ctx.errorCapturingIdentifierExtra()):
        name.name += extra
        name.quote_style = '"'
    return name


@visit.register
def _(ctx: sbp.ErrorIdentContext):
    return ctx.getText()


@visit.register
def _(ctx: sbp.RealIdentContext):
    return ""


@visit.register
def _(ctx: sbp.IdentifierContext):
    return visit(ctx.strictIdentifier())


@visit.register
def _(ctx: sbp.UnquotedIdentifierContext):
    return ast.Name(ctx.getText())


@visit.register
def _(ctx: sbp.ConstantDefaultContext):
    return visit(ctx.constant())


@visit.register
def _(ctx: sbp.NumericLiteralContext):
    return ast.Number(ctx.number().getText())


@visit.register
def _(ctx: sbp.DereferenceContext):
    base = visit(ctx.base)
    field = visit(ctx.fieldName)
    field.namespace = base.name
    base.name = field
    return base


@visit.register
def _(ctx: sbp.FunctionCallContext):
    name = visit(ctx.functionName())
    quantifier = ""
    if quant_ctx := ctx.setQuantifier():
        quantifier = visit(quant_ctx)
    args = visit((ctx.argument))

    return ast.Function(name, args, quantifier=quantifier)


@visit.register
def _(ctx: sbp.FunctionNameContext):
    if qual_name := ctx.qualifiedName():
        return visit(qual_name)
    return ast.Name(ctx.getText())


@visit.register
def _(ctx: sbp.QualifiedNameContext):
    names = visit(ctx.children)
    for i in range(len(names) - 1, 0, -1):
        names[i].namespace = names[i - 1]
    return names[-1]


@visit.register
def _(ctx: sbp.StarContext):

    namespace = None
    if qual_name := ctx.qualifiedName():
        namespace = visit(qual_name)
    star = ast.Wildcard()
    star.name.namespace = namespace
    return star


@visit.register
def _(ctx: sbp.TableNameContext):
    if ctx.temporalClause():
        return
    name = visit(ctx.multipartIdentifier())
    alias, cols = visit(ctx.tableAlias())
    table = ast.Table(name, column_list=cols)
    if alias:
        table.set_alias(ast.Name(alias))
    return table


@visit.register
def _(ctx: sbp.MultipartIdentifierContext):
    names = visit(ctx.children)
    for i in range(len(names) - 1, 0, -1):
        names[i].namespace = names[i - 1]
    return names[-1]


@visit.register
def _(ctx: sbp.QuotedIdentifierAlternativeContext):
    return visit(ctx.quotedIdentifier())


@visit.register
def _(ctx: sbp.QuotedIdentifierContext):
    if ident := ctx.BACKQUOTED_IDENTIFIER():
        return ast.Name(ident.getText()[1:-1], quote_style="`")
    return ast.Name(ctx.DOUBLEQUOTED_STRING().getText()[1:-1], quote_style='"')

@visit.register
def _(ctx: sbp.LateralViewContext):
    outer = bool(ctx.OUTER())
    func_name = visit(ctx.qualifiedName())
    func_args = visit(ctx.expression())
    func = ast.FunctionTable(func_name, func_args)
    table = ast.Table(visit(ctx.tblName))
    columns = [ast.Column(name) for name in visit(ctx.colName)]
    return ast.LateralView(outer, func, table, columns)

@visit.register
def _(ctx: sbp.RelationContext):
    rels = []
    if rel_ext:=ctx.relationExtension():
        rels = visit(rel_ext)
    return visit(ctx.relationPrimary()), rels

@visit.register
def _(ctx: sbp.RelationExtensionContext):
    if join_rel:=ctx.joinRelation():
        return visit(join_rel)
    
@visit.register
def _(ctx: sbp.JoinTypeContext):
    return ctx.getText()
    
@visit.register
def _(ctx: sbp.JoinRelationContext):
    kind = visit(ctx.joinType())
    lateral = bool(ctx.LATERAL())
    natural = bool(ctx.NATURAL())
    criteria = None
    right = visit(ctx.right)
    if join_criteria:=ctx.joinCriteria():
        criteria = visit(join_criteria)
    return ast.Join(kind, right, criteria=criteria, lateral=lateral, natural=natural)

@visit.register
def _(ctx: sbp.TableValuedFunctionContext):
    return visit(ctx.functionTable())

@visit.register
def _(ctx: sbp.FunctionTableContext):
    name = visit(ctx.funcName)
    args = visit(ctx.expression())
    alias, cols = visit(ctx.tableAlias())
    return ast.FunctionTable(name, args = args, column_list = [ast.Column(name) for name in cols]).set_alias(alias)

@visit.register
def _(ctx: sbp.TableAliasContext):
    if ident:=ctx.strictIdentifier():
        identifier_list = visit(ctx.identifierList()) if ctx.identifierList() else []
        return visit(ident), identifier_list
    return None, []

@visit.register
def _(ctx: sbp.IdentifierListContext):
    return visit(ctx.identifierSeq())

@visit.register
def _(ctx: sbp.IdentifierSeqContext):
    return visit(ctx.ident)

@visit.register
def _(ctx: sbp.FromClauseContext):
    relations = visit((ctx.relation()))
    laterals = visit((ctx.lateralView()))
    tables = [rel[0] for rel in relations]
    joins = list(ast.flatten([rel[1] for rel in relations]))
    if ctx.pivotClause() or ctx.unpivotClause():
        return
    return ast.From(tables, joins, laterals)

@visit.register
def _(ctx: sbp.JoinCriteriaContext):
    if expr:=ctx.booleanExpression():
        return ast.JoinCriteria(on=visit(expr))
    return ast.JoinCriteria(using=(visit(ctx.identifierList())))
    
@visit.register
def _(ctx: sbp.ComparisonContext):
    left, right = visit(ctx.left), visit(ctx.right)
    op = visit(ctx.comparisonOperator())
    return ast.BinaryOp(op, left, right)
    
@visit.register
def _(ctx: sbp.ComparisonOperatorContext):
    return ctx.getText()

@visit.register
def _(ctx: sbp.WhereClauseContext):
    return visit(ctx.booleanExpression())

@visit.register
def _(ctx: sbp.StringLiteralContext):
    return ast.String(ctx.getText())

@visit.register
def _(ctx: sbp.CtesContext):
    names = {}
    selects = []
    for namedQuery in ctx.namedQuery():
        if namedQuery.name in names:
            raise SqlSyntaxError(f"Duplicate CTE definition names: {namedQuery.name}")
        select = visit(namedQuery.query().queryTerm())
        select.set_alias(namedQuery.name)
        selects.append(select)
    return selects

@visit.register
def _(ctx: sbp.NamedQueryContext):
    return visit(ctx.query())

@visit.register
def _(ctx: sbp.LogicalBinaryContext):
    return ast.BinaryOp(ctx.operator.text, visit(ctx.left), visit(ctx.right))

@visit.register
def _(ctx: sbp.SubqueryExpressionContext):
    return visit(ctx.query())

@visit.register
def _(ctx: sbp.SetOperationContext):
    left = visit(ctx.left)
    right = visit(ctx.left)
    set_op = ast.SetOp(kind="UNION", table=right)
    left.add_set_op(set_op)
    return left

@visit.register
def _(ctx: sbp.AliasedQueryContext):
    ident, _ = visit(ctx.tableAlias())
    query = visit(ctx.query())
    select = query.select
    select.set_alias(ident)
    return select

@visit.register
def _(ctx: sbp.SimpleCaseContext):
    conditions = []
    results = []
    for when in ctx.whenClause():
        condition, result = visit(when)
        conditions.append(condition)
        results.append(result)
    return ast.Case(
        conditions=conditions,
        else_result=visit(ctx.elseExpression) if ctx.elseExpression else None,
        results=results,
    )

@visit.register
def _(ctx: sbp.SearchedCaseContext):
    conditions = []
    results = []
    for when in ctx.whenClause():
        condition, result = visit(when)
        conditions.append(condition)
        results.append(result)
    return ast.Case(
        conditions=conditions,
        else_result=visit(ctx.elseExpression) if ctx.elseExpression else None,
        results=results,
    )

@visit.register
def _(ctx: sbp.WhenClauseContext):
    condition, result = visit(ctx.condition), visit(ctx.result)
    return condition, result

@visit.register
def _(ctx: sbp.ParenthesizedExpressionContext):
    expr = visit(ctx.expression())
    expr.set_parenthesized(True)
    return expr

@visit.register
def _(ctx: sbp.NullLiteralContext):
    return ast.Null(value=None)

@visit.register
def _(ctx: sbp.CastContext):
    data_type = visit(ctx.dataType())
    expression = visit(ctx.expression())
    return ast.Cast(data_type=data_type, expression=expression)

@visit.register
def _(ctx: sbp.PrimitiveDataTypeContext):
    return visit(ctx.identifier())

@visit.register
def _(ctx: sbp.ExistsContext):
    return ast.UnaryOp(op="EXISTS", expr=ctx.query().queryTerm())

@visit.register
def _(ctx: sbp.LogicalNotContext):
    return ast.UnaryOp(op="NOT", expr=visit(ctx.booleanExpression()))

def parse(sql: str) -> ast.Query:
    """
    Parse a string into a DJ ast using the ANTLR4 backend.
    """
    tree = parse_statement(sql)
    query = visit(tree)
    return query
