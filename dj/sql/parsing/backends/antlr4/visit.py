import antlr4

from dj.sql.parsing import ast
from dj.sql.parsing.backends.antlr4.utils import SqlBaseParser as sbp
from dj.sql.parsing.backends.antlr4.utils import parse_statement, print_tree
from dj.sql.parsing.backends.exceptions import DJParseException


def no_none(f):
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        if result is None:
            raise ValueError("Function %s returned None" % f.name)
        return result

    return wrapper


from functools import singledispatch


@singledispatch
def visit(ctx):
    import pdb

    pdb.set_trace()


def visit_children(ctx, nones=False):
    return list(
        filter(
            lambda child: child is not None if nones == False else True,
            map(visit, ctx.children),
        ),
    )


@no_none
@visit.register
def _(ctx: antlr4.tree.Tree.TerminalNodeImpl):
    return None


@no_none
@visit.register
def _(ctx: sbp.SingleStatementContext):
    return visit(ctx.statement())


@no_none
@visit.register
def _(ctx: sbp.StatementDefaultContext):
    return visit(ctx.query())


@no_none
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


@no_none
@visit.register
def _(ctx: sbp.QueryOrganizationContext):

    order = list(map(visit, ctx.order))
    sort = list(map(visit, ctx.sort))
    org = Organization(order, sort)
    return visit(ctx.limit), org


@no_none
@visit.register
def _(ctx: sbp.SortItemContext):
    expr = visit(ctx.expression())
    order = ""
    if ordering := ctx.ordering:
        order = ordering.text.upper()
    null_order = ""
    if null_order := ctx.nullOrder:
        nulls = "NULLS " + null_order.text
    return SortItem(expr, order, nulls)


@no_none
@visit.register
def _(ctx: sbp.ExpressionContext):
    return visit(ctx.booleanExpression())


@no_none
@visit.register
def _(ctx: sbp.PredicatedContext):
    if value_expr := ctx.valueExpression():
        return visit(value_expr)
    import pdb

    pdb.set_trace()


@no_none
@visit.register
def _(ctx: sbp.ValueExpressionContext):
    if primary := ctx.primaryExpression():
        return visit(primary)
    import pdb

    pdb.set_trace()


@no_none
@visit.register
def _(ctx: sbp.ValueExpressionDefaultContext):
    return visit(ctx.primaryExpression())


@no_none
@visit.register
def _(ctx: sbp.ArithmeticBinaryContext):
    import pdb

    pdb.set_trace()
    return BinaryOp(ctx.operator.text, visit(ctx.left), visit(ctx.right))


@no_none
@visit.register
def _(ctx: sbp.ColumnReferenceContext):
    return Column(visit(ctx.identifier()))


@no_none
@visit.register
def _(ctx: sbp.QueryTermContext):
    # TODO: other branches
    return visit(ctx.queryPrimary())


@no_none
@visit.register
def _(ctx: sbp.QueryPrimaryContext):
    return visit(ctx.querySpecification())


@no_none
@visit.register
def _(ctx: sbp.QuerySpecificationContext):
    quantifier, projection = visit(ctx.selectClause())
    from_ = visit(ctx.fromClause())

    return Select(
        projection=projection,
        from_=from_,
    )


@no_none
@visit.register
def _(ctx: sbp.SelectClauseContext):
    quantifier = ""
    if quant := ctx.setQuantifier():
        quantifier = visit(quant)
    projection = visit(ctx.namedExpressionSeq())
    return quantifier, projection


@no_none
@visit.register
def _(ctx: sbp.SetQuantifierContext):
    if ctx.DISTINCT():
        return "DISTINCT"
    if ctx.ALL():
        return "ALL"
    return ""


@no_none
@visit.register
def _(ctx: sbp.NamedExpressionSeqContext):
    return visit_children(ctx)


@no_none
@visit.register
def _(ctx: sbp.NamedExpressionContext):
    expr = visit(ctx.expression())
    if alias := ctx.name:
        expr.alias = visit(alias)
    return expr


@no_none
@visit.register
def _(ctx: sbp.ErrorCapturingIdentifierContext):
    name = visit(ctx.identifier())
    if extra := visit(ctx.errorCapturingIdentifierExtra()):
        name.name += extra
        name.quote_style = '"'
    return name


@no_none
@visit.register
def _(ctx: sbp.ErrorIdentContext):
    names = list(map(visit, ctx.identifier()))
    return "-".join(name.name for name in names)


@no_none
@visit.register
def _(ctx: sbp.RealIdentContext):
    return ""


@no_none
@visit.register
def _(ctx: sbp.IdentifierContext):
    return visit(ctx.strictIdentifier())


@no_none
@visit.register
def _(ctx: sbp.UnquotedIdentifierContext):
    return Name(ctx.getText())


@no_none
@visit.register
def _(ctx: sbp.ConstantDefaultContext):
    return visit(ctx.constant())


@no_none
@visit.register
def _(ctx: sbp.NumericLiteralContext):
    return Number(ctx.number().getText())


@no_none
@visit.register
def _(ctx: sbp.DereferenceContext):
    base = visit(ctx.base)
    field = visit(ctx.fieldName)
    field.namespace = base.name
    base.name = field
    return base


@no_none
@visit.register
def _(ctx: sbp.FunctionCallContext):
    name = visit(ctx.functionName())
    quantifier = ""
    if quant_ctx := ctx.setQuantifier():
        quantifier = visit(quant_ctx)
    args = list(map(visit, ctx.argument))

    return Function(args, quantifier=quantifier)


@no_none
@visit.register
def _(ctx: sbp.FunctionNameContext):
    if qual_name := ctx.qualifiedName():
        return visit(qual_name)
    return Name(ctx.getText())


@no_none
@visit.register
def _(ctx: sbp.QualifiedNameContext):
    names = visit_children(ctx)
    for i in range(len(names) - 1, 0, -1):
        names[i].namespace = names[i - 1]
    return names[-1]


@no_none
@visit.register
def _(ctx: sbp.StarContext):
    namespace = None
    if qual_name := ctx.qualifiedName():
        namespace = visit(qual_name)
    star = Wildcard()
    star.name.namespace = namespace
    return star


@no_none
@visit.register
def _(ctx: sbp.FromClauseContext):
    relations = list(map(visit, ctx.relation()))
    laterals = list(map(visit, ctx.lateralView()))
    tables = [rel for rel in relations if isinstance(rel, Table)]
    joins = [rel for rel in relations if isinstance(rel, Join)]
    return From(tables, joins, laterals)


@no_none
@visit.register
def _(ctx: sbp.RelationContext):
    return visit(ctx.relationPrimary())


@no_none
@visit.register
def _(ctx: sbp.TableNameContext):
    if ctx.temporalClause():
        import pdb

        pdb.set_trace()

    name = visit(ctx.multipartIdentifier())
    alias = visit(ctx.tableAlias())
    table = Table(name)
    table.alias = alias
    return table


@no_none
@visit.register
def _(ctx: sbp.MultipartIdentifierContext):
    names = visit_children(ctx)
    for i in range(len(names) - 1, 0, -1):
        names[i].namespace = names[i - 1]
    return names[-1]


@no_none
@visit.register
def _(ctx: sbp.TableAliasContext):
    name = None
    if ident := ctx.strictIdentifier():
        name = visit(ident)
    if ctx.identifierList():
        import pdb

        pdb.set_trace()
    return name


@no_none
@visit.register
def _(ctx: sbp.QuotedIdentifierAlternativeContext):
    return visit(ctx.quotedIdentifier())


@no_none
@visit.register
def _(ctx: sbp.QuotedIdentifierContext):
    if ident := ctx.BACKQUOTED_IDENTIFIER():
        return Name(ident.getText()[1:-1], quote_style="`")
    return Name(ctx.DOUBLEQUOTED_STRING().getText()[1:-1], quote_style='"')
