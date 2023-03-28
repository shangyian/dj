"""
Functions for type inference.
"""

# pylint: disable=unused-argument
from decimal import Decimal
from functools import singledispatch
from typing import Callable, Dict

from dj.sql.functions import function_registry
from dj.sql.parsing import ast2
from dj.sql.parsing.backends.exceptions import DJParseException
from dj.typing import ColumnType


@singledispatch
def get_type_of_expression(expression: ast2.Expression) -> ColumnType:
    """
    Get the type of an expression
    """
    raise NotImplementedError(f"Cannot get type of expression {expression}")


@get_type_of_expression.register
def _(expression: ast2.Alias):
    return get_type_of_expression(expression.child)


@get_type_of_expression.register
def _(expression: ast2.Column):
    # column has already determined/stated its type
    if expression._type:  # pylint: disable=W0212
        return expression._type  # pylint: disable=W0212

    # column was derived from some other expression we can get the type of
    if expression.expression:
        type_ = get_type_of_expression(expression.expression)
        expression.add_type(type_)
        return type_

    # column is from a table expression we can look through
    if table_or_alias := expression.table:
        if isinstance(table_or_alias, ast2.Alias):
            table = table_or_alias.child
        else:
            table = table_or_alias
        if isinstance(table, ast2.Table):
            if table.dj_node:
                for col in table.dj_node.columns:  # pragma: no cover
                    if col.name == expression.name.name:
                        expression.add_type(col.type)
                        return col.type
            else:
                raise DJParseException(
                    f"Cannot resolve type of column {expression}. "
                    "column's table does not have a DJ Node.",
                )
        else:
            raise DJParseException(
                f"Cannot resolve type of column {expression}. "
                "DJ does not currently traverse subqueries for type information. "
                "Consider extraction first.",
            )
        # else:#if subquery
        # currently don't even bother checking subqueries.
        # the extract will have built it for us in crucial cases
    raise DJParseException(f"Cannot resolve type of column {expression}.")


@get_type_of_expression.register
def _(expression: ast2.Null):
    return ColumnType.NULL


@get_type_of_expression.register
def _(expression: ast2.String):
    return ColumnType.STR


@get_type_of_expression.register
def _(expression: ast2.Number):
    """
    Determine the type of the numeric expression.
    """
    # expression.__post_init__()

    # We won't assume that anyone wants SHORT by default
    if isinstance(expression.value, int):
        if expression.value <= -2147483648 or expression.value >= 2147483647:
            return ColumnType.LONG
        return ColumnType.INT

    # Arbitrary-precision floating point
    if isinstance(expression.value, Decimal):
        return ColumnType.DECIMAL

    # Double-precision floating point
    if not (1.18e-38 <= abs(expression.value) <= 3.4e38):
        return ColumnType.DOUBLE

    # Single-precision floating point
    return ColumnType.FLOAT


@get_type_of_expression.register
def _(expression: ast2.Boolean):
    return ColumnType.BOOL


@get_type_of_expression.register
def _(expression: ast2.Wildcard):  # pragma: no cover
    return ColumnType.WILDCARD


@get_type_of_expression.register
def _(expression: ast2.Function):  # pragma: no cover
    name = expression.name.name.upper()
    dj_func = function_registry[name]
    return dj_func.infer_type_from_types(
        *(get_type_of_expression(exp) for exp in expression.args)
    )


# @get_type_of_expression.register
# def _(expression: ast2.Raw):
#     return expression.type_


@get_type_of_expression.register
def _(expression: ast2.Subscript):
    type_ = expression.expr.type
    # for _ in expression.index:
    #     type_ = type_.args[1]  # type: ignore
    return type_  # type: ignore


@get_type_of_expression.register
def _(expression: ast2.Cast):
    return expression.data_type


@get_type_of_expression.register
def _(expression: ast2.Case):
    result_types = [
        res.type
        for res in expression.results
        + ([expression.else_result] if expression.else_result else [])
        if res.type != "NULL"
    ]
    if not all(result_types[0] == res for res in result_types):
        raise DJParseException(
            f"Not all the same type in CASE! Found: {', '.join(result_types)}",
        )
    return result_types[0]


@get_type_of_expression.register
def _(expression: ast2.IsNull):
    return ColumnType.BOOL


@get_type_of_expression.register
def _(expression: ast2.In):
    return ColumnType.BOOL


@get_type_of_expression.register
def _(expression: ast2.Select):
    if len(expression.projection) != 1:
        raise DJParseException(
            "Can only infer type of a SELECT when it "
            f"has a single expression in its projection. In {expression}.",
        )
    return get_type_of_expression(expression.projection[0])


@get_type_of_expression.register
def _(expression: ast2.Between):
    expr_type = get_type_of_expression(expression.expr)
    low_type = get_type_of_expression(expression.low)
    high_type = get_type_of_expression(expression.high)
    if expr_type == low_type == high_type:
        return ColumnType.BOOL
    raise DJParseException(
        f"BETWEEN expects all elements to have the same type got "
        f"{expr_type} BETWEEN {low_type} AND {high_type} in {expression}.",
    )


@get_type_of_expression.register
def _(expression: ast2.UnaryOp):
    kind = expression.op
    type_ = get_type_of_expression(expression.expr)

    def raise_unop_exception():
        raise DJParseException(
            "Incompatible type in unary operation "
            f"{expression}. Got {type} in {expression}.",
        )

    UNOP_TYPE_COMBO_LOOKUP: Dict[  # pylint: disable=C0103
        ast2.UnaryOpKind,
        Callable[[ColumnType], ColumnType],
    ] = {
        ast2.UnaryOpKind.Not: lambda type: ColumnType.BOOL
        if type == ColumnType.BOOL
        else raise_unop_exception(),
        ast2.UnaryOpKind.Exists: lambda type: ColumnType.BOOL
        if type == ColumnType.BOOL
        else raise_unop_exception(),
        # ast2.UnaryOpKind.Minus: lambda type: type_
        # if type in (ColumnType.INT, ColumnType.FLOAT)
        # else raise_unop_exception(),
        # ast2.UnaryOpKind.Plus: lambda type: type_
        # if type_ in (ColumnType.INT, ColumnType.FLOAT)
        # else raise_unop_exception(),
    }
    return UNOP_TYPE_COMBO_LOOKUP[kind](type_)


@get_type_of_expression.register
def _(expression: ast2.BinaryOp):
    kind = expression.op
    left_type = get_type_of_expression(expression.left)
    right_type = get_type_of_expression(expression.right)

    def raise_binop_exception():
        raise DJParseException(
            "Incompatible types in binary operation "
            f"{expression}. Got left {left_type}, right {right_type}.",
        )

    BINOP_TYPE_COMBO_LOOKUP: Dict[  # pylint: disable=C0103
        ast2.BinaryOpKind,
        Callable[[ColumnType, ColumnType], ColumnType],
    ] = {
        ast2.BinaryOpKind.And: lambda left, right: ColumnType.BOOL,
        ast2.BinaryOpKind.Or: lambda left, right: ColumnType.BOOL,
        ast2.BinaryOpKind.Is: lambda left, right: ColumnType.BOOL,
        ast2.BinaryOpKind.Eq: lambda left, right: ColumnType.BOOL,
        ast2.BinaryOpKind.NotEq: lambda left, right: ColumnType.BOOL,
        ast2.BinaryOpKind.Gt: lambda left, right: ColumnType.BOOL,
        ast2.BinaryOpKind.Lt: lambda left, right: ColumnType.BOOL,
        ast2.BinaryOpKind.GtEq: lambda left, right: ColumnType.BOOL,
        ast2.BinaryOpKind.LtEq: lambda left, right: ColumnType.BOOL,
        ast2.BinaryOpKind.BitwiseOr: lambda left, right: ColumnType.INT
        if left == right == ColumnType.INT
        else raise_binop_exception(),
        ast2.BinaryOpKind.BitwiseAnd: lambda left, right: ColumnType.INT
        if left == right == ColumnType.INT
        else raise_binop_exception(),
        ast2.BinaryOpKind.BitwiseXor: lambda left, right: ColumnType.INT
        if left == right == ColumnType.INT
        else raise_binop_exception(),
        ast2.BinaryOpKind.Multiply: lambda left, right: left
        if left == right
        else (
            ColumnType.FLOAT
            if {left, right} == {ColumnType.FLOAT, ColumnType.INT}
            else raise_binop_exception()
        ),
        ast2.BinaryOpKind.Divide: lambda left, right: left
        if left == right
        else (
            ColumnType.FLOAT
            if {left, right} == {ColumnType.FLOAT, ColumnType.INT}
            else raise_binop_exception()
        ),
        ast2.BinaryOpKind.Plus: lambda left, right: left
        if left == right
        else (
            ColumnType.FLOAT
            if {left, right} == {ColumnType.FLOAT, ColumnType.INT}
            else raise_binop_exception()
        ),
        ast2.BinaryOpKind.Minus: lambda left, right: left
        if left == right
        else (
            ColumnType.FLOAT
            if {left, right} == {ColumnType.FLOAT, ColumnType.INT}
            else raise_binop_exception()
        ),
        ast2.BinaryOpKind.Modulo: lambda left, right: ColumnType.INT
        if left == right == ColumnType.INT
        else raise_binop_exception(),
        ast2.BinaryOpKind.Like: lambda left, right: ColumnType.BOOL
        if left == right == ColumnType.STR
        else raise_binop_exception(),
    }
    return BINOP_TYPE_COMBO_LOOKUP[kind](left_type, right_type)
