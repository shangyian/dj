"""
Functions for type inference.
"""

# pylint: disable=unused-argument
from decimal import Decimal
from functools import singledispatch
from typing import Callable, Dict

from dj.sql.functions2 import function_registry
from dj.sql.parsing import ast2 as ast
from dj.sql.parsing.backends.exceptions import DJParseException
from dj.typing import ColumnType


@singledispatch
def get_type_of_expression(expression: ast.Expression) -> ColumnType:
    """
    Get the type of an expression
    """
    raise NotImplementedError(f"Cannot get type of expression {expression}")


@get_type_of_expression.register
def _(expression: ast.Alias):
    return get_type_of_expression(expression.child)


@get_type_of_expression.register
def _(expression: ast.Column):
    # column has already determined/stated its type
    if expression._type:  # pylint: disable=W0212
        return expression._type  # pylint: disable=W0212

    # column was derived from some other expression we can get the type of
    if expression.expression:
        type_ = get_type_of_expression(expression.expression)
        expression.add_type(type_)
        return type_

    # Look through a table expression for this column and return its type if found
    if table_or_alias := expression.table:
        table = table_or_alias.child if isinstance(table_or_alias, ast.Alias) else table_or_alias
        if isinstance(table, ast.Table):
            if not table.dj_node:
                raise DJParseException(
                    f"Cannot resolve type of column `{table.name}.{expression.name.name}`: "
                    f"the column's table does not have a DJ Node."
                )
        columns = []
        if isinstance(table, ast.Table):
            columns = table.dj_node.columns
        if isinstance(table, ast.Query):
            columns = table.columns
        for col in columns:  # pragma: no cover
            col_name = col.alias_or_name.name if isinstance(col, ast.Alias) else col.name
            if col_name == expression.name.name:
                expression.add_type(col.type)
                return col.type
        else:
            raise DJParseException(
                f"DJ does not currently traverse subqueries for type "
                f"information. Consider extraction first.",
            )
    raise DJParseException(f"Cannot resolve type of column {expression}.")


@get_type_of_expression.register
def _(expression: ast.Null):
    return ColumnType.NULL


@get_type_of_expression.register
def _(expression: ast.String):
    return ColumnType.STR


@get_type_of_expression.register
def _(expression: ast.Number):
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
def _(expression: ast.Boolean):
    return ColumnType.BOOL


@get_type_of_expression.register
def _(expression: ast.Wildcard):  # pragma: no cover
    return ColumnType.WILDCARD


@get_type_of_expression.register
def _(expression: ast.Function):  # pragma: no cover
    name = expression.name.name.upper()
    dj_func = function_registry[name]
    return dj_func.infer_type(
        *(get_type_of_expression(exp) for exp in expression.args)
    )


# @get_type_of_expression.register
# def _(expression: ast.Raw):
#     return expression.type_


@get_type_of_expression.register
def _(expression: ast.Subscript):
    type_ = expression.expr.type
    return type_.args[1]


@get_type_of_expression.register
def _(expression: ast.Cast):
    return expression.data_type


@get_type_of_expression.register
def _(expression: ast.Case):
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
def _(expression: ast.IsNull):
    return ColumnType.BOOL


@get_type_of_expression.register
def _(expression: ast.In):
    return ColumnType.BOOL


@get_type_of_expression.register
def _(expression: ast.Select):
    if len(expression.projection) != 1:
        raise DJParseException(
            "Can only infer type of a SELECT when it "
            f"has a single expression in its projection. In {expression}.",
        )
    return get_type_of_expression(expression.projection[0])


@get_type_of_expression.register
def _(expression: ast.Query):
    return get_type_of_expression(expression.select)


@get_type_of_expression.register
def _(expression: ast.Between):
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
def _(expression: ast.UnaryOp):
    kind = expression.op
    type_ = get_type_of_expression(expression.expr)

    def raise_unop_exception():
        raise DJParseException(
            "Incompatible type in unary operation "
            f"{expression}. Got {type} in {expression}.",
        )

    UNOP_TYPE_COMBO_LOOKUP: Dict[  # pylint: disable=C0103
        ast.UnaryOpKind,
        Callable[[ColumnType], ColumnType],
    ] = {
        ast.UnaryOpKind.Not: lambda type: ColumnType.BOOL
        if type == ColumnType.BOOL
        else raise_unop_exception(),
        ast.UnaryOpKind.Exists: lambda type: ColumnType.BOOL
        if type == ColumnType.BOOL
        else raise_unop_exception(),
        # ast.UnaryOpKind.Minus: lambda type: type_
        # if type in (ColumnType.INT, ColumnType.FLOAT)
        # else raise_unop_exception(),
        # ast.UnaryOpKind.Plus: lambda type: type_
        # if type_ in (ColumnType.INT, ColumnType.FLOAT)
        # else raise_unop_exception(),
    }
    return UNOP_TYPE_COMBO_LOOKUP[kind](type_)


@get_type_of_expression.register
def _(expression: ast.Like):
    expr_type = get_type_of_expression(expression.expr)
    if expr_type == ColumnType.STR:
        return ColumnType.BOOL
    raise DJParseException(
        f"Incompatible type for {expression}: {expr_type}. Expected STR",
    )


@get_type_of_expression.register
def _(expression: ast.BinaryOp):
    kind = expression.op
    left_type = get_type_of_expression(expression.left)
    right_type = get_type_of_expression(expression.right)

    def raise_binop_exception():
        raise DJParseException(
            "Incompatible types in binary operation "
            f"{expression}. Got left {left_type}, right {right_type}.",
        )

    numeric_types = {
        type_: idx for idx, type_ in
        enumerate([
            ColumnType.DECIMAL, ColumnType.DOUBLE, ColumnType.FLOAT,
            ColumnType.LONG, ColumnType.INT, ColumnType.SHORT,
        ])
    }

    def resolve_numeric_types_binary_operations(left: ColumnType, right: ColumnType):
        if left not in numeric_types or right not in numeric_types:
            raise_binop_exception()
        if left == right:
            return left
        if numeric_types[left] > numeric_types[right]:
            return right
        return left

    BINOP_TYPE_COMBO_LOOKUP: Dict[  # pylint: disable=C0103
        ast.BinaryOpKind,
        Callable[[ColumnType, ColumnType], ColumnType],
    ] = {
        ast.BinaryOpKind.And: lambda left, right: ColumnType.BOOL,
        ast.BinaryOpKind.Or: lambda left, right: ColumnType.BOOL,
        ast.BinaryOpKind.Is: lambda left, right: ColumnType.BOOL,
        ast.BinaryOpKind.Eq: lambda left, right: ColumnType.BOOL,
        ast.BinaryOpKind.NotEq: lambda left, right: ColumnType.BOOL,
        ast.BinaryOpKind.Gt: lambda left, right: ColumnType.BOOL,
        ast.BinaryOpKind.Lt: lambda left, right: ColumnType.BOOL,
        ast.BinaryOpKind.GtEq: lambda left, right: ColumnType.BOOL,
        ast.BinaryOpKind.LtEq: lambda left, right: ColumnType.BOOL,
        ast.BinaryOpKind.BitwiseOr: lambda left, right: ColumnType.INT
        if left == right == ColumnType.INT
        else raise_binop_exception(),
        ast.BinaryOpKind.BitwiseAnd: lambda left, right: ColumnType.INT
        if left == right == ColumnType.INT
        else raise_binop_exception(),
        ast.BinaryOpKind.BitwiseXor: lambda left, right: ColumnType.INT
        if left == right == ColumnType.INT
        else raise_binop_exception(),
        ast.BinaryOpKind.Multiply: resolve_numeric_types_binary_operations,
        ast.BinaryOpKind.Divide: resolve_numeric_types_binary_operations,
        ast.BinaryOpKind.Plus: resolve_numeric_types_binary_operations,
        ast.BinaryOpKind.Minus: resolve_numeric_types_binary_operations,
        ast.BinaryOpKind.Modulo: lambda left, right: ColumnType.INT
        if left == right == ColumnType.INT
        else raise_binop_exception(),
        ast.BinaryOpKind.Like: lambda left, right: ColumnType.BOOL
        if left == right == ColumnType.STR
        else raise_binop_exception(),
    }
    return BINOP_TYPE_COMBO_LOOKUP[kind](left_type, right_type)
