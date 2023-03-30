"""
SQL functions for type inference.

This file holds all the functions that we want to support in the SQL used to define
nodes. The functions are used to infer types.

"""
import re
# pylint: disable=unused-argument, missing-function-docstring, arguments-differ, too-many-return-statements
from typing import ClassVar, List, Union
import abc
import dj.sql.parsing.types as ct
from dj.errors import DJInvalidInputException, DJError, ErrorCode


class Function(abc.ABC):  # pylint: disable=too-few-public-methods
    """
    A DJ function.
    """

    is_aggregation: ClassVar[bool] = False

    @staticmethod
    @abc.abstractmethod
    def infer_type(*args: ct.ColumnType) -> ct.ColumnType:
        """
        Infers the return type of the function based on the input type.
        """
        raise NotImplementedError("Subclass MUST implement infer_type")

class TableFunction(abc.ABC):  # pylint: disable=too-few-public-methods
    """
    A DJ table-valued function.
    """

    @staticmethod
    @abc.abstractmethod
    def infer_type(*args: ct.ColumnType) -> List[ct.ColumnType]:
        """
        Infers the return type of the function based on the input type.
        """
        raise NotImplementedError("Subclass MUST implement infer_type")


class Avg(Function):
    """
    Computes the average of the input column or expression.
    """
    is_aggregation = True

    @staticmethod
    def infer_type(arg) -> ct.DoubleType:
        return ct.DoubleType()


class Min(Function):
    """
    Computes the minimum value of the input column or expression.
    """
    is_aggregation = True

    @staticmethod
    def infer_type(arg) -> ct.ColumnType:
        return arg


class Max(Function):
    """
    Computes the maximum value of the input column or expression.
    """
    is_aggregation = True

    @staticmethod
    def infer_type(arg) -> ct.ColumnType:
        return arg


class Sum(Function):
    """
    Computes the sum of the input column or expression.
    """
    is_aggregation = True

    @staticmethod
    def infer_type(arg) -> ct.ColumnType:
        return arg


class Ceil(Function):
    """
    Computes the smallest integer greater than or equal to the input value.
    """
    @staticmethod
    def infer_type(arg) -> ct.IntegerType:
        return ct.IntegerType()


class Count(Function):
    """
    Counts the number of non-null values in the input column or expression.
    """
    is_aggregation = True

    @staticmethod
    def infer_type(arg) -> ct.LongType:
        return ct.LongType()


class Coalesce(Function):
    """
    Computes the average of the input column or expression.
    """
    is_aggregation = False

    @staticmethod
    def infer_type(*args) -> ct.ColumnType:
        if not args:  # pragma: no cover
            raise DJInvalidInputException(
                message="Wrong number of arguments to function",
                errors=[
                    DJError(
                        code=ErrorCode.INVALID_ARGUMENTS_TO_FUNCTION,
                        message="You need to pass at least one argument to `COALESCE`.",
                    ),
                ],
            )
        for type_ in args:
            if type_ != ct.NullType():
                return type_
        return ct.NullType()


class CurrentDate(Function):
    """
    Returns the current date.
    """
    __name__ = "CURRENT_DATE"

    @staticmethod
    def infer_type() -> ct.DateType:
        return ct.DateType()


class CurrentDatetime(Function):
    """
    Returns the current date and time.
    """

    @staticmethod
    def infer_type() -> ct.TimestampType:
        return ct.TimestampType()


class CurrentTime(Function):
    """
    Returns the current time.
    """

    @staticmethod
    def infer_type() -> ct.TimeType:
        return ct.TimeType()


class CurrentTimestamp(Function):
    """
    Returns the current timestamp.
    """

    @staticmethod
    def infer_type() -> ct.TimestampType:
        return ct.TimestampType()


class Now(Function):
    """
    Returns the current timestamp.
    """

    @staticmethod
    def infer_type() -> ct.TimestampType:
        return ct.TimestampType()


class DateAdd(Function):
    """
    Adds a specified number of days to a date.
    """
    @staticmethod
    def infer_type(*args) -> ct.DateType:
        return ct.DateType()


class DateSub(Function):
    """
    Subtracts a specified number of days from a date.
    """
    @staticmethod
    def infer_type(*args) -> ct.DateType:
        return ct.DateType()


class If(Function):
    """
    If statement

    if(condition, result, else_result): if condition evaluates to true,
    then returns result; otherwise returns else_result.
    """
    @staticmethod
    def infer_type(*args) -> ct.ColumnType:
        if len(args) != 3:
            raise DJInvalidInputException(
                message="Wrong number of arguments passed in, "
                        "expecting: IF(condition, result, else_result)"
            )

        if args[1] != args[2]:
            raise DJInvalidInputException(
                message="The result and else result must match in type!"
                        f"Got {args[1]} and {args[2]}"
            )
        return args[1]


class DateDiff(Function):
    """
    Computes the difference in days between two dates.
    """
    @staticmethod
    def infer_type(*arg) -> ct.IntegerType:
        return ct.IntegerType()


class DatetimeAdd(Function):
    """
    Adds a specified number of seconds to a timestamp.
    """
    @staticmethod
    def infer_type(*arg) -> ct.TimestampType:
        return ct.TimestampType()


class DatetimeSub(Function):
    """
    Subtracts a specified number of seconds from a timestamp.
    """
    @staticmethod
    def infer_type(*arg) -> ct.TimestampType:
        return ct.TimestampType()


class DatetimeDiff(Function):
    """
    Computes the difference in seconds between two timestamps.
    """
    @staticmethod
    def infer_type(*arg) -> ct.IntegerType:
        return ct.IntegerType()


class Extract(Function):
    """
    Returns a specified component of a timestamp, such as year, month or day.
    """
    @staticmethod
    def infer_type(arg) -> ct.IntegerType:
        return ct.IntegerType()


class TimestampAdd(Function):
    """
    Adds a specified amount of time to a timestamp.
    """
    @staticmethod
    def infer_type(*arg) -> ct.TimestampType:
        return ct.TimestampType()


class TimestampSub(Function):
    """
    Subtracts a specified amount of time from a timestamp.
    """
    @staticmethod
    def infer_type(*arg) -> ct.TimestampType:
        return ct.TimestampType()


class TimestampDiff(Function):
    """
    Computes the difference in seconds between two timestamps.
    """
    @staticmethod
    def infer_type(*arg) -> ct.IntegerType:
        return ct.IntegerType()


class TimeAdd(Function):
    """
    Adds a specified amount of time to a time value.
    """
    @staticmethod
    def infer_type(*arg) -> ct.TimestampType:
        return ct.TimestampType()


class TimeSub(Function):
    """
    Subtracts a specified amount of time from a time value.
    """
    @staticmethod
    def infer_type(*arg) -> ct.TimestampType:
        return ct.TimestampType()


class TimeDiff(Function):
    """
    Computes the difference in seconds between two time values.
    """
    @staticmethod
    def infer_type(*arg) -> ct.IntegerType:
        return ct.IntegerType()


class DateStrToDate(Function):
    """
    Converts a date string to a date value.
    """
    @staticmethod
    def infer_type(arg) -> ct.DateType:
        return ct.DateType()


class DateToDateStr(Function):
    """
    Converts a date value to a date string.
    """
    @staticmethod
    def infer_type(arg) -> ct.StringType:
        return ct.StringType()


class DateToDi(Function):
    """
    Returns the day of the year for a specified date.
    """
    @staticmethod
    def infer_type(arg) -> ct.IntegerType:
        return ct.IntegerType()


class Day(Function):
    """
    Returns the day of the month for a specified date.
    """
    @staticmethod
    def infer_type(arg) -> ct.TinyIntType:
        return ct.TinyIntType()


class DiToDate(Function):
    """
    Converts a day of the year and a year to a date value.
    """
    @staticmethod
    def infer_type(arg) -> ct.DateType:
        return ct.DateType()


class Exp(Function):
    """
    Computes the exponential value of a specified number.
    """
    @staticmethod
    def infer_type(arg) -> ct.DoubleType:
        return ct.DoubleType()


class Floor(Function):
    """
    Returns the largest integer less than or equal to a specified number.
    """
    @staticmethod
    def infer_type(arg) -> ct.IntegerType:
        return ct.IntegerType()


class IfNull(Function):
    """
    Returns the second expression if the first expression is null, else returns the first expression.
    """
    @staticmethod
    def infer_type(*args) -> ct.ColumnType:
        return args[0]


class Length(Function):
    """
    Returns the length of a string.
    """
    @staticmethod
    def infer_type(arg) -> ct.LongType:
        return ct.LongType()


class Levenshtein(Function):
    """
    Returns the Levenshtein distance between two strings.
    """
    @staticmethod
    def infer_type(*args) -> ct.IntegerType:
        return ct.IntegerType()


class Ln(Function):
    """
    Returns the natural logarithm of a number.
    """
    @staticmethod
    def infer_type(arg) -> ct.DoubleType:
        return ct.DoubleType()


class Log(Function):
    """
    Returns the logarithm of a number with the specified base.
    """
    @staticmethod
    def infer_type(*args) -> ct.DoubleType:
        return ct.DoubleType()


class Log2(Function):
    """
    Returns the base-2 logarithm of a number.
    """
    @staticmethod
    def infer_type(arg) -> ct.DoubleType:
        return ct.DoubleType()


class Log10(Function):
    """
    Returns the base-10 logarithm of a number.
    """
    @staticmethod
    def infer_type(arg) -> ct.DoubleType:
        return ct.DoubleType()


class Lower(Function):
    """
    Converts a string to lowercase.
    """
    @staticmethod
    def infer_type(arg) -> ct.StringType:
        return ct.StringType()

class Month(Function):
    """
    Extracts the month of a date or timestamp.
    """
    @staticmethod
    def infer_type(arg) -> ct.TinyIntType:
        return ct.TinyIntType()

class Pow(Function):
    """
    Raises a base expression to the power of an exponent expression.
    """
    @staticmethod
    def infer_type(arg1, arg2) -> ct.DoubleType:
        return ct.DoubleType()

class Quantile(Function):
    """
    Computes the quantile of a numerical column or expression.
    """
    is_aggregation = True

    @staticmethod
    def infer_type(arg1, arg2) -> ct.DoubleType:
        return ct.DoubleType()

class ApproxQuantile(Function):
    """
    Computes the approximate quantile of a numerical column or expression.
    """
    is_aggregation = True

    @staticmethod
    def infer_type(arg1, arg2) -> ct.DoubleType:
        return ct.DoubleType()

class RegexpLike(Function):
    """
    Matches a string column or expression against a regular expression pattern.
    """
    @staticmethod
    def infer_type(arg1, arg2) -> ct.BooleanType:
        return ct.BooleanType()

class Round(Function):
    """
    Rounds a numeric column or expression to the specified number of decimal places.
    """
    @staticmethod
    def infer_type(arg1, arg2) -> ct.DoubleType:
        return ct.DoubleType()

class SafeDivide(Function):
    """
    Divides two numeric columns or expressions and returns NULL if the denominator is 0.
    """
    @staticmethod
    def infer_type(arg1, arg2) -> ct.DoubleType:
        return ct.DoubleType()

class Substring(Function):
    """
    Extracts a substring from a string column or expression.
    """
    @staticmethod
    def infer_type(arg1, arg2, arg3) -> ct.StringType:
        return ct.StringType()

class StrPosition(Function):
    """
    Returns the position of the first occurrence of a substring in a string column or expression.
    """
    @staticmethod
    def infer_type(arg1, arg2) -> ct.IntegerType:
        return ct.IntegerType()

class StrToDate(Function):
    """
    Converts a string in a specified format to a date.
    """
    @staticmethod
    def infer_type(arg1, arg2) -> ct.DateType:
        return ct.DateType()


class StrToTime(Function):
    """
    Converts a string in a specified format to a timestamp.
    """
    @staticmethod
    def infer_type(arg1, arg2) -> ct.TimestampType:
        return ct.TimestampType()


class Sqrt(Function):
    """
    Computes the square root of a numeric column or expression.
    """
    @staticmethod
    def infer_type(arg) -> ct.DoubleType:
        return ct.DoubleType()


class Stddev(Function):
    """
    Computes the sample standard deviation of a numerical column or expression.
    """
    is_aggregation = True

    @staticmethod
    def infer_type(arg) -> ct.DoubleType:
        return ct.DoubleType()


class StddevPop(Function):
    """
    Computes the population standard deviation of the input column or expression.
    """
    is_aggregation = True

    @staticmethod
    def infer_type(arg) -> ct.DoubleType:
        return ct.DoubleType()


class StddevSamp(Function):
    """
    Computes the sample standard deviation of the input column or expression.
    """
    is_aggregation = True

    @staticmethod
    def infer_type(arg) -> ct.DoubleType:
        return ct.DoubleType()


class TimeToStr(Function):
    """
    Converts a time value to a string using the specified format.
    """
    @staticmethod
    def infer_type(arg1, arg2) -> ct.StringType:
        return ct.StringType()


class TimeToTimeStr(Function):
    """
    Converts a time value to a string using the specified format.
    """
    @staticmethod
    def infer_type(arg1, arg2) -> ct.StringType:
        return ct.StringType()


class TimeStrToDate(Function):
    """
    Converts a string value to a date.
    """
    @staticmethod
    def infer_type(arg) -> ct.DateType:
        return ct.DateType()


class TimeStrToTime(Function):
    """
    Converts a string value to a time.
    """
    @staticmethod
    def infer_type(arg) -> ct.TimestampType:
        return ct.TimestampType()


class Trim(Function):
    """
    Removes leading and trailing whitespace from a string value.
    """
    @staticmethod
    def infer_type(arg) -> ct.StringType:
        return ct.StringType()


class TsOrDsToDateStr(Function):
    """
    Converts a timestamp or date value to a string using the specified format.
    """
    @staticmethod
    def infer_type(arg1, arg2) -> ct.StringType:
        return ct.StringType()


class TsOrDsToDate(Function):
    """
    Converts a timestamp or date value to a date.
    """
    @staticmethod
    def infer_type(arg) -> ct.DateType:
        return ct.DateType()


class TsOrDiToDi(Function):
    """
    Converts a timestamp or date value to a date.
    """
    @staticmethod
    def infer_type(arg) -> ct.IntegerType:
        return ct.IntegerType()


class UnixToStr(Function):
    """
    Converts a Unix timestamp to a string using the specified format.
    """
    @staticmethod
    def infer_type(arg1, arg2) -> ct.StringType:
        return ct.StringType()


class UnixToTime(Function):
    """
    Converts a Unix timestamp to a time.
    """
    @staticmethod
    def infer_type(arg) -> ct.TimestampType:
        return ct.TimestampType()


class UnixToTimeStr(Function):
    """
    Converts a Unix timestamp to a string using the specified format.
    """
    @staticmethod
    def infer_type(arg1, arg2) -> ct.StringType:
        return ct.StringType()


class Upper(Function):
    """
    Converts a string value to uppercase.
    """
    @staticmethod
    def infer_type(arg) -> ct.StringType:
        return ct.StringType()


class Variance(Function):
    """
    Computes the sample variance of the input column or expression.
    """
    is_aggregation = True

    @staticmethod
    def infer_type(arg) -> ct.DoubleType:
        return ct.DoubleType()


class VariancePop(Function):
    """
    Computes the population variance of the input column or expression.
    """
    is_aggregation = True

    @staticmethod
    def infer_type(arg) -> ct.DoubleType:
        return ct.DoubleType()


class Week(Function):
    """
    Returns the week number of the year of the input date value.
    """
    @staticmethod
    def infer_type(arg) -> ct.TinyIntType:
        return ct.TinyIntType()


class Year(Function):
    """
    Returns the year of the input date value.
    """
    @staticmethod
    def infer_type(arg) -> ct.TinyIntType:
        return ct.TinyIntType()


function_registry = {}
for cls in Function.__subclasses__():
    name = cls.__name__
    snake_cased = re.sub(r'(?<!^)(?=[A-Z])', '_', name)
    function_registry[name.upper()] = cls
    function_registry[snake_cased.upper()] = cls


class Explode(TableFunction):
    """
    Explode function is used to explode the specified array, 
    nested array, or map column into multiple rows. 
    The explode function will generate a new row for each 
    element in the specified column. 
    """
    @staticmethod
    def infer_type(arg) -> Union[ct.ColumnType, List[ct.ColumnType]]:
        if isinstance(arg, ct.ListType):
            return arg.field_type
        if isinstance(arg, ct.MapType):
            return [arg.key, arg.value]
        raise Exception(f"Invalid type for Explode {arg}.")


table_function_registry = {}
for cls in TableFunction.__subclasses__():
    name = cls.__name__
    snake_cased = re.sub(r'(?<!^)(?=[A-Z])', '_', name)
    table_function_registry[name.upper()] = cls
    table_function_registry[snake_cased.upper()] = cls