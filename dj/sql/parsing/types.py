"""DJ Column Types

Example:
    >>> StructType(
                NestedField(
                    name=Name(name='name', quote_style='', namespace=None), 
                    field_type=StringType(), 
                    is_optional=False, 
                    doc="COMMENT'where someone lives'"
                )
            )
"""

from typing import Dict, Optional, Tuple, ClassVar
import dj.sql.parsing.ast2 as ast

class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance


class ColumnType:
    """Base type for all Column Types"""

    _initialized = False

    def __init__(self, type_string: str, repr_string: str):
        self._type_string = type_string
        self._repr_string = repr_string
        self._initialized = True

    def __repr__(self):
        return self._repr_string

    def __str__(self):
        return self._type_string

    @property
    def is_primitive(self) -> bool:
        return isinstance(self, PrimitiveType)


class PrimitiveType(ColumnType):
    """Base class for all Column Primitive Types"""


class FixedType(PrimitiveType):
    """A fixed data type.

    Example:
        >>> FixedType(8)
        FixedType(length=8)
        >>> FixedType(8)==FixedType(8)
        True
    """

    _instances: Dict[int, "FixedType"] = {}

    def __new__(cls, length: int):
        cls._instances[length] = cls._instances.get(length) or object.__new__(cls)
        return cls._instances[length]

    def __init__(self, length: int):
        if not self._initialized:
            super().__init__(f"fixed({length})", f"FixedType(length={length})")
            self._length = length

    @property
    def length(self) -> int:
        return self._length


class DecimalType(PrimitiveType):
    """A fixed data type.

    Example:
        >>> DecimalType(32, 3)
        DecimalType(precision=32, scale=3)
        >>> DecimalType(8, 3)==DecimalType(8, 3)
        True
    """

    _instances: Dict[Tuple[int, int], "DecimalType"] = {}

    def __new__(cls, precision: int, scale: int):
        key = (precision, scale)
        cls._instances[key] = cls._instances.get(key) or object.__new__(cls)
        return cls._instances[key]

    def __init__(self, precision: int, scale: int):
        if not self._initialized:
            super().__init__(
                f"decimal({precision}, {scale})",
                f"DecimalType(precision={precision}, scale={scale})",
            )
            self._precision = precision
            self._scale = scale

    @property
    def precision(self) -> int:
        return self._precision

    @property
    def scale(self) -> int:
        return self._scale


class NestedField(ColumnType):
    """Represents a field of a struct, a map key, a map value, or a list element.

    This is where field IDs, names, docs, and nullability are tracked.
    """

    _instances: Dict[Tuple[bool, int, str, ColumnType, Optional[str]], "NestedField"] = {}

    def __new__(
        cls,
        name: ast.Name,
        field_type: ColumnType,
        is_optional: bool = True,
        doc: Optional[str] = None,
    ):
        key = (is_optional, name, field_type, doc)
        cls._instances[key] = cls._instances.get(key) or object.__new__(cls)
        return cls._instances[key]

    def __init__(
        self,
        name: str,
        field_type: ColumnType,
        is_optional: bool = True,
        doc: Optional[str] = None,
    ):
        if not self._initialized:
            docString = "" if doc is None else f", doc={repr(doc)}"
            super().__init__(
                (
                    f"{name}: {field_type}{' NOT NULL' if not is_optional else ''}" \
                    + (""
                    if doc is None
                    else f" {doc}")
                ),
                f"NestedField(name={repr(name)}, field_type={repr(field_type)}, is_optional={is_optional}"
                f"{docString})",
            )
            self._is_optional = is_optional
            self._name = name
            self._type = field_type
            self._doc = doc

    @property
    def is_optional(self) -> bool:
        return self._is_optional

    @property
    def is_required(self) -> bool:
        return not self._is_optional

    @property
    def name(self) -> str:
        return self._name

    @property
    def doc(self) -> Optional[str]:
        return self._doc

    @property
    def type(self) -> ColumnType:
        return self._type


class StructType(ColumnType):
    """A struct type

    Example:
        >>> StructType(
                NestedField(True, 1, "required_field", StringType()),
                NestedField(False, 2, "optional_field", IntegerType())
        )
    """

    _instances: Dict[Tuple[NestedField, ...], "StructType"] = {}

    def __new__(cls, *fields: NestedField):
        cls._instances[fields] = cls._instances.get(fields) or object.__new__(cls)
        return cls._instances[fields]

    def __init__(self, *fields: NestedField):
        if not self._initialized:
            super().__init__(
                f"struct<{', '.join(map(str, fields))}>",
                f"StructType{repr(fields)}",
            )
            self._fields = fields

    @property
    def fields(self) -> Tuple[NestedField, ...]:
        return self._fields


class ListType(ColumnType):
    """A list type

    Example:
        >>> ListType(element_type=StringType(), element_is_optional=True)
        ListType(element=NestedField(is_optional=True, field_id=3, name='element', field_type=StringType(), doc=None))
    """

    _instances: Dict[Tuple[bool, int, ColumnType], "ListType"] = {}

    def __new__(
        cls,
        element_type: ColumnType,
    ):
        key = (element_type,)
        cls._instances[key] = cls._instances.get(key) or object.__new__(cls)
        return cls._instances[key]

    def __init__(
        self,
        element_type: ColumnType,
    ):
        if not self._initialized:
            super().__init__(
                f"array<{element_type}>",
                f"ListType(element_type={repr(element_type)})",
            )

class MapType(ColumnType):
    """A map type

    Example:
        >>> MapType(key_id=1, key_type=StringType(), value_id=2, value_type=IntegerType(), value_is_optional=True)
        MapType(
            key=NestedField(is_optional=False, field_id=1, name='key', field_type=StringType(), doc=None),
            value=NestedField(is_optional=True, field_id=2, name='value', field_type=IntegerType(), doc=None)
        )
    """

    _instances: Dict[Tuple[int, ColumnType, int, ColumnType, bool], "MapType"] = {}

    def __new__(
        cls,
        key_type: ColumnType,
        value_type: ColumnType,
    ):
        impl_key = (key_type, value_type)
        cls._instances[impl_key] = cls._instances.get(impl_key) or object.__new__(cls)
        return cls._instances[impl_key]

    def __init__(
        self,
        key_type: ColumnType,
        value_type: ColumnType,
    ):
        if not self._initialized:
            super().__init__(
                f"map<{key_type}, {value_type}>",
                f"MapType(key_type={repr(key_type)}, value_type={repr(value_type)})",
            )

    @property
    def key(self) -> NestedField:
        return self._key_field

    @property
    def value(self) -> NestedField:
        return self._value_field


class BooleanType(PrimitiveType, Singleton):
    """A boolean data type can be represented using an instance of this class.

    Example:
        >>> column_foo = BooleanType()
        >>> isinstance(column_foo, BooleanType)
        True
    """

    def __init__(self):
        if not self._initialized:
            super().__init__("boolean", "BooleanType()")


class IntegerType(PrimitiveType, Singleton):
    """An Integer data type can be represented using an instance of this class. Integers are
    32-bit signed and can be promoted to Longs.

    Example:
        >>> column_foo = IntegerType()
        >>> isinstance(column_foo, IntegerType)
        True

    Attributes:
        max (int): The maximum allowed value for Integers, inherited from the canonical Column implementation
          in Java (returns `2147483647`)
        min (int): The minimum allowed value for Integers, inherited from the canonical Column implementation
          in Java (returns `-2147483648`)
    """

    max: ClassVar[int] = 2147483647

    min: ClassVar[int] = -2147483648

    def __init__(self):
        if not self._initialized:
            super().__init__("int", "IntegerType()")


class LongType(PrimitiveType, Singleton):
    """A Long data type can be represented using an instance of this class. Longs are
    64-bit signed integers.

    Example:
        >>> column_foo = LongType()
        >>> isinstance(column_foo, LongType)
        True

    Attributes:
        max (int): The maximum allowed value for Longs, inherited from the canonical Column implementation
          in Java. (returns `9223372036854775807`)
        min (int): The minimum allowed value for Longs, inherited from the canonical Column implementation
          in Java (returns `-9223372036854775808`)
    """

    max: ClassVar[int] = 9223372036854775807

    min: ClassVar[int] = -9223372036854775808

    def __init__(self):
        if not self._initialized:
            super().__init__("long", "LongType()")

class FloatType(PrimitiveType, Singleton):
    """A Float data type can be represented using an instance of this class. Floats are
    32-bit IEEE 754 floating points and can be promoted to Doubles.

    Example:
        >>> column_foo = FloatType()
        >>> isinstance(column_foo, FloatType)
        True
    """

    def __init__(self):
        if not self._initialized:
            super().__init__("float", "FloatType()")


class DoubleType(PrimitiveType, Singleton):
    """A Double data type can be represented using an instance of this class. Doubles are
    64-bit IEEE 754 floating points.

    Example:
        >>> column_foo = DoubleType()
        >>> isinstance(column_foo, DoubleType)
        True
    """

    def __init__(self):
        if not self._initialized:
            super().__init__("double", "DoubleType()")


class DateType(PrimitiveType, Singleton):
    """A Date data type can be represented using an instance of this class. Dates are
    calendar dates without a timezone or time.

    Example:
        >>> column_foo = DateType()
        >>> isinstance(column_foo, DateType)
        True
    """

    def __init__(self):
        if not self._initialized:
            super().__init__("date", "DateType()")


class TimeType(PrimitiveType, Singleton):
    """A Time data type can be represented using an instance of this class. Times
    have microsecond precision and are a time of day without a date or timezone.

    Example:
        >>> column_foo = TimeType()
        >>> isinstance(column_foo, TimeType)
        True
    """

    def __init__(self):
        if not self._initialized:
            super().__init__("time", "TimeType()")


class TimestampType(PrimitiveType, Singleton):
    """A Timestamp data type can be represented using an instance of this class. Timestamps in
    Column have microsecond precision and include a date and a time of day without a timezone.

    Example:
        >>> column_foo = TimestampType()
        >>> isinstance(column_foo, TimestampType)
        True
    """

    def __init__(self):
        if not self._initialized:
            super().__init__("timestamp", "TimestampType()")


class TimestamptzType(PrimitiveType, Singleton):
    """A Timestamptz data type can be represented using an instance of this class. Timestamptzs in
    Column are stored as UTC and include a date and a time of day with a timezone.

    Example:
        >>> column_foo = TimestamptzType()
        >>> isinstance(column_foo, TimestamptzType)
        True
    """

    def __init__(self):
        if not self._initialized:
            super().__init__("timestamptz", "TimestamptzType()")


class StringType(PrimitiveType, Singleton):
    """A String data type can be represented using an instance of this class. Strings in
    Column are arbitrary-length character sequences and are encoded with UTF-8.

    Example:
        >>> column_foo = StringType()
        >>> isinstance(column_foo, StringType)
        True
    """

    def __init__(self):
        if not self._initialized:
            super().__init__("string", "StringType()")


class UUIDType(PrimitiveType, Singleton):
    """A UUID data type can be represented using an instance of this class. UUIDs in
    Column are universally unique identifiers.

    Example:
        >>> column_foo = UUIDType()
        >>> isinstance(column_foo, UUIDType)
        True
    """

    def __init__(self):
        if not self._initialized:
            super().__init__("uuid", "UUIDType()")


class BinaryType(PrimitiveType, Singleton):
    """A Binary data type can be represented using an instance of this class. Binarys in
    Column are arbitrary-length byte arrays.

    Example:
        >>> column_foo = BinaryType()
        >>> isinstance(column_foo, BinaryType)
        True
    """

    def __init__(self):
        if not self._initialized:
            super().__init__("binary", "BinaryType()")
