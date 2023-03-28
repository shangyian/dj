"""Data Types used in DJ Queries

Example:
    >>> str(StructType(
    ...     NestedField(1, "required_field", StringType(), True),
    ...     NestedField(2, "optional_field", IntegerType())
    ... ))
    'struct<1: required_field: optional string, 2: optional_field: optional int>'

"""
from __future__ import annotations

import re
from typing import (
    Any,
    ClassVar,
    Dict,
    Generator,
    Literal,
    Optional,
    Tuple,
)

from pydantic import Field, PrivateAttr
from pydantic.typing import AnyCallable

DECIMAL_REGEX = re.compile(r"decimal\((\d+),\s*(\d+)\)")
FIXED_PARSER = re.compile(rf"fixed\[(\d+)\]")


class DataType(Singleton):
    """Base type for all Data Types

    Example:
        >>> str(DataType())
        'DataType()'
        >>> repr(DataType())
        'DataType()'
    """
    _instances: ClassVar[Dict] = {}  # type: ignore

    def __new__(cls, *args, **kwargs):  # type: ignore
        key = (cls, tuple(args), _convert_to_hashable_type(kwargs))
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    @classmethod
    def __get_validators__(cls) -> Generator[AnyCallable, None, None]:
        # one or more validators may be yielded which will be called in the
        # order to validate the input, each validator will receive as an input
        # the value returned from the previous validator
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> DataType:
        # When Pydantic is unable to determine the subtype
        # In this case we'll help pydantic a bit by parsing the
        # primitive type ourselves, or pointing it at the correct
        # complex type by looking at the type field

        if isinstance(v, str):
            if v.startswith("decimal"):
                return DecimalType.parse(v)
            elif v.startswith("fixed"):
                return FixedType.parse(v)
            else:
                return PRIMITIVE_TYPES[v]
        elif isinstance(v, dict):
            if v.get("type") == "struct":
                return StructType(**v)
            elif v.get("type") == "list":
                return ListType(**v)
            elif v.get("type") == "map":
                return MapType(**v)
            else:
                return NestedField(**v)
        else:
            return v

    @property
    def is_primitive(self) -> bool:
        return isinstance(self, PrimitiveType)

    @property
    def is_struct(self) -> bool:
        return isinstance(self, StructType)


class PrimitiveType(DataType):
    """Base class for all Data Primitive Types"""

    __root__: str = Field()

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __str__(self) -> str:
        return self.__root__


class FixedType(PrimitiveType):
    """A fixed data type in Data.
    Example:
        >>> FixedType(8)
        FixedType(length=8)
        >>> FixedType(8) == FixedType(8)
        True
        >>> FixedType(19) == FixedType(25)
        False
    """

    __root__: str = Field()
    _len: int = PrivateAttr()

    @staticmethod
    def parse(str_repr: str) -> FixedType:
        return FixedType(length=FIXED_PARSER.match(str_repr))

    def __init__(self, length: int):
        super().__init__(__root__=f"fixed[{length}]")
        self._len = length

    def __len__(self) -> int:
        return self._len

    def __repr__(self) -> str:
        return f"FixedType(length={self._len})"


class DecimalType(PrimitiveType):
    """A fixed data type in Data.
    Example:
        >>> DecimalType(32, 3)
        DecimalType(precision=32, scale=3)
        >>> DecimalType(8, 3) == DecimalType(8, 3)
        True
    """

    __root__: str = Field()

    _precision: int = PrivateAttr()
    _scale: int = PrivateAttr()

    @staticmethod
    def parse(str_repr: str) -> DecimalType:
        matches = DECIMAL_REGEX.search(str_repr)
        if matches:
            precision = int(matches.group(1))
            scale = int(matches.group(2))
            return DecimalType(precision, scale)
        else:
            raise ValueError(f"Could not parse {str_repr} into a DecimalType")

    def __init__(self, precision: int, scale: int):
        super().__init__(
            __root__=f"decimal({precision}, {scale})",
        )
        # assert precision < scale, "precision should be smaller than scale"
        self._precision = precision
        self._scale = scale

    @property
    def precision(self) -> int:
        return self._precision

    @property
    def scale(self) -> int:
        return self._scale

    def __repr__(self) -> str:
        return f"DecimalType(precision={self._precision}, scale={self._scale})"


class NestedField(DataType):
    """Represents a field of a struct, a map key, a map value, or a list element.

    This is where field IDs, names, docs, and nullability are tracked.

    Example:
        >>> str(NestedField(
        ...     field_id=1,
        ...     name='foo',
        ...     field_type=FixedType(22),
        ...     required=False,
        ... ))
        '1: foo: optional fixed[22]'
        >>> str(NestedField(
        ...     field_id=2,
        ...     name='bar',
        ...     field_type=LongType(),
        ...     is_optional=False,
        ...     doc="Just a long"
        ... ))
        '2: bar: required long (Just a long)'
    """

    field_id: int = Field(alias="id")
    name: str = Field()
    field_type: DataType = Field(alias="type")
    required: bool = Field(default=True)
    doc: Optional[str] = Field(default=None, repr=False)

    def __init__(
        self,
        field_id: Optional[int] = None,
        name: Optional[str] = None,
        field_type: Optional[DataType] = None,
        required: bool = True,
        doc: Optional[str] = None,
        **data: Any,
    ):
        # We need an init when we want to use positional arguments, but
        # need also to support the aliases.
        data["field_id"] = data["id"] if "id" else field_id
        data["name"] = name
        data["field_type"] = data["type"] if "type" else field_type
        data["required"] = required
        data["doc"] = doc
        super().__init__(**data)

    def __str__(self) -> str:
        doc = "" if not self.doc else f" ({self.doc})"
        req = "required" if self.required else "optional"
        return f"{self.field_id}: {self.name}: {req} {self.field_type}{doc}"

    @property
    def optional(self) -> bool:
        return not self.required


class StructType(DataType):
    """A struct type in Data

    Example:
        >>> str(StructType(
        ...     NestedField(1, "required_field", StringType(), True),
        ...     NestedField(2, "optional_field", IntegerType())
        ... ))
        'struct<1: required_field: optional string, 2: optional_field: optional int>'
    """

    type: Literal["struct"] = "struct"
    fields: Tuple[NestedField, ...] = Field(default_factory=tuple)

    def __init__(self, *fields: NestedField, **data: Any):
        # In case we use positional arguments, instead of keyword args
        if fields:
            data["fields"] = fields
        super().__init__(**data)

    def field(self, field_id: int) -> Optional[NestedField]:
        for field in self.fields:
            if field.field_id == field_id:
                return field
        return None

    def __str__(self) -> str:
        return f"struct<{', '.join(map(str, self.fields))}>"

    def __repr__(self) -> str:
        return f"StructType(fields=({', '.join(map(repr, self.fields))},))"

    def __len__(self) -> int:
        return len(self.fields)


class ListType(DataType):
    """A list type in Data

    Example:
        >>> ListType(element_id=3, element_type=StringType(), element_required=True)
        ListType(element_id=3, element_type=StringType(), element_required=True)
    """

    class Config:
        fields = {"element_field": {"exclude": True}}

    type: Literal["list"] = "list"
    element_id: int = Field(alias="element-id")
    element_type: DataType = Field(alias="element")
    element_required: bool = Field(alias="element-required", default=True)
    element_field: NestedField = Field(init=False, repr=False)

    def __init__(
        self, element_id: Optional[int] = None, element: Optional[DataType] = None, element_required: bool = True, **data: Any
    ):
        data["element_id"] = data["element-id"] if "element-id" else element_id
        data["element_type"] = element or data["element_type"]
        data["element_required"] = data["element-required"] if "element-required" else element_required
        data["element_field"] = NestedField(
            name="element",
            required=data["element_required"],
            field_id=data["element_id"],
            field_type=data["element_type"],
        )
        super().__init__(**data)

    def __str__(self) -> str:
        return f"list<{self.element_type}>"


class MapType(DataType):
    """A map type in Data

    Example:
        >>> MapType(key_id=1, key_type=StringType(), value_id=2, value_type=IntegerType(), value_required=True)
        MapType(key_id=1, key_type=StringType(), value_id=2, value_type=IntegerType(), value_required=True)
    """

    type: Literal["map"] = "map"
    key_id: int = Field(alias="key-id")
    key_type: DataType = Field(alias="key")
    value_id: int = Field(alias="value-id")
    value_type: DataType = Field(alias="value")
    value_required: bool = Field(alias="value-required", default=True)
    key_field: NestedField = Field(init=False, repr=False)
    value_field: NestedField = Field(init=False, repr=False)

    class Config:
        fields = {"key_field": {"exclude": True}, "value_field": {"exclude": True}}

    def __init__(
        self,
        key_id: Optional[int] = None,
        key_type: Optional[DataType] = None,
        value_id: Optional[int] = None,
        value_type: Optional[DataType] = None,
        value_required: bool = True,
        **data: Any,
    ):
        data["key_id"] = key_id or data["key-id"]
        data["key_type"] = key_type or data["key"]
        data["value_id"] = value_id or data["value-id"]
        data["value_type"] = value_type or data["value"]
        data["value_required"] = value_required if value_required is not None else data["value_required"]

        data["key_field"] = NestedField(name="key", field_id=data["key_id"], field_type=data["key_type"], required=True)
        data["value_field"] = NestedField(
            name="value", field_id=data["value_id"], field_type=data["value_type"], required=data["value_required"]
        )
        super().__init__(**data)

    def __str__(self) -> str:
        return f"map<{self.key_type}, {self.value_type}>"


class BooleanType(PrimitiveType):
    """A boolean data type can be represented using an instance of this class.

    Example:
        >>> column_foo = BooleanType()
        >>> isinstance(column_foo, BooleanType)
        True
        >>> column_foo
        BooleanType()
    """

    __root__ = "boolean"


class IntegerType(PrimitiveType):
    """An Integer data type can be represented using an instance of this class. Integers are
    32-bit signed and can be promoted to Longs.

    Example:
        >>> column_foo = IntegerType()
        >>> isinstance(column_foo, IntegerType)
        True

    Attributes:
        max (int): The maximum allowed value for Integers, inherited from the canonical Data implementation
          in Java (returns `2147483647`)
        min (int): The minimum allowed value for Integers, inherited from the canonical Data implementation
          in Java (returns `-2147483648`)
    """

    max: ClassVar[int] = 2147483647
    min: ClassVar[int] = -2147483648

    __root__ = "int"


class LongType(PrimitiveType):
    """A Long data type can be represented using an instance of this class. Longs are
    64-bit signed integers.

    Example:
        >>> column_foo = LongType()
        >>> isinstance(column_foo, LongType)
        True
        >>> column_foo
        LongType()
        >>> str(column_foo)
        'long'

    Attributes:
        max (int): The maximum allowed value for Longs, inherited from the canonical Data implementation
          in Java. (returns `9223372036854775807`)
        min (int): The minimum allowed value for Longs, inherited from the canonical Data implementation
          in Java (returns `-9223372036854775808`)
    """

    max: ClassVar[int] = 9223372036854775807
    min: ClassVar[int] = -9223372036854775808

    __root__ = "long"


class FloatType(PrimitiveType):
    """A Float data type can be represented using an instance of this class. Floats are
    32-bit IEEE 754 floating points and can be promoted to Doubles.

    Example:
        >>> column_foo = FloatType()
        >>> isinstance(column_foo, FloatType)
        True
        >>> column_foo
        FloatType()

    Attributes:
        max (float): The maximum allowed value for Floats, inherited from the canonical Data implementation
          in Java. (returns `3.4028235e38`)
        min (float): The minimum allowed value for Floats, inherited from the canonical Data implementation
          in Java (returns `-3.4028235e38`)
    """

    max: ClassVar[float] = 3.4028235e38
    min: ClassVar[float] = -3.4028235e38

    __root__ = "float"


class DoubleType(PrimitiveType):
    """A Double data type can be represented using an instance of this class. Doubles are
    64-bit IEEE 754 floating points.

    Example:
        >>> column_foo = DoubleType()
        >>> isinstance(column_foo, DoubleType)
        True
        >>> column_foo
        DoubleType()
    """

    __root__ = "double"


class DateType(PrimitiveType):
    """A Date data type can be represented using an instance of this class. Dates are
    calendar dates without a timezone or time.

    Example:
        >>> column_foo = DateType()
        >>> isinstance(column_foo, DateType)
        True
        >>> column_foo
        DateType()
    """

    __root__ = "date"


class TimeType(PrimitiveType):
    """A Time data type can be represented using an instance of this class. Times in Data
    have microsecond precision and are a time of day without a date or timezone.

    Example:
        >>> column_foo = TimeType()
        >>> isinstance(column_foo, TimeType)
        True
        >>> column_foo
        TimeType()
    """

    __root__ = "time"


class TimestampType(PrimitiveType):
    """A Timestamp data type can be represented using an instance of this class. Timestamps in
    Data have microsecond precision and include a date and a time of day without a timezone.

    Example:
        >>> column_foo = TimestampType()
        >>> isinstance(column_foo, TimestampType)
        True
        >>> column_foo
        TimestampType()
    """

    __root__ = "timestamp"


class TimestamptzType(PrimitiveType):
    """A Timestamptz data type can be represented using an instance of this class. Timestamptzs in
    Data are stored as UTC and include a date and a time of day with a timezone.

    Example:
        >>> column_foo = TimestamptzType()
        >>> isinstance(column_foo, TimestamptzType)
        True
        >>> column_foo
        TimestamptzType()
    """

    __root__ = "timestamptz"


class StringType(PrimitiveType):
    """A String data type can be represented using an instance of this class. Strings in
    Data are arbitrary-length character sequences and are encoded with UTF-8.

    Example:
        >>> column_foo = StringType()
        >>> isinstance(column_foo, StringType)
        True
        >>> column_foo
        StringType()
    """

    __root__ = "string"


class UUIDType(PrimitiveType):
    """A UUID data type can be represented using an instance of this class. UUIDs in
    Data are universally unique identifiers.

    Example:
        >>> column_foo = UUIDType()
        >>> isinstance(column_foo, UUIDType)
        True
        >>> column_foo
        UUIDType()
    """

    __root__ = "uuid"


class BinaryType(PrimitiveType):
    """A Binary data type can be represented using an instance of this class. Binaries in
    Data are arbitrary-length byte arrays.

    Example:
        >>> column_foo = BinaryType()
        >>> isinstance(column_foo, BinaryType)
        True
        >>> column_foo
        BinaryType()
    """

    __root__ = "binary"


PRIMITIVE_TYPES: Dict[str, PrimitiveType] = {
    "boolean": BooleanType(),
    "int": IntegerType(),
    "long": LongType(),
    "float": FloatType(),
    "double": DoubleType(),
    "date": DateType(),
    "time": TimeType(),
    "timestamp": TimestampType(),
    "timestamptz": TimestamptzType(),
    "string": StringType(),
    "uuid": UUIDType(),
    "binary": BinaryType(),
}