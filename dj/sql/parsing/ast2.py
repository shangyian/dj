# pylint: disable=R0401,C0302
# pylint: skip-file
# mypy: ignore-errors
import decimal
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field, fields
from enum import Enum
from itertools import chain, zip_longest
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from sqlmodel import Session

from dj.models.node import NodeRevision as DJNode
from dj.models.node import NodeType as DJNodeType
from dj.sql.functions import function_registry
from dj.sql.parsing.backends.exceptions import DJParseException
from dj.typing import ColumnType
from dj.construction.utils import get_dj_node
from dj.errors import DJError, ErrorCode, DJException, DJErrorException

PRIMITIVES = {int, float, str, bool, type(None)}


def flatten(maybe_iterables: Any) -> Iterator:
    """
    Flattens `maybe_iterables` by descending into items that are Iterable
    """

    if not isinstance(maybe_iterables, (list, tuple, set, Iterator)):
        return iter([maybe_iterables])
    return chain.from_iterable(
        (flatten(maybe_iterable) for maybe_iterable in maybe_iterables)
    )


@dataclass
class CompileContext:
    session: Session
    exception: DJException
    query: "Query"


# typevar used for node methods that return self
# so the typesystem can correlate the self type with the return type
TNode = TypeVar("TNode", bound="Node")  # pylint: disable=C0103


class Node(ABC):
    """Base class for all DJ AST nodes.

    DJ nodes are python dataclasses with the following patterns:
        - Attributes are either
            - PRIMITIVES (int, float, str, bool, None)
            - iterable from (list, tuple, set)
            - Enum
            - descendant of `Node`
        - Attributes starting with '_' are "obfuscated" and are not included in `children`

    """

    parent: Optional["Node"] = None
    parent_key: Optional[str] = None

    def __post_init__(self):
        self.add_self_as_parent()

    @property
    def depth(self) -> int:
        if self.parent is None:
            return 0
        return self.parent.depth + 1

    def clear_parent(self: TNode) -> TNode:
        """
        Remove parent from the node
        """
        self.parent = None
        return self

    def set_parent(self: TNode, parent: "Node", parent_key: str) -> TNode:
        """
        Add parent to the node
        """
        self.parent = parent
        self.parent_key = parent_key
        return self

    def add_self_as_parent(self: TNode) -> TNode:
        """
        Adds self as a parent to all children
        """
        for name, child in self.fields(
            flat=True,
            nodes_only=True,
            obfuscated=False,
            nones=False,
            named=True,
        ):
            child.set_parent(self, name)
        return self

    def __setattr__(self, key: str, value: Any):
        """
        Facilitates setting children using `.` syntax ensuring parent is attributed
        """
        if key == "parent":
            object.__setattr__(self, key, value)
            return

        object.__setattr__(self, key, value)
        for child in flatten(value):
            if isinstance(child, Node) and not key.startswith("_"):
                child.set_parent(self, key)

    def swap(self: TNode, other: "Node") -> TNode:
        """
        Swap the Node for another
        """
        if not (self.parent and self.parent_key):
            return self
        parent_attr = getattr(self.parent, self.parent_key)
        if parent_attr is self:
            setattr(self.parent, self.parent_key, other)
            return self.clear_parent()

        new = []
        for iterable_type in (list, tuple, set):
            if isinstance(parent_attr, iterable_type):
                for element in parent_attr:
                    if self is element:
                        new.append(other)
                    else:
                        new.append(element)
                new = iterable_type(new)
                break

        setattr(self.parent, self.parent_key, new)
        return self.clear_parent()

    def copy(self: TNode) -> TNode:
        """
        Create a deep copy of the `self`
        """
        return deepcopy(self)

    def get_nearest_parent_of_type(
        self: "Node",
        node_type: Type[TNode],
    ) -> Optional[TNode]:
        """
        Traverse up the tree until you find a node of `node_type` or hit the root
        """
        if isinstance(self.parent, node_type):
            return self.parent
        if self.parent is None:
            return None
        return self.parent.get_nearest_parent_of_type(node_type)

    def flatten(self) -> Iterator["Node"]:
        """
        Flatten the sub-ast of the node as an iterator
        """
        return self.filter(lambda _: True)

    # pylint: disable=R0913
    def fields(
        self,
        flat: bool = True,
        nodes_only: bool = True,
        obfuscated: bool = False,
        nones: bool = False,
        named: bool = False,
    ) -> Iterator:
        """
        Returns an iterator over fields of a node with particular filters

        Args:
            flat: return a flattened iterator (if children are iterable)
            nodes_only: do not yield children that are not Nodes (trumped by `obfuscated`)
            obfuscated: yield fields that have leading underscores
                (typically accessed via a property)
            nones: yield values that are None
                (optional fields without a value); trumped by `nodes_only`
            named: yield pairs `(field name: str, field value)`
        Returns:
            Iterator: returns all children of a node given filters
                and optional flattening (by default Iterator[Node])
        """

        def make_child_generator():
            """
            Makes a generator enclosing self to return
            not obfuscated fields (fields without starting `_`)
            """
            for self_field in fields(self):
                if (
                    not self_field.name.startswith("_") if not obfuscated else True
                ) and (self_field.name in self.__dict__):
                    value = self.__dict__[self_field.name]
                    values = [value]
                    if flat:
                        values = flatten(value)
                    for value in values:
                        if named:
                            yield (self_field.name, value)
                        else:
                            yield value

        # `iter`s used to satisfy mypy (`child_generator` type changes between generator, filter)
        child_generator = iter(make_child_generator())

        if nodes_only:
            child_generator = iter(
                filter(
                    lambda child: isinstance(child, Node)
                    if not named
                    else isinstance(child[1], Node),
                    child_generator,
                ),
            )

        if not nones:
            child_generator = iter(
                filter(
                    lambda child: (child is not None)
                    if not named
                    else (child[1] is not None),
                    child_generator,
                ),
            )  # pylint: disable=C0301

        return child_generator

    @property
    def children(self) -> Iterator["Node"]:
        """
        Returns an iterator of all nodes that are one
        step from the current node down including through iterables
        """
        return self.fields(
            flat=True,
            nodes_only=True,
            obfuscated=False,
            nones=False,
            named=False,
        )

    def replace(  # pylint: disable=invalid-name
        self,
        from_: "Node",
        to: "Node",
        compare: Optional[Callable[[Any, Any], bool]] = None,
        times: int = -1,
        copy: bool = True,
    ):
        """
        Replace a node `from_` with a node `to` in the subtree
        """
        replacements = 0
        compare_ = (lambda a, b: a is b) if compare is None else compare
        for node in self.flatten():
            if compare_(node, from_):
                node.swap(to.copy() if copy else to)
                replacements += 1
            if replacements == times:
                return

    def filter(self, func: Callable[["Node"], bool]) -> Iterator["Node"]:
        """
        Find all nodes that `func` returns `True` for
        """
        if func(self):
            yield self

        for node in chain(*[child.filter(func) for child in self.children]):
            yield node

    def contains(self, other: "Node") -> bool:
        """
        Checks if the subtree of `self` contains the node
        """
        return any(self.filter(lambda node: node is other))

    def find_all(self, node_type: Type[TNode]) -> Iterator[TNode]:
        """
        Find all nodes of a particular type in the node's sub-ast
        """
        return self.filter(lambda n: isinstance(n, node_type))  # type: ignore

    def apply(self, func: Callable[["Node"], None]):
        """
        Traverse ast and apply func to each Node
        """
        func(self)
        for child in self.children:
            child.apply(func)

    def compare(
        self,
        other: "Node",
    ) -> bool:
        """
        Compare two ASTs for deep equality
        """
        if type(self) != type(other):  # pylint: disable=unidiomatic-typecheck
            return False
        if id(self) == id(other):
            return True
        return hash(self) == hash(other)

    def diff(self, other: "Node") -> List[Tuple["Node", "Node"]]:
        """
        Compare two ASTs for differences and return the pairs of differences
        """

        def _diff(self, other: "Node"):
            if self != other:
                diffs.append((self, other))
            else:
                for child, other_child in zip_longest(self.children, other.children):
                    _diff(child, other_child)

        diffs: List[Tuple["Node", "Node"]] = []
        _diff(self, other)
        return diffs

    def similarity_score(self, other: "Node") -> float:
        """
        Determine how similar two nodes are with a float score
        """
        self_nodes = list(self.flatten())
        other_nodes = list(other.flatten())
        intersection = [
            self_node for self_node in self_nodes if self_node in other_nodes
        ]
        union = (
            [self_node for self_node in self_nodes if self_node not in intersection]
            + [
                other_node
                for other_node in other_nodes
                if other_node not in intersection
            ]
            + intersection
        )
        return len(intersection) / len(union)

    def __eq__(self, other) -> bool:
        """
        Compares two nodes for "top level" equality.

        Checks for type equality and primitive field types for full equality.
        Compares all others for type equality only. No recursing.
        Note: Does not check (sub)AST. See `Node.compare` for comparing (sub)ASTs.
        """
        return type(self) == type(other) and all(  # pylint: disable=C0123
            s == o
            if type(s) in PRIMITIVES  # pylint: disable=C0123
            else type(s) == type(o)  # pylint: disable=C0123
            for s, o in zip(
                (self.fields(False, False, False, True)),
                (other.fields(False, False, False, True)),
            )
        )

    def __hash__(self) -> int:
        """
        Hash a node
        """
        return hash(
            tuple(
                chain(
                    (type(self),),
                    self.fields(
                        flat=True,
                        nodes_only=False,
                        obfuscated=False,
                        nones=True,
                        named=False,
                    ),
                ),
            ),
        )

    @abstractmethod
    def __str__(self) -> str:
        """
        Get the string of a node
        """

    def compile(self, ctx: CompileContext):
        """
        Compile a DJ Node
        """

    def is_compiled(self) -> bool:
        """
        Checks whether a DJ AST Node is compiled
        """
        return True


class DJEnum(Enum):
    """
    A DJ AST enum
    """

    def __repr__(self) -> str:
        return str(self)


@dataclass(eq=False)
class Aliasable(Node):
    """
    A mixin for Nodes that are aliasable
    """

    alias: Optional["Name"] = None
    as_: Optional[bool] = None

    def set_alias(self: TNode, alias: "Name") -> TNode:
        self.alias = alias
        return self

    def set_as(self: TNode, as_: bool) -> TNode:
        self.as_ = as_
        return self

    @property
    def alias_or_name(self) -> "Name":
        if self.alias is not None:
            return self.alias
        elif isinstance(self, Named):
            return self.name
        else:
            raise DJParseException("Node has no alias or name.")

    @property
    def type(self):
        from dj.construction.inference import (  # pylint: disable=C0415
            get_type_of_expression,
        )

        return get_type_of_expression(self)


AliasedType = TypeVar("AliasedType", bound=Node)  # pylint: disable=C0103


@dataclass(eq=False)
class Alias(Aliasable, Generic[AliasedType]):
    """
    Wraps node types with an alias
    """

    child: AliasedType = field(default_factory=Node)

    def __str__(self) -> str:
        as_ = " AS " if self.as_ else " "
        return f"{self.child}{as_}{self.alias}"

    def is_aggregation(self) -> bool:
        return isinstance(self.child, Expression) and self.child.is_aggregation()

    def compile(self, ctx: CompileContext):
        self.child.compile(ctx)


TExpression = TypeVar("TExpression", bound="Expression")  # pylint: disable=C0103


@dataclass(eq=False)
class Expression(Node):
    """
    An expression type simply for type checking
    """

    parenthesized: Optional[bool] = field(init=False, default=None)

    @property
    def type(self) -> ColumnType:
        """
        Return the type of the expression
        """
        from dj.construction.inference import (  # pylint: disable=C0415
            get_type_of_expression,
        )

        return get_type_of_expression(self)

    def is_aggregation(self) -> bool:
        """
        Determines whether an Expression is an aggregation or not
        """
        return all(
            [
                child.is_aggregation()
                for child in self.children
                if isinstance(child, Expression)
            ]
            or [False],
        )

    def set_alias(self: TExpression, alias: "Name") -> Alias[TExpression]:
        return Alias(child=self).set_alias(alias)


@dataclass(eq=False)
class Name(Node):
    """
    The string name specified in sql with quote style
    """

    name: str
    quote_style: str = ""
    namespace: Optional["Name"] = None

    def __str__(self) -> str:
        return self.identifier()

    def identifier(self, quotes: bool = True) -> str:
        quote_style = "" if not quotes else self.quote_style
        namespace = str(self.namespace) + "." if self.namespace else ""
        return (
            f"{namespace}{quote_style}{self.name}{quote_style}"  # pylint: disable=C0301
        )


TNamed = TypeVar("TNamed", bound="Named")  # pylint: disable=C0103


@dataclass(eq=False)  # type: ignore
class Named(Node):
    """
    An Expression that has a name
    """

    name: Name

    @property
    def namespace(self):
        namespace = []
        name = self.name
        while name.namespace:
            namespace.append(name.namespace)
            name = name.namespace
        return namespace[::-1]

    def identifier(self, quotes: bool = True) -> str:
        if quotes:
            return str(self.name)

        return ".".join(
            (
                *(name.name for name in self.namespace),
                self.name.name,
            ),
        )

    @property
    def alias_or_name(self) -> "Name":
        return self.name


@dataclass(eq=False)
class Column(Aliasable, Named, Expression):
    """
    Column used in statements
    """

    _table: Optional[Union[Aliasable, "TableExpression"]] = field(repr=False, default=None)
    _type: Optional["ColumnType"] = field(repr=False, default=None)
    _expression: Optional[Expression] = field(repr=False, default=None)

    def add_type(self, type_: ColumnType) -> "Column":
        """
        Add a referenced type
        """
        self._type = type_
        return self

    @property
    def expression(self) -> Optional[Expression]:
        """
        Return the Expression this node points to in a subquery
        """
        return self._expression

    def add_expression(self, expression: "Expression") -> "Column":
        """
        Add a referenced expression where the column came from
        """
        self._expression = expression
        return self

    def add_table(self, table: "TableExpression"):
        self._table = table

    @property
    def table(self) -> Optional["TableExpression"]:
        """
        Return the table the column was referenced from
        """
        return self._table

    def is_compiled(self):
        return self.table is not None and self.expression is not None

    def compile(self, ctx: CompileContext):
        """
        Compile a column.
        Determines the table from which a column is from.
        """

        if self.is_compiled():
            return

        namespace = (
            self.name.namespace.identifier(False) if self.name.namespace else ""
        )  # a.x -> a
        query = self.get_nearest_parent_of_type(Query)  # the column's parent query
        # things we could check here:
        # - correlated queries: we could break out the check for tables into queries above and below the column
        # - correlated subqueries must appear in a from clause
        # - a column without a namespace can still be unambiguous even if multiple tables have a column of that name in the overall
        #   and we would need to check the immediate FROM only first. Currently, this is not done.
        def check_col(node) -> bool:
            # we're only looking for tables that are in a from clause
            if isinstance(node, TableExpression) and node.in_from():
                if node is not query:
                    # if the column has a namespace we need to check the table identifier to match
                    if namespace:
                        if namespace == node.alias_or_name.identifier(False):
                            # see if the node will accept the column as its own
                            return node.add_ref_column(self)
                        else:
                            return False
                    else:
                        return node.add_ref_column(self)
            return False

        found_sources = list(ctx.query.filter(check_col))
        if len(found_sources) < 1:
            ctx.exception.errors.append(
                DJErrorException(
                    DJError(
                        code=ErrorCode.INVALID_COLUMN,
                        message=f"Column`{self}` does not exist on any valid table.",
                    )
                )
            )
        if len(found_sources) > 1:
            ctx.exception.errors.append(
                DJErrorException(
                    DJError(
                        code=ErrorCode.INVALID_COLUMN,
                        message=f"Column `{self}` found in multiple tables. Consider namespacing.",
                    )
                )
            )

    def __str__(self) -> str:
        as_ = " AS " if self.as_ else " "
        alias = "" if not self.alias else f"{as_}{self.alias}"
        if self.table is not None:
            ret = f"{self.table}.{self.name.quote_style}{self.name}{self.name.quote_style}"
        else:
            ret = str(self.name)
        if self.parenthesized:
            ret = f"({ret})"
        return ret + alias


@dataclass(eq=False)
class Wildcard(Named, Expression):
    """
    Wildcard or '*' expression
    """

    name: Name = field(init=False, repr=False, default=Name("*"))
    _table: Optional["Table"] = field(repr=False, default=None)

    @property
    def table(self) -> Optional["Table"]:
        """
        Return the table the column was referenced from if there's one
        """
        return self._table

    def add_table(self, table: "Table") -> "Wildcard":
        """
        Add a referenced table
        """
        if self._table is None:
            self._table = table
        return self

    def __str__(self) -> str:
        return "*"


@dataclass(eq=False)
class TableExpression(Aliasable, Expression):
    """
    A type for table expressions
    """

    column_list: List[Column] = field(default_factory=list)
    _columns: List[Expression] = field(
        default_factory=list
    )  # all those expressions that can be had from the table
    _ref_columns: Set[Column] = field(init=False, repr=False, default_factory=set)

    @property
    def columns(self) -> Set[Column]:
        """
        Return the columns referenced from this table
        """
        return self._columns

    @property
    def ref_columns(self) -> Set[Column]:
        """
        Return the columns referenced from this table
        """
        return self._ref_columns

    def add_ref_column(self, column: Column) -> bool:
        """
        Add column referenced from this table
        returning True if the table has the column
        and False otherwise
        """
        if not self.is_compiled():
            raise DJParseException(
                "Attempted to add ref column while table expression is not compiled."
            )
        for col in self._columns:
            if isinstance(col, (Aliasable, Named)):
                if column.name.name == col.alias_or_name.identifier(False):
                    self._ref_columns.add(column)
                    column.add_table(self)
                    return True
        return False

    def is_compiled(self) -> bool:
        return bool(self._columns)

    def in_from(self) -> bool:
        """
        Determines if the table expression is referenced in a From clause
        """

        if from_ := self.get_nearest_parent_of_type(From):
            return from_.get_nearest_parent_of_type(
                Select
            ) is self.get_nearest_parent_of_type(Select)
        return False


@dataclass(eq=False)
class Table(TableExpression, Named):
    """
    A type for tables
    """

    _dj_node: Optional[DJNode] = field(repr=False, default=None)

    @property
    def dj_node(self) -> Optional[DJNode]:
        """
        Return the dj_node referenced by this table
        """
        return self._dj_node

    def set_dj_node(self, dj_node: DJNode) -> "Table":
        """
        Set dj_node referenced by this table
        """
        self._dj_node = dj_node
        return self

    def __str__(self) -> str:
        table_str = str(self.name)
        if self.alias:
            as_ = " AS " if self.as_ else " "
            table_str += f"{as_}{self.alias}"
        return table_str

    def is_compiled(self) -> bool:
        return super().is_compiled() and (self.dj_node is not None)

    def compile(self, ctx: CompileContext):
        if self.is_compiled():
            return

        # things we can validate here:
        # - if the node is a dimension in a groupby, is it joinable?
        try:
            dj_node = get_dj_node(
                ctx.session,
                self.identifier(),
                {DJNodeType.SOURCE, DJNodeType.TRANSFORM, DJNodeType.DIMENSION},
            )
            self.set_dj_node(dj_node)
            self._columns = [
                Column(Name(col.name), _type=col.type, _table=self)
                for col in dj_node.columns
            ]
        except DJErrorException as exc:
            ctx.exception.errors.append(exc.dj_error)


class Operation(Expression):
    """
    A type to overarch types that operate on other expressions
    """


# pylint: disable=C0103
class UnaryOpKind(DJEnum):
    """
    The accepted unary operations
    """
    #
    # Plus = "+"
    # Minus = "-"
    Exists = "EXISTS"
    Not = "NOT"


@dataclass(eq=False)
class UnaryOp(Operation):
    """
    An operation that operates on a single expression
    """

    op: UnaryOpKind
    expr: Expression

    def __str__(self) -> str:
        ret = f"{self.op} {self.expr}"
        if self.parenthesized:
            return f"({ret})"
        return ret


# pylint: disable=C0103
class BinaryOpKind(DJEnum):
    """
    The DJ AST accepted binary operations
    """

    And = "AND"
    Or = "OR"
    Is = "IS"
    Eq = "="
    NotEq = "<>"
    Gt = ">"
    Lt = "<"
    GtEq = ">="
    LtEq = "<="
    BitwiseOr = "|"
    BitwiseAnd = "&"
    BitwiseXor = "^"
    Multiply = "*"
    Divide = "/"
    Plus = "+"
    Minus = "-"
    Modulo = "%"
    Like = "LIKE"


@dataclass(eq=False)
class BinaryOp(Operation):
    """
    Represents an operation that operates on two expressions
    """

    op: BinaryOpKind
    left: Expression
    right: Expression

    def __str__(self) -> str:
        ret = f"{self.left} {self.op.value} {self.right}"
        if self.parenthesized:
            return f"({ret})"
        return ret

    def compile(self, ctx: CompileContext):
        self.left.compile(ctx)
        self.right.compile(ctx)


@dataclass(eq=False)
class FrameBound(Expression):
    """
    Represents frame bound in a window function
    """

    start: str
    stop: str

    def __str__(self) -> str:
        return f"{self.start} {self.stop}"


@dataclass(eq=False)
class Frame(Expression):
    """
    Represents frame in window function
    """

    frame_type: str
    start: FrameBound
    end: Optional[FrameBound] = None

    def __str__(self) -> str:
        end = f" AND {self.end}" if self.end else ""
        between = " BETWEEN" if self.end else ""
        return f"{self.frame_type}{between} {self.start}{end}"


@dataclass(eq=False)
class Over(Expression):
    """
    Represents a function used in a statement
    """

    partition_by: List[Expression] = field(default_factory=list)
    order_by: List["SortItem"] = field(default_factory=list)
    window_frame: Optional[Frame] = None

    def __str__(self) -> str:
        partition_by = (  # pragma: no cover
            " PARTITION BY " + ", ".join(str(exp) for exp in self.partition_by)
            if self.partition_by
            else ""
        )
        order_by = (
            " ORDER BY " + ", ".join(str(exp) for exp in self.order_by)
            if self.order_by
            else ""
        )
        consolidated_by = "\n".join(
            po_by for po_by in (partition_by, order_by) if po_by
        )
        window_frame = f" {self.window_frame}" if self.window_frame else ""
        return f"OVER ({consolidated_by}{window_frame})"


@dataclass(eq=False)
class Function(Named, Operation):
    """
    Represents a function used in a statement
    """

    args: List[Expression] = field(default_factory=list)
    quantifier: str = ""
    over: Optional[Over] = None

    def __str__(self) -> str:
        over = f" {self.over}" if self.over else ""

        ret = f"{self.name}({self.quantifier}{', '.join(str(arg) for arg in self.args)}){over}"
        if self.parenthesized:
            ret = f"({ret})"
        return ret

    def is_aggregation(self) -> bool:
        return function_registry[self.name.name.upper()].is_aggregation

    def compile(self, ctx: CompileContext):
        for expr in self.args:
            expr.compile(ctx)


class Value(Expression):
    """
    Base class for all values number, string, boolean
    """


@dataclass(eq=False)
class Null(Value):
    """
    Null value
    """

    def __str__(self) -> str:
        return "NULL"


@dataclass(eq=False)
class Number(Value):
    """
    Number value
    """

    value: Union[float, int, decimal.Decimal]

    def __post_init__(self):
        super().__post_init__()

        if (
            not isinstance(self.value, float) and
            not isinstance(self.value, int) and
            not isinstance(self.value, decimal.Decimal)
        ):
            cast_exceptions = []
            numeric_types = [int, float, decimal.Decimal]
            for cast_type in numeric_types:
                try:
                    self.value = cast_type(self.value)
                    break
                except (ValueError, OverflowError) as exception:
                    cast_exceptions.append(exception)
            if len(cast_exceptions) >= len(numeric_types):
                raise DJException(message="Not a valid number!")

    def __str__(self) -> str:
        return str(self.value)


@dataclass(eq=False)
class String(Value):
    """
    String value
    """

    value: str

    def __str__(self) -> str:
        return f"'{self.value}'"


@dataclass(eq=False)
class Boolean(Value):
    """
    Boolean True/False value
    """

    value: bool

    def __str__(self) -> str:
        return str(self.value)


@dataclass(eq=False)
class IntervalUnit(Value):
    """
    Interval unit value
    """

    unit: str
    value: Optional[Number] = None

    def __str__(self) -> str:
        return f"{self.value or ''} {self.unit}"


@dataclass(eq=False)
class Interval(Value):
    """
    Interval value
    """

    from_: List[IntervalUnit]
    to: Optional[IntervalUnit] = None

    def __str__(self) -> str:
        to = f"TO {self.to}" if self.to else ""
        return f"INTERVAL {' '.join(str(interval) for interval in self.from_)} {to}"


@dataclass(eq=False)
class Struct(Value):
    """
    Struct value
    """

    values: List[Aliasable]

    def __str__(self):
        inner = ", ".join(str(value) for value in self.values)
        return f"STRUCT({inner})"


@dataclass(eq=False)
class Predicate(Operation):
    """
    Represents a predicate
    """

    negated: bool = False


@dataclass(eq=False)
class Between(Predicate):
    """
    A between statement
    """

    expr: Expression = field(default_factory=Expression)
    low: Expression = field(default_factory=Expression)
    high: Expression = field(default_factory=Expression)

    def __str__(self) -> str:
        not_ = "NOT " if self.negated else ""
        between = f"{not_}{self.expr} BETWEEN {self.low} AND {self.high}"
        if self.parenthesized:
            between = f"({between})"
        return between

    def compile(self, ctx: CompileContext):
        self.expr.compile(ctx)
        self.low.compile(ctx)
        self.high.compile(ctx)


@dataclass(eq=False)
class In(Predicate):
    """
    An in expression
    """

    expr: Expression = field(default_factory=Expression)
    source: Union[List[Expression], "Select"] = field(default_factory=Expression)

    def __str__(self) -> str:
        not_ = "NOT " if self.negated else ""
        source = (
            str(self.source)
            if isinstance(self.source, Select)
            else "(" + ", ".join(str(exp) for exp in self.source) + ")"
        )
        return f"{self.expr} {not_}IN {source}"


@dataclass(eq=False)
class Rlike(Predicate):
    """
    A regular expression match statement
    """

    expr: Expression = field(default_factory=Expression)
    pattern: Expression = field(default_factory=Expression)

    def __str__(self) -> str:
        not_ = "NOT " if self.negated else ""
        return f"{not_}{self.expr} RLIKE {self.pattern}"


@dataclass(eq=False)
class Like(Predicate):
    """
    A string pattern matching statement
    """

    expr: Expression = field(default_factory=Expression)
    quantifier: str = ""
    patterns: List[Expression] = field(default_factory=list)
    escape_char: Optional[str] = None
    case_sensitive: Optional[bool] = True

    def __str__(self) -> str:
        not_ = "NOT " if self.negated else ""
        if self.quantifier:  # quantifier means a pattern with multiple elements
            pattern = f'({", ".join(str(p) for p in self.patterns)})'
        else:
            pattern = self.patterns
        escape_char = f" ESCAPE '{self.escape_char}'" if self.escape_char else ""
        quantifier = f"{self.quantifier} " if self.quantifier else ""
        like_type = "LIKE" if self.case_sensitive else "ILIKE"
        return f"{not_}{self.expr} {like_type} {quantifier}{pattern}{escape_char}"

    def compile(self, ctx: CompileContext):
        self.expr.compile(ctx)


@dataclass(eq=False)
class IsNull(Predicate):
    """
    A null check statement
    """

    expr: Expression = field(default_factory=Expression)

    def __str__(self) -> str:
        not_ = "NOT " if self.negated else ""
        return f"{self.expr} IS {not_}NULL"


@dataclass(eq=False)
class IsBoolean(Predicate):
    """
    A boolean check statement
    """

    expr: Expression = field(default_factory=Expression)
    value: str = "UNKNOWN"

    def __str__(self) -> str:
        not_ = "NOT " if self.negated else ""
        return f"{self.expr} IS {not_}{self.value}"


@dataclass(eq=False)
class IsDistinctFrom(Predicate):
    """
    A distinct from check statement
    """

    expr: Expression = field(default_factory=Expression)
    right: Expression = field(default_factory=Expression)

    def __str__(self) -> str:
        not_ = "NOT " if self.negated else ""
        return f"{self.expr} IS {not_}DISTINCT FROM {self.right}"


@dataclass(eq=False)
class Case(Expression):
    """
    A case statement of branches
    """

    expr: Optional[Expression] = None
    conditions: List[Expression] = field(default_factory=list)
    else_result: Optional[Expression] = None
    operand: Optional[Expression] = None
    results: List[Expression] = field(default_factory=list)

    def __str__(self) -> str:
        branches = "\n\tWHEN ".join(
            f"{(cond)} THEN {(result)}"
            for cond, result in zip(self.conditions, self.results)
        )
        else_ = f"ELSE {(self.else_result)}" if self.else_result else ""
        expr = "" if self.expr is None else f" {self.expr} "
        ret = f"""CASE {expr}
        WHEN {branches}
        {else_}
    END"""
        if self.parenthesized:
            return f"({ret})"
        return ret

    def is_aggregation(self) -> bool:
        return all(result.is_aggregation() for result in self.results) and (
            self.else_result.is_aggregation() if self.else_result else True
        )

    def compile(self, ctx: CompileContext):
        if self.expr:
            self.expr.compile(ctx)
        for cond in self.conditions:
            cond.compile(ctx)
        if self.else_result:
            self.else_result.compile(ctx)
        if self.operand:
            self.operand.compile(ctx)
        for result in self.results:
            result.compile(ctx)


@dataclass(eq=False)
class Subscript(Expression):
    """
    Represents a subscript expression
    """

    expr: Expression
    index: Expression

    def __str__(self) -> str:
        return f"{self.expr}[{self.index}]"

    def compile(self, ctx: CompileContext):
        self.expr.compile(ctx)


@dataclass(eq=False)
class Lambda(Expression):
    """
    Represents a lambda expression
    """

    identifiers: List[Named]
    expr: Expression

    def __str__(self) -> str:
        if len(self.identifiers) == 1:
            id_str = self.identifiers[0]
        else:
            id_str = "(" + ", ".join(str(iden) for iden in self.identifiers) + ")"
        return f"{id_str} -> {self.expr}"


@dataclass(eq=False)
class JoinCriteria(Node):
    """
    Represents the criteria for a join relation in a FROM clause
    """

    on: Optional[Expression] = None
    using: Optional[List[Named]] = None

    def __str__(self) -> str:
        if self.on:
            return f"ON {self.on}"
        else:
            id_list = ", ".join(str(iden) for iden in self.using)
            return f"USING ({id_list})"


@dataclass(eq=False)
class Join(Node):
    """
    Represents a join relation in a FROM clause
    """

    join_type: str
    right: Expression
    criteria: Optional[JoinCriteria] = None
    lateral: bool = False
    natural: bool = False

    def __str__(self) -> str:
        parts = []
        if self.natural:
            parts.append("NATURAL ")
        if self.join_type:
            parts.append(f"{self.join_type} ")
        parts.append("JOIN ")
        if self.lateral:
            parts.append("LATERAL ")
        parts.append(str(self.right))
        if self.criteria:
            parts.append(f" {self.criteria}")
        return "".join(parts)


@dataclass(eq=False)
class FunctionTableExpression(TableExpression, Named, Operation):
    """
    An uninitializable Type for FunctionTable for use as a
    default where a FunctionTable is required but succeeds optional fields
    """

    args: List[Expression] = field(default_factory=list)


class FunctionTable(FunctionTableExpression):
    """
    Represents a table-valued function used in a statement
    """

    def __str__(self) -> str:
        alias = f" AS {self.alias}" if self.alias else ""
        cols = f"({', '.join(str(col) for col in self.column_list)})"
        return f"{self.name}{alias}{cols}"


@dataclass(eq=False)
class LateralView(Expression):
    """
    Represents a lateral view expression
    """

    outer: bool = False
    func: FunctionTableExpression = field(default_factory=FunctionTableExpression)
    table: TableExpression = field(default_factory=TableExpression)
    columns: List[Column] = field(default_factory=list)

    def __str__(self) -> str:
        parts = ["LATERAL VIEW"]
        if self.outer:
            parts.append(" OUTER")
        parts.append(f" {self.func}")
        parts.append(f" {self.table}")
        if self.columns:
            parts.append(f" AS {', '.join(str(col) for col in self.columns)}")
        return "".join(parts)


@dataclass(eq=False)
class Relation(Node):
    """
    Represents a relation
    """

    primary: Expression
    extensions: List[Join] = field(default_factory=list)

    def __str__(self) -> str:
        if self.extensions:
            extensions = " " + "\n".join([str(ext) for ext in self.extensions])
        else:
            extensions = ""
        return f"{self.primary}{extensions}"

    def compile(self, ctx: CompileContext):
        self.primary.compile(ctx)


@dataclass(eq=False)
class From(Node):
    """
    Represents the FROM clause of a SELECT statement
    """

    relations: List[Relation] = field(default_factory=list)
    lateral_views: List[LateralView] = field(default_factory=list)

    def __str__(self) -> str:
        parts = ["FROM "]
        parts += ",\n".join([str(r) for r in self.relations])
        for view in self.lateral_views:
            parts.append(f"\n{view}")
        return "".join(parts)

    def compile(self, ctx: CompileContext):
        for relation in self.relations:
            relation.compile(ctx)


@dataclass(eq=False)
class SetOp(TableExpression):
    """
    A column wrapper for ordering
    """

    kind: str = ""  # Union, intersect, ...
    left: Optional[TableExpression] = None
    right: Optional[TableExpression] = None

    def __str__(self) -> str:
        return f"{self.left}\n{self.kind}\n{self.right}"

    def compile(self, ctx: CompileContext):
        """
        Compile a set operation
        """
        if self.is_compiled():
            return
        # things we could check here:
        # - all table expressions in the setop must have the same number of columns and the pairwise types must be the same

        # column names are taken from the leading query i.e. left
        left = cast(self.left, TableExpression)
        left.compile(ctx)
        if left.is_compiled():
            self._columns = left._columns[:]


@dataclass(eq=False)
class Cast(Expression):
    """
    A cast to a specified type
    """

    data_type: str
    expression: Expression

    def __str__(self) -> str:
        return f"CAST({self.expression} AS {self.data_type})"

    @property
    def type(self) -> ColumnType:
        """
        Return the type of the expression
        """
        return ColumnType[self.data_type]


@dataclass(eq=False)
class SelectExpression(Aliasable, Expression):
    """
    An uninitializable Type for Select for use as a
    default where a Select is required but succeeds optional fields
    """

    quantifier: str = ""  # Distinct, All
    projection: List[Union[Aliasable, Expression]] = field(default_factory=list)
    from_: Optional[From] = None
    group_by: List[Expression] = field(default_factory=list)
    having: Optional[Expression] = None
    where: Optional[Expression] = None
    set_op: List[SetOp] = field(default_factory=list)


class Select(SelectExpression):
    """
    A single select statement type
    """

    def add_set_op(self, set_op: SetOp):
        """
        Add a set op such as UNION, UNION ALL or INTERSECT
        """
        self.set_op.append(set_op)

    def add_aliases_to_unnamed_columns(self) -> None:
        """
        Add an alias to any unnamed columns in the projection (`col{n}`)
        """
        projection = []
        for i, expression in enumerate(self.projection):
            if not isinstance(expression, Aliasable):
                name = f"col{i}"
                # only replace those that are identical in memory
                projection.append(expression.set_alias(Name(name)))
            else:
                projection.append(expression)
        self.projection = projection

    def __str__(self) -> str:
        parts = ["SELECT "]
        if self.quantifier:
            parts.append(f"{self.quantifier}\n")
        parts.append(",\n\t".join(str(exp) for exp in self.projection))
        if self.from_ is not None:
            parts.extend(("\n", str(self.from_), "\n"))
        if self.where is not None:
            parts.extend(("WHERE ", str(self.where), "\n"))
        if self.group_by:
            parts.extend(("GROUP BY ", ", ".join(str(exp) for exp in self.group_by)))
        if self.having is not None:
            parts.extend(("HAVING ", str(self.having), "\n"))

        select = " ".join(parts).strip()

        # Add set operations
        if self.set_op:
            if self.parenthesized:
                select = f"({select})"  # Add additional parentheses inclusive of set operations
            select += "\n" + "\n".join([str(so) for so in self.set_op])

        if self.parenthesized:
            select = f"({select})"

        if self.alias:
            as_ = " AS " if self.as_ else " "
            return f"{select}{as_}{self.alias}"
        return select


@dataclass(eq=False)
class SortItem(Node):
    """
    Defines a sort item of an expression
    """

    expr: Expression
    asc: str
    nulls: str

    def __str__(self) -> str:
        return f"{self.expr} {self.asc} {self.nulls}".strip()


@dataclass(eq=False)
class Organization(Node):
    """
    A column wrapper for ordering
    """

    order: List[SortItem]
    sort: List[SortItem]

    def __str__(self) -> str:
        ret = ""
        ret += f"ORDER BY {', '.join(str(i) for i in self.order)}" if self.order else ""
        if ret:
            ret += "\n"
        ret += f"SORT BY {', '.join(str(i) for i in self.sort)}" if self.sort else ""
        return ret


@dataclass(eq=False)
class Query(TableExpression):
    """
    Overarching query type
    """

    select: SelectExpression = field(default_factory=SelectExpression)
    ctes: List["Query"] = field(default_factory=list)
    limit: Optional[Expression] = None
    organization: Optional[Organization] = None

    def compile(self, ctx: CompileContext):
        if self.is_compiled():
            return
        if self.select.from_:
            self.select.from_.compile(ctx)
        for projection in self.select.projection:
            projection.compile(ctx)
        self._columns = self.select.projection[:]

    def bake_ctes(self):
        """
        Add ctes into the select and return the select

        Note: This destroys the structure of the query which cannot be undone
        you may want to deepcopy it first
        """
        for cte in self.ctes:
            table = Table(cte.name, cte.namespace)
            self.replace(table, cte)

    def __str__(self) -> str:
        is_cte = self.parent is not None and self.parent_key == "ctes"
        ctes = ",\n".join(str(cte) for cte in self.ctes)
        with_ = "WITH" if ctes else ""

        parts = [f"{with_}\n{ctes}\n{self.select}\n"]
        if self.organization:
            parts.append(str(self.organization))
        if self.limit is not None:
            limit = f"LIMIT {self.limit}"
            parts.append(limit)
        query = "".join(parts)
        if self.parenthesized:
            query = f"({query})"
        if self.alias:
            as_ = " AS " if self.as_ else " "
            if is_cte:
                query = f"{self.alias}{as_}{query}"
            else:
                query = f"{query}{as_}{self.alias}"
        return query
