"""
Utilities used around construction
"""

from dataclasses import dataclass
from string import ascii_letters, digits
from typing import TYPE_CHECKING, List, Optional, Set

from sqlalchemy.orm.exc import NoResultFound
from sqlmodel import Session, select

from dj.construction.exceptions import CompoundBuildException
from dj.errors import DJError, ErrorCode, DJException, DJErrorException
from dj.models.node import Node, NodeRevision, NodeType

if TYPE_CHECKING:
    from dj.sql.parsing.ast import Namespace


def make_name(namespace: Optional["Namespace"], name="") -> str:
    """utility taking a namespace and name to make a possible name of a DJ Node"""
    ret = ""
    if namespace:
        ret += ".".join(n.name for n in namespace.names)
    if name:
        ret += ("." if ret else "") + name
    return ret


def get_dj_node(
    session: Session,
    node_name: str,
    kinds: Optional[Set[NodeType]] = None,
) -> NodeRevision:
    """Return the DJ Node with a given name from a set of node types"""
    query = select(Node).filter(Node.name == node_name)
    match = None
    try:
        match = session.exec(query).one()
    except NoResultFound:
        kind_msg = " or ".join(str(k) for k in kinds) if kinds else ""
        raise DJErrorException(DJError(
                code=ErrorCode.UNKNOWN_NODE,
                message=f"No node `{node_name}` exists of kind {kind_msg}.",
            )) 

    # found a node but it's not the right kind
    if match and kinds and (match.type not in kinds):
        raise DJErrorException(DJError(
                code=ErrorCode.NODE_TYPE_ERROR,
                message=(
                    f"Node `{match.name}` is of type `{str(match.type).upper()}`. "
                    f"Expected kind to be of {' or '.join(str(k) for k in kinds)}."
                ),
            ),
        )

    return match.current


ACCEPTABLE_CHARS = set(ascii_letters + digits + "_")
LOOKUP_CHARS = {
    ".": "DOT",
    "'": "QUOTE",
    '"': "DQUOTE",
    "`": "BTICK",
    "!": "EXCL",
    "@": "AT",
    "#": "HASH",
    "$": "DOLLAR",
    "%": "PERC",
    "^": "CARAT",
    "&": "AMP",
    "*": "STAR",
    "(": "LPAREN",
    ")": "RPAREN",
    "[": "LBRACK",
    "]": "RBRACK",
    "-": "MINUS",
    "+": "PLUS",
    "=": "EQ",
    "/": "FSLSH",
    "\\": "BSLSH",
    "|": "PIPE",
    "~": "TILDE",
}


def amenable_name(name: str) -> str:
    """Takes a string and makes it have only alphanumerics"""
    ret: List[str] = []
    cont: List[str] = []
    for char in name:
        if char in ACCEPTABLE_CHARS:
            cont.append(char)
        else:
            ret.append("".join(cont))
            ret.append(LOOKUP_CHARS.get(char, "UNK"))
            cont = []

    return ("_".join(ret) + "_" + "".join(cont)).strip("_")
