"""
A DataJunction client for connecting to a DataJunction server
"""
from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

from datajunction._internal import (
    Cube,
    Dimension,
    Metric,
    Namespace,
    Node,
    Source,
    Transform,
)
from datajunction.client import DJReader, DJWriter
from datajunction.models import (
    AvailabilityState,
    ColumnAttribute,
    Engine,
    MaterializationConfig,
    NodeMode,
)

try:
    # Change here if project is renamed and does not equal the package name
    DIST_NAME = __name__
    __version__ = version(DIST_NAME)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError


__all__ = [
    "DJReader",
    "DJWriter",
    "AvailabilityState",
    "ColumnAttribute",
    "Source",
    "Dimension",
    "Transform",
    "MaterializationConfig",
    "Metric",
    "Cube",
    "Node",
    "NodeMode",
    "Namespace",
    "Engine",
]
