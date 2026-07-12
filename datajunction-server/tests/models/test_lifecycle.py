"""Unit tests for LifecycleState and resolve_lifecycle (pure, no DB)."""

import pytest

from datajunction_server.models.node import (
    LifecycleState,
    NodeMode,
    resolve_lifecycle,
)


@pytest.mark.parametrize(
    "state, expected_mode",
    [
        (LifecycleState.DEV, NodeMode.DRAFT),
        (LifecycleState.EXPERIMENTAL, NodeMode.DRAFT),
        (LifecycleState.STABLE, NodeMode.PUBLISHED),
        (LifecycleState.DEPRECATED, NodeMode.PUBLISHED),
        (LifecycleState.RETIRED, NodeMode.PUBLISHED),
    ],
)
def test_to_mode(state, expected_mode):
    assert state.to_mode() == expected_mode


@pytest.mark.parametrize(
    "mode, expected_state",
    [
        (NodeMode.DRAFT, LifecycleState.DEV),
        (NodeMode.PUBLISHED, LifecycleState.STABLE),
    ],
)
def test_from_mode(mode, expected_state):
    assert LifecycleState.from_mode(mode) == expected_state


def test_resolve_prefers_explicit_lifecycle():
    # lifecycle given → it wins and mode is derived from it
    assert resolve_lifecycle(NodeMode.PUBLISHED, LifecycleState.EXPERIMENTAL) == (
        LifecycleState.EXPERIMENTAL,
        NodeMode.DRAFT,
    )


def test_resolve_derives_lifecycle_from_mode_when_absent():
    # back-compat: only mode given → lifecycle derived, mode unchanged
    assert resolve_lifecycle(NodeMode.DRAFT, None) == (
        LifecycleState.DEV,
        NodeMode.DRAFT,
    )
    assert resolve_lifecycle(NodeMode.PUBLISHED, None) == (
        LifecycleState.STABLE,
        NodeMode.PUBLISHED,
    )
