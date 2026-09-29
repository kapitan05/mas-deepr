"""``MCPToolClient``'s persistent-connection behavior (fixed 2026-09-26):
one ``fastmcp.Client`` per ``MCPToolClient`` instance, reused across every
call, instead of a fresh one constructed per call.

Mocks ``fastmcp.Client`` itself (not a real MCP server) -- this is purely
about whether ``MCPToolClient`` constructs/reuses that object correctly,
not about the wire protocol.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mas_deepr.tools.mcp_client import MCPToolClient


class _FakeResult:
    def __init__(self, data: Any) -> None:
        self.data = data


def _make_fake_client_cls() -> tuple[MagicMock, list[MagicMock]]:
    """A fake fastmcp.Client class: each instantiation returns a distinct
    mock instance (so we can tell whether MCPToolClient made one client or
    several), each with async __aenter__/__aexit__/call_tool."""
    instances: list[MagicMock] = []

    def _new_instance(*args: object, **kwargs: object) -> MagicMock:
        instance = MagicMock()
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=None)
        instance.call_tool = AsyncMock(return_value=_FakeResult({"ok": True}))
        instances.append(instance)
        return instance

    client_cls = MagicMock(side_effect=_new_instance)
    return client_cls, instances


@pytest.mark.asyncio
async def test_constructs_exactly_one_client_for_many_calls() -> None:
    """Regression test: the old code did `Client(self._target())` inside
    _execute, a fresh connection every call. Confirms only ONE Client is
    ever constructed, no matter how many tool calls happen."""
    client_cls, instances = _make_fake_client_cls()
    with patch("mas_deepr.tools.mcp_client.Client", client_cls):
        mcp_client = MCPToolClient(url="http://unused")
        for _ in range(5):
            await mcp_client.call("some_tool", query="x")

    assert client_cls.call_count == 1
    assert len(instances) == 1
    assert instances[0].call_tool.call_count == 5


@pytest.mark.asyncio
async def test_each_call_still_enters_and_exits_the_shared_client() -> None:
    """Without an outer async-with holder, every call still enters/exits
    the (now-shared) client itself -- fastmcp's own reentrant Client
    self-manages this correctly regardless."""
    client_cls, instances = _make_fake_client_cls()
    with patch("mas_deepr.tools.mcp_client.Client", client_cls):
        mcp_client = MCPToolClient(url="http://unused")
        await mcp_client.call("some_tool", query="x")
        await mcp_client.call("some_tool", query="y")

    instance = instances[0]
    assert instance.__aenter__.call_count == 2
    assert instance.__aexit__.call_count == 2


@pytest.mark.asyncio
async def test_outer_context_manager_delegates_to_underlying_client() -> None:
    client_cls, instances = _make_fake_client_cls()
    with patch("mas_deepr.tools.mcp_client.Client", client_cls):
        mcp_client = MCPToolClient(url="http://unused")
        async with mcp_client as entered:
            assert entered is mcp_client
            await mcp_client.call("some_tool", query="x")

    instance = instances[0]
    # One outer enter/exit, plus one inner enter/exit from the call inside
    # it -- both against the SAME client instance (reentrant), never a
    # second Client constructed.
    assert client_cls.call_count == 1
    assert instance.__aenter__.call_count == 2
    assert instance.__aexit__.call_count == 2
