"""MCP client adapter: mas-deepr's minimal MCPMixin-equivalent
(docs/dr-tulu-agent-infra.md ss2), retargeted at MAF's ``@tool`` framework
instead of DR-Tulu's own ``BaseTool``.

Per-provider ``asyncio.Semaphore``s (not one shared limiter) bound
concurrent calls to the MCP server -- SearXNG gets its own, lower default
since it proxies/scrapes upstream search engines and risks rate-limiting
at high concurrency, unlike the other providers. Tenacity retries only
transport-layer failures; ``call()`` never raises into the agent loop,
mirroring ``fetch_page``'s pre-existing "never raise" convention -- no
DR-Tulu-style multi-mode error enum, since nothing downstream is built to
catch a raised tool exception.

One persistent ``fastmcp.Client`` per instance, not a fresh one per call
(fixed 2026-09-26) -- confirmed directly against the installed fastmcp's
own source (``client/client.py::_connect``/``_disconnect``): a ``Client``
is an explicitly reentrant, reference-counted context manager -- entering
it while an outer ``async with`` on the SAME object is already open just
increments a counter and reuses the live session/transport; the full
connect (background session task, MCP initialize handshake, transport-
level connection) only happens when nesting drops to and from zero. DR
Tulu's own ``MCPMixin`` relies on exactly this (one cached client per tool
instance, "ping once" via a ``self.pinged`` flag) -- our old code
defeated it by constructing a brand-new ``Client(url)`` every single call
(``async with semaphore, Client(self._target()) as client:``), so every
tool call paid a full reconnect regardless. ``__aenter__``/``__aexit__``
here let a caller hold one long-lived outer context around a whole
pipeline run (``agents/topology.py::run_pipeline`` does exactly this) so
every tool call inside it reuses one connection; without an outer holder,
``_execute``'s own ``async with self._client:`` still self-manages
correctly (connects, then tears down at that call's exit) -- same
behavior as before this fix, just never worse.
"""

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

from fastmcp import Client
from fastmcp.exceptions import FastMCPError
from mcp.shared.exceptions import MCPError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

_TRANSPORT_ERRORS = (ConnectionError, TimeoutError, MCPError, FastMCPError, OSError)

# One tool -> its own concurrency pool. Deliberately per-provider, not one
# shared semaphore, so SearXNG's conservative cap can't starve (or be
# starved by) the free-of-that-risk providers -- see module docstring.
_SEARXNG_TOOL_NAMES = frozenset({"web_search"})


class MCPToolClient:
    """One instance per pipeline build (shared across all Browser tool
    calls for that model), holding the per-provider semaphores and one
    persistent ``fastmcp.Client`` connection (see module docstring)."""

    def __init__(
        self,
        *,
        url: str,
        transport: str = "http",
        concurrency: int = 8,
        searxng_concurrency: int = 2,
    ) -> None:
        self._url = url
        self._transport = transport
        self._default_semaphore = asyncio.Semaphore(concurrency)
        self._searxng_semaphore = asyncio.Semaphore(searxng_concurrency)
        # Constructing a Client doesn't connect anything yet (fastmcp
        # connects lazily on first __aenter__) -- safe to build even for a
        # pipeline that ends up never calling a tool.
        self._client = Client(self._target())

    async def __aenter__(self) -> "MCPToolClient":
        """Hold one connection open for everything inside this block --
        see module docstring. Reentrant: safe to nest with individual
        ``_execute`` calls' own ``async with self._client:``."""
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._client.__aexit__(*exc_info)

    def _semaphore_for(self, tool_name: str) -> asyncio.Semaphore:
        return (
            self._searxng_semaphore
            if tool_name in _SEARXNG_TOOL_NAMES
            else self._default_semaphore
        )

    def _target(self) -> str:
        # stdio transport isn't wired up client-side yet (HTTP is the
        # recommended path -- see the MCP-adoption plan's transport
        # discussion); keeping this as one seam if stdio is ever needed.
        return self._url

    @retry(
        retry=retry_if_exception_type(_TRANSPORT_ERRORS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _execute(self, tool_name: str, params: Mapping[str, Any]) -> Any:
        semaphore = self._semaphore_for(tool_name)
        async with semaphore, self._client:
            result = await self._client.call_tool(tool_name, dict(params))
            return result.data

    async def call(self, tool_name: str, **params: Any) -> Any:
        """Call an MCP tool; never raises -- returns an inline
        ``"[mcp_call_failed: ...]"`` string on any failure (transport
        error after retries exhausted, or a tool-side error FastMCP
        surfaces as an exception)."""
        try:
            return await self._execute(tool_name, params)
        except Exception as e:
            logger.warning(
                "mcp call failed tool=%s error=%s: %s", tool_name, type(e).__name__, e
            )
            return f"[mcp_call_failed: {type(e).__name__}: {e}]"
