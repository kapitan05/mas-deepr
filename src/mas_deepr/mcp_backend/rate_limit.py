"""Server-side rate limiting for MCP providers.

Deliberately separate from ``tools/mcp_client.py``'s ``asyncio.Semaphore``s.
A concurrency semaphore bounds how many requests are *in flight at once*;
it does NOT bound *requests per second over time* -- 8 fast, sequential
calls that each complete in 200ms can blow through a hard "3 requests/sec"
limit (PubMed's real, documented ceiling) even at concurrency=1. A rate
limiter (token bucket, via ``aiolimiter``) is the right primitive for
"don't exceed N requests per T seconds", which is what actually prevents
a 429/ban from an external provider.

Also deliberately server-side, not client-side: the MCP server is the one
process that actually makes the outbound HTTP call to SearXNG/PubMed/
Semantic Scholar, regardless of how many separate client processes
(eval script, GRPO training, a second eval running concurrently) are
asking it to. A per-client-process limiter can't see requests from other
processes; a per-provider limiter living here, in the server, can -- one
shared bucket per provider, per server process, is the actual chokepoint
that matches where the real ban risk lives.
"""

import asyncio
import functools
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from aiolimiter import AsyncLimiter

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])

# One limiter/semaphore per (provider, event loop) -- shared across every
# call that provider gets for the life of *that loop*. Keyed on the loop
# too, not just the provider, because both AsyncLimiter and
# asyncio.Semaphore bind to whichever event loop first uses them; reusing
# one across a *different* loop is explicitly undefined behavior for
# AsyncLimiter (aiolimiter warns on it) and a hard RuntimeError for
# Semaphore on some versions. In production this changes nothing -- the
# MCP server is one process with exactly one event loop for its whole
# lifetime, so the key always resolves to the same single entry. It
# matters for tests: pytest-asyncio gives each test function its own
# event loop, so a module-level cache keyed on provider alone silently
# handed a later test a limiter still bound to an earlier, already-closed
# loop -- confirmed live 2026-09-29 as the cause of an intermittent (CI
# only, not reproducible locally) empty-result failure in the SearXNG ->
# Tavily fallback tests.
_LIMITERS: dict[tuple[str, int], AsyncLimiter] = {}
_SEMAPHORES: dict[tuple[str, int], asyncio.Semaphore] = {}


def _limiter_for(provider: str, max_rate: float, time_period: float) -> AsyncLimiter:
    key = (provider, id(asyncio.get_running_loop()))
    if key not in _LIMITERS:
        _LIMITERS[key] = AsyncLimiter(max_rate, time_period)
    return _LIMITERS[key]


def _semaphore_for(provider: str, max_concurrent: int) -> asyncio.Semaphore:
    key = (provider, id(asyncio.get_running_loop()))
    if key not in _SEMAPHORES:
        _SEMAPHORES[key] = asyncio.Semaphore(max_concurrent)
    return _SEMAPHORES[key]


def rate_limited(
    provider: str, *, max_rate: float, time_period: float = 1.0
) -> Callable[[F], F]:
    """Decorator: cap ``provider``'s calls to ``max_rate`` per ``time_period``
    seconds, shared across all callers within this server process (see
    module docstring for why the underlying limiter is looked up per call,
    not captured once at decoration time)."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            limiter = _limiter_for(provider, max_rate, time_period)
            async with limiter:
                return await func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def concurrency_limited(provider: str, *, max_concurrent: int) -> Callable[[F], F]:
    """Decorator: cap ``provider``'s *simultaneous in-flight* calls to
    ``max_concurrent``, shared across all callers within this server
    process. For local-resource-bound providers (Crawl4AI's headless
    browser) rather than external-rate-limit-bound ones -- see
    ``rate_limited`` for those."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            semaphore = _semaphore_for(provider, max_concurrent)
            async with semaphore:
                return await func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
