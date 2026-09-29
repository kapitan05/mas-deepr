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

# One limiter/semaphore per provider, module-level -- shared across every
# call that provider gets for the life of the server process, regardless
# of which client (or how many concurrent client processes) issued it.
_LIMITERS: dict[str, AsyncLimiter] = {}
_SEMAPHORES: dict[str, asyncio.Semaphore] = {}


def rate_limited(
    provider: str, *, max_rate: float, time_period: float = 1.0
) -> Callable[[F], F]:
    """Decorator: cap ``provider``'s calls to ``max_rate`` per ``time_period``
    seconds, shared across all callers within this server process."""
    if provider not in _LIMITERS:
        _LIMITERS[provider] = AsyncLimiter(max_rate, time_period)
    limiter = _LIMITERS[provider]

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
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
    if provider not in _SEMAPHORES:
        _SEMAPHORES[provider] = asyncio.Semaphore(max_concurrent)
    semaphore = _SEMAPHORES[provider]

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with semaphore:
                return await func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
