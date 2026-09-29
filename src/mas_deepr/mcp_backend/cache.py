"""Signature-aware cache decorator for MCP provider functions.

The one genuinely-worth-porting idea from DR-Tulu's own
``mcp_backend/cache.py``: bind a call's actual arguments against the
wrapped function's real signature (defaults included) before hashing, so
``search(q="x")`` and ``search("x")`` collide correctly. Everything else
about that file (diskcache, TTL) is deliberately NOT ported -- this reuses
``tools/cache.py::WebCache`` (SQLite) and its documented indefinite-TTL,
cache-forever-for-reproducibility intent unchanged.
"""

import functools
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from mas_deepr.tools.cache import WebCache

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


def cached(kind: str, cache: WebCache) -> Callable[[F], F]:
    """Decorator: cache an async provider function's result in ``cache``,
    keyed by ``kind`` + its bound arguments (signature-normalized)."""

    def decorator(func: F) -> F:
        sig = inspect.signature(func)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            key = cache.make_key(kind, **bound.arguments)
            hit = cache.get(key)
            if hit is not None:
                return hit["value"]
            result = await func(*args, **kwargs)
            cache.set(key, kind, {"value": result})
            return result

        return wrapper  # type: ignore[return-value]

    return decorator
