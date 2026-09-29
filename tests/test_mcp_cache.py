"""``mcp_backend/cache.py``'s signature-aware ``@cached`` decorator,
tested in isolation against a fake async provider function -- not through
the server, so there's no functools.partial-binding-timing subtlety to
worry about (see test_mcp_server.py's docstring)."""

from pathlib import Path

import pytest

from mas_deepr.mcp_backend.cache import cached
from mas_deepr.tools.cache import WebCache


@pytest.mark.asyncio
async def test_cached_hits_backend_once_then_serves_from_cache(
    tmp_path: Path,
) -> None:
    calls = []
    cache = WebCache(tmp_path / "cache.sqlite3")

    @cached("search", cache)
    async def fake_search(query: str, *, max_results: int) -> list[str]:
        calls.append(query)
        return [f"hit for {query}"]

    r1 = await fake_search("france", max_results=2)
    r2 = await fake_search("france", max_results=2)

    assert len(calls) == 1
    assert r1 == r2 == ["hit for france"]


@pytest.mark.asyncio
async def test_cached_key_is_signature_normalized(tmp_path: Path) -> None:
    """Positional and keyword calls with the same effective arguments (and
    the same defaults applied) must collide on one cache key."""
    calls = []
    cache = WebCache(tmp_path / "cache.sqlite3")

    @cached("search", cache)
    async def fake_search(query: str, *, max_results: int = 5) -> list[str]:
        calls.append(query)
        return [query]

    await fake_search("q", max_results=5)
    await fake_search(query="q")  # same effective call: max_results defaults to 5

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_cached_distinguishes_different_arguments(tmp_path: Path) -> None:
    calls = []
    cache = WebCache(tmp_path / "cache.sqlite3")

    @cached("search", cache)
    async def fake_search(query: str, *, max_results: int) -> list[str]:
        calls.append((query, max_results))
        return [query]

    await fake_search("q", max_results=2)
    await fake_search("q", max_results=3)

    assert len(calls) == 2
