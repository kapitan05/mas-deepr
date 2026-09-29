"""``web_search``'s Tavily fallback (mcp_backend/server.py) -- confirmed
live that just offering tavily_search as a separate tool doesn't work (the
Browser never calls it on its own), so the fallback has to be transparent
and server-side.

Providers must be monkeypatched *before* ``build_server()`` runs (per
test_mcp_server.py's own docstring: the cache decorator binds a direct
reference to the provider function at server-build time), so these tests
build their own server/client instead of using the ``mcp_test_client``
fixture.
"""

from pathlib import Path

import pytest
from fastmcp import Client

from mas_deepr.config import Settings
from mas_deepr.mcp_backend import build_server
from mas_deepr.mcp_backend.providers import searxng, tavily


async def _build_client(settings: Settings) -> Client:
    return Client(build_server(settings))


@pytest.mark.asyncio
async def test_web_search_falls_back_to_tavily_on_empty_searxng_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def empty_searxng(*args: object, **kwargs: object) -> list[dict[str, str]]:
        return []

    async def fake_tavily(*args: object, **kwargs: object) -> list[dict[str, str]]:
        return [{"title": "from tavily", "url": "https://x", "snippet": "s"}]

    monkeypatch.setattr(searxng, "search", empty_searxng)
    monkeypatch.setattr(tavily, "search", fake_tavily)

    settings = Settings(
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        cache_db=tmp_path / "cache.sqlite3",
        searxng_base_url="http://searxng.invalid",
        tavily_api_key="tvly-fake",
    )
    async with await _build_client(settings) as client:
        result = await client.call_tool("web_search", {"query": "q", "max_results": 1})
    assert result.data == [{"title": "from tavily", "url": "https://x", "snippet": "s"}]


@pytest.mark.asyncio
async def test_web_search_falls_back_to_tavily_on_searxng_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def broken_searxng(*args: object, **kwargs: object) -> list[dict[str, str]]:
        raise ValueError("SearXNG response has no 'results' key")

    async def fake_tavily(*args: object, **kwargs: object) -> list[dict[str, str]]:
        return [{"title": "from tavily", "url": "https://x", "snippet": "s"}]

    monkeypatch.setattr(searxng, "search", broken_searxng)
    monkeypatch.setattr(tavily, "search", fake_tavily)

    settings = Settings(
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        cache_db=tmp_path / "cache.sqlite3",
        searxng_base_url="http://searxng.invalid",
        tavily_api_key="tvly-fake",
    )
    async with await _build_client(settings) as client:
        result = await client.call_tool("web_search", {"query": "q", "max_results": 1})
    assert result.data == [{"title": "from tavily", "url": "https://x", "snippet": "s"}]


@pytest.mark.asyncio
async def test_web_search_no_fallback_when_tavily_not_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No API key -- e.g. a wiki_paper-scope server config -- must return
    the empty result as-is, never attempt Tavily."""
    calls = {"tavily": 0}

    async def empty_searxng(*args: object, **kwargs: object) -> list[dict[str, str]]:
        return []

    async def counting_tavily(*args: object, **kwargs: object) -> list[dict[str, str]]:
        calls["tavily"] += 1
        return [{"title": "should not happen", "url": "", "snippet": ""}]

    monkeypatch.setattr(searxng, "search", empty_searxng)
    monkeypatch.setattr(tavily, "search", counting_tavily)
    # A real TAVILY_API_KEY otherwise wins via *two* independent paths: (1)
    # settings.py's module-level load_dotenv() already populated os.environ
    # at import time, and (2) Settings' own env_file=".env" source reads the
    # real file directly, regardless of os.environ. delenv defeats (1);
    # _env_file=None below defeats (2) -- either alone is insufficient.
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        cache_db=tmp_path / "cache.sqlite3",
        searxng_base_url="http://searxng.invalid",
        tavily_api_key="",
    )
    async with await _build_client(settings) as client:
        result = await client.call_tool("web_search", {"query": "q", "max_results": 1})
    assert result.data == []
    assert calls["tavily"] == 0


@pytest.mark.asyncio
async def test_web_search_no_fallback_needed_when_searxng_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = {"tavily": 0}

    async def real_searxng(*args: object, **kwargs: object) -> list[dict[str, str]]:
        return [{"title": "from searxng", "url": "https://y", "snippet": "s"}]

    async def counting_tavily(*args: object, **kwargs: object) -> list[dict[str, str]]:
        calls["tavily"] += 1
        return [{"title": "should not happen", "url": "", "snippet": ""}]

    monkeypatch.setattr(searxng, "search", real_searxng)
    monkeypatch.setattr(tavily, "search", counting_tavily)

    settings = Settings(
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        cache_db=tmp_path / "cache.sqlite3",
        searxng_base_url="http://searxng.invalid",
        tavily_api_key="tvly-fake",
    )
    async with await _build_client(settings) as client:
        result = await client.call_tool("web_search", {"query": "q", "max_results": 1})
    expected = [{"title": "from searxng", "url": "https://y", "snippet": "s"}]
    assert result.data == expected
    assert calls["tavily"] == 0
