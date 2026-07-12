from pathlib import Path
from typing import Any

import pytest

from mas_deepr.tools import search as search_module
from mas_deepr.tools.cache import WebCache
from mas_deepr.tools.search import web_search


@pytest.mark.asyncio
async def test_web_search_hits_backend_once_then_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_tavily_search(client: Any, query: str, max_results: int) -> dict[str, Any]:
        calls.append(query)
        return {
            "results": [
                {"title": "A", "url": "http://a.com", "content": "snippet a"},
                {"title": "B", "url": "http://b.com", "content": "snippet b"},
            ]
        }

    monkeypatch.setattr(search_module, "_tavily_search", fake_tavily_search)
    cache = WebCache(tmp_path / "cache.sqlite3")

    hits1 = await web_search(
        "capital of france", max_results=2, api_key="fake", cache=cache
    )
    hits2 = await web_search(
        "capital of france", max_results=2, api_key="fake", cache=cache
    )

    assert len(calls) == 1  # second call served from cache
    assert hits1 == hits2
    assert hits1 == [
        {"title": "A", "url": "http://a.com", "snippet": "snippet a"},
        {"title": "B", "url": "http://b.com", "snippet": "snippet b"},
    ]


@pytest.mark.asyncio
async def test_web_search_respects_max_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_tavily_search(client: Any, query: str, max_results: int) -> dict[str, Any]:
        return {
            "results": [
                {"title": str(i), "url": f"http://{i}.com", "content": ""}
                for i in range(5)
            ]
        }

    monkeypatch.setattr(search_module, "_tavily_search", fake_tavily_search)
    cache = WebCache(tmp_path / "cache.sqlite3")

    hits = await web_search("q", max_results=2, api_key="fake", cache=cache)
    assert len(hits) == 2
