"""Tests for the opt-in Tavily fallback provider. See test_search.py for
the primary SearXNG provider."""

from typing import Any

import pytest

from mas_deepr.mcp_backend.providers import tavily as tavily_module
from mas_deepr.mcp_backend.providers.tavily import search as tavily_search


@pytest.mark.asyncio
async def test_tavily_search_normalizes_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_tavily_search(client: Any, query: str, max_results: int) -> dict[str, Any]:
        calls.append(query)
        return {
            "results": [
                {"title": "A", "url": "http://a.com", "content": "snippet a"},
                {"title": "B", "url": "http://b.com", "content": "snippet b"},
            ]
        }

    monkeypatch.setattr(tavily_module, "_tavily_search", fake_tavily_search)

    hits = await tavily_search("capital of france", max_results=2, api_key="fake")

    assert len(calls) == 1
    assert hits == [
        {"title": "A", "url": "http://a.com", "snippet": "snippet a"},
        {"title": "B", "url": "http://b.com", "snippet": "snippet b"},
    ]


@pytest.mark.asyncio
async def test_tavily_search_respects_max_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_tavily_search(client: Any, query: str, max_results: int) -> dict[str, Any]:
        return {
            "results": [
                {"title": str(i), "url": f"http://{i}.com", "content": ""}
                for i in range(5)
            ]
        }

    monkeypatch.setattr(tavily_module, "_tavily_search", fake_tavily_search)

    hits = await tavily_search("q", max_results=2, api_key="fake")
    assert len(hits) == 2
