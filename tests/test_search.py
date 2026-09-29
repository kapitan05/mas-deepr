"""Tests for the primary search provider (SearXNG). See test_tavily.py for
the opt-in fallback provider."""

import httpx
import pytest
import respx

from mas_deepr.mcp_backend.providers.searxng import search as searxng_search


@pytest.mark.asyncio
@respx.mock
async def test_searxng_search_normalizes_hits() -> None:
    route = respx.get("http://searxng.local/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"title": "A", "url": "http://a.com", "content": "snippet a"},
                    {"title": "B", "url": "http://b.com", "content": "snippet b"},
                ]
            },
        )
    )

    hits = await searxng_search(
        "capital of france", max_results=2, base_url="http://searxng.local"
    )

    assert route.called
    assert hits == [
        {"title": "A", "url": "http://a.com", "snippet": "snippet a"},
        {"title": "B", "url": "http://b.com", "snippet": "snippet b"},
    ]


@pytest.mark.asyncio
@respx.mock
async def test_searxng_search_respects_max_results() -> None:
    respx.get("http://searxng.local/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"title": str(i), "url": f"http://{i}.com", "content": ""}
                    for i in range(5)
                ]
            },
        )
    )

    hits = await searxng_search("q", max_results=2, base_url="http://searxng.local")
    assert len(hits) == 2


@pytest.mark.asyncio
async def test_searxng_search_requires_base_url() -> None:
    with pytest.raises(ValueError, match="not configured"):
        await searxng_search("q", max_results=2, base_url="")


@pytest.mark.asyncio
@respx.mock
async def test_searxng_search_raises_on_missing_json_format() -> None:
    respx.get("http://searxng.local/search").mock(
        return_value=httpx.Response(200, json={"unrelated": "shape"})
    )

    with pytest.raises(ValueError, match="results"):
        await searxng_search("q", max_results=2, base_url="http://searxng.local")
