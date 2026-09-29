"""Tests for the direct Wikipedia provider (not routed through SearXNG --
see mcp_backend/providers/wikipedia.py's module docstring for why it's
separate)."""

import httpx
import pytest
import respx

from mas_deepr.mcp_backend.providers.wikipedia import search as wikipedia_search


@pytest.mark.asyncio
@respx.mock
async def test_wikipedia_search_normalizes_hits_and_strips_html() -> None:
    respx.get("https://en.wikipedia.org/w/rest.php/v1/search/page").mock(
        return_value=httpx.Response(
            200,
            json={
                "pages": [
                    {
                        "key": "Reinforcement_learning",
                        "title": "Reinforcement learning",
                        "excerpt": (
                            'In machine <span class="searchmatch">learning</span>...'
                        ),
                        "description": "Field of machine learning",
                    }
                ]
            },
        )
    )

    hits = await wikipedia_search("reinforcement learning", max_results=1)

    assert hits == [
        {
            "title": "Reinforcement learning",
            "url": "https://en.wikipedia.org/wiki/Reinforcement_learning",
            "snippet": "In machine learning...",
        }
    ]


@pytest.mark.asyncio
@respx.mock
async def test_wikipedia_search_falls_back_to_description_when_no_excerpt() -> None:
    respx.get("https://en.wikipedia.org/w/rest.php/v1/search/page").mock(
        return_value=httpx.Response(
            200,
            json={
                "pages": [
                    {
                        "key": "Example",
                        "title": "Example",
                        "excerpt": "",
                        "description": "A description",
                    }
                ]
            },
        )
    )

    hits = await wikipedia_search("example", max_results=1)
    assert hits[0]["snippet"] == "A description"


@pytest.mark.asyncio
@respx.mock
async def test_wikipedia_search_respects_max_results() -> None:
    respx.get("https://en.wikipedia.org/w/rest.php/v1/search/page").mock(
        return_value=httpx.Response(
            200,
            json={
                "pages": [
                    {"key": str(i), "title": str(i), "excerpt": "", "description": ""}
                    for i in range(5)
                ]
            },
        )
    )

    hits = await wikipedia_search("q", max_results=2)
    assert len(hits) == 2


@pytest.mark.asyncio
@respx.mock
async def test_wikipedia_search_handles_no_results() -> None:
    respx.get("https://en.wikipedia.org/w/rest.php/v1/search/page").mock(
        return_value=httpx.Response(200, json={"pages": []})
    )

    hits = await wikipedia_search("nonsense query xyz", max_results=5)
    assert hits == []
