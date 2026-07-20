"""Web search tool (Tavily), cached and retried.

The plain ``web_search`` coroutine holds the logic and is what tests call
directly. ``web_search_tool`` wraps it with ``@tool`` so MAF agents can call
it; the docstring and type hints below become the tool's JSON schema, so
keep them precise -- the SLM reads them to decide how to call this.
"""

import asyncio
from typing import Annotated, Any

from agent_framework import FunctionTool, tool
from tavily import TavilyClient
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from mas_deepr.tools.cache import WebCache


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _tavily_search(
    client: TavilyClient, query: str, max_results: int
) -> dict[str, Any]:
    result: dict[str, Any] = client.search(
        query=query, max_results=max_results, search_depth="basic"
    )
    return result


async def web_search(
    query: str,
    *,
    max_results: int,
    api_key: str,
    cache: WebCache,
) -> list[dict[str, str]]:
    """Run a cached web search and return normalized hits.

    Returns a list of ``{"title", "url", "snippet"}`` dicts, capped at
    ``max_results``. Results are cached indefinitely by (query, max_results)
    so repeated eval runs are reproducible and cheap.
    """
    cache_key = cache.make_key("search", query=query, max_results=max_results)
    cached = cache.get(cache_key)
    if cached is not None:
        hits: list[dict[str, str]] = cached["hits"]
        return hits

    client = TavilyClient(api_key=api_key)
    raw = await asyncio.to_thread(_tavily_search, client, query, max_results)
    hits = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        }
        for r in raw.get("results", [])[:max_results]
    ]
    cache.set(cache_key, "search", {"hits": hits})
    return hits


def make_web_search_tool(
    *, api_key: str, cache: WebCache, max_results: int
) -> FunctionTool:
    """Build the MAF-callable search tool bound to concrete settings/cache."""

    @tool(
        name="web_search",
        description="Search the web and return titles, URLs, and snippets.",
    )
    async def web_search_tool(
        query: Annotated[str, "The search query, in natural language."],
    ) -> list[dict[str, str]]:
        return await web_search(
            query, max_results=max_results, api_key=api_key, cache=cache
        )

    return web_search_tool
